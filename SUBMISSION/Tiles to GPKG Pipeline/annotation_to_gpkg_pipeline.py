"""
MoPR Hackathon – Annotation → GeoTIFF Mosaic → GPKG Pipeline
=============================================================
Converts the tiled PNG annotations in Training/1. Tiles/annotations/
into a georeferenced GeoTIFF mosaic (COG) and a GeoPackage (GPKG) with
per-class polygon vectors.

This pipeline is designed to run on Google Colab with an A100 GPU,
but the vectorisation steps are CPU-only.

Designed so the same script can swap in AI model logit outputs for the
annotation PNGs – just point ANNOTATION_DIR at your model predictions.

SETUP (run these cells first in Colab):
    !pip install rasterio rio-cogeo shapely geopandas tqdm
    !apt-get install -y gdal-bin python3-gdal   # for gdal_polygonize

Author: MoPR Hackathon Team
"""

# ─────────────────────────────────────────────────────────
#  0.  CONFIGURATION  –  edit these paths before running
# ─────────────────────────────────────────────────────────
import os, re, struct
from pathlib import Path

# -- Paths -------------------------------------------------------
# Root of Training/1. Tiles  (change this to your Colab mount path)
DRIVE_BASE   = Path('/content/drive/MyDrive/mopr-hackathon/gpkg_pipeline')
TILES_ROOT   = DRIVE_BASE / 'training' 
OUTPUT_DIR   = DRIVE_BASE / 'output'

ANNOTATION_DIR  = TILES_ROOT / "annotations"   # source annotation PNGs
IMAGES_DIR      = TILES_ROOT / "images"        # source GeoTIFF tiles (for georef)

# Output folder (will be created if missing)
MOSAIC_PATH     = OUTPUT_DIR / "annotation_mosaic.tif"   # raw mosaic GeoTIFF
COG_PATH        = OUTPUT_DIR / "annotation_mosaic_cog.tif"
GPKG_PATH       = OUTPUT_DIR / "annotation_vectors.gpkg"
GEOREF_DIR      = OUTPUT_DIR / "georeferenced_tiles"     # intermediate per-tile GeoTIFFs

# -- Class mapping -----------------------------------------------
# Pixel value → semantic class name
# 0   = background (excluded from vectors)
# 1   = Building (roof_type encoded separately in Built_Up_Area)
# 3   = Road
# 4   = Water Body
# 5   = Utility / Infrastructure
# 255 = No-data / ignore boundary  (excluded from vectors)
CLASS_MAP = {
    1: "road",
    2: "bridge",
    3: "waterbody",
    4: "utility_poly",
    5: "Built_Up_Area"
}

# Minimum polygon area to keep (m²) – removes noise
MIN_AREA_M2 = 2.0

# Confidence value stamped on ground-truth annotations
# (will be replaced with real model scores when using AI outputs)
DEFAULT_CONFIDENCE = 1.0

# ─────────────────────────────────────────────────────────
#  1.  HELPERS
# ─────────────────────────────────────────────────────────

def read_geotiff_transform(tif_path: Path):
    """
    Read the affine geotransform and CRS EPSG from a GeoTIFF using only PIL.
    Returns (transform_dict, epsg_int).
    transform_dict keys: origin_x, origin_y, pixel_w, pixel_h
    """
    from PIL import Image

    img = Image.open(tif_path)
    tags = img.tag_v2

    # ModelPixelScaleTag (33550): (dx, dy, dz)
    pixel_scale = tags.get(33550)
    # ModelTiepointTag (33922): (i, j, k, X, Y, Z)  – upper-left corner
    tiepoint = tags.get(33922)

    if pixel_scale is None or tiepoint is None:
        raise ValueError(f"No georeference tags in {tif_path}")

    dx, dy, _ = pixel_scale
    tx, ty = tiepoint[3], tiepoint[4]   # Easting, Northing of top-left pixel

    # GeoKeyDirectoryTag (34735) → ProjectedCSTypeGeoKey = key 3072
    geo_keys = tags.get(34735, ())
    epsg = None
    keys = list(geo_keys)
    # Parse GeoKey directory: [KeyDirVersion, KeyRevision, MinorRevision, NumberOfKeys,
    #   key_id, TIFFTagLocation, count, value_offset, ...]
    n_keys = keys[3] if len(keys) > 3 else 0
    for i in range(n_keys):
        base = 4 + i * 4
        if base + 3 < len(keys) and keys[base] == 3072:   # ProjectedCSTypeGeoKey
            epsg = keys[base + 3]
            break

    return {
        "origin_x": tx,
        "origin_y": ty,
        "pixel_w":  dx,    # positive  (west → east)
        "pixel_h": -dy,    # negative  (north → south, standard raster convention)
    }, epsg or 32644        # default EPSG:32644 = WGS84/UTM zone 44N


def write_geotiff(
    array,            # 2-D numpy uint8 array
    transform: dict,  # from read_geotiff_transform
    epsg: int,
    out_path: Path,
):
    """Write a single-band uint8 GeoTIFF with the given affine transform."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    h, w = array.shape
    affine = from_origin(
        transform["origin_x"],
        transform["origin_y"],
        transform["pixel_w"],
        -transform["pixel_h"],   # from_origin expects positive pixel height
    )
    crs = CRS.from_epsg(epsg)

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=h, width=w,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=affine,
        compress="lzw",
        nodata=255,
    ) as dst:
        dst.write(array[None, :, :])    # add band dimension


# ─────────────────────────────────────────────────────────
#  2.  STEP 1 – Georeference every annotation tile
# ─────────────────────────────────────────────────────────

def step1_georeference_tiles(annotation_dir, images_dir, georef_dir):
    """
    For each annotation PNG, read the geotransform from the matching TIF
    and write out a georeferenced GeoTIFF.

    Matching rule: bade_tile_XXXXXX.png  ←→  bade_tile_XXXXXX.tif
    (generalises: strip extension, match basename)
    """
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    georef_dir = Path(georef_dir)
    georef_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(Path(annotation_dir).glob("*.png"))
    print(f"[Step 1] Found {len(ann_files)} annotation tiles to georeference.")

    skipped = []
    for ann_path in tqdm(ann_files, desc="Georeferencing tiles"):
        # Find the matching TIF
        tif_path = Path(images_dir) / (ann_path.stem + ".tif")
        if not tif_path.exists():
            skipped.append(ann_path.name)
            continue

        out_path = georef_dir / (ann_path.stem + "_georef.tif")
        if out_path.exists():
            continue   # already done – resume-safe

        transform, epsg = read_geotiff_transform(tif_path)
        arr = np.array(Image.open(ann_path))

        write_geotiff(arr, transform, epsg, out_path)

    if skipped:
        print(f"  [WARNING] {len(skipped)} tiles had no matching TIF and were skipped: {skipped[:5]}…")

    print(f"[Step 1] Done. Georeferenced tiles written to: {georef_dir}")


# ─────────────────────────────────────────────────────────
#  3.  STEP 2 – Mosaic all georeferenced tiles → COG
# ─────────────────────────────────────────────────────────

def step2_mosaic_to_cog(georef_dir, mosaic_path, cog_path):
    """
    Merge all per-tile GeoTIFFs into a single raster mosaic using rasterio.merge,
    then convert to a Cloud-Optimized GeoTIFF (COG) using rio-cogeo.

    Where tiles overlap (64 px overlap from tiling), the LAST tile written wins
    (rasterio.merge default = 'first', here we use 'last' to match inference).
    For ground-truth annotations the tiles should be non-overlapping, so this
    doesn't matter – but keep it consistent with the inference pipeline.
    """
    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from tqdm import tqdm
    from rio_cogeo.cogeo import cog_translate
    from rio_cogeo.profiles import cog_profiles

    georef_files = sorted(Path(georef_dir).glob("*_georef.tif"))
    print(f"[Step 2] Merging {len(georef_files)} georeferenced tiles into mosaic…")

    # Open all datasets
    datasets = [rasterio.open(f) for f in tqdm(georef_files, desc="Opening tiles")]
    mosaic_arr, mosaic_transform = merge(datasets, method="last", nodata=255)
    mosaic_meta = datasets[0].meta.copy()
    for ds in datasets:
        ds.close()

    mosaic_meta.update({
        "driver": "GTiff",
        "height": mosaic_arr.shape[1],
        "width":  mosaic_arr.shape[2],
        "transform": mosaic_transform,
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "nodata": 255,
    })

    mosaic_path = Path(mosaic_path)
    mosaic_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Step 2] Writing raw mosaic → {mosaic_path}")
    with rasterio.open(mosaic_path, "w", **mosaic_meta) as dst:
        dst.write(mosaic_arr)

    print(f"[Step 2] Converting to COG → {cog_path}")
    cog_profile = cog_profiles.get("deflate")
    cog_translate(
        str(mosaic_path),
        str(cog_path),
        cog_profile,
        in_memory=True,
        quiet=False,
        nodata=255,
        overview_resampling="nearest",   # nearest-neighbour for class labels
    )

    print(f"[Step 2] Done. COG written to: {cog_path}")
    return cog_path


# ─────────────────────────────────────────────────────────
#  4.  STEP 2b – Validate the COG
# ─────────────────────────────────────────────────────────

def step2b_validate_cog(cog_path):
    """Run rio-cogeo validation and print the report."""
    from rio_cogeo.cogeo import cog_validate
    is_valid, errors, warnings = cog_validate(str(cog_path))
    print(f"\n[COG Validation] {cog_path.name}")
    print(f"  Valid  : {is_valid}")
    if errors:
        print(f"  Errors : {errors}")
    if warnings:
        print(f"  Warnings: {warnings}")
    return is_valid


# ─────────────────────────────────────────────────────────
#  5.  STEP 3 – Polygonise per class → GPKG
# ─────────────────────────────────────────────────────────

def step3_polygonise_to_gpkg(cog_path, gpkg_path, class_map, min_area_m2, confidence):
    """
    For each class in class_map:
      1. Mask the mosaic to the single class value
      2. Run rasterio.features.shapes to get polygon geometries
      3. Build a GeoDataFrame with attributes: class_id, class_name, area_m2, confidence
      4. Apply morphological cleanup (remove slivers, fill holes)
      5. Append to the GPKG layer 'segments'

    All classes end up in one GPKG layer for easy QGIS loading.
    """
    import numpy as np
    import rasterio
    from rasterio.features import shapes
    import geopandas as gpd
    import shapely
    from shapely.geometry import shape, MultiPolygon
    from shapely.ops import unary_union
    from tqdm import tqdm

    print(f"\n[Step 3] Polygonising classes from {cog_path.name}…")
    gpkg_path = Path(gpkg_path)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    all_frames = []

    with rasterio.open(cog_path) as src:
        data = src.read(1)                 # full mosaic in memory (uint8, ~few MB)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata or 255

    for class_id, class_name in tqdm(class_map.items(), desc="Polygonising classes"):
        # Binary mask: 1 where this class, 0 everywhere else
        mask = (data == class_id).astype(np.uint8)

        if mask.sum() == 0:
            print(f"  Class {class_id} ({class_name}): no pixels found – skipping")
            continue

        # Run shapely-compatible polygon extraction via rasterio.features.shapes
        # shapes() yields (geometry_dict, pixel_value) for each connected component
        polys = []
        for geom_dict, val in shapes(mask, mask=mask, transform=transform):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty:
                continue

            # ── Morphological cleanup ──────────────────────────────
            # 1. Remove slivers (area below threshold)
            if geom.area < min_area_m2:
                continue
            # 2. Simplify to reduce vertices (tolerance = 1/3 pixel width)
            pixel_size = transform.a        # pixel width in map units (metres)
            geom = geom.simplify(pixel_size / 3, preserve_topology=True)
            # 3. Fill small interior holes (< 1 m² donut artefacts)
            if geom.geom_type == "Polygon":
                holes_kept = [r for r in geom.interiors if shapely.geometry.Polygon(r).area >= 1.0]
                geom = shapely.geometry.Polygon(geom.exterior, holes_kept)
            # 4. Buffer by zero to fix any self-intersections
            geom = geom.buffer(0)

            polys.append({
                "geometry":   geom,
                "class_id":   class_id,
                "class_name": class_name,
                "area_m2":    round(geom.area, 4),
                "confidence": confidence,
            })

        if not polys:
            print(f"  Class {class_id} ({class_name}): all polygons below min area – skipping")
            continue

        gdf = gpd.GeoDataFrame(polys, crs=crs)
        all_frames.append(gdf)
        print(f"  Class {class_id:3d} ({class_name:20s}): {len(gdf):5d} polygons  "
              f"(total area = {gdf['area_m2'].sum():.1f} m²)")

    if not all_frames:
        print("[Step 3] ERROR: No polygons produced – check class values and mask.")
        return

    combined = gpd.GeoDataFrame(
        gpd.pd.concat(all_frames, ignore_index=True),
        crs=all_frames[0].crs,
    )

    print(f"\n[Step 3] Writing {len(combined)} total polygons to GPKG…")
    combined.to_file(
        gpkg_path,
        layer="segments",
        driver="GPKG",
    )
    print(f"[Step 3] Done. GPKG written to: {gpkg_path}")
    return combined


# ─────────────────────────────────────────────────────────
#  6.  STEP 4 – QGIS verification helper
# ─────────────────────────────────────────────────────────

QGIS_README = """
HOW TO VERIFY IN QGIS
=====================

1.  Open QGIS Desktop (3.x).

2.  Load the COG mosaic:
      Layer → Add Layer → Add Raster Layer → browse to annotation_mosaic_cog.tif
      In the layer Symbology, choose:
        Render type = Paletted/Unique values
        Classify → assign class colours:
          0   → transparent
          1   → x (road)
          2   → x (bridge)
          3   → x (waterbody)
          4   → x (utility)
          5   → x (built_up_area)
          255 → transparent

3.  Load the GPKG vectors:
      Layer → Add Layer → Add Vector Layer → browse to annotation_vectors.gpkg
      Choose layer = segments
      Go to Symbology, and change from Single Symbol -> Categorized. Under value choose class_name and click Classify.

4.  Overlay with the original orthophoto tiles (drag any .tif from images/ folder).

5.  Visual QA checks:
      ✓  Building footprints align with visible structures
      ✓  Road polygons follow road corridors
      ✓  Water body polygons cover water surfaces
      ✓  No major misalignment between raster mask and GPKG vectors
      ✓  area_m2 values look physically reasonable (~10–500 m² per building)

6.  To measure CRS alignment:
      Enable the Identify Features tool, click a polygon vertex, then click the
      matching pixel in the COG – coordinates should agree to within 1 pixel
      (~3–5 cm for this 3.5 cm/px data).
"""


# ─────────────────────────────────────────────────────────
#  7.  MAIN – run all steps in sequence
# ─────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("  MoPR Annotation → COG + GPKG Pipeline")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: georeference every annotation PNG using its matching TIF
    step1_georeference_tiles(ANNOTATION_DIR, IMAGES_DIR, GEOREF_DIR)

    # Step 2: mosaic → COG
    cog_path = step2_mosaic_to_cog(GEOREF_DIR, MOSAIC_PATH, COG_PATH)

    # Step 2b: validate
    step2b_validate_cog(Path(cog_path))

    # Step 3: polygonise → GPKG
    step3_polygonise_to_gpkg(
        cog_path=Path(cog_path),
        gpkg_path=GPKG_PATH,
        class_map=CLASS_MAP,
        min_area_m2=MIN_AREA_M2,
        confidence=DEFAULT_CONFIDENCE,
    )

    # Print QGIS instructions
    print("\n" + QGIS_README)

    print("\nOutput files:")
    for p in [COG_PATH, GPKG_PATH]:
        p = Path(p)
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.name:<40s}  {size_mb:.1f} MB")

    print("\nDone ✓")


if __name__ == "__main__":
    run_pipeline()
