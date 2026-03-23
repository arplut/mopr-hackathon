#!/usr/bin/env python3
"""
mask_to_gpkg.py: Polygonize segmentation masks and create feature layers in GeoPackage format.

This script reads a single-band segmentation mask GeoTIFF and creates a GeoPackage with:
  - buildings: RCC/tile/tin/thatched roof polygons with area, confidence, and village_id
  - roads: Road features (pucca/kaccha) with length and village_id
  - water_bodies: Water polygon features with area and village_id
  - vegetation: Vegetation polygon features with area and village_id

Features are simplified with configurable tolerance and filtered by minimum area.

Optionally reads a companion confidence raster (*_conf.tif, float32, per-pixel max softmax)
to populate the confidence attribute on buildings. Falls back to 1.0 if unavailable.

Author: MoPR Hackathon Team
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape as shapely_shape, LineString, MultiLineString, mapping
from shapely.ops import unary_union
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Class to feature-layer mapping
# class_id -> (layer_name, subtype_value)
CLASS_MAPPING = {
    1: ("buildings", "RCC"),
    2: ("buildings", "tile"),
    3: ("buildings", "tin"),
    4: ("buildings", "thatched"),
    5: ("roads", "pucca"),
    6: ("roads", "kaccha"),
    7: ("water_bodies", None),
    8: ("vegetation", None),
}


def mask_to_gpkg(
    mask_path: str,
    output_path: str,
    village_id: str,
    confidence_raster_path: str = None,
    simplify_tolerance: float = 0.3,
    min_area_m2: float = 2.0,
) -> dict:
    """
    Polygonize segmentation mask and create GeoPackage with feature layers.

    Args:
        mask_path: Path to single-band uint8 segmentation mask GeoTIFF
        output_path: Path to save output GeoPackage
        village_id: Village identifier for attribute population
        confidence_raster_path: Optional path to float32 confidence raster (per-pixel max softmax)
        simplify_tolerance: Simplification tolerance in CRS units (default 0.3m for projected CRS)
        min_area_m2: Minimum polygon area in m² to keep (default 2.0)

    Returns:
        Dictionary with feature statistics
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading mask: {mask_path}")

    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        crs = src.crs
        transform = src.transform

        if crs is None:
            raise ValueError("Mask file must have valid CRS")

        logger.info(f"Mask shape: {mask_data.shape}, CRS: {crs}")

    # Load confidence raster if available
    confidence_data = None
    if confidence_raster_path:
        conf_path = Path(confidence_raster_path)
        if conf_path.exists():
            with rasterio.open(conf_path) as conf_src:
                confidence_data = conf_src.read(1)
                logger.info(f"Loaded confidence raster: {conf_path}")
        else:
            logger.warning(f"Confidence raster not found: {conf_path}, using 1.0")

    # Auto-detect confidence raster from naming convention
    if confidence_data is None:
        auto_conf = mask_path.with_name(mask_path.stem.replace("_segmentation", "_confidence") + ".tif")
        if auto_conf.exists():
            with rasterio.open(auto_conf) as conf_src:
                confidence_data = conf_src.read(1)
                logger.info(f"Auto-detected confidence raster: {auto_conf}")

    # Compute pixel area in CRS units² (assumes projected CRS with meter units)
    pixel_area = abs(transform.a * transform.e)
    logger.info(f"Pixel area: {pixel_area:.4f} CRS-units²")

    # Initialize feature collections
    buildings_list = []
    roads_list = []
    water_list = []
    vegetation_list = []

    unique_classes = np.unique(mask_data)
    logger.info(f"Classes found in mask: {unique_classes.tolist()}")

    for class_id in tqdm(unique_classes, desc="Polygonizing classes"):
        if class_id == 0:  # Skip background
            continue

        if class_id not in CLASS_MAPPING:
            logger.warning(f"Unknown class ID: {class_id}, skipping")
            continue

        layer_type, subtype = CLASS_MAPPING[class_id]

        # Create binary mask for this class
        binary_mask = (mask_data == class_id).astype(np.uint8)

        # Polygonize using rasterio — returns GeoJSON-like geometry dicts
        for geom_dict, value in rasterio_shapes(binary_mask, transform=transform):
            if value == 0:  # Skip background holes
                continue

            try:
                # Use shapely.geometry.shape to handle all geometry types
                # (Polygon, MultiPolygon, with holes)
                geom = shapely_shape(geom_dict)

                # Fix invalid geometries
                if not geom.is_valid:
                    geom = geom.buffer(0)

                if geom.is_empty:
                    continue

                # Filter by minimum area
                if geom.area < min_area_m2:
                    continue

                # Simplify
                geom = geom.simplify(simplify_tolerance, preserve_topology=True)

                if geom.is_empty:
                    continue

                # Compute mean confidence for this polygon
                poly_confidence = _compute_polygon_confidence(
                    geom, confidence_data, transform, mask_data.shape
                ) if confidence_data is not None else 1.0

                # Route to appropriate layer
                if layer_type == "buildings":
                    buildings_list.append({
                        "geometry": geom,
                        "roof_type": subtype,
                        "area_m2": round(geom.area, 2),
                        "confidence": round(poly_confidence, 3),
                        "village_id": village_id,
                    })

                elif layer_type == "roads":
                    # Extract centerline from road polygon
                    centerline = _extract_road_centerline(geom)
                    if centerline is not None and not centerline.is_empty:
                        roads_list.append({
                            "geometry": centerline,
                            "road_type": subtype,
                            "length_m": round(centerline.length, 2),
                            "village_id": village_id,
                        })

                elif layer_type == "water_bodies":
                    water_list.append({
                        "geometry": geom,
                        "area_m2": round(geom.area, 2),
                        "village_id": village_id,
                    })

                elif layer_type == "vegetation":
                    vegetation_list.append({
                        "geometry": geom,
                        "area_m2": round(geom.area, 2),
                        "village_id": village_id,
                    })

            except Exception as e:
                logger.debug(f"Error processing geometry for class {class_id}: {e}")
                continue

    # Write GeoPackage layers
    logger.info(f"Creating GeoPackage: {output_path}")

    # Remove existing file to avoid append conflicts
    if output_path.exists():
        output_path.unlink()

    _write_gpkg_layer(buildings_list, crs, output_path, "buildings")
    _write_gpkg_layer(roads_list, crs, output_path, "roads")
    _write_gpkg_layer(water_list, crs, output_path, "water_bodies")
    _write_gpkg_layer(vegetation_list, crs, output_path, "vegetation")

    logger.info(f"GeoPackage saved: {output_path}")

    return {
        "buildings_count": len(buildings_list),
        "roads_count": len(roads_list),
        "water_count": len(water_list),
        "vegetation_count": len(vegetation_list),
        "output_path": str(output_path),
        "crs": str(crs),
    }


def _write_gpkg_layer(features: list, crs, output_path: Path, layer_name: str):
    """Write a list of feature dicts to a GPKG layer."""
    if not features:
        logger.info(f"  {layer_name}: 0 features (skipped)")
        return
    gdf = gpd.GeoDataFrame(features, crs=crs)
    # Use append mode if file already exists
    mode = "a" if output_path.exists() else "w"
    gdf.to_file(output_path, layer=layer_name, driver="GPKG", mode=mode)
    logger.info(f"  {layer_name}: {len(gdf)} features written")


def _compute_polygon_confidence(
    geom,
    confidence_data: np.ndarray,
    transform,
    raster_shape: tuple,
) -> float:
    """Compute mean confidence within a polygon from the confidence raster."""
    try:
        from rasterio.features import geometry_mask
        mask = geometry_mask(
            [mapping(geom)],
            out_shape=raster_shape,
            transform=transform,
            invert=True,  # True inside polygon
        )
        values = confidence_data[mask]
        if len(values) > 0:
            return float(np.mean(values))
    except Exception:
        pass
    return 1.0


def _extract_road_centerline(polygon) -> LineString:
    """
    Extract approximate centerline from a road polygon using morphological skeletonization.

    Falls back to a simplified boundary-based approach if skimage is unavailable.
    """
    try:
        from skimage.morphology import skeletonize
        from rasterio.features import geometry_mask
        from rasterio.transform import from_bounds

        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        width_m = bounds[2] - bounds[0]
        height_m = bounds[3] - bounds[1]

        if width_m < 0.5 or height_m < 0.5:
            # Too small — return a simple line between two extremes
            coords = list(polygon.exterior.coords)
            if len(coords) >= 2:
                return LineString([coords[0], coords[len(coords) // 2]])
            return None

        # Rasterize at ~0.2m resolution for skeletonization
        pixel_size = 0.2
        ncols = max(int(width_m / pixel_size), 3)
        nrows = max(int(height_m / pixel_size), 3)

        local_transform = from_bounds(*bounds, ncols, nrows)

        # Create binary mask of the polygon
        mask = geometry_mask(
            [mapping(polygon)],
            out_shape=(nrows, ncols),
            transform=local_transform,
            invert=True,
        )

        # Skeletonize
        skeleton = skeletonize(mask)

        # Extract skeleton pixel coordinates and convert to geographic
        ys, xs = np.where(skeleton)
        if len(xs) < 2:
            # Skeleton too small
            coords = list(polygon.exterior.coords)
            if len(coords) >= 2:
                return LineString([coords[0], coords[len(coords) // 2]])
            return None

        # Convert pixel coords to geographic coords
        geo_coords = [
            local_transform * (int(x), int(y)) for x, y in zip(xs, ys)
        ]

        # Sort by distance from first point to create a continuous line
        if len(geo_coords) >= 2:
            sorted_coords = _sort_skeleton_points(geo_coords)
            line = LineString(sorted_coords)
            return line.simplify(0.3, preserve_topology=True)

        return None

    except ImportError:
        # Fallback: use polygon medial axis approximation
        return _fallback_centerline(polygon)
    except Exception as e:
        logger.debug(f"Skeletonize failed: {e}")
        return _fallback_centerline(polygon)


def _sort_skeleton_points(coords: list) -> list:
    """Sort skeleton points into a roughly continuous path using nearest-neighbor."""
    remaining = list(coords)
    sorted_pts = [remaining.pop(0)]

    while remaining:
        last = sorted_pts[-1]
        dists = [((p[0] - last[0])**2 + (p[1] - last[1])**2) for p in remaining]
        nearest_idx = int(np.argmin(dists))
        sorted_pts.append(remaining.pop(nearest_idx))

    return sorted_pts


def _fallback_centerline(polygon) -> LineString:
    """Simple centerline: line along the major axis through the polygon."""
    try:
        centroid = polygon.centroid
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        if width > height:
            p1 = (bounds[0], centroid.y)
            p2 = (bounds[2], centroid.y)
        else:
            p1 = (centroid.x, bounds[1])
            p2 = (centroid.x, bounds[3])

        line = LineString([p1, p2])
        clipped = line.intersection(polygon)

        if clipped.is_empty:
            return line
        if isinstance(clipped, (LineString, MultiLineString)):
            return clipped
        return line

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Polygonize segmentation masks and create GeoPackage feature layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_to_gpkg.py --mask segmentation.tif --output features.gpkg --village_id VILL001
  python mask_to_gpkg.py --mask mask.tif --output out.gpkg --village_id VILL001 \\
    --confidence_raster confidence.tif --simplify_tolerance 0.5
        """,
    )

    parser.add_argument("--mask", type=str, required=True, help="Path to segmentation mask GeoTIFF")
    parser.add_argument("--output", type=str, required=True, help="Path to save output GeoPackage")
    parser.add_argument("--village_id", type=str, required=True, help="Village identifier")
    parser.add_argument(
        "--confidence_raster", type=str, default=None,
        help="Optional path to float32 confidence raster (per-pixel max softmax)",
    )
    parser.add_argument(
        "--simplify_tolerance", type=float, default=0.3,
        help="Simplification tolerance in CRS units (default: 0.3)",
    )
    parser.add_argument(
        "--min_area", type=float, default=2.0,
        help="Minimum polygon area in CRS-units² (default: 2.0)",
    )

    args = parser.parse_args()

    try:
        result = mask_to_gpkg(
            mask_path=args.mask,
            output_path=args.output,
            village_id=args.village_id,
            confidence_raster_path=args.confidence_raster,
            simplify_tolerance=args.simplify_tolerance,
            min_area_m2=args.min_area,
        )

        print(f"\n{'='*70}")
        print(f"POLYGONIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"Buildings:    {result['buildings_count']}")
        print(f"Roads:        {result['roads_count']}")
        print(f"Water bodies: {result['water_count']}")
        print(f"Vegetation:   {result['vegetation_count']}")
        print(f"Output:       {result['output_path']}")
        print(f"CRS:          {result['crs']}")
        print(f"{'='*70}\n")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
