#!/usr/bin/env python3
"""
mask_to_gpkg.py: Polygonize segmentation masks and create feature layers in GeoPackage format.

This script reads a single-band segmentation mask GeoTIFF and creates a GeoPackage with:
  - buildings: RCC/tile/tin/thatched roof polygons with area and confidence
  - roads: Road centerlines (pucca/kaccha) with length
  - water_bodies: Water polygon features
  - vegetation: Vegetation polygon features

Features are simplified with 0.3m tolerance and filtered (min 2 m²).

Author: MoPR Hackathon Team
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize
from shapely.strtree import STRtree
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Class to feature mapping
CLASS_MAPPING = {
    1: ("buildings", "RCC_roof"),
    2: ("buildings", "tile_roof"),
    3: ("buildings", "tin_roof"),
    4: ("buildings", "thatched_roof"),
    5: ("roads", "pucca"),
    6: ("roads", "kaccha"),
    7: ("water_bodies", None),
    8: ("vegetation", None),
}


def mask_to_gpkg(
    mask_path: str,
    output_path: str,
    village_id: str,
    simplify_tolerance: float = 0.3,
    min_area_m2: float = 2.0,
) -> dict:
    """
    Polygonize segmentation mask and create GeoPackage with feature layers.

    Args:
        mask_path: Path to single-band uint8 segmentation mask GeoTIFF
        output_path: Path to save output GeoPackage
        village_id: Village identifier for attribute population
        simplify_tolerance: Simplification tolerance in meters (default 0.3)
        min_area_m2: Minimum polygon area in m² to keep (default 2.0)

    Returns:
        Dictionary with feature statistics

    Raises:
        FileNotFoundError: If mask file does not exist
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
            logger.error("Mask file has no CRS. Cannot proceed.")
            raise ValueError("Mask file must have valid CRS")

        logger.info(f"Mask shape: {mask_data.shape}, CRS: {crs}")

    # Initialize feature collections
    buildings_list = []
    roads_list = []
    water_list = []
    vegetation_list = []

    # Get pixel area in m²
    pixel_area = abs(transform.a * transform.e)
    logger.info(f"Pixel area: {pixel_area:.4f} m²")

    # Polygonize each class
    logger.info("Polygonizing features...")

    unique_classes = np.unique(mask_data)

    for class_id in tqdm(unique_classes, desc="Polygonizing classes"):
        if class_id == 0:  # Skip background
            continue

        # Create binary mask for this class
        binary_mask = (mask_data == class_id).astype(np.uint8)

        # Polygonize
        try:
            geometries = list(shapes(binary_mask, transform=transform))
        except Exception as e:
            logger.warning(f"Error polygonizing class {class_id}: {e}")
            continue

        if class_id not in CLASS_MAPPING:
            logger.warning(f"Unknown class ID: {class_id}")
            continue

        layer_type, subtype = CLASS_MAPPING[class_id]

        for geom_data, value in geometries:
            if value == 0:  # Skip background
                continue

            try:
                geom = Polygon(geom_data["coordinates"][0])

                # Validate and simplify
                if not geom.is_valid:
                    geom = geom.buffer(0)

                if geom.area < min_area_m2:
                    continue

                geom = geom.simplify(simplify_tolerance)

                # Populate layer
                if layer_type == "buildings":
                    buildings_list.append({
                        "geometry": geom,
                        "roof_type": subtype,
                        "area_m2": geom.area,
                        "confidence": 1.0,
                        "village_id": village_id,
                    })

                elif layer_type == "roads":
                    # Convert to centerline (skeleton)
                    skeleton_geom = _extract_centerline(geom, simplify_tolerance)
                    if skeleton_geom is not None:
                        roads_list.append({
                            "geometry": skeleton_geom,
                            "road_type": subtype,
                            "length_m": skeleton_geom.length,
                            "village_id": village_id,
                        })

                elif layer_type == "water_bodies":
                    water_list.append({
                        "geometry": geom,
                        "area_m2": geom.area,
                        "village_id": village_id,
                    })

                elif layer_type == "vegetation":
                    vegetation_list.append({
                        "geometry": geom,
                        "area_m2": geom.area,
                        "village_id": village_id,
                    })

            except Exception as e:
                logger.debug(f"Error processing geometry for class {class_id}: {e}")
                continue

    # Create GeoDataFrames and save to GeoPackage
    logger.info(f"Creating GeoPackage: {output_path}")

    # Remove existing file if it exists
    if output_path.exists():
        output_path.unlink()

    with tqdm(total=4, desc="Writing layers", unit="layer") as pbar:
        # Buildings layer
        if buildings_list:
            gdf_buildings = gpd.GeoDataFrame(
                buildings_list,
                crs=crs,
            )
            gdf_buildings.to_file(output_path, layer="buildings", driver="GPKG")
            logger.info(f"Wrote {len(gdf_buildings)} building features")
        pbar.update(1)

        # Roads layer
        if roads_list:
            gdf_roads = gpd.GeoDataFrame(
                roads_list,
                crs=crs,
            )
            gdf_roads.to_file(output_path, layer="roads", driver="GPKG")
            logger.info(f"Wrote {len(gdf_roads)} road features")
        pbar.update(1)

        # Water bodies layer
        if water_list:
            gdf_water = gpd.GeoDataFrame(
                water_list,
                crs=crs,
            )
            gdf_water.to_file(output_path, layer="water_bodies", driver="GPKG")
            logger.info(f"Wrote {len(gdf_water)} water features")
        pbar.update(1)

        # Vegetation layer
        if vegetation_list:
            gdf_vegetation = gpd.GeoDataFrame(
                vegetation_list,
                crs=crs,
            )
            gdf_vegetation.to_file(output_path, layer="vegetation", driver="GPKG")
            logger.info(f"Wrote {len(gdf_vegetation)} vegetation features")
        pbar.update(1)

    logger.info(f"GeoPackage saved: {output_path}")

    return {
        "buildings_count": len(buildings_list),
        "roads_count": len(roads_list),
        "water_count": len(water_list),
        "vegetation_count": len(vegetation_list),
        "output_path": str(output_path),
        "crs": str(crs),
    }


def _extract_centerline(
    polygon: Polygon,
    simplify_tolerance: float = 0.3,
) -> LineString:
    """
    Extract centerline (skeleton) from a polygon using morphological operations.
    Approximated using polygon medial axis.

    Args:
        polygon: Input polygon
        simplify_tolerance: Simplification tolerance

    Returns:
        LineString representing approximate centerline, or None if extraction fails
    """
    try:
        # Use polygon medial axis (skeleton)
        # This is approximated by buffering inward and then extracting the spine
        exterior = polygon.exterior
        if len(exterior.coords) < 3:
            return None

        # Create approximate centerline by buffering and intersecting
        # For a more robust approach, use scipy.ndimage.distance_transform_edt
        # but for now use a simple approach
        inset_buffer = polygon.buffer(-simplify_tolerance * 2)

        if inset_buffer.is_empty:
            # Polygon is too small, use centroid line as fallback
            return LineString([(polygon.centroid.x, polygon.centroid.y)])

        # Approximate centerline as line from centroid along major axis
        centroid = polygon.centroid
        bounds = polygon.bounds
        cx, cy = centroid.x, centroid.y
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        if width > height:
            # Horizontal orientation
            x1, y1 = cx - width / 2, cy
            x2, y2 = cx + width / 2, cy
        else:
            # Vertical orientation
            x1, y1 = cx, cy - height / 2
            x2, y2 = cx, cy + height / 2

        centerline = LineString([(x1, y1), (x2, y2)])
        return centerline.intersection(polygon)

    except Exception as e:
        logger.debug(f"Error extracting centerline: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Polygonize segmentation masks and create GeoPackage feature layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_to_gpkg.py --mask segmentation.tif --output features.gpkg --village_id VILL001
  python mask_to_gpkg.py --mask mask.tif --output out.gpkg --village_id VILL001 \\
    --simplify_tolerance 0.5 --min_area 1.0
        """,
    )

    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to segmentation mask GeoTIFF",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output GeoPackage",
    )
    parser.add_argument(
        "--village_id",
        type=str,
        required=True,
        help="Village identifier",
    )
    parser.add_argument(
        "--simplify_tolerance",
        type=float,
        default=0.3,
        help="Simplification tolerance in meters (default: 0.3)",
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=2.0,
        help="Minimum polygon area in m² (default: 2.0)",
    )

    args = parser.parse_args()

    try:
        result = mask_to_gpkg(
            mask_path=args.mask,
            output_path=args.output,
            village_id=args.village_id,
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

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
