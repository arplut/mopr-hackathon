#!/usr/bin/env python3
"""
solar_potential.py: Estimate solar potential for RCC roof buildings.

This script reads a GeoPackage buildings layer and a DSM raster, then estimates
solar potential for each RCC roof building based on unshaded area (cells within
0.5m of maximum elevation in the polygon footprint).

Solar potential is classified as:
  - high: unshaded_area > 30 m²
  - medium: 15-30 m²
  - low: < 15 m²

Author: MoPR Hackathon Team
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def solar_potential(
    gpkg_path: str,
    dsm_path: str,
    elevation_threshold: float = 0.5,
    high_threshold: float = 30.0,
    medium_threshold: float = 15.0,
) -> dict:
    """
    Estimate solar potential for RCC roof buildings.

    Args:
        gpkg_path: Path to GeoPackage with buildings layer
        dsm_path: Path to DSM raster (meters)
        elevation_threshold: Threshold for detecting obstructions (default 0.5m)
        high_threshold: Minimum area for high potential (default 30 m²)
        medium_threshold: Minimum area for medium potential (default 15 m²)

    Returns:
        Dictionary with solar potential statistics

    Raises:
        FileNotFoundError: If files do not exist
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    dsm_path = Path(dsm_path)
    if not dsm_path.exists():
        raise FileNotFoundError(f"DSM file not found: {dsm_path}")

    logger.info(f"Reading GeoPackage: {gpkg_path}")
    gdf_buildings = gpd.read_file(gpkg_path, layer="buildings")

    # Filter for RCC roofs
    rcc_buildings = gdf_buildings[gdf_buildings["roof_type"] == "RCC_roof"].copy()
    logger.info(f"Found {len(rcc_buildings)} RCC roof buildings")

    if len(rcc_buildings) == 0:
        logger.warning("No RCC roof buildings found")
        return {
            "num_rcc_buildings": 0,
            "high_potential": 0,
            "medium_potential": 0,
            "low_potential": 0,
            "output_path": str(gpkg_path),
        }

    # Open DSM
    logger.info(f"Reading DSM: {dsm_path}")
    with rasterio.open(dsm_path) as dsm_src:
        dsm_transform = dsm_src.transform
        dsm_crs = dsm_src.crs

        # Check CRS match
        if dsm_crs != rcc_buildings.crs:
            logger.warning(
                f"CRS mismatch: GeoPackage uses {rcc_buildings.crs}, "
                f"DSM uses {dsm_crs}"
            )

        # Initialize solar potential column
        rcc_buildings["solar_potential"] = "unknown"
        rcc_buildings["unshaded_area_m2"] = 0.0

        high_count = 0
        medium_count = 0
        low_count = 0

        # Process each RCC building
        with tqdm(rcc_buildings.iterrows(), total=len(rcc_buildings), desc="Computing solar potential") as pbar:
            for idx, building in pbar:
                try:
                    geom = building.geometry

                    # Extract DSM pixels within building footprint
                    with rasterio.open(dsm_path) as src:
                        try:
                            cropped, cropped_transform = rio_mask(
                                src,
                                [geom],
                                crop=True,
                            )
                        except ValueError:
                            # Geometry outside raster bounds
                            rcc_buildings.at[idx, "solar_potential"] = "no_data"
                            pbar.update(1)
                            continue

                    # Get elevation values
                    dsm_values = cropped[0]
                    valid_mask = ~np.isnan(dsm_values) & (dsm_values > 0)

                    if not valid_mask.any():
                        rcc_buildings.at[idx, "solar_potential"] = "no_data"
                        pbar.update(1)
                        continue

                    # Find maximum elevation in building
                    max_elevation = np.max(dsm_values[valid_mask])

                    # Find unshaded area (within threshold of max)
                    unshaded_mask = (
                        np.abs(dsm_values - max_elevation) <= elevation_threshold
                    ) & valid_mask

                    # Calculate unshaded area
                    pixel_area = abs(dsm_transform.a * dsm_transform.e)
                    unshaded_area = np.sum(unshaded_mask) * pixel_area

                    # Classify solar potential
                    if unshaded_area > high_threshold:
                        solar_class = "high"
                        high_count += 1
                    elif unshaded_area > medium_threshold:
                        solar_class = "medium"
                        medium_count += 1
                    else:
                        solar_class = "low"
                        low_count += 1

                    rcc_buildings.at[idx, "solar_potential"] = solar_class
                    rcc_buildings.at[idx, "unshaded_area_m2"] = float(unshaded_area)

                except Exception as e:
                    logger.debug(f"Error processing building {idx}: {e}")
                    rcc_buildings.at[idx, "solar_potential"] = "error"
                    pbar.update(1)
                    continue

                pbar.update(0)

        # Save updated GeoPackage
        logger.info(f"Saving updated GeoPackage: {gpkg_path}")
        gdf_buildings.update(rcc_buildings)
        gdf_buildings.to_file(gpkg_path, layer="buildings", driver="GPKG")

        logger.info(f"Solar potential estimation complete")

        return {
            "num_rcc_buildings": len(rcc_buildings),
            "high_potential": high_count,
            "medium_potential": medium_count,
            "low_potential": low_count,
            "output_path": str(gpkg_path),
            "statistics": {
                "mean_unshaded_area": float(
                    rcc_buildings["unshaded_area_m2"].mean()
                ),
                "median_unshaded_area": float(
                    rcc_buildings["unshaded_area_m2"].median()
                ),
                "max_unshaded_area": float(
                    rcc_buildings["unshaded_area_m2"].max()
                ),
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate solar potential for RCC roof buildings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python solar_potential.py --gpkg features.gpkg --dsm dsm.tif
  python solar_potential.py --gpkg village_features.gpkg --dsm village_dsm.tif \\
    --elevation_threshold 0.5 --high_threshold 30 --medium_threshold 15
        """,
    )

    parser.add_argument(
        "--gpkg",
        type=str,
        required=True,
        help="Path to GeoPackage with buildings layer",
    )
    parser.add_argument(
        "--dsm",
        type=str,
        required=True,
        help="Path to DSM raster (meters)",
    )
    parser.add_argument(
        "--elevation_threshold",
        type=float,
        default=0.5,
        help="Elevation threshold for obstruction detection (default: 0.5m)",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=30.0,
        help="Minimum area for high solar potential (default: 30 m²)",
    )
    parser.add_argument(
        "--medium_threshold",
        type=float,
        default=15.0,
        help="Minimum area for medium solar potential (default: 15 m²)",
    )

    args = parser.parse_args()

    try:
        result = solar_potential(
            gpkg_path=args.gpkg,
            dsm_path=args.dsm,
            elevation_threshold=args.elevation_threshold,
            high_threshold=args.high_threshold,
            medium_threshold=args.medium_threshold,
        )

        print(f"\n{'='*70}")
        print(f"SOLAR POTENTIAL SUMMARY")
        print(f"{'='*70}")
        print(f"RCC roof buildings:      {result['num_rcc_buildings']}")
        print(f"High potential (>30m²):  {result['high_potential']}")
        print(f"Medium potential (15-30m²): {result['medium_potential']}")
        print(f"Low potential (<15m²):   {result['low_potential']}")

        if result["statistics"]:
            print(f"\nUnshaded Area Statistics:")
            print(f"  Mean:   {result['statistics']['mean_unshaded_area']:.2f} m²")
            print(f"  Median: {result['statistics']['median_unshaded_area']:.2f} m²")
            print(f"  Max:    {result['statistics']['max_unshaded_area']:.2f} m²")

        print(f"\nOutput: {result['output_path']}")
        print(f"{'='*70}\n")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
