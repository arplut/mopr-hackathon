#!/usr/bin/env python3
"""
village_statistics.py: Generate summary statistics from GeoPackage feature files.

This script reads GeoPackage files (one per village) and generates a CSV summary
with per-village statistics including building counts by type, road lengths,
water area, and vegetation metrics.

Author: MoPR Hackathon Team
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def village_statistics(
    gpkg_dir: str,
    output_path: str,
) -> dict:
    """
    Generate summary statistics from GeoPackage files.

    Args:
        gpkg_dir: Directory containing GeoPackage files (one per village)
        output_path: Path to save CSV summary

    Returns:
        Dictionary with statistics

    Raises:
        FileNotFoundError: If gpkg_dir does not exist or contains no .gpkg files
    """
    gpkg_dir = Path(gpkg_dir)
    if not gpkg_dir.exists():
        raise FileNotFoundError(f"GeoPackage directory not found: {gpkg_dir}")

    # Find all .gpkg files
    gpkg_files = sorted(gpkg_dir.glob("**/*.gpkg"))
    if not gpkg_files:
        raise FileNotFoundError(f"No .gpkg files found in {gpkg_dir}")

    logger.info(f"Found {len(gpkg_files)} GeoPackage files")

    stats_list = []

    with tqdm(gpkg_files, desc="Processing GeoPackages", unit="file") as pbar:
        for gpkg_file in pbar:
            try:
                village_id = gpkg_file.stem

                stats = _extract_village_stats(gpkg_file, village_id)
                stats_list.append(stats)

            except Exception as e:
                logger.warning(f"Error processing {gpkg_file.name}: {e}")
                pbar.update(1)
                continue

            pbar.update(0)

    if not stats_list:
        raise RuntimeError("Could not extract statistics from any GeoPackage files")

    # Create DataFrame
    df = pd.DataFrame(stats_list)

    # Sort by village ID
    df = df.sort_values("village_id").reset_index(drop=True)

    # Save CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved statistics to {output_path}")

    return {"num_villages": len(stats_list), "output_path": str(output_path), "data": df}


def _extract_village_stats(gpkg_path: Path, village_id: str) -> dict:
    """Extract statistics from a single GeoPackage file."""
    stats = {
        "village_id": village_id,
        "total_buildings": 0,
        "rcc_count": 0,
        "tile_count": 0,
        "tin_count": 0,
        "thatched_count": 0,
        "other_count": 0,
        "dominant_roof_type": None,
        "total_roof_area_m2": 0.0,
        "total_road_length_km": 0.0,
        "pucca_road_km": 0.0,
        "kaccha_road_km": 0.0,
        "total_water_area_m2": 0.0,
        "total_vegetation_area_m2": 0.0,
    }

    # Read buildings layer
    try:
        gdf_buildings = gpd.read_file(gpkg_path, layer="buildings")
        stats["total_buildings"] = len(gdf_buildings)
        stats["total_roof_area_m2"] = float(gdf_buildings["area_m2"].sum())

        if len(gdf_buildings) > 0:
            # Count by roof type
            roof_counts = gdf_buildings["roof_type"].value_counts()
            stats["rcc_count"] = int(roof_counts.get("RCC_roof", 0))
            stats["tile_count"] = int(roof_counts.get("tile_roof", 0))
            stats["tin_count"] = int(roof_counts.get("tin_roof", 0))
            stats["thatched_count"] = int(roof_counts.get("thatched_roof", 0))
            stats["other_count"] = len(gdf_buildings) - (
                stats["rcc_count"]
                + stats["tile_count"]
                + stats["tin_count"]
                + stats["thatched_count"]
            )

            # Dominant roof type
            if len(roof_counts) > 0:
                stats["dominant_roof_type"] = str(roof_counts.index[0])
    except Exception as e:
        logger.debug(f"Could not read buildings layer from {gpkg_path}: {e}")

    # Read roads layer
    try:
        gdf_roads = gpd.read_file(gpkg_path, layer="roads")
        stats["total_road_length_km"] = float(gdf_roads["length_m"].sum() / 1000.0)

        if len(gdf_roads) > 0:
            road_counts = gdf_roads["road_type"].value_counts()
            stats["pucca_road_km"] = float(
                gdf_roads[gdf_roads["road_type"] == "pucca"]["length_m"].sum() / 1000.0
            )
            stats["kaccha_road_km"] = float(
                gdf_roads[gdf_roads["road_type"] == "kaccha"]["length_m"].sum() / 1000.0
            )
    except Exception as e:
        logger.debug(f"Could not read roads layer from {gpkg_path}: {e}")

    # Read water bodies layer
    try:
        gdf_water = gpd.read_file(gpkg_path, layer="water_bodies")
        stats["total_water_area_m2"] = float(gdf_water["area_m2"].sum())
    except Exception as e:
        logger.debug(f"Could not read water_bodies layer from {gpkg_path}: {e}")

    # Read vegetation layer
    try:
        gdf_vegetation = gpd.read_file(gpkg_path, layer="vegetation")
        stats["total_vegetation_area_m2"] = float(gdf_vegetation["area_m2"].sum())
    except Exception as e:
        logger.debug(f"Could not read vegetation layer from {gpkg_path}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary statistics from GeoPackage feature files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python village_statistics.py --gpkg_dir outputs/ --output village_statistics.csv
  python village_statistics.py --gpkg_dir /data/features --output stats.csv
        """,
    )

    parser.add_argument(
        "--gpkg_dir",
        type=str,
        required=True,
        help="Directory containing GeoPackage files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save CSV summary",
    )

    args = parser.parse_args()

    try:
        result = village_statistics(
            gpkg_dir=args.gpkg_dir,
            output_path=args.output,
        )

        df = result["data"]

        # Print formatted table
        print(f"\n{'='*140}")
        print(f"VILLAGE STATISTICS SUMMARY")
        print(f"{'='*140}\n")

        print(f"{'Village':<12} {'Buildings':<12} {'RCC':<8} {'Tile':<8} {'Tin':<8} {'Thatch':<8} {'Roof Area':<12} {'Dominant':<15}")
        print(f"{'-'*140}")

        for _, row in df.iterrows():
            print(
                f"{row['village_id']:<12} {row['total_buildings']:<12.0f} "
                f"{row['rcc_count']:<8.0f} {row['tile_count']:<8.0f} "
                f"{row['tin_count']:<8.0f} {row['thatched_count']:<8.0f} "
                f"{row['total_roof_area_m2']:<12.0f} {str(row['dominant_roof_type']):<15}"
            )

        print(f"\n{'Village':<12} {'Road (km)':<12} {'Pucca':<12} {'Kaccha':<12} {'Water (m²)':<15} {'Vegetation (m²)':<15}")
        print(f"{'-'*140}")

        for _, row in df.iterrows():
            print(
                f"{row['village_id']:<12} {row['total_road_length_km']:<12.2f} "
                f"{row['pucca_road_km']:<12.2f} {row['kaccha_road_km']:<12.2f} "
                f"{row['total_water_area_m2']:<15.0f} {row['total_vegetation_area_m2']:<15.0f}"
            )

        print(f"{'='*140}\n")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
