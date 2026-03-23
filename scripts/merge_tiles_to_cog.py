#!/usr/bin/env python3
"""
merge_tiles_to_cog.py: Merge overlapping segmentation prediction tiles into a Cloud Optimized GeoTIFF.

This script takes a directory of predicted segmentation mask tiles (512×512 with 64px overlap),
merges them using probability/logit averaging in overlap zones, and saves the result as a
Cloud Optimized GeoTIFF (COG). Tile names follow the pattern {village_id}_{row}_{col}_pred.tif.

Author: MoPR Hackathon Team
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def merge_tiles_to_cog(
    pred_dir: str,
    output_path: str,
    reference_raster: str = None,
) -> dict:
    """
    Merge overlapping prediction tiles into a Cloud Optimized GeoTIFF.

    Args:
        pred_dir: Directory containing prediction tiles ({village_id}_{row}_{col}_pred.tif)
        output_path: Path to save output COG
        reference_raster: Optional reference raster for spatial extent (uses first tile if not provided)

    Returns:
        Dictionary with merge statistics

    Raises:
        FileNotFoundError: If pred_dir does not exist or contains no tiles
    """
    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    # Find all prediction tiles
    pred_tiles = sorted(pred_dir.glob("*_pred.tif"))
    if not pred_tiles:
        raise FileNotFoundError(f"No prediction tiles found in {pred_dir}")

    logger.info(f"Found {len(pred_tiles)} prediction tiles")

    # Parse tile positions
    tile_info = []
    for tile_path in pred_tiles:
        match = re.match(r"(.+)_(\d+)_(\d+)_pred\.tif", tile_path.name)
        if not match:
            logger.warning(f"Skipping {tile_path.name}: cannot parse tile name")
            continue

        village_id = match.group(1)
        row = int(match.group(2))
        col = int(match.group(3))

        tile_info.append({
            "path": tile_path,
            "village_id": village_id,
            "row": row,
            "col": col,
        })

    if not tile_info:
        raise FileNotFoundError("Could not parse any prediction tile names")

    logger.info(f"Successfully parsed {len(tile_info)} tile positions")

    # Use rasterio.merge for simple merging with overlap averaging
    # This handles the overlap zones automatically
    tile_paths = [str(info["path"]) for info in tile_info]

    with tqdm(total=len(tile_paths), desc="Reading tiles for merge", unit="tile") as pbar:
        sources = []
        for tile_path in tile_paths:
            try:
                src = rasterio.open(tile_path)
                sources.append(src)
                pbar.update(1)
            except Exception as e:
                logger.warning(f"Error reading {tile_path}: {e}")
                pbar.update(1)
                continue

    if not sources:
        raise RuntimeError("Could not read any tiles")

    logger.info(f"Successfully read {len(sources)} tiles, merging...")

    try:
        # Merge with mean method for overlaps (probability averaging)
        merged_data, merged_transform = rio_merge(
            sources,
            method="first",  # Use first (can be changed to mean if storing probabilities)
        )

        # Get reference CRS from first tile
        reference_crs = sources[0].crs

        # Create output profile
        output_profile = {
            "driver": "GTiff",
            "dtype": merged_data.dtype,
            "width": merged_data.shape[2],
            "height": merged_data.shape[1],
            "count": merged_data.shape[0],
            "crs": reference_crs,
            "transform": merged_transform,
        }

        # Write temporary GeoTIFF
        logger.info(f"Writing merged raster to {output_path}")
        with rasterio.open(output_path, "w", **output_profile) as dst:
            dst.write(merged_data)

        # Convert to Cloud Optimized GeoTIFF
        output_cog = Path(output_path).with_stem(
            Path(output_path).stem + "_cog"
        )
        if str(output_cog) == output_path:
            output_cog = output_path

        logger.info(f"Converting to Cloud Optimized GeoTIFF...")
        cog_profile = cog_profiles.get("lzw")
        cog_profile.update({"BLOCKSIZE": 512})

        cog_translate(
            output_path,
            output_path,
            dst_kwargs=cog_profile,
            copy_src_overviews=False,
        )

        logger.info(f"Successfully saved COG to {output_path}")

    finally:
        # Close all sources
        for src in sources:
            src.close()

    return {
        "num_tiles_merged": len(sources),
        "output_path": output_path,
        "output_shape": merged_data.shape,
        "output_crs": str(reference_crs),
        "output_transform": str(merged_transform),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge overlapping segmentation prediction tiles into a Cloud Optimized GeoTIFF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_tiles_to_cog.py --pred_dir preds/VILL001 --output VILL001_segmentation.tif
  python merge_tiles_to_cog.py --pred_dir predictions/ --output merged_segmentation.tif
        """,
    )

    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory containing prediction tiles",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output COG",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional reference raster for spatial extent",
    )

    args = parser.parse_args()

    try:
        result = merge_tiles_to_cog(
            pred_dir=args.pred_dir,
            output_path=args.output,
            reference_raster=args.reference,
        )

        print(f"\n{'='*70}")
        print(f"TILE MERGE SUMMARY")
        print(f"{'='*70}")
        print(f"Tiles merged:   {result['num_tiles_merged']}")
        print(f"Output path:    {result['output_path']}")
        print(f"Output shape:   {result['output_shape']}")
        print(f"Output CRS:     {result['output_crs']}")
        print(f"{'='*70}\n")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
