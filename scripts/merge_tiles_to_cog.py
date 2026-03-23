#!/usr/bin/env python3
"""
merge_tiles_to_cog.py: Merge overlapping segmentation prediction tiles into a Cloud Optimized GeoTIFF.

This script takes a directory of predicted segmentation probability tiles (512×512 with 64px overlap),
merges them using probability averaging in overlap zones, and saves the result as a
Cloud Optimized GeoTIFF (COG).

Supports two tile naming patterns:
  - {village_id}_{row}_{col}_pred.tif  (argmax masks, uint8)
  - {village_id}_{row}_{col}_prob.tif  (softmax probabilities, float32 x num_classes)

When probability tiles are available, overlap zones are averaged before argmax.
When only argmax tiles exist, rasterio.merge with method='first' is used (no averaging possible).

Author: MoPR Hackathon Team
"""

import argparse
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Regex patterns for supported tile naming conventions
TILE_PATTERNS = [
    re.compile(r"(.+?)_(\d{4})_(\d{4})_pred\.tif$"),   # village_0001_0002_pred.tif
    re.compile(r"(.+?)_(\d+)_(\d+)_pred\.tif$"),        # village_1_2_pred.tif
    re.compile(r"(.+?)_(\d{4})_(\d{4})_prob\.tif$"),    # village_0001_0002_prob.tif
]


def _parse_tile_name(filename: str) -> tuple:
    """Parse village_id, row, col from tile filename. Returns None if no match."""
    for pattern in TILE_PATTERNS:
        match = pattern.match(filename)
        if match:
            return match.group(1), int(match.group(2)), int(match.group(3))
    return None


def merge_tiles_to_cog(
    pred_dir: str,
    output_path: str,
    num_classes: int = 9,
) -> dict:
    """
    Merge overlapping prediction tiles into a Cloud Optimized GeoTIFF.

    Looks for probability tiles (*_prob.tif) first. If found, performs proper
    probability averaging in overlap zones before argmax. Falls back to
    argmax tiles (*_pred.tif) with rasterio.merge.

    Args:
        pred_dir: Directory containing prediction tiles
        output_path: Path to save output COG
        num_classes: Number of segmentation classes (default 9)

    Returns:
        Dictionary with merge statistics
    """
    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    # Prefer probability tiles for proper overlap averaging
    prob_tiles = sorted(pred_dir.glob("*_prob.tif"))
    pred_tiles = sorted(pred_dir.glob("*_pred.tif"))

    if prob_tiles:
        logger.info(f"Found {len(prob_tiles)} probability tiles — will average overlaps")
        return _merge_probability_tiles(prob_tiles, output_path, num_classes)
    elif pred_tiles:
        logger.warning(
            f"Found {len(pred_tiles)} argmax tiles only — overlap averaging not possible. "
            "For better results, save probability tiles during inference."
        )
        return _merge_argmax_tiles(pred_tiles, output_path)
    else:
        raise FileNotFoundError(f"No *_pred.tif or *_prob.tif tiles found in {pred_dir}")


def _merge_probability_tiles(
    prob_tiles: list,
    output_path: str,
    num_classes: int,
) -> dict:
    """
    Merge multi-band probability tiles with averaging in overlap zones.
    Each tile has shape (num_classes, H, W) with softmax probabilities.
    """
    # Parse tile positions and collect metadata
    tile_info = []
    for tile_path in prob_tiles:
        parsed = _parse_tile_name(tile_path.name)
        if parsed is None:
            logger.warning(f"Skipping {tile_path.name}: cannot parse tile name")
            continue
        tile_info.append({"path": tile_path, "village_id": parsed[0], "row": parsed[1], "col": parsed[2]})

    if not tile_info:
        raise FileNotFoundError("Could not parse any probability tile names")

    # Use rasterio.merge with method='sum' on probability tiles,
    # plus a count raster to divide by for averaging
    sources = []
    count_sources = []

    try:
        for info in tqdm(tile_info, desc="Reading probability tiles"):
            src = rasterio.open(info["path"])
            sources.append(src)

        # Merge probabilities using sum
        logger.info("Merging probability tiles (sum)...")
        sum_data, sum_transform = rio_merge(sources, method="sum")

        # Create count rasters (1 where data exists, 0 where nodata)
        # Re-read as binary masks
        for src in sources:
            src.seek(0)  # Reset read position

        # Build count by merging binary masks
        count_data, _ = rio_merge(
            sources,
            method="sum",
            dtype="float32",
        )
        # count_data has same shape but we need per-pixel count
        # Since sum of probabilities at a pixel = N (number of overlapping tiles)
        # when each tile contributes probabilities summing to ~1.0,
        # we can estimate count from the sum of band 0..N probabilities.
        # More robust: create explicit count rasters
        # For simplicity, use the sum across classes as the count
        # (each tile's probabilities sum to 1.0 per pixel)
        pixel_count = sum_data.sum(axis=0, keepdims=True)  # (1, H, W)
        pixel_count = np.maximum(pixel_count, 1.0)  # avoid division by zero

        # Average probabilities
        avg_probs = sum_data / pixel_count  # (C, H, W)

        # Argmax to get final class labels
        merged_labels = avg_probs.argmax(axis=0).astype(np.uint8)  # (H, W)

        # Get reference metadata
        reference_crs = sources[0].crs

    finally:
        for src in sources:
            src.close()

    # Write merged result and convert to COG
    _write_cog(merged_labels, sum_transform, reference_crs, output_path)

    return {
        "num_tiles_merged": len(sources),
        "merge_method": "probability_averaging",
        "output_path": str(output_path),
        "output_shape": merged_labels.shape,
        "output_crs": str(reference_crs),
    }


def _merge_argmax_tiles(pred_tiles: list, output_path: str) -> dict:
    """Merge argmax prediction tiles using rasterio.merge (no overlap averaging)."""
    sources = []
    try:
        for tile_path in tqdm(pred_tiles, desc="Reading argmax tiles"):
            sources.append(rasterio.open(tile_path))

        if not sources:
            raise RuntimeError("Could not read any tiles")

        # method='first' is the only sensible option for categorical data
        merged_data, merged_transform = rio_merge(sources, method="first")
        reference_crs = sources[0].crs
    finally:
        for src in sources:
            src.close()

    merged_labels = merged_data[0].astype(np.uint8)  # (H, W)

    _write_cog(merged_labels, merged_transform, reference_crs, output_path)

    return {
        "num_tiles_merged": len(sources),
        "merge_method": "first_tile",
        "output_path": str(output_path),
        "output_shape": merged_labels.shape,
        "output_crs": str(reference_crs),
    }


def _write_cog(
    data: np.ndarray,
    transform,
    crs,
    output_path: str,
):
    """Write a 2D uint8 array as a Cloud Optimized GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file first, then COG-translate to final path
    # (cog_translate cannot safely read and write the same file)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
    }

    logger.info(f"Writing temporary raster ({data.shape[1]}x{data.shape[0]})...")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data, 1)

    logger.info("Converting to Cloud Optimized GeoTIFF...")
    cog_profile = cog_profiles.get("lzw")
    cog_profile.update({"BLOCKSIZE": 512})

    cog_translate(
        tmp_path,
        str(output_path),
        dst_kwargs=cog_profile,
        copy_src_overviews=False,
    )

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)
    logger.info(f"Saved COG: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge overlapping segmentation prediction tiles into a Cloud Optimized GeoTIFF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_tiles_to_cog.py --pred_dir preds/VILL001 --output VILL001_segmentation.tif
  python merge_tiles_to_cog.py --pred_dir predictions/ --output merged.tif --num_classes 9
        """,
    )

    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing prediction tiles")
    parser.add_argument("--output", type=str, required=True, help="Path to save output COG")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of segmentation classes (default: 9)")

    args = parser.parse_args()

    try:
        result = merge_tiles_to_cog(
            pred_dir=args.pred_dir,
            output_path=args.output,
            num_classes=args.num_classes,
        )

        print(f"\n{'='*70}")
        print(f"TILE MERGE SUMMARY")
        print(f"{'='*70}")
        print(f"Tiles merged:   {result['num_tiles_merged']}")
        print(f"Merge method:   {result['merge_method']}")
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
