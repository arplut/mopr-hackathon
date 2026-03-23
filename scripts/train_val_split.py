#!/usr/bin/env python3
"""
train_val_split.py: Create train/validation split by village for semantic segmentation.

This script splits villages into train and validation sets (not individual tiles).
It outputs train_tiles.txt and val_tiles.txt with full paths to all tiles in each set,
and optionally computes class distribution separately for train and val sets.

Author: MoPR Hackathon Team
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "background",
    1: "RCC_roof",
    2: "tile_roof",
    3: "tin_roof",
    4: "thatched_roof",
    5: "road_pucca",
    6: "road_kaccha",
    7: "water_body",
    8: "vegetation",
}


def train_val_split(
    tile_dir: str,
    output_dir: str,
    mask_dir: str = None,
    val_villages: list = None,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """
    Create train/validation split by village.

    Args:
        tile_dir: Directory containing village subdirectories with tiles
        output_dir: Directory to save train_tiles.txt and val_tiles.txt
        mask_dir: Optional directory with mask files to compute class distribution
        val_villages: List of village IDs to use as validation set (overrides val_ratio)
        val_ratio: Fraction of villages to use as validation (default 0.2)
        random_seed: Random seed for reproducibility (default 42)

    Returns:
        Dictionary with split statistics

    Raises:
        FileNotFoundError: If tile_dir does not exist
    """
    tile_dir = Path(tile_dir)
    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mask_dir is not None:
        mask_dir = Path(mask_dir)
        if not mask_dir.exists():
            logger.warning(f"Mask directory not found: {mask_dir}. Skipping stats.")
            mask_dir = None

    # Discover villages
    village_dirs = sorted([d for d in tile_dir.iterdir() if d.is_dir()])

    if not village_dirs:
        raise FileNotFoundError(f"No village subdirectories found in {tile_dir}")

    logger.info(f"Found {len(village_dirs)} villages")

    # Extract village IDs from directory names
    all_village_ids = []
    village_to_dir = {}

    for village_dir in village_dirs:
        village_id = village_dir.name
        village_to_dir[village_id] = village_dir
        all_village_ids.append(village_id)

    # Determine validation villages
    if val_villages is not None:
        val_village_ids = set(val_villages)
        # Verify all specified villages exist
        missing = val_village_ids - set(all_village_ids)
        if missing:
            logger.warning(
                f"Specified validation villages not found: {missing}. "
                f"Available: {all_village_ids}"
            )
    else:
        # Random split
        random.seed(random_seed)
        num_val = max(1, int(len(all_village_ids) * val_ratio))
        val_village_ids = set(random.sample(all_village_ids, num_val))

    train_village_ids = set(all_village_ids) - val_village_ids

    logger.info(
        f"Split: {len(train_village_ids)} train, {len(val_village_ids)} validation"
    )
    logger.info(f"Train villages: {sorted(train_village_ids)}")
    logger.info(f"Val villages:   {sorted(val_village_ids)}")

    # Collect tiles
    train_tiles = []
    val_tiles = []

    for village_id in all_village_ids:
        village_dir = village_to_dir[village_id]
        tiles = sorted(village_dir.glob("*.tif"))

        if not tiles:
            logger.warning(f"No tiles found for village {village_id}")
            continue

        tile_paths = [str(t.resolve()) for t in tiles]

        if village_id in train_village_ids:
            train_tiles.extend(tile_paths)
        else:
            val_tiles.extend(tile_paths)

    logger.info(f"Total train tiles: {len(train_tiles)}")
    logger.info(f"Total val tiles: {len(val_tiles)}")

    # Write tile lists
    train_tiles_path = output_dir / "train_tiles.txt"
    val_tiles_path = output_dir / "val_tiles.txt"

    with open(train_tiles_path, "w") as f:
        for tile_path in sorted(train_tiles):
            f.write(tile_path + "\n")

    with open(val_tiles_path, "w") as f:
        for tile_path in sorted(val_tiles):
            f.write(tile_path + "\n")

    logger.info(f"Saved train tiles to {train_tiles_path}")
    logger.info(f"Saved val tiles to {val_tiles_path}")

    # Compute class distribution if mask_dir provided
    train_stats = None
    val_stats = None

    if mask_dir is not None:
        logger.info("Computing class distribution for train and val sets...")

        train_stats = _compute_split_stats(train_tiles, mask_dir, "train")
        val_stats = _compute_split_stats(val_tiles, mask_dir, "val")

    return {
        "num_train_villages": len(train_village_ids),
        "num_val_villages": len(val_village_ids),
        "num_train_tiles": len(train_tiles),
        "num_val_tiles": len(val_tiles),
        "train_village_ids": sorted(train_village_ids),
        "val_village_ids": sorted(val_village_ids),
        "train_stats": train_stats,
        "val_stats": val_stats,
    }


def _compute_split_stats(
    tile_paths: list,
    mask_dir: Path,
    split_name: str,
    num_classes: int = 9,
) -> dict:
    """Compute class distribution for a set of tiles."""
    class_counts = np.zeros(num_classes, dtype=np.int64)

    with tqdm(tile_paths, desc=f"Processing {split_name} masks", unit="tile") as pbar:
        for tile_path in pbar:
            tile_path = Path(tile_path)
            tile_stem = tile_path.stem

            # Try to find corresponding mask
            mask_path = mask_dir / f"{tile_stem}.tif"
            if not mask_path.exists():
                # Try alternative naming
                parts = tile_stem.split("_")
                if len(parts) >= 2:
                    village_id = "_".join(parts[:-2])
                    mask_path = mask_dir / f"{village_id}_mask.tif"

            if not mask_path.exists():
                pbar.update(1)
                continue

            try:
                with rasterio.open(mask_path) as src:
                    mask_data = src.read(1)
                    for class_id in range(num_classes):
                        class_counts[class_id] += np.sum(mask_data == class_id)
            except Exception as e:
                logger.warning(f"Error reading mask {mask_path.name}: {e}")

            pbar.update(0)

    total_pixels = np.sum(class_counts)
    if total_pixels == 0:
        return None

    class_percentages = 100.0 * class_counts / total_pixels

    return {
        "class_counts": class_counts.tolist(),
        "class_percentages": class_percentages.tolist(),
        "total_pixels": int(total_pixels),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create train/validation split by village.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_val_split.py --tile_dir tiles/ --output_dir splits/
  python train_val_split.py --tile_dir tiles/ --mask_dir masks/ --output_dir splits/ \\
    --val_villages VILL009 VILL010
  python train_val_split.py --tile_dir tiles/ --output_dir splits/ --val_ratio 0.15
        """,
    )

    parser.add_argument(
        "--tile_dir",
        type=str,
        required=True,
        help="Directory containing village subdirectories with tiles",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for train_tiles.txt and val_tiles.txt",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Optional directory with mask files for class distribution stats",
    )
    parser.add_argument(
        "--val_villages",
        type=str,
        nargs="+",
        default=None,
        help="Village IDs to use as validation set (overrides val_ratio)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of villages for validation (default: 0.2)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    try:
        result = train_val_split(
            tile_dir=args.tile_dir,
            output_dir=args.output_dir,
            mask_dir=args.mask_dir,
            val_villages=args.val_villages,
            val_ratio=args.val_ratio,
            random_seed=args.random_seed,
        )

        # Print summary
        print(f"\n{'='*70}")
        print(f"TRAIN/VALIDATION SPLIT SUMMARY")
        print(f"{'='*70}")
        print(f"Train villages: {result['num_train_villages']}")
        print(f"Val villages:   {result['num_val_villages']}")
        print(f"Train tiles:    {result['num_train_tiles']}")
        print(f"Val tiles:      {result['num_val_tiles']}")
        print(f"\nTrain village IDs: {result['train_village_ids']}")
        print(f"Val village IDs:   {result['val_village_ids']}")
        print(f"{'='*70}\n")

        # Print class distribution if available
        if result["train_stats"] is not None:
            print(f"{'TRAIN SET CLASS DISTRIBUTION':<40}")
            print(f"{'-'*70}")
            print(
                f"{'Class':<20} {'Pixels':>15} {'Percentage':>12}"
            )
            print(f"{'-'*70}")
            for i in range(len(result["train_stats"]["class_counts"])):
                print(
                    f"{CLASS_NAMES.get(i, f'class_{i}'):<20} "
                    f"{result['train_stats']['class_counts'][i]:>15,} "
                    f"{result['train_stats']['class_percentages'][i]:>11.2f}%"
                )
            print()

        if result["val_stats"] is not None:
            print(f"{'VALIDATION SET CLASS DISTRIBUTION':<40}")
            print(f"{'-'*70}")
            print(
                f"{'Class':<20} {'Pixels':>15} {'Percentage':>12}"
            )
            print(f"{'-'*70}")
            for i in range(len(result["val_stats"]["class_counts"])):
                print(
                    f"{CLASS_NAMES.get(i, f'class_{i}'):<20} "
                    f"{result['val_stats']['class_counts'][i]:>15,} "
                    f"{result['val_stats']['class_percentages'][i]:>11.2f}%"
                )
            print()

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
