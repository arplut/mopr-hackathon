#!/usr/bin/env python3
"""
dataset_stats.py: Compute class distribution statistics from segmentation mask GeoTIFFs.

This script reads all single-band uint8 GeoTIFF files from a directory,
counts pixel frequency per class (0-8), and generates:
  - CSV file with class names, pixel counts, and percentages
  - Matplotlib bar chart visualization
  - Class weights for loss function weighting (inverse frequency, normalized to sum to 9)

Author: MoPR Hackathon Team
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Class definitions for 9-class segmentation task
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


def compute_dataset_stats(
    mask_dir: str,
    output_dir: str,
    num_classes: int = 9,
) -> dict:
    """
    Compute class distribution statistics from segmentation mask GeoTIFFs.

    Args:
        mask_dir: Directory containing single-band uint8 mask GeoTIFFs
        output_dir: Directory to save CSV and chart outputs
        num_classes: Number of classes (default 9)

    Returns:
        Dictionary with class statistics

    Raises:
        FileNotFoundError: If mask_dir does not exist or contains no .tif files
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .tif files
    mask_files = sorted(mask_dir.glob("**/*.tif"))
    if not mask_files:
        raise FileNotFoundError(f"No .tif files found in {mask_dir}")

    logger.info(f"Found {len(mask_files)} mask files in {mask_dir}")

    # Initialize class counters
    class_counts = np.zeros(num_classes, dtype=np.int64)

    # Process each mask file
    with tqdm(mask_files, desc="Processing masks", unit="file") as pbar:
        for mask_file in pbar:
            try:
                with rasterio.open(mask_file) as src:
                    if src.count != 1:
                        logger.warning(
                            f"Skipping {mask_file.name}: expected 1 band, "
                            f"got {src.count}"
                        )
                        pbar.update(1)
                        continue

                    mask_data = src.read(1)

                    # Count classes efficiently with bincount
                    flat = mask_data.ravel()
                    # Clip to valid range to avoid out-of-bounds
                    flat = flat[flat < num_classes]
                    counts = np.bincount(flat, minlength=num_classes)
                    class_counts += counts[:num_classes]

            except Exception as e:
                logger.warning(f"Error reading {mask_file.name}: {e}")
                pbar.update(1)
                continue

            pbar.update(0)

    # Calculate statistics
    total_pixels = np.sum(class_counts)
    class_percentages = 100.0 * class_counts / total_pixels

    # Calculate class weights (inverse frequency, normalized to sum to num_classes)
    class_weights = np.zeros(num_classes)
    for class_id in range(num_classes):
        if class_counts[class_id] > 0:
            class_weights[class_id] = 1.0 / class_counts[class_id]
    class_weights = (class_weights / class_weights.sum()) * num_classes

    # Create statistics dictionary
    stats = {
        "class_id": list(range(num_classes)),
        "class_name": [CLASS_NAMES.get(i, f"class_{i}") for i in range(num_classes)],
        "pixel_count": class_counts.tolist(),
        "percentage": class_percentages.tolist(),
        "weight": class_weights.tolist(),
    }

    # Save CSV
    df = pd.DataFrame(stats)
    csv_path = output_dir / "class_distribution.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved class distribution to {csv_path}")

    # Create and save visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    bars = ax.bar(
        df["class_name"],
        df["percentage"],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of Pixels (%)", fontsize=12, fontweight="bold")
    ax.set_title("Class Distribution in Segmentation Masks", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    chart_path = output_dir / "class_distribution.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved class distribution chart to {chart_path}")
    plt.close(fig)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute class distribution statistics from segmentation masks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_stats.py --mask_dir masks/ --output_dir stats/
  python dataset_stats.py --mask_dir /data/masks --output_dir /data/stats
        """,
    )

    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory containing single-band uint8 mask GeoTIFFs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for CSV and chart",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=9,
        help="Number of classes (default: 9)",
    )

    args = parser.parse_args()

    try:
        stats = compute_dataset_stats(
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
        )

        # Print class weights
        print(f"\n{'='*70}")
        print(f"CLASS DISTRIBUTION AND WEIGHTS")
        print(f"{'='*70}")
        print(
            f"{'Class':<20} {'Pixels':>15} {'Percentage':>12} {'Weight':>12}"
        )
        print(f"{'-'*70}")
        for i in range(len(stats["class_id"])):
            print(
                f"{stats['class_name'][i]:<20} {stats['pixel_count'][i]:>15,} "
                f"{stats['percentage'][i]:>11.2f}% {stats['weight'][i]:>11.4f}"
            )
        print(f"{'='*70}\n")

        print("Class weights (for loss function weighting):")
        weights_str = "[" + ", ".join(f"{w:.4f}" for w in stats["weight"]) + "]"
        print(weights_str)
        print()

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
