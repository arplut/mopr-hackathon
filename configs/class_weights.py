"""
Class definitions, colors, and weights for MoPR semantic segmentation.
Includes utilities for computing weights from dataset statistics.
"""

import csv
from typing import Dict, List, Tuple
import numpy as np
import torch


# Class names (9 classes)
CLASS_NAMES = [
    "background",      # 0
    "RCC_roof",        # 1
    "tile_roof",       # 2
    "tin_roof",        # 3
    "thatched_roof",   # 4
    "road_pucca",      # 5
    "road_kaccha",     # 6
    "water_body",      # 7
    "vegetation",      # 8
]

# Class color palette (RGB) for visualization
# Using a distinct, perceptually diverse palette suitable for segmentation
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "background": (0, 0, 0),           # Black
    "RCC_roof": (255, 0, 0),           # Red
    "tile_roof": (255, 128, 0),        # Orange
    "tin_roof": (255, 255, 0),         # Yellow
    "thatched_roof": (165, 42, 42),    # Brown
    "road_pucca": (128, 128, 128),     # Gray
    "road_kaccha": (192, 192, 192),    # Light Gray
    "water_body": (0, 0, 255),         # Blue
    "vegetation": (0, 128, 0),         # Green
}

# Default class weights (inverse frequency estimates)
# Based on typical Indian village drone imagery:
# - background: very common, weight down
# - roof types: relatively common, weight moderately
# - thatched_roof: less common but important target, weight up
# - road_kaccha: less common, weight up
# - roads_pucca: medium, weight moderately
# - water_body: rare, weight down
# - vegetation: medium, weight down
DEFAULT_CLASS_WEIGHTS = torch.tensor([
    1.0,    # background
    3.0,    # RCC_roof
    3.0,    # tile_roof
    4.0,    # tin_roof
    5.0,    # thatched_roof (most important, rare)
    2.0,    # road_pucca
    3.0,    # road_kaccha
    1.5,    # water_body
    1.0,    # vegetation
], dtype=torch.float32)


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Get RGB color for a class by ID."""
    return CLASS_COLORS[CLASS_NAMES[class_id]]


def create_colormap() -> np.ndarray:
    """
    Create a 256x3 colormap array for visualization.
    Returns numpy array of shape (256, 3) with RGB values for each class ID.
    Classes 0-8 get their assigned colors; remaining indices are black.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for class_id, class_name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS[class_name]
        colormap[class_id] = color
    return colormap


def compute_weights_from_stats(csv_path: str, method: str = "inverse_frequency") -> torch.Tensor:
    """
    Compute class weights from class distribution statistics CSV.

    CSV format: class_name, pixel_count

    Args:
        csv_path: Path to class_distribution.csv from dataset_stats.py
        method: "inverse_frequency" (default) or "effective_number"

    Returns:
        torch.Tensor of shape (num_classes,) with normalized weights
    """
    # Initialize counts
    class_counts = {name: 0 for name in CLASS_NAMES}

    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row['class_name'].strip()
            pixel_count = int(row['pixel_count'])
            if class_name in class_counts:
                class_counts[class_name] = pixel_count

    # Convert to counts array in order of CLASS_NAMES
    counts = np.array([class_counts[name] for name in CLASS_NAMES], dtype=np.float32)

    if method == "inverse_frequency":
        # Weight = 1 / frequency
        # Avoid division by zero
        weights = np.where(counts > 0, 1.0 / counts, 0.0)
    elif method == "effective_number":
        # Effective number of samples: (1 - beta^N) / (1 - beta)
        # where beta is typically 0.9999 and N is class count
        beta = 0.9999
        weights = (1.0 - beta) / np.where(counts > 0, 1.0 - np.power(beta, counts), 1e-6)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize to sum to num_classes (so average weight is 1.0)
    weights = weights / weights.sum() * len(CLASS_NAMES)

    # Convert to tensor
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Example: print class info
    print("Class Definitions for MoPR Semantic Segmentation")
    print("=" * 60)
    print(f"{'ID':<3} {'Class Name':<20} {'RGB Color':<30} {'Default Weight':<15}")
    print("-" * 60)
    for idx, (name, color) in enumerate(zip(CLASS_NAMES, [CLASS_COLORS[n] for n in CLASS_NAMES])):
        weight = DEFAULT_CLASS_WEIGHTS[idx].item()
        print(f"{idx:<3} {name:<20} {str(color):<30} {weight:<15.2f}")
    print()
    print(f"Total classes: {len(CLASS_NAMES)}")
    print(f"Sum of default weights: {DEFAULT_CLASS_WEIGHTS.sum():.2f}")
