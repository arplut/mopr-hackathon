#!/usr/bin/env python3
"""
batch_inference.py: Run SegFormer-B2 inference on tiled orthophotos with test-time augmentation.

This script loads a trained SegFormer-B2 model and runs sliding-window inference on
village tile directories. It applies 4-rotation + horizontal flip TTA and merges tiles
into Cloud Optimized GeoTIFFs per village.

Expected input: 4-channel tiles (RGB + DSM)
Output: Segmentation masks (uint8, values 0-8)

Author: MoPR Hackathon Team
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSemanticSegmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TileDataset(Dataset):
    """Dataset for loading GeoTIFF tiles."""

    def __init__(self, tile_paths: list, patch_size: int = 512):
        """
        Initialize TileDataset.

        Args:
            tile_paths: List of paths to tile GeoTIFF files
            patch_size: Expected patch size (default 512)
        """
        self.tile_paths = tile_paths
        self.patch_size = patch_size

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]

        with rasterio.open(tile_path) as src:
            # Read 4 channels: RGB + DSM
            if src.count >= 4:
                data = src.read([1, 2, 3, 4])  # RGB + DSM
            else:
                logger.warning(
                    f"Tile {tile_path} has {src.count} bands, expected 4. Padding with zeros."
                )
                data = src.read()
                # Pad to 4 channels if needed
                if data.shape[0] < 4:
                    padding = np.zeros(
                        (4 - data.shape[0], data.shape[1], data.shape[2]),
                        dtype=data.dtype,
                    )
                    data = np.vstack([data, padding])

            # Normalize to [0, 1]
            data = data.astype(np.float32) / 255.0

            # Get metadata
            crs = src.crs
            transform = src.transform

        # Convert to tensor
        tensor = torch.from_numpy(data)

        return {
            "image": tensor,
            "path": str(tile_path),
            "crs": crs,
            "transform": transform,
        }


def apply_tta(model, image_tensor: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Apply test-time augmentation: 4 rotations × 2 flips.

    Args:
        model: Segmentation model
        image_tensor: Input tensor [B, C, H, W]
        device: Device to run on

    Returns:
        Averaged prediction tensor
    """
    augmented_preds = []

    # Original
    augmented_preds.append(_infer_and_aggregate(model, image_tensor, device))

    # 90° rotation
    rotated_90 = torch.rot90(image_tensor, k=1, dims=(2, 3))
    augmented_preds.append(torch.rot90(_infer_and_aggregate(model, rotated_90, device), k=-1, dims=(2, 3)))

    # 180° rotation
    rotated_180 = torch.rot90(image_tensor, k=2, dims=(2, 3))
    augmented_preds.append(torch.rot90(_infer_and_aggregate(model, rotated_180, device), k=-2, dims=(2, 3)))

    # 270° rotation
    rotated_270 = torch.rot90(image_tensor, k=3, dims=(2, 3))
    augmented_preds.append(torch.rot90(_infer_and_aggregate(model, rotated_270, device), k=-3, dims=(2, 3)))

    # Horizontal flip
    flipped = torch.flip(image_tensor, dims=(3,))
    augmented_preds.append(torch.flip(_infer_and_aggregate(model, flipped, device), dims=(3,)))

    # Stack and average
    stacked = torch.stack(augmented_preds, dim=0)
    averaged = torch.mean(stacked, dim=0)

    return averaged


def _infer_and_aggregate(model, image_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Run inference and return argmax predictions."""
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device)).logits
        # logits shape: [1, num_classes, H, W]
        preds = torch.argmax(logits, dim=1)  # [1, H, W]
    return preds.squeeze(0).cpu()


def batch_inference(
    checkpoint_path: str,
    tile_dir: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 4,
    use_tta: bool = True,
) -> dict:
    """
    Run batch inference on village tile directories.

    Args:
        checkpoint_path: Path to model checkpoint (.pth)
        tile_dir: Directory containing village subdirectories with tiles
        output_dir: Output directory for predictions
        device: Device to run on (default cuda)
        batch_size: Batch size for inference (default 4)
        use_tta: Enable test-time augmentation (default True)

    Returns:
        Dictionary with inference statistics

    Raises:
        FileNotFoundError: If checkpoint or tile_dir not found
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tile_dir = Path(tile_dir)
    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    try:
        # Load SegFormer-B2 from HuggingFace
        model = AutoModelForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=9,
        )
        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on {device}")

    # Find village directories
    village_dirs = sorted([d for d in tile_dir.iterdir() if d.is_dir()])
    if not village_dirs:
        raise FileNotFoundError(f"No village subdirectories found in {tile_dir}")

    logger.info(f"Found {len(village_dirs)} villages")

    total_tiles = 0
    total_predictions = 0

    # Process each village
    with tqdm(village_dirs, desc="Processing villages", unit="village") as village_pbar:
        for village_dir in village_dirs:
            village_id = village_dir.name

            # Find tiles in this village
            tile_paths = sorted(village_dir.glob("*.tif"))
            if not tile_paths:
                logger.warning(f"No tiles found in {village_id}")
                village_pbar.update(1)
                continue

            logger.info(f"Processing {village_id}: {len(tile_paths)} tiles")
            total_tiles += len(tile_paths)

            # Create output directory for village
            village_output_dir = output_dir / village_id
            village_output_dir.mkdir(parents=True, exist_ok=True)

            # Create dataset and dataloader
            dataset = TileDataset(tile_paths)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Run inference
            with tqdm(
                dataloader,
                desc=f"Inferring {village_id}",
                unit="batch",
                leave=False,
            ) as batch_pbar:
                for batch in batch_pbar:
                    images = batch["image"].to(device)
                    paths = batch["path"]
                    crs_list = batch["crs"]
                    transform_list = batch["transform"]

                    with torch.no_grad():
                        # Apply TTA if enabled
                        if use_tta:
                            predictions = []
                            for i in range(len(images)):
                                img = images[i : i + 1]
                                aug_pred = apply_tta(model, img, device)
                                predictions.append(aug_pred.numpy())
                            predictions = np.array(predictions)
                        else:
                            # Standard inference
                            logits = model(images).logits
                            predictions = (
                                torch.argmax(logits, dim=1).cpu().numpy()
                            )

                    # Save predictions
                    for i in range(len(predictions)):
                        tile_path = Path(paths[i])
                        pred_mask = predictions[i].astype(np.uint8)
                        crs = crs_list[i]
                        transform = transform_list[i]

                        # Create output filename
                        output_filename = (
                            village_output_dir / f"{tile_path.stem}_pred.tif"
                        )

                        # Save as GeoTIFF
                        profile = {
                            "driver": "GTiff",
                            "height": pred_mask.shape[0],
                            "width": pred_mask.shape[1],
                            "count": 1,
                            "dtype": np.uint8,
                            "crs": crs,
                            "transform": transform,
                        }

                        with rasterio.open(output_filename, "w", **profile) as dst:
                            dst.write(pred_mask, 1)

                        total_predictions += 1

                    batch_pbar.update(1)

            # Try to merge tiles for this village
            try:
                from merge_tiles_to_cog import merge_tiles_to_cog

                merged_output = village_output_dir / f"{village_id}_segmentation.tif"
                merge_tiles_to_cog(str(village_output_dir), str(merged_output))
            except Exception as e:
                logger.warning(f"Could not merge tiles for {village_id}: {e}")

            village_pbar.update(1)

    logger.info(f"Inference complete")

    return {
        "num_villages": len(village_dirs),
        "total_tiles_processed": total_tiles,
        "total_predictions_saved": total_predictions,
        "output_dir": str(output_dir),
        "device": device,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SegFormer-B2 batch inference with test-time augmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_inference.py --checkpoint model_best.pth --tile_dir tiles/ --output_dir outputs/
  python batch_inference.py --checkpoint model.pth --tile_dir tiles/ --output_dir preds/ \\
    --device cuda --batch_size 8 --use_tta
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
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
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        default=True,
        help="Enable test-time augmentation (default: enabled)",
    )

    args = parser.parse_args()

    try:
        result = batch_inference(
            checkpoint_path=args.checkpoint,
            tile_dir=args.tile_dir,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            use_tta=args.use_tta,
        )

        print(f"\n{'='*70}")
        print(f"BATCH INFERENCE SUMMARY")
        print(f"{'='*70}")
        print(f"Villages processed:      {result['num_villages']}")
        print(f"Total tiles processed:   {result['total_tiles_processed']}")
        print(f"Total predictions saved: {result['total_predictions_saved']}")
        print(f"Output directory:        {result['output_dir']}")
        print(f"Device:                  {result['device']}")
        print(f"{'='*70}\n")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
