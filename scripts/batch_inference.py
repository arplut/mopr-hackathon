#!/usr/bin/env python3
"""
batch_inference.py: Run SegFormer-B2 inference on tiled orthophotos with test-time augmentation.

This script loads a trained SegFormer-B2 model and runs inference on village tile directories.
It applies 4-rotation + horizontal flip TTA by averaging LOGITS (not argmax labels), then
saves both argmax masks and optionally probability/confidence rasters.

Expected input structure:
  tile_dir/
    village_001/
      village_001_0000_0000.tif   (4-channel: RGB + DSM)
      village_001_0000_0001.tif
      ...
    village_002/
      ...

Output structure:
  output_dir/
    village_001/
      village_001_0000_0000_pred.tif  (uint8, argmax class)
      village_001_0000_0000_prob.tif  (float32, num_classes bands — softmax)
      village_001_0000_0000_conf.tif  (float32, per-pixel max softmax)
    ...

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
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ImageNet normalization for RGB channels
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TileDataset(Dataset):
    """Dataset for loading 4-channel GeoTIFF tiles (RGB + DSM)."""

    def __init__(self, tile_paths: list):
        self.tile_paths = tile_paths

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]

        with rasterio.open(tile_path) as src:
            num_bands = src.count

            if num_bands >= 4:
                data = src.read([1, 2, 3, 4]).astype(np.float32)  # (4, H, W)
            elif num_bands == 3:
                rgb = src.read([1, 2, 3]).astype(np.float32)
                dsm_zeros = np.zeros((1, rgb.shape[1], rgb.shape[2]), dtype=np.float32)
                data = np.concatenate([rgb, dsm_zeros], axis=0)
                logger.debug(f"Tile {tile_path.name} has 3 bands, padding DSM with zeros")
            else:
                raise ValueError(f"Tile {tile_path.name} has {num_bands} bands, expected >= 3")

            # Normalize RGB channels (0-255 -> ImageNet normalized)
            for c in range(3):
                data[c] = (data[c] / 255.0 - RGB_MEAN[c]) / RGB_STD[c]

            # Normalize DSM separately: per-tile z-score
            dsm = data[3]
            dsm_valid = dsm[dsm != 0]  # Exclude nodata (0)
            if len(dsm_valid) > 0:
                dsm_mean = dsm_valid.mean()
                dsm_std = dsm_valid.std()
                if dsm_std > 1e-6:
                    data[3] = (dsm - dsm_mean) / dsm_std
                else:
                    data[3] = dsm - dsm_mean
            # else: leave DSM as zeros

            # Store metadata as serializable types (not rasterio objects)
            crs_str = str(src.crs) if src.crs else ""
            transform_list = list(src.transform)[:6]  # affine coefficients

        return {
            "image": torch.from_numpy(data),
            "path": str(tile_path),
            "crs_str": crs_str,
            "transform_coeffs": torch.tensor(transform_list, dtype=torch.float64),
        }


def _reconstruct_transform(coeffs):
    """Reconstruct rasterio Affine from 6 coefficients."""
    from rasterio.transform import Affine
    return Affine(*coeffs.tolist())


def apply_tta_logits(
    model,
    image_batch: torch.Tensor,
    device: str,
    num_classes: int = 9,
) -> torch.Tensor:
    """
    Apply test-time augmentation by averaging LOGITS across augmentations.

    Augmentations: original + 3 rotations (90/180/270) + horizontal flip = 5 views.
    Returns averaged logits of shape (B, num_classes, H, W).
    """
    augmentations = [
        (lambda x: x, lambda x: x),                                          # original
        (lambda x: torch.rot90(x, 1, [2, 3]), lambda x: torch.rot90(x, -1, [2, 3])),  # 90°
        (lambda x: torch.rot90(x, 2, [2, 3]), lambda x: torch.rot90(x, -2, [2, 3])),  # 180°
        (lambda x: torch.rot90(x, 3, [2, 3]), lambda x: torch.rot90(x, -3, [2, 3])),  # 270°
        (lambda x: torch.flip(x, [3]),         lambda x: torch.flip(x, [3])),          # h-flip
    ]

    accumulated_logits = None

    for forward_aug, inverse_aug in augmentations:
        augmented_input = forward_aug(image_batch)

        with torch.no_grad():
            outputs = model(augmented_input.to(device))
            logits = outputs.logits  # (B, C, H', W') — SegFormer may output smaller

            # Upsample logits to input resolution if needed
            if logits.shape[2:] != image_batch.shape[2:]:
                logits = F.interpolate(
                    logits,
                    size=image_batch.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

        # Reverse the augmentation on the logits
        reversed_logits = inverse_aug(logits.cpu())

        if accumulated_logits is None:
            accumulated_logits = reversed_logits
        else:
            accumulated_logits += reversed_logits

    # Average
    averaged_logits = accumulated_logits / len(augmentations)
    return averaged_logits


def batch_inference(
    checkpoint_path: str,
    tile_dir: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 4,
    use_tta: bool = True,
    save_probabilities: bool = True,
    num_classes: int = 9,
) -> dict:
    """
    Run batch inference on village tile directories.

    Args:
        checkpoint_path: Path to model checkpoint (.pt or .pth)
        tile_dir: Directory containing village subdirectories with tiles
        output_dir: Output directory for predictions
        device: Device to run on (default cuda)
        batch_size: Batch size for inference (default 4)
        use_tta: Enable test-time augmentation (default True)
        save_probabilities: Save probability and confidence rasters (default True)
        num_classes: Number of segmentation classes (default 9)

    Returns:
        Dictionary with inference statistics
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
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # Adapt first conv layer for 4 channels (RGB + DSM)
    _adapt_model_to_4ch(model)

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}, TTA={'on' if use_tta else 'off'}")

    # Find village directories (or flat tile directory)
    village_dirs = sorted([d for d in tile_dir.iterdir() if d.is_dir()])
    if not village_dirs:
        # Flat directory of tiles — treat as single village
        village_dirs = [tile_dir]

    logger.info(f"Found {len(village_dirs)} village(s)")

    total_tiles = 0
    total_predictions = 0

    for village_dir in tqdm(village_dirs, desc="Villages", unit="village"):
        village_id = village_dir.name
        tile_paths = sorted(village_dir.glob("*.tif"))

        # Exclude any existing prediction files
        tile_paths = [p for p in tile_paths if "_pred" not in p.stem and "_prob" not in p.stem and "_conf" not in p.stem]

        if not tile_paths:
            logger.warning(f"No input tiles in {village_id}")
            continue

        logger.info(f"Processing {village_id}: {len(tile_paths)} tiles")
        total_tiles += len(tile_paths)

        village_output_dir = output_dir / village_id
        village_output_dir.mkdir(parents=True, exist_ok=True)

        dataset = TileDataset(tile_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues with rasterio
            pin_memory=True if device == "cuda" else False,
        )

        for batch in tqdm(dataloader, desc=f"  {village_id}", leave=False, unit="batch"):
            images = batch["image"]  # (B, 4, H, W)
            paths = batch["path"]
            crs_strs = batch["crs_str"]
            transform_coeffs = batch["transform_coeffs"]  # (B, 6)

            if use_tta:
                logits = apply_tta_logits(model, images, device, num_classes)
            else:
                with torch.no_grad():
                    outputs = model(images.to(device))
                    logits = outputs.logits.cpu()
                    if logits.shape[2:] != images.shape[2:]:
                        logits = F.interpolate(
                            logits,
                            size=images.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )

            # Convert logits to predictions
            probs = torch.softmax(logits, dim=1)          # (B, C, H, W)
            pred_classes = probs.argmax(dim=1).numpy().astype(np.uint8)  # (B, H, W)
            confidence = probs.max(dim=1).values.numpy().astype(np.float32)  # (B, H, W)
            probs_np = probs.numpy().astype(np.float32)   # (B, C, H, W)

            for i in range(len(paths)):
                tile_path = Path(paths[i])
                tile_stem = tile_path.stem
                crs_str = crs_strs[i]
                t_coeffs = transform_coeffs[i]
                tile_transform = _reconstruct_transform(t_coeffs)

                h, w = pred_classes[i].shape

                base_profile = {
                    "driver": "GTiff",
                    "height": h,
                    "width": w,
                    "crs": crs_str if crs_str else None,
                    "transform": tile_transform,
                }

                # Save argmax prediction
                pred_path = village_output_dir / f"{tile_stem}_pred.tif"
                with rasterio.open(pred_path, "w", **{**base_profile, "count": 1, "dtype": "uint8"}) as dst:
                    dst.write(pred_classes[i], 1)

                if save_probabilities:
                    # Save full probability raster (for overlap merging)
                    prob_path = village_output_dir / f"{tile_stem}_prob.tif"
                    with rasterio.open(prob_path, "w", **{**base_profile, "count": num_classes, "dtype": "float32"}) as dst:
                        for c in range(num_classes):
                            dst.write(probs_np[i, c], c + 1)

                    # Save confidence raster (for GPKG attribute)
                    conf_path = village_output_dir / f"{tile_stem}_conf.tif"
                    with rasterio.open(conf_path, "w", **{**base_profile, "count": 1, "dtype": "float32"}) as dst:
                        dst.write(confidence[i], 1)

                total_predictions += 1

    logger.info("Inference complete")

    return {
        "num_villages": len(village_dirs),
        "total_tiles_processed": total_tiles,
        "total_predictions_saved": total_predictions,
        "output_dir": str(output_dir),
        "device": device,
        "tta_enabled": use_tta,
        "probabilities_saved": save_probabilities,
    }


def _adapt_model_to_4ch(model):
    """Modify SegFormer first conv layer to accept 4 input channels."""
    original_conv = model.segformer.encoder.patch_embeddings[0].proj
    original_weight = original_conv.weight.data  # (out_ch, 3, kH, kW)

    new_conv = nn.Conv2d(
        4, original_weight.shape[0],
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )

    # Initialize: copy RGB weights, init DSM channel as mean of RGB
    new_weight = torch.zeros(
        original_weight.shape[0], 4,
        original_weight.shape[2], original_weight.shape[3],
        dtype=original_weight.dtype, device=original_weight.device,
    )
    new_weight[:, :3, :, :] = original_weight
    new_weight[:, 3:, :, :] = original_weight.mean(dim=1, keepdim=True)

    new_conv.weight = nn.Parameter(new_weight)
    if original_conv.bias is not None:
        new_conv.bias = original_conv.bias

    model.segformer.encoder.patch_embeddings[0].proj = new_conv


def main():
    parser = argparse.ArgumentParser(
        description="Run SegFormer-B2 batch inference with test-time augmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_inference.py --checkpoint best_model.pt --tile_dir tiles/ --output_dir preds/
  python batch_inference.py --checkpoint model.pt --tile_dir tiles/ --output_dir preds/ --no_tta
        """,
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tile_dir", type=str, required=True, help="Directory with village tile subdirs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for predictions")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--no_probabilities", action="store_true", help="Don't save probability rasters")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of classes (default: 9)")

    args = parser.parse_args()

    try:
        result = batch_inference(
            checkpoint_path=args.checkpoint,
            tile_dir=args.tile_dir,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            use_tta=not args.no_tta,
            save_probabilities=not args.no_probabilities,
            num_classes=args.num_classes,
        )

        print(f"\n{'='*70}")
        print(f"BATCH INFERENCE SUMMARY")
        print(f"{'='*70}")
        print(f"Villages processed:      {result['num_villages']}")
        print(f"Total tiles processed:   {result['total_tiles_processed']}")
        print(f"Total predictions saved: {result['total_predictions_saved']}")
        print(f"TTA enabled:             {result['tta_enabled']}")
        print(f"Probabilities saved:     {result['probabilities_saved']}")
        print(f"Output directory:        {result['output_dir']}")
        print(f"{'='*70}\n")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
