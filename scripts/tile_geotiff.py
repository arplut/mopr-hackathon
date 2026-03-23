#!/usr/bin/env python3
"""
tile_geotiff.py: Tile large GeoTIFF orthophotos into 512×512 patches with 64px overlap.

This script reads a large GeoTIFF file (typically 2-5cm/pixel drone orthophotography)
and tiles it into smaller 512×512 patches with 64-pixel overlap for model inference.
Tiles with >80% nodata are skipped. Each tile preserves CRS and geotransform metadata.

Author: MoPR Hackathon Team
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def tile_geotiff(
    input_path: str,
    output_dir: str,
    village_id: str,
    tile_size: int = 512,
    overlap: int = 64,
    nodata_threshold: float = 0.8,
) -> tuple[int, int]:
    """
    Tile a large GeoTIFF into smaller patches.

    Args:
        input_path: Path to input GeoTIFF file
        output_dir: Directory to save output tiles
        village_id: Village identifier for tile naming
        tile_size: Size of output tiles in pixels (default 512)
        overlap: Overlap between tiles in pixels (default 64)
        nodata_threshold: Fraction of nodata pixels to skip tile (default 0.8)

    Returns:
        Tuple of (total_tiles_generated, tiles_skipped)

    Raises:
        FileNotFoundError: If input file does not exist
        rasterio.errors.RasterioIOError: If file cannot be read as GeoTIFF
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stride = tile_size - overlap

    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        crs = src.crs
        transform = src.transform
        profile = src.profile.copy()

        # Calculate number of tiles
        n_rows = (height - tile_size) // stride + 1
        n_cols = (width - tile_size) // stride + 1

        logger.info(
            f"Input GeoTIFF: {input_path.name} "
            f"({width}×{height} pixels, {src.count} bands)"
        )
        logger.info(
            f"Tiling parameters: tile_size={tile_size}, overlap={overlap}, "
            f"stride={stride}"
        )
        logger.info(
            f"Expected tile grid: {n_cols} columns × {n_rows} rows = "
            f"{n_cols * n_rows} tiles"
        )

        tiles_generated = 0
        tiles_skipped = 0

        with tqdm(
            total=n_rows * n_cols,
            desc="Tiling progress",
            unit="tile",
        ) as pbar:
            for row in range(n_rows):
                for col in range(n_cols):
                    # Calculate window
                    row_start = row * stride
                    col_start = col * stride

                    window = Window(col_start, row_start, tile_size, tile_size)

                    # Read tile data
                    try:
                        tile_data = src.read(window=window)
                    except Exception as e:
                        logger.warning(
                            f"Failed to read tile ({row}, {col}): {e}"
                        )
                        pbar.update(1)
                        continue

                    # Check for nodata
                    nodata_count = np.sum(
                        (tile_data == 0) | np.isnan(tile_data)
                    )
                    nodata_fraction = nodata_count / tile_data.size

                    if nodata_fraction > nodata_threshold:
                        tiles_skipped += 1
                        pbar.update(1)
                        continue

                    # Prepare output profile
                    out_transform = rasterio.transform.from_bounds(
                        *rasterio.transform.xy(
                            transform,
                            col_start,
                            row_start,
                            offset="ul",
                        ),
                        *rasterio.transform.xy(
                            transform,
                            col_start + tile_size,
                            row_start + tile_size,
                            offset="ul",
                        ),
                        tile_size,
                        tile_size,
                    )

                    output_profile = {
                        **profile,
                        "height": tile_size,
                        "width": tile_size,
                        "transform": out_transform,
                        "crs": crs,
                    }

                    # Write tile
                    output_filename = (
                        output_dir / f"{village_id}_{row:04d}_{col:04d}.tif"
                    )

                    with rasterio.open(output_filename, "w", **output_profile) as dst:
                        dst.write(tile_data)

                    tiles_generated += 1
                    pbar.update(1)

    logger.info(f"Tiling complete.")
    print(f"\n{'='*60}")
    print(f"TILING SUMMARY")
    print(f"{'='*60}")
    print(f"Input file:       {input_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Tiles generated:  {tiles_generated}")
    print(f"Tiles skipped:    {tiles_skipped} (>80% nodata)")
    print(f"Total processed:  {tiles_generated + tiles_skipped}")
    print(f"{'='*60}\n")

    return tiles_generated, tiles_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Tile a large GeoTIFF into 512×512 patches with 64px overlap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tile_geotiff.py --input ortho.tif --output_dir tiles/VILL001 --village_id VILL001
  python tile_geotiff.py --input ortho.tif --output_dir tiles/ --village_id VILL001 \\
    --tile_size 512 --overlap 64
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input GeoTIFF file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for tiles",
    )
    parser.add_argument(
        "--village_id",
        type=str,
        required=True,
        help="Village identifier for tile naming",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tile size in pixels (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap between tiles in pixels (default: 64)",
    )
    parser.add_argument(
        "--nodata_threshold",
        type=float,
        default=0.8,
        help="Skip tiles with >this fraction of nodata (default: 0.8)",
    )

    args = parser.parse_args()

    try:
        tile_geotiff(
            input_path=args.input,
            output_dir=args.output_dir,
            village_id=args.village_id,
            tile_size=args.tile_size,
            overlap=args.overlap,
            nodata_threshold=args.nodata_threshold,
        )
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
