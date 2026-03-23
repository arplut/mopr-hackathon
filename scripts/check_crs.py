#!/usr/bin/env python3
"""
check_crs.py: Verify CRS and geotransform consistency across GeoTIFF files.

This script reads all GeoTIFF files from a directory, extracts CRS and transform
metadata, and identifies any mismatches. Output includes a formatted table
printed to console and a CSV report file.

Author: MoPR Hackathon Team
"""

import argparse
import csv
import logging
from pathlib import Path

import rasterio
from rasterio.io import MemoryFile
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_crs_consistency(
    data_dir: str,
    output_path: str = "crs_report.csv",
) -> dict:
    """
    Check CRS and geotransform consistency across GeoTIFF files.

    Args:
        data_dir: Directory containing GeoTIFF files
        output_path: Path to save CSV report

    Returns:
        Dictionary with CRS consistency information

    Raises:
        FileNotFoundError: If data_dir does not exist or contains no .tif files
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all .tif files
    tif_files = sorted(data_dir.glob("**/*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {data_dir}")

    logger.info(f"Found {len(tif_files)} GeoTIFF files in {data_dir}")

    # Collect metadata
    crs_data = []
    crs_set = set()
    epsg_codes = {}

    with tqdm(tif_files, desc="Checking CRS", unit="file") as pbar:
        for tif_file in pbar:
            try:
                with rasterio.open(tif_file) as src:
                    crs = src.crs
                    transform = src.transform
                    bounds = src.bounds

                    # Extract EPSG code if available
                    epsg_code = None
                    if crs is not None and crs.is_epsg_code:
                        epsg_code = crs.to_epsg()

                    # Get pixel size (assuming square pixels)
                    pixel_x = abs(transform.a)
                    pixel_y = abs(transform.e)

                    crs_str = str(crs) if crs is not None else "None"
                    crs_set.add(crs_str)

                    row = {
                        "filename": tif_file.name,
                        "path": str(tif_file),
                        "crs": crs_str,
                        "epsg": epsg_code or "N/A",
                        "pixel_x": f"{pixel_x:.6f}",
                        "pixel_y": f"{pixel_y:.6f}",
                        "bounds_left": f"{bounds.left:.2f}",
                        "bounds_bottom": f"{bounds.bottom:.2f}",
                        "bounds_right": f"{bounds.right:.2f}",
                        "bounds_top": f"{bounds.top:.2f}",
                    }

                    crs_data.append(row)
                    if epsg_code:
                        epsg_codes[epsg_code] = epsg_codes.get(epsg_code, 0) + 1

            except Exception as e:
                logger.warning(f"Error reading {tif_file.name}: {e}")
                pbar.update(1)
                continue

            pbar.update(0)

    # Check for consistency
    has_mismatch = len(crs_set) > 1
    if has_mismatch:
        logger.warning(f"CRS MISMATCH DETECTED: {len(crs_set)} different CRS found!")
        for crs_str in crs_set:
            logger.warning(f"  - {crs_str}")
    else:
        logger.info(f"All files share consistent CRS: {list(crs_set)[0]}")

    # Save CSV report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "filename",
            "path",
            "crs",
            "epsg",
            "pixel_x",
            "pixel_y",
            "bounds_left",
            "bounds_bottom",
            "bounds_right",
            "bounds_top",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(crs_data)

    logger.info(f"Saved CRS report to {output_path}")

    return {
        "num_files": len(crs_data),
        "crs_count": len(crs_set),
        "has_mismatch": has_mismatch,
        "crs_set": crs_set,
        "epsg_codes": epsg_codes,
        "data": crs_data,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check CRS and geotransform consistency across GeoTIFF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_crs.py --data_dir geotiffs/ --output crs_report.csv
  python check_crs.py --data_dir /data/tiles
        """,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing GeoTIFF files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="crs_report.csv",
        help="Path to save CSV report (default: crs_report.csv)",
    )

    args = parser.parse_args()

    try:
        result = check_crs_consistency(
            data_dir=args.data_dir,
            output_path=args.output,
        )

        # Print formatted table
        print(f"\n{'='*130}")
        print(f"CRS AND GEOTRANSFORM REPORT")
        print(f"{'='*130}")

        # Print summary
        print(
            f"\nTotal files: {result['num_files']} | "
            f"Unique CRS: {result['crs_count']}"
        )

        if result["has_mismatch"]:
            print("\n⚠️  WARNING: CRS MISMATCH DETECTED!")
            for crs_str in result["crs_set"]:
                count = sum(1 for d in result["data"] if d["crs"] == crs_str)
                print(f"   {crs_str}: {count} files")
        else:
            print(f"✓ All files use consistent CRS: {list(result['crs_set'])[0]}")

        # Print table
        print(f"\n{'-'*130}")
        print(
            f"{'Filename':<40} {'EPSG':<8} {'Pixel X':<12} {'Pixel Y':<12} "
            f"{'CRS':<40}"
        )
        print(f"{'-'*130}")

        for row in result["data"]:
            print(
                f"{row['filename']:<40} {str(row['epsg']):<8} "
                f"{row['pixel_x']:<12} {row['pixel_y']:<12} {row['crs']:<40}"
            )

        print(f"{'='*130}\n")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
