#!/bin/bash
###############################################################################
# validate_outputs.sh: Validate Cloud Optimized GeoTIFF and GeoPackage files
#
# This script validates all .tif and .gpkg files in a directory:
#   - GeoTIFFs: Checks COG validity using 'rio cogeo validate'
#   - GeoPackages: Verifies geometry validity, required attributes, and CRS
#
# Usage: ./validate_outputs.sh <output_directory>
#
# Author: MoPR Hackathon Team
###############################################################################

set -euo pipefail

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: ./validate_outputs.sh <output_directory>"
    echo ""
    echo "Example:"
    echo "  ./validate_outputs.sh outputs/"
    exit 1
fi

OUTPUT_DIR="$1"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

echo "=========================================="
echo "OUTPUT VALIDATION REPORT"
echo "=========================================="
echo ""

# Initialize counters
tif_pass=0
tif_fail=0
tif_total=0

# Validate GeoTIFFs
echo "Validating Cloud Optimized GeoTIFFs (.tif)..."
echo "-------------------------------------------"

while IFS= read -r tif_file; do
    tif_total=$((tif_total + 1))
    filename=$(basename "$tif_file")

    if rio cogeo validate "$tif_file" > /dev/null 2>&1; then
        echo "  PASS: $filename"
        tif_pass=$((tif_pass + 1))
    else
        echo "  FAIL: $filename (COG validation failed)"
        tif_fail=$((tif_fail + 1))
    fi
done < <(find "$OUTPUT_DIR" -type f -name "*_segmentation.tif" 2>/dev/null)

if [ "$tif_total" -eq 0 ]; then
    echo "  No *_segmentation.tif files found."
fi

echo ""
echo "GeoTIFF Summary: $tif_pass passed, $tif_fail failed out of $tif_total"
echo ""

# Validate GeoPackages using Python (pass OUTPUT_DIR as argument)
echo "Validating GeoPackages (.gpkg)..."
echo "-------------------------------------------"

python3 - "$OUTPUT_DIR" << 'PYTHON_SCRIPT'
import sys
from pathlib import Path

try:
    import fiona
    import geopandas as gpd
except ImportError as e:
    print(f"  ERROR: Required package not installed: {e}")
    sys.exit(1)

output_dir = sys.argv[1]

gpkg_pass = 0
gpkg_fail = 0

for gpkg_file in sorted(Path(output_dir).glob("**/*.gpkg")):
    filename = gpkg_file.name

    try:
        layers = fiona.listlayers(str(gpkg_file))

        valid = True
        issues = []

        for layer in layers:
            try:
                gdf = gpd.read_file(str(gpkg_file), layer=layer)

                # Check for valid geometries
                invalid_geoms = ~gdf.geometry.is_valid
                if invalid_geoms.any():
                    invalid_count = invalid_geoms.sum()
                    issues.append(f"Layer '{layer}': {invalid_count} invalid geometries")
                    valid = False

                # Check for CRS
                if gdf.crs is None:
                    issues.append(f"Layer '{layer}': Missing CRS")
                    valid = False

                # Check for empty geometries
                empty_geoms = gdf.geometry.is_empty
                if empty_geoms.any():
                    issues.append(f"Layer '{layer}': {empty_geoms.sum()} empty geometries")

                # Check required attributes by layer type
                if layer == "buildings":
                    required_attrs = ["roof_type", "area_m2", "confidence", "village_id"]
                    missing = [a for a in required_attrs if a not in gdf.columns]
                    if missing:
                        issues.append(f"Layer '{layer}': Missing attributes {missing}")
                        valid = False

                elif layer == "roads":
                    required_attrs = ["road_type", "length_m", "village_id"]
                    missing = [a for a in required_attrs if a not in gdf.columns]
                    if missing:
                        issues.append(f"Layer '{layer}': Missing attributes {missing}")
                        valid = False

                elif layer in ("water_bodies", "vegetation"):
                    required_attrs = ["area_m2", "village_id"]
                    missing = [a for a in required_attrs if a not in gdf.columns]
                    if missing:
                        issues.append(f"Layer '{layer}': Missing attributes {missing}")
                        valid = False

            except Exception as e:
                issues.append(f"Layer '{layer}': {str(e)}")
                valid = False

        if valid:
            print(f"  PASS: {filename} ({len(layers)} layers: {', '.join(layers)})")
            gpkg_pass += 1
        else:
            print(f"  FAIL: {filename}")
            for issue in issues:
                print(f"    - {issue}")
            gpkg_fail += 1

    except Exception as e:
        print(f"  FAIL: {filename} ({str(e)})")
        gpkg_fail += 1

gpkg_total = gpkg_pass + gpkg_fail
if gpkg_total == 0:
    print("  No .gpkg files found.")

print(f"")
print(f"GeoPackage Summary: {gpkg_pass} passed, {gpkg_fail} failed out of {gpkg_total}")

# Exit with code for bash to capture
sys.exit(1 if gpkg_fail > 0 else 0)
PYTHON_SCRIPT

gpkg_exit=$?

echo ""
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo "GeoTIFFs:    $tif_pass passed, $tif_fail failed"
echo "=========================================="
echo ""

# Exit with error code if any validation failed
if [ $tif_fail -gt 0 ] || [ $gpkg_exit -ne 0 ]; then
    echo "WARNING: Some files failed validation."
    exit 1
else
    echo "All files passed validation."
    exit 0
fi
