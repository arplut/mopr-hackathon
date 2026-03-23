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

set -e

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
gpkg_pass=0
gpkg_fail=0

# Validate GeoTIFFs
echo "Validating Cloud Optimized GeoTIFFs (.tif)..."
echo "-------------------------------------------"

while IFS= read -r tif_file; do
    filename=$(basename "$tif_file")

    if rio cogeo validate "$tif_file" > /dev/null 2>&1; then
        echo "✓ PASS: $filename"
        ((tif_pass++))
    else
        echo "✗ FAIL: $filename (COG validation failed)"
        ((tif_fail++))
    fi
done < <(find "$OUTPUT_DIR" -type f -name "*.tif" 2>/dev/null)

echo ""
echo "Validating GeoPackages (.gpkg)..."
echo "-------------------------------------------"

# Validate GeoPackages using Python
python3 << 'PYTHON_SCRIPT'
import sys
from pathlib import Path
import geopandas as gpd

output_dir = sys.argv[1] if len(sys.argv) > 1 else "."

gpkg_pass = 0
gpkg_fail = 0

for gpkg_file in sorted(Path(output_dir).glob("**/*.gpkg")):
    filename = gpkg_file.name

    try:
        # Try to read each layer
        layers = gpd.layers(str(gpkg_file))

        valid = True
        issues = []

        for layer in layers:
            try:
                gdf = gpd.read_file(str(gpkg_file), layer=layer)

                # Check for valid geometries
                if not gdf.geometry.is_valid.all():
                    invalid_count = (~gdf.geometry.is_valid).sum()
                    issues.append(f"Layer '{layer}': {invalid_count} invalid geometries")
                    valid = False

                # Check for CRS
                if gdf.crs is None:
                    issues.append(f"Layer '{layer}': Missing CRS")
                    valid = False

                # Check for required attributes based on layer type
                if layer == "buildings":
                    required_attrs = ["roof_type", "area_m2", "village_id"]
                    missing = [attr for attr in required_attrs if attr not in gdf.columns]
                    if missing:
                        issues.append(f"Layer '{layer}': Missing attributes {missing}")
                        valid = False

                elif layer == "roads":
                    required_attrs = ["road_type", "length_m", "village_id"]
                    missing = [attr for attr in required_attrs if attr not in gdf.columns]
                    if missing:
                        issues.append(f"Layer '{layer}': Missing attributes {missing}")
                        valid = False

                elif layer in ["water_bodies", "vegetation"]:
                    required_attrs = ["area_m2", "village_id"]
                    missing = [attr for attr in required_attrs if attr not in gdf.columns]
                    if missing:
                        issues.append(f"Layer '{layer}': Missing attributes {missing}")
                        valid = False

            except Exception as e:
                issues.append(f"Layer '{layer}': {str(e)}")
                valid = False

        if valid:
            print(f"✓ PASS: {filename}")
            gpkg_pass += 1
        else:
            print(f"✗ FAIL: {filename}")
            for issue in issues:
                print(f"    {issue}")
            gpkg_fail += 1

    except Exception as e:
        print(f"✗ FAIL: {filename} ({str(e)})")
        gpkg_fail += 1

print(f"")
print(f"GeoPackage Summary: {gpkg_pass} passed, {gpkg_fail} failed")
PYTHON_SCRIPT

# Get results from Python
python3 << 'PYTHON_SCRIPT_SUMMARY'
import sys
from pathlib import Path
import geopandas as gpd

output_dir = sys.argv[1] if len(sys.argv) > 1 else "."

gpkg_files = list(Path(output_dir).glob("**/*.gpkg"))
print(len(gpkg_files))
PYTHON_SCRIPT_SUMMARY gpkg_pass gpkg_fail

echo ""
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo "GeoTIFFs:    $tif_pass passed, $tif_fail failed"
echo "GeoPackages: $gpkg_pass passed, $gpkg_fail failed"
echo "=========================================="
echo ""

# Exit with error code if any validation failed
if [ $tif_fail -gt 0 ] || [ $gpkg_fail -gt 0 ]; then
    echo "⚠ Some files failed validation"
    exit 1
else
    echo "✓ All files passed validation"
    exit 0
fi
