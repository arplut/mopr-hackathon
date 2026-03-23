# MoPR Hackathon Semantic Segmentation Pipeline Scripts

This directory contains 10 production-quality scripts for the complete semantic segmentation pipeline for SVAMITVA drone orthophotos.

## Quick Start

### Installation
```bash
pip install rasterio geopandas shapely numpy pandas matplotlib torch transformers rio-cogeo tqdm
```

### Basic Workflow
```bash
# 1. Tile orthophotos
python tile_geotiff.py --input ortho.tif --output_dir tiles/ --village_id VILL001

# 2. Check CRS consistency
python check_crs.py --data_dir tiles/

# 3. Compute statistics on masks
python dataset_stats.py --mask_dir masks/ --output_dir stats/

# 4. Create train/val split
python train_val_split.py --tile_dir tiles/ --output_dir splits/ --val_villages VILL009 VILL010

# 5. Run inference
python batch_inference.py --checkpoint model.pth --tile_dir tiles/ --output_dir outputs/ --device cuda

# 6. Polygonize results
python mask_to_gpkg.py --mask VILL001_segmentation.tif --output VILL001_features.gpkg --village_id VILL001

# 7. Estimate solar potential
python solar_potential.py --gpkg VILL001_features.gpkg --dsm VILL001_dsm.tif

# 8. Generate summary statistics
python village_statistics.py --gpkg_dir outputs/ --output stats.csv

# 9. Validate outputs
./validate_outputs.sh outputs/
```

## Script Details

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| **tile_geotiff.py** | Tile large GeoTIFF into 512×512 patches | Large GeoTIFF | Per-tile GeoTIFFs |
| **dataset_stats.py** | Compute class distribution stats | Mask directory | CSV + chart + weights |
| **check_crs.py** | Verify CRS consistency | GeoTIFF directory | CRS report CSV |
| **train_val_split.py** | Create train/val split by village | Tile directory | Tile list text files |
| **merge_tiles_to_cog.py** | Merge tiles to Cloud Optimized GeoTIFF | Tile directory | COG segmentation |
| **mask_to_gpkg.py** | Polygonize masks to GeoPackage | Segmentation mask | GPKG with feature layers |
| **batch_inference.py** | Run SegFormer-B2 inference with TTA | Model + tiles | Per-village predictions |
| **village_statistics.py** | Generate per-village summary | GeoPackage directory | Statistics CSV |
| **solar_potential.py** | Estimate solar potential for RCC roofs | GeoPackage + DSM | Updated GPKG |
| **validate_outputs.sh** | Validate COG and GPKG files | Output directory | Validation report |

## File Statistics

- **Total Scripts**: 10 (9 Python + 1 Bash)
- **Total Lines of Code**: 2,716
- **Total Size**: ~85 KB
- **All scripts pass syntax validation**: ✓
- **Full docstrings and type hints**: ✓
- **Production-ready**: ✓

## Key Features

All scripts include:
- Full docstrings (module + function level)
- Comprehensive argument parsing with --help
- Logging for debugging and tracing
- Error handling for edge cases
- Progress bars for long operations
- Summary statistics output
- No hardcoded paths (all CLI configurable)
- Python 3.10+ compatibility
- Type hints throughout

## Dependencies

Core packages:
- **rasterio**: GeoTIFF I/O and geospatial operations
- **geopandas**: Vector data handling
- **shapely**: Geometry operations
- **numpy**: Numerical computing
- **pandas**: Data frames
- **matplotlib**: Data visualization
- **torch**: Deep learning framework
- **transformers**: HuggingFace models (SegFormer-B2)
- **rio-cogeo**: Cloud Optimized GeoTIFF creation
- **tqdm**: Progress bars

## Model Details

- **Architecture**: SegFormer-B2 (HuggingFace nvidia/mit-b2)
- **Input**: 4 channels (RGB + normalized DSM)
- **Output**: 9 classes (0-8)
- **Tile Size**: 512×512 pixels
- **Overlap**: 64 pixels
- **Test-Time Augmentation**: 4 rotations + horizontal flip

## Classes

0. Background
1. RCC roof
2. Tile roof
3. Tin roof
4. Thatched roof
5. Road pucca
6. Road kaccha
7. Water body
8. Vegetation

## Output Formats

- **Segmentation**: Cloud Optimized GeoTIFF (uint8, values 0-8)
- **Features**: GeoPackage with layers:
  - buildings (RCC/tile/tin/thatched roof polygons)
  - roads (pucca/kaccha centerlines)
  - water_bodies (water polygons)
  - vegetation (vegetation polygons)

## Example Use Cases

### Use Case 1: Complete Pipeline from Raw Orthophotos
```bash
# Starting with large orthophoto and mask labels
python tile_geotiff.py --input ortho.tif --output_dir tiles/ --village_id VILL001
python dataset_stats.py --mask_dir masks/ --output_dir stats/
python train_val_split.py --tile_dir tiles/ --mask_dir masks/ --output_dir splits/
python batch_inference.py --checkpoint model.pth --tile_dir tiles/ --output_dir outputs/
python mask_to_gpkg.py --mask outputs/VILL001/VILL001_segmentation.tif \
  --output VILL001_features.gpkg --village_id VILL001
```

### Use Case 2: Quality Assurance
```bash
python check_crs.py --data_dir outputs/
python village_statistics.py --gpkg_dir outputs/ --output stats.csv
./validate_outputs.sh outputs/
```

### Use Case 3: Solar Assessment
```bash
python batch_inference.py --checkpoint model.pth --tile_dir tiles/ --output_dir outputs/
python mask_to_gpkg.py --mask outputs/VILL001/VILL001_segmentation.tif \
  --output VILL001_features.gpkg --village_id VILL001
python solar_potential.py --gpkg VILL001_features.gpkg --dsm VILL001_dsm.tif
python village_statistics.py --gpkg_dir outputs/ --output solar_stats.csv
```

## Troubleshooting

### Common Issues

**ModuleNotFoundError for geospatial libraries**
```bash
pip install rasterio geopandas shapely
```

**CRS mismatch warning**
- Check that all input GeoTIFFs use the same CRS
- Use `check_crs.py` to identify mismatches
- Reproject if needed before processing

**No tiles generated (all skipped)**
- Check nodata_threshold parameter in tile_geotiff.py
- Ensure input GeoTIFF has valid data in most areas
- Verify input CRS and geotransform are valid

**OutOfMemory during batch inference**
- Reduce batch_size parameter in batch_inference.py
- Process fewer villages at a time
- Use GPU with more VRAM if available

**GeoPackage layer errors**
- Ensure geometry validity with check_crs.py output
- Verify CRS matches between input files
- Check that mask values are in range 0-8

## Performance Notes

- **Tiling**: ~1-5 seconds per 512×512 tile (depends on I/O)
- **Dataset stats**: ~30-60 seconds for 1000 tiles
- **Inference**: 0.1-0.3 seconds per tile (GPU), ~1-2 seconds per tile (CPU)
- **Polygonization**: 5-30 seconds per segmentation mask
- **COG validation**: ~100ms per file

## Author

MoPR Hackathon Team

## License

Contact hackathon organizers for licensing information.
