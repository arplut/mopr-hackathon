# Days 2–3 Operational Guide: GPU Training & Model Selection

**Timeline:** March 24–25, 2026 (deadline: March 30)
**Compute:** Google Colab Pro — A100 GPU
**Goal:** Train a production-quality SegFormer-B2 checkpoint with the best class-level metrics achievable in 2 days.

---

## Cell 1 — Run this first in every session

Before anything else, mount Drive and add the scripts folder to Python's path. Every `from X import Y` in the rest of this notebook depends on this cell having run.

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Add scripts folder to path once — all subsequent imports will work
import sys
SCRIPTS = '/content/drive/MyDrive/mopr-hackathon/scripts'
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

# Convenience base paths
DATA  = '/content/drive/MyDrive/mopr-hackathon'
REPO  = '/content/drive/MyDrive/mopr-hackathon'
```

> **Why not `importlib` for everything?** `importlib` is only needed when you don't know the path at import time or are debugging a path issue. Once Drive is mounted and `sys.path` is set correctly here, normal `from X import Y` works fine for the rest of the notebook and is much more readable.

---

## Pre-flight checklist (before touching GPU)

These steps run on CPU and should be done first to avoid wasting GPU hours.

### 1. Download data from GramAnchitra

Go to gramanchitra.gov.in and download the orthophotos + annotation masks for all 20 villages. You need:

- RGB orthophotos (large GeoTIFFs, 2–5 cm/pixel)
- Annotation masks (single-band uint8, class IDs 0–8)
- DSM rasters (if available separately; some villages bundle DSM as band 4)

Upload everything to Google Drive under a structure like:

```
drive/MyDrive/mopr-hackathon/
  raw/
    village_001/
      village_001_ortho.tif
      village_001_mask.tif
      village_001_dsm.tif   (if separate)
    village_002/
      ...
```

### 2. Verify CRS consistency

On Colab (CPU runtime is fine for this), run:

```python
!pip install rasterio tqdm --quiet

import importlib.util
spec = importlib.util.spec_from_file_location(
    "check_crs",
    '/content/drive/MyDrive/mopr-hackathon/scripts/check_crs.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

result = module.check_crs_consistency(
    data_dir='/content/drive/MyDrive/mopr-hackathon/raw',
    output_path='/content/crs_report.csv',
)
```

> **Note:** Use `importlib` rather than `sys.path.insert` + `from ... import`. The latter
> fails silently if Drive hasn't fully synced the folder, giving a misleading
> `ModuleNotFoundError`. `importlib` loads from an explicit file path and always shows
> the real error.

#### Known CRS situation for this dataset

Running the check on the current data reveals **4 different CRS** across the 20 files:

| CRS | EPSG | Files affected |
|---|---|---|
| UTM Zone 43N | EPSG:32643 | Set 2 Punjab orthos (standard, keep as-is) |
| UTM Zone 44N | EPSG:32644 | Set 1 Chhattisgarh orthos (standard, keep as-is) |
| Web Mercator | EPSG:3857 | `fattu_bhila_37458`, `bagga_37774` (Punjab files with `_3857` suffix) |
| Geographic WGS84 | EPSG:4326 | `gudbheli_445483`, `chanabhata_445476` (Chhattisgarh inference), `diwana_40082` (Punjab inference) |

There is also **one corrupted TIF and one unavailable village**:
- `KUTRU_451189_AAKLANKA_451163_ORTHO.tif` fails with `TIFFReadDirectory: Failed to read directory at offset 4937974012` — the file is truncated. A companion ECW exists but **ECW is not supported in Colab's GDAL build** (confirmed: `gdalinfo --formats | grep ECW` returns nothing, and `!apt-get install gdal-bin` does not add ECW support). This village must be **skipped entirely**. You have 9 usable training villages remaining, which is sufficient.

#### What needs fixing before tiling

EPSG:3857 (Web Mercator) and EPSG:4326 (geographic degrees) are not suitable for pixel-level work — distances and areas are distorted. Reproject those files to the correct UTM zone for their region before tiling.

**Fix EPSG:3857 Punjab files → EPSG:32643:**

```python
import subprocess, glob

files_3857 = [
    '/content/drive/MyDrive/mopr-hackathon/raw/training/set2_punjab/fattu_bhila_37458/37458_fattu_bhila_ortho_3857.tif',
    '/content/drive/MyDrive/mopr-hackathon/raw/training/set2_punjab/bagga_37774/37774_bagga_ortho_3857.tif',
]

for src in files_3857:
    dst = src.replace('_3857.tif', '_utm43n.tif')
    subprocess.run([
        'gdalwarp', '-t_srs', 'EPSG:32643',
        '-r', 'bilinear', '-co', 'COMPRESS=DEFLATE',
        src, dst
    ], check=True)
    print(f"Reprojected: {dst}")
```

**Fix EPSG:4326 inference files — three known files:**

All three are inference-only villages (no annotations affected):

```bash
# GUDBHELI — Chhattisgarh → UTM 44N
!gdalwarp -t_srs EPSG:32644 -r bilinear -co COMPRESS=DEFLATE \
  "/content/drive/MyDrive/mopr-hackathon/raw/inference/set2_chhattisgarh/gudbheli_445483/GUDBHELI_445483_Ortho.tif" \
  "/content/drive/MyDrive/mopr-hackathon/raw/inference/set2_chhattisgarh/gudbheli_445483/GUDBHELI_445483_Ortho_utm44n.tif"

# CHANABHATA — Chhattisgarh → UTM 44N
!gdalwarp -t_srs EPSG:32644 -r bilinear -co COMPRESS=DEFLATE \
  "/content/drive/MyDrive/mopr-hackathon/raw/inference/set2_chhattisgarh/chanabhata_445476/CHANABHATA_445476_Ortho.tif" \
  "/content/drive/MyDrive/mopr-hackathon/raw/inference/set2_chhattisgarh/chanabhata_445476/CHANABHATA_445476_Ortho_utm44n.tif"

# DIWANA — Punjab → UTM 43N
!gdalwarp -t_srs EPSG:32643 -r bilinear -co COMPRESS=DEFLATE \
  "/content/drive/MyDrive/mopr-hackathon/raw/inference/set1_punjab/diwana_40082/DIWANA_BARNALA_40082_ORTHO.tif" \
  "/content/drive/MyDrive/mopr-hackathon/raw/inference/set1_punjab/diwana_40082/DIWANA_BARNALA_40082_ORTHO_utm43n.tif"
```

Use the `_utm44n.tif` / `_utm43n.tif` outputs for inference — point the batch inference script at these files, not the originals.

**KUTRU village — skip entirely:**

The TIF is corrupted and the ECW fallback cannot be read on Colab. Exclude it from all file globs:

```python
import glob

all_orthos = sorted(glob.glob(
    '/content/drive/MyDrive/mopr-hackathon/raw/training/**/*.tif',
    recursive=True
))
SKIP = ['kutru_aaklanka_451189']
orthos = [f for f in all_orthos if not any(s in f for s in SKIP)]
print(f"Using {len(orthos)} training orthos")  # expect 9
```

#### CRS strategy for training

You do **not** need all villages in a single CRS for the model to train correctly — SegFormer operates on pixel arrays and does not use geographic coordinates. CRS only matters when merging tiles back into a full-scene COG and when exporting to GPKG. The practical approach is:

- Process each village's tiles independently (each village stays in its own UTM zone)
- During tile merging (Day 4), `rasterio.merge` handles per-village merging within the same CRS automatically
- Only report per-village outputs; do not attempt a cross-zone mosaic

### 3. Run the tiling script

Switch to a GPU runtime now. Mount Drive, then tile all villages:

```python
!pip install rasterio rio-cogeo tqdm --quiet

from tile_geotiff import tile_geotiff

import glob
ortho_files = sorted(glob.glob('/content/drive/MyDrive/mopr-hackathon/raw/*/ortho*.tif'))

for ortho in ortho_files:
    village_id = ortho.split('/')[-2]
    # Tile the orthophoto
    tile_geotiff(
        input_path=ortho,
        output_dir=f'/content/drive/MyDrive/mopr-hackathon/tiles/images/{village_id}',
        tile_size=512,
        overlap=64,
        nodata_threshold=0.8,
    )
    # Tile the corresponding mask (use overlap=0 for masks, or match overlap if using for training)
    mask_path = ortho.replace('ortho', 'mask')
    tile_geotiff(
        input_path=mask_path,
        output_dir=f'/content/drive/MyDrive/mopr-hackathon/tiles/masks/{village_id}',
        tile_size=512,
        overlap=64,
        nodata_threshold=0.8,
    )
```

Adapt the file naming to match your actual filenames. The key is that image tiles and mask tiles have matching names.

### 4. Compute class statistics and weights

```python
from dataset_stats import compute_dataset_stats

stats = compute_dataset_stats(
    mask_dir='/content/drive/MyDrive/mopr-hackathon/tiles/masks',
    output_dir='/content/drive/MyDrive/mopr-hackathon/stats',
)

# Copy the printed weights — you'll need them for training
print("Class weights:", stats['weight'])
```

Save the output. These weights go directly into `configs/class_weights.py` and your training config.

### 5. Create train/val split

```python
from train_val_split import create_train_val_split

create_train_val_split(
    tile_dir='/content/drive/MyDrive/mopr-hackathon/tiles/images',
    mask_dir='/content/drive/MyDrive/mopr-hackathon/tiles/masks',
    output_dir='/content/drive/MyDrive/mopr-hackathon/splits',
    val_ratio=0.2,     # 8 train villages, 2 val villages
    by_village=True,    # Split at village level, not tile level
)
```

Splitting by village (not randomly by tile) is critical — it prevents data leakage from spatial autocorrelation.

---

## Day 2: Baseline training (RGB only, 50 epochs)

This is your anchor result. Everything else is measured against it.

### Setup the Colab environment

Open `notebooks/01_training.ipynb` on Colab. The first cells install dependencies:

```bash
!pip install torch torchvision --quiet
!pip install transformers accelerate --quiet
!pip install rasterio albumentations --quiet
!pip install wandb --quiet
!wandb login YOUR_API_KEY
```

### Configure the baseline run

In the training config cell, set:

```python
from configs.training_config import TrainingConfig, ModelConfig, DataConfig

model_cfg = ModelConfig(
    backbone="nvidia/mit-b2",
    num_classes=9,
    in_channels=3,          # RGB only for baseline
    pretrained=True,
)

data_cfg = DataConfig(
    train_dir="/content/drive/MyDrive/mopr-hackathon/splits/train/images",
    train_mask_dir="/content/drive/MyDrive/mopr-hackathon/splits/train/masks",
    val_dir="/content/drive/MyDrive/mopr-hackathon/splits/val/images",
    val_mask_dir="/content/drive/MyDrive/mopr-hackathon/splits/val/masks",
    tile_size=512,
    batch_size=8,
)

train_cfg = TrainingConfig(
    epochs=50,
    learning_rate=6e-5,
    optimizer="adamw",
    scheduler="cosine",
    fp16=True,
    class_weights=[...],   # Paste weights from dataset_stats output
    loss_type="dice_focal",
    wandb_project="mopr-hackathon",
    wandb_run_name="baseline-rgb-50ep",
)
```

### What to watch in W&B during training

Keep the W&B dashboard open. Key signals:

- **Training loss** should decrease steadily for the first 20 epochs, then slow down
- **Val mIoU** is the most important metric. If it plateaus before epoch 30, you may be overfitting
- **Per-class IoU** — background and water will converge fast; thatched and kaccha road will be last
- **Learning rate curve** — cosine decay should smoothly reach near-zero by epoch 50

If training loss drops but val mIoU stalls or drops (gap > 15 points), you're overfitting. Solutions in priority order: more augmentation, lower LR, weight decay increase.

### Baseline expected results

Based on the dataset characteristics (high class imbalance, varied village morphology), expect roughly:

| Class | Expected IoU range |
|---|---|
| Background | 90–95% |
| RCC roof | 65–75% |
| Tile roof | 50–65% |
| Tin roof | 35–50% |
| Thatched roof | 25–40% |
| Road pucca | 60–70% |
| Road kaccha | 35–50% |
| Water body | 75–85% |
| Vegetation | 75–85% |

Overall mIoU around 55–65%. This is your baseline to beat.

### Save the baseline checkpoint

The training notebook auto-saves the best checkpoint (highest val mIoU). Verify:

```python
import torch
ckpt = torch.load('/content/drive/MyDrive/mopr-hackathon/checkpoints/baseline_rgb_best.pt')
print("Best epoch:", ckpt['epoch'])
print("Best val mIoU:", ckpt['best_miou'])
```

---

## Day 3: Ablation experiments + best config

Run these experiments sequentially. Each one builds on what you learned from the previous.

### Ablation A: RGB + DSM (4-channel input)

This is the single highest-impact change expected. DSM (height data) helps the model distinguish roof materials that look similar in RGB but differ in height.

**What to change from baseline:**

```python
model_cfg.in_channels = 4   # RGB + DSM

train_cfg.wandb_run_name = "ablation-a-rgbdsm-50ep"
train_cfg.epochs = 50
```

You also need to prepare 4-channel tiles. If your DSM is a separate file, stack it with the orthophoto:

```python
import rasterio
import numpy as np

def stack_rgb_dsm(rgb_path, dsm_path, output_path):
    with rasterio.open(rgb_path) as rgb_src:
        rgb = rgb_src.read()  # (3, H, W)
        profile = rgb_src.profile.copy()

    with rasterio.open(dsm_path) as dsm_src:
        dsm = dsm_src.read(1)  # (H, W)

    # Normalize DSM to 0-1 range for stacking
    dsm_valid = dsm[dsm > 0]
    if len(dsm_valid) > 0:
        dsm_norm = (dsm - dsm_valid.min()) / (dsm_valid.max() - dsm_valid.min() + 1e-6)
    else:
        dsm_norm = np.zeros_like(dsm)

    stacked = np.concatenate([rgb, dsm_norm[np.newaxis]], axis=0)  # (4, H, W)

    profile.update(count=4, dtype='float32')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(stacked.astype(np.float32))
```

**Expected improvement:** +5–10 mIoU on roof material classes. RCC vs tile discrimination should improve significantly because RCC roofs are typically flat (low DSM variance) while tile roofs are ridged.

**Decision point:** If mIoU improves by more than 3 points, keep 4-channel for all subsequent runs.

### Ablation B: Focal + Dice loss (if not already used in baseline)

If your baseline used cross-entropy, switch to combined Dice + Focal:

```python
train_cfg.loss_type = "dice_focal"
train_cfg.wandb_run_name = "ablation-b-focal-dice-50ep"
```

The focal loss down-weights easy examples (background, water) and focuses learning on hard examples (thatched, kaccha road). Dice loss directly optimizes the IoU-like metric.

**Expected improvement:** +5–15 F1 on minority classes (thatched, tin, kaccha road). Overall mIoU might not change much because the gain on rare classes is offset by slight drops on easy classes.

**Decision point:** Compare per-class F1 side by side. If any minority class gains > 5 F1 points, keep this loss.

### Ablation C: Copy-paste augmentation for rare classes

Only run this if Ablation B still leaves minority class F1 below 50%. Copy-paste augmentation extracts instances of rare classes from one tile and pastes them into other tiles, synthetically balancing the dataset.

```python
train_cfg.augmentation.copy_paste = True
train_cfg.augmentation.copy_paste_classes = [3, 4]  # tin, thatched
train_cfg.wandb_run_name = "ablation-c-copypaste-50ep"
```

This requires adding a CopyPaste transform to your Albumentations pipeline. The implementation is in the training notebook.

### Interpreting ablation results in W&B

Open the W&B comparison view (select all runs → "Compare"). Look at:

1. **Overall mIoU bar chart** — which run wins overall?
2. **Per-class IoU table** — which run wins on the hardest classes (tin, thatched, kaccha)?
3. **Training curves overlay** — does any run overfit faster?

The best config is usually: **RGB+DSM + Focal+Dice + standard augmentation**. Copy-paste helps only if rare classes are severely underrepresented (< 1% of pixels).

### Best config full run (100 epochs)

Once you've identified the winning combination, do a longer training run:

```python
model_cfg.in_channels = 4       # or 3, based on ablation A result

train_cfg.epochs = 100
train_cfg.scheduler = "cosine"
train_cfg.fp16 = True
train_cfg.wandb_run_name = "best-config-100ep"
```

This run will take 3–4 hours on an A100. Start it before lunch or before bed — don't waste GPU time watching it.

### Save the final model checkpoint

```python
# The training loop saves the best automatically, but also save the final
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'best_miou': best_miou,
    'class_ious': class_ious,
    'config': {
        'in_channels': model_cfg.in_channels,
        'num_classes': model_cfg.num_classes,
        'backbone': model_cfg.backbone,
    }
}, '/content/drive/MyDrive/mopr-hackathon/checkpoints/best_model.pt')
```

---

## Key troubleshooting scenarios

### "CUDA out of memory"

Reduce batch size first (8 → 4 → 2). If still OOM, reduce tile size from 512 to 384. Do NOT reduce model size (B2 → B0) — the accuracy drop isn't worth it.

### "Val mIoU stuck at 40% after 30 epochs"

Check your data loader. Common causes: mask tiles don't align with image tiles (tiling was done separately with different parameters), class IDs in masks don't match your config (0-indexed vs 1-indexed), or DSM channel is all zeros.

Quick diagnostic:

```python
# Visualize a random training sample
import matplotlib.pyplot as plt
img, mask = dataset[42]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img[:3].permute(1,2,0).numpy() * 0.5 + 0.5)  # Denormalize RGB
ax2.imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=8)
plt.show()
```

If the mask looks wrong or doesn't align with the image, your tiling has a bug.

### "Thatched class IoU is 0%"

This means the class has too few pixels for the model to learn. Solutions: increase class weight for thatched (try 10x), use copy-paste augmentation, or merge thatched with "other_roof" and reduce to 8 classes. Honestly reporting this limitation in your submission document is better than faking a result.

### "W&B not logging"

Make sure `wandb.init()` is called before training starts, and `wandb.log()` is called inside the training loop. Check that your API key is set:

```python
import wandb
wandb.login(key="YOUR_KEY")
wandb.init(project="mopr-hackathon", name="run-name")
```

---

## Days 4–5: GIS pipeline execution and submission prep

Once you have the best checkpoint, the remaining work is mostly automated by the scripts from Day 1.

### Day 4: Run inference on all 20 villages

**Step 1 — Batch inference:**

```python
from scripts.batch_inference import batch_inference

result = batch_inference(
    checkpoint_path='/content/drive/MyDrive/mopr-hackathon/checkpoints/best_model.pt',
    tile_dir='/content/drive/MyDrive/mopr-hackathon/tiles/images',
    output_dir='/content/drive/MyDrive/mopr-hackathon/predictions',
    device='cuda',
    batch_size=4,
    use_tta=True,           # 5x augmentations, slower but better
    save_probabilities=True, # Needed for overlap merging
)
```

This processes all village tile directories and saves prediction masks, probability rasters, and confidence maps.

**Step 2 — Merge tiles to full-scene COG:**

```python
from scripts.merge_tiles_to_cog import merge_tiles_to_cog

import glob
village_dirs = sorted(glob.glob('/content/drive/MyDrive/mopr-hackathon/predictions/village_*'))

for vdir in village_dirs:
    village_id = vdir.split('/')[-1]
    merge_tiles_to_cog(
        tile_dir=vdir,
        output_path=f'/content/drive/MyDrive/mopr-hackathon/outputs/{village_id}_segmentation.tif',
        tile_size=512,
        overlap=64,
    )
```

**Step 3 — Polygonize to GeoPackage:**

```python
from scripts.mask_to_gpkg import mask_to_gpkg

import glob
cog_files = sorted(glob.glob('/content/drive/MyDrive/mopr-hackathon/outputs/*_segmentation.tif'))

for cog in cog_files:
    village_id = cog.split('/')[-1].replace('_segmentation.tif', '')
    mask_to_gpkg(
        mask_path=cog,
        output_path=f'/content/drive/MyDrive/mopr-hackathon/outputs/{village_id}_features.gpkg',
        village_id=village_id,
    )
```

**Step 4 — Validate all outputs:**

```bash
bash scripts/validate_outputs.sh /content/drive/MyDrive/mopr-hackathon/outputs/
```

Every COG must pass `rio cogeo validate`. Every GPKG must have valid geometries, correct CRS, and required attributes.

**Step 5 — Solar potential (bonus):**

```python
from scripts.solar_potential import solar_potential

for gpkg in sorted(glob.glob('/content/drive/MyDrive/mopr-hackathon/outputs/*_features.gpkg')):
    village_id = gpkg.split('/')[-1].replace('_features.gpkg', '')
    dsm_path = f'/content/drive/MyDrive/mopr-hackathon/raw/{village_id}/{village_id}_dsm.tif'
    solar_potential(gpkg_path=gpkg, dsm_path=dsm_path)
```

**Step 6 — Village statistics:**

```python
from scripts.village_statistics import generate_village_statistics

generate_village_statistics(
    gpkg_dir='/content/drive/MyDrive/mopr-hackathon/outputs',
    output_csv='/content/drive/MyDrive/mopr-hackathon/outputs/village_statistics.csv',
)
```

### Day 5: Polish and submission

**Morning — Documentation finalization:**

1. Fill in actual metrics from W&B into `docs/submission_document.md` Section 4
2. Add team member names in Section 2
3. Insert W&B run comparison screenshot into Section 5 (Uniqueness)
4. Add per-village statistics table to Section 7 (Expected Impact)
5. Add GitHub repo URL and W&B public report URL to Section 13
6. Update `docs/model_card.md` with final metrics

**Afternoon — Final QA:**

1. Open each GPKG in QGIS — spot check 3–5 villages visually
2. Run QGIS topology checker on building layers
3. Create qualitative overlay images (prediction mask on top of RGB) for 5 representative tiles
4. Verify the GitHub repo README instructions actually work on a fresh Colab instance
5. Export the submission document as PDF

**Submission checklist:**

- [ ] Best model checkpoint (.pt) uploaded
- [ ] All 20 village COGs pass `rio cogeo validate`
- [ ] All 20 village GPKGs pass validation script
- [ ] Per-village statistics CSV generated
- [ ] Submission document PDF with all 13 sections complete
- [ ] Model card filled with actual metrics
- [ ] GitHub repo is public with working README
- [ ] W&B report is public and linked in submission
- [ ] QGIS screen recording (optional but recommended)

---

## Time budget summary

| Task | Estimated time | Day |
|---|---|---|
| Data download + CRS check | 1–2 hours | Day 2 morning |
| Tiling + statistics | 1 hour | Day 2 morning |
| Baseline training (50 epochs) | 2–3 hours | Day 2 afternoon |
| Ablation A: RGB+DSM | 2–3 hours | Day 3 morning |
| Ablation B: Focal+Dice | 2–3 hours | Day 3 morning (parallel if 2 GPUs, else sequential) |
| Best config full run (100 epochs) | 3–4 hours | Day 3 afternoon/evening |
| Batch inference (20 villages, TTA) | 2–3 hours | Day 4 morning |
| Merge + polygonize + validate | 1–2 hours | Day 4 afternoon |
| Solar potential + statistics | 30 min | Day 4 afternoon |
| Documentation finalization | 3–4 hours | Day 5 morning |
| Final QA + submission prep | 2–3 hours | Day 5 afternoon |

Total GPU time needed: approximately 12–16 hours across Days 2–3. A single Colab Pro A100 session can handle this if you manage disconnections (save checkpoints frequently to Drive, not just Colab local storage).

---

## Critical reminders

1. **Save everything to Google Drive, not Colab's local `/content/` directory.** Colab sessions disconnect without warning. Any unsaved work on local disk is lost.

2. **The training notebook auto-saves the best checkpoint by val mIoU.** Don't rely on the final epoch — the best model is often from epoch 70–80, not 100.

3. **TTA at inference time is slow (5x) but improves mIoU by 1–3 points.** Always use it for final outputs, skip it for quick debugging.

4. **Don't spend more than 4 hours on ablations.** If RGB+DSM + Focal+Dice gives you mIoU > 60%, move on to the GIS pipeline. Diminishing returns hit fast.

5. **The deadline is March 30.** You have 7 days. Days 2–3 are the GPU bottleneck — everything after that is mostly scripted.
