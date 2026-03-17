# MoPR Geospatial Intelligence Hackathon — Problem Statement 1
## Project Plan & Execution Roadmap

---

## 0. Context Summary

### What this project is

This is a submission for the **AI/ML Hackathon by the Ministry of Panchayati Raj (MoPR)**, organized by the Geo-Intel Lab at IIT Tirupati Navavishkar I-Hub Foundation (IITTNiF), in partnership with OGC and NIC. The hackathon focuses on AI/ML-based geospatial analysis of drone imagery for smart rural planning.

**Problem Statement 1 — AI-Based Feature Extraction from Drone Images (SVAMITVA Scheme)**
> Develop an AI/ML model to identify and classify key features from SVAMITVA drone orthophotos across Indian villages, including building footprints with roof material classification (RCC, Tile, Tin, etc.), road networks, waterbodies, and infrastructure assets.

**Team size:** 2 members  
**Roles:** AI/CV Engineer + Documentation & Research Lead  
**Timeline:** 17 Feb – 30 March 2026 (competition window); this plan covers a focused **10-day execution sprint**

---

### Key deliverables required

| Deliverable | Format | Notes |
|---|---|---|
| Trained & optimized AI model | `.pth` checkpoint + config | For orthophoto feature extraction |
| Feature-extracted datasets | GeoPackage (GPKG) | Buildings, roads, waterbodies, assets — per village |
| Raster outputs | Cloud Optimized GeoTIFF (COG) | Segmentation masks, probability maps |
| Technical documentation | PDF / Doc | Model design, training, deployment |
| Final report | PDF / Doc | Accuracy metrics + improvement recommendations |

---

### Critical insights & considerations

#### What judges actually care about (from briefing notes)
- **Methodology depth over UI polish** — no dashboard/interface required; focus on the ML pipeline and output quality
- **Open-source preference** — avoid commercial tools; prefer reproducible, vendor-neutral stacks
- **OGC standards alignment** — outputs in COG + GPKG formats signals governance-readiness; explicitly mention standard names in docs
- **"Innovation without standards will not scale"** — this was literally said in the briefing; judges are standards-aware
- **Going beyond the 3 stated features** is explicitly called out as a differentiator (solar potential, property tax, flood risk were mentioned)
- **Honest limitations + recommendations** are a required deliverable — don't oversell
- **Reproducible workflows** — clean documentation so any government agency could re-run it
- **Avoid foreign data centres** for model training/hosting if possible

#### Technical considerations
- **95% accuracy claim** in the problem statement refers to overall pixel accuracy, NOT mIoU — report both, but lead with per-class IoU and F1
- **DSM + RGB fusion** is the single highest-impact experiment to run (height data resolves RCC vs tile ambiguity)
- **Class imbalance** is severe — focal loss or copy-paste augmentation for rare classes (thatched, plastic) is essential
- **Kaccha roads are the hardest class** — edge detection fails on unpaved surfaces; set expectations accordingly
- **Geographic performance gaps** will exist — report mIoU per village, not just aggregate, to show awareness
- **Tiling large GeoTIFFs** with overlap is mandatory; naive tiling creates visible seam artifacts in outputs
- **COG output must be validated** with `rio cogeo validate` before submission

#### Data context
- Input: drone orthophotos (2cm–5cm/pixel) for **20 villages total** — 10 for training/validation, 10 for test
- Data is hosted on GramAnchitra (gramanchitra.gov.in) — the government's own geospatial platform
- Files are large GeoTIFFs that need to be broken into tiles before model ingestion
- CRS consistency across villages must be verified before training

#### Bonus differentiators (worth doing if time permits)
- Export training annotations in **TrainingDML-AI format** using `pytdml` (OGC standard for ML training data)
- Add **solar potential** attribute to RCC rooftops in GPKG output
- Add **per-village summary statistics** CSV (building counts by roof type, road length, water area)
- Write a **model adaptability note** explaining how to add new classes with minimal retraining

---

### Recommended technology stack

| Layer | Tool | Why |
|---|---|---|
| Model architecture | SegFormer-B2 | Best accuracy/compute tradeoff for aerial imagery |
| Framework | MMSegmentation or HuggingFace Transformers | Config-driven, pretrained weights ready |
| Pretrained backbone | SatMAE or ImageNet Swin-T | Domain-aligned pretraining for aerial data |
| Loss function | Dice + Focal combined | Handles class imbalance natively |
| Input modality | RGB + DSM (4-channel) | Height channel dramatically improves roof material discrimination |
| Augmentation | Albumentations | Rotation-invariant, multi-scale crops |
| Geospatial stack | GDAL + rasterio + geopandas + shapely | Full OSS pipeline, COG + GPKG output |
| Annotation tool | Label Studio (if corrections needed) | OSS, has active learning backend |
| Experiment tracking | Weights & Biases (W&B) | Shareable run URLs for submission |
| Compute | Google Colab Pro (A100) or Kaggle (T4 free) | Accessible, no infra needed |
| Post-processing | rasterio.features.shapes + morphological cleanup | Polygonize masks to GPKG |

---

## 1. Phase 1 — Data Pipeline & Setup (Days 1–2)

**Goal:** A clean, reproducible data pipeline where the model can ingest tiles and produce predictions. No model quality focus yet — just infrastructure.

---

### AI/CV Engineer tasks

- [ ] Set up MMSegmentation (or HuggingFace) on Colab Pro with GPU; verify CUDA environment
- [ ] Write **tiling script** using `rasterio`:
  - 512×512 tiles with 64px overlap
  - Preserve CRS and georeferencing per tile
  - Output: `village_id/tile_row_col.tif` naming convention
- [ ] Verify tile-to-annotation alignment — open both in QGIS, check pixel correspondence
- [ ] Write **dataset statistics script**: pixel count per class, tile count, void/null tile flagging
- [ ] Set up **train/val split**: 8 villages train, 2 held out as validation (pick geographically distinct ones)
- [ ] Run one training epoch on 5 tiles just to confirm the pipeline doesn't crash

### Documentation & Research Lead tasks

- [ ] Set up **W&B project**: define logging schema (mIoU, per-class F1, precision, recall, loss curves)
- [ ] Read **LoveDA and ISPRS Vaihingen benchmark papers** — note which augmentations and architectures performed best on aerial data
- [ ] Explore **TrainingDML-AI / pytdml** repository — understand if annotated data can be exported in that format
- [ ] Begin documenting the data pipeline (inputs, processing steps, output structure) — this feeds Section 6 of the submission
- [ ] Verify CRS consistency across all village files; flag any mismatches

### Checkpoint
> **End of Day 2:** Tiling script runs without errors. One training epoch completes. W&B logs appear. Class distribution chart produced. Train/val split documented.

---

### AI-assisted workflows — Phase 1

#### Code generation
Use an AI coding assistant (Claude in chat or Cursor IDE) for all script writing. The key is to be specific about inputs and outputs:

**Prompt — tiling script:**
```
Write a Python script using rasterio that tiles a large GeoTIFF into 512×512 patches 
with 64 pixels of overlap. Each tile should preserve its geotransform and CRS. 
Save tiles as GeoTIFFs named {village_id}_{row}_{col}.tif. Skip tiles that are 
more than 80% nodata. Input: path to orthophoto GeoTIFF. Output: directory of tiles.
```

**Prompt — dataset statistics:**
```
Write a script using rasterio and numpy that reads a directory of label mask GeoTIFFs 
(single-band, uint8, values 0–8 representing classes), counts the pixel frequency 
per class across all files, and outputs a CSV and a matplotlib bar chart. 
Classes: [background, RCC, tile, tin, thatched, road_pucca, road_kaccha, water, vegetation]
```

**Prompt — environment debugging:**
```
I'm getting this error on Google Colab when running rasterio: [PASTE FULL TRACEBACK]
My setup: [PASTE YOUR PIP INSTALL COMMANDS AND PYTHON VERSION]
What's causing this and how do I fix it?
```

---

## 2. Phase 2 — Baseline & Core Experiments (Days 3–5)

**Goal:** A first meaningful mIoU number, then systematic ablations to find the 2–3 highest-impact changes. By end of Day 5, have the best model checkpoint saved.

---

### AI/CV Engineer tasks

#### Day 3 — Baseline (RGB only)
- [ ] Train **SegFormer-B2** (ImageNet pretrain), Dice+CE loss, 50 epochs
- [ ] Apply standard augmentation: H/V flip, 90°/180°/270° rotation, brightness/contrast jitter
- [ ] Run inference on val set with **sliding window + overlap logit averaging** (not hard argmax per tile)
- [ ] Generate confusion matrix + per-class IoU — log to W&B

#### Day 4 — Key ablations
- [ ] **Ablation A:** Add normalized DSM as 4th channel — retrain, compare per-class F1
- [ ] **Ablation B:** Swap loss to Focal+Dice — compare minority class (Tin, Thatched) F1 specifically
- [ ] **Ablation C:** Copy-paste augmentation for rare classes
- [ ] Pick best combination based on per-class results, note which delta was largest

#### Day 5 — Best config full run
- [ ] Full training run, 100 epochs, cosine LR decay, mixed precision FP16
- [ ] Apply **TTA (test-time augmentation)**: 4 rotations + horizontal flip, average predictions
- [ ] Save model checkpoint (`.pth`) + full inference config
- [ ] **Reproducibility test:** run full pipeline from scratch on a fresh Colab instance

### Documentation & Research Lead tasks

- [ ] Create **W&B comparison table** of all ablations — screenshot for report (Section 4)
- [ ] Write **Section 5 (Uniqueness & Innovation)** based on what ablations reveal about RGB+DSM fusion and rare class handling
- [ ] Produce **per-village accuracy breakdown** from val set — check for geographic performance gaps
- [ ] Write **Section 4 (Proposed Solution)** draft — 200–250 words, focused on methodology
- [ ] Find 2–3 relevant prior works (SpaceNet, INRIA, LoveDA) for citations in the technical report
- [ ] Draft **qualitative overlay images**: prediction mask overlaid on RGB for 5 representative tiles

### Checkpoint
> **End of Day 5:** Best model checkpoint saved and validated. Per-class and per-village metrics documented in W&B. Sections 4 and 5 drafted.

---

### Notes — ablation prioritization

Run experiments in this order (highest expected delta first):

1. **RGB → RGB+DSM** (expected: +5–10 mIoU on roof material classes, especially tile vs RCC)
2. **CE loss → Focal+Dice** (expected: +5–15 F1 on minority classes — largest effect on Thatched and Tin)
3. **No augmentation → rotation-invariant aug** (expected: +2–4 mIoU overall, free wins)
4. **Copy-paste for rare classes** (only worth doing if step 2 still leaves minority class F1 < 50%)

Do **not** run architecture comparisons (SegFormer vs U-Net vs DeepLab) unless you have GPU budget to spare — SegFormer-B2 is likely to win and confirming this costs 2 days.

---

### AI-assisted workflows — Phase 2

**Prompt — MMSegmentation config generation:**
```
Generate a complete MMSegmentation config file for SegFormer-B2 with the following:
- 9 classes: background, RCC_roof, tile_roof, tin_roof, thatched_roof, 
  road_pucca, road_kaccha, water_body, vegetation
- Input: 4-channel (RGB + DSM normalized 0–1)
- Loss: combined Dice + Focal loss with class weights [1, 3, 3, 4, 5, 2, 3, 1.5, 1]
- Augmentation: random flip, random rotate 90°, random brightness contrast
- Batch size: 8, image size: 512×512
- Pretrained backbone: mit_b2 from ImageNet
- Optimizer: AdamW, LR 6e-5, cosine decay
- Mixed precision: FP16
```

**Prompt — interpreting results:**
```
My SegFormer-B2 model trained on drone orthophotos of Indian villages has these 
per-class IoU results after 50 epochs:
background: 94%, RCC: 71%, tile: 58%, tin: 43%, thatched: 31%, 
road_pucca: 67%, road_kaccha: 44%, water: 82%, vegetation: 79%

The confusion matrix shows tin is often predicted as thatched, and kaccha road 
is often predicted as bare soil/background. What are the likely causes and 
what should I try first to improve the low-performing classes?
```

**Prompt — training curve diagnosis:**
```
My training loss is still decreasing at epoch 50 but val mIoU has plateaued 
for the last 15 epochs. Training mIoU: 81%, Val mIoU: 67%. 
What does this suggest and what should I adjust? 
My dataset has 8 training villages (~12,000 tiles) and 2 val villages (~3,000 tiles).
```

---

## 3. Phase 3 — Output Pipeline & GIS Integration (Days 6–7)

**Goal:** Model predictions converted to valid, government-ready GeoPackage and COG outputs for all 20 villages. Outputs must open correctly in QGIS with proper attributes and CRS.

---

### AI/CV Engineer tasks

#### Day 6 — Prediction → GeoPackage pipeline
- [ ] Write **tile-merging script**: stitch predicted tiles back to full-scene COG using `rasterio` + GDAL
  - Use logit averaging in overlap zones (not argmax per tile)
  - Output: full-scene segmentation mask as COG
- [ ] **Polygonize** segmentation mask per class: `rasterio.features.shapes`
- [ ] Apply **morphological cleanup**: remove isolated polygons < 2m², smooth boundaries with 0.3m tolerance
- [ ] Write **GPKG export**: building layer with attributes: `roof_type` (string), `area_m2` (float), `confidence` (float), `village_id` (string)
- [ ] Run `rio cogeo validate` on all raster outputs — fix any invalid COGs

#### Day 7 — Batch inference on all test villages
- [ ] Batch inference script for all 10 test villages — parallelized tile inference
- [ ] Output per village: one COG + one GPKG, consistently named: `{village_id}_segmentation.tif`, `{village_id}_features.gpkg`
- [ ] Sanity checks: polygon count per class, coverage percentage, no blank outputs
- [ ] Optional: connected-component labeling for individual building instance IDs

### Documentation & Research Lead tasks

- [ ] Load GPKG in QGIS — verify geometry validity, CRS, attribute schema
- [ ] Run **QGIS topology check** — flag any self-intersecting polygons; document findings
- [ ] **Validate COG outputs**: confirm `rio cogeo validate` passes for every file
- [ ] Generate **per-village summary statistics**: building count by roof type, road length (km), water area (m²)
- [ ] Create a formatted summary table — this is the "feature-extracted dataset" deliverable
- [ ] Cross-check output file naming against OGC guide specification
- [ ] Draft **Section 7 (Expected Impact)** using the village statistics as concrete evidence

### Checkpoint
> **End of Day 7:** All 10 test village GPKGs + COGs generated, valid, and consistently named. QGIS verification passed. Per-village statistics table complete.

---

### Notes — output format requirements (from OGC guide)

**COG (Cloud Optimized GeoTIFF) — for all raster outputs:**
- Segmentation masks, probability maps, DTM-derived surfaces
- Convert using: `rio cogeo create input.tif output_cog.tif --cog-profile deflate`
- Validate: `rio cogeo validate output_cog.tif`
- Reference: https://cogeo.org / https://www.ogc.org/standards/ogc-cloud-optimized-geotiff/

**GPKG (GeoPackage) — for all vector outputs:**
- Building footprints (with `roof_type` attribute), roads, waterbodies, infrastructure assets
- Single-file, multiple layers, CRS embedded
- Reference: https://www.geopackage.org / https://www.ogc.org/standards/geopackage/

**Required GPKG layers:**
| Layer name | Geometry | Key attributes |
|---|---|---|
| `buildings` | Polygon | `roof_type`, `area_m2`, `confidence`, `village_id` |
| `roads` | LineString | `road_type` (pucca/kaccha), `length_m`, `village_id` |
| `water_bodies` | Polygon | `area_m2`, `village_id` |
| `infrastructure` | Point | `asset_type` (DT/OHT/well), `village_id` |

---

### AI-assisted workflows — Phase 3

**Prompt — tile merging + COG output:**
```
Write a Python script using rasterio that:
1. Takes a directory of predicted segmentation mask tiles (512×512, uint8, with 64px overlap)
2. Merges them back into a full-scene raster using logit/probability averaging in overlap zones
3. Saves the result as a Cloud Optimized GeoTIFF (COG) using rio-cogeo
4. The output should preserve the original orthophoto's CRS and spatial extent
The tiles are named {village_id}_{row}_{col}_pred.tif and each contains 
the georeferencing metadata from the original tiling step.
```

**Prompt — mask to GPKG pipeline:**
```
Write a Python script using rasterio, shapely, and geopandas that:
1. Reads a single-band segmentation mask GeoTIFF (uint8, values 0–8)
2. Polygonizes each class separately using rasterio.features.shapes
3. Applies simplification with 0.3m tolerance and removes polygons < 2 square meters
4. Creates a GeoPackage with separate layers: buildings, roads, water_bodies
5. For buildings layer, adds attribute 'roof_type' based on class value mapping:
   {1: 'RCC', 2: 'tile', 3: 'tin', 4: 'thatched'}
6. Adds 'area_m2' computed from polygon geometry
7. CRS should match the input raster
```

**Prompt — QGIS validation check:**
```
I have a GeoPackage file with building polygon features from a semantic segmentation 
model. What QGIS checks should I run to validate it before submission? 
List the specific tools/menu paths in QGIS and what errors to look for.
```

---

## 4. Phase 4 — Beyond Scope & Polish (Days 8–9)

**Goal:** Add at least one differentiator beyond the three stated features. Complete all submission document sections. GitHub repo ready and public.

---

### AI/CV Engineer tasks

- [ ] **Solar potential estimate**: flag RCC rooftops > 20m² unshaded area using DSM; add `solar_potential` attribute (`high`/`medium`/`low`) to buildings layer in GPKG
- [ ] **Road type attribute**: add `road_type` (pucca/kaccha) to road layer — use spectral signature difference + texture feature as discriminator
- [ ] **Proximity analysis** (optional): compute distance from each building centroid to nearest water body; add `water_proximity_m` attribute
- [ ] **TrainingDML-AI packaging** (bonus): if pytdml worked, export training annotations in OGC TrainingDML-AI format
- [ ] Record a **screen recording** of QGIS showing GPKG layers loaded with styled symbology (roof types in different colors)
- [ ] **Final reproducibility check**: run the entire pipeline from data → GPKG on a fresh Colab instance; document any issues

### Documentation & Research Lead tasks

- [ ] Write **"model adaptability" note**: how to add new classes (solar panels, distribution transformers) with minimal retraining — describe the annotation + fine-tuning process
- [ ] Draft **Section 8 (Implementation Roadmap)**: 3-month and 12-month scaling plan — from 20 villages to 3.5 lakh villages across 20 states
- [ ] Draft **Section 10 (Scalability & Sustainability)**: domain adaptation across agro-climatic zones, federated model approach
- [ ] Draft **Section 11 (Stakeholders)**: MoPR, state revenue departments, SVAMITVA, NIC, solar agencies (PM-KUSUM), NDMA
- [ ] Prepare **GitHub repository**:
  - `README.md`: setup instructions, data format, how to run inference
  - `requirements.txt`
  - `notebooks/`: training notebook, inference notebook
  - `scripts/`: tiling, inference, GPKG export
  - `docs/`: model card, class taxonomy with examples
  - `outputs/sample/`: one sample village output (COG + GPKG)
- [ ] Complete all 13 submission document sections; cross-check against word limits

### Checkpoint
> **End of Day 9:** At least one beyond-scope output in GPKG. All 13 document sections drafted. GitHub repo public with working README.

---

### Notes — Section-by-section writing guide

| Section | Word limit | Key points to hit |
|---|---|---|
| 1. Title | 15 words | Specific, includes "SVAMITVA", "multi-class", "rural" |
| 2. Team / Affiliation | 50 words | Names, institution, contact |
| 3. Problem Statement | 100–150 words | Ground surveys are slow + expensive; SVAMITVA produces 2 petabytes; MoPR needs automated extraction |
| 4. Proposed Solution | 200–250 words | SegFormer-B2 + DSM fusion; pipeline from orthophoto to GPKG; OGC-compliant outputs |
| 5. Uniqueness | 150–200 words | RGB+DSM fusion for India; handling India's geographic diversity; beyond-scope solar/tax attributes; TrainingDML-AI packaging |
| 6. Technology Stack | 150–200 words | MMSegmentation, rasterio, GDAL, geopandas; COG + GPKG outputs; W&B tracking; open-source throughout |
| 7. Expected Impact | 100–150 words | PM Awas Yojana targeting, property tax, solar deployment — use village statistics as evidence |
| 8. Implementation Roadmap | 150–200 words | Month 1: validate on 20 villages; Month 3: pilot 100 villages; Month 12: 3.5 lakh village pipeline |
| 9. Required Resources | 100–150 words | GPU compute, annotated data per new region, SVAMITVA drone schedule integration |
| 10. Scalability | 100–150 words | Domain adaptation across zones; state-level fine-tuning; federated model architecture |
| 11. Stakeholders | 100–150 words | MoPR, NIC, GramAnchitra, state revenue depts, solar agencies |
| 12. Development Stage | 50–100 words | Choose "Prototype" — trained model, validated outputs, reproducible pipeline |
| 13. Supporting Materials | No limit | GitHub link, W&B public report URL, QGIS screen recording link |

---

### AI-assisted workflows — Phase 4

**Prompt — solar potential attribute:**
```
I have a GeoPackage buildings layer with polygon geometries in UTM CRS, and a 
corresponding DSM raster (same CRS, values in meters above ground). Write a 
geopandas + rasterio script that:
1. For each building polygon with roof_type == 'RCC'
2. Samples the DSM within the polygon footprint
3. Estimates unshaded area by flagging cells where DSM value is within 0.5m of max (i.e., no taller obstruction nearby)
4. Classifies solar_potential as 'high' (unshaded_area > 30m²), 'medium' (15–30m²), 'low' (<15m²)
5. Writes this attribute back to the GeoPackage buildings layer
```

**Prompt — document section drafting:**
```
Draft Section 4 (Proposed Solution, 200–250 words) for a hackathon submission to 
India's Ministry of Panchayati Raj. The audience is IIT faculty + government officials.

Our approach:
- SegFormer-B2 architecture with SatMAE pretrained backbone
- 4-channel input: RGB orthophoto + normalized DSM
- 9 classes: RCC roof, tile roof, tin roof, thatched roof, pucca road, kaccha road, 
  water body, vegetation, background
- Training: Dice + Focal combined loss to handle class imbalance
- Output: COG raster masks + GPKG vector features with attributes
- Outputs are OGC-compliant and plug directly into GramAnchitra

Emphasize: reproducibility, OGC standards compliance, governance-readiness.
Do NOT mention any UI or dashboard.
```

**Prompt — per-village statistics script:**
```
Write a Python script using geopandas that reads a directory of GeoPackage files 
(one per village, each with layers: buildings, roads, water_bodies), and generates 
a summary CSV with one row per village containing:
- village_id
- total_buildings
- rcc_count, tile_count, tin_count, thatched_count, other_count
- total_road_length_km
- total_water_area_m2
- dominant_roof_type
Print a formatted table to console and save as village_statistics.csv
```

---

## 5. Phase 5 — Final QA & Submission (Day 10)

**Goal:** Everything validated, packaged, and submitted. No last-minute surprises.

---

### AI/CV Engineer tasks

- [ ] Open every test village GPKG in QGIS — verify attributes populated, no null geometries
- [ ] Run `rio cogeo validate` on all COG outputs — fix any failures
- [ ] Upload all outputs + model checkpoint to shared drive (Google Drive or institutional storage)
- [ ] Verify all shareable links work (W&B report, GitHub, drive outputs)
- [ ] Final read of all technical sections — check for metric inconsistencies
- [ ] Archive: final model `.pth`, inference config, W&B run URLs, all outputs

### Documentation & Research Lead tasks

- [ ] Final document review against all 13 sections — word counts within limits
- [ ] Confirm all OGC standard names are explicitly cited in Section 6 (COG, GPKG, TrainingDML-AI if applicable)
- [ ] Confirm GitHub repo is public, README works, sample output is accessible
- [ ] **Submit on hackathon portal** — confirm acknowledgment email received
- [ ] Cross-check submission against the official checklist below

### Final submission checklist

- [ ] Trained model checkpoint (`.pth`) with inference config
- [ ] Feature-extracted datasets: 10 test village GPKGs (`{village_id}_features.gpkg`)
- [ ] Raster outputs: 10 test village COGs (`{village_id}_segmentation.tif`)
- [ ] Per-village summary statistics CSV
- [ ] Technical documentation PDF (all 13 sections)
- [ ] Final accuracy report with per-class metrics and improvement recommendations
- [ ] W&B experiment report (public URL)
- [ ] GitHub repository (public, with README, scripts, notebooks)
- [ ] Screen recording / demo video (optional but recommended)
- [ ] TrainingDML-AI encoded training data (optional bonus)

### Checkpoint
> **Day 10 complete:** Submission portal acknowledgment received. Local archive of everything backed up.

---

## 6. Ongoing: AI-Assisted Workflow Reference

This section is a persistent reference for using AI tools throughout the project.

### Principles

1. **Specificity beats brevity.** The more precise your prompt (input format, output format, library versions, error messages), the better the output. Never paste just the last line of an error — always include the full traceback + the code that caused it.

2. **Use AI for decision latency, not just typing speed.** When unsure what to do next (which experiment to run, why a metric is dropping, what a confusion matrix means), ask. This is where time is actually saved.

3. **Maintain full context.** Share relevant background when switching topics: "We are building a segmentation model for Indian drone orthophotos, 9 classes, SegFormer-B2 architecture. We are currently debugging..." This prevents generic outputs.

4. **Iterate with results.** Paste W&B metric tables, confusion matrices, and training curves into chat for interpretation. Ask for targeted hypotheses, not general advice.

5. **Document as you go.** After each significant coding session, paste your final working script and ask: "Write a docstring + README section explaining what this script does, its inputs, outputs, and how to run it."

---

### Prompt library — quick reference

#### Environment & setup
```
I'm setting up a Python environment on Google Colab for semantic segmentation 
of aerial imagery using MMSegmentation. What pip install commands do I need 
for: MMSegmentation, rasterio, geopandas, rio-cogeo, wandb, albumentations?
List them in the correct install order to avoid conflicts.
```

#### Debugging errors
```
I'm getting this error in my segmentation training pipeline:
[PASTE FULL TRACEBACK — all lines, not just the last one]

My environment: Google Colab, Python 3.10, [list relevant packages + versions]
My relevant code: [paste 10–20 lines around the error]
What is causing this and how do I fix it?
```

#### Interpreting W&B metrics
```
My semantic segmentation model (SegFormer-B2, 9 classes, Indian drone orthophotos) 
shows these per-class IoU values after 100 epochs of training:
[PASTE YOUR METRICS TABLE]

The confusion matrix shows high confusion between: [LIST PAIRS]
What are the most likely causes for each confusion pair, 
and what specific interventions would address each one?
```

#### Config generation
```
Generate a complete MMSegmentation config for [ARCHITECTURE] with:
- Classes: [LIST]
- Input channels: [N] (RGB + [extra channels])
- Loss: [combination]
- Augmentation: [list]
- Hardware: [GPU type, VRAM]
- Dataset format: [format]
Include all required config sections: model, dataset, schedule, runtime.
```

#### Document writing
```
Draft Section [N] ([title], [word limit] words) for a government-academic 
hackathon submission. Audience: IIT faculty + Ministry of Panchayati Raj officials.

Context: [2–3 sentence description of your approach and results]
Key points to include: [bullet list]
Tone: technical but accessible, governance-oriented, no marketing language.
Do not exceed [word limit] words.
```

#### Code documentation
```
Write a complete docstring and a README.md section for the following Python script.
Include: purpose, inputs (with types and formats), outputs (with types and formats), 
dependencies, and example usage command.

[PASTE YOUR SCRIPT]
```

---

## 7. Reference Links

### Standards & formats
- COG Community: https://cogeo.org/
- OGC COG Standard: https://www.ogc.org/standards/ogc-cloud-optimized-geotiff/
- GeoPackage Community: https://www.geopackage.org/
- OGC GeoPackage Standard: https://www.ogc.org/standards/geopackage/
- OGC LAS Standard: https://www.ogc.org/standards/las/
- TrainingDML-AI (pytdml): https://github.com/openrsgis/pytdml

### Data & platform
- GramAnchitra (government data platform): https://grammanchitra.gov.in/gm4MVC
- Hackathon page: https://geo.intel.iittnif.com/activitiesinitiatives/mopr-hackathon
- Geo-Intel Lab: https://geo.intel.iittnif.com

### Tools & frameworks
- MMSegmentation: https://github.com/open-mmlab/mmsegmentation
- MMSeg Colab tutorial: https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb
- HuggingFace SegFormer: https://huggingface.co/docs/transformers/model_doc/segformer
- rio-cogeo (COG tools): https://cogeotiff.github.io/rio-cogeo/
- rasterio docs: https://rasterio.readthedocs.io/
- Label Studio (annotation): https://labelstud.io/
- Weights & Biases: https://wandb.ai/

### Relevant benchmarks / prior work
- LoveDA dataset (urban/rural aerial segmentation): https://github.com/Junjue-Wang/LoveDA
- ISPRS Vaihingen benchmark: https://www.isprs.org/education/benchmarks/UrbanSemLab/
- SpaceNet challenges: https://spacenet.ai/
- SatMAE (aerial pretrained backbone): https://github.com/sustainlab-group/SatMAE

### Contact
- Hackathon email: geointel.mopr@iittnif.com

---

*Last updated: Day 0 — pre-sprint planning*  
*Next review: End of Phase 2 (Day 5)*
