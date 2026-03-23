# Detailed Class Taxonomy for SVAMITVA Segmentation

## Overview

This document defines the 9 semantic classes used in this project for segmenting SVAMITVA drone orthophotos. Each class is accompanied by visual characteristics, examples specific to Indian rural context, annotation guidelines, and common confusion pairs.

**Resolution context**: Training data acquired at 2–5 cm GSD (ground sample distance), typical of SVAMITVA surveys. At this resolution, individual roof tiles, road surface textures, and vegetation gaps are visible but fine details (e.g., individual brick bonds) may not be.

---

## Class 0: Background

### Definition
All non-feature areas: bare soil, stones, waste ground, concrete/asphalt without deliberate human modification (e.g., open courtyards, threshing areas), rocky outcrops, and natural/disturbed earth.

### Visual Characteristics (2–5 cm GSD)
- **Color**: Brown, tan, reddish (iron oxide soils), or gray (concrete/cement dust)
- **Texture**: Uniform or granular; lacks regular pattern
- **Patterns**: No geometric structure; organic boundary with surrounding features
- **Spectral**: Low NIR reflectance (soil); high albedo if lime-treated

### Indian Rural Examples
- **Threshing grounds** (khaḷiyā) — leveled earth areas where grain is spread to dry
- **Open courtyards** (chakār) in village centers
- **Dry riverbeds** (nālī) and seasonal watercourse beds
- **Village waste areas** (kharāb zamīn)
- **Rocky slopes** in hilly villages (Himachal, Uttarakhand)
- **Concrete paving** without vegetation or structures (bus stands, cattle sheds without roofs)

### Annotation Guidelines

1. **Include**:
   - Bare soil between buildings and fields
   - Dust/cement piles waiting for construction
   - Uncovered concrete/asphalt areas
   - Rocks and stone outcrops
   - Sand banks and exposed soil

2. **Exclude**:
   - Any pixel under dense vegetation shadow
   - Water surfaces (classify as water_body)
   - Road surfaces, even if soil-colored (classify as road_kaccha if unpaved)
   - Building rubble (reddish material) if identifiable as from a roof or wall → classify by original material

3. **Minimum polygon size**: 20 pixels (~100 m² at 5 cm GSD)

4. **Edge cases**:
   - **Dusty roads**: If road is identifiable by line/path geometry → road_kaccha; if isolated dust patch → background
   - **Tilled fields**: Recently plowed → background; if crops visible → vegetation
   - **Mixed soil/gravel**: Predominantly coarse → background; if clearly compacted with rut patterns → road_kaccha

### Common Confusion Pairs

| Confused With | Visual Reason | Disambiguation |
|---------------|---------------|-----------------|
| Road (pucca/kaccha) | Similar color (both tan/brown) | Roads have linear geometry, wheel ruts; background is amorphous |
| Tin roof | Gray dust on tin looks similar | Tin has regular geometric boundary; background lacks pattern |
| Vegetation | Dark shadows under trees | Shadows show dark but bounded beneath canopy; background is lit soil |

---

## Roof Classes (1–4): RCC, Tile, Tin, Thatched

These four classes represent building structures—the most critical for SVAMITVA's property formalization mission.

### Key Differentiation Strategy

**At 2–5 cm resolution, roofs are distinguished by**:

1. **Spectral properties** (RGB): Reflectance, color saturation
2. **Surface texture**: Regularity, grain size, reflections
3. **Shadow pattern**: Flat roofs cast hard edges; sloped roofs cast gradient shadows
4. **Elevation profile**: DSM slope (flat vs. sloped)
5. **Geometry**: Rectangularity, edge sharpness

---

## Class 1: RCC_roof (Reinforced Concrete)

### Definition
Modern buildings with reinforced concrete (RCC) or Portland cement concrete slabs, typically flat or nearly flat roofs. Most common in government buildings, schools, health centers, cooperative societies, and wealthier households.

### Visual Characteristics

- **Color**: Light gray to medium gray; uniform due to cement aging
- **Texture**: Smooth, finely granular (cement surface); may show toolmarks, tar sealing
- **Reflectance**: Medium-to-high in visible spectrum; low in NIR (concrete absorbs)
- **Shadow**: Hard, sharp shadow line at roof edge (due to flat roof)
- **Geometry**: Regular rectangles, sharp corners; often with parapet walls (dark strips at roof perimeter)
- **Accessories**: Water tanks (cylindrical), solar panels (dark rectangles), communication antennas visible on top

### Indian Rural Examples

- **Government buildings**: Panchayat offices, Anganwadi centers, post offices, police stations
- **Educational institutions**: Government schools, colleges, private schools (increasingly common)
- **Health facilities**: Primary Health Centers (PHCs), Auxiliary Nurse Midwife (ANM) centers
- **Cooperative societies**: Gram Vikas Kendras, dairy cooperatives, agricultural input stores
- **Upper-income private homes**: In prosperous villages (Punjab, Haryana, parts of Maharashtra)

### Spectral Signature

```
Red   : 120–150 (medium gray)
Green : 120–150 (same as red; no hue)
Blue  : 120–150 (same as red)
DSM   : Flat profile (negligible slope), height 4–8 m
```

### Annotation Guidelines

1. **Include**:
   - All visible RCC roof surface
   - Parapet walls (dark edges) if part of roof structure
   - Tar-sealed or bituminous roof coatings (common in older RCC)
   - Slightly sloped RCC (slope < 10° for drainage)

2. **Exclude**:
   - Water tanks or structures *on top* of RCC roof (separately digitize if prominent)
   - Roof-mounted solar panels (dark; can be excluded if <10% of roof area, else mix with RCC)
   - Shadows cast *by* RCC roof onto adjacent ground
   - Interior courtyard open spaces within RCC-walled compounds

3. **Minimum polygon size**: 20 m² (moderate-sized room)

4. **Common variants**:
   - **Flat with slight crown** (convex center for drainage): Still RCC_roof
   - **Terraced RCC** (step-like appearance): Still RCC_roof
   - **RCC with terracotta coping tiles** (thin clay tiles at edges): RCC_roof (coping is negligible % of area)

### Common Confusion Pairs

| Confused With | Visual Reason | Disambiguation |
|---------------|---------------|-----------------|
| Tile_roof | Both gray; older concrete tiles look similar to cement | Tile has visible individual tile edges/lines; RCC is continuous |
| Road_pucca | Both gray; RCC roofs on ground level look like pavement | RCC is elevated (DSM shows height > 0); road is at ground level |
| Tin_roof | Gray corrugated tin can resemble dark RCC | Tin has regular corrugation ridges; RCC is smooth |
| Water (tarred) | Old tar-sealed RCC can be very dark | DSM elevation distinguishes roof (elevated) from water (ground level) |

---

## Class 2: Tile_roof (Clay/Concrete Tiles)

### Definition
Roofs made of individual clay tiles (traditional terracotta) or concrete tiles (interlocking, modern). Very common in Indian villages, especially in semi-arid and humid zones. Tiles are sloped (typically 30–45° slope) to shed water.

### Visual Characteristics

- **Color**: Red-orange to brown (clay tiles, traditional) or brown-gray (concrete tiles, modern)
- **Texture**: Highly regular ridge-and-furrow pattern from individual overlapping tiles; visible tile seams
- **Reflectance**: Medium; some specular reflection from rounded tile edges
- **Shadow**: Distinctive gradient shadow pattern due to sloped surface and tile ridges; light at high point, dark in furrows
- **Geometry**: Sloped triangular profile (ridge visible as bright line); saw-tooth shadow pattern at eaves
- **Orientation**: Ridge line runs perpendicular to primary slope (usually N–S or E–W in villages)

### Indian Rural Examples

- **Older residential houses** (30–50 year old structures in all zones)
- **Temples and religious structures** (traditional red clay tiles)
- **School buildings** (especially in semi-arid zones)
- **Agricultural buildings**: Grain stores, hay sheds, poultry houses
- **Community halls** (sāmūhik sammelān kendras)

### Spectral Signature

```
Red   : 140–180 (warm, reddish or brown)
Green : 100–130 (lower than red; warm hue)
Blue  : 80–110 (lowest; warm color)
DSM   : Sloped profile (5–45°), height increases toward ridge
Ridge visibility: Bright line along ridge crest
```

### Annotation Guidelines

1. **Include**:
   - All tile roof surface (both sunny and shaded sides of slope)
   - Ridge line and terminal edges
   - Partial tiles at roof edges (even if <50% visible)

2. **Exclude**:
   - Vegetation growing on old tile roofs (if more than 30% of pixel → reclassify as vegetation)
   - Broken/missing sections exposing rafter wood (if large: mark as separate small polygon, classify as background for now)
   - Gutters and downpipes (negligible area)
   - Chimney structures (stacks); if prominent, separate polygon

3. **Minimum polygon size**: 20 m²

4. **Edge cases**:
   - **Partially collapsed roof**: If >50% intact tiles visible → tile_roof; if <50% → background
   - **Tile roof with solar panels**: If panels cover >20% → mixed class; currently assign to tile_roof (panels not primary feature)
   - **Whitewashed tile roof** (lime-washed for heat reflection, common in Rajasthan): Still tile_roof (underlying structure unchanged)

### Common Confusion Pairs

| Confused With | Visual Reason | Disambiguation |
|---------------|---------------|-----------------|
| RCC_roof | Both brown/gray when viewed from nadir angle | Tile shows ridge-furrow texture; RCC is smooth. DSM slope >5° → tile |
| Vegetation | Old tiles with moss/lichen growth appear greenish | True vegetation has irregular, diffuse boundary; moss is continuous with tile edges |
| Tin_roof | Both have linear ridge patterns | Tile ridges are fine, <5 cm apart; tin corrugations are 10–20 cm. Tile color warmer |
| Thatched_roof | Both are sloped | Thatch is loosely structured, organic shape; tile edges are sharp and geometric |

---

## Class 3: Tin_roof (Corrugated Metal)

### Definition
Roofs made of corrugated galvanized steel or aluminum sheets, common in temporary/informal structures, older buildings, and lower-income households. Also used for agricultural shelters and boundary walls.

### Visual Characteristics

- **Color**: Silvery gray (new galvanized) to rust-red (aged, oxidized steel); aluminum is silver-white
- **Texture**: Bold, regular corrugation ridges (9–12 ridges per meter typically); highly reflective, specular reflection visible
- **Reflectance**: Very high specular reflection (shiny) when new; diffuse becomes rust-colored when aged
- **Shadow**: Regular parallel shadow stripes from corrugations; high contrast light/dark alternation
- **Geometry**: Sloped (typically 20–35° for water shedding); corrugation direction usually perpendicular to slope
- **Flapping/sound response**: Visible dents or ripples from wind load (if strong enough at 5 cm GSD)

### Indian Rural Examples

- **Informal/squatter structures** (basti areas)
- **Migrant worker shelters** and temporary camps
- **Agricultural storage**: Hay sheds, grain drying racks, poultry coops
- **Boundary walls** with tin roofing
- **Auto/mechanic shops** and small informal businesses
- **Older lower-income residential** (declining, being replaced by RCC)

### Spectral Signature

```
New Galvanized:
  Red   : 200–220 (very bright, silvery)
  Green : 200–220 (same as red)
  Blue  : 200–220 (same as red)

Aged/Rusted:
  Red   : 150–180 (reddish from oxidation)
  Green : 100–120 (lower than red)
  Blue  : 100–120 (brownish hue)

DSM: Sloped, often irregular surface (ripples/dents from wind)
```

### Annotation Guidelines

1. **Include**:
   - All visible corrugated metal sheets
   - Both new (shiny) and aged (rusty) metal
   - Temporary repairs with patches
   - Sloped or flat tin roofs (both count)

2. **Exclude**:
   - Tin boundary walls without roof function (if no roof overhead) → classify as background
   - Tin fences/railings on top of other roofs
   - Isolated corrugated sheet on ground (not overhead) → background

3. **Minimum polygon size**: 15 m² (smaller than other roof types due to less material)

4. **Edge cases**:
   - **Tin over thatch hybrid**: Common in transitional structures. If tin dominates (>50% of surface) → tin_roof; else → thatched_roof
   - **Painted tin roofs**: Tin painted green/blue still classified as tin_roof (underlying material is corrugated metal)
   - **Tin with major rust holes**: If roof is <70% continuous → background (fails to provide shelter); else → tin_roof

### Common Confusion Pairs

| Confused With | Visual Reason | Disambiguation |
|---------------|---------------|-----------------|
| Road_pucca | Both reflective, gray metallic | Tin has regular perpendicular corrugation ridges; roads lack this pattern |
| Tile_roof | Both sloped with ridges | Tile ridges are irregular, warm-colored; tin is regular, cold-gray with high specularity |
| Vegetation (mossy) | Heavily oxidized tin can look greenish | True vegetation has soft diffuse boundary; tin retains edge sharpness |
| Thatched_roof | Both informal roofing | Thatch is organic, lumpy shape; tin is geometric with bold parallel ridges |

---

## Class 4: Thatched_roof (Straw/Dried Grass)

### Definition
Roofs made of bundled straw, dried grass, palm fronds, or other organic plant material. Largely disappearing from India but still present in poorest villages and some traditional structures. High fire risk; being phased out under PM Awas Yojana.

### Visual Characteristics

- **Color**: Golden brown, straw-tan (dried), or dark brown (aged, weathered)
- **Texture**: Coarse, lumpy, fibrous appearance; uneven surface with visible thatching strands
- **Reflectance**: Low-to-medium; matte finish (no specularity)
- **Shadow**: Soft, diffuse shadow (due to irregular surface); no sharp shadow lines
- **Geometry**: Sloped (steep, 40–60° for rapid water shedding of porous material); irregular outline with "fuzzy" edges from protruding straw
- **Edge definition**: Boundary is soft, ragged at eaves; not sharp like tile or metal
- **Surface variation**: Visible clumps, bundles, and repair patches from seasonal maintenance

### Indian Rural Examples

- **Very low-income households** (Dalit bastis, landless laborer dwellings)
- **Remote/tribal villages** (Jharkhand, Odisha, Chhattisgarh)
- **Temporary seasonal shelters** (migrant camps, monsoon recovery shelters)
- **Agricultural structures**: Hay storage, temporary field huts for guarding crops
- **Some temples in traditional zones** (though being replaced)

### Spectral Signature

```
Golden (Fresh):
  Red   : 160–190 (warm tan)
  Green : 130–150 (lower than red)
  Blue  : 80–100 (much lower; golden hue)

Aged/Dark (Weathered):
  Red   : 80–100 (dark brown)
  Green : 60–80
  Blue  : 50–70

DSM: Steep slope (40–60°), highest variability due to uneven thatch layer
Textural roughness: High; surface relief visible at 5 cm GSD
```

### Annotation Guidelines

1. **Include**:
   - All thatched roof surface
   - Recent patches and repair sections (may be lighter colored)
   - Moss or lichen growth on thatch (still thatched_roof)
   - Both overhanging eaves and main roof body

2. **Exclude**:
   - Underlying wooden rafters if thatch is <50% (if roof is failing) → classify as background
   - Smoke vent holes (small)
   - Straw stored *next to* building but not overhead → vegetation/background
   - Palm trees next to thatched house → vegetation

3. **Minimum polygon size**: 15 m²

4. **Edge cases**:
   - **Thatch + tin hybrid** (common in modern upgrades): If tin component covers >50% → tin_roof; else → thatched_roof
   - **Mostly collapsed thatch**: If <30% coverage remains → background; if 30–70% → thatched_roof; if >70% → thatched_roof
   - **Whitewashed thatch**: Rare but possible (lime wash applied for insulation); classify by underlying structure → thatched_roof

### Common Confusion Pairs

| Confused With | Visual Reason | Disambiguation |
|---------------|---------------|-----------------|
| Vegetation | Both tan/brown and fibrous | Thatch has sharp roof geometric boundary; vegetation spreads diffusely beyond building perimeter |
| Background | Weathered thatch can be very dark | Thatch shows roof geometry and organized slope; background is amorphous |
| Tin_roof | Both used in informal structures | Tin has bold regular corrugation ridges; thatch is organic and lumpy |
| Tile_roof (moss-covered) | Both can be brown with texture | Tile has regular overlapping geometry; thatch is loosely fibrous, frayed edges |

---

## Road Classes (5–6): Pucca and Kaccha

Roads and paths are critical infrastructure in SVAMITVA, enabling mobility and commerce in villages.

---

## Class 5: Road_pucca (Paved/All-Weather Road)

### Definition
Asphalted, concrete, or well-maintained compacted roads designed to withstand monsoons and heavy traffic. Includes village main streets, government roads, and recently constructed panchayat roads.

### Visual Characteristics

- **Color**: Gray (asphalt or concrete), sometimes dark gray to black (asphalt when wet or aged)
- **Texture**: Smooth or finely textured; no coarse grain
- **Reflectance**: Medium (asphalt is darker than concrete); some specular reflection, especially when wet
- **Linearity**: Straight lines or gentle curves; well-defined edges (road boundary is sharp)
- **Markings**: May show lane markings, white/yellow stripes (less common in villages)
- **Width**: Typically 4–8 m for village main roads; 2–4 m for secondary roads
- **Drainage**: Often has camber (slight curve) visible as shading gradient; edge delineation clear
- **Condition**: Surface appears whole; cracks may be visible but road remains continuous

### Indian Rural Examples

- **Government roads** (NH, State Highways passing through villages)
- **Panchayat-constructed roads** (PMGSY — Pradhan Mantri Gram Sadak Yojana funded)
- **Recently constructed approach roads** (SVAMITVA-era improvements)
- **School/health center access roads**
- **Market square pavement**

### Spectral Signature

```
Asphalt:
  Red   : 60–100 (dark gray)
  Green : 60–100
  Blue  : 60–100

Concrete:
  Red   : 120–160 (light gray)
  Green : 120–160
  Blue  : 120–160

DSM: Flat (elevation ~0 slope for drainage <2°)
Width: Consistent, 2–8 m
```

### Annotation Guidelines

1. **Include**:
   - All asphalt or concrete road surface
   - Road markings (white/yellow stripes)
   - Camber (sloped surface for drainage)
   - Worn or slightly patched asphalt (if still continuous)
   - Concrete interlock/permeable paving roads (increasing in villages)

2. **Exclude**:
   - Shadows cast by trees on the road → exclude shadow area (classify as background where shadow falls)
   - Parked vehicles → exclude vehicle footprint; if vehicle is parked long-term (structural) → include
   - Puddles/water on road → water_body if clearly standing water; if just wet surface → road_pucca
   - Road markings only (no asphalt beneath) → exclude from road_pucca if isolated

3. **Minimum polygon size**: Continuous road segment ≥50 m or ≥20 m² (whichever applies)

4. **Edge cases**:
   - **Road under construction** (partially laid): If <50% laid → background; if ≥50% laid but unpaved → road_kaccha; if ≥50% paved → road_pucca
   - **Cracked asphalt with weeds**: If <20% vegetation → road_pucca; if ≥20% vegetation → mixed (currently assign to road_pucca)
   - **Concrete slabs with gaps**: If gaps are minor and road is continuous → road_pucca; if gaps are major → consider road_kaccha

### Common Confusion Pairs

| Confused With | Visual Reason | Disambiguation |
|---------------|---------------|-----------------|
| Road_kaccha | Both are roads; color can overlap | Pucca is uniform gray/black with sharp edge; kaccha is tan/brown, fuzzy boundary |
| RCC_roof | Both gray | RCC is elevated (DSM); road is ground-level (DSM ≈ 0) |
| Background | Both can be gray | Road has linear geometry, consistent width; background is amorphous |
| Tin_roof | Both reflective and gray | Tin has perpendicular corrugation ridges; road lacks this |
| Water (wet roads) | Wet asphalt appears very dark | Water pixels show standing water edges; wet road is still linear |

---

## Class 6: Road_kaccha (Unpaved/Earth Road)

### Definition
Unpaved village paths and roads made of compacted earth, gravel, or sand. Impassable during monsoons; common as secondary village paths connecting homes and fields.

### Visual Characteristics

- **Color**: Tan, brown, reddish (varied soil colors); lighter than asphalt, darker than background soil
- **Texture**: Granular; shows wheel ruts, footprint marks, and surface roughness
- **Reflectance**: Low-to-medium (similar to background soil but slightly darker from compaction and moisture)
- **Linearity**: Visible as path, often with dual wheel ruts or centerline worn smooth
- **Geometry**: Irregular width and wavering path (not as straight as pucca roads); fuzzy boundaries with adjacent soil
- **Drainage**: Often shows grooves/channels from water runoff; may be elevated as embankment
- **Seasonal state**: Mud/wet in monsoon; dust-covered in dry season

### Indian Rural Examples

- **Inter-village connectors** (between hamlets, not yet paved)
- **Field access paths** (kisān panthāvāl)
- **Cattle tracks** (pasture access routes)
- **Secondary settlement roads** (within slums or scattered housing)
- **Seasonal paths** (appear/disappear based on monsoon)
- **Cart tracks** (traditional bullock cart routes)

### Spectral Signature

```
Dry Season:
  Red   : 120–160 (tan, dusty)
  Green : 100–140 (slightly lower than red)
  Blue  : 80–120 (brownish hue)

Wet/Monsoon:
  Red   : 80–120 (darker, mud-colored)
  Green : 70–110
  Blue  : 60–100

DSM: Flat or slight crown (≤3° slope); surface may show ruts (5–10 cm relief)
Width: Variable, 1–4 m (narrower than pucca roads)
```

### Annotation Guidelines

1. **Include**:
   - Compacted earth/gravel paths
   - Paths with visible wheel ruts
   - Temporary mud roads (even if wet)
   - Elevated embankment paths (if primarily unpaved)
   - Stone/pebble filled roads (unpaved, permeable)

2. **Exclude**:
   - Vegetated field boundaries (even if used as paths) → vegetation
   - Pure background soil (no evidence of compaction/regular traffic)
   - Shadows on roads → exclude shadow
   - Standing water on road → water_body
   - Heavily vegetated paths (if vegetation covers >50%) → vegetation

3. **Minimum polygon size**: 15 m² or 30 m continuous segment

4. **Edge cases**:
   - **Path transitioning to pucca**: Partially graveled/partially paved → assign majority class
   - **Wet road with mud splashes**: If central wear line visible → road_kaccha; if uniformly muddy → background
   - **Recently paved road**: If asphalt is fresh and shiny → road_pucca; if top layer is only 10% coverage → background/kaccha

### Common Confusion Pairs

| Confused With | Visual Reason | Disambiguation |
|---------------|---------------|-----------------|
| Background | Both tan/brown | Road shows linear continuity and wheel ruts; background lacks pattern |
| Road_pucca | Both are roads | Pucca is uniform gray/hard edge; kaccha is tan/brown/fuzzy boundary |
| Vegetation (path through crops) | Worn crop area from walking | Crop area has diffuse green vegetation; worn path shows brown soil |
| Water (mud puddles) | Muddy surface can be very dark | True water is reflective, flat, and pooled; mud is matte and dispersed |

---

## Class 7: Water_body (Water Features)

### Definition
Permanent and seasonal water bodies: tanks, ponds, irrigation channels, streams, and wetlands. Critical for assessing water resources and flood risk.

### Visual Characteristics

- **Color**: Dark blue to black (water absorbs visible light); murky green-brown (algae-rich or silt-laden water)
- **Reflectance**: Very low visible reflectance (water absorbs); high NIR reflectance if water is clear (due to bottom reflection); low NIR if turbid
- **Specularity**: Calm water shows specular reflection (bright spots if sun angle favorable); rough water is diffuse
- **Boundary**: Sharp edge between water and land (except marshy areas with gradual transition)
- **Vegetation**: Aquatic plants (reeds, lotus, water hyacinth) at margins; emergent vegetation counted as vegetation, not water
- **Bottom**: Visible at shallow edges (<1 m); deeper water appears uniformly dark

### Indian Rural Examples

- **Village tanks** (pokhar, vāv) — main water harvesting structures
- **Irrigation channels** (nālā) and distributaries
- **Seasonal streams** (nazdik) flowing during monsoon
- **Ponds** (johad) for cattle/livestock
- **Wetlands** (kkār) and marshes (in humid zones)
- **Artificially created ponds** (recharge tanks for groundwater)
- **Rice paddies during crop season** (temporary water)

### Spectral Signature

```
Clean Water:
  Red   : 10–50 (very dark)
  Green : 10–50
  Blue  : 20–80 (slightly higher in visible range)

Turbid/Algae-Rich:
  Red   : 40–80 (greenish-brown)
  Green : 50–100 (higher than red from algae)
  Blue  : 30–70

DSM: Flat surface (water always horizontal at ~0 slope)
NIR (if available): Very low (clear water); medium (turbid water with bottom reflection)
```

### Annotation Guidelines

1. **Include**:
   - All water surface pixels (tanks, ponds, streams, channels)
   - Seasonal water bodies (even if temporary)
   - Turbid/algae-covered water
   - Water with visible emergent reeds (if the reedbase is in water)
   - Shallow water where bottom is visible

2. **Exclude**:
   - Aquatic vegetation (water hyacinth, lotus leaves) → if >30% of visible pixel, classify as vegetation
   - Margins with terrestrial vegetation (reed edges, grass) → vegetation
   - Shadows cast by trees on water → exclude shadow area (water below)
   - Muddy/slushy areas at water edge (not standing water) → background
   - Puddles on roads (if <1 m² and on a road) → road class

3. **Minimum polygon size**: 10 m² (smaller threshold for water due to importance)

4. **Edge cases**:
   - **Partially dry tank** (exposing muddy bottom): If water component is >50% → water_body; if <50% → background
   - **Water with surface scum**: Still water_body (scum is algae film, negligible thickness)
   - **Flooded field during monsoon**: If rainwater pooling on crops → water_body; if crops still visible beneath shallow water → ambiguous (assign to water_body if depth >5 cm estimated)
   - **Construction site with stagnant water**: Still water_body (even if undesired)

### Common Confusion Pairs

| Confused With | Visual Reason | Disambiguation |
|---------------|---------------|-----------------|
| Background (wet/dark soil) | Both very dark | Water has sharp boundary and reflective edge; wet soil is matte and gradual transition |
| Tin_roof (wet/reflective) | Both dark and potentially reflective | Water is always flat and horizontal; tin shows sloped geometry and corrugation |
| Road_pucca (wet asphalt) | Both dark and potentially reflective | Water is pooled with flat surface; wet road shows linear directional geometry |
| Vegetation (algae/water plants) | Water hyacinth and algae are greenish | Pure water is blue/black; vegetated water is green (classify green pixels as vegetation) |
| Shadow (under tree canopy) | Both very dark | Water has liquid reflectance at edges; shadow is matte and not bounded |

---

## Class 8: Vegetation (Green Cover)

### Definition
All visible green vegetation: trees, shrubs, crops, grasses, and any natural or cultivated plant cover. Includes forests, orchards, agricultural fields, and grasslands.

### Visual Characteristics

- **Color**: Green (due to chlorophyll absorption of red light and reflection of NIR and green light); ranges from bright lime (irrigated crops, tender leaves) to dark olive (forests, mature leaves)
- **Texture**: Fine (grain crops, turf), medium (shrubs, small trees), coarse (forest canopy, large trees)
- **Pattern**: Individual tree crowns visible as circular/elliptical shapes; crop fields show field-scale geometry; forests show canopy texture
- **Reflectance**: High NIR reflectance (50–70% of vegetated pixels); moderate visible reflectance
- **Shadows**: Under-canopy shadows visible in forests; individual tree shadows cast on ground
- **Leaf moisture**: Wet vegetation (just after rain or irrigation) appears darker but still green

### Indian Rural Examples

- **Cultivated crop fields**: Rice (monsoon), wheat (winter), sugarcane, cotton
- **Orchards**: Mango, coconut, guava, citrus groves
- **Tree-based systems**: Agroforestry, shelterbelt trees, boundary plantations
- **Common lands**: Pasture, community forests, village woodlots
- **Homestead vegetation**: Trees around houses, gardens
- **Urban/peri-urban vegetation**: Park, avenue trees

### Spectral Signature

```
Healthy/Irrigated Crops:
  Red   : 50–100 (low; absorbed by chlorophyll)
  Green : 100–150 (higher than red due to green light reflection)
  Blue  : 50–100
  NDVI  ≈ 0.5–0.8 (if multispectral available)

Dry/Stressed Vegetation:
  Red   : 80–120 (higher, less chlorophyll)
  Green : 100–140
  Blue  : 80–120
  NDVI  ≈ 0.3–0.5

Forest Canopy:
  Red   : 30–60 (very low; dense canopy absorption)
  Green : 40–80
  Blue  : 20–60
  Very rough texture from canopy heterogeneity
```

### Annotation Guidelines

1. **Include**:
   - All pixels with green vegetation (any % cover if color is distinctly green)
   - Individual tree crowns (even isolated trees)
   - Crop fields and pastures
   - Forest canopy and understory green
   - Shrubs and saplings
   - Grass verges and median strips on roads (if green)
   - Green roofs/living walls (if visible)

2. **Exclude**:
   - Dead/brown vegetation (standing dry stalks) → background
   - Tree trunks and exposed wood (visible in sparse forest) → background if no green visible
   - Shadows beneath trees (dark area) → unless pixels are visibly green → vegetation
   - Fences and support structures for vines (only if vines visible and green)
   - Dried hay and straw (brown) → background

3. **Minimum polygon size**: 10 m² (smaller for small trees and shrubs)

4. **Edge cases**:
   - **Crops ready for harvest** (golden/brown but still rooted): If any green is visible → vegetation; if fully brown/dry → background
   - **Newly sprouted crops** (bright green, sparse): Even if sparse → vegetation
   - **Vegetation on roof** (moss/lichens on thatch or tile): If green pixels are >10% → vegetation; else → original roof class
   - **Mixed urban greening** (garden amidst buildings): Individual trees and patches → vegetation

### Common Confusion Pairs

| Confused With | Visual Reason | Disambiguation |
|---------------|---------------|-----------------|
| Background | Dry vegetation can be brownish | Dry but living trees/crops have structural continuity; true background is amorphous |
| Water (algae) | Algae-rich water appears greenish | Pure vegetation shows terrestrial plant morphology (distinct tree crowns, linear rows); algae is diffuse |
| Shadow (under canopy) | Both dark in forest | Vegetation under-canopy shows green hue; shadow is grayscale dark |
| Thatched_roof (aged, green) | Old thatch with moss appears greenish-brown | Thatch shows roof geometry (slopes to edge); vegetation spreads beyond roof perimeter |
| Tile_roof (with moss) | Moss-covered tiles appear brownish-green | Tile shows regular geometric pattern of overlaps; mossy vegetation is diffuse |

---

## Annotation Quality Standards

### General Requirements

1. **Boundary Precision**
   - Object boundaries must be digitized to ±1 pixel (~5 cm at 5 cm GSD)
   - No self-intersecting polygons
   - Closed rings with consistent direction (counterclockwise for outer boundary)

2. **Minimum Polygon Size**
   - Standard minimum: 20 pixels (~100 m² at 5 cm GSD)
   - Water bodies: 10 pixels (due to importance for hydrology)
   - Tin roofs, small structures: 15 pixels
   - Rationale: Smaller polygons are unreliable and difficult to validate in field

3. **Sliver Removal**
   - Remove slivers (polygons with area <20 m² or aspect ratio >10:1)
   - Merge slivers with adjacent larger polygon of same class

4. **No "NoData" Classes**
   - Every pixel must be assigned to one of the 9 classes
   - Ambiguous pixels (e.g., 50% roof/50% shadow): Assign to roof if roofing material is identifiable

5. **Attribute Data**
   - class_id: Integer 0–8
   - class_name: Text (standardized spelling)
   - confidence: [0–1] flag if known from annotator notes

### Per-Tile Validation Checklist

Before finalizing a village's annotation:

- [ ] All 9 classes are represented (if present in village)
- [ ] No "salt-and-pepper" noise (isolated single-pixel anomalies)
- [ ] Building shadows not misclassified as separate structures
- [ ] Roads are continuous (not fragmented into separate polygons)
- [ ] Water bodies are connected (no false fragmentation)
- [ ] Vegetation is grouped by tree/field, not pixelated
- [ ] Edge of orthophoto is classified (no "unknown" margins)
- [ ] Total coverage = 100% of orthophoto area (no gaps)

### Inter-Rater Agreement Protocol

When multiple annotators work on the same village:

1. **Calculation**: Cohen's κ (kappa) for per-class agreement
2. **Threshold**: κ ≥ 0.75 required; if lower, reconcile discrepancies
3. **Reconciliation method**: Third-party review; if still disagreed, reclassify to majority class or simpler class (e.g., ambiguous roof → background)

---

## Common Annotation Errors & Corrections

| Error | Example | Fix |
|-------|---------|-----|
| Over-segmentation | Each roof tile digitized separately | Merge into single roof_* polygon |
| Under-segmentation | Entire street digitized as one polygon | Split by road type (pucca vs kaccha sections) |
| Boundary sliver | Thin strip of wrong class along road edge | Merge with road or adjacent larger polygon |
| Shadow misclassification | Tree shadow marked as separate object | Reassign to ground class (background/road/vegetation) |
| Seasonal confusion | Rainwater puddle marked as permanent water | Verify in multiple-date imagery; assign to background if temporary |
| Partial structure | Edge-of-image building, 50% visible | Include if >50% of structure is visible; exclude if <50% |

---

## References & Training Materials

1. **SVAMITVA Guidelines**: Ministry of Panchayati Raj official documentation
2. **Ground Truth Field Visits**: Photos of annotated structures for validator training
3. **Inter-Rater Agreement Study**: Cohen's κ results from pilot annotations
4. **Confusion Matrix**: Detailed breakdown of common misclassifications from model training

---

**Document Version**: 1.0
**Last Updated**: March 2026
**Status**: Finalized — Ready for annotation and model training
