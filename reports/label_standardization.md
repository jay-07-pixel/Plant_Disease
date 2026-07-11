# Label Standardization Report

**Generated:** 2026-07-11T03:51:51.286406+00:00  
**Label source:** `reports\dataset_statistics.json`  
**Datasets:** 2  
**Raw labels:** 65  
**Mapped labels:** 65  
**Canonical labels:** 43  

> Read-only label mapping. No images or datasets were modified.

## Universal Label Schema

Every raw class label is mapped to:

| Field | Description | Example |
|-------|-------------|---------|
| `plant` | Canonical crop name | `Tomato` |
| `disease` | Canonical disease name | `Early Blight` |
| `is_healthy` | Healthy sample flag | `false` |
| `canonical_key` | Cross-dataset slug | `tomato|early_blight` |

## Mapping Examples

### PlantVillage

| Raw Label | Standard Label |
|-----------|----------------|
| `Tomato___Early_blight` | plant=`Tomato`, disease=`Early Blight`, healthy=No |
| `Apple___healthy` | plant=`Apple`, disease=`Healthy`, healthy=Yes |

### PlantDoc

| Raw Label | Standard Label |
|-----------|----------------|
| `Tomato Early blight leaf` | plant=`Tomato`, disease=`Early Blight`, healthy=No |
| `Tomato leaf` | plant=`Tomato`, disease=`Healthy`, healthy=Yes |

## Issue Summary

- **Unknown labels:** 0
- **Ambiguous labels:** 0
- **Duplicate mappings:** 22
- **Unmapped labels:** 0

## Unknown Labels

None.

## Ambiguous Labels

None.

## Duplicate Mappings

- **`cross-dataset`** / `apple|apple_scab` — 2 raw labels map to canonical key 'apple|apple_scab'.
  - Candidates: `plantdoc:Apple Scab Leaf`, `plantvillage:Apple___Apple_scab`
- **`cross-dataset`** / `apple|healthy` — 2 raw labels map to canonical key 'apple|healthy'.
  - Candidates: `plantdoc:Apple leaf`, `plantvillage:Apple___healthy`
- **`cross-dataset`** / `blueberry|healthy` — 2 raw labels map to canonical key 'blueberry|healthy'.
  - Candidates: `plantdoc:Blueberry leaf`, `plantvillage:Blueberry___healthy`
- **`cross-dataset`** / `cherry|healthy` — 2 raw labels map to canonical key 'cherry|healthy'.
  - Candidates: `plantdoc:Cherry leaf`, `plantvillage:Cherry_(including_sour)___healthy`
- **`cross-dataset`** / `grape|black_rot` — 2 raw labels map to canonical key 'grape|black_rot'.
  - Candidates: `plantdoc:grape leaf black rot`, `plantvillage:Grape___Black_rot`
- **`cross-dataset`** / `grape|healthy` — 2 raw labels map to canonical key 'grape|healthy'.
  - Candidates: `plantdoc:grape leaf`, `plantvillage:Grape___healthy`
- **`cross-dataset`** / `peach|healthy` — 2 raw labels map to canonical key 'peach|healthy'.
  - Candidates: `plantdoc:Peach leaf`, `plantvillage:Peach___healthy`
- **`cross-dataset`** / `pepper|healthy` — 2 raw labels map to canonical key 'pepper|healthy'.
  - Candidates: `plantdoc:Bell_pepper leaf`, `plantvillage:Pepper,_bell___healthy`
- **`cross-dataset`** / `potato|early_blight` — 2 raw labels map to canonical key 'potato|early_blight'.
  - Candidates: `plantdoc:Potato leaf early blight`, `plantvillage:Potato___Early_blight`
- **`cross-dataset`** / `potato|late_blight` — 2 raw labels map to canonical key 'potato|late_blight'.
  - Candidates: `plantdoc:Potato leaf late blight`, `plantvillage:Potato___Late_blight`
- **`cross-dataset`** / `raspberry|healthy` — 2 raw labels map to canonical key 'raspberry|healthy'.
  - Candidates: `plantdoc:Raspberry leaf`, `plantvillage:Raspberry___healthy`
- **`cross-dataset`** / `soybean|healthy` — 2 raw labels map to canonical key 'soybean|healthy'.
  - Candidates: `plantdoc:Soyabean leaf`, `plantvillage:Soybean___healthy`
- **`cross-dataset`** / `squash|powdery_mildew` — 2 raw labels map to canonical key 'squash|powdery_mildew'.
  - Candidates: `plantdoc:Squash Powdery mildew leaf`, `plantvillage:Squash___Powdery_mildew`
- **`cross-dataset`** / `strawberry|healthy` — 2 raw labels map to canonical key 'strawberry|healthy'.
  - Candidates: `plantdoc:Strawberry leaf`, `plantvillage:Strawberry___healthy`
- **`cross-dataset`** / `tomato|bacterial_spot` — 2 raw labels map to canonical key 'tomato|bacterial_spot'.
  - Candidates: `plantdoc:Tomato leaf bacterial spot`, `plantvillage:Tomato___Bacterial_spot`
- **`cross-dataset`** / `tomato|early_blight` — 2 raw labels map to canonical key 'tomato|early_blight'.
  - Candidates: `plantdoc:Tomato Early blight leaf`, `plantvillage:Tomato___Early_blight`
- **`cross-dataset`** / `tomato|healthy` — 2 raw labels map to canonical key 'tomato|healthy'.
  - Candidates: `plantdoc:Tomato leaf`, `plantvillage:Tomato___healthy`
- **`cross-dataset`** / `tomato|late_blight` — 2 raw labels map to canonical key 'tomato|late_blight'.
  - Candidates: `plantdoc:Tomato leaf late blight`, `plantvillage:Tomato___Late_blight`
- **`cross-dataset`** / `tomato|leaf_mold` — 2 raw labels map to canonical key 'tomato|leaf_mold'.
  - Candidates: `plantdoc:Tomato mold leaf`, `plantvillage:Tomato___Leaf_Mold`
- **`cross-dataset`** / `tomato|septoria_leaf_spot` — 2 raw labels map to canonical key 'tomato|septoria_leaf_spot'.
  - Candidates: `plantdoc:Tomato Septoria leaf spot`, `plantvillage:Tomato___Septoria_leaf_spot`
- **`cross-dataset`** / `tomato|tomato_mosaic_virus` — 2 raw labels map to canonical key 'tomato|tomato_mosaic_virus'.
  - Candidates: `plantdoc:Tomato leaf mosaic virus`, `plantvillage:Tomato___Tomato_mosaic_virus`
- **`cross-dataset`** / `tomato|tomato_yellow_leaf_curl_virus` — 2 raw labels map to canonical key 'tomato|tomato_yellow_leaf_curl_virus'.
  - Candidates: `plantdoc:Tomato leaf yellow virus`, `plantvillage:Tomato___Tomato_Yellow_Leaf_Curl_Virus`

## Unmapped Labels

None.

## Canonical Labels

- `apple|apple_scab` → plant=`Apple`, disease=`Apple Scab`, healthy=No
- `apple|black_rot` → plant=`Apple`, disease=`Black Rot`, healthy=No
- `apple|cedar_apple_rust` → plant=`Apple`, disease=`Cedar Apple Rust`, healthy=No
- `apple|healthy` → plant=`Apple`, disease=`Healthy`, healthy=Yes
- `apple|rust` → plant=`Apple`, disease=`Rust`, healthy=No
- `blueberry|healthy` → plant=`Blueberry`, disease=`Healthy`, healthy=Yes
- `cherry|healthy` → plant=`Cherry`, disease=`Healthy`, healthy=Yes
- `cherry|powdery_mildew` → plant=`Cherry`, disease=`Powdery Mildew`, healthy=No
- `corn|blight` → plant=`Corn`, disease=`Blight`, healthy=No
- `corn|cercospora_leaf_spot_gray_leaf_spot` → plant=`Corn`, disease=`Cercospora Leaf Spot Gray Leaf Spot`, healthy=No
- `corn|common_rust` → plant=`Corn`, disease=`Common Rust`, healthy=No
- `corn|gray_leaf_spot` → plant=`Corn`, disease=`Gray Leaf Spot`, healthy=No
- `corn|healthy` → plant=`Corn`, disease=`Healthy`, healthy=Yes
- `corn|northern_leaf_blight` → plant=`Corn`, disease=`Northern Leaf Blight`, healthy=No
- `corn|rust` → plant=`Corn`, disease=`Rust`, healthy=No
- `grape|black_rot` → plant=`Grape`, disease=`Black Rot`, healthy=No
- `grape|esca_(black_measles)` → plant=`Grape`, disease=`Esca (Black Measles)`, healthy=No
- `grape|healthy` → plant=`Grape`, disease=`Healthy`, healthy=Yes
- `grape|leaf_blight_(isariopsis_leaf_spot)` → plant=`Grape`, disease=`Leaf Blight (Isariopsis Leaf Spot)`, healthy=No
- `orange|huanglongbing_(citrus_greening)` → plant=`Orange`, disease=`Huanglongbing (Citrus Greening)`, healthy=No
- `peach|bacterial_spot` → plant=`Peach`, disease=`Bacterial Spot`, healthy=No
- `peach|healthy` → plant=`Peach`, disease=`Healthy`, healthy=Yes
- `pepper|bacterial_spot` → plant=`Pepper`, disease=`Bacterial Spot`, healthy=No
- `pepper|healthy` → plant=`Pepper`, disease=`Healthy`, healthy=Yes
- `pepper|leaf_spot` → plant=`Pepper`, disease=`Leaf Spot`, healthy=No
- `potato|early_blight` → plant=`Potato`, disease=`Early Blight`, healthy=No
- `potato|healthy` → plant=`Potato`, disease=`Healthy`, healthy=Yes
- `potato|late_blight` → plant=`Potato`, disease=`Late Blight`, healthy=No
- `raspberry|healthy` → plant=`Raspberry`, disease=`Healthy`, healthy=Yes
- `soybean|healthy` → plant=`Soybean`, disease=`Healthy`, healthy=Yes
- `squash|powdery_mildew` → plant=`Squash`, disease=`Powdery Mildew`, healthy=No
- `strawberry|healthy` → plant=`Strawberry`, disease=`Healthy`, healthy=Yes
- `strawberry|leaf_scorch` → plant=`Strawberry`, disease=`Leaf Scorch`, healthy=No
- `tomato|bacterial_spot` → plant=`Tomato`, disease=`Bacterial Spot`, healthy=No
- `tomato|early_blight` → plant=`Tomato`, disease=`Early Blight`, healthy=No
- `tomato|healthy` → plant=`Tomato`, disease=`Healthy`, healthy=Yes
- `tomato|late_blight` → plant=`Tomato`, disease=`Late Blight`, healthy=No
- `tomato|leaf_mold` → plant=`Tomato`, disease=`Leaf Mold`, healthy=No
- `tomato|septoria_leaf_spot` → plant=`Tomato`, disease=`Septoria Leaf Spot`, healthy=No
- `tomato|spider_mites_two_spotted_spider_mite` → plant=`Tomato`, disease=`Spider Mites Two Spotted Spider Mite`, healthy=No
- `tomato|target_spot` → plant=`Tomato`, disease=`Target Spot`, healthy=No
- `tomato|tomato_mosaic_virus` → plant=`Tomato`, disease=`Tomato Mosaic Virus`, healthy=No
- `tomato|tomato_yellow_leaf_curl_virus` → plant=`Tomato`, disease=`Tomato Yellow Leaf Curl Virus`, healthy=No

## Dataset: `plantdoc`

- Raw labels: 27
- Mapped: 27

| Raw Label | Plant | Disease | Healthy | Status | Images |
|-----------|-------|---------|---------|--------|--------|
| `Apple Scab Leaf` | Apple | Apple Scab | No | mapped | 87 |
| `Apple leaf` | Apple | Healthy | Yes | mapped | 91 |
| `Apple rust leaf` | Apple | Rust | No | mapped | 89 |
| `Bell_pepper leaf` | Pepper | Healthy | Yes | mapped | 61 |
| `Bell_pepper leaf spot` | Pepper | Leaf Spot | No | mapped | 71 |
| `Blueberry leaf` | Blueberry | Healthy | Yes | mapped | 116 |
| `Cherry leaf` | Cherry | Healthy | Yes | mapped | 57 |
| `Corn Gray leaf spot` | Corn | Gray Leaf Spot | No | mapped | 68 |
| `Corn leaf blight` | Corn | Blight | No | mapped | 192 |
| `Corn rust leaf` | Corn | Rust | No | mapped | 116 |
| `Peach leaf` | Peach | Healthy | Yes | mapped | 112 |
| `Potato leaf early blight` | Potato | Early Blight | No | mapped | 117 |
| `Potato leaf late blight` | Potato | Late Blight | No | mapped | 105 |
| `Raspberry leaf` | Raspberry | Healthy | Yes | mapped | 116 |
| `Soyabean leaf` | Soybean | Healthy | Yes | mapped | 65 |
| `Squash Powdery mildew leaf` | Squash | Powdery Mildew | No | mapped | 129 |
| `Strawberry leaf` | Strawberry | Healthy | Yes | mapped | 96 |
| `Tomato Early blight leaf` | Tomato | Early Blight | No | mapped | 83 |
| `Tomato Septoria leaf spot` | Tomato | Septoria Leaf Spot | No | mapped | 148 |
| `Tomato leaf` | Tomato | Healthy | Yes | mapped | 62 |
| `Tomato leaf bacterial spot` | Tomato | Bacterial Spot | No | mapped | 107 |
| `Tomato leaf late blight` | Tomato | Late Blight | No | mapped | 111 |
| `Tomato leaf mosaic virus` | Tomato | Tomato Mosaic Virus | No | mapped | 54 |
| `Tomato leaf yellow virus` | Tomato | Tomato Yellow Leaf Curl Virus | No | mapped | 75 |
| `Tomato mold leaf` | Tomato | Leaf Mold | No | mapped | 91 |
| `grape leaf` | Grape | Healthy | Yes | mapped | 69 |
| `grape leaf black rot` | Grape | Black Rot | No | mapped | 64 |

## Dataset: `plantvillage`

- Raw labels: 38
- Mapped: 38

| Raw Label | Plant | Disease | Healthy | Status | Images |
|-----------|-------|---------|---------|--------|--------|
| `Apple___Apple_scab` | Apple | Apple Scab | No | mapped | 630 |
| `Apple___Black_rot` | Apple | Black Rot | No | mapped | 621 |
| `Apple___Cedar_apple_rust` | Apple | Cedar Apple Rust | No | mapped | 275 |
| `Apple___healthy` | Apple | Healthy | Yes | mapped | 1645 |
| `Blueberry___healthy` | Blueberry | Healthy | Yes | mapped | 1502 |
| `Cherry_(including_sour)___Powdery_mildew` | Cherry | Powdery Mildew | No | mapped | 1052 |
| `Cherry_(including_sour)___healthy` | Cherry | Healthy | Yes | mapped | 854 |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Corn | Cercospora Leaf Spot Gray Leaf Spot | No | mapped | 513 |
| `Corn_(maize)___Common_rust_` | Corn | Common Rust | No | mapped | 1192 |
| `Corn_(maize)___Northern_Leaf_Blight` | Corn | Northern Leaf Blight | No | mapped | 985 |
| `Corn_(maize)___healthy` | Corn | Healthy | Yes | mapped | 1162 |
| `Grape___Black_rot` | Grape | Black Rot | No | mapped | 1180 |
| `Grape___Esca_(Black_Measles)` | Grape | Esca (Black Measles) | No | mapped | 1383 |
| `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Grape | Leaf Blight (Isariopsis Leaf Spot) | No | mapped | 1076 |
| `Grape___healthy` | Grape | Healthy | Yes | mapped | 423 |
| `Orange___Haunglongbing_(Citrus_greening)` | Orange | Huanglongbing (Citrus Greening) | No | mapped | 5507 |
| `Peach___Bacterial_spot` | Peach | Bacterial Spot | No | mapped | 2297 |
| `Peach___healthy` | Peach | Healthy | Yes | mapped | 360 |
| `Pepper,_bell___Bacterial_spot` | Pepper | Bacterial Spot | No | mapped | 997 |
| `Pepper,_bell___healthy` | Pepper | Healthy | Yes | mapped | 1478 |
| `Potato___Early_blight` | Potato | Early Blight | No | mapped | 1000 |
| `Potato___Late_blight` | Potato | Late Blight | No | mapped | 1000 |
| `Potato___healthy` | Potato | Healthy | Yes | mapped | 152 |
| `Raspberry___healthy` | Raspberry | Healthy | Yes | mapped | 371 |
| `Soybean___healthy` | Soybean | Healthy | Yes | mapped | 5090 |
| `Squash___Powdery_mildew` | Squash | Powdery Mildew | No | mapped | 1835 |
| `Strawberry___Leaf_scorch` | Strawberry | Leaf Scorch | No | mapped | 1109 |
| `Strawberry___healthy` | Strawberry | Healthy | Yes | mapped | 456 |
| `Tomato___Bacterial_spot` | Tomato | Bacterial Spot | No | mapped | 2127 |
| `Tomato___Early_blight` | Tomato | Early Blight | No | mapped | 1000 |
| `Tomato___Late_blight` | Tomato | Late Blight | No | mapped | 1909 |
| `Tomato___Leaf_Mold` | Tomato | Leaf Mold | No | mapped | 952 |
| `Tomato___Septoria_leaf_spot` | Tomato | Septoria Leaf Spot | No | mapped | 1771 |
| `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato | Spider Mites Two Spotted Spider Mite | No | mapped | 1676 |
| `Tomato___Target_Spot` | Tomato | Target Spot | No | mapped | 1404 |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Tomato Yellow Leaf Curl Virus | No | mapped | 5357 |
| `Tomato___Tomato_mosaic_virus` | Tomato | Tomato Mosaic Virus | No | mapped | 373 |
| `Tomato___healthy` | Tomato | Healthy | Yes | mapped | 1591 |
