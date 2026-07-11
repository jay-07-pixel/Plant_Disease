# Dataset Comparison Report

**Generated:** 2026-07-11T02:22:40.669991+00:00  
**Source directory:** `C:\Projects\PlantDiseaseAI\datasets\external`  
**Datasets compared:** 2  

> Read-only audit. No images were modified, merged, or preprocessed.

## Summary Table

| Dataset | Images | Classes | Avg Width | Avg Height | Avg MP | Formats |
|---------|--------|---------|-----------|------------|--------|---------|
| `plantdoc` | 2552 | 54 | 1050 | 897 | 1.499 | jpeg, mpo, png |
| `plantvillage` | 54305 | 76 | 256 | 256 | 0.066 | jpeg, png |

## Class Overlap

- **Common classes (all datasets):** 0

### Unique Classes per Dataset

- **`plantdoc`:** 27 unique class(es)
  - `apple leaf`, `apple rust leaf`, `apple scab leaf`, `bell_pepper leaf`, `bell_pepper leaf spot`, `blueberry leaf`, `cherry leaf`, `corn gray leaf spot`, `corn leaf blight`, `corn rust leaf`
  - ... and 17 more
- **`plantvillage`:** 38 unique class(es)
  - `apple___apple_scab`, `apple___black_rot`, `apple___cedar_apple_rust`, `apple___healthy`, `blueberry___healthy`, `cherry_(including_sour)___healthy`, `cherry_(including_sour)___powdery_mildew`, `corn_(maize)___cercospora_leaf_spot gray_leaf_spot`, `corn_(maize)___common_rust_`, `corn_(maize)___healthy`
  - ... and 28 more

### Missing Classes Between Datasets

#### Missing from `plantdoc`

- Present in `plantvillage` but not `plantdoc`: 38 class(es)
  - `apple___apple_scab`, `apple___black_rot`, `apple___cedar_apple_rust`, `apple___healthy`, `blueberry___healthy`, `cherry_(including_sour)___healthy`, `cherry_(including_sour)___powdery_mildew`, `corn_(maize)___cercospora_leaf_spot gray_leaf_spot`, `corn_(maize)___common_rust_`, `corn_(maize)___healthy`
  - ... and 28 more

#### Missing from `plantvillage`

- Present in `plantdoc` but not `plantvillage`: 27 class(es)
  - `apple leaf`, `apple rust leaf`, `apple scab leaf`, `bell_pepper leaf`, `bell_pepper leaf spot`, `blueberry leaf`, `cherry leaf`, `corn gray leaf spot`, `corn leaf blight`, `corn rust leaf`
  - ... and 17 more

## Per-Dataset Details

### `plantdoc`

- **Images:** 2552
- **Classes:** 54
- **Average resolution:** 1049.8 × 896.7 px
- **Average megapixels:** 1.499
- **Image formats:** `jpeg` (2545), `mpo` (2), `png` (5)
- **Corrupted images:** 0
- **Empty folders:** 0
- **Duplicate filename groups:** 236
- **Max directory depth:** 3

**Class names:**
- `PlantDoc-Dataset\test\Apple Scab Leaf`: 10 images
- `PlantDoc-Dataset\test\Apple leaf`: 9 images
- `PlantDoc-Dataset\test\Apple rust leaf`: 10 images
- `PlantDoc-Dataset\test\Bell_pepper leaf`: 8 images
- `PlantDoc-Dataset\test\Bell_pepper leaf spot`: 9 images
- `PlantDoc-Dataset\test\Blueberry leaf`: 11 images
- `PlantDoc-Dataset\test\Cherry leaf`: 10 images
- `PlantDoc-Dataset\test\Corn Gray leaf spot`: 4 images
- `PlantDoc-Dataset\test\Corn leaf blight`: 12 images
- `PlantDoc-Dataset\test\Corn rust leaf`: 10 images
- `PlantDoc-Dataset\test\Peach leaf`: 9 images
- `PlantDoc-Dataset\test\Potato leaf early blight`: 8 images
- `PlantDoc-Dataset\test\Potato leaf late blight`: 8 images
- `PlantDoc-Dataset\test\Raspberry leaf`: 7 images
- `PlantDoc-Dataset\test\Soyabean leaf`: 8 images
- `PlantDoc-Dataset\test\Squash Powdery mildew leaf`: 6 images
- `PlantDoc-Dataset\test\Strawberry leaf`: 8 images
- `PlantDoc-Dataset\test\Tomato Early blight leaf`: 9 images
- `PlantDoc-Dataset\test\Tomato Septoria leaf spot`: 11 images
- `PlantDoc-Dataset\test\Tomato leaf`: 8 images
- `PlantDoc-Dataset\test\Tomato leaf bacterial spot`: 9 images
- `PlantDoc-Dataset\test\Tomato leaf late blight`: 10 images
- `PlantDoc-Dataset\test\Tomato leaf mosaic virus`: 10 images
- `PlantDoc-Dataset\test\Tomato leaf yellow virus`: 6 images
- `PlantDoc-Dataset\test\Tomato mold leaf`: 6 images
- `PlantDoc-Dataset\test\grape leaf`: 12 images
- `PlantDoc-Dataset\test\grape leaf black rot`: 8 images
- `PlantDoc-Dataset\train\Apple Scab Leaf`: 77 images
- `PlantDoc-Dataset\train\Apple leaf`: 82 images
- `PlantDoc-Dataset\train\Apple rust leaf`: 79 images
- `PlantDoc-Dataset\train\Bell_pepper leaf`: 53 images
- `PlantDoc-Dataset\train\Bell_pepper leaf spot`: 62 images
- `PlantDoc-Dataset\train\Blueberry leaf`: 105 images
- `PlantDoc-Dataset\train\Cherry leaf`: 47 images
- `PlantDoc-Dataset\train\Corn Gray leaf spot`: 64 images
- `PlantDoc-Dataset\train\Corn leaf blight`: 180 images
- `PlantDoc-Dataset\train\Corn rust leaf`: 106 images
- `PlantDoc-Dataset\train\Peach leaf`: 103 images
- `PlantDoc-Dataset\train\Potato leaf early blight`: 109 images
- `PlantDoc-Dataset\train\Potato leaf late blight`: 97 images
- `PlantDoc-Dataset\train\Raspberry leaf`: 109 images
- `PlantDoc-Dataset\train\Soyabean leaf`: 57 images
- `PlantDoc-Dataset\train\Squash Powdery mildew leaf`: 123 images
- `PlantDoc-Dataset\train\Strawberry leaf`: 88 images
- `PlantDoc-Dataset\train\Tomato Early blight leaf`: 74 images
- `PlantDoc-Dataset\train\Tomato Septoria leaf spot`: 137 images
- `PlantDoc-Dataset\train\Tomato leaf`: 54 images
- `PlantDoc-Dataset\train\Tomato leaf bacterial spot`: 98 images
- `PlantDoc-Dataset\train\Tomato leaf late blight`: 101 images
- `PlantDoc-Dataset\train\Tomato leaf mosaic virus`: 44 images
- `PlantDoc-Dataset\train\Tomato leaf yellow virus`: 69 images
- `PlantDoc-Dataset\train\Tomato mold leaf`: 85 images
- `PlantDoc-Dataset\train\grape leaf`: 57 images
- `PlantDoc-Dataset\train\grape leaf black rot`: 56 images

### `plantvillage`

- **Images:** 54305
- **Classes:** 76
- **Average resolution:** 256.0 × 256.0 px
- **Average megapixels:** 0.066
- **Image formats:** `jpeg` (54304), `png` (1)
- **Corrupted images:** 0
- **Empty folders:** 0
- **Duplicate filename groups:** 0
- **Max directory depth:** 3

**Class names:**
- `PlantVillage\train\Apple___Apple_scab`: 504 images
- `PlantVillage\train\Apple___Black_rot`: 496 images
- `PlantVillage\train\Apple___Cedar_apple_rust`: 220 images
- `PlantVillage\train\Apple___healthy`: 1316 images
- `PlantVillage\train\Blueberry___healthy`: 1202 images
- `PlantVillage\train\Cherry_(including_sour)___Powdery_mildew`: 842 images
- `PlantVillage\train\Cherry_(including_sour)___healthy`: 684 images
- `PlantVillage\train\Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`: 410 images
- `PlantVillage\train\Corn_(maize)___Common_rust_`: 953 images
- `PlantVillage\train\Corn_(maize)___Northern_Leaf_Blight`: 788 images
- `PlantVillage\train\Corn_(maize)___healthy`: 929 images
- `PlantVillage\train\Grape___Black_rot`: 944 images
- `PlantVillage\train\Grape___Esca_(Black_Measles)`: 1107 images
- `PlantVillage\train\Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`: 861 images
- `PlantVillage\train\Grape___healthy`: 339 images
- `PlantVillage\train\Orange___Haunglongbing_(Citrus_greening)`: 4405 images
- `PlantVillage\train\Peach___Bacterial_spot`: 1838 images
- `PlantVillage\train\Peach___healthy`: 288 images
- `PlantVillage\train\Pepper,_bell___Bacterial_spot`: 797 images
- `PlantVillage\train\Pepper,_bell___healthy`: 1183 images
- `PlantVillage\train\Potato___Early_blight`: 800 images
- `PlantVillage\train\Potato___Late_blight`: 800 images
- `PlantVillage\train\Potato___healthy`: 121 images
- `PlantVillage\train\Raspberry___healthy`: 297 images
- `PlantVillage\train\Soybean___healthy`: 4072 images
- `PlantVillage\train\Squash___Powdery_mildew`: 1468 images
- `PlantVillage\train\Strawberry___Leaf_scorch`: 887 images
- `PlantVillage\train\Strawberry___healthy`: 364 images
- `PlantVillage\train\Tomato___Bacterial_spot`: 1702 images
- `PlantVillage\train\Tomato___Early_blight`: 800 images
- `PlantVillage\train\Tomato___Late_blight`: 1527 images
- `PlantVillage\train\Tomato___Leaf_Mold`: 761 images
- `PlantVillage\train\Tomato___Septoria_leaf_spot`: 1417 images
- `PlantVillage\train\Tomato___Spider_mites Two-spotted_spider_mite`: 1341 images
- `PlantVillage\train\Tomato___Target_Spot`: 1123 images
- `PlantVillage\train\Tomato___Tomato_Yellow_Leaf_Curl_Virus`: 4286 images
- `PlantVillage\train\Tomato___Tomato_mosaic_virus`: 299 images
- `PlantVillage\train\Tomato___healthy`: 1273 images
- `PlantVillage\val\Apple___Apple_scab`: 126 images
- `PlantVillage\val\Apple___Black_rot`: 125 images
- `PlantVillage\val\Apple___Cedar_apple_rust`: 55 images
- `PlantVillage\val\Apple___healthy`: 329 images
- `PlantVillage\val\Blueberry___healthy`: 300 images
- `PlantVillage\val\Cherry_(including_sour)___Powdery_mildew`: 210 images
- `PlantVillage\val\Cherry_(including_sour)___healthy`: 170 images
- `PlantVillage\val\Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`: 103 images
- `PlantVillage\val\Corn_(maize)___Common_rust_`: 239 images
- `PlantVillage\val\Corn_(maize)___Northern_Leaf_Blight`: 197 images
- `PlantVillage\val\Corn_(maize)___healthy`: 233 images
- `PlantVillage\val\Grape___Black_rot`: 236 images
- `PlantVillage\val\Grape___Esca_(Black_Measles)`: 276 images
- `PlantVillage\val\Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`: 215 images
- `PlantVillage\val\Grape___healthy`: 84 images
- `PlantVillage\val\Orange___Haunglongbing_(Citrus_greening)`: 1102 images
- `PlantVillage\val\Peach___Bacterial_spot`: 459 images
- `PlantVillage\val\Peach___healthy`: 72 images
- `PlantVillage\val\Pepper,_bell___Bacterial_spot`: 200 images
- `PlantVillage\val\Pepper,_bell___healthy`: 295 images
- `PlantVillage\val\Potato___Early_blight`: 200 images
- `PlantVillage\val\Potato___Late_blight`: 200 images
- `PlantVillage\val\Potato___healthy`: 31 images
- `PlantVillage\val\Raspberry___healthy`: 74 images
- `PlantVillage\val\Soybean___healthy`: 1018 images
- `PlantVillage\val\Squash___Powdery_mildew`: 367 images
- `PlantVillage\val\Strawberry___Leaf_scorch`: 222 images
- `PlantVillage\val\Strawberry___healthy`: 92 images
- `PlantVillage\val\Tomato___Bacterial_spot`: 425 images
- `PlantVillage\val\Tomato___Early_blight`: 200 images
- `PlantVillage\val\Tomato___Late_blight`: 382 images
- `PlantVillage\val\Tomato___Leaf_Mold`: 191 images
- `PlantVillage\val\Tomato___Septoria_leaf_spot`: 354 images
- `PlantVillage\val\Tomato___Spider_mites Two-spotted_spider_mite`: 335 images
- `PlantVillage\val\Tomato___Target_Spot`: 281 images
- `PlantVillage\val\Tomato___Tomato_Yellow_Leaf_Curl_Virus`: 1071 images
- `PlantVillage\val\Tomato___Tomato_mosaic_virus`: 74 images
- `PlantVillage\val\Tomato___healthy`: 318 images

## Visualizations

| Chart | File |
|-------|------|
| Dataset sizes | `dataset_sizes.png` |
| Class overlap | `class_overlap.png` |
| Resolution comparison | `resolution_comparison.png` |
