# Dataset Audit Report

**Generated:** 2026-07-11T02:22:34.970625+00:00  
**Source directory:** `C:\Projects\PlantDiseaseAI\datasets\external`  
**Total images:** 56857  
**Total classes (aggregate):** 130  

> This audit is read-only. No images were modified, renamed, or preprocessed.

## Visualizations

| Chart | File |
|-------|------|
| Class distribution (all, top 20, bottom 20) | `class_distribution.png` |
| Resolution histogram | `image_resolution.png` |
| Format distribution | `image_formats.png` |

## Aggregate Summary

### Format Distribution

- `jpeg`: 56849
- `mpo`: 2
- `png`: 6

## Dataset: `plantdoc`

**Path:** `C:\Projects\PlantDiseaseAI\datasets\external\plantdoc`  
**Total images:** 2552  
**Total classes:** 54  
**Max directory depth:** 3  

### Class Distribution

- Majority class: `PlantDoc-Dataset\train\Corn leaf blight` (180 images)
- Minority class: `PlantDoc-Dataset\test\Corn Gray leaf spot` (4 images)
- Imbalance ratio (max/min): 45.00
- Coefficient of variation: 0.934


### Class Names

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

### Resolution Statistics

- Images measured: 2552
- Width (px): min=115, max=6000, mean=1049.8, median=800.0
- Height (px): min=69, max=6000, mean=896.7, median=655.5
- Megapixels: min=0.009, max=24.000, mean=1.499, median=0.493


### Format Statistics

- `jpeg`: 2545
- `mpo`: 2
- `png`: 5

### Data Quality

- Empty folders: 0
- Non-image files: 0
- Corrupted images: 0

### Duplicates

- Perceptual duplicate groups: 54
- Duplicate filename groups: 236

**Perceptual duplicate groups (sample, max 5):**

- 2 images: PlantDoc-Dataset\test\Blueberry leaf\Blueberry leaf (1).jpg, PlantDoc-Dataset\train\Blueberry leaf\Blueberry leaf (49).jpg
- 2 images: PlantDoc-Dataset\test\Corn Gray leaf spot\Corn Gray leaf spot (1).jpg, PlantDoc-Dataset\train\Corn leaf blight\Corn leaf blight (121).jpg
- 2 images: PlantDoc-Dataset\test\Corn leaf blight\Corn leaf blight (10).jpg, PlantDoc-Dataset\train\Corn Gray leaf spot\Corn Gray leaf spot (20).jpg
- 2 images: PlantDoc-Dataset\test\Corn leaf blight\Corn leaf blight (7).jpg, PlantDoc-Dataset\train\Corn Gray leaf spot\Corn Gray leaf spot (12).jpg
- 2 images: PlantDoc-Dataset\test\Potato leaf early blight\Potato leaf early blight (2).jpg, PlantDoc-Dataset\train\Potato leaf late blight\Potato leaf late blight (19).jpg

**Duplicate filenames (sample, max 5):**

- `Apple Scab Leaf (1).jpg`: 2 occurrences
- `Apple Scab Leaf (10).jpg`: 2 occurrences
- `Apple Scab Leaf (2).jpg`: 2 occurrences
- `Apple Scab Leaf (3).jpg`: 2 occurrences
- `Apple Scab Leaf (4).jpg`: 2 occurrences

---

## Dataset: `plantvillage`

**Path:** `C:\Projects\PlantDiseaseAI\datasets\external\plantvillage`  
**Total images:** 54305  
**Total classes:** 76  
**Max directory depth:** 3  

### Class Distribution

- Majority class: `PlantVillage\train\Orange___Haunglongbing_(Citrus_greening)` (4405 images)
- Minority class: `PlantVillage\val\Potato___healthy` (31 images)
- Imbalance ratio (max/min): 142.10
- Coefficient of variation: 1.187


### Class Names

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

### Resolution Statistics

- Images measured: 54305
- Width (px): min=256, max=256, mean=256.0, median=256.0
- Height (px): min=256, max=256, mean=256.0, median=256.0
- Megapixels: min=0.066, max=0.066, mean=0.066, median=0.066


### Format Statistics

- `jpeg`: 54304
- `png`: 1

### Data Quality

- Empty folders: 0
- Non-image files: 0
- Corrupted images: 0

### Duplicates

- Perceptual duplicate groups: 22
- Duplicate filename groups: 0

**Perceptual duplicate groups (sample, max 5):**

- 2 images: PlantVillage\train\Apple___healthy\11beda66-01e9-4bfd-be37-c0f8646d1478___RS_HL 6271.JPG, PlantVillage\val\Apple___healthy\13298d36-4425-437d-ae8e-c7d70e200084___RS_HL 6271.JPG
- 2 images: PlantVillage\train\Apple___healthy\14896dc0-688d-456f-b5ec-a037695b0193___RS_HL 6268.JPG, PlantVillage\train\Apple___healthy\c21cf428-bfc3-4710-b5d2-69d1c0e94748___RS_HL 6268.JPG
- 2 images: PlantVillage\train\Apple___healthy\1ab5e019-e5f0-4d8e-a252-94cb0aab8b0a___RS_HL 6269.JPG, PlantVillage\train\Apple___healthy\3673d121-b5de-481c-b057-d4ee5b4959b1___RS_HL 6269.JPG
- 2 images: PlantVillage\train\Apple___healthy\9b75de13-d4b0-4b3f-988c-3e9926eef957___RS_HL 6273.JPG, PlantVillage\val\Apple___healthy\5192db55-4aa7-421c-92d4-c2dac79e7379___RS_HL 6273.JPG
- 2 images: PlantVillage\train\Apple___healthy\acb21cc2-8d65-4880-a7bb-dcc1eab1564b___RS_HL 6272.JPG, PlantVillage\val\Apple___healthy\3d075f90-7002-4c45-abc0-4f35ee49aa79___RS_HL 6272.JPG

---
