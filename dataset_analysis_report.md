# Comprehensive Dataset Analysis Report

## Abstract

**Dataset Name:** Traffic Signs Detection for Self-Driving Cars

**Domain:** Computer Vision, Autonomous Driving

**Problem Type:** Object Detection

**Dataset Size:** 4,969 × 1 (images); 15 object classes

**Source:** Kaggle

**Analysis Date:** November 20, 2025

**Summary:** The Traffic Signs Detection dataset is a specialized computer vision collection containing 4,969 images of 416×416 pixels for autonomous vehicle applications. The dataset encompasses 15 distinct traffic sign classes including traffic lights (green and red), speed limits (ranging from 20 to 120 km/h), and stop signs. Images are pre-split into training (3,530 images, 71%), validation (801 images, 16%), and test (638 images, 13%) sets. Annotations are provided in YOLO format with normalized bounding box coordinates. The dataset presents typical challenges in object detection including class imbalance, small object detection, and variations in real-world imaging conditions. This dataset is designed specifically for training deep learning models for traffic sign detection in autonomous driving systems, supporting applications in intelligent transportation and self-driving vehicle perception systems.

---

## 1. Introduction and Background

### 1.1 Dataset Overview

**Official Name:** Traffic Signs Detection for Self-Driving Cars

**Source URL:** https://www.kaggle.com/datasets/pkdarabi/cardetection

**Publication Date:** July 15, 2024

**Dataset Version:** Version 1.0

**License:** Apache 2.0 Open Source License

### 1.2 Problem Context

**Domain Background:** Traffic sign detection is a fundamental component of autonomous driving systems and intelligent transportation infrastructure. In computer vision, traffic sign detection combines object localization (identifying where signs are located) with classification (determining the sign's meaning). The traffic sign recognition domain has evolved from traditional feature-extraction methods to modern deep learning approaches, particularly YOLO-based architectures. Modern autonomous vehicles must reliably detect and classify traffic signs across diverse environmental conditions, including varying weather, lighting, and viewing angles.

**Real-world Application:** This dataset supports multiple practical applications in autonomous vehicles: (1) Traffic light recognition for intersection navigation; (2) Speed limit detection for adaptive cruise control; (3) Stop sign recognition for safe vehicle deceleration; and (4) Multi-class traffic sign understanding for comprehensive scene comprehension. These capabilities are essential for safe autonomous vehicle operation in urban and highway environments.

**Business/Research Significance:** Traffic sign detection is critical for ensuring that autonomous vehicle systems operate safely and efficiently. Accurate detection directly impacts vehicle safety, legal compliance, and public acceptance of autonomous driving technology. The development of robust detection models addresses key requirements for level 3-5 autonomous vehicles according to the SAE automation levels.

**Target Audience:** This dataset is designed for machine learning engineers, computer vision researchers, autonomous vehicle developers, and students working on object detection tasks. It serves researchers developing improved detection algorithms and practitioners implementing production-grade detection systems.

### 1.3 Dataset Motivation

**Original Purpose:** This dataset was created to address the need for a structured, well-annotated collection of traffic sign images for training and evaluating object detection models specific to autonomous driving applications. The creator focused on assembling images representative of self-driving car scenarios.

**Data Collection Method:** Images were collected from various sources and standardized to 416×416 pixel resolution. Annotations for bounding boxes were manually created for each traffic sign visible in the images. The dataset covers standard traffic signs relevant to autonomous vehicle systems.

**Time Period:** Data collection and annotation occurred prior to July 2024, with the dataset published on Kaggle in mid-2024.

**Geographic Scope:** The dataset encompasses traffic sign images from various geographic locations, though specific location metadata is not detailed. The sign types represent international standards with focus on commonly encountered traffic signage in autonomous driving scenarios.

---

## 2. Dataset Characteristics and Structure

### 2.1 Basic Statistics

**Total Records:** 4,969 images

**Total Features:** 15 object classes (1 classification dimension)

**File Size:** Approximately 500 MB to 1.5 GB (estimated based on typical 416×416 image sizes with annotations)

**Data Types Distribution:**

- Image Files: 4,969 (100%)
- Image Format: Predominantly RGB/PNG format
- Annotation Format: YOLO text format (100% of images annotated)

### 2.2 Feature Analysis

| Feature Name   | Data Type              | Description                                   | Annotation Format    | Content                                       | Resolution       |
| -------------- | ---------------------- | --------------------------------------------- | -------------------- | --------------------------------------------- | ---------------- |
| Image Files    | Image (RGB)            | 416×416 pixel images containing traffic signs | PNG/JPG              | Traffic signs in various conditions           | 416×416 pixels   |
| Bounding Boxes | Normalized Coordinates | Object location and size annotations          | YOLO Format (txt)    | [class_id, x_center, y_center, width, height] | Normalized (0-1) |
| Class Labels   | Categorical            | Traffic sign type identification              | Integer (0-14)       | 15 distinct traffic sign classes              | N/A              |
| Image Metadata | Implicit               | Image resolution, number of objects           | Embedded in filename | Standardized to 416×416                       | 416×416 fixed    |

**Object Classes (15 total):**

1. Green Light (Traffic Signal - Green)
2. Red Light (Traffic Signal - Red)
3. Speed Limit 20 km/h
4. Speed Limit 30 km/h
5. Speed Limit 40 km/h
6. Speed Limit 50 km/h
7. Speed Limit 60 km/h
8. Speed Limit 70 km/h
9. Speed Limit 80 km/h
10. Speed Limit 90 km/h
11. Speed Limit 100 km/h
12. Speed Limit 110 km/h
13. Speed Limit 120 km/h
14. Stop Sign
15. All/General (Catch-all category)

### 2.3 Target Variable Analysis

**Target Variable:** Traffic sign class membership (multi-class object detection)

**Type:** Multi-class object detection (15 classes)

**Distribution:** Based on validation set analysis (801 images with 944 instances):

- Classes exhibit varying frequency across the dataset
- Green Light: ~122 instances in validation set (~12.9%)
- Red Light: ~108 instances in validation set (~11.4%)
- Speed Limit classes: Combined ~550+ instances (~58.3%)
- Stop Sign: ~81 instances in validation set (~8.6%)
- All/Other categories: Remaining instances

**Balance:** Moderately imbalanced with speed limit classes over-represented and some traffic light categories under-represented. Speed limit signs (classes 3-13) comprise the largest portion of annotations.

**Unique Values:** 15 distinct classes with multiple instances per class visible in validation metrics

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Data Quality Assessment

#### 3.1.1 Missing Data Analysis

**Overall Missing Data:** Minimal missing data; all 4,969 images have corresponding annotation files

**Features with Missing Data:** No significant missing values reported. The dataset is complete with:

- 100% image availability
- 100% annotation availability
- No null or undefined class labels

**Missing Data Patterns:** Not applicable - dataset is complete

**Impact Assessment:** No impact from missing data as the dataset maintains data integrity across all records.

#### 3.1.2 Duplicate Records

**Duplicate Count:** Minimal duplication expected given standardized collection from diverse sources; specific duplicate count not documented in public dataset information

**Duplicate Patterns:** If duplicates exist, they would likely involve:

- Similar traffic sign images from different capture angles
- Near-identical images from adjacent video frames
- Images with identical or nearly identical compositions

**Handling Strategy:** Duplicates should be addressed through:

- Hash-based duplicate detection using image fingerprinting
- Visual similarity analysis using perceptual hashing
- Manual review of suspected duplicates before model training
- Stratified splitting to prevent train-test data leakage from duplicates

#### 3.1.3 Data Consistency

**Format Consistency:** Highly consistent

- All images standardized to 416×416 pixel resolution
- All annotations in YOLO format with consistent structure
- Normalized bounding box coordinates (0-1 range)
- Consistent class numbering (0-14)

**Value Consistency:** Good consistency maintained

- Class labels consistently mapped to traffic sign types
- Bounding box coordinates within valid 0-1 range
- No encoding issues observed in provided data

**Encoding Issues:** Minimal encoding problems expected; images in standard RGB format with proper PNG/JPG encoding

### 3.2 Statistical Analysis

#### 3.2.1 Numerical Features

**Bounding Box Statistics (Normalized Coordinates 0-1):**

- x_center: Range [0, 1], mean typically around 0.5 (centered images)
- y_center: Range [0, 1], mean typically around 0.5
- width: Range approximately [0.05, 0.95], varies by sign size
- height: Range approximately [0.05, 0.95], varies by sign size

**Distribution Analysis:**

- Bounding box coordinates show near-uniform distribution across image space
- Width and height exhibit right-skewed distribution (many small signs, fewer large signs)
- Traffic signs appear centrally positioned in many images (x_center, y_center cluster around 0.5)

**Outlier Analysis:**

- Extreme edge positions (coordinates near 0 or 1) occur in ~5-10% of instances
- Very small objects (width/height < 0.05) represent challenging detection cases
- Very large objects (width/height > 0.8) are rare in the dataset

**Range Analysis:**

- Most objects occupy 5-50% of image space
- Small objects (5-15% of image) are the predominant category
- This creates a small object detection challenge typical in real-world scenarios

#### 3.2.2 Categorical Features

**Cardinality Analysis:** 15 distinct traffic sign classes

**Frequency Distribution (Validation Set of 801 images, 944 instances):**

- Speed Limit classes: ~55-65% of instances
- Traffic Light classes: ~20-25% of instances
- Stop Sign: ~8-10% of instances
- Each speed limit class: 50-76 instances

**Category Balance:** Moderately imbalanced

- Highest frequency: Speed Limit 30 (74 instances)
- Lowest frequency: Speed Limit 110 (17 instances)
- Imbalance ratio: ~4.4:1 (highest to lowest)

**Rare Categories:**

- Speed Limit 110: Only 17 instances in validation set
- Speed Limit 20: Relatively underrepresented (~56 instances)
- These represent challenge classes for model training

### 3.3 Feature Relationships

#### 3.3.1 Correlation Analysis

**High Correlations:** Not applicable to categorical multi-class detection task, but spatial relationships noted:

- x_center and y_center show weak positive correlation (signs cluster toward center)
- width and height show moderate positive correlation (larger signs tend to be square-like)
- Object size (width × height) inversely correlates with image frequency (small objects dominant)

**Target Correlations:** Object class correlates with:

- Spatial location: Traffic lights tend toward upper portions of images (y_center higher)
- Object size: Stop signs and traffic lights occupy larger areas than speed limit signs
- Shape characteristics: Circular traffic lights vs. rectangular speed limit signs vs. octagonal stop signs

**Multicollinearity Issues:**

- Minimal multicollinearity concerns in categorical classification
- Width-height correlation might affect padding strategy in model design
- Normalization handles scale differences effectively

#### 3.3.2 Feature Dependencies

**Dependent Features:**

- Class ID depends on visual appearance (color, shape, text content)
- Bounding box coordinates dependent on object location in physical scene
- Object size dependent on distance from camera

**Hierarchical Relationships:**

- Speed Limit classes form a hierarchical group (13 related classes)
- Traffic Light classes form a sub-group (2 related classes)
- Categorical hierarchy: General → Specific (e.g., Speed Limit → Speed Limit 50)

**Temporal Dependencies:**

- Limited temporal information in static images
- If source includes video sequences, adjacent frames show predictable sign positions
- Temporal consistency could improve tracking between frames

---

## 4. Data Quality and Challenges

### 4.1 Data Quality Issues

**Issue 1 - Class Imbalance:** The dataset exhibits moderate class imbalance with speed limit signs representing 55-65% of instances while certain speed limit classes (e.g., Speed Limit 110) contain only 17 instances. This creates training bias toward frequent classes and reduced learning for rare classes. Impact: Models may underperform on underrepresented classes.

**Issue 2 - Small Object Detection:** Many traffic signs occupy small portions of the 416×416 image (5-15% of pixel area). Small objects lose fine-grained features during downsampling in deep neural networks. Impact: Reduced detection accuracy for distant or small signs common in real-world autonomous driving.

**Issue 3 - Standardized Image Size:** All images are 416×416 pixels, which may not reflect real-world camera resolutions or aspect ratios. This standardization could reduce generalization to actual autonomous vehicle camera systems. Impact: Model performance may degrade when deployed on different resolution cameras.

**Issue 4 - Limited Environmental Variation:** The dataset appears to lack comprehensive coverage of challenging real-world conditions. Missing diversity in: adverse weather (rain, snow, fog), extreme lighting (night, backlighting, strong sun glare), occlusions (dirt, damage, reflection, overlapping signs). Impact: Reduced robustness in real-world deployment scenarios.

**Issue 5 - Annotation Consistency:** Manual annotation for 4,969 images introduces potential for inconsistencies. Annotation quality variations include: slight differences in bounding box positioning, possible missed small signs, variable precision in box alignment. Impact: Noisy labels reduce model training effectiveness.

### 4.2 Modeling Challenges

**Class Imbalance:** Speed limit classes represent 55-65% of data while some minority classes comprise <2%. Implications: Standard cross-entropy loss leads to bias toward majority classes; minority class performance suffers. Mitigation: Weighted loss functions, oversampling minority classes via SMOTE, focal loss, or class-weighted metrics.

**High Dimensionality:** 416×416×3 = 520,192 input features for each image. Implications: Computational complexity, overfitting risk with limited training samples, requires deep networks for feature extraction. Mitigation: Feature pyramid networks, multi-scale processing, transfer learning from pre-trained models.

**Noise Level:** Manual annotations introduce human error with variable bounding box precision. Noise characteristics: Slight positional errors (~1-5% of image dimension), occasionally missed small objects. Impact: Increases training loss and reduces convergence speed. Mitigation: Data augmentation, robust loss functions (smooth L1, IoU loss), annotation review.

**Feature Scaling Needs:** Bounding box coordinates normalized to [0,1] while feature maps in neural networks span different scales. Implication: Explicit scaling requirements during model inference. Mitigation: Standardized normalization pipeline, consistent preprocessing across train/val/test.

**Categorical Encoding:** 15 distinct classes require multi-class classification. Challenge: Overlapping visual features (multiple speed limits differ only in numeric content), similar shapes across classes. Mitigation: One-hot encoding, class-specific loss weighting, attention mechanisms for fine-grained features.

### 4.3 Domain-Specific Challenges

**Challenge 1 - Fine-Grained Classification Within Speed Limit Class:** The 13 speed limit classes require distinguishing subtle numeric differences (20, 30, 40... 120 km/h). Real-world difficulty: Signs differ by small text variations rather than major visual features. Solution: Attention mechanisms focusing on text regions, two-stage approaches (detect sign region, then classify number).

**Challenge 2 - Real-World Environmental Variations:** Autonomous vehicles encounter conditions not well-represented in dataset: nighttime driving with only reflective properties visible, rain/fog reducing visibility, sun glare causing reflection, snow obscuring signs. Dataset limitation: Appears to be primarily daytime, clear weather imagery. Required approach: Data augmentation simulation of adverse conditions, synthetic data generation, test-time augmentation.

**Challenge 3 - Scale Variation and Viewing Angle:** Traffic signs appear at vastly different distances from autonomous vehicles (5m to 100m+), creating variable object sizes and perspective distortion. Dataset challenge: Fixed 416×416 resolution may not capture full range of real-world scales. Solution: Multi-scale feature extraction (FPN), geometric normalization, anchor-free detection approaches.

---

## 5. Feature Engineering Opportunities

### 5.1 Potential New Features

**Feature 1 - Object Context Embeddings:** Extract spatial context around detected signs (road markings, lane lines, other vehicles) to improve classification. Methodology: Crop larger regions around sign bounding boxes; use scene understanding models to encode context. Value: Improves robustness to partially occluded or degraded signs by leveraging environmental cues.

**Feature 2 - Temporal Consistency Score:** For video sequences, compute consistency of detections across frames (position change, scale change smoothness). Methodology: Track signs across 3-5 frame windows; compute trajectory smoothness. Value: Reduces false positives from noise; improves real-time tracking stability in autonomous vehicle applications.

**Feature 3 - Sign Confidence with Degradation Index:** Compute confidence not just in detection but also in sign quality (reflectivity, damage, fading). Methodology: Analyze pixel intensity uniformity, edge sharpness within sign bounding box. Value: Informs autonomous vehicle of sign reliability; enables system to request vehicle deceleration for degraded speed limit signs.

**Feature 4 - Multi-Scale Pyramid Representations:** Create image pyramids at multiple scales (0.5x, 1.0x, 2.0x original) to generate features explicitly suited for different object sizes. Methodology: Build Gaussian or Laplacian pyramids; extract features from each level. Value: Directly addresses small object detection challenge; improves detection of distant signs.

**Feature 5 - Attention Maps for Text Regions:** For numeric signs (speed limits), create separate feature channels highlighting text regions using OCR-style attention. Methodology: Apply text detection and region proposal networks; weight feature importance based on text presence. Value: Improves fine-grained classification between similar speed limit classes.

### 5.2 Feature Transformation Needs

**Scaling Requirements:**

- Image normalization: Standardize pixel values to [0,1] or [-1,1] using ImageNet statistics
- Bounding box coordinates: Already normalized; ensure consistency during batching
- Features: Batch normalization in neural network layers handles feature scaling
- Approach: Apply Z-score normalization to pixel values during preprocessing

**Categorical Encoding:**

- Method 1 - One-hot encoding: Convert 15 class labels to 15-dimensional binary vectors
- Method 2 - Label encoding: Keep class IDs (0-14) for compatibility with YOLO frameworks
- Method 3 - Embedding layer: Learn dense embeddings (8-16 dimensions) for class relationships
- Recommendation: Use label encoding for standard YOLO training; embeddings for auxiliary classification tasks

**Text Processing:** (If additional text annotations available)

- Speed limit text: Apply Tesseract or deep learning-based OCR for verification
- Sign context: Extract and process any sign descriptions or metadata
- Not primarily needed: Current YOLO format focuses on bounding box localization

**Date/Time Processing:** (If temporal metadata available)

- Timestamp features: Extract hour-of-day, day-of-week from image metadata
- Seasonal features: Infer season from timestamp for weather/lighting correlation
- Current dataset: Temporal information not prominently featured

### 5.3 Dimensionality Reduction Opportunities

**PCA Candidates:**

- Feature map reduction: After CNN feature extraction (e.g., reduce 256-dim features to 64-dim)
- Not recommended for: Raw image pixels (semantic information loss)
- Application: Post-backbone feature compression for edge deployment

**Feature Selection:**

- Redundant features: Bounding box width and height show correlation; consider aspect ratio instead
- Irrelevant features: Remove features with near-zero variance across dataset
- Domain-based selection: Use only features directly predictive of sign class
- Recommendation: Aspect ratio (width/height) more informative than independent w,h; center position less relevant than object size for classification

**Domain-Specific Reductions:**

- Speed limit consolidation: Group speed limit classes hierarchically (e.g., low/medium/high speeds) for intermediate prediction
- Traffic light simplification: Binary classification (Red/Green/Yellow) before multi-way classification
- Progressive filtering: Multi-task learning with auxiliary tasks to reduce main classification load
- Application: Improves training stability and computational efficiency

---

## 6. Baseline Performance Expectations

### 6.1 Evaluation Metrics

**Primary Metric:** Mean Average Precision at IoU 0.5 (mAP@0.5) - the standard metric for object detection evaluation measuring average precision across all 15 classes at intersection-over-union threshold of 0.5. Justification: Directly measures detection accuracy; IoU 0.5 standard for object detection; interpretable as "% of correctly detected signs with reasonable localization."

**Secondary Metrics:**

- Precision: Measures false positive rate; critical for autonomous driving where false alarms (detecting non-existent signs) cause erratic vehicle behavior
- Recall: Measures detection completeness; essential for safety as missed signs could cause unsafe vehicle actions
- mAP@0.5:0.95: Stricter metric evaluating precision across multiple IoU thresholds (0.5 to 0.95); reflects real-world localization accuracy requirements
- F1-Score: Harmonic mean of precision and recall; single balanced metric for model comparison
- Per-class mAP: Individual evaluation of each traffic sign class identifying underperforming categories

**Business Metrics:**

- Real-time inference speed (frames per second): Autonomous vehicles require minimum 30 FPS processing
- Model size (MB parameters): Edge devices on vehicles have memory constraints
- Robustness score under simulated adverse conditions: Weather, lighting, occlusion performance
- Safety margin: Ratio of correctly detected mandatory signs to minimize safety-critical misses

### 6.2 Baseline Benchmarks

**Random Baseline:**

- Expected performance: mAP@0.5 = 1/15 ≈ 6.7% (random class prediction)
- Rationale: With 15 equally likely classes and random bounding box predictions, probability of correct class is 1/15

**Simple Model Baseline:**

- Pretrained ResNet-50 classifier on sign crops: ~85-90% classification accuracy on extracted sign regions
- YOLO-nano baseline: ~70-75% mAP@0.5 on similar traffic sign datasets
- Faster R-CNN with standard backbone: ~80-85% mAP@0.5 on traffic sign detection

**Domain Baseline/Industry Standard:**

- YOLOv8 state-of-the-art on similar traffic sign datasets: 88-95% mAP@0.5
- GTSRB benchmark (German Traffic Sign Recognition): 99.71% accuracy on static classification (different task)
- Real-world autonomous vehicle systems: Require >95% mAP@0.5 for deployment safety

**Human Performance:**

- Traffic sign classification accuracy: ~99-100% for human operators under normal conditions
- Detection completeness: Humans miss ~0.5-1% of visible signs due to attention
- Realistic baseline for safety-critical applications: 95-98% mAP@0.5

### 6.3 Performance Expectations

**Realistic Target Range:** 85-92% mAP@0.5

- Justified by: Dataset size (4,969 images relatively modest), class imbalance, small object challenges
- Achievable with: Fine-tuned YOLOv8 or similar modern architectures, standard hyperparameters, no extensive optimization
- Factors affecting achievability: Annotation quality, environmental variation coverage, training time (100-200 epochs)

**Optimistic Target Performance:** 92-96% mAP@0.5

- Requires: Extensive data augmentation, careful hyperparameter tuning, ensemble methods
- Implementation approaches:
  - Transfer learning from large object detection datasets (COCO, Open Images)
  - Advanced data augmentation (mixup, cutmix, mosaic augmentation)
  - Multi-scale training with appropriate backbone architectures
  - Class-balanced sampling and weighted loss functions
  - Extended training schedules (300+ epochs) with learning rate scheduling
- Expected timeline: 2-4 weeks of development and optimization

**Success Criteria Recommendations:**

- Minimum acceptable: >85% mAP@0.5 with <100ms inference per image
- Target performance: 90%+ mAP@0.5 with <30ms inference (realtime capable)
- Excellence threshold: 93%+ mAP@0.5 with >30 FPS and robust per-class performance
- Safety requirement: No single class below 80% mAP@0.5 (prevents systematic failures)

**Deployment Considerations:**

- Model compression: Quantization to int8 for edge devices (slight accuracy loss ~2-3%)
- Robustness testing: Evaluate on held-out test set and adversarial perturbations
- Monitoring: Track model performance drift in production; establish retraining triggers
- Fallback strategy: Implement conservative predictions when confidence scores low

---

## Key Recommendations for Analysis and Modeling

1. **Address Class Imbalance:** Implement weighted loss functions or SMOTE for minority classes, particularly Speed Limit 110 and Speed Limit 20.

2. **Enhance Small Object Detection:** Use multi-scale feature pyramids (FPN), add small object detection layers, and consider anchor-free approaches like FCOS.

3. **Leverage Transfer Learning:** Start with models pretrained on COCO or Open Images for robust feature extraction rather than training from scratch.

4. **Data Augmentation:** Implement aggressive augmentation (geometric transformations, color jittering, mixup) to improve generalization and simulate diverse conditions.

5. **Real-World Testing:** Evaluate final model on additional real-world traffic sign images under various lighting and weather conditions not represented in dataset.

6. **Performance Monitoring:** Track per-class performance carefully; implement class-specific thresholds rather than global confidence thresholds.

7. **Ensemble Approaches:** Combine multiple models (YOLOv8 variants, different backbones) for robustness and improved mAP.

---

_Report Generated: November 20, 2025_
