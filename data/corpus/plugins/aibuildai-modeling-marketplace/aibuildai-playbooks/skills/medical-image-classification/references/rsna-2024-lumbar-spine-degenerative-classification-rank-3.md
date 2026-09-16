# 3rd Place Solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #3
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453

# 3rd Place Solution

First and foremost, we would like to express our deepest gratitude to Kaggle and the competition organizers for providing this wonderful opportunity. We also thank all the participants for making this competition engaging and insightful.

## Summary

We constructed a general two-stage pipeline:

- **Stage 1**: Crop sagittal images at each disc level and crop axial images using disc level assignments and spinal canal positions.
- **Stage 2**: Use a **Center Classifier** to classify the severity of Spinal Canal Stenosis and a **Side Classifier** to classify the severity of Neural Foraminal Narrowing and Subarticular Stenosis.

[overview]

We will explain each process in detail below.

## Stage 1

The responsibility of Stage 1 is to extract the information necessary for estimating disease severity from the input data.

### 1. Disc Level Keypoint Detector (CenterNet)

We built a CenterNet-based 2D keypoint detector using EfficientNetB6 as the backbone and FPN as the neck. By inputting sagittal images near the center of the body, we estimate the coordinates of each disc level. For training data, we used sagittal images from RSNA2024 that have coordinates of Spinal Canal Stenosis at all levels, as well as the [Coordinate Pretraining Dataset](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/524500). By using the trained model to generate pseudo-labels on unused RSNA2024 data, we ultimately utilized all RSNA2024 data. Recognizing from several discussions that label noise existed, we manually reviewed all annotations and corrected erroneous labels by hand.

### 2. Crop Level

We cropped the sagittal images at each disc level. To ensure diversity in the input data, we adopted multiple cropping settings. There was almost no difference in accuracy due to cropping settings.

[crop_sagittal]

### 3. Assign Level

Using the output of the Disc Level Keypoint Detector, we assign arbitrary disc levels to the axial slices. The processing flow is as follows:

1. Convert the image coordinates of disc levels to real-world coordinates.
2. Estimate the vertebral positions of L1, L2, ..., S1 from the midpoints of each disc level (e.g., L1/L2, L2/L3, etc.). Since we cannot obtain coordinates for T12/L1 and S1/S2, we pseudo-calculate the coordinates for L1 and S1.
3. Calculate the intersection points between the line segments connecting adjacent vertebrae and the axial planes, and assign the corresponding disc levels.

### 4. Spinal Canal Keypoint Detector (CenterNet)

We constructed a CenterNet-based 2D keypoint detector using EfficientNetB4 as the backbone and FPN as the neck. By inputting axial images, we estimate the coordinates of the spinal canal. Since the Y-coordinate can be accurately estimated from the results of the Disc Level Keypoint Detector but estimating the X-coordinate is challenging, we introduced this detector. For training data, we used axial images from RSNA2024 that have coordinates of Spinal Canal Stenosis.

### 5. Crop Spinal

Using the output from the Spinal Canal Keypoint Detector, we crop the necessary regions centered on the spinal canal. To ensure diversity in the input data, we cropped at multiple sizes. There was almost no difference in accuracy due to cropping methods.

[crop_axial]

## Stage 2

The responsibility of Stage 2 is to estimate the severity of each condition using the outputs from Stage 1.

### 6. Center Classifier (2D-Encoder + Attention)

We created a classification model to estimate the severity of Spinal Canal Stenosis from sagittal T1, sagittal T2/STIR, and axial T2 images. For sagittal T1 and sagittal T2/STIR, we input 15 slices at equal intervals into an encoder to generate feature representations for each slice. For axial T2, we input 10 slices at equal intervals. These slice features are then input into an attention mechanism to learn the relationships between slices. To ensure model diversity, we created two models with different head structures. To improve accuracy, we used auxiliary losses such as the severity of other conditions and slice-level predictions. Additionally, increasing the loss weight for the Severe class, due to the metric specifications, was effective. Test-time augmentation (TTA) by flipping axial images also improved the leaderboard score.

[classifier]

After verifying various patterns of input-output combinations for the model, we found that the most effective approach was to input groups of sagittal T1, sagittal T2, and axial T2 slices at arbitrary disc levels to estimate the severity of Spinal Canal Stenosis. During training, we treated each group of slices as an independent data point without depending on the disc level. In other words, the model is designed to consistently learn the characteristics of Spinal Canal Stenosis from slice groups at any disc level and predict the severity based on those features, without specializing in any specific disc level. This allowed us to secure five times the amount of data per condition, which we believe contributed to the improvement in accuracy.

We used the following encoders:

- ResNet18 (160x160, 224x224)
- MNasNet-S (224x224)
- EfficientNet-B4 (224x224)
- EfficientNetV2-RW (224x224)
- EfficientNetV2-S (224x224)
- ConvNeXt-N (224x224, 320x320)
- ConvNeXt-T (224x224, 320x320)
- MaxViT-N (256x256)

#### Training

The basic training settings are as follows:

- 10–20 epochs
- AdamW with learning rate `lr=0.000025`, OneCycleLR scheduler (Warmup for 3/10 steps of the total)
- Batch size: 2–8
- Cross-Entropy Weight: `[1.0, 2.0, 4.0]`
- `drop_path_rate` = 0.2 or 0.3
- Augmentations:
  - `RandomBrightnessContrast`
  - `Blur`
  - `Distortion`
  - `ShiftScaleRotate`
  - `CoarseDropout`
  - `Mixup` (Optional)

### 7. Split LR

We designed preprocessing steps for training and inference of the Side Classifier. We split the sagittal and axial images into the left and right sides of the body. For the right-side data, we reversed the order of sagittal slices and horizontally flipped the axial images. This allowed us to handle the left and right sides uniformly and effectively doubled the amount of data available for training.

### 8. Side Classifier (2D-Encoder + Attention)

We created a classification model to estimate the severity of Neural Foraminal Narrowing and Subarticular Stenosis from sagittal T1, sagittal T2/STIR, and axial T2 images. The model structure and the number of input slices are identical to those of the Center Classifier.

After verifying various patterns of input-output combinations for the model, we found that the most effective approach was to input groups of sagittal T1, sagittal T2, and axial T2 slices at arbitrary disc levels and arbitrary sides (left or right) to estimate the severity of Neural Foraminal Narrowing and Subarticular Stenosis. By applying the Split LR preprocessing and flipping the right-side images to increase data, prediction accuracy improved. During training, we treated each group of slices as an independent data point without distinguishing disc levels or sides of the body. In other words, the model is designed to learn the characteristics of these conditions from the input slice groups, regardless of disc level or side, and estimate the severity based on those features. This allowed us to secure ten times the amount of data per condition, which we believe contributed to the improvement in accuracy.

## Team Validation Strategy

- **StratifiedKFold**
  - `y`: Number of moderate or higher severity cases included in one study
  - `groups`: `study_id`
- Reference code: [Lumbar RSNA 2024 EDA + 3D Visualizationn](https://www.kaggle.com/code/artemtprv/lumbar-rsna-2024-eda-3d-visualization)

## Pseudo Labeling

For items without ground truth labels, we used the predictions of the trained model as soft labels. The change in accuracy due to the presence or absence of pseudo-labels was not significant, but we introduced the use of pseudo-labels as an option to ensure model diversity.

## Ensemble

The highest CV score for a single model was **0.3858**, achieved by combining the Center Classifier (Type B) ConvNeXt-T and the Side Classifier (Type B) ConvNeXt-N.

The ensemble CV score was **0.3643**, obtained by simply averaging 30 models (15 Center models and 15 Side models) with diversity in input image types, model architectures, data augmentations, auxiliary losses, pseudo-labels, etc.

## Post-processing

We applied Temperature Scaling with a temperature of **0.91** to the logits of Spinal Canal Stenosis, sharpening the predicted probabilities.

[temperature_scaling]

## What Didn't Work

- One-stage solution
- Multi-level Multi-disease models
- Multi-level Single-disease models
- Models specialized for each disc level
- Models specialized for each side of the body
- 3D-CNN
- 2.5D-CNN + Attention
- 2D-CNN + LSTM
- Focal Loss
- Long epochs

## Code (Updated on 2024-10-27)

https://github.com/Moyasii/Kaggle-2024-RSNA-Pub

## Video (Updated on 2024-10-28)

https://www.youtube.com/watch?v=e2uRj5f9Lms&ab_channel=sugupoko
