# 18th Place Solution

Competition: rsna-intracranial-aneurysm-detection
Rank: #18
Source: https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/18th-place-solution

# First of all
I would like to express my gratitude to the hosts for providing such a rewarding competition, to the Kaggle staff, and to all the participants who shared their insights in the discussions.

# Strategy
From the early stage of the competition, based on the provided dataset, I devised a strategy to generate ROIs using a segmentation model and then classify them.

# Pipeline overview
1. Stage1: Vessel Mask Prediction and ROI at the vessel-class level Extraction by a 2.5D Segmentation Model
2. Stage2: ROI Classification
3. Stage3: ROI Scoring and Series-level aggregation

# Learning Process
##  Stage1 (Segment Model)
- We performed slice-level mask prediction using smp.Unet with an EfficientNet-B0 backbone.
- For preprocessing, windowing was applied to CT images and percentile normalization to MRI images. Basic data augmentation techniques were also used.
- The input size was (C, H, W) = (3, 512, 512).
- To address class imbalance among negative, positive, and vessel-specific positive samples, we used a random sampler with slice-level weighting.
- Using 3-channel inputs achieved better accuracy than 1-channel inputs.
- The final Dice score was approximately 0.66.

## Stage1 (Vessel-class-level ROI Extraction）
- We extracted ROIs using a segmentation model.
Each ROI was generated for every vessel class as the minimum bounding rectangle of the predicted mask with an added margin.
When multiple separated regions of the same class were detected within a slice, individual ROIs were created for each object (e.g., class_i_object_j).
- For aneurysm-positive slices, ROIs were extracted according to the above definition, and those containing the aneurysm coordinates were labeled as positive ROIs, achieving approximately 94% hit rate.
- Negative ROIs were sampled from both negative series and regions located more than 40 mm (in xyz space) away from aneurysm coordinates in positive series.

## Stage2
- For ROI classification, a 2.5D input was created by applying the same bounding box to the ROI slice and its adjacent slices.
The image was resized while maintaining its aspect ratio, by scaling it so that the longer side matched the target size.
The final input size was (C, H, W) = (3, 224, 224), and padding was applied as necessary.
- Multiple backbone networks were evaluated, including EfficientNet-B0 and EfficientNet-V2-S.
The model achieved an ROI-level AUC of approximately 0.94.

## Stage3
- For each ROI, the aneurysm probability (prob) predicted by the ROI classification model was multiplied by the vessel class probabilities (class_scores) obtained from the segmentation model to compute the 13 class-wise scores.
The binary aneurysm score was taken directly from the ROI probability (prob) without any modification.
- Series-level aggregation was performed using the top-k mean of ROI-level scores.
- The local validation AUC reached approximately 0.81.

# Submission Process
- The DICOM slices were aligned in the LPS coordinate system, and only the last 150 mm along the z-axis (corresponding to the head region) were used for inference.
To limit the number of slices per series, we set max_slice = 200.
The best public leaderboard score achieved was 0.81.

# What Worked Well
- We successfully implemented ROI extraction and ROI-level classification.
- The derivation of class-wise scores also worked as intended.
# What Didn’t Work Well
- The overall score decreased after series-level aggregation.
Several post-processing methods were explored to filter out false-positive ROIs, but none of them were effective.
We should have considered classification methods that incorporate spatial context, such as 2.5D LSTM or 3D CNN.
- The performance on MRI T2 images remained consistently low, and even when using a dedicated model for this modality, no improvement was observed.

> Thank You for Reading!
