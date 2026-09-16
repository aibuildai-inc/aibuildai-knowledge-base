# 1st Place Solution

Competition: rsna-intracranial-aneurysm-detection
Rank: #1
Source: https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution

I thank RSNA, the organizers, the contributing radiologists and institutions, and the Kaggle team for hosting this impactful challenge. This write-up summarizes my 1st place solution. The core of my approach is a robust, coarse-to-fine pipeline that uses vessel segmentation to guide a region-of-interest (ROI) based classifier, producing location-aware predictions.

## Solution Overview

- **High-Level Pipeline**
  1.  **Preprocessing:** Convert and standardize DICOM series into NIfTI volumes.
  2.  **Vessel Segmentation & ROI Extraction:** Use a coarse-to-fine nnU-Net approach. A fast, low-resolution model first finds a candidate region. This allows the more detailed, high-resolution models to focus only on this ROI, which improves both accuracy and speed.
  3.  **ROI Classification:** A 3D classification model, using the detailed vessel masks as input, predicts the probabilities for the 13 anatomical locations and the overall aneurysm presence.

- **Key Design Principles**
    - **Coarse-to-Fine Efficiency:** A fast, low-resolution model first scans the entire volume to find a candidate region. This allows the more detailed, high-resolution models to focus only on this ROI. This improves both accuracy and speed.
    - **Using Segmentation as a Structural Guide:** By providing the classification model with an explicit vessel mask, I give it a detailed map of the vessel structure. This helps the model to focus on the vessels, where aneurysms are located.

## Data Preparation
- I excluded approximately 60 series due to data quality issues, such as orientation anomalies, corrupted DICOM files, and implausible slice spacing.
- I used multilabel-stratified 5-fold cross-validation. 

## Pipeline

### 1. Preprocessing
- **Filter Slices:** Within each series, I retained only the images matching the majority `Rows × Cols` and `PixelSpacing` configuration and filtered out slices with outlier interslice spacing to ensure consistency.
- **Convert DICOM to NIfTI:** I used `dcm2niix` for conversion. If it failed, I first ran `gdcmconv --raw` as a fallback before retrying the conversion [1].
- **Standardize Orientation:** All volumes were reoriented to a consistent anatomical orientation using nnU-Net’s `SimpleITKIOWithReorient`.
- **Normalize Intensity:** I applied nnU-Net’s standard per-volume z-score normalization to standardize image intensities.

### 2. nnU-Net Segmentation + ROI Extraction (Coarse-to-Fine)
This stage uses a sequence of three nnU-Net v2 [2] models (`nnUNetResEncUNetMPlans`, `3d_fullres` configuration) to first locate a coarse ROI and then produce detailed vessel segmentations within it.

- **Model 1: Coarse Vessel Localization**
  - **Spacing:** (1.0, 1.0, 1.0) mm
  - **Classes:** 3 vessel groups (Posterior+Basilar / MCA / Other)
  - **Loss:** Dice + Cross-Entropy
  - **Purpose:** To perform a fast, low-resolution scan to efficiently find a single ROI candidate for the high-resolution models. This model also supports an optional orientation correction step described later.

- **Model 2: Fine Segmentation (Balanced)**
  - **Spacing:** (0.80, 0.45, 0.44) mm
  - **Loss:** Dice + Cross-Entropy + SkeletonRecall (weight=1) [3]
  - **Purpose:** To generate a precise vessel segmentation. The SkeletonRecall loss improves the connectivity of thin vessels, which standard losses might miss.

- **Model 3: Fine Segmentation (Recall-Focused)**
  - **Spacing:** (0.80, 0.45, 0.44) mm
  - **Loss:** Tversky + Cross-Entropy + SkeletonRecall (weight=3)
  - **Purpose:** To complement Model 2 by prioritizing recall, making it more sensitive to detecting hard-to-find vessel segments.

- **Augmentation Strategy**
  - I disabled left–right mirroring for the fine models (2 and 3) to preserve anatomical asymmetry.
  - I used stronger intensity and geometric augmentations [4].
  - I added low-resolution simulation transforms to make the models robust to scans with thick slices.

- **Inference Process (Two-Stage)**
  - **Stage 1 (Coarse Scan):** I run Model 1 with a sliding window (overlap=0.2) and binarize the output to get a foreground mask. I then apply DBSCAN clustering to the mask to remove scattered false positives. The centroid of the largest cluster is used to crop a fixed-size ROI (140×140×140 mm).
  - **Stage 2 (Fine Inference):** I run Models 2 and 3 with a higher overlap (0.3) only within the coarse ROI. The vessel segmentation from Model 2 is used to compute a tight bounding box, which is then re-cropped with margins to create the final ROI for the classifier.

The detailed segmentations from this stage are passed to the classification model as masks.





### 3. ROI Classification (13 locations + Aneurysm Present)
Once the ROI and vessel masks are prepared, a 3D classification model predicts the final probabilities.

- **Model Architecture**
  - **Input Size:** The model takes ROI volumes of size 128 × 256 × 256 voxels.
  - **Backbone:** The core of the model is an nnU-Net pre-trained for the vessel segmentation task. This approach was more accurate and faster to train than standard 2.5D or 3D timm backbones.
      - **Decoder Simplification:** To improve efficiency, I simplified the decoder by removing its final, computationally-heavy block. This had no negative impact on performance.
  - **Auxiliary Detection Task:** To help the model learn specific features of aneurysms, I added an auxiliary task that uses the decoder features to reconstruct a small binary sphere (5-pixel radius) at the location of each annotated aneurysm.
  - **Location-Specific Prediction Head:** To predict the 13 location-specific probabilities, I use the following process:
      1.  **Per-Location Feature Pooling:** A "Vessel Region-Masked Pooling" layer uses the vessel masks to extract feature vectors corresponding to each of the 13 anatomical locations from the decoder's feature maps. (*Note: In practice, I apply this pooling using masks from both fine segmentation models and concatenate the results for more complete features.*)
      2.  **Feature Fusion:** These 13 feature vectors are combined with a global feature vector from the encoder (via Global Average Pooling).
      3.  **Inter-Location Modeling:** The combined features are fed into a "Location-Aware Transformer" to model relationships between different vessel locations.
      4.  **Classification:** Finally, an MLP head predicts the probability for each of the 13 locations.
  - **"Aneurysm Present" Prediction Head:** For the overall presence prediction, I pool features over the entire vessel structure (a union of all vessel masks) and combine them with the encoder's global features. This aggregated feature vector is passed to a separate MLP head.
  - **Output Design:** I treated each of the 14 labels as an independent binary classification problem. This design helps with the severe class imbalance, as positive cases for any single location are very rare.

- **Training Details**
  - **Loss Functions:**
      - **13 Locations:** `BCEWithLogitsLoss`.
      - **Aneurysm Present:** `BCEWithLogitsLoss`.
      - **Auxiliary Sphere Segmentation:** A combination of Balanced BCE [5] and Focal-Tversky++ loss [6, 7]. This combination works well for highly sparse targets and helps prevent the over-confidence that can occur with Dice-like losses.
      - **Loss Weights:** The final loss was a weighted sum of the three components. I set the weights to 0.1 for the 13 location losses, 0.05 for the Aneurysm Present loss, and 1.0 for the auxiliary sphere segmentation loss. The main goal was to prioritize learning the precise location of aneurysms, so the sphere segmentation task had the highest weight. I found that higher weights on the classification losses led to overfitting, so this balance was important.
  - **Data Augmentation**
      - **Intensity Transforms:** Gaussian noise, Gaussian smoothing, intensity shift/scale, contrast adjustment, Gaussian sharpening, and intensity inversion.
      - **Geometric Transforms:** Random flips (z, y, x axes), small rotations (±10°), scaling/shearing (±10%), mild grid distortions, and a simulated low-resolution transform.
  - **Optimizer and Schedule:** I used the AdamW optimizer with a learning rate of 1e-4 and an effective batch size of 8 (achieved with gradient accumulation). A standard cosine annealing schedule with a warmup period was used.
  - **EMA Weights:** I used the Exponential Moving Average (EMA) of the model weights for inference.

- **Inference**
  - **Ensembling:** The final predictions are an average of the models from 4 of the 5 cross-validation folds.
  - **Test-Time Augmentation (TTA):** I averaged the predictions from the original volume and a left-right flipped version of it.







### 4. Other Details
- **Fail-safe Mechanism:** If an anomaly occurred during the segmentation or ROI extraction steps, the pipeline would not attempt a prediction. Instead, it fell back to a set of pre-determined probabilities. For each class, this probability was the mean of the out-of-fold predictions from my cross-validation set.
- **Optional Orientation Correction:** I implemented a method to fix misoriented scans. It analyzed the spatial arrangement of the three vessel groups from the coarse segmentation. By comparing the relative positions of these groups to their expected anatomical locations, it estimated the correct axis permutation. This worked perfectly on the training set, fixing all identified orientation issues. However, it had no measurable effect on the leaderboard score, likely because the test set did not contain such orientation errors.



## Processing Time

- **Training**
  - The training times below were measured on a single NVIDIA RTX 4090.
  - The 96×192×192 input size was mainly used for faster experimentation and tuning.

| Model | Input size | Epochs | Training Time |
|---|---|---|---|
| nnU-Net (Model 2) | 64×192×192 | 1000 | 14 h |
| ROI classifier | 96×192×192 | 30 | 6 h |
| ROI classifier | 128×256×256 | 30 | 12 h |


- **Inference**
  - Inference times were measured on the Kaggle notebook using two T4 GPUs.
  - The times reported are an average over 100 samples.

| Step | Time per Series |
|---|---|
| Preprocessing | 4.03 ± 3.78 s |
| Vessel segmentation | 10.65 ± 4.10 s |
| ROI classification | 3.33 ± 0.09 s |
 
 
## Ablation Study
To validate the effectiveness of the key components in my ROI classification model, I conducted an ablation study. Note that this was a simplified evaluation; hyperparameters such as the number of epochs and learning rate were not re-tuned for each experiment. For efficiency, these experiments were run on folds 0, 1, and 2, using a reduced input size of 96×192×192.

| Model | Input Size | AUC (Aneurysm Present) | AUC (13 Locations) | Score |
|---|---|---|---|---|
| Final model (full resolution) | 128×256×256 | 0.915 | 0.916 | 0.916 |
| Final model | 96×192×192 | 0.907 | 0.898 | 0.902 |
| Without Location-Aware Transformer | 96×192×192 | 0.899 | 0.894 | 0.896 |
| Using Dice instead of FocalTversky++ | 96×192×192 | 0.902 | 0.896 | 0.899 |
| Setting all loss weights to 1.0 | 96×192×192 | 0.890 | 0.877 | 0.884 |
| Without backbone pretraining | 96×192×192 | 0.777 | 0.811 | 0.794 |
| Without segmentation model 3 | 96×192×192 | 0.899 | 0.883 | 0.891 |
| Without auxiliary segmentation loss | 96×192×192 | 0.880 | 0.871 | 0.876 |

The results show several key points:
- Pretraining the backbone on the vessel segmentation task was the most important factor. It greatly improved the score and helped the model train much faster. This was very helpful for running many experiments.
- The Location-Aware Transformer and the FocalTversky++ loss seemed to help at first, but their final contribution to the score was small. This is likely because other improvements and tuning had a larger overall effect.


## Design Journey
My final design was the result of several iterations:

1.  I initially tried a single, end-to-end 3D classifier that had auxiliary heads for vessel segmentation and aneurysm localization. While it could detect the presence of an aneurysm, it failed to predict the 13 specific locations accurately.
2.  I then observed that a standard nnU-Net for vessel segmentation trained easily and generalized well across all modalities. This led me to change to a two-stage, vessel-first pipeline, where the segmentation acts as a strong guide for the subsequent classification task.
3.  I also experimented with a simpler model that took a 2-channel input: the image volume concatenated with a single binary vessel mask. To get a prediction for a specific label, I would feed the model the corresponding mask (e.g., the mask for one location, or the union mask for "Aneurysm Present"). Although the model itself was simple, this approach required running 14 separate forward passes to get all predictions for a single patient series, which was too computationally expensive.
4.  This led to my final approach using a single backbone pass with the region-masked pooling, which provided a good balance of accuracy and computational efficiency.


## References
[1] https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/598083
[2] Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
[3] Kirchhoff, Yannick, et al. "Skeleton recall loss for connectivity conserving and resource efficient segmentation of thin tubular structures." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.
[4] https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/writeups/mic-dkfz-2nd-place-solution-3d-nnu-net-blob-regres
[5] https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/yu4u-tattaka-4th-place-solution-source-codes-submi
[6] Yeung, Michael, et al. "Calibrating the dice loss to handle neural network overconfidence for biomedical image segmentation." Journal of Digital Imaging 36.2 (2023): 739-752.
[7] https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/tomoon33-6th-place-solution

## Code Release
- Training code: https://github.com/uchiyama33/rsna2025_1st_place
- Submission notebook: https://www.kaggle.com/code/tomoon33/rsna2025-submission-1st-place
