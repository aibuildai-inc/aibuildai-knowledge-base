# 9th Place Solution

Competition: physionet-ecg-image-digitization
Rank: #9
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/9th-place-solution

First of all, I would like to pay tribute to all the participants who worked on this competition.
I would also like to thank the hosts for organizing this interesting task competition.


# Overview

My solution consists of a two-stage pipeline: image rectify and signal reconstruction.

For image rectify, I first apply coarse alignment using a homography transformation based on image matching, then detect grid point coordinates and perform precise alignment using Piecewise Affine transformation.

For signal reconstruction, I extend the decoder of a UNet with a signal reconstruction module that directly outputs signals from images, and train the model to optimize SNR.

# Pipeline
## 1. Image rectify
### 1.a Image Type Classification
- Since performance is more stable when some image types (006) are handled separately, images are first classified by type.
- Only competition data is used for training, and a simple classification model with convnext_large_384_in22ft1k as the backbone is trained.
- There is nothing particularly special here, but the model achieves about 99.9% accuracy in CV.

## 1.b Image Matching

- Feature point matching with a template image is performed using ALIKED + LightGlue, followed by coarse alignment via a homography transformation estimated with RANSAC.
  - The template image is created by averaging competition images of image type 001.
  - Publicly available pretrained weights are used as-is for both ALIKED and LightGlue.
- To simultaneously correct rotation, the input image is rotated by 0°, 90°, 180°, 270°, matching is performed for each rotation, and the rotation with the largest number of matches is selected.
- Although some distortion remains at this stage, almost all images are aligned to the same orientation and composition as the template (001), which stabilizes downstream training and inference.

## 1.c Grid Detection
- A model is built to detect grid point coordinates from the projectively transformed images.
    - A UNet with ResNeSt-14d as the encoder is used.
- Only about 22,000 synthetic images are used for training.
    - Since image matching already roughly normalizes the input images, augmentation on synthetic data alone is sufficient to generalize to real data.
    - No annotation on real data is performed, significantly reducing annotation cost.
- For image type 006, false detections sometimes occur due to moiré patterns, so grid detection is applied after moiré removal using [UHDM](https://github.com/CVMI-Lab/UHDM).

## 1.d Precise Alignment
- Finally, precise alignment is performed using Piecewise Affine transformation based on the outputs of image matching and grid detection.
1. From the correspondence points obtained by image matching, regions with small reprojection error and minimal distortion are selected as initial regions. Detected grid points in these regions are matched to ideal grid points using the Hungarian algorithm.
2. Starting from the initial correspondences, assuming a grid spacing of approximately 40 px, correspondences are incrementally expanded to neighboring grid points using breadth-first search in the up/down/left/right directions.
3. The final set of correspondences is used as control points to apply a Piecewise Affine transformation to correct distortion over the entire image.


## 2. Signal Reconstruction

### Architecture
- Based on a UNet with a ResNeSt-14d backbone, an architecture is designed by adding a dedicated module to directly estimate signals from the decoder output.
    - Feature maps from the UNet decoder are expanded 6× along the temporal axis (W direction), then concatenated with sampling frequency (fs) and lead type as features.
    - After concatenation, convolution along the W direction is applied to output logits representing “signal-likeness” along the y direction for each x column.
    - From the y-direction logits, two soft expectations are computed: one biased toward the top edge and one toward the bottom edge, and a gate function predicts (0–1) which to use for each x.
    - The final estimated y (pixel coordinates) is converted to a signal value in mV based on ECG drawing geometry (paper size / resolution / row layout).

### Inputs, Outputs, and Loss Functions
- Inputs
  - Cropped images for each lead
      - Fixed crop size: H×W = 600×491
      - RGB images with positional encoding added for x and y directions, resulting in 5-channel input
      - The long lead (II) is split into 4 parts to match the temporal length of other leads
  - Target sampling frequency (fs)
  - Lead type
      - Adding target fs and lead type as input features improved CV performance by about +1.8 dB, but unfortunately did not yield a clear improvement on the LB.

- Outputs
  - Segmentation maps
      - Three classes: background, target signal to be reconstructed, and non-target signals
  - Reconstructed signal waveform

- Loss
  - Segmentation loss: Dice + CE, weighted 0.5 : 0.5
  - Waveform loss: after interpolating the predicted waveform to match the GT length, an SNR-based loss is used as the main loss, with L1 loss added as an auxiliary term
  - The final loss is a weighted sum, with emphasis on waveform reconstruction (SNR loss)


### Multi-stage Training
Training is performed in three stages, gradually switching datasets and augmentation strategies.

This multi-stage training yields approximately +1.0 dB from 1st-stage pretraining and an additional +0.2 dB from image-type-specific tuning in the 3rd stage, consistently on both Public and Private sets.

#### 1st Stage Training (Pretraining with Synthetic Data)
- Trained using about 21,000 synthetic images generated from PTB-XL (500 Hz), excluding overlaps with competition data.
- Image-only augmentations include blur, brightness changes, grid coordinate shifts up to 1 pixel via Piecewise Affine, and custom augmentations such as DirtPatch / CreaseWrinkles.
- Image-and-signal joint augmentations include random segment dropout and vertical/horizontal flips.
- GT signals are randomly resampled using scipy.resample_poly to one of
[250, 256, 512, 500, 1000, 1025].
- Trained for 15 epochs with lr = 4e-4.

#### 2nd Stage Training (Main Training)
- Initialized with weights from the 1st stage and trained using competition data only.
- No image-only augmentation is applied; only image+signal augmentations from the 1st stage are used.
- Trained for 30 epochs with lr = 1e-4.

#### 3rd Stage Training (Image-Type-Specific Fine-tuning)
- Initialized with weights from the 2nd stage and further trained using only image type 006 data.
- Augmentation settings are the same as in the 2nd stage.
- Trained for 50 epochs with lr = 1e-4.
- Although CV performance improved for other image types as well, LB improvements were unstable, so the final submission uses the 3rd-stage model only for image type 006, and the 2nd-stage model for others.

### TTA
- During inference, predictions are made on the original image plus left-right flip, top-bottom flip, and both flips, totaling 4 patterns, and the results are averaged.
- This TTA consistently provides about +0.15–0.2 dB improvement on both Public and Private sets.


# Score
| # | synthetic_data pretrain (1st stage train) | 006 fine tuning (3rd stage train) | TTA | full data train | use target_fs | use_lead_id | Public LB | Private LB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  | 20.17 | 20.11 |
| 2 |  |  | ✅ |  |  |  | 20.32 | 20.26 |
| 3 | ✅ |  | ✅ |  |  |  | 21.43 | 21.30 |
| 4 | ✅ |  | ✅ | ✅ |  |  | 21.54 | 21.38 |
| 5 | ✅ | ✅ | ✅ | ✅ |  |  | 21.80 | 21.60 |
| 6 | ✅ | ✅ | ✅ | ✅ | ✅ |  | 21.66 | 21.56 |
| 7 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 21.81 | 21.71 |

**final submission**  
Ensemble #5 + #6 + #7
- Public: 22.04  
- Private: 21.92
