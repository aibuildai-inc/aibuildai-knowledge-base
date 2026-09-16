# 15th place solution

Competition: physionet-ecg-image-digitization
Rank: #15
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/15th-place-solution

Team mate: @fatliuyun 
# Overview
**ECG image → signal mask prediction → post-processed waveform extraction**

The core idea is to separate signal extraction from signal reconstruction. By learning to segment ECG traces at the pixel level, the model becomes significantly more robust to real-world artifacts such as grid distortion, scanning noise, paper folds, ink fading, and occlusions.

To capture both global ECG layout and fine-grained waveform details, two complementary training strategies are used:

- Full-crop training, which preserves the complete ECG context and long-range signal continuity.
- Half-crop training, which focuses on localized waveform structure and improves sensitivity to subtle signal variations.

Both strategies employ lightweight ResNet-based encoders (ResNet-18 and ResNet-34d) and share a common decoding head. Their predictions are combined during inference to improve robustness and generalization


# Rectified image and Mask generation
## Rectified image 
The early rectification version from @hengck23 suffered from a subtle boundary issue where the predicted grid did not fully span the image extent, causing the last 1–2 pixel columns to be mis-mapped or border-filled during warping. This was fixed by enforcing a fixed output canvas, normalizing grid points using (W−1, H−1), and enabling corner-aligned interpolation (align_corners=True) when upsampling the deformation field. As a result, the dense grid now covers the full image width, eliminating edge pixel loss and significantly improving pixel-level alignment between rectified images and supervision masks, especially at waveform endpoints.


` 



    # Trim unstable grid columns
    gridpoint_xy = gridpoint_xy[:, :56]

     # Empirical offsets to correct scanner margins
    offset_y = 7
    offset_x = 34

    # Remove noisy right-side region
    image = image[:, :2166]

    # Reference output size
    H, W = 1700 - offset_y, 2200 - offset_x

    # Allocate padded output canvas
    empty_image = np.zeros((H + offset_y, W, 3), np.uint8)

    # Normalize grid points to [-1, 1]
    H1, W1 = image.shape[:2]
    sparse_map = gridpoint_xy / [[[W1 - 1, H1 - 1]]] * 2 - 1
    sparse_map = torch.from_numpy(
        np.ascontiguousarray(sparse_map.transpose(2, 0, 1))
    ).unsqueeze(0).float()

    # Interpolate sparse grid to dense deformation field
    dense_map = F.interpolate(
        sparse_map,
        size=(H, W),
        mode="bilinear",
        align_corners=True
    )

    # Apply grid-based warping
    distort = torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    ).unsqueeze(0).float()

    rectified = F.grid_sample(
        distort,
        dense_map.permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )

    # Convert back to image format
    rectified = rectified.data.cpu().numpy()
    rectified = rectified[0].transpose(1, 2, 0).astype(np.uint8)

    # Restore vertical offset for alignment
    empty_image[offset_y:, :, :] = rectified

    return empty_image
`

## Mask generation
Number of series: 4

Output mask shape: (4, 1696, 4352)
(height × width corresponds to the rectified ECG image)
|  Series|Zero-MV  | mV --> Pixel
| --- | --- |
| 0 | 707  | 78.0
| 1 | 991  | 79.5
| 2 | 1273  | 78.5
| 3 | 1533  | 78.0

# Training
To balance global ECG layout understanding with local waveform precision, two complementary training strategies were employed: Full-Crop training and Half-Crop training. Each addresses a different failure mode observed during early experiments.

## Full-Crop Training (Global Context)

In full-crop training, the model is trained on the entire rectified ECG image, preserving the complete temporal span and vertical alignment of all leads.


- Captures long-range temporal continuity of ECG signals

- Preserves relative lead positioning and baseline consistency

- Improves stability near signal boundaries (start/end regions)

- Helps the model learn overall ECG structure and layout

## Half-Crop Training (Local Detail)

Half-crop training uses random horizontal crops covering approximately half the image width, while preserving full vertical resolution. Cropped regions are then resized back to the original dimensions before training.

Why half-crop matters:

- Forces the model to focus on local waveform morphology

- Increases effective resolution per signal segment

- Improves robustness to local noise, breaks, and occlusions

- Acts as a strong form of data augmentation

DATA AUGMENTATION STRATEGY
==========================

GEOMETRIC AUGMENTATIONS
-----------------------
Augmentation              | Parameters                                              | Probability
--------------------------|---------------------------------------------------------|------------
Affine transform          | Scale 0.90–1.10, Rotate ±8°, Translate ≤4%              | 0.60
Perspective transform     | Scale 0.02–0.07                                         | 0.40
Optical distortion        | Distort ≤0.03, Shift ≤0.02                              | 0.15
Elastic transform         | Alpha = 10, Sigma = 3                                   | 0.03
Grid distortion           | Steps = 3, Distort limit = 0.03                         | 0.05
Affine (shear)            | Shear ±3°                                               | 0.15

PHOTOMETRIC AUGMENTATIONS
-------------------------
Augmentation              | Parameters                                | Probability
--------------------------|-------------------------------------------|------------
Brightness / Contrast     | ±0.15                                     | 0.50 (OneOf)
CLAHE                     | Clip = 2.0, Grid = 8×8                    | 0.30 (OneOf)
Random Gamma              | 0.8–1.2                                   | 0.20 (OneOf)
Gaussian Noise            | Variance 5–20                             | 0.50 (OneOf)
ISO Noise                 | —                                         | 0.30 (OneOf)
Multiplicative Noise      | 0.9–1.1                                   | 0.20 (OneOf)
Hue / Saturation / Value  | H±2, S±5, V±5                             | 0.15

BLUR & SHARPENING
-----------------
Augmentation              | Parameters              | Probability
--------------------------|-------------------------|------------
Sharpen                   | Alpha 0.05–0.15         | 0.20
Gaussian Blur             | Kernel size 3–5         | 0.60 (OneOf)
Motion Blur               | Kernel size 3–5         | 0.40 (OneOf)

ECG-SPECIFIC OCCLUSIONS
----------------------
Augmentation              | Purpose                                   | Probability
--------------------------|-------------------------------------------|------------
CoarseDropout (small)     | Ink breaks / stains                       | 0.25
CoarseDropout (vertical)  | Printer / scan vertical lines             | 0.20
Perlin noise patches      | Paper dirt / aging artifacts              | ~0.50 (custom)



# POST-PROCESSING & SIGNAL REFINEMENT (STAGE-2)
======================================

After segmentation-based signal extraction, additional temporal refinement and gap filling are applied to recover physiologically consistent ECG waveforms. This post-processing focuses on handling missing segments, enforcing periodic consistency, and stabilizing QRS morphology.

1) MISSING-SEGMENT DETECTION & FILLING
-------------------------------------
- A binary mask indicates missing points (mask = 1).
- Consecutive missing runs longer than a minimum length are detected per lead.
- These runs are filled with the mean of nearby observed values in the same lead.
- Prevents long flat gaps while preserving baseline consistency.

2) BEAT-PHASE–AWARE IMPUTATION (KEY STEP)
-----------------------------------------
- ECG is divided into fixed 1250-sample segments.
- Global R-peaks are detected across all leads.
- For a missing point t:
  * Find nearest R-peak in the same segment.
  * Compute phase offset k = t − R_peak.
  * Collect corresponding points (R_j + k) from other cycles in the same segment.
  * If multiple valid values exist, average them to fill t.
- Enforces periodic consistency and preserves waveform morphology.

3) ROBUST R-PEAK DETECTION
-------------------------
- Band-pass filtering (5–18 Hz) emphasizes QRS complexes.
- Pan-Tompkins–style moving window integration (MWI) is applied.
- Energy envelopes are fused across leads.
- Adaptive thresholding with search-back logic recovers missed beats.
- Peaks are refined to the true positive QRS maximum.
- A reference lead with strongest QRS energy is selected automatically.

4) MODEL-LEVEL TEMPORAL REFINEMENT
---------------------------------
- Multi-scale temporal convolution (TCN) captures local and long-range dependencies.
- Fourier harmonic features model dominant cardiac periodicity.
- Gated fusion blends time-domain and frequency-domain information.
- Residual bidirectional LSTM enforces temporal smoothness.
- The model learns corrections only for missing regions; observed points are preserved.

5) OUTPUT GUARANTEES
-------------------
- Observed (non-missing) points are never modified.
- Filled values respect local continuity and global cardiac rhythm.
- No aggressive low-pass filtering is applied; sharp QRS morphology is preserved.
