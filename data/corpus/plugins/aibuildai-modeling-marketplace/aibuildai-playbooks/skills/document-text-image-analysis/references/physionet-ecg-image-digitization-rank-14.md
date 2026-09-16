# 14th Place Solution

Competition: physionet-ecg-image-digitization
Rank: #14
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/14th-place-solution

# PhysioNet ECG Digitization Challenge - Solution Writeup

## Overview
This solution addresses the PhysioNet challenge of digitizing ECG images—converting printed/photographed ECG recordings back into numerical waveform data. The approach uses a **3-stage deep learning pipeline** that progressively transforms raw ECG images into high-fidelity digital waveforms.

**Final Public Leaderboard Score: 21.56 SNR**
**Final Private Leaderboard Score: 21.37 SNR**

---

## Problem Statement

ECG images from various sources (printed records, mobile phone photos, scans of damaged documents) need to be converted back to digital waveforms. The challenge involves:

1. **Diverse image sources**: High-quality prints, mobile photos, screen captures, damaged/moldy documents
2. **Geometric distortions**: Perspective warps, rotations, varying aspect ratios
3. **Visual degradations**: Water stains, fold creases, mold, low resolution, compression artifacts
4. **Layout parsing**: Standard 4-strip ECG layout with 12 leads arranged in specific positions

---
## Solution Architecture

### Three-Stage Pipeline

[Pipeline Diagram]

### Stage 0: Normalization

**Purpose**: Detect lead markers and correct image orientation

- **Architecture**: ResNet18 encoder + U-Net decoder
- **Outputs**: 
  - Lead marker segmentation (13 leads + background)
  - Image orientation classification (8 orientations)
- **Processing**: Applies homography to normalize image to canonical size (3024×4032)

> **Note**: For this stage, I utilized the pre-trained Stage 0 checkpoint provided in the [original baseline](https://www.kaggle.com/code/hengck23/demo-submission).

### Stage 1: Rectification

**Purpose**: Detect and correct grid distortions

- **Architecture**: ResNet34 encoder + U-Net decoder  
- **Outputs**:
  - Grid point detection (keypoints for perspective correction)
  - Horizontal line classification (44 classes)
  - Vertical line classification (57 classes)
- **Processing**: Uses detected grid points to apply perspective transformation, producing a perfectly aligned rectified image

> **Note**: Similar to Stage 0, I used the pre-trained Stage 1 checkpoint from the [original baseline](https://www.kaggle.com/code/hengck23/demo-submission).

### Stage 2: Waveform Prediction

**Purpose**: Extract ECG waveforms from rectified images

This is the main prediction stage where most of the innovation lies.

#### Model Architecture

[Stage 2 Architecture]

#### Key Design Decisions

1. **HRNet-W48 Backbone**: Maintains high-resolution representations throughout the network via parallel branches, crucial for precise pixel-level waveform prediction

2. **CoordConv Decoder**: Injects normalized 2D coordinates at each decoder block, enabling position-aware predictions essential for ECG layout understanding

3. **RGB Skip Connection**: Direct connection from input image to prediction head preserves fine spatial details that may be lost in encoder compression

4. **PixelShuffle Upsampling**: Memory-efficient learnable upsampling using sub-pixel convolution instead of transposed convolution

5. **GroupNorm**: Enables stable training with batch size 1 (necessary for high-resolution 1696×4352 images on limited GPU memory)

---

## Training Strategy

### Loss Function

The model uses **Binary Cross-Entropy (BCE) with positive class weighting** for pixel-level supervision:

```python
loss = F.binary_cross_entropy_with_logits(
    pixel_logit,
    target_mask,
    pos_weight=10.0  # Handle sparse ECG line pixels
)
```

This was found to outperform regression-based losses (MSE, SNR) for this task.

### Data Augmentation

A comprehensive **segment-aware augmentation pipeline** was developed to match the diverse training data distribution:

#### Universal Augmentations
| Augmentation | Probability | Purpose |
|-------------|-------------|---------|
| Perlin Noise | 85% | Simulate scanner noise, paper texture |
| Texture Noise | 70% | Grid-like artifacts, scan lines |
| Color Jitter | 80% | Brightness, contrast, saturation, hue variation |
| Gaussian Noise | 50% | Scanner ISO noise |
| Gaussian Blur | 35% | Low-quality scans, out-of-focus photos |
| Shadow Overlay | 35% | Uneven lighting |
| Paper Texture | 30% | Physical paper grain simulation |
| JPEG Artifacts | 30% | Compression artifacts |
| Horizontal Flip | 50% | Data augmentation (with label flip) |

#### Segment-Specific Augmentations

The training data contains 12 segment types with distinct characteristics. The augmentation pipeline applies targeted degradations based on segment type:

| Segment | Characteristics | Applied Augmentations |
|---------|-----------------|----------------------|
| 0003 | Standard color scans | Baseline augmentations only |
| 0004 | Black & white scans | Grayscale conversion + paper tint |
| 0005 | Mobile phone photos | Vignetting + shadow gradients |
| 0006 | Screen photos | Moiré patterns + screen glare |
| 0009 | Water-damaged | Water stains + color bleeding |
| 0010 | Extensively damaged | Fold creases + heavy noise |
| 0011, 0012 | Moldy scans | Mold patches + grayscale |

#### Augmentation Caching

To avoid expensive on-the-fly augmentation operations (especially Perlin noise generation), a **memory-mapped caching system** pre-computes augmentation masks:

- Perlin noise cache: 512 pre-computed noise maps
- Texture cache: 256 patterns
- Shadow/vignette cache: 256 gradient patterns

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 1 (4352×1696 full resolution) |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.05 |
| Epochs | 80 |
| EMA Decay | 0.999 |
| Train/Val Split | 100/0 (full training set) |
| Mixed Precision | FP16 (AMP) |
| Gradient Checkpointing | Enabled for decoder |

---

## Inference Pipeline

### Test-Time Augmentation (TTA)

Horizontal flip TTA improves robustness:

1. Predict on original image
2. Horizontally flip input image
3. Predict on flipped image
4. Flip prediction back to original orientation
5. Average both predictions

### Einthoven's Law Correction

A physics-based post-processing step enforces the electrocardiogram constraint:

**Einthoven's Law**: `Lead II = Lead I + Lead III`

The correction redistributes violation errors across all three limb leads:

```python
error = II - (I + III)
I_corrected = I + α × error
III_corrected = III + α × error
II_corrected = II - α × error  # α = 0.33
```

### Waveform Extraction

The 4-strip pixel predictions are converted to waveforms via:

1. Apply softmax on y-dimension to get probability distributions
2. Compute weighted centroid (argmax or soft-argmax) for each x-column
3. Convert pixel coordinates to mV using calibration parameters
4. Split each strip into constituent leads based on temporal layout
5. Resample to target sample lengths specified in test.csv

---

## Ablation Studies

> **Note**: These scores are approximate. Changes were not isolated individually to facilitate faster iteration and save compute costs.

| Configuration | Public LB Score |
|---------------|-----------------|
| Baseline (BCE loss + comprehensive augmentations, 90/10 split) | ~15.31 |
| + High Resolution (1696×4352) | ~18.22 |
| + HRNet-W48 backbone (replacing ResNet) | ~20.90 |
| + 100/0 train/val split (use all training data) | ~21.01 |
| + Horizontal Flip TTA | ~21.28 |
| + Einthoven's Law correction | ~21.32 |
| + RGB Skip + PixelShuffle + Fusion Head | ~21.56 |

### Key Insights

1. **Resolution matters**: Increasing from standard resolution to 1696×4352 provided the biggest single improvement (~3 SNR)

2. **HRNet superiority**: The multi-resolution parallel branches preserve spatial precision better than traditional encoder-decoder architectures

3. **Full training data**: With comprehensive augmentation, using 100% of data for training outperformed keeping a validation set

4. **Physics constraints help**: Einthoven correction provides consistent small improvements by enforcing known ECG relationships

---

## Technical Implementation Details

### Memory Optimization

Training on high-resolution images (1696×4352) with batch size 1 required careful memory management:

- **Gradient Checkpointing**: Enabled for decoder blocks to trade compute for memory
- **GroupNorm**: Used instead of BatchNorm for stable single-sample training
- **PixelShuffle**: Used instead of ConvTranspose2d for 2× upsampling (more memory efficient)
- **Mixed Precision (FP16)**: Enabled via PyTorch AMP

### Distributed Training

- DDP (Distributed Data Parallel) support for multi-GPU training
- Per-worker augmentation cache to avoid contention
- Memory-mapped cache files shared across workers

---

## Repository Structure

```
├── stage0_model.py          # Stage 0: Marker detection & normalization
├── stage1_model.py          # Stage 1: Grid detection & rectification
├── stage2_model.py          # Stage 2: Waveform prediction (main model)
├── predictor_dataset.py     # Training data loading & preprocessing
├── predictor_trainer.py     # Training loop with EMA, metrics
├── augmentations.py         # Comprehensive augmentation pipeline
├── augmentation_cache.py    # Memory-mapped augmentation caching
├── demo-submission.py       # End-to-end inference & submission generation
├── config_predictor.yaml    # Training configuration
└── archive/
    ├── stage0-last.checkpoint.pth  # Stage 0 pretrained weights
    └── stage1-last.checkpoint.pth  # Stage 1 pretrained weights
```

---

## Code

- **GitHub Repository**: [starrynites/PhysioNet_Digitization_of_ECG_Images](https://github.com/starrynites/PhysioNet_Digitization_of_ECG_Images)
- **Kaggle Inference Notebook**: [sshiyu/physionet-final-submission](https://www.kaggle.com/code/sshiyu/physionet-final-submission)

## Conclusion

This solution achieves strong performance on ECG digitization through:

1. **Robust preprocessing**: Stage 0+1 pipeline handles diverse image sources and geometric distortions
2. **High-resolution prediction**: Full-resolution HRNet enables precise pixel-level waveform detection
3. **Comprehensive augmentation**: Segment-aware degradation simulation improves generalization
4. **Physics-informed post-processing**: Einthoven correction enforces known ECG constraints
5. **Efficient architecture**: RGB skip connection + PixelShuffle fusion preserves fine details while maintaining training efficiency

Hope you enjoyed my writeup. Thanks for reading!

---

## Acknowledgements

This solution builds upon the excellent work shared by the Kaggle community:

- [hengck23/demo-submission](https://www.kaggle.com/code/hengck23/demo-submission) — Original low resolution baseline
- [wasupandceacar/physio-v2-3-public](https://www.kaggle.com/code/wasupandceacar/physio-v2-3-public) — High resolution baseline reference
- [tonylica/physionet-ecg-streamlined-inference](https://www.kaggle.com/code/tonylica/physionet-ecg-streamlined-inference) — Einthoven's law correction idea

### References

- Shivashankara KK, Deepanshi, Shervedani AM, Reyna MA, Clifford GD, Sameni R. ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization. Physiological Measurement 2024; 45:055019. DOI: 10.1088/1361-6579/ad4954
- Reyna MA, Deepanshi, Weigle J, Koscova Z, Campbell K, Shivashankara KK, Saghafi S, Nikookar S, Motie-Shirazi M, Kiarashi Y, Seyedi S, Hassannia M, Bjørnstad AM, Stenhede E, Ranjbar A, Clifford GD, and Sameni R. ECG-Image-Database: A dataset of ECG images with real-world imaging and scanning artifacts; a foundation for computerized ECG image digitization and analysis, 2024. DOI: 10.48550/arXiv.2409.16612.
- Reyna MA, Deepanshi, Weigle J, Koscova Z, Campbell K, Seyedi S, Elola A, Bahrami Rad A, Shah AJ, Bhatia NK, Clifford GD, Sameni R. Digitization and Classification of ECG Images: The George B. Moody PhysioNet Challenge 2024; Computing in Cardiology 2024; 51: 1-4.
- Matthew A. Reyna, Deepanshi, James Weigle, Zuzana Koscova, Kiersten Campbell, Salman Seyedi, Andoni Elola, Ali Bahrami Rad, Amit J Shah, Neal K. Bhatia, Yao Yan, Sohier Dane, Addison Howard, Gari D. Clifford, and Reza Sameni. PhysioNet - Digitization of ECG Images. https://kaggle.com/competitions/physionet-ecg-image-digitization, 2025. Kaggle.

---
