# 3rd place solution

Competition: physionet-ecg-image-digitization
Rank: #3
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/3rd-place-solution

Thanks to PhysioNet and Kaggle for organizing this competition. Special thanks to @hengck23 for sharing the image rectification pipeline.

### Overview

- **High-Resolution Mapping Rectification:** Calculate geometric parameters using low-resolution images, then scale them up and apply them directly to the original high-resolution images. This helps avoid losing details.
- **Training:** Use strong data augmentation to make the model more robust.
- **Sub-pixel Precision:** A parabolic refinement method is used to turn pixel-level predictions into accurate floating-point values.

For full implementation details and source code, please refer to our GitHub repository and notebook:
- Training Code (GitHub): https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place
- Submition notebook: https://www.kaggle.com/code/hirotetsu/physionet-submission-3rd-place


### Mask Preparation
For mask generation, we draw a one-pixel-wide line using the signal data. We also tried making the line wider by adding a Gaussian blur to create softer labels, but this did not improve the results. Therefore, we decided to keep the one-pixel mask and intentionally trained an “overfitted” model so that it predicts thin waveforms with high confidence.

| Image Size |SNR (dB)  | Notes|
| --- | --- |
| 2200 x 1700 | 24.2235 | Standard dimension (200 DPI) |
| 4400 x 1700 | 31.2271 | Selected dimension |



1. **Wider Images:**
Increasing the width to 4400 pixels greatly improves SNR. ECG signals are time-series data, so more horizontal pixels mean better sampling and less quantization error.
    
2. **Fixed Height:** 
The height is kept the same. Keeping the waveform thickness close to 1 pixel helps the model predict the signal position more accurately and makes post-processing more stable.

We evaluated SNR for different mask sizes and found that larger masks usually give better SNR. However, due to limited inference memory and training results, we chose to double the width while keeping the original height. the problem here, is how can we keep the quality of images in the rectification process, which we addressed by separating the image transformation step from the rectification process.

### High-Resolution Mapping Rectification
The rectification model output relatively small size rectified images, which can cause quality loss during the rectification process. To avoid degradation caused by repeated resizing, we split the process into two steps:
- Parameter estimation (low resolution)
- Image transformation (high resolution)


The homography matrix and grid coordinates are scaled to match the size of the original raw image. This approach resulted in a significant improvement in the experiments

### Signal Segmentation Model
- **encoder:**
    - ConvNeXt V2
    - HRNet
    - EfficientNet V2

    In our experiments, the performance ranking was ConvNeXt V2 > HRNet > EfficientNet V2. For the final submission, we ensembled the ConvNeXt V2 and HRNet models by taking a weighted average of the post-processed signals, which will be introduced later, instead of directly averaging the segmentation probabilities. This signal-level ensembling produced better results.

- **decoder:**
    - UnetDecoder

- **Augmentation:**
    - GridDropout
    - CoarseDropout
    - AddPerlinDirt
    - MotionBlur
    - Downscale
    - GaussianBlur
    - ImageCompression
    - GaussNoise
    - ISONoise
    - ToGray
    - RandomBrightnessContrast
    - HueSaturationValue
    - RandomShadow
    - ElasticTransform
    - HorizontalFlip
    - VerticalFlip

    Considering that some samples in the hidden test data may be highly noisy, we applied very aggressive data augmentation during training. We added a small ElasticTransform to simulate distortions in the unwrapped images, where both the waveforms and grid lines may be slightly warped by the unwrapping process itself.

- **Loss:**
We trained the model using BCE loss. In practice, even at the lowest loss, the predicted waveforms were still too thick. To obtain thin, nearly one-pixel-wide waveform boundaries, we deliberately continued training an “overfitted” model and used the Dice score and CV score as the main criteria for model selection.

### Sub-pixel Signal Processing

We experimented with different strategies to convert pixel locations into time-series values. Two effective approaches are:
- Weighted Centroid：We selected the top-𝐾 pixels and computed the final position using the predicted probabilities as weights. This approach produced good results, but it is sensitive to the choice of 𝐾 and the weighting exponent.
- Parabolic Interpolation: It refines the integer argmax by fitting a local parabola to the peak and its two neighbors, giving a sub-pixel (floating-point) peak position.

We chose parabolic interpolation because it consistently delivered the best performance in our experiments. We believe this is because the model outputs very thin, sharp lines, for which parabolic interpolation provides a reliable way to estimate the true position without requiring hyperparameter tuning.



The offset is computed as:

delta = (y_left - y_right) / (2 * (y_left - 2*y_center + y_right))

The final position is:

Real Coordinate = Integer Index + delta

### Others
- **Lead II Fusion:** Merges the short Lead II prediction with the long rhythm strip (Series 3) by averaging the overlapping head region.
- **Einthoven Correction:** Applies Einthoven Correction to adjust Leads I, II, and III while enforcing the physical constraint II=I+III. Lead II is given higher importance (weight = 2), while Leads I and III use equal lower weights (weight = 1).
- **TTA:** original, horizontal flip, vertical flip, and both flips
