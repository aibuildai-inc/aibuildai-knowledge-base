# 2nd place solution

Competition: physionet-ecg-image-digitization
Rank: #2
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/2nd-place-solution

# Acknowledgements

I would like to thank the organizers and Kaggle staff for hosting and running this excellent competition. I also extend my sincere gratitude to @hengck23 for publishing outstanding baseline notebooks and discussions.

# Overview

My pipeline uses @hengck23's [public implementation](https://www.kaggle.com/code/hengck23/demo-submission) for stage 0 and stage 1 without modifications. Therefore, this solution primarily focuses on stage 2 segmentation and post-processing techniques.

Key innovations:
- Replace competition time-series data with original PTB-XL Dataset signals (500Hz)
- Predict signal sampling positions directly using sparse masks
- Build a 2.5D segmentation model (series model) that fuses phase and amplitude information across leads
- Develop a whole-image segmentation model (whole model) combining timm encoders with @hengck23's `MyCoordUnetDecoder`

**Note**: In this solution writeup, I refer to each row of the standard 12-lead ECG as "series (0-3)".

# Strategy

The competition metric computes SNR for each image, averages it in the linear SNR domain, and then converts it to SNR(dB)([discussion](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/discussion/663901)).

As a result, pushing medium-to-high SNR images higher tends to improve the final score more than spending effort on the hardest low-SNR cases. After analyzing out-of-fold (OOF) predictions, I therefore focused on improving medium-difficulty and easy images, which were also more numerous and easier to improve.

To minimize prediction errors by staying close to sampling points, I refined my approach in three key areas:
- Segmentation mask creation
- Modeling architecture
- Post-processing methods

# Data

## Competition Data

The competition data is a subset of the [PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.3/).

While the PTB-XL dataset provides all time-series data at a 500Hz sampling frequency, the competition data uses multiple frequencies: 250, 256, 500, 512, 1000, and 1025Hz. This indicates that the competition organizers resampled the original 500Hz signals to these different frequencies.

I matched the competition data back to the original PTB-XL time-series by computing correlation coefficients, and replaced all signals with their 500Hz originals.

[500hz]

This replacement improved consistency in sampling positions, which led to noticeable CV improvements during training. A single fold model's LB score boosted from 21.67dB to 22.49dB.

## Synthetic Data

Since I achieved satisfactory scores with the competition data alone, and further improvements from synthetic data were marginal, I did not invest much time in additional data synthesis (using PTB-XL dataset and ECG-Image-Kit).

Considering that the private dataset might contain difficult `0015` type wrinkled data, I added a small amount of randomly selected `0015` data to the training set each epoch.

## Segmentation Mask

The method for creating segmentation masks is crucial. This is because the SNR achieved when reconstructing signals through the pipeline (time-series → segmentation mask → time-series) directly correlates with the model's upper performance limit.

Dense masks (covering the entire signal line) did not yield high reconstruction SNR. Instead, I created sparse masks that annotate at most 2 pixels per column.

To enable reconstruction at sub-pixel precision in the post-processing stage, I distribute labels to two pixels (the integer part and integer part + 1) according to the fractional part of the y-coordinate. During reconstruction, I convert these to time-series data by computing a weighted average of y-coordinates using label values as weights.

[mask_recostruction]

To map from 500Hz (sig_len=5000) to masks without resampling, I set the mask width to 5600 pixels and drew masks in the range [301:5301], covering 5000 pixels.

Below are the created masks and OOF prediction results from a model trained on these masks (yellow: GT, green: prediction).

[mask_prediction_rev3]

# Models

I used two types of models:

- whole model : A U-Net model combining timm encoders with @hengck23's `MyCoordUnetDecoder`.
- series model : A 2.5D segmentation model that fuses series images. I use this to share phase and amplitude information between leads.

## Whole Model Architecture

I based the whole model architecture on @hengck23's model. I changed the encoder to timm encoders from segmentation_models_pytorch. I cropped the top portion of the image and resized it to (1280, 5600).

## Series Model Architecture

Specific ECG leads have strong correlations (e.g., Einthoven's Law). To incorporate these relationships into the model, I built a 2.5D model that takes four series as input.

I crop each series within a ±3mV (240 pixel) range centered on the zero mV y-coordinate position. A shared U-Net encoder processes each series image, then I fuse features across series in the connection paths to each U-Net decoder layer. The decoder extracts features that the segmentation head converts to mask predictions.

[model_v5]

### Fusion Module

I tested three fusion module variants:
- conv2d
- shared conv2d
- conv3d

Typically, 2.5D models use conv3d, LSTM, or transformers to extract depth-wise information. LSTM did not work well (possibly due to poor parameter settings). I did not try transformers due to limited experience.

Since I wanted to mix all depth (series stacking order) information together, I also experimented with feature fusion using conv2d. While differences between variants were small, conv2d showed the best CV results.

#### conv2d
I apply conv2d blocks (Conv2d→BatchNorm2d→ReLU) for feature reduction and feature fusion.

[conv2d_fusion]

#### shared conv2d
To save parameters, shared conv2d shares the reduce conv2d block across all series.

#### conv3d
I reshape features to (C, 4, H, W) and apply conv3d blocks (Conv3d→BatchNorm3d→LeakyReLU) multiple times.

## Model Comparison

To verify the fusion module's effectiveness, I compared prediction results.

The top row shows images with unmasked signals and GT masks. The middle and bottom rows show images with masked regions (simulating image artifacts) and predictions from the whole model and series model, respectively.

While the whole model fails to predict the masked regions, the series model shares information across leads, enabling it to reasonably predict peak phase and amplitude.

[masked_prediction]

# Training Strategy

## Parameters
- Input image: Stage 1 rectified image
  - whole model: input shape = (3, 1280, 5600)
  - series model: input shape = (4, 3, 480, 5600)
- Loss: BCEWithLogitsLoss(pos_weight=20)
- Epoch: 50
- Batch size: 4
- Optimizer: AdamW
  - Learning rate: 5e-4 ~ 1e-3
  - Weight decay: 0.01
- Scheduler: CosineAnnealingLR

**Tip: Gradient checkpointing is effective for reducing activation memory consumption with high-resolution images**

## Augmentation
```python
image_only_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.2),
    A.RandomShadow(p=0.2),
    A.GaussianBlur(p=0.2),
    A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(0.01, 0.1), hole_width_range=(0.01, 0.05), fill=0, p=0.1),
    A.ToGray(p=0.25),
])
```
```python
image_and_mask_aug = A.HorizontalFlip(p=0.5)
```

# Post-Processing

## Mask Post-Processing
I compute a weighted average for each column of the mask using prediction results to obtain sub-pixel level predictions.

## Resampling Method
I tested:
- scipy.signal.resample
- scipy.signal.resample_poly(padtype='line')
- torch.nn.functional.interpolate(mode="linear")

scipy.signal.resample gave the best results.

## Pixel to Series
Code for the complete mask→time-series conversion pipeline, including mask post-processing and resampling:

```python
def pixel_to_series(pixel, length):
    _, H, W = pixel.shape
    eps=1e-8
    y_idx = np.arange(H, dtype=np.float32)[:, None]

    series = []
    for j in [0, 1, 2, 3]:
        p = pixel[j]
        denom = p.sum(axis=0)
        y_exp = (p * y_idx).sum(axis=0) / (denom + eps)
        series.append(y_exp)
    series = np.stack(series).astype(np.float32)

    if length!=W:
        resampled_series = []
        for s in series:
            rs = signal.resample(s, length).astype(np.float32)
            resampled_series.append(rs)
        series = np.stack(resampled_series)
    return series
```

# Inference

## TTA
- Horizontal flip

## Results

### Submission A (6 models)
**Public LB:** 23.37, **Private LB:** 23.27

### Submission B (11 models)
**Public LB:** 23.34, **Private LB:** 23.23

| Encoder                       | Model (fusion module)    | Synthetic Data | Public LB | Private LB | Submission A | Submission B |
|-------------------------------|------------------------|----------------|-----------|------------|-------|-------|
| EfficientNet B7               | whole                  |                | 22.93     | 22.65      | ☑️    | ☑️    |
| EfficientNetV2 L             | whole                  | ✅             | 22.60     | 22.55      | ☑️    | ☑️    |
| EfficientNetV2 M             | whole                  |                | 22.52     | 22.38      |       | ☑️    |
| EfficientNetV2 M             | whole                  | ✅             | 22.58     | 22.39      |       | ☑️    |
| EfficientNet B6               | series (shared conv2d) | ✅             | 23.10     | 22.92      | ☑️    | ☑️    |
| EfficientNet B6               | series (shared conv2d) |                | 23.00     | 22.81      | ☑️    | ☑️    |
| EfficientNet B4               | series (shared conv2d) | ✅             | 22.93     | 22.73      |       | ☑️    |
| EfficientNetV2 L             | series (conv3d)        |                | 22.92     | 22.77      | ☑️    | ☑️    |
| EfficientNetV2 L             | series (conv2d)        |                | 22.85     | 22.70      | ☑️    | ☑️    |
| EfficientNetV2 M             | series (conv3d)        | ✅             | 22.73     | 22.49      |       | ☑️    |
| EfficientNetV2 M             | series (conv2d)        | ✅             | 22.76     | 22.55      |       | ☑️    |

# References

- https://www.kaggle.com/code/hengck23/demo-submission
- Shivashankara KK, Deepanshi, Shervedani AM, Reyna MA, Clifford GD, Sameni R. ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization. Physiological Measurement 2024; 45:055019. DOI: 10.1088/1361-6579/ad4954
- Reyna MA, Deepanshi, Weigle J, Koscova Z, Campbell K, Shivashankara KK, Saghafi S, Nikookar S, Motie-Shirazi M, Kiarashi Y, Seyedi S, Hassannia M, Bjørnstad AM, Stenhede E, Ranjbar A, Clifford GD, and Sameni R. ECG-Image-Database: A dataset of ECG images with real-world imaging and scanning artifacts; a foundation for computerized ECG image digitization and analysis, 2024. DOI: 10.48550/arXiv.2409.16612.
- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kfzx-aw45
- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020), PTB-XL: A Large Publicly Available ECG Dataset. Scientific Data. https://doi.org/10.1038/s41597-020-0495-6
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

# Code Availability

- Training code: https://github.com/someya-takashi/physionet-image-digitization/tree/main
- Inference code: https://www.kaggle.com/code/takashisomeya/physionet-2nd-place-submission
