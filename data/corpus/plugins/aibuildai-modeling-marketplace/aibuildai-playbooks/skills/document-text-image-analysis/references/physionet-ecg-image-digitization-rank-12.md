# 12th place solution

Competition: physionet-ecg-image-digitization
Rank: #12
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/12th-place-solution

## Acknowledgements

We would like to thank Kaggle and the competition organizers for providing the dataset and the opportunity to participate. We also appreciate the community for discussions and sharing ideas that helped improve our approach.

---

## 1. Overall Pipeline

1. Orientation detection and correction using OCR
2. Detect the paper region using OpenCV and crop it
3. Detect 17 grid points using a keypoint detection model
4. For the 17 detected keypoints, extend the four corner keypoints and add four additional points. Then, Delaunay triangulation is applied to a predefined Type1 keypoint layout, and each triangular region from the target image, defined by the corresponding detected keypoints, is warped onto this layout. To reduce information loss for high-resolution images, the reference keypoints are upscaled by ×2 during warping (as shown in the figures below).
5. Crop each lead region from the result of step 4, and predict the signal for each lead using a signal extraction model.

- Visualization of predicted and extended keypoints.  


- Visualization showing the image after Delaunay triangulation and warping. The bounding boxes indicate each lead's ROI used as model input.  


---

## 2. Data

Only competition data was used (no external or synthetic data).

---

## 3. Grid Keypoint Detection

1. Predict 17 individual keypoint masks (17 output channels) using a segmentation model.
2. Extract keypoint coordinates using connected components (`scipy.ndimage.label`) with weighted average.

### Model Architecture

A simple U-Net segmentation model with a `convnext_small.dinov3_lvd1689m` backbone, an additional decoder-head for PAF auxiliary loss, and an orientation detection head.  

### Loss  

Hybrid loss consisting of:
- Dice Loss  
- BCE Loss  
- PAF Loss  
- Cross-Entropy Loss for orientation classification  

### Training Process

1. Repeat the following process for N = 350, 450, ..., 700:
    - Manually annotate 17 grid keypoints for N images
    - Train the model using images with manually annotated ground truth labels (initialize weights from previous checkpoint if available)
    - Predict keypoints for all images and fine-tune the model using both ground truth and predicted pseudo keypoints
2. At this stage, accuracy was generally good except for the right edge points. Therefore, pseudo keypoints were imported back into the annotation tool, and only the wrongly predicted right-edge keypoints were manually corrected. The model was then fine-tuned using the complete annotation to obtain the final model.

### Augmentation

- OneOf:
    - ElasticTransform
    - GridDistortion
    - OpticalDistortion
- OneOf:
    - ShiftScaleRotate
    - Perspective
- RandomBrightnessContrast
- GaussianNoise
- ImageCompression
- CoarseDropout
- Rotation 90/180/270 degrees

---

## 4. Signal Extraction

1. Signal segmentation using a U-Net segmentation module
2. Apply warping to the mask using a learnable warp field (output of another U-Net decoder branch) via 2D grid sampling
3. Predict signal time series from the warped segmentation mask using a 1D CNN-based regression head
4. Resample to 2.5× sampling frequency using a learnable optimized resampler with 1D grid sampling

All modules above were trained jointly.  
Since phase shifts caused by resampling were significant, I introduced a learnable resampler to optimize resampling.  
(For example, in experiments where only the last point of the final predicted `number_of_rows` was trimmed or padded and then interpolated back to `number_of_rows`, the public LB degraded from 21.49 → 20.12/20.00.)

### Model Architecture



- Backbone: `convnext_tiny.dinov3_lvd1689m` / `convnext_small.dinov3_lvd1689m`

### Loss  

Hybrid loss consisting of:
- Dice Loss for signal segmentation
- BCE Loss for signal segmentation
- SNR Loss for signal regression
- Warp smoothness loss for grid warping regularization

### Augmentation

- OneOf:
    - ElasticTransform
    - GridDistortion
    - OpticalDistortion
- RandomBrightnessContrast
- GaussianNoise
- ImageCompression
- CoarseDropout
- Use pseudo (predicted) keypoints instead of ground truth (manual) keypoints for region cropping
- Add Gaussian noise to keypoint coordinates
- Additional right edge keypoint noise
- Randomly select sampling frequency and resample GT signal

**Note:** The above augmentations are not applied to signal segmentation ground truth masks. The goal is to make the model learn distortion correction via the warping module.

### Preprocessing / Training

- For model input images, crop each lead's ROI using GT/predicted keypoints with constant height. Resize each to `(3, H_lead=384, W_LEAD=1536)` and stack to `(n_leads=16, 3, H_lead, W_LEAD)`. Additionally, top region images `(n_leads_top=4, 3, H_lead, W_LEAD)` are prepared similarly.
- For signal segmentation GT masks, after drawing the mask of the target lead, the target region including `y=0` and ±1 regions (as shown in the figure below) is cropped and resized to `(3, 3 × H_lead=1152, W_LEAD=1536)`, then stacked for all leads to `(n_leads=16, 3, 3×H_lead, W_LEAD)`.
- To handle cases where the signal extends beyond the target lead region, the model concatenates features from adjacent y ±1 regions in the encoder feature space, so the final single-lead predicted segmentation height is `3×H_lead`, matching the GT mask.

GT mask example for the V5 lead. The red bounding box indicates the cropped region used for training.


---

## 5. Ensemble

A simple weighted average ensemble worked best.  
Ultimately, three models (grid keypoint detection model was a single model) were used. With 90-degree rotation TTA, a total of 3 × 2 = 6 predictions were combined via weighted average for the best result.

- Single models

| Model Name | Backbone | Use FS PE | Public LB | Private LB |
| ---- | ---- | ---- | ---- | ---- |
| A | convnext_tiny.dinov3_lvd1689m | No | 21.49 | 21.18 |
| B | convnext_small.dinov3_lvd1689m | Yes | 21.12 | 20.84 |
| C (derived from A) | convnext_tiny.dinov3_lvd1689m | No | 21.36 | 21.03 |

- Ensemble

| No | A weight | B weight | C weight | (Keypoint Detection) Rot90 TTA | Public LB | Private LB |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1 | 0.45 | 0.4 | 0.15 | Yes | 21.86 | 21.57 |
| 2 | 0.4 | 0.4 | 0.2 | No | 21.81 | 21.50 |
| 3 | 0.5 | 0.2 | 0.3 | No | 21.76 | 21.45 |
