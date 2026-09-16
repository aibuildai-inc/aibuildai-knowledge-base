# 3rd place solution

Competition: rsna-intracranial-aneurysm-detection
Rank: #3
Source: https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution

Thanks to RSNA and Kaggle for hosting this competition—it was a great opportunity to work with real-world medical data. I’m also grateful to my teammates @yosukeyama, @nmstopen, @heroalchem, and @dainagao; our collaboration directly contributed to the results we achieved.

### Overview

Our main solution consists of two stages:

1. 3D vessel region detection
1. 3D ROI classification

### Stage 1: Vessel Region Detection

Because the target vessels occupy only a limited portion of the field of view, we first detect whole vessel regions before downstream analysis. We also experimented with vessel segmentation, but it was not sufficiently robust across cases.
Aneurysm locations were relatively consistent in XY coordinates on the axial plane across cases, so we used the middle slice from the sagittal and coronal planes as input images for detection. We computed MIPs of the segmentation masks along the sagittal and coronal directions, then constructed 2D, axis-aligned bounding boxes by taking the minimum and maximum mask coordinates in each view. We used YOLOv8n and YOLOv8m for detection, achieving over 0.95 mAP@0.5 on the validation set. After detection, we reconstructed a 3D ROI by combining results from the sagittal and coronal views, and then cropped a fixed-size 3D bounding box of 90×90×90 mm (or 120×120×120 mm) centered on each detection to generate analysis patches.
Examples of detection results. Green = prediction; red = ground truth.



### Stage 2: 3D ROI Classification
#### Training
Using the 3D vessel ROIs from Stage 1, we trained 3D ResNet-18 backbones, implemented with the `timm-3d` library.

Our models are inspired by [BYU 4th place solution](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/writeups/daddies-4th-place-simple-resnet18-classification). 
We attached a 14-class classification head to **each feature map (not whole volume)** and optimized with weighted BCE loss (multi-label setting). In the default 3D configuration (input 128×128×128), the network produced a 4×4×4 feature map, which did not yield good results. Increasing spatial resolution helped: we changed the stride from 2 to 1 in selected convolution layers to obtain larger feature maps.

We explored the input volume from 128×128×128 up to 224×224×224, and feature-map sizes from 8×8×8 to 48×48×48. Notably, increasing the feature map from 8×8×8 to 25×25×25 and the image size from 128×128×128 to 196×196×196 improved the LB score from 0.77 to 0.81 in a single-fold model.

#### Inference & Aggregation

For inference, we aggregated feature-map predictions into a per-case prediction. Specifically, for each class we sorted the `Aneurysm Present` scores across spatial positions and averaged the top-N scores (Top-N mean) to produce the final class prediction. N depends on model configuration.


#### Model ensemble
We built an ensemble of 11 models with different crop sizes, image resolutions, and variants with reduced stride in selected convolution layers. All models used a 3D ResNet-18 backbone. In this ensemble, the public/private LB scores were 0.86/0.84.

### Missing DICOM tags & Fallbacks
There were many missing DICOM tags in the test data, so we built a model to estimate voxel spacing along the X, Y, and Z axes. To preserve XY spacing, each slice was padded and center-cropped to 512×512 pixels, and 10 central slices were sampled along the Z-axis. Each slice was stacked with its adjacent slices to form 2.5D RGB inputs for capturing Z-axis information. Using these inputs, an EfficientNet V2 S regression model predicted voxel spacing from image appearance, with final values obtained by averaging outputs across the 10 slices. On internal validation, the model achieved MAE = 0.015 mm (X), 0.020 mm (Y), and 0.071 mm (Z), improving our private LB from 0.84 to 0.85.

### What didn't work for us
- Vessel segmentation
- Modality and Plane heads for auxiliary loss
- Complicated model like MIL, LSTMs and decorder heads

[Inference code](https://www.kaggle.com/code/tamotamo/rsna2025-3rd-place-inference)
[Training code](https://github.com/tamotamo17/RSNA2025-3rd-place-solution)
