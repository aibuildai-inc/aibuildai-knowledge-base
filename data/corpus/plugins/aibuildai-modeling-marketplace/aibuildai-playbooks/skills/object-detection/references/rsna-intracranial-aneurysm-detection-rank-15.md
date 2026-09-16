# 15th Place Solution

Competition: rsna-intracranial-aneurysm-detection
Rank: #15
Source: https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution

Thanks to RSNA and the Kaggle team for hosting such an interesting competition.
Through this competition, I was able to learn a great deal about image processing and artificial intelligence.
Here, let me share my solution overview.

## Overview
- Ensemble of two Transformers (weighted average of 14 outputs: 13 locations + aneurysm presence)
- Each Transformer aggregates RoIs extracted by the first-stage Faster R-CNN
- Model 1: 2.5D Faster R-CNN (3-channel input, ResNet-50 backbone) + DeiT-Small Transformer
- Model 2: 2.5D Faster R-CNN (5-channel input, ConvNeXtV2-Tiny backbone) + DeiT-Small Transformer

## Strategy
As a radiologist working at a general hospital, I wanted to develop a model leveraging domain knowledge.
Through exploratory data analysis, I found that the dataset contained axial, coronal, and sagittal images.
I also discovered errors in the direction cosine matrix of some volume data.
Since the orientation of all images could not be determined solely based on DICOM tags,
I decided to develop a model that can make inferences regardless of image orientation.

## Data Pipeline
- All volumes were resampled to the same voxel spacing (2.0 mm slice thickness, 0.5 mm in-plane pixel spacing)
- Cropped and padded to depth × 384 px × 384 px by a simple rule-based method
- Training: all anatomical planes (axial, coronal, sagittal) extracted from each volume
- Inference: only the finest anatomical plane (lesser interpolation) extracted

## Stage 1: 2.5D Faster R-CNN
### Common Training Strategy
- Loss: Default of torchvision Faster R-CNN
- Optimizer: AdamW
- Scheduler: OneCycleLR
- Batch size: 32
- Sampling: All positive slices + 3–10 negative slices (gradually increased during training)
- Augmentation:
    - Affine, ElasticTransform, GridDistortion, MotionBlur, RandomBrightnessContrast, GaussNoise using albumentations
    - Horizontal flip with label swapping (left ↔ right)

### Model 1
- Backbone: torchvision/fasterrcnn_resnet50_fpn_v2
- Input: 3-channel (center slice + two adjacent slices, 384 × 384 px)
- Ground truth: 96 px (48 mm) bounding box around aneurysm center
- Anchors: 5 sizes (64, 80, 96, 112, 128 px) × 3 aspect ratios (0.5, 1.0, 2.0)
- Learning rate: 1e-4 → max 1e-3

### Model 2
- Backbone: timm/convnextv2_tiny.fcmae_ft_in22k_in1k
- Input: 5-channel (center slice + four adjacent slices, 384 × 384 px)
- Ground truth: 64 px (32 mm) bounding box around aneurysm center
- Anchors: 3 sizes (56, 64, 72 px) × 3 aspect ratios (0.8, 1.0, 1.2)
- Learning rate: 2e-5 → max 2e-4

## Stage 2: DeiT-Small Transformer
- Pre-trained weights: timm/deit_small_patch16_224
- Flow: 1024-dim Faster R-CNN RoI features → 384-dim ViT embeddings → 3D sinusoidal positional encoding (based on Faster R-CNN outputs)
- Location head: 13-way multi-label classification (BCEWithLogitsLoss with label smoothing)
- Presence head: Binary classification (BCEWithLogitsLoss with label smoothing)
- Loss weighting: 1.0 × location loss + 0.1 × presence loss
- Two-stage training:
    1. Freeze all ViT backbone layers
    2. Unfreeze all weights
- Optimizer: AdamW
- Scheduler: OneCycleLR (stage 2)
- Layer-wise learning rates (stage 2):
    - Classification heads: 4e-4 (max learning rate)
    - ViT top 4 blocks: 2e-4 (max learning rate)
    - ViT bottom blocks: 1e-4 (max learning rate)
    - RoI projector: 1e-4 (max learning rate)
- Batch size: 32
- Models were trained using 20× augmented cached RoI features

## Scores
| Model | Public | Private |
|---|---|---|
| Model 1 (Faster R-CNN only) \* | 0.78602 | 0.77517 |
| Model 2 (Faster R-CNN only) \* | 0.78177 | 0.75816 |
| Model 1 (Faster R-CNN + Transformer) | 0.80415 | 0.78740 |
| Model 2 (Faster R-CNN + Transformer) | 0.80140 | 0.77955 |
| Final Ensemble (weighted average) | 0.82494 | 0.80592 |
\*: To calculate 14 probabilities, I extracted max probability per location and used max as aneurysm presence.
