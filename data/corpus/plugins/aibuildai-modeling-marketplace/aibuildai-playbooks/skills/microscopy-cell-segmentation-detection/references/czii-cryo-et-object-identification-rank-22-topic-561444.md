# 22nd Place Solution

Competition: czii-cryo-et-object-identification
Rank: #22
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561444

**22nd Place Solution**

## Pipeline

1. 2D-3D Semantic Segmentation
2. Estimation of particle centroid coordinates using cc3d  
3. Consolidation of particle coordinates from multiple models with WBF-based post-processing  
4. 2D false-positive suppression using Minislab (apo-ferritin, ribosome)  

## Model

I used a slightly customized version of @hengck23 ’s excellent [2D-3D semantic segmentation model](https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder).

- **Input**: 48×352×352 or 48×320×320  
- **Output**: 6×d×h×w  
- **Loss**: BCE + 2×TverskyLoss  

The best single-model score was **0.737 / 0.730**.

## Data

I performed pre-training on DS-10441 and then fine-tuned on DS-10440. Although the pre-training did not improve the leaderboard score, it helped reduce the training time for fine-tuning.

The radius setting for the segmentation labels had a significant impact on accuracy. I used either 0.5× or 0.6× the particle radius for each particle.

| Radius                       | CV      |
|-----------------------------|---------|
| Particle radius × 0.9       | 0.6484  |
| Particle radius × 0.8       | 0.7340  |
| Particle radius × 0.7       | 0.7645  |
| Particle radius × 0.6       | 0.7807  |
| Particle radius × 0.5       | 0.7727  |
| Particle radius × 0.4       | 0.7675  |

## Post-Processing

- Used WBF-based NMS to merge the particle centroid coordinates from each model (**+0.003 to +0.005**).  
- Created a “minislab” around each predicted particle coordinate, applied a 2D classification to determine particle vs. noise, and removed false positives (**+0.003 to +0.005**).

**Minislab** example:

```python
image = volume[z_start:z_end, y_start:y_end, x_start:x_end]
image = image.mean(axis=0)
```

## Computing Resources

* RTX 3090 × 1.2 (occasionally borrowed from Vast AI).

## Inference Code

https://www.kaggle.com/code/akinosora/czii2024-22nd-place-inference-code/notebook
