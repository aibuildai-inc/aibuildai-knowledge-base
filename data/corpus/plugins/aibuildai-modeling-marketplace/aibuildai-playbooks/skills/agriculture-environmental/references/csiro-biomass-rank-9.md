# 9th Place Solution

Competition: csiro-biomass
Rank: #9
Source: https://www.kaggle.com/c/csiro-biomass/writeups/9th-place-solution

Our solution is a weighted ensemble of **three model variants**, all built on the DINOv3 ViT-Huge backbone with LocalMamba fusion blocks. The key thing that helped was addressing the challenging `Dry_Dead_g` target through ratio-based prediction and multi-head architecture variations.
**Final Ensemble:** 3 model architectures × 4 folds.
---

## Model Architecture

### Shared Components
All models share the following core architecture:
```
Input: Image (2000×1000) → Split into Left (1000×1000) + Right (1000×1000)
    ↓
DINOv3 ViT-Huge Backbone (vit_huge_plus_patch16_dinov3.lvd1689m)
    - Pretrained weights from timm
    - Gradient checkpointing enabled for memory efficiency
    - global_pool='' to retain patch tokens
    ↓
Feature Concatenation: [Left Tokens, Right Tokens] → (B, 2N, D)
    ↓
LocalMamba Fusion (2 blocks)
    - LayerNorm → Gated Linear → DepthwiseConv1d → Projection
    - Residual connections
    ↓
Adaptive Average Pooling → (B, D)
    ↓
Multi-Head Regression with Softplus activation (non-negative outputs)
```
---
### Model 1: Improved Dead Prediction (Ratio-Based)
**Key Innovation:** Predict `Dry_Dead_g` as a ratio of `Dry_Total_g`
| Head | Output | Activation |
|------|--------|------------|
| `head_green` | Dry_Green_g | Softplus |
| `head_clover` | Dry_Clover_g | Softplus |
| `head_total` | Dry_Total_g | Softplus |
| `head_dead_ratio` | Dead/Total ratio | **Sigmoid** [0,1] |
**Derivation:**
- `Dry_Dead_g = head_dead_ratio × Dry_Total_g`
- `GDM_g = Dry_Green_g + Dry_Clover_g`
**Special Handling:**
- State-specific Dead adjustment via learnable bias (WA state has Dead≈0)
- Multi-task loss with auxiliary ratio loss
- Scale-invariant log-space loss for Dead
**Training Config:**
- Image Size: 1024×1024
- Loss Weights: [0.12, **0.25**, 0.12, 0.16, 0.35] (increased Dead weight)

---

### Model 2: Standard Multi-Head
**Architecture:** Direct prediction with derived targets
| Head | Predicted | Derived |
|------|-----------|---------|
| `head_green` | Dry_Green_g | - |
| `head_dead` | Dry_Dead_g | - |
| `head_clover` | Dry_Clover_g | - |
| - | GDM_g | Green + Clover |
| - | Dry_Total_g | GDM + Dead |
**Training Config:**
- Image Size: 1024×1024
- Loss Weights: [0.15, 0.15, 0.15, 0.2, 0.35]

---

### Model 3: Total-First Prediction
**Key Difference:** Predict Total and GDM directly, derive Dead
| Head | Predicted | Derived |
|------|-----------|---------|
| `head_green` | Dry_Green_g | - |
| `head_clover` | Dry_Clover_g | - |
| `head_gdm` | GDM_g | - |
| `head_total` | Dry_Total_g | - |
| - | Dry_Dead_g | **Total - GDM** |
**Training Config:**
- Image Size: 896×896
- Loss Weights: [0.15, 0.15, 0.15, 0.2, 0.35]

---

## Training Strategy
### Data Preprocessing
**Image Split:** Panoramic images (2000×1000) split into Left half (0:mid) and Right half (mid:end), processed separately through the backbone.
### Cross-Validation Strategy
The CV Strategy we used was suggested by someone in the discussion. I am not able to find that thread, but thanks to him.
we used **StratifiedGroupKFold** (4 folds) with a carefully designed stratification and grouping scheme:
**Stratification Label:** Based on presence/absence of challenging targets
```python
stratify_col = f"{has_clover}_{has_dead}"
# Ensures each fold has balanced representation of:
#   - Samples WITH Clover (Dry_Clover_g > 0)
#   - Samples WITH Dead (Dry_Dead_g > 0)
```
**Group Key:** Prevents data leakage by grouping samples from same collection session
```python
group_key = f"{day}/{month}_{state}"
# Examples: "15/3_NSW", "22/6_WA", "8/11_Vic"
# Ensures samples collected on the same date in the same state stay together
```
**Why This Matters:**
1. **Stratification by Clover/Dead:** `Dry_Clover_g` is often zero (sparse), and `Dry_Dead_g` has high variance. Stratifying ensures each fold sees similar distributions of these challenging targets.
2. **Grouping by Date+State:** Samples collected on the same day in the same location likely have similar conditions (weather, pasture state). Keeping them in the same fold prevents the model from "memorizing" collection-day patterns.
```python
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
for fold, (_, val_idx) in enumerate(
    sgkf.split(df, df['stratify_col'], groups=df['group_key'])
):
    df.loc[val_idx, 'fold'] = fold
```

### Augmentations (Training)
```python
A.Resize(IMG_SIZE, IMG_SIZE)
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.5)
A.RandomRotate90(p=0.5)
A.Rotate(limit=(-15, 15), p=0.3, border_mode=REFLECT)
A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5)
A.GaussNoise(var_limit=(5.0, 20.0), p=0.2)
A.Normalize(ImageNet stats)
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 2 |
| Gradient Accumulation | 8 (effective batch = 16) |
| Epochs | 100 (early stopping) |
| Freeze Epochs | 10 |
| Warmup Epochs | 3 |
| LR Backbone | 1e-5 |
| LR Head | 1e-4 |
| Weight Decay | 1e-2 |
| EMA Decay | 0.998 |
| Patience | 15 |
| Gradient Clipping | 1.0 |
| Mixed Precision | FP16 |
### Loss Function
**Weighted Huber Loss (β=5.0):**
```python
loss = Σ(weight_i × SmoothL1Loss(pred_i, target_i, beta=5.0))
```
**Model 1 Additional Losses:**
- Auxiliary MSE loss on Dead/Total ratio (weight=0.15)
- Scale-invariant log-space loss for Dead (weight=0.05)

---

## Inference & Ensemble
### Validation TTA
4-way TTA during validation:
1. Original
2. Horizontal Flip
3. Vertical Flip
4. Horizontal + Vertical Flip
### Ensemble Strategy
**Weighted Average with Fold-Specific Weights:**
```python
weight = {
    fold_0: 1.25,
    fold_1: 0.75,
    fold_2: 1.25,
    fold_3: 0.75
}
ensemble = Σ(weight × fold_predictions) / Σ(weights)
```
### Post-Processing
```python
if target_name == "Dry_Clover_g":
    prediction *= 0.8
elif target_name == "Dry_Dead_g":
    if prediction > 20:
        prediction *= 1.1
    elif prediction < 10:
        prediction *= 0.9
```

---

## Key Insights
1. **Dead Prediction is Hard:** `Dry_Dead_g` has the highest variance and lowest correlation with other features. Ratio-based prediction (Dead/Total) significantly improved stability.
2. **State Matters for Dead:** WA samples consistently have Dead=0. Learnable state-specific bias helped the model learn this pattern.
3. **Resolution Trade-off:** 1024×1024 performed slightly better but 896×896 was faster; combining both helped. In fact, for the models trained on 896, inferring them on 1024 performed better than inferring them on 896.

---

## Results
| Model / Ensemble | Public Score | Private Score |
|------------------|--------------|---------------|
| Model 1: Ratio-Based Dead | 0.76760 | 0.65329 |
| Model 2: Standard Multi-Head | **0.77157** | 0.64714 |
| Model 3: 4-Heads (Total-First) | 0.76685 | 0.64803 |
| Ensemble (with fold weights + post-processing) | 0.77142 | **0.65179** |
| Ensemble (no fold weights, no post-processing) | 0.76772 | 0.64855 |

---

## What Didn't Work
- Metadata conditioning (State/Season embeddings) did not work
- We did not use TTA for the submission of ensemble as it was exceeding 9 hrs.
- Higher image resolutions (>1024) 
- More Mamba blocks (>2) 

---
