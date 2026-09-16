# 12th place solution

Competition: vesuvius-challenge-surface-detection
Rank: #12
Source: https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/12th-place-solution

Thank you for the organizers and Kaggle team for this great competition!

## 1. Summary

nnUNet-based 3D segmentation.
4-model ensemble of 3d_lowres and 3d_fullres + opening/closing post-processing
- Public LB: 0.578
- Private LB: 0.613

---

## 2. Solution Overview

```
[Pipeline]

Input (3D CT volume)
    ↓
nnUNet preprocessing (normalization, resampling)
    ↓
4-model inference (2x T4 GPU parallel)
  ├─ 3d_lowres fold_0 (2000ep) × 0.2
  ├─ 3d_lowres fold_1 (4000ep) × 0.2
  ├─ 3d_fullres fold_0 (4000ep) × 0.3
  └─ 3d_fullres fold_1 (2000ep) × 0.3
    ↓
Weighted average of probability maps
    ↓
Post-processing
  ├─ Hysteresis thresholding (t_low=0.3, t_high=0.85)
  ├─ Opening (noise removal)
  └─ Closing (hole filling)
    ↓
Output (3D segmentation mask)
```

**Key Points**
- Used nnUNet with mostly default settings
- Combined lowres + fullres for different scales
- Post-processing significantly improved TopoScore (topology quality)

---

## 3. Model & Training

**Configuration Comparison**

|              | 3d_lowres | 3d_fullres |
|--------------|-----------|------------|
| Patch size   | 128³      | 128³       |
| Spacing      | 1.56 mm   | 1.0 mm     |
| Effective FOV| ~200 mm³  | ~128 mm³   |
| Image size   | 205³      | 320³       |

*Same patch size but different physical coverage due to spacing differences

**Fold Strategy**
- Used only 2 folds from nnUNet's default 5-fold CV (fold_0, fold_1)
- Reason: Time constraints, and 2 folds provided sufficient diversity
- Training data: 628 cases, validation data: 157-158 cases/fold

**Epochs**
- Base: 2000 epochs
- Some models extended to 4000 epochs
- Wanted to train all models to 4000 epochs but ran out of competition deadline
- Final 4 models used:

| Config     | Fold   | Epochs | CV (opening_closing) |
|------------|--------|--------|----------------------|
| 3d_lowres  | fold_0 | 2000   | 0.606                |
| 3d_lowres  | fold_1 | 4000   | 0.603                |
| 3d_fullres | fold_0 | 4000   | 0.606                |
| 3d_fullres | fold_1 | 2000   | 0.603                |

**Effect of Epochs**
- 1000ep → 2000ep:
  - 3d_lowres fold_0: 0.601 → 0.605 (+0.004)
- 2000ep → 4000ep:
  - 3d_fullres fold_0: 0.603 → 0.606 (+0.003)
  - 3d_lowres fold_1: 0.594 → 0.603 (+0.009)

---

## 4. Inference & Post-processing

**Sliding Window Settings**
- step_size = 0.3 (70% overlap)

**TTA (Test Time Augmentation)**
- 8-direction mirroring
- Used TTA for inference, switched to non-TTA when approaching 9-hour time limit

**Post-processing Pipeline**
```
Probability map → Hysteresis → Opening → Closing → Dust Removal → Final mask
```

**Step 1: 3D Hysteresis Thresholding**
- t_high = 0.85: High confidence regions (seeds)
- t_low = 0.30: Wide coverage
- Structuring element: generate_binary_structure(3, 3) - 26-connectivity
- Process: binary_propagation from strong to weak regions

**Step 2: Opening (Noise Removal)**
- Structuring element: generate_binary_structure(3, 1) - 6-connectivity
- Effect: Removes small protrusions and noise

**Step 3: Anisotropic Closing (Hole Filling)**
- Structuring element: z_radius=2, xy_radius=1 (anisotropic)
- Larger z-direction to strengthen inter-slice connectivity

**Step 4: Dust Removal**
- min_size = 100 voxels
- Effect: Removes small isolated regions

**Post-processing Effect Comparison** (2000ep fold_0)

| Method | Competition Score | TopoScore | Improvement |
|--------|-------------------|-----------|-------------|
| None (argmax) | 0.571 | 0.246 | - |
| Hysteresis only | 0.592 | 0.322 | +0.021 |
| + Opening/Closing | 0.606 | 0.342 | +0.035 |

---

## 5. Scores

**Score Progression**

| Stage | Changes | Public | Private |
|-------|---------|--------|---------|
| Baseline | 1000ep, 1 model, argmax | 0.530 | 0.552 |
| +Post-processing | hysteresis | 0.549 | 0.570 |
| +2-fold | fold_0+1 ensemble | 0.565 | 0.590 |
| +opening_closing | Improved post-processing | 0.565 | 0.593 |
| +2000ep | Increased epochs | 0.568 | 0.593 |
| +fullres | fullres 2 models only | 0.575 | 0.598 |
| +4 models | lowres+fullres ensemble | 0.582 | 0.606 |
| +Weight tuning | fullres 60%, lowres 40% | 0.583 | 0.605 |
| +TTA | 8-direction mirroring | 0.584 | 0.607 |
| +4000ep | Final submission | 0.578 | 0.613 |


---

## 6. What Worked / What Didn't Work

### What Worked

- nnUNet default settings
- 4-model ensemble (lowres + fullres fold0, 1)
- Post-processing (Hysteresis + Opening/Closing)
- Epoch increase (1000→2000→4000)
- TTA (Test Time Augmentation)

### What Didn't Work

- Increased inference parallelism (npp=2, nps=2): Counterproductive due to T4 memory constraints (+50min slower)
- step_size changes: 0.3 was best, 0.2 and 0.5 worsened Public score
