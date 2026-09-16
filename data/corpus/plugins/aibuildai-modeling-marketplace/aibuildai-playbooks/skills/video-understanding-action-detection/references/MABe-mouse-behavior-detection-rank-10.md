# 10th Place Solution (ST-GCN + Transformer)

Competition: MABe-mouse-behavior-detection
Rank: #10
Source: https://www.kaggle.com/c/MABe-mouse-behavior-detection/writeups/10th-place-solution-st-gcn-transformer

# 10th Place Solution

## Overview

- ST-GCN + Transformer
  - Model all 16 (agent, target) pairs jointly
  - Directly optimize competition metric (Macro Soft F1 Loss)
  - Ensemble models with different keypoints (4-7) and sequence lengths

## Preprocessing

### FPS Normalization
- Resample all videos to **30 FPS** (linear interpolation)
- Original FPS passed to model as additional feature

### Coordinate Normalization (per lab)
- Normalize (x, y) coordinates per lab: `x_norm = (x - mean) / std`

### Keypoint Mapping
- Labs have different keypoint names → map to unified names
  - head → nose
  - spine_1 → neck
  - hip_left/right → lateral_left/right

### Feature Engineering (24-dim per keypoint)

From normalized (x, y) coordinates, compute 24 features:

```
Per-keypoint features (8 dim):
  - x, y         (2): normalized coordinates
  - vx, vy       (2): velocity (Δt=2 frames)
  - ax, ay       (2): acceleration (Δt=2 frames)

Inter-mouse features (16 dim, relative to other 3 mice):
  - rel_x, rel_y (6): relative position to mouse i (i=1,2,3)
  - rel_vx, rel_vy (6): relative velocity to mouse i
  - dist         (3): Euclidean distance to mouse i
  - approach_vel (3): approach velocity (d(dist)/dt)

Total: 8 + 16 = 24 features
```

## Cross Validation

- **4-fold StratifiedGroupKFold**
  - Group: video_id
  - Stratify: lab_id
  - CalMS21, CRIM13 labs always in train
- **OOF evaluation**: Train on 3 folds, validate on 1 fold
- **Final submission**: Train on all data (no validation split)

## Model Architecture

### ST-GCN + Transformer

```
Input:           [B, 4, K, 24, F]   # batch, mice, keypoints, features, frames
    ↓
ST-GCN (×4) + Temporal Pooling:
                 [B, 4, K, 128, F']  # spatial graph conv + temporal conv + pool (F'=F/p)
    ↓
Keypoint Attention Pooling:
                 [B, 4, 128, F']    # learnable attention over keypoints
    ↓
Pairwise Feature Extraction:
  concat(agent, target, agent-target, target-agent) for 4×4 pairs
                 [B, 16, 512, F']
    ↓
Concat Embeddings:
  + Lab emb (16) + FPS emb (16) + Action emb (32)
                 [B, 16, 576, F']
    ↓
Feature Compression:
                 [B, 16, 192, F']
    ↓
Transformer with RoPE (per pair):
                 [B, 16, F', 192]
    ↓
Temporal Upsample:
                 [B, 16, F, 192]    # restore to original frame length
    ↓
Cross-Pair Attention:
  16 pairs attend to each other
                 [B, 4, 4, F, 192]
    ↓
Classifier:      [B, 4, 4, 38, F]   # agent, target, classes, frames
```

Note: Non-pairwise model (without Pairwise Feature Extraction / Cross-Pair Attention) also used in ensemble.

## Loss Function

logits: [B, 4, 4, 38, F] → flatten → [N, 152]  # N = B × 4 agents × F

Triplet Mask (from behaviors_labeled):
  - Set logits[invalid] = -inf before softmax
  - Only (agent, target, action) in behaviors_labeled are valid

Loss = α × CE + (1-α) × MacroSoftF1

  - CE: Cross-Entropy over all (target × action: 152) per agent
  - MacroSoftF1: Differentiable approximation of competition metric (soft TP/FP/FN, excludes background)
  - Schedule: α transitions 0.2 → 0 over 32 epochs (pure F1 at end)

## Data Augmentation

**Training**
- Affine transform: rotation, scale, translation, shear, horizontal/vertical flip
- Mouse shuffle: permute mouse IDs
- CutMix: mix temporal segments (same lab & num_mice)
- Keypoint dropout

**TTA**
- Horizontal flip averaging

## Ensemble

- Different keypoints and sequence lengths (160-224)
  - 4kp: ear×2, nose, tail_base
  - 5kp: + neck
  - 7kp: + lateral×2
- Pairwise model (LB: ~0.525, CV: ~0.510) + non-pairwise models (LB: ~0.500, CV: ~0.500)
- Probability averaging across models

## What Didn't Work

- Pseudo-labeling on test data
- Semi-supervised learning with MABe22 unlabeled data
- Merging rare classes per lab in training (e.g., AdaptableSnail: avoid → escape)
