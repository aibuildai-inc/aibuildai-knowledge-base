# 2nd place solution

Competition: MABe-mouse-behavior-detection
Rank: #2
Source: https://www.kaggle.com/c/MABe-mouse-behavior-detection/writeups/2nd-place-solution

We would like to sincerely thank the organizers and the Kaggle team for running such an engaging and enjoyable competition.

# Overview

We train **separate models for solo and pair actions**.

- Input: time-series of shape **(T = 512, num_features)**
- Loss: **Binary Cross Entropy**
- Final predictions are obtained using **lab ×actions–specific thresholds**

---

## Data Preprocessing
- All tracking data are converted to **30 FPS**
    - Features: linear interpolation
    - Labels: nearest-neighbor interpolation
- Distance / velocity / acceleration are standardized using precomputed statistics
- Missing values are filled by linear interpolation
    - Missing body-part features are replaced with the mean of existing body-part features in the same frame.

### Handling AdaptableSnail’s 25 FPS Tracking Data
Reference:
https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/discussion/612531#3305943

Issues:
- Labels are provided at **30 FPS**
- **Mouse ID swapping** occurs

Fixes:
- Label frame rates were adjusted accordingly
- Mouse IDs were manually corrected by visually inspecting tracking visualization videos

## Input Features
### Meta Features (shared across frames)
| Feature | Channels |
| --- | --- |
| Sex | solo=1, pair=2 |
| Frame-rate scale | 1 |
| Lab ID(passed through an embedding layer) | 16 |
| Candidate action label | num_classes（solo=11, pair=26） |

### Per-frame Features
**Solo**
| Feature | Channels |
| --- | --- |
| Body-part distances | num_bodyparts² |
| Velocity | num_bodyparts |
| Acceleration | num_bodyparts |

<br>
**Pair**
| Feature | Channels |
| --- | --- |
| Cross-mouse body-part distances | num_bodyparts² |
| Velocity | 2 × num_bodyparts |
| Acceleration | 2 × num_bodyparts |

---
## Model Architectures
We used two architecture families.

**CNN + RNN / Transformer**
```
Input
 → Multi-scale Conv (k =3,5,7,9)
 → Conv Fusion
 → Bi-GRU or Transformer
 → Linear
 → Output
```

**SqueezeFormer**
```
Input
 → SqueezeFormer
 → Output
```
We trained **5 variants** with different depths and hyperparameters.

---

## Training

- **5-fold stratified cross-validation**
    - Folds are created using only data from lab IDs that appear in the test set
    - Lab IDs are stratified so that their distributions are consistent across folds
    - Labs from CalMS21* and CRIM13 are always included in the training set
- One-vs-rest training per actions with BCE
- EMA with decay = 0.999
- Cosine LR schedule with warmup
- At each epoch, optimize the thresholds for each lab–action pair on the validation set and apply early stopping based on the resulting score.

## Data Augmentation

- Gaussian noise
- Time stretching
- Frame shift (shift features only; labels unchanged)
- Left/right body-part flipping
- Mixup

---

## Inference

- Sliding-window inference with **stride = 128**
- We discard **32 frames at both window boundaries** and average the remaining predictions.
- Flip test-time augmentation (TTA) 

---

## Post-processing

### Thresholding

1. Optimal thresholds are computed independently for each fold and for each lab × action pair, and the final threshold is obtained by averaging the fold-specific thresholds.
2. For each frame, among actions above threshold, the **highest-scoring action** is selected

### AdaptableSnail 25 FPS Adjustment

- `start_frame` and `stop_frame` are multiplied by **30 / 25**
- When mouse ID swaps are present in the tracking data, this adjustment is not meaningful
- The private dataset appears to include tracking sequences without mouse ID swaps
    - For these sequences, predictions are aligned to the 30 FPS labels
    - This resulted in a **+0.009 improvement in the private score**

---

## Results

| Model | CV (Solo) | CV (Pair) | Public | Private |
| --- | --- | --- | --- | --- |
| Ensemble | 0.069 | 0.499 | 0.542 | 0.521 |
| + AdaptableSnail post-processing | — | — | 0.542 | **0.530** |
