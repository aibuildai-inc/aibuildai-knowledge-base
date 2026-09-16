# Key Improvements in 17th Solution: A Technical Summary

Competition: MABe-mouse-behavior-detection
Rank: #17
Source: https://www.kaggle.com/c/MABe-mouse-behavior-detection/writeups/key-improvements-in-our-solution-a-technical-summ

## Overview

This solution uses the approach in https://www.kaggle.com/code/ravaghi/social-action-recognition-in-mice-xgboost as its baseline and achieves competitive performance through advanced feature engineering and robust post-processing. Below, I summarize the key technical improvements that differentiate this approach from the baseline implementation.

Special thanks to @**ravaghi** for providing the baseline!



## 1. Data Curation Strategy

### Strict Data Exclusion Rules

This solution implements **more aggressive data filtering** compared to baseline approaches:

**Excluded Data:**

```python

UNUSE_LAB = [
    "CalMS21_supplemental",
    "CalMS21_task1",
    "CalMS21_task2",
    "MABe22_keypoints",
    "MABe22_movies"
]

```
### Additionally exclude:

#### AdaptableSnail lab with fps == 25`

**Baseline approach only excludes:**

```python

train_df = train_df[~((train_df["lab_id"] == "AdaptableSnail") & (train_df["frames_per_second"] == 25))]

```

### Why This Matters

**CalMS21 and MABe22 supplemental/task data:**

- Different annotation protocols from main competition
- May introduce conflicting behavioral definitions
- Potential label noise that hurts model generalization

**AdaptableSnail fps==25 filtering:**

- Identified as problematic data through validation
- Inconsistent with standard 30 fps videos
- May have tracking or annotation quality issues

**Impact:** Cleaner training data leads to:

- Better model convergence (fewer conflicting signals)
- Improved cross-validation stability
- More reliable feature engineering (consistent fps assumptions)
- Higher competition score (2-4 point F1 improvement estimated)

**Trade-off:** Less training data, but higher quality. This is a deliberate choice prioritizing data quality over quantity.

---

## 2. Advanced Feature Engineering Architecture

### Modular Generator System

Implemented **15 specialized feature generators for single-mouse behaviors** and **14 generators for pair interactions**, each capturing specific behavioral aspects:

**Single Mouse Features:**

- Basic motion: Distance, Speed, Acceleration between all body parts
- Cross-temporal features: Speed between different body parts at different time points
- Body mechanics: Elongation ratio, body angle, curvature
- Movement patterns: Multi-scale speed statistics (20/40/60/80 frame windows)
- Behavioral states: Speed discretization into 4 states with transition tracking
- Long-range dynamics: Deviations from moving averages (30/60/120 frames)

**Pair Interaction Features:**

- Spatial relationships: All pairwise distances between mice
- Relative orientation: Directional alignment between mice
- Approach dynamics: Whether mice are moving toward/away from each other
- Leading indicators: Which mouse is initiating approach
- Chase features: Combined approach and following behavior
- Speed correlation: Synchronized movement patterns across time windows

### Key Innovation: Cross-Speed Features

```python
# Example: Captures coordination between different body parts over time
sp_ear_left_tail_base = distance(ear_left[t], tail_base[t-10])

```

This reveals temporal coordination that simple instantaneous features miss.

---

## 3. Sophisticated Post-Processing Pipeline

### Three-Parameter Optimization

Unlike baseline approaches that only tune prediction threshold, this solution optimizes **three parameters simultaneously using Optuna**:

```python
threshold: 0.0-1.0  # Probability cutoff (step 0.01)
W: 0-60 frames      # Gap-filling window
M: 0-30 frames      # Minimum segment length

```

**Gap-Filling (W):**

- Fills small gaps between predictions of the same behavior
- Addresses fragmentation from noisy frame-by-frame predictions
- Example: [1,1,1,0,0,1,1,1] with W=2 becomes [1,1,1,1,1,1,1,1]

**Noise Removal (M):**

- Removes isolated short predictions
- Filters out spurious detections
- Example: [0,0,1,1,0,0,0] with M=3 becomes [0,0,0,0,0,0,0]

**Impact:** This typically improves F1 score by 2-5 points by producing temporally coherent predictions that match annotator behavior patterns.

---

## 4. Lab-Specific Processing Strategy

### Why Lab-Level Organization Matters

Different research laboratories have:

- Different camera setups and tracking quality
- Different behavioral annotation protocols
- Different mouse strains and experimental conditions
- Different body part tracking configurations

**Implementation:**

```python
for lab_id in lab_list:
    train_subset = train[train.lab_id == lab_id]
    # Process all videos from this lab together
    # Train lab-specific models

```

**Benefits:**

- Models learn lab-specific patterns and biases
- Better handles heterogeneous data quality
- More robust to tracking system differences

---

## 5. Robust Missing Data Handling

### Automatic Body Center Computation

```python
def add_body_center_if_needed(ppvid):
    if 'body_center' exists: return as-is
    elif 'nose' and 'tail_base' exist:
        body_center = (nose + tail_base) / 2
    elif 'head' and 'tail_base' exist:
        body_center = (head + tail_base) / 2
    else: fill with NaN

```

### Feature Generator Validation

Each feature generator validates required body parts before computation:

```python
class BaseGenerator:
    def validate(self) -> bool:
        # Check if required body parts exist
        # Skip gracefully if not available

```

**Impact:** Handles 20+ different body part configurations across labs without crashes or manual intervention.

---

## 6. Model Persistence and Incremental Learning

### Smart Checkpoint System

```python
def load_saved_trainer(model_path, section, action,
                       do_modeling=False, do_tuning=False):
    # Load pre-trained models if available
    # Skip training/tuning for completed actions

```

**Benefits:**

- Experiment with post-processing without retraining
- Resume interrupted training runs
- Selectively retrain specific behaviors
- Essential for Kaggle's time-limited environment

---

## 7. Memory Optimization

### Automatic Dtype Optimization

```python
def reduce_mem_usage(df):
    # int64 → int32 → int16 → int8
    # float64 → float32 → float16
    # object → category
    # Typical reduction: 50-75%

```

**Impact:** Enables processing 2-3x more data within Kaggle's memory limits.

---

## 8. Enhancements to the Code Architecture

### Object-Oriented Design

- **DataLoader:** Centralized file I/O with caching
- **Preprocessor:** Data normalization and transformation
- **BehaviorData:** Encapsulates features, labels, and metadata
- **FeatureStore:** Manages generator pipeline execution

### Benefits of This Architecture

- **Maintainability:** Each component has single responsibility
- **Testability:** Can unit test generators independently
- **Extensibility:** Add new features without modifying existing code
- **Reusability:** Components work in other MABe projects

---

## Performance Comparison

| Aspect | Baseline | This Solution | Improvement |
| --- | --- | --- | --- |
| Feature Count | ~50-80 | ~150-250 | 2-3x more |
| Post-processing | Threshold only | Threshold + W + M | +2-5 F1 points |
| Missing Data | Try-catch errors | Proactive validation | No crashes |
| Code Lines | ~900 | ~2000 | Better organized |
| Memory Usage | Default dtypes | Optimized dtypes | 50-75% reduction |

---

## Code Availability

https://www.kaggle.com/code/yono18/mabe-boosting-model-prediction-500-iter

The complete implementation, including all feature generators and the full processing pipeline, is available in my competition notebook.

Feel free to adapt these techniques for your own solutions.

If you have any feedback or suggestions, please leave a comment.

Thank you for reading until the end!
