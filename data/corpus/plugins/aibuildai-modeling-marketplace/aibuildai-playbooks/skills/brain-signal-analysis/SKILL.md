---
description: >-
  ML playbook for brain signal analysis competitions. Use when tackling a Kaggle-style competition involving brain signal analysis. Teaches how to reason about choose signal representation based on data modality and task horizon, choose frequency bands for the task, use multi-resolution temporal windows to capture both context and detail, cross-validate by patient/subject and recording sequence. 46 top-solution writeups across 8 competitions: Parkinson's FOG (12 writeups), Seizure Prediction (1 writeup), plus additional competitions covering EEG classification, seizure detection, and brain-computer interfaces.
---

# Brain Signal Analysis Playbook

Brain signal analysis competitions involve time-series classification, detection, or prediction tasks using physiological signals—primarily EEG (electroencephalography) from multiple scalp electrodes, and occasionally accelerometer data from wearable sensors. The core challenge is extracting meaningful patterns from noisy, high-dimensional temporal data where the effective sample size is often very small (e.g., only 4 seizures per patient) and signal characteristics vary drastically across subjects and recording conditions.

**Source material:** 46 top-solution writeups across 8 competitions: Parkinson's FOG (12 writeups), Seizure Prediction (1 writeup), plus additional competitions covering EEG classification, seizure detection, and brain-computer interfaces.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Signal Representation Based on Data Modality and Task Horizon | Parkinson's FOG solutions modeled accelerometer data as 1D sequences or patches, with the 6th-place team adding spectrogram and wavelet models; seizure prediction solutions turned EEG into hand-crafted band-power and correlation features. The choice depends on data modality and on whether the task requires fine temporal resolution or frequency patterns. | principles/01.md |
| 2 | Choose Frequency Bands for the Task | Seizure prediction solutions computed power in standard EEG bands from delta up to high gamma (up to 180Hz); several Melbourne models applied no filtering, others a wide Butterworth or 60Hz notch filter. The useful frequency range is task-specific. | principles/02.md |
| 3 | Use Multi-Resolution Temporal Windows to Capture Both Context and Detail | Parkinson's FOG solutions train on short sequences (1000-5000 steps) but infer on long (15000-30000); several Melbourne seizure prediction solutions scored short epochs (20-30s) inside each 10-minute file, and one joined features from several epoch lengths. | principles/03.md |
| 4 | Cross-Validate by Patient/Subject and Recording Sequence | Several Parkinson's FOG solutions used GroupKFold or StratifiedGroupKFold with Subject as the group; Melbourne seizure prediction solutions kept each 1-hour sequence in one fold, and one found that ignoring sequences made local AUC about 0.1 higher than the leaderboard. | principles/04.md |
| 5 | Augment Signals to Exploit Domain Symmetries and Regularize Small Datasets | Several Parkinson's FOG solutions augmented accelerometer data (time stretch, Gaussian noise, pitch shift, wave scaling, random low-pass filtering, positional-encoding roll); one applied heavy stretching, cropping, ablation and accumulated noise to every sequence. Diversity in augmentation across models improves ensemble. | principles/05.md |
| 6 | Ensemble Diverse Representations and Frequency Ranges for Complementary Strengths | Most source solutions ensemble multiple models (the Melbourne seizure prediction winner rank-averaged 11 models). Parkinson's FOG: the 6th-place ensemble combined spectrogram, wavelet, and 1D conv models; spectrograms good at StartHesitation/Turn, wavelets good at Walking. | principles/06.md |
| 7 | Reduce Overfitting with Extreme Regularization When Sample Size Is Tiny | The seizure prediction winner (4 seizures per dog) used a heavily regularized LS ensemble; the Melbourne seizure prediction winner used subject-specific XGB, SVM, KNN, and L2 logistic regression on hand-crafted features; larger datasets such as Parkinson's FOG used deep learning with dropout/augmentation. | principles/07.md |
| 8 | Handle Variable Sequence Lengths with Patches or Adaptive Chunking | Parkinson's FOG competition (variable 5k-200k steps): 4/5 solutions chunked sequences into fixed-length windows with overlap. Transformer solutions (2/5) used patch-based encoding to reduce sequence length. | principles/08.md |
| 9 | Use Positional Bias or Relative Position Encoding for Long-Range Temporal Dependencies | 2/5 transformer solutions bias attention toward local context; FOG 1st place used learnable positional encoding with random roll augmentation; FOG 6th place biased attention with distance masks. | principles/09.md |
| 10 | Treat Multiple Datasets with Different Characteristics as Separate Modeling Problems | Parkinson's FOG: 5/5 solutions trained separate models for tdcsfog (128Hz, lab) vs defog (100Hz, home); only one used defog's 'notype' unlabeled data via pseudo-labeling. | principles/10.md |
| 11 | Probe Test Set Distribution and Align Validation Strategy Accordingly | Several solutions compared CV with the public LB to find mismatches: Melbourne seizure prediction teams found file-level CV did not track the LB, and a Parkinson's FOG team traced public-to-private shake to one subject with long recordings in the public data. | principles/11.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

Brain signal analysis rewards diversity: combine 1D sequences, spectral features, and time-frequency images, multiple frequency ranges, and dataset-specific models to cover the full signal space. Validate rigorously by patient and recording sequence, and invest in augmentation and regularization to combat the curse of small sample sizes.
