---
description: >-
  ML playbook for natural disaster geophysics prediction competitions. Use when tackling a Kaggle-style competition involving natural disaster geophysics prediction. Teaches how to reason about align train and test distributions before building models, design cross-validation to respect physical event boundaries, engineer features from multiple signal representations, minimize feature count through aggressive selection. 39 top-solution writeups across 5 competitions: LANL Earthquake Prediction, How Much Did It Rain I/II, ClimSim Atmospheric Physics, and Geophysical Waveform Inversion
---

# Natural Disaster and Geophysics Prediction Playbook

Natural disaster and geophysics prediction competitions involve forecasting physical phenomena (earthquakes, rainfall) or reconstructing subsurface structures (velocity models) from sensor data. The core challenge is extracting predictive signals from noisy geophysical measurements while handling temporal dependencies, distribution shifts between train and test, and limited data from rare catastrophic events.

**Source material:** 39 top-solution writeups across 5 competitions: LANL Earthquake Prediction, How Much Did It Rain I/II, ClimSim Atmospheric Physics, and Geophysical Waveform Inversion

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Align Train and Test Distributions Before Building Models | 10/15 earthquake solutions, 3/5 rainfall solutions explicitly mentioned this | principles/01.md |
| 2 | Design Cross-Validation to Respect Physical Event Boundaries | 15/15 earthquake solutions used event-aware CV | principles/02.md |
| 3 | Engineer Features from Multiple Signal Representations | 12/15 earthquake solutions, 2/2 rainfall solutions | principles/03.md |
| 4 | Minimize Feature Count Through Aggressive Selection | 5/15 earthquake solutions emphasized this (1st: 4 features, 7th: 7+4, 15th: 7 features) | principles/04.md |
| 5 | Use Multi-Stage Pseudo-Labeling on Unlabeled Test Data | 4 waveform inversion solutions (1st/10th/14th/24th) built pseudo-labeled pairs from test predictions through the forward model; ClimSim 10th also trained on pseudo-labeled test data | principles/05.md |
| 6 | Leverage Physics Simulators for Data Augmentation and Post-Processing | 8/10 waveform inversion solutions, 1/1 ClimSim top solution (forward model to refine predictions) | principles/06.md |
| 7 | Choose Spectrogram vs Tabular Features Based on Data Scale and Model Type | Most earthquake solutions used hand-crafted tabular features; the 19th-place team added STFT-spectrogram 2D-CNNs and 1D-CNNs only as ensemble members | principles/07.md |
| 8 | Apply Domain-Specific Augmentations Carefully with Physical Validity | 5/10 waveform inversion solutions used domain-specific augmentation to preserve physical validity. | principles/08.md |
| 9 | Use Multi-Task Learning with Auxiliary Targets to Regularize Predictions | 3/5 earthquake top solutions (1st place NN, 12th place RNN), 1/1 ClimSim 1st place | principles/09.md |
| 10 | Ensemble Diverse Model Families, Not Just Multiple Seeds | 15/15 earthquake solutions, 10/10 waveform solutions - all ensembled | principles/10.md |
| 11 | Validate with Target Distribution Matching, Not Just Metric Scores | 4/15 earthquake solutions (1st/2nd/12th/15th explicitly compared prediction distributions) | principles/11.md |
| 12 | Use Test-Time Augmentation (TTA) for Invariant Predictions | 6/10 waveform inversion solutions used TTA. | principles/12.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together: start by aligning distributions and designing robust CV, then engineer features appropriate to your data scale (tabular vs spectrograms), train diverse models with physical augmentations and multi-task learning, ensemble them carefully, and refine with physics-based post-processing where applicable. Success comes from understanding the physical system well enough to avoid breaking it with naive ML techniques.
