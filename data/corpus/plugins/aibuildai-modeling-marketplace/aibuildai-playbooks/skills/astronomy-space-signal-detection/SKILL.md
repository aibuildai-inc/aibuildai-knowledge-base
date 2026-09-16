---
description: >-
  ML playbook for astronomy space signal detection competitions. Use when tackling a Kaggle-style competition involving astronomy space signal detection. Teaches how to reason about choose signal representation based on physics and sequence length, apply physics-informed preprocessing before learning, generate synthetic training data from physics simulators, handle multi-detector/multi-channel data with physics-aware architecture. 86 top-solution writeups across 7 competitions: g2net-gravitational-wave-detection, PLAsTiCC-2018, ariel-data-challenge-2024/2025, g2net-detecting-continuous-gravitational-waves, galaxy-zoo-the-galaxy-challenge, flavours-of-physics
---

# Astronomy Space Signal Detection Playbook

Astronomy space signal detection tasks require identifying rare physical phenomena (gravitational waves, exoplanet transit spectra, supernovae and other transients, galaxy morphologies, rare particle decays) in extremely noisy sensor data. The core challenge is extracting weak signals with low signal-to-noise ratios while handling multi-channel observations, temporal evolution, and substantial test-train distribution shifts caused by synthetic vs real data or instrument variations.

**Source material:** 86 top-solution writeups across 7 competitions: g2net-gravitational-wave-detection, PLAsTiCC-2018, ariel-data-challenge-2024/2025, g2net-detecting-continuous-gravitational-waves, galaxy-zoo-the-galaxy-challenge, flavours-of-physics

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Signal Representation Based on Physics and Sequence Length | Strong pattern across gravitational wave (15/19 top solutions used both) and spectroscopy competitions | principles/01.md |
| 2 | Apply Physics-Informed Preprocessing Before Learning | Critical in 18/19 gravitational wave solutions, 15/19 Ariel solutions, universally mentioned across competitions | principles/02.md |
| 3 | Generate Synthetic Training Data from Physics Simulators | Top solutions in g2net (1st place) and multiple others; 10+ solutions explicitly described simulation pipelines | principles/03.md |
| 4 | Handle Multi-Detector/Multi-Channel Data with Physics-Aware Architecture | Critical pattern in gravitational wave (3rd place showed 2bps gain from proper handling), Ariel (multiple channels with different PSFs) | principles/04.md |
| 5 | Address Train-Test Distribution Shift Aggressively | Major theme in PLAsTiCC (top solutions degraded training data to match test), Ariel (foreground correction) | principles/05.md |
| 6 | Choose Between Deep Learning and Classical Methods by Uncertainty Requirements | Ariel 2nd place used pure Bayesian inference with NO deep learning and scored 0.742; multiple solutions combined both approaches | principles/06.md |
| 7 | Engineer Domain Features for Tabular Models, Let CNNs Learn Them | PLAsTiCC 3rd place used extensive feature engineering (template fitting, luminosity, time differences); gravitational wave solutions split - tabular used features, CNNs used raw signals | principles/07.md |
| 8 | Use Test-Time Augmentation for Invariant Transformations | Gravitational wave solutions: 2bps gain from channel swap + vertical flip + time shift; nearly universal across top solutions | principles/08.md |
| 9 | Pretrain on Synthetic Data Then Finetune on Real Data | G2net 3rd place: 2-8bps gain from GW pretraining; multiple solutions across competitions used this pattern | principles/09.md |
| 10 | Use Pseudo-Labeling Selectively by Signal Type and Confidence | PLAsTiCC 3rd place: class90-only pseudo-labeling worked, all-class failed; g2net 3rd place used soft pseudo-labels; pattern across 8+ solutions | principles/10.md |
| 11 | Model Irregular Sampling Explicitly | PLAsTiCC top solutions handled sparse, irregularly sampled multi-band light curves with Gaussian-process interpolation (1st place) or with recurrent and convolutional encoders instead of treating them as regular time series; one team found GP-based features added nothing to their CV | principles/11.md |
| 12 | Ensemble Diverse Models with CMA-ES or Stacking | G2net 3rd place used CMA-ES optimization; nearly universal pattern | principles/12.md |
| 13 | Validate Preprocessing Choices by Checking PSD and Augmenting Based on It | G2net 1st place reverse-engineered PSD to generate noise; multiple solutions compared train vs test PSD; fundamental to signal processing approaches | principles/13.md |
| 14 | Handle Class Imbalance and Low-SNR Positives with Weighting and Careful Sampling | PLAsTiCC solutions tuned per-class or per-sample weights for the weighted log loss (one gained 0.04 LB from optimized sample weights), and the g2net 1st place noted that because of the SNR wall some positive samples are effectively just noise | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together: start by understanding the physics and choosing your signal representation (1D/2D/irregularly sampled), apply domain-specific preprocessing, then decide between deep learning and classical methods based on your data volume and uncertainty requirements. Augment with physics simulations, handle distribution shifts aggressively, and ensemble diverse approaches for robustness.
