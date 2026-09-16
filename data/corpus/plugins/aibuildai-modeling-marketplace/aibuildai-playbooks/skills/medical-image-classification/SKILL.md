---
description: >-
  ML playbook for medical image classification competitions. Use when tackling a Kaggle-style competition involving medical image classification. Teaches how to reason about choose preprocessing based on imaging modality, handle sequential 3d medical data with hybrid 2d+rnn models, apply tile-based multiple instance learning for gigapixel histopathology, formulate ordinal disease grading as regression, not classification. Analysis of 187 top-solution writeups across 16 medical imaging competitions, including RSNA intracranial hemorrhage, PANDA prostate cancer grading, and UBC ovarian cancer classification.
---

# Medical Image Classification Playbook

Medical image classification spans diverse imaging modalities (CT, MRI, whole-slide histopathology, cellular microscopy, fundus and skin photography) and clinical tasks (cancer detection, disease grading, anatomical classification). The core challenge is handling domain-specific preprocessing requirements, extreme class imbalance, label noise from inter-rater variability, and limited training data for rare pathologies.

**Source material:** Analysis of 187 top-solution writeups across 16 medical imaging competitions, including RSNA intracranial hemorrhage, PANDA prostate cancer grading, and UBC ovarian cancer classification.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Preprocessing Based on Imaging Modality | Most top solutions apply modality-specific preprocessing | principles/01.md |
| 2 | Handle Sequential 3D Medical Data with Hybrid 2D+RNN Models | 3/3 top RSNA CT competitions used this; 1st and 2nd place intracranial hemorrhage both used LSTM on 2D embeddings | principles/02.md |
| 3 | Apply Tile-Based Multiple Instance Learning for Gigapixel Histopathology | 15/15 top solutions in PANDA, UBC-OCEAN used tiling; 10+ used attention/MIL | principles/03.md |
| 4 | Formulate Ordinal Disease Grading as Regression, Not Classification | At least 10/15 prostate grading (PANDA) writeups used regression, binned or ordinal targets, or a kappa loss instead of plain classification; one team found SmoothL1 regression no better than binned BCE | principles/04.md |
| 5 | Build Stable Validation with Multi-Year Data When Test Set is Small | Skin-lesion (ISIC 2024) solutions pooled earlier ISIC releases for training, and one also used them for validation because validation loss on the competition data alone was unstable. | principles/05.md |
| 6 | Apply Multi-Stage Training: External Pretraining Then Competition Fine-Tuning | External or prior-release data was common in protein atlas (HPA), skin-lesion (ISIC 2024), and ovarian cancer (UBC-OCEAN) writeups; reported gains ranged from large to none | principles/06.md |
| 7 | Handle Extreme Class Imbalance with Upsampling and Multi-Class Reformulation | Several solutions oversampled positive or rare-class examples; 6 skin-lesion (ISIC 2024) solutions added finer diagnosis classes or diagnosis-based auxiliary targets to the binary target. | principles/07.md |
| 8 | Filter Noisy Labels with Train-Predict-Remove-Retrain | PANDA 1st place (noisy label filtering was key breakthrough); 2+ other competitions used variants | principles/08.md |
| 9 | Use GeM Pooling Instead of Average Pooling | 10+ top solutions explicitly mentioned GeM pooling, including prostate grading (PANDA) and abdominal trauma solutions | principles/09.md |
| 10 | Maximize Batch Size for Stability with Rare Diseases | Several solutions reported that larger batches helped, including batch size 64 in a stroke clot-origin (Mayo STRIP AI) solution and gradient checkpointing for bigger batches in an abdominal trauma CT solution | principles/10.md |
| 11 | Apply Pseudo-Labeling in Stage 3 with Soft Labels and Head-Only Fine-Tuning | 12+ top solutions used pseudo-labeling effectively | principles/11.md |
| 12 | Ensemble Predictions with Rank Normalization Before Averaging | 4 solutions explicitly used rank-based ensembling, including the 1st place skin-lesion (ISIC 2024) solution | principles/12.md |
| 13 | Choose Test-Time Augmentation Based on Medical Imaging Physics | 30+ solutions used TTA; specific augmentations varied by modality | principles/13.md |
| 14 | Train for Fewer Epochs with Heavy Regularization to Combat Overfitting | Several solutions trained for only a few epochs, and some reported that longer training made results worse | principles/14.md |
| 15 | Integrate Metadata Carefully, or Skip It Entirely | 19/21 skin-lesion (ISIC 2024) writeups built GBDT or other tabular models on lesion and patient metadata, often with image-model predictions as extra features. | principles/15.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles form a decision framework, not a fixed recipe. Start with modality-specific preprocessing and validation strategy, then layer in multi-stage training and class balancing. Test every major choice (metadata, pseudo-labeling, TTA strategy) with ablations on your validation set. Medical imaging rewards domain knowledge - understand the anatomy and pathology before choosing techniques.
