---
description: >-
  ML playbook for multimodal multilabel competitions. Use when tackling a Kaggle-style competition involving multimodal multilabel. Teaches how to reason about apply domain-specific preprocessing before generic normalization, validate against the test distribution shift, not just random folds, reduce dimensionality of high-dimensional targets before regression, handle label noise through soft targets, not just data cleaning. 35 top-solution writeups across 5 competitions (iMet-2019 artwork classification, Open Problems single-cell integration, Stable Diffusion image-to-prompts, YouTube-8M video understanding, Multi-modal representation learning)
---

# Multimodal Multilabel Prediction Playbook

Multimodal multilabel tasks require predicting multiple labels across different data modalities (images, text, video, biological signals). The core challenge is handling the interaction between modalities, dealing with noisy or incomplete labels, and optimizing for metrics that balance precision and recall across diverse label distributions. These tasks range from artwork classification to single-cell biology to image-prompt inversion.

**Source material:** 35 top-solution writeups across 5 competitions (iMet-2019 artwork classification, Open Problems single-cell integration, Stable Diffusion image-to-prompts, YouTube-8M video understanding, Multi-modal representation learning)

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Apply Domain-Specific Preprocessing Before Generic Normalization | 4/5 competitions - iMet used RandomCropIfNeeded, Open Problems used CLR/LSI/BM25, Stable Diffusion used CLIP preprocessing, YouTube-8M used precomputed embeddings | principles/01.md |
| 2 | Validate Against the Test Distribution Shift, Not Just Random Folds | 5/5 competitions - Open Problems had day+donor shift, iMet had second-stage size shift, Stable Diffusion needed diverse prompts, YouTube-8M had temporal distribution | principles/02.md |
| 3 | Reduce Dimensionality of High-Dimensional Targets Before Regression | 3/3 regression competitions - Open Problems used tSVD on targets (128-1024 dims), then inverse-transform predictions | principles/03.md |
| 4 | Handle Label Noise Through Soft Targets, Not Just Data Cleaning | 3/5 - iMet used knowledge distillation and pseudo-labeling, Open Problems used model predictions as soft targets, single-cell data inherently noisy | principles/04.md |
| 5 | Use Different Decision Thresholds Per Label Group When Label Characteristics Differ | 4/5 - iMet separated cultures (0-3 labels) from tags (1-8 labels) with different thresholds; Open Problems treated Multiome and CITEseq separately | principles/05.md |
| 6 | Pretrain on Large Noisy Data, Then Finetune on Small High-Quality Data | 4/5 - Stable Diffusion winners ALL used two-stage (6.6M low-quality → 2M high-quality), iMet used progressive training, Open Problems used batch-specific finetuning | principles/06.md |
| 7 | Ensemble Diverse Feature Representations, Not Just Diverse Models | 5/5 - Open Problems ensembled tSVD + UMAP + correlation features; Stable Diffusion ensembled text retrieval + vision encoders; iMet ensembled CNN features + ImageNet predictions + image stats | principles/07.md |
| 8 | Use Focal Loss or Class Balancing for Imbalanced Multilabel Data | 3/5 - iMet 1st and 4th place used focal loss, 9th used focal + F-beta loss; YouTube-8M dealt with extreme imbalance across 4716 classes | principles/08.md |
| 9 | For Image-to-Text or Cross-Modal Tasks, Leverage Pretrained CLIP Embeddings | 4/4 relevant competitions - Stable Diffusion ALL top solutions used CLIP ViT models, Open Problems used pretrained embeddings for sentence similarity | principles/09.md |
| 10 | Tune Image Resolution to Match Task Complexity | 3/3 vision competitions - Stable Diffusion winners found 336+ better than 224, iMet used 320-331, YouTube-8M noted resolution matters | principles/10.md |
| 11 | Generate Synthetic Training Data When Real Data is Scarce | 2/2 relevant competitions - Stable Diffusion winners generated 1M-10M images from prompts, Open Problems explored but didn't use heavily | principles/11.md |
| 12 | For Biology/Count Data, Use Correlation Loss Instead of MSE | 3/3 biological competitions - Open Problems winners ALL used correlation loss for CITEseq, metrics were correlation-based | principles/12.md |
| 13 | Extract Domain-Specific Features to Complement Learned Representations | 4/5 - iMet used ImageNet predictions + image stats, Open Problems used gene pathway features + cluster means, Stable Diffusion used CLIP embeddings + text retrieval | principles/13.md |
| 14 | Use KNN or Retrieval as a Non-Parametric Baseline or Ensemble Component | 2/5 - iMet 4th place used KNN-based tag relevance prediction, Stable Diffusion used text retrieval from 56M embeddings | principles/14.md |
| 15 | Apply Test-Time Augmentation Strategically Based on Compute Budget | 4/5 - iMet used minimal TTA (horizontal flip), Stable Diffusion used different normalizations, Open Problems mostly skipped TTA due to data size | principles/15.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles compose into a full workflow: validate against the test shift, apply domain preprocessing, reduce target dimensionality if needed, handle label noise via soft targets, pretrain then finetune, ensemble diverse feature representations, tune thresholds per label group, and leverage domain priors. Prioritize the principles that match your data characteristics and computational budget.
