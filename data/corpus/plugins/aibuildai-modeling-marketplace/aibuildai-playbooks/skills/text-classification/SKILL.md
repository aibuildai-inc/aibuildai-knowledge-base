---
description: >-
  ML playbook for text classification competitions. Use when tackling a Kaggle-style competition involving text classification. Teaches how to reason about choose architecture based on data scale and domain characteristics, maximize embedding coverage to minimize out-of-vocabulary impact, generate diverse synthetic data when test distribution is unknown, use back-translation for multilingual robustness. 69 top-solution writeups across 11 competitions including llm-detect-ai-generated-text, quora-insincere-questions-classification, and jigsaw-multilingual-toxic-comment-classification
---

# Text Classification Playbook

Text classification spans diverse challenges from toxic comment detection to LLM-generated text identification. The core challenge is choosing the right architecture-data-training triangle: traditional embeddings excel with constrained compute and clear patterns, transformers dominate when data is abundant and patterns are subtle, and LLM-based approaches shine when distinguishing machine- vs human-generated text. Success depends on matching your technical choices to data characteristics and bridging train-test distribution gaps.

**Source material:** 69 top-solution writeups across 11 competitions including llm-detect-ai-generated-text, quora-insincere-questions-classification, and jigsaw-multilingual-toxic-comment-classification

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Architecture Based on Data Scale and Domain Characteristics | Seen across the source competitions - Quora winners used embeddings+RNN (2018-2019), LLM detection winners used transformers+LLMs (2023+) | principles/01.md |
| 2 | Maximize Embedding Coverage to Minimize Out-of-Vocabulary Impact | Several embedding-based Quora solutions emphasize this - 1st place Quora reduced OOV from 50k to near-zero | principles/02.md |
| 3 | Generate Diverse Synthetic Data When Test Distribution is Unknown | 5/5 LLM detection solutions - 1st place generated 160k samples across 40+ models/prompting strategies | principles/03.md |
| 4 | Use Back-Translation for Multilingual Robustness | Half of the Jigsaw multilingual writeups (8/16) trained on translated data; 3 also averaged predictions over translated test text | principles/04.md |
| 5 | Bridge Train-Test Distribution Gaps with Pseudo-Labeling | Top solutions in 3 of the 11 competitions used this (Jigsaw multilingual, LLM detection, Jigsaw community rules) - 11th place community rules gained +0.00155 private score | principles/05.md |
| 6 | Optimize Thresholds Using Cross-Fold Stability, Not Single-Fold Optima | 3/3 imbalanced-class competitions - 1st place Quora found stable threshold across 10 folds | principles/06.md |
| 7 | Optimize Batch Processing for Efficiency Without Sacrificing Accuracy | Common in the kernel-limited Quora competition - 2nd place Quora reduced runtime 40% with adaptive padding | principles/07.md |
| 8 | Use Custom Loss Functions to Match the Metric and Class Imbalance | Several solutions changed the loss to match the metric or the label structure - 20th place Quora gained +0.002 CV by adding a soft F1 loss to BCE | principles/08.md |
| 9 | Train Language-Specific Models for Multilingual Tasks, Then Ensemble | 2/2 multilingual winners - 1st place trained 6 separate monolingual models | principles/09.md |
| 10 | Handle Data Corruption with Targeted Augmentation, Not Cleaning | 3/5 LLM detection solutions - 4th place embraced typos instead of fixing them | principles/10.md |
| 11 | Ensemble Through Architectural and Process Diversity, Not Just Seeds | Top solutions in most source competitions ensemble, best gains from diverse approaches | principles/11.md |
| 12 | Use Perplexity and Token Probability Features for LLM Detection | 3/5 LLM detection winners - 8th place scored 0.956 with perplexity+GLTR alone | principles/12.md |
| 13 | Leverage Domain Adaptation at Test Time for Unknown Distributions | 1/5 LLM detection solutions - 5th place used student models trained on test pseudo-labels | principles/13.md |
| 14 | Train with Statistical and Linguistic Features for Robustness | Several Quora solutions used auxiliary statistical features - 2nd place Quora gained 0.00097 from 92 statistical features | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together as a decision tree: start with architecture selection based on your data scale and domain, then build a data strategy (generation/augmentation/pseudo-labeling) to bridge distribution gaps, optimize training for efficiency and for the metric, and finally ensemble diverse models with stable thresholds. The best solutions combine 5-7 of these principles, not all 13.
