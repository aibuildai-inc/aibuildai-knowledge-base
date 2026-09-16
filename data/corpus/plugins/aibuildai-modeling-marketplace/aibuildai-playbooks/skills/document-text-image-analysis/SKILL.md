---
description: >-
  ML playbook for document text image analysis competitions. Use when tackling a Kaggle-style competition involving document text image analysis. Teaches how to reason about generate synthetic training data to cover unseen combinations, choose two-stage (detect + classify) vs end-to-end based on localization difficulty, preserve full resolution and aspect ratio when information density is high, apply task-specific augmentations that match test distribution. 65 top-solution writeups across 5 competitions: Bengali grapheme classification, Kannada handwritten digit recognition, graph data extraction, ECG digitization, and ancient scroll surface detection.
---

# Document Text Image Analysis Playbook

Document text image analysis tasks require extracting structured information from images containing text, symbols, graphs, or handwriting. The core challenge is bridging the gap between visual representation and symbolic/semantic content while handling diversity in fonts, styles, noise levels, and unseen combinations. Success depends on choosing the right abstraction level (pixels → symbols → semantics), handling distribution shift between training and real-world documents, and generating sufficient training diversity.

**Source material:** 65 top-solution writeups across 5 competitions: Bengali grapheme classification, Kannada handwritten digit recognition, graph data extraction, ECG digitization, and ancient scroll surface detection.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Generate Synthetic Training Data to Cover Unseen Combinations | Recurring in top Bengali and Benetech graph solutions. Critical for competitions with combinatorial output spaces or domain shift. | principles/01.md |
| 2 | Choose Two-Stage (Detect + Classify) vs End-to-End Based on Localization Difficulty | Recurring in top graph solutions. Two-stage wins when localization is easier than recognition; end-to-end (image-to-text) wins when both are hard or tightly coupled. | principles/02.md |
| 3 | Preserve Full Resolution and Aspect Ratio When Information Density is High | Recurring in top solutions, including Bengali 1st/2nd. Cropping/aggressive resizing lost critical details in dense documents. | principles/03.md |
| 4 | Apply Task-Specific Augmentations That Match Test Distribution | Recurring in top solutions. Cutmix/fmix for Bengali, noise/blur/compression for scanned or photographed ECG printouts, color/blur for graphs. Generic augmentations (flips, rotates) often hurt. | principles/04.md |
| 5 | Predict at Component Level When Output Space is Compositional | 5/8 Bengali top solutions. Component-level prediction outperformed direct grapheme prediction for compositional outputs. | principles/05.md |
| 6 | Use Multi-Stage Training: Domain Adaptation then Task Specialization | Recurring in top Benetech solutions. Two-phase training (synthetic→real, coarse→fine, or general→specialized) consistently outperformed single-stage. | principles/06.md |
| 7 | Post-Process Predictions Using Domain Constraints | Recurring in top solutions: ECG 1st and 24th corrected leads with Einthoven's law (II = I + III), graph solutions mapped detected points to values through tick labels (3rd, 14th), Bengali 2nd turned whole-grapheme probabilities into component scores. | principles/07.md |
| 8 | Analyze and Close Train-Test Distribution Gaps Early | Most top solutions explicitly mentioned gap analysis. CV-LB correlation was the primary validation signal. | principles/08.md |
| 9 | Ensemble Diverse Architectures and Training Regimes, Not Just Seeds | Recurring in top solutions. Diversity came from different backbones (CNN vs ViT), input or patch sizes, loss functions, and training data. | principles/09.md |
| 10 | Use Pseudo-Labeling on Test Data for Writer/Style Adaptation | 4/6 Kannada-MNIST top solutions pseudo-labeled test data; the top 3 graph solutions pseudo-labeled extra unlabeled chart images. Confident predictions added to train improved the leaderboard score. | principles/10.md |
| 11 | Validate Structured Outputs with Auxiliary Models or Checks | Seen in several top graph solutions. Used auxiliary point-count models or geometry checks to filter/correct predictions. | principles/11.md |
| 12 | Leverage External Datasets with Careful Annotation Alignment | 8/10 graph solutions. ICDAR, WikiTables, and domain-specific corpora helped, but required annotation cleaning/reformatting. | principles/12.md |
| 13 | Train Coarse-to-Fine or Use Progressive Resizing for Efficiency | Seen in top Bengali and ECG solutions. Train at low resolution for many epochs, then fine-tune at higher resolution (Bengali 7th: 128 then 224; ECG 7th: five size stages). | principles/13.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together as a decision framework: start with distribution gap analysis and synthetic data generation to handle unseen cases, choose your architecture abstraction level (two-stage vs end-to-end, component vs full output), then optimize training (multi-stage, progressive resizing, domain-specific augmentations) and post-process with domain constraints. Prioritize the principles matching your task's specific characteristics—compositional outputs, localization difficulty, and train-test distribution shift.
