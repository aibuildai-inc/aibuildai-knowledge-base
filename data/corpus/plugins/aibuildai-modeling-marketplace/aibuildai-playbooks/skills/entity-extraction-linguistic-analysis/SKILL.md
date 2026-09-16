---
description: >-
  ML playbook for entity extraction linguistic analysis competitions. Use when tackling a Kaggle-style competition involving entity extraction linguistic analysis. Teaches how to reason about choose prediction granularity based on annotation boundary alignment, pre-train on task-aligned data even when transfer seems indirect, use soft pseudo-labels on unlabeled data with confidence filtering, analyze annotation consistency and adapt loss weighting accordingly. 97 top-solution writeups across 6 competitions: Coleridge Initiative (dataset extraction), NBME (clinical notes), PII Detection, Feedback Prize (argumentative elements), and others
---

# Entity Extraction & Linguistic Analysis Playbook

Entity extraction competitions require identifying and extracting specific text spans from documents—whether dataset mentions in scientific papers, clinical features in patient notes, or personally identifiable information in essays. The core challenge is handling the gap between what entities mean semantically and how they appear syntactically: the same entity can be expressed in countless surface forms, annotations are often incomplete or inconsistent, and token boundaries rarely align with entity boundaries. Success requires balancing powerful contextual models with careful boundary prediction and extensive post-processing.

**Source material:** 97 top-solution writeups across 6 competitions: Coleridge Initiative (dataset extraction), NBME (clinical notes), PII Detection, Feedback Prize (argumentative elements), and others

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Prediction Granularity Based on Annotation Boundary Alignment | Seen across competitions - NBME (1st, 3rd, 11th, 18th, 19th) mapped token predictions to characters and repaired whitespace offsets; Feedback Prize (4th, 11th) blended different tokenizers at character level; token-level with boundary repair in others | principles/01.md |
| 2 | Pre-train on Task-Aligned Data Even When Transfer Seems Indirect | Seen across competitions - SQuAD-trained QA models reused for span extraction in Coleridge (47th) and Gendered Pronoun Resolution (9th); MLM on domain data in NBME (1st, 3rd, 19th, 24th) | principles/02.md |
| 3 | Use Soft Pseudo-Labels on Unlabeled Data with Confidence Filtering | Seen mainly in NBME - pseudo-labels on unlabeled patient notes in NBME (1st, 3rd, 4th, 5th, and most other top solutions), soft labels in NBME (1st, 19th, 20th) | principles/03.md |
| 4 | Analyze Annotation Consistency and Adapt Loss Weighting Accordingly | Seen in several competitions - NBME (1st, 2nd, multiple), PII (1st, 5th), Feedback Prize | principles/04.md |
| 5 | Ensemble for Diversity in Architecture, Not Just Seeds | Recurring across competitions - architectural diversity emphasized in most top solutions | principles/05.md |
| 6 | Invest Heavily in Post-Processing, Especially Boundary and Format Repair | Recurring across competitions - extensive post-processing in most top solutions, often worth as much as a new model | principles/06.md |
| 7 | Use Joint Start-End Prediction or Beam Search for Coherent Spans | Limited direct evidence - QA-style start-end span extraction in Coleridge (47th) and Gendered Pronoun Resolution (9th); joint start-end decoding is standard QA practice for valid spans | principles/07.md |
| 8 | Detect and Exploit Domain-Specific Structural Patterns | Seen in several competitions - Coleridge (2nd, 4th, and others used acronym detection), NBME (medical acronym tokens), PII (name propagation) | principles/08.md |
| 9 | Validate Cross-Validation Strategy Against Public LB Distribution | Seen in several competitions - Coleridge (4th place LB probing), NBME (many solutions noted CV-LB mismatch) | principles/09.md |
| 10 | Tune Per-Label Thresholds and Use Label-Specific Post-Processing | Seen in several competitions - PII (1st, 5th), NBME (9th), Feedback Prize (multiple), Coleridge (2nd, 4th) | principles/10.md |
| 11 | Use Adversarial Training or Stochastic Weight Averaging for Robustness | Seen in several competitions - Feedback Prize (1st: AWP), NBME (1st: AWP, 2nd: tried multiple adversarial methods), PII (some solutions) | principles/11.md |
| 12 | When External Data is Abundant, Prioritize Quality Over Quantity | Seen in several competitions - PII (1st, 5th used curated LLM-generated data), NBME (external data less useful), Coleridge (mixed results) | principles/12.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

Entity extraction requires balancing powerful pre-trained models with careful error analysis and domain knowledge. Start with a strong baseline (DeBERTa + SQuAD pre-training + proper CV), analyze errors to guide post-processing, build diversity into ensembles, and validate that your local improvements transfer to the test distribution. No single principle dominates—the highest scores come from systematically applying all of them.
