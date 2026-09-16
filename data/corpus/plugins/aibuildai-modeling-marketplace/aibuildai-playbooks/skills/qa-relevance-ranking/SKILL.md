---
description: >-
  ML playbook for qa relevance ranking competitions. Use when tackling a Kaggle-style competition involving qa relevance ranking. Teaches how to reason about branch architecture by task structure, not just data type, invest in domain-specific pre-training or data when base models plateau, use retrieval + reranking pipeline for knowledge-heavy tasks, measure cv-lb agreement before trusting either on small or noisy data. 67 top-solution writeups across 8 competitions including Home Depot Product Search, Kaggle LLM Science Exam, and Eedi Math Misconceptions
---

# Question-Answer Relevance Ranking Playbook

QA relevance ranking encompasses diverse tasks: matching search queries to products, ranking search results for a user, selecting correct options, and retrieving or classifying student misconceptions. The core challenge is measuring semantic alignment between questions and candidate answers across different formats (text matching, multiple choice, retrieval, or classification). Success requires choosing the right architectural pattern for your specific task type and data characteristics.

**Source material:** 67 top-solution writeups across 8 competitions including Home Depot Product Search, Kaggle LLM Science Exam, and Eedi Math Misconceptions

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Branch Architecture by Task Structure, Not Just Data Type | 8/8 competitions show distinct architectural families | principles/01.md |
| 2 | Invest in Domain-Specific Pre-training or Data When Base Models Plateau | Critical for LLM Science (Wikipedia contexts, generated STEM questions), Eedi and MAP (synthetic math data) | principles/02.md |
| 3 | Use Retrieval + Reranking Pipeline for Knowledge-Heavy Tasks | 5/5 for tasks requiring external knowledge (LLM Science, Eedi top solutions) | principles/03.md |
| 4 | Measure CV-LB Agreement Before Trusting Either on Small or Noisy Data | Mixed in the source writeups: Eedi solutions saw CV gains that did not reach the LB, while a MAP solution with noisy labels chose its submission by out-of-fold score instead of the close public LB | principles/04.md |
| 5 | Post-Process Predictions to Match Target Distribution and Metric | Large impact in Eedi (rescaling seen versus unseen misconceptions) and MAP (filtering labels by question) | principles/05.md |
| 6 | Ensemble Diverse Architectures, Not Just Different Seeds | Common in the source writeups - LLM Science, MAP and Home Depot top solutions ensemble multiple model families | principles/06.md |
| 7 | Handle Ordinal Relevance Targets with Binary or Classification Decomposition | Rare in the source writeups: one Home Depot top solution stacked regression-via-classification models among its level-1 models | principles/07.md |
| 8 | Use Siamese Encoders to Search Many Candidates and Joint Inputs to Rank the Shortlist | Eedi and LLM Science top solutions encoded questions and candidates separately with embedding models to retrieve, then read the shortlist jointly with the question (Eedi 1st: retrievers chosen by recall@32, then a 14B ranker) | principles/08.md |
| 9 | Generate High-Quality Synthetic Data for Unseen Categories | Critical for Eedi (1st place), LLM Science | principles/09.md |
| 10 | Validate with GroupKFold on the Right Grouping Variable | Critical decision affects CV-LB correlation | principles/10.md |
| 11 | Augment the Parts of a Text Input That Can Change at Test Time | Recurring in the source writeups: shuffling answer or candidate order in training (LLM Science 7th, Eedi 3rd and 14th) and training on noisy retrieved contexts (LLM Science 1st); token masking appears in one solution, and another found CutOut and CutMix did not help | principles/11.md |
| 12 | Set Learning Rates Per Parameter Group and Tune the Head Ratio | Several LLM top solutions set separate learning rates per parameter group; where a head rate was reported, it was lower than the LoRA or backbone rate (LLM Science 1st, MAP 31st, Eedi 14th) | principles/12.md |
| 13 | Parse and Normalize Text Carefully for Feature Engineering Tasks | 2/2 for pre-transformer feature engineering competitions (Home Depot, CrowdFlower) | principles/13.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together in stages: identify your task structure and branch to the right architecture family, then apply domain-specific techniques (pre-training/retrieval for knowledge needs, validation strategy for your data size, post-processing for your metric). The meta-lesson: QA relevance ranking is not one task but a family - success comes from recognizing which sub-family you're in and adapting accordingly.
