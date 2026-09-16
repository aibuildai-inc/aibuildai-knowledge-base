---
description: >-
  ML playbook for text normalization generation competitions. Use when tackling a Kaggle-style competition involving text normalization generation. Teaches how to reason about branch early by task structure: rule-based vs. learned assessment, diagnose train-test distribution shift early via prompt/score analysis, establish cv-lb correlation before trusting optimization, use deberta as default backbone for writing assessment. 124 top-solution writeups across 6 competitions (summary evaluation, readability scoring, machine translation, argument effectiveness, language-learning scoring, writing-process quality)
---

# Text Normalization & Writing Assessment Playbook

This category encompasses two distinct task families: (1) text transformation and generation—for example, translating inconsistently transliterated ancient texts into English, where normalization rules and learned sequence-to-sequence models work together, and (2) student writing assessment—scoring essays, summaries, readability, and argumentative writing on quality dimensions. The core challenge is adapting to data scarcity and distribution shifts between training and test sets, whether tackling rule-assisted text transformation or evaluating subjective human writing quality.

**Source material:** 124 top-solution writeups across 6 competitions (summary evaluation, readability scoring, machine translation, argument effectiveness, language-learning scoring, writing-process quality)

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Branch Early by Task Structure: Rule-Based vs. Learned Assessment | Deep Past translation solutions combined rule-based normalization and cleanup (orthography rules, name standardization, repeated n-gram removal) with learned ByT5 or LLM translators; most writing assessment solutions use transformer-based models | principles/01.md |
| 2 | Diagnose Train-Test Distribution Shift Early via Prompt/Score Analysis | Several commonlit-evaluate-student-summaries top solutions analyzed prompt-level differences between training, public, and private data | principles/02.md |
| 3 | Establish CV-LB Correlation Before Trusting Optimization | Several top linking-writing-processes-to-writing-quality solutions emphasize trusting a well-correlated CV over the public leaderboard as critical | principles/03.md |
| 4 | Use DeBERTa as Default Backbone for Writing Assessment | Most writing assessment solutions use DeBERTa (v2/v3, base/large/xlarge), including nearly all top-5 solutions | principles/04.md |
| 5 | Expand Data via External Corpora + Pseudo-Labeling | Many top solutions use external data or pseudo-labels; several use meta pseudo-labeling (summaries 1st, Feedback effectiveness 4th, ELL 20th) | principles/05.md |
| 6 | Apply Two-Stage Training When Distribution Shift Exists | Several feedback-prize-english-language-learning solutions pretrained on pseudo-labeled data from earlier Feedback competitions, then fine-tuned on the target set (one gained 0.002-0.005 CV); a Deep Past translation solution found continual pretraining then fine-tuning consistently beat direct fine-tuning | principles/06.md |
| 7 | Engineer Task-Specific Inputs: Add Prompts, Questions, and Special Tokens | Many solutions report gains from adding prompts; several use custom special tokens like <question>, <summary> | principles/07.md |
| 8 | Extend Inference Max Length Beyond Training Length | A few solutions report gains by using longer max_length at inference than training | principles/08.md |
| 9 | Stabilize Training with Multi-Seed Averaging and Full-Train | Many solutions average several seeds; several use full-train (single model on all data) over k-fold | principles/09.md |
| 10 | Generate Synthetic Data with LLMs When External Corpora Are Unavailable | Several solutions use LLMs to generate synthetic texts; the 1st-place summary solution used meta pseudo-labeling with LLM-generated prompts | principles/10.md |
| 11 | Ensemble with Simple Mean to Avoid Overfitting, Use Negative Weights Carefully | Many solutions use a simple mean; several use hill-climbing or Nelder-Mead, some allowing negative weights | principles/11.md |
| 12 | Apply Metric-Specific Post-Processing Only When It Survives Validation | Readability and language-learning scoring solutions rescaled predictions by predicted-value range (Nelder-Mead coefficients per bin, interval mean matching) for small gains; several other language-learning and summary solutions found post-processing did not help | principles/12.md |
| 13 | Diversify Ensemble via Architecture Variations, Not Just Hyperparameters | Many solutions vary pooling (CLS, mean, GEM, LSTM, attention); several vary max_length; some use auxiliary losses | principles/13.md |
| 14 | Use MLM Pretraining on Domain Data to Adapt Backbones | Several solutions use MLM or Span MLM on essay data before fine-tuning | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together as a pipeline: branch by task type early, diagnose distribution shifts, establish CV-LB correlation, then layer in data expansion, architecture choices, training stabilization, and ensembling. For writing assessment, the 80/20 is: DeBERTa-v3 + external data/pseudo-labeling + two-stage training (if shift exists) + simple mean ensemble.
