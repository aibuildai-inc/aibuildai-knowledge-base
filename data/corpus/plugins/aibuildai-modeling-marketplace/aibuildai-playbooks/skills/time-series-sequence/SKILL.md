---
description: >-
  ML playbook for time series sequence competitions. Use when tackling a Kaggle-style competition involving time series sequence. Teaches how to reason about design time-aware cross-validation that mirrors test conditions, classify your task type to choose the right approach, engineer temporal features that respect the forecasting horizon, apply domain-specific preprocessing before feature engineering. Analysis of 219 top-solution writeups across 29 competitions including M5 Forecasting, Optiver Trading, Liverpool Ion Switching, Jane Street Market Prediction, Child Mind Sleep Detection, and Career-Con surface classification.
---

# Time Series Sequence Playbook

Time series sequence tasks involve predicting future values, states, or classifications from sequential temporal data. The core challenge is handling temporal dependencies while preventing data leakage—models must never see future information during training. These competitions span diverse domains: market prediction, sales forecasting, signal classification, sensor-based activity detection, and medical time series analysis. Success requires careful validation design, domain-appropriate feature engineering, and architectures that capture both short-term patterns and long-term dependencies.

**Source material:** Analysis of 219 top-solution writeups across 29 competitions including M5 Forecasting, Optiver Trading, Liverpool Ion Switching, Jane Street Market Prediction, Child Mind Sleep Detection, and Career-Con surface classification.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Design Time-Aware Cross-Validation That Mirrors Test Conditions | Universal across all top solutions. Every winning solution uses time-respecting CV. | principles/01.md |
| 2 | Classify Your Task Type to Choose the Right Approach | Top solutions explicitly identify their task type early. M5 (forecasting), Liverpool (signal classification), Optiver (market prediction) use completely different techniques. | principles/02.md |
| 3 | Engineer Temporal Features That Respect the Forecasting Horizon | Every top solution uses lag and rolling features, but incorrectly computed ones cause the most common leakage failures. M5 top-3, Favorita 1st, Jane Street 1st all emphasize this. | principles/03.md |
| 4 | Apply Domain-Specific Preprocessing Before Feature Engineering | Signal tasks (Liverpool 1st, Sleep 1st) all preprocess heavily. Market tasks do neutralization. Forecasting tasks do minimal preprocessing. | principles/04.md |
| 5 | Choose Model Architecture Based on Data Structure and Forecasting Horizon | GBDT dominates tabular forecasting (M5, Favorita), RNNs dominate sequence modeling (Sleep), CNNs excel at signal patterns (Liverpool). Top solutions often ensemble across architectures. | principles/05.md |
| 6 | Handle Non-Stationarity Through Differencing or Adaptive Windowing | M5 1st removed first 85 days due to variance shift. Web Traffic 1st used median normalization. Liverpool solutions all handled drift. | principles/06.md |
| 7 | Use Target Encoding Carefully to Avoid Leakage | Liverpool 11th credited target encoding as a major boost. Jane Street 1st used supervised autoencoder. But Liverpool also had examples of leakage failures from bad target encoding. | principles/07.md |
| 8 | Detect and Handle Data Leakage Systematically | Liverpool had a 'drift-free data' leakage. Career-Con had a group_id leakage. Jane Street emphasized purging. Every competition has leakage traps. | principles/08.md |
| 9 | Build Ensembles Across Diverse Architectures and Training Schemes | Optiver 1st combined CatBoost + GRU + Transformer. M5 3rd used DeepAR. Every top solution ensembles. | principles/09.md |
| 10 | Apply Task-Specific Post-Processing to Optimize Metrics | Sleep 1st used greedy post-processing to extract events (huge boost). Optiver used weighted mean subtraction. Critical for top ranks. | principles/10.md |
| 11 | Use Online Learning or Incremental Updates for Live/Streaming Data | Optiver 1st retrained every 12 days during test period (massive boost). Jane Street had real-time inference. G-Research used online updates. | principles/11.md |
| 12 | Validate That Your CV Score Aligns With Leaderboard | Universal concern. M5 3rd explicitly selected models by past 14 periods. Optiver said 'CV aligns with LB very well'. All top solutions monitor CV/LB gap. | principles/12.md |
| 13 | Leverage Domain Knowledge and Competition-Specific Patterns | Liverpool found device noise patterns. Career-Con detected minute bias. Top solutions always do EDA to find domain patterns. | principles/13.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together—start with validation design and data understanding, then build features and models, finally ensemble and post-process. The most common mistake is jumping to modeling before understanding time dependencies and leakage risks. Always validate that your CV matches your LB before optimizing metrics.
