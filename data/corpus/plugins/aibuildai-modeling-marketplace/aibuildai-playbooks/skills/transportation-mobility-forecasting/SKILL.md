---
description: >-
  ML playbook for transportation mobility forecasting competitions. Use when tackling a Kaggle-style competition involving transportation mobility forecasting. Teaches how to reason about choose modeling approach based on problem structure, engineer spatial features via clustering, not raw coordinates, create context-specific aggregations as historical baselines, decompose time into multiple granularities and encode numerically. 9 top-solution writeups across 6 competitions: NYC Taxi Trip Duration, PKDD-15 Taxi Trip Time, BigQuery-Geotab Intersection Congestion, Restaurant Revenue Prediction, Santa's Stolen Sleigh, and Walmart Sales in Stormy Weather.
---

# Transportation Mobility Forecasting Playbook

Transportation mobility forecasting encompasses predicting traffic patterns, trip durations, congestion levels, and location-based demand. The core challenge is modeling complex spatial-temporal interactions where movement patterns depend on location, time, network topology, and contextual factors like weather or events. Unlike pure time series, these problems require understanding how geographic relationships and network effects shape outcomes.

**Source material:** 9 top-solution writeups across 6 competitions: NYC Taxi Trip Duration, PKDD-15 Taxi Trip Time, BigQuery-Geotab Intersection Congestion, Restaurant Revenue Prediction, Santa's Stolen Sleigh, and Walmart Sales in Stormy Weather.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Modeling Approach Based on Problem Structure | All solutions: 5 used regression for time-based prediction, 1 used combinatorial optimization for routing | principles/01.md |
| 2 | Engineer Spatial Features via Clustering, Not Raw Coordinates | 4/5 regression solutions (NYC Taxi, PKDD-15, Geotab, Restaurant) | principles/02.md |
| 3 | Create Context-Specific Aggregations as Historical Baselines | 5/5 regression solutions | principles/03.md |
| 4 | Decompose Time into Multiple Granularities and Encode Numerically | 4/5 regression solutions (NYC, Walmart, Geotab, Restaurant) | principles/04.md |
| 5 | Use Nearest-Neighbor Features for Partial Trajectory Prediction | 2/2 taxi trip time competitions (PKDD-15: ranks #2, #3) | principles/05.md |
| 6 | Model Network Topology Explicitly for Congestion Problems | 1/1 congestion competition (Geotab #2 solution) | principles/06.md |
| 7 | Apply Log Transformation and Predict in Log Space for RMSLE Metrics | 4/4 competitions using RMSLE (NYC, PKDD-15, Walmart) | principles/07.md |
| 8 | Build Per-Entity Baseline Trends and Model Residuals | 2/3 demand forecasting solutions (Walmart #1, Restaurant #1) | principles/08.md |
| 9 | Handle Structural Zeros Explicitly with Successive-Zero Analysis | 2/2 sparse demand forecasting (Walmart #1, Restaurant #1) | principles/09.md |
| 10 | Blend Models via Stacking or Bagged Ensembles with Random Hyperparameters | 4/5 regression solutions (NYC, Walmart #3, Restaurant #1 implicitly, Geotab) | principles/10.md |
| 11 | Extract Route/Path Features via Dimensionality Reduction on Sparse Encodings | 1/2 taxi competitions with route data (NYC Taxi #4) | principles/11.md |
| 12 | Validate with Time-Aware or Stratified Splits Matching Test Distribution | 3/5 solutions explicitly mentioned (PKDD-15 #3, Walmart #3, Restaurant #1) | principles/12.md |
| 13 | Treat Weather and External Data as Weak Signals Requiring Careful Integration | 3/3 competitions with weather data (Walmart #1, Walmart #3, Geotab #2) | principles/13.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

Apply these principles in sequence: choose your modeling paradigm first, then engineer spatial and temporal features, build entity-specific baselines, handle structural zeros, and ensemble diverse models. Validate rigorously with splits matching test conditions, and treat external data as context, not oracle.
