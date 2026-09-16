---
description: >-
  ML playbook for location positioning trajectory competitions. Use when tackling a Kaggle-style competition involving location positioning trajectory. Teaches how to reason about classify problem type to choose methodology, apply spatial partitioning when the location space is large, fuse multi-modal data at the prediction level, not feature level, choose continuous vs discrete optimization based on constraint density. 78 top-solution writeups across 8 competitions: Facebook Check-ins, Lyft Motion Prediction, Google Smartphone Decimeter, Indoor Location Navigation, NFL Big Data Bowl, PKDD Taxi Trajectory, PKU Autonomous Driving, and Foursquare Location Matching
---

# Location Positioning & Trajectory Prediction Playbook

Location positioning and trajectory tasks span from static location prediction (check-ins, indoor positioning) to dynamic motion forecasting (autonomous vehicles, sports analytics) to precision GNSS refinement. The core challenge is handling multi-modal, noisy sensor data across spatial and temporal dimensions while respecting physical constraints. Success depends on correctly classifying your problem type—absolute positioning, trajectory prediction, or hybrid—and choosing the right balance between continuous optimization and discrete constraint satisfaction.

**Source material:** 78 top-solution writeups across 8 competitions: Facebook Check-ins, Lyft Motion Prediction, Google Smartphone Decimeter, Indoor Location Navigation, NFL Big Data Bowl, PKDD Taxi Trajectory, PKU Autonomous Driving, and Foursquare Location Matching

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Classify Problem Type to Choose Methodology | All competitions fall into 3 clusters, each requiring fundamentally different approaches | principles/01.md |
| 2 | Apply Spatial Partitioning When the Location Space is Large | 5/5 absolute positioning solutions, 3/3 hybrid solutions | principles/02.md |
| 3 | Fuse Multi-Modal Data at the Prediction Level, Not Feature Level | 4/5 top indoor navigation solutions, 3/5 GNSS solutions | principles/03.md |
| 4 | Choose Continuous vs Discrete Optimization Based on Constraint Density | Indoor navigation top-3 all used discrete; trajectory prediction top-3 used continuous | principles/04.md |
| 5 | Rasterize Spatial Context for CNN-Based Trajectory Prediction | 5/5 top Lyft solutions, 2/2 top autonomous driving solutions | principles/05.md |
| 6 | Align Training Data Distribution to Test via Filtering and Sampling | 4/4 Lyft solutions, 2/2 indoor navigation solutions that discussed validation | principles/06.md |
| 7 | Engineer Relative Motion Features for Trajectory Prediction | 3/3 indoor navigation sensor models, 2/2 NFL trajectory solutions | principles/07.md |
| 8 | Use ImageNet Pretraining for Vision-Based Models | 5/5 Lyft solutions, 1/1 NFL vision solution that mentioned it | principles/08.md |
| 9 | Ensemble Multi-Mode Predictions via GMM or Set Transformer | 3/3 top Lyft solutions, 1/1 NFL multi-mode solution | principles/09.md |
| 10 | Optimize Trajectory-Level Objectives with Beam Search or Global Optimization | 2/2 top indoor navigation solutions, 1/1 top GNSS solution | principles/10.md |
| 11 | Train with Cosine Annealing to Near-Zero Learning Rate | 4/4 Lyft solutions that discussed LR schedules, 1/1 NFL solution | principles/11.md |
| 12 | Handle Missing or Unreliable Sensor Data via Uncertainty Weighting | 2/2 top indoor navigation solutions, 1/1 GNSS solution with quality analysis | principles/12.md |
| 13 | Generate Additional Waypoints for Discrete Optimization | 2/2 top indoor navigation solutions | principles/13.md |
| 14 | Validate on Temporally Split or Chopped Trajectories, Not Random Splits | 3/3 Lyft solutions, 2/2 indoor navigation solutions that discussed validation | principles/14.md |
| 15 | Profile and Optimize Data Loading Bottlenecks | 3/3 Lyft solutions, 1/1 that mentioned rasterizer optimization | principles/15.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

Apply problem-type classification first—it determines your entire approach. For absolute positioning, focus on spatial partitioning and multi-modal fusion. For trajectory prediction, invest in rasterization, pretraining, and data loading optimization. For hybrid tasks, combine both: multi-modal models for absolute anchoring + discrete optimization for trajectory-level constraints. The methods don't transfer across problem types, but the reasoning framework does.
