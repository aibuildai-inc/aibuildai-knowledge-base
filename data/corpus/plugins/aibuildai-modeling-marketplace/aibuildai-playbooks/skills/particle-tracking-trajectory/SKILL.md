---
description: >-
  ML playbook for particle tracking trajectory competitions. Use when tackling a Kaggle-style competition involving particle tracking trajectory. Teaches how to reason about optimize the reconstructed partition, not individual hit links, choose a reconstruction paradigm from geometry, occupancy, and compute, represent hits in physics- and detector-aware coordinates, control combinatorics with a coarse-to-fine hypothesis funnel. Six top-solution writeups from one competition, the TrackML Particle Tracking Challenge (ranks 1, 2, 3, 7, 9, and 11). Because all evidence comes from one detector and metric, consensus counts describe these six solutions rather than universal laws; the principles are generalized through conditional decision criteria.
---

# Particle Tracking Trajectory Reconstruction Playbook

Particle tracking is a structured reconstruction problem: infer a disjoint set of physically plausible trajectories from a dense, noisy 3D hit cloud arranged on detector surfaces. The central challenge is to suppress combinatorics without discarding recoverable tracks, then balance track purity, particle coverage, detector consistency, and runtime under the exact evaluation metric.

**Source material:** Six top-solution writeups from one competition, the TrackML Particle Tracking Challenge (ranks 1, 2, 3, 7, 9, and 11). Because all evidence comes from one detector and metric, consensus counts describe these six solutions rather than universal laws; the principles are generalized through conditional decision criteria.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Optimize the Reconstructed Partition, Not Individual Hit Links | 6/6 solutions ultimately reconstructed complete tracks; the winning solution explicitly measured score after every stage. | principles/01.md |
| 2 | Choose a Reconstruction Paradigm from Geometry, Occupancy, and Compute | 6/6 used a structured decomposition, but in three methodological clusters: local geometric seed-and-extend (ranks 1 and 3), learned pair affinity followed by curve reconstruction (rank 2), and transformed-space clustering with extension or merging (ranks 7, 9, and 11). | principles/02.md |
| 3 | Represent Hits in Physics- and Detector-Aware Coordinates | 6/6 exploited curve geometry; at least 5/6 also used layer, volume, duplicate-surface, or cell-direction information rather than raw Cartesian distance alone. | principles/03.md |
| 4 | Control Combinatorics with a Coarse-to-Fine Hypothesis Funnel | 6/6 reduced the search in stages—through pairs or tracklets, parameter hypotheses, clustering, extension, pruning, and final assignment. | principles/04.md |
| 5 | Branch the Seeding Strategy on Origin and Trajectory Regime | 4/6 explicitly exploited proximity to the interaction vertex or z-axis; rank 3 then ran separate searches for non-vertex and lower-momentum tracks. | principles/05.md |
| 6 | Prefer Locally Updated Motion Models When Global Physics Is Misspecified | 4/6 explicitly improved candidates with curve fitting, local three-hit helices, z-axis constraints, or magnetic-field corrections; ranks 1 and 3 emphasized local propagation. | principles/06.md |
| 7 | Use Supervised Learning at Narrow Ambiguity Points | 4/6 used supervised models in a central role: logistic regression for pair/triple pruning, a deep pair classifier, ML candidate scoring, or gradient-boosted track extension. Other high-ranked solutions showed that geometry can remain the main engine. | principles/07.md |
| 8 | Calibrate Gates to Local Occupancy and Model Uncertainty | 4/6 made this explicit through outlier-density thresholds or adaptive clustering widths; the winner modeled chance outliers rather than relying on an unavailable exact inlier-noise model. | principles/08.md |
| 9 | Rank Candidates by Detector Coverage and Path Plausibility | 5/6 explicitly ranked or filtered complete candidates; rank 9 found distinct volume coverage and learned layer-sequence plausibility more effective than raw hit count. | principles/09.md |
| 10 | Resolve Shared Hits Globally and Peel Conservatively | 4/6 explicitly described best-candidate conflict resolution followed by hit removal; several others merged or reconstructed tracks from overlapping candidate evidence. | principles/10.md |
| 11 | Recover Missing and Duplicate Hits Only After a High-Purity Core Exists | 4/6 explicitly used track prolongation or supervised/geometric extension; the winner added same-layer duplicate hits in a separate late stage. | principles/11.md |
| 12 | Ensemble Complementary Search Passes, Not Redundant Copies | 3/6 clearly combined multiple passes or models: rank 3 searched detector regions and regimes sequentially, rank 7 merged long parameter sweeps, and rank 9 merged a specialized inner-detector model with broader models. | principles/12.md |
| 13 | Instrument Stagewise Oracle Recall and Runtime Before Tuning Details | The winner explicitly built load/score checkpoints and upper-bound metrics after every stage; at least 4/6 writeups reported stage gains or runtimes, making this a high-value winning practice despite limited explicit documentation. | principles/13.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use these principles as a sequence: derive the track-level objective, choose a geometry-compatible candidate generator, preserve recall through a measured hypothesis funnel, and only then learn, rank, assign, recover, and ensemble. The strongest design is usually hybrid—physics constrains what is possible, detector topology constrains what is plausible, and statistical learning resolves the ambiguity that remains.
