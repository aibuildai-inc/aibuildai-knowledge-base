---
description: >-
  ML playbook for time series dynamics competitions. Use when tackling a Kaggle-style competition involving time series dynamics. Teaches how to reason about classify the prediction interface before choosing the model, preserve natural axes and sequence boundaries, expose dynamics at multiple scales, combine learned models with mechanistic identities and regime routing. 27 top-solution writeups across the 5 specified competitions: LEAP Atmospheric Physics, Web Traffic Forecasting, Jane Street Real-Time Forecasting, and COVID-19 Global Forecasting Weeks 4 and 5. The sources cover physical profile emulation, large panel forecasting, streaming financial prediction, point forecasting, and probabilistic forecasting.
---

# Time Series Dynamics Competition Playbook

Time series dynamics tasks share temporal dependence, but their prediction interfaces differ sharply: some forecast long horizons, some emulate structured physical profiles, and some operate as non-stationary online streams. The core challenge is therefore not choosing a fashionable sequence model; it is matching representation, validation, objective, and inference behavior to the information flow and constraints of the particular dynamical system.

**Source material:** 27 top-solution writeups across the 5 specified competitions: LEAP Atmospheric Physics, Web Traffic Forecasting, Jane Street Real-Time Forecasting, and COVID-19 Global Forecasting Weeks 4 and 5. The sources cover physical profile emulation, large panel forecasting, streaming financial prediction, point forecasting, and probabilistic forecasting.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Classify the Prediction Interface Before Choosing the Model | 5/5 competitions; every successful approach reflected the task's information interface, although the resulting model families differed substantially. | principles/01.md |
| 2 | Preserve Natural Axes and Sequence Boundaries | 5/5 competitions used meaningful grouping or ordering; several explicitly built recurrent, convolutional, attention, or state-space processing around it. | principles/02.md |
| 3 | Expose Dynamics at Multiple Scales | 5/5 competitions engineered lags, differences, integrals, rolling summaries, periodic references, profile derivatives, or cross-sectional aggregates. | principles/03.md |
| 4 | Combine Learned Models with Mechanistic Identities and Regime Routing | 3/5 competitions had decisive gains from exact formulas, stable baselines, or regime-dependent blending; the pattern was strongest in climate, web, and epidemic tasks. | principles/04.md |
| 5 | Pool Across Series, but Preserve Entity and Regime Context | Most source competitions benefited from global models trained across many series; local/statistical experts remained useful for sparse or exceptional regimes. | principles/05.md |
| 6 | Make Validation Reproduce Deployment Information Flow | 5/5 competitions treated validation design as central; all used chronological cutoffs or gaps, appropriate to their deployment information flow. | principles/06.md |
| 7 | Align Target Parameterization and Loss with the Metric | 5/5 competitions changed the loss, target representation, masking, or weighting to better reflect the evaluator. | principles/07.md |
| 8 | Match the Training Distribution to Recency, Season, and Data Scale | 4/5 competitions reported major gains from full-data scaling, random temporal windows, recent-only sampling, or time-decayed weighting. | principles/08.md |
| 9 | Use Auxiliary Targets to Teach Structure, Not Just Add Tasks | 2/5 competitions—LEAP and Jane Street—reported clear gains from derivative, related-responder, confidence, or group-specific auxiliary objectives. | principles/09.md |
| 10 | Regularize with Valid Invariances and Robust Representations | 3/5 competitions explicitly benefited from log/robust transforms, soft clipping, or feature coarsening. | principles/10.md |
| 11 | Choose Direct, Recursive, or Autoregressive Decoding by Horizon Stability | 3/3 long-horizon forecasting competitions made this choice explicit: web traffic and both COVID formulations used different direct, recursive, or sequence-decoder strategies. | principles/11.md |
| 12 | Ensemble Distinct Error Modes, Not Just More Models | 5/5 competitions used seed, fold, checkpoint, architecture, horizon, or mechanism/model blending; gains often exceeded gains from another small architecture tweak. | principles/12.md |
| 13 | Adapt Online Only When Delayed Labels Are Legally Available | 1/5 competitions, Jane Street, used this decisively; it is included because streaming feedback changes the optimal reasoning framework rather than being a generic trick. | principles/13.md |
| 14 | Project Predictions onto the Feasible Output Space | 4/5 competitions applied important post-processing constraints; the unconstrained Jane Street target was the main exception. | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use the principles as a sequence of decisions: identify the information interface and natural axis, design deployment-faithful validation, then choose representations, objectives, and adaptation strategies that match the dynamics. Architecture matters, but the reviewed solutions repeatedly won through correct problem framing, structural targets, robust validation, diverse blending, and constraint-aware inference.
