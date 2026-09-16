---
description: >-
  ML playbook for protein function property competitions. Use when tackling a Kaggle-style competition involving protein function property. Teaches how to reason about branch first on the target's mathematical and biological structure, build the representation around the relevant biological invariance, use biological retrieval as a first-class expert, not merely a feature, fuse orthogonal evidence, but gate it by coverage and failure mode. 14 solution writeups across 3 competitions: 6 from CAFA 5, 3 from CAFA 6, and 5 from Novozymes Enzyme Stability Prediction. The sources include top-5 solutions in all three competitions and expose two methodological clusters: prospective GO-term multilabel prediction and single-variant thermostability ranking.
---

# Protein Function & Property Prediction Playbook

Protein competitions split into two distinct regimes: structured, open-world function annotation over an ontology, and scalar property prediction for sequence variants. The central challenge is to match the representation, validation, and inference logic to the biological observation process—future annotations for function tasks, versus mutation effects and relative ranking for stability tasks—while combining complementary evidence without leaking identity or overfitting a small leaderboard.

**Source material:** 14 solution writeups across 3 competitions: 6 from CAFA 5, 3 from CAFA 6, and 5 from Novozymes Enzyme Stability Prediction. The sources include top-5 solutions in all three competitions and expose two methodological clusters: prospective GO-term multilabel prediction and single-variant thermostability ranking.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Branch First on the Target's Mathematical and Biological Structure | 14/14: the 9 CAFA writeups and 5 stability writeups used fundamentally different formulations. | principles/01.md |
| 2 | Build the Representation Around the Relevant Biological Invariance | 9/9 function writeups used sequence-derived information or homology; 4/5 stability writeups explicitly centered local mutation environments or mutant-minus-wild-type features. | principles/02.md |
| 3 | Use Biological Retrieval as a First-Class Expert, Not Merely a Feature | 3/9 CAFA writeups explicitly built sequence-, structure-, or PPI-neighbor label transfer; it is included despite lower frequency because these independently strong experts appeared in top-5 solutions in both CAFA rounds. | principles/03.md |
| 4 | Fuse Orthogonal Evidence, but Gate It by Coverage and Failure Mode | 13/14 detailed solutions combined multiple feature families or prediction sources; every competition's leading writeups relied on complementary evidence. | principles/04.md |
| 5 | Treat Gene Ontology as a Structured Output Space | 6/9 CAFA writeups explicitly enforced hierarchy through graph models, hierarchy-aware modules, conditional factorization, or post-processing. | principles/05.md |
| 6 | Allocate Model Capacity Across the Long Tail Deliberately | 6/9 CAFA writeups explicitly varied frequency cutoffs, target counts, or label universes; several winning systems paired a strong frequent-label model with a much wider linear model. | principles/06.md |
| 7 | Model Annotation Provenance Instead of Treating All Labels as Truth | 7/9 CAFA writeups explicitly used evidence codes, non-experimental annotations, or filtered GOA/QuickGO records; top solutions repeatedly found these signals valuable. | principles/07.md |
| 8 | Validate Under the Scientific Shift, Not Just Random Rows | 12/14 writeups discussed cross-validation, temporal holdouts, source/pH grouping, duplicate control, or a train-test distribution mismatch. | principles/08.md |
| 9 | Align Supervision with the Metric and Observation Process | 7/14 writeups explicitly used IA weighting, ontology-conditional training, rank targets, or rank-transformed components; the competition metrics make this principle broadly applicable. | principles/09.md |
| 10 | Prefer Simple, Diverse Learners Before Unvalidated Architectural Complexity | 10/14 writeups reported strong linear models, boosting, random forests, simple MLPs, local descriptors, or physics scores; several complex alternatives failed to beat them. | principles/10.md |
| 11 | Ensemble Across Mechanisms, Not Cosmetic Variants | 14/14 final solutions used multiple models, seeds, representations, physical scores, or external prediction sources. | principles/11.md |
| 12 | Enforce Biological Consistency at the Cheapest Reliable Stage | 8/14 explicitly used GO true-path constraints, taxon filters, geometric equivariance, or WT↔mutant antisymmetry. | principles/12.md |
| 13 | Make Inference Policy Part of the Model | 8/14 writeups explicitly tuned propagation, top-N selection, score thresholds, smoothing, or rank-space aggregation. | principles/13.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use the principles as a decision chain: identify the target regime, choose biologically invariant representations and evidence sources, validate against the real shift, then align training and inference with the metric. The common winning pattern is not a single architecture but a conservative fusion of sequence knowledge, domain mechanisms, realistic validation, and explicit structural constraints.
