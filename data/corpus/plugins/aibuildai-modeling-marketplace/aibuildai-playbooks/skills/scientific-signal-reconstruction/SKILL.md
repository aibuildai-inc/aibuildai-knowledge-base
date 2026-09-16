---
description: >-
  ML playbook for scientific signal reconstruction competitions. Use when tackling a Kaggle-style competition involving scientific signal reconstruction. Teaches how to reason about classify the measurement-to-target geometry before choosing a model, match local and global receptive fields to the physical dependency range, encode physical coordinates, symmetries, and target manifolds explicitly, promote a trusted forward operator to a first-class model component. 17 solution writeups across 2 competitions: 15 from Geophysical Waveform Inversion and 2 from Flavours of Physics. Detailed methodological consensus is strongest in Waveform Inversion; Flavours of Physics provides a boundary case for constraint-aware scientific classification.
---

# Scientific Signal Reconstruction Competition Playbook

Scientific signal reconstruction spans several geometries, from inverse mapping of waveforms to fields to event classification under scientific validity checks. The core challenge is not merely fitting labels; it is choosing a representation, objective, validation design, and inference procedure that respect the acquisition process, target geometry, simulator fidelity, and deployment constraints.

**Source material:** 17 solution writeups across 2 competitions: 15 from Geophysical Waveform Inversion and 2 from Flavours of Physics. Detailed methodological consensus is strongest in Waveform Inversion; Flavours of Physics provides a boundary case for constraint-aware scientific classification.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Classify the Measurement-to-Target Geometry Before Choosing a Model | Both competitions require this decision, but they occupy different methodological branches rather than sharing one architecture. | principles/01.md |
| 2 | Match Local and Global Receptive Fields to the Physical Dependency Range | 5/6 top waveform solutions used ViT/CAFormer/ConvFormer-style global models or a global learned initializer. | principles/02.md |
| 3 | Encode Physical Coordinates, Symmetries, and Target Manifolds Explicitly | Coordinate realignment and geometry-aware embeddings appeared in waveform solutions. | principles/03.md |
| 4 | Promote a Trusted Forward Operator to a First-Class Model Component | 13/15 waveform writeups referenced forward modeling, simulation, or generated data; it was central to all top-6 waveform solutions. This is specialized but decisive when a usable operator exists. | principles/04.md |
| 5 | Introduce Specialists Only for Observable Physical Regimes | 5/15 waveform writeups explicitly routed by data family. | principles/05.md |
| 6 | Make Validation Reproduce Acquisition Groups, Difficulty Strata, and Scientific Checks | Both competitions required metric-specific validation; top writeups used large held-out subsets, family-wise scores, or supplied control datasets. | principles/06.md |
| 7 | Optimize the Metric’s Geometry, but Stabilize It with a Compatible Surrogate | Both competitions had materially different objectives; top waveform models generally trained with MAE to match evaluation. | principles/07.md |
| 8 | Augment the Latent Physical State, Then Propagate the Transformation Consistently | All top-6 waveform solutions used augmentation or synthetic generation. | principles/08.md |
| 9 | Audit Simulator Fidelity and Keep an Anchor to Real Measurements | Both competitions exposed simulator or acquisition-domain artifacts: waveform dtype/boundary mismatches, and Flavours of Physics real-versus-simulated agreement checks. | principles/09.md |
| 10 | Engineer Throughput Around Information Density, Not Nominal Tensor Size | 6/6 top waveform solutions depended on large-scale simulation or long training. | principles/10.md |
| 11 | Scale Resolution, Model Size, and Data in Stages | 5/6 top waveform solutions used very large synthetic corpora or progressive resolution. | principles/11.md |
| 12 | Average Only Valid Stochastic or Symmetry-Related Views | 3 waveform solutions (1st/4th/14th) used horizontal-flip TTA, mapping each flipped prediction back before averaging. | principles/12.md |
| 13 | Ensemble for Error Diversity, Then Learn Routing Only if Disagreement Is Predictable | Ensembling, blending, stacking, or model selection is common among the source writeups; the largest gains came from family-aware waveform blends. | principles/13.md |
| 14 | Use Prediction-Time Constraints as Projections or Regularized Refinement | 4/15 waveform writeups explicitly optimized predictions through the forward model, including the 2nd-, 5th-, 10th-, and 14th-place solutions; output-domain projection is relevant to both competitions. | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use these principles as a sequence: identify geometry and valid symmetries, select the dependency structure, design leakage-resistant validation, then decide how much physics, simulation, scaling, and inference refinement the evidence justifies. The strongest pipelines combine learned statistical inversion with physical consistency while remaining skeptical of simulator artifacts and unvalidated priors.
