# 7th Place Solution

Competition: stanford-rna-3d-folding-2
Rank: #7
Source: https://www.kaggle.com/c/stanford-rna-3d-folding-2/writeups/7th-place-solution

Thanks to the organizers for a great competition!

## Overview

3-phase ensemble pipeline combining Template-Based Modeling (TBM), Protenix (template-free) prediction, and RNAPro (NVIDIA's Part 1 winner).

```
 test_sequences.csv
       |
       v
 +-----------+    pairwise alignment, top_n=50
 |  Phase 1  |--> TBM Predictions (up to 5 slots)
 |   TBM     |
 +-----------+
       |  targets with remaining slots
       v
 +-----------+    2-5 samples (adaptive)
 |  Phase 2  |--> Protenix (template-free)
 |  Protenix |
 +-----------+
       |  combine TBM + Protenix into 5 slots
       v
 convert_templates_to_pt_files.py --> templates.pt
       |
       v
 +-----------+
 |  Phase 3  |    RNAPro (NVIDIA, Part 1 winner)
 |  RNAPro   |    N_sample=1, N_step=200, max_len=1000
 +-----------+
       |  merge (slots 1-4: RNAPro, slot 5: Phase 1-2 best)
       v
 submission.csv (final)
```



## Motivation: Why Multi-Model Ensemble?

Public notebooks built on TBM + Protenix were scoring 0.43–0.44 on the public LB. When I forked and resubmitted the same notebooks, **scores varied by ~0.02–0.03 across reruns** due to Protenix's stochastic diffusion. This got me thinking — how much of these scores is real accuracy vs. lucky sampling?

Since most participants seemed to be on TBM + Protenix, I figured I needed a different model to stand out. That led me to RNAPro.



## Phase 1: Template-Based Modeling (TBM)

The core TBM (alignment parameters, constraints, slot perturbations) is mostly the same as public notebooks. The one addition is **dynamic slot allocation** — adjusting the TBM vs Protenix ratio based on template quality.

For each test target, find similar sequences from the template pool (train + validation) using pairwise alignment, ranked by normalized alignment score.

### Slot Strategy

Instead of filling all 5 slots with TBM, allocate slots based on how good the best template is:

Allocate 5 prediction slots based on the best template's quality. The table below shows the maximum TBM slots — the actual count may be lower if the template pool lacks enough sequences passing the similarity/identity thresholds. Remaining slots are filled by Protenix.

```
 Template Quality Assessment
 ------------------------------------------------------------
 pct_identity ≥ 80%          -->  HIGH    --> up to 5 TBM
 AND match_rate ≥ 90%

 pct_identity ≥ 50%          -->  MEDIUM  --> up to 3 TBM

 otherwise                   -->  LOW     --> up to 1 TBM
 ------------------------------------------------------------
```

In practice, most targets classified as HIGH still had fewer than 5 qualifying templates — e.g., only 1 out of 28 targets achieved a full 5 TBM slots in this run.

Each TBM slot applies a different perturbation to diversify predictions:

```
 Slot 1: Best template (direct adaptation)
 Slot 2: + Gaussian noise (scaled by dissimilarity)
 Slot 3: + Hinge motion (longest chain segment)
 Slot 4: + Per-chain jitter
 Slot 5: + Smooth wiggle deformation
         |
         v
 adaptive_rna_constraints()  <-- stereochemical regularization
```

Key parameters: `top_n=50`, `MIN_PERCENT_IDENTITY=50%`


## Phase 2: Protenix (Template-Free)

For targets with remaining empty slots (medium/low quality templates), run **Protenix** (qiweiyin's adjusted weights) for template-free structure prediction, with MSA and template disabled. The number of Protenix samples is adaptive per target — 2 for medium-quality templates, 4 for low-quality (or 5 if no qualifying TBM template exists). TBM and Protenix predictions are combined into 5 slots per target:

```
 Per target:
 +-----------------------------------------------------+
 | Slot 1  | Slot 2  | Slot 3  | Slot 4  | Slot 5    |
 |---------|---------|---------|---------|-----------|
 |  TBM    |  TBM    |  TBM    | Protenix| Protenix  |  <- medium quality
 |  TBM    |  TBM    |  TBM    |  TBM    |  TBM      |  <- high quality
 |  TBM    | Protenix| Protenix| Protenix| Protenix  |  <- low quality
 +-----------------------------------------------------+
```


## Phase 3: RNAPro Enhancement (Key Differentiator)

Run **RNAPro** (NVIDIA, Part 1 winner, score 0.640) using Phase 1-2 predictions as template input. The RNAPro inference pipeline is based on [theoviel's public notebook](https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-rnapro-inference), adapted to use our Phase 1-2 output as templates. For targets where RNAPro succeeds (≤1000nt), slots 1-4 are replaced with RNAPro predictions while slot 5 retains the Phase 1-2 best prediction. Targets exceeding 1000nt keep Phase 1-2 predictions entirely.

```
 Phase 1-2 submission.csv
         |
         v
 convert_templates_to_pt_files.py  -->  templates.pt
                                              |
                                              v
                               +-------------------------+
                               |        RNAPro           |
                               |  checkpoint: 500M       |
                               |  + RibonanzaNet2 (MSA)  |
                               |                         |
                               |  N_sample=1             |
                               |  N_step=200, N_cycle=10 |
                               |  max_len=1000           |
                               +------------+------------+
                                            |
                               +------------v------------+
                               |   Per-target merge:     |
                               |                         |
                               |   RNAPro OK (≤1000nt):  |
                               |     Slots 1-4 <- RNAPro |
                               |     Slot 5   <- Phase1-2|
                               |                  best   |
                               |                         |
                               |   RNAPro skip (>1000nt) |
                               |   or fail/timeout:      |
                               |     -> KEEP Phase 1-2    |
                               +------------+------------+
                                            |
                                            v
                                   submission.csv (final)
```

RNAPro runs sequentially on a single GPU in the submitted notebook.


## What I Tried

### Protenix Parameter Tuning

Before pursuing multi-model ensembles, I tried many variations on the Protenix + TBM pipeline. None of the changes showed clear improvement:

- Protenix settings: SDPA/Flash Attention, MSA, multi-chain input, ligand SMILES, seed changes
- Inference scaling: multi-seed strategies, N_model_seed=2, ranking score-based sample selection
- Protenix version upgrade: v1.0.9 backport (SDPA, mol_type fix, chain reindex)
- Template pool expansion: Ribonanza templates (+167k), post-cutoff PDB, full PDB-RNA (+23k)

### Alternative Models Explored

After running out of ideas on Protenix tuning, I tried other models:

- **Chai-1** — AF3-family; slow and weak on validation; rejected
- **RhoFold+** — MSA-based; similar results to Chai-1; rejected
- **DRfold2** — RNA language model; fast but weak standalone; rejected
- **Boltz** — AF3-family; considered but deprioritized
- **RNAPro** — Part 1 winner (0.640); the only model that reached competitive scores on public LB

### Limitations After RNAPro Introduction

After integrating RNAPro into the pipeline, I was unable to conduct thorough ablation studies or hyperparameter tuning. The Kaggle GPU weekly quota was nearly exhausted from the preceding experiments, and the competition deadline left little room for systematic exploration of ensemble strategies (e.g., weighted merging, per-target model selection, or increasing RNAPro's `N_sample`).


## Compute

```
 +-----------------------------------------------------+
 | Hardware: Kaggle P100 x1    Time limit: 8 hours     |
 +-------------+--------------+------------------------+
 | Phase       | Time         | Notes                  |
 +-------------+--------------+------------------------+
 | Phase 1 TBM | ~11 min      | CPU-only               |
 | Phase 2 Ptx | ~79 min      | GPU (diffusion model)  |
 | Phase 3 RNAPro | ~282 min  | GPU (sequential)       |
 +-------------+--------------+------------------------+
 | Total       | ~6.2 hours   | within 8h limit        |
 +-------------+--------------+------------------------+
```


## Acknowledgments

Thank you to the Stanford and Kaggle organizers for hosting this competition, and to all participants who shared their code publicly.



## References

- **[RNAPro](https://github.com/NVIDIA-Digital-Bio/RNAPro)** (NVIDIA / Theo Viel) — Part 1 winning model; weights and inference code
- **[Protenix](https://github.com/bytedance/Protenix)** (ByteDance) — Diffusion-based structure prediction; adjusted weights by qiweiyin
- **[theoviel's RNAPro inference notebook](https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-rnapro-inference)** — Reference for Phase 3 (RNAPro) pipeline
