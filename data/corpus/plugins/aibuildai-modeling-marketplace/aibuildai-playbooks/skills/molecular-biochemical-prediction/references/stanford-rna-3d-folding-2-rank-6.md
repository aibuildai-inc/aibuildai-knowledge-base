# 6th Place Solution

Competition: stanford-rna-3d-folding-2
Rank: #6
Source: https://www.kaggle.com/c/stanford-rna-3d-folding-2/writeups/6th-place-solution

` Thanks everyone for a great competition. Congrats to all the winners!

---

## The Core Idea

The competition scores the best of 5 predictions, meaning diversity matters as much as accuracy. My strategy was to first generate diverse structural hypotheses cheaply using Template-Based Modeling (TBM). I then used Protenix in no-MSA, no-template mode to produce independent neural predictions. These outputs were later fed into RNAPro as templates, allowing the model to explore different regions of structure space. This cross-pollination between models turned out to be the key.

---

## Pipeline Overview

```
                         ┌──────────────────┐
                         │    Test Seqs     │
                         └────────┬─────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
           seq > 1000 nt                   seq ≤ 1000 nt
                  │                               │
                  │                ┌──────────────┴──────────────┐
                  │                ▼                             ▼
                  │         ┌─────────────┐             ┌─────────────┐
                  │         │     TBM     │             │   Protenix  │
                  │         │  (fast, 5   │             │  (no TBM,   │
                  │         │   diverse)  │             │  no MSA)    │
                  │         └──────┬──────┘             └──────┬──────┘
                  │                │                           │
                  │         TBM[0..4]              Protenix[0..1] as templates
                  │                │                           │
                  │                └─────────────┬─────────────┘
                  │                              ▼
                  │                      ┌──────────────┐
                  │                      │    RNAPro    │
                  │                      │  (refine +   │
                  │                      │   MSA +      │
                  │                      │ RibonanzaNet)│
                  │                      └──────┬───────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 ▼
          ┌──────────────────────────────────────────┐
          │              5 Predictions               │
          ├──────────────────────────────────────────┤
          │  P1 : RNAPro  +  TBM template [0]        │
          │  P2 : RNAPro  +  TBM template [1]        │
          │  P3 : RNAPro  +  Protenix template [0]   │
          │  P4 : RNAPro  +  Protenix template [1]   │
          │  P5 : Pure Protenix [0]  (no RNAPro)     │
          ├──────────────────────────────────────────┤
          │  ⚠️  seq > 1000 nt → TBM × 5 (all slots) │
          └──────────────────────────────────────────┘
```

---

## Phase 1 — Template-Based Modeling (TBM)

This runs first and serves two roles: a standalone fallback for long sequences, and a source of structural templates for RNAPro.

**Template search:** I use BioPython's `PairwiseAligner` in global mode. The  strong gap penalties (`open: -8, extend: -0.4`) including at the terminals. 

**Coordinate transfer:** Once aligned, C1' coordinates from the training structure are mapped onto the query residues directly. Gaps are filled by linear interpolation between neighboring residues, or extrapolated at the ends with a fixed 3Å step.

**Geometry refinement:** After transfer I apply `adaptive_rna_constraints` which runs within each chain segment independently (important for multi-chain targets — you don't want fake bonds across chain breaks):
- i↔i+1 bond → ~5.95 Å
- i↔i+2 soft angle → ~10.20 Å
- Laplacian smoothing to kill kinks
- Light steric self-avoidance for chains ≥ 25 residues

Correction strength scales with `(1 - similarity)` — if the template is a close match, barely touch it. If it's a rough match, correct more aggressively.

**Getting 5 diverse predictions from one template:**

| Pred | Transform |
|------|-----------|
| 0 | Best template, untouched |
| 1 | Mild Gaussian noise scaled by (1 − similarity) |
| 2 | Hinge rotation on the longest chain segment |
| 3 | Independent rigid-body jitter per chain |
| 4 | Smooth low-frequency wiggle via interpolated control points |

---

## Phase 2 — Protenix

I used [Protenix](https://github.com/bytedance/Protenix) (ByteDance's AlphaFold3-style model) in **no-MSA, no-template** mode. The goal here wasn't to get the best individual prediction — it was to get something structurally *different* from TBM that RNAPro could use as an alternative starting point.

- Applied to sequences ≤ 1000 nt (truncated to 512 for featurization)
- 5 samples generated; first 2 are passed forward as templates
- C1' atoms extracted via `centre_atom_mask` with fallback to `atom_to_tokatom_idx`

---

## Phase 3 — RNAPro

This is where the cross-pollination happens. I precompute a 4-slot template file combining TBM and Protenix outputs, then run RNAPro once per slot:

| Slot | Template Source |
|------|----------------|
| 0 | TBM prediction 1 |
| 1 | TBM prediction 2 |
| 2 | Protenix prediction 1 |
| 3 | Protenix prediction 2 |

RNAPro runs with MSA + RibonanzaNet2 embeddings, 10 recycling cycles, 200 diffusion steps. Each run is conditioned on a different template, so you get 4 structurally distinct refined outputs.

Prediction 5 is just raw Protenix[0] — no RNAPro. I kept it because on some targets (well-studied motifs with close training analogs) the direct Protenix output was hard to beat.

---

## What Actually Made the Difference


**Feeding Protenix outputs into RNAPro as templates.** Protenix and RNAPro have genuinely different inductive biases. RNAPro conditioned on a Protenix template explores a different part of structure space than RNAPro conditioned on a TBM template. This gave predictions 3 and 4 real independence from predictions 1 and 2.

**Not throwing away raw Protenix.** Prediction 5 being unrefined saved several targets where RNAPro overcorrected.

---

## What Didn't Work

- Running Protenix with MSA — didn't help and cost time
- More than 2 TBM template slots in RNAPro — performance plateaued after slot 2
- Ensemble averaging of coordinates — worse than best-of-5 strategy for this metric

---

## Hardware 

-  GPU P100 (Kaggle)
`
