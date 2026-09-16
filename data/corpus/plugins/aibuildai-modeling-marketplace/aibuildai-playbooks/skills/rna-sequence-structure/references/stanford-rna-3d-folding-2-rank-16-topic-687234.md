# 16th Place Solution

Competition: stanford-rna-3d-folding-2
Rank: #16
Source: https://www.kaggle.com/c/stanford-rna-3d-folding-2/writeups/16th-place-solution

Thank you to the organizers at Stanford, HHMI, NVIDIA, and Kaggle for this incredible competition. As someone with no formal biology or bioinformatics background, this was my first encounter with RNA structure prediction — and it turned into one of the most rewarding learning experiences I've had. Congratulations to all participants 

## The Core Idea
Use Protenix as a **template generator**, not as the final predictor. Protenix produces one structural prediction per target, which gets fed into RNAPro as template conditioning. RNAPro then refines it using RibonanzaNet2 embeddings and its Pairformer architecture. TBM fills the remaining template slots to give RNAPro structural diversity to work with.

The time constraint shaped the design: running Protenix with N_SAMPLE=5 was too slow, so I used N_SAMPLE=1 and let RNAPro do the heavy lifting for diversity.

## Pipeline Overview

```
                    ┌──────────────┐
                    │   Test Seqs  │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌──────────────┐         ┌──────────────┐
      │     TBM      │         │   Protenix   │
      │  (top 5 by   │         │  (N_SAMPLE=1 │
      │  alignment)  │         │  per target) │
      └──────┬───────┘         └──────┬───────┘
             │                        │
        TBM[1..4]              Protenix[0]
             │                        │
             └───────────┬────────────┘
                         ▼
              ┌─────────────────────┐
              │   Template CSV      │
              │  Slot 1: Protenix   │
              │  Slot 2-5: TBM     │
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │      RNAPro         │
              │  (template_idx=0,   │
              │   RibonanzaNet2,    │
              │   N_SAMPLE=3)       │
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │   5 Predictions     │
              ├─────────────────────┤
              │ P1-P3: RNAPro       │
              │ P4-P5: Protenix/TBM │
              │        fallback     │
              └─────────────────────┘
```

## Phase 1 — Template-Based Modeling

TBM serves two purposes: backup predictions and template slots 2-5 for RNAPro input.

**Template search:** BioPython `PairwiseAligner` in global mode with strong gap penalties (`open: -8, extend: -0.4`). Searched ~5744 training+validation structures. Length ratio filter skips candidates with >30% length difference.

**Coordinate transfer:** C1' coordinates mapped from template to query via alignment. Gaps filled by linear interpolation between nearest aligned positions, or extrapolated at ends with 3.8Å steps.

**No quality threshold filtering** — all templates passed through to the CSV. RNAPro handles template quality internally through its confidence mechanism.

## Phase 2 — Protenix (Template Generation)

Protenix ran in minimal mode — the goal was one decent structural prediction per target, not five polished ones.

- **N_SAMPLE=1** (single prediction per target)
- **Single seed** (42)
- **No MSA, RNA MSA enabled**
- **Chunking** for sequences >512nt: overlapping windows with Kabsch alignment and cosine blending at boundaries (MAX_SEQ_LEN=512, CHUNK_OVERLAP=128)
- C1' extraction via `centre_atom_mask` with `atom_to_tokatom_idx` fallback

A Protenix-only backup `submission.csv` was saved before RNAPro ran — if RNAPro timed out, the notebook still produced a valid submission.

## Phase 3 — Template CSV Construction

For each target, the 5-slot template CSV was built as:

| Slot | Source | Purpose |
|------|--------|---------|
| 1 | Protenix prediction | Primary template — RNAPro refines this |
| 2-5 | TBM templates | Structural diversity (or noise-augmented copies if <4 templates) |

If Protenix failed for a target, slot 1 fell back to the best TBM template.

The CSV was converted to `.pt` format via RNAPro's `convert_templates_to_pt_files.py`.

## Phase 4 — RNAPro Inference

RNAPro conditioned on the Protenix template via `--template_idx 0` (uses only the top template slot — our Protenix prediction). Its architecture combines:

- **RibonanzaNet2** as frozen encoder — captures RNA-specific co-evolutionary features
- **Template conditioning** — Protenix structural prior guides the diffusion
- **Pairformer** — RNA-adapted structure module

Params were aggressive to fit the 8-hour time budget:

| Parameter | Value |
|-----------|-------|
| N_SAMPLE | 3 |
| N_STEP | 100 |
| N_CYCLE | 4 |
| MAX_LEN | 500 |
| MSA | disabled |
| Seed | 42 |
| template_idx | 0 |

## Phase 5 — Submission Assembly

Simple priority filling:
1. RNAPro CIF outputs → slots 1-3 (if available)
2. Protenix/TBM from backup CSV → remaining slots
3. Helix fallback → last resort (never triggered on public test)

No ensemble selection or post-processing — just straight slot filling.

## What Made the Difference

**Protenix as template generator, not final predictor.** The Protenix→RNAPro pipeline scored 0.464 private vs 0.410 for standalone Protenix. RNAPro's RibonanzaNet2 embeddings added RNA-specific knowledge that base Protenix lacks, especially on novel targets without close homologs.

**Time-budget-driven design.** Using Protenix N_SAMPLE=1 instead of 5 saved time, which was exactly the time needed for RNAPro to run. The constraint that felt limiting actually forced a better architecture.

## What Didn't Work

- **MSA in RNAPro** — disabled to save time; didn't noticeably help when tested
- **Protenix N_SAMPLE=5 as templates** — too slow, didn't finish within 8h
- **Running Protenix with MSA** — no improvement, extra time cost
- **Post-processing constraints on neural net output** — slightly degraded predictions; Protenix/RNAPro geometry is already valid
- **RBSSA (embedding-based template search)** — RibonanzaNet2 embeddings + Smith-Waterman. Hurt when it replaced Protenix (0.33), marginal when additive(0.425) on public data
- **Finetuned Protenix (RNA3DB checkpoint)** — scored 0.392; base model was better
- **DRfold2 / Boltz ensemble** — architecturally different but not better on any target; selector correctly ignored them

## Key Lessons

1. **Template quality is the foundation.** Every DL model performed better with good structural starting points. Protenix-generated templates were better than pure TBM for RNAPro conditioning.
2. **DL models are template refiners on Kaggle hardware.** On P100 (16GB), no model reliably folded RNA from scratch. They performed best refining structural starting points.
3. **Time budget shapes architecture.** The 8h limit forced Protenix N_SAMPLE=1, which turned out to be the right call — one good template + RNAPro refinement beat five mediocre Protenix predictions.

---

## Notebook 2: Protenix + TBM (Private: 0.410)

Brief summary of the second selected notebook:
- TBM with 50% identity threshold, diversity transforms (hinge, jitter, wiggle)
- Protenix multi-seed [42, 137] for targets below threshold
- **Consistency-based ensemble selection**: anchor on consensus Protenix prediction (lowest avg RMSD to others), guarantee one TBM slot, fill remaining by RMSD consistency. This was the single biggest improvement in the Protenix pipeline (0.428→0.440 on public).

## Acknowledgments

Thanks to the community for shared resources — Protenix notebooks, RNAPro from NVIDIA, and TBM approaches from Part 1 winners. Excited about the CASP17 collaboration opportunity.

**Kaggle:** [@error1249x](https://www.kaggle.com/error1249x)
