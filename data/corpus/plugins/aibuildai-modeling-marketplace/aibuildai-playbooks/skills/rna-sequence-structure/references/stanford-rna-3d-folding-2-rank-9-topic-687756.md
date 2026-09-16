# 9th place solution: Optimize diversity when representation is limited

Competition: stanford-rna-3d-folding-2
Rank: #9
Source: https://www.kaggle.com/c/stanford-rna-3d-folding-2/writeups/optimize-diversity-when-representation-is-limited

First, I would like to thank Kaggle and the host for this interesting challenge. Second, I want to give credits to authors of these incredible public notebooks, on which my solution was built: [Boltz2_baseline](https://www.kaggle.com/code/lbugnon/boltz2-baseline), [Protenix+TBM](https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm), [RNAPro inference with TBM
](https://www.kaggle.com/code/jaejohn/rnapro-inference-with-tbm), [RNAPro Inference](https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-rnapro-inference)

## Templates
The representation of RNA that is available is its sequence. That's the only thing I can work with really. So, to find the best templates, I need to first understand what templates are "good" in terms of their sequence representation. Thus, I did a templates audit on `validation_sequences`, trying to find their best templates from `train_sequences`. Here is an example of what I found: for sequence `8ZNQ` (30nt) the best templates seems to be `1AKX`, which have a TM score of 0.27864. However, using biopython sequence aligner, the sequence similarity will be -0.175, which means there is no chance to find this template only using sequence aligner. This is what referred to as short "template-free" targets discussed by the host. For longer sequences, aligner seems to work better.

The conclusion here is clear, given representation of RNA only as sequences, there is a limit to the quality of templates I can find. I did experiment other TBM methods (primary chain + side chain search). Some worked well in my local validation but performs poorly on the LB when combined with DL models so I did not end up using. Because "hard" sequences cannot be found by aligner and "easy" sequences can almost be found by any aligner, the TBM used in my final submission are public TBM methods with very little modification.

| method | local validation | Public LB (before rerun)  | Private LB (before rerun)
| --- | --- | --- | --- |
| primary chain (selected) | 0.33840 | 0.32006 | 0.46044 |
| primary chain john ver. (selected) | 0.31873 | 0.34932 | 0.45131 |
| primary chain + side chain | 0.40150 | 0.32790 | 0.45089 |

## Overall prediction pipeline
With a standard TBM method, my model then pivots to create as much diversity in my predictions as possible. The final model is very similar to other shared solutions (TBM+protenix+boltz2+rnapro refinement). The difference is that for TBM and Protenix generated solutions, I select the ones that are most different from each other, thus maximizing diversity in my 5 predictions. More details can be found in the flowchart below.


```
┌─────────────────────────────────────────────────────────┐
│              📥  Input: Test RNA Sequences               │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  PHASE 1: Template-Based Modeling (TBM)  │
│                                                         │
│  1. Align each test sequence against 5,744 training     │
│     structures using global pairwise alignment          │
│                                                         │
│  2. Filter by similarity & percent identity (≥50%)      │
│                                                         │
│  3. Adapt template coordinates → query via alignment    │
│     (interpolate/extrapolate gaps)                      │
│                                                         │
│  4. Apply diversity transforms to generate variants:    │
│     • Hinge bending at random pivot points              │
│     • Chain jittering (rotate + translate per chain)    │
│     • Smooth wiggle (spline-based displacement)         │
│                                                         │
│  5. Greedy max-diversity selection                      │
│     (quality × diversity trade-off)                     │
│                                                         │
│  Slot allocation by length:                             │
│    Short (≤512 nt) → max 2 TBM, min 3 Protenix        │
│    Long  (>512 nt) → max 5 TBM, 0 min Protenix        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Remaining slots =     │
              │  5 − TBM count         │
              └─────┬─────────┬────────┘
                    │         │
          ┌─────────┘         └──────────┐
          ▼                              ▼
┌───────────────────────┐  ┌──────────────────────────────┐
│  PHASE 2: Protenix    │  │  PHASE 2.5: Boltz2           │
│  (2× T4 GPUs)         │  │  (2× T4 GPUs, after Protenix)│
│                       │  │                              │
│  IF n ≥ 3 slots:      │  │  Eligible: seq < 900 nt     │
│  ┌──────────────────┐ │  │  & slots still available     │
│  │ Phase A: Ensemble │ │  │                              │
│  │ GPU 0 → MSA run  │ │  │  1. Write multi-chain FASTA  │
│  │ GPU 1 → noMSA run│ │  │                              │
│  │       ↓          │ │  │  2. Persistent GPU workers   │
│  │ Kabsch-align +   │ │  │     (model loaded once,      │
│  │ weighted average  │ │  │      jobs run sequentially)  │
│  └──────────────────┘ │  │                              │
│                       │  │  3. Extract C1' coords       │
│  IF n < 3 slots:      │  │     from PDB output          │
│  ┌──────────────────┐ │  │                              │
│  │ Phase B: Split   │ │  │  ⚠ No chunking — full       │
│  │ Round-robin      │ │  │    sequence per predict call  │
│  │ across GPUs      │ │  └──────────────────────────────┘
│  │ (both use MSA)   │ │
│  └──────────────────┘ │
│                       │
│  Long seq handling:   │
│  ┌──────────────────┐ │
│  │ Chunk (512 nt,   │ │
│  │ 64 overlap)      │ │
│  │      ↓           │ │
│  │ Predict chunks   │ │
│  │      ↓           │ │
│  │ Kabsch-align     │ │
│  │ overlaps + blend │ │
│  │ → reassemble     │ │
│  └──────────────────┘ │
└───────────┬───────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│           🔗  Build Combined Predictions (5 slots)       │
│                                                         │
│   Slot priority:                                        │
│     1. TBM predictions                                  │
│     2. Protenix predictions (w/ RNA constraints)        │
│     3. Boltz2 predictions  (w/ RNA constraints)         │
│     4. De novo fallback    (idealized A-form helix)     │
│                                                         │
│   All predictions pass through adaptive_rna_constraints │
│   (bond lengths, angles, smoothing, self-avoidance)     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│         PHASE 3: RNAPro Refinement (2× T4 GPUs)         │
│                                                         │
│  Skip if sequence > 1,000 nt                            │
│                                                         │
│  1. Export all 5 combined predictions as .pt templates  │
│                                                         │
│  2. Split targets across GPUs                           │
│                                                         │
│  3. Run RNAPro inference in parallel                    │
│     (RibonanzaNet2 embeddings + precomputed templates)  │
│                                                         │
│  4. Replace combined predictions with RNAPro output     │
│     (keep originals for any failed targets)             │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               PHASE 4: Save Submission                   │
│                                                         │
│  📤  submission.csv                                      │
│  Format: 5 samples × (x, y, z) per residue C1' atom    │
│  Coordinates clipped to [−999.999, 9999.999]            │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

- **5 prediction slots** per target — filled greedily by TBM → Protenix → Boltz2 → de novo
- **Length-adaptive strategy** — short sequences favor model diversity (Protenix + Boltz2); long sequences favor template coverage
- **MSA/noMSA ensemble** — for targets needing ≥3 Protenix slots, two independent runs are Kabsch-aligned and averaged for better accuracy
- **Diversity selection** — TBM candidates are filtered by a greedy algorithm balancing structural diversity (TM-score) against template quality
- **Multi-GPU parallelism** — every phase distributes work across both T4 GPUs
- **RNAPro as final refinement** — treats all 5 combined predictions as templates and re-predicts, improving structural quality

## What I wished to do if have more time:
- Explore embedding based search
