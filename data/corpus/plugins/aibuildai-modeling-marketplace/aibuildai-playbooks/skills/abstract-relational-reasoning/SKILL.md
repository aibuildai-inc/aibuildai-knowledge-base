---
description: >-
  ML playbook for abstract relational reasoning competitions. Use when tackling a Kaggle-style competition involving abstract relational reasoning. Teaches how to reason about identify the relational regime before choosing the solver, represent roles and relations, not incidental values, separate proposal, verification, and set construction, bound the search with complementary high-recall proposal channels. 30 top-solution writeups across 2 Kaggle competitions: 12 from Abstraction and Reasoning Challenge (few-shot visual program induction) and 18 from Foursquare Location Matching (large-scale noisy entity resolution).
---

# Abstract Relational Reasoning Competition Playbook

Abstract relational reasoning tasks require inferring latent rules that connect observations: either a transformation program from a handful of demonstrations or an equivalence relation over many noisy records. The central challenge is to expose the right relational structure, generate plausible hypotheses without an intractable search, and convert local evidence into metric-aligned, globally consistent predictions.

**Source material:** 30 top-solution writeups across 2 Kaggle competitions: 12 from Abstraction and Reasoning Challenge (few-shot visual program induction) and 18 from Foursquare Location Matching (large-scale noisy entity resolution).

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Identify the Relational Regime Before Choosing the Solver | 2/2 competitions exhibited distinct methodological clusters; nearly every substantive writeup followed the cluster implied by its data regime. | principles/01.md |
| 2 | Represent Roles and Relations, Not Incidental Values | Explicit canonicalization or relational representation appeared in at least 8/12 ARC writeups and 14/16 substantive Foursquare solution writeups. | principles/02.md |
| 3 | Separate Proposal, Verification, and Set Construction | 10/12 ARC writeups and 15/16 substantive Foursquare writeups used an explicit or implicit multi-stage pipeline of this form. | principles/03.md |
| 4 | Bound the Search with Complementary High-Recall Proposal Channels | 11/12 ARC writeups searched a bounded DSL or operation family; 15/16 substantive Foursquare writeups combined one or more blocking channels. | principles/04.md |
| 5 | Route by Observable Geometry and Output Topology | At least 8/12 ARC writeups branched on object, tile, grid, or size relations; at least 11/16 substantive Foursquare writeups used graph-aware assembly or group-dependent decisions. | principles/05.md |
| 6 | Match Model Capacity to the Amount and Kind of Supervision | 10/12 ARC writeups favored DSLs, simple trees, or explicit operations; 14/16 substantive Foursquare writeups used learned pair models after blocking. | principles/06.md |
| 7 | Validate on Unseen Relation Groups and the Real Candidate Universe | Group-aware validation was explicitly discussed in at least 8/16 substantive Foursquare writeups; leading ARC writeups also measured train/evaluation transfer and observed distribution shift. | principles/07.md |
| 8 | Learn from Ambiguous Near-Misses, Not Easy Negatives | Hard-example generation or ambiguity-focused augmentation appeared in at least 6/12 ARC and 7/16 substantive Foursquare writeups. | principles/08.md |
| 9 | Make Search and Inference Resource-Aware by Construction | Explicit acceleration, caching, deduplication, cascading, or batching appeared in at least 6/12 ARC and 10/16 substantive Foursquare writeups. | principles/09.md |
| 10 | Optimize the Proposal Ceiling and Final Metric Separately | Metric structure drove both competitions: ARC's top-3 exact-match ranking and Foursquare's candidate max-IoU/Jaccard diagnostics were repeatedly reported by leading solutions. | principles/10.md |
| 11 | Enforce Global Consistency, but Re-Verify Inferred Relations | Cross-demonstration consistency was central to at least 10/12 ARC approaches; graph consistency or traversal appeared in at least 11/16 substantive Foursquare solutions. | principles/11.md |
| 12 | Use the Output Budget for Complementary Hypotheses | Complementary solver components were combined in at least 8/12 ARC writeups and multiple leading Foursquare teams blended retrieval views or model families; the benefit was strongest when errors were non-overlapping. | principles/12.md |
| 13 | Audit Distribution Shift and Leakage as Part of Model Selection | 11/18 Foursquare writeups explicitly discussed the train-test overlap leak; leading ARC writeups also noted train/evaluation/leaderboard shift and, in one case, exploited public-equals-private feedback. | principles/13.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use these principles as a sequence: diagnose the relation and representation, build a bounded proposal-and-verification architecture, validate at the correct relational unit, and only then tune metric-aware assembly and consistency. The strongest transferable systems make each ceiling and assumption observable, so they can change methods when a new competition's supervision, topology, or deployment regime differs.
