# PS S6E4 retrospective: +42 rank jump to Top 0.7% (rank 30/4316)

Competition: playground-series-s6e4
Rank: #30
Source: https://www.kaggle.com/c/playground-series-s6e4/writeups/ps-s6e4-retrospective-42-rank-jump-to-top-0-7

I wrote up the slot-selection rule that drove the +42 jump as a Notebook. Posting a short summary here for the Discussion side, with the full version (with code, plots, and the failure stories) linked at the bottom.

## The framing — and an honest disclosure up front

The single thread of the writeup is one question:

> **Among the two final slots we picked, why did the conservative one — with a much lower Public LB — survive Private while the aggressive one collapsed?**

This is *not* a claim that the conservative slot was the global optimum. It wasn't: a non-selected submission of ours landed at Private **0.98067** — the highest Private score across everything we generated. Picking it would have moved us toward rank ~20. We had no reliable signal at submission time that it would jump, so we did not pick it. The honest reading is that *some* of our non-selected submissions were better than what we picked, by chance more than by craft.

So the question this writeup actually answers is narrower: **among the two slots we actually picked, why did the conservative one survive?**

## Why the aggressive slot lost

Our aggressive slot was a public-NB-derived ensemble (a tiny tweak on top of nina2025's v.11 schema) that scored Public **0.98152** at rank 51. On Private it dropped to **0.98051**.

It was not a lonely casualty. The 54-team cluster I'll call the **commodity zone** — Public LB 0.98148–0.98152, mostly faithful reproductions of the same public ensemble — collapsed on Private to 0.98049–0.98051 *in lockstep*, about −0.00100pt across the board. Reproducing a heavily-shared public NB does not differentiate you on Private.

## Why the conservative slot won — slot selection

We operationalized "trust your CV" with one rule: **define `gap = OOF_bACC − Public_LB` and refuse to pick anything where the gap was negative.**

Three of our four slot finalists had `gap < 0` (Public LB exceeded OOF — the classic over-fit shape). Only one had `gap = +0.00009` — a positive number small enough to look like a real signal rather than under-training. We picked it as the conservative slot for that reason alone. On Private it landed at exactly the same 0.98059 as Public.

Three checklist items mattered:
1. **Gap positive and small** (`0 < gap ≲ 0.0001pt`) — large positive gap usually means under-trained, not robust.
2. **Independent aggregation** from the aggressive slot (we paired an LR-meta stacker against a CSV-vote attack — different aggregation families).
3. **High count drift ≤ ±100 rows** vs your stable baseline (motivated by a balanced_accuracy asymmetry calc — High flips are 4.4× wrong-favored).

## Why the conservative slot won — building the candidate

The architecture was inspired by Mikhail Naumov's NB (LB 0.98107): heterogeneous base pool with a `LogisticRegression` meta-learner. The non-trivial part was making LR meta actually work.

Our first attempt — **probabilities** as LR input — crashed to LB 0.97900. Sweeping 25 LR configs on a single fold made the failure mode legible: **three axes have to all be true at once**:

- Input as **logits**, not probabilities (probs flatten `max_p > 0.99` from 0.74 → 0.65, killing post-process bias-tune headroom)
- `class_weight='balanced'` (else High recall is starved)
- `multinomial` solver (default in sklearn 1.8+)

`C` value barely matters within `{0.01, 0.1, 1.0}`. With those three axes locked in, gap stayed at +0.00009pt on Public and the submission survived Private unchanged.

## Three failed paths the writeup also covers

- **Reproducing Edison Jiang's pipeline line-for-line → LB 0.96796** (vs his 0.98132). Most likely sklearn 1.8 vs older sklearn LR multinomial behavior change. **Lesson**: faithful code reproduction does not guarantee LB reproduction; pin library versions to the source environment.
- **Replacing TE columns with a rule-based logit** (3 columns ranked 1/2/3 in importance, 45% combined importance share) → base bACC dropped 0.033pt. **Lesson**: feature importance ≠ prediction quality.
- **NN-family override on a NN-base ensemble** (hypothesis: same-architecture-family → compatible boundary judgments) → wrong; damaged LB at the same per-row rate as GBDT-family overrides. The accidental find was the balanced_accuracy asymmetry numbers used in the slot-selection checklist.

## Full notebook (with plots, code, and 30+ acknowledgments)

📓 https://www.kaggle.com/code/shotafubasami/ps-s6e4-42-rank-jump-conservative-slot

## Postscript — looking for community

I ran this competition with Claude Code as an AI co-pilot (specialized subagents, persistent knowledge file, cross-LLM adversarial review). If you're running Kaggle competitions with AI co-pilots (Claude Code, Cursor, Codex, etc.) and have built a workflow that worked for you — or you know of an existing community for this kind of thing — I'd genuinely like to hear about it. Drop a comment.
