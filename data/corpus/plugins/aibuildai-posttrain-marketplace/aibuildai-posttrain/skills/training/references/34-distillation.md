# Distillation: carrying a stronger model's knowledge into the student

Read in step 4 when a stronger source of supervision exists, and in step 11 when choosing the second stage.

Distillation is any training signal that comes from a stronger model: its written outputs, its judgments, or its token distributions. It is not one method. Sort the family by where each round's training data comes from, because that decides the pipeline shape, the cost, and the failure mode; the loss function is a detail inside one stage.

## Three data flows

| Flow | Where the training data comes from | Methods (methodology cards) | Entry conditions |
|---|---|---|---|
| **D1: fixed teacher data** | a stronger model's outputs, either a public dataset of its traces or a teacher served locally that writes responses for seed prompts; the data does not depend on the student | SFT on distilled datasets; teacher generation; `Distilling Step-by-Step`; `SeqKD`; `Context Distillation` | a teacher or dataset exists in the task's format; the student is too weak to produce useful samples itself |
| **D2: on-policy student sampling** | the student's own samples, scored by a verifier (keep the correct ones) or by the teacher's per-token distribution (match it) | `RFT`, `STaR`, `ReST`; `GKD` and the on-policy distillation recipe; `MiniLLM`; `DistiLLM`; `SKD`; `OPSD`, `SDFT` | the student already succeeds sometimes; a verifier or a teacher that fits beside the student on the GPU; a regression guard between rounds |
| **D3: preference distillation** | pairs or rankings of student samples judged by a teacher or a reward model, trained with `DPO` or reduced to the best sample and trained with SFT (`RAFT`, `BOND`) | `RAFT`, `RLAIF`, `BOND`, `DPO` with a teacher judge | no exact answer; a judge that agrees with the real grader on a held-out sample; candidates that differ enough for the judge to separate |

Ask the flow question before the method question. D1 and D2 are different pipelines: D1 changes the data and retrains from the same start, D2 feeds the current checkpoint's samples back into training and therefore needs a rollback rule. D3 is a branch after an SFT stage, not a loop.

## What the trainers ship

- D1 needs only `SFTTrainer`; the work is in the data: answer verification, per-problem trace selection, and dedup (`10-deduplication.md`, `12-verifier-filtering.md`).
- D2 with a verifier is `SFTTrainer` on the kept samples. D2 with teacher token distributions is trl's experimental `gkd` (GKD, teacher log-probs on student samples), `gold` (on-policy distillation with a separately loaded teacher), and `minillm` (reverse-KL distillation) modules; check the installed trl version, since these live under `trl.experimental` and can change between releases. `SDFT` needs a newer trl than most pinned environments carry.
- D3 is `DPOTrainer` on the pairs, or `SFTTrainer` on the best-of-N sample for RAFT.

## What the post-training records show

These come from an audit of documented post-training trajectories on small student models, each run under a fixed time budget on one GPU. The findings below are drawn only from the trajectories that carry a written write-up.

- Every math, code, and tool-calling run used D1 with public distilled data; every writing and health run used D1 with a locally served teacher; no run used D2 with teacher token distributions, and no run used a larger model as the sampler in D2.
- On math-contest tasks the largest lever was trace selection inside D1: keep one shortest correct trace per problem, verify the boxed answer, cap the trace length, then anneal on hard-but-concise traces. Selection alone could multiply a student's score, and cutting the trace cap helped where truncation, not ability, was the binding constraint.
- D2 with a verifier was the workhorse on math word problems for the larger students, but regressed on smaller ones; the shared cause was answer-checked-only samples that were noisy, and duplicating them collapsed one run through looping output. Deduplicate kept samples and cap their share of the mixture.
- A second self-training round regressed in every trajectory that tried one. Plan one D2 or D3 round, compare it against its parent with a matched evaluation, and stop.
- In D3, a teacher scoring candidates 1 to 10 saturated near the top of the scale and separated nothing; the same teacher ranking four shuffled candidates gave usable discrimination. DPO on those rankings beat RAFT on the same data for one writing student, while RAFT beat DPO for another whose preference margins were very small.
- Calibrate the judge against the real grader first: a reward model that agreed with the grader on most held-out pairs produced useful DPO rounds; a teacher judge that disagreed with the grader made RAFT regress.
- A larger teacher is not always better in D1: on a rubric-scored health task a small student learned more from a mid-sized teacher than from a much larger one, because the smaller teacher's conversational style matched the rubric and the larger one wrote report-style answers.
- Souping the SFT parent with its D3 child was often the best single checkpoint, above both the SFT parent and the DPO child.

## Directions

- State the flow first, then the method, then the loss. A design that says "distillation" without naming the flow has not decided anything.
- For D1, budget the teacher's generation time from a measured small batch before committing; teacher inference and training share the one GPU and alternate.
- For D2, write the rollback rule before the first round: which score, on which fixed sample, sends the run back to the parent checkpoint.
- For D3, run the judge calibration before generating pairs, and prefer listwise ranking over pointwise scores.
- Record every distillation candidate the knowledge base returned, with the reason each was chosen or rejected for this task, model, and budget.

## Do not copy when

- The task has verifiable answers and the student already scores above 90 percent: `33-decision-order.md` sends you to error-driven continuation, not to more distillation.
- The only available teacher is a hosted API: the task rules may forbid it, and its outputs cannot be regenerated on demand inside the budget.
- The student and teacher use different tokenizers: token-distribution methods (GKD, MiniLLM, DistiLLM, SKD) need a shared vocabulary or an alignment step; fall back to D1 text-level distillation.
