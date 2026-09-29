# Building a held-out legal evaluation

Read in step 1 when no public benchmark matches the legal target, or when the target benchmark has no public test split, and in step 12 when choosing between checkpoints.

A post-training run needs a number it can trust at every checkpoint. For legal work product, the public benchmarks are often too small, have no split, grade only one jurisdiction, or share text with the training data. This card is the procedure for building your own.

## 1. Fix what is being measured

Write down the task type (see the task table in `legal-dataset`), the jurisdiction and the date of the law, and the form of output (label, extracted terms, memo, drafted or marked-up document). An evaluation that mixes jurisdictions needs a jurisdiction field on every item, or its average means nothing.

## 2. Split by matter, not by item

Legal tasks come in families that share documents: scenarios of one matter, variants graded by different lawyers, several questions over one contract. Put a whole family on one side of the split.

- RedlineBench's variants `a`, `b`, `c` share the identical input [1]; LAB's scenarios share a task family and templates [2]; every MAUD `test` contract also appears in `train` (see `legal-dataset`).
- Hash the source documents across your train and test sides before trusting the split.
- Screen every training source against the evaluation with the n-gram procedure in `legal-dataset/references/finding-more.md`.

## 3. Choose the grading to match the output

| Output | Grade with | Card |
|---|---|---|
| a label, letter, yes/no, number | answer key, balanced accuracy or macro-F1 | `answer-key-scoring.md` |
| clauses, terms, articles, passages | set or span matching with precision and recall | `extraction-and-set-scoring.md` |
| an open answer with one right shape | judge with a reference answer | `reference-anchored-scoring.md` |
| work product | lawyer-written binary criteria, weighted, with negative criteria for errors | `rubric-scoring.md` |
| anything that cites authority | an existence check on every citation, support checked on a sample | `citations-and-grounding.md` |

When comparing two checkpoints rather than scoring one, pairwise preference recovered quality better than rubric scores for both attorneys and LLM graders in JudgmentBench, in less than half the annotation time [3]. Harvey's BigLaw Bench Arena collects pairwise lawyer votes and aggregates them to Elo [4].

## 4. Write the rubric like the rubric benchmarks do

- Atomic, checkable criteria, each binary: "identifies that the non-compete exceeds the statutory maximum in [state]", not "good analysis".
- Negative criteria for the errors that matter in law: invented authority, wrong jurisdiction, a missed deadline, harmful advice. PLawBench forfeits all points for harmful or illegal advice [5].
- Weights for importance, and a score normalised by positive weights, as PRBench, LexRubric and RedlineBench do; RedlineBench also clamps each task to [0, 1] (`rubric-scoring.md`).
- Written by someone legally trained. PRBench's experts agreed 93.9% that its criteria were valid [6].

## 5. Calibrate the grader before trusting it

Have a lawyer grade 30-50 outputs, stratified by task type and by model, and measure agreement with the LLM judge (Cohen's κ for binary criteria, Spearman for task scores). The published range runs from κ 0.605, about expert-expert level [6], to an LLM grader that gave 88/100 where examiners gave 37/100 [7]. Use several judges from different model families where the budget allows (`llm-judges.md`).

## 6. Report so the number can be reproduced

- The benchmark commit, the judge models and prompt, the renderer for documents, and the sampling settings.
- For rubric tasks: the criterion pass rate and the stricter task-level score together.
- Repeated runs or bootstrap intervals: legal evaluation sets are small (KoBLEX has 226 items, Magis-Bench 74), and single-run differences of a few points are noise.
- The hallucinated-citation rate beside any quality score.

## Sources

Every source was read on 2026-09-29.

[1] RedlineBench dataset card. https://huggingface.co/datasets/crosbylegal/RedlineBench

[2] Harvey LAB repository at commit `83fe609bcfc350833ff0ac33a94f21df2d748d56`. https://github.com/harveyai/harvey-labs

[3] Yang et al., "JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment". https://arxiv.org/abs/2605.25240

[4] Harvey, "Introducing BigLaw Bench: Arena" (2025-11-07). https://www.harvey.ai/blog/introducing-biglaw-bench-arena

[5] "PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice". https://arxiv.org/html/2601.16669

[6] "PRBench: Large-Scale Expert Rubrics for Evaluating High-Stakes Professional Reasoning". https://arxiv.org/html/2511.11562

[7] Study of LLMs on the Polish National Appeal Chamber qualification exam. https://arxiv.org/abs/2511.04205
