# Rubric scoring: legal work product graded criterion by criterion

Read in step 1 when the legal target is work product: a memo, a draft agreement, a redline, a diligence table, an advice letter, an exam essay. Read again before turning a rubric into a training reward: `llm-judges.md` in this skill covers the rubric-reward papers, and the methodology skill holds the RL methods (`grpo.md`) and reward-model cards (`grm.md`, `rrm.md`).

A lawyer writes a list of checkable statements for each task ("identifies that the indemnity cap excludes fraud", "does not cite a repealed provision"), and a grader (usually an LLM judge, sometimes a lawyer) marks each one. The benchmarks differ on three choices, and each one changes what training should aim at: what the judge is shown, how many judges vote, and how criterion verdicts become a task score.

## The benchmarks

| Benchmark | Work product | Rubric | Judge | Task score | Read in |
|---|---|---|---|---|---|
| Harvey LAB (`harveyai/harvey-labs`) | agent writes `.docx`, `.xlsx`, `.pptx`, `.md` or `.pdf` files from a matter folder | lawyer-written binary criteria, dozens per task | one call per criterion; two judges (`claude-sonnet-4-6`, `gpt-5.5`) grade independently | **all-pass**: 1 only if every criterion passes; headline = mean of the two judges' all-pass rates, so a task can score 0.5 | `lab_core/evaluation/` [1] |
| RedlineBench (`crosbylegal/RedlineBench`) | a `.docx` with tracked changes and comments over a four-turn negotiation | attorney-written, weighted, some negative | three small judges from other model families; strict majority per criterion; all criteria in one call | `clamp((earned − penalty) / total positive, 0, 1)`; output that fails a validity gate scores 0 | `src/judging.py`, `src/panel.py` [2] |
| BigLaw Bench | transactional and litigation answers | attorney-written; positive points for requirements, negative for hallucination and extraneous statements | not stated | answer score = (positive − negative) / total positive; a separate source score | blog, sample rubrics [3][4] |
| PRBench, legal and finance (1,100 tasks, 19,356 criteria) | professional advice | expert-written; weights −10 to +10 | `o4-mini`, binary per criterion | Σ weight of met criteria / Σ positive weights, floored at 0 | paper [5] |
| PLawBench, Chinese practice (850 questions, about 12,500 items) | consultation, analysis, document drafting | practitioners; penalties, with all points forfeit for harmful or illegal advice | Gemini-3.0-Pro-Preview | Σ earned / Σ maximum | paper [6] |
| LexRubric, Chinese (649 items, 12,337 criteria) | consultation and judicial-exam answers | atomic criteria weighted −10 to +10 | Qwen3.6-27B at temperature 0 | Σ weight of met criteria / Σ positive weights | paper [7] |
| Magis-Bench, Brazilian magistrate exams | essay answers and sentence drafts | the exam boards' official rubrics, 0-10 | four frontier judges | rubric score | paper [8] |
| Vals Legal AI Report, first round | extraction, Q&A, summary, redlining, chronology, research | per-element, against a reference response | LLM judge, pass or fail per element; humans reviewed failures | accuracy per task | report [9] |

Vendor studies that report rubric scores without naming the grader (for example HAQQ's legal agent study [10]) cannot be compared with these.

## What the aggregation does

- **All-pass is sparse.** With dozens of criteria, one miss zeroes the task. Harvey's own RL run on its legal tasks moved the criterion pass rate from 0.853 to 0.913 while all-pass moved from 0.059 to 0.126 [11]. As a training reward, all-pass gives almost no gradient early; the criterion fraction, or a shaped mix, is the usual reward, with all-pass reported beside it.
- **Weighted sums with penalties reward avoiding errors as much as adding content.** Under BigLaw Bench, RedlineBench, PRBench and LexRubric, a wrong statement subtracts. Longer answers carry more chances to be penalised for "extraneous" or wrong material [4].
- **Scores from different aggregations are not comparable.** An all-pass rate of 10% and a weighted score of 90% can describe the same outputs.

## What the judge is shown, and why it matters

- LAB's judge prompt carries the task title, not the full instructions, and by default renders `.docx` with tracked changes accepted, so a criterion about what was deleted cannot be seen unless that criterion sets `include_docx_redlines` in its evaluation options [1]. Read the renderer before training a model to redline.
- RedlineBench's judge sees each criterion's weight and the attorney's justification, and grades structural edits, not comments: a comment proposing a change without making it fails [2]. A policy trained on that judge learns to edit, not to explain.
- A multi-judge average (LAB) and a majority vote (RedlineBench) produce different scores from the same verdicts. Pin the judge models and the benchmark commit; both repositories version their grading.

## Splitting and reporting

- Variants of one task share inputs. RedlineBench's variants `a`, `b`, `c` of a group have the identical input and differ only in which attorney's rubric grades them [2]; LAB's scenarios of one task share the task family and templates [1]. Split train and test by task family or input group, never by variant.
- Report the criterion fraction, the all-pass rate (or task score as defined), the judge models and the commit. Harvey chose `GPT-5 Mini` as its training grader only after checking it against a frontier-model consensus [11]; where the budget allows, also score checkpoints with a judge other than the reward judge.

## Sources

Every source was read on 2026-09-29.

[1] Harvey LAB repository at commit `83fe609bcfc350833ff0ac33a94f21df2d748d56`: `lab_core/evaluation/judge.py`, `scoring.py`, `run_eval.py`. https://github.com/harveyai/harvey-labs

[2] RedlineBench repository at commit `98340458f2e36c4f3ea59c8fdb41e1399df62f79`: `src/judging.py`, `src/panel.py`, `docs/EVALUATION.md`; dataset card https://huggingface.co/datasets/crosbylegal/RedlineBench . https://github.com/crosbylegal/redline-bench

[3] Harvey, "Introducing BigLaw Bench" (2024-08-29). https://www.harvey.ai/blog/introducing-biglaw-bench

[4] BigLaw Bench sample rubrics, `core-samples.csv`. https://github.com/harveyai/biglaw-bench

[5] "PRBench: Large-Scale Expert Rubrics for Evaluating High-Stakes Professional Reasoning". https://arxiv.org/abs/2511.11562

[6] "PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice". https://arxiv.org/abs/2601.16669

[7] "LexRubric: A Rubric-Guided Diagnostic Benchmark for Open-Ended Legal Tasks". https://arxiv.org/abs/2606.09389

[8] Magis-Bench paper. https://arxiv.org/abs/2605.08437

[9] Vals AI, Legal AI Report, first round (updated 2025-02-27). https://www.vals.ai/industry-reports/vlair-2-27-25

[10] HAQQ, legal agent study (2026-05-06). https://www.haqq.ai/blog/haqq-legal-agent-benchmark

[11] Harvey, "Training a Legal Agent With Applied Compute" (2026-06-22). https://www.harvey.ai/blog/training-a-legal-agent-with-applied-compute
