# JudgmentBench

JudgmentBench: 30 legal tasks (16 transactional, 14 litigation), 2,274 model outputs, and practising attorneys (53 recruited, 51 in the analyses) who gave 1,539 rubric scores and 1,530 pairwise preference judgments [1][2]. Not a model benchmark: a test of grading methods, with outputs of known relative quality.

**Grades**: compares rubric scoring with pairwise preference, for attorneys and for LLM autograders.

**Score**: how well each grading method recovers the constructed quality ordering: mean Spearman 0.908 for pairwise judgment against 0.150 for rubrics, and pairwise win rate 0.669 against 0.542; pairwise took less than half the annotation time [1].

**Judge**: attorneys; LLM autograders showed the same pattern, and their pairwise orderings agreed with the attorneys' better than their rubric orderings did [1].

**Access and licence**: open, MIT, HF `judgmentbench/JudgmentBench` (85,907 rows) [2].

**Use it for**: choosing how to grade a new legal evaluation, and calibrating an LLM grader against attorney judgments on the same outputs.

**Trap**: quality differences were induced by prompting, so they may be larger or more systematic than differences between two checkpoints of one model.

## Sources

Every source was read on 2026-09-29.

[1] Yang et al., "JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment". https://arxiv.org/abs/2605.25240

[2] Hugging Face Hub API record and card. https://huggingface.co/api/datasets/judgmentbench/JudgmentBench
