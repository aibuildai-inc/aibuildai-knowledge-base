# LexEval

LexEval: 23 Chinese legal tasks, 14,150 questions, organised by a six-level ability taxonomy from memorisation to ethics [1][2]. A broad Chinese benchmark mixing multiple choice and generation.

**Grades**: answer key (multiple choice) and lexical similarity (generation).

**Score**: accuracy or F1 on multiple choice; ROUGE-L, BERTScore and BARTScore on generation tasks [1][2].

**Judge**: none.

**Access and licence**: open, MIT; 23 task files in `data/` [2].

**Use it for**: coverage across Chinese legal abilities; the ability taxonomy is a useful checklist for a Chinese legal model.

**Trap**: generation tasks use lexical overlap, which tracks expert judgement poorly on legal text (`reference-anchored-scoring.md`); do not select checkpoints on those numbers alone.

## Sources

Every source was read on 2026-09-29.

[1] LexEval paper, NeurIPS 2024. https://arxiv.org/abs/2409.20288

[2] LexEval repository at commit `044c695f62894deef41ad9a30797e1da0404945c`: README, LICENSE. https://github.com/CSHaitao/LexEval
