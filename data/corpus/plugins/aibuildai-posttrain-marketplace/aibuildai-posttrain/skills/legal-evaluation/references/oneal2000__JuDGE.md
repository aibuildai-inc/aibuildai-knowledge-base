# JuDGE

JuDGE: generate a full Chinese criminal judgment from the case's fact description, graded against the real judgment with no LLM in the loop [1][2]. A deterministic benchmark for judgment drafting, from a corpus of more than 100,000 judgments.

**Grades**: extraction and set scoring, plus lexical similarity.

**Score**: charges and cited penal-code articles are extracted from the generated judgment with regular expressions and scored by precision, recall and F1; prison term and fine by `1 − |gold − generated| / max`, zero on a sign or death-penalty mismatch; METEOR and BERTScore (`bert-base-chinese`) on the reasoning and judgment sections [2].

**Judge**: none.

**Access and licence**: open, MIT. `train.json` has 2,004 lines and `test.json` 501 (this skill's count of the repository files); the full corpus is on Google Drive [2].

**Use it for**: a verifiable reward for Chinese criminal-judgment generation: charge, article and sentence correctness can be computed without a judge.

**Trap**: the extraction patterns define what counts. Output that states the right charge in a non-court format scores zero, so training must match court phrasing.

## Sources

Every source was read on 2026-09-29.

[1] JuDGE paper, SIGIR 2025. https://arxiv.org/abs/2503.14258

[2] JuDGE repository at commit `6f2c4c888fed2f084cca75bf77523dd6ebd1fdd1`: README, LICENSE, `evaluation/calc.py`, `evaluation/calc_rel.py`, `data/`. https://github.com/oneal2000/JuDGE
