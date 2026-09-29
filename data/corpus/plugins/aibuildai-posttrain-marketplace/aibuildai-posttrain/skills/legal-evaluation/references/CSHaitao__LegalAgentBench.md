# LegalAgentBench

LegalAgentBench: 300 Chinese legal agent tasks over 17 corpora and 37 tools, from multi-hop lookups to writing tasks [1][2]. An agentic legal benchmark graded without a judge.

**Grades**: keyword keys (reference-anchored, deterministic).

**Score**: `key_answer` = fraction of gold key strings found verbatim in the final answer; `key_middle` (process rate) = fraction of answer and intermediate keys found in the trajectory; an optional BERTScore mode calls a remote service [2].

**Judge**: none.

**Access and licence**: open data; MIT per a README badge, with no LICENSE file [2]. Tools call a remote `law_api`, so running the tasks depends on that service [2].

**Use it for**: tool use over Chinese legal databases (companies, courts, statutes); the process rate rewards finding the right intermediate facts, not only the final answer.

**Trap**: exact substrings reward copying retrieved text and penalise correct paraphrase. A policy trained on this score learns to paste entities and numbers.

## Sources

Every source was read on 2026-09-29.

[1] LegalAgentBench paper. https://arxiv.org/abs/2412.17259

[2] LegalAgentBench repository at commit `ec14d8ab1fc439bfd99cbefba08d8e27dafca534`: README, `src/evaluation/eval.py`. https://github.com/CSHaitao/LegalAgentBench
