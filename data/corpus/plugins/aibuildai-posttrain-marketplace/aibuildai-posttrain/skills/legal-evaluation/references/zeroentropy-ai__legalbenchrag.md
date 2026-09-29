# LegalBench-RAG

LegalBench-RAG: 6,858 query-answer pairs over a corpus of more than 79 million characters of contracts, NDAs, merger agreements and privacy policies, built from ContractNLI, CUAD, MAUD and PrivacyQA [1][2]. Retrieval only: it scores which text spans a system returns, not the answer.

**Grades**: extraction, character-level overlap.

**Score**: precision = overlapping characters / retrieved characters; recall = overlapping characters / gold characters; averaged over queries [2].

**Judge**: none.

**Access and licence**: code MIT; data on Dropbox, derived from the four source datasets, whose usage terms the README asks users to accept [2].

**Use it for**: evaluating the retrieval step of a contract-review system at the span level, separately from generation.

**Trap**: returning larger chunks raises recall and lowers precision; report both at a fixed number of retrieved chunks. Its sources are the same contracts as LegalBench's `cuad_*`, `maud_*` and `contract_nli_*` tasks and their training sets (see `legal-dataset`).

## Sources

Every source was read on 2026-09-29.

[1] LegalBench-RAG paper. https://arxiv.org/abs/2408.10343

[2] LegalBench-RAG repository at commit `431bc8f2488a81569ab7259fa633dcc50ab77f9a`: README, LICENSE, `run_benchmark.py`. https://github.com/zeroentropy-ai/legalbenchrag
