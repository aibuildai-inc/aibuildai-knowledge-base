# LegalCiteBench

LegalCiteBench: about 24,000 closed-book citation tasks built from 1,000 U.S. judicial opinions in the Case Law Access Project: citation retrieval, citation completion, citation error detection, case matching, and case verification or correction [1][2].

**Grades**: citation accuracy against the authorities the real opinion cites.

**Score**: overlap between generated and ground-truth citations, reported 0-100; a misleading answer rate for low-scoring answers that still give concrete citations [1].

**Judge**: none for the main score.

**Access and licence**: open, CC0 1.0, HF `legalcitebench/LegalCiteBench`. The card says the paper is under anonymous review [2].

**Use it for**: measuring whether a model invents U.S. case citations when it has no documents, and whether it abstains instead.

**Trap**: top models score below 7 out of 100 on retrieval and completion [1]; a small gain is noise. Its main value is the misleading answer rate: train for abstention and watch that number.

## Sources

Every source was read on 2026-09-29.

[1] "LegalCiteBench: Evaluating Citation Reliability in Legal Language Models". https://arxiv.org/abs/2605.10186

[2] Hugging Face Hub API record and card. https://huggingface.co/api/datasets/legalcitebench/LegalCiteBench ; https://huggingface.co/datasets/legalcitebench/LegalCiteBench
