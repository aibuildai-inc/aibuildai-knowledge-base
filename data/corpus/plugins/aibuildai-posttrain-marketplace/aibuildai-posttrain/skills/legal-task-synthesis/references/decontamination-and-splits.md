# Decontamination and splits for synthesized legal tasks

Read before choosing which documents to synthesize from, and again before any benchmark number is reported. `training` card `11-decontamination.md` gives the general rule and `legal-dataset` gives the hold-out table for existing datasets. This card covers the case those do not: tasks you generate yourself.

A synthesized task is new text, so an n-gram screen of its question against a benchmark's questions will usually come back clean. The leak is one level down. Legal benchmarks are built from a small number of public corpora, and those are the same corpora a synthesis pipeline reaches for. Writing a new question about a document that is in a test set teaches the model that document.

## Which source recreates which benchmark

| If you synthesize from | you can recreate | because |
|---|---|---|
| EDGAR commercial contracts (`sec-edgar.md`) | CUAD; LegalBench `cuad_*` and `contract_qa`; LegalBench-RAG | CUAD's 510 contracts were filed on EDGAR; LegalBench-RAG is built from CUAD, MAUD, ContractNLI and PrivacyQA [1] |
| public-company merger agreements | MAUD; LegalBench `maud_*`; LegalBench-RAG | MAUD annotates merger agreements (`legal-dataset`) |
| non-disclosure agreements | ContractNLI; LegalBench `contract_nli_*`; LegalBench-RAG | ContractNLI is built from NDAs collected from public sources (`legal-dataset`) |
| U.S. opinions, holding parentheticals | CaseHOLD; LexGLUE `case_hold` | CaseHOLD extracts parentheticals that begin with "holding" from Harvard's corpus, 1965 on [2] |
| U.S. opinions, masked citations or continued analysis | CLERC; LePaRD | both are built from citations in context in U.S. opinions [3][4] |
| Supreme Court and appeals court opinions, coded outcomes | Lawma `test` splits; CaselawQA | both use the Supreme Court Database and the Songer database (`legal-dataset`) |
| U.S. Congressional and California bills with summaries | BillSum `test` and `ca_test` | same documents, same official summaries (`legal-dataset`) |
| EU legal acts with summaries or EuroVoc labels | EUR-Lex-Sum; MultiEURLEX | same acts (`legal-dataset`) |
| U.S. state housing statutes | HousingQA | its statutes are 2021 state housing law (`legal-dataset`) |
| Chinese criminal judgments | CAIL2018; LawBench 3-1, 3-3, 3-4 | LawBench draws on CAIL2018 (`legal-dataset`) |
| terms of service; privacy policies | LegalBench and LexGLUE `unfair_tos`; LegalBench `opp115_*`, `privacy_policy_*` | CLAUDETTE and OPP-115 are collections of those documents (`legal-dataset`: `finding-more.md`) |

Benchmarks whose documents were written for them (Harvey LAB, RedlineBench, APEX-Agents) cannot be recreated from a public corpus. They leak another way: by being used as seeds, examples or templates for generation.

## Rules

1. **Exclude by document, not by item.** Before generating, remove from the source pool every document that appears in a benchmark you will report. Match on a stable identifier where there is one (EDGAR accession number, case citation, CELEX number, bill number) and on a text hash where there is not.
2. **Use the date.** `legal-dataset` uses the rule that a training file committed before a benchmark's release cannot contain it. The mirror rule holds for sources: a contract filed, or an opinion issued, after a benchmark was released cannot be one of its documents. Recent documents are the simplest clean pool.
3. **Never seed from a benchmark.** Do not use a benchmark's tasks, instructions, rubrics, documents or model trajectories on them as seeds, few-shot examples or style references for a generator. If a held-out number will be claimed, the generator should not have seen the benchmark at all. Public write-ups that do train on a benchmark's own tasks report on a separate held-out split of it: Applied Compute's results on Harvey LAB are "on a held-out test split" of 180 items [5].
4. **Do not copy one benchmark's layout.** Matching a benchmark's task schema, practice-area list and rubric phrasing is fitting to its form even when no item is shared. Vary shape across the set (`task-anatomy.md`) and evaluate on more than one benchmark family (`legal-evaluation`: evaluations by task type).
5. **Split by document group.** Every task generated from one contract, one docket, one matter folder or one statute section goes on one side of the split. Amendments and restatements of an agreement, and the opinions in one case's appeal history, belong to the same group.
6. **Screen the finished set anyway.** Run the n-gram containment check in `legal-dataset`: `finding-more.md`, with its three controls, against every benchmark that shares the set's jurisdiction and language. Screen the source passages kept on each row, not only the questions.
7. **Watch for the model's memory.** A model may have read the source during pretraining. BLT renumbers transcript lines and uses invented terms so that answers must come from the supplied text, and its amended-statute task tests exactly that [6]. Changing names, amounts and dates in a real form does the same for contracts.

## Reporting

Report the benchmarks screened, the exclusion rule used for each, and the date range of the source pool, alongside the scores. A reader needs the rule that was applied in order to trust a held-out claim.

## Sources

Every source was read on 2026-10-03.

[1] "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain". https://arxiv.org/html/2408.10343

[2] Zheng et al., "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset". https://arxiv.org/html/2104.08671

[3] "CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation". https://arxiv.org/html/2406.17186

[4] Mahari, Stammbach, Ash and Pentland, "LePaRD: A Large-Scale Dataset of Judges Citing Precedents". https://arxiv.org/abs/2311.09356

[5] Applied Compute, "Training a State-of-the-Art Legal Agent with Harvey". https://www.appliedcompute.com/case-studies/harvey

[6] Blair-Stanek, Holzenberger and Van Durme, "BLT: Can Large Language Models Handle Basic Legal Text?". https://arxiv.org/html/2311.09693
