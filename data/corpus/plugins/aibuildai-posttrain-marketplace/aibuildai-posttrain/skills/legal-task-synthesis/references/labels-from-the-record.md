# Answers taken from the record: tasks no model has to answer

Read when choosing how the answer for a synthesized task will be obtained. This is the first option to consider, because the answer is written by a court, a legislature, an official summariser or an expert annotator, and a model only reformats it.

Legal records label themselves. A later opinion says what an earlier case held. A judgment states the charge and the sentence. A bill arrives with an official summary. A coded database records how a case was decided. Each of these has been turned into a task set, and the recipes generalise to new documents from the same sources.

## Published recipes

| Recipe | Source text | How the answer is obtained | Reported in |
|---|---|---|---|
| Holding from a citing parenthetical | Harvard's case law corpus, 1965 on | "extract the holding statement from citations (parenthetical text that begins with 'holding')"; the prompt is the text before it; wrong choices are the most TF-IDF-similar other holdings, under a similarity ceiling of 0.75 "to rule out indistinguishable holding statements" | CaseHOLD, 53,137 items [1] |
| Masked citation | 1.84 million federal cases from the Caselaw Access Project | take "a window of 300 words centered on this citation" and remove the central citation, or all citations; the cited case is the answer | CLERC retrieval [2] |
| Continue the analysis | the same corpus | sample reference paragraphs from "the last 33.3% of all paragraphs"; give the model "all texts preceding the reference" plus the most salient passages of the cases that paragraph cites; the real paragraph is the reference | CLERC generation, 6k passages [2] |
| Passage a later court relied on | U.S. federal opinions | "judicial citations to precedent in context": predict the precedent passage from the argument around it | LePaRD [3] |
| Questions from case metadata | 5,000 cases from each level of the federal judiciary | join each case to tabular metadata (Caselaw Access Project, Supreme Court Database, appeals courts database, Shepard's) and ask for a field: existence, court, citation, author, disposition, quotation, authority, overruling year, doctrinal agreement | "Large Legal Fictions" [4] |
| False-premise questions | the same metadata | ask about a dissent the judge did not write, or the year a never-overruled case was overruled; the correct answer rejects the premise | "Large Legal Fictions" [4] |
| Labels coded by researchers | Supreme Court Database, Songer Court of Appeals Database | the database codebook's answer for each opinion | Lawma, 260 tasks (`legal-dataset`) |
| Exact lookup | 33,176 lines of real deposition transcripts; U.S. Code sections; generated sections | "What is the exact text of just line __ of page __ above?" and the reverse; where a term is defined; answer computed by the script that built the prompt | BLT, 11 tasks, 1,000 train and 100 test prompts each [5] |
| Official summary | U.S. Congressional and California bills; EU legal acts | the summary published with the document | BillSum, EUR-Lex-Sum (`legal-dataset`) |
| The judgment's own fields | Chinese criminal judgments | charge, cited articles and prison term stated in the judgment | CAIL2018 (`legal-dataset`) |
| Expert labels traced to spans | CUAD, MAUD, ContractNLI, PrivacyQA | trace each annotated segment "back to its original location within the source corpus" as character indices; turn the label category into a question | LegalBench-RAG, 6,889 pairs over 714 documents (the abstract says 6,858) [6] |

One more recipe follows from a source's features, though no benchmark read for this skill uses it: two dated versions of one regulation, with the published comparison as the gold list of changes (`us-federal-law.md`).

## What the recipes have in common

- **The answer exists before any model runs.** Generation only writes the question and formats the item.
- **A uniqueness filter is part of the recipe.** BLT keeps a line only if "there is only one clearly correct answer": it drops lines with fewer than four words, lines contained in other lines, and lines at a Levenshtein distance under four from another [5]. CaseHOLD caps how similar a distractor may be to the answer [1].
- **Programmatic tasks can be made unmemorisable.** BLT strips the page and line numbers from transcripts and assigns new ones "in case the LLM saw the original numbering during training", and builds sections whose defined terms "are nonces that are not real words but are pronounceable", because "LLMs have doubtless seen all of the U.S. Code". Its `cite→amended` task changes statutory text to test "applying a given newly-amended statute, rather than its old version" [5].
- **Simple tasks still train.** Fine-tuning GPT-3.5-turbo on BLT's smallest training set raised transcript line retrieval from 32% to 99%, and accuracy on the SARA statutory-reasoning set from 54.3% to 60.9% [5].

## Limits

- The shapes are narrow: labels, spans, lookups, continuations. They do not produce a memo or a marked-up agreement. Use them for the reading and citing skills that work product depends on, and build work-product tasks separately (`document-sets.md`, `writing-rubrics.md`).
- Record answers reward the record's phrasing. A continuation scored against the real paragraph rewards writing like that court (`legal-evaluation`: `reference-anchored-scoring.md`).
- Almost every recipe above is already a benchmark. Re-running it on the same corpus recreates test items. `decontamination-and-splits.md` lists the pairs.

## What to do

1. For each skill the target needs, ask whether the record already holds the answer. If it does, use a recipe here before paying for a model-written answer.
2. Apply the recipe to documents outside the benchmark's corpus or after its date.
3. Add false-premise items. A model trained only on answerable questions learns to answer everything.
4. Keep a uniqueness filter and report how many candidates it removed.

## Sources

Every source was read on 2026-10-03.

[1] Zheng et al., "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset". https://arxiv.org/html/2104.08671

[2] "CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation". https://arxiv.org/html/2406.17186

[3] Mahari, Stammbach, Ash and Pentland, "LePaRD: A Large-Scale Dataset of Judges Citing Precedents". https://arxiv.org/abs/2311.09356

[4] Dahl, Magesh, Suzgun and Ho, "Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models". https://arxiv.org/html/2401.01301

[5] Blair-Stanek, Holzenberger and Van Durme, "BLT: Can Large Language Models Handle Basic Legal Text?". https://arxiv.org/html/2311.09693

[6] "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain". https://arxiv.org/html/2408.10343
