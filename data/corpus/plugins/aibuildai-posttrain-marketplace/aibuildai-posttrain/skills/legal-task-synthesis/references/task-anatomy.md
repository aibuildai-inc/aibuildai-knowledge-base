# The parts of a synthesized legal task

Read first, before choosing a source or a generator. It fixes what has to exist for one training task to be usable, and how the published legal and professional task sets differ in shape, so that no single benchmark's layout becomes the template.

## Five parts

Every synthesized legal task needs all five. A task missing the last one cannot be decontaminated, filtered by jurisdiction, or split safely.

1. **Inputs**: the text or files the model works from. A clause, a provision, an opinion, or a folder of documents.
2. **Instruction**: what is asked, in the voice the target uses. Harvey LAB writes instructions "as a request for work from a partner to an associate" [1]; exam sets ask a question; extraction sets name a clause type.
3. **Output form**: a label, a span, a short answer, a memo, a marked-up document, a table. This is what the grader reads, so it is chosen from the target's scorer (the `legal-evaluation` skill), not from what is easy to generate.
4. **Answer or rubric**: an answer key, gold spans, a reference answer, or a list of binary criteria. Where it comes from decides how far it can be trusted: from the record itself (`labels-from-the-record.md`), from a model (`model-written-tasks.md`), from planted facts (`document-sets.md`), or from a lawyer.
5. **Provenance record**: jurisdiction; date of the law or the document; source URL or identifier; source licence as stated; the generator model and prompt version if a model wrote any part; the document group the task belongs to, for splitting; and a flag saying whether the answer was checked and how.

## Shapes, from short to long

| Shape | Inputs | Output | Answer comes from | Graded as (`legal-evaluation`) |
|---|---|---|---|---|
| Label or short answer | a clause, a passage, a fact pattern | label, letter, yes or no | the record, or an expert | answer key |
| Extraction | one long document | spans, terms, a filled schema | expert annotation, or the document's own structure | set and span matching |
| Grounded question | a provision or passage, plus a question | an answer that cites the passage | a model, checked against the passage | reference-anchored |
| Single work product | one or a few documents | a memo, summary, clause or draft | a reference text or a rubric | reference-anchored or rubric |
| Work over a document set | a folder of files, with tools | one or more files | a rubric, often with planted facts | rubric |

## How large the published long-form sets are

These numbers show the range. They are not a specification to copy.

| Set | Unit | Inputs | Criteria per task | Source |
|---|---|---|---|---|
| Harvey LAB | "more than 1,200 agent tasks across 24 legal practice areas" | a client matter whose files "mix key and peripheral files and embed issues across multiple documents"; the 90th-percentile task has nearly 100,000 tokens of source documents and the largest over 200,000 | "over 75,000 expert-written rubric criteria" in total | [1][2] |
| APEX-Agents | 480 tasks in 33 worlds, 12 of them in law | "On average, each world contains 166 files" | "between 1 and 10 criteria per task, with a mean of 4.06" | [3] |
| GDPval | "1,320 tasks across 44 occupations", lawyers among them | up to 17 reference files in the gold subset and 38 in the full set | graded by blinded expert pairwise comparison, not a rubric | [4] |
| PRBench, legal and finance | 1,100 tasks | a professional question, no files | 10 to 30 per task, 19,356 in total | [5] |
| RedlineBench | 140 tasks over 3 scenarios and 4 turns | a contract as `.docx`, with a playbook | attorney-written weighted rubrics | `legal-dataset` card |

Two things follow. A rubric can have four criteria or sixty, and file sets run from none to hundreds, so the shape should come from the work being trained, not from one benchmark. And long-form sets are small because experts wrote them; synthesis is how a training set of that shape gets large enough to train on.

## What to do

1. Read the target's scorer first (`training` step 1, `legal-evaluation`). Write down the output form and the grading family.
2. Pick the shape row above that matches, then the answer source you can afford to trust.
3. Define the provenance record before generating anything, and reject any item that lacks a field.
4. Vary instruction voice, document count, length and criteria count across the set. A set where every task has the same layout trains the layout.
5. Never take a benchmark's tasks, instructions or rubrics as seeds or few-shot examples for generation (`decontamination-and-splits.md`).

## Sources

Every source was read on 2026-10-03.

[1] Harvey, "Open-Sourcing Harvey's Long Horizon Legal Agent Benchmark". https://www.harvey.ai/blog/introducing-harveys-legal-agent-benchmark

[2] Applied Compute, "Training a State-of-the-Art Legal Agent with Harvey" (source-document token counts). https://www.appliedcompute.com/case-studies/harvey

[3] "APEX-Agents". https://arxiv.org/html/2601.14242

[4] "GDPval". https://arxiv.org/html/2510.04374

[5] "PRBench: Large-Scale Expert Rubrics for Evaluating High-Stakes Professional Reasoning". https://arxiv.org/html/2511.11562
