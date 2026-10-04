# Checking synthesized legal tasks before training on them

Read after generation and before any synthesized task reaches the trainer. It is the legal form of `training` cards `10-deduplication.md` and `12-verifier-filtering.md`: a sequence of checks, cheapest first, each of which can only remove items.

No check here proves a legal answer is right. Together they remove the items that are provably wrong, unanswerable or repeated, and they tell you how much of the rest a lawyer has looked at.

## The gate

| Stage | Question | How it is answered | What published pipelines found |
|---|---|---|---|
| 1 Provenance | does the item carry jurisdiction, date, source, licence and document group? | reject on a missing field (`task-anatomy.md`) | |
| 2 Answer present | is the answer actually in the supplied text? | for spans, exact string match after normalising whitespace; for exhibits, confirm the span is not a redaction or an omitted schedule (`sec-edgar.md`) | |
| 3 Unique answer | is there exactly one right answer? | BLT drops lines with fewer than four words, lines that are substrings of others, and lines at a Levenshtein distance under four from another [1]; CaseHOLD caps distractor similarity at 0.75 [2] | |
| 4 Support | does the supplied text support the model-written answer? | a reviewer model that sees only the passage and the candidate, and may only discard | 31,777 candidates in, 23,905 out, in the German statute study [3] |
| 5 Citations exist | is every cited case real? | extract citations with eyecite, look them up against CourtListener (`courtlistener-recap.md`) | the lookup covers case law only: not statutes, journals, `id.` or `supra` [4] |
| 6 Citations say what is claimed | right page, right quote, right proposition? | compare quoted text with the opinion; read the cited passage | see below: this is where automated checking is weakest |
| 7 Duplication | how many distinct items are there? | exact and near-duplicate counts on content, not on ids (`training`: `10-deduplication.md`) | one legal preference set is 16 distinct rows repeated 6,250 times (`legal-dataset`) |
| 8 Solvable | can the task be done from what is supplied? | someone, or a strong model, who did not write it does it from scratch | independent experts found a defect in 10% of the tasks they re-did [5] |
| 9 Not trivially solvable | does the instruction give the answer away? | run a model without the documents; an item it passes is leaking | |
| 10 Contamination | is the item, or its source document, in a benchmark? | `decontamination-and-splits.md` | |
| 11 Lawyer sample | is the law right? | a law-trained reader reviews a stratified sample; record the sample size and the error rate | none of the model-written legal pipelines in `model-written-tasks.md` reports this |

## Why stage 6 needs more than a lookup

An existence check catches one kind of bad citation. LePhantomCite builds five: a citation that does not exist, a real case name on the wrong citation, a wrong pin cite, a misquote, and a real case cited for something it does not hold [6]. Only the first fails a bare lookup, and the second fails one that also compares the returned case name. On that benchmark GPT-5 in an agentic setup reached 84.4% recall at 55.0% F1 and took an average of 15.3 steps per excerpt; the highest F1 the paper reports for any system is 68.8% [6]. Checking that a citation supports its sentence is expensive and unreliable even for a frontier model.

For synthesized data this means: where an answer cites authority, prefer tasks in which the authority is supplied in the inputs, so that a quote can be string-matched against the supplied text. Answers that cite cases from memory should go to stage 11 or be dropped.

## Reading a 404

The citation lookup returns 404 for a citation that is well-formed but not in CourtListener's database [4]. The database is large, not complete. Treat a 404 as "unverified": sample them, and look a few up by hand before deciding the generator is inventing cases.

## What to report with a synthesized set

- Candidates generated and items kept, per stage.
- The reviewer model and prompt version.
- Distinct items after deduplication.
- The share of items with at least one citation, and the share of citations verified.
- The size of the lawyer-reviewed sample and what it found.
- A plain statement of what was not checked.

## What to do

1. Run the stages in order and stop an item at the first failure; record which stage removed it.
2. Do not let any stage rewrite an item. A filter that repairs answers is a second generator and needs its own check.
3. Keep the removed items. Items that failed stage 4 or 5 are ready-made negatives for training a verifier or calibrating a judge.
4. If stage 8 or stage 11 finds a pattern, fix the generator and regenerate; do not patch items one by one.

## Sources

Every source was read on 2026-10-03.

[1] Blair-Stanek, Holzenberger and Van Durme, "BLT: Can Large Language Models Handle Basic Legal Text?". https://arxiv.org/html/2311.09693

[2] Zheng et al., "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset". https://arxiv.org/html/2104.08671

[3] Bashir et al., "Domain-Adaptation through Synthetic Data: Fine-Tuning Large Language Models for German Law". https://arxiv.org/html/2601.14160

[4] CourtListener, "Citation Lookup and Verification API". https://www.courtlistener.com/help/api/rest/citation-lookup/

[5] "APEX-Agents". https://arxiv.org/html/2601.14242

[6] Liu, Stammbach and Henderson, "Who Checks the Citations? Benchmarking Legal Hallucination Detection". https://arxiv.org/html/2606.21155
