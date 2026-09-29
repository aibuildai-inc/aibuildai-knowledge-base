# Citations and grounding: checking the authority a legal answer rests on

Read in step 1 when the legal target cites cases, statutes or documents, and again before trusting any judge-based score on such a task.

A legal answer that is fluent and well organised can rest on a case that does not exist, or on a real case that does not say what the answer claims. LLM judges miss this (`llm-judges.md`), and overlap metrics reward it (`reference-anchored-scoring.md`). So citation checking is its own measurement with three levels, each harder than the last:

1. **Existence**: does the cited authority exist?
2. **Accuracy**: are the details right: reporter, volume, page, court, year, pin cite, quoted text?
3. **Support**: does the authority say what the answer uses it for?

## Tools and definitions

| Tool or study | Level | How it checks | Read in |
|---|---|---|---|
| `eyecite` (Free Law Project) | parsing, not checking | extracts full and short case citations, statutes, law journals, `supra` and `id.`, with volume, reporter, page, pin cite, year, court and parties; resolves `supra` and `id.` to their antecedents | README [1] |
| CourtListener citation lookup API | existence | parses text with eyecite and returns per citation: 200 found, 300 several matches, 400 unknown reporter, 404 valid form but not in the database, 429 over the limit; 60 valid citations a minute, 250 per request, 64,000 characters per request; token required | API documentation [2] |
| "Large Legal Fictions" (RegLab) | existence and accuracy | reference-based questions about real federal cases, graded against case metadata: citations by exact match with eyecite, authors and quotations by fuzzy match; a reference-free variant flags contradictions between two samples | paper, code [3][4] |
| "Hallucination-Free?" (RegLab) | support | lawyers read the cited sources; each answer is labelled correct, incorrect or refusal, and grounded, ungrounded (no citation) or misgrounded (the source does not support it); hallucinated = incorrect or misgrounded; κ 0.77 between raters | paper [5] |
| CLERC | accuracy against the real opinion | citation recall and precision against the citations of the real analysis paragraph, and a citation false-positive rate for citations not in the supplied sources | paper [6] |
| LegalCiteBench | existence and accuracy, closed-book | about 24,000 items from 1,000 U.S. opinions; overlap between generated and true authorities; a misleading answer rate for low-scoring answers that still give concrete citations | paper [7] |
| BigLaw Bench source score | support | share of correct statements backed by an accurate source | blog [8] |

## What the measurements show

- **Closed-book citation is mostly wrong.** Pooled hallucination rates on verifiable case questions ran from 58% for GPT-4 to 88% for Llama 2 [3]. On LegalCiteBench, 20 of 21 models had a misleading answer rate above 94% on the retrieval-heavy tasks: when they did not know the authority, they still named one [7].
- **Retrieval helps, and does not solve support.** Commercial legal research tools built on retrieval still hallucinated in 17% to 33% of answers under expert review [5].
- **Refusal is not accuracy.** Under the RegLab labels, a refusal is incomplete, not hallucinated, so a system that refuses often looks safe; report accuracy beside the hallucination rate [5].

## Using it in a legal run

- **Automate existence; sample support.** Parse every output with eyecite, look citations up (CourtListener for U.S. case law), and count 404s and unknown reporters as fabrications. Support needs a reader (a lawyer, or a judge given the cited text) on a sample.
- **Make fabrication cost something in training.** A rubric that does not mention citations lets a policy invent them freely. Add a negative criterion or a separate penalty for citations that fail the existence check, and keep abstention an allowed answer, or the policy learns that a concrete wrong citation beats "I could not find authority".
- **Respect the API limits** in any reward loop: 60 valid citations a minute will throttle RL rollouts; cache lookups, or check against a local copy of case-law metadata.
- **Jurisdiction.** Eyecite and CourtListener cover U.S. citations. Other jurisdictions need their own citation grammar and database.

## Sources

Every source was read on 2026-09-29.

[1] eyecite repository at commit `3d83354ec5f18beaebc74d1efd02a98d4d6c891a`, README. https://github.com/freelawproject/eyecite

[2] CourtListener, Citation Lookup and Verification API. https://www.courtlistener.com/help/api/rest/citation-lookup/

[3] Dahl et al., "Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models". https://arxiv.org/abs/2401.01301

[4] RegLab `legal_hallucinations` repository at commit `620734a1ccb84efc6827635ac76a2fb6504b2f64`, `correctness_checks.py`. https://github.com/reglab/legal_hallucinations

[5] Magesh et al., "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools". https://arxiv.org/abs/2405.20362

[6] CLERC paper. https://arxiv.org/abs/2406.17186

[7] "LegalCiteBench: Evaluating Citation Reliability in Legal Language Models". https://arxiv.org/abs/2605.10186

[8] Harvey, "Introducing BigLaw Bench" (2024-08-29). https://www.harvey.ai/blog/introducing-biglaw-bench
