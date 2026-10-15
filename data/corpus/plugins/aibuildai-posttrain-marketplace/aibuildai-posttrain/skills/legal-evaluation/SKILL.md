---
description: >-
  How legal model output is graded, and what each way of grading rewards.
  Covers the five grading families in legal evaluation - answer keys,
  extraction and set matching, comparison with a reference answer,
  criterion-level rubrics for work product, and citation checks - with the
  extraction rules, aggregation and judges each benchmark actually uses,
  read from its code where the code is public. 15 cards on legal
  evaluations not carded in legal-dataset (LEXam, PRBench, BigLaw Bench,
  PLawBench, LexRubric, CaseGen, JuDGE, LexEval, LegalAgentBench,
  LegalBench-RAG, KoBLEX, Magis-Bench, LegalCiteBench, JudgmentBench, the
  Vals Legal AI Report), seven method cards, and an index of 33 legal
  evaluations by grading family. Also covers how far LLM judges agree with
  lawyers, rubric rewards, and building a held-out legal evaluation. Use it
  before training for any legal benchmark, when choosing a reward for legal
  work product, and when building a legal evaluation of your own.
---

# Legal evaluation

Step 1 of the `training` skill says to read the scorer before choosing any data. In legal work that step decides more than usual, for three reasons.

**The grading rule often matters more than legal knowledge.** LegalBench marks "Yes, because the clause..." wrong when the label is "Yes". LawBench marks a correct letter wrong if the explanation mentions another option letter. Harvey LAB scores a task 0 when one of its dozens of criteria fails. CUAD counts any clause returned where none exists as a false positive. A model can know the law and still score poorly under a rule it was not trained for.

**Legal judges vary widely, and they do not check authority.** Published agreement between LLM judges and lawyers ranges from expert-to-expert level (Cohen's κ 0.605 on PRBench) down to an LLM grader that gave 88/100 to an exam answer examiners scored 37/100. Judges read fluency. They do not look up whether a cited case exists. An LLM-judged legal score without a citation check measures polish.

**The right answer depends on jurisdiction, date and reference.** A benchmark grades against one jurisdiction's law at one date, and often against one reference answer. A different correct argument can lose points: LEXam's judge penalises content that is not in the reference.

## The five grading families

| Family | Graded by | Examples | What it rewards | Check first | Card |
|---|---|---|---|---|---|
| Answer key | exact label, letter or number; balanced accuracy, macro-F1 | LegalBench, LexGLUE, CaseHOLD, LawBench and LEXam multiple choice, CaselawQA, Bar Exam QA | the bare final answer in the exact form the extractor reads; reasoning earns nothing | the extraction rule, and how class imbalance is handled | `references/answer-key-scoring.md` |
| Extraction and set matching | overlap between returned and gold spans, clauses, articles or passages | CUAD, MAUD, LawBench article and sentence tasks, JuDGE, LegalBench-RAG | verbatim spans, saying nothing when nothing applies, court-style phrasing | the matching threshold and what an extra or empty answer costs | `references/extraction-and-set-scoring.md` |
| Reference-anchored | ROUGE, BERTScore, keyword keys, or a judge given a model answer | LEXam open questions, CaseGen, KoBLEX, LexEval, CLERC, LegalAgentBench | closeness to the reference answer, not legal correctness in general | whether the metric has been validated against lawyers (lexical metrics usually have not) | `references/reference-anchored-scoring.md` |
| Rubric | an LLM judge or a lawyer marks lawyer-written criteria; weighted sum or all-pass | Harvey LAB, RedlineBench, BigLaw Bench, PRBench, PLawBench, LexRubric, Magis-Bench | covering every required point without a wrong statement | what the judge is shown, how criteria are aggregated, which judges vote | `references/rubric-scoring.md` |
| Citation and grounding | existence, accuracy and support of cited authority | RegLab hallucination studies, CLERC, LegalCiteBench, BigLaw Bench source score | real, correctly cited authority that supports the claim, or an honest abstention | whether refusals count as errors | `references/citations-and-grounding.md` |

Many benchmarks combine families. LawBench mixes all of the first three. BigLaw Bench pairs a rubric with a source score. Read the scorer, not the family label.

## When to consult this skill

- **Before choosing data for a legal target.** Find the target in `references/index.md`, read its family card, and write down the extraction rule, the aggregation and the judge. The data choice follows: `legal-dataset` holds the training data and the hold-out table.
- **When choosing a reward for legal work product.** Read `references/rubric-scoring.md` for aggregation and `references/llm-judges.md` for judge reliability and the rubric-reward recipe. Also read `references/citations-and-grounding.md`, because a reward that ignores citations teaches the policy to invent them.
- **When the target has no public test split, or checkpoints must be compared.** Read `references/building-a-legal-eval.md`.
- **Before reporting a legal number.** Pin the commit and the judges, and report the criterion rate beside the task rate and the hallucinated-citation rate beside quality.

## Reading a legal scorer

Step 1 of the `training` skill applies as written. Legal scorers add these questions:

1. **What exactly is extracted from the output?** Is it the whole string, the last marked letter, any option letter anywhere, a regular expression for charges or articles, or a `.docx` rendered to text?
2. **What does the judge see?** Does it get the full instructions or only a title? Does it get a reference answer? Does it see tracked changes and comments, or only the accepted text? (LAB accepts tracked changes by default. RedlineBench renders deletions and insertions.)
3. **How do criterion verdicts become a task score?** The options are all-pass, a weighted sum normalised by positive weights, a clamp to [0, 1], or forfeiting everything on a disqualifying error.
4. **Who grades, and has that grader been checked against lawyers?** `references/llm-judges.md` lists the published agreement numbers.
5. **How are refusals and citations treated?** A refusal can score as safe (RegLab) or as a failure (all-pass rubrics). Citations may not be checked at all.
6. **Whose law, as of when?** Jurisdiction and date apply to the evaluation as much as to the training data.
7. **Which items share inputs?** Variants graded by different lawyers and scenarios of one matter must stay on one side of any split.

## What to train on and what to report

| Family | Reward or training target | Report |
|---|---|---|
| Answer key | the exact final form, as SFT targets or a verifiable reward | the benchmark metric, with and without reasoning |
| Extraction | span-level F1 or overlap; include items with no answer | precision and recall, not only F1 |
| Reference-anchored | a judge with a reference, not ROUGE | the judge score, plus a lawyer-checked sample |
| Rubric | the criterion pass fraction or a weighted sum, from a cheaper judge | the task score (all-pass or weighted), the criterion rate and the judges; where affordable, also a judge other than the reward's |
| Citation | a penalty for citations that fail an existence check; allow abstention | the hallucinated-citation rate and the accuracy rate together |

The reason for the rubric row is measured (see `references/rubric-scoring.md`). When Harvey trained a legal agent with RL, the criterion pass rate moved from 0.853 to 0.913 while all-pass moved from 0.059 to 0.126. All-pass alone gives little signal to learn from.

## Evaluations by legal task type

These are the task types from `legal-dataset`, each with evaluations from more than one jurisdiction or benchmark family, so that no single benchmark defines progress.

| Task type | Evaluations |
|---|---|
| Contract review and extraction | CUAD, LegalBench `cuad_*` and `contract_nli_*`, MAUD, LegalBench-RAG |
| Drafting and marking up documents | RedlineBench, Harvey LAB `draft` tasks, CaseGen, JuDGE, Magis-Bench sentence drafts |
| Advice and consultation | PRBench, PLawBench, LexRubric, Vals legal research |
| Case law, research and citation | CLERC, LegalCiteBench, RegLab hallucination studies, CaselawQA, CaseHOLD |
| Regulation and statute | HousingQA, KoBLEX, LegalBench `sara_*` (U.S. tax statutes) |
| Exam-style reasoning | LEXam, LegalBench, LawBench, LexEval, Bar Exam QA |
| Agentic work over document sets | Harvey LAB, RedlineBench, LegalAgentBench |

## The full list

`references/index.md` lists 33 legal evaluations grouped by grading family. For each it gives jurisdiction, metric, judge and the card that covers it. Evaluations that are also Hugging Face datasets have their cards in the `legal-dataset` skill, with load lines and hold-out notes. The others have grading cards here.

## How to read a card

A benchmark card opens with what the benchmark is. Then come bold lines:

- **Grades**: the family
- **Score**: the formula
- **Judge**: the grader, and its measured agreement with experts where published
- **Access and licence**: both as stated, including conflicts between repository and Hub
- **Use it for**
- **Trap**

A short section on how it is graded follows, then numbered sources. Method cards explain one grading family across benchmarks and end with what to do in training. A fact read in a benchmark's code names the file and the pinned commit. A count made by this skill says so.

## Scope

- **Legal only.** General evaluation and judge guidance is in the `training` skill's step 1 and its cards `02-evaluator-recon.md` and `31-per-task-map.md`. Reward methods are in `methodology`.
- **Covers how evaluations grade, not how models rank on them.** Leaderboard positions change monthly and are not recorded here.
- **Jurisdictions:** U.S., EU, ECHR, Switzerland, China, Korea, Brazil and multinational sets. Many jurisdictions have legal evaluations not listed here.
- **Only what was read.** Grading details come from each benchmark's code at a pinned commit where the code is public, and otherwise from its paper, card or report. Where a source does not say who grades (BigLaw Bench, HAQQ's study), the card says so.
- Every claim carries a source and the check date, 2026-09-29.
