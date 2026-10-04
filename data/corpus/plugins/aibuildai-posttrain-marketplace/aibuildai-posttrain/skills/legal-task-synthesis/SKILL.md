---
description: >-
  How to build legal training tasks from public documents when no dataset
  has the shape the target needs. Covers where the documents come from
  (SEC EDGAR, CourtListener and RECAP, U.S. federal statutes and
  regulations, EUR-Lex, UK legislation and case law, Chinese judgments,
  and Australian, Canadian, Indian and ECHR sources), each with its access
  rules and reuse terms as stated; the three ways to get an answer (taken
  from the record, written by a model, or fixed by building the
  documents); how to write and test a rubric; the checks to run before
  training, including citation lookup; and decontamination when the
  source corpus is the same one benchmarks are built from. 7 source cards,
  8 method cards, and an index of 15 sources and 31 published synthesis
  recipes. Use it when legal-dataset has no data of the needed shape, when
  planning synthetic or self-generated legal data, when building an RL
  environment for legal work product, and before training on any legal
  data a model wrote.
---

# Legal task synthesis

The `legal-dataset` skill ends its task table with a gap: agentic work over files "is a shape no legal dataset here has", and multiple-choice and classification sets transfer poorly to work product. The `legal-evaluation` skill shows that the benchmarks which matter most for legal practice grade memos, markups and extracted terms against criteria. Between the two sits the data that has to be made. This skill is how to make it.

Law changes three things about synthetic data.

**The raw material is public and enormous, and so is the overlap with benchmarks.** Real contracts, opinions, statutes and filings are free to read in bulk. They are also what CUAD, MAUD, CaseHOLD, CLERC, BillSum, EUR-Lex-Sum and CAIL2018 were built from. A new question about a document that sits in a test set is still a leak. Choosing source documents is the first decontamination step, not the last.

**A model-written legal answer has no built-in check.** Code can be run and arithmetic can be verified. A fluent paragraph of legal analysis can cite a case that does not exist, and nothing fails. Every recipe that worked ties the answer to something outside the generator: the record itself, a supplied passage, or facts planted in the documents.

**Public does not mean free to mine.** Sources differ on automated access and on reuse, sometimes within one country: UK legislation is under an open licence, while UK judgments exclude computational analysis without a further licence. Court records name private people. Each source card gives the terms in the source's own words.

## The pipeline

| Step | Decision | Card |
|---|---|---|
| 1 | What must one task contain, and what shape does the target's grader need? | `references/task-anatomy.md`; the `legal-evaluation` skill for the grader |
| 2 | Which documents, from where, under what terms? | the source cards; `references/index.md` |
| 3 | Which of those documents are inside a benchmark? Remove them now. | `references/decontamination-and-splits.md` |
| 4 | Where does the answer come from? | the table below |
| 5 | If the output is work product: write the rubric and test it. | `references/writing-rubrics.md` |
| 6 | Filter: provenance, answer present, support, citations, duplicates, solvable. | `references/checking-synthesized-tasks.md` |
| 7 | Replace private people's names; record licence and generator per item. | `references/personal-data-and-terms.md` |
| 8 | Split by document group, screen the finished set, report what was not checked. | `references/decontamination-and-splits.md` |

## Where the answer comes from

This is the decision that sets the quality of a synthesized set. Take the first row that fits.

| Answer source | How it works | Trust | Shapes it reaches | Card |
|---|---|---|---|---|
| The record | a later court states the holding, the judgment states the sentence, the bill has an official summary, a script knows which line it picked | high: no model wrote the answer | labels, spans, lookups, continuations, summaries | `references/labels-from-the-record.md` |
| Built documents | write or alter the documents from a facts ledger, plant the issues, and derive the rubric from the ledger | high for the planted facts; the documents may read as too clean | work product over a document set | `references/document-sets.md` |
| A model, grounded | a model writes question and answer from a stored passage; a second pass that sees only the passage discards unsupported items | medium: support is checked, the law is not | grounded Q&A, extraction, scenario application | `references/model-written-tasks.md` |
| A model, ungrounded | a model answers from memory | low: nothing checks it | avoid, or send every item to a law-trained reviewer | `references/model-written-tasks.md` |

A measured example of why the row matters: in one study of German statute questions, graded generation with a discard-only reviewer raised a 12B model from 36.8% to 76.4% on the authors' test, while single-prompt unfiltered generation from the same statutes gave mixed results and lowered one model's open-ended score by 7.6 points (`references/model-written-tasks.md`).

## Long tasks over document sets

Legal work is mostly reading a set of documents and producing something from them, and public benchmarks now test that directly: Harvey LAB hands an agent a client matter and grades the work product against expert criteria, RedlineBench grades contract markups over a negotiation, and APEX-Agents and GDPval include legal work among other professions. Their task sets were written by experts and are small. `references/task-anatomy.md` compares their shapes, and `references/document-sets.md` and `references/writing-rubrics.md` cover building training tasks of that kind from public or written documents.

The target is the kind of work, not any one benchmark. Vary practice area, jurisdiction, document count, length, instruction voice and rubric size, and never use a benchmark's tasks or rubrics as seeds or templates. A set that copies one benchmark's layout fits its form even when no item is shared.

## When to consult this skill

- **The designer, when `legal-dataset` has nothing of the needed shape.** Read `references/task-anatomy.md`, pick the answer source above, then the source cards for the jurisdiction.
- **Anyone planning self-generated or teacher-generated legal data.** Read `references/model-written-tasks.md` and `references/checking-synthesized-tasks.md` before the first generation run. `training` card `34-distillation.md` covers the training side.
- **Anyone building an RL environment for legal work product.** Read `references/document-sets.md` and `references/writing-rubrics.md`; choose the reward's aggregation and judge from `legal-evaluation`.
- **Before training on a legal dataset that a model wrote**, whoever made it. The checks in `references/checking-synthesized-tasks.md` apply to found data as much as to your own.
- **Before reporting.** `references/decontamination-and-splits.md` lists which public corpus recreates which benchmark.

## The full list

`references/index.md` lists the 15 public sources by jurisdiction with their terms and the benchmarks built from each, and the 31 published recipes grouped by where the answer comes from, with how much of each source this skill read.

## How to read a card

A source card opens with what the source is. Then come bold lines:

- **Holds**: the documents
- **Access**: how to get them in bulk, and the stated limits
- **Terms**: copyright and reuse, in the source's own words
- **Use it for**
- **Feeds these benchmarks**: what a set built from this source can recreate
- **Trap**

A method card explains one step across the published examples. Both kinds end with **What to do** and numbered sources. A quotation is the source's wording on the check date. Where a page could not be read, the card says so and does not state what the page says.

## Scope

- **Legal only.** General synthetic-data, distillation and filtering guidance is in the `training` skill (`12-verifier-filtering.md`, `34-distillation.md`) and the `methodology` cards. The four general-domain generation methods in `references/model-written-tasks.md` are there because they start from raw documents.
- **How to build tasks, not a dataset.** This skill ships no generated data and no generator prompts.
- **Jurisdictions:** U.S., EU, UK, China, Australia, Canada, India and the ECHR. Many others publish their law openly and are not covered.
- **Not legal advice on reuse.** Terms are quoted as the sources state them on the check date. Whether a use is permitted is for the project to decide.
- **No signal here says a synthesized legal answer is correct.** The checks remove items that are unsupported, unverifiable or repeated. Only review by someone legally trained addresses whether the law is right, and none of the model-written legal pipelines read for this skill reports it.
- **Only what was read.** Each entry in `references/index.md` records whether a paper, an abstract, a post or a dataset card was read. Pages that refused automated readers are named in the cards as not read: CourtListener's terms, CanLII's terms, the Chinese national law database, EUR-Lex's legal notice, and the U.S. Code download page.
- Every claim carries a source and the check date, 2026-10-03.
