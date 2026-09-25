---
description: >-
  Data that post-trains a language model for legal work, and how to judge
  whether one is fit to use. 48 legal dataset cards - U.S., Australian,
  Canadian, EU, Chinese, Indian, French and Swedish, plus legal evaluation
  sets from short-answer benchmarks to agentic ones - each carrying the
  licence as the card states it and as the sources' own terms state it, the
  exact columns and splits, a revision-pinned load line, a real sample row,
  and a screening record. The body maps legal task types to the datasets that
  train them, names the benchmarks that are built from public training data
  and what to hold out for each, and gives a short recommended set;
  references/index.md lists every dataset by kind and marks the evaluation
  sets; references/finding-more.md is the legal-specific search and screening
  procedure. Use it when choosing legal training data, before reporting any
  legal benchmark, and before trusting a legal dataset a plan already names.
---

# Legal post-training data

Legal post-training data comes in the same shapes as any other - plain text for continued pretraining, instruction-answer pairs, preference pairs, labelled task data - and the parent `dataset` skill's first rule still decides most choices: match the shape of answer the grader reads. Law adds three things that decide a choice as often.

**Benchmarks are made of training data.** LegalBench, LexGLUE, CaselawQA and LawBench are built from public legal datasets whose training splits are also on the Hub - CUAD, MAUD, ContractNLI, CaseHOLD, the CLAUDETTE terms of service, r/legaladvice, CAIL2018, the Chinese judicial exam. A legal training mix assembled from "the good legal datasets" will usually contain part of the benchmark it is scored on. The hold-out table below names the pairs.

**Jurisdiction and date are part of the answer.** A correct answer under Chinese criminal law, 2021 Alabama housing law, or pre-2000 Canadian case law is a wrong answer elsewhere or later. Every card names the jurisdiction and, where the source states one, the date the law was captured.

**Licences are per source, and cards understate them.** Court decisions carry the court's own terms. Several datasets here carry a permissive licence tag on the repository and non-commercial terms in every row, or state two different licences in metadata and body. Each card's **Licence** line gives both.

## Legal task types and the data that trains them

Legal work is more than answering legal questions. Much of it is reading a set of documents and producing work product: finding exact terms in agreements, spotting issues, drafting or marking up a document, writing a memo. Recent benchmarks test that directly - Harvey LAB (`harveyai/harvey-labs`) gives an agent a folder of matter documents and grades the memo, markup or term sheet it writes against a lawyer-written rubric, and RedlineBench grades contract redlines over a negotiation - while older ones test short answers. Choose data by the task type the target needs:

| Task type | What the model must do | Train on | Notes |
|---|---|---|---|
| Contract review and extraction | find clauses and exact terms (parties, dates, amounts, covenants) in long agreements | `theatticusproject/cuad-qa`, `theatticusproject/maud`, `kiddothe2b/contract-nli`, `TheTokenFactory/sec-contracts-financial-extraction-instructions` (silver labels from a 2B model) | each overlaps a LegalBench family; see the hold-out table |
| Transactional language | read and write agreements in their own register | continued pretraining on `chenghao/sec-material-contracts`, `theatticusproject/cuad` raw contracts, `pile-of-law/pile-of-law` configs `edgar` and `atticus_contracts` | raw filing text; strip markup |
| Regulation and statute | know and apply the rules of a jurisdiction | `pile-of-law/pile-of-law` config `state_codes`; `nvidia/Nemotron-Pretraining-Legal-v1` config `eCFR`; `open-agreements/legal-practice-library`; `isaacus/open-australian-legal-corpus` legislation | pin a capture date |
| Case law and litigation | read opinions, find holdings, retrieve and cite precedent | `common-pile/caselaw_access_project`, `HFforLegal/case-law`, `ricdomolm/lawma-tasks`, `jhu-clsp/CLERC`, `casehold/casehold` | several double as evaluation sets |
| Summarisation of long documents | condense opinions and legislation | `FiscalNote/billsum`, `ChicagoHAI/CaseSumm`, `dennlinger/eur-lex-sum` | targets written by officials, not models |
| Legal Q&A and advice | answer a stated question, ideally from supplied law | `ymoslem/Law-StackExchange`, `isaacus/open-australian-legal-qa`, `louisbrulenaudet/legalkit`, English U.S. files of `lawinstruct/lawinstruct` | read rows first; answers vary in quality |

Two points hold across the table. First, agentic work - multi-turn tool use over files, producing a `.docx` - is a shape no legal dataset here has; it comes from the parent `dataset` skill's tool-calling and agent cards, with legal data mixed in for content. Second, multiple-choice and label-classification sets (`casehold`, `lawma-*`, `caselawqa-8k`, `barexam_qa`, `housing_qa`) train a letter or a label, which transfers poorly to work-product tasks.

## When to consult this skill

- The designer, when the target task is legal: pick the task type above, then choose data by shape, jurisdiction and language from the recommended set, then read the cards.
- Anyone building a legal training mix: check the hold-out table before adding a source, and drop the benchmark families it feeds from any report.
- The implementer, before the load line: several legal datasets load wrongly by default - swapped config names, merged files, script-only loading, a documented column that does not exist. The card's **Trap** line names the one that applies.

## Recommended set

The strongest datasets per shape, as the cards show them. Every other dataset is in `references/index.md`.

| Dataset | Shape | Jurisdiction | Rows | Licence | The one restriction | File |
|---|---|---|---|---|---|---|
| `common-pile/caselaw_access_project` | text | U.S. | 6,919,240 declared | per-row Public Domain | the viewer copy is partial; stream the repository files | `references/common-pile__caselaw_access_project.md` |
| `pile-of-law/pile-of-law` | text | U.S. (+EU, CA) | 256GB per the paper | cc-by-nc-sa-4.0 | non-commercial; not decontaminated - its `tos` config is the source of LegalBench `unfair_tos` | `references/pile-of-law__pile-of-law.md` |
| `chenghao/sec-material-contracts` | text | U.S. contracts | 1,141,632 | cc-by-sa-4.0 | raw HTML and SGML; strip markup before training | `references/chenghao__sec-material-contracts.md` |
| `isaacus/open-australian-legal-corpus` | text | Australia | 232,560 declared | other (per source) | most decisions are non-commercial; legislation is the permissive part | `references/isaacus__open-australian-legal-corpus.md` |
| `a2aj/canadian-case-law` | text | Canada (EN/FR) | 226,019 | mit tag, upstream terms per row | every row's `upstream_license` includes non-commercial restrictions | `references/a2aj__canadian-case-law.md` |
| `ricdomolm/lawma-tasks` | SFT, classification over full opinions | U.S. | 692,483 | mit | hold out every `test` split: CaselawQA is sampled from them | `references/ricdomolm__lawma-tasks.md` |
| `theatticusproject/cuad-qa` | SFT, contract clause extraction | U.S. contracts | 26,632 questions | cc-by-4.0 | LegalBench's `cuad_*` tasks come from these contracts; drop them from reports | `references/theatticusproject__cuad-qa.md` |
| `coastalcph/lex_glue` | SFT, seven classification tasks | ECHR, U.S., EU | 236,714 | cc-by-4.0 | hold out all seven `test` splits | `references/coastalcph__lex_glue.md` |
| `ymoslem/Law-StackExchange` | SFT or score-ranked pairs, human Q&A | mixed | 24,370 | cc-by-sa-4.0 | bodies are HTML; share-alike | `references/ymoslem__Law-StackExchange.md` |
| `FiscalNote/billsum` | SFT, legislative summarisation | U.S. federal, California | 23,455 | cc0-1.0 | hold out `test` and `ca_test` | `references/FiscalNote__billsum.md` |
| `ShengbinYue/DISC-Law-SFT` | SFT, Chinese, many task shapes | China | 285,781 measured | apache-2.0 | strip the `exam-*` rows: they carry LawBench and JEC-QA exam questions | `references/ShengbinYue__DISC-Law-SFT.md` |
| `isaacus/open-australian-legal-qa` | SFT, grounded Q&A | Australia | 2,124 | other | answers written by `gpt-4` | `references/isaacus__open-australian-legal-qa.md` |
| `louisbrulenaudet/legalkit` | retrieval or grounded SFT | France | 53,000 | cc-by-4.0 | queries written by LLaMA-3-70B | `references/louisbrulenaudet__legalkit.md` |

**No legal preference dataset here is fit to recommend.** The largest by row count, `stindardlogic/legal-reasoning-dpo-100k`, is 16 distinct rows repeated 6,250 times each; `mb7419/legal-advice-reddit_preference` has no licence and sensitive personal posts; `FredrikBL/Legal-Snigel-DPO` is 1,998 Swedish summaries with no licence. Build preference pairs from `ymoslem/Law-StackExchange` answer scores, or generate and judge your own - and read the parent `dataset` skill's preference cards for general-domain pairs.

## Benchmarks and what to hold out

**Evaluation sets:** `nguha/legalbench`, the `test` splits of `coastalcph/lex_glue`, `ricdomolm/caselawqa-8k`, `reglab/housing_qa`, `reglab/barexam_qa` `test`, `isaacus/legal-rag-bench`, `reglab/legal_rag_hallucinations`, `doolayer/LawBench`, `hails/agieval-jec-qa-kd`, `AdaptLLM/law-tasks`, `lighteval/legal_summarization`, and the agentic sets `harveyai/harvey-labs` and `crosbylegal/RedlineBench`.

Several of them are built from training data in this skill. If you report the benchmark family on the left, keep the sources on the right out of the training mix:

| If you report | do not train on |
|---|---|
| LegalBench `maud_*` | `theatticusproject/maud` `train`; LawInstruct's MAUD files |
| LegalBench `cuad_*`, `contract_qa` | `theatticusproject/cuad-qa`, `theatticusproject/cuad`; Nemotron config `LegalBench-CUAD-v2` |
| LegalBench `contract_nli_*` | `kiddothe2b/contract-nli` `train`; LawInstruct's ContractNLI file |
| LegalBench `international_citizenship_questions` | LawInstruct's International Citizenship files; Nemotron config `GlobalCit` |
| LegalBench `unfair_tos`, LexGLUE `unfair_tos` | Pile of Law config `tos`; LawInstruct's LexGLUE `unfair_tos` file |
| LegalBench `learned_hands_*` | Pile of Law config `r_legaladvice` |
| CaseHOLD, LexGLUE `case_hold` | Nemotron config named `Case-Law-Summary` (it holds CaseHOLD, despite the name) |
| CaselawQA | `ricdomolm/lawma-tasks` `test` splits |
| LawBench 3-1, 3-3, 3-4 | `china-ai-law-challenge/cail2018` `first_stage_train` |
| LawBench 1-2, 3-6; AGIEval JEC-QA-KD | `ShengbinYue/DISC-Law-SFT` `exam-*` rows |

Two datasets also leak within themselves: many CAIL2018 `exercise_contest_test` facts are in its own `first_stage_train`, and every MAUD `test` row's contract also appears in MAUD `train`. A benchmark published after a training set's last commit cannot be inside it; for newer sets, `references/finding-more.md` gives the screening procedure.

## The full list

`references/index.md` holds all 48 datasets grouped by kind, with rows, the licence field, whether the dataset may be trained on (`train`, `both`, `eval`), who wrote the text, and each card's flag. `references/index.json` holds the same list with the measured columns, splits, pinned commit and licence fields, for matching by field.

## Finding one that is not listed

`references/finding-more.md` adapts the parent skill's search procedure to law: the compound-name searches that find legal datasets, the benchmark-source list to screen any find against, and the legal traps that answer HTTP 200 with something wrong. Use it when the index has no fit - legal datasets for many jurisdictions and languages are on the Hub and not here.

## How to read a card

Cards follow the parent `dataset` skill's shape: the opening paragraph, the bolded `Use it for`, `Licence`, `Shape`, `Hold out` and `Trap` lines, the pinned load line, a real row, the sources, and the screening record. Legal cards add jurisdiction and, where the source states it, the date the law was captured. Where a number came from this skill's own count rather than a source, the card's source list says so and names what was downloaded and counted.

## Scope

- The Hugging Face Hub, plus the GitHub release CUAD's loading script downloads and the Harvey LAB repository on GitHub. A dataset's language does not limit scope; the index has Chinese, French, Swedish and multilingual sets.
- Law-specific, not general: general instruction, preference and agent data are in the parent `dataset` skill.
- A dataset whose viewer copy is `partial`, disabled, or script-only is reported with the count its card declares or a count measured by downloading files, and the card says which.
- **No signal here says whether a legal answer is correct.** Overlap, duplication and licence can be checked; the legal accuracy of a model-written answer cannot, and several SFT sets here are model-written. Read rows before training on them.
- Every card claim carries a URL and the check date: 2026-09-23 for 43 cards, 2026-09-24 for five added later. Numbers that no source states - split leakage, duplication, row counts of downloaded files - are this skill's own counts, and the card says so.
