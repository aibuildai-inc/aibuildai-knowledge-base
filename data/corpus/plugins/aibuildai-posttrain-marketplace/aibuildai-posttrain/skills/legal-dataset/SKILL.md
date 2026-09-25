---
description: >-
  Data that post-trains a language model for law, and how to judge whether one
  is fit to use, with Harvey LAB (the Legal Agent Benchmark) as the target
  benchmark. 48 legal dataset cards - U.S., Australian, Canadian, EU,
  Chinese, Indian, French and Swedish, plus the LAB and RedlineBench agentic
  benchmarks - each carrying the licence as the card states it and as the
  sources' own terms state it, the exact columns and splits, a revision-pinned
  load line, a real sample row, and a screening record. Legal benchmarks are
  built from public legal datasets, and LAB itself has been copied to the Hub,
  so this skill also measures which training sets contain which benchmark's
  test items: references/contamination.md holds those measurements, their
  controls, and the script to rerun them. The body says which datasets fit
  LAB, how to carve a LAB split, a short recommended set and the
  contamination map; references/index.md lists every dataset by kind and marks
  the benchmarks; references/finding-more.md is the legal-specific search and
  screening procedure. Use it when choosing legal training data, before
  reporting LAB or any legal benchmark, and before trusting a legal dataset a
  plan already names.
---

# Legal post-training data

Legal post-training data comes in the same shapes as any other - plain text for continued pretraining, instruction-answer pairs, preference pairs, labelled task data - and the parent `dataset` skill's first rule still decides most choices: match the shape of answer the grader reads. Law adds three things that decide a choice as often.

**Benchmarks are made of training data.** LegalBench, LexGLUE, CaselawQA and LawBench are built from public legal datasets whose training splits are also on the Hub - CUAD, MAUD, ContractNLI, CaseHOLD, the CLAUDETTE terms of service, r/legaladvice, CAIL2018, the Chinese judicial exam. A legal training mix assembled from "the good legal datasets" will usually contain part of the benchmark it is scored on. This skill measured it: see the contamination map below.

**Jurisdiction and date are part of the answer.** A correct answer under Chinese criminal law, 2021 Alabama housing law, or pre-2000 Canadian case law is a wrong answer elsewhere or later. Every card names the jurisdiction and, where the source states one, the date the law was captured.

**Licences are per source, and cards understate them.** Court decisions carry the court's own terms. Several datasets here carry a permissive licence tag on the repository and non-commercial terms in every row, or state two different licences in metadata and body. Each card's **Licence** line gives both.

## Target benchmark: Harvey LAB

The maintainers set Harvey LAB (`harveyai/harvey-labs`, card `references/harveyai__harvey-labs.md`) as this skill's target. LAB is not a question-and-answer benchmark. An agent gets a folder of synthetic matter documents (median 8 files, mostly `.docx`, `.xlsx` and `.eml`), works through them with file tools, and writes a deliverable, usually a `.docx` memo, markup or term sheet. Two LLM judges grade it against a median of 54 pass/fail criteria, and the task passes only if every criterion does. Four things follow for data.

1. **What transfers is reading and writing documents, not recall.** Of 2,010 tasks, 498 are contract work, 161 M&A, 147 IP and 97 corporate governance; by work type, 488 analyze, 444 draft, 306 review. Data that trains finding exact terms in long agreements, spotting issues in them and writing structured work product transfers. Multiple-choice and label-classification data mostly does not: LAB never asks for a letter or a label.
2. **The shape is agentic, and no legal dataset here has it.** Multi-turn tool use over files and producing a `.docx` come from the parent `dataset` skill's tool-calling and agent cards (for example `THUDM/AgentInstruct`), not from this skill. Mix legal data in for content.
3. **The law is mostly U.S. and transactional.** Delaware appears in 29% of tasks, Texas 13%, the SEC or the Securities and Exchange Acts 10%, the EU or GDPR 9%. Chinese, Indian, French, Swedish, Canadian and Australian sets add little.
4. **Contamination comes from copies of LAB on the Hub, not from this skill's older datasets.** LAB went public on 2026-05-06. Of the 43 datasets carded before LAB was the target, 39 have no file committed since then, so they cannot contain it, and the ones measured hold no LAB item. The Hub copies do: a full copy of 1,242 tasks with their rubrics, GLM-5.2 trajectories on 394 tasks, and ten repositories that each hold all 250 `firm-knowledge` rubrics (contamination map below).

### Fit for LAB

| Fit | Datasets | Why |
|---|---|---|
| Train on first | `theatticusproject/cuad-qa`, `theatticusproject/maud`, `kiddothe2b/contract-nli`, `TheTokenFactory/sec-contracts-financial-extraction-instructions` (silver labels from a 2B model) | Contract review and exact-term extraction from long agreements, LAB's largest task families |
| Continued pretraining on transactional text | `chenghao/sec-material-contracts`, `theatticusproject/cuad` (raw contracts), `pile-of-law/pile-of-law` configs `edgar`, `atticus_contracts` and `state_codes`, `nvidia/Nemotron-Pretraining-Legal-v1` config `eCFR` | Real U.S. agreements and regulations: the register LAB's synthetic documents imitate |
| Supporting | `open-agreements/legal-practice-library`, `ymoslem/Law-StackExchange`, `jhu-clsp/CLERC`, `FiscalNote/billsum`, `ChicagoHAI/CaseSumm`, `common-pile/caselaw_access_project`, `coastalcph/lex_glue` LEDGAR, English U.S. files of `lawinstruct/lawinstruct` | Long-form legal writing, summarisation of long documents, U.S. doctrine for litigation and disputes tasks |
| Little transfer | Chinese, EU, Indian, French, Swedish, Canadian and Australian sets; multiple-choice and classification sets (`casehold`, `lawma-*`, `caselawqa-8k`, `barexam_qa`, `housing_qa`); the three preference sets | Wrong jurisdiction or wrong answer shape for LAB |
| Evaluation only | `harveyai/harvey-labs`, `crosbylegal/RedlineBench`, and the evaluation sets listed below | Hold out |

### Carving a LAB split

LAB has no training split. If the run trains or runs RL on LAB tasks, split by task family (the path without `/scenario-NN`), keep `firm-knowledge` whole on one side because its 250 tasks share one document store, and report only the held-out side. Scenario variants within a family are distinct matters, not paraphrases (rubric 8-gram Jaccard median 0.00 over 669 pairs, no shared document; `references/harveyai__harvey-labs.md`), so a family split is conservative rather than required. Never let an agent read `task.json`: the rubric is in it. And never train on the Hub copies of LAB in the contamination map below: they carry LAB's tasks, rubrics and model runs.

## When to consult this skill

- The designer, when the target task is legal: for LAB, start from the "Fit for LAB" table above; otherwise choose data by shape, jurisdiction and language from the table below, then read the cards.
- Anyone building a legal training mix: check the contamination map before adding a source, and drop the benchmark families it contains from any report.
- The implementer, before the load line: several legal datasets load wrongly by default - swapped config names, merged files, script-only loading, a documented column that does not exist. The card's **Trap** line names the one that applies.

## Recommended set

The strongest datasets per shape, as the cards show them. Every other dataset is in `references/index.md`.

| Dataset | Shape | Jurisdiction | Rows | Licence | The one restriction | File |
|---|---|---|---|---|---|---|
| `common-pile/caselaw_access_project` | text | U.S. | 6,919,240 declared | per-row Public Domain | the viewer copy is partial; stream the repository files | `references/common-pile__caselaw_access_project.md` |
| `pile-of-law/pile-of-law` | text | U.S. (+EU, CA) | 256GB per the paper | cc-by-nc-sa-4.0 | non-commercial; not decontaminated - its `tos` config holds 2,931 of LegalBench's 3,614 `unfair_tos` items | `references/pile-of-law__pile-of-law.md` |
| `isaacus/open-australian-legal-corpus` | text | Australia | 232,560 declared | other (per source) | most decisions are non-commercial; legislation is the permissive part | `references/isaacus__open-australian-legal-corpus.md` |
| `a2aj/canadian-case-law` | text | Canada (EN/FR) | 226,019 | mit tag, upstream terms per row | every row's `upstream_license` includes non-commercial restrictions | `references/a2aj__canadian-case-law.md` |
| `ricdomolm/lawma-tasks` | SFT, classification over full opinions | U.S. | 692,483 | mit | hold out every `test` split: CaselawQA is sampled from them | `references/ricdomolm__lawma-tasks.md` |
| `theatticusproject/cuad-qa` | SFT, contract clause extraction | U.S. contracts | 26,632 questions | cc-by-4.0 | its training contracts hold 13,674 of 18,060 LegalBench CUAD items; drop LegalBench `cuad_*` from reports | `references/theatticusproject__cuad-qa.md` |
| `coastalcph/lex_glue` | SFT, seven classification tasks | ECHR, U.S., EU | 236,714 | cc-by-4.0 | hold out all seven `test` splits | `references/coastalcph__lex_glue.md` |
| `ymoslem/Law-StackExchange` | SFT or score-ranked pairs, human Q&A | mixed | 24,370 | cc-by-sa-4.0 | bodies are HTML; share-alike | `references/ymoslem__Law-StackExchange.md` |
| `FiscalNote/billsum` | SFT, legislative summarisation | U.S. federal, California | 23,455 | cc0-1.0 | hold out `test` and `ca_test` | `references/FiscalNote__billsum.md` |
| `ShengbinYue/DISC-Law-SFT` | SFT, Chinese, many task shapes | China | 285,781 measured | apache-2.0 | strip the `exam-*` rows: they hold all of LawBench 1-2 and 3-6 and 902 of 1,000 JEC-QA-KD items | `references/ShengbinYue__DISC-Law-SFT.md` |
| `isaacus/open-australian-legal-qa` | SFT, grounded Q&A | Australia | 2,124 | other | answers written by `gpt-4` | `references/isaacus__open-australian-legal-qa.md` |
| `louisbrulenaudet/legalkit` | retrieval or grounded SFT | France | 53,000 | cc-by-4.0 | queries written by LLaMA-3-70B | `references/louisbrulenaudet__legalkit.md` |

**No legal preference dataset here is fit to recommend.** The largest by row count, `stindardlogic/legal-reasoning-dpo-100k`, is 16 distinct rows repeated 6,250 times each; `mb7419/legal-advice-reddit_preference` has no licence and sensitive personal posts; `FredrikBL/Legal-Snigel-DPO` is 1,998 Swedish summaries with no licence. Build preference pairs from `ymoslem/Law-StackExchange` answer scores, or generate and judge your own - and read the parent `dataset` skill's preference cards for general-domain pairs.

**Evaluation sets to hold out:** `harveyai/harvey-labs` (every task; the target), `crosbylegal/RedlineBench`, `nguha/legalbench`, the `test` splits of `coastalcph/lex_glue`, `ricdomolm/caselawqa-8k`, `reglab/housing_qa`, `reglab/barexam_qa` `test`, `isaacus/legal-rag-bench`, `reglab/legal_rag_hallucinations`, `doolayer/LawBench`, `hails/agieval-jec-qa-kd`, `AdaptLLM/law-tasks`, `lighteval/legal_summarization`.

## The contamination map

Measured with n-gram containment and three controls on the check date; an item counts as contained when at least half its sampled n-grams occur in the source. The method, every number, and the script are in `references/contamination.md`.

| If you train on | it contains, of the test items of | Contained |
|---|---|---|
| `lawinstruct/lawinstruct` International Citizenship files | LegalBench `international_citizenship_questions` | 9,306 / 9,306 |
| `nvidia/Nemotron-Pretraining-Legal-v1` config named `...Case-Law-Summary` | LexGLUE `case_hold` / `casehold/casehold` `all/test` | 3,600 / 3,600; 5,312 / 5,314 |
| `nvidia/Nemotron-Pretraining-Legal-v1` `GlobalCit` | LegalBench `international_citizenship_questions` | 8,895 / 9,306 |
| `theatticusproject/maud` `train` (and LawInstruct's MAUD files) | LegalBench `maud_*` | 4,530 / 4,598 |
| `theatticusproject/cuad-qa` `train` contracts | LegalBench `cuad_*` and `contract_qa` | 13,674 / 18,060 |
| `pile-of-law/pile-of-law` `tos` | LegalBench `unfair_tos` | 2,931 / 3,614 |
| `pile-of-law/pile-of-law` `r_legaladvice` | LegalBench `learned_hands_*` | 2,200 / 11,109 |
| `kiddothe2b/contract-nli` `train` (and LawInstruct's copy) | LegalBench `contract_nli_*` | 303 / 1,927 |
| `china-ai-law-challenge/cail2018` `first_stage_train` | LawBench 3-1, 3-3, 3-4 | 1,499 / 1,500 |
| `ShengbinYue/DISC-Law-SFT` `DISC-Law-SFT-Pair.jsonl` | LawBench 1-2 and 3-6 / AGIEval JEC-QA-KD | 1,000 / 1,000; 902 / 1,000 |

Two datasets also leak within themselves: 56.56% of CAIL2018 `exercise_contest_test` facts are in its own `first_stage_train`, and every MAUD `test` row's contract also appears in MAUD `train`.

**Harvey LAB, the target.** Of LAB's 2,010 tasks (rubric, instructions and documents each measured):

| If you train on | it contains, of LAB's | Contained |
|---|---|---|
| `irfanjamil/Harvey-LAB` (a copy of LAB, re-split `train` / `eval`) | rubrics; instructions | 1,242 / 2,010; 1,242 / 2,010 |
| `violetxi/harvey-eval-gpt56sol-*` (10 repositories) | `firm-knowledge` rubrics | 250 / 250 in each |
| `ShubyM/harvey-lab-glm-traces` (GLM-5.2 trajectories) | instructions; documents | 394 / 2,010; 2,659 / 48,687 |
| `violetxi/harvey-kl-ground-sessions`, `violetxi/harvey-note-conditioned-rollouts` | documents, almost all from the `firm-knowledge` store | 6,795 and 5,735 / 48,687 |
| any of this skill's original 43 datasets | anything | 0 where measured; 39 were frozen before LAB existed |

## The full list

`references/index.md` holds all 48 datasets grouped by kind, with rows, the licence field, whether the dataset may be trained on (`train`, `both`, `eval`), who wrote the text, and each card's flag. `references/index.json` holds the same list with the measured columns, splits, pinned commit and licence fields, for matching by field.

## Finding one that is not listed

`references/finding-more.md` adapts the parent skill's search procedure to law: the compound-name searches that find legal datasets, the benchmark-source list to screen any find against, and the legal traps that answer HTTP 200 with something wrong. Use it when the index has no fit - legal datasets for many jurisdictions and languages are on the Hub and not here.

## How to read a card

Cards follow the parent `dataset` skill's shape: the opening paragraph, the bolded `Use it for`, `Licence`, `Shape`, `Hold out` and `Trap` lines, the pinned load line, a real row, the sources, and the screening record. Legal cards add jurisdiction and, where the source states it, the date the law was captured. Where a number came from this skill's own measurement rather than a source, the card cites `references/contamination.md` or names the file that was downloaded and counted.

## Scope

- The Hugging Face Hub, plus the GitHub release CUAD's loading script downloads and the GitHub repository of the target benchmark, `harveyai/harvey-labs`. A dataset's language does not limit scope; the index has Chinese, French, Swedish and multilingual sets.
- Law-specific, not general: general instruction and preference data are in the parent `dataset` skill.
- A dataset whose viewer copy is `partial`, disabled, or script-only is reported with the count its card declares or a count measured by downloading files, and the card says which.
- **No signal here says whether a legal answer is correct.** Containment, duplication and licence can be measured; the legal accuracy of a model-written answer cannot, and several SFT sets here are model-written. Read rows before training on them.
- Every card claim carries a URL and the check date: 2026-09-23 for the first 43 cards, 2026-09-24 for the five added for LAB. Numbers that no source states - containment, split leakage, duplication, row counts of downloaded files - are this skill's own measurements, and the card says so.
