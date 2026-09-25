# Every legal post-training dataset in this skill

This is the full list: 48 datasets, read at 2026-09-23; the five added for the Harvey LAB target were read at 2026-09-24. The SKILL.md body carries only the recommended set and the contamination map, so the common case never opens this file. Each row points at its card, and the card holds the licence as stated and as the sources' own terms state it, the exact columns and splits, the pinned load line, and a real sample row.

Groups are by the shape of the data. `Use` is the verdict recorded with the selection: `train` means it may be trained on, `eval` means hold it out, `both` means part of it is a benchmark split or it contains benchmark items; the card names which. `Origin` says who wrote the text: `human`, `model`, `mixed`, or `unknown`. `Flag` repeats the screening row's flag, when there is one.

**Read this before you build a training mix.** 11 of these 48 datasets are marked `eval`, including the target benchmark `harveyai/harvey-labs`: listed so they can be recognised and kept out. Another 17 are marked `both`. Several `train` and `both` datasets contain benchmark test items; `references/contamination.md` has the measurements.

## Agentic evaluation sets (2)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `harveyai/harvey-labs` (GitHub) | 2,010 tasks at commit `1dd8140` | MIT | eval | mixed | copies-on-hub | `harveyai__harvey-labs.md` |
| `crosbylegal/RedlineBench` | 140 tasks | cc-by-4.0 | eval | mixed |  | `crosbylegal__RedlineBench.md` |

## Plain text (continued pretraining) (9)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `chenghao/sec-material-contracts` | 1,141,632 | cc-by-sa-4.0 | train | human | not-deduplicated | `chenghao__sec-material-contracts.md` |
| `a2aj/canadian-case-law` | 226,019 | mit (upstream terms per row) | train | human | licence-per-source | `a2aj__canadian-case-law.md` |
| `common-pile/caselaw_access_project` | 6,919,240 declared on the card | none on the card (rows tagged Public Domain) | train | human |  | `common-pile__caselaw_access_project.md` |
| `HFforLegal/case-law` | 541,371 declared on the card | cc-by-4.0 | train | human |  | `HFforLegal__case-law.md` |
| `isaacus/open-australian-legal-corpus` | 232,560 declared on the card | other (CC BY 4.0 collection, per-source terms) | train | human | licence-per-source | `isaacus__open-australian-legal-corpus.md` |
| `joelniklaus/Multi_Legal_Pile` | viewer serves no rows; 689GB per the card | cc-by-nc-sa-4.0 (per-source licences inside) | train | mixed | script-loaded | `joelniklaus__Multi_Legal_Pile.md` |
| `open-agreements/legal-practice-library` | 119 (plus about 1,600 repository files) | cc-by-4.0 | train | unknown | unreviewed | `open-agreements__legal-practice-library.md` |
| `nvidia/Nemotron-Pretraining-Legal-v1` | 9,616,568 | cc-by-4.0 | both | mixed | contains-benchmark-items | `nvidia__Nemotron-Pretraining-Legal-v1.md` |
| `pile-of-law/pile-of-law` | viewer disabled; 256GB per the paper | cc-by-nc-sa-4.0 | train | human | not-decontaminated | `pile-of-law__pile-of-law.md` |

## Instruction and answer pairs (SFT) (11)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `Alignment-Lab-AI/Lawyer-Instruct` | 9,241 | apache-2.0 | train | model | off-topic | `Alignment-Lab-AI__Lawyer-Instruct.md` |
| `dzunggg/legal-qa-v1` | 3,742 | none on the card | train | unknown | no-licence | `dzunggg__legal-qa-v1.md` |
| `isaacus/open-australian-legal-qa` | 2,124 | other (corpus licence) | train | model |  | `isaacus__open-australian-legal-qa.md` |
| `lawinstruct/lawinstruct` | viewer serves no rows; 142 files, 10.6 GB | mit (per-source licences inside) | both | mixed | contains-benchmark-items | `lawinstruct__lawinstruct.md` |
| `louisbrulenaudet/legalkit` | 53,000 | cc-by-4.0 | train | mixed |  | `louisbrulenaudet__legalkit.md` |
| `nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions` | 4,394 | apache-2.0 | train | unknown | low-diversity | `nisaar__LLAMA2_Legal_Dataset_4.4k_Instructions.md` |
| `Prarabdha/indian-legal-supervised-fine-tuning-data` | 6,055,371 | apache-2.0 | train | mixed |  | `Prarabdha__indian-legal-supervised-fine-tuning-data.md` |
| `ricdomolm/lawma-instructions` | 554,419 | none on the card | train | human | no-licence | `ricdomolm__lawma-instructions.md` |
| `ShengbinYue/DISC-Law-SFT` | 285,781 (measured, 4 files) | apache-2.0 | both | mixed | contains-benchmark-items | `ShengbinYue__DISC-Law-SFT.md` |
| `TheTokenFactory/sec-contracts-financial-extraction-instructions` | 23,049 (7,683 examples x 3 formats) | cc-by-4.0 | train | mixed | model-written-labels | `TheTokenFactory__sec-contracts-financial-extraction-instructions.md` |
| `ymoslem/Law-StackExchange` | 24,370 | cc-by-sa-4.0 | train | human |  | `ymoslem__Law-StackExchange.md` |

## Preference pairs (3)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `FredrikBL/Legal-Snigel-DPO` | 1,998 | none on the card | both | model | no-licence | `FredrikBL__Legal-Snigel-DPO.md` |
| `mb7419/legal-advice-reddit_preference` | 70,324 | none on the card | train | human | no-licence | `mb7419__legal-advice-reddit_preference.md` |
| `stindardlogic/legal-reasoning-dpo-100k` | 100,000 (16 unique) | apache-2.0 | train | model | massive-duplication | `stindardlogic__legal-reasoning-dpo-100k.md` |

## Multiple-choice question sets (8)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `AdaptLLM/law-tasks` | 5,216 | none on the card | eval | human |  | `AdaptLLM__law-tasks.md` |
| `casehold/casehold` | 584,507 across all configs (53,137 in `all`) | none on the card | both | human | no-licence | `casehold__casehold.md` |
| `doolayer/LawBench` | 2,500 | none on the card | eval | human | contained-in-training-sets | `doolayer__LawBench.md` |
| `hails/agieval-jec-qa-kd` | 1,000 | none on the card | eval | human |  | `hails__agieval-jec-qa-kd.md` |
| `reglab/barexam_qa` | 1,195 questions (measured) | cc-by-sa-4.0 | both | human |  | `reglab__barexam_qa.md` |
| `ricdomolm/caselawqa-8k` | 22,000 | none on the card | eval | human |  | `ricdomolm__caselawqa-8k.md` |
| `ricdomolm/lawma-tasks` | 692,483 | mit | both | human |  | `ricdomolm__lawma-tasks.md` |
| `theatticusproject/maud` | 39,231 | cc-by-4.0 | both | human | overlaps-LegalBench | `theatticusproject__maud.md` |

## Classification task sets (4)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `china-ai-law-challenge/cail2018` | 2,168,025 | unknown | both | human | split-leakage | `china-ai-law-challenge__cail2018.md` |
| `coastalcph/lex_glue` | 236,714 | cc-by-4.0 | both | human |  | `coastalcph__lex_glue.md` |
| `coastalcph/multi_eurlex` | viewer serves no rows; 65,000 docs per full language | cc-by-sa-4.0 (body says cc-by-4.0) | both | human | script-loaded | `coastalcph__multi_eurlex.md` |
| `nguha/legalbench` | 91,750 | cc-by-4.0 (per-task licences) | eval | mixed |  | `nguha__legalbench.md` |

## Natural language inference (1)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `kiddothe2b/contract-nli` | 20,107 | cc-by-nc-sa-4.0 | both | human |  | `kiddothe2b__contract-nli.md` |

## Question answering and RAG sets (4)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `isaacus/legal-rag-bench` | 4,976 | cc-by-nc-sa-4.0 | eval | human |  | `isaacus__legal-rag-bench.md` |
| `reglab/housing_qa` | 6,853 (+9,297 aux) | cc-by-sa-4.0 | eval | human |  | `reglab__housing_qa.md` |
| `reglab/legal_rag_hallucinations` | 500 (400 responses + 100 questions) | cc-by-4.0 | eval | mixed |  | `reglab__legal_rag_hallucinations.md` |
| `theatticusproject/cuad-qa` | 26,632 questions (card); viewer serves no rows | cc-by-4.0 | both | human | overlaps-LegalBench | `theatticusproject__cuad-qa.md` |

## Summarisation sets (4)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `ChicagoHAI/CaseSumm` | 27,071 | cc-by-nc-3.0 (card body: NC 4.0 / NC-SA 4.0) | train | human | licence-conflict | `ChicagoHAI__CaseSumm.md` |
| `dennlinger/eur-lex-sum` | viewer serves no rows; English 1,504 | cc-by-4.0 (body says cc-by-sa-4.0) | both | human | script-loaded | `dennlinger__eur-lex-sum.md` |
| `FiscalNote/billsum` | 23,455 | cc0-1.0 | both | human |  | `FiscalNote__billsum.md` |
| `lighteval/legal_summarization` | 26,860 | none on the card | eval | human | no-licence | `lighteval__legal_summarization.md` |

## Retrieval sets (1)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `jhu-clsp/CLERC` | generation test 1,000 (measured); viewer partial | none on the card | both | human | no-licence | `jhu-clsp__CLERC.md` |

## Raw file releases (1)

| Dataset | Rows | Licence field | Use | Origin | Flag | Card |
|---|---|---|---|---|---|---|
| `theatticusproject/cuad` | 510 contracts (viewer: 84,325 lines) | cc-by-4.0 | both | human |  | `theatticusproject__cuad.md` |

## Screened out

| Dataset | Why it has no card |
|---|---|
| `nguha/legal_hallucinations_subset` | stage 0 refusal: the datasets-server answers "No (supported) data files found"; the repository holds `save_to_disk` Arrow files that `load_dataset` cannot read (a 5,444-row subset of `reglab/legal_hallucinations`, per its card) |
| `common-pile/caselaw_access_project_filtered` | covered as a neighbor on the `common-pile/caselaw_access_project` card |
| `jonathanli/law-stack-exchange` | covered as a neighbor on the `ymoslem/Law-StackExchange` card (questions only, topic labels) |
| `ricdomolm/lawma-instructions_*` variants | per-tokenizer truncations of `ricdomolm/lawma-instructions`; named on its card |
| `irfanjamil/Harvey-LAB` | a copy of Harvey LAB (1,242 tasks with rubrics and documents); never train on it if you report LAB. Measured in `contamination.md`, "Harvey LAB" |
| `ShubyM/harvey-lab-glm-traces` | GLM-5.2 trajectories on 394 LAB tasks; the same |
| `violetxi/harvey-*` (29 repositories) | evaluation sets, sessions, rollouts and notes on LAB's `firm-knowledge` tasks; ten hold all 250 of its rubrics |
| `narcolepticchicken/harvey-qwen35-isft`, `Hanno-Labs/harvey-labs-llm-artifact-analysis` | training-harness files and classifier features built on LAB outputs; small LAB fragments inside |
| `paperinstruments/diligence-bench`, `TryDotAtwo/legal-corpus-raw-batches` | found by the LAB searches; a financial-analysis benchmark, and an unscreened raw legal corpus |
| per-country statute scrapes (`endomorphosis/ipfs_*_laws`, `justicedao/ipfs_*`), smart-contract code sets, retrieval-benchmark repackagings (`mteb/*`) | found by the search, but not post-training data for legal ability, or copies of a carded source |

## How this list was built

Searched on 2026-09-23 on the Hugging Face Hub: 46 keyword searches (`legal`, `law`, `court`, `contract`, `caselaw`, `statute`, and benchmark and dataset names such as `legalbench`, `lex_glue`, `casehold`, `cuad`, `pile-of-law`, `lawinstruct`, `lawma`), up to 100 results each sorted by downloads, returned 1,984 distinct repositories; a second pass of 48 compound-name searches (`housing_qa`, `Law-StackExchange`, `legal_dpo`, `LawBench`, `CLERC`, `ContractNLI`, ...) added smaller named releases. 46 candidates were read in full - Hub API record, datasets-server size, info, splits and first rows, README, file tree, model search - and 43 earned a card. Script-only and viewer-disabled datasets were counted by downloading their files. Contamination, split leakage and duplication were then measured on downloaded data (`references/contamination.md`).

Two limits a reader must know. GitHub, Zenodo, Dataverse and the other DOI repositories were not searched for this list, apart from the CUAD GitHub release that `theatticusproject/cuad-qa` loads; `finding-more.md` has the procedure for them. And every number is a snapshot at one commit; several of these datasets are updated continuously.
