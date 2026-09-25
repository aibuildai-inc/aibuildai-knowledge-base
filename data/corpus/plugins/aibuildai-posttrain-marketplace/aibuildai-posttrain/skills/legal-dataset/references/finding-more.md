# Finding a legal dataset that is not on the list

This file teaches an action, because the list is a snapshot taken on 2026-09-23 and legal datasets appear, change and disappear. It extends the parent `dataset` skill's `finding-more.md`, which holds the general search order, rate limits and quality gate; read that first. What follows is what law changes.

## Search by benchmark and source name, not by "legal"

The generic words find a minority. Measured on this skill's list: the Hub searches `?search=legal` and `?search=law`, 100 results each sorted by downloads, found 16 of the 43 datasets that earned a card. The other 27 were found only by a name: `casehold`, `lex_glue`, `cuad`, `maud`, `contractnli`, `billsum`, `eurlex`, `caselaw`, `lawma`, `lawbench`, `barexam`, `housing_qa`, the author `reglab`. Two reasons: the search tokenizes repository ids, so `casehold/casehold` and `coastalcph/lex_glue` contain no token `legal`; and a 100-result cap sorted by downloads cuts off small releases such as `dzunggg/legal-qa-v1` whose ids do contain it.

So search in this order:

1. **The benchmark and source names** in the table below - every well-known legal dataset is named after its source.
2. **The authors** who publish legal data: `reglab`, `nguha`, `pile-of-law`, `joelniklaus`, `lawinstruct`, `coastalcph`, `theatticusproject`, `isaacus`, `ricdomolm`, `HFforLegal`, `a2aj`, `ShengbinYue`.
3. **Compound names with separators**, because the id tokenizer splits on them: `legal-qa`, `legal_qa`, `legal-instruct`, `legal-dpo`, `legal-reasoning`, `lawyer`, `case_law`.
4. **Full text** (`/api/search/full-text?q=<phrase>&type=dataset`) for absence claims, exactly as the parent file says.

## The benchmark-source list to screen every find against

A legal dataset found by search must be screened against the benchmarks it could contain before it goes in a mix. These are the public sources that legal benchmarks are built from; a find whose card names one of them, or whose text matches, carries benchmark items.

| Source | Feeds these evaluation sets | Training copies measured in this skill |
|---|---|---|
| CUAD contracts | LegalBench `cuad_*`, `contract_qa` | `theatticusproject/cuad-qa`, `theatticusproject/cuad`, Nemotron `LegalBench-CUAD-v2` |
| MAUD merger agreements | LegalBench `maud_*` | `theatticusproject/maud`, LawInstruct `MAUD-*` |
| ContractNLI NDAs | LegalBench `contract_nli_*` | `kiddothe2b/contract-nli`, LawInstruct `ContractNLI` |
| CaseHOLD | LexGLUE `case_hold`, `casehold/casehold` test, `AdaptLLM/law-tasks` | Nemotron config named `Case-Law-Summary`, LawInstruct `LexGLUE-case_hold` |
| CLAUDETTE terms of service | LexGLUE `unfair_tos`, LegalBench `unfair_tos` | Pile of Law `tos`, LawInstruct `LexGLUE-unfair_tos` |
| r/legaladvice (LearnedHands) | LegalBench `learned_hands_*` | Pile of Law `r_legaladvice`, `mb7419/legal-advice-reddit_preference` (not measured) |
| GLOBALCIT citizenship law | LegalBench `international_citizenship_questions` | LawInstruct `InternationalCitizenshipLawQuestions`, Nemotron `GlobalCit` |
| OPP-115 / PrivacyQA | LegalBench `opp115_*`, `privacy_policy_*` | LawInstruct `PrivacyQA` |
| Supreme Court Database, Songer database | CaselawQA | `ricdomolm/lawma-tasks` `test`, and any SCDB-coded training set |
| CAIL2018 | LawBench 3-1, 3-3, 3-4 | `china-ai-law-challenge/cail2018`, DISC-Law-SFT judgment rows |
| Chinese judicial exam (JEC-QA) | LawBench 1-2, 3-6; AGIEval JEC-QA | `ShengbinYue/DISC-Law-SFT` `exam-*` rows |
| Harvey LAB tasks, rubrics and synthetic documents (GitHub, public since 2026-05-06) | Harvey LAB, the target | `irfanjamil/Harvey-LAB`, `ShubyM/harvey-lab-glm-traces`, the `violetxi/harvey-*` repositories (`firm-knowledge` rubrics, sessions and rollouts), `narcolepticchicken/harvey-qwen35-isft`, `Hanno-Labs/harvey-labs-llm-artifact-analysis` (documents only) |

For LAB, date comes first: a repository whose files were all last committed before 2026-05-06 cannot contain it, which the tree API answers in one call (`/api/datasets/<id>/tree/<sha>?recursive=true&expand=true`, field `lastCommit.date`). Everything newer that mentions Harvey, LAB, or legal agent traces gets the scan in `contamination.md`, section "Harvey LAB", with `irfanjamil/Harvey-LAB` as the positive control.

To screen a find, run the containment check in `contamination.md` with the relevant evaluation set, **with its positive control**. A zero is evidence only when a source known to contain the items scores above zero in the same run.

## Traps that return a wrong answer without failing

Each of these answered HTTP 200 with something believable while this list was built.

1. **The card documents a column the files do not have.** LawInstruct's card documents one `text` field; its files have `instruction`, `prompt` and `answer`. Reading `text` returns empty strings, and a contamination check over empty strings reports zero overlap with every benchmark. Read one real row before writing any reader.
2. **Config names that are swapped.** In `nvidia/Nemotron-Pretraining-Legal-v1`, the config named `Case-Law-Summary` holds the CaseHOLD questions and the config named `CaseHOLD` holds case-law summaries, and the per-row `metadata.category` repeats the wrong name. Check a few rows of every config you select or drop by name.
3. **The viewer merges files into one split.** `reglab/legal_rag_hallucinations` concatenates a 400-row response file and a 100-row question file; `ShengbinYue/DISC-Law-SFT` mixes two schemas; `theatticusproject/cuad` becomes 84,325 lines of text; `jhu-clsp/CLERC` serves an IR schema no task uses. When a repository holds several files, load each with `data_files`.
4. **Split names that are not a split design.** CAIL2018's `exercise_contest_test` shares 56.56% of its facts with `first_stage_train`; MAUD's `test` contracts all appear in `train`; CaseHOLD's `fold_1/test` row 0 is `all/train` row 0. Hash the text field across splits before trusting a held-out number.
5. **A permissive tag over restrictive rows.** `a2aj/canadian-case-law` is tagged MIT and every sampled row's `upstream_license` includes non-commercial restrictions; the Australian corpus's summary says most documents allow commercial use while its licence file restricts the Federal and High Court decisions. Read the per-row licence field and the licence file, not only `cardData.license`.
6. **Metadata and body disagree.** `coastalcph/multi_eurlex` says CC BY-SA 4.0 in metadata and CC BY 4.0 in its body; `dennlinger/eur-lex-sum` says the reverse; `ChicagoHAI/CaseSumm` states three licences. Record both, and resolve before redistribution.
7. **Unique ids over duplicate content.** `stindardlogic/legal-reasoning-dpo-100k` gives each of its 100,000 rows a fresh UUID over 16 distinct triples. Deduplicate on content.
8. **The first-rows endpoint truncates long cells.** For long legal documents the response carries `truncated: true` and cells are cut - one served `EurLexSum` article is 100 characters. A length measured from served rows is a lower bound.
9. **Law has a date.** HousingQA is "accurate as of 2021"; the Australian legislation was scraped on 10 March 2025; Swedish statutes in `FredrikBL/Legal-Snigel-DPO` carry repeal markers. A dataset without a capture date teaches law of unknown vintage.

## The quality gate, with the legal stage

Run the parent skill's gate - readable, kind, licence, duplication, contamination, overlap, origin and shape - and add one stage between licence and duplication:

| Stage | What it asks | How it is answered |
|---|---|---|
| 2b jurisdiction and date | whose law, as of when | the card's source list and capture date; the per-row jurisdiction field when there is one (`jurisdiction` in the Australian corpus, `state` in HFforLegal and HousingQA, `dataset` court codes in A2AJ). A dataset that mixes jurisdictions without a field for it cannot be filtered to the target. |

And at the contamination stage, screen against every benchmark in the table above that shares the find's jurisdiction and language, not only the one you plan to report: legal training sets are routinely assembled from the same few public sources.

## Environment note

On a macOS machine using the python.org Python build, `urllib` requests to the Hub fail with `CERTIFICATE_VERIFY_FAILED` until the bundled certificate installer is run; `curl` works. The scripts behind this skill called `curl` through `subprocess` for that reason.

## After you find one

Write a card in the shape the other cards in this skill use - opening, the bolded lines, a pinned load line, a real row, sources, and the screening record - and give it the jurisdiction and capture date. A dataset found by search is not exempt from the gate.
