# Measured contamination, split leakage and duplication

This file holds the measurements this skill ran itself, on 2026-09-23 (and on 2026-09-24 for Harvey LAB), and the method to rerun them. The parent `dataset` skill's cards take every number from a source; legal post-training cannot, because the question that matters most - does this training set carry the benchmark I will report? - is answered by no dataset card. So these numbers are measured, and every card that cites them points here.

## What was measured

1. **Benchmark containment.** For each evaluation set, how many of its test items appear inside a training-side source. Three evaluation families: LegalBench (all 162 tasks, `test` splits, 90,894 rows), CaseHOLD (LexGLUE `case_hold` test and `casehold/casehold` `all/test`), and the Chinese pair LawBench (the five tasks in `doolayer/LawBench`) plus AGIEval JEC-QA-KD.
2. **Split leakage.** Exact matches between a dataset's own test and train splits.
3. **Duplication.** Exact repeats inside a training split, on the key that matters for that dataset.
4. **Harvey LAB exposure** (added 2026-09-24, when LAB became the target). Which Hub repositories carry LAB's rubrics, instructions or documents, and which carded datasets could: section "Harvey LAB".

## Method

**Containment.** English text is lowercased and reduced to alphanumeric tokens; an item's fingerprint is its word 8-grams. Chinese text has whitespace and Unicode punctuation and symbols removed; its fingerprint is character 15-grams. An evaluation item is the concatenation of every string field except the label (`answer`, `index`, `label`), so a LegalBench item is its text plus any question or hypothesis field. Each item keeps at most 200 evenly spaced n-grams. The training-side source is streamed row by row; every n-gram of every row is looked up in the evaluation index. An item's **coverage** is the share of its sampled n-grams found anywhere in the source. An item counts as contained at coverage ≥ 0.5, and strongly contained at ≥ 0.8. Items shorter than one n-gram (fewer than 8 English tokens) have no fingerprint and are left out, which is why LegalBench counts 90,336 items rather than 90,894.

**Controls.** Following the parent skill's rule - three controls or the zeros mean nothing - every run has:

- *self*: the evaluation set against its own rows. Expect every item at 100%. LegalBench: 90,336 of 90,336 at ≥ 0.5.
- *words sorted*: each item's sampled n-grams with their words sorted, against the item index. Expect 0. Every evaluation set: 0.
- *positive*: a source known to hold the items. LegalBench against `lawinstruct/lawinstruct`'s LexGLUE `unfair_tos` file and against `coastalcph/lex_glue` `unfair_tos` train must agree, because LawInstruct copied LexGLUE - both give 210 items. A first run that read LawInstruct's rows through a `text` column (the column its card documents) found 0 everywhere, failed this control, and exposed that the files have `instruction`/`prompt`/`answer` columns instead; the numbers below are the corrected run.

**Split leakage** hashes one normalised field per row (named in each line) and counts test rows whose hash is in the train set. **Duplication** does the same inside one split.

**Limits.** Containment says the text is there, not that the label is: a LegalBench CUAD item found inside a CUAD training contract has its clause in the training data, and whether the same clause label is also there depends on how the training set was built. Coverage between 0.5 and 0.8 on long items can be shared boilerplate (NDA recitals, court captions); read the ≥ 0.8 column alongside. No MinHash near-duplicate pass was run.

## Benchmark containment: LegalBench

| evaluation set | training-side source | source rows | eval items ≥50% | eval items ≥80% | by task family (≥50%) |
| --- | --- | ---: | ---: | ---: | --- |
| LegalBench test (162 tasks) | CONTROL self | 90,894 | 90,336 / 90,336 | 89,961 | every family |
| LegalBench test (162 tasks) | CONTROL words-sorted self | 90,336 | 0 / 90,336 | 0 |  |
| LegalBench test (162 tasks) | nvidia/Nemotron-Pretraining-Legal-v1 Nemotron-Pretraining-Legal-Case-Law-Summary | 53,137 | 52 / 90,336 | 11 | overruling 21/2,227; citation_prediction 9/161; contract_qa 9/80; definition 6/2,018; cuad 4/17,980; function_of_decision_section 2/363 |
| LegalBench test (162 tasks) | nvidia/Nemotron-Pretraining-Legal-v1 Nemotron-Pretraining-Legal-LegalBench-CUAD-v2 | 460,031 | 1,655 / 90,336 | 326 | cuad 1,594/17,980; contract_qa 39/80; unfair_tos 17/3,614; maud 3/4,598; contract_nli 2/1,927 |
| LegalBench test (162 tasks) | nvidia/Nemotron-Pretraining-Legal-v1 Nemotron-Pretraining-Legal-Definition-Classification | 10,200 | 14 / 90,336 | 8 | definition 7/2,018; overruling 4/2,227; citation_prediction 2/161; proa 1/95 |
| LegalBench test (162 tasks) | nvidia/Nemotron-Pretraining-Legal-v1 Nemotron-Pretraining-Legal-Diversity-Jurisdiction | 6,480 | 1 / 90,336 | 0 | diversity 1/1,800 |
| LegalBench test (162 tasks) | nvidia/Nemotron-Pretraining-Legal-v1 Nemotron-Pretraining-Legal-Function-Of-Decision | 70,039 | 19 / 90,336 | 7 | overruling 10/2,227; citation_prediction 3/161; definition 3/2,018; function_of_decision_section 3/363 |
| LegalBench test (162 tasks) | nvidia/Nemotron-Pretraining-Legal-v1 Nemotron-Pretraining-Legal-GlobalCit | 88,898 | 8,895 / 90,336 | 4,558 | international_citizenship_questions 8,895/9,306 |
| LegalBench test (162 tasks) | nvidia/Nemotron-Pretraining-Legal-v1 Nemotron-Pretraining-Legal-NYCourts-Judicial-Ethics-Opinions | 5,511 | 70 / 90,336 | 13 | nys_judicial_ethics 69/292; overruling 1/2,227 |
| LegalBench test (162 tasks) | nvidia/Nemotron-Pretraining-Legal-v1 Nemotron-Pretraining-Legal-ToS-Clause-Understanding | 6,831 | 19 / 90,336 | 7 | unfair_tos 19/3,614 |
| LegalBench test (162 tasks) | nvidia/Nemotron-Pretraining-Legal-v1 Nemotron-Pretraining-Legal-ToSDR-QA | 7,569 | 1 / 90,336 | 0 | unfair_tos 1/3,614 |
| LegalBench test (162 tasks) | kiddothe2b/contract-nli train (a+b) | 14,010 | 454 / 90,336 | 138 | contract_nli 303/1,927; cuad 112/17,980; contract_qa 33/80; unfair_tos 6/3,614 |
| LegalBench test (162 tasks) | theatticusproject/maud train | 25,827 | 4,530 / 90,336 | 4,420 | maud 4,530/4,598 |
| LegalBench test (162 tasks) | coastalcph/lex_glue unfair_tos train | 5,532 | 210 / 90,336 | 97 | unfair_tos 190/3,614; cuad 9/17,980; opp115 5/9,187; consumer_contracts_qa 4/396; contract_qa 1/80; privacy_policy 1/15,258 |
| LegalBench test (162 tasks) | theatticusproject/cuad (CUAD_v1 text lines) | 84,325 | 7,141 / 90,336 | 2,875 | cuad 7,088/17,980; contract_qa 33/80; unfair_tos 15/3,614; maud 5/4,598 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/ContractNLI-contract_nli-train-0.jsonl.xz | 14,010 | 454 / 90,336 | 138 | contract_nli 303/1,927; cuad 112/17,980; contract_qa 33/80; unfair_tos 6/3,614 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/InternationalCitizenshipLawQuestions-international_citizenship_law_questions_mode_acq-train-0.jsonl.xz | 6,460 | 6,457 / 90,336 | 6,457 | international_citizenship_questions 6,457/9,306 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/InternationalCitizenshipLawQuestions-international_citizenship_law_questions_mode_loss-train-0.jsonl.xz | 2,850 | 2,849 / 90,336 | 2,849 | international_citizenship_questions 2,849/9,306 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/LexGLUE-unfair_tos-train-0.jsonl.xz | 5,532 | 210 / 90,336 | 97 | unfair_tos 190/3,614; cuad 9/17,980; opp115 5/9,187; consumer_contracts_qa 4/396; contract_qa 1/80; privacy_policy 1/15,258 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/LexGLUE-case_hold-train-0.jsonl.xz | 45,000 | 39 / 90,336 | 7 | overruling 21/2,227; citation_prediction 9/161; definition 6/2,018; function_of_decision_section 2/363; unfair_tos 1/3,614 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/MAUD-answer-train-0.jsonl.xz | 10,751 | 4,085 / 90,336 | 3,786 | maud 4,085/4,598 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/MAUD-category-train-0.jsonl.xz | 25,827 | 4,530 / 90,336 | 4,420 | maud 4,530/4,598 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/MAUD-question-train-0.jsonl.xz | 25,827 | 4,530 / 90,336 | 4,420 | maud 4,530/4,598 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/MAUD-text_type-train-0.jsonl.xz | 25,827 | 4,530 / 90,336 | 4,420 | maud 4,530/4,598 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/PrivacyQA-privacy_qa-train-0.jsonl.xz | 185,200 | 403 / 90,336 | 101 | opp115 272/9,187; privacy_policy 124/15,258; unfair_tos 7/3,614 |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/Sara-sara_entailment-train-0.jsonl.xz | 176 | 0 / 90,336 | 0 |  |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/Sara-sara_tax_liability-train-0.jsonl.xz | 160 | 0 / 90,336 | 0 |  |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/SaraProlog-sara_prolog_facts-train-0.jsonl.xz | 376 | 0 / 90,336 | 0 |  |
| LegalBench test (162 tasks) | lawinstruct/lawinstruct data/SaraProlog-sara_prolog_statute-train-0.jsonl.xz | 9 | 91 / 90,336 | 0 | sara 91/368 |
| LegalBench test (162 tasks) | pile-of-law/pile-of-law data/train.r_legaldvice.jsonl.xz | 109,740 | 2,240 / 90,336 | 2,176 | learned_hands 2,200/11,109; unfair_tos 28/3,614; contract_qa 7/80; consumer_contracts_qa 2/396; function_of_decision_section 2/363; definition 1/2,018 |
| LegalBench test (162 tasks) | pile-of-law/pile-of-law data/train.tos.jsonl.xz | 37 | 2,976 / 90,336 | 2,920 | unfair_tos 2,931/3,614; consumer_contracts_qa 30/396; cuad 9/17,980; opp115 5/9,187; contract_qa 1/80 |
| LegalBench test (162 tasks) | pile-of-law/pile-of-law data/train.examoutlines.jsonl.xz | 12 | 1 / 90,336 | 0 | overruling 1/2,227 |
| LegalBench cuad_* test tasks | CUAD (GitHub data.zip, used by theatticusproject/cuad-qa) train_separate_questions.json contract texts | 408 | 13,674 / 18,060 | 5,612 | cuad_license_grant 1,073/1,396; cuad_cap_on_liability 964/1,246; cuad_audit_rights 921/1,216; cuad_anti-assignment 847/1,172; cuad_insurance 746/1,030; cuad_governing_law 658/876 |
| LegalBench cuad_* test tasks | CUAD (GitHub data.zip, used by theatticusproject/cuad-qa) test.json contract texts | 102 | 3,286 / 18,060 | 1,299 | cuad_license_grant 263/1,396; cuad_cap_on_liability 228/1,246; cuad_anti-assignment 226/1,172; cuad_audit_rights 215/1,216; cuad_insurance 192/1,030; cuad_governing_law 176/876 |

Read this table by task family. The containments that change what a LegalBench score means:

- `international_citizenship_questions` (9,306 test items): LawInstruct's two International Citizenship Law Questions files hold 6,457 + 2,849 = 9,306 of them, every item; the Nemotron `GlobalCit` config holds 8,895.
- `maud_*` (4,598 items): `theatticusproject/maud` `train` holds 4,530; LawInstruct's MAUD files hold the same 4,530.
- `cuad_*` (17,980 items): the CUAD training contracts that `theatticusproject/cuad-qa` loads hold 13,674 of the 18,060 CUAD-plus-`contract_qa` items; the CUAD test contracts hold 3,286; the Nemotron `LegalBench-CUAD-v2` config holds 1,594.
- `unfair_tos` (3,614 items): Pile of Law's 37-row `tos` config - the CLAUDETTE corpus - holds 2,931.
- `learned_hands_*` (11,109 items): Pile of Law's `r_legaladvice` config holds 2,200.
- `contract_nli_*` (1,927 items): ContractNLI `train` (and LawInstruct's copy of it) holds 303.

## Benchmark containment: CaseHOLD

| evaluation set | training-side source | source rows | eval items ≥50% | eval items ≥80% | by task family (≥50%) |
| --- | --- | ---: | ---: | ---: | --- |
| LexGLUE case_hold test | CONTROL words-sorted self | 3,600 | 0 / 3,600 | 0 |  |
| LexGLUE case_hold test | casehold/casehold all train | 42,509 | 349 / 3,600 | 32 | case_hold 349/3,600 |
| LexGLUE case_hold test | coastalcph/lex_glue case_hold train | 45,000 | 365 / 3,600 | 36 | case_hold 365/3,600 |
| LexGLUE case_hold test | nvidia Nemotron-Pretraining-Legal-Case-Law-Summary (holds CaseHOLD MCQ) | 53,137 | 3,600 / 3,600 | 3,258 | case_hold 3,600/3,600 |
| casehold/casehold all test | CONTROL words-sorted self | 5,314 | 0 / 5,314 | 0 |  |
| casehold/casehold all test | casehold/casehold all train | 42,509 | 534 / 5,314 | 69 | all 534/5,314 |
| casehold/casehold all test | coastalcph/lex_glue case_hold train | 45,000 | 534 / 5,314 | 52 | all 534/5,314 |
| casehold/casehold all test | nvidia Nemotron-Pretraining-Legal-Case-Law-Summary (holds CaseHOLD MCQ) | 53,137 | 5,312 / 5,314 | 4,793 | all 5,312/5,314 |

Every LexGLUE `case_hold` test item, and 5,312 of 5,314 `casehold/casehold` `all/test` items, is inside the Nemotron config named `Nemotron-Pretraining-Legal-Case-Law-Summary` - which, despite its name, holds the reformatted CaseHOLD multiple-choice questions (see that dataset's card). CaseHOLD's own `train` holds about 10% of test items at ≥ 0.5 but only about 1% at ≥ 0.8, and exact-match leakage is 0.15% (next table): its prompts are excerpts of opinions, and different excerpts of one opinion share text without being the same question.

## Benchmark containment: LawBench and AGIEval JEC-QA-KD

| evaluation set | training-side source | source rows | eval items ≥50% | eval items ≥80% | by task family (≥50%) |
| --- | --- | ---: | ---: | ---: | --- |
| LawBench (5 tasks) + AGIEval JEC-QA-KD | CONTROL words-sorted self | 3,500 | 0 / 3,500 | 0 |  |
| LawBench (5 tasks) + AGIEval JEC-QA-KD | china-ai-law-challenge/cail2018 exercise_contest_train | 154,592 | 538 / 3,500 | 472 | LawBench 3-3 190/500; LawBench 3-1 175/500; LawBench 3-4 173/500 |
| LawBench (5 tasks) + AGIEval JEC-QA-KD | china-ai-law-challenge/cail2018 first_stage_train | 1,710,856 | 1,499 / 3,500 | 1,474 | LawBench 3-1 500/500; LawBench 3-3 500/500; LawBench 3-4 499/500 |
| LawBench (5 tasks) + AGIEval JEC-QA-KD | ShengbinYue/DISC-Law-SFT DISC-Law-SFT-Pair-QA-released.jsonl | 79,692 | 20 / 3,500 | 2 | LawBench 3-6 11/500; AGIEval JEC-QA-KD 5/1,000; LawBench 1-2 4/500 |
| LawBench (5 tasks) + AGIEval JEC-QA-KD | ShengbinYue/DISC-Law-SFT DISC-Law-SFT-Pair.jsonl | 166,758 | 1,975 / 3,500 | 1,430 | AGIEval JEC-QA-KD 902/1,000; LawBench 1-2 500/500; LawBench 3-6 500/500; LawBench 3-3 30/500; LawBench 3-1 22/500; LawBench 3-4 21/500 |
| LawBench (5 tasks) + AGIEval JEC-QA-KD | ShengbinYue/DISC-Law-SFT DISC-Law-SFT-Triplet-QA-released.jsonl | 23,331 | 4 / 3,500 | 0 | AGIEval JEC-QA-KD 2/1,000; LawBench 1-2 2/500 |
| LawBench (5 tasks) + AGIEval JEC-QA-KD | ShengbinYue/DISC-Law-SFT DISC-Law-SFT-Triplet-released.jsonl | 16,000 | 62 / 3,500 | 50 | LawBench 3-1 21/500; LawBench 3-3 21/500; LawBench 3-4 20/500 |

LawBench tasks 3-1, 3-3 and 3-4 are CAIL2018 fact descriptions: `first_stage_train` holds 1,499 of their 1,500 items. LawBench 1-2 and 3-6 and JEC-QA-KD are judicial-examination questions: `DISC-Law-SFT-Pair.jsonl` holds all 1,000 LawBench items and 902 of 1,000 JEC-QA-KD items.

## Split leakage

| check | test rows | test rows found in train | % |
| --- | ---: | ---: | ---: |
| cail2018 exercise_contest_test vs first_stage_train (fact) | 32,508 | 18,388 | 56.56 |
| cail2018 first_stage_test vs first_stage_train (fact) | 217,016 | 13,182 | 6.07 |
| cail2018 final_test vs first_stage_train (fact) | 35,922 | 223 | 0.62 |
| cail2018 exercise_contest_train vs first_stage_train (fact) | 154,592 | 76,572 | 49.53 |
| cail2018 exercise_contest_test vs exercise_contest_train (fact) | 32,508 | 836 | 2.57 |
| casehold all/test vs all/train (citing_prompt) | 5,314 | 8 | 0.15 |
| casehold all/validation vs all/train (citing_prompt) | 5,314 | 9 | 0.17 |
| lex_glue case_hold test vs casehold all/train (context) | 3,600 | 5 | 0.14 |
| lex_glue case_hold test vs lex_glue case_hold train (context) | 3,600 | 8 | 0.22 |
| casehold all/test vs lex_glue case_hold train | 5,314 | 11 | 0.21 |
| contract-nli contractnli_a test vs train (premise+hypothesis) | 1,991 | 72 | 3.62 |
| contract-nli contractnli_a test vs train (premise only) | 1,991 | 116 | 5.83 |
| contract-nli contractnli_b test vs train (premise+hypothesis) | 2,091 | 17 | 0.81 |
| contract-nli contractnli_b test vs train (premise only) | 2,091 | 17 | 0.81 |
| maud test vs train (text+question+subquestion) | 6,651 | 81 | 1.22 |
| maud test vs train (contract_name) | 6,651 | 6,651 | 100.0 |
| billsum test vs train (text) | 3,269 | 4 | 0.12 |
| Legal-Snigel-DPO test vs train (prompt) | 400 | 1 | 0.25 |

CAIL2018's contest splits are not disjoint from its first-stage training data: 56.56% of `exercise_contest_test` facts appear verbatim in `first_stage_train`, and 49.53% of `exercise_contest_train` does too. MAUD's `test` rows share no exact item with `train` beyond 81, but every one of the 6,651 `test` rows comes from a contract that also appears in `train` - the split is by question, not by contract.

## Duplication

| dataset | key | rows | unique | repeated rows | % | largest group |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `stindardlogic/legal-reasoning-dpo-100k` | prompt | 100,000 | 16 | 99,984 | 99.98 | 6,250 |
| `stindardlogic/legal-reasoning-dpo-100k` | chosen | 100,000 | 16 | 99,984 | 99.98 | 6,250 |
| `stindardlogic/legal-reasoning-dpo-100k` | rejected | 100,000 | 16 | 99,984 | 99.98 | 6,250 |
| `stindardlogic/legal-reasoning-dpo-100k` | whole row | 100,000 | 16 | 99,984 | 99.98 | 6,250 |
| `Alignment-Lab-AI/Lawyer-Instruct` | instruction+output | 9,241 | 9,191 | 50 | 0.54 | 9 |
| `Alignment-Lab-AI/Lawyer-Instruct` | instruction | 9,241 | 9,051 | 190 | 2.06 | 15 |
| `dzunggg/legal-qa-v1` | question+answer | 3,742 | 3,741 | 1 | 0.03 | 2 |
| `dzunggg/legal-qa-v1` | question | 3,742 | 3,732 | 10 | 0.27 | 2 |
| `nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions` | instruction+input+output | 4,394 | 4,227 | 167 | 3.8 | 3 |
| `nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions` | instruction | 4,394 | 268 | 4,126 | 93.9 | 1,083 |
| `isaacus/open-australian-legal-qa` | question | 2,124 | 2,122 | 2 | 0.09 | 2 |
| `louisbrulenaudet/legalkit` | input+output | 53,000 | 52,930 | 70 | 0.13 | 2 |
| `louisbrulenaudet/legalkit` | query | 53,000 | 52,921 | 79 | 0.15 | 4 |
| `ymoslem/Law-StackExchange` | question_id | 24,370 | 24,370 | 0 | 0.0 | 1 |
| `mb7419/legal-advice-reddit_preference` | whole row | 70,324 | 70,269 | 55 | 0.08 | 4 |
| `mb7419/legal-advice-reddit_preference` | post_id | 70,324 | 24,986 | 45,338 | 64.47 | 5 |
| `FredrikBL/Legal-Snigel-DPO` | prompt (train) | 1,598 | 1,597 | 1 | 0.06 | 2 |
| `theatticusproject/maud` | text+question+subquestion (train) | 25,827 | 21,691 | 4,136 | 16.01 | 3 |
| `FiscalNote/billsum` | text (train) | 18,949 | 18,906 | 43 | 0.23 | 2 |
| `DISC-Law-SFT DISC-Law-SFT-Pair-QA-released.jsonl` | input+output (id excluded) | 79,692 | 76,231 | 3,461 | 4.34 | 3 |
| `DISC-Law-SFT DISC-Law-SFT-Pair.jsonl` | input+output (id excluded) | 166,758 | 161,704 | 5,054 | 3.03 | 55 |
| `DISC-Law-SFT DISC-Law-SFT-Triplet-QA-released.jsonl` | input+output (id excluded) | 23,331 | 23,331 | 0 | 0.0 | 1 |
| `DISC-Law-SFT DISC-Law-SFT-Triplet-released.jsonl` | input+output (id excluded) | 16,000 | 16,000 | 0 | 0.0 | 1 |

## Harvey LAB

Measured on 2026-09-24, after the maintainers named Harvey LAB (`harveyai/harvey-labs`, commit `1dd8140`) as this skill's target benchmark. LAB is different from the benchmarks above in two ways that change what the measurement has to cover. It has no training split, and it went public on 2026-05-06, after most of this skill's datasets were frozen. So the question is not "which training set holds the test split" but "which Hub repository copies LAB, or holds model runs on it".

**Evaluation items.** Three kinds per task, 52,707 items in all: the rubric (every `match_criteria` joined, 2,010 items), the instructions (2,010), and every source document with extractable text (`.docx`, `.eml`, `.txt`, `.json`; 48,687 items). `.xlsx` and `.pptx` sources were not read.

**Method.** The same as above: lowercased alphanumeric tokens, word 8-grams, coverage at ≥ 0.5 and ≥ 0.8, every string field of a source row joined. One change for size: documents keep at most 50 evenly spaced 8-grams instead of 200 (rubrics and instructions keep 200), so the 2.8 million-entry index fits in a laptop's memory. Matching uses an Aho-Corasick automaton over the whole row text rather than a hash lookup per n-gram; the result is the same set of hits.

**Controls.**

- *self*: LAB's own text against the index: 2,010 / 2,010 rubrics, 2,010 / 2,010 instructions, 48,687 / 48,687 documents at ≥ 0.8.
- *words sorted*: each item's sampled 8-grams with their words sorted: 0 of 52,707.
- *positive*: `irfanjamil/Harvey-LAB`, a Hub copy of LAB with 1,243 task rows. It returns 1,242 rubrics and 1,242 instructions at ≥ 0.5, which is every LAB task in the 24 practice areas it copied (1,251 today) except 9, consistent with tasks added or renamed after the copy was made; it has no `contracts`, `diligence` or `firm-knowledge` task. Its rubrics at ≥ 0.8 are 957: the copy predates later rubric edits.

**Which datasets could hold LAB at all.** A file last committed before 2026-05-06 cannot contain LAB. Reading each carded dataset's file tree at its pinned revision (`/api/datasets/<id>/tree/<sha>?expand=true`), 39 of this skill's original 43 datasets have no file committed after that date. The other four were scanned file by file, only the files committed after it:

| carded dataset | files after 2026-05-06 | scanned | rubrics ≥50% | instructions ≥50% | documents ≥50% |
| --- | --- | --- | ---: | ---: | ---: |
| `stindardlogic/legal-reasoning-dpo-100k` | 1 data file, 0.37 GB | all, 100,000 rows | 0 | 0 | 0 |
| `nvidia/Nemotron-Pretraining-Legal-v1` | 21 data files, 6.99 GB | pending: scan running at this commit | | | |
| `pile-of-law/pile-of-law` | 3 data files, 3.16 GB (`courtlisteneropinions` 5 and 9, `courtlistenerdocketentries` validation 0) | not scanned | | | |
| `a2aj/canadian-case-law` | 30 per-court files, 4.27 GB | not scanned | | | |

The two unscanned sets are real court opinions and docket entries; LAB's documents are synthetic matters, so a match is not plausible, but it was not measured.

**Hub repositories that copy or run LAB.** Found by searching the Hub for `harvey`, `harvey_lab`, `harvey-lab`, `legal-agent` and `legal_agent` (2026-09-24), then scanned in full at their current revision:

| repository | what it is | rows | rubrics ≥50% | instructions ≥50% | documents ≥50% |
| --- | --- | ---: | ---: | ---: | ---: |
| `irfanjamil/Harvey-LAB` | LAB itself: task, instructions, criteria and documents, re-split 1,057 `train` / 186 `eval` | 1,243 | 1,242 | 1,242 | 6,841 |
| `ShubyM/harvey-lab-glm-traces` | GLM-5.2 agent trajectories on LAB tasks, raw and SFT-formatted | 1,787 | 0 | 394 | 2,659 |
| `violetxi/harvey-eval-gpt56sol-*` (10 repos) | evaluation sets and runs for the `firm-knowledge` tasks | 3,278 to 12,772 each | 250 each | 250 each | 27 to 276 |
| `violetxi/harvey-kl-ground-sessions` | agent sessions over the `firm-knowledge` document store | 44,115 | 0 | 0 | 6,795 |
| `violetxi/harvey-note-conditioned-rollouts` | 31,000 Qwen3.5-9B trajectories over the same store | 292,078 | 0 | 0 | 5,735 |
| `violetxi/harvey-notes-v4` | notes distilled from those rollouts | 1,159,338 | 0 | 0 | 5 |
| `Hanno-Labs/harvey-labs-llm-artifact-analysis` | refusal-classifier features built from LAB `.docx` outputs | 67,924 | 0 | 0 | 265 |
| `narcolepticchicken/harvey-qwen35-isft` | training harness files; two `corporate-ma` tasks inside | 2,653 | 0 | 2 | 8 |
| `violetxi/harvey-closed-book-*` (9 repos), `violetxi/harvey-eval-recall-*` (7 repos) | closed-book and recall evaluations | 10,111 / 15,870 each | 0 | 0 | 0 |
| `narcolepticchicken/legal-agent-traces-v5`, `narcolepticchicken/legal-agent-router-dataset-v4` | legal agent traces and router data | 2,011 / 15,524 | 0 | 0 | 0 |

Every `violetxi/harvey-eval-gpt56sol-*` repository holds the same 250 rubrics: all of them are `firm-knowledge` tasks, as are 6,697 of the 6,795 documents in `harvey-kl-ground-sessions` and 5,640 of the 5,735 in `harvey-note-conditioned-rollouts`. `ShubyM/harvey-lab-glm-traces` holds no rubric text, but 394 task instructions (387 at ≥ 0.8) across 25 areas, led by `contracts` 97 and `corporate-ma` 44, and the documents the agent read.

**New candidate datasets.** Scanned in full: `crosbylegal/RedlineBench` (9,888 rows) 0 items; `open-agreements/legal-practice-library` (1,622 rows from 140 non-Markdown files; its Markdown explainers and templates were not read) 0 items; `TheTokenFactory/sec-contracts-financial-extraction-instructions` (23,050 rows) one document at exactly 0.5 coverage, `diligence/rail-horizontal-merger/.../sox-302-404-certifications-2019-2024-11.docx`, the boilerplate text of a SOX certification. `chenghao/sec-material-contracts` (39.7 GB) was not scanned; its files were last committed on 2025-08-14, before LAB existed.

**What to do with this.** Never train on any repository in the second table if you report LAB, and treat a Hub repository whose name mentions Harvey or LAB as contaminated until measured. The `firm-knowledge` split is the most exposed: its whole rubric set is public in at least ten repositories besides LAB itself. The older legal datasets in this skill are not a LAB contamination risk.

## Rerunning it

The measurement is about sixty lines. Load an evaluation set's rows into `Eval`, then call `run` once per training-side source; the three controls are `run` on the evaluation rows themselves and `run` on `sorted_control()`.

```python
import re, unicodedata
TOK = re.compile(r"[a-z0-9]+")
def en_grams(s, n=8):
    t = TOK.findall(s.lower()); return [" ".join(t[i:i+n]) for i in range(len(t)-n+1)]
def zh_grams(s, n=15):
    t = "".join(c for c in s if not c.isspace() and unicodedata.category(c)[0] not in "PSZ")
    return [t[i:i+n] for i in range(len(t)-n+1)]
def sample(g, k=200):
    return g if len(g) <= k else [g[int(j*len(g)/k)] for j in range(k)]

class Eval:
    def __init__(self, items, lang="en"):          # items: [(group, text)]
        self.gf = en_grams if lang == "en" else zh_grams
        self.items = [(grp, sample(self.gf(t))) for grp, t in items]
        self.items = [(grp, g) for grp, g in self.items if g]
        self.index = {}
        for i, (_, g) in enumerate(self.items):
            for x in g: self.index.setdefault(x, set()).add(i)
    def run(self, texts):                          # texts: iterable of training rows as strings
        hit = {}
        for t in texts:
            for x in self.gf(t):
                for i in self.index.get(x, ()):
                    hit.setdefault(i, set()).add(x)
        cov = [len(hit.get(i, ())) / len(g) for i, (_, g) in enumerate(self.items)]
        return sum(c >= 0.5 for c in cov), sum(c >= 0.8 for c in cov), len(cov)
    def sorted_control(self):
        return [" ".join(sorted(" ".join(g).split())) for _, g in self.items]
```

The full scripts used for this file - downloading through the datasets-server `/parquet` endpoint, the evaluation-set builders, the split-leakage and duplication checks - ran on the check date against the revisions pinned in each card.
