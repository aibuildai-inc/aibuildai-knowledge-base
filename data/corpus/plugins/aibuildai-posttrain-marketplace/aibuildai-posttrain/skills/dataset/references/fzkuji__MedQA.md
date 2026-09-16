# fzkuji/MedQA

https://huggingface.co/datasets/fzkuji/MedQA

272,634 rows of multiple-choice medical-exam questions across 14 Hugging Face configs - English, mainland-Chinese and Taiwanese USMLE-style exams, each in a raw "source" schema and a normalized "bigbio_qa" schema, with train/test/validation splits.

**fzkuji/MedQA** is an auto-converted, previewable Parquet mirror of the BigBio community's `bigbio/med_qa` loading script, which wraps the original MedQA release introduced in "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams" [1]. The upstream questions were drawn from professional medical board exams in three exam systems - US (English), Mainland China (simplified Chinese), and Taiwan (traditional Chinese) - and the paper reports 12,723, 34,251 and 14,123 questions for the three languages respectively; it also reports that its strongest baseline reaches only 36.7%, 42.0% and 70.1% test accuracy on the English, traditional-Chinese and simplified-Chinese questions respectively, framing the task as unsolved at release time [2]. The task shape is single-answer multiple-choice QA: a clinical-vignette or fact question, a set of answer options, and one correct option. **The dataset viewer on the upstream `bigbio/med_qa` repository errors out because that repository loads data through an arbitrary Python script (`med_qa.py`), so this converted copy is the one a viewer or `datasets-server` client can actually read rows from** [3]. The repository card states no licence beyond `unknown` [4].

**Use it for**: multiple-choice medical QA - SFT on the "source" configs' `question`/`options`/`answer` fields (formatted as a prompt-plus-choices instruction), or as a held-out multiple-choice reasoning benchmark; not a preference-pair dataset. See the SFT method card. The "source" configs map directly to an instruction-QA prompt format; the "bigbio_qa" configs are the same questions renormalized into BigBio's generic QA schema (free-text `answer` list, no structured options) and need the `choices` field turned back into a lettered option list before use as multiple-choice SFT data.

**Licence**: unknown (`cardData.license` is `"unknown"`, tag `license:unknown`, `bigbio_license_shortname` is `"UNKNOWN"`) [4]. Ungated, public repository (`"gated": false`, `"private": false`) [5]. The one catch: no licence text or grant appears anywhere in the card body, so there is no stated permission to redistribute or train on this copy beyond what Hugging Face's default terms imply.

**Shape**: 272,634 rows across 14 configs (`med_qa_en_source`, `med_qa_en_4options_source`, `med_qa_en_bigbio_qa`, `med_qa_en_4options_bigbio_qa`, `med_qa_tw_source`, `med_qa_tw_en_source`, `med_qa_tw_zh_source`, `med_qa_tw_bigbio_qa`, `med_qa_tw_en_bigbio_qa`, `med_qa_tw_zh_bigbio_qa`, `med_qa_zh_source`, `med_qa_zh_4options_source`, `med_qa_zh_bigbio_qa`, `med_qa_zh_4options_bigbio_qa`), each split train/test/validation [4][6].

**Hold out**: nothing beyond the card's own `test` splits, held out per config (see the Shape table below); no source found here states any overlap with a public evaluation benchmark beyond MedQA's own official test set.

**Origin**: converted and published to the Hub by user `fzkuji`, from the BigBio wrapper of the original human-collected MedQA exam corpus [4][2]; 676 downloads, 3 likes as of the check date, 2026-08-11 [5].

**Trained-on-by**: `medalpaca/medical_meadow_medqa` (10,178 rows, matching this repository's `med_qa_en_source` train split) reformats the English train split into an instruction/input/output SFT schema [7]; the MedAlpaca paper names the MedQA benchmark as one of the corpora it fine-tunes its models on, citing the same origin paper as this card [8]. No other adoption evidence for this specific Hub copy (`fzkuji/MedQA`) was found; the MedAlpaca and other adopters listed here trained on the underlying MedQA corpus via their own reformatted copies, not this repository directly.

**Introduced by**: [1] (Jin et al.), reformatted into BigBio's schema by the `bigbio/med_qa` community loader, whose own card carries no separate paper [3].

## Shape

Rows per config and split (Hub `cardData.dataset_info`, cross-checked against `datasets-server` `/info`) [4][6]:

| config | train | test | validation |
| --- | --- | --- | --- |
| `med_qa_en_source` (default) | 10,178 | 1,273 | 1,272 |
| `med_qa_en_4options_source` | 10,178 | 1,273 | 1,272 |
| `med_qa_en_bigbio_qa` | 10,178 | 1,273 | 1,272 |
| `med_qa_en_4options_bigbio_qa` | 10,178 | 1,273 | 1,272 |
| `med_qa_tw_source` | 11,298 | 1,413 | 1,412 |
| `med_qa_tw_en_source` | 11,298 | 1,413 | 1,412 |
| `med_qa_tw_zh_source` | 11,298 | 1,413 | 1,412 |
| `med_qa_tw_bigbio_qa` | 11,298 | 1,413 | 1,412 |
| `med_qa_tw_en_bigbio_qa` | 11,298 | 1,413 | 1,412 |
| `med_qa_tw_zh_bigbio_qa` | 11,298 | 1,413 | 1,412 |
| `med_qa_zh_source` | 27,400 | 3,426 | 3,425 |
| `med_qa_zh_4options_source` | 27,400 | 3,426 | 3,425 |
| `med_qa_zh_bigbio_qa` | 27,400 | 3,426 | 3,425 |
| `med_qa_zh_4options_bigbio_qa` | 27,400 | 3,426 | 3,425 |

Total across all 14 configs: 272,634 rows, 79,498,383 bytes as Parquet (equal to the original download size, since the source is already Parquet), 136,664,202 bytes decoded in memory [6]. No source states sequence-length or token statistics for this repository.

Two column families recur across configs, verified by reading one row from each shape (see A row, below) [4][9][10]:

- **`*_source`** configs: `meta_info` (string, exam section), `question` (string), `answer_idx` (string), `answer` (string), `options` (list of `{key, value}` structs). The `en_4options_source` variant adds a `metamap_phrases` column (list of extracted medical phrases) and trims `options` to 4 entries instead of the base config's 5; confirmed by reading row 0 of `med_qa_en_source` (5 options, no `metamap_phrases`) against row 0 of `med_qa_en_4options_source` (4 options, `metamap_phrases` present) [9].
- **`*_bigbio_qa`** configs: `id`, `question_id`, `document_id` (strings), `question` (string), `type` (string, always `multiple_choice` in the row read), `choices` (list of strings, the same option texts with the letter keys stripped), `context` (string, empty in the row read), `answer` (list of strings, the single correct choice text) [10].

The shortlist's merged 12-column view (`meta_info`, `question`, `answer_idx`, `answer`, `options`, `id`, `question_id`, `document_id`, `type`, `choices`, `context`, `metamap_phrases`) is the union of these two families plus the `4options_source` extra column; no single config carries all 12 [4].

## Quality

- The questions were drawn from professional medical board exams [2]. No source read here states how the correct-answer key was derived or verified.
- No source read here states a measured contamination, duplication, or annotation-error rate for this repository or its BigBio/original upstream.
- The repository card carries no quality caveats beyond noting that the uploader ran an automatic conversion of the dataset into a Hub-previewable format [4].

## Load it

Default config is `med_qa_en_source`; every other config must be named explicitly. Pin the revision this card's numbers were read at [5]:

```python
import datasets

REV = "d7453aabe1e3d58e44896e6a27a1cf944ba5bebe"
train = datasets.load_dataset("fzkuji/MedQA", "med_qa_en_source", revision=REV, split="train")  # 10,178 rows
test = datasets.load_dataset("fzkuji/MedQA", "med_qa_en_source", revision=REV, split="test")     # 1,273 rows - hold out
```

**Trap**: passing no config name silently loads `med_qa_en_source` (English, 5-option, no MetaMap phrases) - not the 4-option variant used in the original paper's reported results, and not the Chinese or Taiwanese exams. Each language/option-count/schema combination is a separate `config_name` argument; there is no `data_dir` merge here as there is for some BigBio mirrors, so a training run that wants multiple languages or both schema families must load and concatenate configs explicitly.

## Neighbors

- `bigbio/med_qa` - the upstream community loader this repository converts; same question set, but its viewer and `datasets-server` endpoints return an error because it runs a Python loading script rather than serving Parquet directly, so it cannot be read as a live source the way this repository can [3].
- `GBaker/MedQA-USMLE-4-options` - English-only, 4-option USMLE questions, 11,451 rows (10,178 train + 1,273 test, no validation split), under an explicit `cc-by-4.0` licence tag - unlike this repository's `unknown` licence - with `options` reshaped into a flat `{A, B, C, D}` struct instead of this repository's list-of-`{key,value}` structs [11][12]. Prefer it over this repository's `med_qa_en_4options_source` when a stated licence matters and only the English 4-option USMLE set is needed.
- `medalpaca/medical_meadow_medqa` - the English train split (10,178 rows, matching this repository's `med_qa_en_source` train count) reformatted into `instruction`/`input`/`output` SFT triples [7]; use it when the target trainer wants Alpaca-style instruction fields directly instead of `question`/`options`/`answer`.
- No neighbor found that already dedups or merges all three languages and both schema families the way this repository's 14 configs do; for multilingual multi-schema access this repository is the only one-stop copy identified.

## A row

Two distinct served shapes exist; one row from each, both from `train`, row index 0.

`config="med_qa_en_source"`, `split="train"` [9]:

```json
{
  "meta_info": "step2&3",
  "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?",
  "answer_idx": "E",
  "answer": "Nitrofurantoin",
  "options": [
    {"key": "A", "value": "Ampicillin"},
    {"key": "B", "value": "Ceftriaxone"},
    {"key": "C", "value": "Ciprofloxacin"},
    {"key": "D", "value": "Doxycycline"},
    {"key": "E", "value": "Nitrofurantoin"}
  ]
}
```

`config="med_qa_en_bigbio_qa"`, `split="train"` [10]:

```json
{
  "id": "0",
  "question_id": "0",
  "document_id": "0",
  "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?",
  "type": "multiple_choice",
  "choices": ["Ampicillin", "Ceftriaxone", "Ciprofloxacin", "Doxycycline", "Nitrofurantoin"],
  "context": "",
  "answer": ["Nitrofurantoin"]
}
```

Both rows share the same underlying question (`answer_idx: E` / `Nitrofurantoin` matches `answer: ["Nitrofurantoin"]`); the `_source` shape keeps lettered options and an explicit index, the `_bigbio_qa` shape drops letters and stores the answer as free text inside a list.

## Where it came from

The original MedQA corpus was collected from professional medical board exam questions in three exam systems - US, Mainland China, and Taiwan - by the authors of the origin paper, who also released an accompanying medical-textbook corpus for retrieval-augmented QA [4]. The BigBio initiative wrapped this corpus in a standardized loading script (`bigbio/med_qa`), exposing both a "source" schema (close to the original exam-question format) and a normalized "bigbio_qa" schema per language and option-count variant [3][4]. This repository, `fzkuji/MedQA`, was produced by running that BigBio loader and writing every resulting config/split to Parquet so it can be browsed and streamed without executing a Python script, per the repository card's own description of the conversion [4].

## Sources

Fetched on the check date, 2026-08-11. Hugging Face Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision `d7453aabe1e3d58e44896e6a27a1cf944ba5bebe`; the `datasets-server` size/info/first-rows endpoints take no revision parameter and so are reported as live reads at the check date, not pinned to that commit.

[1] Jin, Pan, Oufattole, Weng, Fang, Szolovits, "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams", 2020. https://arxiv.org/abs/2009.13081 - origin paper, current title read from the live abs page. Fetched 2026-08-11.

[2] Same paper as [1]; question counts per language, professional-exam source description, and baseline accuracy figures, read from the live arXiv abstract page. https://arxiv.org/abs/2009.13081 Fetched 2026-08-11.

[3] `bigbio/med_qa` Hugging Face API record and `datasets-server` size/info endpoints, showing the repository's file list includes `med_qa.py` (a loading script) and that the viewer endpoints return "The dataset viewer doesn't support this dataset because it runs arbitrary python code." https://huggingface.co/api/datasets/bigbio/med_qa?full=true and https://datasets-server.huggingface.co/size?dataset=bigbio%2Fmed_qa Fetched 2026-08-11.

[4] `fzkuji/MedQA` dataset card (README), including `cardData` front matter (`license`, `dataset_info` per config, `configs` with default flag) and the conversion note. https://huggingface.co/datasets/fzkuji/MedQA/raw/main/README.md Fetched 2026-08-11.

[5] Hugging Face Hub API record for `fzkuji/MedQA`. https://huggingface.co/api/datasets/fzkuji/MedQA?full=true - `sha`, `private`, `gated`, `downloads`, `likes`, `lastModified`. Fetched 2026-08-11.

[6] `datasets-server` size and info endpoints for `fzkuji/MedQA`. https://datasets-server.huggingface.co/size?dataset=fzkuji%2FMedQA and https://datasets-server.huggingface.co/info?dataset=fzkuji%2FMedQA Fetched 2026-08-11.

[7] `medalpaca/medical_meadow_medqa` dataset card and `datasets-server` info endpoint - `instruction`/`input`/`output` schema, single `train` split of 10,178 rows. https://huggingface.co/datasets/medalpaca/medical_meadow_medqa/raw/main/README.md and https://datasets-server.huggingface.co/info?dataset=medalpaca%2Fmedical_meadow_medqa Fetched 2026-08-11.

[8] Han et al., "MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data", 2023. https://arxiv.org/abs/2304.08247 - lists "training data from the MedQA benchmark" among its fine-tuning corpora, citing the same origin paper as [1]. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2304.08247). Fetched 2026-08-11.

[9] `datasets-server` first-rows endpoint, `config=med_qa_en_source`, `split=train`. https://datasets-server.huggingface.co/first-rows?dataset=fzkuji%2FMedQA&config=med_qa_en_source&split=train Fetched 2026-08-11.

[10] `datasets-server` first-rows endpoint, `config=med_qa_en_bigbio_qa`, `split=train`. https://datasets-server.huggingface.co/first-rows?dataset=fzkuji%2FMedQA&config=med_qa_en_bigbio_qa&split=train Fetched 2026-08-11.

[11] `GBaker/MedQA-USMLE-4-options` dataset card and `datasets-server` info/size endpoints - `cc-by-4.0` licence tag, flat `{A,B,C,D}` options schema, 10,178/1,273 train/test split. https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/raw/main/README.md, https://datasets-server.huggingface.co/info?dataset=GBaker%2FMedQA-USMLE-4-options and https://datasets-server.huggingface.co/size?dataset=GBaker%2FMedQA-USMLE-4-options Fetched 2026-08-11.

[12] Hugging Face Hub dataset search API, confirming `GBaker/MedQA-USMLE-4-options` and other MedQA-named repositories exist and their download/like counts as of the check date. https://huggingface.co/api/datasets?search=MedQA&limit=30 Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as a multiple-choice medical QA source, in either its raw "source" schema or its normalized "bigbio_qa" schema, across English, mainland-Chinese, and Taiwanese exam questions. The screening row's own note (below) supplies the reason this repository exists as a distinct item from its upstream: the upstream `bigbio/med_qa` loader cannot be read through the viewer or `datasets-server` endpoints, which return an error because that repository runs a Python loading script rather than serving data directly, confirmed above [3]; this repository's Parquet conversion is readable through those same endpoints, as the first-rows and info calls used throughout this card demonstrate [4][9][10].

### The screening row

The row's own note: "The BigBio MedQA release auto-converted into previewable parquet, 272,634 rows across English, Taiwanese and mainland Chinese configs in four-option and full-option forms, each with train, test and validation splits; the questions come from professional medical board exams, and the upstream bigbio/med_qa serves no viewer rows so this converted copy is the runnable one." The row carries no flag.
