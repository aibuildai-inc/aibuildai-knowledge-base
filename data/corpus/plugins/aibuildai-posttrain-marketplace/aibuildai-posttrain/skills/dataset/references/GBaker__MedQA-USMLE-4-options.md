# GBaker/MedQA-USMLE-4-options

11,451 four-option USMLE-style medical board-exam questions, split into a 10,178-row `train` and a 1,273-row `test`, each with a MetaMap phrase list alongside the question, options and answer key.

**GBaker/MedQA-USMLE-4-options** repackages the English, four-option subset of MedQA, the open-domain multiple-choice medical-exam question-answering dataset introduced by Jin et al. in "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams" [1]. Each row is one USMLE-style board-exam question with four answer choices (A-D), the correct choice, and a list of medical phrases extracted from the question with the MetaMap tool, per the original GitHub repository's description as quoted in a sibling dataset's card [2]. It lives at https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options .

**The `test` split (1,273 rows) is the split the screening note calls "the number people report" for this benchmark [3]; hold it out of any training run and use only `train` (10,178 rows) for fine-tuning.**

**Use it for**: multiple-choice medical QA - question, four labeled options and an answer key, suited to instruction/reasoning-trace SFT where the target is the correct option (optionally with a rationale you add yourself, since none is provided here) or to held-out benchmark scoring; not preference pairs, since there is only one labeled answer per row. Maps to a plain question-with-choices instruction format - see the SFT method card. **Train on `train` only; `test` is the reported benchmark split.**

**Licence**: CC-BY-4.0 (`cardData.license` is `"cc-by-4.0"`), ungated (`"gated": false`, `"private": false`) [4]. The one catch: a same-builder sibling repo of this data (`GBaker/MedQA-USMLE-4-options-hf`) declares a different licence, CC-BY-SA-4.0 [5][6] - use the licence on the specific repo you load.

**Shape**: 11,451 rows in one config (`default`), split `train` 10,178 / `test` 1,273, six columns (`question`, `answer`, `options`, `meta_info`, `answer_idx`, `metamap_phrases`) [7][8].

**Hold out**: `test` (1,273 rows) - the screening note names it as the split people report scores against [3].

**Origin**: uploaded to the Hub by user GBaker [4]; the questions and answer keys are human-authored USMLE board-exam material, not model-generated [1]. Hub API at the check date: `downloads` 25,769, `likes` 99 [4].

**Trained-on-by**: a Hub model search for this dataset's id returns only two low-signal community fine-tunes (`guicondor/prometheus-7b-v2.0_MedQA-USMLE-4-options_finetuned_test` and `Machlovi/Phi4_MedQA_USMLE_4_Options`), both with 0 recorded downloads and 0 likes at the check date; no widely-adopted model or recipe naming this specific repo was found [9].

**Introduced by**: [1] (Jin et al.).

## Shape

Rows served and splits (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 10,178 |
| `test` | 1,273 |
| total | 11,451 |

One config, `default`, with six columns (datasets-server `/info`) [8]:

| column | dtype |
| --- | --- |
| `question` | string |
| `answer` | string |
| `options` | struct\<A: string, B: string, C: string, D: string\> |
| `meta_info` | string |
| `answer_idx` | string |
| `metamap_phrases` | list\<string\> |

`meta_info` records which USMLE step the question is drawn from; of the first 100 rows read at offset 0 in each split, both `train` and `test` contain a mix of `step1` and `step2&3` values (train: 58 `step1` / 42 `step2&3`; test: 53 `step1` / 47 `step2&3`) [7][8]. No source states sequence-length or token statistics for this release.

Sizes (datasets-server `/size`) [7]: 18,286,756 bytes of original JSON download, 8,887,601 bytes as Parquet, 17,121,864 bytes decoded in memory.

A same-builder sibling, `GBaker/MedQA-USMLE-4-options-hf`, reformats the same question content into a HellaSwag-style multiple-choice schema and serves 12,723 rows across three splits - `train` 10,178, `validation` 1,272, `test` 1,273 [6] - which sums to the 12,723 English questions the origin paper reports [1]. This repository's 11,451 rows (`train` + `test` only, no `validation`) are 1,272 short of that total, so the 1,272-row development/validation portion of the original English MedQA split is not included here [1][4][6].

## Quality

- No source states a measured contamination rate, duplicate rate, or answer-accuracy check for this specific repackaging.
- The `metamap_phrases` column holds medical phrases extracted from each question with the MetaMap tool, per the original dataset's own description as quoted in a sibling dataset's card: "Those files in the 'metamap' folders are extracted medical related phrases using the Metamap tool" [2].
- A sibling card (`medalpaca/medical_meadow_medqa`) states the original dataset "provide[s] an official random split into train, dev, and test sets" [2], consistent with the three-way split this repo's `-hf` sibling exposes and this repo's own two-way (`train`/`test`) subset of it [6].
- The `answer` field is human-readable text (e.g. "Nitrofurantoin") and `answer_idx` is the matching letter (e.g. "D"); of the rows sampled above, `answer_idx` took only the values A-D, consistent with the four-option design [7][8].

## Load it

Train on `train`, hold out `test`, and pin `load_dataset` to the Hub API's `sha` for `main` at the check date (the repo was last modified 2023-01-24) [4]. That pin covers the repo content `load_dataset` fetches; it does not cover this card's row counts, byte sizes, or sampled rows, which came from the datasets-server `/size`, `/info`, and `/first-rows` endpoints [7][8] - those endpoints take no `revision` parameter and are read live off `main`, not pinned to the sha below:

```python
import datasets

REV = "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"  # main at the check date
train = datasets.load_dataset("GBaker/MedQA-USMLE-4-options", revision=REV, split="train")  # 10,178 rows
test = datasets.load_dataset("GBaker/MedQA-USMLE-4-options", revision=REV, split="test")    # 1,273 rows - hold out
```

**Trap**: `options` is a struct of four separate string fields (`A`, `B`, `C`, `D`), not a list - a template that expects a list of choices must build one from `options["A"]`..`options["D"]` before formatting a prompt.

## Neighbors

- `GBaker/MedQA-USMLE-4-options-hf` - the same builder's reformatting of this same question set into a `sent1`/`sent2`/`ending0..3`/`label` HellaSwag-style schema, with an extra `validation` split (1,272 rows) that this repository omits; licensed CC-BY-SA-4.0, not CC-BY-4.0 [5][6]. A fetched training row confirms it carries exactly four options (`ending0`..`ending3`) with the identical question and option text as this repo's row 0, so the two repos are the same underlying rows in different schemas, not different data [6].
- `bigbio/med_qa` - the BigBio-community wrapper around the full original MedQA release (English, simplified Chinese, traditional Chinese; 12,723 / 34,251 / 14,123 questions respectively), loaded through a Python loading script rather than served Parquet, so its rows could not be read via the datasets-server API; license is listed as unknown on its card [10].
- `medalpaca/medical_meadow_medqa` - an instruction-tuning reformatting with the same 10,178 `train` row count as this repo's `train` split, but a fetched row shows it keeps the original **five**-option version of the question (options A-E) rather than the four-option version here, so despite the matching row count it is not the same content [11][12].

This corpus prefers this repository (`GBaker/MedQA-USMLE-4-options`) for the plain four-option question/options/answer shape; use `-hf` only if a HellaSwag-style multiple-choice collator is required, and avoid mixing `medical_meadow_medqa`'s five-option rows into a run built on this four-option set.

## A row

One config, one schema shared by both splits (`train` and `test` return the same six columns; a row was fetched from each split to confirm). From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], with `metamap_phrases` truncated:

```json
{
  "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?",
  "answer": "Nitrofurantoin",
  "options": {
    "A": "Ampicillin",
    "B": "Ceftriaxone",
    "C": "Doxycycline",
    "D": "Nitrofurantoin"
  },
  "meta_info": "step2&3",
  "answer_idx": "D",
  "metamap_phrases": ["23 year old pregnant woman", "weeks presents", "burning", "urination", "..."]
}
```

From `config="default"`, `split="test"`, `row_idx=0` [8], same schema, different content:

```json
{
  "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. [...] Which of the following is the correct next action for the resident to take?",
  "answer": "Tell the attending that he cannot fail to disclose this mistake",
  "options": {
    "A": "Disclose the error to the patient and put it in the operative report",
    "B": "Tell the attending that he cannot fail to disclose this mistake",
    "C": "Report the physician to the ethics committee",
    "D": "Refuse to dictate the operative report"
  },
  "meta_info": "step1",
  "answer_idx": "B",
  "metamap_phrases": ["junior orthopaedic surgery resident", "completing", "carpal tunnel repair", "..."]
}
```

## Where it came from

The underlying question set was collected from professional medical board exams (USMLE) by Jin et al. for MedQA, the first free-form multiple-choice open-domain QA dataset for medical problems, covering English, simplified Chinese and traditional Chinese exams [1]. This Hub repository, uploaded by user GBaker [4], carries only the English four-option subset, split into `train` and `test` JSONL files, plus a `metamap_phrases` field per question extracted with the MetaMap tool - a description the original GitHub repository gives and that a sibling dataset's card quotes [2]. The README states no further detail about who performed the repackaging or when beyond the citation to the origin paper [13].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Jin, Pan, Oufattole, Weng, Fang, Szolovits, "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams", arXiv:2009.13081, 2020. https://arxiv.org/abs/2009.13081 - the origin paper; current title and abstract (12,723/34,251/14,123 questions by language) read from the live abs page. Fetched 2026-08-11.

[2] medalpaca/medical_meadow_medqa dataset card (README). https://huggingface.co/datasets/medalpaca/medical_meadow_medqa/raw/main/README.md - quotes the original jind11/MedQA GitHub repository's description of the official train/dev/test split and the MetaMap-extracted phrase files. Fetched 2026-08-11.

[3] The corpus screening row for `GBaker/MedQA-USMLE-4-options`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[4] Hugging Face Hub API record for GBaker/MedQA-USMLE-4-options. https://huggingface.co/api/datasets/GBaker/MedQA-USMLE-4-options?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date, sibling files. Fetched 2026-08-11.

[5] GBaker/MedQA-USMLE-4-options-hf dataset card (README). https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf/raw/main/README.md - licence frontmatter (cc-by-sa-4.0), citation to the origin paper. Fetched 2026-08-11.

[6] datasets-server info and size endpoints for GBaker/MedQA-USMLE-4-options-hf. https://datasets-server.huggingface.co/info?dataset=GBaker%2FMedQA-USMLE-4-options-hf and https://datasets-server.huggingface.co/size?dataset=GBaker%2FMedQA-USMLE-4-options-hf - schema (`sent1`/`sent2`/`ending0..3`/`label`), split row counts (10,178/1,272/1,273); a sampled `train` row via https://datasets-server.huggingface.co/first-rows?dataset=GBaker%2FMedQA-USMLE-4-options-hf&config=default&split=train confirms identical question/options content to this repo's row 0. Fetched 2026-08-11.

[7] datasets-server size and first-rows endpoints for GBaker/MedQA-USMLE-4-options, `train` split. https://datasets-server.huggingface.co/size?dataset=GBaker%2FMedQA-USMLE-4-options and https://datasets-server.huggingface.co/first-rows?dataset=GBaker%2FMedQA-USMLE-4-options&config=default&split=train - row/byte counts, sampled row, `meta_info`/`answer_idx` value counts over the first 100 rows at offset 0. Fetched 2026-08-11.

[8] datasets-server info and first-rows endpoints for GBaker/MedQA-USMLE-4-options, `test` split. https://datasets-server.huggingface.co/info?dataset=GBaker%2FMedQA-USMLE-4-options and https://datasets-server.huggingface.co/first-rows?dataset=GBaker%2FMedQA-USMLE-4-options&config=default&split=test - column schema, sampled row, `meta_info`/`answer_idx` value counts over the first 100 rows at offset 0. Fetched 2026-08-11.

[9] Hugging Face Hub model search for datasets tagged with this repository's id. https://huggingface.co/api/models?search=MedQA-USMLE-4-options - live search, unpinned; returned only two zero-download, zero-like community fine-tunes. Fetched 2026-08-11.

[10] bigbio/med_qa dataset card (README). https://huggingface.co/datasets/bigbio/med_qa/raw/main/README.md - homepage (https://github.com/jind11/MedQA), per-language question counts, licence "unknown". The dataset viewer could not serve this repo's rows because it runs a Python loading script (datasets-server `/info` and `/size` returned an error to that effect). Fetched 2026-08-11.

[11] medalpaca/medical_meadow_medqa datasets-server info and size endpoints. https://datasets-server.huggingface.co/info?dataset=medalpaca%2Fmedical_meadow_medqa and https://datasets-server.huggingface.co/size?dataset=medalpaca%2Fmedical_meadow_medqa - schema (`input`/`instruction`/`output`), 10,178-row single `train` split. Fetched 2026-08-11.

[12] medalpaca/medical_meadow_medqa first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=medalpaca%2Fmedical_meadow_medqa&config=default&split=train - sampled row shows a five-option (A-E) version of the same question as this repo's row 0. Fetched 2026-08-11.

[13] GBaker/MedQA-USMLE-4-options dataset card (README). https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/raw/main/README.md - citation to the origin paper, licence frontmatter, no further build detail. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for SFT-style training on multiple-choice medical QA, restricted to `train`: the `test` split is the split reported as this benchmark's score, per the screening note, so it must be held out of training [3]. No licence or content restriction beyond that split boundary was found in the fetched sources.

### The screening row

The row's own note [3]: "the standard 4-option MedQA/USMLE set; train (10,178) is safe, `test` is the number people report." The row carries no flag.
