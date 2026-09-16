# HPAI-BSC/medical-specialities

13,775 four-option medical multiple-choice questions, merged from nine dataset splits across five medical QA benchmark families (test splits, plus one validation split) and sorted into 35 medical-specialty buckets by an LLM classifier.

**HPAI-BSC/medical-specialities** was built by the Barcelona Supercomputing Center's HPAI group as an evaluation resource for medical language models: it merges questions from CareQA, HeadQA, MedQA-USMLE, MedMCQA and five MMLU medical subsets into one schema, then uses Llama-3-70B-Instruct to classify each question, with a chain-of-thought prompt, into one of 35 specialty categories, so a model's per-specialty accuracy can be inspected directly [1]. It is cited from the paper "The Aloe Family Recipe for Open and Specialized Healthcare LLMs" [2] via the Hub repository's `arxiv:2505.04388` tag [3], though the paper itself is about the Aloe Beta model family and its training/evaluation recipe, not a description of this dataset specifically. **This is an evaluation-only merge of other benchmarks' held-out splits (test splits for CareQA, HeadQA, MedQA and the five MMLU-medical subsets; the validation split for MedMCQA) [1]: decontaminate any training corpus against this dataset and against its nine source splits before scoring a model on it.** It lives at https://huggingface.co/datasets/HPAI-BSC/medical-specialities .

**Use it for**: multiple-choice medical QA evaluation, one config per specialty, so a harness can report accuracy broken out by specialty; never for training (see the restriction above). The data format is four-option multiple choice - `question`, `op1`-`op4`, and `cop` as the 1-indexed correct option [1] - not a chat or preference format, so it maps to a multiple-choice QA eval harness rather than the SFT or DPO method cards.

**Licence**: not stated - the Hub repository's `cardData` carries no `license` key, and the README body (fetched in full) states no license line or section [1][3]. Ungated: the Hub API reports `gated: false` and `private: false`, so the repository can be loaded without an access request [3].

**Shape**: 13,775 rows across 35 configs, one `test` split each, 11 string/int/float columns per row [4][5].

**Hold out**: all 13,775 rows, across every config - the entire dataset must be excluded from training, because it is itself a merge of other benchmarks' test (and one validation) splits, listed above. There is no separate held-out portion within the dataset because the whole thing is a held-out eval set already [1].

**Origin**: built by HPAI-BSC (Barcelona Supercomputing Center); questions come from nine dataset splits across five existing human-authored medical QA benchmark families, and the specialty label on each question is a machine judgment from Llama-3-70B-Instruct, not a human annotation [1]. Hub API as of 2026-08-12: `downloads` 244, `downloadsAllTime` 12,145, `likes` 7 [3][6].

**Trained-on-by**: none found. The Aloe Beta 8B model card lists its training-data collections explicitly and does not name `medical-specialities` among them [7], consistent with this dataset being built from other benchmarks' test/validation splits and intended for evaluation rather than training [1].

**Introduced by**: no paper describes this dataset specifically - the dataset card [1] is the only source for its construction; the Hub repository links to the Aloe Family paper [2] as its citation target, but that paper is about the resulting Aloe Beta models, not this merge.

## Shape

Rows served and splits, one `test` split per specialty config, from the datasets-server `/size` endpoint [4]:

| config | rows |
| --- | --- |
| Allergy | 100 |
| Anatomy | 594 |
| Anesthesiology | 163 |
| Biochemistry | 1,679 |
| Cardiology | 440 |
| Chemistry | 518 |
| Dermatology | 188 |
| Emergency | 202 |
| Endocrinology | 399 |
| Gastroenterology | 423 |
| Genetics | 544 |
| Geriatrics | 69 |
| Gynecology | 136 |
| Hematology | 509 |
| Microbiology | 959 |
| Nephrology | 273 |
| Neurology | 457 |
| Nursing | 198 |
| Obstetrics | 354 |
| Odontology | 1,001 |
| Oncology | 245 |
| Ophthalmology | 140 |
| Orthopedics | 217 |
| Otorhinolaryngology | 209 |
| Pathology | 97 |
| Pediatrics | 294 |
| Pharmacology | 808 |
| Physiology | 297 |
| Psychiatry | 942 |
| Psychology | 484 |
| Radiology | 89 |
| Respiratory | 350 |
| Rheumatology | 109 |
| Surgery | 178 |
| Urology | 110 |

Total across the 35 configs: 13,775 [4]. Every config carries the same 11 columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `id` | string |
| `question` | string |
| `op1` | string |
| `op2` | string |
| `op3` | string |
| `op4` | string |
| `cop` | int64 |
| `dataset` | string |
| `medical_field` | string |
| `cot_medical_field` | string |
| `cumulative_logprob_cot_medical_field` | float64 |

The repository also carries a 36th file, `None.json`, which is not exposed as a Hub config or counted in the totals above; it holds the 936 questions the classifier could not confidently sort into any of the 35 specialties, each labeled `medical_field: "None"` [8]. No source states sequence-length or token statistics for this dataset; "not stated" here, not invented. Original download size across the 35 served configs is 12,635,663 bytes, 4,948,315 bytes as Parquet, 9,172,879 bytes decoded in memory [4].

## Quality

- The specialty label (`medical_field`) on every row is a machine classification, not a human annotation: the dataset card states it was produced by prompting Llama-3-70B-Instruct with a system prompt and few-shot examples, asking for a step-by-step chain-of-thought ending in "The category is: <category>", extracted with a regex [1]. The `cot_medical_field` column holds that reasoning text and `cumulative_logprob_cot_medical_field` holds the model's cumulative log-probability for the generated classification, per row [1][9].
- No source states a measured classification-accuracy or human-validation rate for the specialty labels; the process is fully automated per the dataset card [1].
- The 936 rows the classifier routed to "None" (the underlying question was judged unclear, incomplete, or not clearly medical, per the classifier's own chain-of-thought text on those rows) are excluded from the 35 served configs, so the served 13,775 rows are the subset the classifier placed with confidence into a named specialty [8].
- The `dataset` column names the exact source pool per question; the two configs sampled here (`Cardiology`, `Urology`) each drew only from `mmlu_professional_medicine_test` and `medqa_4options_test` respectively in their first ten rows, matching the dataset card's list of nine constituent source splits and confirming rows carry their provenance rather than being anonymized [1][9].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main`, last modified 2025-11-18) [3]:

```python
import datasets

REV = "bb9f40beb4f19b81d404d3f8f094aab4606e27e1"  # main at the check date
cardiology = datasets.load_dataset("HPAI-BSC/medical-specialities", "Cardiology", revision=REV, split="test")  # 440 rows
```

**Trap**: there is no single default config - each of the 35 specialties is its own config name (e.g. `"Cardiology"`, `"Urology"`), and each has exactly one split, `test`; loading without naming a config fails. The 36th file, `None.json`, is not registered as a config at all and cannot be loaded through `load_dataset` with a config name - it must be fetched directly from the repository if wanted [1][5][8].

## Neighbors

None found: a Hub search for "medical-specialities" returns only this repository [10]. It is not a re-release of any single upstream dataset but a merge of nine splits from five benchmark families; those source splits remain independently available on the Hub - CareQA (`HPAI-BSC/CareQA`, test split `CareQA_en`) [11], HeadQA and MedMCQA (`openlifescienceai/headqa` test split, `openlifescienceai/medmcqa` validation split), MedQA (`GBaker/MedQA-USMLE-4-options-hf` test split), and five MMLU medical subsets (`openlifescienceai/mmlu_anatomy`, `mmlu_clinical_knowledge`, `mmlu_college_medicine`, `mmlu_medical_genetics`, `mmlu_professional_medicine`, each test split) [1]. Choosing one of those instead gives an unclassified, single-source benchmark; choosing this dataset instead gives the same underlying questions pre-sorted by specialty and merged into one schema.

## A row

All 35 configs share the same 11-column schema, so one row covers the served shape. From config `Cardiology`, split `test`, row index 0 (datasets-server `/first-rows`) [9]:

```json
{
  "id": "8cba7b22-c808-45e3-856c-05fc9f133ab7",
  "question": "A 9-year-old boy is brought to the office by his parents for a well-child examination. [...] Cardiac examination discloses a grade 3/6 systolic murmur audible along the left sternal border at the third and fourth intercostal spaces. Femoral pulses are weak and brachial pulses are strong; there is a radiofemoral delay. Chest xray discloses mild cardiomegaly with left ventricular prominence. ECG shows left ventricular hypertrophy. This patient is at greatest risk for which of the following complications?",
  "op1": "Atrial fibrillation",
  "op2": "Cor pulmonale",
  "op3": "Systemic hypertension",
  "op4": "Tricuspid valve regurgitation",
  "cop": 3,
  "dataset": "mmlu_professional_medicine_test",
  "medical_field": "Cardiology",
  "cot_medical_field": "The question describes a patient with a heart murmur, cardiomegaly, and left ventricular hypertrophy, which are all related to heart conditions. Heart conditions are categorized under Cardiology. The category is: Cardiology",
  "cumulative_logprob_cot_medical_field": -3.778287880122661
}
```

`id` format varies by source pool within a config - this MMLU-sourced row uses a UUID, while MedQA-sourced rows in the same repository use a `test-XXXXX` counter, as seen in `Urology/test` row 0 (`id: "test-00006"`, `dataset: "medqa_4options_test"`) [9]; the column set and types are identical either way.

## Where it came from

Built by HPAI-BSC. Per the dataset card's "Dataset Creation" section, the pipeline downloads nine dataset splits from five medical QA benchmark families from the Hub, classifies each question into one of 35 predefined medical fields using Llama-3-70B-Instruct with a chain-of-thought prompt and regex-extracted final category (falling back to "None" when the model is unsure or the question does not fit), then merges the results into per-specialty files [1]. The nine source splits are: CareQA (`CareQA_en.json`) [11], HeadQA's test split, MedMCQA's validation split, MedQA-USMLE-4-options' test split, and five MMLU medical test splits (anatomy, clinical knowledge, college medicine, medical genetics, professional medicine, each a test split) [1]. The dataset card points to a companion GitHub repository, https://github.com/HPAI-BSC/medical-specialities, for the full construction code [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] HPAI-BSC/medical-specialities dataset card (README). https://huggingface.co/datasets/HPAI-BSC/medical-specialities/raw/main/README.md - dataset summary, data fields, construction pipeline, prompt configuration, source-dataset list, citation. Fetched 2026-08-12.

[2] Garcia-Gasulla et al., "The Aloe Family Recipe for Open and Specialized Healthcare LLMs", 2025. https://arxiv.org/abs/2505.04388 - title and abstract read from the live abs page. Fetched 2026-08-12.

[3] Hugging Face Hub API record for HPAI-BSC/medical-specialities. https://huggingface.co/api/datasets/HPAI-BSC/medical-specialities?full=true - `sha`, `cardData`, `downloads`, `likes`, `lastModified`. Fetched 2026-08-12.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=HPAI-BSC%2Fmedical-specialities - per-config row counts and byte sizes. Fetched 2026-08-12.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=HPAI-BSC%2Fmedical-specialities - per-config column schema. Fetched 2026-08-12.

[6] Hugging Face Hub API record with `downloadsAllTime` expansion. https://huggingface.co/api/datasets/HPAI-BSC/medical-specialities?expand[]=downloadsAllTime Fetched 2026-08-12.

[7] HPAI-BSC/Llama3.1-Aloe-Beta-8B model card. https://huggingface.co/HPAI-BSC/Llama3.1-Aloe-Beta-8B/raw/main/README.md - `cardData.datasets` list of training-data collections, which does not include `medical-specialities`. Fetched 2026-08-12.

[8] `None.json` from the HPAI-BSC/medical-specialities repository. https://huggingface.co/datasets/HPAI-BSC/medical-specialities/resolve/main/None.json - 936 rows, each with `medical_field: "None"`, not served as a Hub config. Fetched 2026-08-12.

[9] datasets-server first-rows endpoint, called once per config. https://datasets-server.huggingface.co/first-rows?dataset=HPAI-BSC%2Fmedical-specialities&config=Cardiology&split=test and the same with `config=Urology`. Fetched 2026-08-12.

[10] Hugging Face Hub dataset search API. https://huggingface.co/api/datasets?search=medical-specialities&limit=20 - live search, no revision parameter. Fetched 2026-08-12.

[11] HPAI-BSC/CareQA dataset card (README). https://huggingface.co/datasets/HPAI-BSC/CareQA/raw/main/README.md - confirms `CareQA_en` is a `test`-split config. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as evaluation data only, and only outside any training corpus. The two facts that decide it are already established above: the dataset is a merge of other benchmarks' test (and one validation) splits, sorted into specialties by a Llama-3-70B-Instruct classifier, with no human relabeling [1], and its own screening note calls it a pure evaluation set that carries other benchmarks' test items.

### The screening row

The row's own note: "test-only merge of the HeadQA, MedQA, MMLU-medical and related TEST splits, sorted into 35 specialties by Llama-3-70B-Instruct; pure evaluation and it carries other benchmarks' test items." The row carries no flag.
