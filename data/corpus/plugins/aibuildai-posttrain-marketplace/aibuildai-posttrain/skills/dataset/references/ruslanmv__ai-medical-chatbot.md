# ruslanmv/ai-medical-chatbot

256,916 single-turn patient/doctor consultations - a question summary plus the full patient message and the doctor's reply - built from the UCSD MedDialog-EN corpus of real online medical consultations.

**ruslanmv/ai-medical-chatbot** packages patient-doctor text consultations into a flat `Description`/`Patient`/`Doctor` table for chatbot fine-tuning [1]. The repository's own build notebook says the rows are produced by merging per-file CSVs found under a folder named `Medical-Dialogue-System` into one `dialogues.csv` of 256,916 rows [2] - the name and shape match `MedDialog-EN`, the English half of the MedDialog corpus introduced by He et al., which the paper describes as 257,454 patient-doctor consultations crawled from icliniq.com and healthcaremagic.com, each consultation holding a description of the patient's condition plus the patient/doctor exchange [3]. **The same two source sites (icliniq.com and healthcaremagic.com) were independently scraped by the ChatDoctor project for its `HealthCareMagic-100k` (~100k conversations) and `iCliniq` (~10k conversations) datasets [4], so training on this dataset alongside either of those risks overlapping source material even though no exact-duplicate rows were confirmed here** - see Hold out.

**Use it for**: single-turn SFT on medical Q&A - the `Description` (question summary) and `Patient` (full message) fields form the prompt, `Doctor` the target reply; this is a prompt-completion shape, not a preference pair (no chosen/rejected column exists) [1]. Column names are capitalized (`Description`, `Patient`, `Doctor`) rather than the lowercase `prompt`/`completion` most SFT trainers expect, so rename or map them first. See the SFT method card. Screen against `lavita/ChatDoctor-HealthCareMagic-100k` and `lavita/ChatDoctor-iCliniq` before combining, per the source-overlap restriction above.

**Licence**: not stated - the dataset repository's `cardData` carries no `license` field and no `license:` tag, and the README states none [1]. Ungated (`"gated": false`, `"private": false`) [5]. The catch: the linked GitHub project is Apache-2.0 licensed, but that covers the application code, not this data repository, and no source states a data licence or reuse grant [6].

**Shape**: 256,916 rows, one config (`default`), one split (`train`), three string columns [7][8].

**Hold out**: no internal eval split exists (train only, 256,916 rows) [8]. The hold-out risk here is external: this corpus and `lavita/ChatDoctor-HealthCareMagic-100k` (112,165 rows) and `lavita/ChatDoctor-iCliniq` (7,321 rows) were scraped from the same two websites by different teams [3][4], so a model evaluated on either ChatDoctor set after training on this one (or vice versa) risks source-level contamination; no source states row-level deduplication between them, and none was confirmed in the sampling done for this card (see Quality).

**Origin**: built and published by ruslanmv from a real patient-doctor consultation scrape (human-authored questions and answers, not model-generated) [2][3]. Hub API at the check date: `downloads` 3,416, `downloadsAllTime` 210,797, `likes` 249 [5].

**Trained-on-by**: `ruslanmv/Medical-Llama3-8B`, whose model card names this dataset directly as its fine-tuning data [9], with several public derivatives and GGUF/GPTQ conversions built on it (e.g. `sethuiyer/Medichat-Llama3-8B`, `QuantFactory/Medichat-Llama3-8B-GGUF`) found via the Hub's dataset-usage listing for this repository [10].

**Introduced by**: no paper for this specific repackaging - the dataset card [1] and the builder's own data-construction notes [2]; its content traces to He et al.'s MedDialog corpus paper [3].

## Shape

Rows and split, from the datasets-server size endpoint [7]:

| split | rows |
| --- | --- |
| `train` | 256,916 |

One config, `default`, three columns, all strings (datasets-server `/info`) [8]:

| column | dtype |
| --- | --- |
| `Description` | string |
| `Patient` | string |
| `Doctor` | string |

Byte sizes (datasets-server `/size`) [7]: 141,665,910 bytes of Parquet on disk, 268,685,077 bytes decoded in memory. No source states sequence-length or token statistics for this release; the MedDialog-EN paper gives a related but not identical figure for its own (larger, 257,454-row) source corpus: 514,908 total utterances, split evenly between doctor and patient turns [3].

## Quality

- Of the first 100 rows read at offset 0 in `train`, only 71 are distinct `(Description, Patient, Doctor)` triples: the other 29 are exact repeats, falling into five groups of byte-identical rows (row indices 0/8/25/41/58/68/74/91 repeat the same abutment-of-the-nerve-root triple eight times; four more groups repeat six or seven times each), so 34 of the 100 sampled rows are exact within-dataset duplicates of one of these five triples [11]. No source states a deduplication step in the build pipeline [2], and no source states a corpus-wide duplicate rate; this 34-of-100 figure is this card's own count from the 100 rows read at offset 0 and is not extrapolated beyond that sample.
- Of those same 100 rows, all 100 have non-empty `Description`, `Patient`, and `Doctor` fields; Doctor-field length in that sample ranges from 54 to 1,714 characters [11].
- Of those same 100 rows, 66 `Doctor` replies end in a dangling `-->` arrow (for example, "...consult a neurologist online -->"), which reads as a leftover HTML/link fragment from the original scrape rather than a cut-off sentence, since the text before it is otherwise complete [11]. No source documents this artifact or a cleaning pass that removes it; it is reported here as read, from 100 rows at offset 0 only.
- No source states a measured contamination rate or annotator-agreement figure for this release. A keyword search of `lavita/ChatDoctor-HealthCareMagic-100k` and `lavita/ChatDoctor-iCliniq` for terms from one of these sampled rows ("abutment"/"abutting") returned thousands of topically related rows in each (7,024 and 751 of their respective totals) [12], showing both corpora cover the same kind of spine-pain queries at the source-website level, established already in the opening paragraph; this was a keyword search, not a row-by-row exact-match comparison, so it does not show whether any specific row is an exact duplicate across datasets.

## Load it

Only one split exists; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-03-23) [5]:

```python
import datasets

REV = "138c99336a3afce0df88ffe6fd67bd231df25d36"  # main at the check date
train = datasets.load_dataset("ruslanmv/ai-medical-chatbot", revision=REV, split="train")  # 256,916 rows
```

**Trap**: the columns are `Description`, `Patient`, `Doctor` - capitalized, and with no `prompt`/`completion` or `chosen`/`rejected` equivalents - so a trainer expecting lowercase field names (trl's SFT/DPO trainers, for instance) will not find its expected columns until you rename or map these three yourself [1][8].

## Neighbors

- `lavita/ChatDoctor-HealthCareMagic-100k` - 112,165 rows in `instruction`/`input`/`output` form, independently scraped from healthcaremagic.com for the ChatDoctor project; source-site overlap with this dataset, not a lineage descendant of it [4][13].
- `lavita/ChatDoctor-iCliniq` - 7,321 rows in `input`/`answer_icliniq`/`answer_chatgpt`/`answer_chatdoctor` form, independently scraped from icliniq.com; same source-site overlap [4][13].
- `ruslanmv/icliniq-7k` - published by the same account as this dataset, but its row count and byte sizes are byte-identical to `lavita/ChatDoctor-iCliniq` above (7,321 rows, 9,373,080 original bytes), so it is a re-upload of that ChatDoctor file, not part of this dataset's own lineage [14].
- `DrBenjamin/ai-medical-chatbot` - the same row count (256,916) and the same column count (three) as this dataset; its README carries no `dataset_info`/`features` block, so its actual column names and dtypes were not confirmed [15]. Its card adds an `apache-2.0` license tag that this repository's own metadata does not carry [15]. Prefer the original here for provenance; use the mirror only if the added licence tag matters to you, since no source confirms it was actually granted by the original builder.

This corpus prefers this original release: it is the one the widely-adopted `ruslanmv/Medical-Llama3-8B` model card names directly [9], and the mirror above adds a licence claim not present upstream.

## A row

One config, one split, one schema. From `config="default"`, `split="train"`, `row_idx=1` (datasets-server `/first-rows`) [11]:

```json
{
  "Description": "Q. What should I do to reduce my weight gained due to genetic hypothyroidism?",
  "Patient": "Hi doctor, I am a 22-year-old female who was diagnosed with hypothyroidism (genetic) when I was 12. Over the past five years, I have become around 50 pounds overweight and all of my attempts to lose have seemed to fail so I have given up, but my weight has stayed the same. There is so much information put there about losing weight with hypothyroidism but it all seems to conflict. [...] What can I do? I am currently on Levothyroxine, Buspar, and Benedryl.",
  "Doctor": "Hi. You have really done well with the hypothyroidism problem. Your levels are normal with less medications which are very good. [...] Follow up after 15 days."
}
```

## Where it came from

Published by ruslanmv (Ruslan Magana Vsevolodovna). The account's own GitHub project describes the build pipeline: raw per-file CSVs are loaded from a directory tree named `Medical-Dialogue-System`, cleaned, and merged into a single `dialogues.csv` of 256,916 dialogues [2]. That directory name and near-matching row count point to He et al.'s MedDialog-EN, the English half of the MedDialog corpus, described as 257,454 real patient-doctor consultations crawled from the online consultation platforms icliniq.com and healthcaremagic.com, each pairing a patient's condition description with the patient/doctor exchange [3]. No source explains the 538-row difference between MedDialog-EN's stated 257,454 and this repository's 256,916; it is consistent with the cleaning/merge step the builder describes but is not itself documented [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] ruslanmv/ai-medical-chatbot dataset card (README) and `cardData`. https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot - column schema, split declaration, absence of a license field, no stated usage restriction. Fetched 2026-08-11.

[2] ruslanmv/ai-medical-chatbot GitHub project, data-construction notes. https://raw.githubusercontent.com/ruslanmv/ai-medical-chatbot/master/2-Data/README.md - describes loading CSVs from a `Medical-Dialogue-System` folder and merging them into `dialogues.csv` (256,916 dialogues). Fetched 2026-08-11.

[3] He, Chen, Ju, Dong, Fang, Wang, Yang, Zeng, Zhang, Zhang, Zhou, Zhu, Xie, "MedDialog: Two Large-scale Medical Dialogue Datasets", 2020. https://arxiv.org/abs/2004.03329 - MedDialog-EN's 257,454 consultations, crawl sources icliniq.com and healthcaremagic.com, consultation structure, utterance counts. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2004.03329). Fetched 2026-08-11.

[4] Li, Li, Zhang, Dan, Jiang, Zhang, "ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge", Cureus, 2023. https://arxiv.org/abs/2303.14070 - ~100k conversations scraped from healthcaremagic.com (`HealthCareMagic100k`) and ~10k from iCliniq.com, used to build `HealthCareMagic-100k` and the `iCliniq` test set. Read via the arXiv PDF (https://arxiv.org/pdf/2303.14070v5). Fetched 2026-08-11.

[5] Hugging Face Hub API record for ruslanmv/ai-medical-chatbot. https://huggingface.co/api/datasets/ruslanmv/ai-medical-chatbot?full=true - `gated`, `private`, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] ruslanmv/ai-medical-chatbot GitHub project README (license badge). https://raw.githubusercontent.com/ruslanmv/ai-medical-chatbot/master/README.md - Apache 2.0 license badge shown for the application/project code. Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=ruslanmv%2Fai-medical-chatbot Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=ruslanmv%2Fai-medical-chatbot Fetched 2026-08-11.

[9] ruslanmv/Medical-Llama3-8B model card. https://huggingface.co/ruslanmv/Medical-Llama3-8B/raw/main/README.md - names `ruslanmv/ai-medical-chatbot` as its training dataset (`datasets:` field and prose). Fetched 2026-08-11.

[10] Hugging Face Hub API model listing filtered by this dataset. https://huggingface.co/api/models?filter=dataset:ruslanmv/ai-medical-chatbot&limit=50 - lists models and derivatives declaring use of this dataset. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=ruslanmv%2Fai-medical-chatbot&config=default&split=train - 100 rows at offset 0, used for the sampled row shown and the Quality-section counts. Fetched 2026-08-11.

[12] datasets-server search endpoint, keyword query "abutment"/"abutting" against `lavita/ChatDoctor-HealthCareMagic-100k` and `lavita/ChatDoctor-iCliniq`. https://datasets-server.huggingface.co/search?dataset=lavita%2FChatDoctor-HealthCareMagic-100k&config=default&split=train&query=abutment%20of%20the%20nerve%20root , https://datasets-server.huggingface.co/search?dataset=lavita%2FChatDoctor-iCliniq&config=default&split=train&query=abutment%20of%20the%20nerve%20root - a fuzzy keyword search returning `num_rows_total` counts (7,024 and 751), not an exact-row-match comparison. Fetched 2026-08-11.

[13] Neighbor dataset cards and size endpoint for `lavita/ChatDoctor-HealthCareMagic-100k` and `lavita/ChatDoctor-iCliniq`: https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k/raw/main/README.md , https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=lavita%2FChatDoctor-HealthCareMagic-100k , https://datasets-server.huggingface.co/size?dataset=lavita%2FChatDoctor-iCliniq - row counts and column schemas. Fetched 2026-08-11.

[14] ruslanmv/icliniq-7k dataset card and size endpoint. https://huggingface.co/datasets/ruslanmv/icliniq-7k/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=ruslanmv%2Ficliniq-7k - identical schema, row count, and byte sizes to `lavita/ChatDoctor-iCliniq`. Fetched 2026-08-11.

[15] DrBenjamin/ai-medical-chatbot dataset card and size endpoint. https://huggingface.co/datasets/DrBenjamin/ai-medical-chatbot/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=DrBenjamin%2Fai-medical-chatbot - identical row count (256,916) and column count (3) to this dataset; the README's front matter carries only a bare `license: apache-2.0` line with no `dataset_info`/`features` block, so column names/dtypes were not confirmed. Fetched 2026-08-11.

[16] The corpus screening row for `ruslanmv/ai-medical-chatbot`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as human-authored, single-turn SFT data for medical question-answering, with a source-overlap caveat rather than an internal split to hold out: the dataset's own build trail ties it to the same icliniq.com/healthcaremagic.com scrape that also underlies the ChatDoctor `HealthCareMagic-100k` and `iCliniq` datasets (established above) [2][3][4], so the screening row's note to treat it as overlapping ChatDoctor's sources [16] is confirmed at the source-website level. No exact-duplicate cross-dataset row check was attempted for this card, only the keyword search reported in Quality [12], so whether any specific row is shared verbatim with either ChatDoctor set remains unchecked.

### The screening row

The row's own note [16]: "~250k real patient question / doctor answer consultations scraped from iCliniq-style sites; overlaps ChatDoctor's sources." The row carries no flag.
