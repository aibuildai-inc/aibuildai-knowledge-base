# CohereForAI/aya_dataset

205,568 human-written prompt-completion pairs across 65 languages, crowd-sourced by volunteer annotators through an open web platform, one of the two core releases from the Aya multilingual instruction-tuning project.

**CohereForAI/aya_dataset** is a human-annotated multilingual instruction-following dataset built by contributors to the Aya Open Science Initiative through the Aya Annotation Platform, introduced in "Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning" [1]. Each row is either an original prompt-completion pair written from scratch by an annotator, or a human re-annotation (edit) of a machine-translated or templated prompt-completion pair drawn from existing open-source NLP datasets [1][2]. The repository is served under the id `CohereForAI/aya_dataset`, but the Hub now resolves it to `CohereLabs/aya_dataset` - every API and download call in this card followed that redirect [3]. It lives at https://huggingface.co/datasets/CohereLabs/aya_dataset . **Hold out the 1,750-row `test` split before any scored run: its rows are, byte-for-byte on the sampled row, the same text as the `aya_human_annotated` config of the sibling `CohereLabs/aya_evaluation_suite` eval set, so training on `test` would contaminate that published benchmark [4][5].**

**Use it for**: SFT on prompt-completion pairs - each row is already `inputs`/`targets`, a direct fit for the SFT method card's single-turn chat format, with the restriction that the `test` split must stay held out for evaluation, not training [2][6]. The separate `demographics` config holds anonymized per-annotator metadata (age range, gender, country, languages, dialects), not training text, and is not part of the instruction-tuning shape [2].

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [3]. The one catch: the README's own licensing note says the data "can be used for any purpose, whether academic or commercial" [2] - no restriction beyond the license itself is stated for the `default` config; the `demographics` config carries anonymized annotator attributes (age, gender, country) rather than free text, which the card describes only as anonymized, with no further use restriction given [2].

**Shape**: 205,568 rows total, two configs - `default` (`train` 202,362 / `test` 1,750, 6 columns) and `demographics` (`train` 1,456, 6 columns) [6][7].

**Hold out**: `test` (1,750 rows) - held out because it duplicates a published eval set, `CohereLabs/aya_evaluation_suite`'s `aya_human_annotated` config, confirmed by comparing sampled rows from both [4][5]. `demographics` is not training text and is not a contamination risk either way [2].

**Origin**: built by Cohere Labs and the volunteer contributors of the Aya Open Science Initiative via the Aya Annotation Platform; both prompts and completions are human-written, not model-generated [1][2]. Hub API at the check date (2026-08-11): `downloads` 15,866, `downloadsAllTime` 141,328, `likes` 364 [3].

**Trained-on-by**: the Aya-101 model (`CohereForAI/aya-101`) - its own model card lists "Aya Dataset" among the datasets it was fine-tuned on, alongside xP3x, Aya Collection, a Data Provenance subset, and ShareGPT-Command [8]. Aya-101 is the model introduced in "Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model," which reports it outperforms mT0 and BLOOMZ on the majority of tasks it was evaluated on [9].

**Introduced by**: [1] (Singh et al.).

## Shape

Rows and columns served, live at the check date (datasets-server `/size` and `/info`) [6][7]:

| config | split | rows | columns |
| --- | --- | --- | --- |
| `default` | `train` | 202,362 | 6 |
| `default` | `test` | 1,750 | 6 |
| `demographics` | `train` | 1,456 | 6 |

`default` columns: `inputs` (string), `targets` (string), `language` (string), `language_code` (string), `annotation_type` (string), `user_id` (string) [7]. `demographics` columns: `user_id` (string), `age_range` (sequence of int64), `gender` (string), `country` (string), `languages` (sequence of string), `dialects` (sequence of string) [7].

Byte sizes disagree between two sources and neither supersedes the other. The README's own YAML metadata, read at the pinned commit `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`, states `default` download_size 275,359,572 bytes and dataset_size (in-memory) 256,374,059 bytes [2]. The live datasets-server `/size` endpoint, which takes no revision parameter, reports `default` at 138,173,317 bytes of parquet and 229,536,913 bytes in memory - roughly half the README's download-size figure [6]. Row counts agree exactly between the two (202,362 / 1,750 / 1,456); only the byte totals differ. No source states sequence-length or token statistics for either config.

The README's own prose tables are internally inconsistent with its own YAML metadata by a small margin: the "Data Splits" markdown table states 202,364 `train` rows (two more than the YAML's 202,362 and the live count) [2], and the "Statistics" table's annotation-type breakdown sums to 204,114 (138,844 original + 65,270 re-annotations), two more than the `default` config's actual row total of 204,112 (202,362 + 1,750) [2][6].

## Quality

- Both prompts and completions are entirely human-authored or human-edited: original annotations are new prompt-completion pairs written from scratch by annotators, and re-annotations are human edits of machine-translated or templated source pairs [2].
- The origin paper describes one explicit dedup control: re-annotations were only accepted into the release if the character-level Levenshtein edit distance between the original machine-generated pair and the human-edited pair was at least 5, specifically to avoid releasing duplicates of the source data [1].
- The paper also describes a per-language inclusion floor: a language was only included in the final release once it had reached at least 50 contributions, chosen as a tradeoff between per-language data quality and covering more languages [1].
- The README's "Known Limitations" section states the Aya Annotation Platform had no dedicated toxic-speech flag and relied on human verification and peer review instead, with no guarantee that all offensive content was removed, and that the platform lacks a re-labeling capability, so some rows may carry incorrect language labels or non-compliant formatting [2].
- No source states a measured contamination rate or inter-annotator agreement figure for this dataset; none is invented here.

## Load it

Pin the revision this card's numbers were read at - the commit sha shared by the Hub API's `sha` field and the shortlist's pinned commit - and note that the repository id has changed:

```python
import datasets

REV = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"  # pinned commit, main at the check date
train = datasets.load_dataset("CohereLabs/aya_dataset", revision=REV, split="train")  # 202,362 rows
test = datasets.load_dataset("CohereLabs/aya_dataset", revision=REV, split="test")    # 1,750 rows - hold out
demographics = datasets.load_dataset("CohereLabs/aya_dataset", "demographics", revision=REV, split="train")  # 1,456 rows, anonymized annotator metadata
```

**Trap**: the shortlist and this card's filename carry the id `CohereForAI/aya_dataset`, but that id now redirects to `CohereLabs/aya_dataset` - both the Hub API and the datasets-server endpoints used for this card resolved through that redirect [3]. `load_dataset("CohereForAI/aya_dataset", ...)` still works because the Hub follows the redirect, but the canonical id to write in new code is `CohereLabs/aya_dataset`. Passing no config name loads `default`; the `demographics` config must be requested explicitly as the second positional argument, as shown above, or it will not load [2].

## Neighbors

The README's own "Aya Datasets Family" table names the sibling releases from the same project; this card reports the row counts fetched live from each at the check date [2][10][5][11].

- `CohereLabs/aya_collection` - a much larger sibling built by applying instruction templates to 44 existing datasets and translating 19 of them into 101 languages; 513,755,590 rows live, versus 205,568 here [2][10]. Prefer `aya_dataset` when only human-written data is wanted; `aya_collection` is templated/translated, not human-authored, outside its own re-annotation subset.
- `CohereLabs/aya_evaluation_suite` - an evaluation set, not for training. Its `aya_human_annotated` config holds exactly 1,750 rows, and the first served row (`inputs`/`targets` for a Standard Arabic prompt about a companion of the Prophet) is identical in text to the first row of this dataset's `test` split, confirming the `test` split is this eval set's source data [4][5]. Its other two configs, `dolly_human_edited` (1,200 rows) and `dolly_machine_translated` (23,800 rows), are unrelated Dolly-derived evaluation data, not drawn from `aya_dataset` [5].
- `CohereLabs/aya_redteaming` - a red-teaming prompt set in 8 languages, per the README's family table [2]; its Hub dataset viewer is disabled, so no live row count could be fetched for this card [11].
- `CohereLabs/aya_collection_language_split` - the `aya_collection` data restructured into per-language subsets, per the README's family table; not independently checked here [2].

## A row

Two distinct schemas are served: `default` (`train`/`test`, sharing one schema) and `demographics` (`train` only). One sample of each, fetched live at the check date.

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`), with the Somali song lyrics in `targets` truncated [12]:

```json
{
  "inputs": "Heestan waxaa qada Khalid Haref Ahmed \nOO ku Jiray Kooxdii Dur Dur!",
  "targets": "Habeen ma hurdoo\nAday horjoogoo\nDharaar ma hargalo\nAduun baabay helayee\nRuntii ku helayoo\nCaawaan iman iman\nOonkaan u liitay\nIga ba'ay harraadkiisa\n\nHannaan wanaageey\n... [truncated]",
  "language": "Somali",
  "language_code": "som",
  "annotation_type": "original-annotations",
  "user_id": "f0ff69570af705b75c5a0851883e502feab2bc874c5e98d59145659bd18ca635"
}
```

From `config="demographics"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [13]:

```json
{
  "user_id": "1a82d8a60ee60f34a47d4d66cb098dec06d2241ea4dded0cc7ef0c729ecb9a9e",
  "age_range": [25, 35],
  "gender": "male",
  "country": "China",
  "languages": ["Japanese", "Simplified Chinese"],
  "dialects": null
}
```

`demographics.user_id` and `default.user_id` are the join key between the two configs, both served as plain strings [7].

## Where it came from

Built by Cohere Labs and released by the Aya Open Science Initiative, an open-science community effort [2]. Annotators registered on the Aya Annotation Platform, selected the languages they were fluent in, and either wrote new prompt-completion pairs or edited machine-translated/templated pairs drawn from 19 public instruction-style data sources, translated using the NLLB 3.3B machine translation model [1][2]. Collection ran from roughly mid-2023 through December 2023, and the release covers 65 of the languages the platform supported, chosen by which languages passed the 50-contribution floor described above [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed or renamed - this one was), which is why Load it pins the revision. The datasets-server endpoints used for Shape, Quality, and A row take no revision parameter and are live, not pinned to the commit above.

[1] Singh et al., "Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning", 2024. https://arxiv.org/abs/2402.06619 - the origin paper; current title read from the live abs page. Read via ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2402.06619). Fetched 2026-08-11.

[2] CohereForAI/aya_dataset dataset card (README), which resolves to https://huggingface.co/datasets/CohereLabs/aya_dataset . https://huggingface.co/datasets/CohereForAI/aya_dataset/raw/main/README.md - dataset description, data fields, splits, statistics, licensing, provenance, known limitations, family table. Fetched 2026-08-11.

[3] Hugging Face Hub API record. https://huggingface.co/api/datasets/CohereForAI/aya_dataset?full=true (redirects to the `CohereLabs/aya_dataset` record) - id, sha, license, gate, downloads, likes, last-modified; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server first-rows endpoint, `default` config, `test` split. https://datasets-server.huggingface.co/first-rows?dataset=CohereLabs%2Faya_dataset&config=default&split=test Fetched 2026-08-11.

[5] datasets-server size, info, and first-rows endpoints, sibling dataset, `aya_human_annotated` config. https://datasets-server.huggingface.co/size?dataset=CohereLabs%2Faya_evaluation_suite , https://datasets-server.huggingface.co/info?dataset=CohereLabs%2Faya_evaluation_suite&config=aya_human_annotated , https://datasets-server.huggingface.co/first-rows?dataset=CohereLabs%2Faya_evaluation_suite&config=aya_human_annotated&split=test - row counts, schema, and the sample row compared against this dataset's `test` split. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=CohereLabs%2Faya_dataset (the `CohereForAI` form returns "The dataset has been renamed"). Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=CohereLabs%2Faya_dataset Fetched 2026-08-11.

[8] CohereForAI/aya-101 model card (README). https://huggingface.co/CohereForAI/aya-101/raw/main/README.md - lists Aya Dataset among the model's training datasets. Fetched 2026-08-11.

[9] Üstün et al., "Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model", 2024. https://arxiv.org/abs/2402.07827 - abstract, read from the live abs page (the ar5iv HTML rendering failed mid-conversion and was not used beyond confirming the abstract text). Fetched 2026-08-11.

[10] datasets-server size endpoint, sibling dataset. https://datasets-server.huggingface.co/size?dataset=CohereLabs%2Faya_collection - live, no revision parameter. Fetched 2026-08-11.

[11] datasets-server size endpoint, sibling dataset. https://datasets-server.huggingface.co/size?dataset=CohereLabs%2Faya_redteaming - returned "Not supported: dataset viewer is disabled", so no live row count is available. Fetched 2026-08-11.

[12] datasets-server first-rows endpoint, `default` config, `train` split. https://datasets-server.huggingface.co/first-rows?dataset=CohereLabs%2Faya_dataset&config=default&split=train Fetched 2026-08-11.

[13] datasets-server first-rows endpoint, `demographics` config, `train` split. https://datasets-server.huggingface.co/first-rows?dataset=CohereLabs%2Faya_dataset&config=demographics&split=train Fetched 2026-08-11.

[14] The corpus screening row for `CohereForAI/aya_dataset`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for SFT on the `default` config's `train` split, with `test` held out as an eval set. Two facts decide it: the rows are direct human-written `inputs`/`targets` pairs with no stated usage restriction beyond the Apache-2.0 license [2], and the `test` split's own content matches a published sibling eval set, `aya_evaluation_suite`'s `aya_human_annotated` config, confirmed by comparing sampled rows [4][5] and consistent with the screening row's note [14].

### The screening row

The row's own note [14]: "204k human-written prompt-completion pairs across 65 languages, crowd-sourced by the Aya Open Science community via the Aya Annotation Platform (repo now resolves to CohereLabs/aya_dataset); train on the 202,362-row train split only - the 1,750-row test split is its own held-out eval." The row carries no flag.
