# cais/mmlu

231,400 multiple-choice questions - a 57-subject knowledge benchmark plus one large auxiliary training pool - packaged as 59 Hugging Face configs by the MMLU authors' own group, CAIS.

**cais/mmlu** is the canonical Hugging Face release of the Measuring Massive Multitask Language Understanding benchmark introduced by Hendrycks et al. in "Measuring Massive Multitask Language Understanding" (ICLR 2021) [1], a multiple-choice test spanning humanities, social sciences, STEM, and other subjects that people learn, built to check whether a model has broad world knowledge and problem-solving ability [1]. Each of the 57 subject configs (`abstract_algebra`, `anatomy`, ... `world_religions`) carries `test`, `validation`, and `dev` splits of the same four-field question format; a separate `all` config is the union of all 57 subjects' `test`/`validation`/`dev` plus a 99,842-row `auxiliary_train` split of unrelated multiple-choice questions drawn from ARC, MC_TEST, OBQA, RACE and similar sources, meant for few-shot or auxiliary training rather than as MMLU content itself [2]. **The `test`, `validation`, and `dev` splits, in every config that carries them, are the evaluation benchmark and must be held out of any training run; only the `auxiliary_train` data (also served standalone as the `auxiliary_train` config, same rows reshaped) is trainable.** It lives at https://huggingface.co/datasets/cais/mmlu .

**Use it for**: SFT-style multiple-choice-QA training data drawn from `auxiliary_train` only (99,842 rows, question/choices/answer, sourced from ARC/MC_TEST/OBQA/RACE, not the MMLU subject questions) [2] - see the SFT method card. Every other split, across all 59 configs, is benchmark data to evaluate on, never to train on.

**Licence**: MIT, per the card's `license: mit` metadata and its "Licensing Information" section, which links the MIT License at the upstream `hendrycks/test` GitHub repository [2]; ungated (`"gated": false`, `"private": false`) [3]. No further catch stated.

**Shape**: 231,400 rows across 59 configs - 57 single-subject configs (`test`+`validation`+`dev`, 15,858 rows combined), one `all` config (the union of those plus `auxiliary_train`, 115,700 rows), and one standalone `auxiliary_train` config (99,842 rows, same questions as `all`'s `auxiliary_train` split, reshaped) [3][4].

**Hold out**: every `test`, `validation`, and `dev` split, in the `all` config and in each of the 57 subject configs (15,858 rows total, each subject's rows counted once) - this is the benchmark itself. Only `auxiliary_train` (99,842 rows) is trainable [2]; the shortlist screening row states this directly, calling `auxiliary_train` "the only trainable split" [5].

**Origin**: built by the CAIS (Center for AI Safety) organization account from Dan Hendrycks and coauthors' original MMLU release; questions are human-written exam-style multiple-choice items, not model-generated [1][2]. Hub API at the check date: `downloads` 473,575, `downloadsAllTime` 42,757,824, `likes` 810 [3].

**Trained-on-by**: no source states a specific model trained on the `auxiliary_train` split - MMLU is overwhelmingly used as an evaluation benchmark rather than a training source. The upstream GitHub leaderboard reports MMLU test-set scores (not training use) for GPT-3, GPT-2, UnifiedQA, Chinchilla, Gopher, and the Flan-T5 family [6]. "None found" for training adoption.

**Introduced by**: [1] (Hendrycks et al.).

## Shape

Row counts by config group, read from the live datasets-server `/size` endpoint and cross-checked against the shortlist's per-split counts [3][4]:

| config group | configs | splits present | rows |
| --- | --- | --- | --- |
| 57 subject configs | `abstract_algebra` ... `world_religions` | `test`, `validation`, `dev` | 15,858 |
| `all` | 1 | `test`, `validation`, `dev`, `auxiliary_train` | 115,700 |
| `auxiliary_train` | 1 | `train` | 99,842 |
| **Total** | **59** | | **231,400** |

Two schemas are served. The `all` config and each of the 57 subject configs use a flat schema - `question` (string), `subject` (string), `choices` (list of 4 strings), `answer` (`ClassLabel`, values A/B/C/D) - confirmed live for `all/auxiliary_train` as well as for the subject splits [4][7]. The standalone `auxiliary_train` config nests the same fields differently: a single `train` column holding a struct of `answer` (int64, not a `ClassLabel`), `choices` (list of strings), `question` (string), and `subject` (string) [4][7].

Total size at the check date: 103,650,732 bytes of original/parquet files, 259,482,395 bytes decoded in memory, 231,400 rows, read from the live `/size` endpoint [3]. The dataset's `dataset_info` YAML block in the card (pinned to the same commit) reports different per-config byte totals for `all` (168,856,915 bytes decoded) than the live `/info` endpoint does for the same config (128,216,338 bytes decoded), while both agree on row counts (14,042/1,531/285/99,842); no source explains the mismatch, so both are reported here rather than reconciled [2][4]. No source states sequence-length or token statistics for this release.

## Quality

- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this release; none is invented here.
- The card documents no annotation process beyond calling the questions "expert-generated" in its metadata tags; the "Annotations" and "Source Data" sections of the README are marked "[More Information Needed]" [2].
- The card's own leaderboard table reports few-shot/zero-shot MMLU accuracy for baseline models at the time of the original release: UnifiedQA 48.9% average, GPT-3 (few-shot) 43.9%, GPT-2 32.4%, against a 25.0% random baseline - showing the benchmark separates model capability from chance at that baseline generation [2].
- The first 10 served rows of the standalone `auxiliary_train` config all carry an empty `subject` string, consistent with those rows being drawn from external datasets (ARC, MC_TEST, OBQA, RACE) rather than being subject-tagged MMLU items [4][2].

## Load it

Train only on `auxiliary_train`; hold out every `test`/`validation`/`dev` split. Pin the revision this card's numbers were read at (the Hub API's `sha`, matching the shortlist's commit; the repo was last modified 2024-03-08) [3]:

```python
import datasets

REV = "c30699e8356da336a370243923dbaf21066bb9fe"  # main at the check date

# Trainable: 99,842 rows, nested under a `train` struct column
aux = datasets.load_dataset("cais/mmlu", "auxiliary_train", revision=REV, split="train")

# Benchmark only - hold out, never train on these:
subject_test = datasets.load_dataset("cais/mmlu", "abstract_algebra", revision=REV, split="test")
all_test = datasets.load_dataset("cais/mmlu", "all", revision=REV, split="test")
```

**Trap**: loading the `all` config's `auxiliary_train` split gives the flat question/subject/choices/answer schema (same shape as every benchmark split), while loading the standalone `auxiliary_train` config's `train` split gives the same underlying rows nested one level deeper under a `train` struct column with a plain int64 `answer` instead of a `ClassLabel` [4][7] - a collator written for one will not read the other without unwrapping the struct.

## Neighbors

- `hails/mmlu_no_train` - this same 15,858-row benchmark union (`test`+`validation`+`dev` across the 57 subjects, matching `all`'s non-`auxiliary_train` rows exactly) repackaged without the `auxiliary_train` split, built so that loading many subject configs does not repeatedly regenerate that large split; its card states it "contains a copy of the `cais/mmlu` HF dataset but without the `auxiliary_train` split" [8]. Prefer it over this release when only the benchmark is needed and `auxiliary_train` would otherwise be downloaded and discarded.
- `openai/MMMLU` - OpenAI's professional human translation of the MMLU test set into 14 languages (Arabic, Bengali, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Brazilian Portuguese, Swahili, Yoruba, simplified Chinese), 393,176 rows live at the check date, MIT-licensed [9][10]. Use it for multilingual MMLU evaluation, not as additional English training data.
- `TIGER-Lab/MMLU-Pro` - a harder successor benchmark from a different team (TIGER-Lab), with 10-way options instead of 4, 12,102 rows live at the check date, introduced in its own paper [11][9]; not a re-release of this dataset and not a substitute for holding out this release's own test set.
- `lukaemon/mmlu` - an older community loader with a different flat schema (`input`, `A`, `B`, `C`, `D`, `target` columns) built on a custom loading script rather than served Parquet; its declared `dataset_info` reports the same per-subject row counts as this release's `test`/`validation` splits, with `dev` renamed `train` [12]. This corpus prefers `cais/mmlu` (the canonical release, Parquet-served) over this loader.

## A row

Two distinct served shapes; one row from each, fetched live via datasets-server `/first-rows` [4].

Flat shape, `config="abstract_algebra"`, `split="dev"`, `row_idx=0`:

```json
{
  "question": "Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.",
  "subject": "abstract_algebra",
  "choices": ["0", "1", "2", "3"],
  "answer": 1
}
```

Nested shape, `config="auxiliary_train"`, `split="train"`, `row_idx=0`:

```json
{
  "train": {
    "answer": 1,
    "choices": ["Adams only.", "Brooks only.", "Case only.", "Adams and Brooks"],
    "question": "Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were",
    "subject": ""
  }
}
```

## Where it came from

Built and released on the Hub by the `cais` (Center for AI Safety) organization account, mirroring Dan Hendrycks and coauthors' original MMLU test [1][2]. The 57 subjects' `test`/`validation`/`dev` questions are the human-written exam-style items introduced by that paper [1]. The `auxiliary_train` split is described in the card as "auxiliary multiple-choice training questions from ARC, MC_TEST, OBQA, RACE, etc." - external multiple-choice datasets, not MMLU subject content, bundled in for few-shot or auxiliary training use [2]. The README's "Source Data", "Annotations", and "Personal and Sensitive Information" subsections are all marked "[More Information Needed]" [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the datasets-server `/size`, `/info`, and `/first-rows` endpoints used for Shape, A row, and the Neighbors row counts take no revision parameter and are live, not pinned to the commit above.

[1] Hendrycks, Burns, Basart, Zou, Mazeika, Song, Steinhardt, "Measuring Massive Multitask Language Understanding", ICLR 2021. https://arxiv.org/abs/2009.03300 - the origin paper; current title read from the live abs page. Fetched 2026-08-11.

[2] cais/mmlu dataset card (README). https://huggingface.co/datasets/cais/mmlu/raw/main/README.md - dataset summary, task list, data instances/fields/splits, curation rationale, licensing, citation. Fetched 2026-08-11.

[3] Hugging Face Hub API record for cais/mmlu. https://huggingface.co/api/datasets/cais/mmlu?full=true - `sha`, `private`, `gated`, `downloads`, `likes`, `lastModified`, `cardData.license`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size, info, and first-rows endpoints for cais/mmlu. https://datasets-server.huggingface.co/size?dataset=cais%2Fmmlu , https://datasets-server.huggingface.co/info?dataset=cais%2Fmmlu , https://datasets-server.huggingface.co/first-rows?dataset=cais%2Fmmlu&config=abstract_algebra&split=dev , https://datasets-server.huggingface.co/first-rows?dataset=cais%2Fmmlu&config=auxiliary_train&split=train , https://datasets-server.huggingface.co/first-rows?dataset=cais%2Fmmlu&config=all&split=auxiliary_train Fetched 2026-08-11.

[5] The corpus screening row for `cais/mmlu`, supplied with this card's request - its `note`, read back in the appendix. Checked 2026-08-11.

[6] `hendrycks/test` GitHub repository README. https://raw.githubusercontent.com/hendrycks/test/master/README.md - the MMLU leaderboard table of evaluated (not trained) models. Fetched 2026-08-11.

[7] datasets-server info endpoint, `all` and `auxiliary_train` config feature schemas. https://datasets-server.huggingface.co/info?dataset=cais%2Fmmlu Fetched 2026-08-11.

[8] `hails/mmlu_no_train` dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/hails/mmlu_no_train/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=hails%2Fmmlu_no_train Fetched 2026-08-11.

[9] datasets-server size endpoint, live row counts for `openai/MMMLU` and `TIGER-Lab/MMLU-Pro`. https://datasets-server.huggingface.co/size?dataset=openai%2FMMMLU , https://datasets-server.huggingface.co/size?dataset=TIGER-Lab%2FMMLU-Pro - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[10] `openai/MMMLU` dataset card (README). https://huggingface.co/datasets/openai/MMMLU/raw/main/README.md - translation description, locale list, licence. Fetched 2026-08-11.

[11] `TIGER-Lab/MMLU-Pro` dataset card (README). https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/raw/main/README.md - dataset description, feature schema, paper link. Fetched 2026-08-11.

[12] `lukaemon/mmlu` Hub API record. https://huggingface.co/api/datasets/lukaemon/mmlu - declared `dataset_info` schema and per-config row counts. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable, and only as multiple-choice-QA training data drawn from `auxiliary_train`: every `test`/`validation`/`dev` split, in every config, is the MMLU benchmark and must be held out. This rests on the card's own description of `auxiliary_train` as external auxiliary training questions distinct from the MMLU subject content [2], and on the screening row's note, which calls `auxiliary_train` the dataset's only trainable split [5].

### The screening row

The row's own note [5]: "The canonical MMLU; `auxiliary_train` is the only trainable split and everything else is the benchmark." The row carries no flag.
