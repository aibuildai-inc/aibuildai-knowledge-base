# medalpaca/medical_meadow_cord19

821,007 instruction-formatted rows that pair a COVID-19 research-paper abstract with its title, reformatted from the CORD-19 corpus into Alpaca-style `instruction`/`input`/`output` triples.

**medalpaca/medical_meadow_cord19** (https://huggingface.co/datasets/medalpaca/medical_meadow_cord19) is the CORD-19 Open Research Dataset [1] repackaged by the MedAlpaca team, whose "Medical Meadow" collection was introduced in "MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data" [2], for Alpaca-style instruction tuning; the Hub card describes it as "a processed version of the dataset, where we removed some empty entries and formated it to be compatible with the alpaca training" [3]. Each row is a single-turn abstract-to-title generation instance, task-tagged `summarization` on the Hub [4]. **The underlying CORD-19 abstracts carry mixed, per-paper licences (from CC to more restrictive COVID-19-specific copyright grants) rather than one uniform grant [1], and this Hub repo declares no licence of its own [4].**

**Use it for**: single-turn SFT on a fixed instruction ("summerize" a given abstract into a title) - the SFT method card, feeding the `instruction`+`input`+`output` triple as a prompt/completion pair (concatenate `instruction` and `input` as the prompt, `output` as the target). Not preference data; there is nothing to hold out for that purpose here.

**Licence**: no SPDX id (`cardData` carries no `license` key; repo tags list none) [4], ungated (`"gated": false`, `"private": false`) [4]. The catch: CORD-19's own paper says its canonical-entry selection favors "the most permissive license" among duplicates but that licences "differ greatly across papers" and range from CC terms to "more restrictive COVID-19-speciﬁc copyright license[s]" [1] - so the abstracts here are not under one uniform redistribution grant even though the Hub repo shows none.

**Shape**: 821,007 rows, one config (`default`), one split (`train`), three string columns (`instruction`, `input`, `output`) [5][6].

**Hold out**: nothing named by any source reviewed. The dataset ships a single `train` split with no author-designated test split and no stated overlap with an evaluation benchmark; carve a held-out sample yourself before training if you need one.

**Origin**: built by the MedAlpaca team (Han, Adams, Papaioannou, Grundmann, Oberhauser, Figueroa, Löser, Truhn, Bressem) [2] from the CORD-19 corpus, whose own abstracts and titles come from the corpus's Allen Institute for AI / White House coalition source papers [1]; no model-generated content in this file - both `input` and `output` are copied from the source papers, not synthesized. Hub downloads 797, `downloadsAllTime` 9,409, likes 10 as of the check date [4].

**Trained-on-by**: the MedAlpaca paper's Section 2.1.4 lists CORD-19 among the "medical NLP Benchmarks" data the Medical Meadow collection "additionally use[s]" [2], but the `medalpaca/medalpaca-7b` model card's own training-data table - which enumerates ChatDoc, Wikidoc, three Stack Exchange categories, Anki flashcards, and Wikidoc patient information by row count - does not list CORD-19 among its sources [7]. No other adopter was found.

**Introduced by**: [2] (Han et al., "MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data").

## Shape

Rows and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 821,007 |

One config, `default`, three columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |

Sizes (datasets-server `/size`) [5]: 1,377,062,197 bytes original JSON download, 752,832,517 bytes as Parquet, 1,336,834,621 bytes decoded in memory. No source states sequence-length or token statistics for this file; none is invented here.

## Quality

- Of 100 rows read at offset 0 of `train` (datasets-server `/first-rows`), every row carries the identical `instruction` string, "Please summerize the given abstract to a title" (the misspelling is in the served data) [8]. This sample does not establish that the instruction is identical for all 821,007 rows.
- The Hub card's only stated processing note is that "some empty entries" were removed when converting the source corpus to this Alpaca-compatible format [3]; no source states a measured empty-entry rate, duplicate rate, or other quality metric for the resulting file.
- The source CORD-19 paper notes its own document-clustering and canonical-metadata process can occasionally merge or split entries under ID conflicts [1]; no source ties this to a measured error rate in this reformatted file.

## Load it

Single split, no split to hold out declared by any source; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-04-06) [4]:

```python
import datasets

REV = "f48ea381cb3ae21178d8a573bb1a8995d4ab6b11"  # main at the check date
train = datasets.load_dataset("medalpaca/medical_meadow_cord19", revision=REV, split="train")  # 821,007 rows
```

**Trap**: the repository's only data file is `medical_meadow_cord19.json` (`tree_only_files` on the shortlist row), a single 1.38 GB JSON array loaded whole by the `json` builder - there is no streaming-friendly sharded layout, so `load_dataset` with default settings materializes the full file before any filtering [9].

## Neighbors

- `supergoose/buzz_sources_009_medical_meadow_cord19` - row 0 of its `train` split carries the same abstract as `system`, the same fixed "Please summerize the given abstract to a title" as `human`, and the same title as `gpt` seen in this dataset's row 0, confirming it reshapes these abstract/title pairs into a `conversations` (ShareGPT-style) list plus `source` and `stack` columns [10]; it has 755,200 rows against 821,007 here - fewer rows, likely after further filtering; check before assuming row-for-row parity [10].
- `allenai/cord19` and the several `pyterrier-quality`, `macavaney`, and `irds` CORD-19 mirrors on the Hub are the raw, un-instruction-formatted corpus (full documents or retrieval-indexed forms), not Alpaca triples - useful only if this instruction format is not what is needed [11].
- The other nine `medalpaca/medical_meadow_*` repositories (health advice, MedQA, PubMed Causal, Wikidoc, Wikidoc patient information, medical flashcards, MMLU, USMLE self-assessment, MEDIQA) are sibling Medical Meadow releases from the same builder covering different source tasks, not overlapping content with this abstract-to-title file [12].
- This corpus prefers the original `medalpaca/medical_meadow_cord19` for the full 821,007-row set; reach for `supergoose/buzz_sources_009_medical_meadow_cord19` only if a ShareGPT-style `conversations` column is what the training pipeline needs.

## A row

One config, one split, one schema. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [8], with the long abstract truncated:

```json
{
  "instruction": "Please summerize the given abstract to a title",
  "input": "INTRODUCTION: Infection with the new coronavirus responsible for Severe Acute Respiratory Syndrome (SARS-CoV-2) continues to spread worldwide. In Brazil, there are already more than 230 thousand dead, many of these older adults. OBJECTIVE: To present the clinical characteristics of older Brazilian adults infected by COVID-19 [...] Age, sex, sore throat, dyspnea, respiratory discomfort, O2 saturation <95%, neurological disease, pneumopathy, immunodeficiency, and kidney disease were significantly associated with risk of death from COVID-19.",
  "output": "Clinical characteristics of 1544 Brazilians aged 60 years and over with laboratory evidence for SARS-CoV-2"
}
```

## Where it came from

Built by the MedAlpaca team from the CORD-19 Open Research Dataset [1][2]. CORD-19 itself was assembled by the Allen Institute for AI, the Chan Zuckerberg Initiative, Microsoft Research, Georgetown University, the NIH, and the White House, pooling "over 1,000,000 scholarly articles, including over 400,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses" from multiple upstream publisher and preprint sources, clustered and deduplicated by the CORD-19 paper's own pipeline [1]. The Hub card states this repository is "a processed version of the dataset, where we removed some empty entries and formated it to be compatible with the alpaca training" - pairing each abstract as `input` with the paper's own title as `output` under a single fixed summarization `instruction` [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Wang et al., "CORD-19: The COVID-19 Open Research Dataset", Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020. https://www.aclweb.org/anthology/2020.nlpcovid19-acl.1 - dataset scale, licence-mixing and canonical-entry selection, clustering process. Read from the ACL Anthology PDF (aclanthology.org/2020.nlpcovid19-acl.1.pdf). Fetched 2026-08-11.

[2] Han et al., "MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data", 2023. https://arxiv.org/abs/2304.08247 - the origin paper for the Medical Meadow collection and this dataset; current title read from the live abs page; body text read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2304.08247). Fetched 2026-08-11.

[3] medalpaca/medical_meadow_cord19 dataset card (README). https://huggingface.co/datasets/medalpaca/medical_meadow_cord19/raw/main/README.md - dataset summary, processing note, Kaggle homepage link, citation. Fetched 2026-08-11.

[4] Hugging Face Hub API record for medalpaca/medical_meadow_cord19. https://huggingface.co/api/datasets/medalpaca/medical_meadow_cord19?full=true - `cardData`, gate/private status, `sha`, downloads, likes, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=medalpaca%2Fmedical_meadow_cord19 Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=medalpaca%2Fmedical_meadow_cord19 Fetched 2026-08-11.

[7] medalpaca/medalpaca-7b model card (README). https://huggingface.co/medalpaca/medalpaca-7b/raw/main/README.md - training-data source table. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=medalpaca%2Fmedical_meadow_cord19&config=default&split=train Fetched 2026-08-11.

[9] Hugging Face Hub tree API for medalpaca/medical_meadow_cord19 at `main`. https://huggingface.co/api/datasets/medalpaca/medical_meadow_cord19/tree/main - file listing, showing the single `medical_meadow_cord19.json` data file. Fetched 2026-08-11.

[10] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=supergoose%2Fbuzz_sources_009_medical_meadow_cord19 - row count; its dataset card, https://huggingface.co/datasets/supergoose/buzz_sources_009_medical_meadow_cord19/raw/main/README.md, for the `conversations`/`source`/`stack` schema; and its first-rows endpoint, https://datasets-server.huggingface.co/first-rows?dataset=supergoose%2Fbuzz_sources_009_medical_meadow_cord19&config=default&split=train, whose row 0 was compared against this dataset's row 0 to confirm shared content. Fetched 2026-08-11.

[11] Hugging Face Hub dataset search for "cord19". https://huggingface.co/api/datasets?search=cord19&limit=50 - listing of raw/retrieval-indexed CORD-19 mirrors on the Hub. Fetched 2026-08-11.

[12] Hugging Face Hub dataset listing for the medalpaca author. https://huggingface.co/api/datasets?author=medalpaca&limit=100 - the nine other `medical_meadow_*` sibling repositories. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as single-turn SFT data for abstract-to-title generation, with no held-out split named by any source and no licence declared on the Hub repo. The screening row's own note describes the task in the same terms as the card above: summarizing CORD-19 research abstracts.

### The screening row

The row's own note: "summarize COVID-19 research abstracts from CORD-19 papers." The row carries no flag.
