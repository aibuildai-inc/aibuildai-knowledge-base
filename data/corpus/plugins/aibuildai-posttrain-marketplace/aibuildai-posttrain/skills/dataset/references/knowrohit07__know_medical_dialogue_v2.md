# knowrohit07/know_medical_dialogue_v2

6,307 single-turn instruction/output pairs of patient questions and doctor-style answers on general medical topics, served as one JSON file from a Hugging Face repository at https://huggingface.co/datasets/knowrohit07/know_medical_dialogue_v2 .

**knowrohit07/know_medical_dialogue_v2** is a synthetic expansion of the same author's earlier, smaller `knowrohit07/know_medical_dialogues` release: the card states that "conversational seed tasks or exchanges were collected from anonymized patient-doctor interactions and synthetically made using GPT4" [1]. The card gives no paper and no separate collection-methodology paper; it states only that "the data was meticulously curated to ensure no personally identifiable information remained" [1]. The task shape is single-turn instruction-following: an `instruction` field carries the patient's question, `output` carries the doctor-style answer, and `input` is served empty in every row read [2]. **The card's own stated intended use restricts the data to LLM fine-tuning for medically-informed dialogue, and it explicitly frames model outputs as a supplement to, not a substitute for, professional medical consultation** [1].

**Use it for**: single-turn instruction/response SFT (`instruction` and `output`, `input` unused) - the SFT method card, formatted as an Alpaca-style instruction dataset. The card's stated restriction is that outputs should not be treated as a substitute for professional medical advice [1].

**Licence**: the repo's `cardData` and tags set `license: openrail` [3]; the repository is ungated (`"gated": false`) [3]. The README carries no further licence text and does not name a specific OpenRAIL variant (e.g. RAIL-M vs RAIL-S) [1]. The one catch: no source states which OpenRAIL variant applies.

**Shape**: 6,307 rows, one config (`default`), one split (`train`), three string columns (`instruction`, `input`, `output`) [4][5].

**Hold out**: nothing - no source, including the screening note, states or implies an overlap with any evaluation set [6].

**Origin**: built and released by the Hugging Face user knowrohit07; answers are GPT-4-generated expansions seeded from anonymized human patient-doctor exchanges, per the card [1]. Hub API at the check date: `downloads` 544, `downloadsAllTime` 3,757, `likes` 30 [3][7].

**Trained-on-by**: none found - the Hub models API, filtered for models tagged with this dataset, returned no results [8].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows served and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 6,307 |
| total | 6,307 |

One config, `default`, with three columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `output` | string |
| `input` | string |
| `instruction` | string |

The repository serves a single file, `know_med_v4.json` [9]. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this release.
- The card's only stated curation claim is that the data "was meticulously curated to ensure no personally identifiable information remained," and that conversations are "representative of general concerns and advice, without specific case details" [1].
- The card's stated limitation: the dataset "doesn't cover every medical scenario," and models trained on it "should be viewed as an additional resource, not a substitute for professional medical consultation" [1].
- Of the first 100 served rows (offset 0, `train` split), all 100 had an empty `input` field; whether this holds for the remaining 6,207 rows is not stated and was not read [2].

## Load it

Pin the revision when loading (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-12-18) [3]. The row counts, schema, and sample row given elsewhere in this card come from the datasets-server `/size`, `/info`, and `/first-rows` endpoints, which take no revision parameter and always reflect the current `main` [4][5][2]; they are a live view current as of the check date, and the revision pin below covers only what `load_dataset` fetches at that exact commit, not those endpoints:

```python
import datasets

REV = "48045b0d30f2104614dd3e7e073e2d54f0e6117e"  # main at the check date
train = datasets.load_dataset("knowrohit07/know_medical_dialogue_v2", revision=REV, split="train")  # 6,307 rows
```

**Trap**: the `input` field is present in the schema but served empty in every one of the first 100 rows read - a collator that concatenates `instruction` + `input` + `output` (the standard Alpaca template) will silently produce a blank input section for at least those rows; do not assume `input` carries content without checking further rows yourself.

## Neighbors

- `knowrohit07/know_medical_dialogues` - the same author's earlier, smaller release: 480 rows, one `train` split, the same three-column schema (`instruction`, `output`, `input`) [10][11]. Its own card describes the same collection method - anonymized patient-doctor exchanges "synthetically made using GPT4" [11] - and this `_v2` release appears to be its expansion; prefer `know_medical_dialogue_v2` for more rows unless the smaller set is specifically needed.
- No other same-author or third-party re-release of this corpus was found.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [2]:

```json
{
  "output": "If she actually has a HR of 170 that is accurate, ongoing and persistent, she needs to be seen in the ED immediately.",
  "input": "",
  "instruction": "My daughter ( F, 18 y/o, 5'5', 165lbs) has been feeling poorly for a 6-8 months. She had COVID a couple of months ago and symptoms have are much worse in the last month or so. Symptoms seem POTS-like. She feels light headed, breathless, dizzy, HR goes from ~65 lying down to ~155-160 on standing. Today she tells me HR has been around 170 all day and she feels really lousy. [...] She's away at school if Boston, what to do? Thank you"
}
```

## Where it came from

Built and released by the Hugging Face user knowrohit07. The card states that conversational seed exchanges were collected from anonymized patient-doctor interactions and then expanded synthetically using GPT-4, and that curation removed personally identifiable information so that conversations represent "general concerns and advice, without specific case details" [1]. No further detail on the seed-collection process, seed count, or GPT-4 prompting method is given.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision for `load_dataset`. The row counts, schema, and sample row cited to the datasets-server endpoints ([2], [4], [5], [10]) are a separate, live view: those endpoints take no revision parameter and reflect `main` as of the check date, not the pinned commit.

[1] knowrohit07/know_medical_dialogue_v2 dataset card (README). https://huggingface.co/datasets/knowrohit07/know_medical_dialogue_v2/raw/main/README.md - description, intended use, limitations, data source, collection methodology, ethical considerations. Fetched 2026-08-11.

[2] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=knowrohit07%2Fknow_medical_dialogue_v2&config=default&split=train Fetched 2026-08-11.

[3] Hugging Face Hub API record for knowrohit07/know_medical_dialogue_v2. https://huggingface.co/api/datasets/knowrohit07/know_medical_dialogue_v2?full=true - `cardData.license`, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=knowrohit07%2Fknow_medical_dialogue_v2 Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=knowrohit07%2Fknow_medical_dialogue_v2 Fetched 2026-08-11.

[6] The corpus screening row for `knowrohit07/know_medical_dialogue_v2`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[7] Hugging Face Hub API record with `downloadsAllTime` expanded. https://huggingface.co/api/datasets/knowrohit07/know_medical_dialogue_v2?expand[]=downloadsAllTime Fetched 2026-08-11.

[8] Hugging Face Hub models API filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:knowrohit07/know_medical_dialogue_v2 - empty result. Fetched 2026-08-11.

[9] Hugging Face Hub tree API for the repository at `main`. https://huggingface.co/api/datasets/knowrohit07/know_medical_dialogue_v2/tree/main - lists `.gitattributes`, `README.md`, `know_med_v4.json`. Fetched 2026-08-11.

[10] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=knowrohit07%2Fknow_medical_dialogues Fetched 2026-08-11.

[11] knowrohit07/know_medical_dialogues dataset card (README) and Hub API record. https://huggingface.co/datasets/knowrohit07/know_medical_dialogues/raw/main/README.md and https://huggingface.co/api/datasets/knowrohit07/know_medical_dialogues?full=true Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as single-turn instruction/response SFT data (`instruction` -> `output`, `input` unused). This rests on the card's own description of the data as synthetically-expanded, PII-scrubbed patient-doctor exchanges intended for LLM fine-tuning (quoted in the opening paragraph) [1], and the screening row's note, which states the same origin without flagging any overlap or restriction risk [6].

### The screening row

The row's own note [6]: "anonymized patient-doctor exchanges used as seeds, then expanded synthetically with GPT-4." The row carries no flag.
