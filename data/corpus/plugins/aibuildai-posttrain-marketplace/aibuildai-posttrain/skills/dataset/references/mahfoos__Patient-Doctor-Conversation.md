# mahfoos/Patient-Doctor-Conversation

3,325 single-turn patient-question/doctor-answer pairs in one CSV, each tagged with a three-level severity label - a small, uncredited online-consultation dataset with no dataset card.

**mahfoos/Patient-Doctor-Conversation** is a Hugging Face dataset of 3,325 rows drawn from what appears to be an online medical-consultation forum: each row holds a short patient query (`Description`, restated at greater length in `Patient`), a doctor's reply (`Doctor`), and a `Status` field reading "low severity", "medium severity", or "high severity" [1]. The repository carries no README - fetching it returns "Entry not found" - so the builder states no origin, no collection method, no license, and no intended use anywhere in the repo [1][2]. It lives at https://huggingface.co/datasets/mahfoos/Patient-Doctor-Conversation . **No license is declared anywhere in the repository: the Hub API's license field is empty and no license tag is present, so reuse terms are not established by any source [2].**

**Use it for**: single-turn SFT on patient-question-to-doctor-answer pairs (map `Patient` to the prompt and `Doctor` to the target response), the shape the SFT method card expects, after filtering rows with a null `Doctor` field (5 of the 100 rows sampled at offset 0-99) [5]; `Status` could additionally serve as a severity/triage classification target, but no source describes it as validated for that use. **No license is stated, so the licensing basis for training on this data is not established by any source [2].**

**Licence**: none stated - the Hub API's `license` field is empty and the repo carries no `license:` tag; ungated, public (`"gated": false`, `"private": false`) [2].

**Shape**: 3,325 rows, one config (`default`), one split (`train`), four string columns (`Description`, `Doctor`, `Patient`, `Status`) [3][4].

**Hold out**: nothing - the repository has a single `train` split and no source names an evaluation set or overlap risk.

**Origin**: no source names a builder, a generating model, or a labeling process; the repo's only file besides `.gitattributes` is `pred_status.csv` [1][2]. Hub API at the check date: 332 downloads, 3,656 all-time downloads, 17 likes [2].

**Trained-on-by**: none found - no source describes a model or training recipe consuming this specific dataset.

**Introduced by**: no paper and no dataset card - the repository itself is the only source, and it states no origin [1][2].

## Shape

Rows and splits (datasets-server `/size`) [3]:

| split | rows |
| --- | --- |
| `train` | 3,325 |

One config, `default`, four columns, all strings (datasets-server `/info`) [4]:

| column | dtype |
| --- | --- |
| `Description` | string |
| `Doctor` | string |
| `Patient` | string |
| `Status` | string |

Sizes (datasets-server `/size`) [3]: 4,477,974 bytes as the original CSV, 2,564,501 bytes as Parquet, 4,514,926 bytes decoded in memory. No source states sequence-length or token statistics for this dataset.

## Quality

- Across the rows sampled below, `Status` takes exactly three values - "low severity", "medium severity", "high severity" - at row indices 0-99, 1000-1004, and 3315-3324 (114 rows total, three non-contiguous ranges of the single `train` split) [5]; no source states the full-split distribution across these three labels.
- `Description` is a short restatement of the longer `Patient` field in every one of those 114 sampled rows, and many `Doctor` replies end with a stock referral phrase such as "for further information consult a neurologist online" [5].
- Of the 100 rows sampled at offset 0-99, 5 (row indices 61, 82, 86, 98, 99) have a null `Doctor` field - a 5% missing-target rate in that sample - while `Patient` is populated in all 100 [5]. This bears directly on the SFT mapping in Use it for: a `Patient`-to-`Doctor` training run must filter or otherwise handle rows with a null `Doctor` value.
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset; none is invented here.
- No source states who wrote the `Doctor` replies (a clinician, a forum moderator, or a model) or how the `Status` label was assigned.

## Load it

Single split, single config, no license gate. `load_dataset` accepts a `revision`, so pin it to the commit this card's row/byte counts were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-05-26) [2]:

```python
import datasets

REV = "d225c93b6430ec425f7264975ee23f44a94620fd"  # main at the check date
train = datasets.load_dataset("mahfoos/Patient-Doctor-Conversation", revision=REV, split="train")  # 3,325 rows
```

This `revision` pin covers only what `load_dataset` itself fetches from the repository at that commit. The row/byte counts and sampled rows quoted elsewhere on this card (Shape, Quality, A row) come from the datasets-server `/size`, `/info`, `/first-rows`, and `/rows` endpoints, none of which take a `revision` parameter - they always answer for the current default branch, so those figures are live reads as of the check date, not reads pinned to `REV` [3][4][1][5].

**Trap**: the repository has no dataset card, so `load_dataset` builds the `default` config straight from `pred_status.csv` with no declared license, no declared task, and no declared collection method - every one of those facts is absent from the source, not merely omitted from this card [1][2].

## Neighbors

Two other Hub repositories carry this exact content, confirmed by comparing a fetched row from each against row 0 of this dataset ("what does abutment of the nerve root mean") [6]:

- `pavanmantha/doctor_patient_conversation` - the identical 3,325 rows, reformatted into three columns (`description`, `conversation`, `status`) where `conversation` merges the `Patient` and `Doctor` text into one `"Patient: ...\nDoctor: ..."` string; row 0 matches this dataset's row 0 verbatim [6].
- `supergoose/buzz_sources_103_Patient-Doctor-Conversation` - 3,244 rows (81 fewer than this dataset), reformatted into a ShareGPT-style `conversations` list tagged `"source": "Patient-Doctor-Conversation"`; its row 0 human/gpt turns match this dataset's `Patient`/`Doctor` text for row 0 verbatim [6].

Neither neighbor's card states which repository is the original, and whether this repository predates either neighbor is not established by any source. Prefer this repository for the raw four-column shape (`Description`/`Doctor`/`Patient`/`Status`); reach for `pavanmantha/doctor_patient_conversation` only if a pre-merged single-string conversation column is wanted, or for `supergoose/buzz_sources_103_Patient-Doctor-Conversation` only if ShareGPT-format turns are wanted - but note its row count is short of this dataset's by 81 rows, so it is not a complete copy.

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [1]:

```json
{
  "Description": "what does abutment of the nerve root mean",
  "Doctor": "hi I have gone through your query with diligence and would like you to know that I am here to help you for further information consult a neurologist online",
  "Patient": "hi doctor I am just wondering what is abutting and abutment of the nerve root means in a back issue please explain what treatment is required for annular bulging and tear",
  "Status": "medium severity"
}
```

## Where it came from

No source states a builder, an upstream pool, a collection method, or a generating model for this dataset. The repository has no README (fetching it returns "Entry not found") [1], and the Hub API record carries no `cardData`, no license, and no citation fields [2]. The only content file in the repository tree is `pred_status.csv`, alongside `.gitattributes` [2].

## Sources

Checked 2026-08-11. Hugging Face Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision above; that pin covers only what `load_dataset` reads at that commit. The datasets-server `/size`, `/info`, `/first-rows`, and `/rows` endpoints used for the row counts, byte sizes, and sampled rows below take no `revision` parameter and always answer for the live default branch, so those figures are current as of the check date rather than pinned to the commit above.

[1] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mahfoos%2FPatient-Doctor-Conversation&config=default&split=train - features, row 0. Also: https://huggingface.co/datasets/mahfoos/Patient-Doctor-Conversation/raw/main/README.md, which returns "Entry not found". Fetched 2026-08-11.

[2] Hugging Face Hub API record for mahfoos/Patient-Doctor-Conversation. https://huggingface.co/api/datasets/mahfoos/Patient-Doctor-Conversation?full=true - `sha`, `license` (empty), `tags`, `siblings` (`.gitattributes`, `pred_status.csv`), `downloads`, `likes`, `lastModified`, `gated`, `private`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. https://huggingface.co/datasets/mahfoos/Patient-Doctor-Conversation - the dataset's own Hub page. Fetched 2026-08-11.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mahfoos%2FPatient-Doctor-Conversation Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=mahfoos%2FPatient-Doctor-Conversation Fetched 2026-08-11.

[5] datasets-server rows endpoint, three offsets into the single `train` split: `offset=0&length=100` (via first-rows [1]), `offset=1000&length=5`, and `offset=3315&length=10`. https://datasets-server.huggingface.co/rows?dataset=mahfoos%2FPatient-Doctor-Conversation&config=default&split=train&offset=<n>&length=<n> Fetched 2026-08-11.

[6] datasets-server first-rows and size endpoints for the two neighbor repositories. https://datasets-server.huggingface.co/first-rows?dataset=pavanmantha%2Fdoctor_patient_conversation&config=default&split=train, https://datasets-server.huggingface.co/size?dataset=pavanmantha%2Fdoctor_patient_conversation, https://datasets-server.huggingface.co/first-rows?dataset=supergoose%2Fbuzz_sources_103_Patient-Doctor-Conversation&config=default&split=train, https://datasets-server.huggingface.co/size?dataset=supergoose%2Fbuzz_sources_103_Patient-Doctor-Conversation - row-0 content and row counts. Fetched 2026-08-11.

[7] The corpus screening row for `mahfoos/Patient-Doctor-Conversation`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for single-turn SFT on patient-question-to-doctor-answer pairs, with no license established by any source - a fact the Load it and Licence sections above already carry. The screening row's note matches what the repository itself shows: no card, no stated origin, a `Description`/`Doctor`/`Patient`/`Status` shape [7].

### The screening row

The row's own note [7]: "patient question, doctor answer and a severity label; the repo has no card and states no origin." The row carries no flag.
