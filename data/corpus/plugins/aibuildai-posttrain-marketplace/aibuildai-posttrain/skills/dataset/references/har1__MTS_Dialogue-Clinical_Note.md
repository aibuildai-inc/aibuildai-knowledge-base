# har1/MTS_Dialogue-Clinical_Note

1,301 short doctor-patient dialogues paired with structured clinical-note summaries, a reformatted repackaging of the MTS-Dialog train and validation sets into a fixed four-field note template.

**har1/MTS_Dialogue-Clinical_Note** repackages the MTS-Dialog corpus introduced in "An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters" [1], a collection of short doctor-patient conversations paired with clinician-written section summaries built for the MEDIQA-Chat and MEDIQA-Sum 2023 shared tasks [2]. The Hub card states this repository merges the original 1,201-pair training set and 100-pair validation set, and rewrites every summary into a fixed template of Symptoms/Diagnosis/History of Patient/Plan of Action (with `N/A` where a section has no content), a reformatting done to fit the `har1/HealthScribe-Clinical_Note_Generator` fine-tuning project [3]. **The two official MEDIQA test sets (200 rows each) are not included here** [2]; **the appended validation rows sit inside the single `train` split with no split marker separating them from the training rows**, established by reading the row `ID` column across the split (below). It lives at https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note .

**Use it for**: dialogue-to-note summarization SFT (sequence-to-sequence: `dialogue` in, `section_text` out) — the shape used to fine-tune `facebook/bart-large-cnn` in `har1/HealthScribe-Clinical_Note_Generator` [4]. Not a chat or preference format; maps to the SFT method card as a plain text-in/text-out pair, no `data_dir` or config needed. **Hold out the last 100 rows** (the appended original MTS-Dialog validation set) if evaluating against that published set.

**Licence**: MIT per the Hub card metadata (`license: mit`, tag `license:mit`) [5]. The one catch: the upstream MTS-Dialog repository this data derives from is licensed CC BY 4.0 and asks that its paper be cited [2]; no source here explains the relicensing to MIT.

**Shape**: 1,301 rows, one config (`default`), one split — `train` — with four columns (`ID` int64, `section_header` string, `section_text` string, `dialogue` string) [6][7].

**Hold out**: rows at index 1201–1300 (`ID` 0–99, a second time) — these are the original MTS-Dialog validation set appended after the 1,201 training rows, confirmed by reading rows at offset 0 and offset 1195–1214 of `train` and seeing the `ID` counter restart at row 1201 [8]. Hold these out before evaluating against the published MTS-Dialog validation split. The official MEDIQA-Chat and MEDIQA-Sum test sets (200 rows each) are absent from this repository entirely, so they carry no leakage risk here [2].

**Origin**: dialogues and section summaries originate from the MTS-Dialog corpus [1]; this Hub repository's reformatting into the four-field template is credited to five named contributors for the HealthScribe project, not the original MTS-Dialog authors [3]. Hub API at the check date: `downloads` 365, `downloadsAllTime` 6,086, `likes` 12 [5].

**Trained-on-by**: `har1/HealthScribe-Clinical_Note_Generator`, a fine-tune of `facebook/bart-large-cnn` trained on this dataset's 1,201 training and 100 validation rows, reaching Rouge1 54.32 / Rouge2 34.27 / RougeL 46.58 on its final epoch [4]. A Hub model search filtered to `datasets:har1/MTS_Dialogue-Clinical_Note` returns only this one model [9]. No other adoption found.

**Introduced by**: [1] (Ben Abacha et al., the original MTS-Dialog paper); this repackaged Hub version has no paper of its own — introduced by the dataset card [3].

## Shape

One config, one split (datasets-server `/size`, `/info`) [6][7]:

| split | rows | columns |
| --- | --- | --- |
| `train` | 1,301 | `ID` (int64), `section_header` (string), `section_text` (string), `dialogue` (string) |

Sizes (datasets-server `/size`) [6]: 1,092,878 bytes of original CSV download (two files: a 1,018,127-byte training CSV and a 74,751-byte validation CSV), 1,103,300 bytes decoded in memory. No source states token or word-length statistics for this repackaged release.

The `/size` and `/info` endpoints take no revision parameter, so these row, column, and byte counts are live reads at the check date, not pinned to the `sha` in Load it below; a later push to `main` could change them without changing the pinned load call's own row count, which datasets-server does not compute per-revision.

## Quality

- The `section_text` field is not free text taken verbatim from the original MTS-Dialog summaries; it is rewritten into the fixed Symptoms/Diagnosis/History of Patient/Plan of Action template regardless of the row's `section_header`, per the card's own description [3]. Reading the first 100 rows (`train`, offset 0) confirms this: every row's `section_text` follows that four-line template with `N/A` filling empty sections, even for headers like `MEDICATIONS`, `CC`, and `FAM/SOCHX` that in the original corpus carry free-text summaries in a different shape — for example, the same row 0 (`ID` 0, `GENHX`) reads as free prose in the unmodified corpus mirror `lyumengxian/MTS-Dialog` ("The patient is a 76-year-old white female who presents to the clinic today...") [10] versus the four-field bullet form here ("Symptoms: no fever...\nDiagnosis: hypertension...") [8].
- Of the 100 rows read at offset 0, `section_header` takes 16 distinct values (`FAM/SOCHX`, `GENHX`, `PASTMEDICALHX`, `CC`, `PASTSURGICAL`, `MEDICATIONS`, `ALLERGY`, `EXAM`, `ASSESSMENT`, `ROS`, `DISPOSITION`, `EDCOURSE`, `OTHER_HISTORY`, `PLAN`, `DIAGNOSIS`, `IMMUNIZATIONS`), matching the 20-header scheme documented for the underlying MTS-Dialog corpus [2].
- No source states a measured contamination, duplication, or annotator-agreement rate for this specific 1,301-row repackaging; the origin paper reports manual fact-based quality scoring on the 100-row validation set of the original corpus (Factual P/R/F1, hallucination and omission rates), but those figures were computed on model outputs, not on the source data itself, so they are not repeated here [1].

## Load it

Single config, single split; pin the revision the row counts above were read at (the Hub API's `sha` for `main`, last modified 2024-04-01) [5]:

```python
import datasets

REV = "bbf6a8c1e48e18726d49b5b265787844ab0824b2"  # main at the check date
ds = datasets.load_dataset("har1/MTS_Dialogue-Clinical_Note", revision=REV, split="train")  # 1,301 rows
```

**Trap**: this call loads all 1,301 rows as one undivided `train` split — the last 100 (row index 1201–1300) are the original MTS-Dialog validation set appended without a marker (see Hold out above); slice them off before using any of this data as a held-out set.

## Neighbors

- `X7-qP2-mN9-v4/MTS_Dialogue-Clinical_Note` — an exact duplicate: same 1,301 rows, same four columns, same README text, read live at the check date [11][12].
- `lyumengxian/MTS-Dialog` — the same underlying corpus kept in its original, unreformatted shape and with proper `train`/`validation`/`test` splits (1,201/100/200 rows), including the 200-row official test set this repository lacks; its `section_text` is free clinician prose rather than the four-field template [13]. Prefer this neighbor over the present repository when the official test split or the original summary wording is needed.
- `ChuGyouk/Ko-MTS-Dialog` — 1,201 rows with six columns, a Korean-language variant of the training set only, read live at the check date [11].
- The original upstream is the GitHub repository `abachaa/MTS-Dialog`, which also hosts a 3,603-pair back-translation-augmented training set (English↔French↔Spanish) not mirrored in any of the Hub repositories checked here [2].

## A row

One config, one split, so one shape. From `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [8]:

```json
{
  "ID": 0,
  "section_header": "GENHX",
  "section_text": "Symptoms: no fever, no chills, no cough, no congestion, no nausea, no vomiting, no chest pain, no chest pressure.\nDiagnosis: hypertension, osteoarthritis, osteoporosis, hypothyroidism, allergic rhinitis, kidney stones\nHistory of Patient: 76-year-old white female, presents to the clinic today originally for hypertension and a med check, followed by Dr. Kumar, issues stable\nPlan of Action: N/A",
  "dialogue": "Doctor: What brings you back into the clinic today, miss? \nPatient: I came in for a refill of my blood pressure medicine. \nDoctor: It looks like Doctor Kumar followed up with you last time regarding your hypertension, osteoarthritis, osteoporosis, hypothyroidism, allergic rhinitis and kidney stones. [...] Doctor: I am seventy six years old and identify as a white female."
}
```

## Where it came from

The dialogues and original section summaries come from MTS-Dialog, collected and released for the EACL 2023 paper "An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters" [1], and distributed on GitHub under CC BY 4.0 [2]. This Hub repository takes the original 1,201-row training set and 100-row validation set, concatenates them into a single `train` split, and rewrites every `section_text` into a fixed Symptoms/Diagnosis/History of Patient/Plan of Action template; the card credits five named contributors (Aleena Patani, Amigashabnam F, Harikrishnan K C, Sreeja S, Sujith Jayaprakash) with this modification, done to prepare training data for the `har1/HealthScribe-Clinical_Note_Generator` fine-tune of `facebook/bart-large-cnn` [3][4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Ben Abacha, Yim, Fan, Lin, "An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters", EACL 2023. https://aclanthology.org/2023.eacl-main.168/ — the origin paper; current title read from the live ACL Anthology page. Fetched 2026-08-11.

[2] GitHub repository `abachaa/MTS-Dialog` README. https://github.com/abachaa/MTS-Dialog — split sizes (1,201/100, two 200-row test sets), 20-header scheme, augmented-data description, CC BY 4.0 licence and citation request. Fetched 2026-08-11.

[3] har1/MTS_Dialogue-Clinical_Note dataset card (README). https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note/raw/main/README.md — reformatting description, contributor credits, link to the HealthScribe model. Fetched 2026-08-11.

[4] har1/HealthScribe-Clinical_Note_Generator model card (README). https://huggingface.co/har1/HealthScribe-Clinical_Note_Generator/raw/main/README.md — base model, training row counts, Rouge metrics. Fetched 2026-08-11.

[5] Hugging Face Hub API record for har1/MTS_Dialogue-Clinical_Note. https://huggingface.co/api/datasets/har1/MTS_Dialogue-Clinical_Note?full=true — licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=har1%2FMTS_Dialogue-Clinical_Note Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=har1%2FMTS_Dialogue-Clinical_Note Fetched 2026-08-11.

[8] datasets-server first-rows and rows endpoints. https://datasets-server.huggingface.co/first-rows?dataset=har1%2FMTS_Dialogue-Clinical_Note&config=default&split=train and https://datasets-server.huggingface.co/rows?dataset=har1%2FMTS_Dialogue-Clinical_Note&config=default&split=train&offset=1195&length=20 — row 0 sample; `ID` counter restart between row index 1200 and 1201. Fetched 2026-08-11.

[9] Hugging Face Hub model-search API. https://huggingface.co/api/models?filter=dataset:har1/MTS_Dialogue-Clinical_Note — models declaring this dataset in their card metadata. Fetched 2026-08-11.

[10] datasets-server first-rows endpoint for the neighbor. https://datasets-server.huggingface.co/first-rows?dataset=lyumengxian%2FMTS-Dialog&config=default&split=train — row 0's unmodified `section_text`. Fetched 2026-08-11.

[11] datasets-server size endpoint, one call per neighbor: `lyumengxian/MTS-Dialog`, `X7-qP2-mN9-v4/MTS_Dialogue-Clinical_Note`, `ChuGyouk/Ko-MTS-Dialog`. https://datasets-server.huggingface.co/size?dataset=<id> — this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[12] X7-qP2-mN9-v4/MTS_Dialogue-Clinical_Note dataset card (README). https://huggingface.co/datasets/X7-qP2-mN9-v4/MTS_Dialogue-Clinical_Note/raw/main/README.md — identical text to [3], confirming the duplicate. Fetched 2026-08-11.

[13] lyumengxian/MTS-Dialog dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/lyumengxian/MTS-Dialog/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=lyumengxian%2FMTS-Dialog — split names and row counts (train 1,201 / validation 100 / test 200). Fetched 2026-08-11.

[14] The corpus screening row for `har1/MTS_Dialogue-Clinical_Note`, supplied with this card's request — its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as summarization SFT data, with two shape caveats already established above: the official MEDIQA test sets are absent from this repository [2], and the appended 100-row original validation set sits unmarked inside `train` and must be sliced off before use as a holdout [8]. Both rest on facts fetched directly from the Hub and datasets-server, and agree with the screening row's own note.

### The screening row

The row's own note [14]: "1,301 real short doctor-patient conversations with human-written section summaries (MTS-Dialog train+valid; the MEDIQA-Chat test set is absent)." The row carries no flag.
