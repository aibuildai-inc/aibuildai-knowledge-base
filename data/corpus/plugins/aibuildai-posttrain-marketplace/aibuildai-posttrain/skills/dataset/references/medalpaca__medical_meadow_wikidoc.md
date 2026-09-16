# medalpaca/medical_meadow_wikidoc

10,000 medical instruction/input/output triplets - one fixed instruction string, a question, and an answer paragraph - machine-generated from the WikiDoc "Living Textbook" medical wiki.

**medalpaca/medical_meadow_wikidoc** is part of Medical Meadow, the training-data collection behind MedAlpaca, a LLaMA-based medical question-answering model suite [1]. The builders crawled WikiDoc's "Living Textbook" subsite, a collaborative medical-specialty reference, then used GPT-3.5-Turbo to rephrase each paragraph's heading into a question, pairing it with the paragraph itself as the answer [2]. **The dataset card states it is "still a WIP" and that the GPT-3.5-Turbo rephrasing step "yielded some unsatisfactory results in approximately 30% of cases" [2]; the 10,000 rows served here are also a subsample of a larger 67,704-row Living Textbook pool the same builders report elsewhere, with no documented sampling method for which rows were kept [3].** It lives at https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc .

**Use it for**: instruction-tuned SFT - each row is an Alpaca-style `instruction`/`input`/`output` triplet (the `instruction` field is the constant string "Answer this question truthfully" across every sampled row, so `input` carries the actual question and `output` the answer) [4]. Maps to the SFT method card's instruction-input-output format. Given the card's own ~30% unsatisfactory-conversion admission, treat it as noisy SFT data, not a cleaned gold set [2].

**Licence**: tagged `license:cc` in both the card's YAML front matter and the Hub API, with no Creative Commons variant (no BY/SA/0/NC suffix) named anywhere on the card [2][5]. The catch: this is an unspecified "cc" grant, not a resolvable SPDX id.

**Shape**: 10,000 rows, one config (`default`), one split (`train`); three string columns (`instruction`, `input`, `output`) [6][7].

**Hold out**: nothing named. No source states an evaluation-set overlap risk for this release, and MedAlpaca's own USMLE self-assessment benchmark is a separate, distinct dataset from a different origin [3].

**Origin**: built by the medalpaca team (kbressem et al.); questions and answers are machine-derived (GPT-3.5-Turbo rephrasing of WikiDoc text), not human-written or human-labeled [2]. Hub API at the check date: `downloads` 4,330, `downloadsAllTime` 28,603, `likes` 56 [5].

**Trained-on-by**: MedAlpaca 7b and 13b (and their LoRA variants) were finetuned on the Medical Meadow collection, whose own data table lists this Wikidoc component as 67,704 original rows reduced to 10,000 preprocessed rows - the exact row count served by this repository - so this dataset is part of that finetuning mix [3].

**Introduced by**: the dataset card gives no paper ("Paper: TBA") [2]; the MedAlpaca paper, submitted a week after the card's last edit, separately describes this component's construction and its 67,704-row original size [1].

## Shape

Splits and rows (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 10,000 |

One config, `default`, three columns, all strings (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |

Sizes (datasets-server `/size`) [6]: 10,637,914 bytes of original JSON download, 5,593,193 bytes as Parquet, 10,224,074 bytes decoded in memory. No source states sequence-length or token statistics for this release.

## Quality

- Every question/answer pair is machine-generated: GPT-3.5-Turbo rephrased a WikiDoc paragraph heading into a question, and the paragraph text was used verbatim as the answer, with no human review step stated [2].
- The card's only stated quality signal is its own admission that the GPT-3.5-Turbo rephrasing "yielded some unsatisfactory results in approximately 30% of cases," and that the dataset is still a work in progress being cleaned [2]. No source gives a duplicate-rate or contamination figure.
- Reading the first 100 served rows at offset 0 (`train`, `default`) shows the `instruction` field holding the single fixed string "Answer this question truthfully" in all 100; no source states whether this holds for the remaining 9,900 rows [4].
- The companion project README shows this repository's 10,000 rows come from a 67,704-row Living Textbook pool the same builders report, with no stated criterion for which 10,000 were kept [3].

## Load it

Single JSON file, single split, no config argument needed; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-04-06) [5]:

```python
import datasets

REV = "976bce16fd66f55e6828422b71249a7405a11aa4"  # main at the check date
ds = datasets.load_dataset("medalpaca/medical_meadow_wikidoc", revision=REV, split="train")  # 10,000 rows
```

**Trap**: there is only one split (`train`) and no held-out test rows are provided in this repository; anyone needing an evaluation split must build one themselves. The 10,000 rows are also not the full Living Textbook conversion - a further ~57,700 converted rows exist per the builders' own project table but are not in this Hub repository [3].

## Neighbors

- `medalpaca/medical_meadow_wikidoc_patient_information` - the sibling WikiDoc subsite, "Patient Information," where each section subheading is already phrased as a question, so no GPT-3.5 rephrasing step was needed; 5,942 rows, same three-column instruction/input/output schema, checked live at the check date [8][9]. The dataset card for this repository cross-links to it directly and vice versa [2][8]. Use it alongside this dataset rather than instead of it - it is a different WikiDoc subsite, not a cleaned version of the Living Textbook data.
- No cleaned, deduplicated, or binarized re-release of the Living Textbook data (addressing the card's own ~30% unsatisfactory-conversion admission) was found.

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [4]:

```json
{
  "instruction": "Answer this question truthfully",
  "input": "Can you provide an overview of the lung's squamous cell carcinoma?",
  "output": "Squamous cell carcinoma of the lung may be classified according to the WHO histological classification system into 4 main types: papillary, clear cell, small cell, and basaloid."
}
```

## Where it came from

Built by the medalpaca team (kbressem et al.) as part of the Medical Meadow collection behind MedAlpaca [1][3]. The upstream pool is WikiDoc's "Living Textbook" subsite, a collaborative medical-specialty wiki that the builders crawled; each paragraph heading was rephrased into a question by GPT-3.5-Turbo, and the original paragraph text was kept as the answer unchanged [2]. The card names no human annotation or review step over the generated pairs [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] "MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data," 2023. https://arxiv.org/abs/2304.08247 - MedAlpaca project description and Dataset 3 (Wikidoc) section giving the 67,704-row original Living Textbook count; current title read from the live abs page. Fetched 2026-08-11.

[2] medalpaca/medical_meadow_wikidoc dataset card (README). https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc/raw/main/README.md - construction description, WIP/~30% unsatisfactory-conversion note, license tag, sibling-dataset link, "Paper: TBA." Fetched 2026-08-11.

[3] kbressem/medAlpaca project README (GitHub). https://raw.githubusercontent.com/kbressem/medAlpaca/main/README.md - Medical Meadow data table (Wikidoc: 67,704 original / 10,000 preprocessed rows), list of MedAlpaca 7b/13b and LoRA models finetuned on Medical Meadow. Fetched 2026-08-11.

[4] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=medalpaca%2Fmedical_meadow_wikidoc&config=default&split=train - 100 sampled rows read at offset 0; row 0 used as the example row. Fetched 2026-08-11.

[5] Hugging Face Hub API record for medalpaca/medical_meadow_wikidoc. https://huggingface.co/api/datasets/medalpaca/medical_meadow_wikidoc?full=true - license tag, gate status, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=medalpaca%2Fmedical_meadow_wikidoc Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=medalpaca%2Fmedical_meadow_wikidoc Fetched 2026-08-11.

[8] medalpaca/medical_meadow_wikidoc_patient_information dataset card (README). https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information/raw/main/README.md Fetched 2026-08-11.

[9] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=medalpaca%2Fmedical_meadow_wikidoc_patient_information Fetched 2026-08-11.

[10] The corpus screening row for `medalpaca/medical_meadow_wikidoc`, supplied with this card's request - its `note`, read back in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as noisy instruction-tuning SFT data, with the caveat the card itself states: about 30% of the GPT-3.5-Turbo paragraph-to-QA conversions were unsatisfactory at card-writing time, and this repository's 10,000 rows are a subsample of a larger 67,704-row pool with no documented selection method. Both facts are established above from the dataset card and the companion project README [2][3], matching the screening row's note.

### The screening row

The row's own note [10]: "WikiDoc 'Living Textbook' content turned into QA by GPT-3.5-turbo; the card admits ~30% of conversions were unsatisfactory." The row carries no flag.
