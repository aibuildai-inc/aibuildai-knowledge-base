# medalpaca/medical_meadow_pubmed_causal

2,446 Alpaca-style instruction rows that ask a model to classify a PubMed research-conclusion sentence as no, correlational, conditional-causal, or direct-causal, one split, one file.

**medalpaca/medical_meadow_pubmed_causal** repackages the human-annotated causal-language corpus from Yu, Li, and Wang's "Detecting Causal Language Use in Science Findings" [1] into medalpaca's Medical Meadow instruction-tuning collection, and the dataset's own README states it "is the dataset used in" that paper [2]. Each row is a four-way classification task, cast as an `instruction`/`input`/`output` triple: `instruction` gives the four-option question, `input` is a PubMed conclusion sentence, and `output` is the answer in one of four fixed phrasings [3]. It lives at https://huggingface.co/datasets/medalpaca/medical_meadow_pubmed_causal .

**Use it for**: SFT on a fixed-answer classification task - the instruction/input/output triple maps directly to single-turn SFT-chat format, with `instruction`+`input` as the prompt and `output` as the target completion; see the SFT method card. Not preference pairs and not multi-turn dialogue.

**Licence**: not stated. No `license` tag, no `license` field in `cardData`, and no licence text in the README `[2][4]`; the dataset repo's Hub API record carries no license field at all `[4]`. Ungated, public (`"gated": false`, `"private": false`) `[4]`.

**Shape**: 2,446 rows in one config (`default`), one split (`train`), three string columns (`input`, `output`, `instruction`) `[5][6]`.

**Hold out**: not stated by any fetched source as a required holdout; the origin paper's own evaluation splits (5-fold cross-validation and a separate 20% learning-curve test set) are internal to the paper's experiments and are not exposed as columns or splits in this Hub repository `[1]`.

**Origin**: released by medalpaca as part of the Medical Meadow collection; the underlying sentences and their four-way labels are human-annotated, drawn from the origin paper's corpus `[1][2]`. Hub API at the check date (2026-08-11): `downloads` 362, `downloadsAllTime` 8,892, `likes` 10 `[4][7]`.

**Trained-on-by**: the MedAlpaca paper lists "Training data from the Pubmed Causal Benchmark" among the open NLP datasets it folded into Medical Meadow to fine-tune medalpaca-7b and medalpaca-13b, citing this dataset's origin paper directly `[8]`. The deployed `medalpaca-7b` Hub model card's own training-data table, however, does not list Pubmed Causal among its named sources `[9]` - the paper's text is the only fetched source that states this dataset's use in training, and no other model's card or paper naming this dataset was found.

**Introduced by**: [1] (Yu, Li, and Wang, EMNLP-IJCNLP 2019).

## Shape

Rows and splits (datasets-server `/size`) `[5]`: one config `default`, one split `train`, 2,446 rows, 3 columns.

| column | dtype |
| --- | --- |
| `input` | string |
| `output` | string |
| `instruction` | string |

`[6]`

Sizes (datasets-server `/size`) `[5]`: 935,767 bytes original JSON download, 210,954 bytes as Parquet, 846,695 bytes decoded in memory. No source states sequence-length or token statistics for this Hub release.

## Quality

The origin paper's corpus is larger than what the Hub repository serves: the paper reports annotating 3,126 PubMed conclusion sentences in total, of which 3,061 carried exactly one relation type and 65 carried more than one `[1]`. This Hub dataset serves 2,446 rows `[5]`, and no fetched source states how the 3,126- or 3,061-sentence paper corpus maps onto the 2,446-row Hub file - the difference is not explained by any source read here.

The paper's own single-label distribution across its 3,061 single-class sentences: Correlational 998 (32.6%), Conditional Causal 213 (7.0%), Direct Causal 494 (16.1%), No Relationship 1,356 (44.3%) `[1]`. Of the first 100 rows served by the Hub dataset's `/first-rows` endpoint (config `default`, split `train`, offsets 0-99), the `output` label distribution was: "This is a directly correlative relationship" 49, "This is a conditionally causative relationship" 24, "This no relationship." 18, "This is a causative relationship" 9 `[10]`.

The paper states inter-annotator agreement was measured on a separate sample of 30 abstracts labelled by two annotators, giving a Cohen's Kappa of 0.98 `[1]`. Its best classifier (BioBERT, evaluated by 5-fold cross-validation on the 3,061 single-class sentences) reached 0.901 accuracy and 0.881 macro-F1, ahead of BERT (0.889 accuracy, 0.874 macro-F1), a BiRNN (0.836, 0.811), and a linear SVM (0.772, 0.722) `[1]`. These figures describe the paper's own experiments on its full annotated corpus, not a measurement of the 2,446-row Hub file.

## Load it

```python
import datasets

REV = "8a80e781d33544d22b95d26a8d68cafbe8a6470e"  # main at the check date
ds = datasets.load_dataset("medalpaca/medical_meadow_pubmed_causal", revision=REV, split="train")  # 2,446 rows
```

**Trap**: there is only one split, `train`, and no dev/test split - a reader who needs held-out rows for this task must carve them out manually, since the dataset does not provide one `[5][6]`.

## Neighbors

- `mfmezger/deu-medical_meadow_pubmed_causal` - the same 2,446 rows, machine-translated into German, with 6 columns: German `input`/`output`/`instruction` plus the original English fields renamed `input_org`/`output_org`/`instuction_org` (the typo is preserved from the source) `[11][12]`. Fetching row 0 of each shows the German file's `*_org` fields exactly matching the English dataset's row 0 `input`/`output`/`instruction` `[10][12]`. This repo carries no README/card `[13]`.
- `supergoose/buzz_sources_367_medical_meadow_pubmed_causal` - a very small (4-row, 1,074-byte) derivative reformatted into a `conversations` (from/value) list plus `source` and `stack` columns, a different schema from a chat-collator perspective and not a substitute for the full 2,446-row set `[14]`.
- This corpus prefers the original English `medalpaca/medical_meadow_pubmed_causal` for English-language training; the German neighbor is the choice only when German-language SFT data is needed, and the `supergoose` derivative covers too few rows to use on its own.

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) `[10]`:

```json
{
  "input": "A McMahon score of at least 6 calculated on admission allows for a more sensitive, specific and timely identification of patients who may benefit from high-volume fluid resuscitation.",
  "output": "This is a directly correlative relationship",
  "instruction": "Question: Is this describing a (1) directly correlative relationship, (2) conditionally causative relationship, (3) causative relationship, or (0) no relationship."
}
```

## Where it came from

Yu, Li, and Wang built the origin corpus by sampling structured PubMed abstracts across five health topics (nutrition, diabetes, obesity, breast cancer, cholesterol) and both observational and randomized-controlled-trial study designs, extracting each abstract's conclusion subsection and splitting it into sentences with Stanford CoreNLP, then having annotators label each conclusion sentence as no relationship, correlational, conditional causal, or direct causal `[1]`. medalpaca's README states only that this dataset "is the dataset used in" that paper, without describing its own reformatting step from the paper's sentence-and-label corpus into the `instruction`/`input`/`output` triples served here `[2]`.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable, which is why Load it pins the revision.

[1] Yu, Li, and Wang, "Detecting Causal Language Use in Science Findings", Proceedings of EMNLP-IJCNLP 2019, pages 4664-4674. https://aclanthology.org/D19-1473/ - corpus construction, Table 1 (relation-type definitions), Table 3 (label distribution), inter-annotator agreement, Table 4/5 (model results). Fetched via the ACL Anthology page and the paper PDF (https://aclanthology.org/D19-1473.pdf). Fetched 2026-08-11.

[2] medalpaca/medical_meadow_pubmed_causal dataset card (README). https://huggingface.co/datasets/medalpaca/medical_meadow_pubmed_causal/raw/main/README.md Fetched 2026-08-11.

[3] datasets-server first-rows endpoint (schema shown via row 0). https://datasets-server.huggingface.co/first-rows?dataset=medalpaca%2Fmedical_meadow_pubmed_causal&config=default&split=train Fetched 2026-08-11.

[4] Hugging Face Hub API record for medalpaca/medical_meadow_pubmed_causal. https://huggingface.co/api/datasets/medalpaca/medical_meadow_pubmed_causal?full=true - license fields, gate status, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=medalpaca%2Fmedical_meadow_pubmed_causal Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=medalpaca%2Fmedical_meadow_pubmed_causal Fetched 2026-08-11.

[7] Hugging Face Hub API record with `downloadsAllTime` expanded. https://huggingface.co/api/datasets/medalpaca/medical_meadow_pubmed_causal?expand[]=downloadsAllTime Fetched 2026-08-11.

[8] Han et al., "MedAlpaca - An Open-Source Collection of Medical Conversational AI Models and Training Data", arXiv:2304.08247. https://arxiv.org/abs/2304.08247 - Section 2.1.4 lists "Training data from the Pubmed Causal Benchmark", citing this dataset's origin paper. Read via the PDF (https://arxiv.org/pdf/2304.08247). Fetched 2026-08-11.

[9] medalpaca/medalpaca-7b model card (README). https://huggingface.co/medalpaca/medalpaca-7b/raw/main/README.md - training-data source table, which does not list Pubmed Causal. Fetched 2026-08-11.

[10] datasets-server first-rows endpoint, full 100-row response used for label-distribution counts and row 0. https://datasets-server.huggingface.co/first-rows?dataset=medalpaca%2Fmedical_meadow_pubmed_causal&config=default&split=train Fetched 2026-08-11.

[11] mfmezger/deu-medical_meadow_pubmed_causal dataset, size endpoint. https://datasets-server.huggingface.co/size?dataset=mfmezger%2Fdeu-medical_meadow_pubmed_causal - row count and column count. Fetched 2026-08-11.

[12] mfmezger/deu-medical_meadow_pubmed_causal, first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mfmezger%2Fdeu-medical_meadow_pubmed_causal&config=default&split=train - row 0, columns. Fetched 2026-08-11.

[13] mfmezger/deu-medical_meadow_pubmed_causal README fetch. https://huggingface.co/datasets/mfmezger/deu-medical_meadow_pubmed_causal/raw/main/README.md - returned "Entry not found"; Hub API `cardData` for this repo is also null. Fetched 2026-08-11.

[14] supergoose/buzz_sources_367_medical_meadow_pubmed_causal, Hub API record. https://huggingface.co/api/datasets/supergoose/buzz_sources_367_medical_meadow_pubmed_causal - `cardData.dataset_info` gives schema, row count (4), and byte size (1,074). Fetched 2026-08-11.

[15] The corpus screening row for `medalpaca/medical_meadow_pubmed_causal`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT data for a fixed-answer causal-language classification task. The dataset's own README ties it to a peer-reviewed, human-annotated origin corpus `[1][2]`, and the screening row's note describes the same task shape.

### The screening row

The row's own note `[15]`: "classify causal language in PubMed findings sentences; labels from the EMNLP 2019 human-annotated corpus." The row carries no flag.
