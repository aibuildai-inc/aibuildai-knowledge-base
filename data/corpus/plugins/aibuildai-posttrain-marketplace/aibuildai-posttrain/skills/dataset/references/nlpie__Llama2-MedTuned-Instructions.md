# nlpie/Llama2-MedTuned-Instructions

270,318 instruction/input/output triples for biomedical NLP, assembled by reformatting the training subsets of several expert-annotated corpora (NER, relation extraction, NLI, document classification) plus two QA sources into Alpaca-style instructions.

**Llama2-MedTuned-Instructions** was built by NLPie Research (affiliated with the University of Oxford) and introduced in "Exploring the Effectiveness of Instruction Tuning in Biomedical Language Processing" [1], which used it to instruction-tune Llama 2 7B/13B for named entity recognition (NER), relation extraction (RE), natural language inference (NLI), document classification, and question answering (QA), each example following an Instruction/Input/Output template with 5-10 hand-written instruction variants per source task, one sampled per row [1]. **The dataset is released under CC-BY-NC-4.0, non-commercial use only [2]; and because its QA half is sampled from PMC-Llama-Instructions, which the origin paper says bundles MedQA and PubMedQA [1], decontaminate against those two benchmarks before scoring a model tuned on this corpus.** It lives at https://huggingface.co/datasets/nlpie/Llama2-MedTuned-Instructions .

**Use it for**: SFT on Alpaca-style instruction/input/output triples spanning biomedical NER, RE, NLI, document classification and QA task shapes - the SFT method card. Non-commercial use only (CC-BY-NC-4.0) [2]. Rows load directly as `instruction`/`input`/`output` text with no further template needed; `source` names which upstream task/corpus each row came from [3][4].

**Licence**: CC-BY-NC-4.0 (`cardData.license` is `"cc-by-nc-4.0"`), ungated (`"gated": false`, `"private": false`) [2]. The one catch: non-commercial use only; the card body states no further redistribution terms beyond the SPDX tag [5].

**Shape**: 270,318 rows in one config (`default`), split `train` 200,252 / `validation` 70,066, four string columns (`instruction`, `input`, `output`, `source`) [3][4].

**Hold out**: no dedicated test split is served - only `train` (200,252) and `validation` (70,066) [3]. The paper says each task draws from the *training* subset of its source corpus [1], which should keep the official test sets of NCBI-disease, BC5CDR, BC2GM, JNLPBA, i2b2-2010/2012, MedNLI and HoC out of this release; but PMC-Llama-Instructions, one of its two QA sources, itself bundles MedQA and PubMedQA questions [1], and neither this card nor the paper states whether the 50K sampled rows exclude those benchmarks' test items - hold out MedQA and PubMedQA test sets as a precaution before a scored run.

**Origin**: built by NLPie Research (University of Oxford) [1]; labels are a mix of expert/human annotations inherited from the six NER/RE/NLI/classification source corpora and QA answers drawn from ChatDoctor (patient-doctor conversations) and PMC-Llama-Instructions (exam/QA benchmarks), whose own authorship this card does not state [4][5]. Hub API at the check date: `downloads` 296, `downloadsAllTime` 6,761, `likes` 42 [2].

**Trained-on-by**: the origin paper's own models, `nlpie/Llama2-MedTuned-7b` and `nlpie/Llama2-MedTuned-13b`, instruction-tuned on "approximately 200,000 instruction-focused samples" matching this release's size and citing the same paper [1][6]. No other adoption found.

**Introduced by**: [1] (Rohanian et al., published in *Artificial Intelligence in Medicine*; also posted as an arXiv preprint).

## Shape

Rows served and splits (datasets-server `/size`) [3]:

| split | rows |
| --- | --- |
| `train` | 200,252 |
| `validation` | 70,066 |
| total | 270,318 |

One config, `default`, with four columns (datasets-server `/info`) [4]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |
| `source` | string |

Sizes (datasets-server `/size`) [3]: 97,191,993 bytes of Parquet download, 257,096,594 bytes decoded in memory (`train` 206,159,033 / `validation` 50,937,561). The repository's own YAML `dataset_info` states a different total, 265,683,545 bytes (`train` 206,029,981 / `validation` 59,653,564) [2] - the two disagree on both splits, by about 129,052 bytes on `train` and about 8,716,003 bytes on `validation`; both are reported here rather than picking one silently. No source states sequence-length or token statistics for this release.

`source` distinguishes eleven values across the corpus, sampled directly from the served rows rather than declared anywhere: `NCBI-disease`, `BC5CDR-disease`, `BC5CDR-chem`, `BC2GM`, `JNLPBA`, `i2b2-2010`, `i2b2-2012`, `MedNLI`, `hoc`, `chatdoctor`, `pmc_llama` [7]. The paper additionally describes a twelfth source, GAD, used for relation extraction alongside i2b2-2010 [1], but GAD did not appear in any of the 386 rows sampled across both splits for this card (100 `train` rows at offset 0, 20 at offset 100,000, 20 at offset 199,000; 100 `validation` rows at offset 0 plus six windows of 15-66 rows at offsets 10,000-70,000) [7][8], matching the dataset card's own composition list, which names only i2b2-2010 for relation extraction and does not mention GAD [5].

## Quality

- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this release.
- The paper describes the label/annotation pedigree per task: the five NER corpora use plain BIO tagging with no additional label names, while i2b2-2012 uses six category labels with BIO tagging; i2b2-2010 relation extraction follows a tagged sentence-classification format from the authors' own prior work; MedNLI pairs are labeled Entailment/Contradiction/Neutral; HoC is a multi-class document classification set; ChatDoctor was randomly downsampled from 100K real patient-doctor conversations to 50K, and PMC-Llama-Instructions (itself a compilation of QA sets including MedQA and PubMedQA) was randomly downsampled to 50K [1].
- The paper states the final corpus was produced by concatenating all task subsets and shuffling them [1]. That does not match what the served rows show at the offsets checked for this card: every `validation`-split window sampled (offsets 0, 10,000, 20,000, 35,000, 45,000, 55,000, 70,000, each 15-100 rows) held a single `source` value, while the `train`-split windows checked (offsets 0, 100,000, 199,000) each mixed several `source` values [7][8]. This card only speaks to those checked offsets, not to the full 270,318 rows: the `validation` split may still be block-ordered by source rather than shuffled; a reader who needs a shuffled `validation` split should shuffle it themselves rather than assume the paper's description holds.
- None of the 246 `validation` rows sampled carried the `chatdoctor`, `pmc_llama`, or `MedNLI` source values - only NER/RE/classification sources (`NCBI-disease`, `BC5CDR-chem`, `i2b2-2010`, `i2b2-2012`, `hoc`) appeared in the seven `validation` windows checked [7][8]; whether QA and NLI rows exist elsewhere in `validation` was not verified.
- The paper's own downstream evaluation is mixed, not uniformly positive: on the five NER benchmarks (Table 2), Llama2-MedTuned-7b/13b score below the BioBERT-v1.1 baseline on four of five (NCBI-Disease 87.18/85.69 vs 88.62; BC5CDR-Disease 83.92/85.46 vs 86.67; BC5CDR-Chem 93.88/94.51 vs 94.73; BC2GM 76.46/79.12 vs 87.62) and above it on only one, JNLPBA (82.30/81.31 vs 80.33); on the clinical benchmarks (Table 3), Llama2-MedTuned-13b trails BioClinicalBERT on i2b2-2012 NER (80.64 vs 82.98) but leads it on MedNLI (89.46 vs 82.41). The paper reports the largest single gain from tuning on this corpus on MedNLI, where an untuned Llama2 scored 37.20 accuracy against Llama2-MedTuned-13b's 89.46 [1].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-12-03) [2]:

```python
import datasets

REV = "b4fc72ad5a413b06d81a60dd3ea9a7ebb2708794"  # main at the check date
train = datasets.load_dataset("nlpie/Llama2-MedTuned-Instructions", revision=REV, split="train")            # 200,252 rows
validation = datasets.load_dataset("nlpie/Llama2-MedTuned-Instructions", revision=REV, split="validation")  # 70,066 rows
```

**Trap**: the `source` column is not a split - all task types are folded into `train` and `validation` together, so filtering by `source` (e.g. keeping only `chatdoctor` and `pmc_llama` for a QA-only run) requires an explicit `filter()` call, not a `data_dir` or config argument [3][4].

## Neighbors

- `pites/Llama2-MedTuned-Instructions` - the identical 270,318 rows at the identical 200,252/70,066 split, reformatted from four columns down to two: `source` plus a single `text` field holding the pre-rendered `### Instruction: ... ### Input: ... ### Output: ...` prompt string, confirmed by fetching its first `train` row [9][10]. Prefer this corpus's own four-column form for training, since `pites` only pre-renders the same content into one field and adds nothing.
- `HoangHa/Llama2-MedTuned-Instructions_inout` - 205,048 rows in a single `train` split (no `validation`), three columns (`instruction`, `input`, `output`, no `source`) - a different row count from this release, so not a drop-in copy [9].
- The paper's own footnote links this Hub repository as the dataset's canonical location; no other re-release was found [1].

## A row

Both splits share the same four-column schema, but sampling shows they draw from different source mixes at the checked offsets (see Quality) - one row from each illustrates the two content shapes actually observed.

From `config="default"`, `split="train"`, `row_idx=0`, source `chatdoctor` (datasets-server `/first-rows`), with the long clinical answer truncated [7]:

```json
{
  "instruction": "In your role as a medical professional, address the user's medical questions and concerns.",
  "input": "My relative suffering from secondary lever cancer ( 4th stage as per Allopathic doctor) and primary is in rectum. He is continuously with 103 to 104 degree F fever. Allpathic doctor suggested chemo only after fever subsidises. Is treatment possible at Lavanya & what is the time scale of recover.",
  "output": "Hi, dairy have gone through your question. I can understand your concern. He has rectal cancer with liver metastasis. It is stage 4 cancer. Surgery is not possible at this stage. Only treatment options are chemotherapy and radiotherapy according to type of cancer. Inspite of all treatment prognosis [...]",
  "source": "chatdoctor"
}
```

From `config="default"`, `split="validation"`, `row_idx=1`, source `NCBI-disease` (datasets-server `/first-rows`) [8]:

```json
{
  "instruction": "In the given text, your mission is to identify Named Entities referring to diseases and employ the BIO labeling scheme. Mark the start of a disease-related phrase with B (Begin), followed by I (Inner) [...]",
  "input": "Germline mutations in BRCA1 are responsible for most cases of inherited breast and ovarian cancer .",
  "output": "Germline : O\nmutations : O\nin : O\nBRCA1 : O\nare : O\nresponsible : O\nfor : O\nmost : O\ncases : O\nof : O\ninherited : B\nbreast : I\nand : I\novarian : I\ncancer : I\n. : O",
  "source": "NCBI-disease"
}
```

## Where it came from

Built by NLPie Research, affiliated with the Department of Engineering Science at the University of Oxford [1]. The dataset amalgamates the training subsets of several established biomedical corpora, reformatted into the Instruction/Input/Output template: NCBI-disease, BC5CDR-disease, BC5CDR-chem, BC2GM, JNLPBA and i2b2-2012 for NER (plain BIO tagging, except i2b2-2012's six-category BIO scheme); i2b2-2010 (and, per the paper, GAD - see Shape) for relation extraction, reframed as sentence classification with concept tags; MedNLI for natural language inference; the Hallmarks of Cancer (HoC) dataset for document classification; and two QA sources, ChatDoctor (50K rows randomly sampled from 100K real patient-doctor conversations) and PMC-Llama-Instructions (50K rows randomly sampled from a compilation of QA sets including MedQA and PubMedQA) [1]. The paper states all task subsets were concatenated and shuffled into the final ~200K-sample corpus [1] (see Quality for what the served rows show at the offsets checked).

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Rohanian, O., Nouriborji, M. and Clifton, D. A., "Exploring the Effectiveness of Instruction Tuning in Biomedical Language Processing", *Artificial Intelligence in Medicine*, 2024, DOI 10.1016/j.artmed.2024.103007. https://arxiv.org/abs/2401.00579 - the origin paper, whose own byline lists these three authors; task/source composition, prompting strategy, the concatenate-and-shuffle claim, and the Table 2/3 downstream results, read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2401.00579). The dataset README and model-card bibtex ([5]/[6]) cite this paper under a six-author form (adding Kouchaki, Nooralahzadeh and a second Clifton, Lei) that does not match the paper's own byline. Fetched 2026-08-11.

[2] Hugging Face Hub API record for nlpie/Llama2-MedTuned-Instructions. https://huggingface.co/api/datasets/nlpie/Llama2-MedTuned-Instructions?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant; `dataset_info` byte/row counts read from the embedded `cardData`. Fetched 2026-08-11.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=nlpie%2FLlama2-MedTuned-Instructions Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=nlpie%2FLlama2-MedTuned-Instructions Fetched 2026-08-11.

[5] Dataset card (README) for nlpie/Llama2-MedTuned-Instructions. https://huggingface.co/datasets/nlpie/Llama2-MedTuned-Instructions/raw/main/README.md - dataset description, composition list, prompting-strategy summary, citation. Fetched 2026-08-11.

[6] Model card (README) for nlpie/Llama2-MedTuned-7b. https://huggingface.co/nlpie/Llama2-MedTuned-7b/raw/main/README.md Fetched 2026-08-11.

[7] datasets-server first-rows and rows endpoints for `split=train`: https://datasets-server.huggingface.co/first-rows?dataset=nlpie%2FLlama2-MedTuned-Instructions&config=default&split=train (offset 0, 100 rows) and https://datasets-server.huggingface.co/rows?dataset=nlpie%2FLlama2-MedTuned-Instructions&config=default&split=train&offset=100000&length=20 and &offset=199000&length=20. Fetched 2026-08-11.

[8] datasets-server first-rows and rows endpoints for `split=validation`: https://datasets-server.huggingface.co/first-rows?dataset=nlpie%2FLlama2-MedTuned-Instructions&config=default&split=validation (offset 0, 100 rows), and https://datasets-server.huggingface.co/rows?dataset=nlpie%2FLlama2-MedTuned-Instructions&config=default&split=validation with offset/length 70000/66, 35000/20, 10000/15, 20000/15, 45000/15, 55000/15. Fetched 2026-08-11.

[9] datasets-server size endpoint, one call per neighbor: `pites/Llama2-MedTuned-Instructions` and `HoangHa/Llama2-MedTuned-Instructions_inout`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned; row counts and column counts read from the response. Fetched 2026-08-11.

[10] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=pites%2FLlama2-MedTuned-Instructions&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as instruction-tuning (SFT) data, non-commercial only, with a QA-side contamination caveat: the paper's own composition detail (training subsets only, but PMC-Llama-Instructions bundling MedQA/PubMedQA) supports both the licence restriction and the decontamination note in the opening paragraph [1][2], and the screening row's note already flags the same composition and the two-split-only shape [11].

### The screening row

The row's own note [11]: "biomedical NER/RE/NLI/QA turned into instructions from expert-annotated corpora (NCBI-disease, BC5CDR, BC2GM, JNLPBA, i2b2-2010/2012, MedNLI) plus QA from ChatDoctor and PMC-LLaMA-Instructions, whose authorship the card does not state; train/validation only." The row carries no flag.

[11] The corpus screening row for `nlpie/Llama2-MedTuned-Instructions`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.
