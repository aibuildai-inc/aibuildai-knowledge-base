# lavita/MedQuAD

47,441 consumer-health question-answer rows scraped from 12 NIH/NLM websites, converted to Parquet from the original MedQuAD XML release - and 31,034 of those rows (65.4%) carry an empty `answer` field.

**lavita/MedQuAD** is a Hugging Face user's Parquet conversion of the MedQuAD collection built by Asma Ben Abacha and Dina Demner-Fushman and introduced in "A Question-Entailment Approach to Question Answering" [1]. The original collection extracted 47,457 question-answer pairs from 12 NIH consumer-health sites (e.g. cancer.gov, niddk.nih.gov, GARD, MedlinePlus Health Topics), annotating each with a question type, question focus, UMLS CUI, and semantic type [2]. This repository's dataset card states that four of those source collections - GARD, MPlusHerbsSupplements, ADAM, MPlusDrugs - have had their `answer` text stripped to respect MedlinePlus's copyright, affecting 31,034 records [3]. **The stripped rows are still served, with an empty `answer` string, not removed from the file: they must be filtered out before any QA training run, leaving 16,407 usable rows.** It lives at https://huggingface.co/datasets/lavita/MedQuAD .

**Use it for**: SFT on consumer-health question answering, using `question` as the prompt and `answer` as the target - but only after filtering to the 16,407 rows with a non-empty `answer` (see Quality); maps to the plain instruction/response format used by the SFT method card.

**Licence**: not stated on this Hub repository (no `license` tag or `cardData.license`) [4]; the original GitHub source states the MedQuAD collection is released under Creative Commons Attribution 4.0 International (CC BY) [2] - this Hub repo does not restate that grant itself.

**Shape**: 47,441 rows, one split (`train`), one config, 13 string columns [5][6].

**Hold out**: no source names an evaluation split or a known eval-set overlap for this release, so there is nothing to hold out on contamination grounds; the exclusion required before training is the empty-answer filter above, not an eval holdout.

**Origin**: converted and hosted by Hugging Face user `lavita`; the underlying QA pairs are extracted/curated from NIH-authored web content by the paper's two human authors, not model-generated [1][2]. At the check date the Hub API reports 5,329 downloads (25,153 all-time) and 23 likes [4].

**Trained-on-by**: 24 Hub models declare `lavita/MedQuAD` as a training dataset in their metadata, the most-downloaded being `mradermacher/Mixtral_AI_DeepMedicalMind-GGUF` (1,070 downloads) and `mradermacher/Medra-i1-GGUF` (749 downloads) [7].

**Introduced by**: [1], the origin paper for the MedQuAD collection; this Hub repository's own card is the source for its conversion and copyright-redaction notes [3].

## Shape

Splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 47,441 |

One config, `default`, 13 string columns (datasets-server `/info`) [6]: `document_id`, `document_source`, `document_url`, `category`, `umls_cui`, `umls_semantic_types`, `umls_semantic_group`, `synonyms`, `question_id`, `question_focus`, `question_type`, `question`, `answer`.

Byte sizes differ between the pinned card metadata and the live measurement. The README's front-matter `dataset_info` (read at the pinned revision) states `download_size` 10,718,159 bytes and `dataset_size` (in-memory) 34,989,308 bytes [3]. The live, unpinned datasets-server `/size` and `/info` endpoints report the same 10,718,159-byte download but a larger 58,769,056-byte in-memory size [5][6]; no source explains the gap, and no source states sequence-length or token statistics for this release.

Reading the full served Parquet file directly [8] gives the `document_source` breakdown behind the 47,441 rows:

| `document_source` | rows | rows with empty `answer` |
| --- | --- | --- |
| ADAM | 17,348 | 17,348 |
| MPlusDrugs | 12,889 | 12,889 |
| GHR | 5,430 | 0 |
| GARD | 5,394 | 5 |
| NIDDK | 1,192 | 0 |
| NINDS | 1,088 | 0 |
| MPlusHealthTopics | 981 | 0 |
| MPlusHerbsSupplements | 792 | 792 |
| NIHSeniorHealth | 769 | 0 |
| CancerGov | 729 | 0 |
| NHLBI | 559 | 0 |
| CDC | 270 | 0 |
| **total** | **47,441** | **31,034** |

Twelve distinct `document_source` values are served, matching the original collection's claim of 12 NIH source websites [2]. The `category` column is populated for 32,010 of 47,441 rows (16,256 `Disease`, 13,681 `Drug`, 2,073 `Other`) and null for the rest [8]. `question_type` carries 39 distinct values in the served data; the card's own discrepancy table already documents several places where the dataset's question-type labels do not match the labels used in the origin paper (e.g. dataset `outlook` maps to the paper's `prognosis`) [3].

## Quality

- Of the 47,441 served rows, 31,034 (65.4%) have an empty-string `answer`, read directly from the full Parquet file at the pinned revision [8]. This count matches the card's stated redaction of 31,034 records for the GARD, MPlusHerbsSupplements, ADAM, and MPlusDrugs sources, "to respect the MedlinePlus copyright" [3]. The breakdown above shows the redaction is source-complete for ADAM and MPlusDrugs (100% of their rows are empty-answer) and near-complete for MPlusHerbsSupplements (100%) and GARD (5 of 5,394 rows, 0.09%).
- The `question` field is intact for all 47,441 rows in the sample read; no source states a separate empty-question count.
- No source states a measured duplicate rate, annotator-agreement figure, or automated quality score for this release; none is invented here. The card's only stated quality caveat is the question-type mismatch against the origin paper, given as a mapping table rather than a rate [3].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-12-22) [4]:

```python
import datasets

REV = "84ea67f83cec9692ad254eaa02c9731b24ecfe4c"  # main at the check date
ds = datasets.load_dataset("lavita/MedQuAD", revision=REV, split="train")  # 47,441 rows
usable = ds.filter(lambda r: r["answer"] is not None and r["answer"].strip() != "")  # 16,407 rows
```

**Trap**: loading `train` directly and treating every row as a QA pair silently includes the 31,034 rows (65.4%) whose `answer` is an empty string, sourced from the GARD, MPlusHerbsSupplements, ADAM, and MPlusDrugs subsets [3][8] - filter on a non-empty `answer` first, as above.

## Neighbors

- `keivalya/MedQuad-MedicalQnADataset` - 16,407 rows, 3 columns (`qtype`, `Question`, `Answer`) [9]. Fetching its first three served rows and matching their question/answer text against this release's non-empty-answer subset confirms an exact match, including one two-way-duplicated question ("Who is at risk for Lymphocytic Choriomeningitis (LCM)?") that appears identically in both [8][9]. This dataset IS the pre-filtered, non-empty-answer subset of this release, reduced to three columns; prefer this release's own filtered rows over re-downloading it, since they are the same data.
- `AnonymousSub/MedQuAD_47441_Question_Answer_Pairs` - 47,441 rows, 2 columns (`Questions`, `Answers`) [10], the same row count as this release including its empty-answer rows; only 93 downloads at the check date [11], and no source states whether its `Answers` column preserves the empty strings or was otherwise altered.
- The same Hub user, `lavita`, also hosts `lavita/medical-qa-datasets`, a 27-config collection of other medical QA sources (ChatDoctor, MedMCQA, PubMedQA, USMLE self-assessments, and more); its config list carries no MedQuAD entry, so it does not duplicate this release [12].
- The original collection's home is the GitHub repository `abachaa/MedQuAD`, which also hosts the separate LiveQA-Med test-question judgment file referenced in the origin paper; that repository states the MedQuAD collection's own licence as CC BY 4.0 [2].

## A row

One config and one split are served, so one row covers the shape. From `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [13], an ordinary row with a non-empty answer:

```json
{
  "document_id": "0000559",
  "document_source": "GHR",
  "document_url": "https://ghr.nlm.nih.gov/condition/keratoderma-with-woolly-hair",
  "category": null,
  "umls_cui": "C0343073",
  "umls_semantic_types": "T047",
  "umls_semantic_group": "Disorders",
  "synonyms": "KWWH",
  "question_id": "0000559-1",
  "question_focus": "keratoderma with woolly hair",
  "question_type": "information",
  "question": "What is (are) keratoderma with woolly hair ?",
  "answer": "Keratoderma with woolly hair is a group of related conditions that affect the skin and hair [...] Type IV does not appear to cause cardiomyopathy."
}
```

A row from the redacted majority, the first ADAM-sourced record in the full Parquet file [8], has the same 13 fields but an empty `answer` and null UMLS fields:

```json
{
  "document_id": "0003093",
  "document_source": "ADAM",
  "document_url": "https://www.nlm.nih.gov/medlineplus/ency/article/002589.htm",
  "category": "Other",
  "umls_cui": null,
  "umls_semantic_types": null,
  "umls_semantic_group": null,
  "synonyms": "Pyrethrins poisoning|Lice medicine poisoning",
  "question_id": "0003093-1",
  "question_focus": "Piperonyl butoxide with pyrethrins poisoning",
  "question_type": "information",
  "question": "Do you have information about Piperonyl butoxide with pyrethrins poisoning",
  "answer": null
}
```

## Where it came from

The MedQuAD collection was built by Asma Ben Abacha and Dina Demner-Fushman and introduced in their 2019 BMC Bioinformatics paper, which combines it with a question-entailment (RQE) method to answer new medical questions by mapping them to previously-answered, "similar" questions [1]. The QA pairs themselves were extracted from 12 NIH-run consumer-health websites, with additional annotations (question type, question focus, synonyms, UMLS CUI and semantic type) added by the authors, and a `category` (Disease, Drug, or Other) added for the four MedlinePlus collections [2]. Answers for three of those MedlinePlus-sourced subsets - A.D.A.M. Medical Encyclopedia, MedlinePlus Drug information, and MedlinePlus Herbal medicine/supplement information - were removed from the original GitHub release to respect MedlinePlus's copyright, while the URLs were kept so that answers could be re-crawled [2]. This Hub repository is a third-party Parquet conversion of that release, whose own card additionally lists GARD among the redacted-answer sources and gives a combined count of 31,034 affected records [3], consistent with the source-level counts found in the served Parquet file [8].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Ben Abacha, A. and Demner-Fushman, D., "A question-entailment approach to question answering", BMC Bioinformatics, 2019. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4 - origin paper; abstract, title, and the 47,457-pair MedQuAD collection description read from the live article page. Fetched 2026-08-11.

[2] `abachaa/MedQuAD` GitHub repository `readme.txt`. https://raw.githubusercontent.com/abachaa/MedQuAD/master/readme.txt - 47,457-pair count, 12 NIH sources, 37 question types, the three redacted MedlinePlus subsets, and the CC BY 4.0 licence statement. Fetched 2026-08-11.

[3] `lavita/MedQuAD` dataset card (README). https://huggingface.co/datasets/lavita/MedQuAD/raw/main/README.md - conversion notes, the four redacted sources and 31,034-record count, and the question-type discrepancy table. Fetched 2026-08-11.

[4] Hugging Face Hub API record for lavita/MedQuAD. https://huggingface.co/api/datasets/lavita/MedQuAD?full=true - licence/gate absence, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=lavita%2FMedQuAD Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=lavita%2FMedQuAD Fetched 2026-08-11.

[7] Hugging Face Hub API model search filtered by declared training dataset. https://huggingface.co/api/models?filter=dataset:lavita/MedQuAD - live, unpinned; lists 24 models and their download counts at the check date. Fetched 2026-08-11.

[8] The full `lavita/MedQuAD` Parquet file at the pinned revision, read directly. https://huggingface.co/datasets/lavita/MedQuAD/resolve/main/data/train-00000-of-00001-e36383d177026d53.parquet - all 47,441 rows read; source of the empty-`answer` counts, the `document_source`/`category` breakdowns, the `question_type` count, and the row sample. Fetched 2026-08-11.

[9] `keivalya/MedQuad-MedicalQnADataset` - datasets-server info/size/first-rows endpoints. https://datasets-server.huggingface.co/info?dataset=keivalya%2FMedQuad-MedicalQnADataset , https://datasets-server.huggingface.co/size?dataset=keivalya%2FMedQuad-MedicalQnADataset , https://datasets-server.huggingface.co/first-rows?dataset=keivalya%2FMedQuad-MedicalQnADataset&config=default&split=train - row count, columns, and sampled rows used for the text match against this release. Fetched 2026-08-11.

[10] `AnonymousSub/MedQuAD_47441_Question_Answer_Pairs` - datasets-server info/size endpoints. https://datasets-server.huggingface.co/info?dataset=AnonymousSub%2FMedQuAD_47441_Question_Answer_Pairs , https://datasets-server.huggingface.co/size?dataset=AnonymousSub%2FMedQuAD_47441_Question_Answer_Pairs - row count and columns. Fetched 2026-08-11.

[11] Hugging Face Hub API dataset search. https://huggingface.co/api/datasets?search=medquad - live, unpinned search across dataset repos matching "medquad"; source of the 93-download figure for `AnonymousSub/MedQuAD_47441_Question_Answer_Pairs`, which the info/size endpoints in [10] do not carry. Fetched 2026-08-11.

[12] Hugging Face Hub API record for lavita/medical-qa-datasets. https://huggingface.co/api/datasets/lavita/medical-qa-datasets?full=true - full config list, checked for a MedQuAD config. Fetched 2026-08-11.

[13] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=lavita%2FMedQuAD&config=default&split=train Fetched 2026-08-11.

[14] The corpus screening row for `lavita/MedQuAD`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable, but only after the empty-answer filter this card documents: 47,441 rows are served, and 31,034 of them (65.4%) carry an empty `answer` field, matching the card's own account of copyright-driven redaction across four sources (established above) [3][8]. The screening row's note names the same redaction and gives the same 47k/four-source shape [14].

### The screening row

The row's own note [14]: "47k consumer-health QA curated from NIH/NLM sites; answers from four copyright-restricted sources were removed." The row carries no flag.
