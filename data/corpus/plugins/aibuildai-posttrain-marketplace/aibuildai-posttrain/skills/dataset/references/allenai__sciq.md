# allenai/sciq

13,679 crowdsourced multiple-choice science exam questions, each with a correct answer, three distractors, and - for most rows - a supporting evidence passage.

**allenai/sciq** is the SciQ dataset, released by the Allen Institute for Artificial Intelligence and introduced in "Crowdsourcing Multiple Choice Science Questions" [1]: crowd workers wrote closed-domain, 4-option multiple-choice science questions (Physics, Chemistry, Biology, and others) with the aid of a large domain text corpus and model suggestions for document selection and distractor choice [1]. **The corpus screening note for this dataset flags `test` (1,000 rows) as "the SciQ benchmark"; hold it out of training and use only `train` and `validation`** [2]. It lives at https://huggingface.co/datasets/allenai/sciq .

**Use it for**: closed- or open-domain science multiple-choice QA SFT - present `question` (and `support` when non-empty, for the open-book/reading-comprehension shape) with the four options built from `correct_answer` plus the three `distractor*` fields, and train the model to produce `correct_answer`. This is plain QA/SFT data, not preference pairs; it maps to the SFT method card's instruction-response format, not a DPO-style implicit- or explicit-prompt pair.

**Licence**: CC-BY-NC-3.0 (SPDX id; `cardData.license` is `"cc-by-nc-3.0"`, repo tag `license:cc-by-nc-3.0`), ungated (`"gated": false`, `"private": false`) [3][4]. The one catch: noncommercial only - the README states the dataset "is licensed under the Creative Commons Attribution-NonCommercial 3.0 Unported License" [4].

**Shape**: 13,679 rows in one config (`default`), split `train` 11,679 / `validation` 1,000 / `test` 1,000, six string columns [5][6].

**Hold out**: `test` (1,000 rows); `train` (11,679) and `validation` (1,000) are safe to train on. The screening row's note says so in those words - "`train`+`validation` are safe, `test` is the SciQ benchmark" [2].

**Origin**: built and released by the Allen Institute for AI; questions, distractors, and support passages come from human crowd workers, with automated suggestions aiding document selection and distractor choice, not authorship [1]. Hub API at the check date: `downloads` 148,127, `downloadsAllTime` 3,887,346, `likes` 145 [3].

**Trained-on-by**: the origin paper's own AS Reader and GA Reader models, fine-tuned on 4th- and 8th-grade real science exam questions and then re-trained with SciQ's training portion added as augmentation data, showing the deciding comparison below [1]. No adoption by a named open post-training model or recipe beyond the origin paper's own experiments was found in what was checked.

**Introduced by**: [1] (Welbl, Liu, and Gardner).

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 11,679 |
| `validation` | 1,000 |
| `test` | 1,000 |
| total | 13,679 |

One config, `default`, with six columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `question` | string |
| `distractor3` | string |
| `distractor1` | string |
| `distractor2` | string |
| `correct_answer` | string |
| `support` | string |

Sizes (datasets-server `/size`) [5]: 4,674,410 bytes of original Parquet download, 7,723,117 bytes decoded in memory. No source states sequence-length or token statistics for this release; the origin paper reports question/answer/distractor length only as a figure ("Figure 2: Total counts of question, answer and distractor length, measured in number of tokens"), with no numeric values in the extracted text [1].

## Quality

- The README states an additional supporting passage "is provided" "for the majority of the questions" [4]. Reading the first 100 served rows of `train` at offset 0, 93 of 100 (`support` non-empty) carry a passage; reading the first 100 served rows of `test` at offset 0, 87 of 100 carry a passage [7][8]. These are counts over the first 100 rows read at offset 0 in each split, not a claim about the full 11,679-row `train` or 1,000-row `test` split.
- The origin paper reports a human-vs-crowdsourced discrimination check: judges shown one original exam question and one SciQ question picked the real exam question in 55% of 100 trials, below the paper's own p=0.05 significance threshold against a random-guess null [1].
- The paper's Table 2 gives baseline test-set accuracy on the multiple-choice version of SciQ: Aristo ensemble 77.4, Lucene 80.0, TableILP 31.8, AS Reader 74.1, GA Reader 73.8, against Humans 87.8 ± 0.045 [1] - the multiple-choice systems retrieve their own background text rather than using the `support` field, since the paper excludes the passage from the multiple-choice setting to avoid making the question trivial [1].
- The paper's Table 3, the training-augmentation result, shows the deciding number: on real 4th-grade science exam questions, AS Reader/GA Reader accuracy is 40.7%/37.6% without SciQ and 45.0%/45.4% with SciQ added (+4.3/+7.8 points); on 8th-grade questions, 41.2%/41.0% without SciQ versus 43.0%/44.3% with SciQ added (+1.8/+3.3 points) [1].
- No source states a measured contamination or duplicate-rate figure for this release; none is invented here.

## Load it

Train on `train` (and `validation`, per the origin paper's own splits [1]), hold out `test`, and pin the repo revision this card was written against (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-01-04) [3]. This pin makes `load_dataset` reproducible at the row level; it does not cover any of the other numbers on this card - the download/like counts, licence and gate flags, row counts, byte sizes, column list, or sampled `support`-field percentages are all read from endpoints that ignore the revision parameter and always serve their current state - see the Sources preamble:

```python
import datasets

REV = "2c94ad3e1aafab77146f384e23536f97a4849815"  # main at the check date
train = datasets.load_dataset("allenai/sciq", revision=REV, split="train")            # 11,679 rows
validation = datasets.load_dataset("allenai/sciq", revision=REV, split="validation")   # 1,000 rows
test = datasets.load_dataset("allenai/sciq", revision=REV, split="test")               # 1,000 rows - hold out
```

**Trap**: the `support` field is empty for a share of rows in every split checked above (7% of the first 100 `train` rows, 13% of the first 100 `test` rows) [7][8]; a prompt template that always inserts `support` will insert an empty string on those rows rather than skipping the open-book context, silently degrading a chunk of examples to closed-book without any format change to signal it.

## Neighbors

- `bigbio/sciq` - the same 13,679 rows and splits, offered in two configs: `sciq_source` (identical six-column schema to this release, confirmed by fetching its `/size`) and `sciq_bigbio_qa` (an 8-column BigBio-standardized QA schema); its own card names this repository's homepage (`https://allenai.org/data/sciq`) as its source [9][10]. Same underlying data reshaped for BigBio benchmark unification, not an independent collection - prefer this original repository unless a pipeline specifically needs BigBio's schema.
- `izumi-lab/sciq-ja-mbartm2m` - the same 13,679 rows and splits, machine-translated (mBART) into Japanese, same six-column schema, confirmed by fetching its `/size` and `/info` [11].
- A Hub search for "sciq" at the check date turns up many further small derivative repositories (standardized reformats, interpretability probing sets built on SciQ such as the `quirky_sciq*` family, ad-hoc reuploads) each with under 150 downloads, well below this repository's 148,127 [12]; none showed evidence of independent adoption worth treating as a neighbor here.

## A row

The repository serves one config and one schema across all three splits, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7]:

```json
{
  "question": "What type of organism is commonly used in preparation of foods such as cheese and yogurt?",
  "distractor3": "viruses",
  "distractor1": "protozoa",
  "distractor2": "gymnosperms",
  "correct_answer": "mesophilic organisms",
  "support": "Mesophiles grow best in moderate temperature, typically between 25°C and 40°C (77°F and 104°F). Mesophiles are often found living in or on the bodies of humans or other animals. The optimal growth temperature of many pathogenic mesophiles is 37°C (98°F), the normal human body temperature. Mesophilic organisms have important uses in food preparation, including cheese, yogurt, beer and wine."
}
```

## Where it came from

Built and released by the Allen Institute for Artificial Intelligence. The origin paper describes a crowdsourcing pipeline for generating domain-targeted multiple-choice questions: workers were given a passage from a large domain-specific text corpus plus a small seed set of existing questions, and the method produced automated suggestions for which document to use and which answer-distractor candidates to offer, which workers then edited into final questions and distractors [1]. The paper additionally released a direct-answer variant (passage plus question, no options, 10,481/887/884 rows across train/dev/test) that is smaller than the multiple-choice version because some source passages could not be released, but that variant is not what this repository's `default` config serves - this config's schema (`question`, three `distractor*`, `correct_answer`, `support`) is the multiple-choice version [1][4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision - but the pin covers only what `load_dataset` reads from the repo's files at that commit. Every other live endpoint cited on this card takes no revision parameter and always serves its current state, not the pinned commit: this was verified directly by comparing a call with `&revision=2c94ad3e1aafab77146f384e23536f97a4849815` against a call with no revision parameter, and finding byte-identical JSON, against both the Hub API record [3] and the datasets-server `/info` endpoint [6]. That covers every number sourced from [3] (downloads, likes, licence and gate flags) and from the datasets-server `/size`, `/info`, and `/first-rows` endpoints [5][6][7][8][10][11] - all current as of the check date, not frozen at the pinned sha.

[1] Welbl, Liu, and Gardner, "Crowdsourcing Multiple Choice Science Questions", 2017. https://arxiv.org/abs/1707.06209 - the origin paper; current title read from the live abs page; body text read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/1707.06209). Fetched 2026-08-11.

[2] The corpus screening row for `allenai/sciq`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[3] Hugging Face Hub API record for allenai/sciq. https://huggingface.co/api/datasets/allenai/sciq?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Takes no revision parameter: a call with `&revision=2c94ad3e1aafab77146f384e23536f97a4849815` and a call with no revision parameter returned byte-identical JSON; live, not pinned. Fetched 2026-08-11.

[4] allenai/sciq dataset card (README). https://huggingface.co/datasets/allenai/sciq/raw/main/README.md - dataset summary, licensing information, data splits. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=allenai%2Fsciq - takes no revision parameter (verified against `/info`, below); live, not pinned. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=allenai%2Fsciq - takes no revision parameter; a call with `&revision=2c94ad3e1aafab77146f384e23536f97a4849815` and a call with no revision parameter returned byte-identical JSON; live, not pinned. Fetched 2026-08-11.

[7] datasets-server first-rows endpoint, `train` split. https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Fsciq&config=default&split=train - takes no revision parameter; live, not pinned. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint, `test` split. https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Fsciq&config=default&split=test - takes no revision parameter; live, not pinned. Fetched 2026-08-11.

[9] bigbio/sciq dataset card (README). https://huggingface.co/datasets/bigbio/sciq/raw/main/README.md Fetched 2026-08-11.

[10] datasets-server size and info endpoints for bigbio/sciq. https://datasets-server.huggingface.co/size?dataset=bigbio%2Fsciq and https://datasets-server.huggingface.co/info?dataset=bigbio%2Fsciq - take no revision parameter; live, not pinned. Fetched 2026-08-11.

[11] datasets-server size and info endpoints for izumi-lab/sciq-ja-mbartm2m. https://datasets-server.huggingface.co/size?dataset=izumi-lab%2Fsciq-ja-mbartm2m and https://datasets-server.huggingface.co/info?dataset=izumi-lab%2Fsciq-ja-mbartm2m - take no revision parameter; live, not pinned. Fetched 2026-08-11.

[12] Hugging Face Hub dataset search API for "sciq". https://huggingface.co/api/datasets?search=sciq&limit=30 - this endpoint takes no revision parameter, so these download counts are live, not pinned. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT/QA training data, with `test` held out: train on `train` and `validation`, never `test`. Two facts decide it, and both are already established above - the dataset is plain human-authored multiple-choice science QA with a supporting passage on most rows [1][4], and the screening row's note identifies `test` as the SciQ benchmark split to hold out [2].

### The screening row

The row's own note [2]: "crowdsourced science exam MCQ with supporting text; `train`+`validation` are safe, `test` is the SciQ benchmark." The row carries no flag.
