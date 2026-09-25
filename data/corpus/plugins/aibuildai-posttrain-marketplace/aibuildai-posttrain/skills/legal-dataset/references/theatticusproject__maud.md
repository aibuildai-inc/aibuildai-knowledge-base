# theatticusproject/maud

MAUD: 39,231 expert-annotated multiple-choice questions about the deal points of public-company merger agreements (153 distinct contracts in `train`) - and the source of LegalBench's 34 `maud_*` tasks, whose test items are nearly all inside MAUD's own `train` split.

**theatticusproject/maud** is the Merger Agreement Understanding Dataset, introduced by Wang et al. as "An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding" [1]. Each row is a passage from a merger agreement, a deal-point question and subquestion (for example about the definition of a material adverse effect or a no-shop clause) and the annotated answer with its numeric `label` [2][3]. The repository's README is 27 bytes; the documentation is a PDF, `MAUD v1 README (1).pdf` [4][5]. It lives at https://huggingface.co/datasets/theatticusproject/maud .

**Nearly all of LegalBench's `maud_*` test items are inside MAUD's `train` split, and every one of MAUD's 6,651 `test` rows comes from a contract that also appears in `train`.**

**Use it for**: SFT on reading dense merger-agreement language and answering structured deal-point questions. If you report LegalBench's MAUD tasks, you cannot train on this dataset at all.

**Licence**: CC BY 4.0 in the card metadata [6]. The one catch: the contracts are EDGAR filings, and the licence covers the annotations.

**Shape**: 39,231 rows: `train` 25,827 / `validation` 6,753 / `test` 6,651 [7]; ten columns [2].

**Hold out**: `test` (6,651 rows) - but its contracts are not unseen: all 6,651 `test` rows name a `contract_name` present in `train`, which has 153 distinct contracts [8]. MAUD splits by question, not by contract, so a MAUD test score measures questions on familiar contracts. LegalBench's `maud_*` tasks are built from these questions and nearly all of them sit in `train` [9]; if you report them, do not train on MAUD.

**Origin**: merger agreements, with answers based on the American Bar Association's 2021 Public Target Deal Points Study, per the paper [1]. Hub API at the check date: `downloads` 769, `downloadsAllTime` 6,858, `likes` 4 [6].

**Trained-on-by**: the Hub's dataset tag lists `jbarney/circuit-1.7b` (0 downloads) [10]. LawInstruct converts MAUD `train` into four instruction files [11].

**Introduced by**: [1] (Wang et al.).

## Shape

| split | rows |
| --- | --- |
| `train` | 25,827 |
| `validation` | 6,753 |
| `test` | 6,651 |
| total | 39,231 |

One config, `default` [2]:

| column | dtype |
| --- | --- |
| `data_type` | string |
| `contract_name` | string |
| `text` | string |
| `answer` | string |
| `label` | int64 |
| `question` | string |
| `subquestion` | string |
| `text_type` | string |
| `id` | string |
| `category` | string |

The repository also ships the source files: `MAUD_v1/MAUD_train.csv`, `MAUD_dev.csv`, `MAUD_test.csv` and the full contract texts under `MAUD_v1/contracts/` [5].

## Quality

- Measured duplication inside `train`: 4,136 rows (16.01%) repeat another row's `text`, `question` and `subquestion` [12] - the same passage and question can carry more than one row.
- `data_type` takes three values in the sampled rows - `main`, `abridged` and `rare_answers` (`train`: 95 `main`, 5 `rare_answers`; `test`: 50 `main`, 32 `abridged`) [3]. Read the PDF README before mixing types.

## Load it

Train on `train`, hold out `test` - and exclude MAUD entirely if you will report LegalBench MAUD tasks:

```python
import datasets

REV = "37d5c3b95d18dcd8404cc5ce3fd5069be062392f"  # main at the check date
train = datasets.load_dataset("theatticusproject/maud", revision=REV, split="train")   # 25,827 rows
test = datasets.load_dataset("theatticusproject/maud", revision=REV, split="test")     # 6,651 rows - hold out
```

**Trap**: a held-out MAUD score is not a held-out-contract score: every `test` row's contract is in `train` [8]. To measure generalisation to new agreements, re-split by `contract_name` yourself.

## Neighbors

- `nguha/legalbench` - its 34 `maud_*` tasks are built from MAUD.
- `lawinstruct/lawinstruct` - its `MAUD-*` files are MAUD `train` in instruction form.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3], truncated:

```json
{
  "data_type": "main",
  "contract_name": "contract_41",
  "text": "(i)            Conversion of Company Common Stock. Each Share (including each Restricted Share) issued and outstanding immediately prior to the Effective Time, other than Excluded Shares, shall be cancelled and extinguished and automatically converted into the right to receive $70 in cash, without i [...]",
  "answer": "All Cash",
  "label": 0,
  "question": "Type of Consideration-Answer",
  "subquestion": "<NONE>",
  "text_type": "Type of Consideration",
  "id": "3",
  "category": "General Information"
}
```

## Where it came from

Created by The Atticus Project with Steven Wang, Dan Hendrycks and co-authors, building on the American Bar Association's 2021 Public Target Deal Points Study [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Wang et al., "MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding", arXiv:2301.00876, 2023. https://arxiv.org/abs/2301.00876 - current title read from the live abs page. Fetched 2026-09-23.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=theatticusproject%2Fmaud - column schema; live, no revision parameter. Fetched 2026-09-23.

[3] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=theatticusproject%2Fmaud&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[4] theatticusproject/maud dataset card (README). https://huggingface.co/datasets/theatticusproject/maud/raw/main/README.md. Fetched 2026-09-23.

[5] Repository file tree for theatticusproject/maud. https://huggingface.co/api/datasets/theatticusproject/maud/tree/main?recursive=true. Fetched 2026-09-23.

[6] Hugging Face Hub API record for theatticusproject/maud. https://huggingface.co/api/datasets/theatticusproject/maud?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=theatticusproject%2Fmaud - takes no revision parameter; a live figure. Fetched 2026-09-23.

[8] This skill's own exact split-leakage count at the pinned revision - `contract_name`, and `text`+`question`+`subquestion`. Fetched 2026-09-23.

[9] This skill's own check by n-gram containment against the benchmark's test items (word 8-grams, or character 15-grams for Chinese), with self, words-sorted and positive controls, at the pinned revision. Run 2026-09-23.

[10] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:theatticusproject/maud&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[11] lawinstruct/lawinstruct README file list - `MAUD-answer`, `MAUD-category`, `MAUD-question`, `MAUD-text_type` train files. https://huggingface.co/datasets/lawinstruct/lawinstruct/raw/main/README.md. Fetched 2026-09-23.

[12] This skill's own duplication count at the pinned revision. Run 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as merger-agreement SFT data, but it is LegalBench's MAUD source: training on it voids LegalBench MAUD scores. Its own test split shares every contract with train.

### The screening row

The row's own note: "MAUD; test contracts all in train; train holds LegalBench maud_* items." The row carries the flag `overlaps-LegalBench`.
