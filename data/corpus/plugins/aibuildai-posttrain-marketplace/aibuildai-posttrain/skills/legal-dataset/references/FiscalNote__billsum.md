# FiscalNote/billsum

BillSum: 23,455 U.S. Congressional and California state bills with their official summaries - 18,949 train, 3,269 test, and a 1,237-bill California test set.

**FiscalNote/billsum** is BillSum, "summarization of US Congressional and California state bills" [1], introduced by Kornilova and Eidelman [2]. Each row is a bill's `text`, its human-written `summary`, and its `title` [3]. It lives at https://huggingface.co/datasets/FiscalNote/billsum .

**Use it for**: SFT for long-document legislative summarisation. The targets are summaries written by legislative staff, not by a model.

**Licence**: CC0 1.0 (`license: cc0-1.0`) [4]. The one catch: none.

**Shape**: 23,455 rows: `train` 18,949 / `test` 3,269 / `ca_test` 1,237 [5]; three columns [3].

**Hold out**: `test` and `ca_test`. Measured: 4 of 3,269 `test` bill texts appear in `train` [6]. `lighteval/legal_summarization` repackages the same `train` and `test` as its `BillSum` config [7].

**Origin**: U.S. Congressional and California bills with human-written summaries [1][2]. Hub API at the check date: `downloads` 13,469, `downloadsAllTime` 379,214, `likes` 55 [4].

**Trained-on-by**: the Hub's dataset tag lists summarisation models such as `Anurag33Gaikwad/legal-led-billsum-summarization` (58 downloads) and `luluw/t5-base-finetuned-billsum` (32) [8].

**Introduced by**: [2] (Kornilova and Eidelman).

## Shape

| split | rows |
| --- | --- |
| `train` | 18,949 |
| `test` | 3,269 |
| `ca_test` | 1,237 |
| total | 23,455 |

One config, `default` [3]:

| column | dtype |
| --- | --- |
| `text` | string |
| `summary` | string |
| `title` | string |

## Quality

- Measured duplication: 43 `train` texts (0.23%) repeat [9].
- The card says `ca_test` rows lack some features the U.S. rows have ("features for us bills. ca bills does not have.") [1]; all three splits share the same three columns in the served schema [3].

## Load it

Train on `train`, hold out both test splits:

```python
import datasets

REV = "3d8510441c06a3d9dfb32eb0d7f80151730bcc4f"  # main at the check date
train = datasets.load_dataset("FiscalNote/billsum", revision=REV, split="train")     # 18,949 rows
test = datasets.load_dataset("FiscalNote/billsum", revision=REV, split="test")       # 3,269 rows - hold out
ca_test = datasets.load_dataset("FiscalNote/billsum", revision=REV, split="ca_test") # 1,237 rows - hold out
```

**Trap**: bill text keeps the Government Publishing Office's plain-text layout - fixed-width indentation and doubled backquotes as quotation marks (``like this'') [10]. Normalise both, or the model learns to emit them.

## Neighbors

- `lighteval/legal_summarization` - `BillSum`, `EurLexSum` and `MultiLexSum` in one evaluation-harness format.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [10], truncated:

```json
{
  "text": "SECTION 1. LIABILITY OF BUSINESS ENTITIES PROVIDING USE OF FACILITIES \n              TO NONPROFIT ORGANIZATIONS.\n\n    (a) Definitions.--In this section:\n            (1) Business entity.--The term ``business entity'' means a \n        firm, corporation, association, partnership, consortium, joint \n    [...]",
  "summary": "Shields a business entity from civil liability relating to any injury or death occurring at a facility of that entity in connection with a use of such facility by a nonprofit organization if: (1) the use occurs outside the scope of business of the business entity; (2) such injury or death occurs dur [...]",
  "title": "A bill to limit the civil liability of business entities providing use of facilities to nonprofit organizations."
}
```

## Where it came from

Released by FiscalNote researchers Anastassia Kornilova and Vlad Eidelman; code at https://github.com/FiscalNote/BillSum [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] FiscalNote/billsum dataset card (README). https://huggingface.co/datasets/FiscalNote/billsum/raw/main/README.md. Fetched 2026-09-23.

[2] Kornilova and Eidelman, "BillSum: A Corpus for Automatic Summarization of US Legislation", arXiv:1910.00523, 2019. https://arxiv.org/abs/1910.00523 - current title read from the live abs page. Fetched 2026-09-23.

[3] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=FiscalNote%2Fbillsum - column schema; live, no revision parameter. Fetched 2026-09-23.

[4] Hugging Face Hub API record for FiscalNote/billsum. https://huggingface.co/api/datasets/FiscalNote/billsum?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=FiscalNote%2Fbillsum - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] This skill's exact split-leakage measurement, `references/contamination.md` (split leakage table). Fetched 2026-09-23.

[7] datasets-server size endpoint for lighteval/legal_summarization. https://datasets-server.huggingface.co/size?dataset=lighteval%2Flegal_summarization - `BillSum` `train` 18,949, `test` 3,269. Fetched 2026-09-23.

[8] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:FiscalNote/billsum&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[9] This skill's duplication measurement, `references/contamination.md` (duplication table). Run 2026-09-23.

[10] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=FiscalNote%2Fbillsum&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable: public-domain legislative summarisation data with human summaries. Train on `train`, hold out `test` and `ca_test`.

### The screening row

The row's own note: "BillSum; CC0; human summaries." The row carries no flag.
