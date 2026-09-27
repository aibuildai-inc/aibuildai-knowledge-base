# lighteval/legal_summarization

Three legal summarisation sets in one evaluation-harness format - BillSum, EUR-Lex-Sum (English) and Multi-LexSum - as `article` and `summary` columns, with no card text.

**lighteval/legal_summarization** is a repackaging used by the `lighteval` evaluation harness: three configs, `BillSum`, `EurLexSum` and `MultiLexSum`, each with `article` and `summary` columns [1]. Its README has front matter only [2]. It lives at https://huggingface.co/datasets/lighteval/legal_summarization .

**Use it for**: evaluation of legal summarisation in the harness format. Train on the upstream datasets instead, where the licence and provenance are documented.

**Licence**: none on the card [3][2]. Upstream: BillSum is CC0 1.0, EUR-Lex-Sum is CC BY 4.0 in metadata and CC BY-SA 4.0 in its body [4]. The one catch: no licence here; Multi-LexSum's upstream terms are not stated anywhere in this skill.

**Shape**: 26,860 rows: `BillSum` 18,949 / 3,269; `EurLexSum` 1,129 / 187 / 188; `MultiLexSum` 2,210 / 312 / 616 (`train` / `validation` / `test`) [5].

**Hold out**: every `test` split - they are the upstream test sets. `BillSum` counts match `FiscalNote/billsum` exactly [5][4].

**Origin**: as upstream: BillSum and EUR-Lex-Sum summaries are human-written [4]; this card has no source for Multi-LexSum. Hub API at the check date: `downloads` 339, `downloadsAllTime` 10,659, `likes` 28 [3].

**Trained-on-by**: an evaluation set; no model declares it [6].

**Introduced by**: no paper - a harness repackaging [2].

## Shape

Rows per config and split (datasets-server `/size`) [5]:

| config | split | rows |
| --- | --- | --- |
| `BillSum` | `train` | 18,949 |
| `BillSum` | `test` | 3,269 |
| `EurLexSum` | `train` | 1,129 |
| `EurLexSum` | `test` | 188 |
| `EurLexSum` | `validation` | 187 |
| `MultiLexSum` | `train` | 2,210 |
| `MultiLexSum` | `test` | 616 |
| `MultiLexSum` | `validation` | 312 |
| all 3 configs | `train` 22,288 / `test` 4,073 / `validation` 499 | 26,860 |

All configs share two columns [1]:

| column | dtype |
| --- | --- |
| `article` | string |
| `summary` | string |

## Quality

- `BillSum` text has had its line breaks collapsed relative to `FiscalNote/billsum`: the same bill appears as one run of text [7][8].
- No source states how Multi-LexSum was converted or which of its summary granularities `summary` holds.

## Load it

Evaluate on `test`:

```python
import datasets

REV = "e8b8f0aed467234d29138d540e6163573476276c"  # main at the check date
bs_test = datasets.load_dataset("lighteval/legal_summarization", "BillSum", revision=REV, split="test")  # 3,269 rows
```

**Trap**: scoring BillSum here and training on `FiscalNote/billsum` `train` is fine; training on this repository's `train` splits and scoring on another harness's copy of the same test set is not - they are the same documents.

## Neighbors

- `FiscalNote/billsum`, `dennlinger/eur-lex-sum` - the upstream releases.

## A row

From `config="MultiLexSum"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], truncated:

```json
{
  "article": "On September 15, 2005, the Equal Employment Opportunity Commission (EEOC) filed suit against House of Philadelphia, Inc., on behalf of an employee who was allegedly fired because she was pregnant. Seeking monetary and injunctive relief for the employee (including economic damage, compensation for em [...]",
  "summary": "Equal Employment Opportunity Commission brought a Title VII sex discrimination case against House of Philadelphia, Inc., on behalf of an employee who was allegedly fired because she was pregnant. The EEOC sought monetary and injunctive relief for the employee (including economic damage, compensation [...]"
}
```

## Where it came from

Packaged for the Hugging Face `lighteval` harness [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=lighteval%2Flegal_summarization - column schema; live, no revision parameter. Fetched 2026-09-23.

[2] lighteval/legal_summarization dataset card (README). https://huggingface.co/datasets/lighteval/legal_summarization/raw/main/README.md. Fetched 2026-09-23.

[3] Hugging Face Hub API record for lighteval/legal_summarization. https://huggingface.co/api/datasets/lighteval/legal_summarization?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] Upstream cards in this skill: `FiscalNote__billsum.md` and `dennlinger__eur-lex-sum.md` (licences and split sizes). Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=lighteval%2Flegal_summarization - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:lighteval/legal_summarization&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[7] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=lighteval%2Flegal_summarization&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[8] datasets-server first-rows for FiscalNote/billsum, config `default`, split `train`. https://datasets-server.huggingface.co/first-rows?dataset=FiscalNote%2Fbillsum&config=default&split=train. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

An evaluation repackaging: use its `test` splits for harness evaluation and train on the upstream releases.

### The screening row

The row's own note: "harness copy of BillSum / EUR-Lex-Sum / Multi-LexSum." The row carries the flag `no-licence`.
