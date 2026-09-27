# ChicagoHAI/CaseSumm

CaseSumm: 27,071 U.S. Supreme Court opinions from 1815 to 2019 paired with their official syllabuses - the long-context legal summarisation set with expert-written targets, under a licence its own card states three different ways.

**ChicagoHAI/CaseSumm** was introduced in "CaseSumm: A Large-Scale Dataset for Long-Context Summarization from U.S. Supreme Court Opinions" by Heddaya et al. [1]. Its card explains that each syllabus "is written by an attorney employed by the Court and approved by the Justices" and is "the gold standard for summarizing majority opinions"; opinions come from Public Resource Org's archive and syllabuses from the U.S. Reports hosted by the Library of Congress [2]. It lives at https://huggingface.co/datasets/ChicagoHAI/CaseSumm .

**Use it for**: SFT for long-context summarisation of judicial opinions, with Court-written targets. Opinions are long; plan context length before training.

**Licence**: three statements disagree: the metadata says `cc-by-nc-3.0` [3]; the summary says "Our dataset is available under a CC BY-NC 4.0 license"; the licence section says "All the data and code within this repo are under CC BY-NC-SA 4.0" [2]. The one catch: non-commercial under every reading; share-alike under one.

**Shape**: 27,071 rows in one `train` split [4]; three columns (`citation`, `syllabus`, `opinion`) [5].

**Hold out**: no split is set aside; carve one, ideally by date, since older opinions differ in style.

**Origin**: opinions by Supreme Court Justices; syllabuses by the Court's Reporter of Decisions [2]. Hub API at the check date: `downloads` 193, `downloadsAllTime` 6,231, `likes` 21 [3].

**Trained-on-by**: no model on the Hub declares this dataset through its dataset tag [6].

**Introduced by**: [1] (Heddaya et al.).

## Shape

| split | rows |
| --- | --- |
| `train` | 27,071 |
| total | 27,071 |

One config, `default` [5]:

| column | dtype |
| --- | --- |
| `citation` | string |
| `syllabus` | string |
| `opinion` | string |

## Quality

- Early syllabuses carry OCR errors: row 0 (108 U.S. 30) reads "St. Louis, Iron 3lountain and Southern R. R. Co." and "IExpress Company" [7].
- No source states a duplicate rate.

## Load it

One split; pin the revision and split by citation:

```python
import datasets

REV = "880acec67a69766075ddb42882de4722efdeddb2"  # main at the check date
cs = datasets.load_dataset("ChicagoHAI/CaseSumm", revision=REV, split="train")  # 27,071 rows
```

**Trap**: `citation` is written with dots instead of spaces, for example "108.US.30" [7]; joining it to other Supreme Court data needs normalising to "108 U.S. 30".

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], truncated:

```json
{
  "citation": "108.US.30",
  "syllabus": "1. On the merits of the motion there is no essential difference between this case and the case of the St. Louis, Iron 3lountain and Southern R. R. Co. v. The Southern IExpress Company, just decided. Reference to the master to take and state an account between the parties as to the compensation durin [...]",
  "opinion": "This motion to dismiss is made because, as is alleged, (1) the decree appealed from is not a final decree; and (2) the transcript is not properly certified. 1. As to the decree. The case is in some particulars different from that of the St. Louis, I. M. & S. Ry. Co. v. Southern Express Co., just dec [...]"
}
```

## Where it came from

Built by Mourad Heddaya, Kyle MacMillan, Anup Malani, Hongyuan Mei and Chenhao Tan at the University of Chicago [2][1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Heddaya et al., "CaseSumm: A Large-Scale Dataset for Long-Context Summarization from U.S. Supreme Court Opinions", arXiv:2501.00097, 2024. https://arxiv.org/abs/2501.00097 - current title read from the live abs page. Fetched 2026-09-23.

[2] ChicagoHAI/CaseSumm dataset card (README). https://huggingface.co/datasets/ChicagoHAI/CaseSumm/raw/main/README.md. Fetched 2026-09-23.

[3] Hugging Face Hub API record for ChicagoHAI/CaseSumm. https://huggingface.co/api/datasets/ChicagoHAI/CaseSumm?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=ChicagoHAI%2FCaseSumm - takes no revision parameter; a live figure. Fetched 2026-09-23.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=ChicagoHAI%2FCaseSumm - column schema; live, no revision parameter. Fetched 2026-09-23.

[6] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:ChicagoHAI/CaseSumm&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[7] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=ChicagoHAI%2FCaseSumm&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable for non-commercial long-context summarisation with Court-written targets. Resolve the licence (NC 3.0, NC 4.0, or NC-SA 4.0) before redistribution.

### The screening row

The row's own note: "SCOTUS opinions + official syllabuses; three conflicting licence statements." The row carries the flag `licence-conflict`.
