# common-pile/caselaw_access_project

About 6.9 million U.S. federal and state court opinions from the Caselaw Access Project and CourtListener, as plain text, filtered to public-domain documents - the largest openly licensed body of American case law on the Hub.

**common-pile/caselaw_access_project** is the case-law slice of the Common Pile v0.1, an 8TB corpus built only from public-domain and openly licensed text [1]. Its card describes "6.7 million cases from the Caselaw Access Project and Court Listener" and, in its own statistics table, 6,919,240 documents and 78 UTF-8 GB [2]. The card says only public-domain documents were included, OCR errors were corrected after digitization, and the processing code sits in the common-pile GitHub repository [2]. Every one of the 22 rows the viewer returned carries `metadata.license` "Public Domain" [3]. It lives at https://huggingface.co/datasets/common-pile/caselaw_access_project .

**Use it for**: continued pretraining on U.S. judicial opinions - plain `text`, one opinion per row, no prompt or answer. It teaches the register, citation style, and doctrine of American courts; it does not teach a model to answer questions, so it belongs before an SFT stage, not in place of one. For legal reasoning benchmarks it is the cleanest licence story of any large U.S. corpus here.

**Licence**: no licence field in the card metadata [4]; the documents are tagged "Public Domain" row by row [3], and the card's own License Issues section warns that "license laundering and inaccurate metadata can cause us to erroneously assign the incorrect license to some documents" [2]. Ungated [4]. The one catch: no repository-level licence is declared, so the per-row tag is the whole licence claim.

**Shape**: about 6.9 million documents declared on the card [2]; the viewer indexes only a `partial` 351,619 rows of an estimated 5,520,526 in one `train` split [5]. Six columns (`id`, `source`, `added`, `created`, `metadata`, `text`) [6].

**Hold out**: nothing inside the repository - there is one `train` split. Opinions are the raw material of several benchmarks, so screen before a scored run: CaseHOLD and LexGLUE `case_hold` prompts are excerpts of U.S. opinions [7], and the Nemotron legal set names the filtered version of this corpus as the source of three of its synthetic subsets.

**Origin**: court opinions written by judges, digitized by the Harvard Law School Library's Caselaw Access Project and supplemented from CourtListener [2]. Hub API at the check date: `downloads` 4,944, `downloadsAllTime` 89,919, `likes` 220 [4].

**Trained-on-by**: the Common Pile paper trains Comma v0.1 on the filtered version of this source [1][2]. The Hub's dataset tag lists only small community models, led by `yasserrmd/caselaw-cpt-8b` (28 downloads) [8].

**Introduced by**: [1] (Kandpal et al.); the underlying opinions come from the Caselaw Access Project and CourtListener [2].

## Shape

The viewer reads a partial copy, so these are the rows it indexed, not the dataset (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 351,619 |
| total | 351,619 |

One config, `default` [6]:

| column | dtype |
| --- | --- |
| `id` | string |
| `source` | string |
| `added` | string |
| `created` | string |
| `metadata` | struct<author: string, license: string, url: string> |
| `text` | string |

The repository holds numbered `cap_000NN.jsonl.gz` shards [9]; the viewer's `num_rows` stops at 351,619 because it read only part of them, and its own `estimated_num_rows` is 5,520,526 [5], still short of the 6,919,240 documents the card's statistics table states [2]. No number from the viewer is a whole-dataset figure.

## Quality

- The card states OCR errors were corrected after digitization and that additional post-processing fixed formatting and parsing [2]. The first row still shows OCR residue: "Claude 0. Allen" and "reeross-examination" in a 1973 Ninth Circuit opinion [3].
- All 22 viewer rows are from `source` "Caselaw Access Project"; none of the sampled rows came from the CourtListener portion [3].
- `partial` is true, so no duplicate rate, length distribution, or contamination rate can be read as a whole-split figure from the viewer [5].
- No source states a measured duplicate rate. CourtListener and CAP overlap in coverage, so exact-text deduplication across the two sources is worth running before training.

## Load it

Train on `train`; stream it, because the full corpus is tens of gigabytes. Pin the revision the card was read at:

```python
import datasets

REV = "3c2cb5080b3a16a04d8d8d07b28eaec7c1ba7a90"  # main at the check date
cap = datasets.load_dataset("common-pile/caselaw_access_project", revision=REV, split="train", streaming=True)
for doc in cap.take(3):
    print(doc["metadata"]["license"], doc["text"][:200])
```

**Trap**: the filtered sibling's README points readers to `common-pile/caselaw_access_project_raw` for "the raw version" [10], but the raw release is this repository, `common-pile/caselaw_access_project`, whose own card calls itself "the 'raw' version" [2]. A script that follows the filtered card's link fails. The filtered sibling's `pretty_name` also reads "Biodiversity Heritage Library", a copy-paste from another Common Pile card [11].

## Neighbors

- `common-pile/caselaw_access_project_filtered` - the version used to train Comma v0.1; its card states 6,735,525 documents and 77 GB, and its rows add a `metadata.provenance` field (`CAP-Dolma-0000.json.gz:1` on row 0) [10][12]. Its first row is the same *United States v. Veon* opinion as row 0 here [12][3]. Prefer it when you want the exact training mix of a published model.
- `HFforLegal/case-law` - a separate community collection of U.S. state-court decisions with titles, dockets, and issuing court as columns; see its own card.
- `pile-of-law/pile-of-law` - carries CourtListener opinions as one of its 45 source configs, under CC BY-NC-SA 4.0 rather than public domain.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3], with `text` truncated:

```json
{
  "id": "f2d_474/html/0001-01.html",
  "source": "Caselaw Access Project",
  "added": "2024-08-24T03:29:51.129235",
  "created": "2024-08-24T03:29:51.129683",
  "metadata": {
    "author": "PER CURIAM:",
    "license": "Public Domain",
    "url": "https://static.case.law/"
  },
  "text": "\n    UNITED STATES of America, Appellee, v. Daniel Dee VEON, Appellant.\n    No. 72-1889.\n    United States Court of Appeals, Ninth Circuit.\n    Feb. 12, 1973.\n    \n      Claude 0. Allen, Oakland, Cal., for appellant.\n    James L. Browning, Jr., U. S. Atty., F. Steele Langford and Jerry K. Cimmet, Asst. U. S. Attys., San Francisco, Cal., for appellee.\n    Before TRASK, GOODWIN and WALLACE, Circuit  [...]"
}
```

## Where it came from

The Common Pile team collected public-domain opinions from the Caselaw Access Project (nearly 40 million pages of U.S. federal and state decisions spanning 365 years, per the card) and added over 900 thousand cases scraped from 479 courts by CourtListener [2]. They kept documents in the public domain, corrected OCR errors, and published the code in the common-pile GitHub repository [2][1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Kandpal et al., "The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text", arXiv:2506.05209, 2025. https://arxiv.org/abs/2506.05209 - current title read from the live abs page. Fetched 2026-09-23.

[2] common-pile/caselaw_access_project dataset card (README). https://huggingface.co/datasets/common-pile/caselaw_access_project/raw/main/README.md. Fetched 2026-09-23.

[3] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=common-pile%2Fcaselaw_access_project&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[4] Hugging Face Hub API record for common-pile/caselaw_access_project. https://huggingface.co/api/datasets/common-pile/caselaw_access_project?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=common-pile%2Fcaselaw_access_project - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=common-pile%2Fcaselaw_access_project - column schema; live, no revision parameter. Fetched 2026-09-23.

[7] Zheng et al., "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset", arXiv:2104.08671, 2021. https://arxiv.org/abs/2104.08671 - current title read from the live abs page. Fetched 2026-09-23.

[8] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:common-pile/caselaw_access_project&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[9] Repository file tree for common-pile/caselaw_access_project. https://huggingface.co/api/datasets/common-pile/caselaw_access_project/tree/main?recursive=true. Fetched 2026-09-23.

[10] common-pile/caselaw_access_project_filtered dataset card (README). https://huggingface.co/datasets/common-pile/caselaw_access_project_filtered/raw/main/README.md - document count, the Other Versions link, the Comma v0.1 statement. Fetched 2026-09-23.

[11] Hugging Face Hub API record for common-pile/caselaw_access_project_filtered. https://huggingface.co/api/datasets/common-pile/caselaw_access_project_filtered?full=true - `cardData.pretty_name`. Fetched 2026-09-23.

[12] datasets-server first-rows endpoint for common-pile/caselaw_access_project_filtered. https://datasets-server.huggingface.co/first-rows?dataset=common-pile%2Fcaselaw_access_project_filtered&config=default&split=train. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as continued-pretraining text for U.S. case law, with the strongest licence position of the large U.S. corpora here: every sampled row is tagged public domain, though the card itself warns the tags can be wrong. Not auditable as a whole through the viewer, because `partial` is true.

### The screening row

The row's own note: "public-domain CAP + CourtListener opinions; the licence claim is per-row, not per-repo, and the viewer copy is partial." The row carries no flag.
