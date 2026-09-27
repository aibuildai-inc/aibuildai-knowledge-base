# HFforLegal/case-law

541,371 U.S. state and federal court decisions with title, docket number, state and issuing court as columns, in a single split named `us` - a community collection meant to grow country by country, of which only the U.S. part exists.

**HFforLegal/case-law** is a community dataset from the HFforLegal organisation, credited in its citation entry to Louis Brulé Naudet and Timothy Dolan [1]. Its card describes "a comprehensive collection of legal decisons from various countries, centralized in a common format", organised as "country-based splits" named by ISO 3166-1 alpha-2 code [1]. At the check date there is exactly one split, `us`, which the card's embedded `dataset_info` sizes at 541,371 decisions and 9,138,869,838 bytes [2]. It lives at https://huggingface.co/datasets/HFforLegal/case-law .

**Use it for**: continued pretraining on U.S. court decisions when you want per-document metadata - `state`, `issuer` (the court), `title`, `docket_number` - to filter or balance by jurisdiction. The `document` column is the full decision text. Not SFT data.

**Licence**: CC BY 4.0 (`license: cc-by-4.0` in the card and the Hub tag) [2]. Ungated [2]. The one catch: the card names no upstream source for the decisions, so the CC BY 4.0 grant rests on the uploaders' claim alone [1].

**Shape**: 541,371 rows declared in the card's `dataset_info` for split `us` [2]; the viewer indexes a `partial` 282,390 rows of an estimated 534,289 [3]. Nine columns [4].

**Hold out**: nothing inside the repository. U.S. opinions are the raw material of CaseHOLD and of the Lawma/CaselawQA tasks, so screen before reporting those benchmarks.

**Origin**: decisions written by U.S. courts; the card does not say where they were collected from [1]. Hub API at the check date: `downloads` 2,186, `downloadsAllTime` 25,586, `likes` 32 [2].

**Trained-on-by**: the Hub's dataset tag lists small community models, among them `mradermacher/Mateno-v1.5-2b-it-GGUF` (417 downloads) and a cluster of 125M-parameter "slm" legal base models (for example `asinha08/slm-125m-base`, 151) [5].

**Introduced by**: no paper - the dataset card, with a BibTeX entry crediting Louis Brulé Naudet and Timothy Dolan [1].

## Shape

Rows the viewer indexed, not the dataset (datasets-server `/size`) [3]:

| split | rows |
| --- | --- |
| `us` | 282,390 |
| total | 282,390 |

One config, `default` [4]:

| column | dtype |
| --- | --- |
| `id` | string |
| `title` | string |
| `citation` | string |
| `docket_number` | string |
| `state` | string |
| `issuer` | string |
| `document` | string |
| `hash` | string |
| `timestamp` | string |

The card's own `dataset_info` declares the `us` split as 541,371 examples, 9,138,869,838 bytes, with a 4,597,435,136-byte download [2]. The viewer's copy is partial, so use the declared count.

## Quality

- The `hash` column is a SHA-256 of `document`, and the card ships the snippet that computes it [1]. All 11 sampled rows' hashes match `sha256(document)` exactly [6] - use the column for exact deduplication.
- `citation` is null in the first five sampled rows [6]; do not rely on it as a key.
- The first sampled row shows OCR noise: "oRDER", "fron only", "dismissed," for "dismissed." [6].
- The card's ethics section asks users to "Ensure that all personal information has been properly anonymized" [1], which states a responsibility rather than a step taken.
- `partial` is true, so no whole-split duplicate or length figure can be read from the viewer [3].

## Load it

The only split is `us`. Stream it and pin the revision:

```python
import datasets

REV = "dc0151fad300088cc7c21773bd3d680e0a514311"  # main at the check date
us = datasets.load_dataset("HFforLegal/case-law", revision=REV, split="us", streaming=True)
```

**Trap**: the card's own usage snippet loads the dataset and then reads `dataset['fr']` to "Access the French legal decisions" [1], but no `fr` split exists - the `configs` block maps a single split, `us`, to `data/us-*` [2]. The snippet raises a `KeyError`. Use `split="us"`.

## Neighbors

- `common-pile/caselaw_access_project` - a larger U.S. case-law corpus with a per-row public-domain tag and a stated source.
- `louisbrulenaudet/legalkit` - French law, from the same author; see its card.

## A row

From `config="default"`, `split="us"`, `row_idx=0` (datasets-server `/first-rows`) [6], with `document` truncated:

```json
{
  "id": "8637cc97-865a-4bd5-bb40-93eb099706dc",
  "title": "Ex parte Don Davis",
  "citation": null,
  "docket_number": "1140456",
  "state": "alabama",
  "issuer": "Alabama Supreme Court",
  "document": "STATE OF ALABAMA -- JUDICIAL DEPARTMENT\nTHE SUPREME COURT\nOCTOBER TERM, 2014-2015\n1140456\n\nEx parte Don Davis,\nJudge of Mobile County Probate Court\n\noRDER\n\nThe petition filed in this Court by the Mobile County\nProbate Judge on February 9, 2015, in substance is a request\nfor an advisory opinion. Sect [...]",
  "hash": "b513f47fa3d1518ad1b3372e111f9170de149bc934637678449f598a1553bf52",
  "timestamp": "2015-02-11T00:00:00Z"
}
```

## Where it came from

Uploaded by the HFforLegal community organisation; the citation entry credits Louis Brulé Naudet and Timothy Dolan [1]. The card describes the goal of centralising decisions from many countries under ISO-coded splits and invites contributors to submit new country splits [1]; none beyond `us` had been added at the check date [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] HFforLegal/case-law dataset card (README). https://huggingface.co/datasets/HFforLegal/case-law/raw/main/README.md. Fetched 2026-09-23.

[2] Hugging Face Hub API record for HFforLegal/case-law. https://huggingface.co/api/datasets/HFforLegal/case-law?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=HFforLegal%2Fcase-law - takes no revision parameter; a live figure. Fetched 2026-09-23.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=HFforLegal%2Fcase-law - column schema; live, no revision parameter. Fetched 2026-09-23.

[5] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:HFforLegal/case-law&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[6] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=HFforLegal%2Fcase-law&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as continued-pretraining text for U.S. case law with useful per-court metadata. Its provenance is unstated, which weakens the CC BY 4.0 claim; prefer the Common Pile case-law corpus when licence certainty matters. Not auditable as a whole through the viewer, because `partial` is true.

### The screening row

The row's own note: "US state/federal decisions with court metadata; single `us` split; source of the documents is not stated." The row carries no flag.
