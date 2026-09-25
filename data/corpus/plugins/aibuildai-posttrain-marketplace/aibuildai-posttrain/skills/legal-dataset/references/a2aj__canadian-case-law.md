# a2aj/canadian-case-law

About 226,000 Canadian court and tribunal decisions, English and French side by side in one row where both exist, with a citation network - updated continuously, and licensed per source rather than by the MIT tag on the repository.

**a2aj/canadian-case-law** is maintained by Access to Algorithmic Justice (A2AJ), a research project co-hosted by York University's Osgoode Hall Law School and Toronto Metropolitan University's Lincoln Alexander School of Law [1]. It "provides bulk, open-access full-text decisions from Canadian courts and tribunals", one case per row with English and French text where both are published, and adds `cases_cited`, `cases_citing` and `citing_cases_count` fields [1]. It succeeds an earlier release by the Refugee Law Lab, `refugee-law-lab/canadian-legal-data` [1]. It lives at https://huggingface.co/datasets/a2aj/canadian-case-law .

**The repository is tagged MIT, but every row's `upstream_license` field points to the source court's own terms, which the card says "may include limits on commercial use".**

**Use it for**: continued pretraining on Canadian law in both official languages, or building citation-aware retrieval. The per-court directories (`SCC`, `ONCA`, `FC`, ...) let you load one court at a time [2]. Not SFT data.

**Licence**: `mit` in the card metadata [3]; the card body says the dataset carries an `upstream_license` column "which reflects the licenses through which the A2AJ obtained the document. These upstream licenses may include limits on commercial use, as well as other limitations" [1]. All 10 sampled rows read "See upstream license, including non-commercial use and other restrictions: https://perma.cc/EA5C-R5DK" [4]. The one catch: treat the MIT tag as covering A2AJ's own work, not the decisions.

**Shape**: 226,019 rows in one `train` split, fully indexed (`partial` false) [5]; 21 columns [6]. The card's table gives per-court counts, from 29 (Public Service Disclosure Protection Tribunal) to 52,361 (Supreme Court of British Columbia), and warns "Counts are approximate and will drift as the dataset is updated" [1].

**Hold out**: nothing inside the repository - one `train` split. No external benchmark in this skill is drawn from Canadian decisions, apart from LegalBench's `canada_tax_court_outcomes` task, which uses Tax Court of Canada judgments - a court this corpus covers (`TCC`, 8,124 rows per the card) [1].

**Origin**: decisions written by Canadian courts and tribunals, scraped from their public websites [1]. Hub API at the check date: `downloads` 2,291, `downloadsAllTime` 51,458, `likes` 15; last modified 2026-09-20 [3].

**Trained-on-by**: no model on the Hub declares this dataset through its dataset tag [7].

**Introduced by**: no paper - the dataset card [1].

## Shape

| split | rows |
| --- | --- |
| `train` | 226,019 |
| total | 226,019 |

One config, `default` [6]:

| column | dtype |
| --- | --- |
| `dataset` | string |
| `citation_en` | string |
| `citation2_en` | string |
| `name_en` | string |
| `document_date_en` | timestamp[us, tz=UTC] |
| `url_en` | string |
| `scraped_timestamp_en` | timestamp[us, tz=UTC] |
| `unofficial_text_en` | string |
| `cases_cited_en` | list<string> |
| `cases_citing_en` | list<string> |
| `citation_fr` | string |
| `citation2_fr` | string |
| `name_fr` | string |
| `document_date_fr` | timestamp[us, tz=UTC] |
| `url_fr` | string |
| `scraped_timestamp_fr` | timestamp[us, tz=UTC] |
| `unofficial_text_fr` | string |
| `cases_cited_fr` | list<string> |
| `cases_citing_fr` | list<string> |
| `citing_cases_count` | int64 |
| `upstream_license` | string |

The repository is organised as one directory per court or tribunal, 29 in all (`BCCA` through `YKCA`) [2]. The card's table dates the earliest Supreme Court of Canada decision to 1877-01-15 and the latest to 2026-09-18, three days before the check date [1].

## Quality

- Text fields are named `unofficial_text_en` and `unofficial_text_fr`: the card calls each row "an unofficial reproduction" and warns the data "may contain errors. Always verify data against the official source" [1][4].
- None of the 10 sampled rows (all `BCCA`) has French text [4]; bilingual coverage depends on the court, and British Columbia publishes in English only.
- The citation network matches neutral citations only, so "decisions pre-dating neutral citations (generally pre-2000) are under-represented" [1].
- The dataset is updated continuously - the card's own "Last updated" line reads 2026-09-20 [1] - so row counts move between revisions.

## Load it

Train on `train`. Load one court with `data_dir`, as the card shows, and pin the revision - this repository changes weekly:

```python
import datasets

REV = "c2bf798c3e0b20f7a44f4998f8003f5b5b8e2240"  # main at the check date
scc = datasets.load_dataset("a2aj/canadian-case-law", data_dir="SCC", revision=REV, split="train")
all_courts = datasets.load_dataset("a2aj/canadian-case-law", revision=REV, split="train", streaming=True)
```

**Trap**: English and French live in separate columns of the same row, not in separate rows [6]. A pipeline that reads only a `text` column finds none, and one that concatenates `unofficial_text_en` and `unofficial_text_fr` trains on the same decision twice in two languages. Pick one column per row, or both deliberately.

## Neighbors

- `refugee-law-lab/canadian-legal-data` - the predecessor release by the Refugee Law Lab, which the card says this dataset builds on [1].
- `a2aj/canadian-laws` - Canadian statutes and regulations from the same maintainer, found by the Hub search [8]; not screened here.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [4], with long fields truncated and the empty French fields dropped:

```json
{
  "dataset": "BCCA",
  "citation_en": "2007 BCCA 309",
  "name_en": "R. v. Hill",
  "document_date_en": "2007-06-05T00:00:00",
  "url_en": "https://www.bccourts.ca/jdb-txt/ca/07/03/2007bcca0309.htm",
  "unofficial_text_en": "\n2007 BCCA 309 R. v. Hill\nCOURT OF APPEAL FOR BRITISH COLUMBIA\nCitation:\nR. v. Hill,\n2007 BCCA 309\nDate: 20070605\nDocket: CA033089\nBetween:\nRegina\nRespondent\nAnd\nWarren Hill\nAppellant\nBefore:\nThe Honourable Madam Justice Ryan\nThe Honourable Mr. Justice Smith\nThe Honourable Mr. Justice Chiasson\nJ.B.  [...]",
  "cases_cited_en": [
    "2005 BCSC 975",
    "2006 BCCA 530",
    "2001 BCCA 503",
    "2000 BCCA 480",
    "2003 SCC 30",
    "2001 SCC 32",
    "..."
  ],
  "citing_cases_count": 26,
  "upstream_license": "See upstream license, including non-commercial use and other restrictions: https://perma.cc/EA5C-R5DK. Note: This is an unofficial reproduction of a British Columbia Court of Appeal decision, without endorsement or affiliation by the British Columbia courts."
}
```

## Where it came from

Collected by A2AJ from Canadian court and tribunal websites and maintained as a living dataset; the curators listed include Sean Rehaag, co-director of A2AJ [1]. It extends work begun by the Refugee Law Lab [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] a2aj/canadian-case-law dataset card (README). https://huggingface.co/datasets/a2aj/canadian-case-law/raw/main/README.md. Fetched 2026-09-23.

[2] Repository file tree for a2aj/canadian-case-law. https://huggingface.co/api/datasets/a2aj/canadian-case-law/tree/main?recursive=true. Fetched 2026-09-23.

[3] Hugging Face Hub API record for a2aj/canadian-case-law. https://huggingface.co/api/datasets/a2aj/canadian-case-law?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=a2aj%2Fcanadian-case-law&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=a2aj%2Fcanadian-case-law - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=a2aj%2Fcanadian-case-law - column schema; live, no revision parameter. Fetched 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:a2aj/canadian-case-law&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=law&sort=downloads - lists `a2aj/canadian-laws`. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as continued-pretraining text for Canadian law in English and French. The MIT tag does not describe the decisions: the upstream terms in each row include non-commercial restrictions, so commercial use needs a per-court review.

### The screening row

The row's own note: "Canadian decisions EN/FR with citation network; MIT tag vs per-row upstream non-commercial terms; live-updated." The row carries the flag `licence-per-source`.
