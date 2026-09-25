# chenghao/sec-material-contracts

1,141,632 material contracts (Exhibit 10) filed with the SEC from 1994 to the first quarter of 2025 - credit agreements, employment agreements, leases, licences, merger documents - as raw filing text, 39.7 GB of parquet. The largest open body of real U.S. commercial agreements on the Hub, and continued-pretraining text for Harvey LAB's contract work.

**chenghao/sec-material-contracts** collects "1,141,632 material contracts (Exhibit 10)" from "sec.gov's EDGAR database", "spanning from 1994 to 2025 Q1, sourced from 10-K, 10-Q, and 8-K filings" [1]. Each row is one exhibit with its filing metadata (company, CIK, form type, dates, exhibit description) and its full content [1][2]. It lives at https://huggingface.co/datasets/chenghao/sec-material-contracts .

**Plain text for continued pretraining, not SFT: there are no questions or labels. The content is raw HTML or SGML-wrapped text, and 3,372 rows are PDFs stored as base64; strip markup and drop PDFs before training.**

**Use it for**: continued pretraining on real U.S. commercial agreements - the defined terms, section structure, covenants, change-of-control and assignment clauses that LAB's synthetic data rooms imitate. Filter by `desc` or `doc_type` to weight credit, employment, merger or licence agreements toward the LAB areas you target.

**Licence**: CC BY-SA 4.0 in the card metadata and body [1][3]. The one catch: share-alike, so a dataset you derive from it and publish must carry the same licence. The underlying filings are public SEC records.

**Shape**: 1,141,632 rows, one config (`default`), one split (`train`), 17 columns; 39.7 GB of parquet, about 151.6 GB in memory [2][4]. By extension: `html` 854,986, `txt` 142,454, `txt_filing` 140,820, `pdf` 3,372 [1].

**Hold out**: nothing of its own - no split and no benchmark. LAB went public on 2026-05-06; this revision was last changed on 2025-08-14 [3], so it cannot contain LAB's text.

**Origin**: human-drafted agreements filed by companies with the SEC, collected by a Scrapy crawler (https://github.com/ChenghaoMou/edgar-crawler) [1]. Hub API at the check date: `downloads` 1,431, `downloadsAllTime` 6,747, `likes` 3 [3].

**Trained-on-by**: the Hub's model search by dataset tag returns no model [5].

**Introduced by**: [1] (dataset card; no paper).

## Shape

| config | split | rows | parquet |
| --- | --- | --- | --- |
| `default` | `train` | 1,141,632 | 39,664,203,631 bytes [4] |

Columns [2]:

| column | what it holds [1] |
| --- | --- |
| `index_html_url`, `index_text_url` | the filing's index pages |
| `cik`, `name` | EDGAR company key and company name |
| `type` | filing form: 10-K, 10-Q or 8-K |
| `filing_date`, `report_date` | dates |
| `seq`, `desc`, `doc_type` | exhibit sequence, description, type (for example `EX-10.1`) |
| `size`, `filename`, `file_url`, `file`, `extension` | the document's size, name, SEC URL, private storage URI, format |
| `filing_metadata` | a JSON string of filing facts |
| `file_content` | the document: HTML or text, or base64 for PDF |

Rows per filing year run from 6,795 (1994) through a peak of 54,450 (2005) to 21,314 (2025, first quarter only) [1].

## Quality

- `txt_filing` rows, "most filings before 2000", are cut from the whole text filing rather than the exhibit file [1], so they carry SGML wrappers (`<DOCUMENT>`, `<TYPE>`, `<PAGE>`) around the agreement, as the first served row shows [6].
- The card lists no deduplication. Measured on one shard of 304 (`train-00150-of-00304`, 3,749 rows, the 3,743 non-PDF rows after stripping markup) [7]: 1 exact and 1 cleaned duplicate (0.03%), 18 MinHash near-duplicates at Jaccard 0.8 on the first 2,000 tokens (0.48%). A shard holds a mix of years, so cross-shard repeats of the same form are not counted; this is a sample, not a whole-split figure. The shard's median exhibit is 27,213 characters after tags are stripped.
- Exhibit 10 covers every "material contract", so compensation summaries and promissory notes sit beside credit and merger agreements [6]; filter by `desc` if you want deal documents only.

## Load it

Stream it; do not materialise 151 GB:

```python
import datasets

REV = "bc6a0be3fa359e426733633439316f66a76dbf41"  # main at the check date
ds = datasets.load_dataset("chenghao/sec-material-contracts", revision=REV, split="train", streaming=True)
ds = ds.filter(lambda r: r["extension"] != "pdf")            # 3,372 base64 PDFs
```

**Trap**: `file_content` is markup, not clean text: HTML for 854,986 rows and SGML-wrapped filing text for 140,820 [1]. Train on it unstripped and the model learns tags and page headers; strip HTML and the `<DOCUMENT>...<TEXT>` wrapper first.

## Neighbors

- `theatticusproject/cuad` - 510 EDGAR contracts with expert clause labels; a labelled slice of the same source.
- `pile-of-law/pile-of-law` configs `edgar` and `atticus_contracts` - EDGAR contract text inside a larger legal corpus.
- `TheTokenFactory/sec-contracts-financial-extraction-instructions` - extraction SFT built from 1,028 Exhibit 10 contracts.

## A row

From `split="train"`, `row_idx=0` (datasets-server `/first-rows`), `file_content` truncated [6]:

```json
{
  "cik": "889331",
  "name": "LITTELFUSE INC /DE",
  "type": "10-K",
  "filing_date": "2007-02-27",
  "desc": "SUMMARY OF EXECUTIVE OFFICER COMPENSATION",
  "doc_type": "EX-10.19",
  "extension": "txt",
  "file_url": "https://www.sec.gov/Archives/edgar/data/889331/000095013707002879/c12654exv10w19.txt",
  "file_content": "<DOCUMENT>\n<TYPE>EX-10.19\n<SEQUENCE>7\n<FILENAME>c12654exv10w19.txt\n<DESCRIPTION>SUMMARY OF EXECUTIVE OFFICER COMPENSATION\n<TEXT>\n<PAGE>\n ..."
}
```

## Where it came from

Collected and published by Chenghao Mou; the crawler is public [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-24; that date covers every number, quote, and row above unless a line says it was measured. Hub repositories are mutable, which is why Load it pins the revision.

[1] chenghao/sec-material-contracts dataset card (README). https://huggingface.co/datasets/chenghao/sec-material-contracts/raw/bc6a0be3fa359e426733633439316f66a76dbf41/README.md - summary, field table, year and extension counts, licence. Fetched 2026-09-24.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=chenghao%2Fsec-material-contracts - column schema and split size; live. Fetched 2026-09-24.

[3] Hugging Face Hub API record. https://huggingface.co/api/datasets/chenghao/sec-material-contracts?full=true - licence field, gate, `sha`, `lastModified` 2025-08-14; `downloads`, `downloadsAllTime`, `likes` through the `expand[]` variant. Fetched 2026-09-24.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=chenghao%2Fsec-material-contracts - `num_rows` 1,141,632, `partial` false. Fetched 2026-09-24.

[5] Hub model search by dataset tag. https://huggingface.co/api/models?filter=dataset:chenghao/sec-material-contracts&sort=downloads - empty. Fetched 2026-09-24.

[6] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=chenghao%2Fsec-material-contracts&config=default&split=train - 10 rows served (truncated). Fetched 2026-09-24.

[7] This skill's own measurement on `data/train-00150-of-00304.parquet` at the pinned revision: exact, cleaned (Unicode-aware) and MinHash duplication, extensions and lengths. Run 2026-09-24.

## Appendix: screening record

### Screening verdict

Passed as continued-pretraining text for LAB's transactional areas. Stages 0-2 answered from the Hub records above: readable, not gated, `partial` false, licence field and body agree (CC BY-SA 4.0). Stage 3 (duplication) was measured on one shard only: 0.03% exact, 0.48% MinHash near-duplicates, not a whole-split figure [7]. Stage 5 (overlap with carded datasets) was not run: the split is 39.7 GB of parquet. Stage 4 (contamination): not needed against LAB, whose release post-dates this revision; not measured against LegalBench's CUAD and MAUD items, which come from EDGAR contracts and may overlap - check before reporting LegalBench.

### The screening row

The row's own note: "EDGAR Exhibit 10 contracts, 1994 to 2025 Q1; raw HTML and SGML; not deduplicated; added for the LAB target on 2026-09-24." The row carries the flag `not-deduplicated`.
