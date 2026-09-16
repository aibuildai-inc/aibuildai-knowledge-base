# mjalg/function-code

233 single-turn instruction/output pairs teaching a model to write TypeScript/JavaScript functions, in one `train` split, with no dataset card beyond a license tag.

**mjalg/function-code** is a small Hugging Face dataset uploaded by Hub user `mjalg`, pairing a natural-language description of a function to write with a TypeScript/JavaScript function body that implements it, as seen in the served rows [1][2]. There is no paper and no descriptive card: the repository's `README.md` contains only an Apache-2.0 license front-matter block and no body text [3]. The sole data file, per the Hub API's siblings list, is named `results (2).json`, a name consistent with an ad hoc export rather than a documented, purpose-built release; no source states what produced the instructions or the code, or whether either was model-generated or human-written [4]. **A near-duplicate, larger snapshot of this same data exists in the same author's `mjalg/code-jul-16` repository (see Neighbors); a reader who trains on both duplicates rows.** It lives at https://huggingface.co/datasets/mjalg/function-code .

**Use it for**: single-turn SFT on instruction-to-code generation (the SFT method card), consuming `instruction` as the prompt and `output` as the target completion; there is no `input`/context column, so no augmentation is needed before use. Do not combine with `mjalg/code-jul-16` without deduplicating first (see Neighbors) [5].

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`), ungated, public [4]. The one catch: the license is asserted only as a front-matter tag in an otherwise empty card - no source states the license, generation, or authorship provenance of the underlying code and instructions [3].

**Shape**: 233 rows, one config (`default`), one split (`train`), two string columns (`instruction`, `output`) [6][7].

**Hold out**: nothing - no source states or implies overlap with any evaluation set. The only stated caution is the sibling-dataset duplication risk described above [5].

**Origin**: built and uploaded by Hub user `mjalg`; no source states who or what generated the instructions or the code. Hub API at the check date: `downloads` 35, `downloadsAllTime` 910, `likes` 2 [4][8].

**Trained-on-by**: none found - the repository's own metrics (35 downloads, 2 likes) and its lack of a card or paper turned up no reuse evidence in any source consulted here.

**Introduced by**: no paper - the Hub repository itself, carrying only a license tag and no descriptive text [3].

## Shape

Rows and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 233 |
| total | 233 |

One config, `default`, two columns (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `output` | string |
| `instruction` | string |

Sizes (datasets-server `/size`) [6]: 124,182 bytes of original JSON download, 57,689 bytes as Parquet, 113,360 bytes decoded in memory. No source states sequence-length or token statistics; character counts read from the full 233-row Parquet file give `instruction` lengths of 50-878 characters (mean 285) and `output` lengths of 45-742 characters (mean 194) [1].

## Quality

- Reading the full 233-row Parquet file: all 233 `instruction` values are distinct, and 231 of 233 `output` values are distinct - two `output` values each appear twice, each time paired with a differently worded `instruction` describing the same target function (one duplicate pair both asks for a function that checks a `chartState` variable against `'nodata'` and returns a `NoData` component, phrased two different ways), consistent with intentional instruction paraphrasing rather than accidental duplication [1].
- Of the same 233 rows, 24 outputs contain a JSX-style `return (` block and 26 mention a React type or hook (`React.FC`, `JSX.Element`, `useState`, `useEffect`), so the corpus is a mix of plain TypeScript utility functions and React/TSX component or hook code [1].
- No source states an annotation process, a generating model, a measured error or duplicate rate, or any other quality signal beyond what is derivable from the served rows themselves.

## Load it

```python
import datasets

REV = "40be6ab42a3b6a1fd638290544b0fc1779b7f83c"  # main at the check date
train = datasets.load_dataset("mjalg/function-code", revision=REV, split="train")  # 233 rows
```

**Trap**: the repository has no card body describing the data, its splits, or its intended use - every fact in this card beyond the license tag comes from the served rows and the Hub/datasets-server APIs, not from the dataset's own documentation. There is only one split (`train`); nothing is held out by the repository itself.

## Neighbors

The same author (`mjalg`) has uploaded a series of similarly named, similarly undocumented code datasets; fetching each turned up one clear match [9]:

- `mjalg/code-jul-16` - 1,740 rows, uploaded 2024-07-16 (9 days after this repository), with columns `instruction`, `output`, and an added, empty `input` column. Reading its first 100 served rows and comparing them position-by-position to this dataset's first 100 rows found all 100 `instruction` values identical in the same order, with `output` unchanged and `input` empty; this is consistent with `code-jul-16` being a later, larger superset snapshot of this dataset rather than an independent release, though the comparison covers only the first 100 of each dataset's 233/1,740 rows [9][10]. Its own card is equally empty of prose [9]. Prefer `code-jul-16` over this dataset when a superset is wanted, and never load both without deduplicating.
- `mjalg/proto-code` (991 rows) and `mjalg/code-proto` (1,507 rows) share the `instruction`/`input`/`output` schema but a different content domain - their first rows concern importing a `@protolibrary/components` React library, not the general TS/JS functions seen here - so they are a different corpus, not overlapping snapshots of this one [11].
- `mjalg/reactjs-code` (6 rows) uses an unrelated `Filename`/`Content` schema [9].

## A row

One config, one split, one schema. From `config="default"`, `split="train"`, `row_idx=1` (datasets-server `/first-rows`) [2]:

```json
{
  "output": "const datePickerToApiFormatYear = (date: string) => {\n  // day-month-year --> year month day transformation\n  if (date) {\n    return moment(date, 'DD-MM-yyyy').format('yyyy-MM-dd');\n  }\n}",
  "instruction": "Write a function that transforms a date from the 'day-month-year' format to the 'year-month-day' format using the 'moment' library"
}
```

## Where it came from

Uploaded to the Hub by user `mjalg` under the file name `results (2).json` [4], converted by the Hub into the served Parquet form [1]. No source - not the repository's card, not any linked paper, not the file name itself - states the collection method, the code's origin, or whether the instructions or code were written by a human, generated by a model, or both.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.


[1] Parquet conversion of the dataset, read directly. https://huggingface.co/datasets/mjalg/function-code/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet - all 233 rows, used for the instruction/output pairing, duplicate-count, length, and JSX/React-mention statistics above. Fetched 2026-08-12.

[2] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mjalg%2Ffunction-code&config=default&split=train - first 100 of 233 rows (the endpoint's own cap, flagged `"truncated": true`); used for the sampled row above and cross-checked against the full 233-row Parquet file in [1]. Fetched 2026-08-12.

[3] mjalg/function-code dataset card (README). https://huggingface.co/datasets/mjalg/function-code/raw/main/README.md - a 31-byte file containing only the `license: apache-2.0` front-matter block and no body text. Fetched 2026-08-12.

[4] Hugging Face Hub API record for mjalg/function-code. https://huggingface.co/api/datasets/mjalg/function-code?full=true - license, gate status, `sha`, `downloads`, `likes`, siblings, last-modified date. Fetched 2026-08-12.

[5] Corpus screening row for `mjalg/function-code`, supplied with this card's request. Checked 2026-08-12.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mjalg%2Ffunction-code Fetched 2026-08-12.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=mjalg%2Ffunction-code Fetched 2026-08-12.

[8] Hugging Face Hub API record for mjalg/function-code, expanded for all-time downloads. https://huggingface.co/api/datasets/mjalg/function-code?expand[]=downloadsAllTime Fetched 2026-08-12.

[9] Hugging Face Hub API listing of datasets by author `mjalg`. https://huggingface.co/api/datasets?author=mjalg - used to find `mjalg/code-jul-16`, `mjalg/proto-code`, `mjalg/code-proto`, and `mjalg/reactjs-code`; and the `mjalg/code-jul-16` and `mjalg/reactjs-code` info/size/README endpoints fetched the same way as [3],[6],[7] for their own repositories. Fetched 2026-08-12.

[10] datasets-server first-rows endpoint for mjalg/code-jul-16. https://datasets-server.huggingface.co/first-rows?dataset=mjalg%2Fcode-jul-16&config=default&split=train - first 100 rows, compared positionally against this dataset's first 100 rows from [2]. Fetched 2026-08-12.

[11] datasets-server first-rows endpoint for mjalg/proto-code. https://datasets-server.huggingface.co/first-rows?dataset=mjalg%2Fproto-code&config=default&split=train - first 100 rows, read to confirm a different content domain from this dataset. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as a small SFT source for instruction-to-TypeScript/JavaScript-function generation, but only after deduplicating against `mjalg/code-jul-16`, which this card's own comparison found to reproduce this dataset's first 100 rows verbatim under a larger, later snapshot [10]. The screening row's own note already flags the lack of a card and stated source [5].

### The screening row

The row's own note [5]: "233 TypeScript/JavaScript functions with a written instruction; no card, no stated source." The row carries no flag.
