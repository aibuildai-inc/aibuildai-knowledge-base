# zuom/AIME-solutions

962 AIME competition problems, 1983 through 2024, each paired with its numeric answer and a list of AoPS-style alternate solution write-ups.

**zuom/AIME-solutions** is a tabular archive of American Invitational Mathematics Examination (AIME) problems keyed `1983-I-1` through `2024-II-15`, where each row carries the problem statement, integer answer, and a `Solution` column holding a list of independent worked solutions per problem [1]. The dataset card itself is metadata-only - a YAML `dataset_info`/`configs` block with no prose description, no stated collection method, and no license field [1]. It lives at https://huggingface.co/datasets/zuom/AIME-solutions . **The row-level read below shows the tail (dataset rows 918-961, all 44 of them) is entirely AIME 2024 problems, so any AIME-2024 evaluation must be checked against and held out from this data before training [2].**

**Use it for**: reasoning-trace SFT or RL over math word problems - the `Question`/`Answer` pair gives a gradeable target, and the `Solution` list gives ready-made worked derivations a trainer could turn into chain-of-thought SFT targets - but only after dropping the 44-row 2024 tail (dataset indices 918-961) to avoid AIME-2024 eval contamination. See the reasoning-trace SFT method card. Every row's `Answer` is a bare integer (AIME answers are always integers 0-999), not a full derivation, so the `Solution` strings are the derivation source if one is needed.

**Licence**: not stated - the Hub API record carries no `license` field and the README's YAML front matter has none either [1][3]. Treat it as no explicit grant; the one catch is that use is unclarified, not merely restricted.

**Shape**: 962 rows, one split (`train`), one config (`default`), 7 columns [3][4].

**Hold out**: the 44 rows at dataset indices 918-961, all with `Year` 2024 - hold these out of any training run that will later be scored against AIME 2024 [2]. Within that block, 14 of the 30 distinct 2024-II problems are stored twice (`2024-II-1` through `2024-II-15` except `2024-II-9`); each pair has identical `Answer` and identical `Solution` content, but the `Question` string is never byte-identical between the two copies - checked for all 14 pairs [2]. For 11 of the 14 pairs the difference is small (0-48 characters: LaTeX dollar-sign wrapping or whitespace reflow); for 3 pairs (`2024-II-8`, `2024-II-10`, `2024-II-15`) one copy's `Question` carries 268-470 extra characters of either an Asymptote diagram block (`[asy]...[/asy]`) or a chunk of worked-solution text that the other copy lacks [2]. So the 44 rows cover only 30 distinct problems: the duplication does not add contamination risk beyond those 30 problems, but it does mean naive row-count planning overstates unique 2024 material by 14 rows.

**Origin**: built by Hub user `zuom`; problems and answers are the official AIME contest questions, and the alternate `Solution` write-ups read like the multi-solution format used on the Art of Problem Solving wiki, though the card does not name a source for them [1]. Hub API at the check date: 1,129 downloads, 4,349 all-time downloads, 1 like [3].

**Trained-on-by**: none found - no source cites a model or training recipe using this specific repository.

**Introduced by**: no paper - the dataset card `[1]` is the only description, and it has no prose beyond the YAML schema.

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 962 |

One config, `default`, with seven columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `ID` | string |
| `Year` | int64 |
| `Problem Number` | int64 |
| `Question` | string |
| `Answer` | int64 |
| `Part` | string (nullable) |
| `Solution` | list\<string\> |

Sizes (datasets-server `/size`) [4]: 2,241,873 bytes of original/parquet download, 5,751,406 bytes decoded in memory. No source states sequence-length or token statistics for `Question` or `Solution`.

A full read of all 962 rows, fetched and saved in ten sequential 100-row pages through the datasets-server `/rows` endpoint (offsets 0, 100, 200, ..., 900), gives the year distribution: 1983-1999 carry 9-15 rows per year with `Part` null (one exam per year, 224 rows total); 2000-2023 carry 26-30 rows per year with `Part` set to `I` or `II` (the two-sitting AIME format); 2024 carries 44 rows [2]. `Part` is null for every 1983-1999 row and set to `I` or `II` for every 2000-2024 row [2].

## Quality

- The same full read finds 948 distinct `ID` values across 962 rows: exactly 14 duplicate IDs, all of them `2024-II-*` (every 2024-II problem except `2024-II-9`), each appearing exactly twice with identical `Answer` and `Solution` content but a `Question` string that is never byte-identical between the two copies (checked for all 14 pairs) [2]. Most of these `Question` differences are cosmetic (LaTeX dollar-sign wrapping or whitespace reflow, 0-48 characters), but 3 of the 14 pairs (`2024-II-8`, `2024-II-10`, `2024-II-15`) differ by 268-470 characters, where one copy's `Question` field carries an Asymptote diagram block or an extra chunk of worked-solution text that the other copy is missing [2]. No duplicates exist outside the 2024-II block.
- No source states a measured contamination rate, annotator-agreement figure, or solution-correctness check; the card carries no quality prose at all [1].
- The `Solution` list length varies by problem: the three sampled rows at offset 0 (`1983-I-1`, `1983-I-2`, `1983-I-3`) hold 6, 2, and 4 alternate solutions respectively [6]. This is read from three rows only and is not a claim about the full 962-row distribution.

## Load it

The repository has one split and needs no format conversion before use as reasoning-trace SFT source data, but the 2024 tail must be filtered out of any run that will be scored against AIME 2024. Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2025-04-04) [3]:

```python
import datasets

REV = "d31502c1731d79d7205d86cabc89aca948e547f9"  # main at the check date
ds = datasets.load_dataset("zuom/AIME-solutions", revision=REV, split="train")  # 962 rows
train_only = ds.filter(lambda r: r["Year"] != 2024)  # drops the 44-row 2024 tail (30 distinct problems)
```

**Trap**: filtering on `Year != 2024` alone removes all 44 rows, including all 14 duplicated `2024-II` pairs, and is sufficient by itself - deduplicating on `ID` is a separate, optional cleanup step (removing 14 more rows from the remaining 918, since `ID` is otherwise unique) and does not change which rows carry 2024 contamination, since every duplicated `ID` is itself 2024 data; the two operations commute and can be applied in either order.

## Neighbors

- `zuom/AIME-complete-QA` - same author, same 962 rows, same `ID`/`Year`/`Problem Number`/`Question`/`Answer`/`Part` schema and identical first-row content, but without the `Solution` column [7][8]. This looks like the QA-only sibling of this release; prefer this repository (`AIME-solutions`) when solutions are needed, and the sibling when they are not.
- `di-zhang-fdu/AIME_1983_2024` - 933 rows, `Answer` stored as a string rather than int64, and `ID` values formatted `1983-1` rather than `1983-I-1`; its first row's `Question` text matches this release's `1983-I-1` word for word [9][10]. It has 15 fewer rows than this release's 948 distinct problems, and no `Solution` column; treat it as a related but not identical problem/answer source, not a drop-in replacement.

## A row

One config, one split, one schema - one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/rows`) [2]. The `Solution` array itself holds 6 entries in this row; only the first 2 are shown here, each string also truncated with `[...]`:

```json
{
  "ID": "1983-I-1",
  "Year": 1983,
  "Problem Number": 1,
  "Question": "Let $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_xw=24$ , $\\log_y w = 40$ and $\\log_{xyz}w=12$ . Find $\\log_zw$ .",
  "Answer": 60,
  "Part": null,
  "Solution": [
    "The logarithmic notation doesn't tell us much, so we'll first convert everything to the equivalent exponential forms. [...] $\\log_zw=060.",
    "First we'll convert everything to exponential form. [...] So our answer is $060.",
    "... 4 more Solution entries omitted here, not shown ..."
  ]
}
```

## Where it came from

The Hub API and the dataset card give only a builder and a last-modified date; the card carries no collection-method prose [1][3]. The repository tree holds a `data/` directory (parquet), a `.gitattributes` file, and the 489-byte metadata-only `README.md` - no script or documentation file describing how the problems, answers, or `Solution` write-ups were assembled [11]. The `ID` values match the standard AIME numbering (`<year>-<I|II>-<problem>`), and the `Question` text for the sampled row matches the same problem as it appears in `di-zhang-fdu/AIME_1983_2024`, a different Hub repository of the same contest archive [9][10], which supports the problems being the official AIME set rather than a paraphrase, though no source states the specific origin of the `Solution` write-ups.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision to the `sha` recorded in source [3]. The datasets-server endpoints used throughout Shape, Quality, and A row (sources [2], [4], [5], [6], [8], [10]) take no revision parameter: they always serve the current `main` branch, so the row counts, duplicate-`ID` findings, and sampled row above describe the repository as it stood at fetch time on the check date, not necessarily the pinned sha - though the repo's last-modified date [3] predates the check date, so no divergence is expected.

[1] zuom/AIME-solutions Hub dataset page, https://huggingface.co/datasets/zuom/AIME-solutions , and its raw README source, https://huggingface.co/datasets/zuom/AIME-solutions/raw/main/README.md - full README content is a YAML `dataset_info`/`configs` block, no prose, no license field. Fetched 2026-08-11.

[2] Full read of all 962 rows of `zuom/AIME-solutions`, `config=default`, `split=train`, fetched and saved as ten sequential 100-row pages via the datasets-server `/rows` endpoint at offsets 0, 100, 200, ..., 900 (the last page holds 62 rows). https://datasets-server.huggingface.co/rows?dataset=zuom%2FAIME-solutions&config=default&split=train - year distribution, `Part` null/non-null pattern, duplicate `ID` count and location, the 918-961 boundary of the 2024 block, and the row-by-row `Question`/`Answer`/`Solution` comparison of all 14 duplicate-`ID` pairs. This endpoint takes no revision parameter, so it reflects the `main` branch at fetch time, not the sha pinned in Load it. Fetched 2026-08-11.

[3] Hugging Face Hub API record for zuom/AIME-solutions. https://huggingface.co/api/datasets/zuom/AIME-solutions?full=true - `sha`, `downloads`, `likes`, last-modified date, `cardData`, no `license` field; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=zuom%2FAIME-solutions Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=zuom%2FAIME-solutions Fetched 2026-08-11.

[6] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=zuom%2FAIME-solutions&config=default&split=train Fetched 2026-08-11.

[7] Hugging Face Hub API listing for author `zuom`. https://huggingface.co/api/datasets?author=zuom - lists `zuom/AIME-complete-QA` alongside this repository. Fetched 2026-08-11.

[8] datasets-server info and first-rows endpoints for `zuom/AIME-complete-QA`. https://datasets-server.huggingface.co/info?dataset=zuom%2FAIME-complete-QA and https://datasets-server.huggingface.co/first-rows?dataset=zuom%2FAIME-complete-QA&config=default&split=train - schema (six columns, no `Solution`), row count, and first-row content compared against this release. Fetched 2026-08-11.

[9] Hugging Face Hub search for "AIME". https://huggingface.co/api/datasets?search=AIME&limit=100 - located `di-zhang-fdu/AIME_1983_2024` among other AIME repositories. Fetched 2026-08-11.

[10] datasets-server info and first-rows endpoints for `di-zhang-fdu/AIME_1983_2024`. https://datasets-server.huggingface.co/info?dataset=di-zhang-fdu%2FAIME_1983_2024 and https://datasets-server.huggingface.co/first-rows?dataset=di-zhang-fdu%2FAIME_1983_2024&config=default&split=train - row count (933), `Answer` dtype (string), `ID` format, and first-row `Question` text compared against this release. Fetched 2026-08-11.

[11] Hugging Face Hub tree listing for zuom/AIME-solutions at `main`. https://huggingface.co/api/datasets/zuom/AIME-solutions/tree/main - confirms only `data/`, `.gitattributes`, and a 489-byte `README.md` exist in the repository. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Flagged for contamination, usable only after the tail is removed. The card's own row-level read confirms the screening claim exactly: dataset indices 918-961 (44 rows) are all `Year` 2024, matching the screening flag's count, and that block must be held out of any run scored against AIME 2024 [2].

### The screening row

The row's own note: "AIME archive with AoPS solutions keyed 1983-I-1 onward; the tail contains 44 rows from 2024." The row's flag, under the `contamination` class: "contamination: carries AIME 2024 (44 rows in the tail)."
