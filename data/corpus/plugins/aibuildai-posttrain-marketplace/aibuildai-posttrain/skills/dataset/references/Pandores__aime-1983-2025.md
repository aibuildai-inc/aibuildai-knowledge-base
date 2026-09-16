# Pandores/aime-1983-2025

1,034 American Invitational Mathematics Examination (AIME) problems spanning 1983 to 2025, each with an integer answer and one or more AoPS-forum worked solutions, in a single flat table.

Pandores/aime-1983-2025 packages the AIME problem archive into one table keyed by `year`, `index` (problem number within a contest), `part` (`I`, `II`, or none for years before 2000, since AIME split into two parts starting in 2000), the LaTeX `problem` statement, a list of `solutions`, the numeric `answer`, an `all_answers` list for problems with more than one accepted answer, and an optional `note` [1]. The dataset card states the content is derived from the AIME Problems and Solutions page on the Art of Problem Solving (AoPS) Wiki [1]; no origin paper is linked, so this card cites the dataset card itself as the introduction. It serves a math word-problem question-answering task shape: a model reads `problem` and must produce the integer in `answer`. **Because the archive runs through 2025, it carries the complete year-2024 and year-2025 AIME problem sets - both are commonly used as live held-out evaluation sets for LLM math reasoning - so any training use of this dataset must hold out those 60 rows (see Hold out).** It lives at https://huggingface.co/datasets/Pandores/aime-1983-2025 .

**Use it for**: eval-style math QA (score a model's numeric answer against `answer`) on any year, or reasoning-trace SFT built from `problem`/`solutions` pairs restricted to years before 2024; the SFT method card's chat-format conversion applies once a problem and one selected solution are wrapped into a prompt/completion pair. Never train on the year-2024 or year-2025 rows.

**Licence**: not stated on the Hub - `cardData` and the repo's tag list carry no `license` field [2]; the dataset card instead directs users to check the AoPS Wiki's own terms for the applicable licence version [1], so treat the licence as unresolved custom terms rather than a Hub SPDX grant. Ungated (`"gated": false`) [2].

**Shape**: 1,034 rows in one config (`default`), one split (`train`) [3][4].

**Hold out**: the 60 rows with `year` 2024 or 2025 (30 rows each, split 15/15 across `part` I and II) - fetched directly from the served split at offset 974-1033 and confirmed present [5]; the corpus screening flag for this dataset names the same risk [6].

**Origin**: built by Hub user Pandores from the AoPS Wiki's community-authored AIME problem and solution pages [1]; 485 downloads, 0 likes as of the check date [2].

**Trained-on-by**: none found - a Hugging Face models search filtered to this dataset id returns no results [8].

**Introduced by**: no paper - the dataset card [1].

## Shape

One config (`default`), one split (from datasets-server `/size` and `/info`) [3][4]:

| split | rows |
| --- | --- |
| `train` | 1,034 |

| column | dtype |
| --- | --- |
| `year` | int64 |
| `index` | int64 |
| `part` | string |
| `problem` | string |
| `solutions` | list\<string\> |
| `answer` | int64 |
| `all_answers` | list\<int64\> |
| `note` | string |

Sizes (datasets-server `/size`) [3]: 2,178,082 bytes as the original/parquet download, 4,200,871 bytes decoded in memory. No source states sequence-length or token statistics for this dataset; none is invented here.

## Quality

- The dataset card describes `solutions` as a list of human-made solutions per problem, i.e. community-authored write-ups rather than model output [1].
- Across the 181 rows read directly from the served split (47 rows at offset 0, 74 rows at offset 900-973, 60 rows at offset 974-1033, covering years 1983 and 2021-2025), 8 rows carry a non-null `note`: three from 1983 (index 14; index 8) and 1985 (index 6), and five more from 2021 (index 13, part II), 2022 (index 8, part II), 2023 (index 13 and index 5, both part II), and 2024 (index 5, part II) - each a short editorial aside, e.g. flagging redundant given information, a similarity to another contest problem, or a post-contest scoring clarification [1][5][7].
- One of the 181 sampled rows has more than one value in `all_answers` - year 2022, index 2, part II, with `all_answers` `[80, 81]` - confirming the card's own description of `all_answers` as covering problems that can "accept secondary solutions due to ambiguity" [1][5].
- No source states a measured error rate, duplicate rate, or annotator-agreement figure for this dataset.

## Load it

Everything sits in one `train` split with no built-in year filter, so pin the revision this card's numbers were read at and filter out the held-out years yourself:

```python
import datasets

REV = "57c6aa1e65f3e2c73cca4e3c198170ec6fed35fa"
ds = datasets.load_dataset("Pandores/aime-1983-2025", revision=REV, split="train")  # 1,034 rows
train_safe = ds.filter(lambda r: r["year"] not in (2024, 2025))  # drops the 60 held-out rows
```

**Trap**: a plain `load_dataset(..., split="train")` call pulls in the 2024 and 2025 rows along with every other year - there is no separate split or config that isolates them, so the filter step above is required before using any of this data for training [1][3].

## Neighbors

- `di-zhang-fdu/AIME_1983_2024` - 933 rows covering years 1983-2023 plus AIME 2024 part II only; its own card is a disclaimer reading "Do not using in training!" (sic) and calls itself a benchmark [9][10].
- `AI-MO/aimo-validation-aime` - 90 rows, exactly AIME 2022, 2023, and 2024 (30 each); its card states these years were chosen specifically "to avoid potential overlap with the MATH training set" [9][11].
- `math-ai/aime25` - 30 rows, AIME 2025 only, served as a `test` split [9][12].

Direct row fetches from this dataset's own served split confirm it contains full 30-row sets for 2022, 2023, 2024, and 2025 [5] - so Pandores/aime-1983-2025 duplicates every row in all three smaller, purpose-built releases above. Prefer this archive only for the pre-2022 years none of those releases carry; prefer the smaller releases for an eval on 2022-2025 specifically, since mixing this archive's 2022-2025 rows into a training run risks the same contamination those smaller releases were built to avoid.

## A row

One config and one split, so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], with the `solutions` list truncated to its first entry and that entry truncated:

```json
{
  "year": 1983,
  "index": 1,
  "part": null,
  "problem": "Let $x$, $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_x w = 24$, $\\log_y w = 40$ and $\\log_{xyz} w = 12$. Find $\\log_z w$.",
  "solutions": [
    "The logarithmic notation doesn't tell us much, so we'll first convert everything to the equivalent exponential forms. [...] $\\log_zw=\\boxed{060}$."
  ],
  "answer": 60,
  "all_answers": [60],
  "note": null
}
```

## Where it came from

Built and released by Hub user Pandores from the AoPS Wiki's AIME Problems and Solutions page, a community-maintained archive of past AIME contest problems and forum-contributed solutions [1]. The dataset card gives no separate collection method or generating model beyond this AoPS Wiki source, and tells users to check the Wiki's own terms for the licence that applies to the underlying content [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable, which is why Load it pins the revision to the commit sha (`57c6aa1e65f3e2c73cca4e3c198170ec6fed35fa`, matching source [2]'s `sha` at the check date) that `load_dataset` resolves against. The datasets-server endpoints used for sources [3][4][5][7][9] take a `revision` query parameter, but it has no confirmed pinning effect: `size?dataset=Pandores%2Faime-1983-2025&revision=main` returns the identical body as the call with no `revision` parameter at all, checked live on the check date. Every row count, byte size, and sampled-row figure drawn from those endpoints in this card - for both Pandores/aime-1983-2025 and its neighbors - is therefore a live read as of the check date, not a value pinned to the commit sha; only what `load_dataset(..., revision=REV)` itself resolves is covered by that pin.

[1] Pandores/aime-1983-2025 dataset card (README). https://huggingface.co/datasets/Pandores/aime-1983-2025/raw/main/README.md - feature descriptions, data-source statement, licence note, load example. Fetched 2026-08-11.

[2] Hugging Face Hub API record for Pandores/aime-1983-2025. https://huggingface.co/api/datasets/Pandores/aime-1983-2025?full=true - `sha`, `gated`, `cardData`, `downloads`, `likes`, tag list (no `license` tag). Fetched 2026-08-11.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Pandores%2Faime-1983-2025 Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Pandores%2Faime-1983-2025 Fetched 2026-08-11.

[5] datasets-server rows endpoint, read at offset 900 (length 74, covering years 2021-2023) and offset 974 (length 60, covering years 2024-2025). https://datasets-server.huggingface.co/rows?dataset=Pandores%2Faime-1983-2025&config=default&split=train&offset=900&length=74 and same with offset=974&length=60 - used to count `year`/`part` rows and confirm the 2022, 2023, 2024, and 2025 sets are each complete at 30 rows. Fetched 2026-08-11.

[6] The corpus screening row for `Pandores/aime-1983-2025`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[7] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Pandores%2Faime-1983-2025&config=default&split=train - row 0 sample and the 47 rows at offset 0 used in the Quality note/all_answers count. Fetched 2026-08-11.

[8] Hugging Face models API filtered to this dataset. https://huggingface.co/api/models?filter=dataset:Pandores/aime-1983-2025&limit=20 - empty result. Fetched 2026-08-11.

[9] datasets-server size endpoint, one call per neighbor: `di-zhang-fdu/AIME_1983_2024`, `AI-MO/aimo-validation-aime`, `math-ai/aime25`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[10] di-zhang-fdu/AIME_1983_2024 dataset card (README). https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/raw/main/README.md - year coverage, "Do not using in training!" disclaimer, MIT licence. Fetched 2026-08-11.

[11] AI-MO/aimo-validation-aime dataset card (README). https://huggingface.co/datasets/AI-MO/aimo-validation-aime/raw/main/README.md - year coverage (AIME 22/23/24), stated purpose of avoiding MATH-training-set overlap. Fetched 2026-08-11.

[12] math-ai/aime25 dataset card (README). https://huggingface.co/datasets/math-ai/aime25/raw/main/README.md - AIME 2025 scope, `test` split, Apache-2.0 licence. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable only outside its year-2024 and year-2025 rows: this card's own row fetch confirms both live-eval years are present at full strength (30 rows each) [5], matching the corpus screening flag's contamination warning [6]. Every other year (1983-2023) carries no equivalent live-eval overlap noted by any source read for this card.

### The screening row

The row's own note [6]: "Full AIME archive with AoPS solutions; 30 AIME-2024 and 30 AIME-2025 problems confirmed present, so it carries both live AIME eval sets." Its flag: "contamination: carries AIME 2024 and AIME 2025, both live eval sets."
