# AI-MO/aimo-validation-amc

83 competition-math problems - 43 from AMC 12 2022 and 40 from AMC 12 2023, each rewritten to a single integer answer - built as an internal validation set for the AI Mathematical Olympiad progress prize.

**AI-MO/aimo-validation-amc** was built by the AI-MO team and extracted from the Art of Problem Solving wiki's AMC 12 problem-and-solution pages [1]; the card states no separate paper, so it is introduced by the dataset card itself [1]. Each of the original multiple-choice AMC 12 problems was rewritten so its answer is a single integer instead of one of four choices, and problems that could not be rewritten this way were dropped [1]. The card states the AMC 12 2022/2023 range was chosen specifically "to avoid potential overlap with the MATH training set" [1]. **This is a validation/eval set, and its AMC-2023 half (40 rows) is served, row for row, as the widely-used `amc23` evaluation benchmark: a live row comparison against `math-ai/amc23` (11,185 downloads) found the same problem text, answer, and AoPS URL for a matching row, differing only in column name (`problem` vs `question`) and answer dtype (float vs string) [2][3]. Training on this repository risks contaminating any run that later scores on `amc23`.** It lives at https://huggingface.co/datasets/AI-MO/aimo-validation-amc .

**Use it for**: nothing in a post-training pipeline - it is an evaluation/validation set, not a training corpus; the card describes it only as an internal AIMO-competition validation set, and both the format (single problem/answer pairs with no chat structure) and the contamination risk above rule out routine SFT or preference use. Held out, it can serve as a held-out reasoning-eval set (single free-form integer answer), the same shape as the `amc23` benchmark it doubles as. **Do not include any of its rows, especially the 40 AMC-2023 rows, in a training or RL split.**

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`) [4].

**Shape**: 83 rows, one config (`default`), one split (`train`), four columns (`id`, `problem`, `answer`, `url`) [4][5][6].

**Hold out**: all 83 rows if AMC-2022/2023 problems could appear in an eval elsewhere, and specifically all 40 AMC-2023 rows before any run that will later be scored on the `amc23` benchmark, since that benchmark is this repository's AMC-2023 half [2][3].

**Origin**: built by AI-MO; problems are taken verbatim from AMC 12 competitions and rewritten by the AI-MO team into integer-answer form, with no model or crowd generation involved [1]. Hub API at the check date: `downloads` 4,849, `downloadsAllTime` 44,535, `likes` 19 [4].

**Trained-on-by**: none found - no source confirms a model trained on this repository's rows. Its AMC-2023 half is instead reused, unmodified, as an evaluation benchmark: `math-ai/amc23` (11,185 downloads) serves the same 40 problems [2][3], and `knoveleng/AMC-23` (40 rows, columns `id`/`problem`/`answer`/`url`/`question`) matches this repository's column layout closely enough to be the same lineage, though it was not row-compared here [7].

**Introduced by**: no paper - the dataset card itself [1].

## Shape

Splits and rows (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 83 |

One config, `default`, with four columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `problem` | string |
| `answer` | float64 |
| `url` | string |

Sizes (datasets-server `/size`) [5]: 19,141 bytes as Parquet (original and served size are equal), 32,721 bytes decoded in memory. No source states sequence-length or token statistics for this repository; none is invented here. The `/size`, `/info`, and `/first-rows` endpoints used in this section take no revision parameter, so these row counts and byte sizes are live reads at the check date, not covered by the `sha` pin in Load it [2][5][6].

Reading all 83 served rows (the entire split fits in one `first-rows` call) [2] confirms the card's count split exactly: 43 rows whose `url` names a `2022_AMC_12` page and 40 whose `url` names a `2023_AMC_12` page, matching the screening row's note of "AMC 2022 (43) plus AMC 2023 (40)" [8]. The `id` column is not a dataset-wide key: it restarts at 0 within each year's block, so `id=0` occurs twice - once at served `row_idx=0` (a 2022 problem) and once at served `row_idx=43` (a 2023 problem) [2]. Use `row_idx` from `first-rows`, or the `url`, to address a specific row unambiguously.

## Quality

- The card states its own construction rule: original AMC 12 problems were multiple-choice with four options, and were rewritten to have an integer output; "Those problems whose statement can not be modified are rejected." [1].
- No source states a measured error rate, duplicate rate, or annotator-agreement figure for the rewritten problems or answers; none is invented here.
- The card gives one worked example of the rewrite (the "Flora the frog" problem, turned from a five-way multiple choice into a "find $m+n$" integer-answer question), but does not state how many of the 83 rows needed a similar fraction-to-integer conversion versus a more direct rewrite [1].

## Load it

Single config, single split, no external fetch needed; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2025-05-07) [4]:

```python
import datasets

REV = "69d78a4a2c840e82d69af6bc742bda09005f6316"  # main at the check date
ds = datasets.load_dataset("AI-MO/aimo-validation-amc", revision=REV, split="train")  # 83 rows
```

**Trap**: there is only one split (`train`) and it is the whole dataset - there is no separate eval split to hold back. The entire 83 rows are themselves an evaluation set in the AIMO sense, and 40 of them are the `amc23` benchmark verbatim [2][3]; loading `train` for training data silently trains on that benchmark's held-out problems.

## Neighbors

- `AI-MO/aimo-validation-aime` - a sibling AI-MO validation set of 90 AIME problems (AIME 2022, 2023, 2024), same construction rationale ("avoid potential overlap with the MATH training set") and near-identical schema (`id`, `problem`, `solution`, `answer`, `url`) [9][10].
- `AI-MO/aimo-validation-math-level-4` and `AI-MO/aimo-validation-math-level-5` - further AI-MO validation sets, 754 and 721 rows respectively at the check date, same family [10].
- `math-ai/amc23` - 40 rows, the AMC-2023 half of this repository re-served as a standalone eval benchmark; `math-ai/amc23` row 0 (`config="default"`, `split="test"`) matches this repository's row at served `row_idx=43` (the first 2023-labeled row, itself carrying `id=0` since the `id` column restarts each year) exactly in problem text, answer value, and URL, confirmed by direct comparison [2][3]. Prefer the AMC-2023 rows in this repository as the authoritative copy when checking for contamination, since `math-ai/amc23`'s own README carries no body text describing its provenance [3].
- `knoveleng/AMC-23` - 40 rows with an extra `question` column alongside this repository's `id`/`problem`/`answer`/`url` columns, consistent with (but not row-verified against) the same AMC-2023 subset [7].
- Numerous further `amc23` mirrors exist on the Hub (e.g. `zwhe99/amc23`, `TianHongZXY/amc23`); none was row-compared here, so treat any of them as a possible re-serving of this repository's AMC-2023 half until checked.

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [2]:

```json
{
  "id": 0,
  "problem": "$\\frac{m}{n}$ is the Irreducible fraction value of \\[3+\\frac{1}{3+\\frac{1}{3+\\frac13}}\\], what is the value of $m+n$?",
  "answer": 142.0,
  "url": "https://artofproblemsolving.com/wiki/index.php/2022_AMC_12A_Problems/Problem_1"
}
```

## Where it came from

Built by the AI-MO team from the Art of Problem Solving wiki's AMC 12 problem pages, covering AMC 12 2022 and AMC 12 2023 [1]. The card states the post-2021 window was chosen "to avoid potential overlap with the MATH training set" [1]. Each original multiple-choice problem (four answer choices) was rewritten by the builders into a single-integer-answer form, and problems that resisted this rewrite were dropped, leaving 83 of the available AMC 12 2022/2023 problems [1]. No model or crowd-worker generation is involved; the problems and their answers come from the competitions themselves, adapted by the builders [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] AI-MO/aimo-validation-amc dataset card (README). https://huggingface.co/datasets/AI-MO/aimo-validation-amc/raw/main/README.md - construction rationale, rewrite rule, column descriptions, worked example. Fetched 2026-08-11.

[2] datasets-server first-rows endpoint (returns all 83 rows in one call). https://datasets-server.huggingface.co/first-rows?dataset=AI-MO%2Faimo-validation-amc&config=default&split=train - used to read every served row, confirm the 43/40 AMC-2022/2023 split, and sample row 0. Fetched 2026-08-11.

[3] math-ai/amc23 dataset: dataset card (no body text) and datasets-server first-rows endpoint, compared row-for-row against source [2]. https://huggingface.co/datasets/math-ai/amc23/raw/main/README.md and https://datasets-server.huggingface.co/first-rows?dataset=math-ai%2Famc23&config=default&split=test - used for the row-0 comparison establishing that `amc23` is this repository's AMC-2023 half, and for its download count via the Hub API. Fetched 2026-08-11.

[4] Hugging Face Hub API record for AI-MO/aimo-validation-amc. https://huggingface.co/api/datasets/AI-MO/aimo-validation-amc?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=AI-MO%2Faimo-validation-amc Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=AI-MO%2Faimo-validation-amc Fetched 2026-08-11.

[7] knoveleng/AMC-23 dataset card and datasets-server size endpoint. https://huggingface.co/datasets/knoveleng/AMC-23/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=knoveleng%2FAMC-23 - row count and column schema; not row-compared against source [2]. Fetched 2026-08-11.

[8] The corpus screening row for `AI-MO/aimo-validation-amc`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[9] AI-MO/aimo-validation-aime dataset card (README). https://huggingface.co/datasets/AI-MO/aimo-validation-aime/raw/main/README.md - same construction rationale, schema, and row count for the AIME sibling. Fetched 2026-08-11.

[10] datasets-server size endpoint, one call per neighbor: `AI-MO/aimo-validation-aime`, `AI-MO/aimo-validation-math-level-4`, `AI-MO/aimo-validation-math-level-5`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Flagged for contamination and excluded from any training split: this repository's AMC-2023 half is the `amc23` evaluation benchmark itself, confirmed above by a live row comparison against `math-ai/amc23` [2][3], and the screening row's own flag names the same risk [8].

### The screening row

The row's own note [8]: "AMC 2022 (43) plus AMC 2023 (40); the amc23 benchmark source." Its flag [8]: "contamination: AMC 2022 and 2023, the amc23 eval source."
