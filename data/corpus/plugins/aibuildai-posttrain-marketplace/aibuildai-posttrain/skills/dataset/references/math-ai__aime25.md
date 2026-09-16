# math-ai/aime25

30 problems from the 2025 American Invitational Mathematics Examination, each with a problem statement and a final numeric answer - a small, live competition-math benchmark, not a training corpus.

**math-ai/aime25** packages the American Invitational Mathematics Examination (AIME) 2025 into a single `test` split of problem/answer/id rows; the dataset card names no paper, giving only a BibTeX entry crediting Yifan Zhang and a second author listed as Math-AI, Team [1]. Its three columns hold a competition problem statement, a gold answer, and a row id, not a training signal [1][2]. It lives at https://huggingface.co/datasets/math-ai/aime25 . **This is a live, unmodified eval set: the 30 problems and their gold answers are the actual 2025 AIME I and II competition papers, so any training corpus that contains them (verbatim or paraphrased) contaminates AIME-2025 benchmark results. Hold out the full 30 rows from every training run.**

**Use it for**: evaluation only, never training. The `problem`/`answer`/`id` shape [1][2] is the shape a math-eval harness needs to prompt a model with `problem` and check its output against `answer`; no fetched source states which specific harness or scoring convention the dataset's own maintainers intend. Do not route these rows into an SFT or preference method card; the dataset card gives no instructions for that use [1].

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`; the repository ships a matching `LICENSE` file that opens with "Apache") [1][3]. The one catch: this licence grant is on the Hub repository's packaging, not a statement about the underlying AIME problems' own copyright status - no fetched source addresses that separately [1][3].

**Shape**: 30 rows, one config (`default`), one split (`test`), three string columns (`problem`, `answer`, `id`) [4][2].

**Hold out**: all 30 rows, entirely - there is nothing in this dataset to split into train/test, because the whole release is itself an evaluation set. Contaminating any training corpus with these problems or their answers is what the screening flag warns against [5].

**Origin**: built by the `math-ai` Hub organization (credited to Yifan Zhang and a second author listed as Math-AI, Team in the card's citation block); the problems and gold answers come from the human-authored 2025 AIME competition, not model generation [1]. Hub API at the check date: `downloads` 80,567, `downloadsAllTime` 512,860, `likes` 36 [3].

**Trained-on-by**: none found. No fetched source states that a named model or training recipe trained on this repository; as a live eval benchmark it is the kind of dataset recipes are checked against, not trained on [1].

**Introduced by**: no paper - the dataset card's citation block only, crediting Zhang and a second author listed as Math-AI, Team, with no arXiv link or external announcement [1].

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `test` | 30 |

One config, `default`, three columns (datasets-server `/info`) [2]:

| column | dtype |
| --- | --- |
| `problem` | string |
| `answer` | string |
| `id` | string |

Sizes (datasets-server `/size`) [4]: 15,813 bytes of original JSON download, 13,833 bytes as Parquet, 14,799 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The gold `answer` values come from the official AIME 2025 answer key by way of the human-authored competition itself; no source states a separate verification or annotation process for this specific repository [1].
- No source states a measured contamination rate, duplicate rate, or error rate for this release's 30 rows.
- The repository's only tree entries are `.gitattributes`, `LICENSE`, `README.md`, and `test.jsonl` - no eval-configuration YAML and no held-back "source_datasets" trail beyond the AIME competition itself [6].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2026-01-19) [3]:

```python
import datasets

REV = "563bb8404243c5f09de6ec262f2db674fe5bce9b"  # main at the check date
test = datasets.load_dataset("math-ai/aime25", revision=REV, split="test")  # 30 rows - eval only, never train
```

**Trap**: there is only one split, `test`, and no `train` split to accidentally load instead - the trap here is using this data as training input at all. The dataset card gives no fine-tuning instructions, only a citation block, so anyone loading it should route it straight into an eval harness and never into a training pipeline [1]. The `revision` pin above covers what `load_dataset` reads at that git commit - the README, `LICENSE`, and `test.jsonl` contents [3][6]. It does not cover this card's row-count, byte-size, and sampled-row figures, which come from the datasets-server `/size`, `/info`, and `/first-rows` endpoints; those endpoints take no revision parameter, so they report live state and are only current as of the check date [4][2][7].

## Neighbors

The `math-ai` organization ships sibling AIME/AMC/OlympiadBench eval sets in the same shape; other orgs ship separately-sourced AIME-2025 releases with different columns and splits. Every row count and column list below was read live at the check date.

- `math-ai/aime24` - the 2024 AIME, same organization, 30 rows in one `test` split, but a different schema: `id`, `problem`, `solution`, `url` (no `answer` column) [8].
- `math-ai/aime26` - the 2026 AIME, same organization, same three-column `problem`/`answer`/`id` schema and 30-row single `test` split as this release [9].
- `math-ai/amc23` - the 2023 AMC competition from the same organization, 40 rows in a single `test` split, four columns [10]. `math-ai/gpqa`, `math-ai/math500`, `math-ai/olympiadbench`, `math-ai/minervamath`, `math-ai/BlueMO` are further sibling benchmarks listed under the same organization [11]; their splits and columns were not fetched for this card, so no structural claim is made about them here. Use them for their own competitions/subjects, not as substitutes for AIME 2025.
- `opencompass/AIME2025` - the same 30 AIME-2025 problems from a different builder, but split into two 15-row configs (`AIME2025-I`, `AIME2025-II`) with a `question`/`answer` schema instead of this release's single `problem`/`answer`/`id` `test` split [12][13].
- `yentinglin/aime_2025` - 30 rows in a `default` config (columns `id`, `problem`, `answer`, `solution`, `url`, `year`, plus an index column) served on a `train` split rather than `test`, plus `part1`/`part2` 15-row configs matching AIME I/II; carries worked `solution` text this release does not [14][15].
- `MathArena/aime_2025` - 30 rows on a `train` split with columns `problem_idx`, `problem`, `answer` (int64), `problem_type`, released under CC BY-NC-SA 4.0 rather than this release's Apache-2.0, and tied to the MathArena leaderboard paper [16][17].
- This corpus prefers `math-ai/aime25` for its simple three-column, single-`test`-split shape and permissive Apache-2.0 licence; reach for a neighbor only when its split structure (AIME I vs II separately) or licence terms are what a specific eval harness requires.

## A row

One config and one split, so one row covers it. From `config="default"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [7]:

```json
{
  "problem": "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$",
  "answer": "70",
  "id": "0"
}
```

`problem` holds the LaTeX-formatted competition question, `answer` the gold final answer as a numeric string, and `id` a per-row index string ("0" through "29"). Of the first three served rows, problem length varies widely - the geometry problem at `row_idx=1` runs several sentences describing a labeled figure, while the base-arithmetic problem above is a single sentence [7].

## Where it came from

Built and released by the `math-ai` Hub organization; the dataset card names Yifan Zhang and a second author listed as Math-AI, Team in its citation block and states no separate collection methodology beyond packaging the competition [1]. The problems and gold answers originate from the official 2025 American Invitational Mathematics Examination (AIME I and AIME II), a human-authored, proctored competition; no source states that any model generated or altered the problems or answers in this repository [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] math-ai/aime25 dataset card (README). https://huggingface.co/datasets/math-ai/aime25/raw/main/README.md - title, citation block, licence tag, feature/split declaration; the item's own Hub page, https://huggingface.co/datasets/math-ai/aime25, renders this same card and was separately fetched to confirm it opens. Fetched 2026-08-11.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=math-ai%2Faime25 Fetched 2026-08-11.

[3] Hugging Face Hub API record for math-ai/aime25. https://huggingface.co/api/datasets/math-ai/aime25?full=true and the `expand[]=downloadsAllTime` variant - licence, gate, `sha`, `downloads`, `downloadsAllTime`, `likes`, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=math-ai%2Faime25 Fetched 2026-08-11.

[5] The corpus screening row for `math-ai/aime25`, supplied with this card's request - its `note` and `flag`, read back in their own words in the appendix. Checked 2026-08-11.

[6] Hugging Face Hub tree listing for math-ai/aime25 at `main`. https://huggingface.co/api/datasets/math-ai/aime25/tree/main - confirms the repository's only files are `.gitattributes`, `LICENSE`, `README.md`, `test.jsonl`. Fetched 2026-08-11.

[7] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=math-ai%2Faime25&config=default&split=test Fetched 2026-08-11.

[8] math-ai/aime24 dataset card and datasets-server size endpoint. https://huggingface.co/datasets/math-ai/aime24/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=math-ai%2Faime24 Fetched 2026-08-11.

[9] math-ai/aime26 dataset card and datasets-server size endpoint. https://huggingface.co/datasets/math-ai/aime26/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=math-ai%2Faime26 Fetched 2026-08-11.

[10] datasets-server size endpoint for math-ai/amc23. https://datasets-server.huggingface.co/size?dataset=math-ai%2Famc23 - row count and column count only; the dataset's own README was not fetched for this card. Fetched 2026-08-11.

[11] math-ai Hub organization dataset listing. https://huggingface.co/api/datasets?author=math-ai&limit=100 - names the sibling AIME/AMC/OlympiadBench/GPQA/MATH-500/BlueMO releases under the same organization; no structural detail on gpqa, math500, olympiadbench, minervamath, or BlueMO is claimed beyond their listing here. Fetched 2026-08-11.

[12] opencompass/AIME2025 dataset card. https://huggingface.co/datasets/opencompass/AIME2025/raw/main/README.md Fetched 2026-08-11.

[13] datasets-server size and info endpoints for opencompass/AIME2025. https://datasets-server.huggingface.co/size?dataset=opencompass%2FAIME2025 and https://datasets-server.huggingface.co/info?dataset=opencompass%2FAIME2025 Fetched 2026-08-11.

[14] yentinglin/aime_2025 dataset card. https://huggingface.co/datasets/yentinglin/aime_2025/raw/main/README.md Fetched 2026-08-11.

[15] datasets-server size endpoint for yentinglin/aime_2025. https://datasets-server.huggingface.co/size?dataset=yentinglin%2Faime_2025 Fetched 2026-08-11.

[16] MathArena/aime_2025 dataset card. https://huggingface.co/datasets/MathArena/aime_2025/raw/main/README.md - schema, CC BY-NC-SA 4.0 licence, MathArena leaderboard citation. Fetched 2026-08-11.

[17] datasets-server size endpoint for MathArena/aime_2025. https://datasets-server.huggingface.co/size?dataset=MathArena%2Faime_2025 Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Held out, not trained on: this release is a live, unmodified 30-problem AIME 2025 answer key, and every row must stay out of any training corpus to keep AIME-2025 evaluation results uncontaminated. That rests on the dataset's own single-`test`-split, problem/answer shape (established above) [1][2] and on the screening row's flag naming exactly this risk [5].

### The screening row

The row's own note [5]: "AIME 2025, 30 problems, `test` split; a live eval set." Its flag [5]: "contamination: the AIME 2025 eval set."
