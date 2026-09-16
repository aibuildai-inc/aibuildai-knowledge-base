# ChilleD/SVAMP

1,000 elementary-school math word problems, each with a narrative body, a question, a solving equation and a numeric answer, packaged as a 700/300 train/test parquet mirror of the SVAMP challenge set.

**ChilleD/SVAMP** mirrors the SVAMP challenge set introduced in "Are NLP Models really able to Solve Simple Math Word Problems?" [1], which the authors built by applying targeted variations to examples drawn from the MAWPS and ASDiv-A benchmarks so that surface-level heuristics no longer suffice to answer them [1]. The task shape is single-step-to-multi-step arithmetic word-problem solving: given a body and a question, produce the equation and the numeric answer. **In the paper's own released code, the full 1,000-item set is described as "Complete challenge set to be used for evaluation" alongside a separate MAWPS+ASDiv-A training pool [2]; this repository's own 700/300 split is not that pairing, and no source here states why the 700/300 split was drawn, so treat this repository's split as a repackaging, not the paper's training protocol. Decontaminate against SVAMP before reporting any scored math-reasoning eval, since the same 1,000 items are a widely used benchmark.** It lives at https://huggingface.co/datasets/ChilleD/SVAMP (the row's lowercase `chilleD/SVAMP` redirects at the Hub API [3], but 404s at the datasets-server `/info` and `/size` endpoints, which do not follow that redirect [4]).

**Use it for**: SFT-style supervised fine-tuning on math word-problem solving (body+question in, equation/answer out), or as a held-out evaluation set for a math-reasoning eval harness - not both on the same model. The rows are plain single-turn text fields (`Body`, `Question`, `Equation`, `Answer`, plus a pre-concatenated `question_concat`), not a chat or preference format; map them into the SFT method card's prompt/completion shape by treating `question_concat` (or `Body`+`Question`) as the prompt and `Equation`/`Answer` as the target.

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`) [3]; the GitHub source repo carries the same MIT licence [2]. No further catch found - ungated, public.

**Shape**: 1,000 rows, one config (`default`), split `train` 700 / `test` 300, seven string columns [3][5][6].

**Hold out**: nothing to hold out inside this repository beyond its own declared `test` split (300 rows) - no source here states duplicate or leakage risk within these 1,000 rows. But the whole 1,000-row SVAMP set is a public math-reasoning benchmark used to score other models [1][2], so any external eval run against SVAMP should exclude rows matching these 1,000 `ID`s regardless of which of this repo's splits they fall in.

**Origin**: released on the Hub by user ChilleD as a parquet/JSON mirror of Arkil Patel, Satwik Bhattamishra and Navin Goyal's SVAMP release; the problems and equations are human-authored (adapted by the paper's authors from MAWPS/ASDiv-A source problems), not model-generated [1][2]. Hub API at the check date: `downloads` 21,007 (30-day), `downloadsAllTime` 169,034, `likes` 23 [3].

**Trained-on-by**: none found. The paper's own released code treats the full 1,000-item SVAMP set only as an evaluation challenge set paired with a separate MAWPS+ASDiv-A training pool, not as fine-tuning data in its own right [2]; no source fetched here names a model fine-tuned specifically on this repository's 700-row `train` split.

**Introduced by**: [1] (Patel, Bhattamishra and Goyal, NAACL 2021).

## Shape

Rows and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 700 |
| `test` | 300 |
| total | 1,000 |

One config, `default`, with seven columns (datasets-server `/info`, matching the repo's own `README.md` front matter) [3][6]:

| column | dtype |
| --- | --- |
| `ID` | string |
| `Body` | string |
| `Question` | string |
| `Equation` | string |
| `Answer` | string |
| `Type` | string |
| `question_concat` | string |

Byte sizes (datasets-server `/size`) [5]: 166,226 bytes of original parquet download, 390,461 bytes decoded in memory - 273,253 bytes / 700 rows for `train`, 117,208 bytes / 300 rows for `test`. No source states sequence-length or token counts for either split.

## Quality

- No source fetched here states a measured contamination, duplication, or annotator-agreement rate for this repository's 1,000 rows.
- The origin paper's own quality claim is structural, not a rate: the authors report that existing MWP solvers rely on shallow heuristics to score well on the prior benchmark datasets, built SVAMP by varying MAWPS/ASDiv-A examples to defeat those heuristics, and found state-of-the-art solvers of the time scored substantially lower on SVAMP than on the source benchmarks [1].
- The repository's `README.md` carries only YAML front matter (675 bytes) - no prose quality statement, caveat, or intended-use note beyond the `dataset_info` schema and licence tag [3].
- All rows are served directly (no external fetch needed to complete a row): `Equation` and `Answer` sit in the row itself, so no join step or missing-key problem applies here.

## Load it

```python
import datasets

REV = "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"  # main at the check date
train = datasets.load_dataset("ChilleD/SVAMP", revision=REV, split="train")  # 700 rows
test = datasets.load_dataset("ChilleD/SVAMP", revision=REV, split="test")    # 300 rows
```

**Trap**: the repo id is `ChilleD/SVAMP` (capital C, capital D) - the all-lowercase `chilleD/SVAMP` id returns HTTP 307 from the Hub API and 404 from the datasets-server `/info` and `/size` endpoints, which do not follow the redirect [3][4]. Also, `Equation` is a string like `"( 290.0 / 2.0 )"`, not a pre-evaluated float - a training pipeline that wants a numeric target must parse it or use the separate `Answer` field.

## Neighbors

Every row count below was read live at the check date [7]. This corpus prefers this repository (`ChilleD/SVAMP`) for the plain 700/300 train/test parquet shape; reach for a neighbor only when its format or protocol is what the task needs.

- `arkilpatel/SVAMP` on GitHub - the origin release. Its own README states the 1,000-item set is a complete evaluation challenge set, paired with a training pool built from the full MAWPS (1,921 problems) and ASDiv-A (1,217 problems); it does not split the 1,000 SVAMP items into a 700/300 train/test pair itself [2].
- `cq01/mawps-asdiv-a_svamp` - reproduces the paper's own training protocol on the Hub: 3,138 training rows (from MAWPS+ASDiv-A) and a 1,000-row `validation` split holding the full SVAMP set as evaluation-only, with a richer schema (`Numbers`, `group_nums`, `Variation Type`) than this repository's [7][8].
- `Dahoas/svamp` - the same 700/300 row counts as this repository, but reformatted into `question`/`answer`/`prompt`/`response` columns; its card gives no explanation of how the 700/300 split was drawn, so it is treated here as a parallel repackaging rather than a verified duplicate of this repository's exact rows [7][8].
- `MU-NLPC/Calc-svamp` - a calculator-augmented derivative built from the same GitHub source, holding the full 1,000 items only as a `test` split (no train rows) plus an added `chain` column that spells the solution as a step-by-step calculator trace; its card states it is derived from `arkilpatel/SVAMP` [7][9].
- `tongyx361/svamp` - the same 1,000-row total in a single split, six columns; a further Hub re-export of the same source problems [7].

## A row

The repository serves one config (`default`) with the same seven-column schema in both splits, so one row from each split covers the shapes seen. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [6]:

```json
{
  "ID": "chal-777",
  "Body": "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups",
  "Question": "How big is each group of bananas?",
  "Equation": "( 290.0 / 2.0 )",
  "Answer": "145",
  "Type": "Common-Division",
  "question_concat": "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups How big is each group of bananas?"
}
```

From `config="default"`, `split="test"`, `row_idx=0` [6]:

```json
{
  "ID": "chal-736",
  "Body": "Winter is almost here and most animals are migrating to warmer countries. There are 41 bird families living near the mountain. If 35 bird families flew away to asia and 62 bird families flew away to africa",
  "Question": "How many more bird families flew away to africa than those that flew away to asia?",
  "Equation": "( 62.0 - 35.0 )",
  "Answer": "27",
  "Type": "Subtraction",
  "question_concat": "Winter is almost here and most animals are migrating to warmer countries. There are 41 bird families living near the mountain. If 35 bird families flew away to asia and 62 bird families flew away to africa How many more bird families flew away to africa than those that flew away to asia?"
}
```

Both splits share the identical column set and dtypes; the `test` row above shows the same field shapes as `train`.

## Where it came from

Arkil Patel, Satwik Bhattamishra and Navin Goyal built SVAMP by sampling problems from the existing MAWPS and ASDiv-A math word-problem benchmarks and applying carefully chosen variations - adding structural alterations and reasoning traps - so that the resulting problems could not be solved by the shallow heuristics the authors showed existing benchmarks were vulnerable to [1]. The problems and their solving equations are human-authored, not model-generated [1]. The authors released the 1,000-item set and accompanying code on GitHub, describing it there as a complete challenge set for evaluation, distinct from the MAWPS+ASDiv-A pool they use for training their own baseline solvers [2]. The Hub user ChilleD repackaged these 1,000 items into a `default` config with a 700/300 train/test split, adding `test.json`/`train.json` and Parquet copies to the repository tree; no statement in this repository's README explains how that 700/300 split was drawn from the original 1,000 [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Patel, Bhattamishra and Goyal, "Are NLP Models really able to Solve Simple Math Word Problems?", NAACL 2021. https://arxiv.org/abs/2103.07191 - the origin paper; current title read from the live abs page. Fetched 2026-08-11.

[2] arkilpatel/SVAMP GitHub repository README. https://github.com/arkilpatel/SVAMP - complete-challenge-set description, MAWPS/ASDiv-A training-pool sizes, MIT licence. Fetched 2026-08-11.

[3] Hugging Face Hub API record for ChilleD/SVAMP. https://huggingface.co/api/datasets/ChilleD/SVAMP?full=true - licence, gate, sha, columns/splits from `cardData.dataset_info`, siblings list; `downloads`/`downloadsAllTime`/`likes` read through the same endpoint's `expand[]=downloadsAllTime` variant; the lowercase-id 307 redirect observed via `curl -I` on `https://huggingface.co/api/datasets/chilleD/SVAMP?full=true`. Fetched 2026-08-11.

[4] datasets-server size/info endpoints queried with the lowercase id, both returning HTTP 404. https://datasets-server.huggingface.co/size?dataset=chilleD%2FSVAMP and https://datasets-server.huggingface.co/info?dataset=chilleD%2FSVAMP Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=ChilleD%2FSVAMP Fetched 2026-08-11.

[6] datasets-server info and first-rows endpoints. https://datasets-server.huggingface.co/info?dataset=ChilleD%2FSVAMP and https://datasets-server.huggingface.co/first-rows?dataset=ChilleD%2FSVAMP&config=default&split=train (and `split=test`). Fetched 2026-08-11.

[7] datasets-server size endpoint, one call per neighbor: `cq01/mawps-asdiv-a_svamp`, `Dahoas/svamp`, `MU-NLPC/Calc-svamp`, `tongyx361/svamp`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[8] Neighbor dataset card front matter (READMEs), read for column names and split sizes: `cq01/mawps-asdiv-a_svamp` and `Dahoas/svamp`. https://huggingface.co/datasets/<id>/raw/main/README.md Fetched 2026-08-11.

[9] MU-NLPC/Calc-svamp dataset card (README). https://huggingface.co/datasets/MU-NLPC/Calc-svamp/raw/main/README.md - states the dataset is derived from arkilpatel/SVAMP, describes the `chain` column and the test-only split shape. Fetched 2026-08-11.

[10] The corpus screening row for `chilleD/SVAMP`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as an SFT-shaped word-problem dataset, but only with the decontamination caveat already established above: the full 1,000-row SVAMP set is a public math-reasoning benchmark [1][2], so any external evaluation run against SVAMP should exclude these rows. The screening row's own note gives the shape (1,000 rows, 700/300 split) and flags the id-casing 404 that this card resolves via the Hub redirect [10].

### The screening row

The row's own note [10]: "SVAMP, 1,000 elementary maths word problems with a body, a question, the solving equation and the answer, split 700 train and 300 test; the card carries only front matter, and the live repo is ChilleD/SVAMP, which is why this capitalisation returned 404." The row carries no separate `flag` field.
