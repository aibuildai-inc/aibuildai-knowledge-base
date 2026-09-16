# AI-MO/aimo-validation-math-level-5

721 level-5 (hardest-tier) math competition problems with integer-only answers, held by AI-MO as an internal validation file for the AIMO Progress Prize competition. Item home: https://huggingface.co/datasets/AI-MO/aimo-validation-math-level-5

**AI-MO/aimo-validation-math-level-5** is a subset that AI-MO's own dataset card says was pulled from `lighteval/MATH`, keeping only Level 5 problems and only those whose boxed final answer is an integer [1]. No paper accompanies it; the dataset card is the only source [1]. A related MATH-derived repository from a different maintainer, `lighteval/MATH-Hard`, identifies Level 5 as the hardest difficulty tier in the underlying MATH benchmark [2]. **This corpus's own screening probed the 721 problems against known MATH split membership and matched 3 of 6 MATH-test level-5 probes against 0 of 6 MATH-train level-5 probes, so despite its "validation" name this file is drawn from the MATH test split, not a held-out training-validation slice [3]. Treat it as eval-only: never mix it into a training split, and decontaminate any MATH-derived training corpus (including this builder's own `NuminaMath-CoT`, whose source breakdown lists 7,478 rows tagged `math` [4]) against these 721 problems before scoring on MATH test.**

**Use it for**: evaluation only - scoring a model's integer-answer accuracy on hard MATH-style problems - not any training format; the three columns (`id`, `problem`, `answer`) are a plain eval-harness shape, not a chat, preference, or SFT-trace format, so no method card's format applies. **Hold out** the whole file from training data built from MATH or from any corpus that draws on it.

**Licence**: not stated. `cardData` carries no `license` field and the README body names none [5]; the upstream `lighteval/MATH` repository this file is drawn from now returns 401 Unauthorized without authentication, so its licence cannot be checked from this card [6].

**Shape**: 721 rows, one config (`default`), one split (`train`), three columns [5][7].

**Hold out**: all 721 rows, from any training corpus touching MATH-derived data (train or test), because the screening probe matched this file to the MATH test split, not train [3]. There is no train/test split internal to this repository to hold out separately - the whole file is the thing to exclude.

**Origin**: built and released by AI-MO; problems and answers are drawn from the human-authored MATH benchmark via `lighteval/MATH`, not model-generated [1]. Hub API at the check date: 866 downloads, 8,342 all-time downloads, 11 likes [8].

**Trained-on-by**: none found. A Hub dataset search for this repository's id returns only the repository itself, and a Hub model search for the same id returns no results [9].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and splits (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 721 |

One config, `default`, three columns (datasets-server `/info` and the card's own `dataset_info`) [7][5]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `problem` | string |
| `answer` | string |

Sizes (datasets-server `/size`) [7]: 102,560 bytes of original Parquet download, 184,565 bytes decoded in memory. No source states sequence-length or token statistics for this file.

The repository's file tree also carries `data/test-00000-of-00001.parquet` (198,394 bytes), but the card's `configs.data_files` maps only `data/train-*` to the served `train` split, so this test file is not wired into any config and `/splits` reports only `train` [5][7][10]. The file's name is consistent with the screening finding that these 721 problems come from MATH's test split: the repository appears to hold an earlier, unpublished extraction alongside the filtered, integer-answer file that is actually served, though its contents were not fetched for this card and its exact relationship to the 721-row file is not stated [3][10].

## Quality

- The only stated construction filter is the card's own: final answers were extracted from LaTeX `\boxed{}` markup, and only problems whose extracted answer is an integer were kept [1]. No source states how many candidate MATH Level 5 problems were dropped by this filter.
- No source states a measured duplicate rate, an annotator-agreement figure, or a stated quality complaint for this file.
- This corpus's own screening is the only stated contamination check: probing 6 known MATH-test level-5 problems and 6 known MATH-train level-5 problems against this file's 721 rows matched 3 of the 6 test probes and 0 of the 6 train probes [3].
- A sibling repository from the same builder, `AI-MO/aimo-validation-amc`, states that its own problems were deliberately drawn from AMC12 2022-2023 (dated after 2021) "to avoid potential overlap with the MATH training set" [11]; no equivalent avoidance statement exists for this Level 5 file, because it is itself built directly from MATH.

## Load it

Single split, no train/test choice to make; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-07-10) [8]:

```python
import datasets

REV = "f6a57be5b8b7818f6abc1b91ada912f474466abc"  # main at the check date
eval_set = datasets.load_dataset("AI-MO/aimo-validation-math-level-5", revision=REV, split="train")  # 721 rows
```

**Trap**: the repository's file tree contains a second Parquet file, `data/test-00000-of-00001.parquet`, that looks like a `test` split but is not registered in `configs.data_files` - `load_dataset` never loads it, and passing an explicit `data_files={"test": "data/test-00000-of-00001.parquet"}` would pull in a file the card does not describe or claim as part of the release; its contents were not fetched for this card, so what it contains beyond its size and name is not stated [5][10].

## Neighbors

All from the same builder, AI-MO, as validation files for the AIMO Progress Prize competition; row counts read live at the check date [12].

- `AI-MO/aimo-validation-math-level-4` - the same construction (subset of `lighteval/MATH`, boxed integer answers only) at the level below: 754 rows [12][13].
- `AI-MO/aimo-validation-amc` - 83 AMC12 2022-2023 problems, deliberately drawn from after 2021 to avoid MATH-train overlap; a different provenance path than this file [11][12].
- `AI-MO/aimo-validation-aime` - 90 AIME 2022-2024 problems, same after-2021 avoidance rationale as the AMC file [14][12].
- `AI-MO/NuminaMath-CoT` - this builder's ~860k-row training corpus; its source breakdown table lists 7,478 rows tagged `math`, which is the training-side counterpart this file's 721 test-split problems must be checked against before any run that trains on `NuminaMath-CoT` and evaluates on this file [4].

None of these is a straight duplicate of this file; each serves a different competition tier or a different role (eval file versus training corpus).

## A row

One config and one served split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [15]:

```json
{
  "id": 0,
  "problem": "Find the sum of all integers that satisfy these conditions: \\[\n|x|+1>7\\text{ and }|x+1|\\le7.\n\\]",
  "answer": "-15"
}
```

## Where it came from

Built and released by AI-MO. The card states the construction in one line: a subset of Level 5 problems taken from `lighteval/MATH`, keeping only rows whose boxed final answer could be extracted as an integer [1]. No further collection detail - annotator process, filtering counts, or date of extraction beyond the repository's 2024-07-10 last-modified timestamp - is stated anywhere in the card or the Hub API record [1][8].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] AI-MO/aimo-validation-math-level-5 dataset card (README). https://huggingface.co/datasets/AI-MO/aimo-validation-math-level-5/raw/main/README.md - the one-line construction description, `dataset_info` metadata. Fetched 2026-08-12.

[2] lighteval/MATH-Hard dataset card, read through the Hub dataset-search API's description field. https://huggingface.co/api/datasets?search=lighteval%2FMATH-Hard - states "For MATH-Hard, only the hardest questions were kept (Level 5)". Fetched 2026-08-12.

[3] The corpus screening row for `AI-MO/aimo-validation-math-level-5`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-12.

[4] AI-MO/NuminaMath-CoT dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/raw/main/README.md - source breakdown table listing `math: 7478`, licence, citation. Fetched 2026-08-12.

[5] Hugging Face Hub API record for AI-MO/aimo-validation-math-level-5. https://huggingface.co/api/datasets/AI-MO/aimo-validation-math-level-5?full=true - `cardData` (no `license` field), `configs.data_files` mapping only `data/train-*`, `sha`, `downloads`, `likes`, `lastModified`, `siblings`. Fetched 2026-08-12.

[6] Direct fetch of `lighteval/MATH`'s README and the Hugging Face Hub API record for it both returned 401 Unauthorized ("Invalid username or password"), and the dataset's own datasets-server `/info` endpoint reports it "does not exist, or is not accessible without authentication (private or gated)". https://huggingface.co/datasets/lighteval/MATH/raw/main/README.md ; https://huggingface.co/api/datasets/lighteval/MATH ; https://datasets-server.huggingface.co/info?dataset=lighteval%2FMATH Fetched 2026-08-12.

[7] datasets-server size and info endpoints. https://datasets-server.huggingface.co/size?dataset=AI-MO%2Faimo-validation-math-level-5 ; https://datasets-server.huggingface.co/info?dataset=AI-MO%2Faimo-validation-math-level-5 ; https://datasets-server.huggingface.co/splits?dataset=AI-MO%2Faimo-validation-math-level-5 Fetched 2026-08-12.

[8] Hugging Face Hub API record for AI-MO/aimo-validation-math-level-5, plus its `downloadsAllTime`-expanded variant. https://huggingface.co/api/datasets/AI-MO/aimo-validation-math-level-5?full=true ; https://huggingface.co/api/datasets/AI-MO/aimo-validation-math-level-5?expand[]=downloadsAllTime Fetched 2026-08-12.

[9] Hugging Face Hub dataset-search and model-search APIs, queried for the string `aimo-validation-math-level-5`. https://huggingface.co/api/datasets?search=aimo-validation-math-level-5&limit=100 ; https://huggingface.co/api/models?search=aimo-validation-math-level-5&limit=100 - both return no repository other than this dataset itself. Fetched 2026-08-12.

[10] Hugging Face Hub repository tree for AI-MO/aimo-validation-math-level-5, recursive. https://huggingface.co/api/datasets/AI-MO/aimo-validation-math-level-5/tree/main?recursive=true - lists `data/test-00000-of-00001.parquet` (198,394 bytes) alongside the served `data/train-00000-of-00001.parquet` (102,560 bytes). Fetched 2026-08-12.

[11] AI-MO/aimo-validation-amc dataset card (README). https://huggingface.co/datasets/AI-MO/aimo-validation-amc/raw/main/README.md - "Using data after 2021 is to avoid potential overlap with the MATH training set." Fetched 2026-08-12.

[12] datasets-server size endpoint, one call per neighbor, for every neighbor row count above: `AI-MO/aimo-validation-math-level-4`, `AI-MO/aimo-validation-amc`, `AI-MO/aimo-validation-aime`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-12.

[13] AI-MO/aimo-validation-math-level-4 dataset card (README). https://huggingface.co/datasets/AI-MO/aimo-validation-math-level-4/raw/main/README.md - same one-line construction description as this file, at Level 4. Fetched 2026-08-12.

[14] AI-MO/aimo-validation-aime dataset card (README). https://huggingface.co/datasets/AI-MO/aimo-validation-aime/raw/main/README.md - "Using data after 2021 is to avoid potential overlap with the MATH training set." Fetched 2026-08-12.

[15] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=AI-MO%2Faimo-validation-math-level-5&config=default&split=train Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Flagged for contamination, eval-only. The dataset's own construction note names it as a MATH Level 5 subset [1], and this corpus's screening probe places that subset inside MATH's test split rather than a separate validation slice - 3 of 6 known MATH-test level-5 probes matched, 0 of 6 known MATH-train level-5 probes matched [3]. That result, plus the unwired `data/test-00000-of-00001.parquet` file sitting in the repository tree with a name that matches the finding, is why this card treats the file as benchmark data to hold out rather than as ordinary training-validation data [10].

### The screening row

The row's own note [3]: "Drawn from the MATH TEST split: 3 of 6 MATH test level-5 probes matched, 0 of 6 MATH train level-5 probes matched. Benchmark data despite the harmless name." Its flag: "contamination: drawn from the MATH TEST split (3/6 test probes hit, 0/6 train probes hit)."
