# HuggingFaceH4/aime_2024

30 competition math problems from the 2024 AIME I and II contests, each with a worked solution and a short numeric answer string - a benchmark-sized eval set, not a training corpus.

**HuggingFaceH4/aime_2024** is HuggingFaceH4's re-release of the 2024 subset of `AI-MO/aimo-validation-aime`, a 90-problem internal validation set AI-MO built from the 2022-2024 American Invitational Mathematics Examinations while competing in the AIMO progress-prize competition [1]. It lives at https://huggingface.co/datasets/HuggingFaceH4/aime_2024 [2]. Its own dataset card gives no further introduction beyond naming that source and the two 2024 contests the 30 problems come from [2]. **This is competition-benchmark data, not training data: all 30 rows are held-out AIME 2024 problems, and the only split is misleadingly named `train` - a screening flag on this repository names exactly that as a contamination risk, because a pipeline that treats every `train` split as trainable rows would ingest a public eval set into a training run** [3].

**Use it for**: held-out evaluation of mathematical reasoning (e.g. AIME-style pass@1 or majority-vote scoring), never as SFT or preference rows - no training method card applies to this shape. `problem` is the prompt and `answer` is the exact-match scoring target, kept as a zero-padded string (e.g. `"025"`) rather than an integer, so a scorer must compare against that literal string or explicitly strip the padding [2][4].

**Licence**: not stated on this repository - `cardData` carries no `license` key and the repo has no `license:` tag, and the ungated repository (`"gated": false`) does not carry a licence catch beyond that absence [5]. The upstream `AI-MO/aimo-validation-aime` that this repo's card names as its source is tagged `license:apache-2.0` [1], but no source says that grant extends to this repository.

**Shape**: 30 rows, one config (`default`), one split (`train`), six columns [6][7].

**Hold out**: all 30 rows. There is no train/test split to select from - the entire repository is AIME 2024, so any model that has already seen these problems (directly or through a training corpus that scraped them) cannot be scored cleanly against it; decontaminate a training set against this repository's `problem`/`answer` pairs before using it for evaluation.

**Origin**: released by HuggingFaceH4; problems and reference solutions are human-authored (the AIME contest and the AoPS wiki solvers), not model-generated [1][2]. Hub API at the check date: 59,099 downloads, 693,600 all-time downloads, 63 likes [5][8].

**Trained-on-by**: none found - no source states that a model or training recipe trained on this repository's rows; as a public eval benchmark, the sourced concern is the reverse (leakage into training data), which the flag above already covers [3].

**Introduced by**: no paper - the dataset card [2], drawing on the AI-MO validation-set card [1] for its own upstream description.

## Shape

Rows and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 30 |

One config, `default`, with six columns (datasets-server `/info`, matching the shortlist row's declared columns) [7]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `problem` | string |
| `solution` | string |
| `answer` | string |
| `url` | string |
| `year` | string |

Sizes (datasets-server `/size`) [6]: 81,670 bytes as originally-served parquet, 139,590 bytes decoded in memory. No source states sequence-length or token statistics for this repository.

## Quality

- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this specific repository.
- The `answer` column stores AIME's three-digit convention as a zero-padded string (e.g. `"025"`, `"033"`), confirmed by reading all 30 served rows: seven of the thirty carry a leading zero [4].
- `id` runs 60-89 across the 30 rows, and `url` points at the AoPS wiki page for each problem (`2024_AIME_I_Problems/Problem_N` or `2024_AIME_II_Problems/Problem_N`) [4]. Fifteen rows carry an `AIME_I` url and fifteen carry `AIME_II`, read across all 30 served rows.
- The `solution` field for each row concatenates one or more full worked solutions copied from the AoPS wiki's community solution threads, including solver sign-offs (e.g. "-Failure.net") still present in the text, read across all 30 served rows [4].

## Load it

Only one split is served, and the whole repository is the eval set - there is nothing to hold out beyond using all of it:

```python
import datasets

REV = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"  # main at the check date
ds = datasets.load_dataset("HuggingFaceH4/aime_2024", revision=REV, split="train")  # 30 rows - eval only
```

**Trap**: the split is named `train`. `load_dataset(..., split="train")` succeeds and returns rows shaped like a training split, but every row is a held-out AIME 2024 problem - the screening flag on this repository exists specifically because that name invites a training pipeline to treat it as trainable data [3]. Downloads and likes above are read live and are not covered by the revision pin, since the Hub API's download/like counters take no revision parameter [5].

## Neighbors

- `Maxwell-Jia/AIME_2024` - the same 30 AIME 2024 problems in a differently-shaped `default`/`train` schema (`ID`, `Problem`, `Solution`, `Answer`, with `Answer` as int64 rather than a zero-padded string), 30 rows, MIT-licensed [9][10]. Fetching row 0 of each repository confirms the same underlying problem set: `Maxwell-Jia/AIME_2024` row 0 (`ID` `2024-II-4`, `Answer` 33) matches, problem statement for problem statement and answer for answer, `HuggingFaceH4/aime_2024` row `id` 84 (url `2024_AIME_II_Problems/Problem_4`, `answer` `"033"`) [4][11]. At the check date, `HuggingFaceH4/aime_2024` carries more downloads than `Maxwell-Jia/AIME_2024` (59,099 vs 39,801 in the live Hub API) [5][12]; prefer whichever schema's column names match the harness already in use, since the underlying problem set is identical.
- `AI-MO/aimo-validation-aime` - the 90-problem superset this repository's card names as its source, spanning AIME 2022-2024, Apache-2.0 licensed, with five columns (`id`, `problem`, `solution`, `answer`, `url` - no `year`) versus this repository's six [1]. Reading rows 60-61 of that repository (offset 60, length 2) returns the same `id`, `answer`, and `url` values as `id` 60-61 in this repository, confirming this repository's rows are a 2024-only slice of that superset's content rather than an independent re-collection, even though this repository adds a `year` column the superset does not carry [1][13].

## A row

One config, one split, so one shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [4], with the solution field truncated:

```json
{
  "id": 60,
  "problem": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.",
  "solution": "$\\frac{9}{s} + t = 4$ in hours and $\\frac{9}{s+2} + t = 2.4$ in hours.\nSubtracting the second equation from the first, we get, \n$\\frac{9}{s} - \\frac{9}{s+2} = 1.6$\n[...] Lastly, $s + \\frac{1}{2} = 3$ kilometers per hour, so\n$\\frac{9}{3} + 0.4 = 3.4$ hours, or $\\framebox{204}$ minutes\n-Failure.net\n[... a second worked solution follows, signed -sepehr2010]",
  "answer": "204",
  "url": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_1",
  "year": "2024"
}
```

## Where it came from

HuggingFaceH4 released this repository as a 30-problem, 2024-only re-serving of `AI-MO/aimo-validation-aime`, which that repository's own card says AI-MO built by extracting all 90 problems from AIME 2022, 2023, and 2024 directly off the AoPS wiki's "AIME Problems and Solutions" page, for use as AI-MO's internal validation set while competing in the AIMO progress-prize competition; that card states 2021-and-later contests were chosen specifically to avoid overlap with the MATH training set [1]. This repository's own card adds no further collection detail beyond naming that source and the two 2024 contests [2].

## Sources

Every source below was fetched on the check date, 2026-08-11; that date covers every number, quote, and corpus row on this card. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the download and like counts are read from the live, unpinned Hub API, which takes no revision parameter.

[1] AI-MO/aimo-validation-aime dataset card (README). https://huggingface.co/datasets/AI-MO/aimo-validation-aime/raw/main/README.md - source, collection method, licence tag. Fetched 2026-08-11.

[2] HuggingFaceH4/aime_2024 dataset card (README). https://huggingface.co/datasets/HuggingFaceH4/aime_2024/raw/main/README.md - the item's own introduction and stated source. Fetched 2026-08-11.

[3] The corpus screening row for `HuggingFaceH4/aime_2024`, supplied with this card's request - its `note` and `flag`, read back in their own words in the appendix. Checked 2026-08-11.

[4] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=HuggingFaceH4%2Faime_2024&config=default&split=train - all 30 served rows read from this response. Fetched 2026-08-11.

[5] Hugging Face Hub API record for HuggingFaceH4/aime_2024. https://huggingface.co/api/datasets/HuggingFaceH4/aime_2024?full=true - `sha`, licence absence, gate status, `downloads`, `likes`, `lastModified`. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=HuggingFaceH4%2Faime_2024 Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=HuggingFaceH4%2Faime_2024 Fetched 2026-08-11.

[8] Hugging Face Hub API record for HuggingFaceH4/aime_2024, `downloadsAllTime` variant. https://huggingface.co/api/datasets/HuggingFaceH4/aime_2024?expand%5B%5D=downloadsAllTime Fetched 2026-08-11.

[9] Maxwell-Jia/AIME_2024 dataset card (README). https://huggingface.co/datasets/Maxwell-Jia/AIME_2024/raw/main/README.md - schema, licence, size. Fetched 2026-08-11.

[10] Hugging Face Hub API record for Maxwell-Jia/AIME_2024. https://huggingface.co/api/datasets/Maxwell-Jia/AIME_2024?full=true - `license:mit` tag, `downloads`. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Maxwell-Jia%2FAIME_2024&config=default&split=train - row 0 used for the cross-repository match. Fetched 2026-08-11.

[12] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Maxwell-Jia%2FAIME_2024 Fetched 2026-08-11.

[13] datasets-server rows endpoint, offset 60, length 2. https://datasets-server.huggingface.co/rows?dataset=AI-MO%2Faimo-validation-aime&config=default&split=train&offset=60&length=2 - used to confirm the id-60/61 overlap with this repository. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable only as a held-out evaluation set, never as training data. The card already establishes both facts the verdict rests on: this repository is the complete 2024 AIME I and II problem set with no train/test division (Shape, above), and the screening row's own flag names the risk that its single split is named `train` [3].

### The screening row

The row's own note [3]: "AIME 2024 I and II, 30 records (the same 30 problems as Maxwell-Jia/AIME_2024, which it out-downloads), split named `train`." Its flag, classed as contamination: "contamination: AIME 2024 with the split named train."
