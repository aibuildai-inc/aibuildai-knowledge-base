# math-ai/aime24

30 competition math problems - the complete 2024 AIME I and AIME II contests - each paired with a final boxed numeric answer; a small, fixed evaluation benchmark, not a training corpus.

**math-ai/aime24** packages the 2024 American Invitational Mathematics Examination as a single `test` split of 30 rows, one row per problem from AIME I and AIME II, with columns `id`, `problem`, `solution`, and `url` [1]; reading all 30 served rows shows `solution` holding the boxed final answer and `url` pointing to the problem's Art of Problem Solving wiki page [6]. The README states no origin paper, only a citation block naming Zhang, Yifan and the Math-AI team [1]. It lives at https://huggingface.co/datasets/math-ai/aime24 . **This is a live evaluation set: the corpus screening for this card marks it as an eval set carrying a contamination flag, so training on any of these 30 problems risks contaminating any AIME24-based evaluation of the resulting model [2].**

**Use it for**: nothing in training - this is an eval-only benchmark. Score model completions against the boxed final answer in `solution` for each `problem`; it does not map to any training-shape method card (SFT, preference, or reasoning-trace), and none of the 30 rows should enter a training set for a model that will later be scored on AIME24 [1][2].

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [3].

**Shape**: 30 rows, one config (`default`), one split (`test`), four columns (`id`, `problem`, `solution`, `url`) [3][4].

**Hold out**: all 30 rows, unconditionally. This dataset is itself the AIME 2024 evaluation benchmark, so its rows must never appear in any training set that will later be evaluated on AIME24 - the corpus screening flag names exactly this risk [2].

**Origin**: problems are the official, human-authored 2024 AIME I and AIME II contest problems; `solution` is the final numeric answer, not a worked solution [6]. Hub API at the check date: `downloads` 12,119, `downloadsAllTime` 78,621, `likes` 19 [3].

**Trained-on-by**: none found. No source in this card documents a model or recipe deliberately training on this repository; training on it would itself be the contamination the screening flag warns about [2].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and split, from the datasets-server size endpoint [4]:

| split | rows |
| --- | --- |
| `test` | 30 |

One config, `default`, with four columns, from the datasets-server info endpoint [5]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `problem` | string |
| `solution` | string |
| `url` | string |

`id` runs 60-89, a single contiguous block: reading all 30 served rows, 15 carry an `url` pointing to `2024_AIME_I_Problems` and 15 to `2024_AIME_II_Problems`, so the split is the complete AIME I and AIME II problem sets with no held-back subset [6]. No source states token or sequence-length statistics for this release.

## Quality

- Reading all 30 served rows, every `solution` value is exactly a `\boxed{...}` final-answer string with no worked steps - lengths run 10-11 characters across the set, confirming this is a bare answer key, not a solution corpus [6].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset; competition provenance (official AIME problems) is what the `url` field documents, not an internal quality check [1][6].
- The README states no other quality caveat beyond the Apache-2.0 licence badge and the citation block [1].

## Load it

Pin the revision this card's numbers were read at (the shortlist's commit, matching the Hub API's `sha` for `main` at the check date; last modified 2026-02-20) [3]:

```python
import datasets

REV = "83a7f387baaa524a8bda0022eac0541582297103"  # main at the check date
test = datasets.load_dataset("math-ai/aime24", revision=REV, split="test")  # 30 rows - eval only, never train
```

**Trap**: there is only one split (`test`) and one config (`default`), so no accidental split confusion is possible here - the trap is upstream of loading: any pipeline that folds this repository's rows into a training set (directly, or indirectly through a mirror or re-export) has contaminated the AIME24 eval for every model trained on that set [2].

## Neighbors

- `math-ai/aime25` - the same org's next-year AIME benchmark: 30 rows, one `test` split, three columns; a separate exam year, not an overlapping release of this one [7].
- `HuggingFaceH4/aime_2024` - a differently-shaped re-release of the same 2024 AIME problems: same 30 rows and `id` values (row 0 is `id` 60, matching this repository), but with six columns instead of four, a full worked `solution` (1,313 characters for row 0, versus this repository's 11-character boxed answer) plus a separate short `answer` field, a split named `train` rather than `test`, and a stated upstream source of `AI-MO/aimo-validation-aime`, a 90-problem AIME 2022-2024 pool [8][9]. Prefer this repository (`math-ai/aime24`) when only the boxed answer is needed for scoring; prefer `HuggingFaceH4/aime_2024` when a worked solution is needed, e.g. for solution-based reward modeling.
- `di-zhang-fdu/AIME_1983_2024` - a much larger 933-row pool spanning AIME 1983-2024 with a different schema (`ID`, `Year`, `Problem Number`, `Question`, `Answer`, `Part`); it contains 2024 among its years, so any pipeline using it must exclude 2024 rows to avoid the same AIME24 contamination this card flags [10].
- `Maxwell-Jia/AIME_2024` is a further, more-downloaded re-release of the same 2024 contest under yet another schema; not fetched for this card, so its exact column layout is not stated here [11].

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [6]:

```json
{
  "id": 60,
  "problem": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.",
  "solution": "\\boxed{204}",
  "url": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_1"
}
```

## Where it came from

The repository is published under the `math-ai` Hugging Face org; its only stated authorship is the citation block naming Zhang, Yifan and the Math-AI team, with no linked origin paper [1]. The problems themselves are the official 2024 AIME I and AIME II contest problems, each `url` pointing to that problem's page on the Art of Problem Solving wiki [1][6]; the `solution` field is the final numeric answer these official contests define, not a derived or generated solution.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] math-ai/aime24 dataset card (README). https://huggingface.co/datasets/math-ai/aime24/raw/main/README.md - title, citation, feature/split declaration, licence badge. Fetched 2026-08-11.

[2] The corpus screening row for `math-ai/aime24`, supplied with this card's request - its `flag` ("contamination: the AIME 2024 eval set") and `note`, read back in the appendix. Checked 2026-08-11.

[3] Hugging Face Hub API record for math-ai/aime24. https://huggingface.co/api/datasets/math-ai/aime24?full=true and https://huggingface.co/api/datasets/math-ai/aime24?expand[]=downloadsAllTime - licence, gate, `sha`, `downloads`, `downloadsAllTime`, `likes`, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=math-ai%2Faime24 Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=math-ai%2Faime24 Fetched 2026-08-11.

[6] datasets-server first-rows endpoint, all 30 served rows. https://datasets-server.huggingface.co/first-rows?dataset=math-ai%2Faime24&config=default&split=test Fetched 2026-08-11.

[7] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=math-ai%2Faime25 Fetched 2026-08-11.

[8] HuggingFaceH4/aime_2024 dataset card (README), size, info, and first-rows endpoints. https://huggingface.co/datasets/HuggingFaceH4/aime_2024/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=HuggingFaceH4%2Faime_2024 , https://datasets-server.huggingface.co/info?dataset=HuggingFaceH4%2Faime_2024 , https://datasets-server.huggingface.co/first-rows?dataset=HuggingFaceH4%2Faime_2024&config=default&split=train Fetched 2026-08-11.

[9] AI-MO/aimo-validation-aime, named in [8] as the stated upstream source of HuggingFaceH4/aime_2024; not independently fetched for this card. https://huggingface.co/datasets/AI-MO/aimo-validation-aime

[10] datasets-server size and info endpoints for di-zhang-fdu/AIME_1983_2024. https://datasets-server.huggingface.co/size?dataset=di-zhang-fdu%2FAIME_1983_2024 , https://datasets-server.huggingface.co/info?dataset=di-zhang-fdu%2FAIME_1983_2024 Fetched 2026-08-11.

[11] Hugging Face Hub dataset-search results for "aime24", used to identify neighbor repositories and their download counts, including Maxwell-Jia/AIME_2024 (39,801 downloads, 85 likes at the check date); Maxwell-Jia/AIME_2024 itself was not fetched for this card. https://huggingface.co/api/datasets?search=aime24&limit=50 Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Flagged for contamination, not usable as training data: this repository IS the AIME 2024 evaluation benchmark - all 30 rows must be held out of any training set that will later be scored on AIME24. This rests on the dataset's own content (30 problems from the official 2024 AIME I and II contests with boxed final answers, established above) and on the screening row's flag and note [1][2][6].

### The screening row

The row's own note [2]: "AIME 2024, 30 problems, `test` split; a live eval set." Its flag [2]: "contamination: the AIME 2024 eval set."
