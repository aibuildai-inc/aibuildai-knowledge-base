# Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn

180 GPT-4o-mini solution attempts - two per problem - on the 90 AIME 2022-2024 competition problems, each attempt scored right or wrong against the reference answer.

**Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn** takes the 90 problems of `AI-MO/aimo-validation-aime` - AIME 22-24 problems extracted from the AoPS wiki, held out by AI-MO as an internal validation set for the AIMO progress-prize competition specifically to avoid contaminating the MATH training set [1] - and adds two GPT-4o-mini-generated solution attempts per problem, each split into reasoning steps and marked correct or incorrect against the original AoPS answer. The Hub dataset card for this repository is metadata only: it carries the `dataset_info` YAML block (features and split sizes) and no prose description, so no source states who ran the generation and scoring pipeline or when, only that the columns and row counts are as shown [2]. The repository has no `license` field and no citation. It lives at https://huggingface.co/datasets/Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn . **These are model outputs scored on an eval set: `AI-MO/aimo-validation-aime`'s 90 AIME problems are a standard held-out benchmark for math reasoning, so training on this repository's `response` text or `response_correct` labels risks contaminating any run that later scores against AIME 2022-2024.**

**Use it for**: not preference pairs - grouping by problem `id`, 61 of 90 problems have both GPT-4o-mini attempts wrong and 22 have both right, so only 7 of 90 problems actually pair a correct attempt against an incorrect one for the same prompt [3]. The two shapes it does support are reasoning-trace SFT (train on the `response` text of the 51 correct attempts) or outcome-verifier/reward-model training (`response` as text, `response_correct` as the label) - either way, first decontaminate against AIME 2022-2024 as described above; map to the reasoning-trace SFT method card or the reward-model method card as appropriate.

**Licence**: not stated - `cardData` carries no `license` key, no `license:` tag appears on the repository, and the README's YAML block has no license field [2]. Ungated, public repository [4].

**Shape**: 180 rows, one split (`train`), one config (`default`), 10 columns [4][5].

**Hold out**: all 180 rows, because every one answers one of the 90 `AI-MO/aimo-validation-aime` problems (AIME 2022-2024) that this repository's own `url` column points back to on AoPS [1][3]. The screening row's note and flag both name this as contamination risk against an AIME eval set [6].

**Origin**: repository owned by Hub user Asap7772; the `response` text is a GPT-4o-mini generation and `response_correct` is a right/wrong label against the AoPS reference answer, per the row-level shape read from the served parquet - no source states who ran the generation or scoring pipeline [2][3]. Hub API at the check date: 97 downloads, 884 all-time downloads, 0 likes [4].

**Trained-on-by**: none found - a Hub model search filtered on this dataset returns zero results, and the same Hub user has zero models published on the Hub at all [7][8].

**Introduced by**: no paper - the dataset's Hub card carries only the YAML metadata block, with no prose, citation, or paper link [2]. The underlying 90 problems are AI-MO's own `AI-MO/aimo-validation-aime`, whose card is the closest thing to an "introducing" text for the problem set [1].

## Shape

Rows and split, read from the parquet and confirmed by the datasets-server size endpoint [3][5]:

| split | rows |
| --- | --- |
| `train` | 180 |

One config, `default`, 10 columns (parquet dtypes, matching `cardData.dataset_info` and the datasets-server `/info` response) [2][3][10]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `problem` | string |
| `solution` | string |
| `answer` | string |
| `url` | string |
| `response` | string |
| `response_steps` | list\<string\> |
| `response_steps_cumm` | list\<string\> |
| `response_answer` | string |
| `response_correct` | bool |

`id` runs 0-89 with each value appearing on exactly two rows (90 problems x 2 responses = 180) [3]. `problem`, `solution`, `answer`, and `url` are identical across the two rows sharing an `id`; only `response`, `response_steps`, `response_steps_cumm`, `response_answer`, and `response_correct` differ [3]. No source states token counts for `response`; read across all 180 rows from the served parquet, `response` character length ranges 1,078-3,366 (mean 2,457), `response_steps` holds 1-8 entries per row (mean 3.9), and `response_steps_cumm` holds 2-9 entries per row (mean 4.9, one more than `response_steps` because it prepends an empty first step) [3]. Download size 952,618 bytes; in-memory size 2,905,707 bytes per the datasets-server size endpoint, close to but not identical to the 2,905,685 bytes `cardData.dataset_info` declares [4][5].

## Quality

Read across all 180 rows from the served parquet [3]: 51 of 180 responses (28.3%) are marked `response_correct == True`. Grouped by the 90 problem `id`s, 61 problems have both attempts wrong, 22 have both attempts right, and 7 have exactly one attempt right and one wrong. No source states the scoring method (exact-match on `response_answer` vs. `answer`, or another check), the generation temperature or sampling settings, or any annotator/agreement figure; none is invented here. No source states a duplicate-row or malformed-row rate.

## Load it

```python
import datasets

REV = "edfb29a71cbdf0fcb08b7271b82836f1348763e3"  # main at the check date
ds = datasets.load_dataset(
    "Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn",
    revision=REV,
    split="train",
)  # 180 rows
```

**Trap**: there is only one split (`train`) and one config (`default`), so no `data_dir` or config argument is needed - but that single split IS the eval-contaminated set described above; there is no separate held-out portion inside this repository to train on safely.

## Neighbors

An author-scoped search for the same Hub user's `flatturn`-named repositories, read live at the check date, turns up three siblings [9]:

- `Asap7772/aime_gpt-4o_responses_evaluated_flatturn` - the GPT-4o counterpart: 92,160 rows, one config, 9 columns (no `response_steps_cumm` column) [11]. Far larger sampling budget per problem than this repository's two attempts.
- `Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn_value` - same GPT-4o-mini generations, only 2 rows, with an added `values` float64-sequence column layered on top of this repository's 10 columns [11]. Too small to be a training split; looks like a value-model probe or a smoke-test artifact of the same pipeline.
- `Asap7772/MedQA_gpt-4o-mini_evaluated_flatturn` - the same pipeline applied to MedQA instead of AIME [9]; not fetched further since it is a different problem domain and does not carry the AIME contamination risk this card is about.

None of these has a README beyond the same bare `dataset_info` YAML block, so none resolves the "who built the scoring pipeline and when" gap either [11]. This corpus prefers this repository over `_value` for actual training volume, and over the GPT-4o sibling when the target policy is specifically GPT-4o-mini's own error distribution.

## A row

One config and one split, so one row covers the served shape. From the parquet at `id=0`, first of its two rows (`response_correct == True`) [3]; `solution` and the middle of `response` are truncated with `[...]`:

```json
{
  "id": 0,
  "problem": "Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$",
  "solution": "Let $R(x)=P(x)+Q(x).$ Since the $x^2$-terms of $P(x)$ and $Q(x)$ cancel, we conclude that $R(x)$ is a linear polynomial. [...] Therefore \\[P(0) + Q(0) = 698 + (-582) = \\boxed{116}.\\]\n~Littlemouse",
  "answer": "116",
  "url": "https://artofproblemsolving.com/wiki/index.php/2022_AIME_I_Problems/Problem_1",
  "response": "Let \\( P(x) = 2x^2 + bx + c \\) and \\( Q(x) = -2x^2 + dx + e \\). [...] Therefore, the final answer is: \\(\\boxed{116}\\). I hope it is correct.",
  "response_steps": ["Let \\( P(x) = 2x^2 + bx + c \\) and \\( Q(x) = -2x^2 + dx + e \\). ## Step 1: Set up equations from the points [...]", "..."],
  "response_steps_cumm": ["", "Let \\( P(x) = 2x^2 + bx + c \\) [...]", "..."],
  "response_answer": "116",
  "response_correct": true
}
```

## Where it came from

The 90 source problems are AI-MO's `AI-MO/aimo-validation-aime`, extracted from the AoPS wiki's AIME 2022-2024 pages and released as AI-MO's internal validation set for the AIMO progress-prize competition, chosen from post-2021 problems specifically to avoid overlapping the MATH training set [1]. This repository adds two GPT-4o-mini-generated solution attempts per problem (`response`, split into `response_steps` and the cumulative `response_steps_cumm`), each scored against the AoPS reference `answer` to produce `response_answer` and the boolean `response_correct`. No source states who ran the generation or scoring pipeline, or when, beyond the repository's own creation date of 2025-01-29 [4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] AI-MO/aimo-validation-aime dataset card (README). https://huggingface.co/datasets/AI-MO/aimo-validation-aime/raw/main/README.md - problem source, AIME 22-24 scope, validation-set purpose, MATH-overlap avoidance, column descriptions. Fetched 2026-08-12.

[2] Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn dataset card (README). https://huggingface.co/datasets/Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn/raw/main/README.md - the full card body: `dataset_info` YAML only, no prose, no license field. Fetched 2026-08-12.

[3] The dataset's own parquet file, read in full (all 180 rows). https://huggingface.co/datasets/Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn/resolve/main/data/train-00000-of-00001.parquet - row/column shapes, per-id grouping, correctness counts, response length and step-count statistics, the sampled row. Fetched 2026-08-12.

[4] Hugging Face Hub API record for Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn. https://huggingface.co/api/datasets/Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn?full=true - gate status, `sha`, `downloads`, `likes`, `createdAt`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Asap7772%2Faime_gpt-4o-mini_responses_evaluated_flatturn Fetched 2026-08-12.

[6] The corpus screening row for `Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-12.

[7] Hugging Face Hub model-search API, filtered on this dataset. https://huggingface.co/api/models?filter=dataset:Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn - zero results, live and unpinned. Fetched 2026-08-12.

[8] Hugging Face Hub model-search API, author-scoped, unfiltered. https://huggingface.co/api/models?author=Asap7772 - zero models published by this Hub user, live and unpinned. Fetched 2026-08-12.

[9] Hugging Face Hub dataset-search API, author-scoped. https://huggingface.co/api/datasets?author=Asap7772&search=flatturn - the three sibling repositories named below, live and unpinned. Fetched 2026-08-12.

[10] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Asap7772%2Faime_gpt-4o-mini_responses_evaluated_flatturn Fetched 2026-08-12.

[11] Sibling dataset cards and size endpoints for `Asap7772/aime_gpt-4o_responses_evaluated_flatturn` and `Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn_value`. https://huggingface.co/datasets/<id>/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=<id> - row counts, column lists, absence of prose description. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Flagged, not usable as ordinary training data: this repository is GPT-4o-mini's own scored output on the 90-problem `AI-MO/aimo-validation-aime` benchmark, so training on any part of it risks contaminating a later AIME 2022-2024 evaluation. Two facts decide it, both established above - the source problems are AI-MO's held-out AIME validation set [1], and the screening row's note and flag both name this repository's rows as GPT-4o-mini solutions with correctness labels generated for that eval set [6].

### The screening row

The row's own note [6]: "Read from the parquet: the 90 `AI-MO/aimo-validation-aime` problems (AoPS 2022-2024 AIME urls) with two GPT-4o-mini responses each and a correctness flag. Model outputs scored ON an eval set." Its flag [6]: "contamination: GPT-4o-mini solutions generated for the 90 problems of an AIME eval set, with correctness labels".
