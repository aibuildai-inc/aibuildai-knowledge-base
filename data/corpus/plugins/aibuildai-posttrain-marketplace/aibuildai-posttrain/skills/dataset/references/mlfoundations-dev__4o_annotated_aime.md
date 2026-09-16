# mlfoundations-dev/4o_annotated_aime

3,403 competition-math problems from NuminaMath-CoT's `amc_aime` source, each paired with a DeepSeek reasoning trace, a judge verdict on that trace, and a GPT-4o-mini response - one row per problem, no splits.

**mlfoundations-dev/4o_annotated_aime** is built by mlfoundations-dev from two upstream Hugging Face datasets, joined by the pipeline the repository itself ships as `config/4o_annotated_aime.yaml`: it loads `AI-MO/NuminaMath-CoT`, keeps only the rows whose `source` column is `amc_aime`, loads `mlfoundations-dev/math_stratos_scale_judged_and_annotated` (140,075 rows, columns `problem`/`reasoning`/`deepseek_solution`/`ground_truth_solution`/`correct`/`judge_reasoning`), inner-joins the two on the `problem` text, re-filters to `source == amc_aime`, and calls `gpt-4o-mini` at temperature 1.0 on each surviving problem to fill a `4o_response` column, which is then reshaped into the `conversations` ShareGPT column [1]. The dataset card itself is front matter only, with no prose description [2]. It lives at https://huggingface.co/datasets/mlfoundations-dev/4o_annotated_aime . **The `amc_aime` source label in NuminaMath-CoT names AMC and AIME problems together, and the pipeline config's own `select_rows`/`refilter` steps never remove AIME items from that source before publishing this dataset - the corpus screening record flags this as an `aime2025` rule-risk, i.e. a risk of overlap with an AIME-based evaluation set; decontaminate against AIME25/AIME24 before using this data for a scored run [3].**

**Use it for**: reasoning-trace SFT or preference work built on a math problem plus a full worked solution - most directly the `conversations` column (human/gpt turns pairing `problem` with the GPT-4o-mini `4o_response`), or a custom collator built from `problem` + `reasoning` (the DeepSeek trace) + `correct`/`judge_reasoning` (a judge's correctness verdict on that trace) [4]. **Decontaminate against AIME24/AIME25 before a scored run, per the flag above** [3]. Maps to the reasoning-trace-SFT method card; that card should read from `conversations` or build its own prompt/response pair from `problem` and `4o_response`, not from the raw `messages` string column (see Load it for why).

**Licence**: not stated. `cardData` carries no `license` key and the card body mentions none; the repo is ungated and public (`"gated": false`, `"private": false`) [2][5]. The upstream `AI-MO/NuminaMath-CoT` pool this was built from is Apache-2.0 [6], but no source states that grant carries over to this derived repository.

**Shape**: 3,403 rows, one config (`default`), one split (`train`), 12 columns [5][7].

**Hold out**: nothing declared as a train/test split - there is only `train` [5][7]. The contamination risk is not a holdout inside this repo but an external one: the `amc_aime` source it draws from can contain AIME items that overlap an AIME24/AIME25 evaluation set, per the bolded restriction above [3].

**Origin**: built by mlfoundations-dev; problems and their NuminaMath solutions are sourced from `AI-MO/NuminaMath-CoT`, the `reasoning`/`deepseek_solution`/`correct`/`judge_reasoning` columns come pre-built from `mlfoundations-dev/math_stratos_scale_judged_and_annotated`, and the `4o_response` column is newly generated here by `gpt-4o-mini` [1]. Hub API at the check date: 64 downloads in the trailing window, 672 all-time downloads, 0 likes [5].

**Trained-on-by**: none found. No source in hand states a model or published recipe trained on this specific repository.

**Introduced by**: no paper - the dataset card [2], plus the repository's own pipeline config file [1].

## Shape

Rows and splits (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 3,403 |

One config, `default`, 12 columns (datasets-server `/info` and the Hub API's `cardData.dataset_info`) [5][7]:

| column | dtype |
| --- | --- |
| `problem` | string |
| `reasoning` | string |
| `deepseek_solution` | string |
| `ground_truth_solution` | string |
| `correct` | bool |
| `judge_reasoning` | string |
| `source` | string |
| `solution` | string |
| `messages` | string |
| `4o_response` | string |
| `__original_row_idx` | int64 |
| `conversations` | list\<struct\<from: string, value: string\>\> |

Sizes (datasets-server `/size`) [7]: 39,400,221 bytes as downloaded Parquet, 86,342,831 bytes decoded in memory. No source states token or sequence-length statistics for this dataset.

## Quality

- `correct` is a per-row boolean carried in from the upstream judged-and-annotated pool, judging the `deepseek_solution`/`reasoning` trace against `ground_truth_solution`; `judge_reasoning` is the judge's free-text explanation for that verdict [1]. Of the 10 rows read at offset 0 in `train`, 7 had an un-truncated `judge_reasoning`/`solution` field and of those 7, 6 had `correct = True` and 1 had `correct = False`; this is a read of 7 rows only and is not a dataset-wide rate [8].
- In those same 7 un-truncated rows, `solution` was byte-identical to `ground_truth_solution` in all 7; no source states whether this holds across the full 3,403 rows [8].
- No source states a measured duplicate rate, contamination rate, or annotator-agreement figure for the pipeline's `gpt-4o-mini` annotation step; none is invented here.
- The dataset card carries no quality-caveat prose of its own - it is front matter only [2].

## Load it

Pin the revision this card's numbers were read at (the shortlist row's commit, which matches the Hub API's current `sha` for `main`):

```python
import datasets

REV = "476dac58ece302d3baad217a4d1fdbc946b39f21"
ds = datasets.load_dataset("mlfoundations-dev/4o_annotated_aime", revision=REV, split="train")  # 3,403 rows
```

**Trap**: the `messages` column is a string, not a list of dicts, and it is not valid JSON or a valid Python literal - of the 7 rows read at offset 0 whose `messages` field was not truncated by the API, all 7 render as `[{...}\n {...}]` with a newline where a comma belongs between the two turn-dicts, so both `json.loads` and `ast.literal_eval` raise a syntax error on it as served [8]. Use the `conversations` column instead: it is a proper `list<struct<from, value>>` with two turns per row, `from: "human"` holding `problem` and `from: "gpt"` holding `4o_response` [8].

## Neighbors

- `mlfoundations-dev/r1_annotated_aime` - built from the identical join (same `config/r1_annotated_aime.yaml` pipeline, same `AI-MO/NuminaMath-CoT` `amc_aime` rows joined against the same `mlfoundations-dev/math_stratos_scale_judged_and_annotated`), same 3,403 rows, but its `conversations` column is built from `reasoning` (the DeepSeek trace) instead of `4o_response` [9]. Prefer this repository when the training target is a GPT-4o-mini-style response; prefer `r1_annotated_aime` when the target is the DeepSeek reasoning trace. Do not mix both into one run without deduplicating on `problem` - they cover the same 3,403 problems.
- `mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered` - an earlier, unfiltered `amc_aime` pull of 4,069 rows [10], close to NuminaMath-CoT's own reported `amc_aime` count of 4,072 [6]; this repository's 3,403 rows are the subset of that pool that also matched the judged-and-annotated pool by `problem` text.
- `mlfoundations-dev/math_stratos_scale_judged_and_annotated` - the 140,075-row upstream pool this repository's `reasoning`/`deepseek_solution`/`correct`/`judge_reasoning` columns are drawn from, across all NuminaMath sources, not just `amc_aime` [1].
- `AI-MO/NuminaMath-CoT` - the original problem/solution pool this repository's `problem`, `solution`, `source`, and `messages` columns are drawn from [6].

## A row

One config, one split, so one row covers it. From `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [8], with `judge_reasoning` and `4o_response` truncated:

```json
{
  "problem": "The operation $\\otimes$ is defined for all nonzero numbers by $a\\otimes b =\\frac{a^{2}}{b}$. Determine $[(1\\otimes 2)\\otimes 3]-[1\\otimes (2\\otimes 3)]$.\n$\\text{(A)}\\ -\\frac{2}{3}\\qquad\\text{(B)}\\ -\\frac{1}{4}\\qquad\\text{(C)}\\ 0\\qquad\\text{(D)}\\ \\frac{1}{4}\\qquad\\text{(E)}\\ \\frac{2}{3}$",
  "reasoning": "Okay, so I need to solve this problem where there's an operation defined as a⊗b = a²/b. And the question is asking for [(1⊗2)⊗3] - [1⊗(2⊗3)]. [...]",
  "deepseek_solution": "To solve the problem, we need to evaluate the expression \\([(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]\\) [...]",
  "ground_truth_solution": "1. **Apply the operation $\\otimes$ to the innermost parentheses first:** [...] \\[\\boxed{A}\\]",
  "correct": true,
  "judge_reasoning": "The solution correctly follows the steps of applying the operation [...] hence the answer is indeed \\boxed{A}. [...]",
  "source": "amc_aime",
  "solution": "1. **Apply the operation $\\otimes$ to the innermost parentheses first:** [...] \\[\\boxed{A}\\]",
  "messages": "[{'content': 'The operation $\\\\otimes$ is defined for all nonzero numbers [...]', 'role': 'user'}\n {'content': '1. **Apply the operation $\\\\otimes$ [...]', 'role': 'assistant'}]",
  "4o_response": "To solve for \\( [(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)] \\), we first need to individually compute [...]",
  "__original_row_idx": 0,
  "conversations": [
    {"from": "human", "value": "The operation $\\otimes$ is defined for all nonzero numbers by $a\\otimes b =\\frac{a^{2}}{b}$. Determine $[(1\\otimes 2)\\otimes 3]-[1\\otimes (2\\otimes 3)]$. [...]"},
    {"from": "gpt", "value": "To solve for \\( [(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)] \\), we first need to individually compute [...]"}
  ]
}
```

## Where it came from

mlfoundations-dev built this repository by running its own published pipeline config, `config/4o_annotated_aime.yaml`, hosted in the repository itself: load `AI-MO/NuminaMath-CoT`, filter to `source == amc_aime`, inner-join by exact `problem` text against `mlfoundations-dev/math_stratos_scale_judged_and_annotated` (a 140,075-row pool carrying a DeepSeek reasoning trace, a `deepseek_solution`, a `ground_truth_solution`, a `correct` judge verdict, and `judge_reasoning` for a broader set of NuminaMath problems), re-filter the joined rows to `source == amc_aime` again, then call `gpt-4o-mini` at temperature 1.0 on each `problem` to produce `4o_response`, and finally reshape `problem`/`4o_response` into the `conversations` ShareGPT column [1]. NuminaMath-CoT's own card states its ~860k problems were collected mainly from online exam-paper PDFs and mathematics discussion forums, then processed through OCR, segmentation, translation, and CoT realignment, with `amc_aime` one of nine named source categories (4,072 problems) [6]. No source in hand states which model or process produced the upstream pool's `reasoning`/`deepseek_solution` columns beyond what the column name implies.

## Sources

Fetched on the check date, 2026-08-12; Hub API responses and datasets-server endpoints are live and unpinned except where the revision is stated, so `Load it` records the `sha` this card's numbers were read at.

[1] Repository pipeline config, `config/4o_annotated_aime.yaml`. https://huggingface.co/datasets/mlfoundations-dev/4o_annotated_aime/raw/main/config/4o_annotated_aime.yaml Fetched 2026-08-12.

[2] Dataset card (README) front matter. https://huggingface.co/datasets/mlfoundations-dev/4o_annotated_aime/raw/main/README.md - front matter only, no body prose. Fetched 2026-08-12.

[3] The corpus screening row for `mlfoundations-dev/4o_annotated_aime`, supplied with this card's request - its `flag` field ("rule-risk: aime2025 - Same NuminaMath `amc_aime` problems with GPT-4o responses added; AIME items never removed"), read back in the appendix. Checked 2026-08-12.

[4] trl dataset formats documentation. https://huggingface.co/docs/trl/main/en/dataset_formats - conversational (ShareGPT-style) format for SFT. A `main` build, unpinned and mutable. Fetched 2026-08-12.

[5] Hugging Face Hub API record for mlfoundations-dev/4o_annotated_aime. https://huggingface.co/api/datasets/mlfoundations-dev/4o_annotated_aime?full=true and the same endpoint with `expand[]=downloadsAllTime` - `sha`, `cardData`, `gated`, `private`, `downloads`, `downloadsAllTime`, `likes`. Fetched 2026-08-12.

[6] AI-MO/NuminaMath-CoT dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/raw/main/README.md - source breakdown table (`amc_aime`: 4,072), collection method, Apache-2.0 licence. Fetched 2026-08-12.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2F4o_annotated_aime Fetched 2026-08-12.

[8] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mlfoundations-dev%2F4o_annotated_aime&config=default&split=train - 10 rows at offset 0, `truncated_cells` used to identify which cells the endpoint shortened. Fetched 2026-08-12.

[9] mlfoundations-dev/r1_annotated_aime pipeline config and size. https://huggingface.co/datasets/mlfoundations-dev/r1_annotated_aime/raw/main/config/r1_annotated_aime.yaml and https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fr1_annotated_aime Fetched 2026-08-12.

[10] datasets-server size endpoint for mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fbespokelabs-sky-t1-numina-amc-aime-subset-unfiltered Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable only with the AIME24/AIME25 decontamination step named above: the pipeline config confirms this repository is exactly the `amc_aime`-sourced rows of NuminaMath-CoT joined against a judged reasoning pool, with a GPT-4o-mini response added, and the config's own filter steps never drop the AIME half of that `amc_aime` label [1]. The screening row's flag names the resulting risk directly [3].

### The screening row

The row's own note [3]: "Same amc_aime problems with GPT-4o responses added next to DeepSeek reasoning and a judge verdict." Its flag [3]: "rule-risk: aime2025 - Same NuminaMath `amc_aime` problems with GPT-4o responses added; AIME items never removed."
