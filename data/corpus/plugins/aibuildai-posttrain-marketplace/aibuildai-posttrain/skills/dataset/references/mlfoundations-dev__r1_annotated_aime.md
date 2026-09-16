# mlfoundations-dev/r1_annotated_aime

3,403 competition-math problems, each paired with a DeepSeek-R1 reasoning trace, a judge verdict on whether that trace's answer matches the reference solution, and the same content pre-formatted into two chat-message shapes, in a single train split.

**mlfoundations-dev/r1_annotated_aime** is one of the `mlfoundations-dev` group's intermediate build artifacts: the repository's own `config/r1_annotated_aime.yaml` shows it is built by loading `AI-MO/NuminaMath-CoT`, keeping only rows whose `source` column equals `amc_aime`, and joining those problems (on problem text) against `mlfoundations-dev/math_stratos_scale_judged_and_annotated` - a 140,075-row table pairing NuminaMath problems with reasoning traces and a correctness judgment, whose own dataset card states only its column schema and row count, not which model generated or judged that content [1][2]. It lives at https://huggingface.co/datasets/mlfoundations-dev/r1_annotated_aime . **Despite its name, every one of this repository's 3,403 rows carries NuminaMath's `source` value `amc_aime`, which mixes AMC and AIME problems rather than isolating AIME - confirmed both by the `config/r1_annotated_aime.yaml` filter step and by all 10 rows sampled at offset 0 [1][3]. A separate build config in the same `mlfoundations-dev` org fuzzy-matches a different NuminaMath-derived, rejection-sampled pool (`bespokelabs/sky-t1-numina-rejection-sampled`, not filtered to any `source` category) against `Maxwell-Jia/AIME_2024` before use, showing the org treats AIME overlap as a real risk for NuminaMath-derived pools in general [4]; this repository's own build carries no such decontamination step, so decontaminate against AIME/AMC evaluation sets (e.g. AIME 2024, AIME 2025) before any scored run that uses it.**

**Use it for**: reasoning-trace SFT (distillation) - the `conversations` and `messages` columns are the problem and the `reasoning` column's text pre-formatted as a two-turn human/assistant exchange [1][3]. No licence or usage-shape restriction beyond the AIME/AMC decontamination point above is stated by any fetched source. Maps to the SFT method card's chat-format loader.

**Licence**: not stated - `cardData` in the Hub API record carries no `license` field, the repository's tags carry no `license:` tag, and the README's body is empty of prose beyond its YAML front matter (712 bytes, all metadata) [5][6]. Ungated (`"gated": false`) [5].

**Shape**: 3,403 rows, one config (`default`), one split (`train`) [7][8].

**Hold out**: no internal split is provided to hold out (single `train` split, 3,403 rows) [7]. The risk here is external: decontaminate against AIME/AMC evaluation benchmarks before a scored run, per the bold restriction above and the flag in the appendix.

**Origin**: built by the `mlfoundations-dev` group. The dataset id names "r1" and one column is named `deepseek_solution`, but no fetched source confirms DeepSeek-R1 as the generating model or names the judge model behind `correct`/`judge_reasoning`; neither this repository's card nor its direct upstream `math_stratos_scale_judged_and_annotated`'s card states either model, so both are not stated [6][2]. Hub API at the check date: `downloads` 223, `downloadsAllTime` 1,022, `likes` 0 [5][9].

**Trained-on-by**: none found. No fetched source names a model or training run that consumed this specific repository [1][10][2].

**Introduced by**: no paper - the dataset card carries no description [6], but the repository's own build config documents its construction [1].

## Shape

Rows and splits (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 3,403 |

One config, `default`, with ten columns (datasets-server `/info`) [8]:

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
| `conversations` | list\<struct\<from: string, value: string\>\> |

Sizes (datasets-server `/size`) [7]: 53,240,173 bytes as Parquet (original download), 119,013,156 bytes decoded in memory. No source states sequence-length or token statistics for this repository.

## Quality

- Of the 10 rows read at offset 0 in `train`, 9 carried `correct: true` and 1 carried `correct: false`; no source states a correctness rate for the full 3,403 rows [3].
- Every one of those 10 rows carried `source: amc_aime`; the repository's own build config filters `AI-MO/NuminaMath-CoT` to `source == amc_aime` twice in its pipeline (once before, once after joining the reasoning annotations), so this is not a sampling artifact but the declared construction rule [1][3].
- In the 4 rows read fully (indices 0-3, the only rows datasets-server served without truncation), the `solution` and `ground_truth_solution` fields were byte-identical strings in every case; whether this holds across the remaining 3,399 rows is not stated [3].
- No source states a measured duplicate rate, annotator-agreement figure, or a rate for the AMC/AIME contamination risk described above; none is invented here.

## Load it

Single split, no held-out rows. The Shape and Quality figures above come from datasets-server's `/size`, `/info`, and `/first-rows` endpoints, which take no revision parameter and always read `main` live [7][8][3]; separately, the Hub API's `sha` for `main` at the check date matches the shortlist row's recorded commit, so pin that revision below for reproducible loads even though the datasets-server reads themselves are not revision-pinned (the repo was last modified 2025-01-30) [5]:

```python
import datasets

REV = "8d3e86e734b4ea6139d3091806fc307a13b52e52"  # main at the check date
train = datasets.load_dataset("mlfoundations-dev/r1_annotated_aime", revision=REV, split="train")  # 3,403 rows
```

**Trap**: `messages` is a Python `repr()`-style string (single-quoted dict literals, e.g. `[{'content': ..., 'role': 'user'}, ...]`), not valid JSON - `json.loads` on it will fail; parse it with `ast.literal_eval` instead, or use `conversations`, which is already a proper `from`/`value` struct list [3]. Both columns encode the same problem-plus-reasoning exchange; `conversations`' `gpt` turn is byte-identical to the `reasoning` column and does not include `deepseek_solution` [1][3].

## Neighbors

- `mlfoundations-dev/4o_annotated_aime` - the same-shaped NuminaMath `amc_aime` pool (3,403 rows, matching this repository's row count), annotated with a different teacher model per its name rather than DeepSeek-R1; not opened beyond its row count here [11].
- `mlfoundations-dev/r1_annotated_math`, `mlfoundations-dev/r1_annotated_aops`, `mlfoundations-dev/r1_annotated_olympiads` - sibling category slices of the same build pipeline (`config/r1_annotated_aime.yaml`'s pattern applied to NuminaMath's `math`, `aops_forum`, and `olympiads` sources respectively), at 6,921 / 17,331 / 34,080 rows; not opened beyond row counts here [11].
- `mlfoundations-dev/math_stratos_scale_judged_and_annotated` - the unfiltered, un-split 140,075-row source table this repository's reasoning and judge columns are joined from; prefer it when a category split by NuminaMath source is not needed [1][2].
- `bespokelabs/Bespoke-Stratos-17k` (and its `-35k`/`-50k` siblings) - a separate release by Bespoke Labs, built from NuminaMath's AIME/MATH/Olympiads categories with DeepSeek-R1 as reasoning teacher and gpt-4o-mini judging math solutions, per its own card [10]. No fetched source confirms this shares a build pipeline with `mlfoundations-dev/r1_annotated_aime`; it is listed here because a reader wanting an already-assembled, rejection-sampled training set rather than this category-filtered intermediate table should know it exists.

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`, the only row of the first 10 served fully untruncated) [3], with long fields truncated:

```json
{
  "problem": "The operation $\\otimes$ is defined for all nonzero numbers by $a\\otimes b =\\frac{a^{2}}{b}$. Determine $[(1\\otimes 2)\\otimes 3]-[1\\otimes (2\\otimes 3)]$.\n$\\text{(A)}\\ -\\frac{2}{3}\\qquad\\text{(B)}\\ -\\frac{1}{4}\\qquad\\text{(C)}\\ 0\\qquad\\text{(D)}\\ \\frac{1}{4}\\qquad\\text{(E)}\\ \\frac{2}{3}$",
  "reasoning": "Okay, so I need to solve this problem where there's an operation defined as a⊗b = a²/b. And the question is asking for [(1⊗2)⊗3] - [1⊗(2⊗3)]. The answer choices are given from A to E. Alright, let's break this down step by step. [...]",
  "deepseek_solution": "To solve the problem, we need to evaluate the expression \\([(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]\\) where the operation \\(\\otimes\\) is defined as \\(a \\otimes b = \\frac{a^2}{b}\\). [...]",
  "ground_truth_solution": "1. **Apply the operation $\\otimes$ to the innermost parentheses first:** [...]",
  "correct": true,
  "judge_reasoning": "The solution correctly follows the steps of applying the operation, computes both required parts, and accurately performs the subtraction to arrive at the final result of -2/3, hence the answer is indeed A. [...]",
  "source": "amc_aime",
  "solution": "1. **Apply the operation $\\otimes$ to the innermost parentheses first:** [...]",
  "messages": "[{'content': 'The operation $\\otimes$ is defined for all nonzero numbers by...', 'role': 'user'}, {'content': '1. **Apply the operation $\\otimes$...', 'role': 'assistant'}]",
  "conversations": [
    {"from": "human", "value": "The operation $\\otimes$ is defined for all nonzero numbers by $a\\otimes b =\\frac{a^{2}}{b}$. Determine $[(1\\otimes 2)\\otimes 3]-[1\\otimes (2\\otimes 3)]$. [...]"},
    {"from": "gpt", "value": "Okay, so I need to solve this problem where there's an operation defined as a⊗b = a²/b. [...]"}
  ]
}
```

## Where it came from

Built by the `mlfoundations-dev` group as one node in a data-preparation pipeline whose config file is checked into the repository [1]. The pipeline: (1) load `AI-MO/NuminaMath-CoT`'s `train` split and keep only rows with `source == amc_aime`; (2) load `mlfoundations-dev/math_stratos_scale_judged_and_annotated`'s `train` split, a 140,075-row table pairing NuminaMath problems with reasoning traces (`reasoning`, `deepseek_solution`) and a correctness judgment against the reference solution (`correct`, `judge_reasoning`); (3) inner-join the two on the `problem` text; (4) filter the joined table to `source == amc_aime` again; (5) convert `problem`/`reasoning` into a `conversations` chat format [1]. Neither this repository's own card nor `math_stratos_scale_judged_and_annotated`'s card states which model generated the reasoning traces or which model produced the correctness judgments [6][2]. A separate Bespoke Labs release, `bespokelabs/Bespoke-Stratos-17k`, documents a similarly shaped pipeline over NuminaMath's AIME/MATH/Olympiads categories - DeepSeek-R1 as reasoning teacher, gpt-4o-mini judging math solutions - but no fetched source confirms that pipeline is the one used to build `math_stratos_scale_judged_and_annotated` [10].

## Sources

Fetched on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] `config/r1_annotated_aime.yaml`, the build pipeline checked into the repository. https://huggingface.co/datasets/mlfoundations-dev/r1_annotated_aime/raw/main/config/r1_annotated_aime.yaml Fetched 2026-08-11.

[2] `mlfoundations-dev/math_stratos_scale_judged_and_annotated` dataset card (README). https://huggingface.co/datasets/mlfoundations-dev/math_stratos_scale_judged_and_annotated/raw/main/README.md - column schema and row count only; no prose body. Fetched 2026-08-11.

[3] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mlfoundations-dev%2Fr1_annotated_aime&config=default&split=train - 10 rows read at offset 0. Fetched 2026-08-11.

[4] `config/decontaminate_stratos_numina.yaml`, a build config in the same `mlfoundations-dev` org that loads `bespokelabs/sky-t1-numina-rejection-sampled` (a rejection-sampled NuminaMath pool, not filtered to any `source` category) and fuzzy-matches its `problem` column against `Maxwell-Jia/AIME_2024` and `HuggingFaceH4/MATH-500` at a 95.0 similarity threshold. https://huggingface.co/datasets/mlfoundations-dev/decontaminate_stratos_numina/raw/main/config/decontaminate_stratos_numina.yaml - shows the org treats AIME overlap as a real risk for NuminaMath-derived pools elsewhere in its pipeline; this repository's own build config carries no such decontamination step. Fetched 2026-08-11.

[5] Hugging Face Hub API record for mlfoundations-dev/r1_annotated_aime. https://huggingface.co/api/datasets/mlfoundations-dev/r1_annotated_aime?full=true - `cardData`, `gated`, `sha`, `downloads`, `likes`, `lastModified`, `siblings`. Fetched 2026-08-11.

[6] mlfoundations-dev/r1_annotated_aime dataset card (README). https://huggingface.co/datasets/mlfoundations-dev/r1_annotated_aime/raw/main/README.md - 712 bytes, all YAML front matter, no prose body. Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fr1_annotated_aime Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=mlfoundations-dev%2Fr1_annotated_aime Fetched 2026-08-11.

[9] Hugging Face Hub API record with `downloadsAllTime` expansion. https://huggingface.co/api/datasets/mlfoundations-dev/r1_annotated_aime?expand[]=downloadsAllTime Fetched 2026-08-11.

[10] `bespokelabs/Bespoke-Stratos-17k` dataset card (README). https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k/raw/main/README.md - a separate Bespoke Labs release's description of its own teacher model, judge model, and NuminaMath data source; not confirmed to describe the pipeline behind this repository. Fetched 2026-08-11.

[11] datasets-server size endpoint, one call per neighbor: `mlfoundations-dev/4o_annotated_aime`, `mlfoundations-dev/r1_annotated_math`, `mlfoundations-dev/r1_annotated_aops`, `mlfoundations-dev/r1_annotated_olympiads`. https://datasets-server.huggingface.co/size?dataset=<id> - row counts only, not opened further. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-distillation SFT data, conditional on decontamination: the repository's own build config and the sampled rows confirm the `source` column is uniformly `amc_aime` [1][3], and the org's own decontamination config elsewhere in the same pipeline treats fuzzy overlap with `Maxwell-Jia/AIME_2024` as a real risk for this kind of NuminaMath-derived pool [4] - reason enough to decontaminate against AIME/AMC evaluation sets before a scored run, as this repository's own build carries no such step.

### The screening row

The row's own note: "Same problems with DeepSeek-R1 reasoning plus judge verdicts; the seed of the Sky-T1/s1 line." Its flag, class `rule-risk`: "aime2025 `source` column is 100% `amc_aime` (3403/3403), i.e. NuminaMath's AMC/AIME pool used as training rows."
