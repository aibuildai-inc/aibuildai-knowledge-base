# AI-MO/aimo-validation-aime

90 AIME competition math problems (30 each from 2022, 2023, and 2024), each with the official numeric answer and a worked solution pulled from the AoPS wiki - a small evaluation benchmark, not a training set.

**AI-MO/aimo-validation-aime** was released by the AI-MO team (the org behind the AI Mathematical Olympiad progress prize) as an internal validation set for that competition, with all 90 problems extracted directly from the Art of Problem Solving wiki's AIME pages; the card gives no origin paper [1]. Each row pairs a `problem` statement with an `answer` (the AIME integer answer, 0-999) and a `solution` (one AoPS-forum solution ending in a boxed answer), plus the source `url` [1]. **This is an evaluation benchmark, not training data: it is one of the two or three most common "AIME 2024" scoring sets in published math-reasoning-RL work, so every row (all 90, spanning 2022-2024) must be held out of training data, not just the 2024 third of it.**

**Use it for**: held-out evaluation only, never any training shape - not SFT, not preference pairs, not reward-model data. It maps to a plain question-answer eval format (`problem` prompted, `answer` graded by exact match against the model's final boxed number), the shape multiple published harnesses already parse this exact file into [2][3]. No method card applies to it.

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`), ungated (`"gated": false`) [4]. The one catch: the licence covers only the repository's own compiled rows, not the underlying AIME problem text or the AoPS-forum solutions, whose own copyright the card does not address [1].

**Shape**: 90 rows, one config (`default`), one split (`train`), five columns [1][5][6].

**Hold out**: all 90 rows. The `id` column runs 0-89 in three 30-row contiguous blocks by contest year (0-29 = AIME 2022, 30-59 = AIME 2023, 60-89 = AIME 2024, confirmed by reading every row's `url` field) [6]. External repos commonly treat the 60-89 block alone as "AIME 2024" and mount the whole file at that path, so both the full file and the id 60-89 subset are contamination risks and neither should enter training data [1][7][8].

**Origin**: built by the AI-MO team from AoPS wiki pages; problems are official AIME competition items and solutions are human-written AoPS-forum solutions, not model-generated [1]. Hub API at the check date: `downloads` 30,947, `likes` 69 [4].

**Trained-on-by**: the Hub's own "models trained or fine-tuned on this dataset" list names exactly one model, `IRUCAAI/Opeai_GRPO_Demo_Qwen2.5-7b-Instruct`, a GRPO demo checkpoint [9] - training on it is exactly what the hold-out warning above cautions against. Its dominant sourced use is as an evaluation set, not training data: PRIME-RL/PRIME mounts this exact file at `eval/data/AI-MO/aimo-validation-aime/aimo-validation-aime.jsonl` and reports the resulting score as its "AIME 2024" row [7]; PrimeIntellect-ai/INTELLECT-MATH reports the same "AIME 2024" score-table row [8], and a Sourcegraph search over public GitHub shows its own eval script mounting the same file at the same path (`rl-and-evals/eval/data/AI-MO/aimo-validation-aime/aimo-validation-aime.jsonl`), alongside the same path in `Trae1ounG/BuPO`, `shivamag125/EM_PT`, and a derived AIME-2025 variant in `ReasoningTransfer/Transferability-of-LLM-Reasoning` [10].

**Introduced by**: no paper - the dataset card [1].

## Shape

Splits and rows (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 90 |

One config, `default`, five columns (datasets-server `/info` and the card's `cardData.dataset_info`) [1][6]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `problem` | string |
| `solution` | string |
| `answer` | string |
| `url` | string |

Sizes: 261,038 bytes as the original/Parquet download, 520,443 bytes decoded in memory per the live `/size` endpoint [5], against 520,431 bytes stated in the card's own `cardData.dataset_info.splits[0].num_bytes` [1] - the two disagree by 12 bytes and no source explains the gap. No source states token or sequence-length statistics for `problem` or `solution`; none is invented here.

## Quality

- The `answer` field is the official AIME numeric answer (an integer 0-999, stored as a string); the `solution` field is one worked solution from the AoPS forum, ending in a boxed answer, not an official or verified-unique solution [1].
- No source states a measured contamination, duplication, or answer-error rate for this repository. The card's only stated rationale for its year range is contamination avoidance in the other direction: it restricts to post-2021 AIME problems "to avoid potential overlap with the MATH training set" [1].
- As evidence the benchmark discriminates between models, PRIME-RL's own published table scores four models on this exact file's "AIME 2024" reporting: Eurus-2-7B-PRIME 26.7, Qwen2.5-Math-7B-Instruct 13.3, Llama-3.1-70B-Instruct 16.7, and the pre-RL Eurus-2-7B-SFT checkpoint at 3.3 - a 23.4-point gap between the SFT base and the RL-trained model on the same 30-problem block [7].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2025-05-07) [4]:

```python
import datasets

REV = "13f9e12f613e720c2a2b2f345dd04b998a29494d"  # main at the check date
aime = datasets.load_dataset("AI-MO/aimo-validation-aime", revision=REV, split="train")  # 90 rows, hold out entirely
```

**Trap**: the single `train` split name suggests trainable data, but the card states the file is "an internal validation set" [1] and its dominant external use is as a frozen eval set [7][8] - do not join it into a training mixture because of the split name. If only the 2024-labeled subset is wanted, filter on `id` between 60 and 89 inclusive; the repository does not expose that subset as a separate split or config.

## Neighbors

- `AI-MO/aimo-validation-amc` - the same team's sibling AMC validation set, mounted by PRIME-RL alongside this file at `data/AI-MO/aimo-validation-amc` in the same evaluation run [7]; a different competition (AMC, not AIME), so it is not a substitute or overlap risk for this file's rows.
- `Maxwell-Jia/AIME_2024` - a separate, single-year (2024-only) AIME repository that EleutherAI's `lm-evaluation-harness` uses for its own `aime24` task instead of this repository [2]; a chooser who specifically wants only the 30 AIME-2024 problems without the 2022/2023 rows should check that repository rather than filtering this one.
- `math-ai/aime25` - the 2025 AIME set, cited by `lm-evaluation-harness` as its `aime25` task source [2]; complements rather than overlaps this 2022-2024 file.

## A row

One config, one split, so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [6], with the multi-paragraph solution truncated:

```json
{
  "id": 0,
  "problem": "Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$",
  "solution": "Let $R(x)=P(x)+Q(x).$ Since the $x^2$-terms of $P(x)$ and $Q(x)$ cancel, we conclude that $R(x)$ is a linear polynomial.\nNote that\n\\begin{alignat*}{8} R(16) &= P(16)+Q(16) &&= 54+54 &&= 108, \\\\ R(20) &= P(20)+Q(20) &&= 53+53 &&= 106, \\end{alignat*}\nso the slope of $R(x)$ is $\\frac{106-108}{20-16}=-\\frac12.$ [...] from which $c=\\boxed{116}.$\n~MRENTHUSIASM\n[... additional forum solutions omitted ...]",
  "answer": "116",
  "url": "https://artofproblemsolving.com/wiki/index.php/2022_AIME_I_Problems/Problem_1"
}
```

The `url` field lets every row be traced to its exact contest and problem number; reading all 90 `url` values confirms the id-to-year mapping used in Hold out above (0-29 → 2022, 30-59 → 2023, 60-89 → 2024) [6].

## Where it came from

Built by the AI-MO team by extracting all 90 problems directly from the AoPS wiki's AIME pages (https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions), restricted to AIME 2022 through 2024 so as to postdate the MATH training set and avoid overlap with it [1]. The card states the set served as the team's internal validation set during its participation in the AIMO progress-prize competition [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] AI-MO/aimo-validation-aime dataset card (README), including its YAML front matter. https://huggingface.co/datasets/AI-MO/aimo-validation-aime/raw/main/README.md - origin, column descriptions, licence tag, `cardData.dataset_info` row/byte counts, year-range rationale. Fetched 2026-08-11.

[2] EleutherAI lm-evaluation-harness, `aime` task family. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/aime/aime24.yaml and https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/aime/README.md - shows the harness's `aime24` task sources `Maxwell-Jia/AIME_2024`, not this repository, and its `aime25` task sources `math-ai/aime25`. A `main`-branch file, unpinned and mutable. Fetched 2026-08-11.

[3] PRIME-RL/PRIME evaluation script. https://raw.githubusercontent.com/PRIME-RL/PRIME/main/eval/run.sh - shows the eval harness reading this repository's file and labeling the run "aime_chat(numina)" / reporting it as "AIME 2024". Fetched 2026-08-11.

[4] Hugging Face Hub API record for AI-MO/aimo-validation-aime. https://huggingface.co/api/datasets/AI-MO/aimo-validation-aime?full=true - licence, gate status, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=AI-MO%2Faimo-validation-aime Fetched 2026-08-11.

[6] datasets-server info and first-rows endpoints. https://datasets-server.huggingface.co/info?dataset=AI-MO%2Faimo-validation-aime and https://datasets-server.huggingface.co/first-rows?dataset=AI-MO%2Faimo-validation-aime&config=default&split=train (plus `/rows` at offsets 30 and 60 to read every row's `url` across all 90 rows) - column dtypes, sampled row, id-to-year mapping. Fetched 2026-08-11.

[7] PRIME-RL/PRIME repository. https://github.com/PRIME-RL/PRIME - README's published benchmark table ("AIME 2024" row) at https://raw.githubusercontent.com/PRIME-RL/PRIME/main/README.md, and the mounted file at https://github.com/PRIME-RL/PRIME/blob/main/eval/data/AI-MO/aimo-validation-aime/aimo-validation-aime.jsonl. Fetched 2026-08-11.

[8] PrimeIntellect-ai/INTELLECT-MATH repository README. https://raw.githubusercontent.com/PrimeIntellect-ai/INTELLECT-MATH/main/README.md - published benchmark table's "AIME 2024" row, sourced from the same file path as [7]. Fetched 2026-08-11.

[9] Hugging Face dataset page's "Models trained or fine-tuned on this dataset" panel. https://huggingface.co/datasets/AI-MO/aimo-validation-aime - lists `IRUCAAI/Opeai_GRPO_Demo_Qwen2.5-7b-Instruct` as the only such model. A live, mutable panel. Fetched 2026-08-11.

[10] Sourcegraph public-code search for the path fragment `aimo-validation-aime`. https://sourcegraph.com/search?q=aimo-validation-aime - additional public repositories serving this exact file: `Trae1ounG/BuPO`, `shivamag125/EM_PT`, and `ReasoningTransfer/Transferability-of-LLM-Reasoning` (an AIME-2025 derivative named after this repository). A live, unpinned search index. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Flagged for contamination, correctly: this repository bundles AIME 2022, 2023, and 2024 problems and is widely mounted by external evaluation harnesses as the "AIME 2024" scoring set, as the sourced adoption evidence above shows [7][8]. The screening row's note and flag both name this risk directly, and nothing found while researching this card contradicts it - hold out all 90 rows.

### The screening row

The row's own note [1]: "30 problems each from AIME 2022, 2023 and 2024; the standard AIME24 eval source." Its flag: "contamination: AIME 2022/2023/2024, the standard AIME24 eval source."
