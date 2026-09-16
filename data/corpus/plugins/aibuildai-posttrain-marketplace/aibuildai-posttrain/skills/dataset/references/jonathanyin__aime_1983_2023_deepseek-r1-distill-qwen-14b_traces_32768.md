# jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768

740 chat-templated reasoning traces - one DeepSeek-R1-Distill-Qwen-14B attempt per AIME (1983-2023) problem, kept only when the model's final answer was correct.

**jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768** pairs each problem from the same author's `jonathanyin/AIME_1983_2023` archive (919 problems, years 1983-2023) with a generated solution trace from DeepSeek-R1-Distill-Qwen-14B, one of the distilled reasoning models whose benchmark scores the DeepSeek-R1 paper reports by this exact name [1]. The repository's own README carries no prose beyond its YAML feature/split declaration - no builder statement, no filtering rule, and no citation [2]. It lives at https://huggingface.co/datasets/jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768 [3]. **The corpus screening flag for this row marks a decontamination risk: this dataset draws on the same AIME problem archive as other trace datasets that the corpus uses for training, and AIME items were not removed before release, so decontaminate against any AIME-based evaluation set before a scored run** [4].

**Use it for**: reasoning-trace SFT - the `templated_response` column is a full ChatML transcript in which the assistant turn opens with `<|im_start|>think`, holds the reasoning, then switches to `<|im_start|>answer` for the final boxed solution before a single closing `<|im_end|>`, ready to feed a reasoning-trace SFT method card as-is; decontaminate against AIME-based evals first per the restriction above.

**Licence**: not stated - no `license` tag or `cardData.license` field on this repository, and no license text in the README [2][5].

**Shape**: 740 rows, one config (`default`), one split (`train`); 12 columns [5][6].

**Hold out**: no split beyond `train`, so there is nothing to hold out inside this repository, but see the bold restriction above and the appendix: the corpus screening flag says AIME items shared with other training datasets in the corpus were not removed, and decontamination against an AIME-based evaluation set is left to the user [4].

**Origin**: built by Hub user jonathanyin; the reasoning and answer-attempt columns are generations from DeepSeek-R1-Distill-Qwen-14B, and `Correct` is a programmatic check against the archive's ground-truth answer, not a human label [5][7]. Hub API at the check date: 227 downloads, 955 all-time downloads, 0 likes [5].

**Trained-on-by**: none found - no source describes a model or recipe trained on this specific repository.

**Introduced by**: no paper - the dataset card [2], which is YAML metadata only and states no description of the dataset.

## Shape

Splits and config, from the Hub API's `cardData.dataset_info` and the datasets-server `/info` and `/size` endpoints, which agree [5][6][8]:

| split | rows |
| --- | --- |
| `train` | 740 |

| column | dtype |
| --- | --- |
| `ID` | string |
| `Year` | int64 |
| `Problem Number` | int64 |
| `Question` | string |
| `Answer` | int64 |
| `Part` | string |
| `Reasoning` | string |
| `Solution Attempt` | string |
| `Answer Attempt` | int64 |
| `Correct` | bool |
| `templated_response` | string |
| `Token Count` | int64 |

Sizes: 9,006,078 bytes downloaded (Parquet), 20,464,810 bytes in memory [8]. `Token Count` (the datasets-server `/statistics` endpoint's measured distribution over all 740 rows) ranges 1,207-8,425, mean 5,262, median 5,284 [9] - all comfortably under the 32768 figure in the repository name, which names the generation token budget rather than an observed length.

## Quality

- `Correct` is `True` for all 740 of 740 rows (`/statistics` frequency count) [9] - the deciding number: this repository is not a raw sample of model attempts, it is the subset of DeepSeek-R1-Distill-Qwen-14B's attempts on the 919-problem archive that were graded correct. 740/919 = 80.5% of the archive survives this filter, comparing this repository's row count [6] against the base archive's declared 919 examples [10].
- `Year` spans 1983-2023 (measured min/max over all 740 rows) [9]; none of these years is 2024 or 2025.
- `Part` is null for 197 of 740 rows (26.6%) and otherwise splits 264 `I` / 279 `II` (measured over all 740 rows) [9].
- No source states an annotator-agreement figure, a deduplication check, or a human review step; `Correct` is the only stated quality signal, and it is a programmatic answer-match, not a human judgment [5].

## Load it

Pin the revision this card's numbers were read at (the shortlist's commit, confirmed as the Hub API's `sha` for `main` at the check date; the repo was last modified 2025-04-25) [5]:

```python
import datasets

REV = "82c0336c06ec4c0388d8f8074eb7ac947ba5c966"  # main at the check date
train = datasets.load_dataset(
    "jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768",
    revision=REV, split="train",
)  # 740 rows, all Correct == True
```

**Trap**: this repository has only a `train` split - there is no held-out split inside it, so any eval/train separation against AIME problems has to be arranged by the user, e.g. by filtering on `Year`/`Problem Number` against whatever evaluation benchmark is in use.

## Neighbors

All row counts below were read live at the check date from the same author's sibling repositories, all built on the same `jonathanyin/AIME_1983_2023` problem archive [10][11]:

- `jonathanyin/AIME_1983_2023` - the un-traced problem archive this repository draws on: 919 rows, columns `ID`, `Year`, `Problem Number`, `Question`, `Answer` (string), `Part`; no `Reasoning`/`templated_response`/`Correct` columns, so it is a source of problems, not of training rows [10].
- `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_16384` - same generating model, a 16384-token generation budget instead of 32768: 718 rows [11].
- `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-7b_traces_32768` - the 7B distilled model at the same 32768 budget: 830 rows [11].
- `jonathanyin/aime_1983_2023_deepseek-r1_traces_32768` - the full (non-distilled) DeepSeek-R1 model at the same budget: 877 rows [11].
- `jonathanyin/aime_1983_2023_qwq-32b_traces_32768` - QwQ-32B at the same budget: 889 rows [11].
- Several further siblings use other models (DeepSeek-R1-Distill-Qwen-1.5B, Grok-3-mini-high) or reformatted variants (`_r1_formatted`); this card fetched only the four listed above for a row-count comparison, and does not claim to have checked the rest [12].

This corpus's shortlist row marks this repository's `origin` as `mixed` and its `use` as `train`; no source states that the corpus prefers this repository over its siblings.

## A row

One config, one split, so one row covers the whole repository. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [13], with the `Reasoning`, `Solution Attempt`, and `templated_response` fields truncated:

```json
{
  "ID": "1983-1",
  "Year": 1983,
  "Problem Number": 1,
  "Question": "Let $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_xw=24$ , $\\log_y w = 40$ and $\\log_{xyz}w=12$ . Find $\\log_zw$ .",
  "Answer": 60,
  "Part": null,
  "Reasoning": "Okay, so I have this problem here: \n\nLet \\(x\\), \\(y\\), and \\(z\\) all exceed 1 and let \\(w\\) be a positive number such that \\(\\log_x w = 24\\), \\(\\log_y w = 40\\), and \\(\\log_{xyz} w = 12\\). I need to find \\(\\log_z w\\).\n\nHmm, logarithms can sometimes be tricky, but let me try to break this down. [...]",
  "Solution Attempt": "Given \\(x\\), \\(y\\), and \\(z\\) all exceed 1 and \\(w\\) is a positive number such that \\(\\log_x w = 24\\), \\(\\log_y w = 40\\), and \\(\\log_{xyz} w = 12\\), we need to find \\(\\log_z w\\). [...]",
  "Answer Attempt": 60,
  "Correct": true,
  "templated_response": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nLet $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_xw=24$ , $\\log_y w = 40$ and $\\log_{xyz}w=12$ . Find $\\log_zw$ .\n\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<|im_start|>think\nOkay, so I have this problem here: [...]\n\\boxed{60}\n<|im_start|>answer\nGiven \\(x\\), \\(y\\), and \\(z\\) all exceed 1 [...]\n\\boxed{60}\n<|im_end|>",
  "Token Count": 2372
}
```

## Where it came from

The problems come from the same author's `jonathanyin/AIME_1983_2023` repository, 919 AIME questions spanning 1983-2023 with string-typed `Answer` [10]. No source states how that archive itself was collected. This repository adds, per problem, a `Reasoning` and `Solution Attempt` generated by DeepSeek-R1-Distill-Qwen-14B - one of the distilled reasoning models released alongside DeepSeek-R1 and named directly in that paper's benchmark tables [1] - together with a programmatically extracted `Answer Attempt` and a `Correct` flag comparing it to the archive's `Answer`; `templated_response` wraps the same question and reasoning in a ChatML transcript whose assistant turn opens with `<|im_start|>think`, switches to `<|im_start|>answer` before the final boxed solution, and closes with a single `<|im_end|>` [5][7]. No source states the generation settings (temperature, sampling count) or why the release keeps only `Correct == True` rows.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Guo et al. (DeepSeek-AI), "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025. https://arxiv.org/abs/2501.12948 - the paper whose benchmark tables name DeepSeek-R1-Distill-Qwen-14B; current title read from the live abs page, benchmark-table text read from the arXiv HTML full text (ar5iv rendering, https://ar5iv.labs.arxiv.org/html/2501.12948). Fetched 2026-08-11.

[2] jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768 dataset card (README). https://huggingface.co/datasets/jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768/raw/main/README.md - YAML-only, no prose description. Fetched 2026-08-11.

[3] jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768 Hub dataset page. https://huggingface.co/datasets/jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768 - confirmed to load (HTTP 200). Fetched 2026-08-11.

[4] The corpus screening row for `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768`, supplied with this card's request - its `flag` (`rule-risk: aime2025 - Same AIME archive as train rows with distilled-14B traces; AIME items not removed`), read back in the appendix. Checked 2026-08-11.

[5] Hugging Face Hub API record for jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768. https://huggingface.co/api/datasets/jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768?full=true - license/gate status, `sha`, `cardData.dataset_info`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768 Fetched 2026-08-11.

[7] datasets-server first-rows endpoint, read for the exact `templated_response` transcript structure and the `<|im_start|>think`/`<|im_start|>answer` tags. https://datasets-server.huggingface.co/first-rows?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768&config=default&split=train Fetched 2026-08-11.

[8] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768 Fetched 2026-08-11.

[9] datasets-server statistics endpoint. https://datasets-server.huggingface.co/statistics?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768&config=default&split=train Fetched 2026-08-11.

[10] Hugging Face Hub API record and datasets-server size endpoint for jonathanyin/AIME_1983_2023, the un-traced problem archive. https://huggingface.co/api/datasets/jonathanyin/AIME_1983_2023?full=true and https://datasets-server.huggingface.co/size?dataset=jonathanyin%2FAIME_1983_2023 - row count, columns, README (YAML-only, same as this repository). Fetched 2026-08-11.

[11] datasets-server size endpoint, one call per neighbor, for the row counts above: `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_16384`, `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-7b_traces_32768`, `jonathanyin/aime_1983_2023_deepseek-r1_traces_32768`, `jonathanyin/aime_1983_2023_qwq-32b_traces_32768`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[12] Hugging Face Hub API dataset listing for author jonathanyin. https://huggingface.co/api/datasets?author=jonathanyin&limit=100 - full list of sibling repository ids, used only to identify which siblings exist; the ones not listed in [11] were not fetched for row counts. Fetched 2026-08-11.

[13] datasets-server first-rows endpoint (same call as [7]), read here for the sampled row. https://datasets-server.huggingface.co/first-rows?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data, conditional on decontamination: the corpus marks this row `use: train` and `origin: mixed`, but attaches a `rule-risk` flag rather than a blocking flag. The flag's own text - that this repository shares the AIME problem archive with other training rows in the corpus and that AIME items were not removed before release - is the fact the bold restriction and the Hold out line above rest on [4].

### The screening row

The row's own note [4]: "Same problems with DeepSeek-R1-Distill-Qwen-14B traces." Its flag: "rule-risk: aime2025 - Same AIME archive as train rows with distilled-14B traces; AIME items not removed".
