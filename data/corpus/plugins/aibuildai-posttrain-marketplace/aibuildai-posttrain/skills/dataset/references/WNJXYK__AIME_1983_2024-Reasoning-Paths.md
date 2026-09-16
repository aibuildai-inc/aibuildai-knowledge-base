# WNJXYK/AIME_1983_2024-Reasoning-Paths

Nine per-model, per-temperature JSON files (4.57 GB of LFS data) holding 256 sampled reasoning paths per AIME problem from three small open math LLMs, spanning AIME 1983 through 2024.

**WNJXYK/AIME_1983_2024-Reasoning-Paths** wraps the [di-zhang-fdu/AIME_1983_2024](https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024) problem set with model-generated reasoning paths, released alongside the NeurIPS 2025 paper "A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning" [1], which studies sampling-based test-time scaling (self-consistency and perplexity-based confidence estimation) and introduces a hybrid method, RPC, evaluated on this and sibling releases. Each of the 9 files packs 933 AIME problems by 256 sampled completions from one of DeepSeek-Math-RL-7B, InternLM2-Math-Plus-1.8B, or InternLM2-Math-Plus-7B, at sampling temperature 1.0, 1.1, or 1.3 [2]. It lives at https://huggingface.co/datasets/WNJXYK/AIME_1983_2024-Reasoning-Paths . **The problem set carries 14 of the 15 AIME-2024-II problems (problem 9 is absent), which are part of the standard aime24 evaluation set used across math-reasoning papers: decontaminate against aime24, or drop the 14 affected problems and their 256 sampled paths each per file, before training a model that will later be scored on aime24 [3].**

**Use it for**: reasoning-trace SFT - each problem carries 256 full model completions plus a per-completion boolean `accuracy` label, so a consumer typically filters to correct completions before distillation-style SFT; maps to the SFT method card. Each file is one flat JSON object with parallel arrays keyed by problem index, not row-per-example, so it needs reshaping into prompt/completion pairs before use [2].

**Licence**: MIT (`cardData.license` is `"mit"`), ungated, not private [4]. No catch in this repository's own licence grant; the aime24-overlap restriction above is a contamination risk, not a licence term.

**Shape**: no config/split structure - 9 standalone JSON files (one per model x temperature), each holding the same 933 AIME problems x 256 sampled paths; the Hub dataset viewer returns "dataset viewer is disabled" for this repository [5][6].

**Hold out**: the 14 AIME-2024-II problems at problem index 919-932 of the shared 933-problem ordering (problems numbered 1-8 and 10-15; problem 9 is missing from the source), each carrying 256 sampled paths per file, if the trained model will later be scored on the aime24 benchmark; nothing else in this release needs holding out [3].

**Origin**: built and released by Hugging Face user WNJXYK; the reasoning paths are generations sampled from three existing math LLMs (DeepSeek-Math-RL-7B, InternLM2-Math-Plus-1.8B, InternLM2-Math-Plus-7B), not human-written [2]. Hub API at the check date: `downloads` 552, `likes` 16, `downloadsAllTime` 2,915 [4].

**Trained-on-by**: none found. Neither the dataset card, the origin paper, nor the linked Hugging Face paper page names a model or training recipe that trained on this release; the paper itself uses the sampled paths only for test-time confidence-estimation evaluation, not for training [1][2][7].

**Introduced by**: [1] (Zhou et al., NeurIPS 2025).

## Shape

The 9 data files, read from the Hub tree listing at the pinned commit [8]:

| file | model | temperature (per README) | bytes |
| --- | --- | --- | --- |
| `Deepseek-Math-RL-7B.json` | DeepSeek-Math-RL-7B | 1.0 | 402,531,371 |
| `Deepseek-Math-RL-7B-T=1.1.json` | DeepSeek-Math-RL-7B | 1.1 | 404,577,465 |
| `Deepseek-Math-RL-7B-T=1.3.json` | DeepSeek-Math-RL-7B | 1.3 | 412,357,082 |
| `InternLM2-Math-Plus-1.8B-T=1.0.json` | InternLM2-Math-Plus-1.8B | 1.0 | 517,617,245 |
| `InternLM2-Math-Plus-1.8B-T=1.1.json` | InternLM2-Math-Plus-1.8B | 1.1 | 547,095,446 |
| `InternLM2-Math-Plus-1.8B-T=1.3.json` | InternLM2-Math-Plus-1.8B | 1.3 | 784,781,744 |
| `InternLM2-Math-Plus-7B.json` | InternLM2-Math-Plus-7B | 1.0 | 484,971,281 |
| `InternLM2-Math-Plus-7B-T=1.1.json` | InternLM2-Math-Plus-7B | 1.1 | 490,893,628 |
| `InternLM2-Math-Plus-7B-T=1.3.json` | InternLM2-Math-Plus-7B | 1.3 | 527,502,170 |

Each file is a single JSON object with 8 parallel top-level fields, confirmed by fetching byte ranges of `Deepseek-Math-RL-7B.json` (the viewer being disabled means these were read directly, not through datasets-server) [9]:

| field | shape | content |
| --- | --- | --- |
| `predict` | [933 problems][256 samples] of string | normalized final answer per sample |
| `answer` | [933 problems] of string | ground-truth answer, copied from the upstream problem set |
| `completion` | [933 problems][256 samples] of string | full generated reasoning text per sample |
| `cumulative_logprob` | [933 problems][256 samples] of float | summed log-probability per sample |
| `mean_logprob` | [933 problems][256 samples] of float | log-probability normalized by length |
| `prompt` | [933 problems] of string | the problem text plus a fixed instruction suffix |
| `temperature`, `top_p` | scalar | sampling settings for that file (1.0 and 0.95 confirmed in `Deepseek-Math-RL-7B.json`) |
| `accuracy` | [256 samples][933 problems] of bool | whether `predict` matched `answer`, transposed relative to the other per-sample fields |

933 is the row count of the upstream `di-zhang-fdu/AIME_1983_2024` problem table at the check date [10], and the count of `prompt` strings fetched directly from `Deepseek-Math-RL-7B.json` also came to 933 [9]. No source states a token-length statistic for `prompt` or `completion`; not stated.

## Quality

- Each sampled completion carries a machine-checked `accuracy` boolean (whether the extracted `predict` answer matches the upstream `answer`), not a human or LLM-judge label [2][9].
- No source states a measured contamination, deduplication, or annotator-agreement rate for this release specifically.
- The upstream problem table's own card states, in its own words, "Disclaimer: This is a Benchmark dataset! Do not using in training!" and that it covers "AIME from year 1983~2023, and 2024(part 2)" [10]. This repository's own README does not repeat that disclaimer or otherwise restrict use, but it inherits the same problem set [2][10].
- Reading the upstream table's rows at offset 895-932 (the last 38 of 933 rows) directly confirms the tail is all `2023-I`/`2023-II`/`2024-II` problems, with `2024-II-1` through `2024-II-15` present except `2024-II-9` [10][11]. Together with rows 0-99, this covers 139 of the 933 rows (offsets 0-99 and 895-932); no `2024-I-*` row appears among those 139 read rows, but the remaining 795 rows (offsets 100-894) were not read, so their contents are not stated here [10][11]. `prompt[0]` fetched from `Deepseek-Math-RL-7B.json` matches the upstream row 0 (`1983-1`) question text word for word, and `answer[0]` in that file is `"60"`, matching the upstream row 0 answer, confirming problem-index alignment between this release and the upstream table [9][10].

## Load it

This repository has no `load_dataset`-compatible config (the viewer's own error is "dataset viewer is disabled") [5], so each file must be fetched and parsed as a single JSON object rather than loaded as rows:

```python
import json
from huggingface_hub import hf_hub_download

REV = "de743166007aa18bc7573202cc8624a86bae1404"  # main at the check date
path = hf_hub_download(
    "WNJXYK/AIME_1983_2024-Reasoning-Paths",
    "Deepseek-Math-RL-7B.json",
    repo_type="dataset",
    revision=REV,
)
data = json.load(open(path))  # one 402 MB JSON object; not a row-per-example file
```

**Trap**: `datasets.load_dataset("WNJXYK/AIME_1983_2024-Reasoning-Paths")` fails (the repo carries `viewer: false` and no dataset script or Parquet export) [2][5]; each of the 9 files must be downloaded and parsed individually, and a full file must be held in memory to reach the later fields (`cumulative_logprob`, `mean_logprob`, `prompt`, `accuracy`), since they sit after the large `completion` array in file order [9].

## Neighbors

Three sibling releases from the same author apply the same per-model, per-temperature JSON layout to other math benchmarks instead of AIME; their file listings were fetched directly at the check date [12]:

- `WNJXYK/MATH-Reasoning-Paths` - 8 JSON files over the MATH benchmark, covering the same 3 models but only 2 temperatures (1.0 and 1.1) for InternLM2-Math-Plus-1.8B, versus all 3 temperatures for the other two models.
- `WNJXYK/OlympiadBench-Reasoning-Paths` - the full 9-file, 3-model x 3-temperature layout over OlympiadBench.
- `WNJXYK/MathOdyssey-Reasoning-Paths` - the full 9-file, 3-model x 3-temperature layout over MathOdyssey.

None of these three overlaps this release's AIME problem set; they are a same-shape alternative when the training target is a different benchmark's problems, not a preferred substitute for this one. The problem-only upstream, `di-zhang-fdu/AIME_1983_2024`, is the source of the questions and answers here and carries no model completions [10]; its own card also names `AI-MO/aimo-validation-aime` as the separate source of "2024(part 1)" (AIME-I 2024), which this release and its upstream do not include [10].

## A row

The repository serves one shape across all 9 files (only the model and temperature differ), so one example, reconstructed from directly-fetched byte ranges of `Deepseek-Math-RL-7B.json` (`temperature=1.0`, `top_p=0.95`) at problem index 0, covers it [9]. `completion` is shown for 2 of the 256 samples; `predict`, `cumulative_logprob`, `mean_logprob`, and `accuracy` for 3:

```json
{
  "prompt": "Let $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_xw=24$ , $\\log_y w = 40$ and $\\log_{xyz}w=12$ . Find $\\log_zw$ .\nPlease reason step by step, and put your final answer within \\boxed{}.",
  "answer": "60",
  "predict": ["14", "40", "28", "..."],
  "completion": [
    "\n\nWe can use the properties of logarithms to solve this problem.\n\nFirst, recall that $\\log_a b = c$ is equivalent to $a^c = b$. So, we have $x^{24} = w$, $y^{40} = w$, and $(xyz)^{12} = w$. [...]",
    "..."
  ],
  "cumulative_logprob": [-157.153, -88.101, -85.090, "..."],
  "mean_logprob": [-0.0884, -0.0551, -0.0424, "..."],
  "temperature": 1.0,
  "top_p": 0.95,
  "accuracy": [false, false, false, "..."]
}
```

`accuracy` here is shown transposed to line up sample-by-sample with the other fields for problem index 0 (its native shape is `[256 samples][933 problems]`, so this is `accuracy[0][0]`, `accuracy[1][0]`, `accuracy[2][0]`) [9]. None of `predict[0][0]` ("14"), `predict[0][1]` ("40"), or `predict[0][2]` ("28") match `answer[0]` ("60"), consistent with `accuracy[0][0]`, `accuracy[1][0]`, and `accuracy[2][0]` all being `false`.

## Where it came from

The underlying 933 problems and their answers come from `di-zhang-fdu/AIME_1983_2024`, a CSV of AIME contest problems from 1983 through 2023 plus AIME-II 2024, itself sourced from the Art of Problem Solving wiki [10]. WNJXYK generated the reasoning paths in this repository by sampling 256 completions per problem from three existing, already math-tuned open LLMs - DeepSeek-Math-RL-7B, InternLM2-Math-Plus-1.8B, and InternLM2-Math-Plus-7B - at temperatures 1.0, 1.1, and 1.3, releasing the results for the NeurIPS 2025 paper's test-time-scaling analysis [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable, which is why Load it pins the revision.

[1] Zhou, Tan, Li, Yao, Guo, Li, Ma, "A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning", NeurIPS 2025. https://arxiv.org/abs/2510.15444 - the origin paper; current title read from the live abs page. Fetched 2026-08-11.

[2] WNJXYK/AIME_1983_2024-Reasoning-Paths dataset card (README). https://huggingface.co/datasets/WNJXYK/AIME_1983_2024-Reasoning-Paths/raw/main/README.md - overview, model list, 256-samples-per-problem and temperature description, JSON structure documentation, available-files table. Fetched 2026-08-11.

[3] The corpus screening row for `WNJXYK/AIME_1983_2024-Reasoning-Paths`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix, refined here against the row counts fetched directly from the upstream problem table. Checked 2026-08-11.

[4] Hugging Face Hub API record for WNJXYK/AIME_1983_2024-Reasoning-Paths. https://huggingface.co/api/datasets/WNJXYK/AIME_1983_2024-Reasoning-Paths?full=true - licence, gate, private flag, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=WNJXYK%2FAIME_1983_2024-Reasoning-Paths - returns "Not supported: dataset viewer is disabled in WNJXYK/AIME_1983_2024-Reasoning-Paths configuration." Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=WNJXYK%2FAIME_1983_2024-Reasoning-Paths - same "dataset viewer is disabled" error. Fetched 2026-08-11.

[7] Hugging Face paper page for arXiv:2510.15444. https://huggingface.co/papers/2510.15444 - links this dataset from the paper page; checked for, and found no, statements naming a model trained on it. Fetched 2026-08-11.

[8] Hugging Face Hub tree listing for WNJXYK/AIME_1983_2024-Reasoning-Paths at `main`. https://huggingface.co/api/datasets/WNJXYK/AIME_1983_2024-Reasoning-Paths/tree/main - per-file byte sizes and LFS oids. Fetched 2026-08-11.

[9] Direct byte-range fetches of `Deepseek-Math-RL-7B.json` from https://huggingface.co/datasets/WNJXYK/AIME_1983_2024-Reasoning-Paths/resolve/main/Deepseek-Math-RL-7B.json (the file is 402,531,371 bytes; the dataset viewer being disabled means this was read directly rather than through datasets-server): the first 8,000,001 bytes (covering `predict`, `answer`, and the start of `completion`), a further 6,000,001-byte range ending at byte 383,531,371 (covering `cumulative_logprob`), an 11,000,001-byte range ending at byte 392,531,371 (covering `mean_logprob`), and the final 10,000,000 bytes (covering `prompt`, `temperature`, `top_p`, and `accuracy`). Fetched 2026-08-11.

[10] di-zhang-fdu/AIME_1983_2024 dataset card (README) and datasets-server endpoints. https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=di-zhang-fdu%2FAIME_1983_2024 , https://datasets-server.huggingface.co/first-rows?dataset=di-zhang-fdu%2FAIME_1983_2024&config=default&split=train - the "do not use in training" disclaimer, the "1983~2023, and 2024(part 2)" description, the `AI-MO/aimo-validation-aime` pointer for 2024(part 1), the 933-row count, and rows 0-99. Fetched 2026-08-11.

[11] datasets-server rows endpoint for di-zhang-fdu/AIME_1983_2024, offsets 895 and 913, length 20 each, covering row indices 895-932 (the last 38 rows of 933). https://datasets-server.huggingface.co/rows?dataset=di-zhang-fdu%2FAIME_1983_2024&config=default&split=train&offset=895&length=20 (and offset=913) - used to confirm which AIME-2024 problems are present and which are absent. Fetched 2026-08-11.

[12] Hugging Face Hub API records for the 3 sibling repositories, full=true. https://huggingface.co/api/datasets/WNJXYK/MATH-Reasoning-Paths?full=true , https://huggingface.co/api/datasets/WNJXYK/OlympiadBench-Reasoning-Paths?full=true , https://huggingface.co/api/datasets/WNJXYK/MathOdyssey-Reasoning-Paths?full=true - `cardData` and each repository's `siblings` file listing, read directly to count and name the JSON files present in each. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for reasoning-trace SFT, conditional on decontamination: this release packages 933 AIME problems with 256 sampled model reasoning paths each across 9 model/temperature files, and 14 of those 933 problems are AIME-2024-II problems that overlap the standard aime24 evaluation set. A reader who will later score the trained model on aime24 must drop problem indices 919-932 (and their 256 sampled paths per file) or otherwise decontaminate, as stated in the opening paragraph and the Hold out line above; the screening row's own flag names this same contamination risk [3].

### The screening row

The row's own note [3]: "Sampled reasoning paths over the AIME 1983 to 2024 problem set from di-zhang-fdu/AIME_1983_2024, released with the NeurIPS 2025 RPC paper as per-model json files at several sampling temperatures; the viewer returns 501 for this layout." Its flag [3]: "contamination: it carries the AIME 2024 problems, the standard aime24 eval set, together with sampled solutions."
