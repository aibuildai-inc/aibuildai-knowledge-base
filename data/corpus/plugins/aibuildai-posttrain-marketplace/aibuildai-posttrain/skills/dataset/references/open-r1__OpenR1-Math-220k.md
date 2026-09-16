# open-r1/OpenR1-Math-220k

450,258 rows spread across three overlapping configs of DeepSeek R1 reasoning traces over competition math problems, with per-generation correctness labels and a ready-made SFT `messages` column.

**open-r1/OpenR1-Math-220k** is Hugging Face's open-r1 project's reasoning-trace dataset: problems come from AI-MO's NuminaMath 1.5 problem pool, and for each problem the project prompted DeepSeek R1 to generate two to four chain-of-thought solutions, verified with the Math-Verify tool for most samples and with Llama-3.3-70B-Instruct as an LLM judge for 12% of them, keeping only problems with at least one correct trace [1]. **The upstream NuminaMath 1.5 pool includes 5,872 `amc_aime`-sourced competition problems out of 896,215 total [2], and a live sample of 700 rows read across offsets 0-224,000 in this dataset's `all` config found the `amc_aime` source tag on 11 of them (1.6%) [3] — treat any `amc_aime`-tagged row as a decontamination risk against AMC/AIME-derived benchmarks (including AIME 2025) before a scored eval run.** The `default` and `extended` splits are disjoint by row count (93,733 + 131,396 = 225,129) and the `all` config is their union, so `all` does not add new problems beyond `default` and `extended` combined [4]. It lives at https://huggingface.co/datasets/open-r1/OpenR1-Math-220k .

**Use it for**: reasoning-trace SFT - the `messages` column is a ready two-turn (user problem, assistant `<think>...</think>` trace) chat list usable directly with a chat-SFT trainer; decontaminate `amc_aime`-sourced rows against AIME/AMC benchmarks first, per the restriction above. The `generations`/`correctness_math_verify` columns also let a trainer redo the selection (e.g. rejection sampling across the 2-4 generations per problem) or build DPO pairs from correct-vs-incorrect generations, matching what the dataset card describes as its intended flexibility [1]. See the SFT method card, and the DPO method card for the pairwise use.

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, repo tag `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [5]. The shortlist row's `licence_body_mentions` field lists both "Apache" and "MIT" [6]; the card body states only Apache-2.0 and does not define a second licence [1].

**Shape**: 3 configs (`default`, `all`, `extended`), each a single `train` split; 93,733 / 225,129 / 131,396 rows respectively, 450,258 rows served in total across all three [4][7].

**Hold out**: no source states a held-out eval split for this dataset itself. The contamination risk to hold out against is external: AMC/AIME-derived benchmark problems (e.g. AIME 2025), because the upstream problem pool draws on the `amc_aime` source category, as detailed above [2][3].

**Origin**: built by the Hugging Face open-r1 project; problems are human-authored competition/exercise items from NuminaMath 1.5, solutions are DeepSeek R1 model generations, correctness labels come from the Math-Verify rule-based checker and, for 12% of samples, Llama-3.3-70B-Instruct as judge [1]. Hub API at the check date: `downloads` 60,199, `downloadsAllTime` 481,333, `likes` 780 [5].

**Trained-on-by**: open-r1/OpenR1-Qwen-7B, a fine-tune of Qwen2.5-Math-7B-Instruct, trained on the `default` split for 3 epochs; its own card reports 90.6 on MATH-500, 47.0 on AIME 2024, 33.2 on AIME 2025, and 42.4 on GPQA-Diamond, against 51.3/35.8/52.4 for DeepSeek-R1-Distill-Qwen-7B on AIME 2024/2025/GPQA-D [8]. open-r1/Mixture-of-Thoughts repackages this dataset's `default` split unchanged (93,733 rows) as its `math` config for a multi-domain SFT mix [9].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and bytes per config, from the datasets-server size endpoint [4] and the card's own `dataset_info` [1]:

| config | rows | parquet bytes | in-memory bytes |
| --- | --- | --- | --- |
| `default` | 93,733 | 2,149,897,914 | 5,079,805,007 |
| `all` | 225,129 | 4,221,672,067 | 10,161,169,949 |
| `extended` | 131,396 | 2,063,936,457 | 4,770,393,404 |
| total served | 450,258 | 8,435,506,438 | 20,011,368,360 |

93,733 + 131,396 = 225,129, so `all` is exactly the union of `default` and `extended`; loading `all` on top of the other two configs does not add distinct problems, it duplicates them [4].

Every config shares the same 14 columns [1][7]: `problem`, `solution`, `answer`, `problem_type`, `question_type`, `source` (all strings); `uuid` (string); `is_reasoning_complete`, `correctness_math_verify`, `correctness_llama` (list of bool, one entry per generation); `generations`, `finish_reasons` (list of string); `correctness_count` (int64, count of correct generations); `messages` (list of `{content, role}` structs - the single selected chat trace). The card reports that fine-tuning on the 94k-problem `default` split gives the best downstream SFT performance, while `extended` adds sources such as `cn_k12` for more traces but yields lower SFT performance, which the card attributes to `cn_k12` questions being less difficult than the other sources [1]. No source states sequence-length or token statistics for this dataset; the card states only a 16k-token generation cap and that 75% of problems were solved in under 8k tokens [1].

## Quality

- A live sample of 700 rows read across seven offsets (0, 20,000, 50,000, 100,000, 150,000, 200,000, 224,000) in the `all` config found `correctness_llama` non-null on 86 of them (12.3%), matching the split between Math-Verify and Llama-judge verification stated in the opening paragraph [1][3].
- Source mix in the same 700-row sample: `cn_k12` 290, `olympiads` 287, `cn_contest` 61, `aops_forum` 40, `amc_aime` 11, `inequalities` 7, `olympiads_ref` 3, `number_theory` 1 [3]. This is a sample of `all`, not a full-split count, and does not by itself establish the mix for `default` or `extended` alone.
- The card states two or four R1 generations were produced per problem so that downstream users have room to filter or resample, including rejection sampling and DPO-style preference construction [1]. No source states a measured duplicate-problem rate.

## Load it

```python
import datasets

REV = "e4e141ec9dea9f8326f4d347be56105859b2bd68"  # main at the check date
default = datasets.load_dataset("open-r1/OpenR1-Math-220k", "default", revision=REV, split="train")   # 93,733 rows
extended = datasets.load_dataset("open-r1/OpenR1-Math-220k", "extended", revision=REV, split="train")  # 131,396 rows
```

**Trap**: the `all` config (225,129 rows) is the union of `default` and `extended` at the row-count level [4] - loading `all` together with either `default` or `extended` (or both) duplicates the same underlying problems into a training run rather than adding new data. Pick one of the three configs, not a combination.

## Neighbors

- `open-r1/OpenR1-Math-Raw` - the unfiltered precursor: 516,499 rows, one to eight raw R1 generations per problem, with `problem_is_valid`/`solution_is_valid` fields and no `messages` column, i.e. this dataset before the quality filtering and chat-formatting that produced OpenR1-Math-220k [10]. Prefer this only when you need the raw, unfiltered generation pool.
- `open-r1/Mixture-of-Thoughts` - a multi-domain SFT mixture (math, code, science); its `math` config has exactly 93,733 rows with `messages`, `num_tokens`, and `source` columns and matches `default`'s row count [11]. The open-r1 project's own successor model, OpenR1-Distill-7B, trains on this mixture rather than on OpenR1-Math-220k directly [8]. Prefer this corpus when training a single model across math, code, and science together; prefer OpenR1-Math-220k when math-only and you need the per-generation correctness/verification columns that Mixture-of-Thoughts drops.
- `AI-MO/NuminaMath-1.5` - the upstream problem-and-reference-solution pool this dataset draws its problems from, without R1 traces [2].

## A row

The three configs share one schema; `default`, `all`, and `extended` differ only in which rows they include, not in shape, so one row covers all three. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], with the full reasoning trace truncated:

```json
{
  "problem": "## Task B-1.3.\n\nA ship traveling along a river has covered $24 \\mathrm{~km}$ upstream and $28 \\mathrm{~km}$ downstream. For this journey, it took half an hour less than for traveling $30 \\mathrm{~km}$ upstream and $21 \\mathrm{~km}$ downstream, or half an hour more than for traveling $15 \\mathrm{~km}$ upstream and $42 \\mathrm{~km}$ downstream, assuming that both the ship and the river move uniformly.\n\nDetermine the speed of the ship in still water and the speed of the river.",
  "solution": "## Solution.\n\nLet $t$ be the time required for the boat to travel 24 km upstream and 28 km downstream [...] The speed of the river is $v_{R}=4$ km/h, and the speed of the boat is $v_{B}=10$ km/h.",
  "answer": "v_{R}=4\\mathrm{~}/\\mathrm{},v_{B}=10\\mathrm{~}/\\mathrm{}",
  "problem_type": "Algebra",
  "question_type": "math-word-problem",
  "source": "olympiads",
  "uuid": "586fd646-76d6-5070-8c81-9993ab9d8559",
  "is_reasoning_complete": [true, true],
  "generations": ["<think>\nOkay, so I need to find the speed of the ship in still water [...]", "..."],
  "correctness_math_verify": [true, false],
  "correctness_llama": null,
  "finish_reasons": null,
  "correctness_count": 1,
  "messages": [
    {"role": "user", "content": "## Task B-1.3.\n\nA ship traveling along a river has covered $24 \\mathrm{~km}$ upstream [...] Determine the speed of the ship in still water and the speed of the river."},
    {"role": "assistant", "content": "<think>\nOkay, so I need to find the speed of the ship in still water [...]"}
  ]
}
```

`correctness_llama` and `finish_reasons` are `null` here because this row's 2 generations were resolved by Math-Verify alone, not the LLM judge, matching the card's stated 12%-judge coverage [1][7].

## Where it came from

Built by the Hugging Face open-r1 project. Problems come from AI-MO/NuminaMath 1.5, whose card states it covers about 900k competition-level math problems gathered mainly from online exam-paper PDFs and math discussion forums, ranging from Chinese high-school exercises to US and international olympiad problems [2]. The open-r1 team prompted DeepSeek R1 to generate solutions for 400k of these problems using SGLang on 512 H100 GPUs, following DeepSeek R1's recommended generation parameters and prepending an instruction to box the final answer, with a 16k-token generation cap; generation and filtering code is linked from the dataset card to the open-r1 GitHub repository [1]. Two solutions were generated per problem, and four for some, to support rejection sampling and preference-optimization use [1]. Correctness was checked with the Math-Verify rule-based verifier for most samples and with Llama-3.3-70B-Instruct as an LLM judge for 12% of samples [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. The shortlist row's `commit` field (`e4e141ec9dea9f8326f4d347be56105859b2bd68`) matches the Hub API's current `sha` for `main` at the check date, so the pinned revision above reflects the live repository state as fetched.

[1] open-r1/OpenR1-Math-220k dataset card (README), including `cardData.dataset_info`. https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/raw/main/README.md - description, curation process, generation parameters, split descriptions, licence, per-config schema and row/byte counts. Fetched 2026-08-11.

[2] AI-MO/NuminaMath-1.5 dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-1.5/raw/main/README.md - dataset summary, source breakdown table (including the 5,872-row `amc_aime` count out of 896,215 total). Fetched 2026-08-11.

[3] datasets-server rows endpoint, config `all`, split `train`, sampled at offsets 0, 20000, 50000, 100000, 150000, 200000, 224000, length 100 each (700 rows total). https://datasets-server.huggingface.co/rows?dataset=open-r1%2FOpenR1-Math-220k&config=all&split=train - source-column distribution and `correctness_llama` non-null rate in this live sample. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=open-r1%2FOpenR1-Math-220k - per-config row counts and byte sizes. Fetched 2026-08-11.

[5] Hugging Face Hub API record for open-r1/OpenR1-Math-220k. https://huggingface.co/api/datasets/open-r1/OpenR1-Math-220k?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] The corpus screening row for `open-r1/OpenR1-Math-220k`, supplied with this card's request - its `licence_body_mentions`, `note`, and `flag` fields, read back in the row's own words where quoted in the appendix. Checked 2026-08-11.

[7] datasets-server first-rows/info endpoints. https://datasets-server.huggingface.co/first-rows?dataset=open-r1%2FOpenR1-Math-220k&config=default&split=train and https://datasets-server.huggingface.co/info?dataset=open-r1%2FOpenR1-Math-220k - column schema and the sampled row. Fetched 2026-08-11.

[8] open-r1/OpenR1-Qwen-7B model card (README). https://huggingface.co/open-r1/OpenR1-Qwen-7B/raw/main/README.md - training details (3 epochs on `default` split), and the MATH-500/AIME 2024/AIME 2025/GPQA-Diamond comparison table; note pointing to OpenR1-Distill-7B trained on Mixture-of-Thoughts. Fetched 2026-08-11.

[9] datasets-server size endpoint for open-r1/Mixture-of-Thoughts. https://datasets-server.huggingface.co/size?dataset=open-r1%2FMixture-of-Thoughts - per-config row counts, including the 93,733-row `math` config. Fetched 2026-08-11.

[10] open-r1/OpenR1-Math-Raw dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=open-r1%2FOpenR1-Math-Raw - row count, generation count range, schema. Fetched 2026-08-11.

[11] open-r1/Mixture-of-Thoughts dataset card (README, `cardData.dataset_info`). https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts/raw/main/README.md - per-config schema and row counts. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data, with a decontamination step required before any scored run against AMC/AIME-derived benchmarks. The screening row's note describes the dataset as NuminaMath-1.5 human problems with DeepSeek-R1 reasoning traces verified by Math-Verify, with all three configs (`default`/`all`/`extended`) usable for training [6]. Its flag names the specific risk this card details above: the upstream problem pool carries AMC/AIME competition items, which is a rule-risk calibration case for AIME 2025-style eval overlap [6].

### The screening row

The row's own note [6]: "NuminaMath-1.5 human problems with DeepSeek-R1 reasoning traces verified by Math-Verify; default/all/extended all training." Its flag [6]: "rule-risk: aime2025 - The ruled calibration case: R1 traces over NuminaMath-1.5 problems, whose pool carries AMC/AIME competition items."
