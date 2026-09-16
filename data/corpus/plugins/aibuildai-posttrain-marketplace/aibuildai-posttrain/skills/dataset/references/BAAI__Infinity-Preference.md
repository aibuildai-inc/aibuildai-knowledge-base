# BAAI/Infinity-Preference

59,438 bilingual (English/Chinese) preference pairs - a `prompt` plus a `chosen` and `rejected` two-turn conversation each - sampled from a single policy model and scored by a reward model with task-specific weighting.

**BAAI/Infinity-Preference** packages one instruction per task type, drawn evenly from BAAI's own Infinity-Instruct instruction pool and its capability-labelling system, together with a preference pair sampled from Gemma-2-9B-IT for that instruction; the pair is scored with ArmoRM under attribute weights tuned per task category [1]. No paper is cited on the card - the dataset card is the only source [1]. It lives at https://huggingface.co/datasets/BAAI/Infinity-Preference . **The card's own disclaimer restricts the resources, including this data, to academic research and states they "cannot be used for commercial purposes" [1], which conflicts with the `apache-2.0` licence tag also carried on the repository [2]. Separately, because prompts are sampled from Infinity-Instruct, and the corpus screening for this row flags that Infinity-Instruct's own source table lists TIGER-Lab/MathInstruct (329,254 rows) as carrying MATH competition-style problems, math-category prompts in this dataset carry an unverified overlap risk with AIME/AMC-style benchmarks - decontaminate math-category rows before any scored math evaluation [3].**

**Use it for**: preference-pair training (DPO, SimPO, or a reward model) - never SFT on the chosen side alone, since the card documents no SFT use and every row is a paired comparison, not a single approved response [1]. Each row is the conversational preference format with an implicit prompt: `chosen` and `rejected` are each a two-turn `[{"role": "user", ...}, {"role": "assistant", ...}]` list sharing the same user turn and differing only in the assistant reply, exactly the "implicit prompt" conversational shape TRL documents, and a separate top-level `prompt` string column duplicates the user turn as plain text [4]. TRL's DPOTrainer recommends the explicit-prompt shape, so apply `extract_prompt()` to `chosen`/`rejected` first, as the DPO method card describes [4]. Respect the research-only restriction above.

**Licence**: `apache-2.0` per the repository's `license` tag [2], but the card's disclaimer restricts the data to "academic research purposes only" and bars commercial use [1] - a direct conflict between the SPDX tag and the card's own text; treat the restriction as binding.

**Shape**: 59,438 rows in one config (`default`), split `train` 59,338 / `test` 100, five columns [5][6].

**Hold out**: the `test` split (100 rows) [5]. Separately, hold out or decontaminate any `mathematical_calculation`-category rows before a scored math benchmark run, per the corpus screening flag on Infinity-Instruct's math source (see Quality) [3].

**Origin**: built by BAAI; `chosen`/`rejected` are Gemma-2-9B-IT generations, and the preference label is an automated ArmoRM reward-model score under per-task attribute weights, not a human judgment [1]. Hub API at the check date: 207 downloads, 81 likes [2].

**Trained-on-by**: BAAI's own `BAAI/Gemma2-9B-IT-Simpo-Infinity-Preference`, whose model card states it is "finetuned on Infinity-Preference with Simpo," reporting 73.4% LC win-rate on AlpacaEval 2.0 and 58.1% win-rate on Arena-Hard [7]. A community model, `chenyongxi/Qwen2.5-1.5B-SFT-DPO-InfinityPreference`, declares this dataset in its front matter and states it is a DPO fine-tune "on the [BAAI/Infinity-Preference] dataset," trained with TRL [8].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 59,338 |
| `test` | 100 |
| total | 59,438 |

One config, `default`, with five columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `task_category` | string |
| `prompt` | string |
| `chosen` | list<struct<content: string, role: string>> |
| `rejected` | list<struct<content: string, role: string>> |
| `id` | int64 |

No source states sequence-length or token statistics for this release; "not stated" here.

## Quality

- The preference label on every row is an automated ArmoRM reward-model score, adjusted by per-task-category attribute weights, not a human annotation [1]. No source states a measured agreement rate, contamination rate, or duplicate rate for this release.
- The card's disclaimer states model output "is influenced by uncontrollable variables such as randomness" and that "the accuracy of the output cannot be guaranteed" [1].
- Contamination risk: this release samples its prompts from BAAI's Infinity-Instruct instruction pool [1]. Infinity-Instruct is currently gated on the Hub and its source-composition table could not be fetched for this card [9]. The corpus screening flag for this row states that Infinity-Instruct's source table lists TIGER-Lab/MathInstruct (329,254 rows) as carrying MATH competition-style problems, and marks this a rule-risk against AIME 2025-style benchmarks [3]. Of the first 38 `train` rows and first 36 `test` rows read at offset 0, 2 train rows and 1 test row carry `task_category: "mathematical_calculation"` [10]; one such row, sampled below, is a grade-school arithmetic word problem, not a competition problem [10]. This does not resolve the flag - Infinity-Instruct's own source table was not readable to confirm or rule out AIME/AMC-specific overlap, so the risk stands as flagged and decontamination of math-category rows is the safe default before a scored math eval.

## Load it

Train on `train`, hold out `test`, and pin the revision to `load_dataset` for reproducibility (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-08-30) [2]. The revision pin covers what this call loads; the row counts and sampled row on this card were read from the datasets-server `/size`, `/info`, and `/first-rows` endpoints, which take no revision parameter and were queried live against `main` on the check date, not against the pinned commit [5][6][10]:

```python
import datasets

REV = "2a68f42d60b1a315ea6d82cf294e4ee27b41c3a3"  # main at the check date
train = datasets.load_dataset("BAAI/Infinity-Preference", split="train", revision=REV)  # 59,338 rows
test = datasets.load_dataset("BAAI/Infinity-Preference", split="test", revision=REV)    # 100 rows - hold out
```

**Trap**: `chosen` and `rejected` are two-turn conversations, not bare completions - naively treating them as prompt-completion text (e.g. concatenating for SFT) trains on duplicated user turns and discards the pairing that makes this a preference dataset. Extract the prompt with TRL's `extract_prompt()` before feeding a trainer that expects the explicit-prompt shape [4].

## Neighbors

- `BAAI/Infinity-Instruct` - the instruction pool this dataset's prompts are sampled from; it is gated on the Hub and its README could not be fetched for this card, so its own source-composition table (referenced in the screening flag) could not be independently verified here [9]. Not a preference dataset itself - it is the upstream instruction source, not a substitute.
- `BAAI/Gemma2-9B-IT-Simpo-Infinity-Preference` - a released SimPO checkpoint trained on this dataset, not a data neighbor, but the card's own evidence that the pairs are usable for preference optimization [7].
- No other re-release, cleaned, or binarized sibling of this dataset was found.

## A row

The repository serves one config and one schema, so one row covers it. From `config="default"`, `split="train"`, `row_idx=9` - a `mathematical_calculation`-category row, chosen for relevance to the contamination flag above (datasets-server `/first-rows`) [10]:

```json
{
  "task_category": "mathematical_calculation",
  "id": 17,
  "prompt": "Angela wants to compare her math homework answers with her friends, but not all of them have completed their assignments. Out of 20 problems, Martha has finished 2, Jenna has finished four times the number of problems Martha did minus 2, and Mark has finished half the number of problems Jenna did. If none of the friends worked on any of the same problems, how many problems have not been completed by anyone except Angela?",
  "chosen": [
    {"role": "user", "content": "Angela wants to compare her math homework answers with her friends, but not all of them have completed their assignments. Out of 20 problems, Martha has finished 2, Jenna has finished four times the number of problems Martha did minus 2, and Mark has finished half the number of problems Jenna did. If none of the friends worked on any of the same problems, how many problems have not been completed by anyone except Angela?"},
    {"role": "assistant", "content": "Here's how to solve the problem step-by-step:\n\n* **Jenna's problems:** Jenna finished 4 * 2 - 2 = 6 problems.\n* **Mark's problems:** Mark finished 6 / 2 = 3 problems.\n* **Total completed problems:** Together, they finished 2 + 6 + 3 = 11 problems.\n* **Uncompleted problems:**  There are 20 - 11 = 9 problems not completed.\n\n\n**Answer:** 9 problems have not been completed by anyone except Angela."}
  ],
  "rejected": [
    {"role": "user", "content": "Angela wants to compare her math homework answers with her friends, but not all of them have completed their assignments. Out of 20 problems, Martha has finished 2, Jenna has finished four times the number of problems Martha did minus 2, and Mark has finished half the number of problems Jenna did. If none of the friends worked on any of the same problems, how many problems have not been completed by anyone except Angela?"},
    {"role": "assistant", "content": "Here's how to solve the problem step-by-step:\n\n**1. Find Jenna's completed problems:**\n\n* Martha finished 2 problems.\n* Jenna finished four times that minus 2: (4 * 2) - 2 = 6 problems.\n\n**2. Find Mark's completed problems:**\n\n* Jenna finished 6 problems.\n* Mark finished half of Jenna's problems: 6 / 2 = 3 problems. [...]"}
  ]
}
```

## Where it came from

Built by BAAI. Prompts are sampled evenly across task types from Infinity-Instruct's instruction set, using Infinity-Instruct's own capability-labelling system to stratify the sample [1]. For each sampled instruction, a preference pair is generated by sampling from Gemma-2-9B-IT, then scoring the pair with ArmoRM under attribute weights tuned per task category [1]. The card states the code for building the task-specific weights was not yet released as of the card's last update [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. That pin covers only what `load_dataset(..., revision=REV)` reads. The datasets-server `/size`, `/info`, and `/first-rows` endpoints used for the Shape, Quality, and A-row numbers above [5][6][10] take no `revision` parameter and were called against the live `main` branch, not against the pinned commit; they are reported as true only as of the check date, 2026-08-12, and are not covered by the revision pin.

[1] BAAI/Infinity-Preference dataset card (README). https://huggingface.co/datasets/BAAI/Infinity-Preference/raw/main/README.md - description, disclaimer, generation and scoring method. Fetched 2026-08-12.

[2] Hugging Face Hub API record for BAAI/Infinity-Preference. https://huggingface.co/api/datasets/BAAI/Infinity-Preference?full=true - licence tag, gate status, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-12.

[3] The corpus screening row for `BAAI/Infinity-Preference`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-12.

[4] trl dataset formats documentation. https://huggingface.co/docs/trl/main/en/dataset_formats - conversational preference format with implicit prompt, `extract_prompt()`, DPOTrainer's expected type. A `main` build, unpinned and mutable. Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=BAAI%2FInfinity-Preference Fetched 2026-08-12.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=BAAI%2FInfinity-Preference Fetched 2026-08-12.

[7] BAAI/Gemma2-9B-IT-Simpo-Infinity-Preference model card (README). https://huggingface.co/BAAI/Gemma2-9B-IT-Simpo-Infinity-Preference/raw/main/README.md - states it is finetuned on Infinity-Preference with SimPO, and reports AlpacaEval 2.0 and Arena-Hard scores. Fetched 2026-08-12.

[8] chenyongxi/Qwen2.5-1.5B-SFT-DPO-InfinityPreference model card (README), found via the Hub model-search API (https://huggingface.co/api/models?search=Infinity-Preference). https://huggingface.co/chenyongxi/Qwen2.5-1.5B-SFT-DPO-InfinityPreference/raw/main/README.md - front-matter `datasets: BAAI/Infinity-Preference` and prose stating it is a DPO fine-tune on this dataset, trained with TRL. Fetched 2026-08-12.

[9] BAAI/Infinity-Instruct README fetch attempt. https://huggingface.co/datasets/BAAI/Infinity-Instruct/raw/main/README.md - returned "Access to dataset BAAI/Infinity-Instruct is restricted. You must have access to it and be authenticated to access it," confirming the repository is gated at the check date; its Hub API record's `gated` field reads `"auto"` [https://huggingface.co/api/datasets/BAAI/Infinity-Instruct?full=true]. The source-composition table the screening flag [3] refers to could not be independently read. Fetched 2026-08-12.

[10] datasets-server first-rows endpoint, `train` and `test` splits. https://datasets-server.huggingface.co/first-rows?dataset=BAAI%2FInfinity-Preference&config=default&split=train and https://datasets-server.huggingface.co/first-rows?dataset=BAAI%2FInfinity-Preference&config=default&split=test - task-category counts over the first 38 `train` and first 36 `test` rows served, and the sampled row. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as preference-pair data, with two restrictions carried into the card body: the disclaimer's academic-research-only, non-commercial restriction, which sits in tension with the repository's `apache-2.0` tag [1][2], and the screening flag's math-benchmark contamination risk, inherited from Infinity-Instruct's source composition and not resolvable from sources readable for this card [3][9].

### The screening row

The row's own note [3]: "preference pairs sampled from Gemma-2-9B-IT, scored by ArmoRM with per-task weights." The row's flag [3]: "rule-risk: aime2025 - Prompts sampled from Infinity-Instruct, whose source table lists TIGER-Lab/MathInstruct (329,254 rows) carrying MATH AMC/AIME problems."
