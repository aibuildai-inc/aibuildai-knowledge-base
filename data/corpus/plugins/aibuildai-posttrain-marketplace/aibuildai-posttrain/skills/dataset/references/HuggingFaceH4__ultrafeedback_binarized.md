# HuggingFaceH4/ultrafeedback_binarized

187,405 rows across six splits - preference pairs, SFT dialogues, and generation-only prompts, all derived from the same 61,135+2,000 UltraFeedback examples, built to train Zephyr-7B-beta.

**HuggingFaceH4/ultrafeedback_binarized** is a Hugging Face H4 team preprocessing of the openbmb UltraFeedback dataset, whose 64k prompts each carry four model completions scored by GPT-4 along criteria like helpfulness and honesty [1]. The origin paper is "UltraFeedback: Boosting Language Models with Scaled AI Feedback" [2]. For each prompt, the H4 team's `create_dataset.py` script keeps the completion with the highest `overall_score` as `chosen` and picks one of the remaining three at random (seed 42) as `rejected`, then derives `train_sft`/`test_sft` (the `chosen` dialogue only), `train_prefs`/`test_prefs` (the `chosen`/`rejected` pair), and `train_gen`/`test_gen` (the same pair but with the final assistant turn stripped from `messages`, for rejection sampling or PPO) [3]. **The dataset card documents an earlier revision (`292c16329d921287c4166934cac1a6ad1e13a6c5`) that had several hundred mislabeled completions and prompts sourced from TruthfulQA, contaminating public leaderboards; both are described as fixed in the current version, and the shortlisted commit here (`3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`) matches the Hub API's live `sha` for `main`, i.e. this is that fixed version** [1][4]. It lives at https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized . The build script itself filters out any row whose `source` field is `truthful_qa` or whose prompt matches a TruthfulQA validation question before scoring pairs [3].

**Use it for**: preference-pair training on `train_prefs` (DPO or reward modelling, using `chosen`/`rejected`), SFT on `train_sft` (using `messages`, which equals `chosen`), or generation/rejection-sampling/PPO prompting on `train_gen` (`messages` holds only the user turn) [1]. `chosen`/`rejected` are the implicit-prompt preference format (shared prefix embedded in both fields); `messages` in the `_sft` splits is chat-format SFT data; `messages` in the `_gen` splits is a prompt-only generation format. Maps to the DPO method card for `_prefs`, the SFT method card for `_sft`, and the PPO/rejection-sampling method card for `_gen`.

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated (`gated: false`, `private: false`) [4].

**Shape**: 187,405 rows, one config (`default`), six splits - `train_prefs` 61,135, `train_sft` 61,135, `train_gen` 61,135, `test_prefs` 2,000, `test_sft` 1,000, `test_gen` 1,000 [5][6].

**Hold out**: all three test splits - `test_prefs` (2,000), `test_sft` (1,000), `test_gen` (1,000). These are not 4,000 distinct rows: the build script forms `test_prefs` as the full 2,000-row held-out set from a seeded split, then bisects it again to get `test_sft` (one half, 1,000 rows) and derives `test_gen` from `test_sft` by stripping the last turn - so `test_sft` and `test_gen` rows are a subset of `test_prefs` [3]. No further contamination risk is stated beyond the TruthfulQA fix already resolved at this commit (see opening paragraph) [1][4].

**Origin**: built by the Hugging Face H4 team from openbmb/UltraFeedback; completions come from "a wide variety of open and proprietary models" and are scored by GPT-4, with the chosen/rejected split and the six-way split done by H4's own script [1][3]. Hub API at the check date: `downloads` 16,375, `downloadsAllTime` 699,387, `likes` 344 [4].

**Trained-on-by**: Zephyr-7B-beta - the dataset card states this dataset "was used to train Zephyr-7Β-β" [1], and the Zephyr paper "Zephyr: Direct Distillation of LM Alignment" names UltraFeedback as the source it binarizes into chosen/rejected preference pairs (highest mean score as chosen, one of the remainder as rejected) for its DPO training stage [7].

**Introduced by**: no paper for this specific binarized release - the dataset card [1]; the underlying UltraFeedback data is introduced by [2].

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train_prefs` | 61,135 |
| `train_sft` | 61,135 |
| `train_gen` | 61,135 |
| `test_prefs` | 2,000 |
| `test_sft` | 1,000 |
| `test_gen` | 1,000 |
| total | 187,405 |

One config, `default`, with seven columns shared across all splits (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `prompt` | string |
| `prompt_id` | string |
| `chosen` | list\<struct\<content: string, role: string\>\> |
| `rejected` | list\<struct\<content: string, role: string\>\> |
| `messages` | list\<struct\<content: string, role: string\>\> |
| `score_chosen` | float64 |
| `score_rejected` | float64 |

Download size (Parquet) is 649,967,196 bytes; in-memory size is 1,148,978,231 bytes across all splits [5]. No source states sequence-length or token statistics for this release.

## Quality

- The chosen/rejected preference label is a deterministic pick from GPT-4's per-completion `overall_score`: `chosen` is the argmax completion, `rejected` is a uniformly random draw (seed 42) from the remaining three [3]. No source states a separate human-agreement or contamination-rate measurement for this specific binarized release.
- The dataset card reports, without a numeric rate, that "a few hundred completions" had an incorrect label and that some prompts were sourced from TruthfulQA, causing leaderboard contamination; both are stated as fixed as of the current version [1].
- The build script explicitly filters out any row from the `truthful_qa` source and any prompt matching a TruthfulQA validation question, before pairwise scoring [3].
- Of the 27 `train_prefs` rows and 36 `train_gen` rows served at offset 0 by datasets-server's first-rows endpoint, `score_chosen` equalled `score_rejected` in 4/27 `train_prefs` rows and 6/36 `train_gen` rows [8]; this is consistent with the script's tie-break logic (`rejected` is drawn at random and can share the top score with `chosen`), but this read covers only those first rows of two of the six splits, not the full dataset, so no wider tie rate is claimed [3][8].
- No served row required an external fetch; all columns (`prompt`, `chosen`, `rejected`, `messages`, scores) are self-contained.

## Load it

Load a single split (each of the six is independently loadable). Pin the revision to the Hub API's `sha` for `main` at the check date, which is also the shortlisted commit [4]. The Shape, Quality, and A-row numbers above come from datasets-server's `/size`, `/info`, and `/first-rows` endpoints, which take no effective revision parameter and always serve whatever is on the default branch [5][6][8]; they match this pinned commit only because `main` has not moved since it (last modified 2024-10-16, per the Hub API [4]) - they are live reads, not reads scoped to the commit hash below, and would go stale if `main` advances:

```python
import datasets

REV = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"  # main at the check date
train_prefs = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", revision=REV, split="train_prefs")  # 61,135 rows
train_sft   = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", revision=REV, split="train_sft")    # 61,135 rows
train_gen   = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", revision=REV, split="train_gen")    # 61,135 rows
```

**Trap**: `messages` means different things in different splits, but the schema does not say so - in `_sft` splits `messages` equals `chosen` (a full two-turn dialogue), while in `_gen` splits `messages` has had its final assistant turn stripped (a one-turn, prompt-only list), so code written for one split will silently under- or over-run on another [3][8]. Also, the dataset card explicitly documents an older revision (`292c16329d921287c4166934cac1a6ad1e13a6c5`) with the mislabeled/TruthfulQA-contaminated rows still present; do not pass that revision unless deliberately reproducing the old, uncleaned data [1].

## Neighbors

All row counts below were read live at the check date [9].

- `openbmb/UltraFeedback` - the upstream source: 63,967 rows, one prompt with up to four scored completions each, before binarization into pairs [1][9]. This repository is the H4 team's own binarized derivative and is preferred over the raw upstream when the goal is ready-made preference/SFT/generation splits rather than the full four-way completion set.
- `argilla/ultrafeedback-binarized-preferences-cleaned` - a separate binarization by Argilla, 60,917 rows in a single `train` split, which removes TruthfulQA- and ShareGPT-sourced contaminated rows via a left join against `truthful_qa`, and binarizes using the average of the individual preference ratings rather than the critique `overall_score` this repository uses [10]. Its own card names this repository as one of the formats it aligned to [10].
- `allenai/ultrafeedback_binarized_cleaned` - AllenAI's cleaned binarization, following the same six-split naming as this repository but with fewer rows after decontamination: `train_prefs`/`train_sft`/`train_gen` 60,829 each (versus 61,135 here), `test_prefs` 1,964 (versus 2,000 here), and `test_sft`/`test_gen` 985 each (versus 1,000/1,000 here) [9]. Argilla's card credits AllenAI with first identifying the TruthfulQA contamination that this repository's current revision also fixes [10].

This repository is preferred over both cleaned neighbors when working from H4's own scoring convention and split shapes (used to train Zephyr-7B-beta); reach for the Argilla or AllenAI releases only if a different decontamination or binarization method is specifically wanted, since mixing this repository's `train_prefs` with either neighbor's train split duplicates most of the same underlying UltraFeedback prompts.

## A row

Two distinct served shapes exist. `train_prefs`/`train_sft`/`test_prefs`/`test_sft` share one shape (`messages` equals `chosen`, a full two-turn dialogue); `train_gen`/`test_gen` share a second shape (`messages` holds only the user turn). One row from each, fetched live (datasets-server `/first-rows`) [8], with long fields truncated:

`config="default"`, `split="train_prefs"`, `row_idx=0`:

```json
{
  "prompt": "how can i develop a habit of drawing daily",
  "prompt_id": "086b3e24f29b8956a01059f79c56db35d118a06fb6b844b095737d042795cd43",
  "chosen": [
    {"content": "how can i develop a habit of drawing daily", "role": "user"},
    {"content": "Developing a daily habit of drawing can be challenging but with consistent practice and a few tips... [truncated]", "role": "assistant"}
  ],
  "rejected": [
    {"content": "how can i develop a habit of drawing daily", "role": "user"},
    {"content": "As an AI language model, I cannot personally develop habits for you. But, here are some tips... [truncated]", "role": "assistant"}
  ],
  "messages": [
    {"content": "how can i develop a habit of drawing daily", "role": "user"},
    {"content": "Developing a daily habit of drawing can be challenging but with consistent practice and a few tips... [truncated, identical to chosen]", "role": "assistant"}
  ],
  "score_chosen": 8.5,
  "score_rejected": 8.5
}
```

`config="default"`, `split="train_gen"`, `row_idx=0` (same prompt; `chosen`/`rejected` identical in shape to above and omitted here, `messages` differs):

```json
{
  "prompt": "how can i develop a habit of drawing daily",
  "messages": [
    {"content": "how can i develop a habit of drawing daily", "role": "user"}
  ],
  "score_chosen": 8.5,
  "score_rejected": 8.5
}
```

## Where it came from

The Hugging Face H4 team built this dataset from `openbmb/UltraFeedback` (revision `40b436560ca83a8dba36114c22ab3c66e43f6d5e`), whose prompts are paired with completions from "a wide variety of open and proprietary models" and scored by GPT-4 on criteria including helpfulness and honesty [1]. The H4 build script first drops any row whose `source` field is `truthful_qa` or whose prompt text matches a TruthfulQA (`generation` or `multiple_choice`) validation question, to prevent leaderboard contamination; it then picks the completion with the highest `overall_score` as `chosen` and a uniformly random remaining completion (seed 42) as `rejected`; it splits off a 2,000-row test set (seed 42), bisects that into `test_sft` and the second half of `test_prefs`; and it derives the `_gen` splits from the `_sft` splits by removing the final assistant turn [3]. The dataset was used to train Zephyr-7B-beta [1][7].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] HuggingFaceH4/ultrafeedback_binarized dataset card (README). https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/raw/main/README.md - description, TruthfulQA/mislabel fix note, old-revision hash, split table, usage note, Zephyr training claim. Fetched 2026-08-11.

[2] Cui et al., "UltraFeedback: Boosting Language Models with Scaled AI Feedback", 2023. https://arxiv.org/abs/2310.01377 - the origin paper for the upstream UltraFeedback data; current title read from the live abs page (differs from the README's citation-block title). Fetched 2026-08-11.

[3] `create_dataset.py`, the build script referenced by the dataset card. https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/raw/main/create_dataset.py - TruthfulQA filtering, chosen/rejected selection logic, split construction (`train_test_split` seeds, `test_prefs`/`test_sft`/`test_gen` derivation), `messages` field semantics per split. Fetched 2026-08-11.

[4] Hugging Face Hub API record for HuggingFaceH4/ultrafeedback_binarized. https://huggingface.co/api/datasets/HuggingFaceH4/ultrafeedback_binarized?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=HuggingFaceH4%2Fultrafeedback_binarized - this endpoint takes no effective revision parameter (confirmed by passing a garbage revision string and diffing the response byte-for-byte against the unpinned call: identical), so it always serves whatever is on the default branch, not a pinned commit. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=HuggingFaceH4%2Fultrafeedback_binarized - same live, unpinned behavior as [5]. Fetched 2026-08-11.

[7] Tunstall et al., "Zephyr: Direct Distillation of LM Alignment", 2023. https://arxiv.org/abs/2310.16944 - names UltraFeedback as the preference source binarized (highest mean score as chosen) for Zephyr-7B-beta's DPO stage. Current title read from the live abs page; full text read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2310.16944), since the abs page alone does not mention UltraFeedback. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint, called once per split needed (`train_prefs`, `train_sft`, `train_gen`, `test_prefs`, `test_sft`, `test_gen`). https://datasets-server.huggingface.co/first-rows?dataset=HuggingFaceH4%2Fultrafeedback_binarized&config=default&split=<split> - same live, unpinned behavior as [5]: a call with a garbage revision string returned the identical first row. Fetched 2026-08-11.

[9] datasets-server size endpoint, one call per neighbor: `openbmb/UltraFeedback`, `allenai/ultrafeedback_binarized_cleaned`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[10] argilla/ultrafeedback-binarized-preferences-cleaned dataset card (README) and Hub API record. https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned/raw/main/README.md and https://huggingface.co/api/datasets/argilla/ultrafeedback-binarized-preferences-cleaned?full=true - cleaning method, TruthfulQA/ShareGPT contamination removal, credit to AllenAI, row count. Fetched 2026-08-11.

[11] The corpus screening row for `HuggingFaceH4/ultrafeedback_binarized`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable, across all three training shapes, with the corresponding test split held out for each: `train_prefs` for preference/DPO training against `test_prefs`, `train_sft` for SFT against `test_sft`, `train_gen` for generation/rejection-sampling/PPO against `test_gen`. The screening row's note states the dataset provides "64k prompts with completions from many open and proprietary models scored by GPT-4" and that the three train splits are trainable while the three test splits are held out [11], which matches the split-level detail established above from the build script [3].

### The screening row

The row's own note [11]: "64k prompts with completions from many open and proprietary models scored by GPT-4; train_sft/train_prefs/train_gen are trainable, test_sft/test_prefs/test_gen are held out." The row carries no flag.
