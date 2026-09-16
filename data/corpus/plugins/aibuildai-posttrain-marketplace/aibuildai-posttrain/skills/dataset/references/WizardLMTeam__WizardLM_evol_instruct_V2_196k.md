# WizardLMTeam/WizardLM_evol_instruct_V2_196k

143,000 single-turn instruction/response conversations - a mixture of Evol-Instruct-evolved Alpaca prompts and evolved ShareGPT logs, with ChatGPT-generated answers - served as one `train` split under a `conversations` (ShareGPT-style) chat column.

**WizardLMTeam/WizardLM_evol_instruct_V2_196k** is the WizardLM team's training data release, built on top of the Evol-Instruct method introduced in "WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions" [1]: an LLM (OpenAI's ChatGPT, gpt-3.5-turbo) rewrites seed instructions step by step into more complex or more diverse variants and then generates the corresponding response for each evolved instruction [1]. The dataset's own card describes this repository as "the latest optimized version of Evol-Instruct training data of WizardLM model" and states it is "143K mixture evolved data of Alpaca and ShareGPT" [2]. It lives at https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k .

**The repository name promises 196k rows, but only 143,000 are served here; the card says the remaining rows require merging with the original ShareGPT dataset because of a data usage licence restriction, to reach "around 196k rows of data" [2], and gives no join key or script for that merge - so the extra ~53k rows are not something a reader can recover from this source alone.**

**Use it for**: SFT on chat-formatted instruction/response pairs (the SFT method card). The `conversations` column uses the ShareGPT-style schema - a list of `{"from": ..., "value": ...}` turns with `human`/`gpt` roles - not the `role`/`content` messages format that trainers built around trl's conversational format expect [3]; remap `from`/`value` and `human`/`gpt` to `role`/`content` and `user`/`assistant` before use.

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated (`"gated": false`, `"private": false`) [4]. The one catch: MIT covers only this repository's 143,000 rows; the card's own instruction to merge in the original ShareGPT dataset to reach ~196k rows exists specifically because of a data usage licence restriction [2], and that companion dataset carries its own Apache-2.0 licence [5].

**Shape**: 143,000 rows, one config (`default`), one split (`train`), two columns (`idx` string, `conversations` list of `{from, value}` structs) [6][7].

**Hold out**: Nothing - the repository ships a single `train` split, and no source fetched for this card states or implies overlap with an evaluation set.

**Origin**: built and released by the WizardLM team (WizardLMTeam); the evolved instructions and generated answers both come from OpenAI's ChatGPT (gpt-3.5-turbo) per the Evol-Instruct pipeline [1]. Hub API at the check date: `downloads` 4,097, `downloadsAllTime` 60,260, `likes` 250 [4][8].

**Trained-on-by**: the dataset's own card states it is the training data of the WizardLM model line [2]; the original Evol-Instruct paper's WizardLM models are fine-tuned on this evolved-instruction data (its own 70k sample, the predecessor of this V2 release) [1]. No other model's card was confirmed here to name this exact V2 196k repository.

**Introduced by**: the WizardLM paper introduces the Evol-Instruct method and its original 70k evolved-Alpaca sample [1]; this specific V2, 196k-scale, Alpaca+ShareGPT mixture is not described in that paper - no paper covers it, only the dataset card [2].

## Shape

Rows and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 143,000 |

One config, `default`, two columns (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `idx` | string |
| `conversations` | list\<struct\<from: string, value: string\>\> |

Sizes (datasets-server `/size`) [6]: 374,925,619 bytes of original JSON download, 162,285,753 bytes as Parquet, 336,773,957 bytes decoded in memory. No source fetched for this card states sequence-length or token statistics for this release.

## Quality

- Every one of 80 rows read across four 20-row offsets spanning the full split - offsets 0, 50,000, 100,000, and 142,900 (datasets-server `/rows`) - carried exactly two conversation turns, one `human` and one `gpt` [9]. This card makes no claim about rows outside those four windows.
- Of those same 80 rows, `idx` values split between two visibly distinct forms: an `alpaca_<number>` form (31 of 80 rows sampled) and a short alphanumeric string with no prefix (49 of 80) [9]. This matches the card's own description of a mixture of evolved-Alpaca and evolved-ShareGPT data [2], but no source states the exact provenance split for the full 143,000 rows.
- No source fetched for this card states a measured contamination rate, duplicate rate, or annotator/model-agreement figure for this release.
- The origin paper describes an "elimination evolving" filter step that drops instructions judged to have failed to evolve - no information gain over the seed instruction, or a response too difficult for the LLM to generate, detected in part by response length and the presence of "sorry" - before instructions and responses are kept in the training pool [1]. Whether that filter was applied to the ShareGPT-derived half of this specific release is not stated.

## Load it

```python
import datasets

REV = "8a7d15a83028b5c93915677704f492839e2675f6"  # main at the check date
train = datasets.load_dataset(
    "WizardLMTeam/WizardLM_evol_instruct_V2_196k", revision=REV, split="train"
)  # 143,000 rows
```

**Trap**: this loads only the 143,000 rows served by this repository, not the "196k" the repository name implies. The card's own instructions describe merging in the original ShareGPT dataset to approach 196k rows [2], but give no row-level key or script to do so - `idx` values on the non-Alpaca rows are short opaque strings with no stated correspondence to any field in `anon8231489123/ShareGPT_Vicuna_unfiltered`, so that merge is not something this card's sources make actionable.

## Neighbors

- `WizardLMTeam/WizardLM_evol_instruct_70k` - the predecessor release: 70,000 rows of Evol-Instruct-evolved Alpaca only (no ShareGPT mixture), in a flat `instruction`/`output` schema rather than this release's `conversations` list, also MIT-licensed [10][11]. Its own card states it is "the training data of WizardLM" [10].
- `anon8231489123/ShareGPT_Vicuna_unfiltered` - the original, non-evolved ShareGPT conversation logs this card says to merge in to approach 196k rows; Apache-2.0 licensed [5][2].
- `QuixiAI/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split` - a community repository (previously published under the `ehartford`/`cognitivecomputations` accounts, which now redirect to it) that serves 153,908 rows in the same `conversations`/`from`/`value` schema plus extra per-turn `markdown` fields, closer to but still short of 196k [12][13]. It ships no README and no licence tag, so its filtering and merge process are not documented by any source fetched for this card [12].

This corpus prefers the present repository over `QuixiAI/...merged_split` for its documented, MIT-licensed provenance; a reader who wants closer to the full 196k and can tolerate an undocumented merge may look at the QuixiAI repository instead, but should not mix the two into one training run since their row counts and merge logic do not line up.

## A row

The repository serves one config and one schema. From `config="default"`, `split="train"`, `row_idx=17` (datasets-server `/first-rows`) [9]:

```json
{
  "idx": "alpaca_27566",
  "conversations": [
    {"from": "human", "value": "Give an example of a salty food item"},
    {"from": "gpt", "value": "Potato chips."}
  ]
}
```

## Where it came from

Built and released by the WizardLM team. The underlying method, Evol-Instruct, starts from seed instructions and uses an LLM - OpenAI's ChatGPT, gpt-3.5-turbo, accessed via the OpenAI API - to rewrite each instruction into a more complex or more diverse variant across four evolution epochs, then uses the same model to generate the response to each evolved instruction, feeding the instruction into a ChatGPT request and parsing the returned text as the answer [1]. An elimination-evolving filter step removes instructions that failed to meaningfully evolve or that produced responses signalling the model could not follow them [1]. This repository's own card describes its 143,000 rows as a mixture of that process applied to Alpaca's seed instructions and to ShareGPT logs, and states that the complete ~196k-row training set required merging in the original ShareGPT dataset separately, for licensing reasons [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11. Hugging Face Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] "WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions." https://arxiv.org/abs/2304.12244 - Evol-Instruct method, ChatGPT/gpt-3.5-turbo as the evolving and generating model, four evolution epochs, elimination-evolving filter; current title read from the live abs page; body text read via the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2304.12244. Fetched 2026-08-11.

[2] WizardLMTeam/WizardLM_evol_instruct_V2_196k dataset card (README). https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k/raw/main/README.md - "143K mixture evolved data of Alpaca and ShareGPT", "latest optimized version" description, ShareGPT merge instruction and ~196k figure. Fetched 2026-08-11.

[3] trl dataset formats documentation. https://huggingface.co/docs/trl/main/en/dataset_formats - conversational format defined as `role`/`content` message lists. A `main` build, unpinned and mutable. Fetched 2026-08-11.

[4] Hugging Face Hub API record for WizardLMTeam/WizardLM_evol_instruct_V2_196k. https://huggingface.co/api/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k?full=true - licence, gate, `sha`, `downloads`, `likes`. Fetched 2026-08-11.

[5] anon8231489123/ShareGPT_Vicuna_unfiltered Hub API record. https://huggingface.co/api/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered?full=true - `cardData.license` is `"apache-2.0"`. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=WizardLMTeam%2FWizardLM_evol_instruct_V2_196k Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=WizardLMTeam%2FWizardLM_evol_instruct_V2_196k Fetched 2026-08-11.

[8] Hugging Face Hub API record for WizardLMTeam/WizardLM_evol_instruct_V2_196k with `downloadsAllTime` expanded. https://huggingface.co/api/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k?expand[]=downloadsAllTime Fetched 2026-08-11.

[9] datasets-server first-rows and rows endpoints. https://datasets-server.huggingface.co/first-rows?dataset=WizardLMTeam%2FWizardLM_evol_instruct_V2_196k&config=default&split=train and https://datasets-server.huggingface.co/rows?dataset=WizardLMTeam%2FWizardLM_evol_instruct_V2_196k&config=default&split=train&offset=<0|50000|100000|142900>&length=20 - row shapes, `idx` value forms, the row-17 example. Fetched 2026-08-11.

[10] WizardLMTeam/WizardLM_evol_instruct_70k dataset card (README). https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k/raw/main/README.md - "This is the training data of WizardLM." Fetched 2026-08-11.

[11] datasets-server size and info endpoints for WizardLMTeam/WizardLM_evol_instruct_70k. https://datasets-server.huggingface.co/size?dataset=WizardLMTeam%2FWizardLM_evol_instruct_70k and https://datasets-server.huggingface.co/info?dataset=WizardLMTeam%2FWizardLM_evol_instruct_70k - 70,000 rows, `instruction`/`output` schema. Fetched 2026-08-11.

[12] Hugging Face Hub API record for QuixiAI/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split, reached via redirect from the `ehartford` and `cognitivecomputations` account paths. https://huggingface.co/api/datasets/QuixiAI/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split - no `cardData`, siblings list has no `README.md`. Fetched 2026-08-11.

[13] datasets-server size and info endpoints for QuixiAI/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split. https://datasets-server.huggingface.co/size?dataset=QuixiAI%2FWizardLM_evol_instruct_V2_196k_unfiltered_merged_split and https://datasets-server.huggingface.co/info?dataset=QuixiAI%2FWizardLM_evol_instruct_V2_196k_unfiltered_merged_split - 153,908 rows, `conversations`/`from`/`value`/`markdown` schema. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT chat data, with the licence-driven row-count gap named above as the thing to know before starting: this repository serves 143,000 of the ~196,000 rows its own card describes, and the missing rows are not recoverable from any source fetched for this card. The screening row's note is consistent with what the card itself states about mixed Alpaca/ShareGPT origin and ChatGPT-era answers [2].

### The screening row

The row's own note: "Evol-Instruct evolved Alpaca plus ShareGPT logs; answers are ChatGPT-era." Its flag: "contains WizardLMTeam/WizardLM_evol_instruct_70k plus ShareGPT" (flag class: note).
