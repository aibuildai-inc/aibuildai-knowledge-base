# berkeley-nest/Nectar

182,954 prompts, each carrying seven GPT-4-ranked model responses, described by its own card as the first high-quality 7-wise comparison dataset [1].

**berkeley-nest/Nectar** was developed by Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu and Jiantao Jiao and released with no accompanying paper - the card itself is the only introduction, with a paper listed as "Coming soon!" [1]. Prompts were pooled from six sources (lmsys-chat-1M, ShareGPT, Anthropic HH, UltraFeedback, Evol-Instruct and Flan), and each prompt's seven responses were distilled from GPT-4, GPT-4-0613, GPT-3.5-turbo, GPT-3.5-turbo-instruct, Llama-2-7B-chat and Mistral-7B-Instruct, then ranked 1-7 by GPT-4, giving 3.8M pairwise comparisons in total [1]. It lives at https://huggingface.co/datasets/berkeley-nest/Nectar . **The licence is Apache-2.0 conditioned on not using the dataset to compete with OpenAI, and the card additionally scopes the whole release - dataset, model and demo - to a non-commercial research preview subject to Llama's model licence, OpenAI's Terms of Use and ShareGPT's privacy practices [1].** **A narrower usage-shape flag, carried from screening: 45,000 of the prompts were sampled from lmsys-chat-1M, whose own card says its conversations were collected from "the Vicuna demo and Chatbot Arena website" [6]. Arena-Hard-Auto's README states only that its 2025 v2.0-Preview evaluation set's 250 creative-writing queries are sourced from Chatbot Arena, and separately says Arena-Hard-Auto's prompt-selection pipeline is similar to (not the same as) the one used for Chatbot Arena's own hard-prompt category [7] - it does not say Arena-Hard-Auto's prompts in general come from Chatbot Arena. A reader relying on Arena-Hard for evaluation should still check for overlap against rows whose `source` list contains `lmsys-chat-1m` before treating the two as independent, but the evidence for that overlap is narrower than the screening flag implies.**

**Use it for**: 7-wise (K-wise) preference-ranking training - a reward model trained with a K-wise maximum-likelihood objective, as in Starling-RM-7B-alpha [8], or a listwise/Plackett-Luce preference method. Each row's `answers` field is a ranked list, not a chosen/rejected pair, so a standard pairwise RewardTrainer/DPO trainer needs a conversion step first - either sample one pair per prompt (as HongchengGao/Nectar_binarized does) or expand to all C(7,2)=21 pairs (as kashif/nectar_dpo_pairs does) - before it maps to the implicit-prompt pairwise preference format used by the reward-model method card. Non-commercial research use only [1].

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, ungated, `"private": false`) [2], but the card body narrows this to a non-commercial research preview and warns it must not be used to compete with OpenAI, subject to Llama's model licence, OpenAI's Terms of Use and ShareGPT's privacy practices [1].

**Shape**: 182,954 rows, one split (`train`), one config (`default`) [3][4].

**Hold out**: no eval split ships with this dataset, so there is nothing to hold out on row-count grounds. The screening flag names a possible content-overlap risk between rows whose `source` list contains `lmsys-chat-1m` (up to 45,000 prompts by the card's own sampling step [1]) and Arena-Hard-Auto, since both trace back to Chatbot Arena traffic in some form [6][7]; the evidence for this is narrower than a blanket overlap claim (see the opening paragraph), but a reader planning to evaluate on Arena-Hard should still filter on the `source` column and check before assuming independence.

**Origin**: built by Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu and Jiantao Jiao, whose correspondence email and project blog are hosted on berkeley.edu domains [1][5]; responses are distilled from six named models and the rank label on each row is a GPT-4 judgment, not a human one [1]. Hub API at the check date: `downloads` 1,040, `likes` 296 [2].

**Trained-on-by**: Starling-RM-7B-alpha, a reward model finetuned from Llama2-7B-Chat with a K-wise maximum-likelihood estimator trained directly on this dataset [8]; Starling-LM-7B-alpha, finetuned from Openchat 3.5 against that reward model, which its own card reports scoring 8.09 on MT-Bench and 91.99 on AlpacaEval versus 7.81 and 88.51 for its Openchat-3.5 base [9]. The Hub's list of 99 models tagged `datasets:berkeley-nest/Nectar` includes many third-party requantizations of Starling-LM-7B-alpha (TheBloke's GGUF/AWQ/GPTQ builds, LoneStriker's exl2 builds) but also several repositories whose names indicate independent training runs rather than requantization - Nexusflow/Starling-LM-7B-beta, Nexusflow/Starling-RM-34B, Cornell-AGI/REBEL-OpenChat-3.5, andysalerno/openchat-nectar-0.1/0.5/0.14/0.19, jieliu/Storm-7B, codeIA/GuIA-v2, mightbe/Better-PairRM and pharaouk/starling-RM - none of whose cards were fetched for this card, so their training details are not confirmed here [10].

**Introduced by**: no paper - the dataset card [1], plus the introducing blog at https://starling.cs.berkeley.edu/ [5].

## Shape

Rows and splits (datasets-server `/size`) [3]:

| split | rows |
| --- | --- |
| `train` | 182,954 |

One config, `default`, with six columns (datasets-server `/info`) [4]:

| column | dtype |
| --- | --- |
| `prompt` | string |
| `answers` | list<struct<answer: string, model: string, rank: float64>> |
| `turns` | int64 |
| `num_responses` | int64 |
| `source` | list<string> |
| `good_natured` | bool |

Sizes (datasets-server `/size`) [3]: 517,439,683 bytes as Parquet, 1,207,387,327 bytes decoded in memory. No source states sequence-length or token statistics for this release.

## Quality

- Every rank label is a GPT-4 judgment applied through a rubric that requires harmlessness first and maximizes helpfulness only for prompts GPT-4 itself classified as good-natured; the card quotes the rubric's helpfulness and harmlessness criteria in full [1].
- The `good_natured` column is a by-product of that same GPT-4 classification step, not the authors' own judgment, and the card states this explicitly [1].
- When fewer than seven distinct responses were available for a prompt, the seventh slot could be filled by repeating an existing response or, failing that, by a fixed default reply ("I apologize, but I can’t assist with that request.") regardless of the prompt's content [1].
- The card's own disclaimer states: "This dataset contains conversations and responses that are possibly unsafe, offensive, and/or disturbing." It adds that these are included only to train safer models [1].
- No source states a measured contamination rate, duplicate rate, or inter-rater agreement figure. Of the 26 rows read at offset 0 of the `train` split (the only split), the `source` field distribution was: `lmsys-chat-1m` 13, `sharegpt` 9, `anthropic-hh` 3, `flan_v2_p3` 1 [11]; this describes only those 26 rows, not the full 182,954-row split.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-03-20) [2]:

```python
import datasets

REV = "3c6b4c47fa1cc38869f9f32dce1699f7abad8b06"  # main at the check date
ds = datasets.load_dataset("berkeley-nest/Nectar", revision=REV, split="train")  # 182,954 rows
```

**Trap**: this is a single `train` split with no held-out eval split, and each row bundles all seven ranked responses in one nested `answers` list rather than flat chosen/rejected columns [1][4] - a pairwise RewardTrainer or DPO trainer cannot consume it as-is; convert to pairs first (see Use it for and Neighbors). A second trap: the row count, byte sizes, schema and sampled row on this card come from the datasets-server `/size`, `/info` and `/first-rows` endpoints, none of which accept a revision parameter - they always answer for whatever is on `main` right now, so only `sha` and the licence/gate/download figures in this card are actually covered by the `REV` pin above [2][3][4][11].

## Neighbors

- `kashif/nectar_dpo_pairs` - the full pairwise expansion: 3,842,034 rows, cc-by-nc-4.0, flat `prompt`/`chosen`/`rejected` string columns [12][13]. 182,954 × C(7,2) = 3,842,034 exactly, matching both this row count and Nectar's own "3.8M pairwise comparisons" figure [1][12], so every prompt appears in up to 21 pairs.
- `HongchengGao/Nectar_binarized` - one pair per prompt: 182,954 rows (matching Nectar's prompt count 1:1), `prompt`/`prompt_id`/`chosen`/`rejected` columns where `chosen` and `rejected` are each a one-turn `{content, role}` list [15]; its card says it takes the rank-1 response as chosen and a random response from ranks 2-7 as rejected, following the same recipe as HuggingFaceH4/ultrafeedback_binarized for use with the alignment-handbook DPO pipeline [14]. Its Hub API record carries no `cardData` key at all, so it has no declared licence field [17].
- Prefer this original when the training method can consume a full 7-wise ranking (the K-wise reward-model route Starling-RM-7B-alpha used [8]); reach for `HongchengGao/Nectar_binarized` for a ready one-pair-per-prompt DPO shape, or `kashif/nectar_dpo_pairs` when every pairwise comparison is wanted at the cost of up to 21x row inflation per prompt.

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [16]:

```json
{
  "prompt": "\n\nHuman: 0.002 = 1000 \n1 = x?\n\nAssistant: ",
  "answers": [
    {"answer": "To find the value of x, we can set up a proportion using the given information:\n\n0.002/1000 = 1/x\n\n...\n\nx = 500,000\n\nTherefore, 1 is equal to 500,000 in this proportion.", "model": "gpt-3.5-turbo", "rank": 1.0},
    {"answer": "To solve for x in this equation, you can use cross-multiplication. Here's how:\n\n0.002 / 1 = 1000 / x\n\n...\n\nSo, x equals 500,000.", "model": "gpt-4-0613", "rank": 2.0},
    {"answer": "This looks like a proportion. To solve for x, you can set up a proportion equation:\n\n0.002 / 1 = 1000 / x\n\n...\n\nx = 500,000", "model": "gpt-3.5-turbo", "rank": 3.0},
    {"answer": "If 0.002 equals 1000, then 1 would equal 500,000.", "model": "gpt-4", "rank": 4.0},
    {"answer": "I apologize, but the equation you provided is not correct.\n\n0.002 is equal to 0.002, not 1000.\n\n1 is equal to x, not a number.\n\nCan you please provide a correct equation or problem for me to solve?", "model": "llama-2-7b-chat", "rank": 5.0},
    {"answer": "0.001 = x", "model": "gpt-3.5-turbo-instruct", "rank": 6.0},
    {"answer": "It seems like you are asking for the value of x in the equation x = 1/0.002. \n\n...\n\nTherefore, x = 500.", "model": "mistral-7b-instruct-v0.1", "rank": 7.0}
  ],
  "turns": 1,
  "num_responses": 7,
  "source": ["sharegpt"],
  "good_natured": true
}
```

## Where it came from

Built and released by Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu and Jiantao Jiao [1]. Prompts were collected by grouping each source dataset by prompt, then down-sampling before merging: all ShareGPT prompts, 75,000 randomly-sampled Anthropic HH prompts, all UltraFeedback prompts, and 45,000 randomly-sampled lmsys-chat-1M prompts with more than one answer, keeping only prompts longer than 40 characters with more than one answer and dropping non-English datapoints [1]. Responses were distilled from Llama-2-7B-chat, Mistral-7B-instruct, GPT-4, GPT-4-0613, GPT-3.5-turbo and GPT-3.5-turbo-instruct, with GPT-4-0613 given an additional system prompt; where more than seven candidate responses existed for a prompt, a fixed priority order selected which seven to keep, falling back to repeats or a default refusal string when too few distinct responses were available [1]. GPT-4 then ranked the seven responses per prompt against a rubric requiring harmlessness first and helpfulness for good-natured prompts, with unspecified mitigations against positional bias that the card says will be detailed in a forthcoming paper [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] berkeley-nest/Nectar dataset card (README). https://huggingface.co/datasets/berkeley-nest/Nectar/raw/main/README.md - description, schema, licence terms, prompt/response/ranking process, disclaimer, dataset sources. Fetched 2026-08-11.

[2] Hugging Face Hub API record for berkeley-nest/Nectar. https://huggingface.co/api/datasets/berkeley-nest/Nectar?full=true - licence tag, gate, `sha`, `downloads`, `likes`, `lastModified`. Fetched 2026-08-11.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=berkeley-nest%2FNectar - takes no revision parameter (confirmed by requerying with a nonexistent revision and getting an identical payload), so this is live, not pinned by the sha in Load it. Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=berkeley-nest%2FNectar - takes no revision parameter, so this is live, not pinned by the sha in Load it. Fetched 2026-08-11.

[5] Starling project blog. https://starling.cs.berkeley.edu/ - the introducing blog named in the dataset card's "Dataset Sources" section; hosted on a berkeley.edu domain. Fetched 2026-08-11.

[6] Hugging Face Hub API record for lmsys/lmsys-chat-1m. https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m?full=true - `description` field states the conversations were collected from the Vicuna demo and Chatbot Arena website; the dataset's own README is access-gated and could not be opened directly. Fetched 2026-08-11.

[7] Arena-Hard-Auto README (GitHub). https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/README.md - states that its 2025 v2.0-Preview evaluation set's 250 creative-writing queries are sourced from Chatbot Arena, and separately that its hard-prompt selection pipeline is similar to that used for Chatbot Arena's own hard-prompt category; does not state that its prompts in general come from Chatbot Arena. Fetched 2026-08-11.

[8] Starling-RM-7B-alpha model card (README). https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha/raw/main/README.md - K-wise maximum-likelihood reward model trained on berkeley-nest/Nectar. Fetched 2026-08-11.

[9] Starling-LM-7B-alpha model card (README). https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha/raw/main/README.md - MT-Bench/AlpacaEval/MMLU comparison table against Openchat-3.5 and other models. Fetched 2026-08-11.

[10] Hugging Face Hub models-list endpoint filtered by `datasets:berkeley-nest/Nectar`. https://huggingface.co/api/models?filter=dataset:berkeley-nest/Nectar - the list of repositories tagged as trained on or derived from this dataset. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=berkeley-nest%2FNectar&config=default&split=train - 26 rows read at offset 0; used for the row shown in "A row" and the source-field tally in Quality. Takes no revision parameter, so live, not pinned by the sha in Load it. Fetched 2026-08-11.

[12] Hugging Face Hub API record for kashif/nectar_dpo_pairs. https://huggingface.co/api/datasets/kashif/nectar_dpo_pairs?full=true - `cardData` giving licence, row count and column schema. Fetched 2026-08-11.

[13] datasets-server size endpoint for kashif/nectar_dpo_pairs. https://datasets-server.huggingface.co/size?dataset=kashif%2Fnectar_dpo_pairs - this endpoint takes no revision parameter, so this count is live, not pinned. Fetched 2026-08-11.

[14] HongchengGao/Nectar_binarized dataset card (README). https://huggingface.co/datasets/HongchengGao/Nectar_binarized/raw/main/README.md - states the rank1-chosen, random-rank2-7-rejected binarization method and its intended use with alignment-handbook. Fetched 2026-08-11.

[15] datasets-server size and info endpoints for HongchengGao/Nectar_binarized. https://datasets-server.huggingface.co/size?dataset=HongchengGao%2FNectar_binarized and https://datasets-server.huggingface.co/info?dataset=HongchengGao%2FNectar_binarized - row count and column schema. These endpoints take no revision parameter, so this is live, not pinned. Fetched 2026-08-11.

[16] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=berkeley-nest%2FNectar&config=default&split=train&offset=0 - row 0, used verbatim in "A row" with mid-answer text elided. This endpoint takes no revision parameter, so this row is live, not pinned by the sha in Load it. Fetched 2026-08-11.

[17] Hugging Face Hub API record for HongchengGao/Nectar_binarized. https://huggingface.co/api/datasets/HongchengGao/Nectar_binarized?full=true - the record has no `cardData` key, so no licence field is declared. Live, not pinned. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as 7-wise/K-wise preference training data, non-commercial only, and worth a manual overlap check against Arena-Hard-Auto before any evaluation on it. Three facts decide it, all established above: the card restricts the release to non-commercial research use and bars competing with OpenAI [1]; its `source` column lets a reader identify the 45,000 lmsys-chat-1m-sourced prompts that trace back to Chatbot Arena traffic [1][6], though the fetched Arena-Hard-Auto README only ties Chatbot Arena to a 250-query creative-writing subset of its 2025 v2.0-Preview set, not to its prompts generally [7]; and the screening row itself flags this as a rule risk, worth checking even though the fetched evidence for it is narrower than the flag's wording suggests.

### The screening row

The row's own note: "7-wise preference data: responses distilled from Llama-2-7B-chat, Mistral-7B-instruct, GPT-4, GPT-4-0613, GPT-3.5-turbo and GPT-3.5-turbo-instruct, ranked by GPT-4. Research/non-commercial only." Its flag, class `rule-risk`: "arenahardwriting - Card: 45,000 prompts sampled from lmsys-chat-1m, the Chatbot Arena user-query pool Arena-Hard prompts are themselves drawn from."
