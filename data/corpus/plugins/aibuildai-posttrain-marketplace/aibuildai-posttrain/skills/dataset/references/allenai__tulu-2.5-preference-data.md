# allenai/tulu-2.5-preference-data

2,122,287 preference pairs across 28 named splits, each reformatted into the same chosen/rejected conversation-list schema, built by AI2 to run controlled preference-learning experiments.

**allenai/tulu-2.5-preference-data** packages 20-plus existing and derived preference datasets - AlpacaFarm, Chatbot Arena, HelpSteer, HH-RLHF, Nectar, Orca DPO Pairs, PRM800k, SHP-2, StackExchange, and several UltraFeedback variants - into one common `chosen`/`rejected`/`source` schema, built for "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback" [1]. AI2 reformatted every split so downstream code can load any one file under a single interface, and the README notes that reformatting can leave a split's fields differing from their original upstream layout [2]. It lives at https://huggingface.co/datasets/allenai/tulu-2.5-preference-data . **Two splits, `chatbot_arena_2023` (20,465 rows) and `chatbot_arena_2024` (34,269 rows), draw their prompts from Chatbot Arena conversation logs - the same kind of crowd-sourced pool that Arena-Hard-Auto's BenchBuilder pipeline curates its evaluation prompts from - so decontaminate against Arena-Hard before scoring a model trained on either split** [3][4].

**Use it for**: preference-pair training - DPO directly, or a reward model for PPO, per the training recipes the origin paper compares [1]. Every split shares the same `chosen`/`rejected` list-of-`{role, content}` schema (datasets-server `/info`) [9], and the three splits sampled for this card (`chatbot_arena_2023`, `chatbot_arena_2024`, `ultrafeedback_overall`) each had `chosen` and `rejected` sharing the same leading turn(s) and differing only in the final assistant turn - trl's dataset-formats guide names this shape the conversational implicit-prompt type, and its `extract_prompt` utility splits it into an explicit prompt plus two completions for trainers that need that shape [5][10]. See the DPO method card. Pick one split per run; do not concatenate splits that carry the risk in the previous paragraph into a mixed training set without excluding the eval-overlapping rows.

**Licence**: `odc-by` at the repository level (Hub `cardData.license` and tags both say `license:odc-by`) [6]. The one catch: the README states this ODC-BY licence covers the whole repository, but says "different splits may have additional license details" - two of them (`alpaca_farm_gpt4_pref`, `alpaca_farm_human_pref`) are CC-BY-NC-4.0, and `chatbot_arena_2023`'s model outputs are CC-BY-NC-4.0 while its prompts are CC-BY-4.0, so picking those splits carries a non-commercial restriction the top-level ODC-BY tag does not show [2].

**Shape**: 2,122,287 rows in one config (`default`), split across 28 named splits from 19,465 (`alpaca_farm_gpt4_pref`) to 500,000 (`shp_2`, `stack_exchange_paired`) rows each [6][7].

**Hold out**: hold out `chatbot_arena_2023` (20,465 rows) and `chatbot_arena_2024` (34,269 rows) - or filter their rows out of any split that includes them - before an Arena-Hard-style eval run, per the flag above [3][4]. No other split-versus-eval overlap is stated by any source read for this card.

**Origin**: built and released by AI2 (`allenai`); each split's `chosen`/`rejected` pair comes from that split's own upstream annotation process - human ratings for AlpacaFarm-human, Chatbot Arena, HH-RLHF, HelpSteer, and PRM800k, and model-generated or model-scored labels for AlpacaFarm-GPT4, Nectar, Orca, Capybara, and UltraFeedback [2]. Hub API at the check date: 981 downloads (28-day), 27,967 all-time, 18 likes [6].

**Trained-on-by**: the Tulu 2.5 model suite - the origin paper's own DPO, PPO, and reward-model checkpoints, one released per split (e.g. `allenai/tulu-v2.5-dpo-13b-hh-rlhf`, `allenai/tulu-v2.5-dpo-13b-chatbot-arena-2023`, `allenai/tulu-v2.5-ppo-13b-nectar-60k`), confirmed by a live search of AI2's Hub models [8].

**Introduced by**: [1] (Ivison et al.).

## Shape

Rows and byte sizes per split, read from the datasets-server `/size` endpoint [7]; the schema below is identical across all 28 (datasets-server `/info`) [9]:

| split | rows | parquet bytes |
| --- | --- | --- |
| `alpaca_farm_gpt4_pref` | 19,465 | 8,910,586 |
| `alpaca_farm_human_pref` | 9,686 | 4,438,300 |
| `argilla_dpo_mix` | 6,750 | 21,772,671 |
| `capybara` | 7,563 | 40,632,945 |
| `chatbot_arena_2023` | 20,465 | 23,366,175 |
| `chatbot_arena_2024` | 34,269 | 56,642,273 |
| `helpsteer` | 9,270 | 32,285,362 |
| `hh_rlhf` | 158,530 | 174,888,672 |
| `hh_rlhf_60k` | 60,908 | 67,329,394 |
| `nectar` | 180,099 | 307,382,266 |
| `nectar_60k` | 60,908 | 104,165,103 |
| `orca_dpo_pairs` | 12,859 | 24,023,719 |
| `preference_big_mixture` | 259,851 | 467,240,112 |
| `prm800k_pairs_phase2` | 6,949 | 7,542,773 |
| `shp_2` | 500,000 | 689,958,241 |
| `stack_exchange_60k` | 60,908 | 126,125,496 |
| `stack_exchange_paired` | 500,000 | 1,031,140,440 |
| `ultrafeedback_evol_instruct` | 10,000 | 22,676,483 |
| `ultrafeedback_false_qa` | 2,339 | 1,833,398 |
| `ultrafeedback_flan_v2` | 20,939 | 28,121,349 |
| `ultrafeedback_lower_10k` | 10,000 | 17,889,286 |
| `ultrafeedback_mean_aspects` | 60,908 | 130,038,330 |
| `ultrafeedback_middle_10k` | 10,000 | 21,594,739 |
| `ultrafeedback_overall` | 58,933 | 124,294,375 |
| `ultrafeedback_sharegpt` | 19,948 | 44,783,080 |
| `ultrafeedback_top_10k` | 10,000 | 23,114,209 |
| `ultrafeedback_truthful_qa` | 811 | 788,527 |
| `ultrafeedback_ultrachat` | 9,929 | 29,364,039 |
| **total** | **2,122,287** | **3,632,342,343** |

Columns, identical in every split (datasets-server `/info`) [9]:

| column | dtype |
| --- | --- |
| `chosen` | list\<struct\<role: string, content: string\>\> |
| `rejected` | list\<struct\<role: string, content: string\>\> |
| `source` | string |

`source` names the upstream split each row came from, e.g. `chatbot-arena-conversations`, `lmsys-human-pref-55k`, `h4-ultrafeedback` (read from sampled rows below) [10]. No source states a whole-dataset sequence-length or token-count figure. The README's split-by-split table carries only row counts, and the paper's Table 1 is a benchmark-score table (14 splits x 6 scored categories) whose `#Samples` column repeats a subset of those same row counts alongside the scores; neither table carries a token-length or sequence-length figure [1][2].

## Quality

- `preference_big_mixture` is itself a downsample: the README states it mixes HelpSteer, PRM800k, HH-RLHF, Nectar, StackExchange, and UltraFeedback, "randomly downsampl[ing] StackExchange, HH-RLHF, and Nectar to 60,908 samples each" [2] - so it is not an independent data source, and mixing it with `hh_rlhf_60k`, `nectar_60k`, or `stack_exchange_60k` in one training run double-counts those subsamples.
- The origin paper's Table 1 trains Tulu 2 13B with DPO on 14 of these splits and scores them on a 6-category benchmark suite; two splits stand out on safety - `chatbot_arena_2023` scores 67.3 and `chatbot_arena_2024` scores 58.1, against an SFT baseline of 91.8. Every other scored split scores 90.4 or above except `alpaca_farm_gpt4_pref` at 89.5. The paper attributes the Chatbot Arena drop to Chatbot Arena volunteers generally preferring more toxic completions [1].
- The paper reports that preference-based DPO training raises instruction-following and truthfulness performance over the SFT baseline by over 8 points on its best-performing splits; Table 1's own numbers show the largest gains on `ultrafeedback_mean_aspects` (truthfulness 69.3 vs. the 56.6 SFT baseline, +12.7; instruction-following 52.8 vs. 44.2, +8.6), while factuality stays within 1 point of the SFT baseline across all 14 scored splits - evidence the paper reads as preference data mainly shifting style and helpfulness, not factual knowledge [1].
- No split in this repository needs an outside fetch to train: every row already carries the full `chosen`/`rejected` conversation, so there is no join key to report.

## Load it

Pick one split (each is a leaf under `data/` in the repo) [6], and pin the revision this card's numbers were read at:

```python
import datasets

REV = "6e87e7ba11eeb2a21db459cf13267f62d1feea23"  # main at the check date
ds = datasets.load_dataset(
    "allenai/tulu-2.5-preference-data",
    data_files="data/hh_rlhf-00000-of-00001.jsonl",
    split="train",
    revision=REV,
)  # 158,530 rows
```

**Trap**: `load_dataset("allenai/tulu-2.5-preference-data", revision=REV)` with no `data_files` argument does download and build all 28 splits, but keeps them separate - it returns a `DatasetDict` with one key per source file (e.g. `hh_rlhf`: 158,530 rows), not a merged `train` split, confirmed by running the call above with no `data_files=` argument. The trap is that this pulls and builds the entire ~3.6 GB repository (all 28 splits) even when only one split is wanted; the `DatasetDict` returned by that call already paid the full download and build cost, so indexing it by split name afterward does not avoid that cost - only passing `data_files=` (as above) does. Either way, still exclude `chatbot_arena_2023`/`chatbot_arena_2024` before an Arena-Hard-adjacent eval and avoid mixing `preference_big_mixture` with the full-size splits it downsamples from.

## Neighbors

- `allenai/preference-datasets-tulu` - an earlier, un-downsampled staging repository from the same builder with the same `chosen`/`rejected`/`source` schema: its `shp_2` split alone carries 4,324,531 rows and `stack_exchange_paired` carries 4,999,988, against the 500,000-row downsamples of the same names in this release, confirming this card's 500k splits are subsamples of that larger pool [11].
- `allenai/llama-3.1-tulu-3-8b-preference-mixture` - the Tulu 3 successor mixture, 272,898 rows in a single `train` split with a `prompt`/`chosen`/`rejected`/`id`/`source` schema; its README lists mostly on-policy, model-generated preference sources rather than this release's mix of external human- and synthetic-labeled datasets, so the two are differently constructed, not overlapping releases [12][13].
- The 27 individual upstream datasets this repository reformats (AlpacaFarm, Chatbot Arena, HH-RLHF, Nectar, UltraFeedback, and the rest) are each named with a link in the README's split-by-split list; use one of those directly only if the reformatting to `chosen`/`rejected` message lists done here is not wanted [2].

## A row

All 28 splits share one schema; sampled from `config="default"`, `split="chatbot_arena_2023"`, `row_idx=0` (datasets-server `/first-rows`) [10]:

```json
{
  "chosen": [
    {"role": "user", "content": "Oauth 2.0 define the url to the token endpoint?"},
    {"role": "assistant", "content": "Yes, the OAuth 2.0 specification defines the URL to the token endpoint as follows:\n*   The token endpoint is an endpoint on the authorization server that is used to request new access tokens. [...]"}
  ],
  "rejected": [
    {"role": "user", "content": "Oauth 2.0 define the url to the token endpoint?"},
    {"role": "assistant", "content": "Yes, OAuth 2.0 defines a specific URL to the token endpoint. This URL is typically registered with the authorization server during the client registration process [...]"}
  ],
  "source": "chatbot-arena-conversations"
}
```

Both fields repeat the shared leading turn(s) and differ only in the final assistant turn; of the two other splits sampled while writing this card (`chatbot_arena_2024`, `ultrafeedback_overall`), each first row also carried exactly one user/assistant turn pair per side [10].

## Where it came from

Built and released by AI2 (allenai) as the preference-data release for "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback" [1]. The README states the team cleaned and reformatted every upstream dataset into this repository's shared schema and points to a public conversion script for most splits [2]. Each split's own generating process differs by source: AlpacaFarm's human and GPT-4 preference splits, Chatbot Arena's 2023 and 2024 conversation logs, Anthropic's HH-RLHF, NVIDIA's HelpSteer, and PRM800k's phase-2 annotations are all reused from their respective original releases with light reformatting (per-split links and licences in the README); Nectar, Orca DPO Pairs, Capybara, and the UltraFeedback variants are likewise reused from their published Argilla- or Berkeley-cleaned forms; `shp_2` and `stack_exchange_paired` are random 500,000-row downsamples of their much larger sources; `hh_rlhf_60k`, `nectar_60k`, and `stack_exchange_60k` are random 60,908-row downsamples of the corresponding full-size split, used for the paper's PPO experiments; and `preference_big_mixture` combines HelpSteer, PRM800k, HH-RLHF, Nectar, StackExchange, and UltraFeedback with the same three sources downsampled to 60,908 each [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. All sources were read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Ivison et al., "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback", 2024. https://arxiv.org/abs/2406.09279 - the origin paper; current title read from the live abs page, body text (abstract, Table 1, safety and factuality discussion) read from the ar5iv HTML rendering. Fetched 2026-08-12.

[2] allenai/tulu-2.5-preference-data dataset card (README). https://huggingface.co/datasets/allenai/tulu-2.5-preference-data/raw/main/README.md - per-split descriptions, licences, and construction notes. Fetched 2026-08-12.

[3] The corpus shortlist row for `allenai/tulu-2.5-preference-data`, supplied with this card's request - its `flag` field ("rule-risk: arenahardwriting"), read back in the appendix. Checked 2026-08-12.

[4] Li et al., "From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline", 2024. https://arxiv.org/abs/2406.11939 - the BenchBuilder/Arena-Hard-Auto paper; its abstract states BenchBuilder curates Arena-Hard-Auto's prompts from large crowd-sourced datasets including Chatbot Arena and WildChat-1M. Corroborated by the Arena-Hard-Auto README, which names Chatbot Arena as the source of the 250 creative-writing queries in its newer Arena-Hard-v2.0-Preview set specifically (https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/README.md, a `main`-branch file, unpinned and mutable). Fetched 2026-08-12.

[5] trl dataset formats documentation. https://huggingface.co/docs/trl/main/en/dataset_formats - conversational implicit-prompt preference type and `extract_prompt`. A `main` build, unpinned and mutable. Fetched 2026-08-12.

[6] Hugging Face Hub API record for allenai/tulu-2.5-preference-data. https://huggingface.co/api/datasets/allenai/tulu-2.5-preference-data?full=true - licence, gate status, `sha`, siblings (per-split data files), `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=allenai%2Ftulu-2.5-preference-data - live, unpinned, takes no revision parameter; row and byte counts above hold for the state read at the check date, not necessarily for the pinned `revision` in Load it. Fetched 2026-08-12.

[8] Hugging Face Hub model-search API, `search=tulu-2.5&author=allenai`. https://huggingface.co/api/models?search=tulu-2.5&author=allenai - live listing of Tulu 2.5 DPO/PPO/reward-model checkpoints named after individual splits. Fetched 2026-08-12.

[9] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=allenai%2Ftulu-2.5-preference-data - live, unpinned, takes no revision parameter; the schema read here holds for the state at the check date. Fetched 2026-08-12.

[10] datasets-server first-rows endpoint, one call per split sampled: `chatbot_arena_2023`, `chatbot_arena_2024`, `ultrafeedback_overall`. https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Ftulu-2.5-preference-data&config=default&split=<split> - live, unpinned, takes no revision parameter. Fetched 2026-08-12.

[11] datasets-server size and info endpoints for the neighbor `allenai/preference-datasets-tulu`, plus its Hub API record. https://datasets-server.huggingface.co/size?dataset=allenai%2Fpreference-datasets-tulu and https://huggingface.co/api/datasets/allenai/preference-datasets-tulu - split-level row counts, unpinned and live. Fetched 2026-08-12.

[12] datasets-server size endpoint for the neighbor `allenai/llama-3.1-tulu-3-8b-preference-mixture`. https://datasets-server.huggingface.co/size?dataset=allenai%2Fllama-3.1-tulu-3-8b-preference-mixture - live, unpinned. Fetched 2026-08-12.

[13] allenai/llama-3.1-tulu-3-8b-preference-mixture dataset card (README). https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture/raw/main/README.md - schema and list of source datasets it mixes. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as preference-pair training data, split by split, with the two Chatbot Arena splits held out of any Arena-Hard-adjacent evaluation. This rests on the row's own flag, which names `chatbot_arena_2023` and `chatbot_arena_2024` as carrying the prompt pool Arena-Hard's questions are drawn from [3], corroborated by the Arena-Hard-Auto project's own description of its prompt sourcing [4]; every other split in the repository carries no stated eval-overlap risk.

### The screening row

The row's own note [3]: "20+ preference splits in one format: AlpacaFarm (human and GPT-4), Chatbot Arena, HelpSteer, HH-RLHF, Nectar, PRM800k, SHP-2, StackExchange, UltraFeedback." Its flag, class `rule-risk`: "rule-risk: arenahardwriting - Ships `chatbot_arena_2023` and `chatbot_arena_2024` splits, i.e. the Chatbot Arena prompts Arena-Hard's questions are drawn from."
