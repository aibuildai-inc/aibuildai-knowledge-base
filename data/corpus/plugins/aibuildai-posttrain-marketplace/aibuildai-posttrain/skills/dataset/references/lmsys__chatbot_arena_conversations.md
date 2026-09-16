# lmsys/chatbot_arena_conversations

33,000 cleaned Chatbot Arena battle transcripts with pairwise human votes across 20 LLMs, gated behind an auto-accept terms click.

**lmsys/chatbot_arena_conversations** is LMSYS's (UC Berkeley) release of crowdsourced pairwise-preference conversations from the Chatbot Arena platform, introduced by "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" [1]: each row holds two full model conversations answering the same user turns plus the crowd voter's pick, so the task shape is a pairwise judge/preference comparison, not a single chosen/rejected SFT pair [2]. It lives at https://huggingface.co/datasets/lmsys/chatbot_arena_conversations . **The dataset is gated with auto-accept terms - anonymous requests to both the file and the datasets-server viewer return 401/404 until a Hugging Face account accepts the gate, confirmed live [4][5] - and our corpus screening flags it as the Chatbot Arena user-prompt pool that Arena-Hard-v2.0's 250 creative-writing evaluation questions are drawn from, so training on it risks contaminating that benchmark [6][7].**

**Use it for**: pairwise preference/judge training - each row carries `conversation_a`, `conversation_b` (both full OpenAI-role-format transcripts) and a `winner` label, not an implicit-prompt chosen/rejected pair, so it needs reshaping (pick the winning conversation as chosen, the other as rejected, and drop tie/tie-bad rows) before it fits a DPO-style trainer; maps to the preference-pair method card once reshaped. **Given the rule-risk flag, decontaminate against Arena-Hard-v2.0's creative-writing set before using this for any training whose output will be scored on that benchmark** [6][7].

**Licence**: tagged `cc` at the repository level [3], but the card's own License section splits it: user prompts under CC-BY-4.0, model outputs under CC-BY-NC-4.0 [2] - the catch is that the non-commercial term on outputs covers the full conversation text, since every row bundles prompt and output together. Gated, auto-accept (an `extra_gated_prompt` disclaimer a user must click through; no manual-review fields) [3].

**Shape**: 33,000 rows, one split (`train`), one config (`default`), 13 columns [2][3].

**Hold out**: the dataset ships no train/test split of its own - the risk here is not an in-repo split but the rule-risk flag: hold out (or decontaminate against) Arena-Hard-v2.0's 250 creative-writing questions, which the benchmark's own README says are "sourced from Chatbot Arena" [6], and which the corpus screening row identifies this exact release as the pool for [7].

**Origin**: built and released by LMSYS; the two candidate replies come from 20 different LLMs including GPT-4 and Claude-v1, and the win/lose/tie label is a human Chatbot Arena user's live vote [2]. Hub API at the check date: `downloads` 2,821, `likes` 475 [3]; `downloadsAllTime` 318,872 [8].

**Trained-on-by**: `llm-blender/PairRM` lists `lmsys/chatbot_arena_conversations` among its named training datasets [9]. The origin paper itself fine-tunes a Vicuna-13B judge on a 22,000-vote sample drawn from Chatbot Arena (a different, earlier sample than this 33,000-row release, from the same paper) [1]. No other trained-on adoption found.

**Introduced by**: [1] (Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena").

## Shape

Columns, read from the repository's `cardData.dataset_info` front matter (the datasets-server `/info` endpoint 404s without an accepted gate, so this YAML block is the only structural source available) [3][4]:

| column | dtype |
| --- | --- |
| `question_id` | string |
| `model_a` | string |
| `model_b` | string |
| `winner` | string |
| `judge` | string |
| `conversation_a` | list of `{content: string, role: string}` |
| `conversation_b` | list of `{content: string, role: string}` |
| `turn` | int64 |
| `anony` | bool |
| `language` | string |
| `tstamp` | float64 |
| `openai_moderation` | struct (per-category bool flags + float scores + overall `flagged` bool) |
| `toxic_chat_tag` | struct (`roberta-large` and `t5-large` flag/score sub-structs) |

One split, `train`, 33,000 rows, `num_bytes` 81,159,839, `download_size` 41,572,998 (the single Parquet file under `data/`) [3].

The card's own "Basic Statistics" table states: 33,000 conversations, 20 models, 13,383 users, 96 languages, 1.2 average turns per sample, 52.3 average tokens per prompt, 189.5 average tokens per response [2].

## Quality

- PII removal is a stated best-effort process, not a measured rate: "we have made our best efforts to remove all conversations that contain personally identifiable information (PII)" [2].
- Unsafe conversations are kept intentionally rather than filtered out, to let researchers study real-world LLM safety behavior; each row instead carries the OpenAI moderation API's category flags and scores (`openai_moderation`) and a `toxic_chat_tag` produced by the builders' own T5- and RoBERTa-based toxic-content classifiers, fine-tuned on manually labeled data [2].
- No source states a measured flagged-conversation rate, duplicate rate, or annotator-agreement figure for this release; none is invented here.
- Users consented via the Chatbot Arena "Terms of use" at collection time, per the card [2].

## Load it

The gate requires an authenticated, terms-accepted Hugging Face account - anonymous file and viewer requests both fail (confirmed live at the check date: `GET .../resolve/main/data/train-00000-of-00001-cced8514c7ed782a.parquet` returns HTTP 401 "Access to dataset lmsys/chatbot_arena_conversations is restricted..."; the datasets-server `/info` and `/first-rows` endpoints both return HTTP 404 "The dataset does not exist, or is not accessible without authentication...") [4][5]. Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-09-30) [3]:

```python
import datasets

REV = "1b6335d42a1d2c7e34870c905d03ab964f7f2bd8"  # main at the check date
train = datasets.load_dataset(
    "lmsys/chatbot_arena_conversations",
    revision=REV,
    split="train",
    token=True,  # requires a HF account that has accepted the gate terms
)  # 33,000 rows
```

**Trap**: without `token=True` (or `HF_TOKEN` set) and without the account having clicked through the gate's disclaimer, both the file download and the dataset viewer fail outright rather than returning a partial or cached result - there is no ungated fallback path for this repository [3][4][5].

## Neighbors

- `lmsys/lmsys-chat-1m` - the larger, unpaired raw log this release was drawn from: "one million real-world conversations with 25 state-of-the-art LLMs", collected from "210K unique IP addresses in the wild on the Vicuna demo and Chatbot Arena website from April to August 2023" [10]. It is single-response per row (no `model_b`, no `winner`), also gated auto-accept, so it does not share this release's pairwise-judge shape [10].
- `lmarena-ai/arena-human-preference-55k` - the LMSYS team's ungated successor, built for a Kaggle competition: "over 55,000 real-world user and LLM conversations and user preferences across over 70 state-of-the-art LLMs" [11], 57,477 rows in its `train` split [12]. A live-fetched row shows a different schema: `prompt`, `response_a`, `response_b` as JSON-string lists plus separate `winner_model_a`/`winner_model_b`/`winner_tie` int flags, rather than this release's single `winner` string and role/content list columns [12] - reshape, do not concatenate, if combining the two.
- `lmarena-ai/PPE-Debug` - explicitly built by sampling prompts from this exact dataset, but its own card states "This dataset is meant for benchmarking and evaluation, not for training" [13].

No source states which of these our corpus prefers when they overlap; this shortlist row is the one carried into screening.

## A row

No row could be fetched: the repository is gated, and every anonymous read path returned an authentication error at the check date - `GET .../resolve/main/data/train-00000-of-00001-cced8514c7ed782a.parquet` returned HTTP 401 with body "Access to dataset lmsys/chatbot_arena_conversations is restricted. You must have access to it and be authenticated to access it. Please log in." [4], and the datasets-server `/first-rows` endpoint for `config=default, split=train` returned HTTP 404 with body "The dataset does not exist, or is not accessible without authentication (private or gated). Please check the spelling of the dataset name or retry with authentication." [5]. A reader who needs to see a row before committing to this dataset must first accept the gate's terms with an authenticated Hugging Face account and re-run the load line above with a valid token.

## Where it came from

Built and released by LMSYS. The underlying conversations come from Chatbot Arena, "a crowdsourcing benchmark platform featuring anonymous battles" where a user poses one question to two anonymously-assigned models at once and votes for the better response after the models' identities are revealed [1]. The origin paper reports collecting "around 30K votes" after running the arena for one month, and separately fine-tunes its own 22K-vote-sample Vicuna-13B judge on this raw arena log [1]; this Hub release is the cleaned, PII-scrubbed, 33,000-row card of that same collection process, expanded past the paper's original month of data collection given its later `createdAt` date of 2023-07-18 [1][3]. Release-time cleaning applied OpenAI's moderation API and the builders' own toxic-content classifiers as tags rather than filters, and removed detected PII before publication [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and live-endpoint result above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; this repository is additionally gated, so several endpoints below are pinned only in the sense that they returned an authentication error at that revision, not in the sense that they returned data.

[1] Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", 2023. https://arxiv.org/abs/2306.05685 - the origin paper; collection process, the 30K-vote arena count, the 22K-vote Vicuna-13B fine-tuning experiment. Current title read from the live abs page and the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2306.05685). Fetched 2026-08-11.

[2] lmsys/chatbot_arena_conversations dataset card (README), read via the raw resolve endpoint since the rendered page itself refuses anonymous access. https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/raw/main/README.md - description, Basic Statistics table, Disclaimers and Terms, License section, PII/moderation/toxic-tag statements. Fetched 2026-08-11.

[3] Hugging Face Hub API record for lmsys/chatbot_arena_conversations. https://huggingface.co/api/datasets/lmsys/chatbot_arena_conversations?full=true - `cardData.dataset_info` (columns, split sizes), `license` tag, `gated`, `sha`, `downloads`, `likes`, `lastModified`, `extra_gated_prompt`. Fetched 2026-08-11.

[4] Anonymous resolve request for the repository's data file. https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/resolve/main/data/train-00000-of-00001-cced8514c7ed782a.parquet - returned HTTP 401, confirming the gate blocks unauthenticated file access. Fetched 2026-08-11.

[5] datasets-server first-rows and info endpoints for this dataset. https://datasets-server.huggingface.co/first-rows?dataset=lmsys%2Fchatbot_arena_conversations&config=default&split=train and https://datasets-server.huggingface.co/info?dataset=lmsys%2Fchatbot_arena_conversations - both returned HTTP 404, confirming the gate blocks the anonymous viewer. Fetched 2026-08-11.

[6] lmarena/arena-hard-auto GitHub README. https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/README.md - states Arena-Hard-v2.0-Preview contains "250 creative writing queries sourced from Chatbot Arena". Fetched 2026-08-11.

[7] The corpus screening row for `lmsys/chatbot_arena_conversations`, supplied with this card's request - its `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[8] Hugging Face Hub API record for lmsys/chatbot_arena_conversations with `expand[]=downloadsAllTime`. https://huggingface.co/api/datasets/lmsys/chatbot_arena_conversations?expand[]=downloadsAllTime Fetched 2026-08-11.

[9] llm-blender/PairRM model card (README). https://huggingface.co/llm-blender/PairRM/raw/main/README.md - lists `lmsys/chatbot_arena_conversations` among its training datasets. Fetched 2026-08-11.

[10] lmsys/lmsys-chat-1m Hub API record and description. https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m?full=true - description text (row/IP/collection-window counts, single-response shape), `gated` status; the rendered README itself 401s anonymously, same as this dataset's. Fetched 2026-08-11.

[11] lmarena-ai/arena-human-preference-55k dataset card (README). https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/raw/main/README.md - row-count and model-count description, licence. Fetched 2026-08-11.

[12] datasets-server info and first-rows endpoints for lmarena-ai/arena-human-preference-55k. https://datasets-server.huggingface.co/info?dataset=lmarena-ai%2Farena-human-preference-55k and https://datasets-server.huggingface.co/first-rows?dataset=lmarena-ai%2Farena-human-preference-55k&config=default&split=train - `train` split row count (57,477), column schema, and one live row's fields. Fetched 2026-08-11.

[13] lmarena-ai/PPE-Debug dataset card (README). https://huggingface.co/datasets/lmarena-ai/PPE-Debug/raw/main/README.md - states "The prompts are sampled from lmsys/chatbot_arena_conversations" and "This dataset is meant for benchmarking and evaluation, not for training." Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as a pairwise-preference source, but carrying a rule-risk flag rather than a clean pass: the dataset's own card restricts nothing about training use beyond the CC-BY-NC-4.0 term on model outputs (quoted above) [2], but the corpus screening row flags this exact release as the source pool for Arena-Hard-v2.0's creative-writing evaluation questions, which the benchmark's own README corroborates by describing those 250 questions as "sourced from Chatbot Arena" [6][7]. Any training run drawing on this dataset should decontaminate against that benchmark before being scored on it.

### The screening row

The row's own note [7]: "33,000 cleaned Chatbot Arena conversations from 13,000 unique IP addresses collected April to June 2023, each with two model names, both full conversations and the pairwise human vote; gated with auto-accept, so the viewer returns HTTP 404 until the terms are taken." Its flag [7]: "rule-risk: arenahardwriting - this is the Chatbot Arena user-prompt pool itself, the source the arena-hard-v2.0 creative_writing questions are curated from."
