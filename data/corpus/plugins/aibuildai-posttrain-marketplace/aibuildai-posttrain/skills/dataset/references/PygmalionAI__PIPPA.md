# PygmalionAI/PIPPA

About 26,000 crowdsourced Character.AI roleplay logs, human turns interleaved with the site's LLM replies, shipped as three JSONL variants with no formal train/test split.

**PygmalionAI/PIPPA** (Personal Interaction Pairs between People and AI) is a partially-synthetic conversational and roleplaying dataset built by PygmalionAI from user-submitted Character.AI chat logs, introduced in "PIPPA: A Partially Synthetic Conversational Dataset" [1]. It lives at https://huggingface.co/datasets/PygmalionAI/PIPPA . A community userscript scraped each submitter's own conversation and the character persona ("definitions" and "description" fields) from the Character.AI website, so every conversation pairs a human-written persona with alternating human and Character.AI-model turns [1][2]. **Only logs whose submitter opted in to public redistribution are included, PII was scanned and redacted "to the best of" the maintainers' ability rather than guaranteed complete, and the maintainers recommend the deduplicated file (`pippa_deduped.jsonl`) over the raw one; the content is flagged not-for-all-audiences and the card warns that models trained purely on PIPPA may generate X-rated output [1][2].**

**Use it for**: multi-turn persona roleplay chat SFT. `pippa.jsonl`/`pippa_deduped.jsonl` rows carry a `conversation` list of `{message, is_human}` turns plus `bot_description`/`bot_definitions` persona context, which needs conversion into a chat template before training - the shape the SFT method card expects. `pippa_metharme.jsonl` instead ships pre-formatted single `prompt`/`generation` completion pairs in PygmalionAI's own Metharme instruction format, ready for completion-style SFT without further templating [2][3]. Screen for the not-for-all-audiences content before use.

**Licence**: apache-2.0 (`cardData.license` is `"apache-2.0"`), ungated (`"gated": false`, `"private": false`) [4]. The one catch: the Apache-2.0 grant covers PygmalionAI's redistribution rights to the release; it is not a statement about the completeness of the PII redaction or a licence-level content restriction, both of which are the card's own use-time cautions above [2][4].

**Shape**: no declared splits or configs in the Hub metadata (`"splits": {}` in the shortlist row, and the Hub dataset viewer is disabled) - three flat JSONL files, `pippa.jsonl` (25,960 lines), `pippa_deduped.jsonl` (16,832 lines) and `pippa_metharme.jsonl` (16,832 lines), read directly from the repository [5][6].

**Hold out**: nothing. No source read for this card - the README, the paper, or the corpus screening note - flags any overlap with an evaluation set, and the release ships as flat logs with no train/test split to begin with [1][2][5].

**Origin**: built and released by PygmalionAI; each conversation's human turns are real user input and the bot turns are Character.AI's proprietary LLM output, i.e. partially synthetic [1][2]. Hub API at the check date: `downloads` 734, `downloadsAllTime` 14,161, `likes` 245 [4].

**Trained-on-by**: a community LoRA fine-tune, `DS-Archive/mistral-v0.1-7b-pippa-metharme-lora` (Mistral-7B), tags itself `dataset:PygmalionAI/PIPPA` and cites the origin paper [7]. The `chimbiwide/pippa` and `chimbiwide/pippa_filtered` reformats state their maintainer "trained our Gemma3NPC models" and "Gemma3NPC-Filtered models" respectively using those reformatted copies of this release [8]. The Hub's arXiv-citations API lists eleven further models, all from PJMixers-Dev (the Gemma-3-Earthen, Granite-3.1-Earthen and Gemma-3-Starshine-Earthen families); each cites the origin arXiv id but, checked directly against each model's own tags, none carries `dataset:PygmalionAI/PIPPA` - all eleven instead tag `dataset:grimulkan/PIPPA-augmented-dedup`, a derivative of this release, alongside dozens of unrelated datasets [7][9].

**Introduced by**: [1] (Gosling, Dale & Zheng).

## Shape

Row counts, read directly from each JSONL file at the pinned revision (one line per conversation) [5]:

| file | rows | bytes (full download) |
| --- | --- | --- |
| `pippa.jsonl` | 25,960 | 348,589,691 |
| `pippa_deduped.jsonl` | 16,832 | 256,837,235 |
| `pippa_metharme.jsonl` | 16,832 | 193,294,206 |

The README rounds this to "26,000 conversations" and the paper to "nearly 26,000"; the line count read directly from `pippa.jsonl` is 25,960 [1][2][5]. `pippa_deduped.jsonl` and `pippa_metharme.jsonl` carry the same row count because the metharme file is a reformat of the deduped file, not an independent variant [2].

Columns, read from the repository's `PIPPA.py` loading script [3]. For `pippa` and `pippa_deduped` (identical schema):

| column | dtype |
| --- | --- |
| `submission_timestamp` | timestamp[ms] |
| `categories` | sequence\<string\> |
| `bot_id` | string |
| `bot_name` | string |
| `bot_greeting` | string |
| `bot_definitions` | string |
| `bot_description` | string |
| `conversation` | sequence of `{message: string, is_human: bool}` |

For `pippa_metharme`:

| column | dtype |
| --- | --- |
| `prompt` | string |
| `generation` | string |

The loading script always yields a single `train` split per config, regardless of file [3]. No source states aggregate sequence-length or token statistics for the whole release; the paper does state per-conversation turn statistics (see Quality) but does not say which file variant it measured [1].

## Quality

- Submission was opt-in: the paper states users could "opt out of including their conversations in the public release" and that PIPPA "solely contains logs for which users have explicitly granted permission for public distribution" [1].
- The maintainers ran PII detection and redaction, described in the paper as performed "to the best of our ability" and "not obligatory" but done "as a matter of ethical prudence"; the README separately invites reports of any unredacted PII that slipped through [1][2].
- `pippa_deduped.jsonl` removes duplicate conversations and any conversation with fewer than three turns from `pippa.jsonl`, per the README's file description [2].
- The paper reports conversation-length statistics: median 10 turns, mean 40.41 turns, standard deviation 145 turns, and a longest conversation of 11,491 turns; message-length distribution follows a power law with bot replies generally more verbose than human turns [1].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure; none is invented here.
- The README's content warning: PIPPA "contains conversations, themes and scenarios which can be considered 'not safe for work' (NSFW) and/or heavily disturbing in nature," and models trained purely on it may tend toward X-rated output [2].

## Load it

No config argument loads `pippa_deduped` (the loading script's `DEFAULT_CONFIG_NAME`), pinned to the revision this card's numbers were read at [4][3]:

```python
from datasets import load_dataset

REV = "6412b0cae4d879b678e7a33df3ba076b9581f4d4"  # main at the check date
ds = load_dataset("PygmalionAI/PIPPA", "pippa_deduped", revision=REV, trust_remote_code=True)  # 16,832 rows
```

**Trap**: `PIPPA.py` defines three configs with different row counts and, for `pippa_metharme`, a different schema - passing no config name silently gives you `pippa_deduped` (16,832 rows), `"pippa"` gives the larger, non-deduplicated 25,960-row file, and `"pippa_metharme"` gives a `prompt`/`generation` completion-pair schema instead of the `conversation`-list schema of the other two [3]. Because `cardData.viewer` is `false`, the Hub dataset viewer and the datasets-server `/info`, `/size` and `/first-rows` endpoints all return HTTP 501 for this repository - the counts and rows on this card were read by downloading the JSONL files directly, not through those endpoints [4][10].

## Neighbors

Reformats and derivatives, row counts fetched live at the check date where the source has a working viewer [7][8][11]:

- `kingbri/PIPPA-shareGPT` - a ShareGPT-format conversion of `pippa_deduped.jsonl` for Axolotl fine-tuning, in three variants (`_raw`, plain, and `_trimmed`, the last recommended by its own card); its dataset viewer is disabled, so no live row count [8].
- `KaraKaraWitch/PIPPA-ShareGPT-formatted` and `mpasila/PIPPA-ShareGPT-formatted-named` - further ShareGPT reformats built on `kingbri/PIPPA-shareGPT`, the latter adding randomized character names; both also have their viewer disabled [8].
- `chimbiwide/pippa` - a processed ChatML-style reformat (single `messages` column) of this release, 16,832 rows, matching this release's deduplicated row count [11].
- `chimbiwide/pippa_filtered` - a filtered subset of the above, 3,258 rows [11].
- `grimulkan/PIPPA-augmented-dedup` - built from `kingbri/PIPPA-shareGPT`'s trimmed file, with names substituted via Faker, conversations under 50 tokens dropped, further deduplication keeping the longest unique conversation, and turns forced to alternate user/assistant; 915 rows [11].

This corpus prefers the original `PygmalionAI/PIPPA` (`pippa_deduped.jsonl`) over these reformats: it is the source every reformat above is built from, and mixing an unlisted-revision reformat with this release risks duplicate conversations in training data.

## A row

Two schemas are served. From `pippa_deduped.jsonl` (config `pippa_deduped`, split `train`, row 0, `conversation` truncated from 89 turns) [5]:

```json
{
  "submission_timestamp": 1675002668639,
  "categories": null,
  "bot_id": "1lYx-3u21uWsKPI1ghqWrOTdIvU6T-CEXTXy0arv46A",
  "bot_name": "Kamisato Ayaka",
  "bot_greeting": "Fufu get used to this new Ayaka.",
  "bot_definitions": "{{char}}: Fufu get used to this new Ayaka.\n{{user}}: Akane?\n{{char}}: Yes, did you forget my new name? ... [truncated, 3,140 chars total]",
  "bot_description": "She is Akane the maid of Ayaka. One day Ayaka found a gold ring that grant all the wishes of its wearer. The ring was stuck on Ayaka`s finger. Akane came up with a plan. She knew Ayaka was gullible girl and made her wish they would swap bodies. ...",
  "conversation": [
    {"message": "Fufu get used to this new Ayaka.", "is_human": false},
    {"message": "Who are you?", "is_human": true},
    {"message": "I am your new Mistress! *laugh*", "is_human": false}
    /* ... 86 more turns ... */
  ]
}
```

From `pippa_metharme.jsonl` (config `pippa_metharme`, split `train`, row 0 - the metharme-formatted reformat of the same conversation above) [5]:

```json
{
  "prompt": "<|system|>You are now in roleplay conversation mode. You should act according to this character sheet:\n\nShe is Akane the maid of Ayaka. One day Ayaka found a gold ring that grant all the wishes of its wearer. ...\n\nYou must stay in-character at all times, and generate messages as if you were Kamisato Ayaka.<|model|>Fufu get used to this new Ayaka.<|user|>Who are you?<|model|>I am your new Mistress! *laugh*<|user|>Huh? ... [truncated, 15,909 chars total]",
  "generation": "*She then look at {{user}} with her pretty eyes as she smile kindly.* Then your wish is my order!"
}
```

## Where it came from

PygmalionAI built and released PIPPA from logs submitted through a community-run userscript that scraped a user's own Character.AI conversation history and the character's persona fields ("Definitions" and "Description") from the Character.AI website [1][2]. Contributors could opt out of public release, so the shipped data is limited to logs whose submitters explicitly granted redistribution permission; the maintainers additionally scanned and redacted personally identifiable information before publication [1]. The human side of each conversation is real user text, and the assistant side is Character.AI's own proprietary LLM's generations, which is what the paper's title calls "partially synthetic" [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12. Hub repositories are mutable, which is why Load it pins the revision; this repository's Hub API `sha` for `main` matched the shortlist row's pinned commit at the check date, so the file counts and row/byte figures above are covered by that pin. The datasets-server endpoints (`/info`, `/size`, `/first-rows`) take no revision parameter and are live, disabled-viewer responses, not pinned data.

[1] Gosling, Dale & Zheng, "PIPPA: A Partially Synthetic Conversational Dataset", 2023. https://arxiv.org/abs/2308.05884 - the origin paper; read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2308.05884), current title read from the live abs page. Fetched 2026-08-12.

[2] PygmalionAI/PIPPA dataset card (README). https://huggingface.co/datasets/PygmalionAI/PIPPA/raw/main/README.md - dataset summary, file descriptions, consent/redaction statements, content warning, citation. Fetched 2026-08-12.

[3] `PIPPA.py`, the repository's `datasets` loading script. https://huggingface.co/datasets/PygmalionAI/PIPPA/raw/main/PIPPA.py - config names, `DEFAULT_CONFIG_NAME`, per-config feature schema, and the single `train` split generator. Fetched 2026-08-12.

[4] Hugging Face Hub API record for PygmalionAI/PIPPA. https://huggingface.co/api/datasets/PygmalionAI/PIPPA?full=true - licence, gated/private status, `sha`, `cardData.viewer`, siblings, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[5] Row counts, byte sizes, and sample rows read directly from `pippa.jsonl`, `pippa_deduped.jsonl` and `pippa_metharme.jsonl` at https://huggingface.co/datasets/PygmalionAI/PIPPA/resolve/main/ - full-file line counts via streamed download, byte sizes via the resolved-file `Content-Length`, and sample rows via ranged HTTP GET on the first line of `pippa_deduped.jsonl` and `pippa_metharme.jsonl`. Fetched 2026-08-12.

[6] The corpus screening row for `PygmalionAI/PIPPA`, supplied with this card's request - `splits`, `info_status`, `size_status` and `viewer_gap` fields, consistent with the disabled-viewer endpoints in [10]. Checked 2026-08-12.

[7] Models and datasets citing arXiv:2308.05884, via the Hugging Face arXiv-citations API. https://huggingface.co/api/arxiv/2308.05884/repos - twelve citing models and six citing datasets at the check date, including `DS-Archive/mistral-v0.1-7b-pippa-metharme-lora`'s tags. Fetched 2026-08-12.

[8] Neighbor dataset cards (READMEs) with no live viewer: `kingbri/PIPPA-shareGPT`, `KaraKaraWitch/PIPPA-ShareGPT-formatted`, `mpasila/PIPPA-ShareGPT-formatted-named`. https://huggingface.co/datasets/<id>/raw/main/README.md Fetched 2026-08-12.

[9] Hugging Face Hub API model records for the eleven PJMixers-Dev repos listed as citing arXiv:2308.05884 in [7] (the Gemma-3-Earthen, Granite-3.1-Earthen and Gemma-3-Starshine-Earthen families). https://huggingface.co/api/models/<id> - each model's own `tags` list, checked for a `dataset:` tag; all eleven tag `dataset:grimulkan/PIPPA-augmented-dedup` and none tags `dataset:PygmalionAI/PIPPA`. Fetched 2026-08-12.

[10] datasets-server `/info`, `/size` and `/first-rows` endpoints for PygmalionAI/PIPPA. https://datasets-server.huggingface.co/info?dataset=PygmalionAI%2FPIPPA , https://datasets-server.huggingface.co/size?dataset=PygmalionAI%2FPIPPA , https://datasets-server.huggingface.co/first-rows?dataset=PygmalionAI%2FPIPPA&config=pippa_deduped&split=train - all three return `{"error":"Not supported: dataset viewer is disabled."}`. Fetched 2026-08-12.

[11] Neighbor dataset cards and live datasets-server `/size` endpoints for `chimbiwide/pippa`, `chimbiwide/pippa_filtered` and `grimulkan/PIPPA-augmented-dedup`. https://huggingface.co/datasets/<id>/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=<id> - row counts are live, not pinned, since this endpoint takes no revision parameter. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as multi-turn roleplay/persona chat SFT data, recommended via the deduplicated file, with the not-for-all-audiences content warning carried into the opening paragraph above. This rests on facts already established on this card: the README's own recommendation of `pippa_deduped.jsonl`, its consent and redaction statements, and its content warning [1][2], together with the screening row's note below [6].

### The screening row

The row's own note [6]: "About 26,000 consented roleplay conversation logs from the Pygmalion project, each pairing a community-written character persona with the bot's replies, shipped as raw, deduplicated and Metharme-formatted JSONL; the custom format means the viewer returns HTTP 501." The row carries no flag.
