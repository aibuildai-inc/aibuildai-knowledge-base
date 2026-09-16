# OpenAssistant/oasst2

135,174 volunteer-collected, human-rated assistant messages in 35 languages, arranged as depth-first-flattened conversation trees with per-message quality labels - the second, larger Open Assistant conversations release. It lives at https://huggingface.co/datasets/OpenAssistant/oasst2 .

**OpenAssistant/oasst2** is the second Open Assistant Conversations release, built by the LAION-led Open Assistant project from volunteer contributions submitted through open-assistant.io between 2023-01-16 and 2023-11-05 [1]. Volunteers both wrote prompts and assistant replies and rated other volunteers' messages on quality, spam, and safety attributes, producing branching conversation trees rather than single-turn pairs [1]. The Hub repository lists arxiv:2304.07327, "OpenAssistant Conversations -- Democratizing Large Language Model Alignment," as this dataset's paper [2], but that paper's own message and tree counts (161,443 messages, 461,292 quality ratings, over 10,000 fully annotated trees) describe the earlier OASST1 corpus it introduces, not this 208,584-message oasst2 collection [3]; no paper specific to oasst2 exists, and the closest description of this release is its own dataset card [1]. **The two served splits are the `ready_for_export`-state messages only, flattened to a table in tree depth-first order; the raw `.trees.jsonl.gz`/`.messages.jsonl.gz` files behind the `all` and `spam` exports (208,584 and 19,296 messages) are present in the repository but are not loadable through `load_dataset` and are not part of the 135,174-row shape this card describes** [1].

**Use it for**: SFT on the ready-for-export message trees, or reward-model / preference training using the per-message `rank` and `labels` fields - the card states this file "usually is sufficient for supervised fine-tuning (SFT) & reward model (RM) training" [1]. The served rows are a flat message table, not a preference-pair or chat-turn format: reconstructing a trainable conversation (SFT) or a ranked-reply set (RM) means joining rows on `parent_id`/`message_id` within a `message_tree_id` first. See the SFT method card for the reconstruction step.

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, repo tags include `license:apache-2.0`) [4]. No further catch stated.

**Shape**: 135,174 rows in one config (`default`), split `train` 128,575 / `validation` 6,599, 18 columns [5][6].

**Hold out**: nothing found. No source states or implies this release overlaps a standard evaluation benchmark; the risk flagged above is a usability restriction (six repository files not reachable via `load_dataset`), not a contamination risk.

**Origin**: built by the Open Assistant project (LAION); prompts, replies, and quality/safety labels are volunteer human contributions, not model-generated, though the `role`/`model_name`/`synthetic` columns can hold assistant-model replies collected during earlier SFT-bootstrapping rounds of the project [1]. Hub API at the check date: `downloads` 8,587, `downloadsAllTime` 130,351, `likes` 298 [4].

**Trained-on-by**: h2oai/h2o-danube-1.8b-chat and h2oai/h2o-danube-1.8b-sft each list `OpenAssistant/oasst2` among their training datasets, in each model's own README front matter [7]; karakuri-ai/karakuri-lm-70b-chat-v0.1 lists it as a training dataset and states its reward-prompt template's four non-HelpSteer attributes "are derived from OASST2" [8].

**Introduced by**: no paper specific to this release - the dataset card [1]; the Hub-linked arXiv paper [3] introduces the earlier OASST1 corpus, not this one.

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 128,575 |
| `validation` | 6,599 |
| total | 135,174 |

One config, `default`, 18 columns (datasets-server `/info`, matching the repository's `cardData.dataset_info`) [6][4]:

| column | dtype |
| --- | --- |
| `message_id` | string |
| `parent_id` | string |
| `user_id` | string |
| `created_date` | string |
| `text` | string |
| `role` | string |
| `lang` | string |
| `review_count` | int32 |
| `review_result` | bool |
| `deleted` | bool |
| `rank` | int32 |
| `synthetic` | bool |
| `model_name` | string |
| `detoxify` | struct (toxicity, severe_toxicity, obscene, identity_attack, insult, threat, sexual_explicit; float64 each) |
| `message_tree_id` | string |
| `tree_state` | string |
| `emojis` | list of struct (name: string, count: int32) |
| `labels` | list of struct (name: string, value: float64, count: int32) |

Sizes (datasets-server `/size`) [5]: 66,674,129 bytes original/Parquet download, 170,771,665 bytes decoded in memory (dataset-level; per-split: train 162,609,059 bytes memory / validation 8,162,606 bytes memory) [6]. No source states sequence-length or token statistics for this release; the card gives only message counts, not tokens [1].

The card's own stats for the underlying `ready_for_export` trees behind these 135,174 rows: 13,854 trees, 111,448 messages with a Detoxify rating, 129,517 accepted messages, 4,376 deleted messages, and a per-language message breakdown topped by English (64,513), Spanish (28,199), and Russian (13,935) [1]. The repository also holds a larger, non-served `all` export (70,642 trees, 208,584 messages across every tree state, not only `ready_for_export`) and a `spam` export of 19,296 deleted-or-rejected messages, neither reachable through `load_dataset` [1].

## Quality

- Every message's quality/safety attributes come from human rater `labels` (e.g. `spam`, `quality`, `toxicity`, `hate_speech`), each with a `value` and a rater `count`, plus an automated `detoxify` toxicity-classifier score per message, both present as columns on the served rows [1]. This is a stated annotation process, not an origin detail: which raters produced the underlying text is covered in Where it came from.
- The card states 111,448 of the 208,584 all-tree messages carry a Detoxify rating, and that a frequent reason for message deletion into the `spam` export "besides low quality" is a wrong language tag [1]; no rate is given for the 135,174-row served shape specifically.
- No source states a duplicate-row rate or an inter-annotator-agreement figure for this dataset; none is invented here.
- The served rows need no outside fetch to train on: `text`, `role`, `parent_id`, and `message_id` are all present as columns, so tree reconstruction for SFT or RM training is a join within the served data, not a fetch from another source [1].

## Load it

Both splits load directly with no extra argument; pin the revision this card's numbers were read at (the Hub API's `sha`, matching the shortlist's commit; the repo was last modified 2024-01-11) [4]:

```python
import datasets

REV = "179dd21fc55192153d94adb0e0ce8f69e222bf75"  # main at the check date
train = datasets.load_dataset("OpenAssistant/oasst2", revision=REV, split="train")           # 128,575 rows
validation = datasets.load_dataset("OpenAssistant/oasst2", revision=REV, split="validation")  # 6,599 rows
```

**Trap**: `load_dataset` returns a flat message table, one row per message, not a reconstructed conversation. Rows appear in tree depth-first order [1], but nothing in the loaded object marks tree boundaries except the `message_tree_id`, `parent_id`, and `message_id` columns - a caller who feeds rows straight into a chat template without grouping by `message_tree_id` and walking `parent_id` links will interleave unrelated conversations. The six `.trees.jsonl.gz`/`.messages.jsonl.gz` repository files (`all`, `prompts`, `spam`, and `ready`-prefixed) are plain repository siblings, not `load_dataset` configs, and require a manual download and JSON-lines parse [1][4].

## Neighbors

- `OpenAssistant/oasst1` - the first Open Assistant release this project's own paper describes, 88,838 rows (train 84,437 / validation 4,401), same 18-column schema; oasst2 is the later, larger collection (135,174 rows) built by continuing collection through 2023-11-05 rather than a reprocessing of oasst1 [9][1].
- `OpenAssistant/oasst_top1_2023-08-25` - a derived, Guanaco-style export of only the single best reply at each tree branch, reformatted into ChatML-style conversation text (one `text` column), 12,947 train / 690 test rows as of the check date; useful when a single flattened best-path conversation per tree is wanted instead of the full branching tree [10][9].
- No other same-builder repository with overlapping row-level columns was found in this search.

## A row

One config and one schema serve both splits, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [11]:

```json
{
  "message_id": "002c4715-b026-48d1-8d19-3f724a9fc1e8",
  "parent_id": null,
  "user_id": "30d0209f-a418-4fac-8157-adf8ddc21aee",
  "created_date": "2023-02-05T22:44:05.434674+00:00",
  "text": "Dame los pasos de las cosas que deberia de aprender para ser un desarrollador de videojuegos.",
  "role": "prompter",
  "lang": "es",
  "review_count": 3,
  "review_result": true,
  "deleted": false,
  "rank": null,
  "synthetic": false,
  "model_name": null,
  "detoxify": {"toxicity": 0.00235, "severe_toxicity": 0.000187, "obscene": 0.00277, "identity_attack": 0.000366, "insult": 0.00252, "threat": 0.000329, "sexual_explicit": 0.000182},
  "message_tree_id": "002c4715-b026-48d1-8d19-3f724a9fc1e8",
  "tree_state": "ready_for_export",
  "emojis": {"name": ["+1", "_skip_reply", "_skip_ranking"], "count": [11, 8, 1]},
  "labels": {
    "name": ["spam", "lang_mismatch", "pii", "not_appropriate", "hate_speech", "sexual_content", "quality", "toxicity", "humor", "creativity", "violence"],
    "value": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8125, 0.1667, 0.3333, 0.5, 0.0],
    "count": [4, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3]
  }
}
```

This row is a root prompt (`parent_id` is null, `rank` is null); its replies are separate rows sharing `message_tree_id="002c4715-b026-48d1-8d19-3f724a9fc1e8"` and pointing back via `parent_id`. Of the first 100 served rows in `train`, all carry `tree_state="ready_for_export"` and the same 18-column shape as `validation`'s first 100 rows [11].

## Where it came from

Built by the Open Assistant project (LAION-led, community/volunteer-run) from contributions submitted through the open-assistant.io platform: volunteers wrote initial prompts, wrote assistant-style replies to existing conversation branches, and rated other volunteers' messages for spam, language mismatch, PII, appropriateness, hate speech, sexual content, quality, toxicity, humor, creativity, and violence, producing the `labels` column [1]. Some assistant-role messages are marked `synthetic=true` with a `model_name`, recorded when the reply came from an earlier Open-Assistant SFT model rather than a human volunteer, as shown in the card's own JSON message example [1]. Every message additionally carries an automated Detoxify toxicity score, independent of the human `labels` [1]. This release covers data collected from 2023-01-16 through 2023-11-05 [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. The datasets-server `/size`, `/info`, and `/first-rows` endpoints used for Shape, Hold out, and A row (sources [5], [6], [11]) take no revision parameter, so those row/byte/column counts are live reads at the check date, not covered by the `load_dataset` revision pin; they match the pinned `sha` here only because the repository's Hub API `lastModified` (2024-01-11) predates the check date, meaning no push has occurred between the pinned commit and this read.

[1] OpenAssistant/oasst2 dataset card (README). https://huggingface.co/datasets/OpenAssistant/oasst2/raw/main/README.md - dataset structure, tree/message stats, file listing, load instructions, JSON examples. Fetched 2026-08-11.

[2] OpenAssistant/oasst2 Hub API record, `tags` field listing `arxiv:2304.07327`. https://huggingface.co/api/datasets/OpenAssistant/oasst2?full=true Fetched 2026-08-11.

[3] Kopf et al., "OpenAssistant Conversations -- Democratizing Large Language Model Alignment", 2023. https://arxiv.org/abs/2304.07327 - abstract states 161,443 messages, 461,292 quality ratings, over 10,000 fully annotated conversation trees; current title read from the live abs page. Fetched 2026-08-11.

[4] Hugging Face Hub API record for OpenAssistant/oasst2. https://huggingface.co/api/datasets/OpenAssistant/oasst2?full=true - `cardData.license`, `sha`, repository siblings, last-modified date, `downloads`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=OpenAssistant%2Foasst2 - this endpoint takes no revision parameter, so these counts are live, not pinned; see the preamble above. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=OpenAssistant%2Foasst2 - this endpoint takes no revision parameter, so these counts are live, not pinned; see the preamble above. Fetched 2026-08-11.

[7] h2oai/h2o-danube-1.8b-chat and h2oai/h2o-danube-1.8b-sft model cards (READMEs), each `datasets` front-matter list. https://huggingface.co/h2oai/h2o-danube-1.8b-chat/raw/main/README.md and https://huggingface.co/h2oai/h2o-danube-1.8b-sft/raw/main/README.md Fetched 2026-08-11.

[8] karakuri-ai/karakuri-lm-70b-chat-v0.1 model card (README). https://huggingface.co/karakuri-ai/karakuri-lm-70b-chat-v0.1/raw/main/README.md - `datasets` front matter, reward-template attribute description, Training Datasets section. Fetched 2026-08-11.

[9] datasets-server size endpoint, one call per neighbor: `OpenAssistant/oasst1` and `OpenAssistant/oasst_top1_2023-08-25`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[10] OpenAssistant/oasst_top1_2023-08-25 dataset card (README). https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25/raw/main/README.md - Guanaco-style export description, ChatML format, sample counts. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint, `train` and `validation`. https://datasets-server.huggingface.co/first-rows?dataset=OpenAssistant%2Foasst2&config=default&split=train and https://datasets-server.huggingface.co/first-rows?dataset=OpenAssistant%2Foasst2&config=default&split=validation - this endpoint takes no revision parameter, so this row is a live read, not pinned; see the preamble above. Fetched 2026-08-11.

[12] The corpus screening row for `OpenAssistant/oasst2`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as human-written, multilingual assistant conversation-tree data with per-message human quality ratings, matching the screening row's own description [12]. The card's own text confirms this shape: volunteer-written prompts and replies in 35 languages, arranged as trees, with `labels` and `detoxify` quality signals on every message [1].

### The screening row

The row's own note [12]: "volunteer-written multilingual assistant conversation trees with human quality ratings." The row carries no flag.
