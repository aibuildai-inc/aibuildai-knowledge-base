# zhuzilin/aime-2024

30 single-turn math problems, each row a chat-formatted prompt plus a numeric-string answer label, served under a split literally named `train`.

**zhuzilin/aime-2024** packages 30 math competition problems as a `prompt`/`label` pair per row, where `prompt` is a one-turn `[{"role": "user", "content": ...}]` list and `label` is a short numeric answer string [1]. The repository ships one file, `aime-2024.jsonl`, and carries no README or card metadata (`card_body_bytes` is 0) [2][3], so no builder statement, licence, problem source, or intended-use text exists anywhere in the repo. It lives at https://huggingface.co/datasets/zhuzilin/aime-2024 . The corpus screening record flags this repository as a duplicate of `HuggingFaceH4/aime_2024` [6], whose own card identifies its problem set as the 2024 AIME I and AIME II competition tests [5]; comparing one row from each (row 15 here against row 0 there) shows the same problem stated in different wording, consistent with that flag [4]. **This is an evaluation set, not training data: the screening record flags it as duplicating that 2024 AIME set, and its only split is named `train`, which invites accidental training on eval rows [6].**

**Use it for**: nothing to train on - this is a held-out evaluation benchmark for math reasoning, formatted as single-turn chat prompts with a short-answer `label`; do not point an SFT or RL training run at it. There is no method card this dataset feeds; it is a contamination risk to check against, not a training source.

**Licence**: not stated. The repo carries no `license` card field and no licence text in the body (`licence_card_field` is null, `licence_body_mentions` is empty) [1][2]; nothing in the fetched sources states terms.

**Shape**: 30 rows, one config (`default`), one split, misleadingly named `train`; two columns, `prompt` and `label` [7][8].

**Hold out**: all 30 rows, from any training run - the entire dataset is a competition-eval set that the corpus screening record flags as a duplicate of `HuggingFaceH4/aime_2024` [6]; a single-row check is consistent with that flag [4].

**Origin**: repository owner `zhuzilin`; problems are human-authored competition questions with their official numeric answers, not model-generated [1]. Hub API at the check date: 1,920 `downloads`, 21,064 `downloadsAllTime`, 3 `likes` [2].

**Trained-on-by**: none found - no source fetched for this card names a model or training recipe trained on `zhuzilin/aime-2024`.

**Introduced by**: no paper and no dataset card - the repository has no README (`card_body_bytes` is 0) [2][3].

## Shape

Rows and split, from the datasets-server size endpoint [7]:

| split | rows |
| --- | --- |
| `train` | 30 |

One config, `default`, two columns, from the datasets-server info endpoint [8]:

| column | dtype |
| --- | --- |
| `prompt` | list\<struct\<role: string, content: string\>\> |
| `label` | string |

No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- Reading all 30 served rows (the full dataset, not a sample - `first-rows` reports `"truncated": false`) shows every `prompt` is a single `user` turn with no system or assistant turn, and every `label` is a short numeric string (AIME answers range 0-999) [1].
- No source states a measured contamination or duplicate rate for this specific repository. The comparison in Neighbors below, read from both datasets' own served rows, is the contamination evidence this card has: row 15 here (`label: "204"`) and row 0 of `HuggingFaceH4/aime_2024` (`answer: "204"`) state the same walking-speed problem in different wording, indicating the two repositories carry the same underlying 30-problem set [1][4].
- The repository has no README, so there is no builder statement about collection process, annotation, or known issues [3].

## Load it

```python
import datasets

REV = "1c625e328db94ec7ef7ff169016b097c468d60b9"  # main at the check date
train = datasets.load_dataset("zhuzilin/aime-2024", revision=REV, split="train")  # 30 rows
```

**Trap**: the only split is named `train`, but every row is a held-out competition-eval problem - loading it with `split="train"` and feeding it straight into a training loop trains on eval data. There is no separate eval split to distinguish it from; the split name is the whole dataset [1].

## Neighbors

- `HuggingFaceH4/aime_2024` - flagged by the corpus screening record as this repository's duplicate [6], reformatted into `id`, `problem`, `solution`, `answer`, `url`, `year` columns instead of chat `prompt`/`label`; its own card states its problems are the 2024 AIME I and AIME II tests, drawn from `AI-MO/aimo-validation-aime` [5]. A single-row check (row 0 there, answer `"204"`, versus row 15 here, label `"204"`) shows both state the same walking-speed problem in different phrasing, consistent with that flag though only one problem pair was compared [1][4]. Prefer `HuggingFaceH4/aime_2024` when a `problem`/`solution`/`answer` shape is wanted, or this repository when a ready-made chat `prompt`/`label` shape is wanted; either way, hold out both together, since training on one and evaluating on the other risks leaking the answers.
- `AI-MO/aimo-validation-aime` - 90 rows across AIME 2022, 2023, and 2024, five columns, Apache-2.0 licensed per its Hub tags; `HuggingFaceH4/aime_2024`'s own card names this as its source dataset [5][9]. It is the larger pool this 30-row 2024 subset was drawn from, not a distinct problem set - holding out this repository's 30 rows does not clear the 2024 problems still present inside `AI-MO/aimo-validation-aime`.

## A row

One config, one split, so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [1]:

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "Among the $900$ residents of Aimeville, there are $195$ who own a diamond ring, $367$ who own a set of golf clubs, and $562$ who own a garden spade. In addition, each of the $900$ residents owns a bag of candy hearts. There are $437$ residents who own exactly two of these things, and $234$ residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things."
    }
  ],
  "label": "73"
}
```

## Where it came from

The repository is owned by Hub user `zhuzilin` and contains one file, `aime-2024.jsonl`, with no README or card metadata to describe how it was built [2][3]. Its problems are human-authored, externally administered competition questions, not model-generated content [1]; the shortlist row's own note independently states the same origin - "AIME 2024, 30 rows, split misleadingly named `train`" [6].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=zhuzilin%2Faime-2024&config=default&split=train - all 30 rows, columns, and `"truncated": false`. Fetched 2026-08-11.

[2] Hugging Face Hub API record for zhuzilin/aime-2024. https://huggingface.co/api/datasets/zhuzilin/aime-2024?full=true - `sha`, `downloads`, `likes`, `lastModified`, tags, siblings; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[3] Attempted fetch of the dataset README. https://huggingface.co/datasets/zhuzilin/aime-2024/raw/main/README.md - returns "Entry not found", confirming no card body exists. Fetched 2026-08-11.

[4] datasets-server first-rows endpoint for the neighbor. https://datasets-server.huggingface.co/first-rows?dataset=HuggingFaceH4%2Faime_2024&config=default&split=train - row 0, used for the row-level content comparison. Fetched 2026-08-11.

[5] HuggingFaceH4/aime_2024 dataset card (README) and Hub API record. https://huggingface.co/datasets/HuggingFaceH4/aime_2024/raw/main/README.md and https://huggingface.co/api/datasets/HuggingFaceH4/aime_2024?full=true - states the problems are the 2024 AIME I and AIME II tests, column shape, row count, and that the source is `AI-MO/aimo-validation-aime`. Fetched 2026-08-11.

[6] The corpus screening row for `zhuzilin/aime-2024`, supplied with this card's request - its `note` and `flag`, read back in the appendix. Checked 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=zhuzilin%2Faime-2024 Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=zhuzilin%2Faime-2024 Fetched 2026-08-11.

[9] datasets-server size endpoint and Hub API record for AI-MO/aimo-validation-aime. https://datasets-server.huggingface.co/size?dataset=AI-MO%2Faimo-validation-aime and https://huggingface.co/api/datasets/AI-MO/aimo-validation-aime?full=true - row count, column count, licence tag. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Contamination risk, hold out entirely: the dataset is a competition-eval set for math reasoning, served under a split named `train`, and the corpus screening record flags it as a duplicate of `HuggingFaceH4/aime_2024` [6]. That rests on what this card already established - the screening flag itself [6], the split-naming trap in Load it [1], and a single-row comparison in Neighbors that is consistent with the flag though it checked only one problem pair [4].

### The screening row

The row's own note [6]: "AIME 2024, 30 rows, split misleadingly named `train`." Its flag [6]: "contamination: AIME 2024 with the split named train; duplicate of HuggingFaceH4/aime_2024".
