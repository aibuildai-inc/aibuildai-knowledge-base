# glaiveai/glaive-function-calling-v2

112,960 single- and multi-turn assistant chats teaching function/tool calling, each row a system prompt with JSON function definitions plus a full USER/ASSISTANT/FUNCTION-RESPONSE transcript in one flat string.

**glaiveai/glaive-function-calling-v2** is Glaive AI's second-generation function-calling dataset, generated through Glaive's synthetic data platform [1]; the dataset card gives no description of the release itself beyond its license and task tags [1], but the sibling first-generation release from the same org states the same platform and format for its 52k samples [2]. Each row's `chat` column is a plain-text transcript of `USER:`/`ASSISTANT:` turns, with tool invocations written as `ASSISTANT: <functioncall> {json}` and tool results as `FUNCTION RESPONSE: {json}`, and each row's `system` column holds the system prompt plus the available functions as inline JSON [3]. **The generating model is never named in any fetched source** [1][2][3]. It lives at https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 .

**Use it for**: reasoning-trace / tool-use SFT on flat text transcripts (system + chat), not a chat-template or messages-list format - map `system` and `chat` into whatever prompt template the target model uses, splitting `chat` on its `USER:`/`ASSISTANT:`/`FUNCTION RESPONSE:` markers. See the SFT method card.

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, repo tags include `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [1][6]. No licence catch is stated in any fetched source.

**Shape**: 112,960 rows, one config (`default`), one split (`train`), two string columns (`system`, `chat`) [4][5].

**Hold out**: nothing - no fetched source states or implies any evaluation-set overlap for this release.

**Origin**: built and released by Glaive AI (`glaiveai`); both the function-call outputs and the labels are synthetic, generated through Glaive's platform with no stated human annotation step [1][2]. Hub API at the check date: `downloads` 32,042, `downloadsAllTime` 245,496, `likes` 522 [6].

**Trained-on-by**: Stability AI's `stabilityai/stablelm-2-12b-chat` lists this dataset in its training mix twice - once in the card's `datasets:` metadata and once in its own dataset listing in the body [7]. jondurbin's `bagel-8b-v1.0` model card lists it among its training datasets and separately documents "GlaiveAI function calling" as one of two function-calling prompt formats the model was trained to handle [8].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 112,960 |

One config, `default`, two columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `system` | string |
| `chat` | string |

No source states sequence-length or token statistics for this release; the only size figures available are byte counts (below).

Sizes (datasets-server `/size`) [4]: 271,190,065 bytes of original JSON download, 97,037,816 bytes as Parquet, 254,485,835 bytes decoded in memory. The Hub tree at `main` lists a single data file, `glaive-function-calling-v2.json`, at 271,190,065 bytes [9].

## Quality

- No fetched source states a measured contamination rate, duplicate rate, or human-review figure for this release.
- The sibling v1 release's card states that its 52k samples include "samples which do not have any function invocations, multiple invocations and samples with no functions presented and invoked to keep the data balanced" [2]; no fetched source confirms this same balancing statement for v2 specifically.
- Of the first 75 rows served for `train` (offsets 0-74), 59 contain more than one `USER:` turn (multi-turn) and 16 contain exactly one; 38 of the 75 contain at least one `<functioncall>` invocation [3]. This describes only those 75 rows at offset 0, not the full 112,960-row split.

## Load it

```python
import datasets

REV = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"  # main at the check date
train = datasets.load_dataset("glaiveai/glaive-function-calling-v2", revision=REV, split="train")  # 112,960 rows
```

**Trap**: the repository ships a single raw JSON file (`glaive-function-calling-v2.json`), not a `data/` Parquet layout, so `builder_name` reported by datasets-server is `json` [5]; the whole 271 MB source file is downloaded and parsed on first load. Both `system` and `chat` are single plain-text strings with embedded `USER:`/`ASSISTANT:`/`FUNCTION RESPONSE:` markers and literal `<functioncall>` tags - there is no `messages` list or chat-template structure to consume directly, and any newline- or marker-based parser must be written before this data can feed a chat-template trainer.

## Neighbors

- `glaiveai/glaive-function-calling` (the org's first-generation release) - 52,000 rows, same `SYSTEM:`/`USER:`/`ASSISTANT:`/`FUNCTION RESPONSE:` text format and the same Glaive platform origin, per its own card [2]. This corpus's screening row calls the v2 release here "the canonical" 113k version [10], and v2 is roughly double v1's row count with no stated relationship (superset, disjoint, or resample) between the two in any fetched source.
- No other Glaive-AI-org function-calling re-release, cleaned variant, or binarized successor was found in the sources fetched for this card.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=1` (datasets-server `/first-rows`) [3], chosen because it shows both a function call and a function response over two turns:

```json
{
  "system": "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{\n    \"name\": \"get_news_headlines\",\n    ...\n}\n",
  "chat": "USER: Can you tell me the latest news headlines for the United States?\n\n\nASSISTANT: <functioncall> {\"name\": \"get_news_headlines\", \"arguments\": '{\"country\": \"United States\"}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {\"headlines\": [\"Biden announces new vaccine mandates\", \"Hurricane Ida devastates Louisiana\", \"Apple unveils new iPhone\", \"NASA's Perseverance rover collects first Mars rock sample\"]}\n\n\nASSISTANT: Here are the latest news headlines for the United States:\n1. Biden announces new vaccine mandates\n2. Hurricane Ida devastates Louisiana\n3. Apple unveils new iPhone\n4. NASA's Perseverance rover collects first Mars rock sample <|endoftext|>\n\n\nUSER: That's interesting. What about the news in France?\n\n\nASSISTANT: <functioncall> {\"name\": \"get_news_headlines\", \"arguments\": '{\"country\": \"France\"}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {\"headlines\": [\"France recalls ambassadors to US and Australia\", \"French election: Macron's party braces for tough fight\", \"Louvre Museum to undergo major overhaul\", \"France to offer free birth control to all women under 25\"]}\n\n\nASSISTANT: Here are the latest news headlines for France:\n1. France recalls ambassadors to US and Australia\n2. French election: Macron's party braces for tough fight\n3. Louvre Museum to undergo major overhaul\n4. France to offer free birth control to all women under 25 <|endoftext|>\n\n\n"
}
```

## Where it came from

Built and released by Glaive AI. No fetched source, including this dataset's own card, states a collection method, a generating model, or a human-review step for the v2 release; the only stated origin fact is that it was "generated through Glaive" [2], repeated for the sibling v1 release whose card the same org wrote in fuller prose [2]. The generating model behind Glaive's platform is not named in any fetched source [1][2][3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. The row counts and byte sizes in Shape, and the sampled row in A row, come from the datasets-server `/size`, `/info`, and `/first-rows` endpoints, which take no `revision` parameter and were confirmed live to return identical figures when queried with a nonexistent revision string - so those numbers describe whatever is live on `main` at the check date, not the pinned commit `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` used in Load it.

[1] glaiveai/glaive-function-calling-v2 dataset card (README). https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/raw/main/README.md - the entire body is YAML front matter (license, task_categories, language, size_categories); no prose description follows. Fetched 2026-08-11.

[2] glaiveai/glaive-function-calling (v1) dataset card (README). https://huggingface.co/datasets/glaiveai/glaive-function-calling/raw/main/README.md - platform origin, format description, and the balancing statement for the 52k sibling release. Fetched 2026-08-11.

[3] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=glaiveai%2Fglaive-function-calling-v2&config=default&split=train Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=glaiveai%2Fglaive-function-calling-v2 Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=glaiveai%2Fglaive-function-calling-v2 Fetched 2026-08-11.

[6] Hugging Face Hub API record for glaiveai/glaive-function-calling-v2. https://huggingface.co/api/datasets/glaiveai/glaive-function-calling-v2?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[7] stabilityai/stablelm-2-12b-chat model card (README). https://huggingface.co/stabilityai/stablelm-2-12b-chat/raw/main/README.md - lists `glaiveai/glaive-function-calling-v2` in both the YAML `datasets:` field and the body's training-data list. Fetched 2026-08-11.

[8] jondurbin/bagel-8b-v1.0 model card (README). https://huggingface.co/jondurbin/bagel-8b-v1.0/raw/main/README.md - lists the dataset in its training-data table and documents "GlaiveAI function calling" as one of two supported function-calling prompt formats. Fetched 2026-08-11.

[9] Hugging Face Hub tree API for glaiveai/glaive-function-calling-v2 at `main`. https://huggingface.co/api/datasets/glaiveai/glaive-function-calling-v2/tree/main - file listing and byte sizes. Fetched 2026-08-11.

[10] The corpus screening row for `glaiveai/glaive-function-calling-v2`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace / tool-use SFT data. The screening row's note identifies this as the canonical 113k Glaive v2 release and flags that the generating model is never named [10]; no fetched source states any evaluation-set overlap, licence restriction beyond Apache-2.0, or usage-shape limit for this release.

### The screening row

The row's own note [10]: "the canonical 113k Glaive v2 single- and multi-turn function-calling chats from Glaive's synthetic platform; the generating model is never named." The row carries no flag.
