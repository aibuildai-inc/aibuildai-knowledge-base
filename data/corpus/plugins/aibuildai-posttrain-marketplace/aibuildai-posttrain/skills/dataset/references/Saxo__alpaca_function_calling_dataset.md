# Saxo/alpaca_function_calling_dataset

112,390 single-turn function-calling examples in Alpaca's system/instruction/output shape, an Alpaca-format re-encoding of an existing Llama-3-template dataset rather than a fresh collection.

**Saxo/alpaca_function_calling_dataset** was built by Dr. Ji Yun Sung ("Saxo"), a data scientist at the Korean analytics firm Linkbricks, who reformatted `mzbac/function-calling-llama-3-format-v1.1` - a Llama-3-chat-template dataset - into Alpaca's `system`/`instruction`/`output` columns for function-calling training [1]. That parent dataset is itself described by its own author as "cleaned up" from `glaiveai/glaive-function-calling-v2`, with invalid JSON and function-argument values removed [2]. It lives at https://huggingface.co/datasets/Saxo/alpaca_function_calling_dataset . **The repository's `language` tags claim Korean and English [1], but every row sampled across the split's start, middle, and end is English; treat the dataset as English-only for language-conditioned use.**

**Use it for**: single-turn SFT on function-calling behavior - the `system` column holds a tool-use system prompt (often JSON function definitions), `instruction` holds the user turn, and `output` holds either a direct reply or a `<functioncall> {...}` JSON invocation; this maps to the SFT method card's instruction-response format, not to a preference or chat-multiturn shape. **Do not rely on it for Korean-language function calling** - see the caveat above.

**Licence**: apache-2.0 (`cardData.license` and the `license:apache-2.0` tag) [1], ungated (`"gated": false`, `"private": false`) [3]. No catch found: the grandparent `glaive-function-calling-v2` is also apache-2.0 [4], and the intermediate `mzbac` dataset declares no conflicting licence of its own [5].

**Shape**: 112,390 rows, one split (`train`), one config (`default`), three string columns: `system`, `instruction`, `output` [6][7].

**Hold out**: nothing - the repository serves only a `train` split, and none of the sources fetched (this dataset's own card, the parent's, the grandparent's, or the screening note) flags overlap with any evaluation set [1][2][4][8].

**Origin**: built by Saxo/Linkbricks as a format conversion, not a fresh collection; none of the three cards in the lineage (this one, `mzbac`'s, or `glaiveai`'s) states what model or process generated the underlying function-calling dialogues [1][2][4]. Hub API at the check date: 78 downloads, 0 likes [3].

**Trained-on-by**: none found - no source fetched for this card names a model or recipe trained on this specific re-encoding.

**Introduced by**: no paper - the dataset card [1], reformatting `mzbac/function-calling-llama-3-format-v1.1` [2], which in turn reformats `glaiveai/glaive-function-calling-v2` [4].

## Shape

Rows and split, from the datasets-server size endpoint [6]:

| split | rows |
| --- | --- |
| `train` | 112,390 |

One config, `default`, three columns, all string (datasets-server info endpoint) [7]:

| column | dtype |
| --- | --- |
| `system` | string |
| `instruction` | string |
| `output` | string |

Sizes (datasets-server size endpoint) [6]: 139,311,220 bytes of original CSV download, 45,120,206 bytes as Parquet, 134,801,020 bytes decoded in memory. No source states sequence-length or token statistics for this release.

## Quality

- Of 120 rows sampled across three points in `train` - the first 100 rows at offset 0 [9], 10 rows at offset 56,000 [10], and the final 10 rows at offset 112,380 [11] - every row's `instruction` and `output` text is English, despite the repository's `language:ko` tag [1]. The grandparent `glaive-function-calling-v2` declares `language: en` only, with no Korean [4], consistent with this observation.
- Row 0 of this dataset (`system` naming a `get_exchange_rate` function, `instruction` "Can you book a flight for me from New York to London?", `output` declining and pointing to the exchange-rate function) matches row 0 of the parent `mzbac/function-calling-llama-3-format-v1.1` [12] verbatim once the Llama-3 chat-template tokens (`<|begin_of_text|>`, `<|start_header_id|>...`) are stripped [9], and both datasets report the same 112,390-row count [2][6] - consistent with a row-for-row template conversion rather than resampling.
- No source states a measured contamination rate, duplicate rate, or annotation-agreement figure for this dataset.

## Load it

Pin the `main`-branch revision (the Hub API's `sha` at the check date; the repo was last modified 2024-07-10) [3]:

```python
import datasets

REV = "36769101d375c9a2552dd7da617c54b979a04ed9"  # main at the check date
train = datasets.load_dataset("Saxo/alpaca_function_calling_dataset", revision=REV, split="train")
```

**Trap, and what the pin does and does not cover**: the repository ships a single `processed_dataset.csv` file loaded through the generic `csv` builder [3][7], not a `dataset_info` block with declared features in its own README - the three-column schema comes from the datasets-server `/info` endpoint, not from the card. That endpoint, and every other datasets-server call this card cites ([6], [7], [9], [10], [11]), takes no revision parameter and is live: `/info`'s own `download_checksums` key shows its 112,390-row, 134,801,020-byte figures were computed against `hf://datasets/Saxo/alpaca_function_calling_dataset@71d69d94ffcebda15d8d86d227700d323806c8b6`, an auto-generated parquet-conversion ref, not the `main` sha `36769101d375c9a2552dd7da617c54b979a04ed9` pinned above [7]. `load_dataset(..., revision=REV)` therefore reproduces the CSV file as it exists at `main`, but the shape and row numbers on this card - and the sampled rows in A row and Quality - are only pinned to that separate `71d69d94...` ref, which this card does not control and cannot pass as `revision`.

## Neighbors

- `mzbac/function-calling-llama-3-format-v1.1` - the direct parent, same 112,390 rows in Llama-3 chat-template form (a single `text` column) rather than Alpaca's three columns; prefer it over this release if the target model's chat template is Llama-3's, since no reformatting round-trip is then needed [2].
- `glaiveai/glaive-function-calling-v2` - the grandparent source that `mzbac` cleaned; prefer it only if the `mzbac`/Saxo cleanup (removing invalid JSON and function-argument values) is unwanted [2][4].

## A row

One shape is served (config `default`, split `train`). Row 0, from the datasets-server first-rows endpoint [9], with the function schema in `system` shown in full since it is short:

```json
{
  "system": "You are a helpful assistant with access to the following functions. Use them if required -\n{\n    \"name\": \"get_exchange_rate\",\n    \"description\": \"Get the exchange rate between two currencies\",\n    \"parameters\": {\n        \"type\": \"object\",\n        \"properties\": {\n            \"base_currency\": {\n                \"type\": \"string\",\n                \"description\": \"The currency to convert from\"\n            },\n            \"target_currency\": {\n                \"type\": \"string\",\n                \"description\": \"The currency to convert to\"\n            }\n        },\n        \"required\": [\n            \"base_currency\",\n            \"target_currency\"\n        ]\n    }\n}",
  "instruction": "Can you book a flight for me from New York to London?",
  "output": "I'm sorry, but I don't have the capability to book flights. My current function allows me to get the exchange rate between two currencies. If you need help with that, feel free to ask!"
}
```

A second sampled row (offset 1) shows the alternate `output` shape, a function invocation rather than a text reply: `"output": "<functioncall> {\"name\": \"get_news_headlines\", \"arguments\": {\"country\": \"United States\"}}"` [9].

## Where it came from

Built by Dr. Ji Yun Sung (Saxo) at Linkbricks, a Korean AI and big-data analytics company, as a format conversion: taking `mzbac/function-calling-llama-3-format-v1.1` - a single-column, Llama-3-chat-template dataset - and splitting it into Alpaca's `system`/`instruction`/`output` columns [1]. That parent dataset's own card states it is "cleaned up" from `glaiveai/glaive-function-calling-v2`, with invalid JSON and function-calling argument values removed, and reports the same 112,390-row count as this release [2]. `glaive-function-calling-v2`'s card carries only licence, task-category, and language metadata, with no description of how its dialogues were produced [4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was fetched on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Saxo/alpaca_function_calling_dataset dataset card (README). https://huggingface.co/datasets/Saxo/alpaca_function_calling_dataset/raw/main/README.md - builder, description, language/license tags. Fetched 2026-08-12.

[2] mzbac/function-calling-llama-3-format-v1.1 dataset card (README). https://huggingface.co/datasets/mzbac/function-calling-llama-3-format-v1.1/raw/main/README.md - lineage from glaiveai/glaive-function-calling-v2, cleanup description, row/size figures. Fetched 2026-08-12.

[3] Hugging Face Hub API record for Saxo/alpaca_function_calling_dataset. https://huggingface.co/api/datasets/Saxo/alpaca_function_calling_dataset?full=true - `sha`, `gated`, `private`, `downloads`, `likes`, `lastModified`, siblings (single `processed_dataset.csv` file). Fetched 2026-08-12.

[4] Hugging Face Hub API record and dataset card for glaiveai/glaive-function-calling-v2. https://huggingface.co/api/datasets/glaiveai/glaive-function-calling-v2?full=true and https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/raw/main/README.md - licence, `language: en` tag, absence of any generation-method description. Fetched 2026-08-12.

[5] Hugging Face Hub API record for mzbac/function-calling-llama-3-format-v1.1. https://huggingface.co/api/datasets/mzbac/function-calling-llama-3-format-v1.1?full=true - `cardData` and tags carry no licence field. Fetched 2026-08-12.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Saxo%2Falpaca_function_calling_dataset Fetched 2026-08-12.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Saxo%2Falpaca_function_calling_dataset - column names/dtypes, `csv` builder. Fetched 2026-08-12.

[8] The corpus screening row for `Saxo/alpaca_function_calling_dataset`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-12.

[9] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Saxo%2Falpaca_function_calling_dataset&config=default&split=train - rows 0-99. Fetched 2026-08-12.

[10] datasets-server rows endpoint, offset 56000. https://datasets-server.huggingface.co/rows?dataset=Saxo%2Falpaca_function_calling_dataset&config=default&split=train&offset=56000&length=10 Fetched 2026-08-12.

[11] datasets-server rows endpoint, offset 112380. https://datasets-server.huggingface.co/rows?dataset=Saxo%2Falpaca_function_calling_dataset&config=default&split=train&offset=112380&length=10 Fetched 2026-08-12.

[12] datasets-server first-rows endpoint for mzbac/function-calling-llama-3-format-v1.1. https://datasets-server.huggingface.co/first-rows?dataset=mzbac%2Ffunction-calling-llama-3-format-v1.1&config=default&split=train - row 0, for the row-match comparison. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as English-language, single-turn function-calling SFT data, with no holdout required beyond the dataset's own single `train` split. The card establishes this from the confirmed row/column shape and the absence of any stated evaluation overlap in this dataset's card, its parent's, or its grandparent's [1][2][4]; the screening row's note agrees [8].

### The screening row

The row's own note [8]: "mzbac/function-calling-llama-3-format-v1.1 converted to Alpaca format; Korean and English." The row carries no flag.
