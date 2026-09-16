# BitAgent/tool_calling

551,285 single-turn, single-tool-call conversations, each paired with the JSON schema of the one tool it calls - a mid-size tool-calling SFT corpus with a near-empty dataset card.

**BitAgent/tool_calling** is a Hugging Face dataset published by the org account BitAgent at https://huggingface.co/datasets/BitAgent/tool_calling ; its dataset card carries only the `dataset_info` YAML front matter (features, split sizes, config) and no prose, so no source states who or what generated the conversations [1]. Every sampled row (see Shape and A row) follows the same three-turn shape: a `user` request, one `tool call` turn naming a function and its arguments, and an `assistant` reply - the same shape as the tool-calling SFT rows a chat-format method card expects. **No source establishes the generating model or process; treat the content as unverified synthetic data and see the appendix flag before training on it.**

**Use it for**: single-turn, single-tool tool-calling SFT - one function call per conversation, matching the SFT method card's chat-format training shape, subject to the unverified-provenance caveat above. `conversation` is a JSON string of `{"role": ..., "content": ...}` turns (`user`, `tool call`, `assistant`) and `tools` is a JSON string of the schema(s) available at that turn; both need `json.loads` before use, and the `"tool call"` role label (two words, no underscore) is this dataset's own non-standard token, not a chat-template standard [1][2].

**Licence**: not stated, ungated (`"gated": false`, `"private": false` in the Hub API record) [6]. The card's YAML front matter carries no `license` key, the repo's tag list carries no `license:` tag, and the card body has no licence text [1].

**Shape**: 551,285 rows, one config (`default`), one split (`train`) [1][3][4].

**Hold out**: not stated by any source read for this card - no eval split, no stated overlap with a benchmark, and no contamination note in the card itself. The screening row flags a separate, unresolved provenance question (see appendix); it is not a holdout instruction [5].

**Origin**: published by the BitAgent Hugging Face org; generating process not stated by the dataset card [1]. Hub API at the check date: 583 `downloads` (8,807 `downloadsAllTime`), 10 `likes` [6][7].

**Trained-on-by**: `chatitcloud/UZI1`, a Gemma-3-270M finetune, lists `dataset:BitAgent/tool_calling` in its Hub tags; `mradermacher/UZI1-GGUF` is a GGUF quantization of that same model [8]. No other adopting model or recipe was found.

**Introduced by**: no paper - the dataset card [1].

## Shape

One config, one split, two string columns, per the datasets-server info and size endpoints [3][4]:

| split | rows |
| --- | --- |
| `train` | 551,285 |

| column | dtype |
| --- | --- |
| `conversation` | string (JSON-encoded turn list) |
| `tools` | string (JSON-encoded tool-schema list) |

Sizes, from the datasets-server size endpoint: 79,932,244 bytes of original Parquet download, 510,389,576 bytes decoded in memory [4]. No source states sequence-length or token statistics for this dataset.

## Quality

- The card body is empty prose - only `dataset_info` YAML - so it states no annotation process, no measured quality rate, and no known complaint [1].
- Reading a run of rows at three points in the served split - offsets 0-99 (100 rows), 275,000-275,049 (50 rows), and 551,185-551,284 (100 rows), 250 rows total - every row had the same three-turn shape (`user`, `tool call`, `assistant`) and exactly one tool listed in `tools` [2][9][10]. This is a shape observation over the rows read, not a corpus-wide guarantee; no source states whether every one of the 551,285 rows follows it.
- No source states a duplicate-row rate, a contamination rate, or an annotator-agreement figure for this dataset; none is invented here.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha`, matching the shortlist's pinned commit; the repo was last modified 2024-08-06) [6]:

```python
import json
import datasets

REV = "8969e358964d9a2c516c235306bcd15a0e8b7abc"  # main at the check date
ds = datasets.load_dataset("BitAgent/tool_calling", revision=REV, split="train")  # 551,285 rows, one split, no held-out eval set
row = ds[0]
conversation = json.loads(row["conversation"])  # list of {"role", "content"} turns
tools = json.loads(row["tools"])                 # list of tool schema dicts
```

**Trap**: `conversation` and `tools` are stored as JSON strings, not as `datasets` `Sequence`/`Value(dict)` columns - loading the split does not give you parsed turn lists or schemas; every row needs `json.loads` on both fields before use [3].

## Neighbors

- `BitAgent/tool_calling_shuffle` - same org, same declared shape (551,285 rows, one `train` split, identical byte sizes in its own `dataset_info`) [11]. Rows fetched from both at offset 0 (3 rows) and offset 275,000 (5 rows) matched byte-for-byte in the same row order despite the "shuffle" name; treat it as a copy of this release at the rows checked, not an independently reordered set, and do not concatenate the two into one training run without deduplicating [2][9].
- `BitAgent/tool_shuffle_small` - 100 rows, described on its own card as tasks representative of BFCL evaluation criteria, excluding multi-turn tasks [12][13]. Much smaller and BFCL-shaped, not a drop-in replacement for this release.
- `BitAgent/tool_shuffle_small_test` - 333 rows, same org, size class implies a held-out counterpart to `tool_shuffle_small`; no card prose states the relationship [13][14].
- `BitAgent/bfcl_shuffle_small` and `BitAgent/bfcl_shuffle_full` - the org's own republication of the Berkeley Function Calling Leaderboard; `bfcl_shuffle_full`'s Hub summary describes it as built from BFCL for representative function-calling use cases [15]. This is the BFCL republish the screening flag references as a reason this dataset's seed lineage stays unproven; do not evaluate a model on BFCL after training on this dataset without checking for overlap first.

## A row

The repository serves one config and one shape, so one row covers it. From `config="default"`, `split="train"`, `row_idx=1` (datasets-server `/first-rows`) [2]:

```json
{
  "conversation": "[{\"role\": \"user\", \"content\": \"Delete a service called 'old-service' in the 'dev' namespace.\"}, {\"role\": \"tool call\", \"content\": {\"name\": \"delete_old_service\", \"arguments\": {\"namespace\": \"dev\", \"service_name\": \"old-service\"}}}, {\"role\": \"assistant\", \"content\": \"---\\n\\n'old-service' has been deleted in the 'dev' namespace.\"}]",
  "tools": "[{\"name\": \"delete_old_service\", \"description\": \"Deletes a service in a given Kubernetes namespace. Useful for removing old or unused services.\", \"arguments\": {\"namespace\": {\"required\": true, \"type\": \"str\", \"description\": \"The Kubernetes namespace where the service is deployed.\"}, \"service_name\": {\"required\": true, \"type\": \"str\", \"description\": \"Name of the service to be deleted.\"}}}]"
}
```

Both fields are JSON-encoded strings; `conversation` decodes to a three-turn list (`user`, `tool call`, `assistant`) and `tools` decodes to a one-element list holding the schema the `tool call` turn used.

## Where it came from

The dataset is published under the BitAgent Hugging Face org. No source read for this card - the dataset's own card, the Hub API record, or the datasets-server endpoints - states a builder, a generating model, a human-annotation process, or an upstream data source for the 551,285 conversations [1][6]. The org's sibling repositories show it also republishes the Berkeley Function Calling Leaderboard (`bfcl_shuffle_full`, `bfcl_shuffle_small`) and hosts smaller BFCL-representative sets (`tool_shuffle_small`, `tool_shuffle_small_test`), but none of their cards states a lineage connection to this dataset's own 551,285 rows [12][15].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] BitAgent/tool_calling dataset card (README). https://huggingface.co/datasets/BitAgent/tool_calling/raw/main/README.md - YAML front matter only, no prose, no license field. Fetched 2026-08-11.

[2] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=BitAgent%2Ftool_calling&config=default&split=train - rows 0-99. Fetched 2026-08-11.

[3] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=BitAgent%2Ftool_calling Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=BitAgent%2Ftool_calling Fetched 2026-08-11.

[5] The corpus screening row for `BitAgent/tool_calling`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[6] Hugging Face Hub API record for BitAgent/tool_calling. https://huggingface.co/api/datasets/BitAgent/tool_calling?full=true - `sha`, `downloads`, `likes`, `cardData`, tags, last-modified date. Fetched 2026-08-11.

[7] Hugging Face Hub API `downloadsAllTime` expansion. https://huggingface.co/api/datasets/BitAgent/tool_calling?expand[]=downloadsAllTime Fetched 2026-08-11.

[8] Hugging Face model-search API, filtered on `dataset:BitAgent/tool_calling`. https://huggingface.co/api/models?filter=dataset:BitAgent/tool_calling&limit=100 - returns `chatitcloud/UZI1` and its `mradermacher/UZI1-GGUF` quantization. Fetched 2026-08-11.

[9] datasets-server rows endpoint, offset 275,000, length 50. https://datasets-server.huggingface.co/rows?dataset=BitAgent%2Ftool_calling&config=default&split=train&offset=275000&length=50 Fetched 2026-08-11.

[10] datasets-server rows endpoint, offset 551,185, length 100. https://datasets-server.huggingface.co/rows?dataset=BitAgent%2Ftool_calling&config=default&split=train&offset=551185&length=100 Fetched 2026-08-11.

[11] BitAgent/tool_calling_shuffle dataset card (README) and datasets-server info/size endpoints. https://huggingface.co/datasets/BitAgent/tool_calling_shuffle/raw/main/README.md ; https://datasets-server.huggingface.co/info?dataset=BitAgent%2Ftool_calling_shuffle ; https://datasets-server.huggingface.co/size?dataset=BitAgent%2Ftool_calling_shuffle - same declared row count and byte sizes as BitAgent/tool_calling. Fetched 2026-08-11.

[12] BitAgent/tool_shuffle_small dataset card (README). https://huggingface.co/datasets/BitAgent/tool_shuffle_small/raw/main/README.md - states the dataset represents BFCL evaluation criteria, excluding multi-turn tasks. Fetched 2026-08-11.

[13] datasets-server size endpoint, BitAgent/tool_shuffle_small and BitAgent/tool_shuffle_small_test. https://datasets-server.huggingface.co/size?dataset=BitAgent%2Ftool_shuffle_small ; https://datasets-server.huggingface.co/size?dataset=BitAgent%2Ftool_shuffle_small_test Fetched 2026-08-11.

[14] Hugging Face Hub API listing for the BitAgent org's datasets. https://huggingface.co/api/datasets?author=BitAgent&limit=100 - repo names, tags, and creation dates for `tool_shuffle_small_test`, `bfcl_shuffle_small`, `bfcl_shuffle_full`. Fetched 2026-08-11.

[15] BitAgent/bfcl_shuffle_full Hub listing summary (from the same org-datasets API call as [14]), describing the repository as built from the Berkeley Function Calling Leaderboard. https://huggingface.co/api/datasets?author=BitAgent&limit=100 Fetched 2026-08-11.

[16] GitHub API repository lookup. https://api.github.com/repos/RogueTensor/bitagent_subnet - returned HTTP 404 (repository not found) at the check date. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Unverified provenance: the dataset's own card states nothing about who or what generated its 551,285 conversations, and no external source read for this card fills that gap [1]. The screening flag adds that the generating repository named in its own note, the Bittensor subnet `RogueTensor/bitagent_subnet`, no longer exists - a GitHub API check for `RogueTensor/bitagent_subnet` returned 404 at the check date [16] - and that the same org republishes BFCL, so lineage against BFCL is not settled by the probes the screening row describes.

### The screening row

The row's own note [5]: "551k tool-call conversations with their tool schemas; the repo holds one parquet file and an empty card, so nothing establishes the generator (content reads synthetic)." Its flag [5]: "unverified: - Not establishable: LLM-synthesized tool calls whose generating Bittensor subnet repo (RogueTensor/bitagent_subnet) is deleted; the org also republishes BFCL, so seed lineage stays unproven despite 0/500 question and 0/412 function-name overlap probes."
