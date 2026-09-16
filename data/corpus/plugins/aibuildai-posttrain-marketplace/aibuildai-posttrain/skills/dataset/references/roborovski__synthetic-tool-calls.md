# roborovski/synthetic-tool-calls

6,008 single-turn tool-use records - a tool schema, a natural-language question, a tool call, the tool's returned result, and an assistant answer built from that result - synthetic data with no README prose and no stated generator.

**roborovski/synthetic-tool-calls** is a Hugging Face dataset repository under the user `roborovski`, at https://huggingface.co/datasets/roborovski/synthetic-tool-calls, holding one Parquet file and a YAML dataset-info block with no descriptive text: the `README.md` on the Hub carries only the `dataset_info`/`configs` front matter and no body [1]. No paper, blog post, or card prose names how the records were generated, what model (if any) produced them, or what the tool set was sampled from; the row's own screening note already states the generator is not named [2], and this card invents nothing beyond that. **The generating model is not stated by any source, so treat this as unattributed synthetic data and verify a sample by hand before trusting it as a training signal.**

**Use it for**: single-turn tool-call SFT - each row's `question`, `tool`, `tool_call`, `call_result` and `agent_output` can be assembled into one training example that teaches a model to read a tool schema, emit a call, and phrase an answer from the returned result. There is no chat-template wrapper and no multi-turn structure in the served columns [3][4]; a caller must assemble the turns itself before feeding the SFT method card's chat-format loader. Not preference data - see the sibling `roborovski/synthetic-toolformer-dpo-pairs` below for that shape.

**Licence**: not stated. The Hub API's `cardData` carries no `license` key and the repository has no licence tag [5]; the YAML front matter carries only `dataset_info` and `configs` [1]. Ungated (`"gated": false`, `"private": false`) [5].

**Shape**: 6,008 rows in one config (`default`), one split (`train`), five string columns [3][4].

**Hold out**: nothing found - no source ties this dataset's rows to any named evaluation set, and the dataset itself carries no eval split to withhold.

**Origin**: repository owned by the Hugging Face user `roborovski`; content is synthetic (the row's tool schemas, calls, results and answers show no human-authorship markers, and the shortlist row's own `origin` field records it as model-produced [2]), but no source names the generating model. Hub API at the check date: `downloads` 110, `downloadsAllTime` 1,837, `likes` 2 [5].

**Trained-on-by**: none found - no source cites a named model or training recipe as having trained on this specific repository.

**Introduced by**: no paper - the dataset card carries no descriptive text at all [1], so there is no card-stated introduction either; the repository's existence and its YAML metadata are the only self-description available [1][5].

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 6,008 |
| total | 6,008 |

One config, `default`, with five columns, all strings (datasets-server `/info`) [3]:

| column | dtype |
| --- | --- |
| `tool` | string |
| `question` | string |
| `call_result` | string |
| `tool_call` | string |
| `agent_output` | string |

Sizes (datasets-server `/size`) [4]: 1,787,706 bytes of Parquet download, 5,272,422 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset.
- The `tool`, `tool_call` and `call_result` columns hold Python-repr-style strings (e.g. `"{'name': 'find_timezone', ...}"`, `"'Asia/Tokyo'"`), not JSON - a loader must `ast.literal_eval` or otherwise parse them rather than `json.loads` them directly, based on the 100 first-page rows and a further 18-row sample read at offset 5,990 [6][7].
- Of the 100 rows read from the first page of `train` (offsets 0-99), 89 distinct tool names appear, so the tool vocabulary is broad relative to the sample rather than a handful of tools repeated at scale [6].
- The trailing 18 rows read at offsets 5,990-6,007 show the same five-field shape and repr-string encoding as the first page, so the format is consistent from the start to the end of the split across the 118 rows sampled [6][7].
- No source states whether tool call arguments were validated against the tool schema's declared parameter types before the row was written; none is invented here.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-03-05) [5]:

```python
import datasets

REV = "f88b113a34eab424677256c4440a0e11bddeb4d5"  # main at the check date
train = datasets.load_dataset("roborovski/synthetic-tool-calls", revision=REV, split="train")  # 6,008 rows
```

**Trap**: the `tool`, `tool_call` and `call_result` fields are stored as Python dict/string reprs (single quotes, no field quoting), not valid JSON - calling `json.loads` on them raises a decode error; use `ast.literal_eval` instead [6][7].

## Neighbors

All neighbors below are owned by the same `roborovski` account; row counts were read live at the check date [8].

- `roborovski/synthetic-tool-calls-v2` - the same five-column schema (`tool`, `question`, `call_result`, `tool_call`, `agent_output`) at 414 rows, far smaller than this 6,008-row release; its card also carries no descriptive body [8][9].
- `roborovski/synthetic-toolformer-sharegpt` - 7,793 rows reshaped into a `conversations` list of `{role, content}` turns, a chat-formatted SFT rendering rather than this dataset's flat columns [8][9].
- `roborovski/synthetic-toolformer-dpo-pairs` - 289 rows with `question` plus `_accepted`/`_rejected` variants of `call_result`, `tool_call` and `agent_output`, a preference-pair shape built from the same field names as this dataset [8][9].
- `roborovski/synthetic-toolformer-dpo` - 2,709 rows, same accepted/rejected column shape as `synthetic-toolformer-dpo-pairs` above but a larger row count [8][9].
- `roborovski/synthetic-tool-calls-v2-dpo-pairs` - 8,005 rows combining this dataset's five columns with `_accepted`/`_rejected` pairs in one row, a preference-pair release built on top of the `-v2` tool-call data [8][9].
- No source states which of these releases the account prefers when they overlap; pick the plain SFT shape (this dataset or `-v2`) for tool-call SFT and one of the DPO-named releases for preference training.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [6]:

```json
{
  "tool": "{'name': 'find_timezone', 'description': 'Find the timezone of a given location. Returns the timezone of the location.', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The location for which to find the timezone'}}}, 'required': ['location']}",
  "question": "Find the timezone of a given location",
  "call_result": "'Asia/Tokyo'",
  "tool_call": "{'location': 'Tokyo, Japan'}",
  "agent_output": "The timezone of Tokyo, Japan is Asia/Tokyo."
}
```

## Where it came from

The repository holds only three files - `.gitattributes`, `README.md`, and one Parquet shard under `data/` [5] - and the README's YAML front matter states only the schema, split, and byte sizes, with no description, citation, homepage, or license field filled in (`datasets-server`'s `/info` response echoes the same empty `"description": ""`, `"citation": ""`, `"homepage": ""`, `"license": ""` fields) [3]. No source names a builder team, a generating model, an upstream question or tool pool, or a collection method beyond the account name `roborovski` itself.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] roborovski/synthetic-tool-calls dataset card (README). https://huggingface.co/datasets/roborovski/synthetic-tool-calls/raw/main/README.md - full file is YAML front matter only, no body prose. Fetched 2026-08-11.

[2] The corpus screening row for `roborovski/synthetic-tool-calls`, supplied with this card's request - its `note` and `origin` fields, read back in the row's own words in the appendix. Checked 2026-08-11.

[3] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=roborovski%2Fsynthetic-tool-calls Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=roborovski%2Fsynthetic-tool-calls Fetched 2026-08-11.

[5] Hugging Face Hub API record for roborovski/synthetic-tool-calls. https://huggingface.co/api/datasets/roborovski/synthetic-tool-calls?full=true - license (absent), gate, sha, downloads, likes, last-modified date, siblings; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=roborovski%2Fsynthetic-tool-calls&config=default&split=train - 100 rows read from offset 0. Fetched 2026-08-11.

[7] datasets-server rows endpoint. https://datasets-server.huggingface.co/rows?dataset=roborovski%2Fsynthetic-tool-calls&config=default&split=train&offset=5990&length=18 - 18 rows read from the tail of `train`. Fetched 2026-08-11.

[8] datasets-server size endpoint, one call per neighbor, for every neighbor row count above: `roborovski/synthetic-tool-calls-v2`, `roborovski/synthetic-toolformer-sharegpt`, `roborovski/synthetic-toolformer-dpo-pairs`, `roborovski/synthetic-toolformer-dpo`, `roborovski/synthetic-tool-calls-v2-dpo-pairs`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[9] Neighbor dataset cards (READMEs), read for schema and body content: `roborovski/synthetic-tool-calls-v2`, `roborovski/synthetic-toolformer-sharegpt`, `roborovski/synthetic-toolformer-dpo-pairs`, `roborovski/synthetic-toolformer-dpo`, `roborovski/synthetic-tool-calls-v2-dpo-pairs`. https://huggingface.co/datasets/<id>/raw/main/README.md Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as unattributed synthetic tool-call SFT data, with the generator left unnamed by every available source. The card rests this on two facts already established above: the README carries no descriptive body at all [1], and the screening row's own note says the generator is not named [2].

### The screening row

The row's own note [2]: "6k tool / question / call / result / answer records; synthetic, generator not named." The row carries no flag.
