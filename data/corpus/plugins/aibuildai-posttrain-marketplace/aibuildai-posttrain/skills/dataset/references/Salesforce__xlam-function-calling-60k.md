# Salesforce/xlam-function-calling-60k

Roughly 60,000 execution-verified function-calling examples - a natural-language query, its available tool schemas, and the correct function call(s) - synthesized by Salesforce's APIGen pipeline and gated behind a click-through licence agreement.

**Salesforce/xlam-function-calling-60k** is the dataset release accompanying "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets" [1]: an automated pipeline samples real, executable REST APIs and Python functions, has an LLM generate a query plus a matching function call, and keeps only examples that pass a three-stage format, execution, and semantic check [1]. **The repository is gated with auto-approval: a visitor must submit name, country and affiliation and click through a licence-and-citation acknowledgement before any file - including the README body and the data JSON - becomes readable, so nothing below the quick-facts block could be read from the gated card itself** [2][3]. It lives at https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k .

**Use it for**: single-turn function-calling SFT - each row pairs one user query with the correct tool call(s) over a set of candidate tool schemas, which is the reasoning-trace/tool-call SFT shape (query + tools in the prompt, the verified call(s) as the target), not a preference pair. See the SFT method card. The one usage catch is procedural, not a data restriction: the gate must be accepted (name, country, affiliation, and an acknowledgement to follow the licence and cite APIGen) before the file loads at all [2][3].

**Licence**: CC-BY-4.0 (`cardData.license` is `"cc-by-4.0"`, tag `license:cc-by-4.0`) [3], gated with auto-approval (`"gated": "auto"`) [3]. The one catch: accepting the gate requires agreeing to "follow corresponding license and cite APIGen" [3], which is a citation ask layered on top of the CC-BY-4.0 grant, not a separate licence term.

**Shape**: one config (`dataset`), one `train` split, one file `xlam_function_calling_60k.json`, roughly 60,000 rows [3][4]; no config or split's row count is available from the live datasets-server endpoints because the repository is gated (see Load it).

**Hold out**: nothing found. No source read for this card - the Hub card's own description, the origin paper, or the corpus screening note - states that any row overlaps a named evaluation set; the paper cites the Berkeley Function-Calling Leaderboard (BFCL) only as where the resulting xLAM models were scored, not as a source the training data was drawn from or checked against [1][5].

**Origin**: built by Salesforce; the queries and function calls are LLM-generated (DeepSeek-V2-Chat 236B and Mixtral-8x22B-Instruct), not human-written, though a human-evaluation quality check was run on a sample [1]. Hub API at the check date: `downloads` 16,435, `downloadsAllTime` 175,542, `likes` 671 [3].

**Trained-on-by**: Salesforce's own xLAM function-calling model family, including `Salesforce/xLAM-1b-fc-r` and `Salesforce/xLAM-7b-fc-r`, both of which the origin paper introduces as trained on this data [1][6]. Third-party adoption confirmed by Hub model-card metadata: `MadeAgents/Hammer-7b` (and the smaller Hammer-4b/Hammer-1.5b siblings) declare `Salesforce/xlam-function-calling-60k` in their `datasets:` field alongside a supplementary irrelevance-detection set [7]. A live Hub model search for models tagged with this dataset returned 50+ repositories, including further xLAM quantizations and merges (`bartowski/xLAM-7b-fc-r-GGUF`, `legraphista/xLAM-8x7b-r-IMat-GGUF`) and other third-party fine-tunes (`KishoreK/ActionGemma-9B`) [7].

**Introduced by**: [1] (Liu, Hoang, Zhang, Zhu, Lan, Kokane, Tan, Yao, Liu, Feng, et al.).

## Shape

The Hub API's `cardData.configs` declares one config, `dataset`, mapping the `train` split to the single file `xlam_function_calling_60k.json` [3]; the repository tree lists no other data file [3]. Every row-count-bearing endpoint that would normally confirm this is gated:

| endpoint | result |
| --- | --- |
| `datasets-server` `/size` | 404, "does not exist, or is not accessible without authentication" |
| `datasets-server` `/info` | 404, same message |
| `datasets-server` `/first-rows` | 404, same message |
| `.../resolve/main/xlam_function_calling_60k.json` | "Access to dataset ... is restricted. You must have access to it and be authenticated" |
| `.../raw/main/README.md` | same restricted-access message |

(all four checked live at the check date) [4]. The only row-count figures available without authentication are textual: the Hub API's own truncated card description states "This repo contains 60,000 data collected by APIGen" [3], and the origin paper's Table 1 gives per-generator verified-example counts that were merged to build the release - 33,659 from DeepSeek-V2-Chat (236B) and 26,384 from Mixtral-8x22B-Instruct, summing to 60,043 - with the paper stating the release uses "the two strongest models" of the four it trialed [1]. No source read for this card states column names, byte sizes, or sequence/token statistics for this specific repository; "not stated" for all three. The most likely column set is a query/tools/answers/id shape by analogy with a downstream merge repository built from this dataset's rows (see Neighbors), but that inference is not this repository's own served metadata.

## Quality

- Every kept example passed a three-stage verification: a format checker, an execution checker that runs the generated call against the real API, and an LLM-based semantic checker that confirms the call and its result match the query's intent [1].
- The paper's Table 1 reports per-generator pass rates through that pipeline: DeepSeek-V2-Chat (236B) 84.15%, Mixtral-8x22B-Instruct 65.96%, Mixtral-8x7B-Instruct 38.46%, DeepSeek-Coder-33B-Instruct 34.42% - the release keeps only the two highest-pass-rate generators [1].
- A human evaluation checked 600 sampled examples from the released data using three evaluators judging parameter-value accuracy and call-count appropriateness; 28 of 600 (about 4.7%) had minor issues, so about 95.3% were judged high quality with no issues found [1].
- The paper's own ablation (Fig. 5, a chart with no accompanying text values) reports that adding back the examples the pipeline had filtered out at the execution- and semantic-checking stages degrades downstream BFCL performance, more severely for the smaller model - no numeric values from that figure are stated in the paper's text, so none are given here [1].
- No source read for this card states a decontamination check, a duplicate-rate figure, or an inter-annotator-agreement statistic for this release; none is invented here.

## Load it

The file cannot be loaded without first accepting the gate on the dataset page (name, country, affiliation, and the licence/citation acknowledgement) while authenticated with a Hugging Face token that has accepted access [2][3]:

```python
import datasets

REV = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"  # main at the check date
ds = datasets.load_dataset(
    "Salesforce/xlam-function-calling-60k",
    revision=REV,
    split="train",
)  # requires an authenticated, gate-accepted token; otherwise raises a 401/403
```

**Trap**: every anonymous or non-accepted-token request - the Hub `/api/datasets` viewer info, the datasets-server `/size`, `/info`, and `/first-rows` endpoints, and the raw file `resolve` URL - returns a gated/404 "does not exist, or is not accessible" error rather than any partial data, so a pipeline that treats a 404 here as "dataset absent" will silently skip a real, popular dataset instead of prompting for gate acceptance [4]. This card's own row counts and column claims are therefore drawn only from the Hub API's public metadata and card-description snippet, the origin paper, and downstream repositories built from this dataset's rows - not from the gated file itself.

## Neighbors

- `Salesforce/APIGen-MT-5k` - the multi-turn successor from the same builder and pipeline family, introduced by a separate paper (APIGen-MT) [8]. It holds 5,000 rows in one `train` split with 3 columns, generated by GPT-4o and DeepSeek-V3 over retail and airline tau-bench domains, in a ShareGPT-like multi-turn JSON format, licensed CC-BY-NC-4.0 (non-commercial, unlike this release's CC-BY-4.0) [8]; its README carries the same `extra_gated_*` fields as this release [8], but unlike this release its Hub API record reports `"gated": false`, and its `datasets-server` `/size` endpoint answers live without authentication (5,000 rows, 3 columns), so it is not actually gated - the README's gate-styled fields are present but inactive [9]. Its own card states it is "a subset of the data used to train the xLAM-2 model series" [8]. Choose it for multi-turn agent trajectories; choose this release for single-turn function calling.
- `argilla/apigen-function-calling` - an ungated, 109,402-row merge of `argilla/Synth-APIGen-v0.1` and this dataset, built by Argilla [10]. Its README states the merge is "not ready to do fine tuning" as served, and shows a worked example row carrying `"origin": "xLAM"` with fields `answers`, `query`, `id`, `tools` populated and `func_name`, `func_desc`, `hash_id`, `model_name` left null - those four extra fields are populated only for the `"distilabel"`-origin half of the merge, its own newly generated rows [10]. This is the only source read for this card that shows this dataset's actual column names, and it is a downstream repackaging, not this repository's own served schema.

## A row

No row could be fetched from this repository itself: the resolve URL for `xlam_function_calling_60k.json`, and the datasets-server `/size`, `/info`, and `/first-rows` endpoints, all returned a gated/404 access-restricted response when checked live (see Shape) [4]. The closest verifiable example is the worked row from the `argilla/apigen-function-calling` merge (see Neighbors) labeled `"origin": "xLAM"`, i.e. carried over unmodified from this dataset [10]:

```json
{
  "answers": "[{\"name\": \"split_list\", \"arguments\": {\"lst\": [10, 20, 30, 40, 50, 60], \"chunk_size\": 4}}, {\"name\": \"binary_search\", \"arguments\": {\"arr\": [10, 20, 30, 40, 50, 60], \"target\": 30}}]",
  "query": "Please split the list [10, 20, 30, 40, 50, 60] into chunks of size 4, and also find the index of 30 in this list.",
  "id": 1234,
  "tools": "[{\"name\": \"split_list\", \"description\": \"Splits a list into chunks of a specified size.\", \"parameters\": {\"lst\": {\"description\": \"The input list.\", \"type\": \"List\"}, \"chunk_size\": {\"description\": \"The size of each chunk.\", \"type\": \"int\"}}}, {\"name\": \"binary_search\", \"description\": \"Performs binary search on a sorted list to find the index of a target value.\", \"parameters\": {\"arr\": {\"description\": \"The sorted list of integers.\", \"type\": \"List[int]\"}, \"target\": {\"description\": \"The target value to search for.\", \"type\": \"int\"}}}]",
  "func_name": null,
  "func_desc": null,
  "hash_id": null,
  "model_name": null,
  "origin": "xLAM"
}
```

Separately, the origin paper's own appendix gives an illustrative example of the same query/tools/answers shape (for a parallel-call case), stated in the paper as the pipeline's target JSON format rather than as a sampled row from the released file [1].

## Where it came from

Built by Salesforce. The pipeline sourced 3,539 executable REST APIs by filtering and re-documenting 16,464 APIs pulled from the RapidAPI-backed ToolBench dataset [1], removing APIs that lacked parameters or failed live execution tests, and added 134 hand-written Python functions inspired by BFCL's executable-evaluation categories, for 3,673 APIs across 21 consolidated categories in total [1]. Four base LLMs generated candidate query-plus-call examples against this API pool - DeepSeek-V2-Chat (236B), DeepSeek-Coder-33B-Instruct, Mixtral-8x22B-Instruct, and Mixtral-8x7B-Instruct - each targeted at 40,000 generations at temperature 0.7, then filtered through the format/execution/semantic verification pipeline [1]. The released 60k file keeps only the DeepSeek-V2-Chat and Mixtral-8x22B-Instruct output, the two generators with the highest pass rates [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus reference above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the datasets-server endpoints for this repository take no revision parameter and were unreachable without authentication regardless, so no live-endpoint claim in this card is pinned beyond "gated as of the check date."

[1] Liu, Hoang, Zhang, Zhu, Lan, Kokane, Tan, Yao, Liu, Feng, et al., "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets", 2024. https://arxiv.org/abs/2406.18518 - the origin paper; current title read from the live abs page; body text read from the official PDF (https://arxiv.org/pdf/2406.18518) via text extraction, since the ar5iv/HTML renderings for this paper failed to convert. Fetched 2026-08-11.

[2] Hugging Face Hub dataset page for Salesforce/xlam-function-calling-60k. https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k - rendered HTML confirms the gate wall (`"gated"` state, `extra_gated_heading`, `extra_gated_fields`) and that the full card body is not served to an unauthenticated request. Fetched 2026-08-11.

[3] Hugging Face Hub API record for Salesforce/xlam-function-calling-60k. https://huggingface.co/api/datasets/Salesforce/xlam-function-calling-60k?full=true - `cardData` (license, gated fields, configs), `sha`, `downloads`, `likes`, `lastModified`, `siblings`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] Live access checks against the gated repository: `datasets-server` `/size`, `/info`, and `/first-rows` endpoints (`https://datasets-server.huggingface.co/{size,info,first-rows}?dataset=Salesforce%2Fxlam-function-calling-60k`) and the raw file resolve URL (`https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/resolve/main/xlam_function_calling_60k.json`) and README raw URL (`https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/raw/main/README.md`) - all returned gated/404 access-restricted responses. Fetched 2026-08-11.

[5] The corpus screening row for `Salesforce/xlam-function-calling-60k`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[6] Salesforce/xLAM-1b-fc-r and Salesforce/xLAM-7b-fc-r model listings, Hugging Face Hub. https://huggingface.co/Salesforce/xLAM-1b-fc-r and https://huggingface.co/Salesforce/xLAM-7b-fc-r - confirmed present in the live model search below as models trained on this dataset. Fetched 2026-08-11.

[7] Hugging Face Hub model-search API filtered on this dataset. https://huggingface.co/api/models?filter=dataset:Salesforce/xlam-function-calling-60k&limit=50 - returned 50 models; cross-checked one third-party entry's own card metadata at https://huggingface.co/api/models/MadeAgents/Hammer-7b?full=true, whose `cardData.datasets` lists `Salesforce/xlam-function-calling-60k`. A live, unpinned Hub search, not a revision-pinned record. Fetched 2026-08-11.

[8] Salesforce/APIGen-MT-5k dataset card (README). https://huggingface.co/datasets/Salesforce/APIGen-MT-5k/raw/main/README.md - licence, gating, generator models, domains, format, and the "subset of the data used to train the xLAM-2 model series" statement. Fetched 2026-08-11.

[9] datasets-server size endpoint for the neighbor, https://datasets-server.huggingface.co/size?dataset=Salesforce%2FAPIGen-MT-5k (row and column counts, answered live without authentication), and the neighbor's Hub API record, https://huggingface.co/api/datasets/Salesforce/APIGen-MT-5k?full=true (`"gated": false`). Fetched 2026-08-11.

[10] argilla/apigen-function-calling dataset card (README) and size endpoint. https://huggingface.co/datasets/argilla/apigen-function-calling/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=argilla%2Fapigen-function-calling - merge description, "not ready to do fine tuning" statement, worked example row, row/column counts. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as single-turn function-calling SFT data, gated behind a click-through licence agreement. The card could confirm the pipeline, generator models, verification process, and licence/gating terms entirely from public Hub API metadata and the origin paper, without ever reading the gated card body or the gated data file itself, consistent with the screening note's own summary [1][2][3][5].

### The screening row

The row's own note [5]: "60k execution-verified function-calling records generated by DeepSeek-V2-Chat (ids 0-33658) and Mixtral-8x22B-Inst via Salesforce's APIGen pipeline over 3,673 real executable APIs; single train split, gated with auto-accept, cc-by-4.0; the card cites BFCL only as where the resulting xLAM models were scored, not as a data source." The row carries no flag.
