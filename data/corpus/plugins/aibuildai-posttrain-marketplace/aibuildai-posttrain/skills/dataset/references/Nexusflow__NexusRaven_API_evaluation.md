# Nexusflow/NexusRaven_API_evaluation

1,070 served rows across four Hub-visible configs of single-turn, zero-shot API-calling queries with reference function calls, over five real and simulated tool domains - the evaluation set behind the NexusRaven-13B function-calling model.

**Nexusflow/NexusRaven_API_evaluation** is Nexusflow's own evaluation benchmark, built to score NexusRaven-13B and comparison models (GPT-3.5, GPT-3.5 Instruct, GPT-4, ToolLLM, ToolAlpaca, CodeLlama 13B Instruct) on single-turn zero-shot function calling, over five domains named in the repository's GitHub instructions - `cve_cpe`, `emailrep`, `virustotal`, `toolalpaca`, and `toolllm` [1]. Nexusflow curated the `cve_cpe`, `emailrep`, and `virustotal` domains itself and adapted the `toolalpaca` and `toolllm` domains from the ToolAlpaca [2] and ToolLLM [3] papers' own evaluation data [4]. **This is an evaluation set, not training data: every split is named `train` in the repository's Parquet layout, but the repository card's own License section frames the whole release as curated evaluation data, and Nexusflow's GitHub instructions describe it only as the benchmark used to score models, never as a training corpus [1][4].** It lives at https://huggingface.co/datasets/Nexusflow/NexusRaven_API_evaluation .

**Use it for**: nothing in a training pipeline - hold the entire repository out of any training corpus. It is a single-turn zero-shot function-calling evaluation benchmark: score a model's ability to pick the right Python API call and arguments given a natural-language query, then compare its generated `python_args_dict`/function call against the reference in each row [1][4]. Because every split is literally named `train`, a pipeline that globs for a `train` split and folds it into SFT or DPO training data would silently train on the eval set; there is no held-out counterpart split to train on instead.

**Licence**: no `license` field in the Hub API's `cardData` [5]; the README's License section states the evaluation data is released under CC-BY-NC-4.0, non-commercial, because it blends Nexusflow's own commercially-usable data with GPT-generated ToolAlpaca/ToolLLM data that is not commercially licensable [4]. Repository is ungated (`"gated": false`) [5].

**Shape**: four served configs, each a single `train` split, no other split name: `outputs_in_toolllm_format` 348 rows, `raw_queries` 339 rows, `standardized_api_list` 65 rows, `standardized_queries` 318 rows - 1,070 rows total [6][7]. The card's own YAML also declares a fifth config, `raw_api_list` (2 rows), which is not in this total because no data file for it exists in the repository.

**Hold out**: everything - all four served configs (1,070 rows) and, if it is ever reconstructed, `raw_api_list` (2 rows). The screening row's flag names the reason: an evaluation set whose splits are all named `train` [8].

**Origin**: built and released by Nexusflow.ai; queries and reference calls are Nexusflow's own curation for three domains and an adaptation of ToolAlpaca/ToolLLM's evaluation data for the other two, not model-generated content produced for this card [4]. Hub API at the check date: `downloads` 403, `downloadsAllTime` 16,628, `likes` 17 [5].

**Trained-on-by**: none found - this is stated to be an evaluation set, so no adoption-as-training-data would be expected. As an evaluation benchmark it is used by Nexusflow's own NexusRaven-13B model card, which reports the same headline result as the GitHub repository: a 95% success rate on cybersecurity tools such as CVE/CPE Search and VirusTotal versus 64% for prompted GPT-4, and links this dataset as the evaluation set behind that number [9][1].

**Introduced by**: no paper - the dataset card [4] and the introducing GitHub repository README [1]; the domains it adapts come from ToolAlpaca [2] and ToolLLM [3].

## Shape

Rows served per config (datasets-server `/size`), and the card's own declared row count via `cardData.dataset_info` (Hub API) [6][5]:

| config | split | rows served | rows declared on card |
| --- | --- | --- | --- |
| `outputs_in_toolllm_format` | `train` | 348 | 348 |
| `raw_queries` | `train` | 339 | 339 |
| `standardized_api_list` | `train` | 65 | 65 |
| `standardized_queries` | `train` | 318 | 318 |
| `raw_api_list` | `train` | 0 (no file) | 2 |
| total | | 1,070 | 1,072 |

The repository's file tree at the pinned revision has exactly four Parquet data files, one per served config, plus `README.md` and `.gitattributes` - six files total, and no file under a `raw_api_list/` path [10]. The card's YAML front matter (`cardData.dataset_info`) nonetheless declares a fifth config, `raw_api_list`, with 2 examples and a schema close to `standardized_api_list`'s but with `args_dicts.default` typed as null-only rather than string [5]. That config does not load: `datasets-server`'s `/info` endpoint lists only the four served configs [7], and there is no data file for it in the tree [10]. This is the 1,072-declared-vs-1,070-served gap in the shortlist row.

Columns per config (datasets-server `/info`) [7]:

| config | columns |
| --- | --- |
| `outputs_in_toolllm_format` | `response`: list of struct{`function_call`: string, `query`: string, `task_id`: int64, `timestamp`: float64} |
| `raw_queries` | `dataset`: string, `query_dict`: string |
| `standardized_api_list` | `dataset`: string, `name`: string, `description`: string, `args_dicts`: list of struct{`default`: string, `description`: string, `name`: string, `required`: bool, `type`: string} |
| `standardized_queries` | `dataset`: string, `prompt`: string, `python_function_name`: string, `python_args_dict`: string, `context_functions`: list of string |

Original-file sizes total 246,052 bytes across the four Parquet files; decoded in-memory size is 971,645 bytes (datasets-server `/size`) [6]. No source states token or sequence-length statistics for any config; none is invented here.

## Quality

- No source states a measured contamination, duplication, or annotator-agreement rate for this benchmark.
- The GitHub README states one explicit quality caveat, on the `toolllm` domain only: the ToolLLM evaluation data originally had no ground truths, so Nexusflow curated, filtered, and post-processed it, leaving only 21 usable samples, and it warns that run-to-run non-determinism in model serving can produce up to a one-sample accuracy swing on that domain, translating to roughly a 5% accuracy swing [1].
- Reading the `dataset` column across the configs that carry it: the full 65-row `standardized_api_list` split (read in full, not truncated) covers four of the five named domains - `cve_cpe`, `emailrep`, `toolalpaca`, `virustotal` - with no `toolllm` value present [11]. The first 100 of 339 `raw_queries` rows and the first 100 of 318 `standardized_queries` rows, both read at offset 0, show only `cve_cpe` and `emailrep` in that range; whether `virustotal`, `toolalpaca`, or `toolllm` appear later in either split was not checked here [12][13]. The `toolllm` domain's queries and reference calls instead live in the separate `outputs_in_toolllm_format` config, which carries no `dataset` column at all: the row's first (query) turn carries a non-null `task_id` and a non-null `query`, with `function_call` null, while every later turn in the same row carries `function_call` set and both `task_id` and `query` null - so turns are grouped by their order inside the row's `response` list, not by a shared `task_id` value repeated across turns [14].

## Load it

Every config's only split is named `train`; there is no split to reserve as a genuine holdout, because this repository is itself the holdout. Pin the revision this card's numbers were read at:

```python
import datasets

REV = "85c7c326fcfb9c6587daabd9582e7cfaddcc65bd"  # main at the check date

outputs = datasets.load_dataset("Nexusflow/NexusRaven_API_evaluation", "outputs_in_toolllm_format", revision=REV, split="train")  # 348 rows
raw_q = datasets.load_dataset("Nexusflow/NexusRaven_API_evaluation", "raw_queries", revision=REV, split="train")                  # 339 rows
api_list = datasets.load_dataset("Nexusflow/NexusRaven_API_evaluation", "standardized_api_list", revision=REV, split="train")     # 65 rows
std_q = datasets.load_dataset("Nexusflow/NexusRaven_API_evaluation", "standardized_queries", revision=REV, split="train")         # 318 rows
```

**Trap**: `split="train"` reads exactly like a training split from every other Hub dataset, and a pipeline that treats any `*/train` config as trainable material will pull this whole evaluation set into a training run. Requesting `data_dir="raw_api_list"` or config `"raw_api_list"` will fail - the card's YAML declares that config, but no matching data file exists in the repository tree at this revision [10].

## Neighbors

No other Hub dataset is this exact evaluation set repackaged; the Nexusflow org hosts a family of later, narrower, single-domain benchmarks, three of which were opened for this card [15]. Prefer this repository for the original NexusRaven-13B single-turn evaluation over any of these:

- `Nexusflow/CVECPEAPIBenchmark` - 56 rows, columns `Input`/`Output`, a plain query-string-to-Python-call-string pair; its live first row's `Output` field wraps the call in a `count_cvecpe_items(...)` helper not present in this repository's `cve_cpe` rows, so it is a distinct reformatting rather than a copy [16][17].
- `Nexusflow/NVDLibraryBenchmark` - 78 rows, columns `Input`, `Output`, `verified` (bool) [16][18]. Its live first row's `Input` and `Output` values are byte-identical to this repository's `raw_queries` first row's embedded `query` and `reference` fields, plus a `verified` flag this repository does not carry [18][12]. Its `Output` field also matches this repository's `standardized_queries` first row on the underlying API call and arguments, but that config exposes the called function as a bare `python_function_name` string (`"searchCPE"`) separate from its arguments, rather than as one executable-looking `nvdlib.searchCPE(...)` expression [18][13].
- `Nexusflow/VirusTotalBenchmark` - 151 rows in the same `Input`/`Output`-style single-split shape as the two CVE/CPE siblings above; row count checked live, schema not compared here [16].
- Other Nexusflow-org repositories (`PlacesAPIBenchmark`, `ClimateAPIBenchmark`, `OTXAPIBenchmark`, `VirusTotalMultiple`, `VT_MultiAPIs`, `Function_Call_Definitions`, `ITType0Benchmark`, `ITType1Benchmark`, `TicketTrackingBenchmark`, `LangChainMathBenchmark`, `LangChainMultitoolTypeWriterHard`, `LangChainRelational`, `MultiverseMathHard`, `VirusTotalAgentic`, `HallucinationTMIBenchmark`) exist under the same org and appear to be later, domain-specific function-calling benchmarks; none of these fifteen was opened for this card, so their relationship to this release is not stated [15].

## A row

Four served configs, four distinct schemas, so four rows. All read live from `config`/`split="train"` at `row_idx=0` [12][13][11][14].

`config="raw_queries"`:

```json
{
  "dataset": "cve_cpe",
  "query_dict": "{\"query\": \"I want to check vulnerability related to Microsoft Exchange 2010. Can you provide me with a list of two representative CPEs?\", \"reference\": \"r = nvdlib.searchCPE(keywordSearch = 'Microsoft Exchange 2010', limit = 2)\"}"
}
```

`config="standardized_queries"`:

```json
{
  "dataset": "cve_cpe",
  "prompt": "I want to check vulnerability related to Microsoft Exchange 2010. Can you provide me with a list of two representative CPEs?",
  "python_function_name": "searchCPE",
  "python_args_dict": "{\"keywordSearch\": \"Microsoft Exchange 2010\", \"limit\": 2}",
  "context_functions": ["searchCVE", "searchCPE"]
}
```

`config="standardized_api_list"` (the `description` field truncated):

```json
{
  "dataset": "cve_cpe",
  "name": "searchCVE",
  "description": "Build and send GET request then return list of objects containing a collection of CVEs. For more information on the parameters available, please visit https://nvd.nist.gov/developers/vulnerabilities. Args: cpeName (str): ... [truncated]",
  "args_dicts": "[list of per-argument structs, each with default/description/name/required/type fields - truncated]"
}
```

`config="outputs_in_toolllm_format"` (a four-turn ToolLLM-style trace: one query turn, three function-call turns, sharing `task_id` on the query turn only):

```json
{
  "response": [
    {"function_call": null, "query": "Show me the object descriptors of URLs associated with the domain 'linked.net'. Please use the API key 'linker_api'.", "task_id": 80165, "timestamp": 1695868781.2787745},
    {"function_call": "vt_get_objects_related_to_domain(domain=\"linked.net\", relationship=\"linked_domain\", x_apikey=\"linker_api\", )", "query": null, "task_id": null, "timestamp": 1695868787.7164962},
    {"function_call": "vt_get_domain_report(domain=\"linked.net\", x_apikey=\"linker_api\", )", "query": null, "task_id": null, "timestamp": 1695868792.5056944},
    {"function_call": "vt_get_objects_related_to_domain(domain=\"linked.net\", relationship=\"domain\", x_apikey=\"linker_api\", )", "query": null, "task_id": null, "timestamp": 1695868797.713006}
  ]
}
```

## Where it came from

Built and released by Nexusflow.ai as the evaluation set for NexusRaven-13B [1][9]. The `cve_cpe`, `emailrep`, and `virustotal` domains are Nexusflow's own curated queries and reference Python API calls against real security-tooling APIs (NVD's CVE/CPE search, EmailRep, VirusTotal) [1]. The `toolalpaca` domain is adapted from the ToolAlpaca paper's evaluation data [2], and the `toolllm` domain is adapted from the ToolLLM paper's evaluation data, which the GitHub README says originally shipped with no ground truths, requiring Nexusflow's own curation, filtering, and post-processing down to 21 usable samples before it could serve as a reference-based benchmark [1][3]. The repository's own instructions describe a four-stage pipeline that produced the served configs: process the ToolAlpaca-derived raw queries, upload `raw_queries`, upload `standardized_api_list`, then derive `standardized_queries` from the two prior configs; a separate script converts the standardized data into the `outputs_in_toolllm_format` config plus recorded model responses used for ToolLLM-format evaluation runs [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision to `85c7c326fcfb9c6587daabd9582e7cfaddcc65bd`. That pin covers only the `load_dataset` calls in Load it. Live-tested against sources [6], [7], and [11]-[14], the `datasets-server.huggingface.co` `/size`, `/info`, and `/first-rows` endpoints ignore any `revision` parameter entirely - a request against this dataset's `/size` endpoint with `revision=deadbeef` returned byte-identical output to the same request with no revision argument at all - so every Shape, Quality, and "A row" number and quotation drawn from those endpoints is a live read, current as of the check date, and not covered by the revision pin.

[1] Nexusflow, "NexusRaven: Surpassing the state-of-the-art in open-source function calling LLMs" (GitHub README). https://github.com/nexusflowai/NexusRaven/raw/main/README.md - domain list, evaluation instructions, toolllm curation caveat, dataset-curation pipeline description. Fetched 2026-08-12.

[2] Tang et al., "ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases", 2023. https://arxiv.org/abs/2306.05301 - current title read from the live abs page. Fetched 2026-08-12.

[3] Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs", 2023. https://arxiv.org/abs/2307.16789 - current title read from the live abs page. Fetched 2026-08-12.

[4] Nexusflow/NexusRaven_API_evaluation dataset card (README). https://huggingface.co/datasets/Nexusflow/NexusRaven_API_evaluation/raw/main/README.md - License section, References section. Fetched 2026-08-12.

[5] Hugging Face Hub API record for Nexusflow/NexusRaven_API_evaluation. https://huggingface.co/api/datasets/Nexusflow/NexusRaven_API_evaluation?full=true - `cardData.dataset_info` (declared configs/rows), licence field absence, gate status, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Nexusflow%2FNexusRaven_API_evaluation Fetched 2026-08-12.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Nexusflow%2FNexusRaven_API_evaluation Fetched 2026-08-12.

[8] The corpus screening row for `Nexusflow/NexusRaven_API_evaluation`, supplied with this card's request - its `flag`, read back in the appendix. Checked 2026-08-12.

[9] Nexusflow/NexusRaven-13B model card (README). https://huggingface.co/Nexusflow/NexusRaven-13B/raw/main/README.md - 95%/64% headline comparison and link to this dataset as its evaluation set. Fetched 2026-08-12.

[10] Hugging Face Hub tree API for Nexusflow/NexusRaven_API_evaluation. https://huggingface.co/api/datasets/Nexusflow/NexusRaven_API_evaluation/tree/main?recursive=true - full file listing showing four Parquet files and no `raw_api_list` path. Fetched 2026-08-12.

[11] datasets-server first-rows endpoint, config `standardized_api_list`, split `train` - all 65 rows returned untruncated. https://datasets-server.huggingface.co/first-rows?dataset=Nexusflow%2FNexusRaven_API_evaluation&config=standardized_api_list&split=train Fetched 2026-08-12.

[12] datasets-server first-rows endpoint, config `raw_queries`, split `train` - first 100 of 339 rows, offset 0, truncated. https://datasets-server.huggingface.co/first-rows?dataset=Nexusflow%2FNexusRaven_API_evaluation&config=raw_queries&split=train Fetched 2026-08-12.

[13] datasets-server first-rows endpoint, config `standardized_queries`, split `train` - first 100 of 318 rows, offset 0, truncated. https://datasets-server.huggingface.co/first-rows?dataset=Nexusflow%2FNexusRaven_API_evaluation&config=standardized_queries&split=train Fetched 2026-08-12.

[14] datasets-server first-rows endpoint, config `outputs_in_toolllm_format`, split `train` - first 100 of 348 rows, offset 0, truncated; no `dataset` column present in this config's schema per [7]. https://datasets-server.huggingface.co/first-rows?dataset=Nexusflow%2FNexusRaven_API_evaluation&config=outputs_in_toolllm_format&split=train Fetched 2026-08-12.

[15] Hugging Face Hub API dataset listing for the Nexusflow org. https://huggingface.co/api/datasets?author=Nexusflow&limit=100 - full list of Nexusflow-org dataset repository names; none beyond the two named in this card's Neighbors section was opened. Fetched 2026-08-12.

[16] datasets-server size endpoint, one call per neighbor: `Nexusflow/CVECPEAPIBenchmark`, `Nexusflow/VirusTotalBenchmark`, `Nexusflow/NVDLibraryBenchmark`. https://datasets-server.huggingface.co/size?dataset=<id> - row counts; these endpoints take no revision parameter, so the counts are live, not pinned. Fetched 2026-08-12.

[17] datasets-server first-rows endpoint, `Nexusflow/CVECPEAPIBenchmark`, config `default`, split `train`, row 0. https://datasets-server.huggingface.co/first-rows?dataset=Nexusflow%2FCVECPEAPIBenchmark&config=default&split=train Fetched 2026-08-12.

[18] datasets-server info and first-rows endpoints, `Nexusflow/NVDLibraryBenchmark`, config `default`, split `train`, row 0. https://datasets-server.huggingface.co/info?dataset=Nexusflow%2FNVDLibraryBenchmark and https://datasets-server.huggingface.co/first-rows?dataset=Nexusflow%2FNVDLibraryBenchmark&config=default&split=train - columns and the first row's values. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Not usable as training data under any split name: this repository is Nexusflow's own evaluation benchmark for NexusRaven-13B and comparison models, and every one of its four served configs is stored under a split literally named `train`, which is the exact shape that would let a naive "load the train split" pipeline fold an evaluation set into training data. The card's opening paragraph and Use it for line both name this trap directly; the entire repository (1,070 served rows, plus the 2 declared-but-unfileable `raw_api_list` rows) belongs in a holdout, not a training run.

### The screening row

The row's own note [8]: "NexusRaven API-calling evaluation queries with reference calls; its splits are named `train` but the repository is the evaluation set." Its flag: "trap: an evaluation set whose splits are all named train."
