# gorilla-llm/Berkeley-Function-Calling-Leaderboard

52 files (49 JSON, plus a README, `eval.yaml`, and `.gitattributes`) - question sets, executable ground truths, and multi-turn API-class function documentation - that together are the test data behind the Berkeley Function Calling Leaderboard (BFCL), not a Hub-loadable train/test dataset.

**gorilla-llm/Berkeley-Function-Calling-Leaderboard** is UC Berkeley's evaluation set for BFCL, the benchmark that scores how well an LLM invokes functions/tools, first released with BFCL V1 and extended through V2 ("Live") and V3 ("multi-turn") [1][2][3]. Each of the 25 top-level JSON files is one BFCL test category: a question file pairs a user prompt with one or more function-call documents (or, for the five multi-turn files, an `initial_config` plus a list of `involved_classes`), a matching `possible_answer/` file (where one exists, 16 files) carries the scored ground truth, and the eight `multi_turn_func_doc/` files carry the reusable API-class schemas (`GorillaFileSystem`, `MathAPI`, `MessageAPI`, `TicketAPI`, `TradingBot`, `TravelAPI`, `TwitterAPI`, `VehicleControlAPI`) that the multi-turn questions reference by name [4]. **The whole repository is BFCL's own scored evaluation surface: every file here, including the executable (`exec_*`) files that carry their ground truth inline and the `possible_answer/` files, is data BFCL grades models against, so training on any file in this repository is direct BFCL contamination** [4]. The dataset's own card states it is not compatible with the Hugging Face `datasets` `load_dataset` method [4]. It lives at https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard .

**Use it for**: nothing training-shaped. This is an eval-only benchmark corpus (function-calling / tool-use accuracy), not preference, SFT, or reward-model data - it does not map to any post-training method card. The files are the harness input for BFCL's own AST- and execution-based scorer, run through the `gorilla` GitHub repository's evaluation code, not through `load_dataset` [4][5].

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, repo tag `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [6]. The one catch: the README states "All the models, and data used to train the models are released under Apache 2.0" [4], but that grant carries no field-of-use clause - the do-not-train restriction above is the dataset's own eval-contamination warning, not a licence term.

**Shape**: 52 files total (25 top-level question JSON files, 16 `possible_answer/` JSON files, 8 `multi_turn_func_doc/` JSON files, plus README.md, `eval.yaml`, and `.gitattributes`), no Hub-recognized splits or configs (`datasets-server` `/info` and `/size` both return "No (supported) data files found") [7][8]. Counted directly from the raw files: 5,251 question rows across the 25 top-level test-category files (5 of them multi-turn, 200 rows each), 3,601 matching `possible_answer/` ground-truth rows across 16 files, and 129 function schemas across 8 `multi_turn_func_doc/` files [9].

**Hold out**: everything. There is no non-eval subset to carve out - the whole repository (all 5,251 question rows, all 3,601 possible-answer rows, all 129 function-doc entries) is the live BFCL scoring set, so none of it should be trained on [4].

**Origin**: built and released by the UC Berkeley Gorilla/BFCL team (`gorilla-llm`); V1 is described as an "Expert Curated (Non-live)" set built by that team, V2 adds "live, user-contributed function documentation and queries," and V3 adds multi-turn/multi-step scenarios on top of both [1][2][3][10]. No generating model is named on the card for any category. Hub API at the check date: `downloads` 101,965, `downloadsAllTime` 417,066, `likes` 117 [6].

**Trained-on-by**: none found. No source consulted for this card documents a model or recipe trained on this repository's files; BFCL is designed and used as a held-out scoring benchmark, and training on it would be the contamination this card's Hold-out line warns against [4].

**Introduced by**: no paper - the dataset card [4], with the BFCL V1 [1], V2 [2], and V3 [3] release blogs as the introducing write-ups for each version.

## Shape

Question-bearing files and their row counts, read directly from each file (one JSON object per line) [9]:

| file | rows | notes |
| --- | --- | --- |
| `BFCL_v3_simple.json` | 400 | single function call |
| `BFCL_v3_multiple.json` | 200 | pick 1 of 2-4 functions |
| `BFCL_v3_parallel.json` | 200 | several calls, one query |
| `BFCL_v3_parallel_multiple.json` | 200 | parallel + multiple combined |
| `BFCL_v3_java.json` | 100 | Java-typed functions |
| `BFCL_v3_javascript.json` | 50 | JavaScript-typed functions |
| `BFCL_v3_sql.json` | 100 | `sql.execute`-style calls |
| `BFCL_v3_rest.json` | 70 | real `requests.get` REST calls |
| `BFCL_v3_chatable.json` | 200 | no functions passed; legacy `{question, function}` shape, no `id` |
| `BFCL_v3_irrelevance.json` | 240 | no function should be called |
| `BFCL_v3_exec_simple.json` | 100 | executable, ground truth inline |
| `BFCL_v3_exec_multiple.json` | 50 | executable |
| `BFCL_v3_exec_parallel.json` | 50 | executable |
| `BFCL_v3_exec_parallel_multiple.json` | 40 | executable |
| `BFCL_v3_live_simple.json` | 258 | user-contributed (V2) |
| `BFCL_v3_live_multiple.json` | 1,053 | user-contributed (V2) |
| `BFCL_v3_live_parallel.json` | 16 | user-contributed (V2) |
| `BFCL_v3_live_parallel_multiple.json` | 24 | user-contributed (V2) |
| `BFCL_v3_live_relevance.json` | 18 | user-contributed (V2) |
| `BFCL_v3_live_irrelevance.json` | 882 | user-contributed (V2) |
| `BFCL_v3_multi_turn_base.json` | 200 | multi-turn, `initial_config` |
| `BFCL_v3_multi_turn_composite.json` | 200 | multi-turn, augmented |
| `BFCL_v3_multi_turn_long_context.json` | 200 | multi-turn, augmented |
| `BFCL_v3_multi_turn_miss_func.json` | 200 | multi-turn, augmented |
| `BFCL_v3_multi_turn_miss_param.json` | 200 | multi-turn, augmented |
| **total** | **5,251** | |

`possible_answer/` carries 3,601 ground-truth rows across the 16 categories that have a single scoreable answer form (java, javascript, live_simple, live_multiple, live_parallel, live_parallel_multiple, multiple, parallel, parallel_multiple, simple, sql, and the 5 multi_turn files) [9]. It has no counterpart for `chatable`, `irrelevance`, `live_irrelevance`, `live_relevance`, or `rest` (no fixed correct-call answer), and none for the `exec_*` files, whose ground truth is embedded inline as their own `ground_truth` field [9]. `multi_turn_func_doc/` carries 129 function schemas across 8 files (`gorilla_file_system.json` 18, `math_api.json` 17, `message_api.json` 10, `posting_api.json` 14, `ticket_api.json` 9, `trading_bot.json` 22, `travel_booking.json` 17, `vehicle_control.json` 22) [9]. No source states sequence-length or token statistics for any file; none is invented here.

## Quality

- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this repository.
- The README documents known scope limits per category rather than measured error rates: SQL AST scoring is excluded from the live leaderboard because "the multiplicity of methods to construct SQL function calls achieving identical outcomes" makes a fixed answer set unreliable, and the authors describe SQL evaluation as still in progress [4]. Chat-only scenarios are likewise excluded from the live leaderboard statistics and are described as included "for internal model evaluation," with the authors stating "We currently are working on a better evaluation of chat ability" [4].
- V1's construction is described only as being built "from our learnings to be representative of most users' function calling use-cases" [4]; V2's construction is described as "live, user-contributed function documentation and queries," explicitly framed as a way to avoid dataset contamination and bias in a static benchmark [2]. Neither description states a formal annotation-quality process or inter-annotator agreement figure.
- `eval.yaml` in the repository declares this dataset for the `nemo-evaluator` framework under two task ids, `bfclv3` and `bfclv2`, each pointed at `config: default, split: test` - a harness-level label, not a Hub-recognized dataset split [11].

## Load it

The dataset card explicitly warns against `load_dataset`: "**DO NOT** use the HuggingFace `load_dataset` method to load the dataset as our dataset is not compatible with the HuggingFace datasets package" [4]. It instead points readers to the `gorilla` GitHub repository's evaluation instructions and gives a line-by-line JSON-lines loader [4][5]:

```python
import json
from huggingface_hub import hf_hub_download

REV = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"  # main at the check date

path = hf_hub_download(
    repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
    repo_type="dataset",
    filename="BFCL_v3_simple.json",
    revision=REV,
)

def load_file(file_path: str):
    result = []
    with open(file_path) as f:
        for line in f:
            result.append(json.loads(line))
    return result

rows = load_file(path)  # 400 rows for BFCL_v3_simple.json
```

**Trap**: each of the 52 files must be fetched and parsed individually this way - there is no single config that returns the whole repository, and `datasets-server`'s `/info` and `/size` both fail with "No (supported) data files found," so the usual `load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard")` one-liner does not work at all [7][8].

## Neighbors

- `gorilla-llm/APIBench` - an earlier dataset from the same `gorilla-llm` org, listed via the same Hub API org query used for this card [6], for the original Gorilla API-invocation task; the introducing paper describes it as consisting of HuggingFace, TorchHub, and TensorHub APIs [12]. A different task, not a re-release of this repository.
- Several third-party Hub repositories mirror this dataset under the same name (e.g. `AndyChen123/Berkeley-Function-Calling-Leaderboard-Fix`, `yk12311/Berkeley-Function-Calling-Leaderboard`); the one checked, `AndyChen123/...-Fix`, serves only a single file (`BFCL_v3_simple.json`) and has 74 downloads against this repository's 101,965, so it is a partial, low-adoption mirror, not a preferred alternative [6][13].
- The `ShishirPatil/gorilla` GitHub repository's `berkeley-function-call-leaderboard` directory is the runnable evaluation harness for these files, not a data neighbor [5].

## A row

The repository has no Hub configs or splits; each block below names the source file it was read from. Long fields are truncated with `[...]`.

Single-turn AST shape (`data/BFCL_v3_simple.json`, row 0) [9]:

```json
{
  "id": "simple_0",
  "question": [[{"role": "user", "content": "Find the area of a triangle with a base of 10 units and height of 5 units."}]],
  "function": [{"name": "calculate_triangle_area", "description": "Calculate the area of a triangle given its base and height.", "parameters": {"type": "dict", "properties": {"base": {"type": "integer", "description": "The base of the triangle."}, "height": {"type": "integer", "description": "The height of the triangle."}, "unit": {"type": "string", "description": "The unit of measure (defaults to 'units' if not specified)"}}, "required": ["base", "height"]}}]
}
```

Legacy chat shape (`data/BFCL_v3_chatable.json`, row 0) - no `id` field, `function` is an empty string [9]:

```json
{"question": "Find the area of a triangle with a base of 10 units and height of 5 units.", "function": ""}
```

Executable shape, ground truth inline (`data/BFCL_v3_exec_simple.json`, row 0) [9]:

```json
{
  "id": "exec_simple_0",
  "question": [[{"role": "user", "content": "I've been playing a game where rolling a six is somehow more likely than usual [...]"}]],
  "function": [{"name": "calc_binomial_probability", "description": "Calculates the probability of getting k successes in n trials.", "parameters": {"type": "dict", "properties": {"n": {"type": "integer", "description": "The number of trials."}, "k": {"type": "integer", "description": "The number of successes."}, "p": {"type": "float", "description": "The probability of success."}}, "required": ["n", "k", "p"]}}],
  "execution_result_type": ["exact_match"],
  "ground_truth": ["calc_binomial_probability(n=20, k=5, p=0.6)"]
}
```

Multi-turn question shape (`data/BFCL_v3_multi_turn_base.json`, row 0) [9]:

```json
{
  "id": "multi_turn_base_0",
  "question": [[{"role": "user", "content": "Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory"}], [{"role": "user", "content": "Perform a detailed search using grep [...]"}]],
  "initial_config": {"GorillaFileSystem": {"root": {"workspace": {"type": "directory", "contents": {"document": {"type": "directory", "contents": {"final_report.pdf": {"type": "file", "content": "Year2024 [...]"}}}}}}}, "TwitterAPI": {"tweet_counter": 3, "tweets": {"0": {"id": 0, "username": "analyst_pro", "content": "Just finished analyzing the reports!", "tags": ["#analysis", "#reports"], "mentions": []}}}},
  "involved_classes": ["GorillaFileSystem", "TwitterAPI"],
  "path": "[...]"
}
```

Single-turn possible-answer shape (`possible_answer/BFCL_v3_simple.json`, row 0) - joins to the question file above on `id` [9]:

```json
{"id": "simple_0", "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]}
```

Multi-turn possible-answer shape (`possible_answer/BFCL_v3_multi_turn_base.json`, row 0) - `ground_truth` is a list of per-turn action-string lists, a different structure from the single-turn dict-of-parameters shape above, though it joins the same way, on `id` [9]:

```json
{"id": "multi_turn_base_0", "ground_truth": [["cd(folder='document')", "mkdir(dir_name='temp')", "mv(source='final_report.pdf', destination='temp')"], ["cd(folder='temp')", "grep(file_name='final_report.pdf',pattern='budget analysis')"], ["sort('final_report.pdf')"], ["[...]"]]}
```

Function-schema shape (`multi_turn_func_doc/gorilla_file_system.json`, row 0) - referenced by the multi-turn question files' `involved_classes` list, not joined by `id` [9]:

```json
{
  "name": "cat",
  "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Display the contents of a file of any extension from currrent directory.",
  "parameters": {"type": "dict", "properties": {"file_name": {"type": "string", "description": "The name of the file from current directory to display. No path is allowed. "}}, "required": ["file_name"]},
  "response": {"type": "dict", "properties": {"file_content": {"type": "string", "description": "The content of the file."}}}
}
```

## Where it came from

The repository is built and released by the UC Berkeley Gorilla/BFCL team under the `gorilla-llm` Hub org [4][6]. It is assembled from three releases layered on top of one another: BFCL V1 is an "Expert Curated (Non-live)" set the team built from its own analysis of function-calling use cases (agents, enterprise workflows), covering Python (simple, multiple, parallel, parallel-multiple) and non-Python categories (chat, relevance, REST, SQL, Java, JavaScript), each with AST and, where applicable, hand-written executable evaluation functions [1][4][10]. BFCL V2 "Live" adds a "User Contributed (Live)" pool of function documentation and queries contributed by real users and the OSS community, aimed at reducing the contamination and bias risk of a static, team-authored set [2][10]. BFCL V3 adds multi-turn and multi-step scenarios - built on the V1 and V2 categories - where a model must call functions across several conversational turns against a simulated backend state (`initial_config`), including variants that withhold required parameters or functions to test clarification behavior [3][4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and file row above. Hugging Face Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the `datasets-server` size/info endpoints checked below take no revision parameter and are reported as live, not pinned.

[1] "Berkeley Function Calling Leaderboard" (BFCL V1 release blog). https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html - V1 scope, category composition, "we built this dataset from our learnings" text, hand-written executable functions. Fetched 2026-08-11.

[2] "BFCL V2 • Live" (BFCL V2 release blog). https://gorilla.cs.berkeley.edu/blogs/12_bfcl_v2_live.html - "live, user-contributed function documentation and queries" description, contamination/bias rationale. Fetched 2026-08-11.

[3] "BFCL V3 • Multi-Turn & Multi-Step Function Calling" (BFCL V3 release blog). https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html - multi-turn/multi-step scope, and the "Expert Curated (Non-live)" / "User Contributed (Live)" labels for V1/V2. Fetched 2026-08-11.

[4] gorilla-llm/Berkeley-Function-Calling-Leaderboard dataset card (README). https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/README.md - category descriptions, `load_dataset` incompatibility warning, `load_file` loader, licence statement, chat/SQL scope caveats. Fetched 2026-08-11.

[5] `ShishirPatil/gorilla` GitHub repository, `berkeley-function-call-leaderboard` directory README, linked from [4] as the evaluation instructions. https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/README.md - confirms this directory is the runnable BFCL evaluation harness. Fetched 2026-08-11.

[6] Hugging Face Hub API record for gorilla-llm/Berkeley-Function-Calling-Leaderboard. https://huggingface.co/api/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard?full=true - licence, gate, `sha`, `downloads`, `likes`, siblings list; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Also used, via `https://huggingface.co/api/datasets?author=gorilla-llm` and `?search=berkeley-function-calling`, to check the org's other datasets and third-party mirrors, including `AndyChen123/Berkeley-Function-Calling-Leaderboard-Fix`'s file list and download count. Fetched 2026-08-11.

[7] `datasets-server` size endpoint. https://datasets-server.huggingface.co/size?dataset=gorilla-llm%2FBerkeley-Function-Calling-Leaderboard - returns "No (supported) data files found in gorilla-llm/Berkeley-Function-Calling-Leaderboard". Fetched 2026-08-11.

[8] `datasets-server` info endpoint. https://datasets-server.huggingface.co/info?dataset=gorilla-llm%2FBerkeley-Function-Calling-Leaderboard - same "No (supported) data files found" error. Fetched 2026-08-11.

[9] Every one of the 52 raw files in the repository (25 top-level question files, 16 `possible_answer/` files, 8 `multi_turn_func_doc/` files, plus README.md, `eval.yaml`, and `.gitattributes`), enumerated via the recursive tree endpoint (https://huggingface.co/api/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/tree/main?recursive=true) and each JSON file fetched individually at https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/<path>, then read as newline-delimited JSON to count rows and inspect row shape and keys. Fetched 2026-08-11.

[10] Repeats the V1/"Expert Curated"/V2/"User Contributed" framing already cited at [1][2][3]; grouped here where the README's own version-history section is the proximate source for the three-release structure. Fetched 2026-08-11 (README, https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/README.md).

[11] `eval.yaml` in the repository. https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/eval.yaml - `nemo-evaluator` task declarations `bfclv3` and `bfclv2`, each `config: default, split: test`. Fetched 2026-08-11.

[12] Patil et al., "Gorilla: Large Language Model Connected with Massive APIs", 2023. https://arxiv.org/abs/2305.15334 - the paper introducing APIBench, describing it as consisting of HuggingFace, TorchHub, and TensorHub APIs. Fetched 2026-08-11.

[13] AndyChen123/Berkeley-Function-Calling-Leaderboard-Fix dataset card and Hub API record. https://huggingface.co/datasets/AndyChen123/Berkeley-Function-Calling-Leaderboard-Fix/raw/main/README.md and https://huggingface.co/api/datasets/AndyChen123/Berkeley-Function-Calling-Leaderboard-Fix?full=true - single-file (`BFCL_v3_simple.json`) sibling list, 74 downloads. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Not usable as training data, and correctly so: this repository is BFCL's own evaluation surface end to end. The two facts that decide it are already established above - the dataset's own card blocks `load_dataset` and treats these files as the harness input for BFCL's scorer rather than as a trainable corpus [4], and every category, including the executable and ground-truth files, is data the live leaderboard grades models against, which the screening flag names directly as a contamination trap.

### The screening row

The row's own note: "The BFCL question and function-documentation files themselves (V1 hand-built by the Berkeley team, V2 real-world user-contributed live data, V3 multi-turn), i.e. the data the bfcl benchmark scores against; no generating model is named on the card." Its flag: "trap: the whole repo is the bfcl eval surface - every file, including any train-shaped subset, is bfcl eval data, so training on it is direct bfcl contamination."
