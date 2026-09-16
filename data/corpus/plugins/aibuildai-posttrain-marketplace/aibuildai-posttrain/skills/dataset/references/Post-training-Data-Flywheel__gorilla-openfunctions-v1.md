# Post-training-Data-Flywheel/gorilla-openfunctions-v1

12,125 single-turn function-calling conversations - a user request, a set of candidate API-function documents, and one assistant reply that invokes one or more of them - repackaged from the training data behind Berkeley's Gorilla OpenFunctions v1 model.

**Post-training-Data-Flywheel/gorilla-openfunctions-v1** reformats the training set for `gorilla-openfunctions-v1`, a 7B function-calling model the Gorilla team (UC Berkeley) describes as trained on "a curated set of API documentation, and Question-Answer pairs generated from the API documentations" and released as an open alternative to OpenAI function calling [1]. The Gorilla project itself was introduced in "Gorilla: Large Language Model Connected with Massive APIs" [2], but that paper covers the earlier Gorilla model and its APIBench benchmark, not this OpenFunctions v1 training set specifically. **This is training data, not the Berkeley Function Calling Leaderboard (BFCL): BFCL is a separate evaluation release built by the same Gorilla team [3], and the corpus screening note for this row warns to check for overlap between this training set and BFCL's test prompts before reporting a model's BFCL score [4].** It lives at https://huggingface.co/datasets/Post-training-Data-Flywheel/gorilla-openfunctions-v1 .

**Use it for**: single-turn SFT chat data for function-calling - the SFT method card's chat schema, with a `system` message, a `tools` list of candidate API documents, and a two-turn `messages` list (user request, assistant function call). The restriction above applies: verify no BFCL-prompt overlap before using a model trained on this set to report BFCL numbers [4]. One format trap: each entry in `tools` is a Python `dict` repr string (single-quoted, e.g. `{'name': ..., 'parameters': {...}}`), not JSON - parse with `ast.literal_eval`, not `json.loads` [5].

**Licence**: Apache-2.0 (`cardData.license: apache-2.0`, tag `license:apache-2.0`), ungated [6]. The Gorilla OpenFunctions model card's Contributing section states that "All the models, and data used to train the models is released under Apache 2.0." [8], so the Apache-2.0 tag here matches the upstream team's own stated license for this training data. The one catch: the identical-content mirror of the same training file at `SumanthRH/openfunctions-v1-train` carries no license tag at all [7], so that copy does not surface the license on its own.

**Shape**: 12,125 rows, one split (`train`), one config (`default`), four columns [5][9].

**Hold out**: nothing to hold out inside this file - it ships only a `train` split, no internal dev/test division [9]. The risk instead sits outside the file: this is the Gorilla team's own OpenFunctions v1 training data, and the same team built BFCL for evaluating function-calling models [1][3]; the screening note flags that overlap between this training set and BFCL's prompts is unchecked, so confirm no overlap before scoring a model trained on this data against BFCL [4].

**Origin**: Gorilla OpenFunctions v1 training data was collected by the Gorilla team (UC Berkeley); the blog describes the question-answer pairs as "generated from the API documentations" without naming the generator model, so the specific generating model is not stated [1]. This repackaging is by the Hub org `Post-training-Data-Flywheel`. Hub API at the check date: `downloads` 419, `downloadsAllTime` 2,843, `likes` 0 [6].

**Trained-on-by**: `gorilla-openfunctions-v1`, the Gorilla team's own 7B model (LLaMA-2 base), which the team describes as extending the earlier v0 model to handle multiple API calls and choose between functions [1][8]. A separate mirror, `SumanthRH/openfunctions-v1-train`, carries a file literally named `gorilla_openfunctions_v1_train.json` whose first record matches this dataset's first row word for word [7] - consistent with this being that model's release training file, though no source states outright that this exact Hub copy is the file used to train the released weights. No other downstream model or recipe is named in any source read for this card. On the blog's own function-calling benchmark against a 116-query test set, the sibling `gorilla-openfunctions-v0` model reaches an 87.39% success rate against a roughly 95% success rate for GPT-4 and GPT-3.5-Turbo function calling; the blog states that "All of the results in this blog are generated using `gorilla-openfunctions-v0`." [1], not v1, and gives v1 - the model this dataset actually trains - no equivalent published number, describing it only as an "early pre-view" expected to improve [1].

**Introduced by**: no paper - the dataset card [10] (a bare YAML header with no prose), plus the introducing blog, "OpenFunctions" [1].

## Shape

Rows and splits (datasets-server `/size`) [9]:

| split | rows |
| --- | --- |
| `train` | 12,125 |

One config, `default`, four columns (datasets-server `/info`, `cardData.dataset_info` in the README) [5][11]:

| column | dtype |
| --- | --- |
| `conversation_id` | int64 |
| `system` | string |
| `tools` | list\<string\> |
| `messages` | list\<struct\<content: string, role: string\>\> |

Byte sizes from the README's `dataset_info` and datasets-server `/size`: 9,407,376 bytes download (Parquet), 28,823,306 bytes declared dataset size, 26,397,349 bytes in memory per datasets-server [9][11]. No source states token or sequence-length statistics for this dataset.

## Quality

- Of the first 100 served rows, every row has an empty `system` string and exactly two `messages` (one `user`, one `assistant`) - none carry a system prompt or a multi-turn exchange [12].
- Of those same 100 rows, 85 have an assistant reply spanning more than one line - in every one of the ten inspected, this is the same function call repeated once per requested unit (e.g. a user asking to "create three ArrowArray objects" gets `pyarrow.ArrowArray()` repeated three times), not distinct parallel calls to different functions [12]. This is consistent with the model card's description of `gorilla-openfunctions-v1` as supporting parallel function calls [8], but no source states an overall rate of this pattern across all 12,125 rows.
- The `tools` column stores each candidate function document as a Python `dict` repr (single-quoted keys, e.g. `{'name': 'Torch', ...}`), not valid JSON [12]. No source states this was intentional or flags it as a defect; it is reported here as observed in the served rows.
- No source states a measured contamination rate, duplicate rate, or annotation-agreement figure for this dataset; none is invented here.

## Load it

Only one split and one config; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-08-23) [6]:

```python
import ast
import datasets

REV = "a9eb8453e413d98456f8529be113609c4c70f99a"  # main at the check date
ds = datasets.load_dataset(
    "Post-training-Data-Flywheel/gorilla-openfunctions-v1", revision=REV, split="train"
)  # 12,125 rows

tools = [ast.literal_eval(t) for t in ds[0]["tools"]]  # NOT json.loads - see the trap below
```

**Trap**: `ds[0]["tools"]` is a list of strings, and each string is a Python `dict` literal with single-quoted keys, not JSON - calling `json.loads` on it raises `json.decoder.JSONDecodeError`; use `ast.literal_eval` instead [5][12].

## Neighbors

- `Post-training-Data-Flywheel/gorilla-apibench` - the closest sibling: same Hub org, same four columns (`conversation_id`, `system`, `tools`, `messages`) with the same dtypes, but a different 16,166-row shape spread across five splits (`huggingface_train` 8,191, `huggingface_eval` 911, `tensorflow_train` 6,190, `tensorflow_eval` 688, `torchhub_eval` 186) [14][15]. A fetched row from `huggingface_train` shows the content differs from this dataset despite the matching columns: `tools` holds valid double-quoted JSON here (not this dataset's Python-repr strings), the `user` message is phrased as "Instruction: ..." rather than a plain request, and the `assistant` message is a long structured "Output: <<<domain>>>... <<<explanation>>>... <<<code>>>..." completion rather than this dataset's bare function-call expression [16]. Same builder, overlapping schema, different collator - do not merge the two without accounting for that. It also ships three `_eval` splits (1,785 rows total) that a reader must hold out if training on it directly.
- `SumanthRH/openfunctions-v1-train` - an unformatted mirror of the same underlying data as two files, `gorilla_openfunctions_v1_train.json` and `.jsonl`, with `Instruction`/`Functions` columns instead of `system`/`tools`/`messages`; its first record's instruction text and function documentation match this dataset's row 0 word for word, and it carries no license tag [7]. This corpus prefers the `Post-training-Data-Flywheel` copy for its ready-to-load Parquet format and stated Apache-2.0 license.
- `gorilla-llm/APIBench` - the Gorilla project's original API-invocation benchmark/training corpus from the origin paper [2], covering 1,600+ APIs across HuggingFace, TorchHub and TensorHub; it is a different, earlier collection than this OpenFunctions v1 training set and is not Parquet-served, so no row count is available here [2][13].
- `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (BFCL) - the same team's live function-calling evaluation set, explicitly not loadable with `datasets.load_dataset` and organized as per-category JSON/JSONL files instead [3]. This is the evaluation set named in the Hold out risk above, not a training-data neighbor to mix in.

## A row

One served shape (single config, single split). From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [12], with the two longer `tools` entries truncated:

```json
{
  "conversation_id": 0,
  "system": "",
  "tools": [
    "{'name': 'Torch', 'api_name': 'torch.linspace', 'description': 'Create a one-dimensional tensor with evenly spaced values', 'parameters': {'start': {'type': 'float', ...}, ..., 'requires_grad': {'type': 'bool', 'description': 'Optional flag to enable gradient tracking'}}}\n",
    "{'name': 'RapidAPI', 'api_name': 'requests.get', 'description': 'NOTE: You need an API-Key to use this API. ... The Cancer Imaging Archive (TCIA) is a public repository of cancer images...', 'parameters': [{'name': 'format', 'description': 'Specify output type. Allowed values CSV/HTML/XML/JSON', 'type': 'STRING'}]}\n",
    "{'name': 'pyarrow', 'api_name': 'read_tensor', 'description': 'Read pyarrow.Tensor from pyarrow.NativeFile object from current position', 'parameters': {'required': [{'name': 'source', 'description': 'pyarrow.NativeFile object'}], 'optional': []}}\n",
    "{'name': 'alpha', 'api_name': 'gcloud.alpha.builds.enterprise_config.bitbucketserver.delete', 'description': 'Delete a Bitbucket Server config from Google Cloud Build', 'parameters': [{'name': 'config', ...}, {'name': 'region', ...}]}\n",
    "{'name': 'aws', 'api_name': 'aws.es.describe_domain_auto_tunes', 'description': 'Provides scheduled Auto-Tune action details for the Elasticsearch domain, such as Auto-Tune action type, description, severity, and scheduled date.', 'parameters': [{'name': 'domain_name', ...}, {'name': 'max_results', ...}, {'name': 'next_token', ...}]}"
  ],
  "messages": [
    {
      "content": "I want to create a one-dimensional tensor with evenly spaced values from 0 to 1 using the torch.linspace API.\n",
      "role": "user"
    },
    {
      "content": "torch.linspace(start=0,end=1,steps=10)",
      "role": "assistant"
    }
  ]
}
```

## Where it came from

Built by the Gorilla team (UC Berkeley) as the training data for `gorilla-openfunctions-v1`, a follow-on to the original Gorilla model that adds parallel and multi-function support [1][8]. The OpenFunctions blog post itself is bylined Fanjia Yan, Adam Lee, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez [1] - a different, overlapping byline from the earlier Gorilla/APIBench paper's authors, Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez [2][13]. The team's blog describes the source material as "a curated set of API documentation, and Question-Answer pairs generated from the API documentations" [1], without naming the generating model. This Hub repository, under the `Post-training-Data-Flywheel` org, repackages that training file (which also circulates separately, in `Instruction`/`Functions` form, as `SumanthRH/openfunctions-v1-train` [7]) into a `system`/`tools`/`messages` chat schema and adds an Apache-2.0 license tag.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] "OpenFunctions" blog post, Gorilla team (UC Berkeley). https://gorilla.cs.berkeley.edu/blogs/4_open_functions.html - training-data description, v0/v1 model descriptions, model authorship context. Fetched 2026-08-11.

[2] Patil et al., "Gorilla: Large Language Model Connected with Massive APIs", 2023. https://arxiv.org/abs/2305.15334 - the Gorilla project's origin paper; covers the original Gorilla model and APIBench, not this OpenFunctions v1 training set directly. Current title read from the live abs page. Fetched 2026-08-11.

[3] `gorilla-llm/Berkeley-Function-Calling-Leaderboard` dataset card (README). https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/README.md - BFCL description, non-`load_dataset`-compatible format. Fetched 2026-08-11.

[4] The corpus screening row for `Post-training-Data-Flywheel/gorilla-openfunctions-v1`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Post-training-Data-Flywheel%2Fgorilla-openfunctions-v1 Fetched 2026-08-11.

[6] Hugging Face Hub API record for `Post-training-Data-Flywheel/gorilla-openfunctions-v1`. https://huggingface.co/api/datasets/Post-training-Data-Flywheel/gorilla-openfunctions-v1?full=true - license, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[7] `SumanthRH/openfunctions-v1-train` repository. https://huggingface.co/api/datasets/SumanthRH/openfunctions-v1-train?full=true and https://huggingface.co/datasets/SumanthRH/openfunctions-v1-train/resolve/main/gorilla_openfunctions_v1_train.jsonl - file listing (no license tag), first-record comparison. Fetched 2026-08-11.

[8] `gorilla-llm/gorilla-openfunctions-v1` model card (README). https://huggingface.co/gorilla-llm/gorilla-openfunctions-v1/raw/main/README.md - model description ("+ Parallel functions, and can choose between functions"), Apache-2.0 license tag for the model. Fetched 2026-08-11.

[9] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Post-training-Data-Flywheel%2Fgorilla-openfunctions-v1 Fetched 2026-08-11.

[10] `Post-training-Data-Flywheel/gorilla-openfunctions-v1` dataset card (README). https://huggingface.co/datasets/Post-training-Data-Flywheel/gorilla-openfunctions-v1/raw/main/README.md - YAML-only card with no prose body; source of `cardData.dataset_info` figures. Fetched 2026-08-11.

[11] Same as [10], `cardData.dataset_info` byte-size and split fields.

[12] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Post-training-Data-Flywheel%2Fgorilla-openfunctions-v1&config=default&split=train Fetched 2026-08-11.

[13] `gorilla-llm/APIBench` dataset card (README). https://huggingface.co/datasets/gorilla-llm/APIBench/raw/main/README.md - author list, project description, dataset date. Fetched 2026-08-11.

[14] Hugging Face Hub API record for `Post-training-Data-Flywheel/gorilla-apibench`. https://huggingface.co/api/datasets/Post-training-Data-Flywheel/gorilla-apibench?full=true - license, gate, downloads. Fetched 2026-08-11.

[15] `Post-training-Data-Flywheel/gorilla-apibench` dataset card (README) and datasets-server size/info endpoints. https://huggingface.co/datasets/Post-training-Data-Flywheel/gorilla-apibench/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=Post-training-Data-Flywheel%2Fgorilla-apibench , https://datasets-server.huggingface.co/info?dataset=Post-training-Data-Flywheel%2Fgorilla-apibench - column schema, split names and row counts. Fetched 2026-08-11.

[16] datasets-server first-rows endpoint for `Post-training-Data-Flywheel/gorilla-apibench`. https://datasets-server.huggingface.co/first-rows?dataset=Post-training-Data-Flywheel%2Fgorilla-apibench&config=default&split=huggingface_train - row 0 content, compared against this dataset's row 0. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT function-calling data, with one flag to clear before use: this is the Gorilla team's own OpenFunctions v1 training data, not the Berkeley Function Calling Leaderboard, and the screening note requires checking for overlap between this training set and BFCL before reporting BFCL scores for anything trained on it [1][3][4].

### The screening row

The row's own note [4]: "Gorilla OpenFunctions TRAINING conversations (API doc to a single call) from the same team as the Berkeley Function Calling Leaderboard; not BFCL itself, but check overlap before reporting BFCL scores." The row carries no separate flag field beyond this note.
