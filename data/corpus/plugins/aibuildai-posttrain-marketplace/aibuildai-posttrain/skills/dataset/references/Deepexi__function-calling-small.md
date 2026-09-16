# Deepexi/function-calling-small

24,608 Chinese single-turn function-selection examples - a system prompt listing candidate functions, a user question, and the assistant's chosen function call - built over 700+ real Alibaba Cloud OpenAPI specs.

**Deepexi/function-calling-small** is released by the Deepexi organization on the Hugging Face Hub, with no origin paper; the card itself is the only description of how it was built [1]. It lives at https://huggingface.co/datasets/Deepexi/function-calling-small . Each row's `systemPrompt` names several Alibaba Cloud OpenAPI functions - drawn from "700+" public OpenAPI operations across DataWorks, EMR, DataLake, MaxCompute, Hologres, Realtime Compute for Apache Flink, Quick BI, and DTS - and instructs the model to answer in a fixed `{"function": ..., "arguments": ...}` format; `userPrompt` is a Chinese question naming one of the listed functions; `assistantResponse` is the function call that answers it [1]. The API names, descriptions and argument docs are the real, human-written Alibaba Cloud OpenAPI documentation; the card states nothing about how the `userPrompt` questions themselves were produced. **Of the first 48 rows served (offset 0, the full page the `first-rows` endpoint returns), only 12 distinct `userPrompt` strings appear, each repeated roughly four times with a different candidate-function list in `systemPrompt` and the same target call in `assistantResponse` - this repeats the same query/answer pair with varying distractor pools rather than varying the underlying task [2].**

**Use it for**: single-turn function-calling / tool-use SFT (system + user + assistant, three string columns) - the format this maps to is a plain chat-SFT triple, not a preference pair; see the SFT method card. The templated repetition described above means the 24,608 rows likely contain far fewer distinct query/answer templates than rows, which a trainer should account for before treating row count as example diversity.

**Licence**: CC-BY-4.0 (`cardData.license` is `"cc-by-4.0"`, tag `license:cc-by-4.0`), ungated (`"gated": false`, `"private": false`) [3]. The one catch: CC-BY-4.0 requires attribution and permits commercial use; the API documentation and generated Q&A pairs are not separately licensed.

**Shape**: 24,608 rows, one config (`default`), one split (`train`), three string columns (`systemPrompt`, `userPrompt`, `assistantResponse`) [4][5].

**Hold out**: nothing stated. No source describes an evaluation split, a held-out set, or an overlap with any benchmark; the repository ships a single CSV loaded entirely into `train` [3][5]. The screening row's note raises no contamination concern either [6].

**Origin**: built and released by Deepexi; the API documentation is real, human-written Alibaba Cloud OpenAPI reference text, and the card does not say how the `userPrompt` questions or the distractor-function selection were generated [1]. Hub API at the check date: `downloads` 226, `downloadsAllTime` 2,599, `likes` 21 [3].

**Trained-on-by**: none found - no source names a model or recipe trained on this dataset.

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and split, from the datasets-server size endpoint [4]:

| split | rows |
| --- | --- |
| `train` | 24,608 |
| total | 24,608 |

One config, `default`, three string columns, read via the `csv` builder (datasets-server info endpoint) [5]:

| column | dtype |
| --- | --- |
| `systemPrompt` | string |
| `userPrompt` | string |
| `assistantResponse` | string |

Byte sizes (datasets-server size endpoint) [4]: 111,809,738 bytes as the original CSV download, 25,981,493 bytes as Parquet, 104,263,632 bytes decoded in memory. The size and info endpoints take no revision parameter, so these row and byte counts are what the live `main` branch reports at the check date, not values pinned to the commit used in Load it; they matched the Hub API's `sha` for `main` at fetch time [3][4][5]. No source states sequence-length or token statistics for this dataset; none is invented here.

The repository's tags include `task_categories:feature-extraction` [3], which does not describe this content - every served row is a system/user/assistant function-call triple, not an embedding or extraction task; no source explains the tag.

## Quality

- The API documentation embedded in `systemPrompt` (function names, descriptions, argument specs) is the real, human-written Alibaba Cloud OpenAPI reference [1].
- No source states how the `userPrompt` questions were authored (hand-written, templated, or model-generated) or how the distractor functions listed alongside the target function in each `systemPrompt` were chosen [1].
- Reading the first 48 rows served at offset 0 (config `default`, split `train`) [2] found only 12 distinct `userPrompt` values and 6 distinct `assistantResponse` values among them; each distinct question/answer pair recurs about four times, each time paired with a different set of distractor functions in `systemPrompt`. This is a fact about those 48 rows only; no wider claim about the remaining 24,560 rows is made here.
- No source states a measured contamination rate, duplicate-row rate across the full split, or annotator/QA process for this dataset.

## Load it

```python
import datasets

REV = "8fba7ee941523e7da69f4ff882b52e5566d24288"  # main at the check date
train = datasets.load_dataset("Deepexi/function-calling-small", revision=REV, split="train")  # 24,608 rows
```

**Trap**: the repository ships one CSV file, `function-calling-small_aliyun_openapi_V2.csv`, loaded whole into a single `train` split with no other split or config to select [3][5]; there is no held-out portion to exclude by passing a different split name.

## Neighbors

- `Deepexi/openai-formate-function-calling-small` - the same 700+ Alibaba Cloud OpenAPI functions and the same row count, 24,608, with `systemPrompt` reformatted so the function list matches OpenAI's function-calling schema (`name`/`parameters` instead of `function`/`arguments`), released under Apache-2.0 rather than CC-BY-4.0 [7][8].
- `Deepexi/glaive-function-calling-vicuna`, also from Deepexi, is an SFT-format reformatting (fields `id` and `conversations`) of the separate `glaiveai/glaive-function-calling` dataset - an unrelated English corpus, not a re-release of this one [9].

## A row

One config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server first-rows endpoint), with the function-list block in `systemPrompt` truncated [2]:

```json
{
  "systemPrompt": "你是一个函数筛选助理，如果与问题相关的话,您可以使用下面的函数来获取更多数据以回答用户提出的问题:\n{\"function\": \"CreateExportMigration\", \"description\": \"使用CreateExportMigration，新建DataWorks导出任务且仅创建导出任务。\", \"arguments\": [...] }\n\n{\"function\": \"UpdateTicketNum\", \"description\": \"对用于免登嵌入报表的指定的ticket进行更新票据数量操作。\", \"arguments\": [{\"name\": \"Ticket\", \"type\": \"string\", \"description\": \"三方嵌入的票据值，即URL中的accessTicket值。\"}, {\"name\": \"TicketNum\", \"type\": \"integer\", \"description\": \"票据数。\\n- 取值范围：1~99998，建议值为1。\"}]}\n\n{\"function\": \"CreateSavepoint\", \"description\": \"创建快照\", \"arguments\": [...] }\n\n\"\n    请以如下格式回复：:\n    {\n        \"function\": \"function_name\",\n        \"arguments\": {\n            \"argument1\": value1,\n            \"argument2\": value2\n        }\n    }",
  "userPrompt": " \"更新免登嵌入报表的票据数量为10的票据值为\"abcd1234\"。\" ",
  "assistantResponse": "{\n    \"function\": \"UpdateTicketNum\",\n    \"arguments\": [\n        {\n            \"Ticket\": \"abcd1234\",\n            \"TicketNum\": 10\n        }\n    ]\n}"
}
```

## Where it came from

Built and released by the Deepexi organization. The dataset card states only that it covers "700+" public Alibaba Cloud OpenAPI operations across DataWorks, EMR, DataLake, MaxCompute, Hologres, Realtime Compute for Apache Flink, Quick BI and DTS, packaged so each row presents a candidate-function list, a user question, and the correct function call [1]. The card names no annotators, no generating model, and no collection process for the `userPrompt` questions or for how distractor functions were assigned to each row.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Deepexi/function-calling-small dataset card (README). https://huggingface.co/datasets/Deepexi/function-calling-small/raw/main/README.md - content description, product list, sample format, field descriptions, stated use cases. Fetched 2026-08-11.

[2] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Deepexi%2Ffunction-calling-small&config=default&split=train - 48 rows returned at offset 0; used for the sample row and the userPrompt/assistantResponse duplication count. Fetched 2026-08-11.

[3] Hugging Face Hub API record for Deepexi/function-calling-small. https://huggingface.co/api/datasets/Deepexi/function-calling-small?full=true and https://huggingface.co/api/datasets/Deepexi/function-calling-small?expand[]=downloadsAllTime - licence, gate status, sha, siblings, downloads, likes, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Deepexi%2Ffunction-calling-small - this endpoint takes no revision parameter, so the row and byte counts it returns are live, not covered by the Load it revision pin. Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Deepexi%2Ffunction-calling-small - schema, builder_name, download checksums naming the source CSV file; this endpoint also takes no revision parameter and is live, not pinned. Fetched 2026-08-11.

[6] The corpus screening row for `Deepexi/function-calling-small`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[7] Deepexi/openai-formate-function-calling-small dataset card (README). https://huggingface.co/datasets/Deepexi/openai-formate-function-calling-small/raw/main/README.md Fetched 2026-08-11.

[8] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=Deepexi%2Fopenai-formate-function-calling-small - this endpoint takes no revision parameter, so this row count is live, not pinned. Fetched 2026-08-11.

[9] Hugging Face Hub API listing of datasets by the Deepexi organization. https://huggingface.co/api/datasets?author=Deepexi&limit=100 - lists `Deepexi/glaive-function-calling-vicuna` and its description, converting `glaiveai/glaive-function-calling` to an SFT format with `id` and `conversations` fields. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as function-calling SFT data, with a caveat the card above already establishes: the 24,608 rows are not 24,608 independent tasks. Reading the first 48 served rows found only 12 distinct question/answer pairs recurring under different distractor-function lists, so a trainer sampling this dataset should expect much lower template diversity than the row count suggests [2]. The dataset carries no stated evaluation-set overlap risk and no eval split to hold out.

### The screening row

The row's own note [6]: "Chinese function selection over 700+ real Alibaba Cloud OpenAPI specs; the API docs are human-written, how the questions were produced is not stated." The row carries no flag.
