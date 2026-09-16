# Digital-nimbus/llama-2-oai-function-calling

2,243 single-column text rows, each a full Llama-2-formatted conversation embedding an OpenAI-style tool list and a `[FUNC]...[/FUNC]` block, with no dataset card describing how they were produced.

**Digital-nimbus/llama-2-oai-function-calling** is a Hugging Face dataset repository at https://huggingface.co/datasets/Digital-nimbus/llama-2-oai-function-calling, under the `Digital-nimbus` account [1]. Its file tree holds `.gitattributes`, `README.md`, and one data file, `data/function_call_dataset.json`, a JSON array of 2,243 objects each holding one `text` string [2]. Each string opens with `<s>[FUNC]` followed by a Python-repr list of OpenAI-style `{'type': 'function', 'function': {...}}` tool specs, closes that block with `[/FUNC]`, then continues with a Llama-2-style `[INST]<<SYS>>...<</SYS>>...[/INST]` turn and an assistant reply, ending in `</s>` [3]. The repository's README contains only a `license: mit` front-matter block and no body text describing the source, generation process, or intended use [4]. **No source establishes who or what generated these conversations: the repository carries no card prose, no linked paper, and no named upstream seed set or organization trail, so the generator is unverified [5].**

**Use it for**: nothing, pending verification of origin - the dataset would otherwise fit reasoning-trace/tool-call SFT (each row is a complete instruction-formatted dialogue with an embedded tool call), and each row would need parsing into a chat-format record before use, but no source names the generating model or process, so it is disallowed on origin grounds rather than format grounds [5].

**Licence**: MIT (`cardData.license` is `"mit"`, and the repo carries the `license:mit` tag), ungated (`"gated": false`, `"private": false`) [1]. The one catch: the licence covers only the repository's own MIT grant, not any right to the underlying conversations or tool schemas, since no source states where those originated [1][4].

**Shape**: 2,243 rows in one config (`default`), a single split `train`, one string column (`text`) [6][7].

**Hold out**: not established - no source names a benchmark or evaluation set this dataset might overlap, and the flag on this row states that a BFCL (Berkeley Function-Calling Leaderboard) overlap probe found zero overlap [8].

**Origin**: unknown. No card, paper, or organization page names a builder, a generating model, or a human-annotation process for the conversations or tool-call blocks [4][5]. Hub API at the check date: `downloads` 74 (`downloadsAllTime` 933), `likes` 3 [1].

**Trained-on-by**: none found - no source names a model or training recipe that used this dataset.

**Introduced by**: no paper - the dataset repository itself carries no card body beyond the MIT licence front matter [4].

## Shape

Rows served and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 2,243 |
| total | 2,243 |

One config, `default`, with one column (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `text` | string |

No source states sequence-length or token statistics for this release; "not stated."

## Quality

- No source states a measured contamination rate, duplicate rate, or annotation-agreement figure for this dataset.
- The repository's README has no body text at all beyond the MIT licence front matter (21 bytes total), so it makes no quality claim of any kind [4].
- Reading the first five served rows and the last three rows (offset 2240) shows every row follows the same `<s>[FUNC][...][/FUNC][INST]<<SYS>>...<</SYS>>...[/INST]...</s>` shape, with exactly one `<s>` and one `</s>` marker per row; row lengths in this small sample range from about 3,650 to about 5,366 characters [3][9]. This is a read of 8 of 2,243 rows and does not establish a rate across the split.
- The flag on this shortlist row states that a BFCL (Berkeley Function-Calling Leaderboard) probe found zero overlap between this dataset and BFCL, and that the generator remains unverified, so the row stays disallowed [5].

## Load it

```python
import datasets

REV = "69e08545a668fa692ad5d66e4a89a4a4f6b5fb9a"  # main at the check date
train = datasets.load_dataset("Digital-nimbus/llama-2-oai-function-calling", revision=REV, split="train")  # 2,243 rows
```

**Trap**: the single `text` column is a pre-formatted Llama-2 prompt-and-response string, not separate prompt/response or messages fields - a consumer must parse the `[FUNC]...[/FUNC]`, `[INST]...[/INST]`, and trailing-reply segments out of each string before it can be reshaped into a chat-message or tool-call training record [3]. The repository's only data file, `data/function_call_dataset.json`, is a plain JSON array (not JSON Lines), so it must be loaded as one array of 2,243 objects rather than streamed line by line if read outside `datasets` [2].

## Neighbors

None found - no source names a sibling, cleaned, binarized, or successor release from this builder or elsewhere carrying the same rows.

## A row

The repository serves one config and one schema, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3], with the function-list and reply body truncated:

```json
{
  "text": "<s>[FUNC][{'type': 'function', 'function': {'name': 'list_jira_issues', 'description': 'List Jira issues based on a specified JQL query', 'parameters': {'type': 'object', 'properties': {'jql_query': {'type': 'string', 'description': 'The JQL query string to search for issues'}}, 'required': ['jql_query']}}}, [...8 more tool specs, Jira and Confluence functions...]][/FUNC][INST] <<SYS>>Evaluate the user message and if applicable, choose a tool from the functions and parse the user input into arguments and respond with a json object containing the function name, and args.<</SYS>>Could you please assist in List Jira issues based on a specified JQL query?[/INST]Certainly! To assist you with listing Jira issues based on a specified JQL (Jira Query Language) query, please provide me with the JQL query you would like to use. Once I have the query, I'll be able to proceed with fetching the list of issues for you.</s>"
}
```

## Where it came from

Not established. The repository holds one data file, `data/function_call_dataset.json`, and a README limited to an MIT licence declaration; neither names a builder, a generating model, a human-annotation process, or an upstream pool [2][4]. The shortlist row's own flag states that this is "not establishable": the card carries only a licence field and no repository, paper, or organization trail naming a generator or seed set, and a probe for overlap with the BFCL benchmark found none, so the unknown generator leaves the dataset disallowed [5].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Hugging Face Hub API record for Digital-nimbus/llama-2-oai-function-calling. https://huggingface.co/api/datasets/Digital-nimbus/llama-2-oai-function-calling?full=true - licence, gate, `sha`, `downloads`, `likes`, `siblings`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[2] Repository file tree. https://huggingface.co/api/datasets/Digital-nimbus/llama-2-oai-function-calling/tree/main - lists `.gitattributes`, `README.md`, and `data/function_call_dataset.json`, the repository's only data-bearing file; that file, read directly at https://huggingface.co/datasets/Digital-nimbus/llama-2-oai-function-calling/raw/main/data/function_call_dataset.json, confirms a plain JSON array of 2,243 `{"text": ...}` objects. Fetched 2026-08-11.

[3] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Digital-nimbus%2Fllama-2-oai-function-calling&config=default&split=train Fetched 2026-08-11.

[4] Dataset card (README). https://huggingface.co/datasets/Digital-nimbus/llama-2-oai-function-calling/raw/main/README.md - 21 bytes total, `license: mit` front matter only, no body text. Fetched 2026-08-11.

[5] The corpus screening row for `Digital-nimbus/llama-2-oai-function-calling`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Digital-nimbus%2Fllama-2-oai-function-calling Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Digital-nimbus%2Fllama-2-oai-function-calling Fetched 2026-08-11.

[8] The corpus screening row's `flag` field for `Digital-nimbus/llama-2-oai-function-calling`, stating that a BFCL overlap probe found zero overlap. Checked 2026-08-11.

[9] datasets-server rows endpoint, offset 2240, length 3. https://datasets-server.huggingface.co/rows?dataset=Digital-nimbus%2Fllama-2-oai-function-calling&config=default&split=train&offset=2240&length=3 Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Disallowed. The dataset's own repository carries no card body, no paper, and no named generator or seed set beyond an MIT licence declaration, and the screening row's flag treats an unknown generator as grounds to keep the row disallowed even though a BFCL overlap probe found no overlap [4][5][8].

### The screening row

The row's note [5]: "2.2k Llama-2-formatted [FUNC] OpenAI-style tool-call strings; no card, no stated source." The row's flag [5]: "unverified: - Not establishable: licence-only card and no repo, paper or org trail naming a generator or seed set; BFCL probes zero overlap; an unknown generator stays disallowed."
