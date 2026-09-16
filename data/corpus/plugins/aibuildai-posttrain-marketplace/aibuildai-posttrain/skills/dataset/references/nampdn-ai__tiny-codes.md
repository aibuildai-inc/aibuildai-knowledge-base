# nampdn-ai/tiny-codes

Roughly 1.63 million short, heavily commented synthetic code snippets paired with natural-language instructions, spanning thirteen programming and database languages, gated behind an auto-accept licence click.

**nampdn-ai/tiny-codes** is a synthetic instruction/code dataset released by nampdn-ai, with no dedicated origin paper - the dataset card frames the release as inspired by "Textbooks Are All You Need" [1], the paper behind the phi-1 line of models trained on textbook-quality synthetic data, and by "The Magic of IF: Investigating Causal Reasoning Abilities in Large Language Models of Code" [2]. The card states each example asks a model to produce a code snippet in one of Python, TypeScript, JavaScript, Ruby, Julia, Rust, C++, Bash, Java, C#, Go, Cypher, or SQL, with `if`/`else` control flow deliberately emphasized "to foster the development of effective reasoning skills" [3]. **The card explicitly restricts the intended use: "this dataset is not intended for code-generation purposes, it's intended to boost the reasoning capability of model via logic code"** [3]. **The repository is gated with auto-accept: every unauthenticated request this card's sources tried - the README, the datasets-server info/size/first-rows endpoints, and the parquet resolve/API URLs - returned a 401/404 "GatedRepo" or "not accessible without authentication" error, so native columns and rows below come from two ungated same-content mirrors, not the gated repo itself** [4][5]. It lives at https://huggingface.co/datasets/nampdn-ai/tiny-codes .

**Use it for**: reasoning-trace SFT on `instruction`-`output` pairs that walk through commented, branching code - not code-generation training, per the card's own restriction above [3]. The native shape is a single instruction/response pair per row, not a chat-templated dialogue (no chat dialect is declared on the repo). This maps to the SFT method card's instruction-tuning format once the columns are renamed to that card's prompt/response keys.

**Licence**: MIT (`license: mit` in the card's front matter and in the Hub API's `cardData`) [3][5], gated with auto-approve (`"gated": "auto"`) [5]. The catch: auto-approve still requires an authenticated, gate-accepted request - every anonymous fetch in this session was refused, so a pipeline must attach a Hub token with the gate accepted before it can read anything past the README.

**Shape**: one file tree of nine `.parquet` parts and a README, 981,394,439 bytes of LFS parquet on the Hub tree listing [6]; no split or config name is confirmed from the gated repo, but two independent ungated mirrors that preserve the byte-identical row 0 both report a single `train` split of 1,632,309 rows in one `default` config [7][8]. See Shape below for the discrepancy against the file-naming count.

**Hold out**: nothing. No source read for this card - the card's own README, the two schema-preserving mirrors, or the adopter model cards - states or flags any evaluation-set overlap risk for this release.

**Origin**: built by nampdn-ai; the content is model-generated (synthetic instruction/response pairs), but no source read for this card names which model generated it [3]. Hub API at the check date: `downloads` 2,093, `downloadsAllTime` 31,815, `likes` 299 [5][9].

**Trained-on-by**: the Hub's model-search-by-dataset-tag lists 46 models tagged as trained on `nampdn-ai/tiny-codes` [10]; two with their own README confirming it directly: `daekeun-ml/phi-2-ko-v0.1`, a Korean phi-2 continual-pretrain that lists the tiny-code dataset as a training source [11], and `monsterapi/llama2-code-generation`, whose card states the team "finetuned Llama 2 7B model from Meta on nampdn-ai/tiny-codes for ~ 10,000 steps" [12].

**Introduced by**: no paper - the dataset card [3]; inspired by "Textbooks Are All You Need" [1] and "The Magic of IF" [2].

## Shape

The gated repo's own tree listing shows nine parquet parts plus the README, no `dataset_info` block, and file names that follow a cumulative-count convention - `part_1_200000.parquet`, `part_2_400000.parquet`, ..., `part_9_1632520.parquet` - implying 1,632,520 total rows, but that number is read off the file names, not stated as a row count anywhere on the card [6]:

| file | bytes |
| --- | --- |
| `part_1_200000.parquet` | 120,240,865 |
| `part_2_400000.parquet` | 120,181,361 |
| `part_3_600000.parquet` | 120,140,572 |
| `part_4_800000.parquet` | 120,109,996 |
| `part_5_1000000.parquet` | 120,211,926 |
| `part_6_1200000.parquet` | 120,347,549 |
| `part_7_1400000.parquet` | 120,140,121 |
| `part_8_1600000.parquet` | 120,447,788 |
| `part_9_1632520.parquet` | 19,574,261 |
| total | 981,394,439 |

Because the gated repo's own datasets-server `/info` and `/size` endpoints return 404 without authentication [4], this card confirms shape through two ungated repos that carry the byte-identical row 0 text (verified below): `TinyPixel/tiny-codes`, single `text` column, `train` split, 1,632,309 rows, 3,806,871,049 bytes in memory [7]; and `hubistrauss/tiny-codes-instruct`, twelve columns, `train` split, 1,632,309 rows, 3,908,392,508 bytes in memory [8]. Both mirrors agree with each other on the row count but read 211 rows fewer than the 1,632,520 the original file names imply - not stated why. `hubistrauss/tiny-codes-instruct`'s columns, all large-string except `idx` (int64) and `input` (string) [8]:

| column | content (from the row below) |
| --- | --- |
| `instruction` | the full natural-language coding prompt |
| `main_topic` | e.g. "Sneezing and coughing etiquette" |
| `subtopic` | e.g. "Preventing Spread of Germs" |
| `adjective` | e.g. "High" |
| `action_verb` | e.g. "Determine" |
| `scenario` | e.g. "for Engineer" |
| `target_audience` | e.g. "Experts" |
| `programming_language` | e.g. "Python" |
| `common_sense_topic` | e.g. "Bias" |
| `idx` | integer row index into the original build, e.g. 1230929 |
| `output` | the commented code answer |
| `input` | empty string in the sampled rows (see A row) |

No source read for this card states token or sequence-length statistics for any split.

## Quality

- No source read for this card states a measured contamination rate, duplicate rate, or human-review pass for the generated pairs.
- The card's only stated quality signal is a design intent, not a measurement: examples are "carefully written and commented to ensure maximum readability" and use `if`/`else` branching "to foster the development of effective reasoning skills" [3].
- The card's only stated caveat is the scope restriction quoted above - not for code-generation training [3].
- `input` is an empty string in every one of the five rows read from `hubistrauss/tiny-codes-instruct` (offsets 0-4) [8]; whether that holds for the rest of the 1,632,309 rows is not established by this five-row read.

## Load it

The repo itself is gated with auto-accept, so `load_dataset` needs a Hub token with the gate accepted; the commit this card's numbers were read at is the shortlist's pinned `sha`, `9aebe5ee8b406356d5f5f2d603bc0a1684ee8ce7` [5]:

```python
import datasets

REV = "9aebe5ee8b406356d5f5f2d603bc0a1684ee8ce7"  # main at the check date
ds = datasets.load_dataset("nampdn-ai/tiny-codes", revision=REV, token=True)  # requires an accepted gate
```

**Trap**: without `token=True` (or `HF_TOKEN` set) and without having clicked through the gate on the Hub page first, every request this card's session made - README, `/info`, `/size`, `/first-rows`, the parquet resolve URLs, and the Hub `/parquet` API URLs - returned `401 GatedRepo` or a 404 "not accessible without authentication" error [4]; there is no anonymous path to this repo's rows.

## Neighbors

Every row count below was read live at the check date. `hubistrauss/tiny-codes-instruct` and `layoric/tiny-codes-alpaca` reproduce the twelve-column native schema and the 1,632,309-row count, ungated [8][13]; `TinyPixel/tiny-codes` collapses `instruction`+`output` into one `text` field wrapped in a `<|im_start|>USER:` / `<|im_start|>ASSISTANT:` ChatML template, same 1,632,309 rows [7]; `HydraLM/tiny-codes-standardized` splits each pair into separate `message`/`message_type`/`conversation_id` rows, 3,264,618 rows (roughly double, one row per turn) [14]; `mlfoundations-dev/tiny_codes` is a 50,000-row filtered subset with `instruction_seed`/`response_seed`/`response`/`conversations` columns, no stated relationship to the full set [15]. This corpus's shortlist row points at the original `nampdn-ai/tiny-codes`; use one of the schema-preserving mirrors (`hubistrauss/tiny-codes-instruct` or `layoric/tiny-codes-alpaca`) only if the gate cannot be accepted, and do not combine a mirror with the original - they are the same rows.

## A row

The gated repo serves no row directly; this is `hubistrauss/tiny-codes-instruct`, `config="default"`, `split="train"`, `row_idx=0` [8], confirmed byte-identical in content to row 0 of `TinyPixel/tiny-codes` [7]:

```json
{
  "instruction": "Develop a Python program snippet to Determine High Sneezing and coughing etiquette: Preventing Spread of Germs for Engineer for Experts. Incorporate if/else or switch/case statements to handle various cases related to the Bias. Dry-run, ensure your control flow logic is clear and well-commented.",
  "main_topic": "Sneezing and coughing etiquette",
  "subtopic": "Preventing Spread of Germs",
  "adjective": "High",
  "action_verb": "Determine",
  "scenario": "for Engineer",
  "target_audience": "Experts",
  "programming_language": "Python",
  "common_sense_topic": "Bias",
  "idx": 1230929,
  "output": "Here is python code which determines high sneezing and coughing etiquette based upon different conditions : [...]",
  "input": ""
}
```

## Where it came from

Built and released by nampdn-ai as a synthetic dataset: the card states it is a collection of model-generated code snippets meant to teach reasoning through `if`/`else` branching across natural and programming languages, but does not name the generating model [3]. The card frames the construction as inspired by two external works rather than derived from a stated upstream corpus: "Textbooks Are All You Need" [1], which the card credits with showing that high-quality textbook-like training data improves code-task performance, and "The Magic of IF" [2], a study of causal reasoning in code-trained LLMs. The card also lists this release as the first entry in the author's own "Tiny Series" of small, synthetic, textbook-quality datasets, alongside `tiny-textbooks`, `tiny-orca-textbooks`, `tiny-webtext`, `tiny-lessons`, and `tiny-bridgedict`, and credits "TinyStories" [16] as the paper that "sparked" the series [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. The Hub repo is mutable and gated (a repo can be force-pushed, and gate/auth state can change what an unauthenticated request returns), which is why Load it pins the revision and the Shape section states which fetches were blocked rather than guessing past them.

[1] "Textbooks Are All You Need", 2023. https://arxiv.org/abs/2306.11644 - current title read from the live abs page; cited by the dataset card as the release's inspiration. Fetched 2026-08-11.

[2] "The Magic of IF: Investigating Causal Reasoning Abilities in Large Language Models of Code", ACL Findings 2023. https://aclanthology.org/2023.findings-acl.574/ - current title read from the live ACL Anthology page; no arXiv record found for this paper. Fetched 2026-08-11.

[3] nampdn-ai/tiny-codes dataset card (README), read via the resolve endpoint since the `raw` endpoint is gate-blocked. https://huggingface.co/datasets/nampdn-ai/tiny-codes/resolve/main/README.md - description, use restriction, Tiny Series list. Fetched 2026-08-11.

[4] Direct fetch attempts against the gated repo without authentication: `raw/main/README.md` (returns an access-restricted plain-text message), `datasets-server` `/info`, `/size`, `/splits`, `/first-rows` (each returns a 404 JSON error naming the dataset as private/gated), and the parquet resolve and `/api/datasets/.../parquet/...` URLs (each returns HTTP 401 with `x-error-code: GatedRepo`). Fetched 2026-08-11.

[5] Hugging Face Hub API record for nampdn-ai/tiny-codes. https://huggingface.co/api/datasets/nampdn-ai/tiny-codes?full=true - `sha`, `gated`, `downloads`, `likes`, `lastModified`, `cardData.license`, tree `siblings`. Fetched 2026-08-11.

[6] Hugging Face Hub tree API for nampdn-ai/tiny-codes. https://huggingface.co/api/datasets/nampdn-ai/tiny-codes/tree/main - file names and byte sizes; this endpoint is not gate-blocked, unlike file content and datasets-server. Fetched 2026-08-11.

[7] datasets-server info and size endpoints for the ungated mirror TinyPixel/tiny-codes, plus its first-rows endpoint for row 0. https://datasets-server.huggingface.co/info?dataset=TinyPixel%2Ftiny-codes , https://datasets-server.huggingface.co/size?dataset=TinyPixel%2Ftiny-codes , https://datasets-server.huggingface.co/first-rows?dataset=TinyPixel%2Ftiny-codes&config=default&split=train Fetched 2026-08-11.

[8] datasets-server info endpoint and first-rows endpoint (offsets 0-4) for the ungated, schema-preserving mirror hubistrauss/tiny-codes-instruct. https://datasets-server.huggingface.co/info?dataset=hubistrauss%2Ftiny-codes-instruct , https://datasets-server.huggingface.co/first-rows?dataset=hubistrauss%2Ftiny-codes-instruct&config=default&split=train Fetched 2026-08-11.

[9] Hugging Face Hub API record for nampdn-ai/tiny-codes with `expand[]=downloadsAllTime`. https://huggingface.co/api/datasets/nampdn-ai/tiny-codes?expand[]=downloadsAllTime Fetched 2026-08-11.

[10] Hugging Face Hub models-list API filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:nampdn-ai/tiny-codes&full=true&limit=50 - 46 models returned. Fetched 2026-08-11.

[11] daekeun-ml/phi-2-ko-v0.1 model card. https://huggingface.co/daekeun-ml/phi-2-ko-v0.1/raw/main/README.md - lists nampdn-ai/tiny-codes among training datasets. Fetched 2026-08-11.

[12] monsterapi/llama2-code-generation model card. https://huggingface.co/monsterapi/llama2-code-generation/raw/main/README.md - states the model was finetuned on nampdn-ai/tiny-codes. Fetched 2026-08-11.

[13] datasets-server info endpoint for the ungated mirror layoric/tiny-codes-alpaca. https://datasets-server.huggingface.co/info?dataset=layoric%2Ftiny-codes-alpaca Fetched 2026-08-11.

[14] datasets-server info endpoint for HydraLM/tiny-codes-standardized. https://datasets-server.huggingface.co/info?dataset=HydraLM%2Ftiny-codes-standardized Fetched 2026-08-11.

[15] datasets-server info endpoint and dataset card for mlfoundations-dev/tiny_codes. https://datasets-server.huggingface.co/info?dataset=mlfoundations-dev%2Ftiny_codes , https://huggingface.co/datasets/mlfoundations-dev/tiny_codes/raw/main/README.md Fetched 2026-08-11.

[16] "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?", 2023. https://arxiv.org/abs/2305.07759 - current title read from the live abs page; cited by the dataset card as the inspiration for the author's Tiny Series. Fetched 2026-08-11.

[17] The corpus screening row for `nampdn-ai/tiny-codes`, supplied with this card's request. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data once the Hub gate is accepted: the card's own README restricts the pairs to reasoning training rather than code-generation training (quoted in the opening paragraph) [3], and the screening row's note describes the same content and flags the gate as the practical access barrier [17].

### The screening row

The row's own note: "1.6 million short synthetic code snippets paired with natural-language reasoning across Python, TypeScript, JavaScript, Ruby, Julia, Rust, C++, Bash, Java, C#, Go, Cypher and SQL; gated with auto-accept, so the viewer returns 404 until the gate is accepted." The row carries no separate flag field.
