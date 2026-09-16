# m-a-p/CodeFeedback-Filtered-Instruction

156,526 single-turn code-instruction pairs (`query`/`answer`), complexity-filtered by an LLM judge from four open-source code-instruction pools and spanning many programming languages per row - the seed query set behind the OpenCodeInterpreter project's multi-turn Code-Feedback data.

**m-a-p/CodeFeedback-Filtered-Instruction** was built and released by the OpenCodeInterpreter team and introduced in "OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement" [1] (the dataset's own README links the paper under a different anchor title, "OpenCodeInterpreter: A System for Enhanced Code Generation and Execution" [2], but the arXiv page's own metadata carries the title cited here) as the intermediate single-turn query pool behind their multi-turn `Code-Feedback` dataset. The paper reports aggregating 287k coding queries from four upstream sets - Magicoder-OSS-Instruct, the Python subset of ShareGPT, Magicoder-Evol-Instruct, and Evol-Instruct-Code - then scoring each query and its response for complexity (1-5) with Qwen-72B-Chat and keeping only scores of 4 or 5, across two distinct filtering prompts, leaving 156k queries [1][2]. The card retains the original responses "to provide users with more convenient usage options" even though the paper's own downstream construction used only the queries [2]. **Two of the four upstream pools carry verbatim HumanEval prompts: a live search of the served rows found 85 rows containing the string "fibfib" (HumanEval/163's `even_fibfib`), all labeled `resource="wizardcoder"` (the Evol-Instruct-Code pool), and the corpus screening record separately counts 30 rows containing "separate_paren_groups" (HumanEval/2's helper) [3][4]. Anyone scoring against HumanEval or HumanEval+ must decontaminate against these two sources before using this set.** The card also warns the answers include OpenAI-model output subject to OpenAI's usage policy [2]. It lives at https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction .

**Use it for**: single-turn SFT on coding instructions (`query` as prompt, `answer` as target) - the SFT method card's shape. **Decontaminate against HumanEval/HumanEval+ before any scored run** [3][4]. `query`/`answer` map directly to a prompt/completion SFT format; no reformatting is needed for that shape.

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`), ungated (`gated` is `false`) [5]. The one catch: the answers include OpenAI-model output subject to OpenAI's usage policy, a use restriction layered on top of the Apache-2.0 grant [2].

**Shape**: 156,526 rows, one config (`default`), one split `train`, four string columns (`query`, `answer`, `resource`, `lang`) [6][7].

**Hold out**: no split to hold out (single `train` split) [7], but hold out or filter the 85 `fibfib` rows (resource `wizardcoder`) and the flagged `separate_paren_groups` rows before evaluating on HumanEval/HumanEval+, per the screening flag [3][4].

**Origin**: built by the m-a-p / OpenCodeInterpreter team; queries are aggregated from four open human/LLM-authored instruction sets and answers are the responses shipped in those same upstream sets, filtered (not generated) by Qwen-72B-Chat [1][2]. Hub API at the check date: `downloads` 18,970, `downloadsAllTime` 132,534, `likes` 208 [5][8].

**Trained-on-by**: `dphn/dolphin-2.9-llama3-8b` lists `m-a-p/CodeFeedback-Filtered-Instruction` directly in its training-data set and loads a local copy of this file (`m-a-p_CodeFeedback-Filtered-Instruction-sharegpt-unfiltered.jsonl`) in its axolotl config [9]. `dphn/Dolphin3.0-Llama3.1-8B` also carries this dataset's tag on the Hub, though its README/config was not fetched, so only the tag (not the axolotl-config detail) is confirmed for it [9]. abacusai's Liberated-Qwen1.5-72B/14B models likewise list this dataset (alongside the multi-turn `m-a-p/Code-Feedback`) among their training sources [10]. The paper's own OpenCodeInterpreter models are not shown training on this file directly: they fine-tune on the multi-turn `Code-Feedback` set (built from this pool's queries) blended 2:1 with a separate single-turn "WizardCoder 110k" set [1].

**Introduced by**: [1] (Xiang Yue et al., OpenCodeInterpreter paper); no separate blog post found.

## Shape

Rows and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 156,526 |

One config, `default`, four columns (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `query` | string |
| `answer` | string |
| `resource` | string |
| `lang` | string |

`resource` is a 4-valued label naming which upstream pool a row came from; the datasets-server column statistics over all 156,526 rows give: `wizardcoder` 62,851, `evolinstruct` 48,187, `magicoder` 29,623, `sharegpt` 15,865 [11]. `lang` is free text naming the code language of the row; over the full split its length histogram (character count) has a median of 6 and max of 16, consistent with short language tags like `python` or `csharp` rather than prose [11]. A sample of 1,000 rows read from `train` at ten 100-row offsets spanning the split (0, 20000, 40000, 50000, 60000, 80000, 100000, 120000, 140000, 150000) turned up 27 distinct `lang` values, led by `python` (715 of the 1,000) with the rest split across `sql`, `javascript`, `java`, `cpp`, `php`, `html`, `shell`, `csharp`, `swift`, `rust`, `typescript` and 15 further values each appearing 8 or fewer times; no source enumerates the full set of distinct `lang` values over all 156,526 rows, so more may exist outside this sample [12]. No source maps these four `resource` strings one-to-one onto the four named upstream sets (Magicoder-OSS-Instruct, Magicoder-Evol-Instruct, Evol-Instruct-Code, Python ShareGPT); this card's own HumanEval search (below) found all 85 `fibfib`-matching rows carry `resource="wizardcoder"`, consistent with the flagged Evol-Instruct-Code pool, but that is this card's inference from one query, not a stated mapping [2][4].

No source states sequence-length or token statistics for this release. The datasets-server column statistics do give *character*-length histograms over all 156,526 rows: `query` mean 722.7 characters (min 22, max 16,445, median 540), `answer` mean 1,521.3 characters (min 0, max 11,353, median 1,396) [11].

Byte sizes (datasets-server `/size`) [6]: 371,228,314 bytes of original JSON download, 175,028,271 bytes as Parquet, 356,431,585 bytes decoded in memory.

## Quality

- The complexity label on each row is an LLM judgment (Qwen-72B-Chat, scores 1-5, only 4-5 kept), not a human annotation; the paper reports repeating the filtering with two distinct prompts to make the selection more robust, but states no inter-prompt agreement rate [1].
- No source states a measured duplicate rate for this release.
- Contamination: this card's own live search against the dataset's served rows found 85 rows (all `resource="wizardcoder"`) containing the string "fibfib", matching HumanEval/163's `even_fibfib` function [3]. The corpus screening record for this dataset separately flags 30 rows containing "separate_paren_groups", matching HumanEval/2's helper function, drawn from the Evol-Instruct-Code and Magicoder-Evol upstream pools [4]. Neither this card's search nor the served dataset's full-text search index reliably isolates a single "separate_paren_groups" substring count independently (the served search index tokenizes on underscores and returns thousands of partial-word matches), so the 30-row figure is reported as the screening record's own finding, not independently re-derived here [4].
- The card's only stated content-safety note is the OpenAI usage-policy warning: "The dataset contains part data generated by OpenAI's language models, please pay attention to OpenAI's usage policy when adopting this dataset" [2].

## Load it

Single split, no held-out portion; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-02-26) [5]:

```python
import datasets

REV = "a08c213a9748c66c15d0225814be80a2e77adf4a"  # main at the check date
train = datasets.load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", revision=REV, split="train")  # 156,526 rows
```

**Trap**: the repository ships one JSON file (`CodeFeedback-Filtered-Instruction.jsonl`) with no explicit train/test split defined by the builder - `datasets` auto-assigns everything to a single `train` split, so there is no held-out portion to evaluate against; and the `answer` column is the raw upstream response, not something the paper's own OpenCodeInterpreter models trained on directly, so treat it as SFT target data only if the answer quality (unaudited by this card beyond the LLM complexity filter) is separately checked for your use case [1][2]. The revision pin above guarantees only what `load_dataset` returns; the Shape, Quality, and A-row figures on this card come from the datasets-server `/size`, `/info`, `/statistics`, and `/first-rows` endpoints, none of which accept a `revision` parameter (confirmed by diffing `/size` output with and without one) - those numbers are live as of the check date, not pinned to the revision above [6][7][11][13].

## Neighbors

- `m-a-p/Code-Feedback` - the multi-turn successor from the same paper and team: 66,383 rows (paper reports "68K") of `id`/`messages` conversations built by simulating execution and feedback turns from this pool's queries [1][14]. Different training shape (multi-turn chat messages vs. single-turn query/answer) - not a duplicate of this release, and the corpus does not need to choose between them for the same training run.
- `Leon-Leee/Code-Feedback-decontamination` - a decontaminated fork of `m-a-p/Code-Feedback` (66,369 rows, 14 fewer), removing rows matching only two specific code snippets (a GCD loop and a `sum_to_n` one-liner) via the Magicoder/bigcode substring-match decontamination process against HumanEval, MBPP, APPS, GSM8K and DS-1000 [15]. It decontaminates the multi-turn sibling, not this single-turn release, and its own README does not mention the `fibfib` or `separate_paren_groups` HumanEval matches flagged above, so it does not resolve this dataset's contamination risk [4][15].
- `LimYeri/CodeFeedback-Filtered-Instruction-Python` - a Python-only reshaping of this dataset with 104,848 rows and 5 columns, read live at the check date [16].
- `dlface/CodeFeedback-Filtered-Instruction-Chinese` (155,258 rows, 3 columns) and `nayohan/CodeFeedback-Filtered-Instruction-ko` (156,526 rows, 4 columns) - machine-translated Chinese and Korean renderings of this release, row counts read live at the check date [16].
- CoIR's `CoIR-Retrieval/codefeedback-st-queries-corpus` / `codefeedback-st-qrels` (and `-mt-` variants) - this dataset repurposed as a code-retrieval benchmark (queries/corpus/qrels), a different task shape entirely [16].

## A row

One config, one split, one served shape. From `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [13]:

```json
{
  "query": "Create a nested loop to print every combination of numbers between 0-9, excluding any combination that contains the number 5. Additionally, exclude any combination that contains a repeating digit. Implement the solution without using any built-in functions or libraries to check for repeating digits.",
  "answer": "Here is an example of a nested loop in Python to print every combination of numbers between 0-9, excluding any combination that contains the number 5 or repeating digits:\n\n```python\nfor i in range(10):  # First digit\n    for j in range(10):  # Second digit\n        for k in range(10):  # Third digit\n            # Checking for the conditions\n            if i != 5 and j != 5 and k != 5 and i != j and i != k and j != k:\n                print(i, j, k)\n```\n\nThis code will generate and print every combination of three digits between 0-9 that do not contain [...]",
  "resource": "evolinstruct",
  "lang": "python"
}
```

## Where it came from

Built by the m-a-p / OpenCodeInterpreter team. Coding queries were aggregated from four open-source code-instruction sets - Magicoder-OSS-Instruct, the Python subset of ShareGPT, Magicoder-Evol-Instruct, and Evol-Instruct-Code - totalling 287k queries; each query and its paired response, still in the compiled pool, was scored 1-5 for complexity by the open-source chat model Qwen-72B-Chat, run under two distinct prompts for robustness, and only scores of 4 or 5 were kept, yielding the 156k rows served here [1][2]. The `answer` column is the response as shipped in the corresponding upstream set (not generated fresh for this release); the card notes some of those upstream responses were themselves produced by OpenAI models [2]. This pool's queries (without their responses) were subsequently used to build the sibling multi-turn `Code-Feedback` dataset via three methods - Single-turn Packing, Interaction Simulation, and Code Correction - described in the paper [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Yue et al., "OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement", 2024. https://arxiv.org/abs/2402.14658 - the origin paper; current title read from the live abs page's `citation_title` metadata. 287k/156k query counts, Qwen-72B-Chat filtering process, Code-Feedback construction, and OpenCodeInterpreter training-data blend. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2402.14658). Fetched 2026-08-11.

[2] m-a-p/CodeFeedback-Filtered-Instruction dataset card (README). https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction/raw/main/README.md - dataset description, four upstream sources, retained-responses note, OpenAI usage-policy warning. Fetched 2026-08-11.

[3] datasets-server full-text search endpoint, query "fibfib". https://datasets-server.huggingface.co/search?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction&config=default&split=train&query=fibfib - 85 total matching rows, all `resource="wizardcoder"`. Fetched 2026-08-11.

[4] The corpus screening row for `m-a-p/CodeFeedback-Filtered-Instruction`, supplied with this card's request - its `flag` ("rule-risk: humaneval, humanevalplus - Its Evol-Instruct-Code and Magicoder-Evol sources carry HumanEval prompts: 85 rows `fibfib`, 30 `separate_paren_groups`") and `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[5] Hugging Face Hub API record for m-a-p/CodeFeedback-Filtered-Instruction. https://huggingface.co/api/datasets/m-a-p/CodeFeedback-Filtered-Instruction?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction - this endpoint takes no revision parameter (confirmed by diffing its output with and without one appended), so these counts are live, not pinned. Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction - takes no revision parameter; live, not pinned. Fetched 2026-08-11.

[8] Hugging Face Hub API record, `downloadsAllTime` variant. https://huggingface.co/api/datasets/m-a-p/CodeFeedback-Filtered-Instruction?expand[]=downloadsAllTime Fetched 2026-08-11.

[9] `dphn/dolphin-2.9-llama3-8b` dataset card (README), listing `m-a-p/CodeFeedback-Filtered-Instruction` in its training-data list and axolotl config path `m-a-p_CodeFeedback-Filtered-Instruction-sharegpt-unfiltered.jsonl`. https://huggingface.co/dphn/dolphin-2.9-llama3-8b/raw/main/README.md - found via the Hub API's dataset-tag model filter (https://huggingface.co/api/models?filter=dataset:m-a-p/CodeFeedback-Filtered-Instruction). Fetched 2026-08-11.

[10] `abacusai/Liberated-Qwen1.5-72B` dataset card (README), listing both `m-a-p/Code-Feedback` and `m-a-p/CodeFeedback-Filtered-Instruction` among its training datasets. https://huggingface.co/abacusai/Liberated-Qwen1.5-72B/raw/main/README.md Fetched 2026-08-11.

[11] datasets-server statistics endpoint. https://datasets-server.huggingface.co/statistics?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction&config=default&split=train - `resource` value frequencies and `query`/`answer`/`lang` character-length statistics over all 156,526 rows; takes no revision parameter, so these are live, not pinned. Fetched 2026-08-11.

[12] datasets-server rows endpoint, read at ten 100-row offsets across `train` (0, 20000, 40000, 50000, 60000, 80000, 100000, 120000, 140000, 150000), 1,000 rows total. https://datasets-server.huggingface.co/rows?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction&config=default&split=train&offset=<n>&length=100 - `lang` value frequencies over this sample; takes no revision parameter, so live, not pinned. Fetched 2026-08-11.

[13] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction&config=default&split=train - takes no revision parameter; live, not pinned. Fetched 2026-08-11.

[14] datasets-server size and info endpoints for the sibling dataset. https://datasets-server.huggingface.co/size?dataset=m-a-p%2FCode-Feedback and https://datasets-server.huggingface.co/info?dataset=m-a-p%2FCode-Feedback - row count and `id`/`messages` schema. Fetched 2026-08-11.

[15] `Leon-Leee/Code-Feedback-decontamination` dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/Leon-Leee/Code-Feedback-decontamination/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=Leon-Leee%2FCode-Feedback-decontamination Fetched 2026-08-11.

[16] datasets-server size endpoint, one call per neighbor: `LimYeri/CodeFeedback-Filtered-Instruction-Python`, `dlface/CodeFeedback-Filtered-Instruction-Chinese`, `nayohan/CodeFeedback-Filtered-Instruction-ko`; and a Hub dataset search for "CodeFeedback" listing the CoIR retrieval-benchmark repositories. https://datasets-server.huggingface.co/size?dataset=<id> and https://huggingface.co/api/datasets?search=CodeFeedback - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for single-turn SFT, with a mandatory decontamination step before any HumanEval/HumanEval+ scored run. Two facts decide it, both established above: the corpus screening row's flag identifies specific rule-risk contamination (HumanEval, HumanEval+) with row counts this card confirmed in part via a live search of the served data [3][4], and no other restriction (licence, usage-shape) blocks SFT use of the `query`/`answer` pairs [2][5].

### The screening row

The row's own note [4]: "156k coding queries lifted from Magicoder-OSS-Instruct, Magicoder-Evol-Instruct, Evol-Instruct-Code and ShareGPT-python, then filtered by Qwen-72B-Chat; the card warns part of it is OpenAI-model output." Its flag [4]: "rule-risk: humaneval, humanevalplus - Its Evol-Instruct-Code and Magicoder-Evol sources carry HumanEval prompts: 85 rows `fibfib`, 30 `separate_paren_groups`."
