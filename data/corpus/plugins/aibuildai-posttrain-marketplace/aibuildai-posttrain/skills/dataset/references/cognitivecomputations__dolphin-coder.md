# cognitivecomputations/dolphin-coder

109,118 single-turn coding instruction/response triples - LeetCode-style code generation and cross-language code translation - built from a Kaggle problem collection, with no generating model named for the responses.

**dolphin-coder** was built by Eric Hartford's cognitivecomputations project to train the DolphinCoder models; its README states that it was built from the Kaggle dataset `erichartford/leetcode-rosetta` and used to train the dolphin-coder model [1]. No paper introduces it - the dataset card is the only source [1]. The repository ships two converter scripts and their outputs: `convertLeetcodeRosettaCodegen.py` turns each Kaggle problem into several templated coding-instruction rows (one per solution language, in one of four prompt styles: explain, justify, focus-on-an-aspect, or format-as-language), and `convertLeetcodeRosettaTranslate.py` turns the same problems into templated cross-language "translate this code" rows [2]. **The repository was renamed after this card's commit was read: the Hub now serves it at `QuixiAI/dolphin-coder`, and the old id `cognitivecomputations/dolphin-coder` (and its Hub API/datasets-server calls) resolve to it only via an HTTP redirect [3][4].** The dataset lives at https://huggingface.co/datasets/cognitivecomputations/dolphin-coder [5].

**Use it for**: single-turn SFT on code-generation and code-translation instructions, formatted as a system_prompt/question/response triple that needs conversion into a chat message list (system, user, assistant) before use - see the SFT method card. No source states a generating model for the `response` text; the responses originate in the upstream Kaggle collection, not from an LLM call made by this repository's own scripts [1][2].

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [3]. The upstream Kaggle source is also released under Apache 2.0 [6]; no source states any further catch.

**Shape**: one config (`default`), one split `train`, 109,118 rows, 3 string columns (`system_prompt`, `question`, `response`) [7][8].

**Hold out**: nothing. No source fetched for this card - the dataset card, the converter scripts, or the Kaggle source page - states or flags any overlap with an evaluation set, and the screening row carries no flag [1][2][6][9].

**Origin**: built by cognitivecomputations (Eric Hartford) from the Kaggle dataset `erichartford/leetcode-rosetta`, also credited to Eric Hartford [1][6]; no generating model is named, and the label origin (whether the Kaggle collection's explanations and per-language solutions are human-written or LLM-produced) is not stated by any source read for this card. Hub API at the check date: `downloads` 676, `downloadsAllTime` 19,349, `likes` 62 [3][10].

**Trained-on-by**: `dphn/dolphincoder-starcoder2-15b` and `dphn/dolphincoder-starcoder2-7b`, whose model cards list `cognitivecomputations/dolphin-coder` in their `datasets:` front matter alongside `cognitivecomputations/dolphin` and others, and describe training with qLoRA/Axolotl on the combined mix [11][12].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and columns, from the datasets-server `/info` and `/size` endpoints under the current repo id (`QuixiAI/dolphin-coder`) [7][8]:

| config | split | rows |
| --- | --- | --- |
| `default` | `train` | 109,118 |

| column | dtype |
| --- | --- |
| `system_prompt` | string |
| `question` | string |
| `response` | string |

The single `train` split concatenates the two source files listed in the repo tree: `dolphin-coder-codegen.jsonl` (236,748,284 bytes) and `dolphin-coder-translate.jsonl` (73,566,445 bytes), both matching the 3-column schema above [4]. No source states a per-file row split, nor any sequence-length or token statistic for this release; none is invented here.

## Quality

No source fetched for this card states a measured contamination rate, duplicate rate, or annotator-agreement figure. The only process detail available is structural: the codegen script emits one row per (problem, solution language) pair using one of four randomly chosen instruction templates, and optionally prepends a pseudocode block to the response; the translate script emits one row per randomly chosen (source language, target language) pair per problem, with a fully templated system prompt and instruction [2]. Both scripts read `pseudocode`, `content` (the problem statement), `explanation`, and `code` (a per-language solution map) from the upstream Kaggle file, so the substantive content - problem text, explanation, and code - is inherited unchanged from `erichartford/leetcode-rosetta` and is not itself described on the Kaggle page beyond its Apache-2.0 licence and author credit [2][6].

## Load it

The Hub redirects the shortlisted id/commit to the renamed repo; both the original id and the new id resolve to the same revision (`sha` "18607faa1a9698aa8d4b7683937fb94a0c2c7060" at the check date, matching `last_modified` 2023-12-07) [3][4]:

```python
import datasets

REV = "18607faa1a9698aa8d4b7683937fb94a0c2c7060"  # main at the check date
train = datasets.load_dataset("cognitivecomputations/dolphin-coder", revision=REV, split="train")  # 109,118 rows
```

**Trap**: the repo has been renamed to `QuixiAI/dolphin-coder` on the Hub; the datasets-server viewer and info/size endpoints return a "renamed" error for the old id and only answer under the new one, even though the old id still loads via redirect through `load_dataset` and the raw-file/API routes [3][4][7][8]. Do not treat the old id's 404 from a viewer call as "no data" - the underlying files are intact under the new id.

## Neighbors

A Hub author search for other cognitivecomputations coder-labeled datasets returned no results [13]. The same builder's `cognitivecomputations/dolphin` dataset (also renamed, now served at `QuixiAI/dolphin`) is not a coder-specific re-release: its two configs total 3,731,947 rows across three columns (`instruction`, `input`, `output`), built by augmenting FLANv2 with GPT-4 and GPT-3.5 completions [14]; a sampled row is a general story-continuation instruction, not a coding task [14]. Column names, row count, and content all differ from this repository's `system_prompt`/`question`/`response` LeetCode-derived rows, so the two do not overlap and do not compete as training data. None found beyond that.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0`, read via the datasets-server `/first-rows` endpoint under the current repo id [9], with long fields truncated with `[...]`:

```json
{
  "system_prompt": "You are a coding AI",
  "question": "Please write the following solution using c++:\nGiven an array of integers `nums` and an integer `target`, return _indices of the two numbers such that they add up to `target`_.\n\nYou may assume that each input would have **_exactly_ one solution**, and you may not use the _same_ element twice.\n\nYou can return the answer in any order.\n\n**Example 1:**\n\n**Input:** nums = \\[2,7,11,15\\], target = 9\n**Output:** \\[0,1\\]\n**Explanation:** Because nums\\[0\\] + nums\\[1\\] == 9, we return \\[0, 1\\].\n\n[...]\n\n**Constraints:**\n\n*   `2 <= nums.length <= 104`\n*   `-109 <= nums[i] <= 109`\n*   `-109 <= target <= 109`\n*   **Only one valid answer exists.**\n\n[...]",
  "response": "### Explanation\n The algorithm leverages a hash map (unordered_map in C++, HashMap in Java, dictionary in Python, and Map in JavaScript). It iterates through the given 'nums' array and calculates the complementary value (target - current value). [...]\n\n### Code\n```cpp\n#include <vector>\n#include <unordered_map>\n\nstd::vector<int> twoSum(std::vector<int>& nums, int target) {\n    std::unordered_map<int, int> map;\n    for (int i = 0; i < nums.size(); i++) {\n        int complement = target - nums[i];\n        if (map.find(complement) != map.end())\n            return {map[complement], i};\n        map[nums[i]] = i;\n    }\n    return {};\n}\n```"
}
```

A row further into the split (`row_idx` 109,000, from the translate half of the file) shows the same three columns filled with a code-translation instruction instead of a code-generation one - same schema, different task template [9].

## Where it came from

Built by cognitivecomputations (Eric Hartford), transformed from the Kaggle dataset `erichartford/leetcode-rosetta`, which the Kaggle page also credits to Eric Hartford and releases under Apache 2.0 [1][6]. The repository's two Python scripts perform the transformation: `convertLeetcodeRosettaCodegen.py` reads each Kaggle problem's `content`, `explanation`, `pseudocode`, and per-language `code` map, and emits one templated instruction/response row per solution language; `convertLeetcodeRosettaTranslate.py` emits one templated cross-language translation row per randomly paired source/target language, reusing the same `code` map [2]. Neither script calls an external model API; the underlying problem statements, explanations, pseudocode, and code solutions come from the Kaggle collection as-is, and no source states how that collection itself was produced [1][2][6].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable and, in this case, renameable (a repo id can change and the old id becomes a redirect), which is why Load it pins the revision and Where it came from records the rename.

[1] cognitivecomputations/dolphin-coder dataset card (README), read via redirect to the current repo id. https://huggingface.co/datasets/cognitivecomputations/dolphin-coder/raw/main/README.md - source attribution and stated use. Fetched 2026-08-12.

[2] Repository converter scripts, read via redirect to the current repo id. https://huggingface.co/datasets/cognitivecomputations/dolphin-coder/raw/main/convertLeetcodeRosettaCodegen.py and .../convertLeetcodeRosettaTranslate.py - prompt-template logic and source fields consumed. Fetched 2026-08-12.

[3] Hugging Face Hub API record, read via redirect to the current repo id. https://huggingface.co/api/datasets/cognitivecomputations/dolphin-coder?full=true - id, sha, licence, gate, downloads, likes, last-modified date, file list. Fetched 2026-08-12.

[4] Repository file tree, read via redirect to the current repo id. https://huggingface.co/api/datasets/cognitivecomputations/dolphin-coder/tree/main - per-file byte sizes for the two served JSONL files. Fetched 2026-08-12.

[5] The dataset's canonical Hub page. https://huggingface.co/datasets/cognitivecomputations/dolphin-coder - confirmed to return HTTP 200 (via redirect to the current repo id) at the check date. Fetched 2026-08-12.

[6] Kaggle dataset page for the upstream collection. https://www.kaggle.com/datasets/erichartford/leetcode-rosetta - author credit and Apache-2.0 licence statement. Fetched 2026-08-12.

[7] datasets-server info endpoint, under the current repo id (the old id returns a "renamed" error). https://datasets-server.huggingface.co/info?dataset=QuixiAI%2Fdolphin-coder - column schema. Fetched 2026-08-12.

[8] datasets-server size endpoint, under the current repo id. https://datasets-server.huggingface.co/size?dataset=QuixiAI%2Fdolphin-coder - row and byte counts. Fetched 2026-08-12.

[9] datasets-server first-rows and rows endpoints, under the current repo id. https://datasets-server.huggingface.co/first-rows?dataset=QuixiAI%2Fdolphin-coder&config=default&split=train and https://datasets-server.huggingface.co/rows?dataset=QuixiAI%2Fdolphin-coder&config=default&split=train&offset=109000&length=5 - sampled rows. Fetched 2026-08-12.

[10] Hugging Face Hub API record with `downloadsAllTime` expanded. https://huggingface.co/api/datasets/cognitivecomputations/dolphin-coder?expand[]=downloadsAllTime (redirects to the current repo id). Fetched 2026-08-12.

[11] dphn/dolphincoder-starcoder2-15b model card. https://huggingface.co/dphn/dolphincoder-starcoder2-15b/raw/main/README.md - `datasets:` front matter listing `cognitivecomputations/dolphin-coder`, training description. Fetched 2026-08-12.

[12] dphn/dolphincoder-starcoder2-7b model card. https://huggingface.co/dphn/dolphincoder-starcoder2-7b/raw/main/README.md - `datasets:` front matter listing `cognitivecomputations/dolphin-coder`. Fetched 2026-08-12.

[13] Hugging Face Hub API dataset search for author cognitivecomputations matching "coder". https://huggingface.co/api/datasets?author=cognitivecomputations&search=coder - no results beyond this repository. Fetched 2026-08-12.

[14] cognitivecomputations/dolphin dataset card, size endpoint, and first-rows endpoint, read via redirect to the current repo id `QuixiAI/dolphin`. https://huggingface.co/datasets/cognitivecomputations/dolphin/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=QuixiAI%2Fdolphin , https://datasets-server.huggingface.co/first-rows?dataset=QuixiAI%2Fdolphin&config=flan1m-alpaca-uncensored&split=train - description, config/row/column counts, and a sampled row for the neighbor comparison. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as single-turn code-generation and code-translation SFT data, with two caveats the card already establishes: no source names a generating model for the response text, and the repository has been renamed on the Hub since the shortlisted commit, so the old id only resolves through a redirect [1][3][4]. Both facts match the screening row's note.

### The screening row

The row's own note [3][4]: "The dolphin-coder training data: the Kaggle leetcode-rosetta collection transformed into a codegen file and a translate file; the card states only that source and names no generating model, and the viewer returns 404." The row carries no flag.
