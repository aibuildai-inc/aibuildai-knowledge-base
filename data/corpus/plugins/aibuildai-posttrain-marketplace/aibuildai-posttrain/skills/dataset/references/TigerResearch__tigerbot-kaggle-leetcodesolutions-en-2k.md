# TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k

2,048 English instruction/input/output triples, each a LeetCode problem paired with a solution and explanation in one of four programming languages, released as part of TigerBot's open SFT data collection.

**TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k** was built by TigerResearch, the team behind the TigerBot open-source LLM project, by processing a Kaggle LeetCode-solutions collection into TigerBot's Alpaca-style instruction/input/output SFT schema; the dataset card names the source as the Kaggle dataset `erichartford/leetcode-solutions` [1]. There is no dedicated paper for this dataset - it is documented only by its own Hub card [1] and by its entry in the TigerBot project's data table, where it is one of the "自研*" ("secondary development on existing data") releases feeding TigerBot's roughly 1.2M-pair open instruction set [2]. It serves single-turn, code-generation SFT: given a LeetCode problem statement and a target language, produce working code plus an explanation.

**Use it for**: SFT chat/instruction training, specifically code generation with explanation - concatenate `instruction` and `input` into the prompt and train the model to produce `output` (the SFT method card's Alpaca-style instruction/input/output shape). No usage-shape restriction is stated by any source.

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, repo tags include `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [3]. No licence catch is stated beyond the standard Apache-2.0 grant.

**Shape**: 2,048 rows, one config (`default`), one split (`train`), three string columns (`instruction`, `input`, `output`) [4][5].

**Hold out**: Nothing - the repository serves a single `train` split with no accompanying eval split, and no source (the dataset card, the TigerBot project README, or the screening note) states that any rows overlap a held-out evaluation set [1][2][6].

**Origin**: built by TigerResearch from a Kaggle-sourced problem/solution pool; the code solutions and explanations are machine-generated (per-language code plus a prose "explanation" block), not human-written from scratch [1]. Hub API at the check date: `downloads` 88, `downloadsAllTime` 2,558, `likes` 17 [3].

**Trained-on-by**: TigerResearch's own TigerBot models - the TigerBot project README lists this dataset in its open SFT data table as one of the released instruction sets feeding TigerBot's instruction tuning [2]. No adoption by other named models or recipes was found.

**Introduced by**: no paper - the dataset card [1]; the TigerBot project overview describes the broader collection it belongs to but does not name this dataset specifically [2].

## Shape

Rows and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 2,048 |

One config, `default`, with three columns (datasets-server `/info`) [4]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |

Sizes (datasets-server `/size`) [5]: 5,078,505 bytes of original JSON download, 937,266 bytes as Parquet, 4,847,444 bytes decoded in memory. No source states sequence-length or token statistics for this dataset.

## Quality

- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset.
- The dataset card gives no description of how the code solutions or explanations were produced or checked, beyond calling the repository "加工生成的代码类sft数据集" (a code-category SFT dataset produced by processing the source data) [1].
- The TigerBot project README's general cleaning-rule description (sensitive-word filtering, invalid-input/output filtering, keyword-based text cleanup, instruction/input-duplication cleanup) is stated for the TigerBot open data collection as a whole, not specifically confirmed for this file [2].
- Of the first 84 served rows (offsets 0-83, config `default`, split `train`), the `instruction` field takes one of four distinct values, one per language (c++, java, python, javascript), each of the form "Use \<language\> to solve the following problems, and give an explanation.", each occurring 21 times; each of the 21 distinct `input` problem statements recurs once per language; this pattern is observed only in those 84 rows and is not asserted for the remaining 1,964 rows [7].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-05-31) [3]:

```python
import datasets

REV = "aaf5fce34b1d673a60ea2e1fc49182698c6a65c5"  # main at the check date
ds = datasets.load_dataset(
    "TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k",
    revision=REV,
    split="train",
)  # 2,048 rows
```

**Trap**: the repository ships a single JSON file (`tigerbot-kaggle-leetcodesolutions-en-2k.json`) as its only data file, with no `data_dir` or config split beyond `default`/`train` - there is no held-out split to pass, and the whole file loads as `train` [1][8].

## Neighbors

- `dim/leetcodesolutions_en_2k` - a repository serving the same three columns, the same 2,048-row `train` split, and identical row content at offset 0 (`instruction`, `input`, and `output` for "Two Sum" in C++ match this repository's row 0 exactly), so it is a byte-level mirror of this dataset under a different owner and MIT rather than Apache-2.0 licensing; treat it as a duplicate, not an independent source [9][10].
- `greengerong/leetcode` - 2,360 rows in a different, wide schema (`id`, `slug`, `title`, `difficulty`, `content`, plus one column per language: `java`, `c++`, `python`, `javascript`), one row per LeetCode problem rather than one row per problem-language pair; its row 0 covers the same "Two Sum" problem and the same Java solution text seen in this dataset's row 1, so the two repositories draw on overlapping underlying LeetCode content, but `greengerong/leetcode` is not an Alpaca-style instruction/input/output SFT file and needs reshaping before use [11][12]. This corpus prefers the TigerResearch release for direct SFT loading; `greengerong/leetcode` is worth reaching for only when a wider per-problem, per-language table (with difficulty labels) is needed.

## A row

The repository serves one config and one schema, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7]:

```json
{
  "instruction": "Use c++ to solve the following problems, and give an explanation.",
  "input": "Given an array of integers `nums` and an integer `target`, return _indices of the two numbers such that they add up to `target`_. [...] **Follow-up:** Can you come up with an algorithm that is less than `O(n2)` time complexity?",
  "output": "```cpp\n#include <vector>\n#include <unordered_map>\n\nstd::vector<int> twoSum(std::vector<int>& nums, int target) {\n    std::unordered_map<int, int> map;\n    for (int i = 0; i < nums.size(); i++) {\n        int complement = target - nums[i];\n        if (map.find(complement) != map.end()) {\n            return {map[complement], i};\n        }\n        map[nums[i]] = i;\n    }\n    return {};\n}\n```\nexplanation\nThe algorithm leverages a hash map [...] This approach has a time complexity of O(n) and a space complexity of O(n) as well."
}
```

Row 1 (same config and split, `row_idx=1`) shows the same problem restated for a different language, confirming the per-language repetition pattern noted above: `instruction` reads "Use java to solve the following problems, and give an explanation.", `input` is the identical "Two Sum" statement, and `output` is a Java solution with the same explanation text [7].

## Where it came from

Built by TigerResearch from a Kaggle collection of LeetCode problems and solutions, `erichartford/leetcode-solutions`, named as the source directly in the dataset card [1]. The dataset card and TigerBot project README give no further detail on how the Kaggle source was filtered, deduplicated, or reformatted into the `instruction`/`input`/`output` triples served here, beyond classifying the release as "自研*" ("secondary development on existing data") in the TigerBot project's open-data table - distinct from "自研" (fully self-developed) and "开源" (used as-is from an open source) [1][2]. The Kaggle source page itself was not fetched for this card (it sits behind Kaggle's own access flow); the claim about it is attributed to the TigerBot dataset card that names it, not verified directly [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k, Hugging Face Hub dataset page. https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k - the item's Hub landing page; its rendered card body is the same text fetched from the raw README at https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k/raw/main/README.md - Kaggle source name, description, load instructions. Fetched 2026-08-12.

[2] TigerBot project README (GitHub, `main` branch). https://raw.githubusercontent.com/TigerResearch/TigerBot/main/README.md - open SFT data table listing this dataset, "自研*" category description, general data-cleaning rules. A `main`-branch file, unpinned and mutable. Fetched 2026-08-12.

[3] Hugging Face Hub API record for TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k. https://huggingface.co/api/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k?full=true - licence, gate status, `sha`, `downloads`, `likes`, last-modified date, siblings; `downloadsAllTime` (2,558) read through the same endpoint's `expand[]=downloadsAllTime` variant, distinct from the `downloads` field (88) shown in the base record. Fetched 2026-08-12.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=TigerResearch%2Ftigerbot-kaggle-leetcodesolutions-en-2k Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=TigerResearch%2Ftigerbot-kaggle-leetcodesolutions-en-2k Fetched 2026-08-12.

[6] The corpus screening row for `TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k`, supplied with this card's request. Checked 2026-08-12.

[7] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=TigerResearch%2Ftigerbot-kaggle-leetcodesolutions-en-2k&config=default&split=train - 84 rows returned at offset 0. Fetched 2026-08-12.

[8] Hugging Face Hub repository tree for TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k. https://huggingface.co/api/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k/tree/main - lists `.gitattributes`, `README.md`, and the single data file `tigerbot-kaggle-leetcodesolutions-en-2k.json`. Fetched 2026-08-12.

[9] Hugging Face Hub API record for dim/leetcodesolutions_en_2k. https://huggingface.co/api/datasets/dim/leetcodesolutions_en_2k?full=true - licence, schema, row count, downloads, likes. Fetched 2026-08-12.

[10] datasets-server first-rows endpoint for dim/leetcodesolutions_en_2k. https://datasets-server.huggingface.co/first-rows?dataset=dim%2Fleetcodesolutions_en_2k&config=default&split=train - row 0 compared against this dataset's row 0. Fetched 2026-08-12.

[11] Hugging Face Hub API record and datasets-server size/info for greengerong/leetcode. https://huggingface.co/api/datasets/greengerong/leetcode?full=true , https://datasets-server.huggingface.co/info?dataset=greengerong%2Fleetcode , https://datasets-server.huggingface.co/size?dataset=greengerong%2Fleetcode - licence, schema, row count (2,360), downloads, likes. Fetched 2026-08-12.

[12] datasets-server first-rows endpoint for greengerong/leetcode. https://datasets-server.huggingface.co/first-rows?dataset=greengerong%2Fleetcode&config=default&split=train - row 0 compared against this dataset's rows 0 and 1. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as SFT data for code generation: load the single `train` split, no held-out split needed since none is declared and no source flags evaluation overlap. This rests on the shape and schema established above [1][4][5] and on the screening row's own note [6].

### The screening row

The row's own note [6]: "the same 2,048 Kaggle leetcode-solutions records processed into TigerBot SFT form." The row carries no flag.
