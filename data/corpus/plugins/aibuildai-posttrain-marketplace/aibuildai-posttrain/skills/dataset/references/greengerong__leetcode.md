# greengerong/leetcode

2,360 LeetCode coding problems, each paired with a Markdown problem statement and four language-specific solutions (Java, C++, Python, JavaScript) with an inline explanation - a single-config, single-split Hub dataset.

**greengerong/leetcode** packages LeetCode problem statements with worked solutions and explanations in four programming languages, one row per problem, across a single `default` config and `train` split [1]. The repository carries no description beyond its licence declaration: the README's entire body is the three-line YAML front matter `license: mit`, and no paper, blog post, or homepage is linked anywhere on the page [2]. The Hub API record names the account `greengerong` as the repository's author [3], but no source states who wrote the solutions or explanations, or whether they are human-authored or model-generated [2][3]. It lives at https://huggingface.co/datasets/greengerong/leetcode .

**Use it for**: code-generation SFT - each row's `content` field is a natural-language problem statement that maps to a prompt, and each of the four language columns is a candidate completion; there is no chosen/rejected pairing and no chat turns, so this is raw single-turn data, not the implicit-prompt preference format [1]. Map it to the SFT method card by extracting `content` as the prompt and one language column as the target completion, stripping the leading/trailing indentation each code+explanation field carries (see Quality). No stated non-commercial or preference-only restriction applies.

**Licence**: MIT (`cardData.license` is `"mit"`, repo tag `license:mit`), ungated (`"gated": false`, `"private": false`) [3]. No catch beyond the grant itself: the README states nothing else, and no additional terms are declared anywhere in the repository [2][3].

**Shape**: 2,360 rows in one config (`default`), one split (`train`), nine columns [1][4].

**Hold out**: not stated. The repository has no test/eval split, and no source here - the README, the Hub API record, or the screening note - names an evaluation set this release may overlap [2][3][5].

**Origin**: the Hub API record names `greengerong` as the repository's author [3]; no source states a generating model or human-authorship claim for the solutions or explanations [2]. Hub API at the check date: `downloads` 1,198, `downloadsAllTime` 26,650, `likes` 104 [3].

**Trained-on-by**: none found. No source examined for this card - the dataset's own card, or the Hub API record - states that a named model or training recipe used this release [2][3].

**Introduced by**: no paper - the dataset card [2].

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 2,360 |
| total | 2,360 |

One config, `default`, with nine columns (datasets-server `/info`) [1]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `slug` | string |
| `title` | string |
| `difficulty` | string |
| `content` | string |
| `java` | string |
| `c++` | string |
| `python` | string |
| `javascript` | string |

`id` is not a row index: rows read at offset 0 carry ids 1-10 and rows read at offset 2,350 carry ids 2,603-2,612, so `id` is LeetCode's own problem number and the 2,360 served rows are a subset of a wider, non-contiguous id range spanning at least 1 to 2,612 [6]. Sizes (datasets-server `/size`) [4]: 16,061,888 bytes of original JSON-lines download, 7,038,211 bytes as Parquet, 15,477,592 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- Of the 30 rows read at offset 0, one (`id` 11, "Container With Most Water") has a null `difficulty` value; the other 29 are `Easy`, `Medium`, or `Hard` [6].
- Of the same 30 rows, every one has an identical explanation paragraph repeated verbatim across its `java`, `c++`, `python`, and `javascript` fields, and that shared paragraph itself names all four languages generically (for example: "The algorithm leverages a hash map (unordered_map in C++, HashMap in Java, dictionary in Python, and Map in JavaScript)") regardless of which column it sits in [6]. This pattern was checked only in these 30 rows at offset 0, not the full 2,360.
- Each language column carries a fixed formatting artifact: leading whitespace before the opening ```` ``` ```` fence and trailing whitespace after the explanation, present in every one of the 30 rows read [6]. A prompt/completion split needs to strip this before use.
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset; none is invented here.

## Load it

```python
import datasets

REV = "00f2d466dc0f00f65a0b6938c4c11a57f721db81"  # main at the check date
train = datasets.load_dataset("greengerong/leetcode", revision=REV, split="train")  # 2,360 rows
```

Pin `REV` to the repo's `sha` for `main` at the check date (repo last modified 2023-08-06) [3] for a reproducible `load_dataset` read of the underlying git repository. That pin does **not** cover this card's row-count, schema, or sample-row numbers: those come from the datasets-server `/size`, `/info`, `/first-rows`, and `/rows` endpoints [4][1][6], and every one of those endpoints ignores a `revision` query parameter - passing a nonexistent revision to each returns byte-identical output - so they always serve a live index of the default branch, not the pinned commit [4][1][6].

**Trap**: there is only one split, `train` - there is no held-out split to load by mistake, but also none to evaluate against; any train/test split for this data has to be constructed by the user.

## Neighbors

Two Hub datasets reformat this exact row set; a third is an independently built LeetCode dataset with its own eval-oriented design, not a copy of this one:

- `techandy42/multi_lang_leetcode` - the same 2,360 rows and the same nine columns, plus four added `*_code_only` columns; reading row 0 (`id` 1, Two Sum) from its own `/first-rows` endpoint shows each `*_code_only` field holds exactly the bare code from the matching language column here with the ```` ``` ```` fence, surrounding whitespace, and trailing explanation paragraph removed [7].
- `RayBernard/leetcode` - 2,359 rows (one fewer than here) reformatted into an Alpaca-style `instruction`/`input`/`output`/`text` layout; its `id` 1 row's Python `output` and `input` fields reproduce this release's Two Sum Python solution and explanation text verbatim, confirming the same underlying problem/solution content under a different schema [8].
- `newfacade/LeetCodeDataset` - a separately built, Apache-2.0-licensed, Python-only LeetCode dataset described on its own card as meant for both training and evaluation of code LLMs, with its own arXiv paper; it does not share this release's schema or licence and is not derived from it [9].

Prefer this release when a reader wants all four languages per problem in one row; prefer `techandy42/multi_lang_leetcode` when bare code (no fence, no explanation) is wanted without post-processing.

## A row

The repository serves one config and one schema, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [6], with the four solution fields truncated:

```json
{
  "id": 1,
  "slug": "two-sum",
  "title": "Two Sum",
  "difficulty": "Easy",
  "content": "Given an array of integers `nums` and an integer `target`, return _indices of the two numbers such that they add up to `target`_.\n\nYou may assume that each input would have **_exactly_ one solution**, and you may not use the _same_ element twice. [...] **Follow-up:** Can you come up with an algorithm that is less than `O(n2)` time complexity?",
  "java": "\n    ```java\nimport java.util.HashMap;\nimport java.util.Map;\n\npublic int[] twoSum(int[] nums, int target) {\n    Map<Integer, Integer> map = new HashMap<>();\n    for (int i = 0; i < nums.length; i++) { [...] }\n```\n    \n    The algorithm leverages a hash map (unordered_map in C++, HashMap in Java, dictionary in Python, and Map in JavaScript). [...] This approach has a time complexity of O(n) and a space complexity of O(n) as well.\n    ",
  "c++": "\n    ```cpp\n#include <vector>\n#include <unordered_map>\n\nstd::vector<int> twoSum(std::vector<int>& nums, int target) { [...] }\n```\n    \n    The algorithm leverages a hash map (unordered_map in C++, HashMap in Java, dictionary in Python, and Map in JavaScript). [...]\n    ",
  "python": "\n    ```python\ndef twoSum(nums, target):\n    map = {}\n    for i, num in enumerate(nums): [...]\n```\n    \n    The algorithm leverages a hash map (unordered_map in C++, HashMap in Java, dictionary in Python, and Map in JavaScript). [...]\n    ",
  "javascript": "\n    ```javascript\nfunction twoSum(nums, target) {\n    const map = new Map(); [...] }\n```\n    \n    The algorithm leverages a hash map (unordered_map in C++, HashMap in Java, dictionary in Python, and Map in JavaScript). [...]\n    "
}
```

## Where it came from

The repository is hosted under the Hub account `greengerong`, and its README states nothing about how the problems, solutions, or explanations were produced - no generating model and no data-collection process is named anywhere on the page [2]. The `id` values are LeetCode's own problem numbers rather than a local row index (see Shape), which places the source problem pool as the LeetCode platform, but no source here states whether the problem-statement text was scraped, licensed, or transcribed, nor whether the solutions and explanations are human-written or model-generated [2][6].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision for `load_dataset`; the datasets-server endpoints used elsewhere on this card are unpinned and live (see Load it).

[1] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=greengerong%2Fleetcode - column names and dtypes; confirmed to ignore a `revision` query parameter (identical output with an invalid revision). Fetched 2026-08-11.

[2] greengerong/leetcode dataset card (README). https://huggingface.co/datasets/greengerong/leetcode/raw/main/README.md - the full card body, three lines: `---\nlicense: mit\n---`. Fetched 2026-08-11.

[3] Hugging Face Hub API record for greengerong/leetcode. https://huggingface.co/api/datasets/greengerong/leetcode?full=true and the `expand[]=downloadsAllTime` variant of the same endpoint - `author`, licence, gate/private status, `sha`, siblings, `downloads`, `downloadsAllTime`, `likes`, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=greengerong%2Fleetcode - row counts and byte sizes; confirmed to ignore a `revision` query parameter (identical output with an invalid revision). Fetched 2026-08-11.

[5] The corpus screening row for `greengerong/leetcode`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[6] datasets-server first-rows and rows endpoints. https://datasets-server.huggingface.co/first-rows?dataset=greengerong%2Fleetcode&config=default&split=train (30 rows at offset 0) and https://datasets-server.huggingface.co/rows?dataset=greengerong%2Fleetcode&config=default&split=train&offset=2350&length=10 (10 rows at offset 2,350); confirmed `/first-rows` ignores a `revision` query parameter (identical output with an invalid revision). Fetched 2026-08-11.

[7] techandy42/multi_lang_leetcode dataset card, size endpoint, and first-rows endpoint. https://huggingface.co/datasets/techandy42/multi_lang_leetcode/raw/main/README.md, https://datasets-server.huggingface.co/size?dataset=techandy42%2Fmulti_lang_leetcode, and https://datasets-server.huggingface.co/first-rows?dataset=techandy42%2Fmulti_lang_leetcode&config=default&split=train - column list, row count, and row-0 content confirming what the `*_code_only` columns hold. Live, unpinned endpoints. Fetched 2026-08-11.

[8] RayBernard/leetcode dataset card, size endpoint, and first-rows endpoint. https://huggingface.co/datasets/RayBernard/leetcode/raw/main/README.md, https://datasets-server.huggingface.co/size?dataset=RayBernard%2Fleetcode, and https://datasets-server.huggingface.co/first-rows?dataset=RayBernard%2Fleetcode&config=default&split=train - schema, row count, and row-0 content used to confirm shared underlying content. Live, unpinned endpoints. Fetched 2026-08-11.

[9] newfacade/LeetCodeDataset dataset card. https://huggingface.co/datasets/newfacade/LeetCodeDataset/raw/main/README.md - licence, stated training-and-evaluation purpose, and linked paper, used only to describe this dataset as a distinct, non-derived neighbor. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as raw code-generation SFT source data, with one open gap: the card states nothing about who wrote the solutions and explanations, and no other source examined here fills that gap [2][3]. This rests on the same fact the opening paragraph and the screening row both state: the entire README is a single MIT licence line with no authorship or generation-process claim [2][5].

### The screening row

The row's own note [5]: "LeetCode problem statements (human) with solutions and explanations in Java/C++/Python/JS; the card is one licence line and never says who wrote the solutions." The row carries no flag.
