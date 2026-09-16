# openai/openai_humaneval

164 hand-written Python programming problems, each a function signature plus docstring, a canonical solution, and a unit-test function - the original HumanEval functional-correctness benchmark, one `test` split only.

**openai/openai_humaneval** is OpenAI's HumanEval release, introduced in "Evaluating Large Language Models Trained on Code" [1], and lives at https://huggingface.co/datasets/openai/openai_humaneval . It comprises 164 problems handcrafted by OpenAI engineers and researchers, each pairing a function signature and docstring prompt with a reference solution and a set of assertion-based unit tests [2]. The paper defines its evaluation metric on these problems as pass@k: n ≥ k code samples are generated per problem (the paper uses n = 200, k ≤ 100), a problem counts as solved if any sample passes the unit tests, and pass@k is computed as an unbiased estimator over those samples rather than as a raw solved-fraction [1]. The dataset card states the problems were "handwritten to ensure not to be included in the training set of code generation models" [2], and its curation rationale explains the reasoning further: code models are commonly trained on GitHub dumps, so a benchmark not present in such a dump was needed to evaluate them properly, though the card also warns that "since this dataset was published on GitHub it is likely to be included in future dumps" [2]. **This is an evaluation benchmark, not training data: every one of its 164 rows must be held out of - and decontaminated against - any training corpus, never trained on.**

**Use it for**: nothing in the training sense - this dataset is the eval benchmark itself, and the corpus's own screening flag names it a contamination risk for that reason [3]. Its rows are not a training-data format at all: each row is a `prompt` (function signature and docstring), a `canonical_solution`, a `test` function, and an `entry_point`, structured for pass@k scoring rather than for SFT or preference training. Do not map it to any training-shape method card; use it only as a held-out eval set, and screen any code-training corpus for overlap with it before a scored run.

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated (`"gated": false`, `"private": false`) [4]. No further catch: the README states the licensing information as MIT License with no additional restriction [2].

**Shape**: 164 rows, one split (`test`), one config (`openai_humaneval`), five string columns [5][6].

**Hold out**: all 164 rows, in full - this dataset is the benchmark, not a training pool with a held-out slice. The screening row's note and flag both say so: "HumanEval, the 164 hand-written Python problems models are scored on, `test` only," flagged for "contamination: the HumanEval benchmark" [3].

**Origin**: built and released by OpenAI; the problems and their reference solutions are human-written by OpenAI engineers and researchers, per the dataset card's source-data section [2]. Hub API at the check date: `downloads` 275,041, `downloadsAllTime` 5,633,660, `likes` 399 [4].

**Trained-on-by**: none found - no source fetched for this card cites a model or recipe as having trained on this dataset.

**Introduced by**: [1] (Chen et al.).

## Shape

Rows and split, from datasets-server `/size` [5]:

| split | rows |
| --- | --- |
| `test` | 164 |

One config, `openai_humaneval`, with five columns, from datasets-server `/info` [6]:

| column | dtype |
| --- | --- |
| `task_id` | string |
| `prompt` | string |
| `canonical_solution` | string |
| `test` | string |
| `entry_point` | string |

Sizes, from datasets-server `/size` [5]: 83,920 bytes of original/Parquet download, 194,394 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The dataset card states the problems were handwritten specifically so they would not already appear in code-model training data, and flags that this protection erodes once the data is public on GitHub [2].
- The card's "Considerations for Using the Data" section warns to execute generated code only in a safe environment, since generated code being evaluated could be harmful [2].
- No source states a measured error rate, duplicate rate, or annotator-agreement figure for the 164 problems; several of the card's own subsections (initial data collection, annotation process, discussion of biases) are marked "[More Information Needed]" in the source [2].

## Load it

Single split, no train/test division to choose between - the entire `test` split is the benchmark and must be held out of training. Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the shortlist row's `commit` matches it; the repo was last modified 2024-01-04) [4]:

```python
import datasets

REV = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"  # main at the check date
test = datasets.load_dataset("openai/openai_humaneval", revision=REV, split="test")  # 164 rows - hold out entirely
```

**Trap**: there is no `train` split to load - `load_dataset` on this repository only ever returns the 164-row `test` split, so any pipeline expecting a training partition here is misconfigured; the entire dataset is eval-only [2][6].

## Neighbors

Several repositories extend or repackage this exact 164-problem set; every row count below was read live at the check date [7]. This corpus prefers the original for identity/decontamination checks, since neighbors reshape or extend the problems.

- `evalplus/humanevalplus` (HumanEval+) - 164 rows, five columns matching this dataset's column names; its row 0 (`task_id` `HumanEval/0`) carries the same `has_close_elements` prompt as this dataset's row 0, confirming at least that problem is shared, though the repository is far larger on disk (2,902,210 bytes original vs. 83,920 here), consistent with added test cases per problem; apache-2.0, ungated [7][8][12].
- `bigcode/humanevalpack` - a `python` config of 164 rows whose row 0 (`task_id` `Python/0`, `entry_point` `has_close_elements`) carries the same `has_close_elements` problem as this dataset's row 0 in its `declaration` field, plus five more per-language configs (`cpp`, `go`, `java`, `js`, `rust`, 164 rows each, 984 rows total across six configs), each row carrying 15 columns instead of 5; MIT, ungated [7][9][12].
- `codeparrot/instructhumaneval` - 164 rows reshaped to 9 columns; row content was not fetched for this card, so whether its problems match this dataset's is not confirmed here; licence not stated in its Hub card, ungated [7][10].

## A row

One config and one split, so one row covers the served shape. From `config="openai_humaneval"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [11]:

```json
{
  "task_id": "HumanEval/0",
  "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
  "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
  "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    [...]\n\n",
  "entry_point": "has_close_elements"
}
```

## Where it came from

Built and released by OpenAI, handcrafted by its engineers and researchers rather than mined from an existing corpus, specifically because problems drawn from public code (e.g., GitHub) risk already being in a code model's training data [2]. The paper that introduced it evaluates Codex, a GPT model fine-tuned on public GitHub code, against these 164 problems using the pass@k estimator described above [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Chen et al., "Evaluating Large Language Models Trained on Code", 2021. https://arxiv.org/abs/2107.03374 - the origin paper; current title read from the live abs page, pass@k metric definition and its use against these 164 problems read from the full-text PDF at https://arxiv.org/pdf/2107.03374 (converted with pdftotext). Fetched 2026-08-11.

[2] openai/openai_humaneval dataset card (README). https://huggingface.co/datasets/openai/openai_humaneval/raw/main/README.md - dataset summary, curation rationale, source-data section, data instance example, licensing information, safety consideration. Fetched 2026-08-11.

[3] The corpus screening row for `openai/openai_humaneval`, supplied with this card's request - its `note` and `flag`, read back in their own words in the appendix. Checked 2026-08-11.

[4] Hugging Face Hub API record for openai/openai_humaneval. https://huggingface.co/api/datasets/openai/openai_humaneval?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=openai%2Fopenai_humaneval Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=openai%2Fopenai_humaneval Fetched 2026-08-11.

[7] datasets-server size endpoint, one call per neighbor: `evalplus/humanevalplus`, `bigcode/humanevalpack`, `codeparrot/instructhumaneval`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[8] Hugging Face Hub API record for evalplus/humanevalplus. https://huggingface.co/api/datasets/evalplus/humanevalplus?full=true - licence, gate status. Fetched 2026-08-11.

[9] Hugging Face Hub API record for bigcode/humanevalpack. https://huggingface.co/api/datasets/bigcode/humanevalpack?full=true - licence, gate status. Fetched 2026-08-11.

[10] Hugging Face Hub API record for codeparrot/instructhumaneval. https://huggingface.co/api/datasets/codeparrot/instructhumaneval?full=true - licence field (absent), gate status. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=openai%2Fopenai_humaneval&config=openai_humaneval&split=test Fetched 2026-08-11.

[12] datasets-server first-rows endpoint, one call per neighbor, used to compare row 0's problem content against this dataset's row 0. https://datasets-server.huggingface.co/first-rows?dataset=evalplus%2Fhumanevalplus&config=default&split=test and https://datasets-server.huggingface.co/first-rows?dataset=bigcode%2Fhumanevalpack&config=python&split=test Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Flagged for contamination, not usable as training data: this dataset is the HumanEval benchmark itself, and every fact establishing that is already above - the dataset card's own description of the problems as handwritten to keep them out of training sets [2], and the screening row's flag naming the risk explicitly [3].

### The screening row

The row's own note [3]: "HumanEval, the 164 hand-written Python problems models are scored on, `test` only." Its flag [3]: "contamination: the HumanEval benchmark."
