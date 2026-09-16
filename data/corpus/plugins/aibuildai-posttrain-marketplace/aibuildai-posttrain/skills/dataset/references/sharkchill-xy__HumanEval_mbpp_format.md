# sharkchill-xy/HumanEval_mbpp_format

164 rows, one `train` split, two string columns (`task_id`, `prompt`) - the OpenAI HumanEval coding-eval problems rewritten into MBPP's natural-language-plus-assert-test prompt shape, with no solutions or unit-test code attached.

**sharkchill-xy/HumanEval_mbpp_format** is a community re-upload with no dataset card body beyond the placeholder "More Information needed" [1]; it carries no author statement, licence, or citation of its own. It lives at https://huggingface.co/datasets/sharkchill-xy/HumanEval_mbpp_format . Its `task_id` values run `HumanEval/0` through `HumanEval/163` and its `prompt` text is a natural-language task description followed by `Your code should pass these tests:` and `assert` statements [2][3] - the same shape MBPP's own `text` and `test_list` fields carry [4] - applied to the 164 problems from OpenAI's HumanEval benchmark, introduced in "Evaluating Large Language Models Trained on Code" [5] and served on the Hub as `openai/openai_humaneval` [6]. **This is an evaluation benchmark, not training data: it holds only the reformatted prompts, with no `canonical_solution`, `test`, or `entry_point` field [2][3], and all 164 `task_id` values on both sides, each read in two calls covering offsets 0 and 100, match one-to-one against `openai/openai_humaneval`'s own 164 rows [2][3][12][13] - despite the single split being named `train`, none of these rows should enter a training set for any model later scored on HumanEval.**

**Use it for**: nothing training-shaped. The two columns hold only a task_id and a reformatted prompt, with no target completion, canonical solution, or executable test code, so the rows cannot feed SFT, preference, or reward-model training as they stand; the only sound use is as an alternate-format eval prompt set, and comparing it against the SFT method card's training corpus for decontamination.

**Licence**: not stated. `cardData.license` is absent and the repository's tags carry no licence identifier [1][7]; the upstream HumanEval rows it reformats are MIT-licensed on the Hub [6][8].

**Shape**: one config (`default`), one split (`train`, 164 rows), two columns (`task_id` string, `prompt` string) [7][9].

**Hold out**: all 164 rows, from any training run later evaluated on HumanEval - all 164 rows read (offsets 0 and 100) match a HumanEval `task_id` one-to-one against `openai/openai_humaneval`'s own 164 rows, read the same way [2][3][12][13], so training on this file is training on the eval set under a different prompt wording.

**Origin**: uploaded by Hub user sharkchill-xy; the `task_id`/`prompt` pairs are OpenAI's original human-written HumanEval problems [5] passed through an unstated reformatting step, since the card names no generating model or script [1]. Hub API on the check date: 31 downloads, 0 likes [7].

**Trained-on-by**: none found. No source found in this search names a model or training recipe that trained on this specific re-upload.

**Introduced by**: no paper - the dataset card [1] carries no citation; the underlying 164 problems come from Chen et al., "Evaluating Large Language Models Trained on Code" [5].

## Shape

Rows and bytes, from the datasets-server size endpoint [9]:

| split | rows |
| --- | --- |
| `train` | 164 |

One config, `default`, two columns, per the datasets-server info endpoint [7][10]:

| column | dtype |
| --- | --- |
| `task_id` | string |
| `prompt` | string |

Original download 24,961 bytes, in-memory (decoded) size 58,366 bytes [9][1]. No source states sequence-length or token statistics for this file.

## Quality

- Every row holds only a `task_id` and a `prompt` string; there is no `canonical_solution`, `test`, or `entry_point` column, so none of the 164 rows carries a way to check a generated answer without pulling those fields from elsewhere [2][3][7].
- Reading all 164 served rows, in two calls at offset 0 (rows 0-99) and offset 100 (rows 100-163), against the 164 rows of `openai/openai_humaneval`, read the same way [12][13], shows the `task_id` values match one-to-one, `HumanEval/0` through `HumanEval/163`, so no row here names a problem outside the standard HumanEval set [2][3].
- The card states no measured quality signal (contamination rate, annotation process, or known complaint) of its own [1]; none is invented here.
- The `prompt` field reproduces the original HumanEval task's docstring content as a plain natural-language instruction followed by `assert`-based tests, matching the shape of MBPP's `text` plus `test_list` fields [4], rather than HumanEval's original function-signature-plus-docstring `prompt` format [6].

## Load it

```python
import datasets

REV = "096841382a4065abf5a27cc012201726ae9f90ef"  # main at the check date
ds = datasets.load_dataset("sharkchill-xy/HumanEval_mbpp_format", revision=REV, split="train")  # 164 rows
```

**Trap**: the single split is named `train`, and the two columns (`task_id`, `prompt`) look like a ready-to-train instruction set, but every row is a HumanEval eval problem with no solution or test attached [2][3][12][13] - loading this as a `train` split for SFT or preference training puts eval prompts into the training data of anything later scored on HumanEval.

## Neighbors

- `openai/openai_humaneval` - the original HumanEval release this file reformats: same 164 `task_id` values and split count (as `test`, not `train`), but with the original prompt wording plus `canonical_solution`, `test`, and `entry_point` columns this repository omits [6][11]. Prefer it over this repository whenever a runnable HumanEval eval is needed, since it carries the code needed to check an answer and this repository does not.
- `google-research-datasets/mbpp` - the dataset whose `text`/`test_list` prompt shape this repository copies onto HumanEval's problems; it is a distinct, non-overlapping problem set (different `task_id` scheme, different problems) and not a source of duplicate rows [4].

## A row

The repository serves one config and one split. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server first-rows) [2]:

```json
{
  "task_id": "HumanEval/0",
  "prompt": "Write a python function to check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    \nYour code should pass these tests:\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n"
}
```

## Where it came from

The 164 underlying problems are OpenAI's HumanEval, handwritten by OpenAI engineers and researchers specifically to be excluded from code-model training data, and introduced in "Evaluating Large Language Models Trained on Code" [5]. This repository's own card states nothing about who performed the MBPP-style reformatting or when beyond the repository's creation date, 2024-02-03 [1][7]; no generating model or script is named.

## Sources

Every source below was fetched on the check date, 2026-08-11; Hub repositories are mutable, which is why Load it pins the revision.

[1] sharkchill-xy/HumanEval_mbpp_format dataset card (README). https://huggingface.co/datasets/sharkchill-xy/HumanEval_mbpp_format/raw/main/README.md - card body, `dataset_info` block. Fetched 2026-08-11.

[2] datasets-server first-rows endpoint, offset 0. https://datasets-server.huggingface.co/first-rows?dataset=sharkchill-xy%2FHumanEval_mbpp_format&config=default&split=train - rows 0-99 (this endpoint truncates at 100 rows); `task_id` and `prompt` values. Fetched 2026-08-11.

[3] datasets-server rows endpoint, offset 100. https://datasets-server.huggingface.co/rows?dataset=sharkchill-xy%2FHumanEval_mbpp_format&config=default&split=train&offset=100&length=64 - rows 100-163, completing the read of all 164 served rows; `task_id` and `prompt` values. Fetched 2026-08-11.

[4] datasets-server first-rows endpoint for google-research-datasets/mbpp. https://datasets-server.huggingface.co/first-rows?dataset=google-research-datasets%2Fmbpp&config=full&split=test - `text`/`test_list` field shape. Fetched 2026-08-11.

[5] Chen et al., "Evaluating Large Language Models Trained on Code", 2021. https://arxiv.org/abs/2107.03374 - the HumanEval origin paper, current title read from the openai/openai_humaneval card's citation. Fetched 2026-08-11.

[6] openai/openai_humaneval dataset card (README). https://huggingface.co/datasets/openai/openai_humaneval/raw/main/README.md - split name/count, column names, licence, paper link. Fetched 2026-08-11.

[7] Hugging Face Hub API record for sharkchill-xy/HumanEval_mbpp_format. https://huggingface.co/api/datasets/sharkchill-xy/HumanEval_mbpp_format?full=true - `sha`, `downloads`, `likes`, `cardData`, `siblings`, `createdAt`. Fetched 2026-08-11.

[8] Hugging Face Hub API record for openai/openai_humaneval. https://huggingface.co/api/datasets/openai/openai_humaneval?full=true - `license:mit` tag. Fetched 2026-08-11.

[9] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=sharkchill-xy%2FHumanEval_mbpp_format Fetched 2026-08-11.

[10] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=sharkchill-xy%2FHumanEval_mbpp_format Fetched 2026-08-11.

[11] datasets-server info endpoint for openai/openai_humaneval. https://datasets-server.huggingface.co/info?dataset=openai%2Fopenai_humaneval - column list (`task_id`, `prompt`, `canonical_solution`, `test`, `entry_point`), split name `test`. Fetched 2026-08-11.

[12] datasets-server first-rows endpoint for openai/openai_humaneval, offset 0. https://datasets-server.huggingface.co/first-rows?dataset=openai%2Fopenai_humaneval&config=openai_humaneval&split=test - rows 0-99 (this endpoint truncates at 100 rows); `task_id` values. Fetched 2026-08-11.

[13] datasets-server rows endpoint for openai/openai_humaneval, offset 100. https://datasets-server.huggingface.co/rows?dataset=openai%2Fopenai_humaneval&config=openai_humaneval&split=test&offset=100&length=64 - rows 100-163, completing the read of all 164 openai_humaneval rows; `task_id` values. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Not usable as training data under its own `train` label: the two facts that decide this are already established above - the file holds only `task_id` and `prompt`, with no solution or test code [2][3][7], and all 164 rows read duplicate one of the `openai/openai_humaneval` eval problems by `task_id` [2][3][12][13]. It is a decontamination trap, matching the screening row's flag.

### The screening row

The row's own note [screening file for `sharkchill-xy/HumanEval_mbpp_format`, checked 2026-08-11]: "The same 164 HumanEval tasks reformatted MBPP-style with task ids HumanEval/0 onward; prompts only with no solutions, and the split is named `train`." Its flag: "trap: HumanEval reformatted with the split named train; a prompts-only duplicate of openai/openai_humaneval".
