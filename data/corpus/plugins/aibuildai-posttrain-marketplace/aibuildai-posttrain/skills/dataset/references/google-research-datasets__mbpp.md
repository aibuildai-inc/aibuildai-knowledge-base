# google-research-datasets/mbpp

974 crowd-sourced Python programming problems (plus a 427-row hand-verified "sanitized" subset), each a short task description, a reference solution, and three test-case asserts - the Mostly Basic Python Problems benchmark.

**google-research-datasets/mbpp** is Google Research's Mostly Basic Python Problems (MBPP) dataset, built by asking an internal pool of crowdworkers with basic Python knowledge to each write a short problem statement, a self-contained Python function solving it, and three asserts checking correctness, then to supply a ground-truth solution passing those asserts [1]. It was introduced in "Program Synthesis with Large Language Models" [1] to measure few-shot and fine-tuned program synthesis; a second, smaller `sanitized` version was produced by the paper's authors manually inspecting, editing, and pruning a subset of the questions for signature, clarity, and test-case accuracy, yielding what the paper calls the "edited dataset" [1]. It lives at https://huggingface.co/datasets/google-research-datasets/mbpp [2]. **Both the `full` and `sanitized` configs split rows by `task_id` into the same four bands - `prompt` (1-10, few-shot examples), `test` (11-510, the held-out MBPP benchmark), `validation` (511-600), and `train` (601-974) - so both configs' `test` splits are benchmark rows and must be held out of training, not just one of them [2].**

**Use it for**: SFT on short natural-language-to-Python-function generation - `text`/`prompt` as the instruction, `code` as the target completion, `test_list` usable as a correctness check or a reward signal rather than as training text - training on `train` (374 rows, `full`) or `train` (120 rows, `sanitized`), never on `test` from either config, which is the MBPP few-shot/fine-tuned-eval benchmark itself [1][2]. Maps to the SFT method card's prompt-completion format.

**Licence**: CC-BY-4.0 (`cardData.license: cc-by-4.0`; the README's own Licensing Information section states "CC-BY-4.0"), ungated [3][2]. One catch: the shortlist row that seeded this card also lists "MIT" as a licence-body mention; the README's License section and the linked GitHub repository's LICENSE (Apache-2.0, for the code, not the data) do not contain the string "MIT" anywhere in the copies fetched for this card, so that mention is not corroborated by any source read here [2][4].

**Shape**: two configs, `full` (974 rows: train 374 / test 500 / validation 90 / prompt 10) and `sanitized` (427 rows: train 120 / test 257 / validation 43 / prompt 7) [5][6].

**Hold out**: `full/test` (500 rows) and `sanitized/test` (257 rows) - both cover the same `task_id` 11-510 benchmark band per the GitHub README's evaluation-split description, so both must be excluded from training, not only one [4].

**Origin**: built by an internal Google crowdsourcing effort, with the `sanitized` edits made by the paper's authors [1]; Hub API at the check date: 205,200 downloads (9,827,078 all-time), 233 likes [3].

**Trained-on-by**: the origin paper's own decoder LMs (244M-137B parameters) were fine-tuned on the 374-row `train` split [1]; on the Hub, a handful of community fine-tunes tag this dataset, e.g. `DeryFerd/Qwen2.5-Coder-7B-Instruct-Distill-Phi2-974mbpp`, `arbenilazi/dpo-mbpp-merged`, and `arbenilazi/grpo-mbpp-merged` [7]; no broadly-documented flagship model or training recipe beyond the origin paper was found in the sources fetched.

**Introduced by**: [1] (Austin et al.).

## Shape

Rows and splits, both configs (datasets-server `/size`) [5]:

| config | split | rows |
| --- | --- | --- |
| `full` | train | 374 |
| `full` | test | 500 |
| `full` | validation | 90 |
| `full` | prompt | 10 |
| `sanitized` | train | 120 |
| `sanitized` | test | 257 |
| `sanitized` | validation | 43 |
| `sanitized` | prompt | 7 |
| total | | 1,401 |

Columns (datasets-server `/info`) [6]:

| config | columns |
| --- | --- |
| `full` | `task_id` int32, `text` string, `code` string, `test_list` list\<string\>, `test_setup_code` string, `challenge_test_list` list\<string\> |
| `sanitized` | `source_file` string, `task_id` int32, `prompt` string, `code` string, `test_imports` list\<string\>, `test_list` list\<string\> |

All four splits within a config share this same column set [6]. Sizes (datasets-server `/size`) [5]: `full` 236,069 bytes Parquet / 468,062 bytes in memory; `sanitized` 115,422 bytes Parquet / 219,685 bytes in memory; 351,491 bytes combined original download. No source states sequence-length or token statistics for either config; none is invented here.

The `task_id` bands are stated explicitly in the GitHub repository's README linked from the Hub card: task IDs 11-510 are the test set, 1-10 the few-shot prompting set, 511-600 the validation set used during fine-tuning, and 601-974 the training set, with the sanitized subset inheriting the same groupings [4]. Reading the served rows confirms this for three of the four `full` bands in full: `full/prompt` (10/10 rows read) spans `task_id` 1-10; `full/validation` (90/90 rows read) spans 511-600. The first 100 of 374 `full/train` rows (offset 0) span 601-700, and the first 100 of 500 `full/test` rows (offset 0) span 11-110 - consistent with, but not a full re-verification of, the stated 601-974 and 11-510 bands [8].

## Quality

- The paper's own characterization: crowdworkers were told to write problem descriptions concrete enough for a human to code from without clarification, self-contained code that does not print to the console, and were permitted to use internet references [1]. A random sample of 100 questions, tagged by the authors, showed 58% mathematical in nature, 43% involving list processing, 19% string processing, 9% integer sequences, and 2% other data structures [1].
- The `sanitized` config is the result of the authors manually inspecting, editing, and pruning a subset "to ensure it had a standard Python function signature, that it was unambiguous to a human, and that its test cases accurately reflected the text description" [1]. The paper reports the resulting edited set at 426 hand-verified questions [1]; the Hub `sanitized` config totals 427 rows across its four splits, one more than the paper's stated count [1][5].
- Comparing the two configs at matching `task_id`s (11, read from both `full/test` and `sanitized/test` row 0) shows a byte-for-byte identical `test_list` in both, while `code` differs only in line endings (`\r\n` in `full` versus `\n` in `sanitized`, same logic otherwise) - confirming `sanitized` is a re-edited version of (not a disjoint set from) `full` for at least this row [8].
- Of the 100 `full/test` rows read (offset 0, `task_id` 11-110), 11 carry a non-empty `challenge_test_list`; none of the 290 `full` rows read across `train`/`validation`/`prompt` (offset 0 in each) carry a non-empty `test_setup_code`. Of the 250 `sanitized` rows read (100 `train`, 100 `test`, 43 `validation`, 7 `prompt`, all at offset 0), 7 `sanitized/test` rows carry a non-empty `test_imports`; none of the other sampled `sanitized` splits do [8]. These are sample-level counts at offset 0, not full-split re-verifications.
- The paper's deciding numbers: its largest model (137B) solves 59.6% of `full` MBPP problems few-shot, and fine-tuning on the 374-row `train` split raises performance by about 10 percentage points across most model sizes tested [1]. On the 100 problems shared between `full` and the edited (`sanitized`) subset, few-shot performance rises from 63% on the original wording to 79% on the edited wording [1].
- No source states a measured contamination or duplicate rate against downstream benchmarks; the dataset itself IS a benchmark, which is why its `test` splits are the hold-out.

## Load it

Both configs are loaded by name; hold out `test` in each. Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; last modified 2024-01-04) [3]:

```python
import datasets

REV = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"  # main at the check date
full_train = datasets.load_dataset("google-research-datasets/mbpp", "full", revision=REV, split="train")            # 374 rows
full_test  = datasets.load_dataset("google-research-datasets/mbpp", "full", revision=REV, split="test")             # 500 rows - hold out
san_train  = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", revision=REV, split="train")       # 120 rows
san_test   = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", revision=REV, split="test")        # 257 rows - hold out
```

**Trap**: `load_dataset("google-research-datasets/mbpp")` with no config name defaults to `full` per the repo's `configs` metadata [3], silently skipping the 427-row `sanitized` config; and within either config, loading only `train` or the default split without also checking `test` risks mixing the benchmark rows into a training run, since both configs carry their own `test` band over the same `task_id` 11-510 range [2][4].

## Neighbors

- `Muennighoff/mbpp` mirrors this repository's data under the same `full`/`sanitized` configs and row counts (974 and 427, confirmed live) [9], but its own Data Splits section states it collapses each config to "only one split each (test)" [10] - so all four `task_id` bands, including the 601-974 training band, are folded into a single split named `test`. That makes it unsuitable for extracting a clean train-only slice without re-deriving the `task_id` bands yourself; prefer this repository's native four-way split for training.
- `evalplus/mbppplus` (EvalPlus's MBPP+) is a 378-row, single-`test`-split benchmark extension: the same `task_id`/`prompt`/`code`/`source_file`/`test_imports`/`test_list` columns as `sanitized`, plus one additional `test` column carrying an expanded, auto-generated set of test asserts per problem [11]. It is licensed Apache-2.0 in its own `cardData`, unlike this repository's CC-BY-4.0 [11]. It is an eval-only augmentation of the `sanitized` test band, not a training source.
- No sibling "clean" or "binarized" preference release was found; MBPP is a code-generation SFT/eval corpus, not a preference dataset.

## A row

Two distinct served shapes, one per config; all four splits within a config share the same columns [6]. `test_setup_code`/`test_imports` are empty in these rows but are non-empty for a minority of other sampled rows (see Quality).

`full` config, `train` split, row 0 (datasets-server `/first-rows`) [8]:

```json
{
  "task_id": 601,
  "text": "Write a function to find the longest chain which can be formed from the given set of pairs.",
  "code": "class Pair(object): \r\n\tdef __init__(self, a, b): \r\n\t\tself.a = a \r\n\t\tself.b = b \r\ndef max_chain_length(arr, n): \r\n\tmax = 0\r\n\tmcl = [1 for i in range(n)] \r\n\tfor i in range(1, n): \r\n\t\t...",
  "test_list": ["assert max_chain_length([Pair(5, ...)], ...)"],
  "test_setup_code": "",
  "challenge_test_list": []
}
```

`sanitized` config, `train` split, row 0 (datasets-server `/first-rows`) [8]:

```json
{
  "source_file": "Benchmark Questions Verification V2.ipynb",
  "task_id": 602,
  "prompt": "Write a python function to find the first repeated character in a given string.",
  "code": "def first_repeated_char(str1):\n  for index,c in enumerate(str1):\n    if str1[:index+1].count(c) > 1:\n      return c",
  "test_imports": [],
  "test_list": [
    "assert first_repeated_char(\"abcabc\") == \"a\"",
    "assert first_repeated_char(\"abc\") == None",
    "assert first_repeated_char(\"123123\") == \"1\""
  ]
}
```

## Where it came from

Built by Google Research as an entirely new crowd-sourced dataset: an internal pool of crowdworkers with basic Python knowledge each wrote a problem statement, a self-contained Python function, three correctness-checking asserts, and a ground-truth solution passing those asserts, with internet-reference use permitted [1]. The paper's authors then manually inspected, edited, and pruned a subset of the questions - fixing function signatures, ambiguity, and test-case accuracy - to produce the smaller `sanitized`/"edited" set [1]. The repository is released alongside the GitHub mirror at https://github.com/google-research/google-research/tree/master/mbpp, whose README states the `task_id` split bands used for evaluation in the paper [1][4].

## Sources

Fetched 2026-08-11; Hub repositories are mutable, which is why Load it pins the revision.

[1] Austin et al., "Program Synthesis with Large Language Models", 2021. https://arxiv.org/abs/2108.07732 - the origin paper: dataset construction, crowdworker instructions, sample-tag breakdown, the edited/sanitized subset description and its 426-question count, the `task_id` band sizes (10/500/374/rest), and the few-shot/fine-tuned performance numbers. Current title read from the live abs page; body read via the ar5iv HTML rendering at https://ar5iv.labs.arxiv.org/html/2108.07732. Fetched 2026-08-11.

[2] google-research-datasets/mbpp dataset page and card. Item home: https://huggingface.co/datasets/google-research-datasets/mbpp . README content read from the raw file at https://huggingface.co/datasets/google-research-datasets/mbpp/raw/main/README.md - licence statement, config/split declarations, data-field descriptions. Read at both `main` and the pinned commit `4bb6404fdc6cacfda99d4ac4205087b89d32030c`; identical. Fetched 2026-08-11.

[3] Hugging Face Hub API record for google-research-datasets/mbpp. https://huggingface.co/api/datasets/google-research-datasets/mbpp?full=true and the `expand[]=downloadsAllTime` variant - licence, gate status, `sha`, `downloads`, `downloadsAllTime`, `likes`, last-modified date, default-config metadata. Fetched 2026-08-11.

[4] google-research/google-research GitHub repository, `mbpp/README.md`. https://github.com/google-research/google-research/tree/master/mbpp - the explicit `task_id` evaluation-split bands (11-510 test, 1-10 prompt, 511-600 validation, 601-974 train) and their inheritance by the sanitized subset; read via the raw file at https://raw.githubusercontent.com/google-research/google-research/master/mbpp/README.md. The repository's root LICENSE (https://raw.githubusercontent.com/google-research/google-research/master/LICENSE) is Apache-2.0 and does not mention MIT. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=google-research-datasets%2Fmbpp - also re-queried with `&revision=4bb6404fdc6cacfda99d4ac4205087b89d32030c`; the two responses are byte-for-byte identical, so the figures below are covered by the Load it pin. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=google-research-datasets%2Fmbpp - also re-queried with `&revision=4bb6404fdc6cacfda99d4ac4205087b89d32030c`; the two responses are byte-for-byte identical. Fetched 2026-08-11.

[7] Hugging Face Hub API models-filter endpoint. https://huggingface.co/api/models?filter=dataset:google-research-datasets/mbpp&limit=20 - live, unpinned list of models tagged as using this dataset. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint, called once per config/split (`full` and `sanitized`, each of `train`/`test`/`validation`/`prompt`). https://datasets-server.huggingface.co/first-rows?dataset=google-research-datasets%2Fmbpp&config=<full|sanitized>&split=<split> - the `full/train` call was also re-queried with `&revision=4bb6404fdc6cacfda99d4ac4205087b89d32030c` and returned an identical row 0, confirming this endpoint is covered by the Load it pin. Fetched 2026-08-11.

[9] datasets-server size endpoint for the mirror. https://datasets-server.huggingface.co/size?dataset=Muennighoff%2Fmbpp - live row counts, not pinned. Fetched 2026-08-11.

[10] Muennighoff/mbpp dataset card (README). https://huggingface.co/datasets/Muennighoff/mbpp/raw/main/README.md - states both configs are collapsed to a single `test` split. Fetched 2026-08-11.

[11] evalplus/mbppplus dataset card (front-matter) and datasets-server size endpoint. https://huggingface.co/datasets/evalplus/mbppplus/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=evalplus%2Fmbppplus - row count, columns, licence. Fetched 2026-08-11.

[12] The corpus screening row for `google-research-datasets/mbpp`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for SFT, with the `test` split of both configs held out. The card's own facts settle it: both `full/test` and `sanitized/test` sit inside the `task_id` 11-510 band that the GitHub README names as the MBPP evaluation set [4], matching the screening row's own note [12].

### The screening row

The row's own note [12]: "crowdsourced basic Python problems; `train`, `validation` and `prompt` are safe, `test` is the MBPP benchmark." The row carries no flag.
