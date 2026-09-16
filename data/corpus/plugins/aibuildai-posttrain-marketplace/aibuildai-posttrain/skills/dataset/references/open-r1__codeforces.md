# open-r1/codeforces

Over 10,000 unique CodeForces competitive-programming problems, from the earliest contests through 2025, packaged with problem statements, official test cases, editorials, and Hugging Face-generated checkers and extra test cases for problems that were not otherwise fully verifiable.

**open-r1/codeforces** was built by the Hugging Face open-r1 team from public CodeForces contest pages, covering the platform's full history into 2025 [1]. Because CodeForces truncates any displayed test case beyond about 400 characters and about 30% of problems accept multiple correct answers requiring a custom checker, the card's authors call most existing "verifiable" CodeForces collections not actually verifiable, and built this release specifically to fix that: they ran real human contestant solutions to confirm each problem is executable, used DeepSeek-R1 to generate and validate custom checkers (20-100 attempts per problem) against those human solutions, and used DeepSeek-R1 again to write test-case generators that produce additional large, hard test cases validated by agreement across multiple correct human solutions [1]. **The card ships a `test` split of problems from late 2024 and early 2025 and asks readers to avoid training on it** [1]; NVIDIA's OpenCodeReasoning explicitly excluded this same `test` split when building its own SFT data from this dataset [2]. It lives at https://huggingface.co/datasets/open-r1/codeforces .

**Use it for**: reasoning-trace SFT or RL-style code generation - the `verifiable-prompts` config's `prompt` field is a ready-to-use instruction ("reason step by step... provide a complete implementation") in either Python or C++ per problem, meant to be paired with a model-generated solution and checked against `official_tests`/`generated_tests` via the card's own execution harness [1]. Hold out `test` in every config. The card does not map to a single method shape by itself: it supplies problems and verifiers, not preference pairs or chat turns, so treat it as the reasoning/RL problem-set counterpart to a code-generation SFT or RL method card, not the SFT method card itself.

**Licence**: the Hub metadata field and repo tag both say `cc-by-4.0` [3], but the card's own License section states the data is licensed under the Open Data Commons Attribution License (ODC-By) 4.0, a different named licence instrument [1]. Ungated (`"gated": false`, `"private": false`) [3]. Report both: the machine-readable tag says CC-BY 4.0, the prose says ODC-By 4.0.

**Shape**: three configs sharing most of a 27/29-column schema - `default` (10,024 rows: 9,556 train / 468 test), `verifiable` (8,760 rows: 8,338 train / 422 test, a filtered subset of `default`), `verifiable-prompts` (17,520 rows: 16,676 train / 844 test, `verifiable` with two prompt rows - one Python, one C++ - per problem) [1]. The live datasets-server index is flagged `partial: true` and currently serves only 29,286 of these 36,304 rows, entirely because `verifiable-prompts/train` is under-indexed (9,658 of 16,676 served); `default` and `verifiable` are both fully served [4][5].

**Hold out**: `test` in every config - `default/test` (468 rows), `verifiable/test` (422 rows), `verifiable-prompts/test` (844 rows) - all drawn from late-2024/early-2025 contests per the card's own Splits section [1].

**Origin**: built by the Hugging Face open-r1 team; problem text and official tests come from CodeForces, checkers and extra test cases are DeepSeek-R1 generations, LaTeX-in-image OCR uses Qwen2.5-VL-7B-Instruct [1]. Hub API at the check date: `downloads` 11,499, `downloadsAllTime` 207,957, `likes` 101 [3].

**Trained-on-by**: NVIDIA's OpenCodeReasoning lists `open-r1/codeforces` alongside TACO, APPS, and CodeContests as one of the sources its overall question collections were gathered from, and separately states that it excluded the `test` split of both CodeContests and this dataset [2]; OpenCodeReasoning's own CodeForces slice covers 10,069 questions and 386,948 R1-generated samples [2]. No other adoption by a named model or training recipe was found.

**Introduced by**: no paper - the dataset card [1] gives a citation entry pointing back to the Hugging Face repository itself, with no separate arXiv record.

## Shape

Rows declared in the card's own `dataset_info` YAML (the `rows_declared_source` for this card) versus rows the live datasets-server index actually serves [1][4][5]:

| config | split | declared (card) | served (live, partial) |
| --- | --- | --- | --- |
| `default` | train | 9,556 | 9,556 |
| `default` | test | 468 | 468 |
| `verifiable` | train | 8,338 | 8,338 |
| `verifiable` | test | 422 | 422 |
| `verifiable-prompts` | train | 16,676 | 9,658 |
| `verifiable-prompts` | test | 844 | 844 |

Only `verifiable-prompts/train` is under-served by the live index; every other split matches the card's declared count exactly. The `default` and `verifiable` configs share 27 columns (listed in the shortlist row); `verifiable-prompts` adds `prompt` (string) and `language` (string) on top of the same 27, one row per problem per language [1][6]. `verifiable` is `default` filtered to problems that are `executable` and have either `official_tests_complete` or at least one `generated_tests` row [1] - 8,760 of the roughly 10,000 `default` problems, the number that decides which config gives execution-checkable problems and which does not.

No source states token or sequence-length statistics for any config. Byte sizes from the card's own `dataset_info`: `default` 2.76 GB download / 5.47 GB decoded, `verifiable` 2.46 GB / 4.78 GB, `verifiable-prompts` 4.94 GB / 9.58 GB, summing to about 10.15 GB of parquet download across the three configs [1]. The live (partial) datasets-server index reports a smaller total of 7.92 GB of parquet because `verifiable-prompts/train` is only partly converted there [4] - `load_dataset` against the repository's actual parquet files is not limited by that partial index (see Load it).

The repository separately ships a `generated_tests/` directory of per-contest parquet files (`test_cases_XXXX.parquet`, columns `problem_id`, `input`, `output`, `test_case_i`) holding the DeepSeek-R1-generated large test cases; the card says these are about 110 GB and must be downloaded separately with `huggingface-cli download ... --include='generated_tests/*.parquet'`, joined back to the main table on `problem_id` = the main table's `id` [1].

## Quality

- About 30% of problems require a custom checker because they accept multiple correct answers; the card's checkers were generated by DeepSeek-R1 in 20-100 attempts per problem and kept only once one checker correctly validated real human contestant solutions [1].
- About 60% of problems include an editorial (the contest organizers' own solution writeup) [1].
- Generated test cases went through a four-solution-agreement filter: multiple correct human solutions had to agree on the output for a generated case to be kept, and cases were then ranked by difficulty (how many known-incorrect solutions still pass them) and by input size before the hardest and a spread of others were selected [1].
- `official_tests_complete` marks whether every CodeForces-displayed test case for a problem escaped the platform's ~400-character truncation; `executable` marks whether at least three real human solutions were confirmed runnable against `official_tests` [1]. Neither figure is aggregated into a dataset-wide rate by any source read here.
- No source states a measured duplicate-row or contamination rate for this release; the `aliases` column exists specifically because the same problem commonly appears as a copy across a Div. 1 and Div. 2 contest pair, per the card's own field description [1].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main`, last modified 2025-05-19) [3]:

```python
from datasets import load_dataset

REV = "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"  # main at the check date
train = load_dataset("open-r1/codeforces", name="verifiable", split="train", revision=REV)  # 8,338 rows
test = load_dataset("open-r1/codeforces", name="verifiable", split="test", revision=REV)     # 422 rows - hold out
```

**Trap**: the card's own "Loading the dataset" code block calls `load_dataset("open-r1/codeforces-submissions", ...)` - that is a different, 12.6-million-row repository of raw human contestant submissions, not this problem set [1][7]. Load `"open-r1/codeforces"` (this repository's id), not `"open-r1/codeforces-submissions"`. Second trap: the live datasets-server preview endpoints report `partial: true` and under-serve `verifiable-prompts/train` (9,658 of 16,676) [4][5], but that partial flag describes the server's own preview index, not the repository's actual parquet shards - `load_dataset` reads those shards directly and returns the full declared row count for every split, including all 16,676 rows of `verifiable-prompts/train`.

## Neighbors

- `open-r1/codeforces-submissions` - the companion dataset of raw human contestant submissions (12,591,518 rows in its `default` config as read live at the check date) that this card's checkers and executability checks were validated against; not a problem set and not interchangeable with this card [1][7].
- `deepmind/code_contests` - named by NVIDIA's OpenCodeReasoning as the other CodeForces-adjacent source it drew on alongside this dataset, also with its own `test` split excluded when OpenCodeReasoning was built [2]; not fetched here, so no row-level comparison is given.

## A row

Two distinct served shapes: `default`/`verifiable` share one 27-column schema, and `verifiable-prompts` adds `prompt` and `language`.

From `config="default"`, `split="train"`, `row_idx=0` (also present, unmodified, in `verifiable`) [8]. `official_tests` holds five entries in the source row; only the first is shown here:

```json
{
  "id": "852/A",
  "aliases": null,
  "contest_id": "852",
  "contest_name": "Bubble Cup X - Finals [Online Mirror]",
  "contest_type": "ICPC",
  "contest_start": 1504432800,
  "contest_start_year": 2017,
  "index": "A",
  "time_limit": 1.0,
  "memory_limit": 256.0,
  "title": "Digits",
  "description": "John gave Jack a very hard problem. He wrote a very big positive integer A0 on a piece of paper. [...] Jack must not add leading zeros to intermediate results, but he can put ' + ' signs in front of digit 0. [...]",
  "input_format": "First line contains a positive integer N (1 ≤ N ≤ 200000), representing the number of digits of A0. [...]",
  "output_format": "Output exactly three lines, the steps Jack needs to perform to solve the problem. [...]",
  "interaction_format": null,
  "note": "In the first sample, Jack can't put ' + ' signs anywhere, so he just writes 1 in each line and solves the problem. [...]",
  "examples": [{"input": "1\n1", "output": "1\n1\n1"}, {"input": "4\n5806", "output": "5+8+0+6\n1+9\n1+0"}],
  "editorial": null,
  "rating": 2500,
  "tags": ["brute force", "implementation", "math"],
  "testset_size": 46,
  "official_tests": [{"input": "1\r\n1\r\n", "output": "1\r\n1\r\n1\r\n"}],
  "official_tests_complete": false,
  "input_mode": "stdio",
  "generated_checker": "import sys\n\ndef main(input_path, output_path, submission_path): [...]",
  "executable": true,
  "generated_tests": 19
}
```

From `config="verifiable-prompts"`, `split="train"`, `row_idx=0`, the same problem row (all 27 fields above, unchanged) plus the two added columns [8]:

```json
{
  "language": "cpp",
  "prompt": "You are an expert competitive programmer. You will be given a problem statement, test case constraints and example test inputs and outputs. Please reason step by step about the solution (that must respect memory and time limits), then provide a complete implementation in c++17. [...] Execution time limit: 1.0 seconds\nMemory limit: 256.0 MB\n\n# Problem\nJohn gave Jack a very hard problem. [...]"
}
```

Of the first ten rows served at that offset, all ten carry `language: "cpp"` - the card states each problem gets one Python and one C++ row, but a Python-language row was not seen in this ten-row read [1][8].

## Where it came from

Problem statements, contest metadata, official test cases, and editorials come from public CodeForces contest pages, collected by the Hugging Face open-r1 team [1]. LaTeX equations that CodeForces renders as images were converted to text with `Qwen/Qwen2.5-VL-7B-Instruct` [1]. Real human contestant solutions (hosted separately in `open-r1/codeforces-submissions`) were run against each problem to confirm executability, to generate DeepSeek-R1 custom checkers for problems with multiple valid answers, and to validate DeepSeek-R1-generated additional test cases by requiring agreement across several correct solutions [1].

## Sources

Fetched on the check date, 2026-08-11. Hugging Face Hub repositories are mutable, which is why Load it pins the revision; datasets-server endpoints (`/size`, `/info`, `/first-rows`) take no revision parameter and so are reported as live, not pinned, values.

[1] open-r1/codeforces dataset card (README). https://huggingface.co/datasets/open-r1/codeforces/raw/main/README.md - description, verifiability rationale, subset definitions, field descriptions, loading instructions, license section, citation. Fetched 2026-08-11.

[2] nvidia/OpenCodeReasoning dataset card (README). https://huggingface.co/datasets/nvidia/OpenCodeReasoning/raw/main/README.md - naming open-r1/codeforces as a source, stating the test split was excluded, CodeForces question/sample counts. Fetched 2026-08-11. Technical report: Ahmad et al., "OpenCodeReasoning: Advancing Data Distillation for Competitive Coding", 2025. https://arxiv.org/abs/2504.01943 - current title read from the live abs page. Fetched 2026-08-11.

[3] Hugging Face Hub API record for open-r1/codeforces. https://huggingface.co/api/datasets/open-r1/codeforces?full=true - license tag, gated/private status, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=open-r1%2Fcodeforces - per-split row and byte counts, `partial: true` flag. Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=open-r1%2Fcodeforces - config feature lists, shard lengths, `partial: true` flag. Fetched 2026-08-11.

[6] The shortlist row's own `columns` field, supplied with this card's request, cross-checked against the card's `dataset_info` YAML [1] for the shared 27-column schema and the two extra `verifiable-prompts` columns.

[7] Hugging Face Hub API record and datasets-server size endpoint for open-r1/codeforces-submissions. https://huggingface.co/api/datasets/open-r1/codeforces-submissions?full=true and https://datasets-server.huggingface.co/size?dataset=open-r1%2Fcodeforces-submissions - confirms it is a distinct, 12.6-million-row repository. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint, one call per config/split read for this card: `config=default&split=train`, `config=verifiable&split=train`, `config=verifiable-prompts&split=train`, `config=default&split=test`. https://datasets-server.huggingface.co/first-rows?dataset=open-r1%2Fcodeforces&config=<config>&split=<split> Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as a problem-and-verifier set for reasoning/RL code generation, with the `test` split held out in every config. The card's own instruction to avoid training on `test` [1] and NVIDIA's independent decision to exclude the same split when reusing this dataset [2] agree with the screening row's note.

### The screening row

The row's own note: "CodeForces problems with real contestant solutions plus DeepSeek-R1-generated test-case generators and checkers; it ships a `test` split that NVIDIA deliberately excluded when building OpenCodeReasoning, so train on `train`/`verifiable` only." The row carries no flag.
