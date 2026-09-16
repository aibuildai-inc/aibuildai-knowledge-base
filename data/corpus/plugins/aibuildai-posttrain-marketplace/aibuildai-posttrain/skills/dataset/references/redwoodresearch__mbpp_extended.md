# redwoodresearch/mbpp_extended

38,215 single-turn Python coding problems, each with a GPT-4-written solution and a list of assert-style test cases - a machine-generated expansion of the MBPP benchmark, not a copy of it.

**redwoodresearch/mbpp_extended** is released by Redwood Research at https://huggingface.co/datasets/redwoodresearch/mbpp_extended . Its own dataset card carries no description beyond a placeholder title [1], but the row count identifies it: "Benchmarks for Detecting Measurement Tampering" (Roger et al., Redwood Research) states that the function-correctness dataset it builds draws its problems from three sources - APPS (10,000 problems), MBPP (974 problems), and "AI generated programs inspired by easy MBPP problems (38215 problems)" - the last figure matching this repository's row count exactly [2]. The paper further states that those AI-generated problems' solutions are "0-shot GPT-4 programs", which matches this repository's `gpt4_solution` column, and that their test cases are "AI-generated (inputs were generated, outputs were computed using the solutions)", matching the `test_cases` column [2]. **Because every row is a variant "inspired by" an MBPP problem, and the paper's 974-problem MBPP pool spans MBPP's train, test, validation and prompt splits together, training on this dataset risks contaminating any evaluation run on the MBPP benchmark - see Hold out.**

**Use it for**: SFT on single-turn code-generation (problem description in, function body out) - maps to the SFT method card's plain-text format, one prompt/completion pair per row (`description` as prompt, `gpt4_solution` as completion). **Do not use for evaluation, and hold out MBPP-benchmark evaluation runs for any model trained on it** (see Hold out).

**Licence**: not stated - the dataset card's YAML has no `license` field and the Hub API's `cardData` carries none [1][3]; repository is ungated (`gated: false`) [3].

**Shape**: one config (`default`), one split (`train`, 38,215 rows), four columns [1][4].

**Hold out**: this repository has no internal eval split to hold out (train-only). The risk is external: because every row is a GPT-4-generated variant of an "easy" MBPP problem [2], and MBPP's 974-problem pool spans all of MBPP's splits including its 500-row `test` split [5], any evaluation using google-research-datasets/mbpp's benchmark splits should be treated as at risk of contamination for a model trained on this data. A sample check found no exact overlap (see Quality) but did not rule it out.

**Origin**: builder Redwood Research; problem descriptions are machine-paraphrased from MBPP and solutions are GPT-4 completions, per the paper that matches this repository's row count [2]. Hub API at the check date: `downloads` 343, `downloadsAllTime` 2,151, `likes` 2 [3][6].

**Trained-on-by**: the origin paper's own measurement-tampering-detection experiments consume this AI-generated slice as part of a larger derived training set (redwoodresearch/function_correctness, 113,560 train / 11,706 validation rows), built by pairing these problems and GPT-4 solutions with pass/fail test labels; the paper reports 230,000 of its 280,000 function-correctness examples come from this AI-generated pool [2]. No other trained-on-by adoption found.

**Introduced by**: [2] (Roger et al.), which names this exact problem count and generation method; no separate dataset paper or blog post is linked from the dataset card itself [1].

## Shape

Splits and bytes (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 38,215 |

Columns (datasets-server `/info`) [4][7]:

| column | dtype |
| --- | --- |
| `description` | string |
| `gpt4_solution` | string |
| `function_name` | string |
| `test_cases` | list\<string\> |

The repository's own `cardData.dataset_info` states 50,571,642 bytes in memory for `train` [1], but the live datasets-server `/size` endpoint reports 37,967,710 bytes in memory for the same split, with the on-disk Parquet size agreeing at 11,623,252 bytes in both [1][4] - the static card metadata and the live served bytes disagree on the in-memory figure; both are reported here rather than picking one. No source states token or sequence-length statistics for this dataset.

## Quality

- The paper states the GPT-4-written solutions (`gpt4_solution`) are the ground truth used to grade this data, with an estimated error rate under roughly 5% [2].
- The `description` field is prefixed with the literal text "(AI generated)" in the rows read (see A row below); this prefix, and the paper's own Table 5 of "Random examples of AI generated problems" using the same wording, indicate the descriptions are LLM-generated MBPP-style variants rather than copied MBPP text [2].
- To check for literal overlap with MBPP: a sample of 100 rows read from this dataset's `train` split at offset 0 yielded 44 distinct `function_name` values (`replace_vowels`, `median`, `kebab_to_snake`, `most_common_ngrams`, and 40 others) [8]. Against this, function names were extracted from MBPP's `code` field (every `def` statement, since some MBPP solutions define helper functions alongside the main one) across all of MBPP's `validation` (90 rows) and `prompt` (10 rows) splits plus the first 100 rows each of `train` (of 374) and `test` (of 500) [8]; none of those 100+100+90+10=300 MBPP rows contained a `def` matching any of the 44 names above. Two MBPP problems on related themes were found (`count vowels in a string`, task 667; `median of two sorted arrays`, task 622) but neither matches the sampled `replace_vowels` or `median` problems in this dataset word for word. This check covers only the rows read; it does not rule out overlap in the remaining 674 of MBPP's `test`+`train` rows or the remaining 38,115 rows of this dataset.
- No source states a measured duplicate rate or annotator-agreement figure for this dataset.

## Load it

```python
import datasets

REV = "5151e1ee7f2f8c550a73bd182258d10834a619d0"  # main at the check date
train = datasets.load_dataset("redwoodresearch/mbpp_extended", revision=REV, split="train")  # 38,215 rows
```

**Trap**: there is only one split (`train`); the dataset ships no held-out split of its own, so any evaluation split must be built by the user, and any such split should still be checked against MBPP's own splits given the contamination risk above.

## Neighbors

- redwoodresearch/function_correctness - the derived, larger dataset (113,560 train / 11,706 validation rows) that the origin paper builds by combining this AI-generated problem set with APPS and MBPP problems, pairing each with generated correct and incorrect functions and per-test-case pass/fail labels; its own dataset card says it is built from MBPP and APPS, with most of its problems inspired by MBPP and generated by gpt-3.5-turbo, and most of its functions generated by gpt-3.5-turbo and gpt-4 [9]. It is MIT-licensed, unlike this repository's unstated licence [9]. Use this repository for raw problem/solution pairs; use `function_correctness` only for measurement-tampering-style correctness classification, not general SFT.
- google-research-datasets/mbpp - the original benchmark this repository's problems are "inspired by" [2]; 974 rows across `train`/`test`/`validation`/`prompt` splits in its `full` config, CC-BY-4.0 [5][10]. This is the benchmark to hold out, not a training-data neighbor.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [4]:

```json
{
  "description": "(AI generated) Write a function that takes a string and replaces every vowel with a specified character, given as a parameter to the function. If the input string is empty, return an empty string.",
  "gpt4_solution": "def replace_vowels(s, ch):\n    vowels = \"aeiouAEIOU\"\n    if not s:\n        return \"\"\n    else:\n        return \"\".join([ch if c in vowels else c for c in s])",
  "function_name": "replace_vowels",
  "test_cases": [
    "assert (replace_vowels(\"hello world\", \"*\") == 'h*ll* w*rld')",
    "assert (replace_vowels(\"aeiou\", \"x\") == 'xxxxx')",
    "assert (replace_vowels(\"\", \"x\") == '')",
    "assert (replace_vowels(\"bcd\", \"x\") == 'bcd')",
    "assert (replace_vowels(\"Hello World\", \"*\") == 'H*ll* W*rld')"
  ]
}
```

## Where it came from

Built by Redwood Research. The dataset's own card states only a placeholder title with no construction detail [1]. The matching row count and column semantics in "Benchmarks for Detecting Measurement Tampering" identify the construction: problem descriptions are GPT-generated variants "inspired by easy MBPP problems", each paired with a 0-shot GPT-4 solution treated as ground truth, and with test cases whose inputs were AI-generated and whose expected outputs were computed by running the GPT-4 solution [2]. The upstream pool, MBPP, is itself a set of "around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers", introduced in "Program Synthesis with Large Language Models" [5].

## Sources

Checked 2026-08-11; Hub repositories are mutable, which is why Load it pins the revision `5151e1ee7f2f8c550a73bd182258d10834a619d0`.

[1] redwoodresearch/mbpp_extended dataset card (README). https://huggingface.co/datasets/redwoodresearch/mbpp_extended/raw/main/README.md - YAML `dataset_info`, placeholder body text, no license field. Fetched 2026-08-11.

[2] Roger, Greenblatt, Nadeau, Shlegeris, and Thomas, "Benchmarks for Detecting Measurement Tampering", 2023. https://arxiv.org/abs/2308.15605 - Section 2.4.2 "Data generation" (problem-count breakdown, generation method for the AI-generated MBPP-inspired problems, GPT-4 solution error rate) and Appendix Q (Table 5, example AI-generated problems with the "(AI generated)" prefix). Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2308.15605). Fetched 2026-08-11.

[3] Hugging Face Hub API record for redwoodresearch/mbpp_extended. https://huggingface.co/api/datasets/redwoodresearch/mbpp_extended?full=true - `gated`, `sha`, `downloads`, `likes`, `cardData` (no license key), `lastModified`. Fetched 2026-08-11.

[4] datasets-server size, info, and first-rows endpoints for redwoodresearch/mbpp_extended. https://datasets-server.huggingface.co/size?dataset=redwoodresearch%2Fmbpp_extended , https://datasets-server.huggingface.co/info?dataset=redwoodresearch%2Fmbpp_extended , https://datasets-server.huggingface.co/first-rows?dataset=redwoodresearch%2Fmbpp_extended&config=default&split=train - split row count, in-memory/Parquet bytes, columns, row 0. Fetched 2026-08-11.

[5] google-research-datasets/mbpp dataset card (README) and Austin et al., "Program Synthesis with Large Language Models", 2021, https://arxiv.org/abs/2108.07732 - MBPP's description, split structure, and licence, and the origin-paper citation the card itself gives. README fetched from https://huggingface.co/datasets/google-research-datasets/mbpp/raw/main/README.md. Fetched 2026-08-11.

[6] Hugging Face Hub API record for redwoodresearch/mbpp_extended with `downloadsAllTime` expanded. https://huggingface.co/api/datasets/redwoodresearch/mbpp_extended?expand%5B%5D=downloadsAllTime Fetched 2026-08-11.

[7] datasets-server info endpoint, same call as [4], cited separately for the column dtype table. Fetched 2026-08-11.

[8] datasets-server rows endpoint, one call per split, for the function-name overlap check: redwoodresearch/mbpp_extended `train` (offset 0, length 100), google-research-datasets/mbpp `full` config `test` (offset 0, length 100), `train` (offset 0, length 100), `validation` (offset 0, length 90, all rows), `prompt` (offset 0, length 10, all rows). https://datasets-server.huggingface.co/rows?dataset=<id>&config=<config>&split=<split>&offset=<n>&length=<n> - this endpoint takes no revision parameter, so these reads are live, not pinned. Fetched 2026-08-11.

[9] redwoodresearch/function_correctness dataset card (README) and Hub API record. https://huggingface.co/datasets/redwoodresearch/function_correctness/raw/main/README.md , https://huggingface.co/api/datasets/redwoodresearch/function_correctness?full=true - construction description, licence (MIT), split sizes. Fetched 2026-08-11.

[10] datasets-server size endpoint for google-research-datasets/mbpp. https://datasets-server.huggingface.co/size?dataset=google-research-datasets%2Fmbpp - `full` config split row counts (train 374, test 500, validation 90, prompt 10). Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for SFT, but carrying a real, only-partially-checked contamination risk: every row is a GPT-generated variant "inspired by" an MBPP problem [2], and MBPP's 974-problem source pool spans MBPP's `train`, `test`, `validation`, and `prompt` splits together [5]. A 100-row function-name sample found no exact overlap with a matching sample of MBPP [4][8], but this does not clear the remaining rows on either side. Any consumer of this dataset should hold out MBPP-benchmark evaluation for models trained on it, matching the screening row's flag.

### The screening row

The row's own note, supplied with this card's request: "MBPP-style problems with a `gpt4_solution` column and test cases." Its flag: "contamination risk — built on MBPP, which is a benchmark and is also in this range as google-research-datasets/mbpp."
