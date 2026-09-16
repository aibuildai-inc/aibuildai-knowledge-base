# mlfoundations-dev/multiple_samples_all_numina_aime

7,500 reasoning-trace SFT rows built from 2,500 AMC/AIME-category competition problems, each problem triplicated with three independently sampled DeepSeek-R1-Distill-Llama-70B responses as three separate training rows.

**mlfoundations-dev/multiple_samples_all_numina_aime** is one repository in a large internal ablation sweep by the mlfoundations-dev org (its Hub organization page lists 5,554 datasets and no papers) [1]; the repository's own README carries only YAML metadata and no prose description [2]. The repository's pipeline definition file shows it is built by loading AI-MO/NuminaMath-CoT, keeping only rows whose `source` field is `amc_aime`, deduplicating on the `problem` text, uniformly sampling 2,500 of the deduplicated problems, duplicating each sampled row three times, and annotating each of the resulting 7,500 rows independently with `together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B` at temperature 1.0, before merging the three per-problem responses back onto each row, running a verification step, and converting to a `conversations` chat format [3]. NuminaMath-CoT's own card describes its `amc_aime` source as part of "US and international mathematics olympiad competition problems" collected from exam PDFs and math forums, with no stated collection cutoff date [4]. **Neither this dataset's card nor NuminaMath-CoT's card states that AIME 2024 or AIME 2025 problems were identified or excluded from the `amc_aime` source pool [2][4]; the corpus screening record flags this as an unresolved contamination risk against those benchmarks, and no source here gives a count of how many rows might overlap [5]. Decontaminate against AIME 2024/2025 (and any other AMC/AIME benchmark in use) before a scored eval run.** It lives at https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_all_numina_aime .

**Use it for**: reasoning-trace SFT via the `conversations` column (ShareGPT-style `human`/`gpt` turns) - the SFT method card. Because each of the 2,500 unique problems is repeated as three separate rows with three different sampled responses as the assistant turn, naively deduplicating on `problem` before training would discard the response diversity that is the point of this release, while training on it as-is repeats the same problem context three times per epoch [2][6].

**Licence**: not stated. This repository's `cardData` carries no `license` field and no `license:` tag [2][7]. The upstream pool, AI-MO/NuminaMath-CoT, is released under Apache License 2.0, but this derived repository does not restate that grant [4].

**Shape**: 7,500 rows, one split (`train`), one config (`default`) [2][8].

**Hold out**: not quantifiable from any fetched source. No source gives a row count or list of rows that overlap AIME 2024, AIME 2025, or any other held-out benchmark; the risk is real (see the restriction above) but its size is not stated anywhere fetched for this card [2][4][5].

**Origin**: built by mlfoundations-dev; problem/solution pairs come from AI-MO/NuminaMath-CoT's `amc_aime` source, and the training-target responses are generations from DeepSeek-R1-Distill-Llama-70B run via the Together AI backend [3]. Hub API at the check date: `downloads` 48, `downloadsAllTime` 544, `likes` 0 [7][9].

**Trained-on-by**: none found. No fetched source names a model or training recipe that used this specific repository.

**Introduced by**: no paper - the dataset's own bare card [2], built per the pipeline config file in the same repository [3]. The upstream pool, NuminaMath-CoT, is likewise introduced only by its own dataset card, which links a non-arXiv report PDF rather than a formal paper [4].

## Shape

Rows and split, from the repository's own declared metadata and the live datasets-server size endpoint, which agree on row count [2][8]:

| split | rows |
| --- | --- |
| `train` | 7,500 |

One config, `default`, with seven columns (datasets-server `/info`, matching the repository's own declared features) [2][10]:

| column | dtype |
| --- | --- |
| `problem` | string |
| `solution` | string |
| `r1_distill_70b_response` | list\<string\> |
| `__original_row_idx` | int64 |
| `_majority_responses` | list\<string\> |
| `verified_r1_distill_70b_response` | string |
| `conversations` | list\<struct\<from: string, value: string\>\> |

Byte sizes carry a pinned-vs-live gap: the repository's own `cardData` (read at the pinned commit `fd813c96fc74e7450bcbda6c8f5c4f1b46f9bbff`) states `download_size` 320,593,358 bytes and an in-memory `dataset_size` of 915,171,337 bytes [2]; the live datasets-server `/info` and `/size` endpoints, which take no revision parameter, report the same 320,593,358-byte download but a smaller in-memory size of 868,701,157 bytes [8][10]. No source states sequence-length or token statistics for this dataset.

## Quality

- The pipeline config shows problems are deduplicated on exact `problem` text before sampling, but no source states a measured duplicate rate on the final 7,500 rows [3].
- No source states a measured correctness or contamination rate for this release. A sibling repository from the same builder, `multiple_samples_majority_consensus_numina_aime_math_verify`, carries an added `r1_distill_70b_extracted_answer` column and 6,659 rows (841 fewer than this release's 7,500), consistent with an answer-verification filter being applied there but not here [11]; this release's own card states no such filter [2].
- Of the ten rows read at offset 0 of `train` (datasets-server first-rows), six (rows 0-5) are served without truncation; the other four (rows 6-9) have `r1_distill_70b_response`, `_majority_responses`, `verified_r1_distill_70b_response`, and `conversations` marked truncated by the server, so the observations below cover only rows 0-5 [6].
- In each of those six rows, `_majority_responses` is element-for-element identical to `r1_distill_70b_response`, i.e. the list is carried twice under two column names rather than holding a distinct value [6].
- Within each of those six rows, the three `r1_distill_70b_response` texts themselves are not identical: for row 0 they run 959, 790, and 872 characters and differ in wording throughout, while all three reach the same final boxed answer; this is a six-row observation, not a corpus-wide rate [6].
- The one sampled problem read (row 0) is itself an AMC 8 problem, not an AIME problem, showing that at least this row's `amc_aime` source label is not AIME-exclusive; NuminaMath-CoT's README gives only a bare row count for the category (4,072) with no breakdown by competition level, so whether the label spans multiple levels generally is not established by this single row [6][4].

## Load it

```python
import datasets

REV = "fd813c96fc74e7450bcbda6c8f5c4f1b46f9bbff"  # main at the check date
train = datasets.load_dataset("mlfoundations-dev/multiple_samples_all_numina_aime", revision=REV, split="train")  # 7,500 rows
```

**Trap**: the 7,500 rows are not 7,500 independent problems - they are 2,500 unique problems, each appearing as three rows that share the same `problem`, `solution`, `r1_distill_70b_response` list, and `_majority_responses` list, but differ in `verified_r1_distill_70b_response` and `conversations`, since each of the three duplicate rows uses a different one of the three sampled responses as its assistant turn [6]. Grouping or deduplicating by `__original_row_idx` recovers the three-response structure; deduplicating by `problem` alone silently collapses it back to 2,500 rows and discards the response diversity the repository was built to hold [3][6].

## Neighbors

This repository is one variant in a same-builder family that shares the same 2,500-problem `amc_aime` pool and largely overlapping columns; every row count below is read from each neighbor's own declared `dataset_info` in its README [12].

- `multiple_samples_none_numina_aime` - 4,070 rows, one response per problem (no triplication, `r1_distill_70b_response` is a plain string not a list); the closest thing to a pre-augmentation baseline [13].
- `multiple_samples_random_numina_aime` and `multiple_samples_shortest_numina_aime` - each 2,500 rows, one row per unique problem, picking one of the three sampled responses (by random choice or by shortest length respectively) instead of keeping all three as separate rows [14][15].
- `multiple_samples_ground_truth_numina_aime` - 1,522 rows, filtered down to problems whose sampled response could be checked against the NuminaMath ground-truth solution [16].
- `multiple_samples_majority_consensus_numina_aime_math_verify` - 6,659 rows, same triplicated-row shape as this release plus an added `r1_distill_70b_extracted_answer` column; a live first-rows fetch on this neighbor shows the same three-duplicate-rows-per-problem pattern as this release, so it is this release's answer-verified variant rather than a differently shaped dataset [11].
- `multiple_samples_all_numina_aime_w_openthoughts` - 125,668 rows across a much wider column set (`system`, `reasoning`, `deepseek_solution`, `code`, `correct`, `judge_reasoning`, etc.), a merge of this `amc_aime` pool with OpenThoughts-style data rather than a same-pool sibling [17].

No source states which of these the corpus prefers when they overlap; a reader who wants the full three-response augmentation without an answer-verification filter uses this release, and one who wants the same pool already checked against ground truth uses `multiple_samples_ground_truth_numina_aime` or the `math_verify` neighbor instead.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`), with the middle of `solution` and of each response truncated [6]:

```json
{
  "problem": "At Euclid Middle School the mathematics teachers are Mrs. Germain, Mr. Newton, and Mrs. Young. There are $11$ students in Mrs. Germain's class, $8$ students in Mr. Newton's class, and $9$ students in Mrs. Young's class taking the AMC $8$ this year. How many mathematics students at Euclid Middle School are taking the contest? \n$\\textbf{(A)}\\ 26 \\qquad\\textbf{(B)}\\ 27\\qquad\\textbf{(C)}\\ 28\\qquad\\textbf{(D)}\\ 29\\qquad\\textbf{(E)}\\ 30$",
  "solution": "1. **Identify the number of students in each class:** [...] \\[\n   11 + 8 + 9 = 28\n   \\]\n\n4. **Conclusion:**\n   The total number of mathematics students at Euclid Middle School taking the AMC 8 contest is $\\boxed{\\textbf{(C)}\\ 28}$.",
  "r1_distill_70b_response": [
    "<think>\nTo determine the total number of mathematics students at Euclid Middle School taking the AMC 8 contest, I will add the number of students from each teacher's class. [...] Therefore, the total number of mathematics students at Euclid Middle School taking the contest is \\(\\boxed{28}\\).",
    "<think>\nFirst, note the number of students in each mathematics teacher's AMC 8 class:\n- Mrs. Germain has 11 students.\n- Mr. Newton has 8 students.\n- Mrs. Young has 9 students. [...] Substituting the given values:\n\n\\[\n\\text{Total students} = 11 + 8 + 9 = 28\n\\]\n\n**Answer:** \\(\\boxed{28}\\)",
    "<think>\nTo determine the total number of mathematics students at Euclid Middle School taking the AMC 8 contest, I'll add the number of students from each teacher's class. [...] Therefore, the total number of mathematics students at Euclid Middle School taking the contest is \\(\\boxed{28}\\)."
  ],
  "__original_row_idx": 12,
  "_majority_responses": [
    "<think>\nTo determine the total number of mathematics students at Euclid Middle School taking the AMC 8 contest, I will add the number of students from each teacher's class. [...] Therefore, the total number of mathematics students at Euclid Middle School taking the contest is \\(\\boxed{28}\\).",
    "<think>\nFirst, note the number of students in each mathematics teacher's AMC 8 class:\n- Mrs. Germain has 11 students.\n- Mr. Newton has 8 students.\n- Mrs. Young has 9 students. [...] Substituting the given values:\n\n\\[\n\\text{Total students} = 11 + 8 + 9 = 28\n\\]\n\n**Answer:** \\(\\boxed{28}\\)",
    "<think>\nTo determine the total number of mathematics students at Euclid Middle School taking the AMC 8 contest, I'll add the number of students from each teacher's class. [...] Therefore, the total number of mathematics students at Euclid Middle School taking the contest is \\(\\boxed{28}\\)."
  ],
  "verified_r1_distill_70b_response": "<think>\nTo determine the total number of mathematics students at Euclid Middle School taking the AMC 8 contest, I will add the number of students from each teacher's class. [...] Therefore, the total number of mathematics students at Euclid Middle School taking the contest is \\(\\boxed{28}\\).",
  "conversations": [
    {"from": "human", "value": "At Euclid Middle School the mathematics teachers are Mrs. Germain, Mr. Newton, and Mrs. Young. [...]"},
    {"from": "gpt", "value": "<think>\nTo determine the total number of mathematics students at Euclid Middle School taking the AMC 8 contest, I will add the number of students from each teacher's class. [...] Therefore, the total number of mathematics students at Euclid Middle School taking the contest is \\(\\boxed{28}\\)."}
  ]
}
```

For this row, `_majority_responses` is element-for-element identical to `r1_distill_70b_response`, and `verified_r1_distill_70b_response` and the `conversations` assistant turn both equal `r1_distill_70b_response[0]` (959 characters); the other two sampled responses run 790 and 872 characters and differ in wording throughout while reaching the same boxed answer [6].

## Where it came from

Built by mlfoundations-dev from AI-MO/NuminaMath-CoT, an approximately 860,000-row math problem set whose sources "range from Chinese high school math exercises to US and international mathematics olympiad competition problems", collected primarily from exam-paper PDFs and math discussion forums and processed through OCR, problem/solution segmentation, translation, chain-of-thought realignment, and answer formatting [4]. This repository's pipeline keeps only the 4,072 rows NuminaMath-CoT labels `amc_aime`, deduplicates on problem text, uniformly samples 2,500 of them, triplicates each, and generates one response per duplicate row from `DeepSeek-R1-Distill-Llama-70B` (via the Together AI API, temperature 1.0, up to 32,000 output tokens), before merging the three responses back per problem, running a verification step, and converting problem/response pairs into the `conversations` ShareGPT format [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hugging Face Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the datasets-server endpoints cited take no revision parameter and are reported as live, not pinned.

[1] Hugging Face Hub organizations overview API for mlfoundations-dev. https://huggingface.co/api/organizations/mlfoundations-dev/overview - dataset count, paper count. Fetched 2026-08-12.

[2] mlfoundations-dev/multiple_samples_all_numina_aime dataset card (README), YAML-only. https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_all_numina_aime/raw/main/README.md - declared features, splits, byte sizes, configs; no license field, no prose. Fetched 2026-08-12.

[3] Pipeline config file in the same repository. https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_all_numina_aime/raw/main/config/multiple_samples_all_numina_aime.yaml - build steps: source dataset, source filter, dedup, sampling, triplication, annotation model and backend, merge, verification, ShareGPT conversion. Fetched 2026-08-12.

[4] AI-MO/NuminaMath-CoT dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/raw/main/README.md - dataset description, collection method, source breakdown table (`amc_aime`: 4,072), Apache-2.0 license. Fetched 2026-08-12.

[5] The corpus screening row for `mlfoundations-dev/multiple_samples_all_numina_aime`, supplied with this card's request - its `flag`, read back in the row's own words in the appendix. Checked 2026-08-12.

[6] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mlfoundations-dev%2Fmultiple_samples_all_numina_aime&config=default&split=train - the ten rows read at offset 0, used for the sampled row, the triplicate-row structure, and the AMC-8-not-AIME observation. Fetched 2026-08-12.

[7] Hugging Face Hub API record for mlfoundations-dev/multiple_samples_all_numina_aime. https://huggingface.co/api/datasets/mlfoundations-dev/multiple_samples_all_numina_aime?full=true - `sha`, `downloads`, `likes`, `gated`, `private`, `cardData`, `lastModified`, `createdAt`, tags. Fetched 2026-08-12.

[8] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fmultiple_samples_all_numina_aime Fetched 2026-08-12.

[9] Hugging Face Hub API record with all-time downloads. https://huggingface.co/api/datasets/mlfoundations-dev/multiple_samples_all_numina_aime?expand[]=downloadsAllTime - `downloadsAllTime`. Fetched 2026-08-12.

[10] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=mlfoundations-dev%2Fmultiple_samples_all_numina_aime Fetched 2026-08-12.

[11] `multiple_samples_majority_consensus_numina_aime_math_verify` dataset card and first-rows. https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify/raw/main/README.md and https://datasets-server.huggingface.co/first-rows?dataset=mlfoundations-dev%2Fmultiple_samples_majority_consensus_numina_aime_math_verify&config=default&split=train - row count, columns, triplicate-row structure. Fetched 2026-08-12.

[12] Hugging Face Hub dataset search for the mlfoundations-dev org filtered to "multiple_samples". https://huggingface.co/api/datasets?author=mlfoundations-dev&search=multiple_samples&limit=100 - the family of same-pool sibling repository ids. Fetched 2026-08-12.

[13] `multiple_samples_none_numina_aime` dataset card (README). https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_none_numina_aime/raw/main/README.md Fetched 2026-08-12.

[14] `multiple_samples_random_numina_aime` dataset card (README). https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_random_numina_aime/raw/main/README.md Fetched 2026-08-12.

[15] `multiple_samples_shortest_numina_aime` dataset card (README). https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_shortest_numina_aime/raw/main/README.md Fetched 2026-08-12.

[16] `multiple_samples_ground_truth_numina_aime` dataset card (README). https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_ground_truth_numina_aime/raw/main/README.md Fetched 2026-08-12.

[17] `multiple_samples_all_numina_aime_w_openthoughts` dataset card (README). https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_all_numina_aime_w_openthoughts/raw/main/README.md Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as SFT reasoning-trace data with a use-time restriction, not a rejection: the dataset's own card and its upstream source's card together establish that the `amc_aime` pool this repository draws from carries no stated collection cutoff and no stated AIME 2024/2025 decontamination step, so this is a benchmark-overlap risk to resolve before a scored run rather than a defect in the training data itself [2][4]. That restriction is stated in bold in the opening paragraph and reflected in the Hold out line above.

### The screening row

The row's own flag [5]: "rule-risk: aime2025 - Rows are Numina amc_aime competition problems with R1-Distill samples; card is bare and claims no AIME removal". The row's note [5]: "Numina amc_aime historical problems with several R1-Distill-70B samples each plus the human reference solution."
