# mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify

6,659 DeepSeek-R1-Distill-Llama-70B math reasoning traces on NuminaMath's AMC/AIME competition-problem pool, kept only where the trace's Math-Verify-extracted final answer agrees with the majority answer among three temperature-1.0 samples of the same problem.

**mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify** carries no descriptive prose in its own README - the file is YAML front matter only, with no paper linked and no dataset summary [1]. Its build recipe is instead recorded in a `config/` YAML shipped in the same repository: load `AI-MO/NuminaMath-CoT`'s train split, keep only rows whose `source` field is `amc_aime` (a 4,072-row category inside that 860k-row corpus, which the NuminaMath card does not break down further [2]), keep just the `problem`/`solution` columns, deduplicate on `problem`, uniformly sample 2,500 problems, duplicate each row three times, generate one response per duplicate with `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` at temperature 1.0, extract each response's final answer with the Math-Verify library, keep only the duplicate rows whose extracted answer matches the majority answer among the three duplicates of that problem, and convert the surviving `problem`/verified-response pairs into a two-turn `conversations` list [3]. The repository is one of 27 sibling releases from the same author built on this identical AMC/AIME pool with varying sampling, filtering and verification choices [4]. **Because every sibling draws from the same Numina `amc_aime` pool, and this dataset's majority-consensus/Math-Verify step filters out minority (likely-wrong) answers rather than filtering out AIME items specifically, the corpus screening record flags this release for possible overlap with AIME evaluation benchmarks (named "aime2025" in the flag) - decontaminate against whatever AIME/AMC benchmark you plan to score on before a scored run [5].** It lives at https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify .

**Use it for**: reasoning-trace SFT - the `conversations` column is already a human/gpt two-turn pair (problem, then the verified reasoning trace and final answer) - decontaminated first against any AIME/AMC evaluation set before a scored run [5]. Maps to the SFT method card's chat format directly; no reformatting beyond loading `conversations` is needed.

**Licence**: not stated by this repository - no `license` field in its `cardData` and no `license:` tag on the repo [6]. The upstream `AI-MO/NuminaMath-CoT` pool this was built from is Apache-2.0 [2], but this derivative repository does not itself carry a licence.

**Shape**: 6,659 rows, one config (`default`), one split (`train`) [7][8].

**Hold out**: this repository ships no eval split of its own to hold out. The row-level risk is external: decontaminate against AIME/AMC evaluation sets before scoring on them, per the screening flag discussed above and repeated in the appendix [5].

**Origin**: built by mlfoundations-dev; the reasoning traces are DeepSeek-R1-Distill-Llama-70B generations, and the keep/drop decision on each trace is an automatic majority-vote-plus-Math-Verify check, not a human judgment [3]. Hub API at the check date: `downloads` 109, `downloadsAllTime` 600, `likes` 0 [6].

**Trained-on-by**: none found for this exact Math-Verify variant. mlfoundations-dev's own Llama-3.1-8B model `llama3-1_8b_multiple_samples_majority_consensus_numina_aime` was trained on the sibling dataset that filters by the same majority-consensus logic but extracts answers with an LLM (`gpt-4o-mini`) instead of Math-Verify [9] - evidenced by a TensorBlock GGUF mirror whose `base_model` field names that mlfoundations-dev repo; the original training repository has since been deleted from the Hub [10].

**Introduced by**: no paper - the dataset card [1] (front matter only, no prose, no citation block).

## Shape

Columns and dtypes (datasets-server `/info`) [8]:

| column | dtype |
| --- | --- |
| `problem` | string |
| `solution` | string |
| `r1_distill_70b_response` | list\<string\> |
| `__original_row_idx` | int64 |
| `r1_distill_70b_extracted_answer` | list\<string\> |
| `_majority_responses` | list\<string\> |
| `verified_r1_distill_70b_response` | string |
| `conversations` | list\<struct\<from: string, value: string\>\> |

One split, `train`, 6,659 rows (datasets-server `/size`) [7]: 244,788,482 bytes of Parquet download, 718,888,253 bytes decoded in memory. No source states sequence-length or token statistics for this release; not stated.

## Quality

- The filtering is fully automatic: an LLM generates each candidate trace, the Math-Verify library extracts each trace's final answer, and a majority-vote check across the three same-problem duplicates decides which individual traces survive - no human review step is stated anywhere in the build config or README [3][1].
- Reading the first three served rows (`row_idx` 0-2, all sharing `__original_row_idx` 12) shows the mechanism concretely: all three rows carry an identical 3-item `r1_distill_70b_response` list and an identical `r1_distill_70b_extracted_answer` of `['28','28','28']`, but each row's `verified_r1_distill_70b_response` is a different one of the three list items - row 0's verified field is an exact match to `r1_distill_70b_response[0]` [11]. So each duplicate row keeps one of the three samples as its target completion, and rows survive only when that particular sample's extracted answer matches the per-problem majority.
- No source states a measured error rate, duplicate rate, or agreement rate for this dataset; none is invented here.
- Of the first ten served rows, `__original_row_idx` 12 and 1 (row_idx 0-5) each carry a `_majority_responses` list of length 3; for `__original_row_idx` 15 and 18 (row_idx 6-9) the datasets-server response marks `_majority_responses` (along with several other fields) as truncated, serving a 100-character cut of the field instead of the full value, so the true list length for those four rows was not read and is not claimed here [11].

## Load it

Pin the revision this card's numbers were read at (the shortlist commit, matching the Hub API's current `sha` for `main`) [6]:

```python
import datasets

REV = "00643ccff2d088f7a24f93499a0b6ce5ca82ae3a"
ds = datasets.load_dataset(
    "mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify",
    revision=REV, split="train",
)  # 6,659 rows
```

**Trap**: only one split (`train`) ships, so `load_dataset` gives you every row with no held-out portion; any eval hold-out has to be built by the caller, and the natural one - AIME/AMC problems - is exactly the pool this dataset draws from (see Hold out above). Also, rows with the same `__original_row_idx` share the same `problem`/`solution` and largely-overlapping `r1_distill_70b_response` list (up to three rows per surviving problem); a random row-level split of this single `train` split would leak near-duplicate problems across the split, not novel data.

## Neighbors

All 27 sibling repos share the same 2,500-problem AMC/AIME pool and DeepSeek-R1-Distill-Llama-70B generation step, varying only in the filtering/verification stage; every row count below was read live at the check date [4].

- `mlfoundations-dev/multiple_samples_all_numina_aime` - 7,500 rows: every one of the three duplicate samples per problem is kept (an "all_verification" step that performs no filtering), versus this dataset's post-filter 6,659 [12].
- `mlfoundations-dev/multiple_samples_majority_consensus_numina_aime` - 5,601 rows: the same majority-consensus filtering logic, but the final answer is extracted by an LLM (`gpt-4o-mini`, prompted as `metamath_extract_math_answer`) instead of the Math-Verify library used here [13].
- `mlfoundations-dev/multiple_samples_majority_consensus_pick_one_numina_aime_math_verify` - 2,500 rows, exactly one per problem: the same Math-Verify-based majority filter as this dataset, collapsed to a single representative surviving response per problem instead of keeping every surviving duplicate [14].
- `mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_llm_verifier` - 7,093 rows: a further variant of the family that substitutes an LLM-based verifier for the Math-Verify check [4].
- `mlfoundations-dev/multiple_samples_all_numina_aime_w_openthoughts` and `mlfoundations-dev/multiple_samples_majority_consensus_math_verify_numina_aime_w_openthoughts` - 125,668 and 124,955 rows: later (2025-02-17) revisions of the "all" and "majority-consensus + Math-Verify" pipelines respectively, run against a larger problem pool that also draws from OpenThoughts [4].
- `mlfoundations-dev/dpo_from_multiple_samples_shortest_numina_aime` exists as a DPO-formatted release, but it pairs the separate "shortest" sibling variant, not this Math-Verify majority-consensus one; no DPO-formatted release of this exact dataset was found [4].

This corpus's screening record does not state which sibling it prefers; the choice among them depends on whether the training run wants every sample (`all`), one row per problem (`pick_one`), or Math-Verify versus LLM-based answer extraction, as described above.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [11], with the long response fields truncated:

```json
{
  "problem": "At Euclid Middle School the mathematics teachers are Mrs. Germain, Mr. Newton, and Mrs. Young. There are $11$ students in Mrs. Germain's class, $8$ students in Mr. Newton's class, and $9$ students in Mrs. Young's class taking the AMC $8$ this year. How many mathematics students at Euclid Middle School are taking the contest? [...]",
  "solution": "1. **Identify the number of students in each class:** [...]",
  "r1_distill_70b_response": [
    "<think>\nTo determine the total number of mathematics students at Euclid Middle School taking the AMC 8 contest, I will [...]",
    "<think>\n[second sample, truncated] [...]",
    "<think>\n[third sample, truncated] [...]"
  ],
  "__original_row_idx": 12,
  "r1_distill_70b_extracted_answer": ["28", "28", "28"],
  "_majority_responses": ["<think>\nTo determine the total number [...] [truncated, 3 items for this row]"],
  "verified_r1_distill_70b_response": "<think>\nTo determine the total number of mathematics students at Euclid Middle School taking the AMC 8 contest, I will [...]",
  "conversations": [
    {"from": "human", "value": "At Euclid Middle School the mathematics teachers are Mrs. Germain, Mr. Newton, and Mrs. Young. [...]"},
    {"from": "gpt", "value": "<think>\nTo determine the total number of mathematics students at Euclid Middle School taking the AMC 8 contest, I will [...]"}
  ]
}
```

Every served row has this one shape (one config, one split, one schema); `verified_r1_distill_70b_response` always equals one element of `r1_distill_70b_response`, and `conversations[1].value` always equals `verified_r1_distill_70b_response` [11].

## Where it came from

Built by mlfoundations-dev from `AI-MO/NuminaMath-CoT`'s `amc_aime` source subset - a pool the NuminaMath card describes as part of "US and international mathematics olympiad competition problems" collected from exam PDFs and math forums, then OCR'd, segmented, translated and reformatted into chain-of-thought solutions [2]. mlfoundations-dev's own build config samples 2,500 deduplicated problems from that pool, generates three DeepSeek-R1-Distill-Llama-70B responses per problem at temperature 1.0, extracts each response's final answer with the Math-Verify library, and keeps only the individual responses whose extracted answer agrees with the majority answer for that problem, before converting the surviving problem/response pairs into the served `conversations` format [3]. This release sits in a family of at least 27 sibling repos from the same author built on the identical pool with different sampling, verification, and merge choices [4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision `00643ccff2d088f7a24f93499a0b6ce5ca82ae3a`.

[1] mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify dataset card (README). https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify/raw/main/README.md - YAML front matter only, no descriptive body, no citation. Fetched 2026-08-11.

[2] AI-MO/NuminaMath-CoT dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/raw/main/README.md - source breakdown table (`amc_aime`: 4,072 rows of 859,608 total), licence, collection description. Fetched 2026-08-11.

[3] Build pipeline config for this dataset. https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify/raw/main/config/multiple_samples_majority_consensus_numina_aime_math_verify.yaml - operator graph: source selection, sampling, duplication, generation model/params, Math-Verify extraction, majority-consensus filter, sharegpt conversion. Fetched 2026-08-11.

[4] Hugging Face Hub dataset search for author `mlfoundations-dev`, query `numina_aime`. https://huggingface.co/api/datasets?author=mlfoundations-dev&search=numina_aime&limit=100 - 27 sibling repository ids and metadata; individual sibling row counts and configs cross-checked via the same datasets-server `/size` and repo `config/*.yaml` endpoints as this dataset. This endpoint takes no revision parameter, so the sibling list and counts are live, not pinned. Fetched 2026-08-11.

[5] The corpus screening row for `mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify`, supplied with this card's request - its `flag` field, read back in the appendix. Checked 2026-08-11.

[6] Hugging Face Hub API record for this dataset. https://huggingface.co/api/datasets/mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify?full=true - `cardData` (no licence field), `gated`, `private`, `sha`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fmultiple_samples_majority_consensus_numina_aime_math_verify Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=mlfoundations-dev%2Fmultiple_samples_majority_consensus_numina_aime_math_verify Fetched 2026-08-11.

[9] Build pipeline config for the LLM-extraction sibling. https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_majority_consensus_numina_aime/raw/main/config/multiple_samples_majority_consensus_numina_aime.yaml - shows `gpt-4o-mini`/`metamath_extract_math_answer` in place of the Math-Verify step used by this dataset. Fetched 2026-08-11.

[10] TensorBlock GGUF mirror model card. https://huggingface.co/tensorblock/llama3-1_8b_multiple_samples_majority_consensus_numina_aime-GGUF/raw/main/README.md and its API record https://huggingface.co/api/models/tensorblock/llama3-1_8b_multiple_samples_majority_consensus_numina_aime-GGUF - `base_model: mlfoundations-dev/llama3-1_8b_multiple_samples_majority_consensus_numina_aime`; that base repository itself is no longer reachable via the Hub API. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mlfoundations-dev%2Fmultiple_samples_majority_consensus_numina_aime_math_verify&config=default&split=train - first ten served rows, used for the row-shape check, the `verified_r1_distill_70b_response`-matches-list-item observation, and the `_majority_responses` truncation noted in Quality. Fetched 2026-08-11.

[12] `mlfoundations-dev/multiple_samples_all_numina_aime`: size endpoint (https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fmultiple_samples_all_numina_aime) and build config (https://huggingface.co/datasets/mlfoundations-dev/multiple_samples_all_numina_aime/raw/main/config/multiple_samples_all_numina_aime.yaml). Fetched 2026-08-11.

[13] `mlfoundations-dev/multiple_samples_majority_consensus_numina_aime`: size endpoint. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fmultiple_samples_majority_consensus_numina_aime Fetched 2026-08-11.

[14] `mlfoundations-dev/multiple_samples_majority_consensus_pick_one_numina_aime_math_verify`: size endpoint and info endpoint. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fmultiple_samples_majority_consensus_pick_one_numina_aime_math_verify and https://datasets-server.huggingface.co/info?dataset=mlfoundations-dev%2Fmultiple_samples_majority_consensus_pick_one_numina_aime_math_verify Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data, with a decontamination step owed before any scored AIME/AMC evaluation run. The dataset's own build config shows the filtering step keeps only individually verified-correct-by-consensus traces [3], but that check operates on answer correctness, not on removing AIME items from the shared Numina pool - so the screening record's rule-based risk flag against AIME evaluation sets stands as a usage restriction, not a defect in the data itself [5].

### The screening row

The row's own note [5]: "Same as the 'all' variant, filtered to majority-consensus answers and checked with Math-Verify." Its flag: "rule-risk: aime2025 - Same Numina amc_aime problem pool; majority-consensus and Math-Verify filtering removes wrong answers, not AIME items."
