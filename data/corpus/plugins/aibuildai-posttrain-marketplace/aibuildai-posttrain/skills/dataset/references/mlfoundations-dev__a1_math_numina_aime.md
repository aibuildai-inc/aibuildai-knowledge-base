# mlfoundations-dev/a1_math_numina_aime

31,600 competition-math reasoning-SFT rows built from three merged NuminaMath-1.5 source subsets - `amc_aime`, `olympiad`, and `aops_forum` - each paired with a DeepSeek-R1-generated reasoning trace, held at a fixed pipeline-ablation sample size.

**mlfoundations-dev/a1_math_numina_aime** is built by the mlfoundations-dev group's data-generation pipeline (the repository's `config/a1_math_numina_aime.yaml` names the operators) from `AI-MO/NuminaMath-1.5` [1]. The config selects three of NuminaMath-1.5's source subsets separately - rows where the original `source` field equals `amc_aime`, `olympiad`, or `aops_forum` - and its single `decontaminate` step takes all three selections as input, so the three subsets are merged into one pool before the fixed 31,600-row sample is drawn; no later step in the graph branches or re-splits them, and no source states the resulting per-subset share of the shipped rows [1]. The dataset's own name and its overwritten `source` column both read as AMC/AIME-only, but the config this card cites shows the shipped rows are not that: they also include NuminaMath-1.5's `olympiad` and `aops_forum` material, which are majority-proof-type subsets (32% and 36% proof respectively, versus 3.5% for `amc_aime` alone) [1][7]. The 31,600-row sample size matches the "Question Sourcing" ablation methodology of "OpenThoughts: Data Recipes for Reasoning Models" (the OpenThoughts3 paper), which generates 31,600 examples per candidate question-source configuration and finetunes Qwen2.5-7B-Instruct on each to compare them [2]; DeepSeek-R1 is that paper's default teacher model unless stated otherwise, and this repository's config also calls `deepseek-reasoner` [1][2]. The dataset card itself carries no prose - only YAML front matter - so all of this is read from the pipeline config file, not from a card statement. It lives at https://huggingface.co/datasets/mlfoundations-dev/a1_math_numina_aime . **Do not treat this as an AMC/AIME-only pool: it is a merge of `amc_aime`, `olympiad`, and `aops_forum` rows in an unstated ratio. The pipeline's `decontaminate` step runs fuzzy n-gram matching (threshold 75, 13-grams) against a named list of evaluation sets - including `mlfoundations-dev/AIME2025_combined` (AIME 2025 problems), `Maxwell-Jia/AIME_2024`, `AI-MO/aimo-validation-amc`, and `HuggingFaceH4/MATH-500` - but no source states how many rows were removed or confirms zero overlap remains [1]; decontaminate again against AIME/AMC/olympiad-style benchmarks before a scored run.**

**Use it for**: reasoning-trace SFT - single-turn chat pairs where the assistant turn is a DeepSeek-R1 `<think>...</think>` trace ending in a boxed answer or proof; decontaminate against AIME/AMC/olympiad-family benchmarks first (see above), since the pool is not AMC/AIME-only. Maps to the SFT method card's chat-format loader; the served `conversations` column is already ShareGPT-style `[{"from": "human", ...}, {"from": "gpt", ...}]` pairs, so no reformatting from an implicit-prompt preference layout is needed.

**Licence**: not stated. `cardData` carries no `license` key and the repo has no `license:` tag [3]; ungated (`"gated": false`, `"private": false`) [3]. The upstream `AI-MO/NuminaMath-1.5` problems are Apache-2.0 [1], but no source states that grant extends to this repository's added DeepSeek-R1 traces.

**Shape**: 31,600 rows, one split (`train`), one config (`default`), 14 columns [4][5].

**Hold out**: no split is held out inside this repository - it is one `train` split. The contamination risk is against external competition-math benchmarks, not an internal split: the pipeline's own decontamination step targets `mlfoundations-dev/AIME2025_combined` (30 rows, AIME 2025 I+II) and eleven other eval sets, but no source states a removed-row count, and no source states what share of the 31,600 rows come from `amc_aime` versus `olympiad`/`aops_forum` [1]. Treat any AIME/AMC/olympiad-benchmark evaluation of a model trained on this data as needing its own decontamination check.

**Origin**: built by mlfoundations-dev from `AI-MO/NuminaMath-1.5` problems (merged `amc_aime`/`olympiad`/`aops_forum` subsets), with DeepSeek-R1 (`deepseek-reasoner`) generating the `reasoning`/`deepseek_solution`/`final_reasoning_trace` fields [1]. Hub API at the check date: 256 downloads, 1,100 all-time downloads, 0 likes [3].

**Trained-on-by**: none found. This is a small (31,600-row), unlisted per-source ablation artifact; no source names a released model or recipe trained specifically on this repository. The OpenThoughts3 paper trains many Qwen2.5-7B-Instruct ablation checkpoints on 31,600-row per-source datasets built with this same pipeline, but its public tables report `AI-MO/NuminaMath-1.5` as one aggregate math source and do not name this merged `amc_aime`/`olympiad`/`aops_forum` slice as a separate row [2].

**Introduced by**: no paper - the dataset card carries no prose [6]. The pipeline config that produced it matches the described ablation methodology of [2], but that paper does not name this specific artifact.

## Shape

One config (`default`), one split, 14 columns [4][5]:

| column | dtype |
| --- | --- |
| `instruction_seed` | string |
| `solution` | string |
| `answer` | string |
| `problem_type` | string |
| `question_type` | string |
| `problem_is_valid` | string |
| `solution_is_valid` | string |
| `synthetic` | bool |
| `source` | string |
| `reasoning` | string |
| `deepseek_solution` | string |
| `__original_row_idx` | int64 |
| `final_reasoning_trace` | string |
| `conversations` | list\<struct\<from: string, value: string\>\> |

| split | rows |
| --- | --- |
| `train` | 31,600 |

Sizes: 1,250,736,223 bytes of Parquet download, 2,901,327,487 bytes decoded in memory (datasets-server, live) [4]; `cardData.dataset_info` states 2,892,924,747 bytes decoded, a smaller figure than the live one [5]. No source states sequence-length or token statistics for this repository.

## Quality

- `instruction_seed`, `solution`, `answer`, `problem_type`, `question_type`, `problem_is_valid`, `solution_is_valid`, `source`, and `synthetic` are carried through from `AI-MO/NuminaMath-1.5`'s matching columns (`problem` renamed to `instruction_seed`) [1][7]. Of these, `reasoning`, `deepseek_solution`, `__original_row_idx`, `final_reasoning_trace`, and `conversations` are added by this repository's pipeline [1].
- The `source` column is not the original NuminaMath source label - the pipeline config drops the true per-row `source` value (`amc_aime`, `olympiad`, or `aops_forum`) after the three subsets are merged, and overwrites the whole column with the constant string `AI-MO/NuminaMath-1.5-SUBSETS` [1]; the 10 rows read at offset 0 in `train` all carry that constant value, so the shipped `source` column cannot be used to recover which of the three subsets any given row came from [8].
- Of the first 10 rows read (offset 0, `train`), 5 have `question_type="proof"` [8] - far above the 3.5% proof rate NuminaMath-1.5's card reports for `amc_aime` alone (208/5,872 problems), and closer to the 32%-36% proof rates it reports for `olympiad` (62,970/197,084) and `aops_forum` (24,532/67,841) [7]. That 10-row sample is consistent with `olympiad`/`aops_forum` rows making up a substantial share of this dataset, though no source states the exact share. Also of those 10 rows, 5 have a null `solution` field even though `final_reasoning_trace` and `deepseek_solution` are populated for all 10; one row (index 4) has null `problem_type`, `question_type`, `problem_is_valid`, and `solution_is_valid` while still carrying a non-null `answer` [8]. No source states a dataset-wide rate for any of these; this is only what was observed in that 10-row sample.
- No source states a measured contamination rate, duplicate rate, or human-verification rate specific to this repository.

## Load it

Pin the revision this card's numbers were read at:

```python
import datasets

REV = "0b1b0f4f61b26e9d6769538c708a6b68bfb58a4b"
ds = datasets.load_dataset("mlfoundations-dev/a1_math_numina_aime", revision=REV, split="train")  # 31,600 rows
```

**Trap**: this is a single-split, single-config repository with no held-out eval split of its own - the contamination risk here is external (against AIME/AMC benchmarks), not a `train`/`test` leak inside the repo, so there is nothing to hold out by splitting this dataset itself.

## Neighbors

- The `olympiad` and `aops_forum` NuminaMath-1.5 subsets are not separate sibling releases of this one - this repository's own `config/a1_math_numina_aime.yaml` selects `amc_aime`, `olympiad`, and `aops_forum` rows and merges all three into this single 31,600-row output before decontamination and sampling (see the opening paragraph) [1]. Repository ids matching those subset names, `mlfoundations-dev/a1_math_numina_olympiad` and `mlfoundations-dev/a1_math_numina_aops_forum`, do not resolve as public Hub repositories at the check date: the Hub API returns HTTP 401 with body `{"error":"Invalid username or password."}` for both, the same generic response a deliberately nonexistent repository id returns for an anonymous caller - so there is no separate, single-subset release to compare against [9].
- `open-thoughts/OpenThoughts3-1.2M` - the 1,200,000-row released successor from the same project: `difficulty`, `source`, `domain`, `conversations` columns, one `train` split [10]. Its `source` and `domain` columns are not readable as "does it contain `amc_aime`" without fetching every row; the paper states `AI-MO/NuminaMath-1.5` was the runner-up math question source in its final ablation table, but the served rows read here (first 10, all `domain: code`) do not include a math row to confirm the exact source label used [2][10]. Prefer this repository only for the isolated, single-source signal it gives; prefer OpenThoughts3-1.2M for a deployable, mixed-source training set.
- `AI-MO/NuminaMath-1.5` itself - the 896,215-row upstream pool this repository's `amc_aime` rows (5,872 problems in the upstream source breakdown) are drawn from, without DeepSeek-R1 traces [1].

## A row

One config, one split - one shape. From `split="train"`, row 0 (datasets-server first-rows) [8], with long fields truncated. This particular row is a proof-type geometry problem (null `answer`), illustrating the non-`amc_aime` (`olympiad`/`aops_forum`-consistent) proof-heavy content discussed above rather than a numeric-answer AMC/AIME item:

```json
{
  "instruction_seed": "Given a non-isosceles triangle $ABC$ with incircle $k$ with center $S$. $k$ touches the side $BC,CA,AB$ at $P,Q,R$ respectively. [...]",
  "solution": null,
  "answer": null,
  "problem_type": "Geometry",
  "question_type": "proof",
  "problem_is_valid": "Yes",
  "solution_is_valid": "Yes",
  "synthetic": false,
  "source": "AI-MO/NuminaMath-1.5-SUBSETS",
  "reasoning": "Okay, so I need to prove that points S, L, and M are collinear in this triangle configuration. [...]",
  "deepseek_solution": "Given a non-isosceles triangle \\(ABC\\) with incircle \\(k\\) centered at \\(S\\) [...]",
  "__original_row_idx": 1,
  "final_reasoning_trace": "<think>\nOkay, so I need to prove that points S, L, and M are collinear in this triangle configuration. [...]\n</think>\n\n[...] \\[\n\\boxed{S, L, M \\text{ are collinear}}\n\\]",
  "conversations": [
    {"from": "human", "value": "Given a non-isosceles triangle $ABC$ with incircle $k$ [...]"},
    {"from": "gpt", "value": "<think>\nOkay, so I need to prove that points S, L, and M are collinear [...]\n</think>\n\n[...] \\[\n\\boxed{S, L, M \\text{ are collinear}}\n\\]"}
  ]
}
```

`conversations[1].value` is byte-identical to `final_reasoning_trace` in this row [8].

## Where it came from

Built by mlfoundations-dev. The pipeline config (`config/a1_math_numina_aime.yaml`) loads `AI-MO/NuminaMath-1.5`, renames `problem` to `instruction_seed`, and separately selects rows whose original `source` field equals `amc_aime`, `olympiad`, or `aops_forum`. All three selections feed into a single decontamination step that runs fuzzy n-gram matching (threshold 75, 13-grams) against a named list of evaluation datasets - merging the three subsets into one pool at that point, with no later step separating them back out. The pipeline then discards the original `source` column and replaces it with the constant `AI-MO/NuminaMath-1.5-SUBSETS`, uniformly samples 31,600 rows from the merged, decontaminated pool, generates a reasoning trace and solution for each with DeepSeek-R1 (`deepseek-reasoner`, temperature 1.0, via the DeepSeek API), builds `final_reasoning_trace` from the reasoning and solution, and converts `instruction_seed`/`final_reasoning_trace` into the ShareGPT-style `conversations` column [1].

## Sources

Checked 2026-08-12; Hub repositories are mutable, which is why Load it pins the revision.

[1] mlfoundations-dev/a1_math_numina_aime repository files: dataset card YAML (`README.md`, `card_body_bytes` 906, no prose) and pipeline config `config/a1_math_numina_aime.yaml`. https://huggingface.co/datasets/mlfoundations-dev/a1_math_numina_aime/raw/main/README.md and https://huggingface.co/datasets/mlfoundations-dev/a1_math_numina_aime/raw/main/config/a1_math_numina_aime.yaml Fetched 2026-08-12.

[2] "OpenThoughts: Data Recipes for Reasoning Models" (OpenThoughts3), arXiv:2506.04178 - 31,600-row per-source ablation methodology, DeepSeek-R1 as default teacher, math question source list and Table 33/R.1 descriptions, held-out AIME 2025 benchmark. Read via the arXiv HTML full text. https://arxiv.org/abs/2506.04178 Fetched 2026-08-12.

[3] Hugging Face Hub API record for mlfoundations-dev/a1_math_numina_aime. https://huggingface.co/api/datasets/mlfoundations-dev/a1_math_numina_aime?full=true - licence fields, gate status, `sha`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fa1_math_numina_aime Fetched 2026-08-12.

[5] datasets-server info endpoint, and `cardData.dataset_info` inside [3]. https://datasets-server.huggingface.co/info?dataset=mlfoundations-dev%2Fa1_math_numina_aime Fetched 2026-08-12.

[6] Same as [1]; cited separately for the "no paper" determination in Introduced by.

[7] AI-MO/NuminaMath-1.5 dataset card and datasets-server info endpoint, for the upstream feature schema and the `amc_aime` source-breakdown count (5,872 problems). https://huggingface.co/datasets/AI-MO/NuminaMath-1.5/raw/main/README.md and https://datasets-server.huggingface.co/info?dataset=AI-MO%2FNuminaMath-1.5 Fetched 2026-08-12.

[8] datasets-server first-rows endpoint, `train` split, offset 0, 10 rows. https://datasets-server.huggingface.co/first-rows?dataset=mlfoundations-dev%2Fa1_math_numina_aime&config=default&split=train Fetched 2026-08-12.

[9] Hugging Face Hub API, checked for `mlfoundations-dev/a1_math_numina_olympiad` and `mlfoundations-dev/a1_math_numina_aops_forum` - both return HTTP 401, `{"error":"Invalid username or password."}`, the Hub API's generic response to an anonymous caller for any repository id that does not resolve; verified against a control fetch of a deliberately nonexistent repository id returning the identical body, confirming neither id names an existing public repository. https://huggingface.co/api/datasets/mlfoundations-dev/a1_math_numina_olympiad and https://huggingface.co/api/datasets/mlfoundations-dev/a1_math_numina_aops_forum Fetched 2026-08-12.

[10] open-thoughts/OpenThoughts3-1.2M Hub API record, size endpoint, and first-rows endpoint. https://huggingface.co/api/datasets/open-thoughts/OpenThoughts3-1.2M , https://datasets-server.huggingface.co/size?dataset=open-thoughts%2FOpenThoughts3-1.2M , https://datasets-server.huggingface.co/first-rows?dataset=open-thoughts%2FOpenThoughts3-1.2M&config=default&split=train - this endpoint takes no revision parameter, so these figures are live, not pinned. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Flagged, usable with a required decontamination step before any competition-math benchmark evaluation. The screening note's "amc_aime subset" characterization is only partly accurate: the repository's own pipeline config shows the shipped rows are a merge of NuminaMath-1.5's `amc_aime`, `olympiad`, and `aops_forum` subsets, not `amc_aime` alone (see the opening paragraph and Where it came from) [1]. The screening flag's underlying concern is confirmed above: the served `source` column is a constant pool label that does not itself prove decontamination succeeded, even though the repository's own pipeline config does configure a decontamination step against AIME2025, AIME2024, AMC, MATH-500, and eight other eval sets [1].

### The screening row

The row's own note: "NuminaMath-1.5 amc_aime subset (historical AMC/AIME, including proof-type items) with DeepSeek-R1 reasoning traces." Its flag: "rule-risk: aime2025 - Row `source` field reads AI-MO/NuminaMath-1.5-SUBSETS, the calibration pool carrying AMC/AIME items; card claims no removal."
