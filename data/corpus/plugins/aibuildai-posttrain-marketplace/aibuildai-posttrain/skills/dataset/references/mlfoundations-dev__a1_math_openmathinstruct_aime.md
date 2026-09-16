# mlfoundations-dev/a1_math_openmathinstruct_aime

31,600 competition-style math problems with DeepSeek-R1 reasoning traces, one config/one split, built from AMC/AIME-flavored subsets of NuminaMath-CoT.

Item home: https://huggingface.co/datasets/mlfoundations-dev/a1_math_openmathinstruct_aime

**mlfoundations-dev/a1_math_openmathinstruct_aime** is one release in the mlfoundations-dev `a1_math_*` family, a set of same-shaped math SFT datasets built by an internal pipeline recorded in the repository's own `config/a1_math_openmathinstruct_aime.yaml` [1]. That pipeline loads AI-MO/NuminaMath-CoT [2] and selects its `amc_aime` and `olympiad` source subsets, rewrites each selected problem into a new question with `gpt-4o-mini` ("evolution", `n_repeat: 10`), keeps a uniform sample of 40,000, annotates each with a DeepSeek `deepseek-reasoner` (R1) trace, fuzzy-decontaminates the result against twelve named evaluation sets - including `mlfoundations-dev/AIME2025_combined` and `AI-MO/aimo-validation-amc` - at a similarity threshold of 75 over 13-grams, and finally uniformly samples 31,600 rows and formats them as ShareGPT `conversations` [1]. The repository's README carries only YAML metadata (`dataset_info`/`configs`), no prose, so none of this pipeline is stated in the README itself [3]. **The dataset's own `source` column is a constant string, `"OpenMathInstruct-AIME"`, added by the pipeline as a family label, not a lineage claim: the pipeline config shows the actual upstream is NuminaMath-CoT's `amc_aime`/`olympiad` subsets, not the OpenMathInstruct dataset [1].** Despite the pipeline's own fuzzy decontamination against AIME 2025 and AMC benchmarks, hold out AIME/AMC-style evaluation sets before scoring a model trained on this data (see Hold out below), because the source pool is itself AMC/AIME competition problems and the served rows show verbatim, duplicated problems rather than paraphrased ones (see Quality).

**Use it for**: reasoning-trace SFT - each row pairs a competition-math question (`instruction_seed`) with a DeepSeek-R1 `<think>...</think>` trace plus final solution (`final_reasoning_trace`), and the same pair is pre-formatted as a two-turn `conversations` list (`human`/`gpt`). Use `conversations` directly with a ShareGPT-style SFT chat loader, or build the prompt/completion pair from `instruction_seed`/`final_reasoning_trace` yourself; this is the reasoning-trace SFT method card's shape, not a preference or reward-model format. Hold out AIME/AMC-style benchmarks before evaluation regardless of the pipeline's own decontamination pass (see Hold out).

**Licence**: not stated. `cardData` carries no `license` key, no `license:` tag appears on the repository, and the README body has no licence text [3][4]. The repo is ungated and public (`"gated": false`, `"private": false`) [4]. The one catch: the upstream NuminaMath-CoT pool is released under Apache-2.0 [2], but no source states that this derived, re-formatted release carries that grant forward.

**Shape**: 31,600 rows, one config (`default`), one split (`train`); six columns, no held-out split shipped by the repository itself [3][5][6].

**Hold out**: AIME/AMC-style evaluation sets, by name, even though the pipeline already ran a fuzzy-ngram decontamination pass against them at build time: `mlfoundations-dev/AIME2025_combined` (30 rows) [7][8], `AI-MO/aimo-validation-amc` (83 rows) [1][9], `Maxwell-Jia/AIME_2024`, `HuggingFaceH4/MATH-500`, and the other eval sets the pipeline config names for decontamination [1]. This is the shortlist row's own rule-risk flag: the `source` column reads `OpenMathInstruct-AIME` and the rows are AMC/AIME competition problems, and the README carries no decontamination claim [10] - the claim exists, but only in the repository's config file, not the card, and a fuzzy match at threshold 75 does not guarantee zero overlap with AIME 2025 or AMC problems the model may later be scored on.

**Origin**: built by mlfoundations-dev from NuminaMath-CoT source problems, rewritten by `gpt-4o-mini` and annotated with DeepSeek `deepseek-reasoner` (R1) traces, per the pipeline config [1][2]. Hub API at the check date: 97 downloads, 788 all-time downloads, 0 likes [4].

**Trained-on-by**: no named public model card states it trained on this dataset. The same repository namespace hosts `mlfoundations-dev/a1_math_openmathinstruct_aime_eval_636d`, a sibling dataset that stores cached outputs from an evaluation run, reporting 25.0% AIME24, 66.0% AMC23, and 80.2% MATH500 accuracy over ten runs, which is evidence some model was trained (likely on this data or a close variant in the pipeline) and scored on AIME/AMC benchmarks, but that sibling names no model checkpoint [11]. No other adoption evidence was found.

**Introduced by**: no paper - the dataset card carries only YAML metadata [3], and the construction pipeline is documented solely in the repository's own config file [1].

## Shape

The `/size` and `/info` endpoints below take no revision parameter and reflect whatever the Hub currently indexes, not the pinned commit in Load it; at the check date the live Hub `sha` for `main` [4] matched the shortlist's pinned commit (`938d2aa0...`), so these figures agree with that revision as of 2026-08-12, but the endpoints carry no guarantee of staying pinned to it.

Rows and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 31,600 |

One config, `default`, six columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `instruction_seed` | string |
| `reasoning` | string |
| `deepseek_solution` | string |
| `source` | string |
| `final_reasoning_trace` | string |
| `conversations` | list<struct<from: string, value: string>> |

The repository's `cardData` (embedded in the README's YAML front matter) states `dataset_size: 1,425,045,348` bytes decoded in memory [3], but the live datasets-server `/info` and `/size` endpoints report `dataset_size`/`num_bytes_memory` of 737,424,773 bytes - roughly half the card's figure - while both agree on `download_size` at 622,828,394 bytes [5][6]. No source states sequence-length or token statistics for this release.

## Quality

- The pipeline's rewrite step ("evolution") is meant to generate a new question per selected source problem via `gpt-4o-mini`, `n_repeat: 10` [1], but of the first 20 served rows (offsets 0-19), three distinct `instruction_seed` values repeat verbatim: the "Carlos took 70% of a whole pie..." AMC problem at rows 0 and 3 (a pair), the "You and five friends need to raise $1500..." AMC problem at rows 4, 7, and 8 (a triplicate), and the "Margie's car can go 32 miles on a gallon of gas..." AMC problem at rows 11 and 17 (a pair) [12]. That is 7 of the first 20 rows (35%) sharing a question text with at least one other row in that window. This is read from twenty rows at offset 0 only; it is not a claim about the duplicate rate across all 31,600 rows.
- The `source` column holds the constant string `"OpenMathInstruct-AIME"` for every row inspected (all 20 rows, offsets 0-19); it is a family label the pipeline assigns, not a per-row provenance field [1][12].
- The pipeline's decontamination step (`decontaminate_fuzzy_w_ngram`) runs before the final 31,600-row sample, at similarity threshold 75.0 over 13-grams, against twelve named evaluation datasets including `mlfoundations-dev/AIME2025_combined` and `AI-MO/aimo-validation-amc` [1]. No source states a measured contamination rate after that pass; none is invented here.
- No source states an annotator-agreement figure, since annotation here is a DeepSeek model call, not human labeling; no source flags erroneous or unsolved DeepSeek traces.

## Load it

```python
import datasets

REV = "938d2aa03bdc812da59a11d29bd592ef74a84503"  # main at the check date
train = datasets.load_dataset("mlfoundations-dev/a1_math_openmathinstruct_aime", revision=REV, split="train")  # 31,600 rows
```

**Trap**: there is only one split (`train`) and no repository-provided eval/test split - a caller must carve out its own holdout, and the Hold out line above names the benchmarks to exclude rather than any dataset-internal split [3][5].

## Neighbors

The `mlfoundations-dev` namespace hosts a family of same-shaped `a1_math_*` releases sharing the `*_aime` naming pattern; each row count below was read live at the check date [13]. Only one sibling's pipeline config was fetched and diffed against this release's own config - that is the only one confirmed below to share the same pipeline steps. For the other two, only the Hub API and datasets-server counts were fetched, so their naming and column counts are reported without a pipeline-identity claim.

- `mlfoundations-dev/a1_math_numina_aime` - 31,600 rows, 14 columns (more columns than this release), same row count and naming pattern; no pipeline config was fetched for this sibling, so whether it shares this release's pipeline steps is not stated [13].
- `mlfoundations-dev/a1_math_metamath_aime` - 31,600 rows, 8 columns, same row count and `*_aime` naming pattern; no pipeline config was fetched for this sibling, so its source pool and pipeline steps are not stated here beyond the "metamath" name itself [13].
- `mlfoundations-dev/a1_math_openmathinstruct2_aime` - 31,600 rows, 6 columns (same column count as this release). Its own pipeline config, fetched directly, shows it also loads `AI-MO/NuminaMath-CoT` and selects the same `amc_aime`/`olympiad` subsets through the same `gpt-4o-mini` evolution and DeepSeek `deepseek-reasoner` annotation steps as this release, differing only in one column-rename step; despite the "openmathinstruct2" name, it is not built from the OpenMathInstruct-2 dataset [14].
- `mlfoundations-dev/a1_math_openmathinstruct_aime_eval_636d` - 7,904 rows, a precomputed-model-outputs/evaluation-results dataset over this pipeline's checkpoints (AIME24, AMC23, MATH500, and other benchmarks), not a training corpus [11][13].
- The upstream pool, `AI-MO/NuminaMath-CoT` [2], is the broader ~860k-problem source this and its `a1_math_*` siblings sample from; it carries no reasoning traces of its own and is not decontaminated against AIME/AMC by its own card.

This corpus's shortlist row does not state a preference among the `*_aime` siblings; none is asserted here.

## A row

One config, one split, so one row covers it. From `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [12], with the reasoning trace truncated:

```json
{
  "instruction_seed": "Carlos took $70\\%$ of a whole pie. Maria took one third of the remainder. What portion of the whole pie was left?\n$\\textbf{(A)}\\ 10\\%\\qquad\\textbf{(B)}\\ 15\\%\\qquad\\textbf{(C)}\\ 20\\%\\qquad\\textbf{(D)}\\ 30\\%\\qquad\\textbf{(E)}\\ 35\\%$",
  "reasoning": "Okay, so Carlos took 70% of a whole pie. Let me visualize that. [...] one third of 30%... Let me calculate that.",
  "deepseek_solution": "Carlos took 70% of the whole pie, leaving a remainder of 30%. [...] \\frac{1}{3} \\times 30\\% = 10\\% [...]",
  "source": "OpenMathInstruct-AIME",
  "final_reasoning_trace": "<think>\nOkay, so Carlos took 70% of a whole pie. [...]\n</think>\n\nCarlos took 70% of the whole pie, leaving a remainder of 30%. [...]",
  "conversations": [
    {"from": "human", "value": "Carlos took $70\\%$ of a whole pie. Maria took one third of the remainder. What portion of the whole pie was left?\n$\\textbf{(A)}\\ 10\\%\\qquad\\textbf{(B)}\\ 15\\%\\qquad\\textbf{(C)}\\ 20\\%\\qquad\\textbf{(D)}\\ 30\\%\\qquad\\textbf{(E)}\\ 35\\%$"},
    {"from": "gpt", "value": "<think>\nOkay, so Carlos took 70% of a whole pie. [...] Therefore, the correct answer is C, 20%.\n</think>\n\nCarlos took 70% of the whole pie, leaving a remainder of 30%. [...] \\(\\boxed{C}\\)."}
  ]
}
```

## Where it came from

Built by mlfoundations-dev. Per the repository's own `config/a1_math_openmathinstruct_aime.yaml` [1]: the pipeline loads `AI-MO/NuminaMath-CoT`'s `train` split [2], selects rows whose `source` field reads `amc_aime` or `olympiad`, and rewrites each into a new question with `gpt-4o-mini` at temperature 1.0 (`n_repeat: 10`). It uniformly samples 40,000 of these, then calls DeepSeek's `deepseek-reasoner` (R1) model on each to produce a `reasoning` trace and a `deepseek_solution`, drops all columns except `instruction_seed`/`reasoning`/`deepseek_solution`, and stamps a constant `source` value of `"OpenMathInstruct-AIME"`. It then fuzzy-decontaminates (ngram size 13, similarity threshold 75.0) against twelve named evaluation datasets - `HuggingFaceH4/MATH-500`, `Maxwell-Jia/AIME_2024`, `AI-MO/aimo-validation-amc`, `livecodebench/code_generation_lite`, `mlfoundations-dev/AIME2025_combined`, `cais/hle`, `open-r1/codeforces`, `Idavidrein/gpqa`, `daman1209arora/jeebench`, `mlfoundations-dev/mmlu_pro_eval_full`, `Qwen/CodeElo`, `open-r1/ioi` - uniformly samples down to 31,600, and finally converts the reasoning trace/solution pair into a `final_reasoning_trace` field and a ShareGPT `conversations` list [1]. NuminaMath-CoT's own card states its `amc_aime` source subset (4,072 of its ~860k rows) and its problems come from OCR'd exam PDFs and math forums, translated and reformatted into chain-of-thought, and is released under Apache-2.0 [2]; that card lists its source-column values as `amc_aime`, `aops_forum`, `cn_k12`, `gsm8k`, `math`, `olympiads` (plural), `orca_math`, `synthetic_amc`, and `synthetic_math` [2] - the pipeline's own `olympiad` (singular) selection condition does not literally match that `olympiads` value; no source confirms whether this selection resolved to any rows.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable, which is why Load it pins the revision. The shortlist row is cited by dataset id and check date, never by line or entry number.

[1] Pipeline configuration file in the dataset repository. https://huggingface.co/datasets/mlfoundations-dev/a1_math_openmathinstruct_aime/raw/main/config/a1_math_openmathinstruct_aime.yaml - construction pipeline: source selection, gpt-4o-mini rewrite step, DeepSeek R1 annotation, decontamination step and its eval-set list, final sampling and ShareGPT conversion. Fetched 2026-08-12.

[2] AI-MO/NuminaMath-CoT dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/raw/main/README.md - upstream pool description, source breakdown table, licence. Fetched 2026-08-12.

[3] mlfoundations-dev/a1_math_openmathinstruct_aime dataset card (README). https://huggingface.co/datasets/mlfoundations-dev/a1_math_openmathinstruct_aime/raw/main/README.md - YAML-only `dataset_info`/`configs`, no prose, `dataset_size` figure. Fetched 2026-08-12. Item home: https://huggingface.co/datasets/mlfoundations-dev/a1_math_openmathinstruct_aime

[4] Hugging Face Hub API record. https://huggingface.co/api/datasets/mlfoundations-dev/a1_math_openmathinstruct_aime?full=true - licence absence, gate/private status, `sha`, downloads, likes; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2Fa1_math_openmathinstruct_aime Fetched 2026-08-12.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=mlfoundations-dev%2Fa1_math_openmathinstruct_aime Fetched 2026-08-12.

[7] Hugging Face Hub API record for mlfoundations-dev/AIME2025_combined. https://huggingface.co/api/datasets/mlfoundations-dev/AIME2025_combined - `question`/`answer` schema, 30 rows declared. Fetched 2026-08-12.

[8] datasets-server size endpoint for mlfoundations-dev/AIME2025_combined. https://datasets-server.huggingface.co/size?dataset=mlfoundations-dev%2FAIME2025_combined Fetched 2026-08-12.

[9] datasets-server size endpoint for AI-MO/aimo-validation-amc. https://datasets-server.huggingface.co/size?dataset=AI-MO%2Faimo-validation-amc Fetched 2026-08-12.

[10] The corpus screening row for `mlfoundations-dev/a1_math_openmathinstruct_aime`, supplied with this card's request - its `flag` (rule-risk: aime2025), read back in the appendix. Checked 2026-08-12.

[11] mlfoundations-dev/a1_math_openmathinstruct_aime_eval_636d dataset card (README) and Hub API record. https://huggingface.co/datasets/mlfoundations-dev/a1_math_openmathinstruct_aime_eval_636d/raw/main/README.md and https://huggingface.co/api/datasets/mlfoundations-dev/a1_math_openmathinstruct_aime_eval_636d - cached evaluation-run outputs description, AIME24/AMC23/MATH500 accuracy table. Fetched 2026-08-12.

[12] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mlfoundations-dev%2Fa1_math_openmathinstruct_aime&config=default&split=train - the endpoint's full returned window, rows 0-19, read for duplicate-row and `source`-value checks. Fetched 2026-08-12.

[13] datasets-server size and info endpoints, one call per neighbor, for the row/column counts above: `mlfoundations-dev/a1_math_numina_aime`, `mlfoundations-dev/a1_math_metamath_aime`, `mlfoundations-dev/a1_math_openmathinstruct2_aime`, `mlfoundations-dev/a1_math_openmathinstruct_aime_eval_636d`. https://datasets-server.huggingface.co/size?dataset=<id> - these endpoints take no revision parameter, so these counts are live, not pinned; the same caveat applies to [5] and [6] above, which read this dataset's own headline row/byte/column counts from the same unpinned endpoints. Both this dataset's live Hub `sha` [4] and its shortlist commit match (`938d2aa0...`), so the live reads in [5][6] agree with the pinned revision as of the check date, but neither endpoint itself accepts a revision parameter to guarantee that going forward. Fetched 2026-08-12.

[14] Pipeline configuration file for the neighbor dataset. https://huggingface.co/datasets/mlfoundations-dev/a1_math_openmathinstruct2_aime/raw/main/config/a1_math_openmathinstruct2_aime.yaml - compared line-by-line against [1]; both load `AI-MO/NuminaMath-CoT` and differ only in one column-rename step. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable with a named restriction: train on the full 31,600-row `train` split, but hold out AIME/AMC-style evaluation benchmarks regardless of the pipeline's own decontamination pass. Two facts decide it, both established above: the row pool is drawn from AMC/AIME-flavored NuminaMath-CoT subsets and the served rows show verbatim, duplicated AMC problems rather than paraphrases [1][12], and the pipeline's fuzzy decontamination (threshold 75, 13-grams) reduces but does not eliminate overlap risk with the AIME/AMC benchmarks it targets [1].

### The screening row

The row's own note [10]: "Same historical amc_aime problems with DeepSeek-R1 traces, OpenMathInstruct lineage." Its flag [10]: "rule-risk: aime2025 - Row `source` field reads OpenMathInstruct-AIME and rows are AMC/AIME competition problems; card carries no decontamination claim."
