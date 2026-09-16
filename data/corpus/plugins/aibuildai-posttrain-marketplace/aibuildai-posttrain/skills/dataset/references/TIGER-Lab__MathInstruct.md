# TIGER-Lab/MathInstruct

262,039 instruction/output math-reasoning pairs, single-turn, blending chain-of-thought (CoT) prose and program-of-thought (PoT) Python solutions across 13 source datasets, six of them newly rationale-curated by the releasing paper.

**TIGER-Lab/MathInstruct** is TIGER-Lab's instruction-tuning dataset for the MAmmoTH models, introduced in "MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning" [1]. It is compiled by combining 13 existing math datasets - some with their original human or model rationales, six with new CoT or PoT rationales the authors generated with GPT-4 - into one instruction/output schema mixing natural-language reasoning and executable Python solutions [1][2]. It lives at https://huggingface.co/datasets/TIGER-Lab/MathInstruct . **Two usage-shape restrictions apply. First, the repository's `license: mit` tag covers only the newly curated portions and the MIT-licensed source subsets; the README's own per-subset table lists the Camel-Math subset (about a fifth of all rows) under a noncommercial licence and GSM8K-RFT under no listed licence, so this is not a clean-MIT corpus for commercial use [2]. Second, the training mixture embeds actual TheoremQA benchmark questions with newly written rationales, so a model trained on this data should not be scored against the TheoremQA benchmark, and the MATH-sourced rows carry a similar rule-risk against AIME-style benchmarks such as AIME 2025, since the screening record traces MATH's problems to AMC/AIME-origin competitions [3].**

**Use it for**: reasoning-trace SFT - single-turn instruction-to-solution pairs where the solution is either CoT prose or a PoT Python program, not a multi-turn chat format. The served columns are `instruction`, `output`, `source`; the shortlist's own `chat_dialect` field records "none" for this dataset, a literal value the row does not further define. This maps to the SFT method card; no chat template or prompt wrapper is applied. See the licence and TheoremQA/AIME restrictions above before using for commercial or benchmark-adjacent training.

**Licence**: repo-level tag `mit` (SPDX MIT), ungated [4]. The catch: the README's own per-subset licence table names Camel-Math (about 50K of the paper's declared per-subset counts, roughly a fifth of the corpus) as Attribution-NonCommercial 4.0 International, and GSM8K-RFT as "Non listed" - both under the same MIT-tagged repository [2].

**Shape**: 262,039 rows, one split (`train`), one config (`default`), three string columns [5][6].

**Hold out**: not "nothing" - two named risks. The paper states it synthesized new CoT and PoT rationales for TheoremQA's own questions and folded them into training (600 CoT + 700 PoT rows in the paper's own count) [1], so any TheoremQA benchmark run on a model trained on this data is contaminated; hold those rows out before a TheoremQA score. The PoT file is confirmed as `data/PoT/TheoremQA.json` in the 300-row sample [7]; a parallel `data/CoT/TheoremQA.json` file is inferred from Table 1's 600-row CoT count but was not itself observed in that sample [1][7]. Separately, the corpus screening record traces the MATH-derived rows to Hendrycks' MATH, whose problems it says come from AMC/AIME-style competitions, and flags this as a rule-risk against AIME 2025 scoring even though no row is a literal AIME-2025 problem [3]. The screening row itself states that the GSM8K subsets are train/RFT-derived and that the standard test probe stays clean [3]; a 300-row sample of the served `source` values (offsets 0, 100,000, 200,000) is consistent with that, showing GSM8K entries only under `gsm_train.json`, `gsm_rft.json` and `gsm_gpt4.json`, and MATH entries only under `MATH_train.json` (CoT and PoT) - none named as an official test split - though the sample does not cover every row, so it cannot rule out a test-split file elsewhere in the corpus [7].

**Origin**: built by TIGER-Lab (Xiang Yue et al.); rationale sources are a mix of the original datasets' human solutions, Llama-generated rejection-sampling rationales (GSM8K-RFT), and GPT-4-written or GPT-4-supplemented rationales added by this work [1]. Hub API at the check date: 16,768 downloads, 178,801 all-time downloads, 306 likes [4].

**Trained-on-by**: the paper's own MAmmoTH and MAmmoTH-Coder models (7B/13B/34B/70B, on Llama-2 and Code Llama bases) [2]. Beyond those, the Hub model search for "MathInstruct" returns only smaller, unofficial community fine-tunes (e.g. `Wanfq/MathInstruct-Mistral-7b`, `kyungeun/gemma-2-9b-it-mathinstruct`); no other named model or training recipe is documented as adopting it [8].

**Introduced by**: [1] (Yue et al., "MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning").

## Shape

Split and size (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 262,039 |
| total | 262,039 |

One config, `default`, three columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `source` | string |
| `output` | string |
| `instruction` | string |

The paper's Table 1 lists the 13 source datasets that make up the mixture, with its own declared per-subset sizes, rationale type, annotation origin, and (from the README's licence table) the licence each carries [1][2]. No source states the exact row count per subset in the served 262,039-row file; the counts below are the paper's own approximate figures, and they sum to about 260K against the paper's own stated 260K total:

| source dataset | rationale | declared size | annotation | licence (README) |
| --- | --- | --- | --- | --- |
| GSM8K | CoT | 7K | human | MIT |
| GSM8K | PoT | 14K | GPT-4 + validated | MIT |
| GSM8K-RFT | CoT | 28K | Llama + validated | Non listed |
| AQuA-RAT | CoT | 90K | human | Apache 2.0 |
| AQuA-RAT | PoT | 9.7K | GPT-4 + validated | Apache 2.0 |
| MATH | CoT | 7K | human | MIT |
| MATH | PoT | 7K | GPT-4 + validated | MIT |
| TheoremQA (new rationales) | CoT | 600 | GPT-4 + validated | MIT |
| TheoremQA (new rationales) | PoT | 700 | GPT-4 + validated | MIT |
| Camel-Math | CoT | 50K | GPT-4, unvalidated | CC BY-NC 4.0 |
| College-Math (new) | CoT | 1.8K | GPT-4, unvalidated | MIT ("Our Curated") |
| MathQA | PoT | 25K | human | Apache-2.0 |
| NumGLUE | PoT | 13K | human | Apache-2.0 |

Sampling 300 rows at offsets 0, 100,000 and 200,000 (100 rows each, via datasets-server `/rows`) found AQuA-RAT CoT rows most frequent (about 35% of the sample), Camel-Math CoT next (about 19%), and the remaining eleven `source` values each under 10%; a `data/CoT/number_comparison.json` source not named in the paper's Table 1 also appeared once, likely a NumGLUE subtask file [9]. Original download is 212,488,891 bytes (one JSON file, `MathInstruct.json`); as Parquet it is 99,132,648 bytes; decoded in memory it is 188,742,872 bytes [5]. No source states token or sequence-length statistics for this release.

## Quality

- The paper's own ablation on data source is the number that decides whether the hybrid design earns its name: fine-tuning on the CoT subset alone raised overall accuracy across the paper's nine evaluation datasets from a 27% baseline to 32%, the PoT subset alone raised it to 41%, and combining both as the hybrid mixture reached 47.9% - a further 6.9-point gain over PoT alone and 15.9 points over CoT alone [1].
- The paper reports it filtered its GPT-4-synthesized PoT programs (for MATH, AQuA-RAT, GSM8K, and TheoremQA) by executing them and comparing the result against the human-annotated ground-truth answer, discarding mismatches [1].
- The paper's own composition table marks the Camel-Math and College-Math CoT rationales as "GPT-4 (Unvalidated)", i.e. not passed through that execution-match filter, unlike the other GPT-4-authored subsets [1].
- No source states a measured duplicate rate, contamination rate against the standard GSM8K/MATH test sets, or human-annotator agreement figure for this release; the evidence that GSM8K and MATH rows draw from training rather than test splits is the screening row's own statement plus the sampled `source` filenames described in Hold out above, and it does not cover every row [3][7].

## Load it

Pin the revision this card's numbers were read at - the Hub API's `sha`, which matches the shortlist's recorded commit; the repo was last modified 2024-05-15 [4]:

```python
import datasets

REV = "b4fdc323a7be1379c9c7c0b67b1de72dfee2111a"  # main at the check date
train = datasets.load_dataset("TIGER-Lab/MathInstruct", revision=REV, split="train")  # 262,039 rows
```

**Trap**: the repository ships a single `MathInstruct.json` file and a single `train` split with no held-out eval split of its own - a chooser must carve out their own holdout, and the TheoremQA-sourced and MATH-sourced rows named above need to be identified and excluded before any TheoremQA- or AIME-style benchmark run, since `load_dataset` with no filtering loads everything into one undifferentiated `train` split [5][1].

## Neighbors

- `typeof/TIGER-Lab-MathInstruct_PoT` - 29,344 rows, a PoT-only subset; its own README says "SEE https://huggingface.co/datasets/TIGER-Lab/MathInstruct. This is only here for convenience", i.e. a community-maintained filtered mirror, not an independent release [10][11].
- `jan-hq/math_instruct_binarized` - 262,040 rows reformatted into a `messages` chat-role list and split into `train` (235,836) and `test` (26,204); its card states no split methodology ("More Information needed"), so it is unknown whether that test split avoids the TheoremQA/MATH contamination named above [10][12]. Prefer the original `TIGER-Lab/MathInstruct` and build a holdout using the source-level exclusions above rather than trusting this split without knowing how it was drawn.
- No other same-builder reformatting, cleaned, or successor release was found; other Hub hits for "MathInstruct" are unofficial third-party format conversions or fine-tuning checkpoints, not TIGER-Lab releases [13].

## A row

One config, one split, so both rationale styles are shown since they differ substantially in downstream handling. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`), a CoT (prose) row [14]:

```json
{
  "source": "data/CoT/aqua_rat.json",
  "instruction": "The distance between two stars is 6.52 × 10^5 light years. What is the distance between the two stars in parsecs? (1 parsec = 3.26 light years)\nAnswer Choices: (A) 2 × 10^5 (B) 4 × 10^6 (C) 5 × 10^7 (D) 7 × 10^7 (E) 9 × 10^8",
  "output": "Let's think about the multi-choice question.\n6.52 × 10^5 ly / (3.26 ly/parsec) = 2 x 10^5 persec\nThe answer is A."
}
```

From the same config and split, `row_idx=23`, a PoT (Python) row - one of the TheoremQA-sourced rows named in Hold out above [14]:

```json
{
  "source": "data/PoT/TheoremQA.json",
  "instruction": "The planet Mercury travels around the Sun with a mean orbital radius of 5.8x10^10 m. The mass of the Sun is 1.99x10^30 kg. Use Newton's version of Kepler's third law to determine how long it takes Mercury to orbit the Sun. Give your answer in Earth days. Let's write a Python program to solve it.",
  "output": "import math\ndef solve():\n  G = 6.6743 * 10**(-11) # gravitational constant\n  M = 1.99 * 10**30 # mass of the Sun\n  r = 5.8 * 10**10 # mean orbital radius of Mercury\n  T = 2 * math.pi * math.sqrt(r**3 / (G * M)) # Kepler's third law\n  T = T / (60 * 60 * 24) # convert seconds to days\n  return T\nprint(solve())"
}
```

## Where it came from

Built by TIGER-Lab (Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen) and released alongside the MAmmoTH paper [1]. The paper states the goal was to combine "a few high-quality datasets that are widely adopted" (GSM8K, MATH, AQuA-RAT, Camel-Math, TheoremQA) with newly authored coverage of gaps such as college-level math, and to supplement CoT-only datasets with GPT-4-synthesized PoT programs [1]. Concretely: GSM8K, AQuA-RAT, MATH, MathQA and NumGLUE contribute their original human-written solutions or exam-derived items; GSM8K-RFT contributes Llama-generated, rejection-sampling-filtered rationales from a prior paper; Camel-Math contributes unfiltered GPT-4-generated rationales from its own source repository; and the authors themselves used GPT-4 to write new CoT rationales for TheoremQA's questions and new PoT programs for MATH, AQuA-RAT, GSM8K and TheoremQA, filtering the PoT additions by execution match against ground truth, and wrote College-Math from scratch via Self-Instruct with seed exemplars [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Yue et al., "MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning", 2023. https://arxiv.org/abs/2309.05653 - the origin paper; Table 1 composition table, abstract's deciding numbers, methodology for rationale synthesis and filtering. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2309.05653); current title read from the live abs page. Fetched 2026-08-11.

[2] TIGER-Lab/MathInstruct dataset card (README). https://huggingface.co/datasets/TIGER-Lab/MathInstruct/raw/main/README.md - per-subset licence table, MAmmoTH model list, citation. Fetched 2026-08-11.

[3] The corpus screening row for `TIGER-Lab/MathInstruct`, supplied with this card's request - its `flag` field, read back in the row's own terms in the appendix. Checked 2026-08-11.

[4] Hugging Face Hub API record for TIGER-Lab/MathInstruct. https://huggingface.co/api/datasets/TIGER-Lab/MathInstruct?full=true - licence tag, gate status, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=TIGER-Lab%2FMathInstruct Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=TIGER-Lab%2FMathInstruct Fetched 2026-08-11.

[7] datasets-server rows endpoint, `source` values sampled at offsets 0, 100,000 and 200,000: GSM8K rows under `gsm_train.json`, `gsm_rft.json` and `gsm_gpt4.json`, MATH rows under `MATH_train.json` (CoT and PoT), none named as an official test split. https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FMathInstruct&config=default&split=train Fetched 2026-08-11.

[8] Hugging Face Hub model search for "MathInstruct". https://huggingface.co/api/models?search=MathInstruct&limit=30 - a live, unpinned search endpoint. Fetched 2026-08-11.

[9] datasets-server rows endpoint, three 100-row samples at offsets 0, 100,000 and 200,000. https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FMathInstruct&config=default&split=train&offset=0|100000|200000&length=100 Fetched 2026-08-11.

[10] datasets-server size endpoint, one call per neighbor: `typeof/TIGER-Lab-MathInstruct_PoT`, `jan-hq/math_instruct_binarized`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[11] `typeof/TIGER-Lab-MathInstruct_PoT` dataset card (README). https://huggingface.co/datasets/typeof/TIGER-Lab-MathInstruct_PoT/raw/main/README.md Fetched 2026-08-11.

[12] `jan-hq/math_instruct_binarized` dataset card (README) and first-rows sample. https://huggingface.co/datasets/jan-hq/math_instruct_binarized/raw/main/README.md and https://datasets-server.huggingface.co/first-rows?dataset=jan-hq%2Fmath_instruct_binarized&config=default&split=train Fetched 2026-08-11.

[13] Hugging Face Hub dataset search for "MathInstruct". https://huggingface.co/api/datasets?search=MathInstruct&limit=20 - a live, unpinned search endpoint. Fetched 2026-08-11.

[14] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=TIGER-Lab%2FMathInstruct&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data, with two named holdouts rather than a clean bill of health. The screening row's own flag identifies a minor contamination risk - TheoremQA benchmark questions embedded in the training mixture with new rationales, harmless for scoring GSM8K but not TheoremQA - plus a rule-risk against AIME-style benchmarks (named as AIME 2025) because the MATH subset traces to AMC/AIME-origin competition problems; it also notes the GSM8K subsets are train/RFT-derived and that the standard test probe stays clean [3]. Both risks are carried into the Hold out line and opening paragraph above, alongside the noncommercial-licence catch on the Camel-Math subset found independently in the README [2].

### The screening row

The row's own note [3]: "13 sources blending CoT and program-of-thought rationales; sampling shows aqua_rat, camel, gsm_rft, gsm_train, MATH_train, mathqa, numglue; several rationales GPT-4 written." Its flag [3]: "contamination (minor): TheoremQA benchmark items found inside the training mixture; harmless for a GSM8K score, not for a TheoremQA one; rule-risk: aime2025 - Subset table names MATH (Hendrycks), whose problems come from AMC/AIME competitions; GSM8K subsets are train/RFT, test probe clean."
