# nvidia/OpenMathInstruct-2

21,972,791 rows of math problem-solution pairs across four overlapping training splits, all synthesized by Llama-3.1-405B-Instruct from the GSM8K and MATH training sets.

**nvidia/OpenMathInstruct-2** is NVIDIA's math instruction-tuning dataset, introduced in "OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data" [1]: a teacher LLM (Llama-3.1-405B-Instruct) is prompted to write chain-of-thought solutions for the GSM8K and MATH training-set questions, and separately to generate new questions similar to those training questions together with their own synthesized solutions, yielding a reasoning-trace SFT corpus of problem/solution pairs [1][2]. **The dataset carries a decontamination pass against the GSM8K, MATH, AMC 2023, and AIME 2024 test sets, but not against the Omni-MATH test set, which was released after training finished; a post-hoc check found about 1.4% of Omni-MATH's test questions duplicated in the training data [1]. Hold out Omni-MATH (and re-verify the other four test sets) before using this corpus's models for a scored run.** It lives at https://huggingface.co/datasets/nvidia/OpenMathInstruct-2 .

**Use it for**: reasoning-trace SFT - each row is a `problem` paired with a `generated_solution` chain-of-thought, the shape the SFT method card expects; there is no chat template on these fields, they are plain strings rather than turn lists [3]. The Omni-MATH contamination caveat above is the one restriction to carry into any evaluation-driven use.

**Licence**: CC-BY-4.0 (`cardData.license` is `"cc-by-4.0"`, tag `license:cc-by-4.0`), ungated (`"gated": false`, `"private": false`) [4]. The paper frames this permissive license as a deliberate design choice: it contrasts itself with NuminaMath, whose restrictive license the authors attribute, with a hedge ("likely due to"), to its use of GPT-4o, and with MetaMathQA and MathInstruct, which the paper says only "have also utilized GPT models for data synthesis" without naming which one - this release instead uses only Llama-3.1-405B-Instruct outputs, described by the authors as "commercially permissive" [1].

**Shape**: one config (`default`), four splits that are not disjoint - `train` (13,972,791), `train_1M` (1,000,000), `train_2M` (2,000,000), `train_5M` (5,000,000) - for 21,972,791 rows served in total; four string columns (`problem`, `generated_solution`, `expected_answer`, `problem_source`) [3][5].

**Hold out**: the GSM8K, MATH, AMC 2023, and AIME 2024 test sets, which the LLM decontamination pipeline checked the synthesized questions against; and the Omni-MATH test set, which it did not check, with a stated ~1.4% overlap found afterward [1]. No row counts are given for either the removed or the overlapping questions beyond what the paper states (see Quality).

**Origin**: built by NVIDIA (Toshniwal et al.); the questions and solutions are Llama-3.1-405B-Instruct generations, with correctness against the ground-truth answer used to filter solutions for original GSM8K/MATH questions and majority voting among 32 sampled solutions used as a proxy answer for newly synthesized questions - no human labeling [2]. Hub API as of the check date: `downloads` 94,880, `likes` 252 [4].

**Trained-on-by**: NVIDIA's own OpenMath2-Llama3.1-8B and OpenMath2-Llama3.1-70B, the models the origin paper releases to demonstrate the data's quality; the paper's own headline comparison is that finetuning Llama-3.1-8B-Base on this dataset outperforms Llama-3.1-8B-Instruct on MATH by an absolute 15.9 points (51.9% to 67.8%) [1][2]. AMD's `amd/Instella-3B-Math` and `amd/Instella-3B-Math-SFT` name this dataset in their pinned `datasets:` metadata and describe using it for "Stage 1" instruction tuning on math prompts [6].

**Introduced by**: [1] (Toshniwal et al.).

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 13,972,791 |
| `train_1M` | 1,000,000 |
| `train_2M` | 2,000,000 |
| `train_5M` | 5,000,000 |
| total | 21,972,791 |

One config, `default`, with four columns (datasets-server `/info`) [3]:

| column | dtype |
| --- | --- |
| `problem` | string |
| `generated_solution` | string |
| `expected_answer` | string |
| `problem_source` | string |

The README states `train_1M`, `train_2M`, and `train_5M` are "fair-downsampled" versions of the full training set, released to match the points plotted in the paper's data-scaling curve, not additional data on top of `train` [2]. The paper gives the corpus-level totals as about 14M question-solution pairs over roughly 600K unique questions, of which 592K are synthesized (the remainder are the original GSM8K/MATH training questions) [1].

Byte sizes from the live datasets-server `/size` endpoint [5]: 12,628,023,456 bytes of Parquet download (matching the original-files figure), 25,959,586,261 bytes decoded in memory. No source states sequence-length or token-count statistics for the whole corpus; the one length figure the README gives is qualitative - 564 questions (about 0.1% of the corpus) run longer than 1024 Llama tokens and were not filtered out of the release [2].

## Quality

- Solutions for the original GSM8K/MATH training questions are filtered by matching the model's final answer against the ground-truth answer; solutions for newly synthesized questions (which have no ground truth) are instead kept when they agree with a majority vote across 32 sampled solutions for that question, and that majority answer becomes the row's `expected_answer` [1][2].
- The paper's LLM-based decontamination pipeline - embedding similarity search for the top-5 most similar test questions, followed by an LLM (Llama-3.1-405B-Instruct) paraphrase judgment in both pairing orders - removed about 50K of the 569K newly synthesized questions as paraphrases of GSM8K, MATH, AMC 2023, or AIME 2024 test questions (569K to 519K) [1].
- Omni-MATH was released after the authors finished training and so was not included in that decontamination pass; a post-hoc check found about 1.4% of the Omni-MATH test set duplicated in the training data [1].
- The README's own quality note: 564 questions (about 0.1%) are longer than 1024 Llama tokens and were not filtered from the release; the authors report that removing these in their own experiments did not hurt performance (they saw a minor improvement) and recommend filtering them, pointing to their NeMo-Skills documentation for the exact commands [2].
- No source states a measured duplicate-row rate or annotator-agreement figure for this corpus, and none is invented here; there was no human annotation to measure agreement over (see Origin).

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-11-25) [4]:

```python
import datasets

REV = "469216e3f46f4dacf476b382e192485ea51a143e"  # main at the check date
train = datasets.load_dataset("nvidia/OpenMathInstruct-2", revision=REV, split="train")        # 13,972,791 rows
train_1m = datasets.load_dataset("nvidia/OpenMathInstruct-2", revision=REV, split="train_1M")  # 1,000,000 rows - subset of train, not additional
```

**Trap**: `train_1M`, `train_2M`, and `train_5M` are downsampled subsets drawn from the same pool as `train`, not extra data appended to it [2]. Concatenating more than one of the four splits (or `train` plus any of the three) duplicates rows rather than adding coverage; pick exactly one split for a given training run. The README also warns that loading and converting the full `train` split to JSONL takes 20-30 minutes and about 20GB of RAM [2]. The `sha` above pins what `load_dataset` returns. It does not cover this card's row counts, byte sizes, and sampled row - those come from the datasets-server `/size`, `/info`, and `/first-rows` endpoints [3][5][10], which take no revision parameter and reflect the repository's current state at fetch time rather than the pinned commit; treat those numbers as live, not reproducible against `REV`.

## Neighbors

- `nvidia/OpenMathInstruct-1` - the same builder's predecessor, introduced in "OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset" [7]. Its README states 1.8M problem-solution pairs, but a live fetch of its splits shows 5,752,065 rows in `train` plus 1,127,629 in `validation`, 6,879,694 total - more rows than the paper's headline count, consistent with the served rows being pre-filter generations rather than only the 1.8M pairs kept as correct [8]. Its schema differs from this release: a sampled row carries `question`, `expected_answer`, `predicted_answer`, `error_message`, `is_correct`, `dataset`, `generation_type`, and a `generated_solution` that interleaves Python code blocks with text, because its solutions were produced by Mixtral-8x7B with code execution rather than Llama-3.1-405B-Instruct's pure-text chain-of-thought [8][9]. Its license is `other` / `nvidia-license`, not CC-BY-4.0 [9]. This release supersedes it for anyone wanting Llama-3.1-generated, pure-text CoT solutions under a standard open license; reach for OpenMathInstruct-1 only if code-execution-grounded solutions are specifically wanted.
- The paper names NuminaMath, MetaMathQA, and MathInstruct as related open math instruction datasets with more restrictive, non-commercial-leaning licenses than this release; it attributes NuminaMath's restrictive license, with a hedge, to its use of GPT-4o, and says only that MetaMathQA and MathInstruct "have also utilized GPT models for data synthesis" without naming which model [1].
- The dataset's own repository links a companion contamination-explorer Space (`nvidia/OpenMathInstruct-2-explorer`) for inspecting which training rows are flagged as similar to GSM8K, MATH, AMC 2023, AIME 2024, and Omni-MATH test problems; it is a browsing tool, not an alternative training corpus [2].

## A row

The repository serves one config and one schema across all four splits - the first 100 rows read at offset 0 from each of `train`, `train_1M`, `train_2M`, and `train_5M` all carry the same four columns, so one row covers the shape. The `problem_source` values seen in those same 100-row samples differ by split: `train` and `train_2M` show all four values (`math`, `gsm8k`, `augmented_math`, `augmented_gsm8k`); `train_1M` shows only `augmented_math`, `gsm8k`, and `augmented_gsm8k`; `train_5M` shows only `math`, `augmented_math`, and `augmented_gsm8k` - a sampling effect of reading 100 rows at offset 0 from each, not a claim about the full splits [10]. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [10]:

```json
{
  "problem": "Ava is planning a camping trip with her friends. She wants to make sure they have enough granola bars for snacks. There will be five people total: Ava, her two friends, and her parents. They will spend 3 days and 2 nights at the campsite, and they plan to have 2 granola bars per person for breakfast and 1 granola bar per person for an afternoon snack each day. How many granola bars will Ava need to pack in total for the entire trip?",
  "generated_solution": "There will be a total of 5 people.\nEach person needs 2 granola bars for breakfast and 1 granola bar for snack. This amounts to a total of 3 granola bars per person per day.\nSince the trip is 3 days long, each person will need 3 granola bars/day * 3 days = 9 granola bars.\nSo for 5 people, Ava will need 5 * 9 = 45 granola bars.\nThus, Ava will need to pack \\boxed{45} granola bars in total for the entire trip.",
  "expected_answer": "45",
  "problem_source": "augmented_gsm8k"
}
```

## Where it came from

Built and released by NVIDIA (Toshniwal, Du, Moshkov, Kisacanin, Ayrapetyan, Gitman). The pipeline starts from the training splits of GSM8K and MATH [1][2]. Two generation modes feed the corpus: solution augmentation, where Llama-3.1-405B-Instruct writes new chain-of-thought solutions for the original training questions; and question-solution augmentation, where the same model is few-shot prompted (with example questions and human-written similar questions) to write new questions in the style of the training set, followed by its own solutions for those new questions [1][2]. Solutions for original questions are kept when their final answer matches the known ground truth; solutions for the newly synthesized questions, which have no ground truth, are kept via majority voting across 32 samples per question [1]. The synthesized questions then pass through an LLM-based decontamination pipeline checking them against the GSM8K, MATH, AMC 2023, and AIME 2024 test sets before being folded into the release [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Toshniwal, Du, Moshkov, Kisacanin, Ayrapetyan, Gitman, "OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data", 2024. https://arxiv.org/abs/2410.01560 - the origin paper; current title read from the live abs page, full text read from the arXiv-served PDF (https://arxiv.org/pdf/2410.01560), since the HTML rendering for this paper returns no article content. Fetched 2026-08-11.

[2] nvidia/OpenMathInstruct-2 dataset card (README). https://huggingface.co/datasets/nvidia/OpenMathInstruct-2/raw/main/README.md - dataset description, field descriptions, split descriptions, load instructions, the long-question note, links to the paper and explorer Space. Fetched 2026-08-11.

[3] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=nvidia%2FOpenMathInstruct-2 - column names and dtypes, confirming `problem` and `generated_solution` are plain strings with no role/turn structure. Fetched 2026-08-11.

[4] Hugging Face Hub API record for nvidia/OpenMathInstruct-2. https://huggingface.co/api/datasets/nvidia/OpenMathInstruct-2?full=true - license, gate, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=nvidia%2FOpenMathInstruct-2 Fetched 2026-08-11.

[6] amd/Instella-3B-Math model card (README). https://huggingface.co/amd/Instella-3B-Math/raw/main/README.md - pinned `datasets:` metadata naming `nvidia/OpenMathInstruct-2`, and the "Stage 1: Instruction Tuning with OpenMathInstruct-2" section describing its use. Fetched 2026-08-11.

[7] Toshniwal et al., "OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset", 2024. https://arxiv.org/abs/2402.10176 - current title read from the live abs page. Fetched 2026-08-11.

[8] datasets-server size and first-rows endpoints for nvidia/OpenMathInstruct-1. https://datasets-server.huggingface.co/size?dataset=nvidia%2FOpenMathInstruct-1 and https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FOpenMathInstruct-1&config=default&split=train - live row counts and a sampled row's columns; these endpoints take no revision parameter, so the counts are live, not pinned. Fetched 2026-08-11.

[9] nvidia/OpenMathInstruct-1 dataset card (README). https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/raw/main/README.md - stated pair count, field descriptions, license (`license: other`, `license_name: nvidia-license`). Fetched 2026-08-11.

[10] datasets-server first-rows endpoint, one call per split. https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FOpenMathInstruct-2&config=default&split=train (and `&split=train_1M`, `&split=train_2M`, `&split=train_5M`) - schema and `problem_source` values read from the first 100 rows of each split at offset 0. Fetched 2026-08-11.

[11] The corpus screening row for `nvidia/OpenMathInstruct-2`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT training data, restricted to the four training splits. The screening row's own note already states the basis: the corpus is built from GSM8K and MATH training problems (plus augmented variants derived from them) with Llama-3.1-405B-Instruct chain-of-thought solutions, and the card is explicit that only training splits were used - `train`, `train_1M`, `train_2M`, and `train_5M` are all training subsets, with no eval split shipped [11]. The Omni-MATH contamination gap documented above (Quality, Hold out) is a use-time caveat for anyone evaluating a model trained on this data, not a reason to exclude the dataset from training use.

### The screening row

The row's own note [11]: "GSM8K+MATH TRAIN problems (plus augmented variants) with CoT solutions from Llama-3.1-405B-Instruct; card explicit that only training splits were used; train/train_1M/2M/5M are all training subsets." The row carries no flag.
