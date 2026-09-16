# nvidia/OpenMathInstruct-1

6,879,694 math-solution rows (5,752,065 train / 1,127,629 validation) for GSM8K and MATH training-set problems, each carrying a Mixtral-8x7B-generated code-interpreter solution and an `is_correct` flag - most rows are wrong-answer generations, not the "1.8M" figure in the dataset's own name.

**nvidia/OpenMathInstruct-1** is NVIDIA's synthetic math instruction-tuning corpus introduced in "OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset" [1]: the problems come from the GSM8K and MATH training subsets, and the solutions - a mix of natural-language reasoning and Python code blocks executed by an interpreter - were sampled from the permissively licensed Mixtral-8x7B model [2]. **The served rows are not the "1.8M" dataset the paper names: after merging and deduplicating, the paper counts 787K unique correct solutions for MATH and 1.04M for GSM8K, about 1.83M correct problem-solution pairs total [1]; the Hub repository instead serves every sampled attempt, correct and incorrect together, distinguished only by the row-level `is_correct` boolean, and a live sample of the `train` split shows the transition from `is_correct=True` to `is_correct=False` rows happening somewhere between row 1.4M and row 1.6M of 5,752,065 [3].** Filter to `is_correct == True` before using this as SFT data for correct reasoning traces. It lives at https://huggingface.co/datasets/nvidia/OpenMathInstruct-1 .

**Use it for**: reasoning-trace SFT on math word problems with interleaved code - pair `question` with `generated_solution`, after filtering to `is_correct == True` as described above [1][3]. This is plain instruction-response text, not a chat-templated or preference format; it maps to the SFT method card, not a preference or reward-model card. The `validation` split is not a benchmark: the paper describes it as an internal 1K-problem ablation/hyperparameter set carved out of the GSM8K/MATH training problems (with many sampled solutions per problem), used alongside `train` to train the paper's own released models, not as a held-out evaluation benchmark [1][2].

**Licence**: `license: other` / `license_name: nvidia-license` in the repo's card metadata, ungated (`"gated": false`) [4]. This is a custom NVIDIA License, not MIT - a repository-wide text-match on "MIT" is a false positive on the substring inside "per**mit**s" in the README's licensing sentence [2]. The README states the licence "permits commercial usage" [2]; the LICENSE file's own catch is that redistribution must keep a complete copy of the license and any existing notices, and that bringing a patent claim against a licensor immediately terminates your rights under it [5].

**Shape**: one `default` config, 8 string/bool columns, split `train` 5,752,065 rows / `validation` 1,127,629 rows, 6,879,694 rows total [6][7].

**Hold out**: nothing needs holding out from this corpus for GSM8K/MATH benchmark contamination - both splits are drawn only from the GSM8K and MATH *training* subsets, never their test/eval sets [1][2]. Before training, filter out `is_correct == False` rows (see the opening paragraph and Quality below) - that is a correctness filter, not a contamination hold-out.

**Origin**: built and released by NVIDIA (Toshniwal, Moshkov, Narenthiran, Gitman, Jia, Gitman) [1]; problems are from GSM8K/MATH, solutions are Mixtral-8x7B generations graded by the authors' own scripts, no human labeling [1][2]. Hub API at the check date: `downloads` 4,483, `downloadsAllTime` 50,541, `likes` 251 [4].

**Trained-on-by**: NVIDIA's own OpenMath-CodeLlama (7B/13B/34B/70B), OpenMath-Mistral-7B and OpenMath-Llama-2-70B models, trained on a combination of this dataset's `train` and `validation` splits [2]. Of the 42 entries in the Hub's dataset-filtered model list, 30 are NVIDIA's own checkpoints or third-party GGUF/EXL2/mlx requantizations of them, 3 are the Nextorage entry below and its own GGUF requantizations, and 9 remaining entries (`18Barz/The-Tunnel`, `Zayb1999/Supreme01`, `Kosasih/CyberCog_Revolution`, `razaobj/ttc`, `iAyman/GPT`, `koodi-ai/math-llama-2.5`, `9ai7/Test`, `Dijitaal/DijiHax`, `Kasmic/Plieas-GRPO-test`) carry the dataset tag but were not opened for this card, so what they trained is not stated here [8]. The one adoption case this card verified in full is `Nextorage/Llama-3.1-Swallow-8B-OpenMath-FT`, a fine-tune of `tokyotech-llm/Llama-3.1-Swallow-8B` for Japanese math word-problem solving whose own model card lists `nvidia/OpenMathInstruct-1` in its `datasets:` metadata [9]. That same model card states its actual training data was a 9,772-row, cleaned subset of a Japanese machine translation of the full corpus (1,825,008 rows), not the served English `train`/`validation` rows described elsewhere on this card [9].

**Introduced by**: [1] (Toshniwal et al., "OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset").

## Shape

Rows served and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 5,752,065 |
| `validation` | 1,127,629 |
| total | 6,879,694 |

Both `/size` and `/info` mark their response `"partial": true`, which datasets-server uses for repositories it only partially scanned when computing statistics; despite the flag, the two endpoints' row counts agree exactly with each other (`/info`'s per-split `num_examples` equal `/size`'s per-split `num_rows`) and with the Hub tree's file listing, so this card uses those agreeing counts rather than either endpoint's separate `estimated_num_rows` field, which does not match them [6][7][10].

One config, `default`, with eight columns (datasets-server `/info`, field meanings from the README) [7][2]:

| column | dtype | meaning |
| --- | --- | --- |
| `question` | string | original GSM8K or MATH training-set question |
| `expected_answer` | string | ground-truth answer from the source dataset |
| `predicted_answer` | string | Mixtral's answer, extracted from `\boxed{}` |
| `error_message` | string | `<not_executed>` if no code ran; otherwise empty, a Python exception, or `timeout` |
| `is_correct` | bool | whether the grading script judged the final answer correct |
| `generation_type` | string | `without_reference_solution` or `masked_reference_solution` |
| `dataset` | string | `gsm8k` or `math` |
| `generated_solution` | string | the synthesized solution, mixing text and executed code blocks |

Sizes (datasets-server `/size`) [6]: 2,858,982,712 bytes as Parquet, 5,989,783,861 bytes decoded in memory; `num_bytes_original_files` is not reported. The repository's own (pre-conversion) tree shows the row-level split behind these totals: `correct_solutions/train.jsonl` is 1,329,648,010 bytes against `incorrect_solutions/train.jsonl` at 6,421,849,826 bytes, and `correct_solutions/validation.jsonl` is 203,147,304 bytes against `incorrect_solutions/validation.jsonl` at 981,121,236 bytes - incorrect-solution files are roughly 4-5x the byte size of correct-solution files in both splits [10]. No source states sequence-length or token-count statistics for this dataset; "not stated" here.

## Quality

- The `is_correct` label is machine-graded by the authors' own scripts, not human-annotated; no source states a human-verification or spot-check rate for it [1][2].
- Sampling the served `train` split at offsets 900,000 / 1,200,000 / 1,400,000 / 1,600,000 / 1,800,000 (single-row fetches) returned `is_correct=True` through 1,400,000 and `is_correct=False` from 1,600,000 on, and offsets 2,000,000 and 5,700,000 (5-row fetches each) returned `is_correct=False` for all five rows fetched at each offset, showing the transition from correct to incorrect rows lands between row 1.4M and row 1.6M of the 5,752,065-row split - i.e., the file is not randomly shuffled by correctness, correct rows are concentrated at the front [3].
- The paper reports coverage after its data-selection pipeline: 85.9% of MATH training problems and 99.9% of GSM8K training problems end up with at least one correct solution in the corpus, using the paper's best prompting strategy [1].
- The paper states it did not attempt to filter "semantically noisy" solutions - ones that reach the right final answer through flawed reasoning - calling this "beyond the scope" of the work, though it reports such cases as anecdotally rare in the corpus [1].
- No source states a deduplication or contamination rate against the GSM8K/MATH *test* sets; the paper's own claim is that the problems are sourced only from the training subsets [1].

## Load it

Filter to `is_correct == True` for correct-only SFT, and pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-02-16) [4]:

```python
import datasets

REV = "4627efae6bd2ddcebb8acac00d513ffd8e00775c"  # main at the check date
train = datasets.load_dataset("nvidia/OpenMathInstruct-1", revision=REV, split="train")            # 5,752,065 rows
validation = datasets.load_dataset("nvidia/OpenMathInstruct-1", revision=REV, split="validation")  # 1,127,629 rows
train_correct = train.filter(lambda r: r["is_correct"])
```

**Trap**: loading either split whole and training on it directly mixes in the majority-incorrect rows described above - `is_correct` must be applied as a filter, it is not pre-split out into separate Hub splits (the `correct_solutions/` and `incorrect_solutions/` directories in the raw repository are merged into one `train`/`validation` pair by the Hub's auto-conversion) [2][10].

## Neighbors

- `nvidia/OpenMathInstruct-2` - the same builder's much larger 2024 follow-up: 21,972,791 rows total across a `train` split (13,972,791 rows) plus fixed-size `train_1M`/`train_2M`/`train_5M` subsamples of it, generated with Llama3.1 models instead of Mixtral, in a different four-column schema (`problem`, `generated_solution`, `expected_answer`, `problem_source`) with no `is_correct` column - its abstract describes itself as "nearly eight times larger than the previous largest open-source math reasoning dataset" [11][12]. Prefer OpenMathInstruct-2 for scale and a pre-filtered, single-quality-tier release; prefer OpenMathInstruct-1 when the correct/incorrect contrast itself (e.g., for a verifier or reward model) or Mixtral-specific traces are what's wanted.
- `nvidia/OpenMath-GSM8K-masked` (7,473 rows) and `nvidia/OpenMath-MATH-masked` (7,500 rows) - the masked reference solutions this dataset's own README says were used to produce its `generation_type="masked_reference_solution"` rows; not a training-data alternative, but the upstream input for part of this release [2][13].

## A row

Same eight-column schema in both splits, confirmed by fetching a row from each. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [14]:

```json
{
  "question": "Martha has 18 crayons. She lost half of them, so she bought a new set of 20 crayons. How many crayons in total does Martha have after the purchase?",
  "expected_answer": "29",
  "predicted_answer": "29",
  "error_message": "",
  "is_correct": true,
  "generation_type": "masked_reference_solution",
  "dataset": "gsm8k",
  "generated_solution": "Let's solve this problem using Python code.\n<llm-code>\namount_of_lost_crayons = 18 / 2\namount_of_new_crayons = 20\ntotal_amount = amount_of_lost_crayons + amount_of_new_crayons\ntotal_amount\n</llm-code>\n<llm-code-output>\n29.0\n</llm-code-output>\nThus Martha has \\boxed{29} crayons in total."
}
```

From `config="default"`, `split="validation"`, `row_idx=0` (datasets-server `/first-rows`) [15]:

```json
{
  "question": "A watermelon stand sold 46 watermelons. Seventeen customers bought one melon, three customers bought three melons, and the rest bought two melons. How many customers bought two watermelons?",
  "expected_answer": "10",
  "predicted_answer": "10",
  "error_message": "",
  "is_correct": true,
  "generation_type": "masked_reference_solution",
  "dataset": "gsm8k",
  "generated_solution": "Let's solve this problem using Python code.\n<llm-code>\ncustomers_1 = 17\ncustomers_3 = 3\nsold_melons = 46\n\n# customers that bought one watermelon\nnumber_melons_bought_1 = customers_1\n\n# customers that bought three watermelons\nnumber_melons_bought_3 = customers_3 * 3\n\n# customers that bought two water[...]"
}
```

## Where it came from

Built and released by NVIDIA. The problems are the training-set questions of GSM8K and MATH, two established math reasoning benchmarks; the solutions were synthesized by prompting the permissively licensed Mixtral-8x7B model to interleave natural-language reasoning with Python code blocks executed by an interpreter, using two generation strategies the paper calls `without_reference_solution` and `masked_reference_solution` (the latter using the masked reference solutions released separately as `nvidia/OpenMath-GSM8K-masked` and `nvidia/OpenMath-MATH-masked`) [1][2]. Each generated solution was graded by the authors' own script against the ground-truth `expected_answer` to produce the `is_correct` label; the correct and incorrect generations were then merged, deduplicated (for the correct subset), and released together, split into `train` and a smaller `validation` set carved out for the paper's own ablation and hyperparameter-selection experiments [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the datasets-server `/size`, `/info`, `/rows`, and `/first-rows` endpoints and the Hub tree/model-list APIs take no revision parameter, so their figures are live as of the check date, not pinned to the commit.

[1] Toshniwal, Moshkov, Narenthiran, Gitman, Jia, Gitman, "OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset", 2024. https://arxiv.org/abs/2402.10176 - the origin paper; current title, authors, and abstract read from the live abs page and its HTML rendering. Fetched 2026-08-11.

[2] nvidia/OpenMathInstruct-1 dataset card (README). https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/raw/main/README.md - dataset description, field definitions, masked-solutions links, model table, licence sentence. Fetched 2026-08-11.

[3] datasets-server rows endpoint, single-row fetches at offsets 900,000; 1,200,000; 1,400,000; 1,600,000; 1,800,000; 2,000,000; 5,700,000 on `split=train`. https://datasets-server.huggingface.co/rows?dataset=nvidia%2FOpenMathInstruct-1&config=default&split=train&offset=<n>&length=1 (or 5) - used to locate the correct/incorrect transition point. Fetched 2026-08-11.

[4] Hugging Face Hub API record for nvidia/OpenMathInstruct-1. https://huggingface.co/api/datasets/nvidia/OpenMathInstruct-1?full=true (and `?expand[]=downloadsAllTime`) - licence field, gate status, `sha`, `downloads`, `downloadsAllTime`, `likes`, last-modified date. Fetched 2026-08-11.

[5] NVIDIA License text (LICENSE file). https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/raw/main/LICENSE - redistribution and patent-termination terms. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=nvidia%2FOpenMathInstruct-1 Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=nvidia%2FOpenMathInstruct-1 Fetched 2026-08-11.

[8] Hugging Face Hub models API filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:nvidia/OpenMathInstruct-1&limit=50 - list of models tagged as trained on this dataset. Fetched 2026-08-11.

[9] Nextorage/Llama-3.1-Swallow-8B-OpenMath-FT model card (README). https://huggingface.co/Nextorage/Llama-3.1-Swallow-8B-OpenMath-FT/raw/main/README.md - third-party fine-tune naming this dataset in its `datasets:` metadata. Fetched 2026-08-11.

[10] Hugging Face Hub repository tree (recursive). https://huggingface.co/api/datasets/nvidia/OpenMathInstruct-1/tree/main?recursive=true - raw file listing and byte sizes of `correct_solutions/` and `incorrect_solutions/` train/validation files. Fetched 2026-08-11.

[11] Toshniwal et al., "OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data", 2024. https://arxiv.org/abs/2410.01560 - neighbor paper; current title and abstract read from the live abs page. Fetched 2026-08-11.

[12] datasets-server size and first-rows endpoints for nvidia/OpenMathInstruct-2. https://datasets-server.huggingface.co/size?dataset=nvidia%2FOpenMathInstruct-2 and https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FOpenMathInstruct-2&config=default&split=train - split row counts and column schema. Fetched 2026-08-11.

[13] nvidia/OpenMath-GSM8K-masked and nvidia/OpenMath-MATH-masked dataset cards and size endpoints. https://huggingface.co/datasets/nvidia/OpenMath-GSM8K-masked/raw/main/README.md, https://huggingface.co/datasets/nvidia/OpenMath-MATH-masked/raw/main/README.md, and their datasets-server `/size` endpoints - row counts. Fetched 2026-08-11.

[14] datasets-server first-rows endpoint, `split=train`. https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FOpenMathInstruct-1&config=default&split=train Fetched 2026-08-11.

[15] datasets-server first-rows endpoint, `split=validation`. https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FOpenMathInstruct-1&config=default&split=validation Fetched 2026-08-11.

[16] The corpus screening row for `nvidia/OpenMathInstruct-1`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data once filtered: the served rows mix correct and incorrect solution attempts under the `is_correct` flag, so training must select `is_correct == True` first, as established above from the paper's own 1.8M-correct-solution figure and this card's live sampling of the row order [1][3]. No benchmark-contamination hold-out is needed, since both splits draw only from the GSM8K/MATH training subsets, matching the screening row's own note [1][16].

### The screening row

The row's own note [16]: "Same GSM8K+MATH TRAIN problems with code-interleaved solutions from Mixtral-8x7B; its `validation` split is an internal ablation split, not a benchmark." The row carries no flag.
