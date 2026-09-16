# TIGER-Lab/MMLU-Pro

12,102 multiple-choice knowledge-and-reasoning questions across 14 disciplines - a `test` split of 12,032 scored questions plus a `validation` split of 70 chain-of-thought few-shot exemplars - the MMLU-Pro benchmark.

**TIGER-Lab/MMLU-Pro** is a harder, more option-rich successor to MMLU, introduced in "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark" [1]. It pools retained MMLU questions with new questions hand-picked from a STEM website, TheoremQA, and SciBench, expands each question from four options up to ten with GPT-4-generated distractors, and puts every question and option set through expert review [2]. It is a question-answering benchmark, not a training corpus. **The entire dataset is an evaluation benchmark: both splits must be held out of any training run, and training corpora should be decontaminated against it before a scored run.** It lives at https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro .

**Use it for**: nothing in training - this is an eval-only benchmark, so the usable action is decontamination (screen training corpora for near-duplicates of the `question` field) and scored evaluation, not any training shape (SFT, preference, or reward-model). The repository ships its own `eval.yaml` for the `inspect-ai` framework, pointing the `multiple_choice` solver and `choice` scorer at the `default` config's `test` split with `question` as input, `options` as choices, and `answer` as target [3]. No method card in this corpus consumes it as training data.

**Licence**: MIT (`cardData.license` is `"mit"`, repo tag `license:mit`) [4]. The one catch: MIT covers reuse of the data, not a licence to train on it undetected - the dataset's evaluation purpose is a usage-shape fact stated above, not a licence term.

**Shape**: 12,102 rows in one config (`default`), split `test` 12,032 / `validation` 70, eight columns (`question_id`, `question`, `options`, `answer`, `answer_index`, `cot_content`, `category`, `src`) [5][6].

**Hold out**: everything - all 12,102 rows across both splits. This is the benchmark itself, so there is no train/test split to carve inside it; the risk is contamination of some other training corpus by this benchmark, not the reverse.

**Origin**: built by TIGER-Lab, mixing retained human-authored MMLU questions, hand-picked STEM-website/TheoremQA/SciBench questions, GPT-4-generated distractor options, and human expert review [2]. Hub API at the check date: `downloads` 169,025, `downloadsAllTime` 2,059,715, `likes` 510 [4].

**Trained-on-by**: none found - this is an evaluation benchmark and the dataset card's own leaderboard section describes running many models (GPT-4o, Claude-3-Opus/Sonnet, Gemini 1.5 Flash, Llama-3-70B-Instruct, Mixtral-8x7B, and others) *on* it for scoring, not training *on* it [2].

**Introduced by**: [1] (Wang et al.).

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `test` | 12,032 |
| `validation` | 70 |
| total | 12,102 |

One config, `default`, with eight columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `question_id` | int64 |
| `question` | string |
| `options` | list\<string\> |
| `answer` | string |
| `answer_index` | int64 |
| `cot_content` | string |
| `category` | string |
| `src` | string |

`category` takes one of 14 values across the dataset - math, physics, chemistry, law, engineering, other, economics, health, psychology, business, biology, philosophy, computer science, history - and the README's own discipline table gives the `test`-split count for each, from 1,351 (math) down to 381 (history), splitting each into "From Original MMLU" and "Newly Added" counts [2]. No source states sequence-length or token statistics for either split.

The two splits differ in what fills their shared columns, not just in row count: in the 100 `test` rows read at offset 0, `cot_content` is empty (`""`) in all 100 and `src` values start with `ori_mmlu-`, `stemez-`, or `theoremQA-` [7]; in all 70 `validation` rows, `cot_content` holds a full worked chain-of-thought answer ending "The answer is (X)." and every `src` value starts with `cot_lib-` [8]. `validation` carries five rows per `category` (70 / 14), matching the README's note that leaderboard runs are "normally" 5-shot [2].

## Quality

- The dataset card documents an ongoing correction process rather than a one-time measured error rate: entries record specific `question_id` fixes (e.g., 15 medical-domain answers corrected with input from Mayo Clinic subspecialists; six specific IDs corrected on 2024-09-06; several math/engineering IDs corrected on 2024-08-07; one answer changed 2024-07-08; another 2024-07-05) and a 2026-01-18 fix for "a leading space issue in answer options" that "could have been exploited as a shortcut" in chemistry, physics, and other STEM subsets [2].
- The card states the option-augmentation step used GPT-4 to expand four-option MMLU questions to ten options, and that "over ten experts" then reviewed every question and its options for accuracy and fairness [2].
- The card reports its own robustness signal: across 24 tested prompt styles, model-score sensitivity to prompt variation was 2% on MMLU-Pro versus 4-5% on the original MMLU [2].
- No source states an annotator-agreement figure or a measured duplicate/contamination rate for MMLU-Pro itself.

## Load it

Both splits are served directly - no external fetch is needed to use either one. Pin the revision this card's numbers were read at (the shortlist row's commit, which matches the Hub API's live `sha` for `main` at the check date; the repo was last modified 2026-05-02) [4]:

```python
import datasets

REV = "b189ec765aa7ed75c8acfea42df31fdae71f97be"  # main at the check date
test = datasets.load_dataset("TIGER-Lab/MMLU-Pro", revision=REV, split="test")              # 12,032 rows - the scored benchmark
validation = datasets.load_dataset("TIGER-Lab/MMLU-Pro", revision=REV, split="validation")  # 70 rows - CoT few-shot exemplars
```

**Trap**: the dataset card's own YAML front matter states `download_size: 121157475` and `dataset_size: 8775905` bytes [2], while the live datasets-server `/size` endpoint reports 4,187,042 bytes of Parquet download and 8,006,031 bytes decoded in memory for the same commit [5] - the two sources disagree on download size by roughly 29x. Neither source explains the gap; do not assume either figure describes the other's quantity.

## Neighbors

- `cais/mmlu` - the original MMLU benchmark this dataset is built from and explicitly compares against in its own README [2]; the `all` config's `test` split holds 14,042 rows (versus 12,032 four-times-smaller-option-set questions here), plus a 1,531-row `validation` split, a 285-row `dev` split used for few-shot exemplars, and a separate 99,842-row `auxiliary_train` split not present in this repository at all [9]. This corpus's own README frames MMLU-Pro as the harder, ten-option, more CoT-dependent successor - use this repository, not `cais/mmlu`, when the goal is the harder benchmark; use `cais/mmlu`'s `auxiliary_train` only if training data is actually wanted, since neither split of MMLU-Pro is training data.
- No cleaned, binarized, or GPT-4-distilled successor release of MMLU-Pro itself was found in this search.

## A row

Two distinct served shapes - `test` rows carry an empty `cot_content`, `validation` rows carry a full worked answer - so both are shown.

From `config="default"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [7]:

```json
{
  "question_id": 70,
  "question": "Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.",
  "options": ["Safe practices, Fear, Jealousy, Trivial", "Unsafe practices, Distress, Joy, Trivial", "Safe practices, Wants, Jealousy, Trivial", "Safe practices, Distress, Fear, Trivial", "Unsafe practices, Wants, Jealousy, Serious", "Safe practices, Distress, Jealousy, Serious", "Safe practices, Wants, Fear, Serious", "Unsafe practices, Wants, Fear, Trivial", "Unsafe practices, Distress, Fear, Serious"],
  "answer": "I",
  "answer_index": 8,
  "cot_content": "",
  "category": "business",
  "src": "ori_mmlu-business_ethics"
}
```

From `config="default"`, `split="validation"`, `row_idx=0` (datasets-server `/first-rows`) [8]:

```json
{
  "question_id": 0,
  "question": "The symmetric group $S_n$ has $\n\\factorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of the ring 2Z.",
  "options": ["0", "30", "3", "10", "12", "50", "2", "100", "20", "5"],
  "answer": "A",
  "answer_index": 0,
  "cot_content": "A: Let's think step by step. A characteristic of a ring is R is $n$ if the statement $ka = 0$ for all $a\\in 2Z$ implies that $k$ is a multiple of $n$. Assume that $ka = 0$ for all $a\\in 2Z$ for some $k$. In particular $2k = 0$. Hence $k=0$ and $n=0$. The answer is (A).",
  "category": "math",
  "src": "cot_lib-abstract_algebra"
}
```

## Where it came from

Built by TIGER-Lab. The card describes construction in four stages: an initial filtering pass over the original MMLU dataset to keep only sufficiently difficult and relevant questions; collection of additional questions from a STEM website, TheoremQA, and SciBench, chosen for their ability to challenge advanced models; option augmentation, where GPT-4 expanded each question from four options to ten by generating distractors meant to require discriminative reasoning; and expert review, where "a panel of over ten experts" checked every question and option set for accuracy and fairness [2]. The `src` column preserves provenance per row, tagged `ori_mmlu-`, `stemez-`, `theoremQA-`, or (in `validation`) `cot_lib-` by source [7][8].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the datasets-server endpoints used here take no revision parameter and reflect the live `main` branch, which at the check date shares the same `sha` as the pinned revision.

[1] Wang et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark", 2024. https://arxiv.org/abs/2406.01574 - the origin paper; current title read from the live abs page. Fetched 2026-08-11.

[2] TIGER-Lab/MMLU-Pro dataset card (README). https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/raw/main/README.md - construction process, discipline table, correction log, prompt-sensitivity figures, YAML front-matter byte sizes. Fetched 2026-08-11.

[3] eval.yaml in the repository. https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/raw/main/eval.yaml - the inspect-ai evaluation config pointing at `default`/`test`. Fetched 2026-08-11.

[4] Hugging Face Hub API record for TIGER-Lab/MMLU-Pro. https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=TIGER-Lab%2FMMLU-Pro Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=TIGER-Lab%2FMMLU-Pro Fetched 2026-08-11.

[7] datasets-server first-rows endpoint, `split=test`. https://datasets-server.huggingface.co/first-rows?dataset=TIGER-Lab%2FMMLU-Pro&config=default&split=test Fetched 2026-08-11.

[8] datasets-server first-rows endpoint, `split=validation`. https://datasets-server.huggingface.co/first-rows?dataset=TIGER-Lab%2FMMLU-Pro&config=default&split=validation - returns all 70 rows. Fetched 2026-08-11.

[9] datasets-server size endpoint for the `all` config of cais/mmlu. https://datasets-server.huggingface.co/size?dataset=cais%2Fmmlu - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[10] The corpus screening row for `TIGER-Lab/MMLU-Pro`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Eval-only, hold out entirely. The dataset's own README frames it as a benchmark with a ready-made `inspect-ai` evaluation config [2][3], and the screening row's note names it as the MMLU-Pro benchmark with a small CoT-exemplar validation split - both facts already established above.

### The screening row

The row's own note [10]: "The MMLU-Pro benchmark; `test` plus a small `validation` of CoT exemplars." The row carries no flag.
