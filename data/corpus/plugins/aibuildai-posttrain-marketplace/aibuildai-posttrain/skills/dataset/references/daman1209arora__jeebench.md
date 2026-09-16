# daman1209arora/jeebench

515 human-authored physics, chemistry and mathematics problems drawn from the IIT JEE-Advanced entrance exam, served as a single `test` split with four distinct answer-format types.

**daman1209arora/jeebench** is JEEBench, built by Daman Arora, Himanshu Singh and Mausam and introduced in "Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models" (EMNLP 2023) [1], which curates 515 pre-engineering problems from the IIT JEE-Advanced exam to test long-horizon reasoning over deep in-domain knowledge [1]. Each problem is one of four answer types - `MCQ` (single correct option), `MCQ(multiple)` (one or more correct options), `Integer`, or `Numeric` - and the repository's own scorer grades each type differently: exact set match for `MCQ`, partial credit for `MCQ(multiple)`, and a ±0.01 absolute-tolerance match for `Integer`/`Numeric` [2]. **This is an evaluation-only benchmark: it ships one `test` split and no train split, so the full 515 rows must be held out of any training run, not fed into it.** It lives at https://huggingface.co/datasets/daman1209arora/jeebench [14].

**Use it for**: scoring a trained model's math/physics/chemistry reasoning at eval time, never for training - there is no train split to draw from [3]. It does not map to a training method card; it belongs in an eval harness that prompts each row's `question`, parses the model's final answer by `type`, and scores it against `gold` with the type-specific rule above [2].

**Licence**: MIT (`cardData.license` is `"mit"`, repo tags include `license:mit`), ungated (`"gated": false`, `"private": false`) [3]. The one catch: the word "proprietary" that turns up in a scan of the repository is from the paper abstract embedded in the README's citation block ("evaluation on various open-source and proprietary models") [4], describing the models the paper tested, not a licence clause - no source imposes any restriction beyond the MIT grant.

**Shape**: 515 rows, one config (`default`), one split - `test` only, 6 columns [3][5].

**Hold out**: all 515 rows. The repository declares a single `test` split with no train counterpart [5], and the screening row's own note confirms the benchmark is `test` only [6] - there is nothing to partition, the entire file is held-out eval data.

**Origin**: built and released by the paper's authors from the real IIT JEE-Advanced exam papers [1]; questions and correct answers are human-authored (exam-setter content), not model-generated. Hub API at the check date: `downloads` 3,542, `downloadsAllTime` 15,651, `likes` 8 [3].

**Trained-on-by**: none found - as an eval-only benchmark it should not appear in training data. The paper's abstract states it evaluates open-source and proprietary models with self-consistency, self-refinement and chain-of-thought prompting, and that the best model, GPT-4, scores under 40% [1]. The repository's own scorer additionally names GPT-3, GPT-3.5, and a one-shot GPT-4 variant among the models it computed results for (`GPT3_normal`, `GPT3.5_normal`, `GPT4_CoT+OneShot`) [2].

**Introduced by**: [1] (Arora, Singh and Mausam, EMNLP 2023).

## Shape

Rows and the sole split (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `test` | 515 |

Columns, all in the `default` config (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `subject` | string |
| `description` | string |
| `gold` | string |
| `index` | int64 |
| `type` | string |
| `question` | string |

Read from the complete 515-row file (all rows, not a sample) [8]:

- `subject`: `math` 236, `chem` 156, `phy` 123.
- `type`: `MCQ(multiple)` 186, `Numeric` 137, `MCQ` 110, `Integer` 82.
- `description`: 16 distinct exam sittings, `JEE Adv 2016 Paper 1` through `JEE Adv 2023 Paper 2` (two papers per year, 2016-2023), 26-42 rows each.

No source states sequence-length or token statistics for this dataset; not stated.

Byte sizes (datasets-server `/size`) [5]: 374,599 bytes as the original `test.json`, 135,694 bytes as auto-converted Parquet, 299,941 bytes decoded in memory.

## Quality

- The deciding number is the paper's own evaluation ceiling: the best model it tests, GPT-4, scores under 40% even combined with self-consistency, self-refinement and chain-of-thought prompting [1].
- The paper further reports that by mere prompting, GPT-4 cannot properly weigh the exam's negative-marking risk, motivating a post-hoc confidence-thresholding method the paper builds on top of self-consistency [1].
- The repository's own scorer (`compute_metrics.py`) defines exact grading: `MCQ` needs the exact single-letter set to match; `MCQ(multiple)` gives 1.0 for an exact match, 0.25 per correctly-selected option when the response is a subset of the gold set, and 0.0 if any selected option is wrong; `Integer`/`Numeric` needs the response within absolute difference 0.01 of `gold` [2].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset; none is invented here.

## Load it

Only `test` exists - there is no `train` split to request. Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-12-18) [3]:

```python
import datasets

REV = "d070d1113ce365df05903763a2017ec2d0d56118"  # main at the check date
test = datasets.load_dataset("daman1209arora/jeebench", revision=REV, split="test")  # 515 rows - eval only, never train
```

**Trap**: the repository's file tree also carries `data.zip` (8,069,293 bytes) alongside `test.json`, `compute_metrics.py` and `inference.py` [9]; `load_dataset` only reads `test.json` through the auto-converted `default`/`test` config and never touches `data.zip`. The repo's own `inference.py` loads a separate `data/few_shot_examples.json` from inside that archive to build few-shot prompts [10] - that file is not exposed through `load_dataset` or the datasets-server API, so any few-shot prompting setup needs to unzip `data.zip` directly from the repo rather than relying on the served split.

## Neighbors

Four re-releases were found by Hub search and confirmed by fetching a row from each; all row counts below are live, unpinned reads [11].

- `nirantk/jeebench` - 1,330 rows in a single `train` split, apache-2.0 licensed, with the same six columns plus an added `solution` field. Its first row is byte-identical to this dataset's first row (`phy`, `JEE Adv 2016 Paper 1`, `index` 1, gold `B`, same question text) but that row's `solution` is `None`, and the split holds far more rows than this benchmark's 515, so it is not a plain re-host - treat it as a separate, larger collection that happens to start with the same rows [12].
- `macabdul9/jeebench_math` - 236 rows, exactly this dataset's `math`-subject count; its first served row (`index` 37, gold `C`, same question text as this dataset's math question 37) confirms it is the math-only subset of this benchmark, with an added `id` column [12].
- `guanning-ai/jeebench-math` - also 236 rows, matching the same math question 37 and gold `C` as `macabdul9/jeebench_math`, but reshaped to `problem`/`answer`/`datasource` columns instead of this dataset's `question`/`gold` [12].
- `daman1209arora/jeebench_numeric` - 137 rows from the same author, exactly this dataset's `Numeric`-type count; its first row (`phy`, answer `2`) is reshaped to `subject`/`problem`/`answer` columns [12].

This original release is the one to prefer for benchmarking, since it is the one the origin paper's own scorer (`compute_metrics.py`) [2] is written against, with all four question types and both grading rules intact; the subject- and type-restricted neighbors drop question types and only cover part of the scoring logic.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [13]:

```json
{
  "subject": "phy",
  "description": "JEE Adv 2016 Paper 1",
  "gold": "B",
  "index": 1,
  "type": "MCQ",
  "question": "In a historical experiment to determine Planck's constant, a metal surface was irradiated with light of different wavelengths. The emitted photoelectron energies were measured by applying a stopping potential. [...] Given that $c=3 \\times 10^{8} \\mathrm{~m} \\mathrm{~s}^{-1}$ and $e=1.6 \\times 10^{-19} \\mathrm{C}$, Planck's constant (in units of $J \\mathrm{~s}$ ) found from such an experiment is\n\n(A) $6.0 \\times 10^{-34}$\n\n(B) $6.4 \\times 10^{-34}$\n\n(C) $6.6 \\times 10^{-34}$\n\n(D) $6.8 \\times 10^{-34}$"
}
```

`gold` holds a bare option letter for `MCQ` rows; a full 515-row read shows it holds a concatenated letter string for `MCQ(multiple)` (e.g. `"ABD"`) and a plain number string for `Integer`/`Numeric` rows (e.g. `"9"`, `"2"`) [8].

## Where it came from

Built and released by the paper's authors, who curated the 515 problems directly from the IIT JEE-Advanced exam, one of the most competitive pre-engineering entrance exams, spanning physics, chemistry and mathematics questions from 2016 through 2023 [1]. Questions and their correct answers are the exam's own human-authored content; the paper's contribution is the curation, the four-way answer-type scoring scheme, and the evaluation of LLMs against it, not the generation of new questions [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Arora, Singh and Mausam, "Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models", EMNLP 2023. https://aclanthology.org/2023.emnlp-main.468/ - the origin paper; no arXiv mirror is linked from this page or the dataset's README. Fetched 2026-08-11.

[2] `compute_metrics.py` in the dataset repository. https://huggingface.co/datasets/daman1209arora/jeebench/raw/main/compute_metrics.py - the scoring rules per question type. Fetched 2026-08-11.

[3] Hugging Face Hub API record for daman1209arora/jeebench. https://huggingface.co/api/datasets/daman1209arora/jeebench?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] daman1209arora/jeebench dataset card (README). https://huggingface.co/datasets/daman1209arora/jeebench/raw/main/README.md - licence tag, pretty name, and the BibTeX-embedded paper abstract containing "proprietary". Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=daman1209arora%2Fjeebench Fetched 2026-08-11.

[6] The corpus screening row for `daman1209arora/jeebench`, supplied with this card's request - its `note`, read back in the appendix. Checked 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=daman1209arora%2Fjeebench Fetched 2026-08-11.

[8] `test.json`, the full 515-row dataset file. https://huggingface.co/datasets/daman1209arora/jeebench/resolve/main/test.json - read in full (not sampled) for subject/type/description counts and `gold` format per type. Fetched 2026-08-11.

[9] Repository file tree. https://huggingface.co/api/datasets/daman1209arora/jeebench/tree/main - file listing and byte sizes, including `data.zip` and `test.json`. Fetched 2026-08-11.

[10] `inference.py` in the dataset repository. https://huggingface.co/datasets/daman1209arora/jeebench/raw/main/inference.py - loads `data/few_shot_examples.json` for few-shot prompting. Fetched 2026-08-11.

[11] Hub dataset search for "jeebench". https://huggingface.co/api/datasets?search=jeebench&limit=20 Fetched 2026-08-11.

[12] datasets-server size and first-rows endpoints, one call per neighbor: `nirantk/jeebench`, `macabdul9/jeebench_math`, `guanning-ai/jeebench-math`, `daman1209arora/jeebench_numeric`. https://datasets-server.huggingface.co/size?dataset=<id> and https://datasets-server.huggingface.co/first-rows?dataset=<id>&config=default&split=<split> - these endpoints take no revision parameter, so these counts and rows are live, not pinned. Fetched 2026-08-11.

[13] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=daman1209arora%2Fjeebench&config=default&split=test Fetched 2026-08-11.

[14] daman1209arora/jeebench dataset landing page. https://huggingface.co/datasets/daman1209arora/jeebench - the item's Hub page. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as an evaluation-only benchmark, never as training data: the dataset ships a single `test` split with no train counterpart [5], and its own scorer defines exact grading for all four question types it contains [2] - both facts already established above settle that the full 515 rows are held-out eval data, matching the screening row's note [6].

### The screening row

The row's own note [6]: "JEEBench, IIT-JEE Advanced physics, chemistry and maths questions, `test` only." The row carries no flag.
