# EleutherAI/hendrycks_math

12,500 competition math problems with human-written step-by-step LaTeX solutions, split into seven subject configs, each with its own `train`/`test` split.

**EleutherAI/hendrycks_math** is a Parquet repackaging of the Mathematics Aptitude Test of Heuristics (MATH) dataset introduced in "Measuring Mathematical Problem Solving With the MATH Dataset" [1], competition problems (AMC 10, AMC 12, AIME, and others) each carrying a full derivation that ends in a boxed final answer, intended to teach models to produce worked solutions rather than bare answers [1]. It lives at https://huggingface.co/datasets/EleutherAI/hendrycks_math .

**The seven `*/test` splits (5,000 rows total) are the original MATH benchmark test set: hold them out of training. Only the seven `*/train` splits (7,500 rows total) are safe for reasoning-trace SFT.**

**Use it for**: reasoning-trace SFT on math word problems - `problem` as the prompt and `solution` as the target completion, mapping to a plain instruction/response SFT format (no chat template is present in the columns). Train only on the `*/train` splits. See the SFT method card.

**Licence**: MIT (`cardData.license` is `"mit"`, and the repo carries the `license:mit` tag), ungated (`"gated": false`, `"private": false`) [2]. The card states only the SPDX id with no further terms; the upstream GitHub repository linked in the card's summary [3] carries the same unrestricted MIT license text [4].

**Shape**: 12,500 rows across 7 configs (one per subject: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus), each with a `train` and a `test` split [5][6].

**Hold out**: all seven `*/test` splits, 5,000 rows total (1,187 + 474 + 479 + 903 + 540 + 871 + 546) - this is the MATH benchmark test set that the origin paper reports scores against [1][5], and it is also the source pool for the 500-problem `MATH-500` eval subset (see Neighbors) [7].

**Origin**: repackaged by EleutherAI from Dan Hendrycks et al.'s original release; problems are drawn from real math competitions and solutions are human-written [1]. Hub API at the check date: `downloads` 172,608, `downloadsAllTime` 3,781,537, `likes` 108 [2].

**Trained-on-by**: MetaMath bootstraps new questions by rewriting problems "from multiple perspectives" starting from the GSM8K and MATH training sets (7,500 MATH training problems, 5,000 MATH test problems, matching this release's counts), then fine-tunes LLaMA-2 models on the resulting MetaMathQA data [8]. No source found stating a model trained directly on this Parquet repackaging (as opposed to the original MATH release) rather than a derivative built from it.

**Introduced by**: [1] (Hendrycks, Burns, Kadavath, Arora, Basart, Tang, Song, Steinhardt).

## Shape

Rows and splits (datasets-server `/size`) [5]:

| config | train | test |
| --- | --- | --- |
| algebra | 1,744 | 1,187 |
| counting_and_probability | 771 | 474 |
| geometry | 870 | 479 |
| intermediate_algebra | 1,295 | 903 |
| number_theory | 869 | 540 |
| prealgebra | 1,205 | 871 |
| precalculus | 746 | 546 |
| total | 7,500 | 5,000 |

All seven configs share the same four string columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `problem` | string |
| `level` | string |
| `type` | string |
| `solution` | string |

The origin paper states MATH totals "12,500 problems (7,500 training and 5,000 test)" [1], matching this release exactly. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- Solutions are human-written, ending in a LaTeX `\boxed{}` final answer, and problems are drawn from real math competitions rather than being model-generated [1].
- `level` records human difficulty ratings "Level 1" through "Level 5" and `type` records the subject, both confirmed by the sampled rows below; the origin paper does not give a per-level or per-type breakdown of counts [1].
- No source states a measured duplicate rate, annotator-agreement figure, or contamination rate for this release; none is invented here. The paper itself does not discuss contamination of the test set against pretraining corpora [1].
- No external fetch is needed to use these rows: `problem` and `solution` are both served inline as full text, unlike datasets that store only references to an external source.

## Load it

Train on the seven `*/train` splits, hold out the seven `*/test` splits, and pin the revision this card's numbers were read at:

```python
import datasets

REV = "21a5633873b6a120296cce3e2df9d5550074f4a3"  # main at the check date

configs = ["algebra", "counting_and_probability", "geometry",
           "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

train = datasets.concatenate_datasets([
    datasets.load_dataset("EleutherAI/hendrycks_math", cfg, revision=REV, split="train")
    for cfg in configs
])  # 7,500 rows total - safe for training
test = datasets.concatenate_datasets([
    datasets.load_dataset("EleutherAI/hendrycks_math", cfg, revision=REV, split="test")
    for cfg in configs
])  # 5,000 rows total - hold out, this is the MATH benchmark
```

**Trap**: there is no single default config - `load_dataset("EleutherAI/hendrycks_math")` with no config name fails, because the dataset is split into seven independent per-subject configs rather than one config with a `type` column filter. A reader who loads only `algebra` and calls it "the training set" is silently training on 1,744 of the 7,500 available training rows.

## Neighbors

- `qwedsacf/competition_math` - the same 12,500 problems, one config, same four columns [9]; its own card describes the same MATH dataset and links the same origin paper and GitHub repository [10]. This corpus prefers the `EleutherAI/hendrycks_math` per-subject Parquet release over this one because it splits by subject config rather than requiring a `type`-column filter.
- `nlile/hendrycks-MATH-benchmark` - a re-split of the same problems into one `train`/`test` pair, but with a different split point: 12,000 train and 500 test, versus 7,500/5,000 here, plus extra `answer` and `unique_id` columns and an integer `level` [11]. Because its test split (500 rows) is smaller than this release's 5,000-row test set, some of its "train" rows are this release's held-out test rows; do not mix the two without checking for that overlap.
- `HuggingFaceH4/MATH-500` - a fixed 500-problem subset of the MATH test set, curated by OpenAI for "Let's Verify Step by Step" [7]; its card states it draws from that paper's GitHub repository rather than restating the full test set [7]. Any of these 500 problems that also sit in this release's `*/test` splits must stay out of training for the same reason the full test split does.

## A row

Both `train` and `test` share the same four-column shape, so one row from each shows both. From `config="algebra"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [12]:

```json
{
  "problem": "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous...",
  "level": "Level 5",
  "type": "Algebra",
  "solution": "For the piecewise function to be continuous, the cases must \"meet\" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$... So $a+b=-3+3=\\boxed{0}$."
}
```

From `config="algebra"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [13]:

```json
{
  "problem": "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?",
  "level": "Level 3",
  "type": "Algebra",
  "solution": "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$... Therefore, the graph has $\\boxed{2}$ vertical asymptotes."
}
```

## Where it came from

The problems and human step-by-step solutions were collected by Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt for the MATH benchmark, drawn from real competitions including the AMC 10, AMC 12, and AIME [1]. The Hub card credits the source as the authors' GitHub repository and cites the same paper [3]; that GitHub repository itself points readers to a different Hub dataset, `qwedsacf/competition_math`, for the MATH data download, and to a separate Google Drive file for the AMPS pretraining data, and it licenses its own code under plain MIT [4]. EleutherAI's copy on the Hub reshapes the release into seven per-subject Parquet configs, each with its original `train`/`test` split [3][5][6].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Hendrycks, Burns, Kadavath, Arora, Basart, Tang, Song, Steinhardt, "Measuring Mathematical Problem Solving With the MATH Dataset", 2021. https://arxiv.org/abs/2103.03874 - problem/solution format, subject list, 12,500/7,500/5,000 counts. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2103.03874). Fetched 2026-08-11.

[2] Hugging Face Hub API record for EleutherAI/hendrycks_math. https://huggingface.co/api/datasets/EleutherAI/hendrycks_math?full=true - license, gate status, `sha`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[3] EleutherAI/hendrycks_math dataset card (README). https://huggingface.co/datasets/EleutherAI/hendrycks_math/raw/main/README.md - dataset summary line, license front matter, citation. Fetched 2026-08-11.

[4] hendrycks/math GitHub repository README and LICENSE. https://raw.githubusercontent.com/hendrycks/math/main/README.md and https://raw.githubusercontent.com/hendrycks/math/main/LICENSE - origin repo's own data-download links (to `qwedsacf/competition_math` on the Hub and to a Google Drive AMPS file), plain MIT license text. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=EleutherAI%2Fhendrycks_math Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=EleutherAI%2Fhendrycks_math Fetched 2026-08-11.

[7] HuggingFaceH4/MATH-500 dataset card (README) and size endpoint. https://huggingface.co/datasets/HuggingFaceH4/MATH-500/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=HuggingFaceH4%2FMATH-500 Fetched 2026-08-11.

[8] Yu et al., "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models", 2023. https://arxiv.org/abs/2309.12284 - GSM8K/MATH training-set counts, bootstrapped MetaMathQA construction, LLaMA-2 fine-tuning. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2309.12284). Fetched 2026-08-11.

[9] datasets-server size endpoint for qwedsacf/competition_math. https://datasets-server.huggingface.co/size?dataset=qwedsacf%2Fcompetition_math Fetched 2026-08-11.

[10] qwedsacf/competition_math dataset card (README). https://huggingface.co/datasets/qwedsacf/competition_math/raw/main/README.md Fetched 2026-08-11.

[11] nlile/hendrycks-MATH-benchmark dataset card (README) and size endpoint. https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=nlile%2Fhendrycks-MATH-benchmark Fetched 2026-08-11.

[12] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=EleutherAI%2Fhendrycks_math&config=algebra&split=train Fetched 2026-08-11.

[13] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=EleutherAI%2Fhendrycks_math&config=algebra&split=test Fetched 2026-08-11.

[14] The corpus screening row for `EleutherAI/hendrycks_math`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for reasoning-trace SFT, restricted to the `*/train` splits only. The card already establishes both facts that decide this - the `*/test` splits are the MATH benchmark's own test set that the origin paper scores against [1], and the screening row's note draws the same line between the 7,500 safe training rows and the 5,000 benchmark test rows [14].

### The screening row

The row's own note [14]: "MATH competition problems with human step-by-step solutions split by subject; the seven `*/train` splits (7,500 total) safe, the `*/test` splits (5,000) are the MATH benchmark." The row carries no flag.
