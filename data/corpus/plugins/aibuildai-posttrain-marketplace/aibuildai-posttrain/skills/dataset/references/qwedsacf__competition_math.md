# qwedsacf/competition_math

12,500 competition math problems with step-by-step LaTeX solutions, served as a single `train` split - but that split silently concatenates the original MATH benchmark's train and test problems.

**qwedsacf/competition_math** is a Hugging Face re-upload of the Mathematics Aptitude Test of Heuristics (MATH) dataset introduced in "Measuring Mathematical Problem Solving With the MATH Dataset" [1], a collection of competition mathematics problems (AMC 10, AMC 12, AIME, and others) each paired with a full step-by-step derivation, useful for training models to generate mathematical explanations [2]. It lives at https://huggingface.co/datasets/qwedsacf/competition_math . **Reading the dataset's own served parquet file shows the 12,500-row `train` split is not a fresh sample: rows 0-7,499 match the original MATH training split's per-subject counts exactly, and rows 7,500-12,499 match the original MATH test split's per-subject counts exactly, with row 7,500 an exact match - problem, level, type and solution text - for row 0 of the standard MATH algebra test set [7][8][9]. Do not train on this repository's `train` split and then evaluate on any standard MATH test benchmark: the test problems are inside it.**

**Use it for**: reasoning-trace SFT on competition math (problem plus a `\boxed`-terminated step-by-step solution) only after removing the appended test block; maps to plain prompt/completion SFT format once `problem` and `solution` are paired - see the SFT method card. Do not use the served `train` split as-is for any run later evaluated against MATH.

**Licence**: MIT (`cardData.license` is `"mit"`, repo tag `license:mit`), ungated (`"gated": false`, `"private": false`) [3]. The catch: the card's own "Licensing Information" section states only a link to an external LICENSE file on GitHub and never restates "MIT" as text in the body [2].

**Shape**: 12,500 rows, one config (`default`), one split (`train`), four string columns (`problem`, `level`, `type`, `solution`) [4][5].

**Hold out**: rows 7,500-12,499 of the served `train` split - these are the original MATH test problems, appended after the 7,500 original training problems [7][8]. There is no separate held-out split to point to instead; a reader who wants a clean train/test division should use rows 0-7,499 as train and either discard rows 7,500-12,499 or treat them as the held-out set, never both together.

**Origin**: re-uploaded by Hub user `qwedsacf`; the underlying problems and solutions are unchanged from the original MATH release, itself expert-authored (competition problems and worked solutions), with no re-annotation stated for this upload [2]. Hub API at the check date: `downloads` 17,386, `downloadsAllTime` 121,694, `likes` 138 [3].

**Trained-on-by**: none found for this specific re-upload; no source cites `qwedsacf/competition_math` by name as training data.

**Introduced by**: [1] (Hendrycks et al.); this repository carries no dataset card explanation of its own train/test merge - the merge is established here by reading the served rows [2][7].

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 12,500 |

One config, `default`, four columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `problem` | string |
| `level` | string |
| `type` | string |
| `solution` | string |

Sizes (datasets-server `/size`) [4]: 4,848,345 bytes of original/Parquet download, 6,927,300 bytes decoded in memory. No source states sequence-length or token statistics for this repository.

## Quality

- Reading the full served parquet directly [7] shows the 7,500-row block at indices 0-7,499 has per-`type` counts of Algebra 1,744, Intermediate Algebra 1,295, Prealgebra 1,205, Geometry 870, Number Theory 869, Counting & Probability 771, Precalculus 746 (sum 7,500); the 5,000-row block at indices 7,500-12,499 has Algebra 1,187, Intermediate Algebra 903, Prealgebra 871, Precalculus 546, Number Theory 540, Geometry 479, Counting & Probability 474 (sum 5,000). These exact counts match EleutherAI/hendrycks_math's per-subject `train` and `test` splits respectively (datasets-server `/size` for that repository) [8], confirming the two blocks are the original MATH train and test splits concatenated, not a new sample.
- Row 7,500 (the first row of the second block, and its first `Algebra`-type row) reads "How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?", Level 3, with a solution ending "$\boxed{2}$" [7]; this is an exact match - problem, level, type and solution text - for row 0 of `EleutherAI/hendrycks_math`'s `algebra`/`test` config, fetched live from its `/first-rows` endpoint [9]. That confirms the appended block is the canonical MATH test set, not a re-shuffled or re-sampled copy.
- The card documents no annotation process, no measured error or duplicate rate, and no mention of the train/test merge; its only stated content note is the field descriptions (`level` ranges "Level 1" to "Level 5"; `type` is one of Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus) [2].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-01-28) [3]:

```python
import datasets

REV = "e839825f9ec5c6cfa585c654a59610969ec13993"  # main at the check date
ds = datasets.load_dataset("qwedsacf/competition_math", revision=REV, split="train")  # 12,500 rows
train_only = ds.select(range(7500))   # matches original MATH train, verified above
test_block = ds.select(range(7500, 12500))  # matches original MATH test, verified above - hold out
```

**Trap**: `split="train"` returns all 12,500 rows with no internal marker distinguishing the two blocks - there is no `split` or `source` column to filter on. The boundary is the row index itself: rows 0-7,499 and rows 7,500-12,499, established only by comparing counts and content against `EleutherAI/hendrycks_math` as done above [7][8][9]. Training on the full 12,500 rows and then evaluating against the standard MATH test set (or any benchmark built from it) evaluates on training data.

## Neighbors

- `hendrycks/competition_math` - the original MATH release from the paper's authors, correctly split into `train` (7,500 rows) and `test` (5,000 rows); its Hub API record marks the repository `"disabled": true` at the check date, so it cannot currently be loaded from the Hub [10]. Prefer this repository's row content conceptually (it is what this corpus's `train`/`test` blocks were copied from), but it is not currently loadable.
- `EleutherAI/hendrycks_math` - the same 12,500 problems reorganized into seven per-subject configs (`algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus`), each with its own correctly separated `train`/`test` split; row counts per config were read live from its `/size` endpoint and used above to confirm this repository's contamination [8]. This corpus prefers `EleutherAI/hendrycks_math` over `qwedsacf/competition_math` for any run that needs the train/test boundary intact, since its splits are labelled correctly and its counts match the original MATH release exactly.

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [6]:

```json
{
  "problem": "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).",
  "level": "Level 5",
  "type": "Algebra",
  "solution": "For the piecewise function to be continuous, the cases must \"meet\" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$."
}
```

## Where it came from

Re-uploaded to the Hub by user `qwedsacf`; the card states no collection method of its own beyond restating the original MATH paper's summary and citation [2]. The underlying problems and solutions originate from the MATH benchmark introduced in "Measuring Mathematical Problem Solving With the MATH Dataset" [1], which states the full dataset comprises 12,500 problems, split by the original authors into 7,500 training and 5,000 test problems [1]. This repository's `train` split reproduces both of those original blocks concatenated together, as established under Quality above [7][8][9].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset", 2021. https://arxiv.org/abs/2103.03874 - the origin paper; current title read from the live abs page, and the 12,500/7,500/5,000 split figures read from the paper's HTML full text (ar5iv rendering, https://ar5iv.labs.arxiv.org/html/2103.03874). Fetched 2026-08-11.

[2] qwedsacf/competition_math dataset card (README). https://huggingface.co/datasets/qwedsacf/competition_math/raw/main/README.md - dataset summary, field descriptions, licensing information section, citation. Fetched 2026-08-11.

[3] Hugging Face Hub API record for qwedsacf/competition_math. https://huggingface.co/api/datasets/qwedsacf/competition_math?full=true - licence, gate status, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size endpoint, pinned to the revision above. https://datasets-server.huggingface.co/size?dataset=qwedsacf%2Fcompetition_math&revision=e839825f9ec5c6cfa585c654a59610969ec13993 - row counts and byte sizes reported in Shape. Fetched 2026-08-11.

[5] datasets-server info endpoint, pinned to the revision above. https://datasets-server.huggingface.co/info?dataset=qwedsacf%2Fcompetition_math&revision=e839825f9ec5c6cfa585c654a59610969ec13993 - column names and dtypes. Fetched 2026-08-11.

[6] datasets-server first-rows endpoint, pinned to the revision above. https://datasets-server.huggingface.co/first-rows?dataset=qwedsacf%2Fcompetition_math&config=default&split=train&revision=e839825f9ec5c6cfa585c654a59610969ec13993 - the sampled row-0 content in A row. Fetched 2026-08-11.

[7] The dataset's own served Parquet file, downloaded at the pinned revision. https://huggingface.co/datasets/qwedsacf/competition_math/resolve/e839825f9ec5c6cfa585c654a59610969ec13993/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet - all 12,500 rows read to compute per-block `type` counts and to locate and read the exact text of row index 7,500; byte-identical (same md5) to the unpinned `main` copy fetched the same day, confirmed by downloading both. Fetched 2026-08-11.

[8] datasets-server size endpoint for EleutherAI/hendrycks_math. https://datasets-server.huggingface.co/size?dataset=EleutherAI%2Fhendrycks_math - per-subject-config `train`/`test` row counts, used to confirm the contamination above. This endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[9] datasets-server first-rows endpoint for EleutherAI/hendrycks_math. https://datasets-server.huggingface.co/first-rows?dataset=EleutherAI%2Fhendrycks_math&config=algebra&split=test - row 0, used for the exact-match comparison against this repository's row 7,500. Fetched 2026-08-11.

[10] Hugging Face Hub API record for hendrycks/competition_math. https://huggingface.co/api/datasets/hendrycks/competition_math?full=true - `train`/`test` split sizes (7,500/5,000) and `"disabled": true` status. Fetched 2026-08-11.

[11] The corpus screening row for `qwedsacf/competition_math`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Contaminated: this repository's `train` split is the original MATH train and test splits merged into one split labelled `train`, so training on it and evaluating on the standard MATH test benchmark trains on the evaluation data. The facts establishing this are set out above (rows 7,500-12,499 match the original MATH test split's per-subject counts and content exactly) [7][8][9]; the screening row's flag names the same issue and additionally notes this repository duplicates `EleutherAI/hendrycks_math` [11].

### The screening row

The row's own note [11]: "Re-upload of MATH with train and test MERGED into one 12,500-row split labelled `train`; the first MATH algebra TEST problem confirmed inside it." Its flag [11]: "contamination: MATH train and test merged into one split named train (the first MATH algebra test problem found inside); also a duplicate of EleutherAI/hendrycks_math."
