# simplescaling/aime_nofigures

https://huggingface.co/datasets/simplescaling/aime_nofigures

90 AIME competition math problems (2022-2024) as problem/solution/answer triples, with figure diagrams re-expressed as Asymptote code only where the figure is needed to solve the problem - a benchmark artifact released alongside Stanford's s1 project, not a training set.

**simplescaling/aime_nofigures** was released by the `simplescaling` org (the s1 project, introduced in "s1: Simple test-time scaling" [1]) alongside sibling repos covering the same 90 AIME 2022-2024 problems with and without figure code, and single-year (AIME24-only) versions of each [2][3][4]. Its own card states the scope in one line: "The 90 problems from AIME 2022, 2023, 2024 only with the ASY code for figures when it is necessary to solve the problem. Figure code that is not core to the problem was excluded" [2]. **The dataset is an evaluation set, not training data, and it duplicates 90 rows of AI-MO/aimo-validation-aime almost exactly - same problem ids, urls, and answers on every one of the 30 rows sampled here; of the 16 rows whose `problem` text differs, 14 differ only in minor LaTeX punctuation placement around math delimiters and 2 (id 21, id 29) differ because this repo restores necessary figure code [5][6]. Hold out all 90 rows from any training run, and do not mix this repo with AI-MO/aimo-validation-aime in the same run.**

**Use it for**: scoring a model's competition-math ability, not for training - the rows are raw problem/solution/answer QA triples (`chat_dialect: none`), so map them into whatever prompt template the evaluation harness expects rather than the SFT method card. 30 of these 90 problems, the AIME 2024 set, are the actual "AIME24" benchmark the s1 paper reports and decontaminates its training data against; the paper does not report results on the full 90-problem 2022-2024 span this repo covers [1].

**Licence**: Apache-2.0 (`cardData.license: apache-2.0`, tag `license:apache-2.0`), ungated (`"gated": false`, `"private": false`) [7]. No catch beyond the standard grant is stated in the card [2].

**Shape**: 90 rows, one config (`default`), one split (`train`), five columns [8].

**Hold out**: all 90 rows. This repository is itself an evaluation benchmark, and it duplicates AI-MO/aimo-validation-aime, so training on either one contaminates evaluation on the other [5][6].

**Origin**: released by the `simplescaling` (s1) org; the problems and one forum-sourced solution each come from AIME competitions via the AoPS wiki, i.e. human-authored content, not model-generated [2][6]. Hub API at the check date: 191 downloads, 1 like [7].

**Trained-on-by**: none found. This is an eval artifact, not a training corpus; the s1 paper decontaminates its SFT training data against AIME24 rather than training on it [1].

**Introduced by**: no dedicated paper names this exact 90-problem, figures-partially-restored repo. The card cites "s1: Simple test-time scaling" [1] as its associated work via its bibtex block [2], but that paper's own AIME results are reported only on the 30-problem AIME24 subset, not this 90-problem 2022-2024 collection [1].

## Shape

One split, from the datasets-server size endpoint [8]:

| split | rows |
| --- | --- |
| `train` | 90 |

Columns, from the datasets-server info endpoint [8]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `problem` | string |
| `solution` | string |
| `answer` | string |
| `url` | string |

Sizes [8]: 549,011 bytes of original JSON download, 263,793 bytes as Parquet, 522,180 bytes decoded in memory. No source states sequence-length or token statistics for this repository.

## Quality

- The problems and their forum-derived solutions are human-authored, sourced from the AoPS wiki; the sibling AI-MO/aimo-validation-aime card, which shares the same problem set, describes the `solution` column as "one of the solutions proposed in the forum with \boxed answer" [6].
- No source states a measured error rate, duplicate rate, or annotator-agreement figure for this specific repository; none is invented here.
- Of the 30 rows read at offset 0 (`split=train`, the only rows the datasets-server first-rows endpoint serves before truncating), 2 have `[asy]` figure code in the `problem` field (ids 21, 29) and 4 have it in the `solution` field (ids 8, 13, 17, 21), for 5 distinct rows with figure code somewhere in the row - confirming the card's claim that figure code is kept only where needed rather than on every problem [2][5].

## Load it

Pin the revision this card's numbers were read at (the shortlist's recorded commit, matching the Hub API `sha` at the check date; the repo was last modified 2025-02-03) [2][7]:

```python
import datasets

REV = "e1829989175f0be31baa761134b83cccd542fa73"
ds = datasets.load_dataset("simplescaling/aime_nofigures", revision=REV, split="train")  # 90 rows
```

**Trap**: there is only one split (`train`) and no held-out test split of its own - the entire repository is the thing to hold out of any training run, not something to further split for training. Loading it as if it were a training corpus (e.g. fine-tuning on `problem`/`solution` pairs) trains directly on an evaluation benchmark [1][2].

## Neighbors

All from the same `simplescaling` org, sharing the identical five-column schema [3][4]:

- `simplescaling/aime_figures` - the same 90 problems with all Asymptote figure code kept, none excluded ("The 90 problems from AIME 2022, 2023, 2024 with all ASY code for figures") [3]. This is the figures-complete counterpart to this repo.
- `simplescaling/aime24_nofigures` and `simplescaling/aime24_figures` - the 2024-only 30-problem subset, with and without full figure code respectively [4]. `aime24_nofigures` is the narrower slice that corresponds to the "AIME24" benchmark actually reported in the s1 paper [1].
- `simplescaling/aime25_nofigures` and `simplescaling/aime25_figures` - the same pattern for AIME 2025, not covered by this card's fetches.
- `AI-MO/aimo-validation-aime` - the same 90 problems built independently by AI-MO as an internal AIMO-competition validation set, with figure code stripped from every problem rather than selectively kept [6]. Comparing all 30 rows read at offset 0 from both repositories found identical `id`, `url`, and `answer` values on every row, and identical `problem` text on 14 of 30; of the 16 rows that differ, 14 differ only in minor LaTeX punctuation placement (e.g. row 0: a comma placed inside vs. outside a math delimiter) and 2 (id 21, id 29) differ because this repo restores necessary figure code [5][6]. Prefer this repo (`aime_nofigures`) or `aime_figures` when the eval needs figure code available to the model; prefer `aimo-validation-aime` only if a figure-free baseline is specifically wanted. Do not use both in the same run.

## A row

One shape is served (config `default`, split `train`). Row `id=1`, without a figure, from offset 0 of the datasets-server first-rows endpoint [5]:

```json
{
  "id": 1,
  "problem": "Three spheres with radii $11$, $13$, and $19$ are mutually externally tangent. A plane intersects the spheres in three congruent circles centered at $A$, $B$, and $C$, respectively, and the centers of the spheres all lie on the same side of this plane. Suppose that $AB^2 = 560$. Find $AC^2$.",
  "answer": "756",
  "url": "https://artofproblemsolving.com/wiki/index.php/2022_AIME_I_Problems/Problem_10"
}
```

Row `id=21`, one of the 2 (of 30 read) rows whose `problem` field carries restored Asymptote figure code, illustrating the card's selective-figure-code rule [2][5]:

```json
{
  "id": 21,
  "problem": "Two externally tangent circles $\\omega_1$ and $\\omega_2$ have centers $O_1$ and $O_2$, respectively. A third circle $\\Omega$ passing through $O_1$ and $O_2$ intersects $\\omega_1$ at $B$ and $C$ and $\\omega_2$ at $A$ and $D$, as shown. Suppose that $AB = 2$, $O_1O_2 = 15$, $CD = 16$, and $ABO_1CDO_2$ is a convex hexagon. Find the area of this hexagon. [asy] import geometry; size(10cm); point O1=(0,0),O2=(15,0),B=9*dir(30); circle w1=circle(O1,9),w2=circle(O2,6),o=circle(O1,O2,B); [...]",
  "answer": "140",
  "url": "https://artofproblemsolving.com/wiki/index.php/2022_AIME_II_Problems/Problem_15"
}
```

## Where it came from

Released by the `simplescaling` (s1) org, the group behind "s1: Simple test-time scaling" [1], as one of a set of AIME benchmark repositories (with and without figures, spanning 2022-2025) built for evaluating reasoning models [2][3][4]. The problems and solutions are drawn from AIME 2022-2024 competitions via the AoPS wiki; the sibling AI-MO card, which covers the same problem set, states the problems were "extracted directly from the AOPS wiki page" and the solutions are forum-proposed [6]. This repository's own contribution is selective figure handling: full AoPS diagrams are re-expressed as Asymptote source, and that source is kept in the `problem`/`solution` text only for the subset of problems where the figure is load-bearing for the solution, per the card's own description [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Muennighoff et al., "s1: Simple test-time scaling", 2025. https://arxiv.org/abs/2501.19393 - current title read from the live abs page; AIME24 as the reported benchmark, decontamination against AIME24, and the note that AIME problems rely on figures provided as Asymptote code, read from the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2501.19393). Fetched 2026-08-12.

[2] simplescaling/aime_nofigures - the item's own Hub page, https://huggingface.co/datasets/simplescaling/aime_nofigures, which renders the same card fetched directly as README: https://huggingface.co/datasets/simplescaling/aime_nofigures/raw/main/README.md - scope description, license, citation block, last-modified date corroborated against the Hub API. Fetched 2026-08-12.

[3] simplescaling/aime_figures dataset card (README). https://huggingface.co/datasets/simplescaling/aime_figures/raw/main/README.md - the figures-complete sibling's own description. Fetched 2026-08-12.

[4] simplescaling/aime24_nofigures dataset card (README), and the sibling dataset listing for the `simplescaling` org (https://huggingface.co/api/datasets?author=simplescaling&limit=100) used to enumerate all aime24/aime25 figures/nofigures repos. https://huggingface.co/datasets/simplescaling/aime24_nofigures/raw/main/README.md Fetched 2026-08-12.

[5] datasets-server first-rows endpoint for this repository. https://datasets-server.huggingface.co/first-rows?dataset=simplescaling%2Faime_nofigures&config=default&split=train - serves 30 of 90 rows (truncated) at offset 0; used for the sampled row comparison, the figure-code count, and the two rows quoted above. Fetched 2026-08-12.

[6] AI-MO/aimo-validation-aime dataset card (README) and its datasets-server first-rows endpoint (https://datasets-server.huggingface.co/first-rows?dataset=AI-MO%2Faimo-validation-aime&config=default&split=train, same 30-row-at-offset-0 sample). https://huggingface.co/datasets/AI-MO/aimo-validation-aime/raw/main/README.md - column descriptions and problem/answer/solution values used for the duplicate comparison. Fetched 2026-08-12.

[7] Hugging Face Hub API record for simplescaling/aime_nofigures. https://huggingface.co/api/datasets/simplescaling/aime_nofigures?full=true - license, gate status, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-12.

[8] datasets-server size and info endpoints for this repository. https://datasets-server.huggingface.co/size?dataset=simplescaling%2Faime_nofigures and https://datasets-server.huggingface.co/info?dataset=simplescaling%2Faime_nofigures Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Flagged as contamination risk rather than usable training data: the corpus's own screening row marks this repository `use: eval`, and its `flag_class` is `contamination`, matching what this card independently confirms above - the repository is an evaluation benchmark that duplicates AI-MO/aimo-validation-aime almost row-for-row [5][6].

### The screening row

The row's own note: "The same 90 problems as aimo-validation-aime with figures stripped; s1's AIME eval." Its flag: "contamination: duplicate of AI-MO/aimo-validation-aime with figures removed."
