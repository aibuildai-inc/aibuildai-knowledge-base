# di-zhang-fdu/AIME_1983_2024

933 American Invitational Mathematics Examination (AIME) problems spanning 1983 to 2024 (2024's second sitting only), each a free-response question paired with its numeric answer, in a single CSV/train split - an evaluation benchmark, not a training corpus.

**di-zhang-fdu/AIME_1983_2024** is a compilation, by the Hugging Face account di-zhang-fdu, of AIME contest problems and answers drawn from the Art of Problem Solving wiki, covering every AIME year from 1983 through 2023 plus the second 2024 sitting; the card points readers wanting the first 2024 sitting to a separate dataset, AI-MO/aimo-validation-aime [1]. The card carries no origin paper of its own - its citation block instead lists two unrelated papers by the same author, Di Zhang, that report AIME accuracy as one of several evaluation benchmarks for their own reasoning methods [1][2][3]. **The card states outright that this is a benchmark and must not be used for training** [1], and a byte-for-byte comparison of the underlying CSV file confirms this exact 933-row file is also served, unchanged, by gneubig/aime-1983-2024 and philschmid/AIME_1983_2024 [4][5][6] - so treat it as a widely-mirrored eval set that decontamination must check against, not merely a single repo to exclude. It lives at https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024 .

**Use it for**: measuring AIME-style math-reasoning accuracy (question in, exact numeric answer expected) in an eval harness - never for training, per the card's own disclaimer [1]. The rows are a flat question/answer table (`Question`, `Answer` columns plus `ID`, `Year`, `Problem Number`, `Part`), not a chat or preference format; an eval harness reads `Question` as the prompt and checks the model's final answer against `Answer`. No training method card applies - this dataset feeds evaluation, not a training run.

**Licence**: MIT (`license: mit` in the card's front matter, and the `license:mit` tag) [1][7]. The one catch: the MIT tag covers this compiled CSV, but the card's own disclaimer bars training use outright - a restriction the licence grant itself does not carry and does not override [1].

**Shape**: 933 rows, one config (`default`), one split (`train`), six columns [8][9].

**Hold out**: all 933 rows, for any training run - the card's disclaimer forbids training on this benchmark [1], and the same 933-row file, unchanged, also ships as gneubig/aime-1983-2024 and philschmid/AIME_1983_2024 [4][5][6], so excluding this one repo id is not enough. The years 2022-2024 here also overlap AI-MO/aimo-validation-aime's 90 validation problems (AIME 22-24) [10], a widely used AIME eval set in its own right.

**Origin**: compiled by the Hugging Face account di-zhang-fdu (repository created 2024-05-20) [7]; every row is a human-authored contest problem and its human-published answer, with no generating model involved [1]. Hub API as of the check date: 7,759 downloads, 41 likes [7].

**Trained-on-by**: none found. As a benchmark the card explicitly bars training use [1], and the two papers by the same author that cite AIME evaluate their MCTS-based reasoning methods against it rather than training on it [2][3].

**Introduced by**: no paper - the dataset card [1]. The two arXiv papers in the card's citation block (LLaMA-Berry [2] and the MCT Self-Refine paper [3]) are the uploader's own prior work; both use "AIME" as one of several evaluation benchmarks for their reasoning methods and neither claims to have compiled this dataset.

## Shape

Rows and splits (datasets-server `/size`) [8]:

| split | rows |
| --- | --- |
| `train` | 933 |

One config, `default`, with six columns (datasets-server `/info`) [9]:

| column | dtype |
| --- | --- |
| `ID` | string |
| `Year` | int64 |
| `Problem Number` | int64 |
| `Question` | string |
| `Answer` | string |
| `Part` | string |

Reading the served CSV directly [11]: rows run from `Year` 1983 through 2024. 1983-1999 carry only one sitting per year, 9-15 rows each (`Part` is blank for all of them); 2000-2023 each carry two sittings, `Part` "I" and "II", 26-30 rows per year; 2024 has only the 14 rows tagged `Part` "II" - consistent with the card's statement that only the second 2024 sitting is included [1][11]. No row IDs repeat. Sizes (datasets-server `/size`) [8]: 342,215 bytes of original CSV, 179,316 bytes as Parquet, 358,605 bytes decoded in memory. No source states token or sequence-length statistics for this dataset; none is invented here.

## Quality

- Every answer is the contest's own published numeric answer, and every problem is the contest's own published problem text; no model or automated pipeline generated any field [1].
- No source states a measured error rate, duplicate rate, or annotation-agreement figure for this file; none is invented here.
- The card's only quality-relevant statement is the disclaimer that this is a benchmark and must not be used for training [1] - a usage caveat, not a data-quality measurement.
- The three repositories serving this exact file disagree on stated provenance: this card cites the Art of Problem Solving wiki directly [1], while gneubig/aime-1983-2024's card instead names a Kaggle compilation by Hemish Veeraboina as its source, itself ultimately drawn from the same AoPS wiki [12]. The underlying CSV bytes are identical across all three repos regardless of which provenance story each card tells [4][5][6].

## Load it

The whole file is a `train` split of 933 rows that must be held out of any training run [1]. Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2025-03-03) [7]:

```python
import datasets

REV = "3e2cc86390666c5c756622afc0eeb9e6194496bc"  # main at the check date
aime = datasets.load_dataset("di-zhang-fdu/AIME_1983_2024", revision=REV, split="train")  # 933 rows, eval only
```

**Trap**: nothing in the loaded object marks these rows as an eval set - `load_dataset` will happily hand back a normal `train` split, and a pipeline that globs every "train" split across a corpus of Hub datasets will pull this one straight into a training mix unless it is explicitly excluded by dataset id (and by the ids of its identical mirrors below) [1][4][5][6].

## Neighbors

The exact same 933-row CSV - confirmed by an identical Hub git blob hash for `AIME_Dataset_1983_2024.csv` (`d290c4f0...6111a4`) across all three repos, read live at the check date - is also served as [4][5][6]:

- `gneubig/aime-1983-2024` - byte-identical CSV, same 933 rows and six columns, created 2024-12-21 (later than this repo's 2024-05-20); its own card names a Kaggle source rather than this repo, and adds a `cc0-1.0` licence tag in place of this repo's `mit` [4][12].
- `philschmid/AIME_1983_2024` - byte-identical CSV and an identical `.gitattributes` file to this repo, created 2025-01-21 (later still); its README repeats this repo's exact disclaimer text word for word, including "Do not using in training!" [5].

Neither mirror adds rows or changes the schema, so there is no basis here to prefer one over another for evaluation - but training on more than one of the three does not add data, only redundant risk of leaking this exact benchmark into a training mix.

A related but distinct dataset: `AI-MO/aimo-validation-aime` holds 90 rows covering AIME 2022-2024 only, extracted from the same AoPS wiki but built independently as a validation set for the AIMO competition, with `problem`/`solution`/`answer`/`url` columns rather than this repo's `Question`/`Answer`/`Part` layout [10]. Its own card states it deliberately restricts to post-2021 problems "to avoid potential overlap with the MATH training set" [10] - the same contamination concern this card's Hold out line raises for AIME rows generally.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [13]:

```json
{
  "ID": "1983-1",
  "Year": 1983,
  "Problem Number": 1,
  "Question": "Let $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_xw=24$ , $\\log_y w = 40$ and $\\log_{xyz}w=12$ . Find $\\log_zw$ .",
  "Answer": "60",
  "Part": null
}
```

`Part` is blank for every row from 1983-1999 (single-sitting years); rows from 2000-2023 carry `Part` "I" or "II" for the two sittings held those years; the 14 rows dated 2024 carry only `Part` "II" [11]. `Question` holds LaTeX-formatted problem text; `Answer` is always the contest's plain numeric answer as a string.

## Where it came from

Compiled by the Hugging Face account di-zhang-fdu, whose card states the problems come from the American Invitational Mathematics Examination for 1983 through 2023 plus the second 2024 sitting, drawn from the Art of Problem Solving wiki's AIME problems-and-solutions pages, and that the first 2024 sitting is instead served by a separate dataset, AI-MO/aimo-validation-aime [1]. No generating model is involved anywhere in the pipeline; both the problems and their answers are the contest's own published material [1]. The card's citation block lists two arXiv papers by the same author, Di Zhang - "LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning" [2] and "Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B" [3] - both of which evaluate their own MCTS-based reasoning methods against AIME among several other benchmarks, rather than describing how this dataset itself was compiled [2][3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] di-zhang-fdu/AIME_1983_2024 dataset card (README). https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/raw/main/README.md - disclaimer, year/part coverage, licence, citation block, pointer to AI-MO/aimo-validation-aime. Fetched 2026-08-11.

[2] Zhang et al., "LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning", 2024. https://arxiv.org/abs/2410.02884 - evaluates AIME24 among several benchmarks; current title read from the live abs page. Fetched 2026-08-11.

[3] Zhang et al., "Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B", 2024. https://arxiv.org/abs/2406.07394 - evaluates AIME among several benchmarks; current title read from the live abs page. Fetched 2026-08-11.

[4] Hugging Face tree API and datasets-server size endpoint for gneubig/aime-1983-2024. https://huggingface.co/api/datasets/gneubig/aime-1983-2024/tree/main and https://datasets-server.huggingface.co/size?dataset=gneubig%2Faime-1983-2024 - git blob hash of `AIME_Dataset_1983_2024.csv`, row/byte counts, `createdAt`. Fetched 2026-08-11.

[5] Hugging Face tree API and README for philschmid/AIME_1983_2024. https://huggingface.co/api/datasets/philschmid/AIME_1983_2024/tree/main and https://huggingface.co/datasets/philschmid/AIME_1983_2024/raw/main/README.md - git blob hash of `AIME_Dataset_1983_2024.csv`, identical `.gitattributes` hash, identical disclaimer text, `createdAt`. Fetched 2026-08-11.

[6] Hugging Face tree API for di-zhang-fdu/AIME_1983_2024. https://huggingface.co/api/datasets/di-zhang-fdu/AIME_1983_2024/tree/main - git blob hash of `AIME_Dataset_1983_2024.csv`, used as the comparison basis for [4] and [5]. Fetched 2026-08-11.

[7] Hugging Face Hub API record for di-zhang-fdu/AIME_1983_2024. https://huggingface.co/api/datasets/di-zhang-fdu/AIME_1983_2024?full=true - licence, gate status, `sha`, `downloads`, `likes`, `createdAt`, `lastModified`. Fetched 2026-08-11.

[8] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=di-zhang-fdu%2FAIME_1983_2024 Fetched 2026-08-11.

[9] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=di-zhang-fdu%2FAIME_1983_2024 Fetched 2026-08-11.

[10] AI-MO/aimo-validation-aime dataset card and datasets-server size endpoint. https://huggingface.co/datasets/AI-MO/aimo-validation-aime/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=AI-MO%2Faimo-validation-aime - 90-row AIME 22-24 validation set, post-2021 restriction rationale. Fetched 2026-08-11.

[11] The served CSV file itself. https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/resolve/main/AIME_Dataset_1983_2024.csv - read in full (933 rows) to get the per-year row counts, `Part` values, and duplicate-ID check reported above. Fetched 2026-08-11.

[12] gneubig/aime-1983-2024 dataset card (README). https://huggingface.co/datasets/gneubig/aime-1983-2024/raw/main/README.md - states its source as a Kaggle compilation by Hemish Veeraboina, its own licence (`cc0-1.0`), and `createdAt`. Fetched 2026-08-11.

[13] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=di-zhang-fdu%2FAIME_1983_2024&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Not usable for training: this is an evaluation benchmark whose own card forbids training use, and it is one of at least three Hub repositories serving the identical 933-row file, so decontamination against it means checking all three ids, not just this one. Both facts are established above from the card's disclaimer and the git-blob-hash comparison [1][4][5][6].

### The screening row

The row's own note: "Row-identical copy of gneubig's 933 rows (the 2024-05-20 original; the philschmid re-upload folds into it)." Its flag: "contamination: row-identical duplicate of gneubig/aime-1983-2024; the philschmid re-upload's card says do not train on it."
