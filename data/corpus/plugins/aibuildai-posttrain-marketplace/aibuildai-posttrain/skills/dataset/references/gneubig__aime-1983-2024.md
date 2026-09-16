# gneubig/aime-1983-2024

933 American Invitational Mathematics Examination (AIME) problems spanning contest years 1983 to 2024, each row a problem statement paired with its numeric answer, mirrored to Hugging Face from a Kaggle compilation.

**gneubig/aime-1983-2024** packages AIME problems and answers compiled by Hemish Veeraboina and published on Kaggle as "AIME Problem Set 1983-2024" [1], which this Hugging Face repository mirrors without a separate introducing paper - the dataset card is the only source of provenance [1]. It lives at https://huggingface.co/datasets/gneubig/aime-1983-2024 . Each row is one contest problem: a `Question` string, its `Answer` string, the `Year`, the `Problem Number` within that year's contest, an `ID`, and (from 2000 onward) a `Part` marking AIME I or AIME II [2]. **The compilation runs through 2024 and includes AIME 2024 II problems 1-8 and 10-15 [2] - a live benchmark year that overlaps eval sets many current models were trained or tested against, so this dataset is contamination-risk unless the reader excludes those rows before using it for training or scoring recent-model results.**

**Use it for**: an eval-only math benchmark (question -> short numeric answer), not an SFT source in its own right - the card states no chat template and no reasoning traces, only the bare problem and answer [2]. **Do not train or score on the 2024 rows without checking them against any model's training or eval cutoff**, since AIME 2024 is a current benchmark year [2]. The raw `Question`/`Answer` pair maps to a plain QA/short-answer format; see the SFT or eval-harness method card for how to wrap it into a prompt template.

**Licence**: CC0-1.0 (`license: cc0-1.0` in the card's YAML, tag `license:cc0-1.0`), ungated, public [3]. One catch: the card's own body only speaks of "research purposes" without repeating a licence grant in prose, so the CC0 declaration in the metadata is the only explicit licence statement [1][3].

**Shape**: 933 rows, one config (`default`), one split (`train`); six columns [4][5].

**Hold out**: the 2024 rows - the served AIME 2024 II rows (14 of them, problem 9 missing) [11] - before using this dataset to train or evaluate any model whose training or eval cutoff could include AIME 2024. AIME 2024 I is entirely absent from this repository (see Shape/Where it came from), so that part is not a holdout concern here, but a reader assembling other AIME-2024 sources should not assume this repo is a complete substitute. The compilation is also incomplete elsewhere - see Quality for the full year/part gap count - but none of those other gaps carries a contamination risk of its own.

**Origin**: compiled by Hemish Veeraboina from public AIME problem archives and mirrored to the Hub by user gneubig; no model or human labelling step beyond compilation is stated [1]. Hub API at check date: `downloads` 4,940, `downloadsAllTime` 48,554, `likes` 21 [3].

**Trained-on-by**: none found - no source read for this card names a model or training recipe that used this specific repository.

**Introduced by**: no paper - the dataset card [1], attributing the underlying compilation to Hemish Veeraboina's Kaggle dataset [1].

## Shape

Rows and split, from the datasets-server size endpoint [4]:

| split | rows |
| --- | --- |
| `train` | 933 |
| total | 933 |

One config, `default`, six columns, from the datasets-server info endpoint [5]:

| column | dtype |
| --- | --- |
| `ID` | string |
| `Year` | int64 |
| `Problem Number` | int64 |
| `Question` | string |
| `Answer` | string |
| `Part` | string (null before 2000) |

Sizes from the datasets-server size endpoint [4]: 342,215 bytes original CSV, 179,316 bytes as Parquet, 358,605 bytes decoded in memory. No source states sequence-length or token statistics for this dataset.

The dataset card's prose states "Total Problems: 2,250" [1], but the repository serves 933 rows [4] - a mismatch between the card's stated count and what is actually loadable; this card reports only the served 933.

## Quality

- No source states a measured contamination, duplicate, or answer-error rate for this dataset.
- Reading rows at offset 250-269 (20 rows) shows the `Part` column is null for every row through 2000 and switches to "I"/"II" starting with the 2001-I contest in the served ordering, matching the AIME I/AIME II split that began in 2000 [6].
- Reading the full CSV at the pinned revision (933 rows) and grouping by `Year`+`Part` shows the compilation is missing individual problems throughout, not only at the 2024 tail: 31 of the 66 year/part groups are short of the 15 problems a full AIME sitting has, for a total of 57 missing problem-rows against the 990 a complete 1983-2024 set would hold. The gaps range from a single missing problem (e.g. 1984, 1986, 1987, 1993, 1995, 1996, 1997, most of 2002-2013, 2023-I, and 2024-II) up to the largest gaps, six missing problems each, in 1989 and 1994; 1988 is short five [11]. No 2024-I row (`2024-I-*`) appears anywhere in the 933 served rows [11].
- The card gives no annotation or verification process beyond naming the Kaggle compiler; no known-complaint list is stated [1].

## Load it

```python
import datasets

REV = "5d610df981dec508dd93d0a16333a029ac1739d8"  # main at the check date
ds = datasets.load_dataset("gneubig/aime-1983-2024", revision=REV, split="train")  # 933 rows
```

**Trap**: the single `train` split mixes every contest year 1983-2024 with no year-based split, so a reader must filter on `Year` (and `Part`) themselves to exclude 2024 or any other contamination-sensitive year before use - `load_dataset` alone does not separate them.

## Neighbors

- `di-zhang-fdu/AIME_1983_2024` - byte-identical to this repository: the datasets-server size endpoint reports the same 933 rows, six columns, and the same 342,215/179,316/358,605-byte original/Parquet/memory sizes [7]. Its own card states it is "the Benchmark of AIME from year 1983~2023, and 2024(part 2)" and warns "Do not using in training!", and it names `AI-MO/aimo-validation-aime` as the source for "2024(part 1)" [8] - independently confirming this card's finding that AIME 2024 I is absent from the shared row set. It carries an MIT licence tag, unlike this repository's CC0 declaration [8][3]; the two disagree on licence for what appears to be the same underlying data.
- `AI-MO/aimo-validation-aime` - a separate, smaller (90-row) benchmark covering AIME 2022-2024 problems only, with `problem`/`solution`/`answer`/`url` columns and an Apache-2.0 licence; its card states it was built "to avoid potential overlap with the MATH training set" by using post-2021 AIME years [9]. This is the repository di-zhang-fdu points to for the AIME 2024 I problems missing from both that repo and this one.
- `math-ai/aime25` - AIME 2025 only (30 rows, `test` split, `problem`/`answer`/`id` columns, Apache-2.0) [10]; a newer contest year not covered by this repository at all, useful as a less contamination-exposed AIME eval once 2024 becomes stale.
- None of these three is a strict superset of this repository's 1983-2024 span with AIME 2024 I filled in; a reader who needs the complete 2024 contest must combine this repository (or `di-zhang-fdu/AIME_1983_2024`) with `AI-MO/aimo-validation-aime`'s 2024-I rows and de-duplicate on `Year`+`Problem Number`.

## A row

The repository serves one config and one split, but `Part` differs in kind between pre-2000 (null) and 2000-onward (I/II) contests, so one row of each shape:

Pre-2000 shape, `config="default"`, `split="train"`, `row_idx=0` (datasets-server first-rows) [6]:

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

Post-2000 shape, `config="default"`, `split="train"`, `row_idx=903` (datasets-server rows endpoint, offset 903) [6]:

```json
{
  "ID": "2023-I-14",
  "Year": 2023,
  "Problem Number": 14,
  "Question": "The following analog clock has two hands that can move independently of each other. [asy] ... [/asy] Initially, both hands point to the number $12$ . The clock performs a sequence of hand movements so that on each movement, one of the two hands moves clockwise to the next number on the clock face while the other hand does not move. Let $N$ be the number of sequences of $144$ hand movements such that during the sequence, every possible positioning of the hands appears exactly once, and at the end of the $144$ movements, the hands have returned to their initial position. Find the remainder when $N$ is divided by $1000$ .",
  "Answer": "608",
  "Part": "I"
}
```

## Where it came from

The dataset card names Hemish Veeraboina's Kaggle release "AIME Problem Set 1983-2024" as the source and says the compilation draws on "publicly available AIME problems and their solutions" [1]. No generating model or human-labelling pipeline is described beyond that compilation step. `Part` is null throughout 1983-2000 (AIME had one contest per year through 2000) and becomes "I"/"II" from 2001 onward in the served ordering, matching AIME's move to two annual sittings [6]. The full pinned-revision CSV shows the compilation is incomplete throughout, not just at the edges: 31 of 66 year/part sittings are short at least one problem, 57 problem-rows short of a complete 1983-2024 set overall, and the final contest year, 2024, is present only as AIME II, itself missing problem 9, with no AIME 2024 I rows anywhere in the 933-row set [11].

## Sources

Every source below was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row in this card. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] gneubig/aime-1983-2024 dataset card (README). https://huggingface.co/datasets/gneubig/aime-1983-2024/raw/main/README.md - source attribution, licence prose, stated total-problems count, considerations for use. Fetched 2026-08-11.

[2] datasets-server info and rows endpoints for gneubig/aime-1983-2024, read for column names/types and the 2024-tail row content. https://datasets-server.huggingface.co/info?dataset=gneubig%2Faime-1983-2024 and https://datasets-server.huggingface.co/rows?dataset=gneubig%2Faime-1983-2024&config=default&split=train&offset=903&length=30 Fetched 2026-08-11.

[3] Hugging Face Hub API record for gneubig/aime-1983-2024. https://huggingface.co/api/datasets/gneubig/aime-1983-2024?full=true and the `expand[]=downloadsAllTime` variant - licence tag, gate/private status, `downloads`, `likes`, `downloadsAllTime`. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=gneubig%2Faime-1983-2024 Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=gneubig%2Faime-1983-2024 Fetched 2026-08-11.

[6] datasets-server first-rows and rows endpoints, multiple offsets, read for the pre/post-2000 `Part` transition and the full 2024 tail. https://datasets-server.huggingface.co/first-rows?dataset=gneubig%2Faime-1983-2024&config=default&split=train ; https://datasets-server.huggingface.co/rows?dataset=gneubig%2Faime-1983-2024&config=default&split=train&offset=250&length=20 ; https://datasets-server.huggingface.co/rows?dataset=gneubig%2Faime-1983-2024&config=default&split=train&offset=903&length=30 Fetched 2026-08-11.

[7] datasets-server size endpoint for di-zhang-fdu/AIME_1983_2024. https://datasets-server.huggingface.co/size?dataset=di-zhang-fdu%2FAIME_1983_2024 Fetched 2026-08-11.

[8] di-zhang-fdu/AIME_1983_2024 dataset card (README). https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/raw/main/README.md - benchmark disclaimer, year/part coverage statement, pointer to AI-MO/aimo-validation-aime, MIT licence tag. Fetched 2026-08-11.

[9] AI-MO/aimo-validation-aime dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/AI-MO/aimo-validation-aime/raw/main/README.md ; https://datasets-server.huggingface.co/size?dataset=AI-MO%2Faimo-validation-aime - row count, columns, licence, stated purpose. Fetched 2026-08-11.

[10] math-ai/aime25 dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/math-ai/aime25/raw/main/README.md ; https://datasets-server.huggingface.co/size?dataset=math-ai%2Faime25 - row count, split name, columns, licence. Fetched 2026-08-11.

[11] The full CSV file at the pinned revision. https://huggingface.co/datasets/gneubig/aime-1983-2024/resolve/5d610df981dec508dd93d0a16333a029ac1739d8/AIME_Dataset_1983_2024.csv - all 933 rows, read directly and grouped by `Year`+`Part` to find every missing problem number, including the 2024-II gap and the 2023-I gap. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Flagged for contamination and usable only with that risk managed: the served rows include the AIME 2024 II contest (14 of its 15 problems), confirmed by reading the full pinned-revision CSV above [11], and no source states a training or eval cutoff for the reader's own model that would clear this concern automatically - the reader must exclude or check the 2024 rows themselves before training or scoring on this dataset.

### The screening row

The row's own note: "Kaggle AIME archive, 933 rows, tail confirmed to include 2024." Its flag: "contamination: carries AIME 2024."
