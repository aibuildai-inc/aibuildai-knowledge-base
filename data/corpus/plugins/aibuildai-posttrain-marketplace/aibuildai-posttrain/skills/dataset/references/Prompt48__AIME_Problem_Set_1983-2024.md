# Prompt48/AIME_Problem_Set_1983-2024

919 AIME competition problems spanning 1983-2024, each paired with a worked-solution write-up in the `Solution` column. It lives at https://huggingface.co/datasets/Prompt48/AIME_Problem_Set_1983-2024 [1].

**Prompt48/AIME_Problem_Set_1983-2024** is a Hugging Face dataset of American Invitational Mathematics Examination (AIME) problems and their worked solutions, one row per problem, covering contest years 1983 through 2024 [1]. The repository's README is YAML metadata only - a license field and a `dataset_info`/`configs` block - with no prose describing the builder, collection method, or upstream source, so those are not stated by the dataset's own card [1]. **The final 30 rows are all from the 2024 AIME (both AIME I and AIME II), and 2024 AIME problems are a live eval benchmark elsewhere on the Hub (e.g. `Maxwell-Jia/AIME_2024`, `AI-MO/aimo-validation-aime`), so any run that scores on those benchmarks must exclude this dataset's 2024 rows first** [2][3].

**Use it for**: reasoning-trace SFT - each row's `Question` plus `Solution` is a single-turn problem/worked-solution pair, usable to build prompt-completion training examples after excluding the 2024 rows named below; maps to the SFT method card. There is no chat template or role field, so a builder must format `Question` as the prompt and `Solution` (containing the boxed final answer) as the target completion itself [1].

**Licence**: CC0-1.0, ungated (`"gated": false`, `"private": false`) [4]. The one catch: CC0 is a public-domain dedication with no field-of-use restriction, so the contamination hold-out above is a data-hygiene requirement from this card's screening, not a licence term [4].

**Shape**: 919 rows in one config (`default`), one split (`train`), six columns [1][5].

**Hold out**: the 30 rows at `Year == 2024` (row offsets 889-918 of `train`, split evenly between `Type == "AIME_I"` and `Type == "AIME_II"`), because they duplicate a standard held-out eval set and carry full solutions [2]. Nothing else in the shipped columns is flagged.

**Origin**: builder and collection method not stated in the dataset's own card; the repository is a parquet-only Hub dataset with no linked paper [1]. Hub API at the check date: `downloads` 1,652, `downloadsAllTime` 4,917, `likes` 0 [4].

**Trained-on-by**: none found. The Hub API's model-search-by-dataset filter did not return any model whose tags name this repository, and no other source found during this check cites it [6].

**Introduced by**: no paper - the dataset card [1].

## Shape

Splits and columns (datasets-server `/size` and `/info`) [5][7]:

| split | rows |
| --- | --- |
| `train` | 919 |

| column | dtype |
| --- | --- |
| `Year` | int64 |
| `Type` | string |
| `Problem` | string |
| `Question` | string |
| `Solution` | string |
| `__index_level_0__` | int64 |

`Problem` holds only a label like `"Problem 1"`; the actual problem text is in `Question` [8]. `Type` holds `"AIME"` for the earlier years and splits into `"AIME_I"` / `"AIME_II"` once the contest itself split into two sittings - both values are present in a row sample read at offset 400 (year 2005, `Type` = `"AIME_I"`) and at offset 889-918 (year 2024) [9]. No source states token or sequence-length statistics for this release; not stated.

## Quality

- No source states a measured contamination rate, deduplication rate, or annotator-agreement figure for this release.
- A row sample read at offsets 889-918 (the last 30 of 919 rows) confirms all 30 are `Year == 2024`, 15 `Type == "AIME_I"` and 15 `Type == "AIME_II"`, each with a non-empty `Solution` field. Of those 30, 28 end in a boxed or `\framebox`-wrapped numeric answer; the remaining 2 (`row_idx` 890 and 916) cut off mid-derivation before stating a final answer [2].
- No separate `Answer` column exists; the final numeric answer sits embedded inside the free-text `Solution` field (as `\boxed{...}` or `\framebox{...}`), so extracting a bare answer requires parsing that field [8][2].
- The dataset's own card states no known complaints or caveats, because it carries no prose body [1].

## Load it

```python
import datasets

REV = "31e03f0fca043e2226f0084eef898356b73e4909"  # main at the check date
ds = datasets.load_dataset("Prompt48/AIME_Problem_Set_1983-2024", revision=REV, split="train")  # 919 rows
ds = ds.filter(lambda r: r["Year"] != 2024)  # drop the 30 held-out 2024 rows
```

**Trap**: there is only one split (`train`); nothing in the repository separates the 2024 rows out for you, so the filter above must be applied by hand before any SFT run [1][2].

**Pin coverage**: the revision above pins the parquet file and the repository's `README.md`/YAML metadata (Shape, Licence) [1][4]. The row counts, dtypes, and sampled row contents reported throughout this card - Shape's split/column table [5][7], the offset-889 and offset-400 row samples [2][9], and the offset-0 row shown under A row [8] - were read through the `datasets-server` `/size`, `/info`, `/rows`, and `/first-rows` endpoints, none of which accept a revision parameter; those endpoints serve whatever `datasets-server` has currently indexed for `main` and are not guaranteed to match the pinned commit above if the repository is later force-pushed. They matched the pinned parquet's declared 919 rows and six columns at the check date, but a reader who re-runs this card after a force-push should re-check them rather than assume the pin covers them.

## Neighbors

Several other Hub repositories carry AIME problem archives with the same 1983-2024 coverage; row counts and schemas below were all read live at the check date [10]:

- `di-zhang-fdu/AIME_1983_2024` - 933 rows, columns `ID`/`Year`/`Problem Number`/`Question`/`Answer`/`Part`, MIT-licensed. Its README states it is sourced from the AoPS wiki and explicitly warns "Disclaimer: This is a Benchmark dataset! Do not using in training!" [10][11].
- `gneubig/aime-1983-2024` - served rows carry the identical schema and row 0 content as `di-zhang-fdu/AIME_1983_2024` (933 rows, same `ID`, `Question`, and `Answer` values for `1983-1`), even though its README prose describes a different, four-column CSV structure that does not match what is actually served; its README also names the Kaggle dataset "AIME Problem Set 1983-2024" by Hemish Veeraboina as its source [10][12].
- `Maxwell-Jia/AIME_2024` - 30 rows, 2024 only, columns `ID`/`Problem`/`Solution`/`Answer`; this is the standard 2024-only eval set that this card's contamination hold-out is protecting against [3][10].
- `AI-MO/aimo-validation-aime` - 90 rows, another 2024-focused validation set; `di-zhang-fdu`'s README points readers to it for the remaining part of the 2024 contest [10][11].

This dataset's own `Question`/`Solution` pairing (full worked solutions, not just bare answers) is closer in shape to `di-zhang-fdu`/`gneubig` than to the answer-only `Maxwell-Jia` set, but none of these repositories state a shared build lineage with this one, so treat the overlap as topical (same public AIME problems), not a confirmed shared source [1][10].

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` [8]:

```json
{
  "Year": 1983,
  "Type": "AIME",
  "Problem": "Problem 1",
  "Question": "Let $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_x w = 24$ , $\\log_y w = 40$ and $\\log_{xyz} w = 12$ . Find $\\log_z w$ .",
  "Solution": "Thenotation doesn't tell us much, so we'll first convert everything to the equivalent exponential forms.\n $x^{24}=w$ , $y^{40}=w$ , and $(xyz)^{12}=w$ . [...] $\\log_zw=\\boxed{060}$ .",
  "__index_level_0__": 0
}
```

A 2024 row, from `row_idx=889` (offset 889 of `train`), shown to illustrate the flagged tail and its embedded answer format [2]:

```json
{
  "Year": 2024,
  "Type": "AIME_I",
  "Problem": "Problem 1",
  "Question": "Every morning Aya goes for a $9$ -kilometer-long walk and stops at a coffee shop afterwards. [...] Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.",
  "Solution": "$\\frac{9}{s} + t = 4$ in hours and $\\frac{9}{s+2} + t = 2.4$ in hours. [...] Lastly, $s + \\frac{1}{2} = 3$ kilometers per hour, so\n $\\frac{9}{3} + 0.4 = 3.4$ hours, or $\\framebox{204}$ minutes\n-Failure.net",
  "__index_level_0__": 1199
}
```

## Where it came from

Not stated. The repository's README carries only YAML front matter (license and schema declaration) with no prose describing who compiled the problems, which upstream corpus they were pulled from, or how the solutions were written or sourced; the `card_body_bytes` on this repo is 504, the size of the YAML block alone [1]. The `__index_level_0__` column (a pandas artifact left over from a DataFrame export) and the `Problem`/`Question`/`Solution` naming are consistent with, but not proof of, a scrape of a public AIME archive such as the Art of Problem Solving wiki that other Hub repositories of the same 1983-2024 span cite explicitly [10][11].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable, which is why Load it pins the revision.

[1] Prompt48/AIME_Problem_Set_1983-2024 dataset card (README). https://huggingface.co/datasets/Prompt48/AIME_Problem_Set_1983-2024/raw/main/README.md - full content is YAML front matter only, no prose body. Fetched 2026-08-11.

[2] datasets-server rows endpoint, offset 889, length 30. https://datasets-server.huggingface.co/rows?dataset=Prompt48%2FAIME_Problem_Set_1983-2024&config=default&split=train&offset=889&length=30 - confirms the final 30 rows are all `Year == 2024`, split 15/15 between `AIME_I` and `AIME_II`. Fetched 2026-08-11.

[3] Maxwell-Jia/AIME_2024 dataset card (README) and datasets-server size endpoint. https://huggingface.co/datasets/Maxwell-Jia/AIME_2024/raw/main/README.md ; https://datasets-server.huggingface.co/size?dataset=Maxwell-Jia%2FAIME_2024 - 30-row 2024-only AIME eval set used for LLM math-reasoning evaluation. Fetched 2026-08-11.

[4] Hugging Face Hub API record for Prompt48/AIME_Problem_Set_1983-2024. https://huggingface.co/api/datasets/Prompt48/AIME_Problem_Set_1983-2024?full=true and the `expand[]=downloadsAllTime` variant - license, gate, sha, downloads, likes. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Prompt48%2FAIME_Problem_Set_1983-2024 Fetched 2026-08-11.

[6] Hugging Face Hub API models-by-dataset query. https://huggingface.co/api/models?dataset=Prompt48/AIME_Problem_Set_1983-2024&limit=20 - returned unfiltered trending models, not results scoped to this dataset, so no model was confirmed to train on it through this endpoint. Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Prompt48%2FAIME_Problem_Set_1983-2024 Fetched 2026-08-11.

[8] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Prompt48%2FAIME_Problem_Set_1983-2024&config=default&split=train Fetched 2026-08-11.

[9] datasets-server rows endpoint, offset 400, length 5. https://datasets-server.huggingface.co/rows?dataset=Prompt48%2FAIME_Problem_Set_1983-2024&config=default&split=train&offset=400&length=5 - confirms `Type` value `"AIME_I"` appears by 2005. Fetched 2026-08-11.

[10] datasets-server size and info endpoints, one call per neighbor: `di-zhang-fdu/AIME_1983_2024`, `gneubig/aime-1983-2024`, `Maxwell-Jia/AIME_2024`, `AI-MO/aimo-validation-aime`, plus a first-row read (offset 0, length 1) for `di-zhang-fdu/AIME_1983_2024` and `gneubig/aime-1983-2024`. https://datasets-server.huggingface.co/size?dataset=<id> ; https://datasets-server.huggingface.co/info?dataset=<id> ; https://datasets-server.huggingface.co/rows?dataset=<id>&config=default&split=train&offset=0&length=1 - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[11] di-zhang-fdu/AIME_1983_2024 dataset card (README). https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/raw/main/README.md - AoPS wiki source, benchmark-only disclaimer, pointer to AI-MO/aimo-validation-aime for the remaining 2024 problems. Fetched 2026-08-11.

[12] gneubig/aime-1983-2024 dataset card (README). https://huggingface.co/datasets/gneubig/aime-1983-2024/raw/main/README.md - states a four-column CSV structure and names the Kaggle dataset by Hemish Veeraboina as source; the served rows do not match the README's stated columns. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for reasoning-trace SFT only after the 2024 rows are excluded. This card's own row read confirms the screening flag: the last 30 of 919 rows are all `Year == 2024`, 28 of them carrying a `Solution` that ends in a boxed answer, and 2024 AIME problems are served elsewhere on the Hub as a dedicated eval set - so training on the unfiltered dataset risks contaminating any AIME-2024 evaluation [2][3].

### The screening row

The row's own note: "AIME archive with solutions, 919 rows, 30 of them from 2024." Its flag: "contamination: 30 AIME-2024 rows, with solutions".
