# casehold/casehold

CaseHOLD: 53,137 multiple-choice questions asking which of five holdings a cited U.S. case stands for, in an `all` config plus ten cross-validation folds whose test sets overlap the `all` training split.

**casehold/casehold** is CaseHOLD (Case Holdings On Legal Decisions), "a law dataset comprised of over 53,000+ multiple choice questions to identify the relevant holding of a cited case" [1], introduced in "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset" by Zheng et al. [2]. Each row is a `citing_prompt` - an excerpt of a U.S. opinion with the cited case's holding removed - and five candidate holdings, one correct [1][3]. The repository has no README; a loading script and CSV files are all it holds [4][5]. It lives at https://huggingface.co/datasets/casehold/casehold .

**Use the `all` config and hold out its `test` split. The `fold_*` configs are cross-validation folds: row 0 of `fold_1/test` is row 0 of `all/train`.**

**Use it for**: SFT or evaluation for legal holding identification. As SFT data it teaches reading an opinion excerpt and matching it to a holding parenthetical - a narrow, well-defined skill.

**Licence**: none on the card - no licence field and no README [6][4]. The CaseHOLD paper and its GitHub release are the only terms of reference. The one catch: no licence is stated on this repository.

**Shape**: `all`: `train` 42,509 / `validation` 5,314 / `test` 5,314; ten `fold_N` configs of about 42,560 / 5,260 / 5,314 each [7]. Eight columns [1].

**Hold out**: `all/test` (5,314 rows). The Nemotron legal config named `Nemotron-Pretraining-Legal-Case-Law-Summary` contains these test items [8]; exclude it from a mix you score on CaseHOLD. Exact leakage from `all/test` into `all/train` is 8 rows [9].

**Origin**: excerpts of U.S. court opinions; the correct answer is the cited case's holding, alongside four incorrect holdings [2][1]. Hub API at the check date: `downloads` 2,092, `downloadsAllTime` 35,007, `likes` 26 [6].

**Trained-on-by**: the Hub's dataset tag lists `MMOPD/Qwen3-4B-OT3-law` (432 downloads) [10]. LexGLUE's `case_hold` task is a re-split of the same questions [11].

**Introduced by**: [2] (Zheng et al.).

## Shape

Rows per config and split (datasets-server `/size`) [7]:

| config | split | rows |
| --- | --- | --- |
| `all` | `train` | 42,509 |
| `all` | `validation` | 5,314 |
| `all` | `test` | 5,314 |
| `fold_1` | `train` | 42,562 |
| `fold_1` | `validation` | 5,261 |
| `fold_1` | `test` | 5,314 |
| `fold_10` | `train` | 42,563 |
| `fold_10` | `validation` | 5,261 |
| `fold_10` | `test` | 5,313 |
| `fold_2` | `train` | 42,562 |
| `fold_2` | `validation` | 5,261 |
| `fold_2` | `test` | 5,314 |
| `fold_3` | `train` | 42,562 |
| `fold_3` | `validation` | 5,261 |
| ... 19 more rows | | |
| all 11 configs | `train` 468,132 / `validation` 57,924 / `test` 58,451 | 584,507 |

Columns, shared by all configs [1]:

| column | dtype |
| --- | --- |
| `example_id` | int32 |
| `citing_prompt` | string |
| `holding_0` | string |
| `holding_1` | string |
| `holding_2` | string |
| `holding_3` | string |
| `holding_4` | string |
| `label` | string |

## Quality

- `label` is a string ("0"-"4") in the served schema [1], not an integer.
- The folds overlap `all`: `fold_1/test` row 0 ("Drapeau's cohorts ...") is also `all/train` row 0 [3]. Mixing `all` with any fold duplicates questions across splits.
- About 10% of `all/test` items share at least half their 8-grams with `all/train`, but only 8 match exactly [8][9]; excerpts of the same opinion overlap without being the same question.

## Load it

Use `all`; the script needs remote code, so pin the revision:

```python
import datasets

REV = "acc4532a67e0966e9ed7ec4ca543e8532f983b0c"  # main at the check date
train = datasets.load_dataset("casehold/casehold", "all", revision=REV, split="train", trust_remote_code=True)  # 42,509 rows
test = datasets.load_dataset("casehold/casehold", "all", revision=REV, split="test", trust_remote_code=True)    # 5,314 rows - hold out
```

**Trap**: the script's CSV for validation is `data/all/val.csv` but the split is exposed as `validation` [5][7]; and on `datasets` 4.x, which dropped script loading, read `data/all/*.csv` directly - the parquet conversion the viewer serves is also available through the `/parquet` API.

## Neighbors

- `coastalcph/lex_glue` `case_hold` - the same question pool re-split 45,000 / 3,900 / 3,600 (served); its test set is not `all/test`.
- `nvidia/Nemotron-Pretraining-Legal-v1` - holds all 53,137 questions, reformatted, under a config named `Case-Law-Summary`.

## A row

From `config="all"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3], truncated:

```json
{
  "example_id": 0,
  "citing_prompt": "Drapeau’s cohorts, the cohort would be a “victim” of making the bomb. Further, firebombs are inherently dangerous. There is no peaceful purpose for making a bomb. Felony offenses that involve explosives qualify as “violent crimes” for purposes of enh [...]",
  "holding_0": "holding that possession of a pipe bomb is a crime of violence for purposes of 18 usc  3142f1",
  "holding_1": "holding that bank robbery by force and violence or intimidation under 18 usc  2113a is a crime of violence",
  "holding_2": "holding that sexual assault of a child qualified as crime of violence under 18 usc  16",
  "holding_3": "holding for the purposes of 18 usc  924e that being a felon in possession of a firearm is not a violent felony as defined in 18 usc  924e2b",
  "holding_4": "holding that a court must only look to the statutory definition not the underlying circumstances of the crime to determine whether a given offense is by its nature a crime of violence for purposes of 18 usc  16",
  "label": "0"
}
```

## Where it came from

Built by Lucia Zheng, Neel Guha, Brandon Anderson, Peter Henderson and Daniel Ho at Stanford from the Harvard Caselaw Access Project, by removing holding parentheticals from citing sentences [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=casehold%2Fcasehold - column schema; live, no revision parameter. Fetched 2026-09-23.

[2] Zheng et al., "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset", arXiv:2104.08671, 2021. https://arxiv.org/abs/2104.08671 - current title read from the live abs page. Fetched 2026-09-23.

[3] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=casehold%2Fcasehold&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[4] casehold/casehold dataset card (README). https://huggingface.co/datasets/casehold/casehold/raw/main/README.md. Fetched 2026-09-23.

[5] Repository file tree for casehold/casehold. https://huggingface.co/api/datasets/casehold/casehold/tree/main?recursive=true. Fetched 2026-09-23.

[6] Hugging Face Hub API record for casehold/casehold. https://huggingface.co/api/datasets/casehold/casehold?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=casehold%2Fcasehold - takes no revision parameter; a live figure. Fetched 2026-09-23.

[8] This skill's own check by n-gram containment against the benchmark's test items (word 8-grams, or character 15-grams for Chinese), with self, words-sorted and positive controls, at the pinned revision. Run 2026-09-23.

[9] This skill's own exact split-leakage count at the pinned revision. Fetched 2026-09-23.

[10] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:casehold/casehold&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[11] coastalcph/lex_glue dataset card (README), Data Splits table. https://huggingface.co/datasets/coastalcph/lex_glue/raw/main/README.md. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable: train on `all/train`, hold out `all/test`, and never mix folds with `all`. The whole question pool is inside the Nemotron legal set, so decontaminate against it before reporting CaseHOLD.

### The screening row

The row's own note: "CaseHOLD original; folds overlap `all`; no licence on the repo." The row carries the flag `no-licence`.
