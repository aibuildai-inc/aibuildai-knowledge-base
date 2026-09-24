# coastalcph/lex_glue

LexGLUE: seven English legal classification and multiple-choice tasks - ECtHR A and B, SCOTUS, EUR-LEX, LEDGAR, UNFAIR-ToS and CaseHOLD - with train, validation and test splits; the standard legal NLU benchmark, whose test splits must be held out.

**coastalcph/lex_glue** is LexGLUE, "a benchmark dataset to evaluate the performance of NLP methods in legal tasks", introduced by Chalkidis et al. [1][2]. It gathers seven existing legal datasets "selected using criteria largely from SuperGLUE": European Court of Human Rights cases (two tasks), U.S. Supreme Court opinions, EU legislation, contract provisions from SEC filings (LEDGAR), terms-of-service sentences (UNFAIR-ToS) and CaseHOLD [2]. It lives at https://huggingface.co/datasets/coastalcph/lex_glue .

**Hold out all seven `test` splits (23,607 rows). Two LexGLUE tasks have copies elsewhere in this skill: its CaseHOLD test is entirely inside the NVIDIA Nemotron legal set, and its UNFAIR-ToS sentences come from the CLAUDETTE corpus that Pile of Law also carries.**

**Use it for**: task-specific SFT on the `train` splits (188,532 rows) for classification-style legal skills - article violation prediction, issue-area classification, provision typing, unfair-clause detection, holding selection - and evaluation on `test`. The tasks are classification, so turn labels into text targets with a template.

**Licence**: CC BY 4.0 in the card metadata [3]; the card's Licensing Information reads "[More Information Needed]" [2]. The one catch: each task inherits its source dataset's terms, which the card does not list.

**Shape**: 236,714 rows over seven configs: `train` 188,532 / `validation` 24,575 / `test` 23,607 [4]. Each config has its own columns [5].

**Hold out**: every `test` split. Measured [6]: all 3,600 `case_hold` test items are inside the Nemotron config named `Nemotron-Pretraining-Legal-Case-Law-Summary`; about 10% of `case_hold` test items share half their 8-grams with `case_hold` train (a property of CaseHOLD's opinion excerpts, not an exact leak - exact matches are 8 of 3,600).

**Origin**: human-written legal documents with labels from the original datasets' annotators or databases [2]. Hub API at the check date: `downloads` 23,558, `downloadsAllTime` 867,972, `likes` 83 [3].

**Trained-on-by**: the Hub's dataset tag lists many task-specific classifiers, for example `sidharthjatt/reasonable-doubt-deberta-ledgar` (75 downloads) and `Agreemind/lexglue-legalbert-small-unfair-tos` (43) [7]. LawInstruct copies its training splits [8].

**Introduced by**: [1] (Chalkidis et al.).

## Shape

Rows per config and split (datasets-server `/size`) [4]:

| config | split | rows |
| --- | --- | --- |
| `case_hold` | `train` | 45,000 |
| `case_hold` | `test` | 3,600 |
| `case_hold` | `validation` | 3,900 |
| `ecthr_a` | `train` | 9,000 |
| `ecthr_a` | `test` | 1,000 |
| `ecthr_a` | `validation` | 1,000 |
| `ecthr_b` | `train` | 9,000 |
| `ecthr_b` | `test` | 1,000 |
| `ecthr_b` | `validation` | 1,000 |
| `eurlex` | `train` | 55,000 |
| `eurlex` | `test` | 5,000 |
| `eurlex` | `validation` | 5,000 |
| `ledgar` | `train` | 60,000 |
| `ledgar` | `test` | 10,000 |
| ... 7 more rows | | |
| all 7 configs | `train` 188,532 / `test` 23,607 / `validation` 24,575 | 236,714 |

Columns of `case_hold` [5]:

| column | dtype |
| --- | --- |
| `context` | string |
| `endings` | list<string> |
| `label` | class_label[0, 1, 2, 3, 4] |

Other configs differ: `ecthr_a` and `ecthr_b` have `text` (a list of fact paragraphs) and `labels` (a list of article ids); `unfair_tos` has `text` and a `labels` list; `scotus`, `eurlex` and `ledgar` have `text` with one or more labels [5].

## Quality

- The card's own split table gives CaseHOLD 45,000 / 3,900 / 3,900, but the served `case_hold` `test` split has 3,600 rows [2][4]. The other six tasks match the table.
- Most of the card's standard sections, including source-data collection, annotation and licensing, read "[More Information Needed]" [2].
- Measured exact leakage inside `case_hold`: 8 of 3,600 `test` contexts also appear in `train` [9].

## Load it

Train on `train`, select on `validation`, hold out `test`. Configs are loaded by name:

```python
import datasets

REV = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"  # main at the check date
train = datasets.load_dataset("coastalcph/lex_glue", "unfair_tos", revision=REV, split="train")   # 5,532 rows
test = datasets.load_dataset("coastalcph/lex_glue", "unfair_tos", revision=REV, split="test")     # 1,607 rows - hold out
```

**Trap**: `unfair_tos` `labels` is a list and is empty for fair sentences [5]; a template that takes `labels[0]` crashes on most rows. Map an empty list to an explicit "none" target.

## Neighbors

- `casehold/casehold` - the original CaseHOLD release, with ten cross-validation folds and different split sizes.
- `lighteval/lexglue` - a repackaging found by the Hub search [10]; not screened here.
- `AdaptLLM/law-tasks` - an evaluation set whose CaseHOLD and UNFAIR-ToS items are built from LexGLUE test data.

## A row

From `config="case_hold"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [11], truncated:

```json
{
  "context": "Drapeau’s cohorts, the cohort would be a “victim” of making the bomb. Further, firebombs are inherently dangerous. There is no peaceful purpose for making a bomb. Felony offenses that involve explosives qualify as “violent crimes” for purposes of enhancing the sentences of career offenders. See 18 U [...]",
  "endings": [
    "holding that possession of a pipe bomb is a crime of violence for purposes of 18 usc  3142f1",
    "holding that bank robbery by force and violence or intimidation under 18 usc  2113a is a crime of violence",
    "holding that sexual assault of a child qualified as crime of violence under 18 usc  16",
    "holding for the purposes of 18 usc  924e that being a felon in possession of a firearm is not a violent felony as defined in 18 usc  924e2b",
    "holding that a court must only look to the statutory definition not the underlying circumstances of the crime to determine whether a given offense is by its nature a crime of violence for purposes of 18 usc  16"
  ],
  "label": 0
}
```

## Where it came from

Assembled by Ilias Chalkidis and co-authors from seven published datasets; the benchmark code and leaderboard are at https://github.com/coastalcph/lex-glue [2][1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Chalkidis et al., "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English", arXiv:2110.00976, 2021. https://arxiv.org/abs/2110.00976 - current title read from the live abs page. Fetched 2026-09-23.

[2] coastalcph/lex_glue dataset card (README). https://huggingface.co/datasets/coastalcph/lex_glue/raw/main/README.md. Fetched 2026-09-23.

[3] Hugging Face Hub API record for coastalcph/lex_glue. https://huggingface.co/api/datasets/coastalcph/lex_glue?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=coastalcph%2Flex_glue - takes no revision parameter; a live figure. Fetched 2026-09-23.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=coastalcph%2Flex_glue - column schema; live, no revision parameter. Fetched 2026-09-23.

[6] This skill's own measurement, `references/contamination.md`, row "LexGLUE" - n-gram containment with its three controls; method and script in that file. Run 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:coastalcph/lex_glue&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] lawinstruct/lawinstruct file list in its README - `LexGLUE-<task>-train-0.jsonl.xz` for all seven tasks. https://huggingface.co/datasets/lawinstruct/lawinstruct/raw/main/README.md. Fetched 2026-09-23.

[9] This skill's exact split-leakage measurement, `references/contamination.md` (split leakage table). Fetched 2026-09-23.

[10] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=lexglue&sort=downloads. Fetched 2026-09-23.

[11] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=coastalcph%2Flex_glue&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable: train on `train`, hold out `test`. The `case_hold` test split is fully contained in the Nemotron legal set and the UNFAIR-ToS source is in Pile of Law, so decontaminate any mix that includes those before reporting LexGLUE.

### The screening row

The row's own note: "the LexGLUE benchmark; train safe, test is the reported number." The row carries no flag.
