# dim/competition_math

7,500 competition-math problems with LaTeX step-by-step solutions - a single-split Parquet reformatting of the training half of the MATH dataset.

**dim/competition_math** is a user re-upload of the training split of MATH, the competition-mathematics benchmark introduced in "Measuring Mathematical Problem Solving With the MATH Dataset" [1]: 12,500 problems drawn from competitions such as AMC 10, AMC 12 and AIME, each with a full LaTeX solution ending in a boxed final answer [2]. The repository's own dataset card is the stub "More Information needed" and states no author, collection method, or licence [3]; everything about origin and format below is read from the parent MATH repository and the paper. **This repository serves only the MATH train split (7,500 of the 12,500 problems); the MATH test split (5,000 problems) is a standard held-out evaluation set for competition-math benchmarks and is not present here [4].** It lives at https://huggingface.co/datasets/dim/competition_math .

**Use it for**: reasoning-trace SFT - each row pairs a `problem` with a `solution` that derives the boxed answer step by step, the shape a chat-SFT or reasoning-SFT method card consumes directly as prompt/completion. **Hold out MATH test rows if evaluating on MATH**, since this repository is the train half only [4]. Format is plain `problem`/`solution` string pairs, not a chat-templated dialect (`chat_dialect: none` per the shortlist row) - build the assistant turn from `solution` yourself.

**Licence**: MIT. This repository's own card carries no `license` field (`cardData.license` absent, `licence_body_mentions` empty in the corpus screening row) [3], but both the original loading-script repository `hendrycks/competition_math` and the `qwedsacf/competition_math` mirror of the same MATH data declare `license: mit` in their Hub metadata and card frontmatter [5][11], and `qwedsacf`'s card additionally points to the MIT-licensed text at `github.com/hendrycks/math/blob/main/LICENSE` [5]. Ungated, not private (`gated: false`) [9]. The one catch: this repository does not itself state the licence, so a tool that reads only its own card metadata will see no licence field.

**Shape**: one split, `train`, 7,500 rows, one config, four string columns (`problem`, `level`, `type`, `solution`) [6][7].

**Hold out**: nothing within this repository - it is train-only and holds nothing back itself. The risk sits one level up: this repository's rows are a subset of the standard MATH benchmark, so if a downstream evaluation harness scores against MATH test, rows here do not overlap it, but any pipeline that also draws from a full 12,500-row MATH mirror must not mix its test half back in. The corpus screening row probed three MATH test problems against this repository's rows and found none present [8].

**Origin**: re-uploaded to the Hub by user `dim`; the underlying problems and solutions are the original MATH dataset's, authored by its creators, not model-generated [1][3]. Hub API at the check date: `downloads` 574, `downloadsAllTime` 3,861, `likes` 0 [9].

**Trained-on-by**: none found for this specific re-upload; the underlying MATH benchmark it reformats is a standard SFT and evaluation source across the field, but no source consulted here names a model trained specifically on this repository (`dim/competition_math`) rather than another MATH mirror.

**Introduced by**: the dataset is a reformatting of MATH, introduced by [1] (Hendrycks et al.); this repository's own card states no more than "More Information needed" [3].

## Shape

Rows and split (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 7,500 |

One config, `default`, four columns (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `problem` | string |
| `level` | string |
| `type` | string |
| `solution` | string |

No source states sequence-length or token statistics for this repository or for MATH generally; none is invented here.

## Quality

- No source states a measured contamination, duplicate, or error rate for this repository's rows.
- This repository's own card carries no quality statement at all - it is the stub "More Information needed" [3].
- The parent MATH card documents `level` as difficulty "Level 1" (easiest) to "Level 5" (hardest) within a subject, and `type` as one of seven subjects (Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus) [5]. A read of rows at train offset 100-104 (5 rows) shows three at "Level 4" and two at "Level 2", all "Algebra" - consistent with those two fields but too small a sample to characterize the split's overall level/type distribution [10].
- The Hub disabled the original loading-script repository `hendrycks/competition_math` (`"disabled": true` in its API record) [11]; this Parquet re-upload is one of several third-party mirrors that exist because of that removal (see Neighbors).

## Load it

Pin the revision this card's numbers were read at (the shortlist row's own commit, matching the Hub API's current `sha`, last modified 2023-09-25):

```python
import datasets

REV = "1fc5a1a4b382faf8de65d4ed3d35a128008b0b5f"
train = datasets.load_dataset("dim/competition_math", revision=REV, split="train")  # 7,500 rows
```

**Trap**: there is only one split, `train` - there is no `test` split to request, so scripts that assume a MATH-shaped `train`/`test` pair (as the original `hendrycks/competition_math` loader provided) will fail on this repository unless rewritten to a single split [7][11].

## Neighbors

- `hendrycks/competition_math` - the original loading-script repository this reformats, with both `train` (7,500) and `test` (5,000) splits in its `cardData.dataset_info`; the Hub API marks it `"disabled": true`, so it can no longer be loaded [11].
- `qwedsacf/competition_math` - a different third-party mirror that merges MATH train and test into a single 12,500-row `train` split with no split boundary; a row fetched from its `/info` and `/size` endpoints confirms one config, one split, 12,500 rows, same four columns [12]. Because it erases the train/test boundary, mixing it into training risks the exact held-out MATH test rows this repository excludes; prefer `dim/competition_math` when the train/test separation matters.
- `dim/competition_math_selected` - a same-builder repository with the same four columns and `license: mit` declared in its own card metadata, but a different, smaller `train` split: 3,000 rows per its Hub API record [15]. Its row content was not fetched here, so whether it is a subset of this repository's 7,500 rows or a distinct selection is not verified.
- Numerous other third-party re-uploads exist (e.g. `SuperSecureHuman/competition_math_hf_dataset`, `jeggers/competition_math`, `chiayewken/competition_math`) found via Hub search; none were fetched, so no comparison is made beyond noting they exist [13].

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [14]:

```json
{
  "problem": "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).",
  "level": "Level 5",
  "type": "Algebra",
  "solution": "For the piecewise function to be continuous, the cases must \"meet\" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$."
}
```

## Where it came from

The problems and step-by-step LaTeX solutions originate in MATH, built by Hendrycks et al. from mathematics-competition problems including AMC 10, AMC 12 and AIME [1][2], introduced with an auxiliary pretraining corpus meant to teach models mathematical fundamentals and improve MATH accuracy [1]. This Hub repository is a Parquet re-upload of the 7,500-row train half of that dataset under the account `dim`; its own card gives no further detail on who performed the reformatting or when beyond the repository's creation and last-modified timestamps (2023-09-25) [3][9].

## Sources

Every source below was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row in this card. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset", 2021. https://arxiv.org/abs/2103.03874 - the origin paper; current title and abstract read from the live abs page. Fetched 2026-08-11.

[2] `hendrycks/competition_math` dataset card (README, via API `description` field), stating the problem source (AMC 10, AMC 12, AIME) and step-by-step solution format. https://huggingface.co/api/datasets/hendrycks/competition_math?full=true Fetched 2026-08-11.

[3] `dim/competition_math` dataset card (README). https://huggingface.co/datasets/dim/competition_math/raw/main/README.md - the "More Information needed" stub; no licence, author, or collection-method text present. Fetched 2026-08-11.

[4] The corpus screening row for `dim/competition_math`, supplied with this card's request - its `note`, describing this repository as the MATH train half only. Checked 2026-08-11.

[5] `qwedsacf/competition_math` dataset card (README), which carries the full MATH field descriptions (`level`, `type` subjects) and the licence pointer to the parent GitHub repository that this repository's own stub card omits. https://huggingface.co/datasets/qwedsacf/competition_math/raw/main/README.md Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=dim%2Fcompetition_math Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=dim%2Fcompetition_math Fetched 2026-08-11.

[8] The corpus screening row for `dim/competition_math` - its stated probe of three MATH test problems against this repository, finding none present, read back in the row's own words in the appendix. Checked 2026-08-11.

[9] Hugging Face Hub API record for dim/competition_math. https://huggingface.co/api/datasets/dim/competition_math?full=true - `sha`, `downloads`, `likes`, `createdAt`, `lastModified`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[10] datasets-server rows endpoint, offset 100, length 5. https://datasets-server.huggingface.co/rows?dataset=dim%2Fcompetition_math&config=default&split=train&offset=100&length=5 Fetched 2026-08-11.

[11] Hugging Face Hub API record for hendrycks/competition_math. https://huggingface.co/api/datasets/hendrycks/competition_math?full=true - `"disabled": true`, split-level `cardData.dataset_info` showing `train` 7,500 / `test` 5,000. Fetched 2026-08-11.

[12] datasets-server size and info endpoints for qwedsacf/competition_math. https://datasets-server.huggingface.co/size?dataset=qwedsacf%2Fcompetition_math and https://datasets-server.huggingface.co/info?dataset=qwedsacf%2Fcompetition_math - single `train` split, 12,500 rows, one config, no revision parameter taken (live, not pinned). Fetched 2026-08-11.

[13] Hugging Face Hub dataset search for "competition_math". https://huggingface.co/api/datasets?search=competition_math&limit=20 - listing of other third-party re-uploads found by name only; not individually fetched. Fetched 2026-08-11.

[14] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=dim%2Fcompetition_math&config=default&split=train Fetched 2026-08-11.

[15] Hugging Face Hub API record for dim/competition_math_selected. https://huggingface.co/api/datasets/dim/competition_math_selected?full=true - `cardData.license`, `cardData.dataset_info` split (`train`, 3,000 rows), `downloads` 79, `likes` 1. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for reasoning-trace SFT as the MATH train half. The dataset's own card carries no restriction text (it is a stub) [3], and the screening row's probe found no MATH test problems present in this repository, supporting its "train-only" characterization [8].

### The screening row

The row's own note [4][8]: "The MATH TRAIN half only (7,500 rows); three MATH TEST problems probed, none present." The row carries no flag.
