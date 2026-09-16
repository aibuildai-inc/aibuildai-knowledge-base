# deepmind/aqua_rat

https://huggingface.co/datasets/deepmind/aqua_rat

97,975 crowd-written algebra word problems, each with five multiple-choice options, a step-by-step natural-language rationale, and the correct option letter, served in two parallel configs of the same rows.

**deepmind/aqua_rat** (pretty name "Algebra Question Answering with Rationales", AQuA-RAT) is DeepMind's release of the dataset behind "Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems" [1], built to train a model that generates both a program and a natural-language rationale explaining how it solves an algebraic word problem [2]. Each row is a question, five lettered options, a rationale, and the correct letter [2]. **The 254-row `test` split is the AQuA benchmark itself: it is the fixed evaluation set the paper's leaderboard entry and downstream papers report scores against, so it must be held out of training, not just the paired `validation` split.**

**Use it for**: reasoning-trace SFT - each row already pairs a question with a worked rationale ending in the answer letter, so it maps to a single-turn instruction/response format (prompt = question + options, response = rationale + final answer) rather than to preference pairs or raw pretraining text. See the SFT method card. Hold out `test` (254 rows) before any training or scored run.

**Licence**: Apache-2.0, ungated [3]. The one catch: the shortlist's own scan flagged an "MIT" mention in the card body, but the only text matched is the substring "mit" inside the word "limitations" in the README's boilerplate section headers [2] - the README's actual licence section quotes only the Apache License, Version 2.0, attributed to "Copyright 2017 Google Inc." [2].

**Shape**: two configs, `raw` and `tokenized`, each with the same three splits and row counts - `train` 97,467, `validation` 254, `test` 254 (97,975 rows per config, 195,950 rows served in total) [4][5].

**Hold out**: `test` (254 rows), because it is the AQuA benchmark's own evaluation set [2]; `validation` (254 rows) is a second, separate split and is not itself the benchmark test set, but decontaminate against it too if it is ever used for anything beyond validation.

**Origin**: built by DeepMind; questions, options and rationales are crowdsourced and expert-generated human writing, not model-generated [2]. Hub API at the check date: 29,235 recent downloads, 278,273 all-time downloads, 72 likes [3][6].

**Trained-on-by**: TIGER-Lab's MathInstruct, the training set behind the MAmmoTH model family, lists AQuA-RAT (under Apache 2.0) among its constituent source datasets [7]; MAmmoTH-7B/13B/34B/70B are trained on MathInstruct [7][8]. No source found stating that a specific model trained on this Hub repository's rows directly (as opposed to through MathInstruct's repackaging).

**Introduced by**: [1] (Ling et al.).

## Shape

Rows and byte sizes are identical across the two configs (datasets-server `/size`, `/info`) [4][5]:

| config | split | rows | bytes (parquet) | bytes (in memory) |
| --- | --- | --- | --- | --- |
| `raw` | train | 97,467 | 25,418,552 | 42,831,775 |
| `raw` | validation | 254 | 76,084 | 118,616 |
| `raw` | test | 254 | 74,040 | 116,759 |
| `tokenized` | train | 97,467 | 26,275,499 | 47,011,550 |
| `tokenized` | validation | 254 | 78,185 | 128,853 |
| `tokenized` | test | 254 | 76,189 | 126,263 |

Both configs share the same four columns and dtypes: `question` (string), `options` (list of string, length 5 in the sampled rows), `rationale` (string), `correct` (string) [5]. The two configs differ in content, not schema: `tokenized` is the same text pre-split into whitespace-separated tokens (for example `"43 - km"` and `"P '"` where `raw` has `"43-km"` and `"P'"`), read directly from a sampled row of each [9][10]. No source states sequence-length or token-count statistics for either config; none is invented here.

## Quality

- The rationale is a human-written, natural-language explanation ending in the correct letter, not a model generation [2].
- The paper this dataset comes from describes the questions as algebraic word problems solved via program generation with a natural-language rationale, but no source states a measured error rate, duplicate rate, or annotator-agreement figure for the released rows; none is invented here [1][2].
- The README's own metadata fields for curation rationale, annotation process, and annotator identity are all marked "[Needs More Information]" - the card does not state them [2].

## Load it

Both configs load directly from the served parquet files; the `raw` config is the one built by `configs.default: true` in the card's metadata [2]. Pin the revision this card's numbers were read at:

```python
import datasets

REV = "33301c6a050c96af81f63cad5562cb5363e88971"  # main at the check date
train = datasets.load_dataset("deepmind/aqua_rat", "raw", revision=REV, split="train")       # 97,467 rows
test = datasets.load_dataset("deepmind/aqua_rat", "raw", revision=REV, split="test")         # 254 rows - the AQuA benchmark, hold out
```

**Trap**: `raw` and `tokenized` are not interchangeable text - `tokenized` has already been split on punctuation and case-normalized spacing (e.g. `"P ' s"` instead of `"P's"`), so feeding it to a subword tokenizer built for natural text reproduces that whitespace-tokenization artifact in every training example rather than the original prose in `raw` [9][10].

## Neighbors

- `TIGER-Lab/MathInstruct` compiles AQuA-RAT together with 12 other math rationale datasets into a unified instruction-tuning format for the MAmmoTH models; the mix's own metadata declares it MIT-licensed as a whole, even though its per-source table lists AQuA-RAT's own contribution as Apache 2.0 [7]. It is a downstream mix, not a copy of this repository's rows or schema.
- The original GitHub release at https://github.com/deepmind/AQuA is named as the dataset's homepage and repository in this card [2]; it was not separately fetched here.

## A row

Both configs serve four columns with identical types but different text content, so one row from each config shows the actual difference. From `config="raw"`, `split="train"`, `row_idx=0` [9]:

```json
{
  "question": "Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time. If Friend P's rate is 15% faster than Friend Q's, how many kilometers will Friend P have walked when they pass each other?",
  "options": ["A)21", "B)21.5", "C)22", "D)22.5", "E)23"],
  "rationale": "If Q complete x kilometers, then P completes 1.15x kilometers.\nx + 1.15x = 43\n2.15x=43\nx = 43/2.15 = 20\nThen P will have have walked 1.15*20=23 km.\nThe answer is E.",
  "correct": "E"
}
```

From `config="tokenized"`, `split="train"`, `row_idx=0`, the same underlying problem [10]:

```json
{
  "question": "Two friends plan to walk along a 43 - km trail , starting at opposite ends of the trail at the same time . If Friend P ' s rate is 15 % faster than Friend Q ' s , how many kilometers will Friend P have walked when they pass each other ?",
  "options": ["A ) 21", "B ) 21.5", "C ) 22", "D ) 22.5", "E ) 23"],
  "rationale": "If Q complete x kilometers , then P completes 1.15 x kilometers .\nx + 1.15 x = 43\n2.15 x = 43\nx = 43 / 2.15 = 20\nThen P will have have walked 1.15 * 20 = 23 km .\nThe answer is E .",
  "correct": "E"
}
```

## Where it came from

Built and released by DeepMind, and introduced by "Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems" [1]. The README states the questions, options, and rationales are crowdsourced and expert-generated human writing, in English [2], and gives no further detail on the collection or annotation process beyond that classification [2].

## Sources

Checked 2026-08-11. Hugging Face Hub repositories are mutable, which is why Load it pins the revision (`sha`); the datasets-server endpoints below take no revision parameter and are read live at the check date, so any figure from them describes the repository as it stood on that date, not as pinned by the revision.

[1] Ling, Yogatama, Dyer, Blunsom, "Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems", 2017. https://arxiv.org/abs/1705.04146 - the origin paper; current title read from the live abs page. Fetched 2026-08-11.

[2] deepmind/aqua_rat dataset card (README). https://huggingface.co/datasets/deepmind/aqua_rat/raw/main/README.md - dataset summary, data instance/fields, licence section, homepage/repository/paper links, "Needs More Information" fields. Fetched 2026-08-11.

[3] Hugging Face Hub API record for deepmind/aqua_rat. https://huggingface.co/api/datasets/deepmind/aqua_rat?full=true - `sha`, licence tag, gate status, `downloads`, `likes`, `lastModified`, `cardData.dataset_info`. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=deepmind%2Faqua_rat Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=deepmind%2Faqua_rat Fetched 2026-08-11.

[6] Hugging Face Hub API record for deepmind/aqua_rat, `expand[]=downloadsAllTime` variant. https://huggingface.co/api/datasets/deepmind/aqua_rat?expand%5B%5D=downloadsAllTime Fetched 2026-08-11.

[7] TIGER-Lab/MathInstruct dataset card (README). https://huggingface.co/datasets/TIGER-Lab/MathInstruct/raw/main/README.md - lists AQuA-RAT (Apache 2.0) among its source datasets, names the MAmmoTH paper and model checkpoints trained on it. Fetched 2026-08-11.

[8] Yue et al., "MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning", 2023. arXiv:2309.05653, cited by [7]; not independently fetched here.

[9] datasets-server first-rows endpoint, `raw`/`train`. https://datasets-server.huggingface.co/first-rows?dataset=deepmind%2Faqua_rat&config=raw&split=train Fetched 2026-08-11.

[10] datasets-server first-rows endpoint, `tokenized`/`train`. https://datasets-server.huggingface.co/first-rows?dataset=deepmind%2Faqua_rat&config=tokenized&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for reasoning-trace SFT, with the test split held out. The dataset's own card describes crowd-written algebra word problems with rationales, and its "raw"/"tokenized" pair of configs share one train/validation/test split structure whose test split is the published AQuA benchmark [2][4] - the same two facts the screening row's note names.

### The screening row

The row's own note: "Crowd-written algebra word problems with options and natural-language rationales; train/validation/test, and the test split is the AQuA benchmark." The row carries no flag.
