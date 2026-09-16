# codeparrot/apps

10,000 competitive-programming problems - hand-curated from Codewars, AtCoder, Kattis and Codeforces - each paired with human-written Python ground-truth solutions and input/output test cases, split evenly into a 5,000-row train half and a 5,000-row benchmark test half.

**codeparrot/apps** mirrors the APPS dataset introduced in "Measuring Coding Challenge Competence With APPS" [1]: problems are posed as natural-language specifications, curated by hand from four open-access competitive-programming sites, each carrying one or more human-submitted Python solutions and one or more test cases used to check functional correctness [1]. It serves single-turn code-generation training - a prompt built from the problem text plus any starter code, and a completion drawn from the JSON-encoded list of ground-truth solutions - and, as later work shows, execution-based RL fine-tuning that uses the `input_output` test cases as a pass/fail reward signal [2]. **The 5,000-row test split is the paper's own evaluation set, used to score pass@k across the introductory/interview/competition difficulty tiers [1]; hold it out and train only on the 5,000-row train split.** It lives at https://huggingface.co/datasets/codeparrot/apps .

**Use it for**: code-generation SFT (prompt: problem `question` plus `starter_code`; completion: one string picked from the JSON-encoded `solutions` list) - the SFT method card's shape - and, separately, execution-feedback RL training that samples a program and scores it against the JSON-encoded `input_output` cases, the shape CodeRL fine-tunes on [2]. Train only on the `train` split; hold out `test`.

**Licence**: MIT (`cardData.license` is `"mit"`, repo tag `license:mit`, and the loading script's own `dataset_info` also states `"MIT License"`) [3][4]. Ungated, not private [3]. The one catch: this MIT tag covers the Hub repository's own packaging; no source states whether it also covers redistribution of the underlying problem text pulled from Codewars, AtCoder, Kattis and Codeforces, since those are separate third-party sites the paper describes scraping and reformatting, not licensing [1].

**Shape**: 10,000 rows total. The default config (`all`, what `load_dataset("codeparrot/apps")` returns) has `train` 5,000 / `test` 5,000; three further configs filter by difficulty tier - `introductory` (2,639 train / 1,000 test), `interview` (2,000 train / 3,000 test), `competition` (361 train / 1,000 test) [4].

**Hold out**: `test` (5,000 rows) - APPS's own held-out evaluation split, used in the origin paper to report pass@k across all three difficulty tiers [1]; row count from the `all` config [4].

**Origin**: built by Hendrycks, Basart et al., who describe curating the problems by hand over six months with custom per-site HTML parsers, MathPix equation conversion and tf-idf/SVD deduplication [1]; every problem and solution is human-authored, sourced from programmers who posted on the four source sites [1]. Hosted on the Hub under the `codeparrot` organisation. Hub API at the check date: `downloads` 17,673, `downloadsAllTime` 766,080, `likes` 204 [3].

**Trained-on-by**: CodeRL fine-tunes CodeT5 with reinforcement learning directly on the APPS training set, reporting a new state of the art on the APPS benchmark and strong zero-shot transfer to MBPP from the resulting model [2]. AlphaCode also fine-tunes its own pre-trained models on the APPS training set, reporting pass@k results against GPT-Neo and Codex baselines on the APPS test set [5].

**Introduced by**: [1] (Hendrycks et al., NeurIPS 2021).

## Shape

Rows per config and split (datasets-server `/size`, requested per config since the bare, no-config call errors) [4]. This endpoint takes no revision parameter and does not validate one if supplied, so - like the Neighbors counts below - these figures are live reads against `main` at the check date, not pinned to the `sha` used in Load it; they matched the origin paper's own counts at fetch time (below), which is why this card treats them as reliable [1][4]:

| config | train | test | total |
| --- | --- | --- | --- |
| `all` (default) | 5,000 | 5,000 | 10,000 |
| `introductory` | 2,639 | 1,000 | 3,639 |
| `interview` | 2,000 | 3,000 | 5,000 |
| `competition` | 361 | 1,000 | 1,361 |

The three difficulty configs partition the same 10,000 problems as `all`; their counts match the origin paper's own per-tier breakdown of 3,639 / 5,000 / 1,361 problems (1,000 / 3,000 / 1,000 of each held out for test) [1]. Seven columns in every config: `problem_id` (int64), `question`, `solutions`, `input_output`, `difficulty`, `url`, `starter_code` (all string) [4].

Byte sizes for the `all` config [4]: 1,399,538,125 bytes of original download (`train.jsonl` 107,101,272 bytes, `test.jsonl` 1,292,436,853 bytes, matching the file sizes served at `resolve/main`), 788,750,989 bytes as auto-converted Parquet, and 1,329,350,372 bytes decoded in memory - split unevenly, with `train` at 103,144,035 bytes and `test` alone at 1,226,206,337 bytes, since the held-out test problems carry far more test cases per problem.

Sequence-length statistics the sources state: average problem length 293.2 words [1][6]; on average roughly 23.2 human solutions per problem (232,421 solutions across 10,000 problems, computed from the paper's two totals) [1]; average test cases per test-split problem 21.2 per the paper and the dataset card [1][6], versus 20.99 tests/problem measured across the whole dataset by AlphaCode's own count [5]. No source states token counts for any field.

## Quality

- 131,777 test cases and 232,421 human-written ground-truth solutions across the 10,000 problems [1].
- The dataset card's own statistics: every problem has at least one test case except 195 samples in the train split, and every problem has a ground-truth solution except 1,235 samples in the test split [6].
- The deciding number: AlphaCode's authors manually reviewed 50 APPS problems their own 1B-parameter model had solved, from the split the paper's Table 2 caption calls the "validation" split it evaluated on for APPS, and measured a 60% false-positive rate (programs that pass every given test but are not actually correct) and a 70% false-positive-or-slow rate, at 20.99 tests/problem - against their own CodeContests dataset's 4% false-positive rate at 203.7 tests/problem [5]. Whether that AlphaCode "validation" split is the same as this repository's `test` split is not stated in the source; APPS itself has no separate validation split, only `train` and `test` [1], so the two are likely the same set under different names, but this card does not assert that as a source fact. The dataset card repeats the same false-positive finding, pointing to AlphaCode's observation that sparse test coverage lets many incorrect submissions pass evaluation [6].
- No source states a duplicate-rate or annotator-agreement figure for this release beyond the paper's description of a tf-idf/SVD cosine-similarity deduplication pass during curation [1].

## Load it

Train on `train`, hold out `test`, and pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; last modified 2022-10-20) [3]:

```python
import datasets

REV = "21e74ddf8de1a21436da12e3e653065c5213e9d1"  # main at the check date
train = datasets.load_dataset("codeparrot/apps", split="train", revision=REV, trust_remote_code=True)  # 5,000 rows
test = datasets.load_dataset("codeparrot/apps", split="test", revision=REV, trust_remote_code=True)    # 5,000 rows - hold out
```

**Trap**: the repository ships a legacy Python loading script (`apps.py`), not a data-only Parquet repo [3], so current `datasets` versions require `trust_remote_code=True` or the load raises. The same script dependency makes the Hub's datasets-server `/info` and `/size` endpoints return an HTTP 501 error for a bare, no-config request ("doesn't support this dataset because it runs arbitrary Python code"); both endpoints only resolve once a config name (`all`, `introductory`, `interview` or `competition`) is supplied, reading from an auto-converted Parquet branch the Hub built separately (`refs/convert/parquet`, target commit `0f10e424e13e1c2a69f851e153097b71b6734a1f`) [4][7]. The dataset card's own example passes `difficulties=["competition"]` as a script keyword argument to filter by tier within a split, which is a different mechanism from the Hub config names above [6].

## Neighbors

- `4gate/codeparrot_apps` and `ReactiveAI/codeparrot-apps-reupload`: community Parquet re-uploads of this exact release - `train` 5,000 / `test` 5,000, the same seven-column schema, byte counts matching this card's `all` config within rounding - that load without `trust_remote_code`. Both carry far fewer downloads (142 and 96 respectively) than this original and neither declares an independent license, so they read as loader-compatibility mirrors, not a preferred source [8].
- `pxyyy/codeparrot-apps`: a reformatted, train-only, 5,000-row subset with two columns, not the full seven-field schema - not a drop-in replacement [8].
- `deepmind/code_contests`, the dataset released alongside AlphaCode [5]: a differently-sourced competitive-programming benchmark with far more generated tests per problem (203.7 vs. this dataset's 20.99) and a measured false-positive rate of 4% against this dataset's 60% [5]. The paper's own Table 1 gives its split sizes as 13,328 train / 165 test / 117 validation problems [5]; the live datasets-server `/size` endpoint used for this card's other neighbor counts instead returns a `"partial": true` response with only 3,762 train rows for this repository, undercounting against the paper, so the train figure here is the paper's, not the live endpoint's [8]. Prefer it over this dataset's train split when solution-correctness under evaluation matters more than raw problem count; this corpus keeps both since they cover non-overlapping problem sources [8].

## A row

Both splits share the same seven-column schema, but the test split's `input_output` field is typically far larger (Shape, above), so one row from each illustrates both shapes. From `config="all"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [9], with long fields truncated:

```json
{
  "problem_id": 0,
  "question": "Polycarp has $n$ different binary words. A word called binary if it contains only characters '0' and '1'. For example, these words are binary: \"0001\", \"11\", \"0\" and \"0011100\".\n\nPolycarp wants to offer his set of $n$ binary words to play a game \"words\". [...] Please, help him.\n\n\n-----Input-----\n\n[...]",
  "solutions": "[\"for _ in range(int(input())):\\n    n = int(input())\\n    mass = []\\n    zo = 0\\n    oz = 0\\n [...]\", \"...second solution...\"]",
  "input_output": "{\n  \"inputs\": [\n    \"4\\n4\\n0001\\n1000\\n0011\\n0111\\n3\\n010\\n101\\n0\\n2\\n00000\\n00001\\n4\\n01\\n001\\n0001\\n00001\\n\"\n  ],\n  \"outputs\": [\n    \"1\\n3 \\n-1\\n0\\n\\n2\\n1 2 \\n\"\n  ]\n}",
  "difficulty": "interview",
  "url": "https://codeforces.com/problemset/problem/1259/D",
  "starter_code": ""
}
```

From `config="all"`, `split="test"`, `row_idx=0` [9]:

```json
{
  "problem_id": 0,
  "question": "An accordion is a string (yes, in the real world accordions are musical instruments, but let's forget about it for a while) which can be represented as a concatenation of: an opening bracket [...] Is it possible to obtain an accordion by removing characters from $s$, and if so, what is the maximum possible length of the result?\n\n\n-----Input-----\n\n[...]",
  "solutions": "[\"s = input()\\nn = len(s)\\nind = -1\\nf = False\\n [...]\", \"...more solutions...\"]",
  "input_output": "{\n  \"inputs\": [\n    \"|[a:b:|]\"...\n  ],\n  \"outputs\": [\n    \"4\"...\n  ]\n}",
  "difficulty": "interview",
  "url": "https://codeforces.com/problemset/problem/1101/B",
  "starter_code": ""
}
```

Both rows carry `solutions` and `input_output` as JSON-encoded strings that must be parsed with `json.loads` before use, per the dataset card's own load example [6]. `starter_code` is empty on both sampled rows; the card states only a minority of problems set it [6].

## Where it came from

Built and released by Hendrycks, Basart et al. [1]. The authors manually curated problems from Codewars, AtCoder, Kattis and Codeforces, writing custom HTML parsers per source to normalize LaTeX, lists and sections, converting equation images to LaTeX with the MathPix API, and dropping problems that depend on image figures; they deduplicated with tf-idf features, SVD dimensionality reduction and cosine similarity, and report the curation itself took several graduate and undergraduate authors about six months [1]. Because each source site runs its own judge and difficulty scale, the authors built a unified testing framework merging each site's judging logic and mapped each site's own difficulty scale into three tiers - introductory, interview, competition [1]. The Hub repository is hosted under the `codeparrot` organisation [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Hendrycks, Basart, Kadavath, Mazeika, Arora, Guo, Burns, Puranik, He, Song, Steinhardt, "Measuring Coding Challenge Competence With APPS", NeurIPS 2021. https://arxiv.org/abs/2105.09938 - the origin paper; problem/solution/test-case totals, difficulty-tier counts, dataset-construction method, average problem length. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2105.09938). Fetched 2026-08-11.

[2] Le, Wang, Gotmare, Savarese, Hoi, "CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning", 2022. https://arxiv.org/abs/2207.01780 - CodeT5 fine-tuned with RL on the APPS training set, MBPP zero-shot transfer result. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2207.01780). Fetched 2026-08-11.

[3] Hugging Face Hub API record for codeparrot/apps. https://huggingface.co/api/datasets/codeparrot/apps?full=true - license, gated/private status, `sha`, `siblings` (confirming `apps.py`), `lastModified`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server info and size endpoints, called per config (`all`, `introductory`, `interview`, `competition`). https://datasets-server.huggingface.co/info?dataset=codeparrot%2Fapps&config=all and https://datasets-server.huggingface.co/size?dataset=codeparrot%2Fapps&config=<name> - schema, split row counts, byte sizes; a bare call with no `config` parameter returns an HTTP 501 script-execution error. Fetched 2026-08-11.

[5] Li et al., "Competition-Level Code Generation with AlphaCode", 2022. https://arxiv.org/abs/2203.07814 - Table 2 false-positive-rate comparison (APPS vs. HumanEval vs. CodeContests), CodeContests split sizes. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2203.07814). Fetched 2026-08-11.

[6] codeparrot/apps dataset card (README). https://huggingface.co/datasets/codeparrot/apps/raw/main/README.md - use example, `difficulties` filter example, data-field descriptions, dataset statistics, AlphaCode false-positive caveat. Fetched 2026-08-11.

[7] Hugging Face Hub API refs endpoint for codeparrot/apps. https://huggingface.co/api/datasets/codeparrot/apps/refs - the `main` branch target commit and the `refs/convert/parquet` auto-conversion branch and its target commit. Fetched 2026-08-11.

[8] Hugging Face Hub API and datasets-server size endpoints for each neighbor: `4gate/codeparrot_apps`, `ReactiveAI/codeparrot-apps-reupload`, `pxyyy/codeparrot-apps`, `deepmind/code_contests`. https://huggingface.co/api/datasets/<id> and https://datasets-server.huggingface.co/size?dataset=<id> - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[9] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=codeparrot%2Fapps&config=all&split=train and the same URL with `split=test`. Fetched 2026-08-11.

[10] The corpus screening row for `codeparrot/apps`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as training data, with a split-based restriction: train on `train`, hold out `test`. Two facts decide it, both already established above - the origin paper uses the `test` split to report the benchmark's own pass@k scores across difficulty tiers [1], and the screening row's note draws the same train/test line in its own words [10].

### The screening row

The row's own note [10]: "The APPS code-generation benchmark: 10,000 competitive-programming problems hand-curated from Codewars, AtCoder, Kattis and Codeforces, with human-written Python solutions and I/O test cases; the 5,000-row train half is trainable, the 5,000-row test half is APPS's own benchmark split (not a scored surface of the target benchmarks)." The row carries no flag.
