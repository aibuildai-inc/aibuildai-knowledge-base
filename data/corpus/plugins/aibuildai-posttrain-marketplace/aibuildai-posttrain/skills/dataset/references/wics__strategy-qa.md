# wics/strategy-qa

2,290 yes/no questions requiring implicit multi-hop reasoning, each with a human-written reasoning-step decomposition and supporting facts, served in a single split on the Hugging Face Hub.

**wics/strategy-qa** mirrors the StrategyQA benchmark introduced in "Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies" [1], where crowdworkers wrote strategy questions whose reasoning steps are not stated explicitly in the question itself, then decomposed each question into reasoning steps and matched Wikipedia evidence to each step [1]. The repository's loader script downloads a file literally named `strategyQA_train.json` from a GitHub mirror and yields it as a Hub split called `test` [2]; this served split's row count, 2,290, matches the paper's own Table 4 count for its **train** set exactly, not the paper's 490-question test set [1]. The official code repository points to a separate public leaderboard for scoring that 490-question test set [3], and a copy of the official test file served elsewhere on the Hub, `voidful/StrategyQA`'s `strategyqa_test.json`, holds exactly 490 rows keyed only by `qid` and `question`, with no `answer` field [4] - the 490 test-set answers are not the ones served in this repository and are not bundled with it. **This is the officially-labeled train portion of StrategyQA under a misleading `test` split name, not the paper's held-out evaluation set** - treat it as a benchmark-style eval resource with known public answers, not as data proven disjoint from any other model's training run. It lives at https://huggingface.co/datasets/wics/strategy-qa .

**Use it for**: evaluating implicit multi-hop, strategy-based yes/no question answering - the shortlist's own screening classifies this row's use as `eval` [5]. The rows are plain QA items (question, boolean answer, decomposition steps, supporting facts), not preference pairs or chat turns; they map to an eval/benchmark harness that scores yes/no answers, not to the SFT or preference-pair method cards, though `decomposition` and `facts` could seed a reasoning-trace SFT set if a pipeline chooses to build one - no source here states that as an intended use.

**Licence**: the Hub `license` tag reads `other`, but no source states what "other" means: the README's only content is the four-word `license: other` YAML block [6], and the loader script itself carries an empty `_LICENSE = ""` string next to the comment "TODO: Add the licence for the dataset here if you can find it" [2]. Treat the licence as unstated rather than as a real grant.

**Shape**: 2,290 rows, one split (`test`), one config (`strategyQA`), 7 columns [7][8].

**Hold out**: the entire 2,290-row `test` split - the whole repository is a single eval-only benchmark set with no separate train partition to train on, and its answers are public, so a model evaluated on it afterward should not have been trained on it [5].

**Origin**: this Hub copy was built by the account `wics`, which loads its rows from `https://raw.githubusercontent.com/wicsaax/strategy-qa/main/strategyQA_train.json`, a third-party GitHub mirror of the original data [2]; the underlying questions, decompositions, and facts are human crowdworker annotations collected for the original StrategyQA paper [1]. Hub API at the check date: `downloads` 7,819, `downloadsAllTime` 59,187, `likes` 11 [9].

**Trained-on-by**: none found. Every mention located of StrategyQA in later work uses it as an evaluation benchmark for multi-hop or chain-of-thought reasoning, not as training data; no source fetched here documents a model fine-tuned on this Hub copy specifically.

**Introduced by**: [1] (Geva et al., TACL 2021).

## Shape

The datasets-server `/size`, `/info`, and `/first-rows` endpoints used throughout this section and in A row take no revision parameter and always reflect the current `main` branch, not the pinned commit `f4d03d5ee3d1e302a9b4200e71231c46013eaeb2` used in Load it. The repository has eight commits, six of them titled "Update strategy-qa.py" after the file's creation [10], so it is not a single-commit repository where a live reading and a pinned reading are definitionally the same; a later push to `main` could change these figures without changing this card. Every row count, byte size, and sample row below is a live `main` reading taken on the check date, not a claim scoped to the pinned commit specifically.

Rows and split, from the datasets-server `/size` endpoint [7]:

| split | rows |
| --- | --- |
| `test` | 2,290 |

Columns and dtypes, from the datasets-server `/info` endpoint [8], matching the loader script's declared features [2]:

| column | dtype |
| --- | --- |
| `qid` | string |
| `term` | string |
| `description` | string |
| `question` | string |
| `answer` | bool |
| `facts` | list\<string\> |
| `decomposition` | list\<string\> |

No source fetched here states token-length statistics for this exact Hub serving. The paper's Table 4 gives word/step statistics for its 2,290-question train set, the same count served here: average question length 9.6 words, average decomposition length 2.93 steps, average number of matched paragraphs per question 2.33, and 46.8% of questions answered "yes" [1].

Sizes, from the datasets-server `/size` endpoint [7]: 1,270,268 bytes of original downloaded JSON, 719,905 bytes as Parquet, 1,080,171 bytes decoded in memory - matching the loader script's declared `download_size` and `dataset_size` [2][8].

## Quality

- Annotation is by crowdworkers under a multi-stage pipeline: term-based priming to elicit creative questions, decomposition of each question into reasoning steps, matching of Wikipedia evidence paragraphs to each step, and an adversarial "solver" filtering pass meant to eliminate questions answerable via reasoning shortcuts [1].
- A separate evidence-verification task (EVV) re-answers each decomposition step from only the matched paragraphs, run on a subset of examples during collection to catch pipeline and worker-quality issues [1].
- The paper reports that human solvers reach 87% accuracy on the task while its best baseline model reaches about 66%, the reference gap the benchmark is designed to probe [1].
- No source fetched here states a measured duplicate-question or contamination rate for this Hub copy; none is invented here.

## Load it

Pass the config name explicitly (the repository has one non-default config, `strategyQA`), and pin the revision this card's numbers were read at - the Hub API's `sha` for `main` at the check date, which matches the shortlist row's own `commit` field [9]:

```python
import datasets

REV = "f4d03d5ee3d1e302a9b4200e71231c46013eaeb2"  # main at the check date
data = datasets.load_dataset("wics/strategy-qa", name="strategyQA", revision=REV, split="test")  # 2,290 rows
```

**Trap**: the only split is named `test`, but its 2,290 rows come from a file named `strategyQA_train.json` and match the paper's train-set count exactly [1][2] - do not assume this split is the paper's disjoint, answer-withheld evaluation set; it is the labeled train portion of the original benchmark, republished under a `test` split name.

## Neighbors

- `tasksource/strategy-qa` serves the identical row count and byte sizes (2,290 rows; 1,270,268 bytes original, 1,080,171 bytes in memory) as this repository, in one `default` config with the split correctly named `train` rather than `test` [11]. Same data, more accurately labeled split name; prefer it if the mislabeled split name here is a concern.
- `ChilleD/StrategyQA` reshapes the same 2,290 total questions into an unofficial 1,603/687 `train`/`test` split, drops the `decomposition` column, and stores `facts` as a single string rather than a list [12][13]. Use it only if a ready-made train/test split is wanted; it discards the decomposition annotations that make this repository distinctive.

## A row

The repository serves one config and one split, so one row covers it. From `config="strategyQA"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [14]:

```json
{
  "qid": "b8677742616fef051f00",
  "term": "Genghis Khan",
  "description": "founder and first Great Khan of the Mongol Empire",
  "question": "Are more people today related to Genghis Khan than Julius Caesar?",
  "answer": true,
  "facts": [
    "Julius Caesar had three children.",
    "Genghis Khan had sixteen children.",
    "Modern geneticists have determined that  out of every 200 men today has DNA that can be traced to Genghis Khan."
  ],
  "decomposition": [
    "How many kids did Julius Caesar have?",
    "How many kids did Genghis Khan have?",
    "Is #2 greater than #1?"
  ]
}
```

## Where it came from

The questions, decompositions, and evidence facts originate from the StrategyQA data-collection pipeline described in the paper [1]: crowdworkers were shown a Wikipedia term and description as a priming term, wrote an implicit-reasoning strategy question about it, then a (possibly different) worker decomposed the question into reasoning steps and matched Wikipedia evidence paragraphs to each step, with periodic solver updates and adversarial filtering used to discourage reasoning shortcuts [1]. This Hub repository does not host that pipeline; its loader script downloads a single JSON file from a third-party GitHub mirror (`wicsaax/strategy-qa`) and republishes it as a `datasets` split named `test` [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Geva, Khashabi, Segal, Khot, Roth, and Berant, "Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies", TACL 2021. https://arxiv.org/abs/2101.02235 - the origin paper; Table 4 statistics and abstract accuracy numbers read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2101.02235). Fetched 2026-08-11.

[2] wics/strategy-qa loader script. https://huggingface.co/datasets/wics/strategy-qa/raw/main/strategy-qa.py - declared features, download URL, empty `_LICENSE` field, split name. Fetched 2026-08-11.

[3] Official StrategyQA code repository README (eladsegal/strategyqa). https://raw.githubusercontent.com/eladsegal/strategyqa/main/README.md - links to the StrategyQA leaderboard at https://leaderboard.allenai.org/strategyqa/ for scoring the held-out test set. Fetched 2026-08-11.

[4] voidful/StrategyQA, file `strategyqa_test.json`. https://huggingface.co/datasets/voidful/StrategyQA/raw/main/strategyqa_test.json - 490 rows, each with only `qid` and `question` keys and no `answer` field. Fetched 2026-08-11.

[5] The corpus screening row for `wics/strategy-qa`, supplied with this card's request - its `use` and `note` fields, read back in the appendix. Checked 2026-08-11.

[6] wics/strategy-qa dataset card (README). https://huggingface.co/datasets/wics/strategy-qa/raw/main/README.md - entire content is the four-word `license: other` YAML front matter. Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=wics%2Fstrategy-qa Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=wics%2Fstrategy-qa Fetched 2026-08-11.

[9] Hugging Face Hub API record for wics/strategy-qa. https://huggingface.co/api/datasets/wics/strategy-qa?full=true - `sha`, `downloads`, `likes`, gate/private status; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[10] Hugging Face Hub commits endpoint for wics/strategy-qa. https://huggingface.co/api/datasets/wics/strategy-qa/commits/main - eight commits total, six titled "Update strategy-qa.py"; confirms the repository is not single-commit, so the pinned revision used in Load it and the live datasets-server endpoints used elsewhere in this card are not guaranteed to agree. Fetched 2026-08-11.

[11] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=tasksource%2Fstrategy-qa Fetched 2026-08-11.

[12] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=ChilleD%2FStrategyQA Fetched 2026-08-11.

[13] datasets-server info endpoint for the neighbor. https://datasets-server.huggingface.co/info?dataset=ChilleD%2FStrategyQA Fetched 2026-08-11.

[14] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=wics%2Fstrategy-qa&config=strategyQA&split=test Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as an evaluation benchmark only, and only with the split-naming caveat established above: the served `test` split is 2,290 rows that match the paper's train-set count, not its held-out test set [1][2]. The screening row's own note already states the dataset's shape and its single-split limitation [5].

### The screening row

The row's own note [5]: "StrategyQA implicit multi-hop questions with decompositions and facts; the repo has only a `test` split." The row carries no flag.
