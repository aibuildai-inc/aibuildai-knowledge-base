# dmayhem93/agieval-gaokao-biology

210 Chinese-language, single-answer multiple-choice biology questions from the Chinese college entrance exam (Gaokao), served as a single `test` split - an evaluation-only benchmark, not a training corpus.

**dmayhem93/agieval-gaokao-biology** packages the Gaokao Biology subtask of AGIEval, a human-centric benchmark of official admission and qualification exams introduced by Zhong et al. in "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models" [1]. The dataset card states it is taken from the official AGIEval GitHub repository and processed as in that repo [2]. Each row is a four-option multiple-choice question drawn from the Chinese Gaokao biology exam, with the query, its four choices, and the index of the correct choice [2][3]. **This is an evaluation benchmark, not a training set: it carries a single `test` split with no `train` split, so it has nothing to train a model on directly** [2][4]. It lives at https://huggingface.co/datasets/dmayhem93/agieval-gaokao-biology .

**Use it for**: held-out evaluation of Chinese-language, exam-style multiple-choice reasoning - not for training. Rows are multiple-choice with an implicit prompt (`query`), a `choices` list, and a single-element `gold` index list; EleutherAI's lm-evaluation-harness config for this exact dataset renders it as `output_type: multiple_choice`, scored with `acc` and `acc_norm` on the `test` split [5]. This does not map to any post-training method card - it is a benchmark, not a training-shape corpus.

**Licence**: MIT (`cardData.license` is `"mit"`; the card body reproduces the MIT licence text and attributes it to Microsoft Corporation) [2][6]. No further catch beyond the standard MIT grant.

**Shape**: 210 rows, one config (`default`), one split (`test` only) [4][7].

**Hold out**: nothing to hold out inside this repository - it has only a `test` split and no companion train split to leak against. The dataset itself is a decontamination concern for anyone training a model that might later be evaluated on AGIEval Gaokao-Biology, but that concern is external to this card's own splits [4].

**Origin**: built by dmayhem93, converting the human-authored AGIEval Gaokao-Biology questions (originally sourced from real Chinese Gaokao biology exams) into a Hugging Face parquet dataset [1][2]. Hub API at the check date: `downloads` 1,933, `downloadsAllTime` 9,126, `likes` 2 [6].

**Trained-on-by**: none found - a Hub search for models declaring this dataset in their metadata returns zero results, and no other fetched source states a model trained on it [8]. It is used as an evaluation benchmark instead: EleutherAI's lm-evaluation-harness includes a task, `agieval_gaokao_biology`, that scores models zero/few-shot against this exact question set (via the `hails/agieval-gaokao-biology` mirror), which is evaluation usage, not training [5].

**Introduced by**: [1] (Zhong et al.).

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `test` | 210 |

One config, `default`, with three columns (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `query` | string |
| `choices` | sequence<string> |
| `gold` | sequence<int64> |

The origin paper's Table 1 lists this exact task as `GK-biology`, subject Biology, 210 instances, 141 average tokens per instance - matching this repository's row count [1].

## Quality

- Of the 100 rows served by datasets-server's first-rows endpoint (offset 0 of 210, `test` split), every row has exactly 4 entries in `choices` and exactly 1 index in `gold`, i.e. single-answer four-way multiple choice; the remaining 110 rows were not read, so this description covers only the observed 100 [3].
- No source states a measured error rate, duplicate rate, or annotator-agreement figure for this dataset; none is invented here.
- The dataset card gives no quality narrative beyond naming its origin repository and licence; any construction methodology (how the raw Gaokao exam text was parsed into query/choices/gold) is described in the AGIEval paper and code, not in this card [1][2].
- The origin paper's own zero-shot results table for GK-biology shows the benchmark is not saturated: average human performance is 68 and top human performance is 89, against zero-shot scores of 40.5 (Text-Davinci-003), 52.9 (ChatGPT), and 75.7 (GPT-4), and zero-shot chain-of-thought scores of 30.0, 42.4, and 71.9 for the same three models respectively - so the strongest model tested (GPT-4, zero-shot, 75.7) clears average human performance but stays well below the top human score of 89 [1].

## Load it

Only a `test` split exists; there is nothing to split into train/holdout within this repository. Pin the revision this card's numbers were read at (the commit named in the shortlist row, matching the Hub API's `sha` for `main`; the repo was last modified 2023-06-18) [6]:

```python
import datasets

REV = "91e58112d4022523e02d07cfbc96a950eac9219f"  # main at the check date
test = datasets.load_dataset("dmayhem93/agieval-gaokao-biology", revision=REV, split="test")  # 210 rows
```

**Trap**: there is no `train` split to accidentally load or fine-tune on - the only split is `test`, so any code path that assumes a `train`/`test` pair will fail outright rather than silently mixing eval data into training [4].

## Neighbors

- `hails/agieval-gaokao-biology` - a separate Hub repository with the same 210-row, same-schema (`query`/`choices`/`gold`) content; its own card states it was taken from the same upstream AGIEval GitHub source, following the naming of the `dmayhem93/agieval-*` datasets on the Hub, and names the upstream commit it was read at [9]. Row count and column types match exactly (210 rows, one `test` split) [10]. This is the repository EleutherAI's lm-evaluation-harness task `agieval_gaokao_biology` actually loads by its `dataset_path` [5]; prefer it only if reproducing that harness's exact behavior matters, otherwise this repository and that one serve identical content.
- Sixteen other `dmayhem93/agieval-*` repositories (17 returned by the same author search, minus this one) cover the other AGIEval subtasks (e.g. `agieval-gaokao-chemistry`, `agieval-gaokao-physics`, `agieval-lsat-lr`) [11]; these are different subjects, not alternate releases of this biology set, so they do not overlap with it.

## A row

The repository serves one config and one schema, so one row covers it. From `config="default"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [3]:

```json
{
  "query": "问题：已知(1)酶、(2)抗体、(3)激素、(4)糖原、(5)脂肪、(6)核酸都是人体内有重要作用的物质。下列说法正确的 是  选项：(A)(1)(2)(3)都是由氨基酸通过肽键连接而成的 (B)(3)(4)(5)都是生物大分子, 都以碳链为骨架 (C)(1)(2)(6)都是由含氮的单体连接成的多聚体 (D)(4)(5)(6)都是人体细胞内的主要能源物质\n答案：从A到D, 我们应选择",
  "choices": [
    "(A)(1)(2)(3)都是由氨基酸通过肽键连接而成的",
    "(B)(3)(4)(5)都是生物大分子, 都以碳链为骨架",
    "(C)(1)(2)(6)都是由含氮的单体连接成的多聚体",
    "(D)(4)(5)(6)都是人体细胞内的主要能源物质"
  ],
  "gold": [2]
}
```

`query` embeds the full Chinese question stem and all four lettered options followed by an answer prompt; `choices` repeats those same four option strings separately; `gold` holds the zero-indexed position of the correct option (here, index 2, option C) [3].

## Where it came from

Built by dmayhem93 from the AGIEval benchmark's Gaokao Biology subtask, itself drawn from real Chinese Gaokao (college entrance exam) biology questions and released as part of "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models" [1]. The dataset card states the data was taken from the official AGIEval GitHub repository and processed following that repository's own pipeline [2]; the questions and answer keys are human-authored exam content, not model-generated [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision to `91e58112d4022523e02d07cfbc96a950eac9219f`, the `sha` the Hub API reported for `main` at the check date [6]. That pin covers only what `load_dataset` reads from the repository itself. The row counts, column types, and sampled row in Shape, Quality, and A row come from the datasets-server `/size`, `/info`, and `/first-rows` endpoints [3][4][7], none of which accept a revision parameter and none of which report one in their response - so those figures are read from the live, unpinned endpoint state at the check date, not from the pinned commit, even though this repository has had only the single commit named above since it was created [6].

[1] Zhong et al., "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models", 2023. https://arxiv.org/abs/2304.06364 - the origin paper; Table 1 lists `GK-biology`, 210 instances, 141 average tokens, and Table 2 lists the GK-biology human and zero-shot model scores. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2304.06364); current title read from the live abs page. Fetched 2026-08-12.

[2] dmayhem93/agieval-gaokao-biology dataset card (README). https://huggingface.co/datasets/dmayhem93/agieval-gaokao-biology/raw/main/README.md - origin statement, MIT licence text, citation. Fetched 2026-08-12.

[3] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=dmayhem93%2Fagieval-gaokao-biology&config=default&split=test - 100 rows read, offset 0. Fetched 2026-08-12.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=dmayhem93%2Fagieval-gaokao-biology Fetched 2026-08-12.

[5] EleutherAI lm-evaluation-harness, AGIEval Gaokao-Biology task config. https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/agieval/gaokao-biology.yaml (task metadata inherited from https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/agieval/aqua-rat.yaml) - `dataset_path: hails/agieval-gaokao-biology`, `output_type: multiple_choice`, `test_split: test`, metrics `acc`/`acc_norm`. A `main` build, unpinned and mutable. Fetched 2026-08-12.

[6] Hugging Face Hub API record for dmayhem93/agieval-gaokao-biology. https://huggingface.co/api/datasets/dmayhem93/agieval-gaokao-biology?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=dmayhem93%2Fagieval-gaokao-biology Fetched 2026-08-12.

[8] Hugging Face Hub API models search filtered to this dataset. https://huggingface.co/api/models?filter=dataset:dmayhem93/agieval-gaokao-biology - returned an empty list, the evidence for the "none found" adoption claim. A live, unpinned endpoint. Fetched 2026-08-12.

[9] hails/agieval-gaokao-biology dataset card (README). https://huggingface.co/datasets/hails/agieval-gaokao-biology/raw/main/README.md Fetched 2026-08-12.

[10] datasets-server size and info endpoints for hails/agieval-gaokao-biology. https://datasets-server.huggingface.co/size?dataset=hails%2Fagieval-gaokao-biology and https://huggingface.co/api/datasets/hails/agieval-gaokao-biology?full=true Fetched 2026-08-12.

[11] Hugging Face Hub API dataset listing for author dmayhem93 matching "agieval". https://huggingface.co/api/datasets?author=dmayhem93&search=agieval - 17 repositories total, including this one (16 siblings). Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable only as an evaluation benchmark, not as training data: the repository carries a single `test` split of 210 human-authored Gaokao biology multiple-choice questions, with no train split and no route to a training shape. This rests on the shape established above - one config, one `test` split, 210 rows [4][7] - and on the screening row's own note.

### The screening row

The row's own note [the corpus screening row for `dmayhem93/agieval-gaokao-biology`, supplied with this card's request, checked 2026-08-12]: "AGIEval Gaokao biology exam questions; test split only." The row carries no flag.
