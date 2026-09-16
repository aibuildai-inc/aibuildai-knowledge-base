# dmayhem93/agieval-gaokao-chemistry

https://huggingface.co/datasets/dmayhem93/agieval-gaokao-chemistry

207 Chinese-language, four-choice questions from the chemistry section of the Chinese national college entrance exam (Gaokao), served as a single `test` split - an evaluation-only benchmark, not a training corpus.

**dmayhem93/agieval-gaokao-chemistry** packages the Gaokao-chemistry subtask of AGIEval, the human-centric exam benchmark introduced in "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models" [1], reformatted from the original AGIEval repository into flat `query`/`choices`/`gold` rows for multiple-choice accuracy scoring [2]. **This is a held-out human exam benchmark, not a training set: it exists to measure a model's chemistry-reasoning accuracy against real Gaokao takers, and rows sourced from real exam papers carry web-scrape contamination risk for models trained after these exams were published** [1].

**Use it for**: held-out multiple-choice evaluation of chemistry reasoning in Chinese, never SFT or preference training; the row shape is a flat `{query, choices, gold}` multiple-choice item, with `query` holding the full prompt text and `choices` the four lettered options - EleutherAI's lm-evaluation-harness consumes this exact shape with `doc_to_text: "{{query}}"`, `doc_to_choice: "{{choices}}"`, `doc_to_target: "{{gold}}"` and scores `acc`/`acc_norm` [3]. See the eval-harness method card for how to run it, not an SFT or DPO card.

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated, non-gated repo [4]. No catch beyond the standard MIT grant; the README carries the full MIT licence text of the original AGIEval repository [2].

**Shape**: 207 rows, one config (`default`), one split (`test`); three columns, `query` (string), `choices` (list of string), `gold` (list of int64) [5][6].

**Hold out**: the entire dataset (207 rows) is itself an evaluation set with no separate held-out portion - it has no train split to leak into it, but every row is drawn from a real, publicly-scraped Gaokao exam and so may already sit inside the pretraining corpus of any model trained on general web text after the exam's publication; the origin paper's own contamination check (below) quantifies this for the closest exam years [1].

**Origin**: reformatted by Hugging Face user dmayhem93 from the Microsoft AGIEval GitHub repository [2]; the exam questions and correct answers are the real, human-authored Gaokao chemistry exam, not model-generated [1]. Hub API at the check date: `downloads` 2,157 (last 30 days) / 9,437 all-time, `likes` 1 [4].

**Trained-on-by**: EleutherAI's lm-evaluation-harness ships a ready-made `agieval_gaokao_chemistry` task, though it points at the re-hosted `hails/agieval-gaokao-chemistry` copy of this same 207-row set rather than this exact repo [3]. No source found naming a specific released model as having trained on this repository; AGIEval as a whole is used to evaluate, not train, foundation models, and the origin paper reports GPT-4, ChatGPT and Text-Davinci-003 evaluation scores on it, not training on it [1].

**Introduced by**: [1] (Zhong et al., "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models").

## Shape

Rows and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `test` | 207 |

One config, `default`, three columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `query` | string |
| `choices` | sequence\<string\> |
| `gold` | sequence\<int64\> |

The origin paper's own Table 1 states 207 instances for this exact task ("GK-chemistry / Chemistry / 207") with an average of 113 tokens per instance - this is the only source-stated token statistic and it matches this repo's row count exactly [1].

## Quality

- Every one of the first 100 of 207 served rows (offsets 0-99, the full page returned by the datasets-server `first-rows` endpoint) has a non-empty `query`, exactly four `choices`, and a `gold` list of length one - i.e. single-label four-way multiple choice throughout the sample read [7].
- No source states a measured error, duplicate, or annotator-agreement rate for this repository specifically.
- The origin paper's own contamination analysis (its Table 5) reports that of the Gaokao subjects it checked for timestamp-based leakage, chemistry has 207 total instances of which 64 were released in 2022 or later (post-dating GPT-4/ChatGPT's training cutoff); on that 64-row uncontaminated subset GPT-4's zero-shot accuracy drops from 51.7% (full 207) to 42.2%. The paper's own reading of this pattern, stated across all the Gaokao subjects it checked and not specifically for chemistry, is that "barring the Mathematics subjects, the performance experiences a minor drop in the absence of contamination, yet remains proximate to the performances on the complete datasets," which it takes as evidence the benchmark "still retains its value as a useful and effective human-centric benchmark" despite some contamination risk [1].
- The paper's Table 2 gives the deciding numbers for the full 207-row zero-shot setting: human average accuracy 66%, human top-scorer accuracy 86%, versus zero-shot GPT-4 51.7%, ChatGPT 38.7%, and Text-Davinci-003 27.1% - so GPT-4 clears neither the average nor the top human bar on this task [1].

## Load it

```python
import datasets

REV = "2fb33cf46ce4aeea9409ea3600a3b1d7e5216536"  # main at the check date
test = datasets.load_dataset("dmayhem93/agieval-gaokao-chemistry", revision=REV, split="test")  # 207 rows
```

**Trap**: this repository has only a `test` split - there is no `train` or `validation` to accidentally mix in, but by the same token there is nothing here to fine-tune on; treat every row as held-out evaluation data.

## Neighbors

`hails/agieval-gaokao-chemistry` is a re-hosted copy built from a later commit of the same upstream AGIEval repository, and it - not this repository - is the `dataset_path` that EleutherAI's lm-evaluation-harness actually loads for its `agieval_gaokao_chemistry` task [3]. Fetched live at the check date, it serves the identical 207 rows with the same three columns and identical text and labels on row 0, but a slightly different byte size (`download_size` 77,487 vs 78,411 here), consistent with a separate re-scrape of the same content rather than a different sample [8]. Prefer this dmayhem93 repository when reproducing the original paper's exact release; prefer `hails/agieval-gaokao-chemistry` when running lm-evaluation-harness's built-in AGIEval tasks unmodified, since that is the path its config hard-codes [3]. The same dmayhem93 account also hosts the other 16 AGIEval subtasks (e.g. `agieval-gaokao-biology`, `agieval-gaokao-physics`, `agieval-gaokao-mathqa`, `agieval-lsat-lr`, `agieval-sat-math`) in the same flat `query`/`choices`/`gold` shape [9]. The upstream AGIEval GitHub repository has since moved to a "v1.1" release that it says updates the Gaokao chemistry, biology and physics data with 2023 questions and fixes annotation issues; this Hub repository's row count (207) matches the original paper's v1.0 Table 1 exactly, so it reflects v1.0, not the newer v1.1 revision [10].

## A row

One config, one split, one row shape. From `config="default"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [7]:

```json
{
  "query": "问题：以下是中华民族为人类文明进步做出巨大贡献的几个事例, 运用化学知识对其 进行的分析不合理的是 ( $)$ 选项：(A)四千余年前用谷物酿造出酒和酯, 酿造过程中只发生水解反应 (B)商代后期铸造出工艺精湛的后（司）母戊鼎, 该鼎属于铜合金制品 (C)汉代烧制出\"明如镜、声如磬\"的瓷器，其主要原料为黏土 (D)屠呦呦用乙醚从青蒿中提取出对治疗疘疾有特效的青高素, 该过程包括萃取操作\n答案：从A到D, 我们应选择",
  "choices": [
    "(A)四千余年前用谷物酿造出酒和酯, 酿造过程中只发生水解反应",
    "(B)商代后期铸造出工艺精湛的后（司）母戊鼎, 该鼎属于铜合金制品",
    "(C)汉代烧制出\"明如镜、声如磬\"的瓷器，其主要原料为黏土",
    "(D)屠呦呦用乙醚从青蒿中提取出对治疗疘疾有特效的青高素, 该过程包括萃取操作"
  ],
  "gold": [0]
}
```

`query` embeds the full question and all four lettered options followed by an answer prompt; `choices` repeats the same four options as separate strings for scoring; `gold` holds the zero-based index of the correct option (here `0`, i.e. option A).

## Where it came from

The Gaokao chemistry questions come from the Chinese national college entrance exam and were compiled into AGIEval by the paper's authors as one of nine Gaokao subject subtasks spanning Chinese, math, English, physics, chemistry, biology, history, geography and politics [1]. Hugging Face user dmayhem93 took the processed data from the Microsoft AGIEval GitHub repository and converted it, following that repository's own processing, into this flat `query`/`choices`/`gold` parquet format; the dataset card states this directly ("Dataset taken from https://github.com/microsoft/AGIEval and processed as in that repo") [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Zhong et al., "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models", 2023. https://arxiv.org/abs/2304.06364 - origin paper; abstract, Table 1 (task sizes and average tokens), Table 2 (zero-shot human vs. model accuracy), Table 5 (contamination analysis). Read via the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2304.06364; current title read from the live arXiv abs page. Fetched 2026-08-12.

[2] dmayhem93/agieval-gaokao-chemistry dataset card (README). https://huggingface.co/datasets/dmayhem93/agieval-gaokao-chemistry/raw/main/README.md - provenance statement, MIT licence text, citation. Fetched 2026-08-12.

[3] EleutherAI lm-evaluation-harness AGIEval task configs. https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/agieval/gaokao-chemistry.yaml (names `dataset_path: hails/agieval-gaokao-chemistry`) and https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/agieval/aqua-rat.yaml (shared `doc_to_text`/`doc_to_choice`/`doc_to_target`/metric config included by every AGIEval subtask, including gaokao-chemistry). A `main`-branch build, unpinned and mutable. Fetched 2026-08-12.

[4] Hugging Face Hub API record for dmayhem93/agieval-gaokao-chemistry. https://huggingface.co/api/datasets/dmayhem93/agieval-gaokao-chemistry?full=true - licence, gate status, `sha`, `downloads`, `likes`; all-time downloads read via the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=dmayhem93%2Fagieval-gaokao-chemistry Fetched 2026-08-12.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=dmayhem93%2Fagieval-gaokao-chemistry Fetched 2026-08-12.

[7] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=dmayhem93%2Fagieval-gaokao-chemistry&config=default&split=test - 100 rows returned, offsets 0-99. Fetched 2026-08-12.

[8] hails/agieval-gaokao-chemistry dataset card and size endpoint, read for the neighbor comparison: https://huggingface.co/datasets/hails/agieval-gaokao-chemistry/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=hails%2Fagieval-gaokao-chemistry, cross-checked against a fetched row 0 via https://datasets-server.huggingface.co/first-rows?dataset=hails%2Fagieval-gaokao-chemistry&config=default&split=test - this endpoint takes no revision parameter, so this comparison is live, not pinned. Fetched 2026-08-12.

[9] Hugging Face Hub API dataset listing for author dmayhem93. https://huggingface.co/api/datasets?author=dmayhem93&limit=100 - enumerates the sibling `agieval-*` repositories. Fetched 2026-08-12.

[10] Microsoft AGIEval GitHub repository README (mirrored under ruixiangcui/AGIEval). https://raw.githubusercontent.com/ruixiangcui/AGIEval/master/README.md - states the v1.1 update refreshed Gaokao chemistry/biology/physics with 2023 questions. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as an evaluation-only benchmark: 207 Gaokao-chemistry multiple-choice items in `test`, no train split to hold anything out from. This rests on facts already established above - the dataset's own card states it is a direct reformatting of the AGIEval Gaokao-chemistry subtask [2], and the origin paper defines it as an exam-derived evaluation benchmark, not training data [1].

### The screening row

The row's own note [row supplied with this card's request]: "AGIEval Gaokao chemistry, `test` only." The row carries no flag.
