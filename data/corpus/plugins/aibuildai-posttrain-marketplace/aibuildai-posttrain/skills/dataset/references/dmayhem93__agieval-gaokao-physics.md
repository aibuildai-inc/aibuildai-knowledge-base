# dmayhem93/agieval-gaokao-physics

200 four-choice, Chinese-language physics questions from China's Gaokao (national college entrance exam), a single `test`-only split of the AGIEval benchmark suite.

**dmayhem93/agieval-gaokao-physics** repackages the Gaokao-physics subtask of AGIEval as Hugging Face parquet, reprocessing the data exactly as released in the `microsoft/AGIEval` GitHub repository [1]. AGIEval was introduced by Zhong et al. as "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models", built to score foundation models on human-centric standardized exams such as Gaokao, the SAT, and law-school admission tests [2]. Each row is one multi-choice question with four lettered options and one or more correct indices. **This is an evaluation-only benchmark, not training data, and the paper's own contamination check found only 5 of these 200 questions postdate the September 2021 training-data cutoff shared by ChatGPT and GPT-4 [2] - treat the whole set as likely already present in modern LLM pretraining corpora, so it should never be added to a training corpus and any Gaokao-physics eval score on a recent model should be read with that caveat.** It lives at https://huggingface.co/datasets/dmayhem93/agieval-gaokao-physics .

**Use it for**: scoring multi-choice accuracy on a held-out eval run, never for SFT or any other training - the row's `use` field records it as an eval set [3]. Rows are plain `(query, choices, gold)` triples with no chat template, mapping directly to a multi-choice eval harness format (score `choices[i]` against the index or indices in `gold`); there is no method card this maps to, since it feeds evaluation, not a training objective.

**Licence**: MIT (`license: mit` in `cardData`, tag `license:mit`), ungated, not private [4]. The one catch: the MIT text embedded in the README is Microsoft's software licence, copied verbatim from the `microsoft/AGIEval` GitHub repo into this dataset card [1]; no source separately states a licence or rights clearance for the underlying Gaokao exam questions themselves.

**Shape**: 200 rows, one config (`default`), one split (`test`) [4][5].

**Hold out**: all 200 rows. There is no train split to hold anything back from - the entire dataset is the eval set, and per the usage restriction above it should not appear in any training corpus given the paper's own near-total contamination finding [2][3].

**Origin**: built by Hugging Face user dmayhem93 from the Microsoft-released `microsoft/AGIEval` GitHub repo [1]; the questions themselves are human-authored real Gaokao physics exam items, not model-generated [2]. Hub API at the check date: `downloads` 1,912, `downloadsAllTime` 8,843, `likes` 1 [4].

**Trained-on-by**: none found. As an evaluation benchmark, its documented use in the origin paper is scoring GPT-4, ChatGPT, and Text-Davinci-003 zero/few-shot, not training [2]; no source states any model was trained on it, and doing so would be exactly the contamination the paper warns about.

**Introduced by**: [2] (Zhong et al., "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models").

## Shape

The datasets-server `/size`, `/info`, and `/first-rows` endpoints used throughout this section, Quality, and A row take no `revision` parameter - they always answer for whatever is at `main`, so the counts below are live reads at the check date, not covered by the commit pin (`3f82847f19ead1a682f0b27cc5c829ac964586bb`) shown in Load it [5][6][7].

Rows and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `test` | 200 |

One config, `default`, three columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `query` | string |
| `choices` | list\<string\> |
| `gold` | list\<int64\> |

Byte sizes: 70,363 bytes original parquet download (matches the shortlist's `bytes` field and the README's `download_size`) [3][4]. The README's static `cardData` states `num_bytes: 136757` for the decoded `test` split [1], while the live datasets-server `/size` and `/info` endpoints both report 136,787 decoded bytes [5][6] - a 30-byte difference between the pinned card metadata and the live endpoint that no source explains. The origin paper's Table 1 states an average of 124 tokens per instance for GK-physics, over its own count of 200 instances [2].

## Quality

- Evaluation metric: the paper scores multi-choice tasks like this one by accuracy, reserving Exact Match for the benchmark's separate fill-in-blank tasks (not this one) [2].
- Contamination is the dominant quality signal here. The paper's Appendix B contamination analysis reports a "full-set-size" of 200 for Gaokao-physics with 39% zero-shot GPT-4 accuracy, against an "uncontaminated-set-size" of only 5 questions (those released in 2022, after GPT-4's training cutoff) where GPT-4 scored 40% [2]. That 195-of-200 contaminated share is the deciding number for anyone considering this set a clean held-out eval today.
- The `gold` field is not always a single index. Reading the first 100 of the 200 `test` rows (offset 0, config `default`, via the live, unpinned datasets-server `/first-rows` endpoint) [7]: 86 rows carry exactly one correct index, 12 carry two, and 2 carry three - a multi-select minority of about 14% in that 100-row read that any accuracy-scoring code must handle rather than assuming `gold[0]` is the only answer.
- No source states a duplicate-rate, annotation-agreement figure, or any other quality signal beyond the contamination analysis above; none is invented here.

## Load it

```python
import datasets

REV = "3f82847f19ead1a682f0b27cc5c829ac964586bb"  # main at the check date
test = datasets.load_dataset("dmayhem93/agieval-gaokao-physics", revision=REV, split="test")  # 200 rows at this revision; row/column counts in Shape are separate, unpinned live reads
```

**Trap**: there is only a `test` split - passing `split="train"` fails, since none exists [4][5]. A second trap is scoring: `gold` is a list, not a scalar, and about one in seven rows in the first 100 has more than one correct index (see Quality above), so an accuracy function that compares a model's single predicted letter against `gold[0]` will silently mis-score the multi-select rows [7].

## Neighbors

- `hails/agieval-gaokao-physics` - the same 200 rows: row 0 from its `test` split is byte-identical to row 0 here (same `query`, `choices`, and `gold`) [8]. Its README states it was built "following dmayhem93/agieval-* datasets" from a pinned commit of the `ruixiangcui/AGIEval` GitHub fork, and its Hub `cardData` carries no `license` field, unlike this repo's stated MIT [8]. Prefer this dmayhem93 repo when a stated licence matters; the two are otherwise interchangeable content.
- Sibling subject splits from the same builder cover the rest of AGIEval's Gaokao portion: `dmayhem93/agieval-gaokao-biology`, `-chemistry`, `-chinese`, `-english`, `-geography`, `-history`, and `-mathqa` [1]. These are different exam subjects, not duplicates of this physics set, and combine with it to reconstruct AGIEval's full Chinese Gaokao battery; `hails/agieval-gaokao-mathcloze` fills the one Gaokao subject (math cloze) that dmayhem93 did not upload.
- The original source is the `microsoft/AGIEval` GitHub repository this Hub repo was processed from [1].

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="test"`, `row_idx=0`, read live via the unpinned datasets-server `/first-rows` endpoint (not covered by the Load it revision pin) [7]:

```json
{
  "query": "问题：20 世纪 60 年代, 我国以国防为主的尖端科技取得了突破性的发展。1964 年, 我国第一颗原子弹试爆成 功； 1967 年, 我国第一颗氢弹试爆成功。关于原子弹和氢弹, 下列说法正确的是（ ） 选项：(A)原子弹和氢弹都是根据核裂变原理研制的 (B)原子弹和氢弹都是根据核聚变原理研制的 (C)原子弹是根据核裂变原理研制的，氢弹是根据核聚变原理研制的 (D)原子弹是根据核聚变原理研制的，氢弹是根据核裂变原理研制的\n答案：从A到D, 我们应选择",
  "choices": [
    "(A)原子弹和氢弹都是根据核裂变原理研制的",
    "(B)原子弹和氢弹都是根据核聚变原理研制的",
    "(C)原子弹是根据核裂变原理研制的，氢弹是根据核聚变原理研制的",
    "(D)原子弹是根据核聚变原理研制的，氢弹是根据核裂变原理研制的"
  ],
  "gold": [2]
}
```

`query` bundles the Chinese question stem and its own inline option text plus an answer prompt; `choices` repeats the same four lettered options as separate strings; `gold` holds the zero-based index (or indices) of the correct option(s) - here index 2, option (C).

## Where it came from

Built by Hugging Face user dmayhem93, who took the Gaokao-physics subtask "from https://github.com/microsoft/AGIEval and processed as in that repo" [1]. The underlying questions are real items from China's Gaokao physics exam, collected and curated for AGIEval by Zhong et al. at Microsoft alongside seven other Gaokao subjects, the SAT, LSAT sections, math competitions, and other human-centric exam sources, then released as one benchmark to test foundation models against human-level performance on standardized exams [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was fetched on the check date, 2026-08-12; that date covers every number, quote, and row above. Hub repositories are mutable, which is why Load it pins the revision.

[1] dmayhem93/agieval-gaokao-physics dataset card (README). https://huggingface.co/datasets/dmayhem93/agieval-gaokao-physics/raw/main/README.md - source-repo attribution, MIT licence text, `cardData` byte/row counts, citation. Fetched 2026-08-12.

[2] Zhong, Cui, Guo, Liang, Lu, Wang, Saied, Chen, Duan, "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models", 2023. https://arxiv.org/abs/2304.06364 - abstract read from the live abs page (current title); Table 1 instance/token counts and the Appendix B contamination table read via the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2304.06364. Fetched 2026-08-12.

[3] The corpus screening row for `dmayhem93/agieval-gaokao-physics`, supplied with this card's request - its `use` and `note` fields. Checked 2026-08-12.

[4] Hugging Face Hub API record for dmayhem93/agieval-gaokao-physics. https://huggingface.co/api/datasets/dmayhem93/agieval-gaokao-physics?full=true - licence, gate status, `sha`, `downloads`, `lastModified`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=dmayhem93%2Fagieval-gaokao-physics Fetched 2026-08-12.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=dmayhem93%2Fagieval-gaokao-physics Fetched 2026-08-12.

[7] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=dmayhem93%2Fagieval-gaokao-physics&config=default&split=test - returns up to 100 rows; used for the row sample and the gold-length count over rows 0-99. Fetched 2026-08-12.

[8] hails/agieval-gaokao-physics: dataset card, Hub API record, and datasets-server size/first-rows endpoints. https://huggingface.co/datasets/hails/agieval-gaokao-physics/raw/main/README.md , https://huggingface.co/api/datasets/hails/agieval-gaokao-physics?full=true , https://datasets-server.huggingface.co/size?dataset=hails%2Fagieval-gaokao-physics , https://datasets-server.huggingface.co/first-rows?dataset=hails%2Fagieval-gaokao-physics&config=default&split=test - this endpoint takes no revision parameter, so the row-0 comparison is live, not pinned. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable strictly as an evaluation set, and never as training data: this is a 200-row, `test`-only AGIEval Gaokao-physics benchmark that the paper's own contamination analysis found almost entirely present in pre-2022 web text, and thus in the training data of contemporary LLMs. The screening row's own `use` field already marks it `eval` [3], consistent with the origin paper's use of it to score GPT-4, ChatGPT, and Text-Davinci-003 [2].

### The screening row

The row's own note [3]: "AGIEval Gaokao physics, `test` only." The row carries no flag.
