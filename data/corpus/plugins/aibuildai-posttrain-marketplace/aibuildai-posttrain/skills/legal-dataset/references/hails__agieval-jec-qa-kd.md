# hails/agieval-jec-qa-kd

The 1,000-question knowledge-driven JEC-QA subtask of AGIEval: Chinese national judicial examination multiple-choice questions, test-only.

**hails/agieval-jec-qa-kd** contains "the contents of the JEC-QA-KD subtask of AGIEval", taken from Microsoft's AGIEval repository at a pinned commit and processed as there [1][2]. Each row is a `query` with four lettered `choices` and a `gold` index list [3][4]. It lives at https://huggingface.co/datasets/hails/agieval-jec-qa-kd .

**An evaluation set - and 902 of its 1,000 items are inside `ShengbinYue/DISC-Law-SFT`'s Pair file at ≥ 50% n-gram coverage.**

**Use it for**: evaluating Chinese legal-knowledge multiple choice.

**Licence**: none on the card [5]. The one catch: no licence.

**Shape**: 1,000 rows, one `test` split [6]; three columns [3].

**Hold out**: all of it. Measured [7]: `DISC-Law-SFT-Pair.jsonl` holds 902 items (375 at ≥ 80%). Its first item is also LawBench 1-2's first item [4][8].

**Origin**: questions from the Chinese national judicial examination, via JEC-QA and AGIEval [1]. Hub API at the check date: `downloads` 2,319, `downloadsAllTime` 60,166, `likes` 8 [5].

**Trained-on-by**: an evaluation set; no model declares it [9].

**Introduced by**: [2] (Zhong et al., AGIEval).

## Shape

| split | rows |
| --- | --- |
| `test` | 1,000 |
| total | 1,000 |

One config, `default` [3]:

| column | dtype |
| --- | --- |
| `query` | string |
| `choices` | list<string> |
| `gold` | list<int64> |

## Quality

- `gold` is a list of indices (row 0: `[1]`) [4].

## Load it

Evaluate on `test`:

```python
import datasets

REV = "970133b139bcf1cc07f944851dba0e6b715d5659"  # main at the check date
jec = datasets.load_dataset("hails/agieval-jec-qa-kd", revision=REV, split="test")  # 1,000 rows
```

**Trap**: the same questions reappear in LawBench 1-2 [4][8]; reporting both double-counts one question pool.

## Neighbors

- `hails/agieval-jec-qa-ca` - the case-analysis subtask, found by the Hub search [10]; not screened here.
- `doolayer/LawBench` - shares items with this set.

## A row

From `config="default"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [4], truncated:

```json
{
  "query": "问题：中国商务部决定对原产于马来西亚等八国的橡胶制品展开反补贴调查。根据我国《反补贴条例》以及相关法律法规，下列关于此次反补贴调查的哪项判断是正确的? 选项：(A)我国商务部在确定进口橡胶制品是否存在补贴时必须证明出国(地区)政府直接向出口商提供了现金形式的财政资助 (B)在反补贴调查期间，该八国政府或橡胶制品的出口经营者，可以向中国商务部作出承诺，取消、限制补贴或改变价格 (C)如果我国商务部终 [...]",
  "choices": [
    "(A)我国商务部在确定进口橡胶制品是否存在补贴时必须证明出国(地区)政府直接向出口商提供了现金形式的财政资助",
    "(B)在反补贴调查期间，该八国政府或橡胶制品的出口经营者，可以向中国商务部作出承诺，取消、限制补贴或改变价格",
    "(C)如果我国商务部终局裁定决定对该八国进口橡胶制品征收反补贴税，该反补贴税的征收期限不得超过10年",
    "(D)如果中国橡胶制品进口商对商务部征收反补贴税的终局裁定不服，必须首先向商务部请求行政复审，对行政复审决定还不服，才能向中国有管辖权的法院起诉"
  ],
  "gold": [
    1
  ]
}
```

## Where it came from

Processed by user hails from AGIEval at commit 5c77d073 of the AGIEval repository [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] hails/agieval-jec-qa-kd dataset card (README). https://huggingface.co/datasets/hails/agieval-jec-qa-kd/raw/main/README.md. Fetched 2026-09-23.

[2] Zhong et al., "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models", arXiv:2304.06364, 2023. https://arxiv.org/abs/2304.06364 - current title read from the live abs page. Fetched 2026-09-23.

[3] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=hails%2Fagieval-jec-qa-kd - column schema; live, no revision parameter. Fetched 2026-09-23.

[4] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=hails%2Fagieval-jec-qa-kd&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[5] Hugging Face Hub API record for hails/agieval-jec-qa-kd. https://huggingface.co/api/datasets/hails/agieval-jec-qa-kd?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=hails%2Fagieval-jec-qa-kd - takes no revision parameter; a live figure. Fetched 2026-09-23.

[7] This skill's own measurement, `references/contamination.md`, row "JEC" - n-gram containment with its three controls; method and script in that file - LawBench + JEC-QA-KD vs DISC-Law-SFT files; character 15-grams. Run 2026-09-23.

[8] datasets-server first-rows endpoint for doolayer/LawBench, config `1-2`. https://datasets-server.huggingface.co/first-rows?dataset=doolayer%2FLawBench&config=1-2&split=test. Fetched 2026-09-23.

[9] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:hails/agieval-jec-qa-kd&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[10] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=JEC-QA&sort=downloads. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

An evaluation set: hold out, and strip DISC-Law-SFT's `exam-*` rows from any training mix before reporting it.

### The screening row

The row's own note: "AGIEval JEC-QA-KD, 1,000 Chinese judicial-exam MCQs." The row carries no flag.
