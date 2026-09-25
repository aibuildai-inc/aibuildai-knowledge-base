# doolayer/LawBench

Five of LawBench's twenty Chinese legal tasks, 500 test items each - judicial-exam multiple choice (1-2, 3-6) and CAIL-based judgment prediction (3-1 articles, 3-3 charges, 3-4 prison term) - with no card beyond a link.

**doolayer/LawBench** is a partial upload of LawBench, the OpenCompass Chinese legal benchmark; its README body is a single link to https://github.com/open-compass/LawBench/blob/main/README_EN.md [1]. It holds five configs of 500 `test` items, each with an `instruction` that fixes the answer format ("[正确答案]A<eoa>") , a `question` and an `answer` [2][3][4]. It lives at https://huggingface.co/datasets/doolayer/LawBench .

**An evaluation set whose items are in two training sets here: CAIL2018 `first_stage_train` holds 1,499 of the 1,500 items of tasks 3-1, 3-3 and 3-4, and DISC-Law-SFT's Pair file holds all 1,000 items of tasks 1-2 and 3-6.**

**Use it for**: evaluating Chinese legal judgment prediction and exam knowledge in LawBench's answer format.

**Licence**: none on the card [5][1]. The one catch: no licence on this upload; the upstream GitHub project states its own.

**Shape**: 2,500 rows: configs `1-2`, `3-1`, `3-3`, `3-4`, `3-6`, 500 `test` rows each [2]; three columns [3].

**Hold out**: all of it. Its judgment tasks (3-1, 3-3, 3-4) are drawn from CAIL2018 cases and its exam tasks (1-2, 3-6) also appear in `ShengbinYue/DISC-Law-SFT`'s Pair file [6]; do not train on either if you report LawBench.

**Origin**: judicial-exam questions (1-2, 3-6) and CAIL criminal cases (3-1, 3-3, 3-4) [4][6]. Hub API at the check date: `downloads` 271, `downloadsAllTime` 3,316, `likes` 2 [5].

**Trained-on-by**: an evaluation set; no model declares it [7].

**Introduced by**: LawBench (OpenCompass), linked from the card [1].

## Shape

Rows per config (datasets-server `/size`) [2]:

| config | split | rows |
| --- | --- | --- |
| `1-2` | `test` | 500 |
| `3-1` | `test` | 500 |
| `3-3` | `test` | 500 |
| `3-4` | `test` | 500 |
| `3-6` | `test` | 500 |
| all 5 configs | `test` 2,500 | 2,500 |

All configs share one schema [3]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `question` | string |
| `answer` | string |

## Quality

- Tasks 3-1, 3-3 and 3-4 start from the same case: row 0 of each is the "颜某" phone-theft case, asked three ways (articles, charges, sentence) [4] - and that case is row 0 of CAIL2018 `first_stage_train` [8].
- Only five of LawBench's tasks are here; the card does not say why [1].

## Load it

Evaluate on `test`:

```python
import datasets

REV = "23c262cf1d5593760be72eae0c76b9c0817fef68"  # main at the check date
charges = datasets.load_dataset("doolayer/LawBench", "3-3", revision=REV, split="test")  # 500 rows
```

**Trap**: answers must be wrapped in the task's own markers - `[正确答案]...<eoa>`, `[罪名]...<eoa>`, `[刑期]...<eoa>` - as each `instruction` specifies [4]; a model that answers correctly without them scores zero under the official parser.

## Neighbors

- `china-ai-law-challenge/cail2018` and `ShengbinYue/DISC-Law-SFT` - the training sets that contain these items.
- `hails/agieval-jec-qa-kd` - shares its first item with 1-2.

## A row

From `config="3-3"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [4], truncated:

```json
{
  "instruction": "请你模拟法官依据下面事实给出罪名，只需要给出罪名的名称，将答案写在[罪名]和<eoa>之间。例如[罪名]盗窃;诈骗<eoa>。请你严格按照这个格式回答。",
  "question": "事实:公诉机关指控：2016年3月28日20时许，被告人颜某在本市洪山区马湖新村足球场马路边捡拾到被害人谢某的VIVOX5手机一部，并在同年3月28日21时起，分多次通过支付宝小额免密支付功能，秘密盗走被害人谢某支付宝内人民币3723元。案发后，被告人颜某家属已赔偿被害人全部损失，并取得谅解。公诉机关认为被告人颜某具有退赃、取得谅解、自愿认罪等处罚情节，建议判处被告人颜某一年以下××、××或者××，并处罚金。\r\n",
  "answer": "罪名:盗窃"
}
```

## Where it came from

Uploaded by user doolayer in March 2025 from the OpenCompass LawBench project [5][1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] doolayer/LawBench dataset card (README). https://huggingface.co/datasets/doolayer/LawBench/raw/main/README.md. Fetched 2026-09-23.

[2] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=doolayer%2FLawBench - takes no revision parameter; a live figure. Fetched 2026-09-23.

[3] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=doolayer%2FLawBench - column schema; live, no revision parameter. Fetched 2026-09-23.

[4] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=doolayer%2FLawBench&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[5] Hugging Face Hub API record for doolayer/LawBench. https://huggingface.co/api/datasets/doolayer/LawBench?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[6] This skill's own check by n-gram containment against the benchmark's test items (word 8-grams, or character 15-grams for Chinese), with self, words-sorted and positive controls, at the pinned revision. Run 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:doolayer/LawBench&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] datasets-server first-rows endpoint for china-ai-law-challenge/cail2018, split `first_stage_train`. https://datasets-server.huggingface.co/first-rows?dataset=china-ai-law-challenge%2Fcail2018&config=default&split=first_stage_train. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

An evaluation set: hold out, and never report it after training on CAIL2018 `first_stage_train` or DISC-Law-SFT's Pair file without decontamination.

### The screening row

The row's own note: "5 LawBench tasks; items inside CAIL2018 and DISC-Law-SFT." The row carries the flag `contained-in-training-sets`.
