# china-ai-law-challenge/cail2018

CAIL2018: 2.17 million Chinese criminal cases - fact descriptions with charges, statute articles and sentences - in six splits that overlap each other heavily, and the source of LawBench's judgment tasks.

**china-ai-law-challenge/cail2018** is the Chinese AI and Law Challenge 2018 dataset, introduced as "CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction" by Xiao et al. [1]. Each row is a criminal case's `fact` description with its `accusation` (charges), `relevant_articles` (Criminal Law articles), `imprisonment` in months, `punish_of_money`, `death_penalty`, `life_imprisonment` and `criminals` [2]. Every section of the card body reads "[More Information Needed]" [3]. It lives at https://huggingface.co/datasets/china-ai-law-challenge/cail2018 .

**The splits are not disjoint: 56.56% of `exercise_contest_test` facts appear verbatim in `first_stage_train`. LawBench's judgment tasks (3-1, 3-3, 3-4) are also drawn from `first_stage_train`.**

**Use it for**: SFT for Chinese legal judgment prediction - charge, article and sentence from facts. Choose one split family and hold out LawBench if you train on `first_stage_train`.

**Licence**: `unknown` (`license: unknown`) [4]; the card's Licensing Information is empty [3]. The one catch: no licence.

**Shape**: 2,168,025 rows over six splits [5]; eight columns [2].

**Hold out**: `final_test` is the most separate: 223 of its 35,922 facts (0.62%) appear in `first_stage_train` [6]. `exercise_contest_test` is not held out from `first_stage_train` (56.56% overlap) and `first_stage_test` only partly (6.07%) [6]. LawBench tasks 3-1, 3-3 and 3-4 are drawn from these cases and appear in `first_stage_train` [7]; do not train on it if you report them.

**Origin**: criminal judgments published by Chinese courts (China Judgments Online), per the paper [1]. Hub API at the check date: `downloads` 1,178, `downloadsAllTime` 62,667, `likes` 31 [4].

**Trained-on-by**: the Hub's dataset tag lists one model [8].

**Introduced by**: [1] (Xiao et al.).

## Shape

| split | rows |
| --- | --- |
| `exercise_contest_train` | 154,592 |
| `exercise_contest_valid` | 17,131 |
| `exercise_contest_test` | 32,508 |
| `first_stage_train` | 1,710,856 |
| `first_stage_test` | 217,016 |
| `final_test` | 35,922 |
| total | 2,168,025 |

One config, `default` [2]:

| column | dtype |
| --- | --- |
| `fact` | string |
| `relevant_articles` | list<int32> |
| `accusation` | list<string> |
| `punish_of_money` | float32 |
| `criminals` | list<string> |
| `death_penalty` | bool |
| `imprisonment` | float32 |
| `life_imprisonment` | bool |

Measured, 49.53% of `exercise_contest_train` facts are also in `first_stage_train`, and 2.57% of `exercise_contest_test` facts are in `exercise_contest_train` [6].

## Quality

- Row 0 of `exercise_contest_test` and row 0 of `first_stage_train` are the same case (the "颜某" phone-theft case), and the same fact text is LawBench 3-1's first item [9][10].
- Defendants' names are partly masked ("颜某", "张某某") [9].

## Load it

Pick one split family and pin the revision:

```python
import datasets

REV = "775098da3ba75f033781f8061900b62503e9bea0"  # main at the check date
train = datasets.load_dataset("china-ai-law-challenge/cail2018", revision=REV, split="first_stage_train")  # 1,710,856 rows
test = datasets.load_dataset("china-ai-law-challenge/cail2018", revision=REV, split="final_test")          # 35,922 rows - hold out
```

**Trap**: split names suggest a train/test design, but `exercise_contest_test` shares 18,388 facts with `first_stage_train` [6]. Evaluating on `exercise_contest_test` after training on `first_stage_train` reports memorisation. Use `final_test`, and deduplicate it against your training facts anyway.

## Neighbors

- `doolayer/LawBench` - its tasks 3-1, 3-3 and 3-4 are CAIL facts.
- `ShengbinYue/DISC-Law-SFT` - its judgment-prediction rows are CAIL-style cases.

## A row

From `config="default"`, `split="exercise_contest_valid"`, `row_idx=0` (datasets-server `/first-rows`) [9]:

```json
{
  "fact": "公诉机关起诉指控，被告人张某某秘密窃取他人财物，价值2210元，××数额较大，其行为已触犯《中华人民共和国刑法》××之规定，应当以××罪追究其刑事责任。建议判处被告人张某某××以下刑罚，并处罚金。",
  "relevant_articles": [
    264
  ],
  "accusation": [
    "盗窃"
  ],
  "punish_of_money": 0.0,
  "criminals": [
    "张某某"
  ],
  "death_penalty": false,
  "imprisonment": 2.0,
  "life_imprisonment": false
}
```

## Where it came from

Released for the 2018 Chinese AI and Law Challenge by Chaojun Xiao and co-authors from Tsinghua and partners [1]; added to the Hub by JetRunner [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Xiao et al., "CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction", arXiv:1807.02478, 2018. https://arxiv.org/abs/1807.02478 - current title read from the live abs page. Fetched 2026-09-23.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=china-ai-law-challenge%2Fcail2018 - column schema; live, no revision parameter. Fetched 2026-09-23.

[3] china-ai-law-challenge/cail2018 dataset card (README). https://huggingface.co/datasets/china-ai-law-challenge/cail2018/raw/main/README.md. Fetched 2026-09-23.

[4] Hugging Face Hub API record for china-ai-law-challenge/cail2018. https://huggingface.co/api/datasets/china-ai-law-challenge/cail2018?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=china-ai-law-challenge%2Fcail2018 - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] This skill's own exact split-leakage count at the pinned revision - `fact` with whitespace and punctuation removed. Fetched 2026-09-23.

[7] This skill's own check by n-gram containment against the benchmark's test items (word 8-grams, or character 15-grams for Chinese), with self, words-sorted and positive controls, at the pinned revision. Run 2026-09-23.

[8] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:china-ai-law-challenge/cail2018&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[9] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=china-ai-law-challenge%2Fcail2018&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[10] datasets-server first-rows endpoint for doolayer/LawBench, config `3-1`, split `test`. https://datasets-server.huggingface.co/first-rows?dataset=doolayer%2FLawBench&config=3-1&split=test. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable for Chinese judgment-prediction SFT with `final_test` as the held-out set. Its splits overlap, it has no licence, and its training data contains LawBench's judgment tasks.

### The screening row

The row's own note: "CAIL2018; overlapping splits; LawBench 3-x source." The row carries the flag `split-leakage`.
