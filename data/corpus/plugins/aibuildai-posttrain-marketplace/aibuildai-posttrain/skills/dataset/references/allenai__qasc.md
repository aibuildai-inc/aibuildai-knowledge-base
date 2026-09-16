# allenai/qasc

9,980 crowdsourced 8-way multiple-choice science questions, each answer requiring the composition of two retrieved facts, split 8,134 train / 926 validation / 920 test.

**allenai/qasc** is the Hugging Face release of QASC (Question Answering via Sentence Composition), introduced in "QASC: A Dataset for Question Answering via Sentence Composition" [1]: grade-school-science multiple-choice questions built so that answering requires retrieving two supporting facts from a companion corpus of roughly 17 million sentences and composing them [2]. It lives at https://huggingface.co/datasets/allenai/qasc . **The `test` split carries no answer key or supporting facts: `answerKey`, `fact1`, `fact2`, and `combinedfact` are empty strings in every one of the 100 rows sampled here, so `test` is the blind QASC benchmark split and must be held out, not trained on [3].**

**Use it for**: reasoning-trace SFT on `train`/`validation` - each row already carries the two supporting facts (`fact1`, `fact2`, `combinedfact`) plus the question and gold `answerKey`, so a chain-of-thought style target can be built as "fact1 + fact2 -> combinedfact -> answerKey" without extra annotation; maps to a plain instruction/response SFT format built from `formatted_question` as the prompt and the composed-fact-plus-answer as the target. See the SFT method card. Do not use `test` for training or contamination-free evaluation prep without recovering labels from the original benchmark, since none are served here.

**Licence**: CC BY 4.0 (`cardData.license` is `["cc-by-4.0"]`, repo tag `license:cc-by-4.0`), ungated (`"gated": false`, `"private": false`) [4]. No further licence catch stated beyond CC BY 4.0 attribution.

**Shape**: 9,980 rows in one config (`default`), splits `train` 8,134 / `validation` 926 / `test` 920, eight columns [5][6].

**Hold out**: `test` (920 rows) — its `answerKey`, `fact1`, `fact2`, and `combinedfact` fields are all empty in every sampled row, so it cannot supply a training signal and is also unusable as a self-contained held-out eval without the original labels [3].

**Origin**: built and released by the Allen Institute for AI (allenai); questions and supporting-fact annotations were crowdsourced from the source corpus, not model-generated [1][2]. Hub API at the check date: `downloads` 25,006, `downloadsAllTime` 617,536, `likes` 23 [4].

**Trained-on-by**: none found. UnifiedQA evaluates its pre-trained model on QASC as one of twelve "unseen" generalization datasets, but its own 8 seed datasets used for training are SQuAD1.1, SQuAD2.0, NarrativeQA, RACE, ARC, OBQA, MCTest, and BoolQ - QASC is not among them [7].

**Introduced by**: [1] (Khot et al.).

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows | parquet bytes | in-memory bytes |
| --- | --- | --- | --- |
| `train` | 8,134 | 1,967,904 | 4,870,614 |
| `test` | 920 | 158,241 | 390,534 |
| `validation` | 926 | 223,553 | 559,180 |
| total | 9,980 | 2,349,698 | 5,820,328 |

One config, `default`, with eight columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `id` | string |
| `question` | string |
| `choices` | sequence of struct {text: string, label: string} |
| `answerKey` | string |
| `fact1` | string |
| `fact2` | string |
| `combinedfact` | string |
| `formatted_question` | string |

No source states sequence-length or token statistics for this release.

## Quality

- The dataset card states each question is 8-way multiple choice over grade-school science and ships alongside "a corpus of 17M sentences" that the paper's task uses for fact retrieval [2].
- The `test` split's answer and fact fields are withheld: reading 100 of its 920 rows (the full first-rows page, offset 0) found `answerKey`, `fact1`, `fact2`, and `combinedfact` empty in all 100, while `id`, `question`, `choices`, and `formatted_question` are populated [3]. This is consistent with `test` being the original QASC leaderboard benchmark's blind split; no source states a partial or per-row exception within the sampled 100.
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure; none is invented here.
- The dataset card's Curation Rationale, Annotation process, and bias/limitations sections are all marked "More Information Needed" in the current README [2].

## Load it

Train on `train`, evaluate on `validation`, hold out `test`, and pin the revision this card's numbers were read at:

```python
import datasets

REV = "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"  # main at the check date
train = datasets.load_dataset("allenai/qasc", revision=REV, split="train")            # 8,134 rows
validation = datasets.load_dataset("allenai/qasc", revision=REV, split="validation")  # 926 rows
test = datasets.load_dataset("allenai/qasc", revision=REV, split="test")              # 920 rows - hold out, no labels
```

**Trap**: `test` loads without error and returns 920 well-formed rows with a real `question` and `choices`, but `answerKey`, `fact1`, `fact2`, and `combinedfact` are empty strings, not missing keys - a script that checks only for column presence will not notice the labels are gone [3].

## Neighbors

None found. No sibling, cleaned, binarized, or successor release of QASC was located among the sources fetched for this card.

## A row

The repository serves one config and one schema across all three splits, but `train`/`validation` and `test` differ in which fields are populated, so both shapes are shown.

From `config="default"`, `split="train"`, `row_idx=0` [8]:

```json
{
  "id": "3E7TUJ2EGCLQNOV1WEAJ2NN9ROPD9K",
  "question": "What type of water formation is formed by clouds?",
  "choices": {
    "text": ["pearls", "streams", "shells", "diamonds", "rain", "beads", "cooled", "liquid"],
    "label": ["A", "B", "C", "D", "E", "F", "G", "H"]
  },
  "answerKey": "F",
  "fact1": "beads of water are formed by water vapor condensing",
  "fact2": "Clouds are made of water vapor.",
  "combinedfact": "Beads of water can be formed by clouds.",
  "formatted_question": "What type of water formation is formed by clouds? (A) pearls (B) streams (C) shells (D) diamonds (E) rain (F) beads (G) cooled (H) liquid"
}
```

From `config="default"`, `split="test"`, `row_idx=0` [3]:

```json
{
  "id": "3C44YUNSI1OBFBB8D36GODNOZN9DPA",
  "question": "What type of birth do therian mammals have?",
  "choices": "...populated, same shape as train...",
  "answerKey": "",
  "fact1": "",
  "fact2": "",
  "combinedfact": "",
  "formatted_question": "...populated, same shape as train..."
}
```

## Where it came from

Built by the Allen Institute for AI (allenai) and introduced in "QASC: A Dataset for Question Answering via Sentence Composition" [1]. The dataset card states the questions are crowdsourced and the language is "found" rather than machine-generated [2]. The paper describes QASC as requiring retrieval of two supporting facts from a companion corpus and their composition to answer each multiple-choice question, and reports that the paper's own two-step retrieval-plus-reasoning approach improves over the contemporary state of the art by 11 absolute points while still trailing human performance by 20 points [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Khot, Clark, Guerquin, Jansen, Sabharwal, "QASC: A Dataset for Question Answering via Sentence Composition", 2020 (arXiv:1910.11473). https://arxiv.org/abs/1910.11473 - the origin paper; current title and abstract read from the live abs page. Fetched 2026-08-11.

[2] allenai/qasc dataset card (README). https://huggingface.co/datasets/allenai/qasc/raw/main/README.md - dataset summary, corpus size, licence statement, "More Information Needed" sections. Fetched 2026-08-11.

[3] datasets-server first-rows endpoint, split `test`. https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Fqasc&config=default&split=test - 100 rows read at offset 0, all with empty `answerKey`/`fact1`/`fact2`/`combinedfact`. Fetched 2026-08-11.

[4] Hugging Face Hub API record for allenai/qasc. https://huggingface.co/api/datasets/allenai/qasc?full=true - licence, gate, `sha`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=allenai%2Fqasc Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=allenai%2Fqasc Fetched 2026-08-11.

[7] Khashabi et al., "UnifiedQA: Crossing Format Boundaries With a Single QA System", 2020 (arXiv:2005.00700). Read as the ar5iv HTML full text (https://ar5iv.labs.arxiv.org/html/2005.00700), which names the 8 seed training datasets and places QASC among the 12 unseen generalization datasets, noting it has 8 candidate answers per question. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint, split `train`. https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Fqasc&config=default&split=train Fetched 2026-08-11.

[9] The corpus screening row for `allenai/qasc`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for reasoning-trace SFT on `train` and `validation`; `test` must be held out because its answer and supporting-fact fields are empty, matching the screening row's own note that `test` is the QASC benchmark split [3][9].

### The screening row

The row's own note [9]: "crowdsourced science MCQ built by composing two facts; `train`+`validation` are safe, `test` is the QASC benchmark." The row carries no flag.
