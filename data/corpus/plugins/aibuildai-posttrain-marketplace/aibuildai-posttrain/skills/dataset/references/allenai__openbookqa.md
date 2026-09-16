# allenai/openbookqa

5,957 crowdsourced elementary-science multiple-choice questions paired with a small 1,326-fact "open book," released in two column shapes under one repository.

**allenai/openbookqa** hosts OpenBookQA, introduced by Mihaylov, Clark, Khot and Sabharwal in "Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering" [1]: four-way multiple-choice science questions modeled on open-book exams, each requiring a core science fact plus an additional common-knowledge fact to answer, and originally released with a separate open-book fact set for retrieval-style QA [1]. It lives at https://huggingface.co/datasets/allenai/openbookqa .

**The `test` split (500 rows in each config) is the public OpenBookQA benchmark set and its answer keys are served openly, so it must be held out of training; `train` (4,957 rows) and `validation` (500 rows) are safe [1][2].**

**Use it for**: SFT-style multiple-choice QA training - question stem plus four labeled choices, target the `answerKey` letter - or as a reasoning/fact-grounded QA task; not a preference-pair or chat-turn dataset. See the SFT method card.

**Licence**: `unknown` per the repository's own `license` field and its `license:unknown` tag [2]; the paper's model-training code on GitHub is Apache-2.0, but that covers the code repository, not this data repository, and no source states a licence for the questions or fact list themselves [3].

**Shape**: two configs, `main` (id, question_stem, choices, answerKey) and `additional` (adds fact1, humanScore, clarity, turkIdAnonymized); each has `train` 4,957 / `validation` 500 / `test` 500, 5,957 rows per config, 11,914 rows served total [2][4].

**Hold out**: `test` in both configs (500 + 500 = 1,000 rows) - it is the standard OpenBookQA evaluation benchmark, and the served rows carry visible `answerKey` labels, not masked ones [1][5].

**Origin**: built by the Allen Institute for AI; questions were authored by Amazon Mechanical Turk crowd-workers from a "masters"-qualification pool and partially filtered by experts, over a fact set also assembled by the authors [1]. Hub API record: 161,213 downloads, 135 likes as of the check date [2].

**Trained-on-by**: AI2's UnifiedQA fine-tunes T5 on a pool of 17 QA datasets that explicitly includes OpenBookQA's training data among its multiple-choice sources [6]. No source found training a chat-oriented instruction model on this dataset specifically for SFT; it is more commonly cited as a held-out evaluation benchmark (e.g., by Llama-family and other LLM technical reports) than as training data.

**Introduced by**: [1] (Mihaylov et al., EMNLP 2018).

## Shape

Rows and splits, both configs (datasets-server `/size`) [4]:

| config | train | validation | test | total |
| --- | --- | --- | --- | --- |
| `main` | 4,957 | 500 | 500 | 5,957 |
| `additional` | 4,957 | 500 | 500 | 5,957 |

Columns per config (datasets-server `/info` and the repository's own `dataset_info`) [2][5]:

| config | columns |
| --- | --- |
| `main` | id (string), question_stem (string), choices (list<struct<text: string, label: string>>), answerKey (string) |
| `additional` | main's four columns plus fact1 (string), humanScore (float32), clarity (float32), turkIdAnonymized (string) |

`main` and `additional` carry the same 5,957 underlying questions in the same row order - row 0 of `main/train` and `additional/train` share `id: "7-980"` and an identical question, choices and answer key - so `additional` is `main` with four extra per-question annotation fields, joinable on `id` [7]. The default config on load with no `config_name` argument is `main` [2]. Original download size is 1,393,402 bytes across both configs combined; in-memory size is 2,728,838 bytes [4]. No source states token or sequence-length statistics for the Hub release; the origin paper does report them for its own release of the same question set: 1.08 average question sentences, 11.46 average question tokens, 2.89 average choice tokens, and 9.38 average tokens in the associated core science fact [1].

## Quality

- The paper reports the underlying fact-writing and question-authoring stages produced far more raw material than shipped: crowd-workers collected 8,140 candidate questions, of which 2,183 were discarded during crowdsourced and expert filtering, leaving the 5,957 questions in this release [1].
- Human accuracy on the full question set is reported at close to 92%, versus a 25% random-guess baseline for four-way multiple choice [1].
- The paper's own bias probes show the dataset is gameable to a degree without reading the question or facts: a "plausible answer detector" that ignores the question text scores 49.6% accuracy, and an "odd-one-out" solver that only compares answer choices reaches 50.2%, both well above the 25% random baseline [1]. A "question match" solver using only the question and choices (no external knowledge) also reaches 50.2%, while giving a model the gold science fact plus the question author's chosen additional fact ("f + k" in the paper's oracle setting) raises the same architecture's test score to about 76-80%, still short of the 92% human ceiling [1].
- The `additional` config's `humanScore` and `clarity` fields are per-question crowd-annotation quality scores from that same collection process, at the row level, but no source states aggregate distributions for them [1][5].
- No source states a measured duplicate rate or cross-dataset contamination rate for this release; none is invented here.

## Load it

Pick a config explicitly; `main` is the lightweight QA shape, `additional` carries the fact and annotation columns. Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-01-04) [2]:

```python
import datasets

REV = "388097ea7776314e93a529163e0fea805b8a6454"  # main branch at the check date
train = datasets.load_dataset("allenai/openbookqa", "additional", revision=REV, split="train")        # 4,957 rows
val   = datasets.load_dataset("allenai/openbookqa", "additional", revision=REV, split="validation")    # 500 rows
test  = datasets.load_dataset("allenai/openbookqa", "additional", revision=REV, split="test")          # 500 rows - hold out
```

**Trap**: `load_dataset("allenai/openbookqa")` with no config name silently loads `main`, which drops `fact1`, `humanScore`, `clarity` and `turkIdAnonymized` - if a training recipe needs the gold fact for grounding or wants to filter by `humanScore`/`clarity`, the config name must be `additional`, not the default [2][5].

## Neighbors

- `allenai/ai2_arc` - the AI2 Reasoning Challenge, a sibling elementary/middle-school science MCQ benchmark from an overlapping AllenAI author group, released in `ARC-Easy` and `ARC-Challenge` configs totaling 7,787 rows across train/validation/test; a distinct question set from OpenBookQA, not a duplicate or superset of it [8].
- Several community translations exist on the Hub, none from AllenAI: `projecte-aina/openbookqa_ca` (Catalan), `BSC-LT/openbookqa-es` (Spanish), `hishab/openbookqa-bn` (Bengali), `yuri-no/openbookqa-ita` (Italian), and others found by a Hub search for "openbookqa" [9]. Use one of these only if the target language is not English; this corpus prefers the original `allenai/openbookqa` for English training.
- No cleaned, binarized, or official AllenAI successor release of OpenBookQA itself was found beyond the two configs already in this repository.

## A row

`main` and `additional` share the same underlying question set but differ in columns, so one row from each config shows the two served shapes. Both are row 0 of `train` (datasets-server `/first-rows`) [7].

`config="main"`, `split="train"`:
```json
{
  "id": "7-980",
  "question_stem": "The sun is responsible for",
  "choices": {
    "text": ["puppies learning new tricks", "children growing up and getting old", "flowers wilting in a vase", "plants sprouting, blooming and wilting"],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "D"
}
```

`config="additional"`, `split="train"`:
```json
{
  "id": "7-980",
  "question_stem": "The sun is responsible for",
  "choices": {
    "text": ["puppies learning new tricks", "children growing up and getting old", "flowers wilting in a vase", "plants sprouting, blooming and wilting"],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "D",
  "fact1": "the sun is the source of energy for physical cycles on Earth",
  "humanScore": 1.0,
  "clarity": 2.0,
  "turkIdAnonymized": "b356d338b7"
}
```

`test` rows carry the same shape with a populated `answerKey`, e.g. row 0 of `config="main"`, `split="test"` has `id: "8-343"`, `answerKey: "B"` [10] - confirming the benchmark's answers are served openly, not masked.

## Where it came from

Built by the Allen Institute for AI. The authors first assembled a "book" of 1,326 core elementary-science facts, then ran a multi-stage Amazon Mechanical Turk pipeline (workers required a "masters" qualification): a crowd-worker was shown a core fact, asked to supply a second common-knowledge fact needed to combine with it, then wrote a question and four answer choices exercising that combination; an automatic filter removed questions solvable by retrieval or word-association methods, and a further crowd/expert filtering stage removed low-quality items, cutting 8,140 raw candidates to the 5,957 questions released [1]. The `additional` config's `humanScore`, `clarity`, and `turkIdAnonymized` fields are metadata captured during that same crowdsourcing and filtering process [1][5].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Mihaylov, Clark, Khot, Sabharwal, "Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering", EMNLP 2018. https://arxiv.org/abs/1809.02789 - origin paper; question/fact counts, filtering pipeline, human/random/oracle baseline scores, statistics table. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/1809.02789). Fetched 2026-08-11.

[2] Hugging Face Hub API record for allenai/openbookqa. https://huggingface.co/api/datasets/allenai/openbookqa?full=true - `sha`, `license`, `cardData.dataset_info`, `configs` (default config), `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[3] GitHub repository allenai/OpenBookQA (model/experiment code for the paper). https://github.com/allenai/OpenBookQA - read via the GitHub API license field. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=allenai%2Fopenbookqa Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=allenai%2Fopenbookqa - per-config feature schema. Fetched 2026-08-11.

[6] Khashabi et al., "UnifiedQA: Crossing Format Boundaries With a Single QA System", 2020. https://arxiv.org/abs/2005.00700 - lists OpenBookQA among the multiple-choice datasets used to train UnifiedQA. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2005.00700). Fetched 2026-08-11.

[7] datasets-server first-rows endpoint, `train` split, both configs. https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Fopenbookqa&config=main&split=train and https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Fopenbookqa&config=additional&split=train Fetched 2026-08-11.

[8] datasets-server size endpoint for allenai/ai2_arc. https://datasets-server.huggingface.co/size?dataset=allenai%2Fai2_arc - this endpoint takes no revision parameter, so this row count is live, not pinned. Fetched 2026-08-11.

[9] Hugging Face Hub dataset search for "openbookqa". https://huggingface.co/api/datasets?search=openbookqa&limit=20 - live listing of repositories matching the search term; a live endpoint, not pinned. Fetched 2026-08-11.

[10] datasets-server first-rows endpoint, `main` config, `test` split. https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Fopenbookqa&config=main&split=test Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for MCQ/SFT-style training on `train` and `validation`; `test` must be held out because it is the public OpenBookQA benchmark with its answer keys served openly, matching the screening row's own note [1][5].

### The screening row

The row's own note: "crowdsourced elementary science MCQ over a small fact book; `train`+`validation` are safe, `test` is the OpenBookQA benchmark." The row carries no flag.
