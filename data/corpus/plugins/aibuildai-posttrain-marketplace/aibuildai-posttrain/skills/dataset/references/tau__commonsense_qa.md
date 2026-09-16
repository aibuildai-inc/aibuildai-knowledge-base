# tau/commonsense_qa

12,102 crowdsourced five-choice commonsense questions, each built from a ConceptNet source concept and its distractor target concepts, with a single letter answer key.

**tau/commonsense_qa** (https://huggingface.co/datasets/tau/commonsense_qa) is the Hugging Face re-release of CommonsenseQA, introduced in "CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge" [1]: crowd-workers were shown a source concept and several ConceptNet target concepts sharing a relation to it, and asked to author a question whose correct answer is one target concept while the other targets (plus a crowd-authored and a ConceptNet distractor) become the four wrong choices, yielding five-way multiple-choice questions that require background knowledge beyond the question text [1]. Each row is a `question`, its `question_concept`, five `choices` (label + text), and an `answerKey` letter [2]. **The `test` split carries no gold `answerKey`: a check of all 1,140 rows, read as the first and last 100 rows of the split, found every `answerKey` empty - it cannot be used for supervised training or scoring - and `validation` is the split the paper reports accuracy on, so hold it out of training [1][3].**

**Use it for**: multiple-choice commonsense QA SFT - format each row as a prompt containing the question and its five lettered choices, with the completion being the correct choice (by label or text), on `train` only; hold out `validation` as the reported benchmark and do not train on `test` since it has no labels. Maps to the SFT method card's multiple-choice-QA input shape.

**Licence**: MIT (`license:mit` tag, and the card states "The dataset is licensed under the MIT License" pointing to a GitHub issue as the source of that determination), ungated (`gated: false`) [2][4]. No further catch stated.

**Shape**: one config (`default`), three splits - `train` 9,741 / `validation` 1,221 / `test` 1,140 = 12,102 rows total, five string/struct columns [2][5].

**Hold out**: `validation` (1,221 rows) - the paper's own reported-accuracy split [1]. `test` (1,140 rows) is unusable for supervised training or scoring regardless, since a check of the first 100 and last 100 rows of the split found every `answerKey` empty [3].

**Origin**: built by the Tel Aviv University / AI2 authors of [1] from ConceptNet source/target concepts, with human crowd-workers authoring the questions and one distractor each [1]. Hub API at the check date: `downloads` 73,572, `likes` 152 [4].

**Trained-on-by**: the Hub's own "models trained or fine-tuned on this dataset" listing reports 57 models, naming e.g. `danlou/roberta-large-finetuned-csqa` and `sileod/deberta-v3-base-tasksource-nli` [6]. The origin paper's own baselines were fine-tuned on it: BERT-large and GPT [1].

**Introduced by**: [1] (Talmor et al., NAACL 2019).

## Shape

Splits and rows (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 9,741 |
| `validation` | 1,221 |
| `test` | 1,140 |
| total | 12,102 |

One config, `default`, five columns (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `id` | string |
| `question` | string |
| `question_concept` | string |
| `choices` | list\<struct\<label: string, text: string\>\> |
| `answerKey` | string |

Sizes (datasets-server `/size`) [5]: 1,558,570 bytes of original Parquet download, 2,740,150 bytes decoded in memory. No source states sequence-length or token statistics for this release.

## Quality

- The paper reports its own baselines' accuracy on the (paper-internal) test set: BERT-large 55.9% and GPT 45.5% on the random split, against 88.9% human accuracy from a majority vote of five workers per question, sampled over 100 random questions and not involved in question generation [1] - the gap the paper uses to argue the task requires commonsense knowledge beyond surface pattern matching.
- No source states a measured contamination or duplicate rate for this Hub release; none is invented here.
- The paper states the random split (used by this release, per its README "Random split" description) is harder for trained models than the alternative question-concept split, because the same `question_concept` can appear in both train and validation/test with a different correct answer, which defeats memorization [1].
- All 1,140 `test` rows have an empty `answerKey` string: the first 100 rows (offset 0) and the last 100 rows (offset 1,040) were both checked and every value was empty [3]; the same check on the first row of `train` and `validation` found non-empty, single-letter `answerKey` values [3]. This is read from the served rows themselves, not stated by the README.

## Load it

Train on `train`, hold out `validation`, do not train on `test`, and pin the revision this card's numbers were read at:

```python
import datasets

REV = "94630fe30dad47192a8546eb75f094926d47e155"
train = datasets.load_dataset("tau/commonsense_qa", revision=REV, split="train")           # 9,741 rows
validation = datasets.load_dataset("tau/commonsense_qa", revision=REV, split="validation")  # 1,221 rows - hold out
test = datasets.load_dataset("tau/commonsense_qa", revision=REV, split="test")              # 1,140 rows - no answerKey
```

**Trap**: `test` loads without error and has the same five columns as `train`/`validation`, but its `answerKey` field is an empty string on every row checked [3] - a training or eval loop that reads `answerKey` blindly will silently train or score against blank labels instead of raising.

## Neighbors

- `ChilleD/CommonsenseQA` - the same 9,741/1,221/1,140 split sizes as this release, with one added `question_concept`-derived `question_concat` string column; its `dataset_info` block otherwise matches this release's features [8][9]. Treat it as a reformatted copy of the same corpus, not an independent source - training on both duplicates rows.
- `tasksource/commonsense_qa_2.0` - a different, harder successor dataset from a separate paper, "CommonsenseQA 2.0: Exposing the limits of AI through gamification" [10], licensed CC-BY-4.0, with 9,264 train / 2,541 validation rows and ten columns, no `test` split served [8][10]. Not a re-release of this dataset - a distinct benchmark built by adversarial gamification rather than ConceptNet-seeded question authoring.
- Several single-purpose derivatives exist on the Hub (translations such as `Thanmay/commonsense_qa-translated`, and reformattings such as `liujqian/commonsenseqa_with_content_words`); none is a preferred replacement for this release, reach for one only when its specific transform (translation, retrieval augmentation) is what is needed [8].
- This corpus prefers `tau/commonsense_qa` itself for the original English random-split rows.

## A row

The repository serves one config and one schema across all three splits, so the shapes below cover it in full. From `config="default"`, `split="train"`, `row_idx=0` [3]:

```json
{
  "id": "075e483d21c29a511267ef62bedc0461",
  "question": "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?",
  "question_concept": "punishing",
  "choices": {
    "label": ["A", "B", "C", "D", "E"],
    "text": ["ignore", "enforce", "authoritarian", "yell at", "avoid"]
  },
  "answerKey": "A"
}
```

From `config="default"`, `split="test"`, `row_idx=0`, showing the empty `answerKey` [3]:

```json
{
  "id": "90b30172e645ff91f7171a048582eb8b",
  "question": "The townhouse was a hard sell for the realtor, it was right next to a high rise what?",
  "question_concept": "townhouse",
  "choices": {
    "label": ["A", "B", "C", "D", "E"],
    "text": ["suburban development", "apartment building", "bus stop", "michigan", "suburbs"]
  },
  "answerKey": ""
}
```

## Where it came from

Built by the authors of [1] (Talmor, Herzig, Lourie, Berant), who sampled a source concept and several target concepts sharing a semantic relation to it from ConceptNet, then had crowd-workers author, for each target concept, a question whose correct answer is that target while the other targets serve as two of the four distractors; each worker additionally chose one ConceptNet distractor and authored one distractor themselves [1]. The Hub repository packages this as the `default` config with the random 80/10/10 train/validation/test split the paper calls its primary split, because it is harder for models to exploit than the alternative question-concept split [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Talmor, Herzig, Lourie, Berant, "CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge", NAACL 2019. https://arxiv.org/abs/1811.00937 - origin paper; abstract, split methodology, and Table 5 baseline/human accuracy figures read from the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/1811.00937). Fetched 2026-08-11.

[2] tau/commonsense_qa dataset card (README). https://huggingface.co/datasets/tau/commonsense_qa/raw/main/README.md - column descriptions, split table, licensing statement. Fetched 2026-08-11.

[3] datasets-server first-rows endpoint, config `default`, one call per split (https://datasets-server.huggingface.co/first-rows?dataset=tau%2Fcommonsense_qa&config=default&split=train, and `&split=validation`, `&split=test`), plus the rows endpoint at offset 1,040 for the last 100 `test` rows (https://datasets-server.huggingface.co/rows?dataset=tau%2Fcommonsense_qa&config=default&split=test&offset=1040&length=100) - sample rows and the full-split `test` answerKey check. Fetched 2026-08-11.

[4] Hugging Face Hub API record for tau/commonsense_qa. https://huggingface.co/api/datasets/tau/commonsense_qa?full=true - licence tag, gate status, `sha`, `downloads`, `likes`. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=tau%2Fcommonsense_qa Fetched 2026-08-11.

[6] tau/commonsense_qa dataset page, "Models trained or fine-tuned on this dataset" panel. https://huggingface.co/datasets/tau/commonsense_qa - lists 57 models and names the first two. Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=tau%2Fcommonsense_qa Fetched 2026-08-11.

[8] Hugging Face Hub dataset search for "commonsense_qa". https://huggingface.co/api/datasets?search=commonsense_qa&limit=30 - neighbor discovery list. Fetched 2026-08-11.

[9] ChilleD/CommonsenseQA dataset card and size. https://huggingface.co/datasets/ChilleD/CommonsenseQA/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=ChilleD%2FCommonsenseQA - column list and split row counts. Fetched 2026-08-11.

[10] tasksource/commonsense_qa_2.0 dataset card and size. https://huggingface.co/datasets/tasksource/commonsense_qa_2.0/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=tasksource%2Fcommonsense_qa_2.0 - licence, citation to the CommonsenseQA 2.0 paper, split row counts and column count. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for SFT-style multiple-choice QA on `train`, with `validation` held out as the reported benchmark and `test` unusable for training since it carries no labels. Both facts are established above from the paper's split description [1] and from the empty `answerKey` observed on every sampled `test` row [3], matching the screening row's own note.

### The screening row

The row's own note [screening row for `tau/commonsense_qa`, checked 2026-08-11]: "crowdsourced commonsense MCQ; `train` is safe, `validation` is the reported benchmark, `test` is unlabeled." The row carries no flag.
