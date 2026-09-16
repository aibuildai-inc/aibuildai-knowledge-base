# openlifescienceai/medmcqa

193,155 four-option multiple-choice medical exam questions, most paired with an expert-written explanation, spanning 21 medical subjects.

**openlifescienceai/medmcqa** re-hosts MedMCQA, introduced by Pal, Umapathi and Sankarasubbu in "MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering" [1], as a Hugging Face Hub parquet dataset with a `train`/`test`/`validation` split matching the paper. The questions are drawn from India's AIIMS and NEET postgraduate medical entrance exams; each row carries a question, four options, the correct option index, and (for most rows) an expert explanation of the answer [2]. It lives at https://huggingface.co/datasets/openlifescienceai/medmcqa . **The `test` split's answer key is withheld: every one of the 100 `test` rows sampled at offset 0 here carries a masked correct-option value and an empty explanation, and the MEDITRON paper independently states that MedMCQA does not release test answer keys to the public [3]. Treat `test` as unusable for training or self-scored evaluation, and hold out `validation` too if the target task is any medical-MCQA benchmark, since MEDITRON reports scoring against `validation` in place of the withheld `test` [3].**

**Use it for**: multiple-choice QA SFT, or reasoning-trace SFT on the subset of rows whose `exp` field is non-empty (the correct-option text plus its expert explanation as the training target). Not preference data - `cop` is a single gold label per row, not a pair of ranked responses. See the SFT method card. Building a training example requires joining `cop` (a 0-indexed class label over `opa`-`opd`) to the corresponding option text yourself; the served schema has no column that already holds the correct option's text [4].

**Licence**: Apache-2.0 per the Hub repo's `cardData.license` tag [5], but the README's own Licensing Information section reads "[Needs More Information]" [2] - the card body never states a license. The GitHub code repository for the paper carries a separate MIT `LICENSE.md` for its scripts [6], which is not a license for this dataset.

**Shape**: 193,155 rows in one config (`default`), split `train` 182,822 / `test` 6,150 / `validation` 4,183, eleven columns [7][4].

**Hold out**: `test` (6,150 rows, correct-option label masked to -1 and explanation empty in every sampled row - see above) [3]; the screening row for this dataset likewise names `test` as the MedMCQA benchmark to hold out [8]. Also hold out `validation` (4,183 rows) if targeting a medical-MCQA benchmark score, since MEDITRON reports scoring against `validation` in place of the withheld `test` [3].

**Origin**: built by Ankit Pal, Logesh Kumar Umapathi and Malaikannan Sankarasubbu, who collected the raw historical AIIMS/NEET PG exam questions from open websites and books; the README states the dataset carries no additional annotation on top of that collected material [2]. Hub API at the check date: `downloads` 65,524, `downloadsAllTime` 554,266, `likes` 232 [5].

**Trained-on-by**: the origin paper's own baselines (BERT-base, BioBERT, SciBERT, PubMedBERT) were fine-tuned on this `train` split [1]. MEDITRON-7B/70B fine-tune on the MedMCQA training set for supervised evaluation, and - because MMLU ships no training data - also fine-tune on MedMCQA's four-option training data and evaluate the resulting model's generalization to MMLU-Medical; after removing rows whose explanation is "None", 159,669 training rows remain in their setup [3].

**Introduced by**: [1] (Pal et al.).

## Shape

Rows served and splits (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 182,822 |
| `test` | 6,150 |
| `validation` | 4,183 |
| total | 193,155 |

One config, `default`, with eleven columns (datasets-server `/info`) [4]:

| column | dtype |
| --- | --- |
| `id` | string |
| `question` | string |
| `opa` | string |
| `opb` | string |
| `opc` | string |
| `opd` | string |
| `cop` | class_label (0=a, 1=b, 2=c, 3=d) |
| `choice_type` | string ("single" or "multi") |
| `exp` | string |
| `subject_name` | string |
| `topic_name` | string |

Sizes (datasets-server `/size`) [7]: 88,311,487 bytes of parquet download, 136,323,996 bytes decoded in memory.

The origin paper's Table 2 gives per-split token statistics for its own train/test/dev counts (182,822 / 6,150 / 4,183, matching this release) [1]:

| | Train | Test | Dev | Total |
| --- | --- | --- | --- | --- |
| Vocab | 94,231 | 11,218 | 10,800 | 97,694 |
| Max Q tokens | 220 | 135 | 88 | 220 |
| Max A tokens | 38 | 21 | 25 | 38 |
| Max E tokens | 3,155 | 651 | 695 | 3,155 |
| Avg Q tokens | 12.77 | 9.93 | 14.09 | 12.71 |
| Avg A tokens | 2.69 | 2.58 | 3.19 | 2.70 |
| Avg E tokens | 67.52 | 46.54 | 38.44 | 66.22 |

Here "Dev" is the paper's name for the `validation` split, and "E" is the explanation field (`exp`) [1].

## Quality

- The dataset's own paper reports that the split boundaries are exam-based, not question-based: `train` is built from mock/online test series, `test` from AIIMS PG exam questions (1991-present), and `validation` from NEET PG exam questions (2001-present) [1].
- To limit leakage between splits, the paper computed the Levenshtein distance between every pair of questions in the whole dataset and excluded a question from `test`/`validation` whenever its similarity to another question exceeded 0.9 [1].
- All 100 `test` rows sampled at offset 0 from the served dataset carry `cop=-1` (the ClassLabel missing-value code) and an empty `exp` string; none of the 100 sampled `validation` rows at offset 0 showed a masked `cop` [9]. This matches the MEDITRON paper's statement that MedMCQA withholds its test answer keys from the public [3].
- Of the 100 `train` rows sampled at offset 0, 11 have an empty or "None" `exp` field [9]; MEDITRON's own preprocessing, which drops rows with a "None" explanation from a self-reported 187k-row training set, keeps 159,669 rows [3] - both figures put the empty-explanation rate in `train` in roughly the same 10-15% range.
- Of the 100 `validation` rows sampled at offset 0, 36 have an empty or "None" `exp` field [9] - a higher rate than `train`'s 11/100, so a reasoning-trace SFT run that needs `exp` should not expect `validation` to fill in for `train`.
- The origin paper's own baseline results show the task is hard and domain pretraining matters: with a PubMed-abstract context, PubMedBERT reaches 47% test accuracy versus 33% for BERT-base with no context, and the paper states this is far below the roughly 90% average score of human exam candidates [1].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-01-04) [5]:

```python
import datasets

REV = "91c6572c454088bf71b679ad90aa8dffcd0d5868"  # main at the check date
train = datasets.load_dataset("openlifescienceai/medmcqa", revision=REV, split="train")           # 182,822 rows
validation = datasets.load_dataset("openlifescienceai/medmcqa", revision=REV, split="validation")  # 4,183 rows - hold out for benchmark scoring
test = datasets.load_dataset("openlifescienceai/medmcqa", revision=REV, split="test")               # 6,150 rows - unlabeled, hold out
```

**Trap**: `cop` is a 0-indexed `ClassLabel` (0=a, 1=b, 2=c, 3=d), not the option text - decode it and join to `opa`-`opd` yourself before building a training target [7][4]. On `test`, `cop` is uniformly -1 (missing) and `exp` is empty in every sampled row, so loading `test` and training on it, or scoring against it locally, will silently produce meaningless labels rather than an error [9].

## Neighbors

- `openlifescienceai/medmcqa_formatted` - same builder org, same `train` (182,822) and `validation` (4,183) row counts, no `test` split served. A fetched row (`id` `e9ad821a-c438-4965-9f77-760819dfa155`, identical to this release's `train` row 0) shows the reformatting: `cop`/`opa`-`opd` are replaced by a nested `data` struct with `Question`, `Options` (A-D), `Correct Option` (a letter) and `Correct Answer` (the option text spelled out), and `exp` is renamed `explanation` [10]. Prefer this release for the raw `cop` index and the withheld-`test` shape; prefer `medmcqa_formatted` when a pre-joined letter-and-text answer is wanted and `test` is not needed.
- `lighteval/med_mcqa` - a different org's copy with the same 193,155 rows across the same three splits and the same eleven columns [11], built for use with the `lighteval` evaluation harness rather than as a training source [12].
- No source found evidence of a preference-pair or DPO-formatted re-release of MedMCQA.

## A row

Two distinct served shapes: `train`/`validation` rows carry a real correct-option label and (usually) an explanation; `test` rows do not. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [9]:

```json
{
  "id": "e9ad821a-c438-4965-9f77-760819dfa155",
  "question": "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma",
  "opa": "Hyperplasia",
  "opb": "Hyperophy",
  "opc": "Atrophy",
  "opd": "Dyplasia",
  "cop": 2,
  "choice_type": "single",
  "exp": "Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is used to describe dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine Refer Robbins 7yh/9,1012,9/e. P950",
  "subject_name": "Anatomy",
  "topic_name": "Urinary tract"
}
```

From `config="default"`, `split="test"`, `row_idx=0` [9], showing the masked label and empty explanation:

```json
{
  "id": "84f328d3-fca4-422d-8fb2-19d55eb31503",
  "question": "Which of the following is derived from fibroblast cells ?",
  "opa": "TGF-13",
  "opb": "MMP2",
  "opc": "Collagen",
  "opd": "Angiopoietin",
  "cop": -1,
  "choice_type": "single",
  "exp": "",
  "subject_name": "Pathology",
  "topic_name": null
}
```

## Where it came from

Built by Ankit Pal, Logesh Kumar Umapathi and Malaikannan Sankarasubbu from historical AIIMS and NEET PG entrance exam questions (1991-present), collected from open websites and books [1][2]. The README states the dataset carries no additional annotation on top of that collected material [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Pal, Umapathi, Sankarasubbu, "MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering", Proceedings of the Conference on Health, Inference, and Learning, 2022. https://arxiv.org/abs/2203.14371 - origin paper; current title read from the live abs page, body read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2203.14371). Fetched 2026-08-11.

[2] openlifescienceai/medmcqa dataset card (README). https://huggingface.co/datasets/openlifescienceai/medmcqa/raw/main/README.md - dataset summary, data instance example, field descriptions, split construction, source-data and curator statements, Licensing Information section. Fetched 2026-08-11.

[3] Chen et al., "MEDITRON-70B: Scaling Medical Pretraining for Large Language Models", 2023. https://arxiv.org/abs/2311.16079 - fine-tunes on the MedMCQA training set, states MedMCQA's test answer keys are withheld from the public, and gives the post-filtering training row count. Read via the PDF (https://arxiv.org/pdf/2311.16079), current title read from the live abs page. Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=openlifescienceai%2Fmedmcqa Fetched 2026-08-11.

[5] Hugging Face Hub API record for openlifescienceai/medmcqa. https://huggingface.co/api/datasets/openlifescienceai/medmcqa?full=true - license tag, gate status, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] LICENSE.md of the medmcqa/medmcqa GitHub repository (the paper's code repository, linked from the README). https://api.github.com/repos/medmcqa/medmcqa/license - MIT license text, applying to the code repository, not the Hub dataset. Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=openlifescienceai%2Fmedmcqa Fetched 2026-08-11.

[8] The corpus screening row for `openlifescienceai/medmcqa`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[9] datasets-server first-rows endpoint, `train`, `test`, and `validation` splits of `config=default`. https://datasets-server.huggingface.co/first-rows?dataset=openlifescienceai%2Fmedmcqa&config=default&split=train (and `split=test`, `split=validation`) Fetched 2026-08-11.

[10] openlifescienceai/medmcqa_formatted: dataset card (https://huggingface.co/datasets/openlifescienceai/medmcqa_formatted/raw/main/README.md) for its schema and split sizes, size endpoint (https://datasets-server.huggingface.co/size?dataset=openlifescienceai%2Fmedmcqa_formatted) for row counts, and first-rows endpoint (https://datasets-server.huggingface.co/first-rows?dataset=openlifescienceai%2Fmedmcqa_formatted&config=default&split=train) for the compared row. Fetched 2026-08-11.

[11] datasets-server size endpoint for lighteval/med_mcqa. https://datasets-server.huggingface.co/size?dataset=lighteval%2Fmed_mcqa - this endpoint takes no revision parameter, so this count is live, not pinned. Fetched 2026-08-11.

[12] lighteval/med_mcqa dataset card (README). https://huggingface.co/datasets/lighteval/med_mcqa/raw/main/README.md - states its purpose as a `lighteval`-format copy of MedMCQA. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as multiple-choice QA / reasoning-trace SFT data on `train`, with `test` and `validation` both held out from training - `test` because it is confirmed unusable (masked labels, empty explanations) and `validation` because MEDITRON reports scoring against it in place of the withheld `test` [3][9]. This refines the screening row's own note, which named only `test` as the split to hold out [8]; the additional `validation` recommendation and the masked-`test` finding both come from fetching the served rows and the MEDITRON paper, not from the screening row.

### The screening row

The row's own note [8]: "AIIMS/NEET-PG entrance exam MCQs with expert explanations; train (~182k) is safe, `test` is the MedMCQA benchmark." The row carries no flag.
