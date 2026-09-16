# curaihealth/medical_questions_pairs

3,048 medical question pairs, each pair labeled similar or different, hand-generated and labeled by doctors at Curai Health - a single train split, no chat turns, no free text beyond the two questions.

**curaihealth/medical_questions_pairs** is Curai Health's "MQP" dataset, released alongside "Effective Transfer Learning for Identifying Similar Questions: Matching User Questions to COVID-19 FAQs" [1]. Doctors were given 1,524 patient-asked questions sampled from a public HealthTap crawl and, for each one, wrote one paraphrase preserving the original intent (the similar pair) and one related-but-wrong-answer question (the dissimilar pair), yielding 3,048 rows split evenly between the two labels [2]. The task shape is pairwise binary text classification - semantic question-similarity - not a chat or preference format. It lives at https://huggingface.co/datasets/curaihealth/medical_questions_pairs . **No license is stated anywhere the pipeline could check: the Hub `license` tag and `cardData.license` both read "unknown," the README's own Licensing Information section is empty, and the linked GitHub mirror carries no detected license either [2][3][4].**

**Use it for**: sentence-pair semantic-similarity classification - feed `question_1`/`question_2` with the binary `label` to a cross-encoder-style classifier, or reformat the pair and label into an instruction/answer example for the text-classification SFT method card. Not a preference-pair or chat-SFT format: there is no chosen/rejected structure and no dialogue turns.

**Licence**: unknown (`license:unknown` tag, `cardData.license: ["unknown"]`); ungated (`"gated": false`, `"private": false`) [3]. The one catch: nothing in the README, the Hub metadata, or the linked GitHub repository states any actual license terms, so treat use as unlicensed by default [2][3][4].

**Shape**: 3,048 rows, one config (`default`), one split (`train`); four columns (`dr_id`, `question_1`, `question_2`, `label`) [3][5][6].

**Hold out**: nothing in any source fetched for this card flags this dataset as overlapping a known evaluation benchmark. The dataset ships only the one `train` split (3,048 rows); the README itself says it "consists of only one split(train) but can be split seperately based on the requirement" [2], so a held-out check is left to whoever trains on it.

**Origin**: built by Curai Health's doctors from HealthTap-sourced patient questions; both the paraphrase/dissimilar-question authoring and the similar/different label are human (doctor) judgments, not model output [2]. Hub API at the check date: `downloads` 1,743 (30-day) / 245,339 (all-time), `likes` 50 [3][7].

**Trained-on-by**: the origin paper's own double-fine-tuned BERT classifier is trained and evaluated on this exact MQP dataset [1]. MedINST, a 133-task, 7-million-sample biomedical instruction meta-dataset, lists MQP ("McCreery et al. 2020") as one of its Semantic Similarity source tasks, and its authors report fine-tuning several LLMs on MedINST [8]. No other source fetched for this card names a further downstream model or recipe trained on this release.

**Introduced by**: [1] (McCreery et al.), which names the release "MQP" and links the same GitHub dataset repository the Hub card links [1][2].

## Shape

Rows and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 3,048 |

One config, `default`, four columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `dr_id` | int32 |
| `question_1` | string |
| `question_2` | string |
| `label` | class_label (names `0`, `1`) |

The README defines `label` as 1 when the pair is similar and 0 otherwise, and states `dr_id` ranges 1 to 11 across "11 different doctors" [2]; the 100 rows read live at offset 0 of `train` all carry `dr_id` 1, consistent with rows being grouped by doctor but not itself a check of the full 1-11 range [9].

Sizes: 313,704 bytes as the original/Parquet download, 689,911 bytes decoded in memory per the live datasets-server `/size` endpoint [5]; the README's pinned `dataset_info` block instead states `dataset_size: 701642` for the same split [2] - the two sources disagree by 11,731 bytes on the in-memory figure, and neither is corrected here. No source states sequence-length or token statistics for this dataset.

The card-declared count table in the README's Data Splits section gives the label balance directly: 1,524 "Non similar Question Pairs" and 1,524 "Similar Question Pairs" against `train` [2], matching the exact 50/50 split.

## Quality

- The label is a human (doctor) judgment applied under an explicit two-part writing instruction, not a similarity score: doctors were told to paraphrase for the similar pair and to write a related-but-wrong-answer question for the dissimilar pair, specifically so that "positive question pairs can look very different by superficial metrics, and negative question pairs can conversely look very similar," to keep the task non-trivial [2].
- The origin paper reports the resulting downstream accuracy achieved when models are fine-tuned on this data: pretraining strategies other than the paper's proposed medical-QA pretraining top out below 78.7% accuracy on this task, the proposed double fine-tuning approach reaches 82.6% with the same number of training examples (80.0% with a much smaller training set), and 84.5% when the full corpus of medical question-answer pretraining data is used [1].
- No source fetched for this card states a measured duplicate rate, inter-annotator agreement, or contamination rate for this dataset; none is invented here.
- The paper notes that, in its own experiments, the 1,524 patient-asked questions used to build this dataset were excluded from the separate HealthTap-derived pretraining tasks it also constructed, to keep those pretraining tasks and this fine-tuning dataset disjoint [1]; this describes the paper's own experimental design, not a property of the served rows.

## Load it

One config, one split; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-01-04) [3]:

```python
import datasets

REV = "d91d9da9b2a843fc09ab9d48568d2a93bf58bd0d"  # main at the check date
train = datasets.load_dataset("curaihealth/medical_questions_pairs", revision=REV, split="train")  # 3,048 rows
```

**Trap**: there is no dev/test split - `load_dataset` with `split="train"` returns the entire 3,048-row dataset, and any evaluation split has to be carved out by the caller before training.

## Neighbors

- `ChuGyouk/medical_questions_pairs_ko` - this same 3,048-row set with `question_1`/`question_2` machine-translated into Korean (`question_1_ko`, `question_2_ko`) by "solar-1-mini-translate-enko," alongside the original English columns and `label`/`dr_id`; its own card states it is sourced directly from this repository [10][11].
- `Lots-of-LoRAs/task1645_medical_question_pair_dataset_text_classification` - a Super-NaturalInstructions reformatting of this dataset into `input`/`output`/`id` text-generation examples, split train/valid/test 2,358/295/295 (2,948 rows total, 100 fewer than this release's 3,048) [10][12]. Licensed Apache-2.0 by the reformatter, unlike this release's unknown license [12].
- No other Hub search on this dataset's name or its GitHub repository name surfaced a further re-release; "none found" beyond the two above.

Prefer this original release when you need the doctor-written English pairs and their exact 3,048/1,524/1,524 counts; use the Korean neighbor only if you need a Korean-language training set, and the Natural-Instructions neighbor only if you specifically need its instruction/output text-generation framing.

## A row

One config, one split, so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [9]:

```json
{
  "dr_id": 1,
  "question_1": "After how many hour from drinking an antibiotic can I drink alcohol?",
  "question_2": "I have a party tonight and I took my last dose of Azithromycin this morning. Can I have a few drinks?",
  "label": 1
}
```

The next row served (`row_idx=1`) shares the same `question_1` and `dr_id` but pairs it with a different, unrelated `question_2` and `label: 0`, showing the dataset's one-similar/one-dissimilar-per-question construction directly in the served rows [9].

## Where it came from

Built by Curai Health. Doctors were shown a list of 1,524 patient-asked questions randomly sampled from a publicly available crawl of HealthTap, a medical question-answering site; the crawl itself is credited to a separate GitHub dataset, not authored by Curai Health [2]. For each source question, a doctor wrote one paraphrase that keeps the original intent (allowed to change surface wording and non-essential medical details) and one related question whose answer would be wrong or irrelevant for the original - the two instructions that generate the similar and dissimilar labeled pairs respectively [1][2]. The origin paper frames this as MQP, the fine-tuning target for a "double fine-tuning" approach that first pretrains a BERT model on medical question-answer pairs before fine-tuning on this question-question similarity data, and the paper is also the source of the GitHub repository the Hub card's Repository link points to [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] McCreery, Katariya, Kannan, Chablani, Amatriain, "Effective Transfer Learning for Identifying Similar Questions: Matching User Questions to COVID-19 FAQs," 2020. https://arxiv.org/abs/2008.13546 - the origin paper; current title read from the live abs page, full text read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2008.13546). Fetched 2026-08-11.

[2] curaihealth/medical_questions_pairs dataset card (README). https://huggingface.co/datasets/curaihealth/medical_questions_pairs/raw/main/README.md - dataset summary, construction instructions, data fields, data splits table, licensing section, GitHub repository link, citation. Fetched 2026-08-11.

[3] Hugging Face Hub API record for curaihealth/medical_questions_pairs. https://huggingface.co/api/datasets/curaihealth/medical_questions_pairs?full=true - license tag/cardData, gate status, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[4] GitHub API record for the linked dataset repository. https://api.github.com/repos/curai/medical-question-pair-dataset - `license` field, read `null` (no license detected). Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=curaihealth%2Fmedical_questions_pairs Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=curaihealth%2Fmedical_questions_pairs Fetched 2026-08-11.

[7] Hugging Face Hub API record with `downloadsAllTime` expanded. https://huggingface.co/api/datasets/curaihealth/medical_questions_pairs?expand[]=downloadsAllTime Fetched 2026-08-11.

[8] Han, Fang, Zhang, Yin, Song, Chen, Pechenizkiy, Chen, "MedINST: Meta Dataset of Biomedical Instructions," 2024. https://arxiv.org/abs/2410.13458 - lists MQP ("McCreery et al. 2020") among its Semantic Similarity source tasks and states several LLMs were fine-tuned on MedINST; current title read from the live abs page, full text read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2410.13458). Found via a Semantic Scholar citations lookup of the origin paper (https://api.semanticscholar.org/graph/v1/paper/arXiv:2008.13546/citations). Fetched 2026-08-11.

[9] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=curaihealth%2Fmedical_questions_pairs&config=default&split=train Fetched 2026-08-11.

[10] Hugging Face Hub dataset search for "medical_questions_pairs" and "medical-question-pair," used to find neighbor releases. https://huggingface.co/api/datasets?search=medical_questions_pairs and https://huggingface.co/api/datasets?search=medical-question-pair - this endpoint takes no revision parameter, so the neighbor list is live, not pinned. Fetched 2026-08-11.

[11] ChuGyouk/medical_questions_pairs_ko dataset card and datasets-server size/info endpoints. https://huggingface.co/datasets/ChuGyouk/medical_questions_pairs_ko/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=ChuGyouk%2Fmedical_questions_pairs_ko - source attribution, translation model, row/column counts. Fetched 2026-08-11.

[12] Lots-of-LoRAs/task1645_medical_question_pair_dataset_text_classification dataset card and datasets-server size endpoint. https://huggingface.co/datasets/Lots-of-LoRAs/task1645_medical_question_pair_dataset_text_classification/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=Lots-of-LoRAs%2Ftask1645_medical_question_pair_dataset_text_classification - license, feature schema, split row counts. Fetched 2026-08-11.

[13] The corpus screening row for `curaihealth/medical_questions_pairs`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as pairwise semantic-similarity classification training data, on the single `train` split as served. The dataset's own card gives no license and no held-out split, both restated above [2][3]; the screening row's note independently confirms the dataset's shape and origin [13].

### The screening row

The row's own note [13]: "3,048 similar/dissimilar medical question pairs hand-written and labelled by Curai's own doctors." The row carries no flag.
