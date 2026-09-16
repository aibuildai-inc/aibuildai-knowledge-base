# qiaojin/PubMedQA

273,518 yes/no/maybe research questions over PubMed abstracts, split into a 211,269-row auto-labelled config, a 61,249-row unlabeled config, and a 1,000-row expert-annotated config that carries PubMedQA's official evaluation set.

**qiaojin/PubMedQA** is the Hugging Face mirror of PubMedQA, introduced in "PubMedQA: A Dataset for Biomedical Research Question Answering" [1]: each instance pairs a research question, a PubMed abstract context (with the conclusion held out), a long free-text answer (the conclusion), and a yes/no/maybe label, so the task is to answer a biomedical research question from the abstract that precedes its own conclusion [1]. The repository serves three configs the paper calls PQA-A(rtificial), PQA-U(nlabeled) and PQA-L(abeled) [2][1]. It lives at https://huggingface.co/datasets/qiaojin/PubMedQA . **The 1,000-row `pqa_labeled` config is entirely expert-annotated and entirely PubMedQA's evaluation material: the paper draws its official 500-question test set and a separate 500-question cross-validation set from these same 1,000 instances, and the parquet served here carries no split or column that marks which pubids belong to which slice — it is one `train` split of 1,000 rows [1][2]. Hold out `pqa_labeled` in full; train only on `pqa_artificial` and `pqa_unlabeled`.**

**Use it for**: reasoning-trace or classification-style SFT that maps a (question, abstract) pair to a yes/no/maybe decision plus a supporting long answer — the SFT method card's format, not a preference-pair format. `pqa_artificial`'s heuristically-generated yes/no labels and `pqa_unlabeled`'s unlabeled question-context pairs are the safe training material; never train on `pqa_labeled`, which is held-out evaluation data end to end.

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated (`"gated": false`, `"private": false`) [3]. No further licence catch beyond the evaluation-data restriction above, which is a usage restriction stated by the origin paper, not a licence term [1][3].

**Shape**: 273,518 rows across three configs, each one `train` split: `pqa_artificial` 211,269, `pqa_unlabeled` 61,249, `pqa_labeled` 1,000 [4][5].

**Hold out**: `pqa_labeled` in full (1,000 rows) — the paper's official 500-question test set and its separate 500-question cross-validation set both live inside these 1,000 rows, and this repository does not expose which pubids fall in which slice [1][2].

**Origin**: built by Qiao Jin et al. and released under the `qiaojin` Hub namespace; questions are PubMed article titles (verbatim for `pqa_labeled`/`pqa_unlabeled`, heuristically rewritten from statement titles for `pqa_artificial`), and the yes/no/maybe labels are two qualified M.D. candidates' judgments for `pqa_labeled` and a negation-based heuristic for `pqa_artificial` [1]. Hub API at the check date: 41,994 recent downloads, 777,153 all-time downloads, 336 likes [3].

**Trained-on-by**: BioGPT fine-tunes on `pqa_artificial` and `pqa_unlabeled` (the paper's own PQA-A/PQA-U naming) before evaluating on the `pqa_labeled` test split, reporting 78.2% accuracy against PubMedBERT's 55.8%, BioELECTRa's 64.2%, and BioLinkBERT-large's 72.2% on the same reasoning-required test set [6].

**Introduced by**: [1] (Jin et al., EMNLP-IJCNLP 2019).

## Shape

Rows served per config (datasets-server `/size`) [4]:

| config | split | rows |
| --- | --- | --- |
| `pqa_artificial` | `train` | 211,269 |
| `pqa_unlabeled` | `train` | 61,249 |
| `pqa_labeled` | `train` | 1,000 |
| total | | 273,518 |

Columns differ by config (datasets-server `/info`) [5]:

| config | columns |
| --- | --- |
| `pqa_artificial` | `pubid` (int32), `question` (string), `context.contexts` (string list), `context.labels` (string list), `context.meshes` (string list), `long_answer` (string), `final_decision` (string) |
| `pqa_unlabeled` | same as above, minus `final_decision` |
| `pqa_labeled` | same as `pqa_artificial`, plus `context.reasoning_required_pred` (string list) and `context.reasoning_free_pred` (string list) |

The paper's Table 1 gives per-subset statistics no source restates here at the served-row level: PQA-L 1.0k QA pairs (55.2% yes / 33.8% no / 11.0% maybe), PQA-U 61.2k pairs, PQA-A 211.3k pairs (92.8% yes / 7.2% no / 0.0% maybe); average question length 14.4–16.3 words, average context length 237–239 words, average long-answer length 41–46 words across the three subsets [1]. The paper further states PQA-A's 211.3k instances split into 200k for training and 11.3k for validation, and that PQA-L's 1,000 instances split into 500 for 10-fold cross-validation and 500 for the held-out test set [1] — neither split is reproduced as a column in this repository.

## Quality

- `pqa_labeled`'s labels come from two annotators per the paper's Algorithm 1: annotator 1 sees the long answer (reasoning-free) and annotator 2 does not (reasoning-required); the paper reports single-annotator human performance of 90.40% accuracy / 84.18% macro-F1 reasoning-free and 78.00% accuracy / 72.19% macro-F1 reasoning-required on the test set, against a 55.20%-accuracy majority baseline and a 66.44%–67.66%-accuracy single-phase/multi-phase BioBERT baseline [1].
- `pqa_unlabeled`'s questions are filtered by a rule-based method the paper reports at "over 93% agreement with annotator 1" on identifying yes/no/maybe-answerable questions, not by human review [1].
- `pqa_artificial`'s yes/no labels are generated purely from the negation status of the title's verb phrase, with no human check [1].
- A schema quirk in the served `pqa_labeled` config: `context.reasoning_required_pred` and `context.reasoning_free_pred` are stored as lists of single characters rather than one string per row — reading the first five served rows shows `"yes"` rendered as `['y', 'e', 's']` and `"no"` as `['n', 'o']` in every case checked, so code that expects a scalar label must join these lists first [7].
- No source states a measured contamination or duplicate rate for the served 273,518 rows; none is invented here.

## Load it

Each config loads independently; train on `pqa_artificial` and `pqa_unlabeled`, and pin the revision this card's numbers were read at. The pin below fixes the parquet files `load_dataset` reads; the row counts and sample rows elsewhere on this card come from datasets-server, which takes no revision argument and always reflects the current `main` (see Sources preamble) [3][4][5][7]:

```python
import datasets

REV = "9001f2853fb87cab8d220904e0de81ac6973b318"  # main at the check date
artificial = datasets.load_dataset("qiaojin/PubMedQA", "pqa_artificial", revision=REV, split="train")  # 211,269 rows
unlabeled = datasets.load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", revision=REV, split="train")    # 61,249 rows
labeled = datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled", revision=REV, split="train")        # 1,000 rows - hold out in full
```

**Trap**: `pqa_labeled`'s single `train` split silently mixes the paper's 500-question cross-validation slice with its 500-question test slice — loading `split="train"` here does not give you a clean training partition of PubMedQA's expert-labeled data, it gives you the entire evaluation set [1][2]. There is no `data_dir` or split argument in this repository that separates them.

## Neighbors

- `openlifescienceai/pubmedqa` splits the same 1,000 expert-labeled instances into 450 `train` / 50 `validation` / 500 `test` rows, reformatted into a multiple-choice shape (`Question`, `Context`, `Options` A/B/C, `Correct Option`, `Correct Answer`, `Long Answer`) [8][9]. The origin paper itself states only that the 1,000 `pqa_labeled` instances split into 500 for 10-fold cross-validation and 500 for test, without stating a 450/50 per-fold breakdown [1]; the 450/50 figures come from the BioGPT paper, which reports using "the original train/validation/test split with 450, 50 and 500 respectively" [6], and from the official PubMedQA GitHub repository's split script, which divides the 500 cross-validation rows into 10 folds of 50 and builds each fold's 450-row train set from the other nine folds [10]. `openlifescienceai/pubmedqa`'s 450/50/500 split matches one such fold. A fetched `test`-split row shows the same abstract-and-conclusion content as this repository's `pqa_labeled`, keyed by a generated UUID rather than `pubid` [8][9]. This is the neighbor to use when the official 500-row test set must be isolated from the rest of `pqa_labeled`, since this repository's `pqa_labeled` does not expose that split.
- `bigbio/pubmed_qa` packages the same three PQA-A/PQA-U/PQA-L subsets under the BigBio schema; its card states the same 1k/61.2k/211.3k row counts as this repository, and its README explicitly credits the same origin paper [11]. Its viewer is disabled server-side ("the dataset viewer doesn't support this dataset because it runs arbitrary python code" [12]), so its rows were not independently fetched for this card; prefer this repository's plain-parquet configs unless BigBio's task schema is specifically needed.

## A row

Two distinct served shapes: `pqa_artificial`/`pqa_unlabeled` share one shape (no `final_decision` in `pqa_unlabeled`), and `pqa_labeled` adds the two per-annotator prediction fields. From `config="pqa_labeled"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], abstract text truncated:

```json
{
  "pubid": 21645374,
  "question": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
  "context": {
    "contexts": [
      "Programmed cell death (PCD) is the regulated death of cells within an organism. [...] it has been less studied during PCD in plants.",
      "The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. [...] and that displayed mitochondrial dynamics similar to that of non-PCD cells."
    ],
    "labels": ["BACKGROUND", "RESULTS"],
    "meshes": ["Alismataceae", "Apoptosis", "Cell Differentiation", "Mitochondria", "Plant Leaves"],
    "reasoning_required_pred": ["y", "e", "s"],
    "reasoning_free_pred": ["y", "e", "s"]
  },
  "long_answer": "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, [...] Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant.",
  "final_decision": "yes"
}
```

From `config="pqa_unlabeled"`, `split="train"`, `row_idx=0` [7], showing the missing `final_decision` field and no prediction fields:

```json
{
  "pubid": 14499029,
  "question": "Is naturopathy as effective as conventional therapy for treatment of menopausal symptoms?",
  "context": {
    "contexts": [
      "Although the use of alternative medicine in the United States is increasing, no published studies have documented the effectiveness of naturopathy for treatment of menopausal symptoms compared to women receiving conventional therapy in the clinical setting.",
      "[...]",
      "In univariate analyses, patients treated with naturopathy for menopausal symptoms reported higher monthly incomes ($1848.00 versus $853.60), [...] about as frequently as patients who were treat[ed conventionally]."
    ],
    "labels": ["BACKGROUND", "OBJECTIVE", "DESIGN", "SETTING", "PATIENTS", "MAIN OUTCOME MEASURES", "RESULTS"],
    "meshes": ["Anxiety", "Cohort Studies", "Confidence Intervals", "Estrogen Replacement Therapy", "Female", "..."]
  },
  "long_answer": "Naturopathy appears to be an effective alternative for relief of specific menopausal symptoms compared to conventional therapy."
}
```

## Where it came from

Built by Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu, and released to the Hub under the `qiaojin` namespace [1][3]. The paper collects source PubMed articles that have a question-mark title and a structured abstract with a conclusive part (denoted "pre-PQA-U"); two qualified M.D. candidates then annotate 1,000 of these instances with yes/no/maybe labels to build `pqa_labeled`, one seeing the long answer (reasoning-free) and one not (reasoning-required), and the remaining pre-PQA-U instances whose questions pass a yes/no/maybe-answerability filter become `pqa_unlabeled` [1]. `pqa_artificial` is built separately from PubMed articles whose title is a declarative statement (a specific part-of-speech pattern): the title is rewritten into a question and a yes/no label is generated from the verb's negation status [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision to the `sha` read from the Hub API [3]. That pin covers the parquet files themselves. It does not cover the datasets-server endpoints used for Shape, Quality, and A row ([4], [5], [7], and the neighbor calls [9], [12]): none of those endpoints accepts a revision parameter, so they always serve the current default-branch conversion rather than a pinned commit. At the check date the Hub API's live `sha` for `main` equals the pinned `REV` in Load it, so the two happen to agree here, but a later re-fetch of the datasets-server endpoints would reflect whatever is then current on `main`, not this pinned revision.

[1] Jin, Dhingra, Liu, Cohen, Lu, "PubMedQA: A Dataset for Biomedical Research Question Answering", EMNLP-IJCNLP 2019. https://arxiv.org/abs/1909.06146 - origin paper; collection method for PQA-A/PQA-U/PQA-L, Table 1 statistics, Table 4 human performance, Table 5 model results, the 500-test/500-cross-validation split description (this paper does not itself state a 450/50 per-fold breakdown). Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/1909.06146); current title read from the live abs page. Fetched 2026-08-11.

[2] qiaojin/PubMedQA dataset card (README). https://huggingface.co/datasets/qiaojin/PubMedQA/raw/main/README.md - config names, per-config feature schema and row counts in `dataset_info`, statement that 500 `pqa_labeled` questions are the official test set and can be found at the linked GitHub repository. Fetched 2026-08-11.

[3] Hugging Face Hub API record for qiaojin/PubMedQA. https://huggingface.co/api/datasets/qiaojin/PubMedQA?full=true and the `expand[]=downloadsAllTime` variant - licence, gate, `sha`, `downloads`, `downloadsAllTime`, `likes`, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=qiaojin%2FPubMedQA Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=qiaojin%2FPubMedQA Fetched 2026-08-11.

[6] Luo, Sun, Xia, Qin, Zhang, Poon, Liu, "BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining", 2022. https://arxiv.org/abs/2210.10341 - fine-tunes on PQA-A/PQA-U, evaluates on the PQA-L reasoning-required test split, Table 5 accuracy comparison against PubMedBERT, BioELECTRa, BioLinkBERT. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2210.10341). Fetched 2026-08-11.

[7] datasets-server first-rows endpoint, three calls (one per config). https://datasets-server.huggingface.co/first-rows?dataset=qiaojin%2FPubMedQA&config=pqa_labeled&split=train (and `config=pqa_artificial`, `config=pqa_unlabeled`) - row content and schema per config, the `reasoning_required_pred`/`reasoning_free_pred` character-list check across the first five `pqa_labeled` rows. Fetched 2026-08-11.

[8] openlifescienceai/pubmedqa dataset card (README) and its `dataset_info` YAML. https://huggingface.co/datasets/openlifescienceai/pubmedqa/raw/main/README.md - 450/50/500 train/validation/test split, multiple-choice column shape. Fetched 2026-08-11.

[9] datasets-server size and first-rows endpoints for openlifescienceai/pubmedqa. https://datasets-server.huggingface.co/size?dataset=openlifescienceai%2Fpubmedqa and https://datasets-server.huggingface.co/first-rows?dataset=openlifescienceai%2Fpubmedqa&config=default&split=test - row counts per split, one fetched test-split row. Fetched 2026-08-11.

[10] PubMedQA GitHub repository, `preprocess/split_dataset.py`. https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/preprocess/split_dataset.py - the official split script: divides the 500 `pqa_labeled` cross-validation instances into 10 folds of 50, and builds each fold's 450-row train set from the other nine folds. Fetched 2026-08-11.

[11] bigbio/pubmed_qa dataset card (README). https://huggingface.co/datasets/bigbio/pubmed_qa/raw/main/README.md - same PQA-A/PQA-U/PQA-L row counts, same origin-paper citation. Fetched 2026-08-11.

[12] datasets-server size endpoint for bigbio/pubmed_qa, returning a viewer-disabled error. https://datasets-server.huggingface.co/size?dataset=bigbio%2Fpubmed_qa Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as a training source only through `pqa_artificial` and `pqa_unlabeled`; `pqa_labeled` is evaluation-only and must be held out in full. This rests on the origin paper's own account of how `pqa_labeled`'s 1,000 rows split into the official cross-validation and test slices [1], on the absence of any column or split in this repository that marks that division [2][5], and on the screening row's own note.

### The screening row

The row's own note [screening record for `qiaojin/PubMedQA`, corpus check 2026-08-11]: "yes/no/maybe research QA over PubMed abstracts; pqa_artificial (auto-labelled from conclusions) and pqa_unlabeled are the safe configs, pqa_labeled holds the 500 expert-annotated official test questions." The row carries no flag.
