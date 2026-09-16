# derek-thomas/ScienceQA

21,208 K-12 science multiple-choice questions, each with an answer and a paired lecture/solution explanation, and roughly half carrying a contextual image - a Parquet re-release of the ScienceQA benchmark.

**derek-thomas/ScienceQA** (https://huggingface.co/datasets/derek-thomas/ScienceQA) packages the ScienceQA benchmark introduced in "Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering" [1] as Hub Parquet, with the same three splits and thirteen columns as the paper's release. The questions were extracted from IXL Learning's K-12 science curricula problems, aligned to California Common Core Content Standards, and each item carries a multiple-choice question, optional image and hint, and a lecture plus solution written as the answer's chain-of-thought explanation [2]. **The dataset's own body text states it is "available for non-commercial research purposes only" and gives its licence as CC BY-NC-SA 4.0 [2], even though the repository's machine-readable licence tag says `cc-by-sa-4.0`** [3] **- treat it as non-commercial until that conflict is resolved. It is also only partly multimodal: of the first 100 rows read at offset 0 in each split, an image was present in 43/100 (train), 49/100 (validation) and 48/100 (test) [4], so a text-only model can only use the no-image rows.**

**Use it for**: reasoning-trace SFT - question (+ optional hint/image) in, and a lecture-plus-solution rationale followed by the answer out, matching the paper's own chain-of-thought setup [1]. It is not chat-formatted (no dialogue/turn columns) and has no eval YAML on the repo, so a load-time reformat is needed before training; map it to the SFT method card. Hold out `test`, and if training a text-only model, filter to rows with no `image`.

**Licence**: repository tag `cc-by-sa-4.0` [3], but the card body's "Licensing Information" section names Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0), and its "Annotations" section separately states the data is for non-commercial research use only [2] - the two licence statements on the same page disagree, and the non-commercial one is the more specific, later-written claim.

**Shape**: 21,208 rows, one config (`default`), three splits: `train` 12,726 / `validation` 4,241 / `test` 4,241 [5].

**Hold out**: `test` (4,241 rows) - the screening row's note identifies it as the ScienceQA benchmark test split [6], the same split the origin paper and downstream work report scores against [1][7][8].

**Origin**: questions sourced from IXL Learning's K-12 science curriculum problems and built by Lu et al. (UCLA, Arizona State, Allen Institute for AI); components (question, hint, image, choices, answer, lecture, solution) were extracted by heuristic rule, manually filtered, and reviewed by experts - human-authored source content with human curation, no model generation [2]. Uploaded to the Hub by Derek Thomas [2]. Hub API at the check date: `downloads` 35,643, `downloadsAllTime` 307,044, `likes` 233 [3].

**Trained-on-by**: Multimodal-CoT fine-tunes a sub-1B-parameter model on ScienceQA and reports state-of-the-art accuracy on the ScienceQA benchmark [7]. LLaVA is fine-tuned on ScienceQA and, combined with GPT-4, reaches 92.53% accuracy on it [8].

**Introduced by**: [1] (Lu et al., NeurIPS 2022).

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 12,726 |
| `validation` | 4,241 |
| `test` | 4,241 |
| total | 21,208 |

One config, `default`, with thirteen columns (datasets-server `/info`, and the repository's own `dataset_info` YAML) [5][2]:

| column | dtype |
| --- | --- |
| `image` | Image |
| `question` | string |
| `choices` | sequence\<string\> |
| `answer` | int8 |
| `hint` | string |
| `task` | string |
| `grade` | string |
| `subject` | string |
| `topic` | string |
| `category` | string |
| `skill` | string |
| `lecture` | string |
| `solution` | string |

Total size: 626,082,086 bytes as Parquet, 727,926,379 bytes decoded in memory [5]. No source states sequence-length or token statistics for `question`, `lecture` or `solution`; not stated. The card notes that "some records might be missing any or all of image, lecture, solution" [2], which the sample below confirms for `image` and `hint`.

## Quality

- Questions were downloaded from IXL Learning's problem bank and had their components (questions, hints, images, options, answers, lectures, solutions) extracted by heuristic rule; invalid, faulty, or duplicated questions were manually removed, answer choices were shuffled to remove positional patterns, and the builders "developed a data exploration tool to review examples in the collected dataset," with "incorrect annotations further manually revised by experts" [2].
- Reading the first 100 rows at offset 0 in each split: `hint` is non-empty in 49/100 (train), 55/100 (validation), 49/100 (test); `lecture` is non-empty in 89/100 (train), 86/100 (validation), 87/100 (test); `solution` is non-empty in 86/100 (train), 85/100 (validation), 90/100 (test) [4]. These are counts over the first 100 rows of each split only and do not describe the full splits.
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure; none is invented here.

## Load it

Train on `train`, hold out `test`, and pin the revision this card's numbers were read at:

```python
import datasets

REV = "f18b0a70359ebfb41f658fd564208d0355b013f4"
train = datasets.load_dataset("derek-thomas/ScienceQA", revision=REV, split="train")            # 12,726 rows
validation = datasets.load_dataset("derek-thomas/ScienceQA", revision=REV, split="validation")    # 4,241 rows
test = datasets.load_dataset("derek-thomas/ScienceQA", revision=REV, split="test")                 # 4,241 rows - hold out, it is the benchmark
```

**Trap**: `image` decodes to a PIL image object or `None` per row - almost half the rows in every split carry no image (see the flag above), so a naive collator that always calls into the image column will fail on those rows; filter or branch on `image is None` first. The declared licence tag (`cc-by-sa-4.0`) also disagrees with the card body's non-commercial licensing text, so do not rely on the tag alone for licence clearance [2][3].

## Neighbors

- `tasksource/ScienceQA_text_only` - 10,876 rows (`train` 6,508 / `validation` 2,144 / `test` 2,224), the same non-image columns with the `image` column dropped entirely; its own card describes it as "ScienceQA text-only examples (examples where no image was initially present, which means they should be doable with text-only models)" [9]. Its citation block names a different paper (Saikh et al., "ScienceQA: A Novel Resource for Question Answering on Scholarly Articles") that is not this dataset's origin paper - the row counts and column names match a text-only filter of this release, not that paper's data, so treat the citation in that card as mislabeled rather than following it.
- `lmms-lab/ScienceQA-IMG` - 10,332 rows (`train` 6,218 / `validation` 2,097 / `test` 2,017); its card states it is "a formatted and filtered version of derek-thomas/ScienceQA with only image instances," built for the `lmms-eval` evaluation pipeline [10]. Prefer this one only for one-click multimodal-model evaluation against the image-only subset, not for training.
- `armanc/ScienceQA` - 94,286 rows; a namesake collision, not a re-release of this dataset. Its card identifies it as "the ScientificQA dataset by Saikh et al (2022)," a scholarly-article QA dataset unrelated to the K-12 curriculum content here [11]; do not substitute it for this dataset.
- `HuggingFaceM4/ScienceQA` - an older (2022-12-08), loading-script-based Hub repo with 275 downloads at the check date [12]; its dataset viewer is disabled because the repo runs a custom Python loader, so its row shapes were not read here. Its low adoption relative to this Parquet release (35,643 downloads) suggests this repository is the one downstream work uses.

This corpus prefers this release over the filtered subsets above because it is the complete, unfiltered three-split release the origin paper and both downstream training papers report against [1][7][8].

## A row

Two shapes are served side by side in every split: rows with an image and rows without. From `config="default"`, `split="train"` (datasets-server `/first-rows`) [4]:

Row 0, image-bearing:

```json
{
  "image": "<PIL image, not shown>",
  "question": "Which of these states is farthest north?",
  "choices": ["West Virginia", "Louisiana", "Arizona", "Oklahoma"],
  "answer": 0,
  "hint": "",
  "task": "closed choice",
  "grade": "grade2",
  "subject": "social science",
  "topic": "geography",
  "category": "Geography",
  "skill": "Read a map: cardinal directions",
  "lecture": "Maps have four cardinal directions, or main directions. Those directions are north, south, east, and west.\nA compass rose is a set of arrows that point to the cardinal directions. [...]",
  "solution": "To find the answer, look at the compass rose. Look at which way the north arrow is pointing. West Virginia is farthest north."
}
```

Row 3, text-only (`image` is `None`):

```json
{
  "image": null,
  "question": "Which tense does the sentence use?\nMona will print her name with care.",
  "choices": ["present tense", "future tense", "past tense"],
  "answer": 1,
  "hint": "",
  "task": "closed choice",
  "grade": "grade2",
  "subject": "language science",
  "topic": "verbs",
  "category": "Verb tense",
  "skill": "Is the sentence in the past, present, or future tense?",
  "lecture": "Present tense verbs tell you about something that is happening now.\nMost present-tense verbs are regular. They have no ending, or they end in -s or -es. [...]",
  "solution": "The sentence is in future tense. You can tell because it uses will before the main verb, print. The verb tells you about something that is going to happen."
}
```

A collator written for the text-only shape (row 3) works unchanged on the image-bearing shape (row 0) if it simply ignores or drops the `image` field; the reverse only works if it branches on `image is None`.

## Where it came from

Built by Lu, Mishra, Xia, Qiu, Chang, Zhu, Tafjord, Clark and Kalyan (UCLA, Arizona State University, Allen Institute for AI) [2]. The questions come from open resources managed by IXL Learning, an online K-12 learning platform, aligned to California Common Core Content Standards; the builders downloaded the original problems and extracted question, hint, image, choices, answer, lecture and solution components with heuristic rules, manually removed invalid, faulty or duplicated questions, shuffled answer-choice order, and manually revised incorrect annotations found by their own review tool [2]. The Hub Parquet repository was contributed by Derek Thomas [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Lu et al., "Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering", NeurIPS 2022. https://arxiv.org/abs/2209.09513 - origin paper; abstract states ~21k multimodal MCQs, and the deciding numbers: chain-of-thought improves few-shot GPT-3 by 1.20% and fine-tuned UnifiedQA by 3.99%, feeding explanations as input improves few-shot GPT-3 by 18.96%, and models reach equal performance with 40% of the data when given explanations. Fetched 2026-08-11.

[2] derek-thomas/ScienceQA dataset card (README). https://huggingface.co/datasets/derek-thomas/ScienceQA/raw/main/README.md - dataset_info schema and split sizes, sample instance, data-field descriptions, curation/annotation process, licensing information, curators, contributor credit. Fetched 2026-08-11.

[3] Hugging Face Hub API record for derek-thomas/ScienceQA. https://huggingface.co/api/datasets/derek-thomas/ScienceQA?full=true (and the same endpoint with `expand[]=downloadsAllTime`) - `sha`, `license` tag, `gated`, `private`, `downloads`, `downloadsAllTime`, `likes`, `lastModified`. Fetched 2026-08-11.

[4] datasets-server first-rows endpoint, one call per split. https://datasets-server.huggingface.co/first-rows?dataset=derek-thomas%2FScienceQA&config=default&split=train (and `split=validation`, `split=test`) - 100 rows read at offset 0 in each split; sample rows and image/hint/lecture/solution presence counts above. Fetched 2026-08-11.

[5] datasets-server size and info endpoints. https://datasets-server.huggingface.co/size?dataset=derek-thomas%2FScienceQA and https://datasets-server.huggingface.co/info?dataset=derek-thomas%2FScienceQA Fetched 2026-08-11.

[6] The corpus screening row for `derek-thomas/ScienceQA`, supplied with this card's request - its `note` and `flag`, read back in the appendix. Checked 2026-08-11.

[7] Zhang et al., "Multimodal Chain-of-Thought Reasoning in Language Models", 2023. https://arxiv.org/abs/2302.00923 - abstract states experiments on the ScienceQA benchmark and a sub-1B-parameter model reaching state-of-the-art accuracy on it. Fetched 2026-08-11.

[8] Liu et al., "Visual Instruction Tuning", 2023. https://arxiv.org/abs/2304.08485 - abstract states the model (LLaVA) is fine-tuned on ScienceQA and, combined with GPT-4, reaches 92.53% accuracy. Fetched 2026-08-11.

[9] tasksource/ScienceQA_text_only dataset card and size endpoint. https://huggingface.co/datasets/tasksource/ScienceQA_text_only/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=tasksource%2FScienceQA_text_only Fetched 2026-08-11.

[10] lmms-lab/ScienceQA-IMG dataset card and size endpoint. https://huggingface.co/datasets/lmms-lab/ScienceQA-IMG/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=lmms-lab%2FScienceQA-IMG Fetched 2026-08-11.

[11] armanc/ScienceQA dataset card and size endpoint. https://huggingface.co/datasets/armanc/ScienceQA/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=armanc%2FScienceQA Fetched 2026-08-11.

[12] Hugging Face Hub API record for HuggingFaceM4/ScienceQA. https://huggingface.co/api/datasets/HuggingFaceM4/ScienceQA?full=true - `lastModified`, `downloads`, repository file list (loading-script based, viewer disabled). Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable, with two conditions already established above: treat the licence as non-commercial despite the `cc-by-sa-4.0` tag [2][3], and hold out `test` because it is the ScienceQA benchmark split [6]. The dataset is also only partly multimodal, so a text-only model can train only on the no-image rows [4][6].

### The screening row

The row's own note [6]: "expert/textbook science exam MCQ with images; `train`+`validation` are safe, `test` is the ScienceQA benchmark." Its flag [6]: "partly multimodal — many items carry an image, so a text-only model can use only the text subset."
