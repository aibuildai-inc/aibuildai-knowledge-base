# Muennighoff/natural-instructions

A preprocessed, per-task-file copy of Super-NaturalInstructions - 876 JSON-Lines files (757 under `train/`, 119 under `test/`) totaling roughly 4.3 GB, one file per task, each row a single instance with its task's definition, one input, and one target.

**Muennighoff/natural-instructions** repackages the Super-NaturalInstructions benchmark - 1,616 diverse NLP tasks with expert-written instructions, introduced by Wang et al. in "Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks" [1] - by fetching each task's JSON file from the benchmark's official `splits/default` train/test task lists on GitHub and flattening every (instance, output) pair into one row of `{task_name, id, definition, inputs, targets}` [2][3]. The repository's own README warns that "the same inputs may appear with different outputs", so it tells users to deduplicate by the `id` or `inputs` field before training [2]. **The `test` directory is the benchmark's official held-out evaluation split (119 tasks) used in the origin paper to score cross-task generalization; training on it would contaminate that standard benchmark, so hold it out unless the goal is specifically to evaluate against it [1][2].** It lives at https://huggingface.co/datasets/Muennighoff/natural-instructions .

**Use it for**: SFT-style instruction tuning - each row is a `(definition, inputs) -> targets` example, close to a single-turn instruction/response pair once `definition` and `inputs` are concatenated into a prompt, as the downstream tokenization example in the Neighbors section does; the SFT method card is the fit. **Hold out `test`** (the 119-task benchmark split) unless deliberately evaluating against Super-NaturalInstructions. The data format is plain flat JSON fields, not a chat-templated format - the reader builds the prompt from `definition` and `inputs`.

**Licence**: not stated on this repository - the Hub API's `cardData` carries no `license` key, and the README text contains no license statement; a case-insensitive scan of the README does turn up the substring "MIT", but it comes only from the task names `task1517_limit_classfication` and `task1518_limit_answer_generation` ("...**limit**_clas...", "...**limit**_answ..."), not from a license clause - there is no actual MIT mention [4]. The upstream GitHub benchmark this repo repackages states a different license split: "All the data here (except the instances of each task) are released under Apache-2.0 license. The instances of each tasks are subject to the license under which the original dataset was released" [3] - and that per-instance license field is not carried into this repo's rows, whose schema is only `task_name, id, definition, inputs, targets` [2]. The repo itself is ungated [4].

**Shape**: 757 train task files + 119 test task files = 876 JSON-Lines files, ~4.29 GB of raw file bytes on the Hub tree; exact row totals are not obtainable from this repo directly (see Shape body).

**Hold out**: the `test` directory (119 task files) - it is the benchmark's official held-out evaluation split from the same GitHub source this repo repackages [1][3]. Exact row counts for `train` and `test` are not stated by this repo (viewer 500, see Shape body); a downstream tokenized derivative built by `load_dataset("Muennighoff/natural-instructions")` and mapped 1:1 (no filtering) reports 6,164,188 train rows and 982,285 test rows, which is the best available approximation of this repo's own row counts [5].

**Origin**: repackaged by Niklas Muennighoff from the AllenAI-built Super-NaturalInstructions benchmark; the repository's own Hub metadata tags its `annotations_creators` as `crowdsourced` and `expert-generated` [4]. Hub API at the check date (2026-08-12): `downloads` 6,834, `likes` 85 [4].

**Trained-on-by**: `togethercomputer/RedPajama-INCITE-Instruct-3B-v1`, which lists it among its training datasets [6]; the `jondurbin/bagel-*` model family (e.g. `bagel-dpo-34b-v0.2`), whose card lists it by name and describes it as "Millions of instructions from 1600+ task categories (sampled down substantially, stratified by task type)" [7]; and `wordcab/llama-natural-instructions-13b`, a LoRA fine-tune whose model card tags it as trained on this dataset [8].

**Introduced by**: no paper for this specific repackaging - the dataset card [2]; the underlying benchmark it repackages was introduced by Wang et al. [1].

## Shape

The repository has no dataset-loading script and no `dataset_infos.json`; the `datasets` library infers splits from the two top-level directories, `train/` (757 files) and `test/` (119 files) [9]. Per-file byte sizes summed from the Hub tree API give 3,795,260,826 bytes under `train/` and 495,121,793 bytes under `test/`, 4,290,382,619 bytes total across 876 files [9]. The datasets-server `/info` and `/size` endpoints both return HTTP 500 for this repository at every retry made for this card, so no server-computed row count, column dtype table, or Parquet byte size is available [10][11]; the Hub API's own `size_categories` tag self-declares the bucket `100M<n<1B` (rows, in the standard Hub convention), but that is a builder-chosen tag, not a computed count [4].

The README lists the task names in each of the "Train", "Validation", and "Test" splits [2]. The 757-name train list and 119-file `train/` directory line up (a filename check confirms the directory holds exactly the tasks named), but the README's "Test Tasks" block names 148 tasks while only 119 test files actually exist in the repository tree - checked directly against the full recursive listing, all 119 served test files are named in that list, but 29 of the 148 named test tasks have no file, e.g. `task104_semeval_2019_task10_closed_vocabulary_mathematical_answer_generation` and `task1336_peixian_equity_evaluation_corpus_gender_classifier` [2][9]. The three "Validation Tasks" the README names (`task1333_check_validity_date_ddmmyyyy`, `task1403_check_validity_date_mmddyyyy`, `task291_semeval_2020_task4_commonsense_validation`) are not a separate directory in this repository - all three are also named in the Train Tasks list, and there is no `validation/` folder in the tree - so `load_dataset` on this repo as it stands today exposes only `train` and `test` [2][9].

No source states sequence-length or token statistics for this repackaged release; none is invented here.

## Quality

- The README's only stated data-quality caveat is that "the same inputs may appear with different outputs" across rows within a task, because a `get_ni.py` preprocessing script (bundled in the repo) writes one output row per `(instance, output)` pair rather than per instance - the script's `get_all_prompted_examples_ni` function iterates `task["Instances"]` and, within each, `example["output"]`, appending one row per output [2][12]. The card tells users to deduplicate on `id` or `inputs` to avoid this [2].
- The per-instance license metadata that the upstream GitHub benchmark keeps for each task (an "Instance License" field per the upstream README) is dropped by the preprocessing script - this repo's schema (`task_name, id, definition, inputs, targets`) carries no license field [2][3][12].
- No source states a measured contamination rate, duplicate rate, or inter-annotator agreement figure for this specific repackaged release; none is invented here.

## Load it

```python
import datasets

REV = "a29a9757125f4bb1c26445ad0d2ef7d9b2cc9c4c"  # main at the check date
train = datasets.load_dataset("Muennighoff/natural-instructions", revision=REV, split="train")
test = datasets.load_dataset("Muennighoff/natural-instructions", revision=REV, split="test")  # hold out
```

**Trap**: this repo exposes only `train` and `test` splits today - a script written against an earlier revision that requested `data["validation"]` (one such script, from a downstream derivative, is quoted in Neighbors below) will fail, because no `validation/` directory exists in the current tree [2][9]. Row-level counts cannot be checked before the download completes, because the datasets-server `/info` and `/size` endpoints both return HTTP 500 for this repository [10][11] - budget for a ~4.3 GB download based on the summed file sizes in Shape, not a served row count.

## Neighbors

- `chainyo/natural-instructions-tokenized` - built by `load_dataset("Muennighoff/natural-instructions")` and then tokenizing with a LLaMA tokenizer, per the Python script embedded in its own card; its `dataset_info` YAML states 6,164,188 train rows, 982,285 test rows, and, notably, 5,995 rows in a `validation` split its script reads as `data["validation"]` - a split this repository does not currently expose as a directory, so either the repository changed after that derivative was built or the two do not fully align on split structure [5]. Its row counts are the best fetched approximation of this repository's own train/test totals, cited here as a derivative's stated figures, not confirmed against this repo directly.
- `jayelm/natural-instructions` - its own card states it is "modified from" this repository, adding positive/negative examples and explanations per task, plus an `eval` boolean field marking a balanced held-out subset (100 rows per test task = 11,900 rows; 15 rows per train task = 11,355 rows) for in-domain evaluation [13].
- `wisenut-nlp-team/natural-instructions` - a per-task-config reshaping (`instruction`, `input`, `output`, `domain`, `lang` columns) of Super-NaturalInstructions data rather than this repo's flat per-line files; its own card gives no explicit statement that it derives from this repository [14].
- No neighbor was found that clearly supersedes this repository as the deduplicated or canonical Hub copy of Super-NaturalInstructions.

## A row

Both `train/` and `test/` files share one schema, confirmed by fetching the first line of one file from each directory directly (`train/task022_cosmosqa_passage_inappropriate_binary_train.jsonl` and `test/task020_mctaco_span_based_question_test.jsonl`), since the datasets-server first-rows endpoint is unavailable for this repo [10]:

```json
{
  "task_name": "task022_cosmosqa_passage_inappropriate_binary",
  "id": "task022-ba906f03a1ed4f9dae56dae7e5742031",
  "definition": "Read the given context and if the the context is inappropriate (e.g., pornographic) or nonsensical (e.g., cannot determine what happenings the context is about), indicate via \"yes\". Otherwise, response via \"no\".",
  "inputs": "Context: As I was crawling back into bed I realized I did n't feel like puking . The morning sickness has n't been bad , but it 's there . This morning I laid there for a few minutes waiting for the familiar wave of nausea to hit me . When it did n't I woke up Chris and preceded to get a little scared about it .",
  "targets": "No."
}
```

```json
{
  "task_name": "task020_mctaco_span_based_question",
  "id": "task020-3b643de54d474381854c2499cd185a74",
  "definition": "The answer will be 'yes' if the provided sentence contains an explicit mention that answers the given question. Otherwise, the answer should be 'no'. Instances where the answer is implied from the sentence using \"instinct\" or \"common sense\" (as opposed to being written explicitly in the sentence) should be labeled as 'no'.",
  "inputs": "Sentence: Jerry goes out to the pier and casts his favorite bait : cheese . \nQuestion: How much time did Jerry spend at the pier?",
  "targets": "No."
}
```

## Where it came from

Built by Niklas Muennighoff from Super-NaturalInstructions, the AllenAI-led benchmark of 1,616 tasks and expert-written instructions across 76 task types introduced by Wang et al. [1]. The bundled `get_ni.py` script reads the benchmark's official `splits/default/train_tasks.txt` and `test_tasks.txt` task-name lists from a pinned commit of the `allenai/natural-instructions` GitHub repository, downloads each named task's JSON file from that same commit, and for every instance writes one output row per `(instance, output)` pair with fields `task_name, id, definition, inputs, targets` - dropping every other field the upstream task JSON carries, including the per-instance license field the upstream repository documents [2][3][12]. The repository's own Hub metadata tags its `annotations_creators` as `crowdsourced` and `expert-generated`; the origin paper describes the benchmark's 1,616 tasks and 76 task types but its fetched abstract does not itself state where task instances are drawn from [1][4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable, which is why Load it pins the revision.

[1] Wang et al., "Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks", 2022. https://arxiv.org/abs/2204.07705 - the origin paper for the underlying benchmark; current title and abstract read from the live abs page. Fetched 2026-08-12.

[2] Muennighoff/natural-instructions dataset card (README). https://huggingface.co/datasets/Muennighoff/natural-instructions/raw/main/README.md - description, dedup warning, train/validation/test task-name lists. Fetched 2026-08-12.

[3] allenai/natural-instructions GitHub repository README and LICENSE, at the commit pinned by `get_ni.py` (`6174af63465999768fbc09f5dd8a7f1a5dfe9abc`). https://raw.githubusercontent.com/allenai/natural-instructions/master/README.md and https://raw.githubusercontent.com/allenai/natural-instructions/master/LICENSE - license split between code (Apache-2.0) and per-instance data license; read from the `master` branch as the pinned commit's raw files were not separately re-fetched, so this reflects the current `master` text, cited through the current upstream README rather than the exact historical commit. Fetched 2026-08-12.

[4] Hugging Face Hub API record for Muennighoff/natural-instructions. https://huggingface.co/api/datasets/Muennighoff/natural-instructions?full=true - cardData (no license key), gate status, sha, downloads, likes, last-modified date, siblings count; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[5] chainyo/natural-instructions-tokenized dataset card (README), including its embedded `dataset_info` YAML and Python tokenization script that calls `load_dataset("Muennighoff/natural-instructions")`. https://huggingface.co/datasets/chainyo/natural-instructions-tokenized/raw/main/README.md Fetched 2026-08-12.

[6] togethercomputer/RedPajama-INCITE-Instruct-3B-v1 model card (README) front matter, `datasets:` field. https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1/raw/main/README.md Fetched 2026-08-12.

[7] jondurbin/bagel-dpo-34b-v0.2 model card (README), `datasets:` front-matter field and its dataset list description of `natural_instructions`. https://huggingface.co/jondurbin/bagel-dpo-34b-v0.2/raw/main/README.md Fetched 2026-08-12.

[8] Hugging Face Hub model-search API, filtered by `dataset:Muennighoff/natural-instructions`. https://huggingface.co/api/models?filter=dataset:Muennighoff/natural-instructions - lists `wordcab/llama-natural-instructions-13b` among adopters via its `dataset:` tag. Fetched 2026-08-12.

[9] Hugging Face Hub tree API for this repository, recursive listing of `train/` and `test/` and the repository root. https://huggingface.co/api/datasets/Muennighoff/natural-instructions/tree/main?recursive=false , .../tree/main/train?recursive=false&limit=1000 , .../tree/main/test?recursive=false&limit=1000 - file counts, per-file byte sizes, confirmation of no third top-level directory. Fetched 2026-08-12.

[10] datasets-server info and first-rows endpoints. https://datasets-server.huggingface.co/info?dataset=Muennighoff%2Fnatural-instructions and https://datasets-server.huggingface.co/first-rows?dataset=Muennighoff%2Fnatural-instructions&config=default&split=train - both returned HTTP 500 on every retry made for this card. Fetched 2026-08-12.

[11] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Muennighoff%2Fnatural-instructions - returned HTTP 500 on every retry made for this card. Fetched 2026-08-12.

[12] `get_ni.py`, the preprocessing script bundled in this repository. https://huggingface.co/datasets/Muennighoff/natural-instructions/raw/main/get_ni.py - pinned GitHub commit used for task JSON and split-list URLs, row-flattening logic, output schema. Fetched 2026-08-12.

[13] jayelm/natural-instructions dataset card (README). https://huggingface.co/datasets/jayelm/natural-instructions/raw/main/README.md - states it is modified from this repository, describes its added `eval` field and balanced held-out subset sizes. Fetched 2026-08-12.

[14] wisenut-nlp-team/natural-instructions dataset card (README) front matter. https://huggingface.co/datasets/wisenut-nlp-team/natural-instructions/raw/main/README.md Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable, and only under the noted restriction: train on `train`, hold out `test` because it is the benchmark's official evaluation split. This rests on the origin paper's description of the benchmark's train/test structure [1] and the repository's own README, which repackages that same official split [2] - both already established above - together with the screening row's own note.

### The screening row

The row's own note [screening record for `Muennighoff/natural-instructions`, checked 2026-08-12]: "A preprocessed copy of Super-NaturalInstructions, the crowdsourced and expert-written task collection, laid out as 876 per-task jsonl files under train and test directories; the viewer returns 500 for this layout, and the card warns the same input can appear with different outputs." The row carries no flag.
