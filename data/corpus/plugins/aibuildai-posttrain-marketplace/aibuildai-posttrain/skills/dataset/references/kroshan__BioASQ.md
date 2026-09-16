# kroshan/BioASQ

8,216 biomedical question-context-answer rows in two plain-string columns, split train/validation, packaged with no dataset card.

**kroshan/BioASQ** is a CSV repackaging uploaded by Hub user `kroshan`; the repository carries no README, no license field, and no citation of its own (`card_body_bytes` is 0 and `https://huggingface.co/datasets/kroshan/BioASQ/raw/main/README.md` returns "Entry not found") [1]. Each row pairs a `question` string with a `text` string of the fixed form `<answer> ... <context> ...`, where the answer segment is a short expert-style span and the context segment is a full biomedical abstract [2]. That shape - a short gold answer plus one supporting abstract per question, with the same question repeated across several rows for different abstracts - matches the factoid-question task format that the BioASQ biomedical semantic indexing and question-answering challenge defines, whose benchmark question sets carry gold-standard reference answers prepared by a European team of biomedical experts [3], and that the continuously-updated BioASQ-QA benchmark corpus paper describes as combining questions with linked documents and snippets [4]. No source in this repository states that these particular rows were drawn from an official BioASQ release, what BioASQ year or task version they come from, or how the pairing was built. **Because no source documents the provenance of these specific rows, and because BioASQ's own factoid test questions are a public evaluation benchmark reused across the biomedical-QA literature [3][4], treat any use of this repository as unverified against BioASQ's official test sets and do not assume its `validation` split is safe from overlap with a benchmark you plan to score against.** It lives at https://huggingface.co/datasets/kroshan/BioASQ .

**Use it for**: closed or open-book biomedical question-answering SFT, after splitting each `text` value on the literal `<answer>` and `<context>` markers to recover an `answer` string and a `context` string - the format is not chat-formatted (`chat_dialect` is `none` on this repository) [1]. Maps to the SFT method card once the delimiter is parsed into separate fields.

**Licence**: not stated - the repository has no license tag, no card, and no license text anywhere in its metadata; the repo is ungated [1].

**Shape**: 8,216 rows in one config (`default`), two splits, two string columns (`question`, `text`) [5][6].

**Hold out**: the `validation` split (4,950 rows) from training on `train` (3,266 rows); no source states this repository's exact relationship to any specific BioASQ official test set, so treat any external BioASQ-derived evaluation set as a further, unverified overlap risk per the restriction above.

**Origin**: uploaded by Hub user `kroshan`; the answer spans read as human-expert reference answers in the BioASQ style, not model generations, per the BioASQ challenge's description of its gold-standard answers [3]. Hub API at the check date: `downloads` 1,278, `downloadsAllTime` 11,928, `likes` 18 [1][7].

**Trained-on-by**: none found - no source found for this repository names a model or training recipe that used it.

**Introduced by**: no paper and no dataset card for this specific repackaging. Its row format matches the factoid question-answering task defined by the BioASQ challenge overview [3] and the continuously-extended BioASQ-QA benchmark corpus [4], but neither paper names this Hub repository.

## Shape

Rows served and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 3,266 |
| `validation` | 4,950 |
| total | 8,216 |

One config, `default`, with two columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `question` | string |
| `text` | string |

The repository's file tree holds exactly two data files, `train_bio.csv` (5,580,275 bytes) mapped to the `train` split and `valid_bio.csv` (8,416,918 bytes) mapped to the `validation` split, plus a `.gitattributes` file [8]. Total original download size is 13,997,193 bytes; as Parquet, 2,907,264 bytes; decoded in memory, 14,027,503 bytes [5]. No source states sequence-length or token statistics for this repository.

## Quality

- Of the 100 `train` rows read at offset 0, only 9 distinct `question` values appear, one repeated 52 times; each repeat pairs the same question and answer with a different abstract in the `<context>` segment, so a single BioASQ-style question is flattened across many rows, one per supporting document [2]. The same pattern holds in the 94 `validation` rows read at offset 0: also 9 distinct questions, the most-repeated ("Orteronel was developed for treatment of which cancer?") appearing 38 times [9].
- Within those same 94 `validation` rows, grouping by identical `(question, text)` finds 11 recurring exact-duplicate groups, sized 2, 2, 2, 2, 2, 2, 3, 5, 5, 7, and 16 rows - 48 rows total, 37 of them beyond each group's first occurrence. The Orteronel question alone splits into six separate duplicate-text groups, sized 16, 7, 5, 5, 3, and 2 rows, each pairing the question with a different repeated abstract in `text`; the "Which MAP kinase phosphorylates the transcription factor c-jun?" question contributes four more 2-row duplicate groups, and the Acrokeratosis/Bazex question contributes the remaining 2-row group [9]. This is a fact about the 94 rows actually read at offset 0 in `validation`, not a claim about the full 4,950-row split.
- No source states a measured contamination rate, an annotator-agreement figure, or a construction/collection process for this specific repository; none of that is invented here.
- Every one of the 100 `train` rows and all 94 `validation` rows read match the fixed `<answer> ... <context> ...` template exactly, with no rows deviating from it in the rows sampled [2][9].

## Load it

Train on `train`, hold out `validation`, and pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2021-12-06) [1]:

```python
import datasets

REV = "d6b0152cab800446548aa0f9fcec972560cd903c"  # main at the check date
train = datasets.load_dataset("kroshan/BioASQ", revision=REV, split="train")           # 3,266 rows
validation = datasets.load_dataset("kroshan/BioASQ", revision=REV, split="validation") # 4,950 rows - hold out
```

**Trap**: the `text` column is not a ready answer field - it is the literal string `<answer> <span> <context> <abstract>` and must be split on those two markers before use; loading it as-is and treating it as an answer will feed the abstract text into the target. A minimal parse:

```python
import re
m = re.match(r"^<answer>\s*(.*?)\s*<context>\s*(.*)$", example["text"], re.S)
answer, context = m.group(1), m.group(2)
```

## Neighbors

Several other Hub repositories reformat BioASQ-derived question-answering data differently; row counts below were read live at the check date [10].

- `rag-datasets/rag-mini-bioasq` - a two-config release (4,719-row `question-answer-passages` test split plus a 40,221-row `text-corpus` passage split) built, per its own card, from the official BioASQ Task 11b training data, "we generated our own subset using `generate.py`" [11]. Its card also points to `enelpol/rag-mini-bioasq` as an updated version without NaNs in the passage corpus [11]. Neither shares kroshan/BioASQ's row counts or its combined `<answer><context>` text format.
- `lucadiliello/bioasqqa` - 1,504 rows in a single `test` split with separate `context`, `question`, `answers`, `key`, and `labels` columns, a SQuAD-style extractive-QA layout rather than a combined string [12].
- `pavanmantha/BioASQ_with_context` - 4,719 rows in a single `train` split with separate `question`, `ground_truth`, and `context` columns - the same row count as `rag-mini-bioasq`'s question-answer-passages split, suggesting a shared source, though neither card states the relationship [12].

None of these three match kroshan/BioASQ's row counts (3,266/4,950) or its single-column `<answer>...<context>...` packing; a chooser who wants pre-split answer/context fields should look at these instead of parsing kroshan/BioASQ's delimiter.

## A row

The repository serves one config and one schema across both splits, so one row from each split covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [2]:

```json
{
  "question": "What is the inheritance pattern of Li–Fraumeni syndrome?",
  "text": "<answer> autosomal dominant <context> Balanced t(11;15)(q23;q15) in a TP53+/+ breast cancer patient from a Li-Fraumeni syndrome family. Li-Fraumeni Syndrome (LFS) is characterized by early-onset carcinogenesis involving multiple tumor types and shows autosomal dominant inheritance. [...] These data may implicate the region at breakpoint 11q23 and/or 15q15 as playing a significant role in predisposition to breast cancer development."
}
```

From `config="default"`, `split="validation"`, `row_idx=0` (datasets-server `/first-rows`) [9]:

```json
{
  "question": "Name synonym of Acrokeratosis paraneoplastica.",
  "text": "<answer> Bazex syndrome <context> Acrokeratosis paraneoplastica (Bazex syndrome): report of a case associated with small cell lung carcinoma and review of the literature. Acrokeratosis paraneoplastic (Bazex syndrome) is a rare, but distinctive paraneoplastic dermatosis characterized by erythematosquamous lesions located at the acral sites and is most commonly associated with carcinomas of the upper aerodigestive tract. [...] it is also important for the radiologist to be aware of this entity and its common presentations."
}
```

## Where it came from

No source states who built this specific repository's CSV files or how the question/answer/context triples were extracted or matched; the repository's only metadata is the Hub API record and the CSV files themselves, with no README, dataset script, or citation attached [1]. The content's shape - a short expert-reference answer plus one full PubMed abstract as supporting context per biomedical question - matches the task that the BioASQ challenge defines: BioASQ Task 1b (and its later, larger iterations) uses benchmark question sets with gold-standard reference answers prepared by a team of biomedical experts from around Europe, alongside relevant articles and snippets retrieved from PubMed Central that systems must use as context [3]. The BioASQ-QA benchmark corpus paper describes this same combination - questions with golden answers plus linked documents and snippets - as continuously extended over the life of the BioASQ challenge [4]. Neither paper names or describes this Hub repository.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Hugging Face Hub API record for kroshan/BioASQ. https://huggingface.co/api/datasets/kroshan/BioASQ?full=true - `sha`, `downloads`, `likes`, tags, siblings, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant; card body confirmed empty by fetching https://huggingface.co/datasets/kroshan/BioASQ/raw/main/README.md ("Entry not found"). Fetched 2026-08-12.

[2] datasets-server first-rows endpoint, train split. https://datasets-server.huggingface.co/first-rows?dataset=kroshan%2FBioASQ&config=default&split=train - 100 rows read at offset 0 (response marked `truncated: true`). Fetched 2026-08-12.

[3] Tsatsaronis et al., "An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition", BMC Bioinformatics, 2015. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0564-6 - task definition, gold-standard answer preparation, PubMed article/snippet retrieval. Fetched 2026-08-12.

[4] Krithara et al., "BioASQ-QA: A manually curated corpus for Biomedical Question Answering", Scientific Data, 2023. https://www.nature.com/articles/s41597-023-02068-4 - benchmark corpus description, documents and snippets, continuous extension. Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=kroshan%2FBioASQ Fetched 2026-08-12.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=kroshan%2FBioASQ Fetched 2026-08-12.

[7] Hugging Face Hub API record for kroshan/BioASQ with `downloadsAllTime` expansion. https://huggingface.co/api/datasets/kroshan/BioASQ?expand[]=downloadsAllTime Fetched 2026-08-12.

[8] Hugging Face Hub tree endpoint for kroshan/BioASQ. https://huggingface.co/api/datasets/kroshan/BioASQ/tree/main - file list and per-file byte sizes. Fetched 2026-08-12.

[9] datasets-server first-rows endpoint, validation split. https://datasets-server.huggingface.co/first-rows?dataset=kroshan%2FBioASQ&config=default&split=validation - 94 rows read at offset 0 (response marked `truncated: true`). Fetched 2026-08-12.

[10] datasets-server size endpoint, one call per neighbor: `rag-datasets/rag-mini-bioasq`, `lucadiliello/bioasqqa`, `pavanmantha/BioASQ_with_context`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-12.

[11] rag-datasets/rag-mini-bioasq dataset card (README). https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq/raw/main/README.md - BioASQ Task 11b provenance statement, pointer to the `enelpol/rag-mini-bioasq` update. Fetched 2026-08-12.

[12] datasets-server first-rows endpoint, one call per neighbor: `lucadiliello/bioasqqa` (`config=default`, `split=test`) and `pavanmantha/BioASQ_with_context` (`config=default`, `split=train`) - column names and one sample row from each. https://datasets-server.huggingface.co/first-rows?dataset=<id>&config=default&split=<split> Fetched 2026-08-12.

[13] The corpus screening row for `kroshan/BioASQ`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as biomedical question-answering training data, with an unresolved provenance gap: the repository's `question`/`text` rows match the BioASQ factoid task's short-answer-plus-abstract shape [3][4], and the screening row's own note describes it the same way [13], but no source - including this repository itself - documents which BioASQ release or year these rows were drawn from, so the restriction on unverified overlap with an official BioASQ test set stands until that is resolved. Hold out `validation` (4,950 rows) from `train` (3,266 rows) for any training run.

### The screening row

The row's own note [13]: "BioASQ biomedical questions with expert answers plus retrieved abstract context; train/validation only." The row carries no flag.
