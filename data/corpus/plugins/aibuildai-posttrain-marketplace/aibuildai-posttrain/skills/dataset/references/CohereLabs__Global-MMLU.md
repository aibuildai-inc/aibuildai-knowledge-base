# CohereLabs/Global-MMLU

601,734 multiple-choice knowledge questions - MMLU translated into 42 languages, with a cultural-sensitivity label attached to a fixed subset of each language's questions.

**CohereLabs/Global-MMLU** is Cohere Labs' 42-language release of the MMLU benchmark, introduced in "Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation" [1]. It combines machine translation of the original MMLU questions [2] with professional translation and crowd-sourced post-edits, and it is described as an evaluation set rather than a training corpus [2]. **The dataset carries only `test` and `dev` splits, no `train` split, so it holds evaluation-only content: any use as training data conflicts with its stated purpose and risks contaminating any MMLU-based benchmark score.** It lives at https://huggingface.co/datasets/CohereLabs/Global-MMLU .

**Use it for**: held-out multilingual multiple-choice evaluation - not training data of any shape. Each row is one MMLU-style question with four lettered options (`option_a`.."option_d") and a correct `answer` letter, the same shape MMLU-derived eval harnesses expect [2]. It does not map to any post-training method card; hold the whole dataset out of SFT, preference, or reasoning-trace training sets.

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`), ungated [3]. No catch on the grant itself - the README states the data may be used without restriction under Apache-2.0 [2]; the only real restriction is the usage-shape one above (evaluation-only, no `train` split), which is a stated-purpose limit, not a licence term.

**Shape**: 601,734 rows across 42 per-language configs (`am`, `ar`, ..., `zh`), each with a `test` split of 14,042 rows and a `dev` split of 285 rows; 17 string/bool columns per row [3][4].

**Hold out**: everything - all 601,734 rows. There is no `train` split to draw from, and the dataset's own framing as an evaluation set [2] means the entire release is contamination risk if it enters any training mix; the screening row's note agrees, describing only `test` and `dev` splits [5].

**Origin**: built by Cohere Labs, with professional annotators and Cohere Labs Community volunteers doing the translation post-edits and cultural-bias annotation [2]; Hub API at the check date: `downloads` 26,561, `downloadsAllTime` 346,651, `likes` 163 [3].

**Trained-on-by**: none found - no source in this check states a model was trained on Global-MMLU; as a translated MMLU benchmark it is built to be evaluated against, not trained on [2].

**Introduced by**: [1] (Singh et al., Cohere Labs).

## Shape

Every one of the 42 language configs shares the same two splits and the same 601,734-row total splits into [3][4]:

| split | rows | languages | rows/language |
| --- | --- | --- | --- |
| `test` | 589,764 | 42 | 14,042 |
| `dev` | 11,970 | 42 | 285 |
| total | 601,734 | 42 | 14,327 |

Columns, identical across every config and split (datasets-server `/info`) [4]:

| column | dtype |
| --- | --- |
| `sample_id` | string |
| `subject` | string |
| `subject_category` | string |
| `question` | string |
| `option_a` | string |
| `option_b` | string |
| `option_c` | string |
| `option_d` | string |
| `answer` | string |
| `required_knowledge` | string |
| `time_sensitive` | string |
| `reference` | string |
| `culture` | string |
| `region` | string |
| `country` | string |
| `cultural_sensitivity_label` | string |
| `is_annotated` | bool |

`required_knowledge`, `time_sensitive`, `reference`, `culture`, `region` and `country` are stored as string-encoded Python lists (one entry per human annotator) to avoid a Hub schema conversion issue; the README shows how to convert them back with `ast.literal_eval` [2]. `is_annotated` marks whether a row carries any cultural-bias-study annotation at all, and `cultural_sensitivity_label` is `-` when it does not.

Sizes across all 42 configs (datasets-server `/size`) [3]: 198,876,690 bytes of original Parquet download, 291,901,765 bytes decoded in memory - matching the 198,876,690-byte figure the shortlist reports for this repository. No source states sequence-length or token statistics for this release.

## Quality

- 119,700 of the 601,734 rows (2,850 per language, `is_annotated=True`) carry cultural-bias-study annotations; of those the README's own statistics table reports 33,264 rows labeled Culturally Sensitive and 86,436 labeled Culturally Agnostic across the 42 languages [2].
- The origin paper's headline finding, from evaluating state-of-the-art open and proprietary models on the full MMLU translation versus the culturally-sensitive subset: 28% of all questions require culturally sensitive knowledge to answer, and among questions requiring geographic knowledge, 84.9% focus on North America or Europe; the paper reports that model rankings change depending on which subset is used for scoring, which is the reason the cultural-sensitivity labels exist [1].
- The README's stated limitations: translation post-edits come from community volunteers, most of whom contributed only once or twice, and contribution counts vary widely between languages; the annotation process did not screen for toxic or offensive content, so some may be present; and the six-region geographic taxonomy used for annotation is coarser than the World Bank taxonomy the authors say they would now prefer [2].
- No source in this check states a measured contamination rate or annotator-agreement figure for Global-MMLU itself.

## Load it

Each language is a separate config; loading one returns a `DatasetDict` with `test` and `dev`. Pin the commit sha that was `main`'s HEAD at the check date (the Hub API's `sha`; the repo was last modified 2025-08-14) [3] so a future force-push cannot silently change what loads; the row counts, columns and byte sizes reported above come from the live datasets-server endpoints rather than this pinned commit, as noted in Sources [4][6].

```python
import datasets

REV = "0e619dbeb34206cd48705a1a0ea7fb21cae09993"  # main at the check date
ds = datasets.load_dataset("CohereLabs/Global-MMLU", "en", revision=REV)
test = ds["test"]  # 14,042 rows - evaluation only, do not train on this
dev = ds["dev"]    # 285 rows - evaluation only, do not train on this
```

**Trap**: there is no `train` split in any config - `ds["train"]` raises a `KeyError`. Loading without a config name fails; a language code (e.g. `"en"`, `"am"`) is required [2][4].

## Neighbors

- `CohereLabs/Global-MMLU-Lite` - the same builder's smaller, more balanced release: 200 Culturally-Sensitive and 200 Culturally-Agnostic samples per language, covering 15 of the 42 languages with human (not machine) translation, 14,000 rows total confirmed live at the check date [2][7]. The README recommends it when a smaller, human-translated, class-balanced set is preferred over the full machine-plus-professional-translation release [2].
- `cais/mmlu` - the original English-only MMLU benchmark this release translates; it is the upstream source, not an alternative release, and is covered under Where it came from below [2].

## A row

All 42 configs and both splits share one schema, so one row covers it. From `config="en"`, `split="test"`, `row_idx=0` (datasets-server `/rows`) [6]:

```json
{
  "sample_id": "abstract_algebra/test/0",
  "subject": "abstract_algebra",
  "subject_category": "STEM",
  "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
  "option_a": "0",
  "option_b": "4",
  "option_c": "2",
  "option_d": "6",
  "answer": "B",
  "required_knowledge": "[]",
  "time_sensitive": "[]",
  "reference": "[]",
  "culture": "[]",
  "region": "[]",
  "country": "[]",
  "cultural_sensitivity_label": "-",
  "is_annotated": false
}
```

This row's `is_annotated` is `false` (`cultural_sensitivity_label` is `-`); of the first 100 `en/test` rows read at the same offset, 50 carry `is_annotated=true` [6].

## Where it came from

Built by Cohere Labs from the original MMLU question set [2] (translated into 42 languages) [1]. Translation combined machine translation with professional translation and crowd-sourced post-edits; the README does not break down which languages received which method [2]. Cultural-bias annotation - the `required_knowledge`, `time_sensitive`, `reference`, `culture`, `region`, `country` and `cultural_sensitivity_label` fields - was collected between May 2024 and August 2024 through two purpose-built Hugging Face Spaces: a Cultural Sensitivity Annotation Platform and a Translation Quality Annotation Platform [2]. The README credits professional annotators together with volunteer contributors from the Cohere Labs Community as curators [2].

## Sources

Checked 2026-08-11; Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. The datasets-server `/info`, `/size` and `/rows` endpoints used below ([4] and [6]) take no `revision` parameter and always serve the current `main` HEAD, confirmed by querying them with a bogus revision and getting the same HTTP 200 content back; every row count, column list, byte size and sampled row drawn from them is therefore a live fact about `main` at the check date, not a fact pinned to the commit sha used in Load it.

[1] Singh et al., "Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation", 2024. https://arxiv.org/abs/2412.03304 - the origin paper; current title and abstract (28% culturally sensitive, 84.9% North America/Europe geographic focus, ranking shifts) read from the live abs page. Fetched 2026-08-11.

[2] CohereLabs/Global-MMLU dataset card (README). https://huggingface.co/datasets/CohereLabs/Global-MMLU/raw/main/README.md - dataset summary, licence, Global-MMLU-Lite pointer, data fields, data splits, statistics table, known limitations, provenance, collection dates. Fetched 2026-08-11.

[3] Hugging Face Hub API record for CohereLabs/Global-MMLU. https://huggingface.co/api/datasets/CohereLabs/Global-MMLU?full=true - licence, gate status, `sha`, `lastModified`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server info and size endpoints. https://datasets-server.huggingface.co/info?dataset=CohereLabs%2FGlobal-MMLU and https://datasets-server.huggingface.co/size?dataset=CohereLabs%2FGlobal-MMLU - columns, per-config and total row counts, byte sizes. Tested with a bogus `revision` query parameter and it returned the same content with HTTP 200, confirming this endpoint ignores `revision` and always reflects the current `main` HEAD; the column list, row counts and byte sizes are therefore live facts about `main` at fetch time, not tied to the pinned commit sha in Load it. Fetched 2026-08-11.

[5] The corpus screening row for `CohereLabs/Global-MMLU`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[6] datasets-server rows endpoint. https://datasets-server.huggingface.co/rows?dataset=CohereLabs%2FGlobal-MMLU&config=en&split=test&offset=0&length=100 - same as [4], this endpoint takes no revision parameter and reflects live `main` HEAD, so the sampled row and the 50/100 `is_annotated` count are live, not pinned to the commit sha. Fetched 2026-08-11.

[7] datasets-server size endpoint for the neighbor. https://datasets-server.huggingface.co/size?dataset=CohereLabs%2FGlobal-MMLU-Lite - this endpoint takes no revision parameter, so this row count is live, not pinned. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable only as held-out evaluation data, never for training: the dataset carries no `train` split and its own card frames it as an evaluation set [2], and the screening row's note confirms it is MMLU translated into 42 languages with only `test` and `dev` splits [5].

### The screening row

The row's own note [5]: "MMLU translated into 42 languages, professional human translation for part and machine translation for the rest; only `test` and `dev` splits." The row carries no flag.
