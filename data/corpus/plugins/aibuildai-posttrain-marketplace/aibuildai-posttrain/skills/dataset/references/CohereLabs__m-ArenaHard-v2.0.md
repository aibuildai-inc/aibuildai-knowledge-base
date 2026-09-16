# CohereLabs/m-ArenaHard-v2.0

11,454 multilingual evaluation prompts, no responses, spread across 23 languages and derived from LMArena's Arena-Hard-v2.0 English test set.

**CohereLabs/m-ArenaHard-v2.0** is Cohere Labs' multilingual evaluation-prompt release built from LMArena's `arena-hard-auto-v2.0` test set [2], introduced by the paper "When Life Gives You Samples: The Benefits of Scaling up Inference Compute for Multilingual LLMs" [1]. The builder filtered LMArena's 750 arena-hard-v2.0 prompts down to 498 English-only prompts with the `papluca/xlm-roberta-base-language-detection` model, then machine-translated those 498 prompts into 22 further languages with an in-house translation model, for 498 prompts x 23 languages [2]. **Every row is a bare prompt string with no paired response: the schema carries `question_id`, `category`, `subcategory`, `prompt`, and `language` only, so there is nothing here to fine-tune or preference-train on [2]. Use it only as a judged evaluation benchmark, and hold its 11,454 prompts (and the 498 English prompts they translate) out of any training corpus to avoid contaminating that benchmark.** It lives at https://huggingface.co/datasets/CohereLabs/m-ArenaHard-v2.0 .

**Use it for**: multilingual LLM-judge evaluation (win-rate against a baseline model), not training - there are no response fields to build an SFT, preference, or reward-model example from [2]. It does not map to any training-format method card; treat it as an eval-time prompt source, one `prompt` per `question_id`, matched with a judge and baseline model as the sibling Greek release also notes for its own arena-hard prompts [4].

**Licence**: the README grants unrestricted academic or commercial use under Apache 2.0 [2], but the Hub API's `cardData.license` field is empty (`null`) [3] - the licence is a prose statement, not machine-encoded metadata. Repo is ungated and public (`"gated": false`, `"private": false`) [3].

**Shape**: 23 configs, one per language code, each a single `test` split of 498 rows; 11,454 rows total [2][5].

**Hold out**: all 11,454 rows (and the 498 English source prompts they translate). This is an evaluation benchmark by construction - it carries no responses to train on [2] - so nothing in it is a training row; the risk to guard is contaminating a future eval run, not a training/test split inside this repository.

**Origin**: built by Cohere Labs; the underlying prompts are human-written queries sourced from Chatbot Arena via LMArena's arena-hard-auto-v2.0 curation, and the 22 non-English variants are machine translations from Cohere's own in-house translation model, not human translations [2]. Hub API at the check date: `downloads` 402, `downloadsAllTime` 5,214, `likes` 7 [3].

**Trained-on-by**: none found - there is nothing to train on in this release [2]. The origin paper instead evaluates models against it: combined sampling-and-selection strategies from that paper produce an average +6.8 point win-rate jump for 8B models on m-ArenaHard-v2.0 prompts against proprietary baselines such as Gemini, and a further-scaled Command-A (111B) model shows a +9.0 point win-rate improvement with five samples over single-sample decoding on the same benchmark [1].

**Introduced by**: [1] (Khairi et al.), the paper this dataset card names as the source of its evaluation results [2].

## Shape

Splits and rows, one `test` split per language config (dataset card YAML `dataset_info`, cross-checked against the datasets-server size endpoint) [2][5]:

| config (language) | split | rows |
| --- | --- | --- |
| ar, cs, de, el, en, es, fa, fr, he, hi, id, it, ja, ko, nl, pl, pt, ro, ru, tr, uk, vi, zh (23 configs) | test | 498 each |
| total | - | 11,454 |

Five columns, identical across all 23 configs (datasets-server info endpoint) [6]:

| column | dtype |
| --- | --- |
| `question_id` | string |
| `category` | string |
| `subcategory` | string |
| `prompt` | string |
| `language` | string |

Total size: 9,181,282 bytes on disk as Parquet (original and Parquet sizes are equal), 18,121,964 bytes decoded in memory, across the 23 configs [5]. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The README's own "Source and Quality Note" section says the upstream Arena-Hard v2.0 prompt set is not uniformly clean and that some of its quality issues originate in the original English data and may survive translation and repair; the builder addressed code-preservation problems, corrupted or mixed-language English source prompts, and translation artifacts, while keeping every language's rows aligned to the original Arena-Hard v2.0 prompt order [2].
- No source states a measured contamination rate, duplicate rate, or translation-quality score for this release; none is invented here.
- Of the first 99 rows read at offset 0 in the `en`/`test` split (the datasets-server first-rows page for that config), every row carries `category` "hard_prompt" and `subcategory` "coding" [7]; this is what those 99 rows show and is not extended to the remaining 399 rows of `en`/`test`, which were not read.
- `question_id` values are shared across language configs: the first row of `ar`/`test` (`question_id` `2edbb5f36f5b42be`) matches the first row of `en`/`test`, confirming the README's row-alignment claim for at least this pair [2][7][8].

## Load it

Load a single language config, or every config concatenated, pinned to the revision this card's numbers were read at:

```python
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names

REV = "24c65eff42cec85e30dd5db99d1a702c7ebaa8ab"  # main at the check date
en = load_dataset("CohereLabs/m-ArenaHard-v2.0", "en", revision=REV, split="test")  # 498 rows

moniker = "CohereLabs/m-ArenaHard-v2.0"
all_langs = concatenate_datasets([
    load_dataset(moniker, cfg, revision=REV, split="test")
    for cfg in get_dataset_config_names(moniker, revision=REV)
])  # 11,454 rows, 23 languages
```

**Trap**: `load_dataset(moniker)` with no config name fails - there is no default config, and the README's own single-language example passes `"en"` explicitly [2]. To load every language you must loop over `get_dataset_config_names` and concatenate, as the README's own second snippet does [2]; a plain `load_dataset(moniker)["test"]` call does not return the multilingual union.

## Neighbors

- `CohereLabs/m-ArenaHard` - the direct predecessor: 500 English arena-hard-auto-v0.1 prompts translated into 22 languages with Google Translate API v3, for 11,000 rows total (500 x 22, read live from the datasets-server size endpoint), with a `cluster` column instead of `subcategory` [9][10]. This v2.0 release supersedes it: it starts from the newer arena-hard-auto-v2.0 prompt set, filters to English-verified prompts first, and applies the targeted quality fixes described above [2], none of which the v1 card claims [10].
- `ilsp/m-ArenaHard_greek` - a Greek-only spinoff of the v1 predecessor (not of this v2.0 release): it re-translates the same arena-hard-auto-v0.1 prompts with Claude Sonnet 3.5, post-editing Cohere's original Google-Translate output, and adds a `prompt_en` column plus a `train` split, for 1,000 rows total (500 test + 500 train) [4]. Its base is v1's 500 prompts, not this release's 498.
- `pinzhenchen/m-ArenaHard-new` - a 60-row trial set across six language configs (cs, de, en, pt, ru, zh), far smaller than this release and not documented with a card; treat it as a trial artifact, not an alternative to this release [11].
- The upstream English-only source itself, LMArena's `arena-hard-auto-v2.0`, is distributed as data files in the `lmarena/arena-hard-auto` GitHub repository rather than as its own Hub dataset repo, per this card's own upstream link [2].

## A row

One schema is served across all 23 configs, so one row covers it. From `config="en"`, `split="test"`, `row_idx=0` (datasets-server first-rows) [7], prompt truncated:

```json
{
  "question_id": "2edbb5f36f5b42be",
  "category": "hard_prompt",
  "subcategory": "coding",
  "prompt": "Write me a zig program that solves the following problem from advent of code and reads the input from a file input.txt and prints the answer to stdout.\n```\n--- Day 25: Let It Snow ---\nMerry Christmas! Santa is booting up his weather machine [...] ",
  "language": "en"
}
```

The same `question_id` (`2edbb5f36f5b42be`) is row 0 of `config="ar"`, `split="test"` as well, with `prompt` translated into Arabic and `language` set to `"ar"` [8] - `question_id` is the join key across every language config's row.

## Where it came from

Built and released by Cohere Labs. The English prompts originate as human queries collected on Chatbot Arena and curated into LMArena's `arena-hard-auto-v2.0` test set [2]; Cohere Labs filtered that set's 750 prompts down to 498 English-verified prompts with the `papluca/xlm-roberta-base-language-detection` classifier, then translated those 498 prompts into 22 further languages using an in-house translation model, applying targeted fixes for code-preservation, corrupted or mixed-language source text, and translation artifacts while keeping every language's rows aligned by `question_id` [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Khairi, D'souza, Shen, Kreutzer, Hooker, "When Life Gives You Samples: The Benefits of Scaling up Inference Compute for Multilingual LLMs", 2025. https://arxiv.org/abs/2506.20544 - the origin paper; current title and abstract (win-rate figures) read from the live abs page. Fetched 2026-08-12.

[2] CohereLabs/m-ArenaHard-v2.0 dataset card (README), including its embedded `dataset_info` YAML. https://huggingface.co/datasets/CohereLabs/m-ArenaHard-v2.0/raw/main/README.md - filtering/translation method, quality note, load instructions, field descriptions, licence prose, config/split/byte counts. Fetched 2026-08-12.

[3] Hugging Face Hub API record for CohereLabs/m-ArenaHard-v2.0. https://huggingface.co/api/datasets/CohereLabs/m-ArenaHard-v2.0?full=true - `sha`, `cardData.license`, `gated`, `private`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[4] ilsp/m-ArenaHard_greek dataset card (README). https://huggingface.co/datasets/ilsp/m-ArenaHard_greek/raw/main/README.md - Greek re-translation of the v1 predecessor, `prompt_en` column, split sizes, judge/baseline usage note. Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=CohereLabs%2Fm-ArenaHard-v2.0 Fetched 2026-08-12.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=CohereLabs%2Fm-ArenaHard-v2.0 Fetched 2026-08-12.

[7] datasets-server first-rows endpoint, `config=en`. https://datasets-server.huggingface.co/first-rows?dataset=CohereLabs%2Fm-ArenaHard-v2.0&config=en&split=test Fetched 2026-08-12.

[8] datasets-server first-rows endpoint, `config=ar`. https://datasets-server.huggingface.co/first-rows?dataset=CohereLabs%2Fm-ArenaHard-v2.0&config=ar&split=test Fetched 2026-08-12.

[9] datasets-server size endpoint for the v1 predecessor. https://datasets-server.huggingface.co/size?dataset=CohereLabs%2Fm-ArenaHard Fetched 2026-08-12.

[10] CohereLabs/m-ArenaHard (v1 predecessor) dataset card (README). https://huggingface.co/datasets/CohereLabs/m-ArenaHard/raw/main/README.md - base prompt set (arena-hard-auto-v0.1), Google Translate API v3 method, `cluster` column, licence prose. Fetched 2026-08-12.

[11] datasets-server size endpoint for pinzhenchen/m-ArenaHard-new. https://datasets-server.huggingface.co/size?dataset=pinzhenchen%2Fm-ArenaHard-new - this endpoint takes no revision parameter, so this count is live, not pinned. Fetched 2026-08-12.

[12] The corpus screening row for `CohereLabs/m-ArenaHard-v2.0`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-12.

## Appendix: screening record

### Screening verdict

Correctly screened as eval-only, unusable for training: the dataset's own schema carries a `prompt` column and no response column at all, which the README's field list confirms row by row [2], matching the screening row's own note [12].

### The screening row

The row's own note [12]: "Arena-Hard v2 prompts translated into 23 languages, prompts only with no responses, so there is nothing to train on." The row carries no flag.
