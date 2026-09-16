# flytech/python-codes-25k

A single-split Python instruction-tuning set of 24,813 unique instruction/input/output examples, served by the Hub as 49,626 rows because its two source files duplicate the same content, and whose card never states who or what wrote the pairs.

**flytech/python-codes-25k** is a Hub dataset from the `flytech` organization pairing a short natural-language `instruction` (optionally with a one-line `input` context) with a Python `output` that completes the task, plus a `text` field concatenating all three [1]. The dataset card carries no citation to an origin paper; it states only aggregate statistics - 24,813 total entries, 24,580 unique instructions, 3,666 unique inputs, 24,581 unique outputs, 24,813 unique texts, and an average of 508 tokens per example [1]. It lives at https://huggingface.co/datasets/flytech/python-codes-25k . **The Hub's default JSON loader concatenates the repository's `python-codes-25k.json` and `python-codes-25k.jsonl` files, which hold identical content, so `load_dataset` without a `data_files` argument returns 49,626 rows - exactly two copies of the same 24,813 examples - and the card's own load snippet warns to "double-check if there is ~25k examples instead of almost 50k" [1].**

**Use it for**: single-turn instruction-following SFT (Alpaca-style instruction/input/output), either by reconstructing the prompt/response pair from `instruction`+`input`/`output` or by training directly on the pre-joined `text` field; maps to the SFT method card. Deduplicate to the 24,813 unique examples first (see Load it) - training on the served 49,626 rows repeats every example exactly twice.

**Licence**: MIT, stated only in the README body, under a "License" heading whose next line names MIT [1]; the repository's YAML front matter carries no machine-readable `license` field, and the Hub API's `cardData` block likewise has none [2]. Ungated, public (`"gated": false`, `"private": false`) [2].

**Shape**: one config (`default`), one split (`train`) of 49,626 served rows over four string columns (`output`, `instruction`, `input`, `text`) [3][4]; the card's own statistics describe 24,813 unique underlying entries, not the served row count (see opening paragraph and Load it) [1].

**Hold out**: nothing flagged - no source states or implies overlap with any evaluation benchmark. This does not cover the file-duplication trap in Load it, which is a within-dataset redundancy issue, not a train/eval leak.

**Origin**: built and published by the Hub org `flytech`; no source states whether the instruction/output pairs are human-written or model-generated - the README gives only statistics and no authorship claim [1]. The sibling dataset `flytech/llama-python-codes-30k`, which its own card describes this dataset as "the fully cleaned version" of, states that its underlying data is "a blend of GPT-4 generated content, custom codes, behavioral approaches and tasks extending beyond Python" [5]. Hub API at the check date: 5,240 downloads, 60,902 all-time downloads, 182 likes [2].

**Trained-on-by**: Hugging Face's model-hub filter for `dataset:flytech/python-codes-25k` lists 48 models tagged as trained on it at the check date, including `shahdishank/gemma-2b-it-finetune-python-codes` (6 downloads, 4 likes), `Mr-Vicky-01/Gemma-2B-Finetuined-pythonCode` (15 downloads, 4 likes), and `aigcode/AIGCodeGeek-DS-6.7B` (5 downloads, 4 likes) - small community fine-tunes rather than widely-adopted releases [6].

**Introduced by**: no paper - the dataset card [1].

## Shape

The `/size` and `/info` endpoints below take no revision parameter and read the current `main` branch live; the counts they report match the `sha` pinned in Load it only because the repository has not been pushed to since that revision was recorded [2][3][4].

Rows and split (datasets-server `/size`) [3]:

| split | rows |
| --- | --- |
| `train` | 49,626 |

One config, `default`, four string columns (datasets-server `/info`) [4]:

| column | dtype |
| --- | --- |
| `output` | string |
| `instruction` | string |
| `input` | string |
| `text` | string |

Byte sizes (datasets-server `/size`) [3]: 51,802,237 bytes of original JSON/JSONL download, 23,006,748 bytes as Parquet, 47,543,602 bytes decoded in memory. The README's Dataset Statistics section states an average of 508 tokens per example, measured over its stated 24,813 total entries; it does not say which tokenizer produced that figure, and it predates the duplication - it is not a statistic over the 49,626 served rows [1].

## Quality

- The README's own statistics show internal duplication even within its stated 24,813-entry count: 24,580 unique instructions and 24,581 unique outputs against 24,813 total entries, so roughly 230 instructions and 230 outputs each repeat at least once within the unique set [1].
- No source states a measured code-execution or correctness rate for the `output` field, an annotation or generation process, or a contamination check.
- No source states who or what produced the pairs; the templated instruction/input/output shape and the sibling dataset's own description of a "blend of GPT-4 generated content" [5] are consistent with model generation, but no source confirms it for this dataset specifically.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main`, matching the shortlist's recorded commit; the repo was last modified 2024-05-15) [2]:

```python
import datasets

REV = "0ed98ff2a76c5d133d8c157b814189a5a17ebd20"  # main at the check date
ds = datasets.load_dataset("flytech/python-codes-25k", revision=REV, split="train")  # 49,626 rows as of the check date (live count, see Shape)
```

**Trap**: this returns 49,626 rows, but rows 0-24,812 and rows 24,813-49,625 are pairwise identical - confirmed by fetching offset 0 and offset 24,813 from the served rows and finding matching `instruction`/`output`/`text` at both offsets [7]. The duplication comes from the repository shipping the same content twice, as `python-codes-25k.json` and `python-codes-25k.jsonl` [8], both consumed by the default JSON builder. Deduplicate (e.g. `ds.to_pandas().drop_duplicates(subset="text")`, expected to land near the card's stated 24,813) before training, or the dataset silently doubles every example's weight.

## Neighbors

- `flytech/llama-python-codes-30k` - the same author's earlier, Llama-tokenized release (27,332 rows served at the check date) [9]; its own card calls itself "not cleaned" with "a very low number of unique input entries" and points readers to this dataset as "the fully cleaned version", so this corpus prefers `python-codes-25k` over it [5].
- `badaranta/python-codes-25k` - a third-party re-upload under a different author name; its served size (49,626 rows, 47,543,602 bytes in memory) and its row 0 (`instruction`, `input`, `output`, `text` all byte-identical to this dataset's row 0) match this dataset exactly, so it is a mirror, not a distinct release [10].
- `sg247/python-codes-25k-llama2` - a reformatted, five-column variant (49,626 rows) [10].
- `zurd46/python-codes-25k-de` - a German-translated subset (2,481 rows) [10].
- The dataset card itself states the data "can also be found on kaggle, under the same name but from different author," without a link [1].

## A row

The repository serves one config and one split, and the served rows are pairwise duplicated (see Load it), so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/rows`) [7]:

```json
{
  "output": "```python\ntasks = []\nwhile True:\n    task = input('Enter a task or type 'done' to finish: ')\n    if task == 'done': break\n    tasks.append(task)\nprint(f'Your to-do list for today: {tasks}')\n```",
  "instruction": "Help me set up my daily to-do list!",
  "input": "Setting up your daily to-do list...",
  "text": "Help me set up my daily to-do list! Setting up your daily to-do list... ```python\ntasks = []\nwhile True:\n    task = input('Enter a task or type 'done' to finish: ')\n    if task == 'done': break\n    tasks.append(task)\nprint(f'Your to-do list for today: {tasks}')\n```"
}
```

## Where it came from

Published by the Hub org `flytech`; the README states only that it is "a Cleaned Python Dataset Covering 25,000 Instructional Tasks" with four fields, and gives no collection method, generating model, or annotator description [1]. The one provenance clue on the Hub is indirect: the sibling dataset `flytech/llama-python-codes-30k`, which this dataset's own family relationship (that dataset's card names this one as its cleaned counterpart) ties to it, describes its own underlying data as a blend of GPT-4-generated content and custom code [5]; no source makes that claim about `python-codes-25k` directly.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] flytech/python-codes-25k dataset card (README). https://huggingface.co/datasets/flytech/python-codes-25k/raw/main/README.md - license line, overview, dataset statistics, feature descriptions, usage section, load snippet and its duplication warning, kaggle mention. Fetched 2026-08-11.

[2] Hugging Face Hub API record for flytech/python-codes-25k. https://huggingface.co/api/datasets/flytech/python-codes-25k?full=true (and the same endpoint with `expand[]=downloadsAllTime`) - `cardData`, `sha`, `downloads`, `downloadsAllTime`, `likes`, `gated`, `private`, `lastModified`. Fetched 2026-08-11.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=flytech%2Fpython-codes-25k - this endpoint takes no revision parameter, so its row and byte counts are live, not pinned; they match the `main` `sha` recorded in [2] only because the repository has had no push since that `lastModified` date. Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=flytech%2Fpython-codes-25k - likewise unpinned; same caveat as [3]. Fetched 2026-08-11.

[5] flytech/llama-python-codes-30k dataset card (README). https://huggingface.co/datasets/flytech/llama-python-codes-30k/raw/main/README.md - "not cleaned" / low unique-input warning, pointer to python-codes-25k as "the fully cleaned version", and the "blend of GPT-4 generated content, custom codes, behavioral approaches and tasks extending beyond Python" description of its own underlying data. Fetched 2026-08-11.

[6] Hugging Face Hub API model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:flytech/python-codes-25k - list of 48 model ids with their `downloads` and `likes` fields, all read from this single bulk response. Fetched 2026-08-11.

[7] datasets-server rows endpoint, three calls at offset 0, offset 24,813, and offset 49,623 (length 3 each). https://datasets-server.huggingface.co/rows?dataset=flytech%2Fpython-codes-25k&config=default&split=train&offset=0&length=3 (and offset=24813, offset=49623) - confirms rows at offset 0 and offset 24,813 are identical, i.e. the two halves of the served split duplicate each other. Fetched 2026-08-11.

[8] Hugging Face Hub repository tree for flytech/python-codes-25k at `main`. https://huggingface.co/api/datasets/flytech/python-codes-25k/tree/main - lists `python-codes-25k.json` and `python-codes-25k.jsonl` as the two data files. Fetched 2026-08-11.

[9] datasets-server size endpoint for flytech/llama-python-codes-30k. https://datasets-server.huggingface.co/size?dataset=flytech%2Fllama-python-codes-30k - live row count, no revision parameter. Fetched 2026-08-11.

[10] datasets-server size and rows endpoints for three third-party neighbor repositories: `badaranta/python-codes-25k` (size and row 0), `sg247/python-codes-25k-llama2` (size), `zurd46/python-codes-25k-de` (size). https://datasets-server.huggingface.co/size?dataset=<id> and https://datasets-server.huggingface.co/rows?dataset=badaranta%2Fpython-codes-25k&config=default&split=train&offset=0&length=1 - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT instruction data once deduplicated: the card's own statistics and a direct comparison of served rows at offset 0 and offset 24,813 show the 49,626 served rows are two exact copies of 24,813 unique examples, so training on the raw load doubles every example's weight (see Load it). The dataset's authorship is unverified - the screening note that "the card gives only statistics and never says who or what wrote them" holds after fetching every available source, and the samples' templated, uniform style is consistent with (but not confirmed as) model generation.

### The screening row

The row's own note [as supplied with this card's request]: "25k short Python instruction/output pairs; the card gives only statistics and never says who or what wrote them (the samples read as model output)." The row carries no flag.
