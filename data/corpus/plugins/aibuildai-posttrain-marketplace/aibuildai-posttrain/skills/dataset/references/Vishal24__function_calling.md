# Vishal24/function_calling

16,746 rows of single-turn function-call SFT data - an e-commerce product title paired with a per-category JSON-schema function and the extracted attribute dict a call to it should return - with no dataset card.

**Vishal24/function_calling** is an uncredited Hub upload with no README (`card_body_bytes` is 0, and the raw README path returns "Entry not found") [1][2], so builder, generation process, and licence are not stated anywhere in the repository. Each row poses a product title as a user prompt, offers one callable function named `smp` ("Get the features of product") whose JSON-schema parameters and enum choices are tailored to that row's product category, and gives the target function-call arguments as the assistant response [3]. It lives at https://huggingface.co/datasets/Vishal24/function_calling .

**No licence is stated anywhere in the repository - the Hub API record carries no `cardData` and no `license:` tag** [1]. Treat this as unlicensed for redistribution purposes until a licence can be confirmed elsewhere.

**Use it for**: single-turn function-calling SFT, where the model reads a title and a tool schema and must emit the correct call arguments; this is not the OpenAI/Hermes-style function-calling chat format (no `tools` list, no assistant `tool_calls` field, no multi-turn history) and would need reformatting - see the SFT method card for tool-use data. `functionList` and `assistantResponse` are stored as Python `dict`-repr strings (single quotes, `True`/`False`), not JSON, despite the repo's `format:json` tag - see Load it.

**Licence**: not stated. No `cardData.license`, no `license:` tag on the repo, and no licence text found in a (nonexistent) README [1].

**Shape**: 16,746 rows in one config (`default`), split `train` 14,736 / `test` 2,010, six string columns [4][5].

**Hold out**: not stated - no card states a held-out evaluation use for this data, and no source found here ties any row to a named benchmark. Nothing to hold out beyond the repo's own `test` split (2,010 rows), which is a train/test division, not a stated eval-contamination guard.

**Origin**: unknown - the repo has no dataset card, so no builder or generation process is stated [1][2]. The Hub API shows no linked models under `dataset:Vishal24/function_calling`, and as of the check date (2026-08-11) the API's `downloads` field reads 125 and its separately-fetched `downloadsAllTime` field reads 662; likes are 3 [1][6].

**Trained-on-by**: none found - the Hub's linked-models filter for this dataset returns an empty list; no blog or paper search was performed [1].

**Introduced by**: no paper - the dataset card [1] (which is empty; the id and file listing are the only self-description the repo carries).

## Shape

Rows served and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 14,736 |
| `test` | 2,010 |
| total | 16,746 |

One config, `default`, with six columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `userPrompt` | string |
| `title` | string |
| `functionList` | string |
| `assistantResponse` | string |
| `category` | string |
| `sub_category` | string |

The repository holds two plain JSON files, `train.json` (23,620,467 bytes) and `test.json` (3,237,562 bytes), no Parquet at the source [7]; datasets-server's own Parquet conversion reports 3,529,336 bytes total and 24,097,860 bytes decoded in memory [4]. No source states sequence-length or token statistics.

## Quality

No card and no paper accompany this repository, so no stated annotation process, measured error rate, or known complaint exists to report; none is invented here [1][2].

From a sample of 436 rows read live across both splits (train offsets 0, 3000, 6000, 9000, 12000, 14700 and test offsets 0, 1000, 1900, each up to 50 rows) [8]:

- The function name is always `smp`, but its JSON-schema `parameters` are not one fixed schema: the 436 sampled rows carry 34 distinct sets of parameter-key names, one set per product category (e.g. `whey protein` rows ask for `diet`/`flavor`/`form`/`brand`; `desktops` rows ask for `ram`/`hdd`/`ssd`/`series`/`generation`/`gaming`/`brand` instead) [8].
- 22 distinct `category` values appear in the sample, spanning nutrition/supplement products (`whey protein`, `protein`, `mass gainer`, `pre-workout`), personal care (`skin care`, `makeup`, `eyes`, `oral healthcare`), and appliances/electronics (`desktops`, `electric kettles`, `vacuums & floor care`, `kitchen appliances`) [8].
- `userPrompt` follows the template "Extract the features from the title : " + `title` in 400 of the 436 sampled rows; the remaining 36 use the same template with the space before the colon dropped ("...the title :" immediately followed by the title, no leading space) [8].
- `functionList` and `assistantResponse` are Python `dict.__repr__` strings (single-quoted keys, `True`/`False`/`None` literals), not valid JSON - `json.loads` on a sampled cell raises `Expecting property name enclosed in double quotes`, while Python's `ast.literal_eval` parses it correctly [9].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-09-27) [1]:

```python
import datasets

REV = "0106ee7cc7c7ff9aea2de3471083237e25bb21d1"  # main at the check date
train = datasets.load_dataset("Vishal24/function_calling", revision=REV, split="train")  # 14,736 rows
test = datasets.load_dataset("Vishal24/function_calling", revision=REV, split="test")    # 2,010 rows
```

**Trap**: `functionList` and `assistantResponse` load as plain strings that look like JSON but are Python `dict` reprs (single-quoted keys, `True`/`False`/`None`) - `json.loads` on them raises a parse error; use `ast.literal_eval` instead, or rewrite the quoting before feeding downstream tooling that expects real JSON [9].

## Neighbors

The same author, `Vishal24`, hosts `Vishal24/small_function_calling`, a smaller repository with the identical six columns (`userPrompt`, `title`, `functionList`, `assistantResponse`, `category`, `sub_category`) and a 2,000/500 train/test split (2,500 rows total, against 14,736/2,010 here) [10][11]. Its own repository likewise carries no dataset card, so whether it is a subsample of this release or a separately drawn set is not stated; prefer this larger release unless the smaller size is specifically wanted. The same author's Hub listing carries 34 further repositories - several with names suggesting overlapping product-attribute or category-classification tasks (e.g. `feature_extractor`, `title_category`, `sub_cat_data_all`) - but only `small_function_calling` was opened and checked here; those other repositories were not fetched, so whether any of them is a neighbor of this release is not stated [10].

## A row

One config and one schema, so one row from each split shows the shape; both are chosen to show a different per-category function schema. `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3]:

```json
{
  "userPrompt": "Extract the features from the title : leucine dutch goat milk protein bcaa whey protein isolate valine wpi protein essential amino acid supplement satiety wph parents",
  "title": "leucine dutch goat milk protein bcaa whey protein isolate valine wpi protein essential amino acid supplement satiety wph parents",
  "functionList": "{'name': 'smp', 'description': 'Get the features of product.', 'parameters': {'type': 'object', 'properties': {'diet': {'type': 'array', ...}, 'material free': {...}, 'use': {...}, 'flavor': {...}, 'benefits': {...}, 'form': {'type': 'string', 'description': 'product form', 'enum': ['capsules', 'cream', 'liquid', 'powder', 'spray', 'tablets', 'gels']}, 'brand': {'type': 'string', 'description': 'get the brand of the product'}}, 'required': ['diet', 'material free', 'use', 'flavor', 'benefits', 'form', 'brand']}}",
  "assistantResponse": "{'diet': '', 'material free': '', 'use': '', 'flavor': '', 'benefits': '', 'form': 'powder', 'brand': 'Leucine'}",
  "category": "whey protein",
  "sub_category": "whey protein"
}
```

`config="default"`, `split="train"`, a `desktops`-category row read from the offset-14700 sample [8], showing a differently-shaped `parameters` schema for the same function name:

```json
{
  "userPrompt": "Extract the features from the title :hp s01 s01 - pf0307in desktop (9th gen i5-9400/4gb/1tb hdd/windows 10 home/integrated graphics), jet black and monitor combo",
  "title": "hp s01 s01 - pf0307in desktop (9th gen i5-9400/4gb/1tb hdd/windows 10 home/integrated graphics), jet black and monitor combo",
  "functionList": "{'name': 'smp', 'description': 'Get the features of product.', 'parameters': {'type': 'object', 'properties': {'ram': {'type': 'string', 'description': 'ram capacity', 'enum': ['4gb', ' 8gb']}, 'hdd': {...}, 'ssd': {...}, 'screen': {...}, 'series': {'type': 'string', 'description': 'processor series name', 'enum': ['i5', ' i7', ' ryzen 7', ' apple m2', ' mediatek']}, 'generation': {...}, 'touch': {'type': 'boolean', ...}, 'graphic': {...}, 'graphic series': {...}, 'os': {...}, 'gaming': {'type': 'boolean', ...}, 'count': {...}, 'quantity': {...}, 'unit': {...}, 'brand': {...}}, 'required': ['ram', 'hdd', 'ssd', 'screen', 'series', 'generation', 'touch', 'graphic', 'graphic series', 'os', 'gaming', 'count', 'quantity', 'unit', 'brand']}}",
  "assistantResponse": "{'ram': '4gb', 'hdd': '1tb', 'ssd': '', 'screen': '', 'series': 'i5', 'generation': 9, 'touch': False, 'graphic': '', 'graphic series': '', 'os': 'windows 10 home', 'gaming': False, 'count': 1, 'quantity': 1, 'unit': '', 'brand': 'hp'}",
  "category": "desktops",
  "sub_category": "desktops"
}
```

## Where it came from

Not stated. The repository carries no dataset card, so no builder, upstream data pool, collection method, or generating model is documented for this release [1][2]. The repository's `siblings` list holds only `.gitattributes`, `train.json`, and `test.json` - no data-generation script, notebook, or provenance file [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Hugging Face Hub API record for Vishal24/function_calling. https://huggingface.co/api/datasets/Vishal24/function_calling?full=true - `sha`, `lastModified`, `createdAt`, `downloads`, `likes`, tags, `cardData` absence, siblings list; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant; linked-models check via `https://huggingface.co/api/models?filter=dataset:Vishal24/function_calling` (empty result). Fetched 2026-08-11.

[2] Raw README fetch. https://huggingface.co/datasets/Vishal24/function_calling/raw/main/README.md - returns "Entry not found", confirming no dataset card exists. Fetched 2026-08-11.

[3] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Vishal24%2Ffunction_calling&config=default&split=train - feature list and row 0. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Vishal24%2Ffunction_calling Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Vishal24%2Ffunction_calling Fetched 2026-08-11.

[6] Hugging Face Hub API record with `downloadsAllTime` expansion. https://huggingface.co/api/datasets/Vishal24/function_calling?expand[]=downloadsAllTime Fetched 2026-08-11.

[7] Hugging Face Hub tree API. https://huggingface.co/api/datasets/Vishal24/function_calling/tree/main - file listing and byte sizes for `train.json` and `test.json`. Fetched 2026-08-11.

[8] datasets-server rows endpoint, sampled at multiple offsets. https://datasets-server.huggingface.co/rows?dataset=Vishal24%2Ffunction_calling&config=default&split=train (offsets 0, 3000, 6000, 9000, 12000, 14700, length 50 each) and https://datasets-server.huggingface.co/rows?dataset=Vishal24%2Ffunction_calling&config=default&split=test (offsets 0, 1000, 1900, length 50 each) - 436 rows total read, used for category count, `functionList` parameter-key diversity, and `userPrompt` template check. Fetched 2026-08-11.

[9] Local parsing check against a sampled `functionList`/`assistantResponse` cell: `json.loads` raises `Expecting property name enclosed in double quotes`, `ast.literal_eval` parses it. Checked against a cell fetched in [8] on 2026-08-11.

[10] Hugging Face Hub API listing of the author's other datasets. https://huggingface.co/api/datasets?author=Vishal24 - lists `Vishal24/small_function_calling` among other repos by the same author. Fetched 2026-08-11.

[11] datasets-server size and info endpoints for the neighbor. https://datasets-server.huggingface.co/size?dataset=Vishal24%2Fsmall_function_calling and https://datasets-server.huggingface.co/info?dataset=Vishal24%2Fsmall_function_calling - row counts and column names. Fetched 2026-08-11.

[12] The corpus screening row for `Vishal24/function_calling`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as narrow, single-domain function-calling SFT data, with no licence to rely on for redistribution. The two facts that decide it are already established above: the repository has no dataset card so builder, licence, and provenance are unknown [1][2], and every sampled row (436 of 16,746) resolves to the same `smp` function name applied to a per-category parameter schema, matching the screening row's characterization of a narrow e-commerce attribute-extraction task [8].

### The screening row

The row's own note [12]: "a narrow e-commerce case - pull product attributes out of a title into one fixed function schema; no card." The row carries no flag. This card's row-level check found the function name fixed (`smp`) but its parameter schema varying by product category rather than literally fixed, as detailed under Quality above.
