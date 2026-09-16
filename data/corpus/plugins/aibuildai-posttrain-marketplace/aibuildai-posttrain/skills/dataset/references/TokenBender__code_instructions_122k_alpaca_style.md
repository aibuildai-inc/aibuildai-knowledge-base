# TokenBender/code_instructions_122k_alpaca_style

121,959 Alpaca-style code-instruction rows - `instruction`/`input`/`output` triples plus a precomposed single-line prompt-and-response `text` field - one JSON file re-hosted under an Apache-2.0 tag.

**TokenBender/code_instructions_122k_alpaca_style** is a Hub dataset uploaded by user TokenBender on 2023-07-20 [1]. The repository's README carries no descriptive text at all beyond a one-line `license: apache-2.0` front-matter block (28 bytes total) [2], and no paper introduces it, so nothing here is stated by the dataset's own card about who wrote the instructions or which model generated the outputs. A bundled `convert_to_alpaca.py` script shows only how the repo's `text` column was templated from `instruction`/`input`/`output` [3]. It lives at https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style . **The instruction/input/output content matches, row for row where sampled, another Hub dataset's lineage: `iamtarun/code_instructions_120k_alpaca`, which states on its own card that it is "taken from sahil2801/code_instructions_120k" [4], carries the identical row count (121,959) [5], and returned identical `instruction`/`input`/`output` text at every row checked against this repository (offset 0, 100 rows, and offset 50,000, 3 rows) [4][1]. `sahil2801/code_instructions_120k` itself returned HTTP 401 ("Invalid username or password") at the check date and could not be fetched directly [6].**

**Use it for**: SFT on code instructions - each row is an `instruction`/`input`/`output` triple, the shape the SFT method card expects. The `text` column is a ready-templated alternative (see the Load it trap below) - use one or the other, not both. **Given the row-for-row match with the `sahil2801/code_instructions_120k` lineage described above, do not also train on `iamtarun/code_instructions_120k_alpaca` or the two `*_standardized` explosions of this same set (see Neighbors) without deduplicating first.**

**Licence**: Apache-2.0 (`cardData.license` is `"apache-2.0"`, tag `license:apache-2.0`), ungated [1]. The one catch: this is this repository's own declared licence for its re-hosted copy; no accessible source states what licence `sahil2801/code_instructions_120k` itself carried, since that repository could not be fetched [6].

**Shape**: 121,959 rows, one split (`train`), one config (`default`), four string columns [7][8].

**Hold out**: nothing found stating this dataset overlaps a held-out evaluation set. The risk this card carries is duplication with the training data of siblings named in Neighbors, not evaluation contamination - see the bold restriction above.

**Origin**: builder is Hub user TokenBender; the generation method for the underlying instructions and outputs is not stated by any accessible source - the card carries no prose and the upstream repository it traces to is inaccessible [1][2][6]. Hub API at the check date: `downloads` 2,908 (`downloadsAllTime` 46,375), `likes` 80 [1].

**Trained-on-by**: none found. No model card was located citing this repository as training data, and the Hub's model-search-by-dataset query did not return dataset-relevant results at the check date [9].

**Introduced by**: no paper - the dataset card [2], which itself states nothing beyond the licence tag.

## Shape

Splits and rows (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 121,959 |
| total | 121,959 |

One config, `default`, four columns (datasets-server `/info`) [8]:

| column | dtype |
| --- | --- |
| `input` | string |
| `text` | string |
| `instruction` | string |
| `output` | string |

Byte sizes (datasets-server `/size`) [7]: 169,499,006 bytes original JSON download, 72,231,668 bytes as Parquet, 153,696,684 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The repository's own README states nothing about annotation process, filtering, or measured quality - it is a one-line licence tag with no body text [2].
- The only quality-adjacent signal available is the content match against `iamtarun/code_instructions_120k_alpaca`: both repositories return 121,959 rows, and both returned identical `instruction`, `input`, and `output` text at every row sampled (offset 0, 100 rows; offset 50,000, 3 rows) [4][1]. `iamtarun`'s card states its content is "taken from sahil2801/code_instructions_120k" [4], which places this repository's content in the same lineage, though `sahil2801/code_instructions_120k` could not itself be fetched to confirm a three-way match [6].
- The `text` column bakes a fixed single-line template - "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: ... ### Input: ... ### Output: ..." - into every row, confirmed by reading `convert_to_alpaca.py` [3] and by the served row itself [10]. `iamtarun`'s equivalent `prompt` column uses a different, multi-line template ending in "### Response:" rather than "### Output:" [11] - a formatting difference between the two re-hostings, not a content difference in the underlying triples.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-07-20) [1]:

```python
import datasets

REV = "19b59da67914b5fb2e0a5dff937e9917c0cfb7e4"  # main at the check date
train = datasets.load_dataset("TokenBender/code_instructions_122k_alpaca_style", revision=REV, split="train")  # 121,959 rows
```

**Trap**: there is only one split - `train` - so any held-out or validation slice must be carved out manually before training. The repo also serves a redundant `text` column that already contains a full Alpaca-style prompt-and-response template built from `instruction`/`input`/`output` [3]; loading `text` and then also applying your own SFT prompt template to `instruction`/`input`/`output` in the same run doubles the template. Pick one representation.

## Neighbors

- `iamtarun/code_instructions_120k_alpaca` - same 121,959 rows, same underlying `instruction`/`input`/`output` content confirmed by direct row comparison above, adds a `prompt` column in a different Alpaca template than this repo's `text` column [4][5][11]. Prefer this repository or that one, not both, for the reasons in the bold restriction above.
- `iamtarun/python_code_instructions_18k_alpaca` - a Python-only, 18,612-row subset from the same builder, with substantially higher adoption (25,830 downloads, 347 likes at the check date versus 2,908 downloads and 80 likes here) [12][1].
- `kranthigv/code_instructions_122k_alpaca_style_standardized` and `HydraLM/code_instructions_122k_alpaca_style_standardized` - both 365,877 rows (121,959 x 3) in a long `message`/`message_type`/`message_id`/`conversation_id` format that explodes each instruction/input/output triple into three per-message rows; both repositories serve no README, and the first served message text ("Create a function to calculate the sum of a sequence of integers.") matches this repository's own row 0 `instruction`, confirming they are a reformatting of this exact set rather than an independent collection [13][14][10].
- `hcho22/code_instructions_120k_alpaca_filtered` - a smaller (10K-100K size-category), CSV-formatted filtered variant of the same lineage [15].
- `sahil2801/code_instructions_120k` - named by `iamtarun`'s card as the shared origin of this lineage, but returned HTTP 401 at the check date and could not be fetched to confirm directly [6][4].

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [10]:

```json
{
  "instruction": "Create a function to calculate the sum of a sequence of integers.",
  "input": "[1, 2, 3, 4, 5]",
  "output": "# Python code\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum",
  "text": "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: Create a function to calculate the sum of a sequence of integers. ### Input: [1, 2, 3, 4, 5] ### Output: # Python code\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum"
}
```

## Where it came from

Built and uploaded by Hub user TokenBender on 2023-07-20 [1]. The repository bundles a `convert_to_alpaca.py` script that templates `instruction`/`input`/`output` into the `text` column [3], but states nothing about how the underlying instructions, inputs, and outputs were themselves produced. The content matches, row for row where sampled, the `sahil2801/code_instructions_120k` lineage named by `iamtarun/code_instructions_120k_alpaca`'s card as that repository's own origin [4]; `sahil2801/code_instructions_120k` could not be fetched directly to read its own provenance description, since it returned HTTP 401 at the check date [6].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Hugging Face Hub API record for TokenBender/code_instructions_122k_alpaca_style. https://huggingface.co/api/datasets/TokenBender/code_instructions_122k_alpaca_style?full=true - licence, gate, `sha`, siblings, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[2] TokenBender/code_instructions_122k_alpaca_style dataset card (README). https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style/raw/main/README.md - 28 bytes, licence front matter only, no body text. Fetched 2026-08-11.

[3] convert_to_alpaca.py in the repository. https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style/raw/main/convert_to_alpaca.py - shows the template used to build the `text` column. Fetched 2026-08-11.

[4] iamtarun/code_instructions_120k_alpaca dataset card (README). https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca/raw/main/README.md - states the dataset is taken from sahil2801/code_instructions_120k with an added alpaca-style `prompt` column. Fetched 2026-08-11.

[5] datasets-server size endpoint for iamtarun/code_instructions_120k_alpaca. https://datasets-server.huggingface.co/size?dataset=iamtarun%2Fcode_instructions_120k_alpaca Fetched 2026-08-11.

[6] Direct HTTP request to sahil2801/code_instructions_120k. https://huggingface.co/datasets/sahil2801/code_instructions_120k and https://huggingface.co/api/datasets/sahil2801/code_instructions_120k - both returned HTTP 401 with error message "Invalid username or password", meaning the repository is not accessible without authentication (private, gated, or removed) at the check date. Fetched 2026-08-11.

[7] datasets-server size endpoint for TokenBender/code_instructions_122k_alpaca_style. https://datasets-server.huggingface.co/size?dataset=TokenBender%2Fcode_instructions_122k_alpaca_style Fetched 2026-08-11.

[8] datasets-server info endpoint for TokenBender/code_instructions_122k_alpaca_style. https://datasets-server.huggingface.co/info?dataset=TokenBender%2Fcode_instructions_122k_alpaca_style Fetched 2026-08-11.

[9] Hugging Face Hub models-list API, queried with a `dataset` filter for this repository. https://huggingface.co/api/models?dataset=TokenBender/code_instructions_122k_alpaca_style&limit=20 - the endpoint ignored the filter and returned unrelated trending models, so it supplies no adoption evidence either way. Fetched 2026-08-11.

[10] datasets-server first-rows endpoint for TokenBender/code_instructions_122k_alpaca_style. https://datasets-server.huggingface.co/first-rows?dataset=TokenBender%2Fcode_instructions_122k_alpaca_style&config=default&split=train - row 0, and offset-50,000 rows read via the `/rows` endpoint with `offset=50000&length=3`. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint for iamtarun/code_instructions_120k_alpaca. https://datasets-server.huggingface.co/first-rows?dataset=iamtarun%2Fcode_instructions_120k_alpaca&config=default&split=train - row 0, and offset-50,000 rows read via the `/rows` endpoint. Fetched 2026-08-11.

[12] datasets-server size endpoint for iamtarun/python_code_instructions_18k_alpaca, plus its Hub search listing for downloads/likes. https://datasets-server.huggingface.co/size?dataset=iamtarun%2Fpython_code_instructions_18k_alpaca and https://huggingface.co/api/datasets?search=code_instructions&limit=30 Fetched 2026-08-11.

[13] datasets-server info and first-rows endpoints for kranthigv/code_instructions_122k_alpaca_style_standardized. https://datasets-server.huggingface.co/info?dataset=kranthigv%2Fcode_instructions_122k_alpaca_style_standardized and https://datasets-server.huggingface.co/first-rows?dataset=kranthigv%2Fcode_instructions_122k_alpaca_style_standardized&config=default&split=train - row count, schema, and row-0 message text; its README.md returned "Entry not found". Fetched 2026-08-11.

[14] datasets-server size endpoint for HydraLM/code_instructions_122k_alpaca_style_standardized. https://datasets-server.huggingface.co/size?dataset=HydraLM%2Fcode_instructions_122k_alpaca_style_standardized - same row count as [13]; its README.md also returned "Entry not found". Fetched 2026-08-11.

[15] Hugging Face Hub dataset search listing. https://huggingface.co/api/datasets?search=code_instructions&limit=30 - lists hcho22/code_instructions_120k_alpaca_filtered with its size-category and format tags. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for SFT, but only once, and only in one form: the instruction/input/output content here matches, row for row where sampled, the `sahil2801/code_instructions_120k` lineage also served (with an added template column) by `iamtarun/code_instructions_120k_alpaca` and (exploded into per-message rows) by the two `*_standardized` repositories - training on more than one of these at once duplicates the same underlying data. That duplication is what the screening row's note flags, and it is the fact this card's opening restriction rests on.

### The screening row

The row's own note [screening data supplied with this card's request]: "Alpaca-style code instructions; content identical to sahil2801/code_instructions_120k (see FLAG)." Despite the note's parenthetical, the supplied row carries no separate `flag` field - the note's own text is the only flag-like content present.
