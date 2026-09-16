# Nan-Do/instructional_code-search-net-python

418,545 single-turn instruction/response pairs for Python - each row either asks for a natural-language summary of a given function or asks for the code that implements a given description - built by turning the same set of Python functions into two-way prompts.

**Nan-Do/instructional_code-search-net-python** is a community-built instructional dataset for Python, created by the Hub user Nan-Do from a summarized version of the Python portion of CodeSearchNet, `Nan-Do/code-search-net-python` [1]; that source dataset is itself the Python split of the CodeSearchNet corpus introduced by "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search" [2], with each function's docstring re-summarized by a Salesforce T5 summarization model [1]. This dataset applies templates and NLP techniques to turn each function into two complementary tasks - generating a description of what a given piece of code does, and generating the code that fulfils a given description - as instruction/response pairs, and the card states there are no splits [3]. It lives at https://huggingface.co/datasets/Nan-Do/instructional_code-search-net-python . **The single `train`-labeled block folds in CodeSearchNet's original train, test, and validation partitions with no partition column to separate them again: searching this dataset's own served rows for two functions confirmed present in the source dataset's `test` partition (`delete_file_if_needed`, `check_exc_info`) found both, verbatim, among these 418,545 rows [4][5]. Decontaminate before evaluating any model trained on this data against a CodeSearchNet-derived benchmark.**

**Use it for**: single-turn SFT on an instruction/response pair - INSTRUCTION as the prompt, RESPONSE as the target completion, no chat template applied. Maps directly to the SFT method card's prompt-completion format. **Decontaminate against CodeSearchNet test/validation functions first** [4][5].

**Licence**: Apache-2.0 (`cardData.license: apache-2.0`, tag `license:apache-2.0`), ungated [6]. The one catch: this licence covers Nan-Do's compilation and generated text; neither this card nor its source card states whether the underlying GitHub functions' original per-repository licences were checked or preserved [3][1].

**Shape**: one config (`default`), one split `train`, 418,545 rows, three string columns (`INSTRUCTION`, `RESPONSE`, `SOURCE`) [7][8].

**Hold out**: rows built from CodeSearchNet's original `test` and `validation` partitions (row count not stated - see Quality for what was verified by search), because this dataset carries no partition/split column of its own. Decontaminate by matching the code text embedded in `RESPONSE`/`INSTRUCTION` against `Nan-Do/code-search-net-python`'s `code` or `original_string` column, which does carry a `partition` value of `train`, `test`, or `valid` per row [1][4].

**Origin**: built by the Hub user Nan-Do; text is a mix of template-authored instruction prompts and either the original function code or a Salesforce T5 model's docstring summary as the response [1][3]. Hub API at the check date: `downloads` 1,891, `downloadsAllTime` 9,117, `likes` 36 [6].

**Trained-on-by**: none found - no source states a model or published recipe trained on this dataset.

**Introduced by**: no paper - the dataset card [3]; the underlying corpus it summarizes is introduced by [2].

## Shape

Rows and columns served (datasets-server `/size`, `/info`) [7][8]:

| split | rows |
| --- | --- |
| `train` | 418,545 |

| column | dtype |
| --- | --- |
| `INSTRUCTION` | string |
| `RESPONSE` | string |
| `SOURCE` | string |

Sizes (datasets-server `/size`) [7]: 172,777,462 bytes of original Parquet download, 453,147,370 bytes decoded in memory. The `/size` and `/info` endpoints take no revision parameter, so these row and byte counts are live reads, not pinned to the `Load it` revision. The row count (418,545) and download size (172,777,462 bytes) match the `cardData.dataset_info` declared in the pinned README [3][6]; the decoded in-memory byte figure does not - the README declares 451,473,573 bytes against the live endpoint's 453,147,370 [3][6][7]. The repository has not changed since 2023-05-20 [6], so this is a pre-existing mismatch in the source metadata itself, not a sign the repo moved after the pin. No source states sequence-length or token statistics for this dataset; none is invented here.

## Quality

- `SOURCE` carries a single constant value, `"codesearchnet"`, across every one of 140 rows sampled at offsets 0-99, 200,000-200,019 and 418,525-418,544 [9] - the column records provenance, not a per-row distinguishing fact.
- The source card states the annotations were cleaned to remove repeated or meaningless summaries [1], and this card repeats the same claim [3]; neither states a measured duplicate rate or repetition rate.
- Overlap with CodeSearchNet's evaluation partitions was checked by full-text search, not assumed. `Nan-Do/code-search-net-python` carries a `partition` column; probing its served rows located `partition="test"` functions named `delete_file_if_needed` (row 420,001) and `check_exc_info` (row 420,000) [4]. Searching this dataset's own served rows for those exact function names (datasets-server `/search`) returned an exact-code match for each - `delete_file_if_needed` at row 80,671 and `check_exc_info` at row 80,670 - confirming CodeSearchNet `test`-partition functions are present in this dataset's rows [5]. This checked two functions, not the full overlap; no source states what share of the 418,545 rows come from `test` or `valid`.
- Bisection probing of `Nan-Do/code-search-net-python`'s row order, reading one row at each probed offset, found the partition column laid out in three contiguous blocks: `train` from offset 0 through 410,174 (410,175 rows), `test` from 410,175 through 432,265 (22,091 rows), and `valid` from 432,266 through the final row 455,242 (22,977 rows) - the three block sizes sum to the dataset's declared 455,243 rows [4].

## Load it

```python
import datasets

REV = "98ba5e1cd9f9603eb9860e9409deaffd34829530"  # main at the check date
ds = datasets.load_dataset("Nan-Do/instructional_code-search-net-python", revision=REV, split="train")  # 418,545 rows
```

**Trap**: there is only one split, `train`, and no held-out split to evaluate against - the README itself states the dataset has no splits [3]. Loading it as-is trains on rows that duplicate CodeSearchNet's own `test` and `validation` functions (see Hold out and Quality); build a held-out set before training if you plan to evaluate on any CodeSearchNet-derived benchmark.

## Neighbors

All fetched live at the check date [10].

- `Nan-Do/code-search-net-python` - the direct upstream: 455,243 rows, one function per row with `repo`/`path`/`func_name`/`code`/`docstring`/`summary`/`partition` columns, where `partition` still marks each row's original CodeSearchNet `train`/`test`/`valid` membership [1][10]. Prefer this one over the instructional release when you need to filter by partition before generating training pairs.
- `Nan-Do/reason_code-search-net-python` - a five-task-type sibling built from the same source pool: for each function, a summary of what it does, what its parameters mean, what its return value means, the return type, or the parameter types, tagged by a `TYPE` column (1-5); 429,059 rows [10][11].
- Same-builder, other-language instructional siblings, same two-way code/description construction: `Nan-Do/instructional_code-search-net-java` (467,959 rows), `Nan-Do/instructional_code-search-net-php` (536,632 rows), `Nan-Do/instructional_code-search-net-go` (203,128 rows), `Nan-Do/instructional_code-search-net-javacript` (121,323 rows), `Nan-Do/instructional_code-search-net-ruby` (51,470 rows) [10]. Pick the language that matches your target; none is Python.

## A row

One config, one split, so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=3` (datasets-server `/first-rows`) [9]:

```json
{
  "INSTRUCTION": "Implement a function in Python 3 to\nconvert a dlib rect object to a plain tuple in ( top right bottom left",
  "RESPONSE": "def _rect_to_css(rect):\n    \"\"\"\n    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order\n\n    :param rect: a dlib 'rect' object\n    :return: a plain tuple representation of the rect in (top, right, bottom, left) order\n    \"\"\"\n    return rect.top(), rect.right(), rect.bot [...]",
  "SOURCE": "codesearchnet"
}
```

The reverse direction sits in the same schema: row 0 has `INSTRUCTION` asking what a function does and `RESPONSE` holding a one-line summary, while this row has `INSTRUCTION` holding a description and `RESPONSE` holding the function body - both fit the same prompt/completion collator [9].

## Where it came from

Built by the Hub user Nan-Do in May 2023 [3]. The pipeline has two steps, both documented on the two upstream cards: first, `Nan-Do/code-search-net-python` takes the Python portion of the CodeSearchNet corpus [2] and adds a `summary` column generated by a Salesforce T5 summarization model over each function's docstring [1]; second, this dataset applies templates and natural-language-processing methods over that summarized data to phrase each function as an instruction/response pair in both directions - code-to-description and description-to-code [3]. The `SOURCE` column names this pipeline's single input, `codesearchnet`, on every row [9].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Nan-Do/code-search-net-python dataset card (README). https://huggingface.co/datasets/Nan-Do/code-search-net-python/raw/main/README.md - source-data pointer, T5 annotation process, cleaning statement. Fetched 2026-08-11.

[2] Husain et al., "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search", 2019. https://arxiv.org/abs/1909.09436 - the corpus this dataset's source is drawn from; current title read from the live abs page. Fetched 2026-08-11.

[3] Nan-Do/instructional_code-search-net-python dataset card (README). https://huggingface.co/datasets/Nan-Do/instructional_code-search-net-python/raw/main/README.md - dataset summary, no-splits statement, annotation process, licensing. Fetched 2026-08-11.

[4] datasets-server rows endpoint against `Nan-Do/code-search-net-python`, bisection-probed (each probed offset saved) to locate the exact `partition` column boundaries between `train`/`test` and between `test`/`valid`, and to read two `test`-partition function names. https://datasets-server.huggingface.co/rows?dataset=Nan-Do%2Fcode-search-net-python&config=default&split=train - this endpoint takes no revision parameter, so these offsets and boundaries are live, not pinned. Fetched 2026-08-11.

[5] datasets-server search endpoint against this dataset, queried for the two `test`-partition function names found via [4]. https://datasets-server.huggingface.co/search?dataset=Nan-Do%2Finstructional_code-search-net-python&config=default&split=train&query=<name> Fetched 2026-08-11.

[6] Hugging Face Hub API record for Nan-Do/instructional_code-search-net-python. https://huggingface.co/api/datasets/Nan-Do/instructional_code-search-net-python?full=true - licence, gate, `sha`, `downloads`, `likes`; all-time downloads read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Nan-Do%2Finstructional_code-search-net-python - takes no revision parameter; live, not pinned. Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Nan-Do%2Finstructional_code-search-net-python - takes no revision parameter; live, not pinned. Fetched 2026-08-11.

[9] datasets-server first-rows and rows endpoints against this dataset, sampled at offsets 0, 200,000 and 418,525 for the `SOURCE`-constancy check and at row 0 and row 3 for the example row. https://datasets-server.huggingface.co/first-rows?dataset=Nan-Do%2Finstructional_code-search-net-python&config=default&split=train Fetched 2026-08-11.

[10] datasets-server size endpoint, one call per neighbor: `Nan-Do/code-search-net-python`, `Nan-Do/reason_code-search-net-python`, `Nan-Do/instructional_code-search-net-java`, `Nan-Do/instructional_code-search-net-php`, `Nan-Do/instructional_code-search-net-go`, `Nan-Do/instructional_code-search-net-javacript`, `Nan-Do/instructional_code-search-net-ruby`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[11] Nan-Do/reason_code-search-net-python dataset card (README). https://huggingface.co/datasets/Nan-Do/reason_code-search-net-python/raw/main/README.md - five task types, `TYPE` column, annotation process using the Python AST module and docstring parsing. Fetched 2026-08-11.

[12] The corpus screening row for `Nan-Do/instructional_code-search-net-python`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT prompt/completion data, but not without a held-out set: the two upstream cards establish it as a template-built, two-way instruction/response release over CodeSearchNet Python functions with no split of its own [1][3], and searching this dataset's own served rows confirmed CodeSearchNet `test`-partition functions are present in it [4][5] - a contamination risk the screening row's note does not mention.

### The screening row

The row's own note [12]: "the same functions turned into two-way code<->description instructions using that T5 summary." The row carries no flag.
