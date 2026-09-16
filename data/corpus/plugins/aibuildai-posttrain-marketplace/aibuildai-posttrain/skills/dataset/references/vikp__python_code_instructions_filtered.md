# vikp/python_code_instructions_filtered

170,635 Python code-instruction pairs, a single-split concatenation of five smaller, already-filtered Hub datasets - the merge carries no paper of its own.

**vikp/python_code_instructions_filtered** is a Hub user's ("vikp") merge of five of their own pre-filtered subsets: `xlcost_filtered_2k`, `evol_instruct_code_filtered_39k`, `evol_codealpaca_filtered_87k`, `code_instructions_filtered_7k` and `code_search_net_filtered_34k`, each already "filtered based on quality and learning value" before the merge, per the card's own wording [1]. There is no origin paper; the closest thing to documentation is the dataset card itself [1]. **About 21% of rows (36,326 of 170,635: the `csn` and `xlcost` slices) carry an empty `instruction` field and are code-only completions, not instruction-response pairs [2]. The `evol_alpaca` slice (from `theblackcat102/evol-codealpaca-v1` [3]) contains at least 80 rows built verbatim around HumanEval's `fibfib` problem, confirmed by a live full-text search hit count on this repository's own served rows [2]; decontaminate against HumanEval / HumanEval+ before any scored eval run.** It lives at https://huggingface.co/datasets/vikp/python_code_instructions_filtered .

**Use it for**: Python instruction-following SFT (the SFT method card) using the `instruction`/`output` columns as prompt/completion - but only after filtering to the three `kind` values that actually carry an instruction (`code_instructions`, `evol_alpaca`, `evo_instruct`); the `csn` and `xlcost` rows are code-only and better suited to a code-completion SFT shape, not instruction SFT [2]. **Decontaminate against HumanEval / HumanEval+ first** [4].

**Licence**: not stated - `cardData` carries no `license` key and the repo tags include no `license:` tag [5]; ungated, not private [5]. The catch: one of the five merged components, the CodeSearchNet-derived `csn` slice (34,488 rows, ~20% of the total), was released upstream under CC-BY-4.0 in its own dataset card, a term this merged repo's own (absent) licence field does not carry forward [6].

**Shape**: 170,635 rows, one config (`default`), one split (`train`); three string columns, `output`, `instruction`, `kind` [7][8].

**Hold out**: no split to hold out - this is one `train` split with no declared eval set [7]. The contamination risk is the HumanEval-derived rows named above; hold those out (or filter them) before any HumanEval/HumanEval+-scored run [4].

**Origin**: built and released by Hub user `vikp`, merging their own five prior filtered releases; no separate generation or annotation is described beyond what those five upstream cards state [1][9][3][10][11][12]. Hub API at the check date: `downloads` 119, `downloadsAllTime` 1,540, `likes` 5 [5].

**Trained-on-by**: two of the same author's own models declare this dataset in their Hub tags - `vikp/instruct_llama_7b` and `vikp/llama_coder` [13]. No third-party adoption found.

**Introduced by**: no paper - the dataset card [1].

## Shape

One config (`default`), one split, three columns (datasets-server `/info` and `/size`) [7][8]:

| split | rows |
| --- | --- |
| `train` | 170,635 |

| column | dtype |
| --- | --- |
| `output` | string |
| `instruction` | string |
| `kind` | string |

No sequence-length or token statistics are stated by any source read for this card. Original download size 160,726,948 bytes; in-memory (Parquet-decoded) size 179,603,234 bytes [7][8]. The card's own YAML block additionally states an uncompressed `dataset_size` of 313,731,517 bytes, a figure not reproduced by `/size`'s 179,603,234-byte in-memory figure; both are quoted here rather than reconciled, since neither source explains the discrepancy [1][7].

The `kind` column is not documented in the card but its five values map exactly onto the five merged upstream repos, and their row counts sum exactly to the served total with no further row-level filtering at merge time: `csn` 34,488 = `code_search_net_filtered_34k` [12]; `code_instructions` 7,526 = `code_instructions_filtered_7k` [10]; `evol_alpaca` 87,705 = `evol_codealpaca_filtered_87k` [3]; `evo_instruct` 39,078 = `evol_instruct_code_filtered_39k` [11]; `xlcost` 1,838 = `xlcost_filtered_2k` [9]. Rows are stored in contiguous blocks in that same order, confirmed by sampling the served rows at coarse offsets across the full split [2].

## Quality

- No source states a measured duplicate rate, contamination rate against a named benchmark other than the HumanEval hit found here, or an annotator-agreement figure for this merged repository [1].
- Two of the three columns retained from the upstream repos (`output`, `instruction`) drop the `quality_prob` and `learning_prob` classifier-score columns that all five upstream repos carry, so the reader cannot re-derive which rows scored highest on the upstream quality/learning-value filters from this repository alone [3][9][10][11][12].
- Sampling the first 100 served rows (offset 0) and single rows read at coarse offsets from 5,000 to 170,600 across the full split shows the `csn` and `xlcost` `kind` values consistently carry an empty `instruction` string, while `code_instructions`, `evol_alpaca` and `evo_instruct` consistently carry a non-empty one [2]. A live full-text search against this repository's own served rows returns exactly 80 hits for the token `fibfib`, all under `kind="evol_alpaca"`, each row's `instruction` and `output` built around the HumanEval `fibfib` problem statement and doctests verbatim [2].
- No rows in this repository require an external fetch to train on; all three columns are served in full by the Hub dataset viewer, subject to the truncation the endpoint itself applies to very long fields [2].

## Load it

```python
import datasets

REV = "2d134335b24111315215f106723585e85d3724f2"  # main at the check date
ds = datasets.load_dataset("vikp/python_code_instructions_filtered", revision=REV, split="train")  # 170,635 rows
```

**Trap**: loading the whole `train` split mixes code-only rows (`kind` in `{"csn", "xlcost"}`, empty `instruction`) with instruction-response rows (`kind` in `{"code_instructions", "evol_alpaca", "evo_instruct"}`); an instruction-SFT collator applied to the whole split will train on ~36,326 rows whose "instruction" is an empty string. Filter on `kind` before choosing a collator [2].

## Neighbors

The five upstream repos this dataset merges are also its closest neighbors - each is a subset of this release at the same rows, still carrying the `quality_prob`/`learning_prob` columns this merge drops [3][10][11][12][9]. Pick this merged repository when Python-only instruction-plus-code-completion breadth is wanted in one split; pick a single upstream repo to train on only one source's rows, keep the classifier-score columns, or avoid the empty-instruction `csn`/`xlcost` rows entirely.

- `vikp/evol_codealpaca_filtered_87k` - 87,705 rows, filtered from `theblackcat102/evol-codealpaca-v1` by manual filtering plus quality/learning-value classifiers; this is the source of the `evol_alpaca` kind and its HumanEval-derived rows [3].
- `vikp/evol_instruct_code_filtered_39k` - 39,078 rows, filtered from `nickrosh/Evol-Instruct-Code-80k-v1` the same way; the `evo_instruct` kind [11].
- `vikp/code_instructions_filtered_7k` - 7,526 rows, filtered from `sahil2801/code_instructions_120k`; the `code_instructions` kind [10].
- `vikp/code_search_net_filtered_34k` - 34,488 rows, filtered CodeSearchNet Python (perplexity, quality/learning-value, manual filters), building on `bjoernp/code_search_net_python_processed_400k`, and the only one of the five with its own stated licence, CC-BY-4.0; the `csn` kind [12].
- `vikp/xlcost_filtered_2k` - 1,838 rows; its own card states no filtering method or source beyond the name; the `xlcost` kind [9].

## A row

Two distinct served shapes exist within the single `train` split, split by whether `instruction` is populated; one row of each.

Empty-instruction (code-only) shape, `kind="xlcost"`, `row_idx=169503`, `split="train"` [2]:

```json
{
  "output": "def max_gcd(n):\n    \"\"\"\n    Maximum GCD among all pairs ( i , j ) of first N natural numbers\n    \"\"\"\n    return (n // 2)\n\n\nif __name__ == '__main__':\n    n = 4\n    print(max_gcd(n))\n",
  "instruction": "",
  "kind": "xlcost"
}
```

Instruction-response shape, `kind="code_instructions"`, `row_idx=38008`, `split="train"` [2]:

```json
{
  "output": "\n\nThe solutions to the system of equations are x = 4, y = 5.",
  "instruction": "Suggest a solution to the following system of equations:\n\nx + y = 9\n2x + 3y = 15",
  "kind": "code_instructions"
}
```

## Where it came from

Built by Hub user `vikp` as a straight concatenation of five of their own already-filtered repos, in this order, with the exact per-source row counts read live from each repo's own `/size` endpoint at the check date, which sum exactly to the 170,635 served here [7][9][3][10][11][12]:

| upstream repo | rows | filtering stated on its own card |
| --- | --- | --- |
| `code_search_net_filtered_34k` (`kind="csn"`) | 34,488 | CodeSearchNet Python subset, filtered by perplexity with/without docstring plus quality/learning-value classifiers and manual filtering, building on `bjoernp/code_search_net_python_processed_400k`; CC-BY-4.0 [12] |
| `code_instructions_filtered_7k` | 7,526 | filtered from `sahil2801/code_instructions_120k` by manual, quality and learning-value filters [10] |
| `evol_codealpaca_filtered_87k` | 87,705 | filtered from `theblackcat102/evol-codealpaca-v1` by manual filtering plus quality/learning-value classifiers [3] |
| `evol_instruct_code_filtered_39k` | 39,078 | filtered from `nickrosh/Evol-Instruct-Code-80k-v1` the same way [11] |
| `xlcost_filtered_2k` | 1,838 | no filtering method stated on its own card [9] |

The `code_search_net_filtered_34k` and `xlcost_filtered_2k` slices carry no `instruction` text in this merge; those two upstream cards show pure code (or code plus a `text`/`docstring`-shaped field) rather than an instruction, and the merge drops that distinction into a blank `instruction` string [12][9]. No source states a decontamination step against HumanEval or any other eval benchmark anywhere in this chain [1][3][10][11][12][9].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; Hub repositories are mutable, which is why Load it pins the revision.

[1] vikp/python_code_instructions_filtered dataset card (README). https://huggingface.co/datasets/vikp/python_code_instructions_filtered/raw/main/README.md - lists the five merged sources, states filtering by quality and learning value. Fetched 2026-08-11.

[2] datasets-server rows/search endpoints against vikp/python_code_instructions_filtered, `split=train`: `/rows` sampled at offset 0 (first 100 rows) and at coarse offsets 5,000 through 170,600 across the full split; `/search?query=fibfib` (80 total hits, `kind="evol_alpaca"` throughout the sample checked); individual rows read at `row_idx` 169503 and 38008. https://datasets-server.huggingface.co/rows?dataset=vikp%2Fpython_code_instructions_filtered&config=default&split=train and https://datasets-server.huggingface.co/search?dataset=vikp%2Fpython_code_instructions_filtered&config=default&split=train&query=fibfib Fetched 2026-08-11.

[3] vikp/evol_codealpaca_filtered_87k dataset card and size endpoint. https://huggingface.co/datasets/vikp/evol_codealpaca_filtered_87k/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=vikp%2Fevol_codealpaca_filtered_87k - names `theblackcat102/evol-codealpaca-v1` as source; 87,705 rows. Fetched 2026-08-11.

[4] The corpus screening row for `vikp/python_code_instructions_filtered`, supplied with this card's request - its `flag` field, read back in the appendix. Checked 2026-08-11.

[5] Hugging Face Hub API record for vikp/python_code_instructions_filtered. https://huggingface.co/api/datasets/vikp/python_code_instructions_filtered?full=true - `sha`, `private`, `gated`, `license` absent from `cardData`, `downloads`, `likes`, `lastModified`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] vikp/code_search_net_filtered_34k dataset card, stating `license: cc-by-4.0` in its own YAML front matter, contrasted against [5]'s absent `license` field for the merged repo. https://huggingface.co/datasets/vikp/code_search_net_filtered_34k/raw/main/README.md Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=vikp%2Fpython_code_instructions_filtered Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=vikp%2Fpython_code_instructions_filtered Fetched 2026-08-11.

[9] vikp/xlcost_filtered_2k dataset card and Hub API/size records. https://huggingface.co/datasets/vikp/xlcost_filtered_2k/raw/main/README.md , https://huggingface.co/api/datasets/vikp/xlcost_filtered_2k?full=true , https://datasets-server.huggingface.co/size?dataset=vikp%2Fxlcost_filtered_2k - card states no filtering method; 1,838 rows. Fetched 2026-08-11.

[10] vikp/code_instructions_filtered_7k dataset card and size endpoint. https://huggingface.co/datasets/vikp/code_instructions_filtered_7k/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=vikp%2Fcode_instructions_filtered_7k - names `sahil2801/code_instructions_120k` as source; 7,526 rows. Fetched 2026-08-11.

[11] vikp/evol_instruct_code_filtered_39k dataset card and size endpoint. https://huggingface.co/datasets/vikp/evol_instruct_code_filtered_39k/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=vikp%2Fevol_instruct_code_filtered_39k - names `nickrosh/Evol-Instruct-Code-80k-v1` as source; 39,078 rows. Fetched 2026-08-11.

[12] vikp/code_search_net_filtered_34k dataset card and size endpoint. https://huggingface.co/datasets/vikp/code_search_net_filtered_34k/raw/main/README.md , https://datasets-server.huggingface.co/size?dataset=vikp%2Fcode_search_net_filtered_34k - filtering method, credit to `bjoernp/code_search_net_python_processed_400k`, CC-BY-4.0 licence, 34,488 rows. Fetched 2026-08-11.

[13] Hub models API filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:vikp/python_code_instructions_filtered - returns `vikp/instruct_llama_7b` and `vikp/llama_coder`, both tagged `dataset:vikp/python_code_instructions_filtered`. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT data with two conditions established above: filter to the `kind` values that carry a non-empty `instruction` before training an instruction-following collator, and remove or decontaminate the HumanEval-derived `evol_alpaca` rows before any HumanEval/HumanEval+-scored evaluation. Both conditions rest on facts fetched directly from the repository's own served rows [2], not on the screening row alone.

### The screening row

The row's own note [4]: "quality-filtered blend of xlcost, evol-instruct code, CodeAlpaca, code instructions and CodeSearchNet; inherits their GPT-written answers." Its flag [4]: "rule-risk: humaneval, humanevalplus - Its evol-codealpaca subset carries HumanEval prompts verbatim: 80 rows `fibfib`, 35 `separate_paren_groups`, 1 `has_close_elements`."
