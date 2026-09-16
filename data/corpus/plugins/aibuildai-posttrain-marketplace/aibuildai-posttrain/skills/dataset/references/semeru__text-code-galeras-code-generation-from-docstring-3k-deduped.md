# semeru/text-code-galeras-code-generation-from-docstring-3k-deduped

2,937 real Python functions paired with their GitHub docstrings and AST/complexity metadata, one of the curated testbeds behind the Galeras causal-interpretability benchmark for code LLMs.

**semeru/text-code-galeras-code-generation-from-docstring-3k-deduped** was built by researchers at William & Mary and released as one of five task-specific testbeds inside *Galeras*, a benchmarking strategy for interpreting large language models for code (LLMc) through causal inference, introduced in "Benchmarking Causal Study to Interpret Large Language Models for Source Code" [1]. Each row is a deduplicated `(docstring, code)` pair mined from popular Python GitHub repositories, carrying the function's AST structure, cyclomatic complexity, and token/identifier counts as confounders for the paper's causal analysis [1]. **The paper itself files this exact testbed (its "FromDocString" set) under the SE task it calls "code completion", not "code generation" — that label in the paper's own task table is reserved for a separate commit-message testbed — so the Hub repo's name and the paper's internal taxonomy disagree on what to call this I/O direction (text ⇒ code); the data itself is unaffected by which name is used [1].** It lives at https://huggingface.co/datasets/semeru/text-code-galeras-code-generation-from-docstring-3k-deduped .

**Use it for**: SFT-style function-body generation from a docstring — use `documentation.docstring` (optionally with `fun_name`/`path` as extra context) as the prompt and `code` as the target. Maps to the SFT method card. Rows are plain text/code pairs with no chat template applied, so wrap them in your own instruction format before training.

**Licence**: none declared — the repo carries no `license` tag, no LICENSE file, and no README to state terms; the upstream GitHub repositories the functions were mined from keep their own individual licences, which this dataset does not track [2][3].

**Shape**: 2,937 rows, one config (`default`), one split (`train`), 23 columns [4][5].

**Hold out**: nothing within this repository itself — it is a single `train` split with no eval companion here. The paper's own dedication of this testbed to LLM interpretability evaluation, and the fact that four sibling Galeras testbeds (RandomCut, the docstring-plus-cut-code completion testbed, CommitGen, SummarizationGen) were sampled from the same source-repo pool and commit window, means a reader who separately evaluates a model on those sibling Galeras testbeds should check for overlap before also training on this one [1][6].

**Origin**: built by Rodriguez-Cardenas, Palacio, Khati, Burke and Poshyvanyk (William & Mary); code and docstrings are real GitHub source text, not model-generated — a human-authored corpus [1]. Hub API at the check date (2026-08-12): downloads 118, lifetime downloads 889, likes 0 [2].

**Trained-on-by**: none found. The origin paper uses this testbed to evaluate ChatGPT's code-completion behavior under different prompting strategies via causal analysis; it does not report training a model on it, and no independent adoption of this specific Hub dataset was found [1].

**Introduced by**: [1] (Rodriguez-Cardenas et al.).

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 2,937 |

One config, `default`, 23 columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `d_id` | int64 |
| `commit_id` | string |
| `repo` | string |
| `path` | string |
| `file_name` | string |
| `fun_name` | string |
| `url` | string |
| `commit_message` | string |
| `code` | string |
| `documentation` | struct: `docstring` (string), `language` (string), `n_whitespaces` (int64), `n_words` (int64), `vocab_size` (int64) |
| `n_words` | int64 |
| `n_whitespaces` | int64 |
| `vocab_size` | int64 |
| `language` | string |
| `nloc` | int64 |
| `token_counts` | int64 |
| `n_identifiers` | int64 |
| `n_ast_nodes` | int64 |
| `n_ast_errors` | int64 |
| `ast_errors` | string |
| `ast_levels` | int64 |
| `complexity` | int64 |

Sizes (datasets-server `/size`) [4]: 6,808,178 bytes of original JSON download, 2,202,014 bytes as Parquet, 4,970,447 bytes decoded in memory. No source states sequence-length statistics for this exact 2,937-row release beyond the descriptive table below.

The paper's own descriptive-statistics table for this testbed (its "FromDocString" row, `avg ± std`, n=2,937): `n_whitespaces` 167.96±244.56, `nloc` 16.68±20.91, `token_counts` 100.13±118.36, `n_ast_errors` 0.10±0.59, `ast_levels` 11.38±3.44, `n_ast_nodes` 156.39±180.71, `complexity` 3.48±4.48, `n_identifiers` 14.62±11.94 [1].

## Quality

- The 2,937-row size is the result of Jaccard-similarity deduplication on BPE-tokenized snippets at a 0.7 similarity threshold: the paper reports this testbed started with 63 detected duplicates (2.10% of the pre-dedup sample) that were removed [1].
- The paper's own prose repeatedly gives the deduplicated `RawData` pool as ≈227K rows, but its own Table II lists a literal `RawData` "Dedup" figure of 277,226 for the same pool — the paper is internally inconsistent on this one number, and this card follows the prose figure below because that is the number the paper uses when describing how `FromDocString` was sampled [1].
- The paper's validity filter for docstrings (more than 3 words) reduces the ≈227K-row (prose figure; 277,226 per Table II, see above) deduplicated `RawData` pool to an intermediate ≈77K docstring-bearing rows; a further manual/automated consistency check — comparing the docstring and commit message against the actual code implementation — then removes about 1.9% of those rows, leaving the ≈57K-row `RawDataDocstring` pool reported in Table II [1]. That same validation step, applied by hand to 960 of the ≈227K `RawData` points (the remainder validated automatically), also checked that each commit's push date and each method's update date fell inside the query's Jan 2, 2022 – Jan 1, 2023 window, in addition to the docstring/commit-message consistency check [1]. These validation steps are described for the shared upstream pipeline, not reported separately for this 2,937-row testbed [1].
- The source commits were deliberately restricted to the window Jan 2, 2022 – Jan 1, 2023, chosen because the paper assumed ChatGPT and other LLMs under study had not been trained on commits from that period, as a guard against data snooping in the paper's own evaluation [1].
- No source states a measured contamination or duplicate rate against any external benchmark, nor an annotator-agreement figure (there are no human annotations to disagree over — the docstrings are the original developers' own comments) [1].
- No columns in this dataset require an external fetch to resolve; `code`, `documentation.docstring`, and all metadata columns are served directly, confirmed by reading the first 100 rows of `train` at offset 0 [7].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main`; the repo was last modified 2023-10-01):

```python
import datasets

REV = "3c09919a8f26fff4a0d537c6bb0bf2ec2b1595a1"  # main at the check date
train = datasets.load_dataset(
    "semeru/text-code-galeras-code-generation-from-docstring-3k-deduped",
    revision=REV, split="train",
)  # 2,937 rows
```

**Trap**: the repo's file tree also carries a raw `code_generation_from_docstring_dataset_3k_deduped.json` sibling file alongside the auto-converted Parquet the `datasets` library reads by default [2][3]. `load_dataset(...)` with no extra arguments reads the Parquet conversion (2,937 rows, the numbers on this card); loading the raw JSON file directly instead is a different code path that this card does not validate.

## Neighbors

Four sibling Galeras testbeds, released by the same builder from the same source-repo pool and commit window, each serving a different SE task; row counts read live at the check date [6]. This corpus card is about the docstring-to-code one; reach for a neighbor only when its task or I/O direction is what you need.

- `semeru/Galeras-3k_deduped` — 2,931 rows, carries a `random_cut` column: the paper's "RandomCut" testbed, code-to-code completion where the input is a signature-truncated cut of the same function (no docstring in the input) [1][6].
- `semeru/code-code-galeras-code-completion-from-docstring-3k-deduped` — 2,926 rows, also carries `random_cut` alongside `documentation`: the paper's "WithDocString" testbed, code+docstring context to code completion [1][6].
- `semeru/code-text-galeras-commit-generation-3k-deduped` — 2,919 rows: the paper's "CommitGen" testbed, code to commit-message text — the testbed the paper's own task table labels "code generation" [1][6].
- `semeru/code-text-galeras-code-summarization-3k-deduped` — 2,924 rows: the paper's "SummarizationGen" testbed, code to docstring-style summary text [1][6].
- `semeru/Code-code-galeras-prompting-3k-control` — 3,000 rows, carrying a `predicted_P2` column: ChatGPT outputs collected for the paper's own prompt-engineering causal case study, not raw training data [1][6].
- `semeru/Code-code-galeras-prompting-3k-treatment-1` — 2,926 rows, carrying `T1` and `control` prompt-template columns but no `predicted_P2`; same case-study family as the control repo above, a different row count [1][6].
- `semeru/Code-code-galeras-prompting-3k-treatment-2` — 2,923 rows, carrying `T1`, `T2` and `control` prompt-template columns but no `predicted_P2`; same case-study family, a third distinct row count [1][6].
- `semeru/galeras-causal4se-3k-levenshtein` — carries `propensity_score`, `strata`, and treatment/outcome columns: the paper's final causal-inference dataset (Levenshtein-distance outcomes), not raw training data [1][6].

## A row

One config, one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7]:

```json
{
  "complexity": 1,
  "n_identifiers": 2,
  "code": "def minzoom(self):\n        \n        return self[\"minzoom\"]\n",
  "path": "packages/python/plotly/plotly/graph_objs/layout/mapbox/_layer.py",
  "n_ast_nodes": 22,
  "ast_errors": "",
  "repo": "plotly.py",
  "documentation": {
    "docstring": "\n        Sets the minimum zoom level (mapbox.layer.minzoom). At zoom\n        levels less than the minzoom, the layer will be hidden.\n\n        The 'minzoom' property is a number and may be specified as:\n          - An int or float in the interval [0, 24]\n\n        Returns\n        -------\n        int|float\n        ",
    "language": "en",
    "n_whitespaces": 101,
    "n_words": 42,
    "vocab_size": 37
  },
  "n_words": 4,
  "language": "Python",
  "vocab_size": 4,
  "commit_id": "43e3a4011080911901176aab919c0ecf5046ddd3",
  "file_name": "_layer.py",
  "id": 232037,
  "nloc": 2,
  "token_counts": 11,
  "fun_name": "minzoom",
  "url": "https://github.com/plotly/plotly.py.git",
  "commit_message": "switch to black .22",
  "n_whitespaces": 18,
  "n_ast_errors": 0,
  "d_id": 63481,
  "ast_levels": 7
}
```

## Where it came from

Built by Rodriguez-Cardenas, Palacio, Khati, Burke and Poshyvanyk (William & Mary) as part of the Galeras benchmark [1]. The upstream pool was collected by querying popular Python GitHub repositories (`fork:false`, `size >= 30,000`, `pushed > 2021-12-31`, `stars > 1,000`) and taking every method touched by a commit in the window Jan 2, 2022 – Jan 1, 2023, yielding roughly 338K raw data points; documentation and AST/complexity features were then parsed with the Tree-Sitter library and stored in a relational database, giving an intermediate deduplicated pool (`RawData`) the paper's own prose repeatedly gives as ≈227K rows but its Table II lists as 277,226 rows for the same pool (see Quality), and, after the docstring-validity and consistency filters described in Quality, a separate ≈57K-row subset with a valid docstring (`RawDataDocstring`) [1]. This testbed was then built by sampling 3,000 points from the larger `RawData` pool — not from the smaller `RawDataDocstring` subset — and deduplicating them by Jaccard similarity on BPE-tokenized snippets, leaving 2,937 rows [1]. No model generated any of the text or code in this dataset; the docstrings and commit messages are the original developers' own writing, and the code is their own committed source [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Rodriguez-Cardenas, Palacio, Khati, Burke, Poshyvanyk, "Benchmarking Causal Study to Interpret Large Language Models for Source Code", 2023. https://arxiv.org/abs/2308.12415 — the origin paper: testbed construction pipeline, Table II descriptive statistics, Table III deduplication/task table, causal-analysis case study. Read from the PDF (https://arxiv.org/pdf/2308.12415) and its text extraction. Fetched 2026-08-12.

[2] Hugging Face Hub API record. https://huggingface.co/api/datasets/semeru/text-code-galeras-code-generation-from-docstring-3k-deduped?full=true — tags, gate status, `sha`, `downloads`, `likes`, last-modified date, file tree (`siblings`); `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[3] Attempted fetch of the repository README. https://huggingface.co/datasets/semeru/text-code-galeras-code-generation-from-docstring-3k-deduped/raw/main/README.md — returned "Entry not found": no README/dataset card exists in this repository. Fetched 2026-08-12.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=semeru%2Ftext-code-galeras-code-generation-from-docstring-3k-deduped — called with no revision parameter, so this is a live read of `main`, not pinned to the `Load it` revision; the repository's Hub API `sha`/last-modified date (2023-10-01, source [2]) predates the check date with no intervening push, so no drift is expected, but the endpoint itself carries no pin. Fetched 2026-08-12.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=semeru%2Ftext-code-galeras-code-generation-from-docstring-3k-deduped — same live-`main`, no-revision-parameter caveat as [4]. Fetched 2026-08-12.

[6] datasets-server size and first-rows endpoints, one call per neighbor, for every neighbor row count and column list above: `semeru/Galeras-3k_deduped`, `semeru/code-code-galeras-code-completion-from-docstring-3k-deduped`, `semeru/code-text-galeras-commit-generation-3k-deduped`, `semeru/code-text-galeras-code-summarization-3k-deduped`, `semeru/Code-code-galeras-prompting-3k-control`, `semeru/Code-code-galeras-prompting-3k-treatment-1`, `semeru/Code-code-galeras-prompting-3k-treatment-2`, `semeru/galeras-causal4se-3k-levenshtein`. https://datasets-server.huggingface.co/size?dataset=<id> and https://datasets-server.huggingface.co/first-rows?dataset=<id>&config=default&split=train — these endpoints take no revision parameter, so these counts and column lists are live, not pinned. Fetched 2026-08-12.

[7] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=semeru%2Ftext-code-galeras-code-generation-from-docstring-3k-deduped&config=default&split=train — 100 rows read at offset 0; same live-`main`, no-revision-parameter caveat as [4]. Fetched 2026-08-12.

[8] The corpus screening row for `semeru/text-code-galeras-code-generation-from-docstring-3k-deduped`, supplied with this card's request — its `note`, read back in the row's own words in the appendix. Checked 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as SFT training data for docstring-to-code generation: real GitHub functions paired with their original docstrings, human-authored on both sides, no chat-template or preference structure to preserve. Two facts decide it, both established above — the data is a plain (docstring, code) pair corpus with no restriction stated by its builder against training use [1], and the screening row's note describes exactly this shape [8].

### The screening row

The row's own note [8]: "real GitHub functions with their docstrings and AST/complexity metadata, for docstring-to-code." The row carries no flag.
