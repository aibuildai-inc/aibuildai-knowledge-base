# jinaai/code_search_net_clean

1,914,013 (docstring, code) function pairs across six programming languages, a per-language-split, deduplicated republish of the CodeSearchNet corpus.

**jinaai/code_search_net_clean** is Jina AI's re-release of the CodeSearchNet Corpus introduced in "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search" [1], reshaped into eighteen splits (`train`/`test`/`validation` crossed with Go, Java, JavaScript, PHP, Python and Ruby) and reduced to two columns, `doc` and `code`, holding the function's docstring and its source respectively [2]. The dataset's own card carries no description beyond the Hugging Face auto-generated placeholder - no stated cleaning method, no stated intended task [2]. It lives at https://huggingface.co/datasets/jinaai/code_search_net_clean . **Use only the `train.*` splits for training: the `test.*` splits (92,321 rows) are CodeSearchNet's per-language held-out split, and this dataset's own screening note flags them as CodeSearchNet's retrieval benchmark. MTEB's `CodeSearchNetRetrieval` task is built from the same upstream corpus (`code-search-net/code_search_net`) and organizes its own evaluation data into per-language splits also named "test", 1,000 rows each [3] - consistent with, though not confirmed identical to, reusing that same held-out portion - so training on `test.*` risks contaminating that evaluation.**

**Use it for**: paired natural-language/code training on `train.*` only - either the embedding-training method card's contrastive positive-pair format (`doc` as query, `code` as positive passage) or plain prompt/completion SFT pairs (`doc` as instruction, `code` as completion); the dataset card states no single intended format, so either read is a fit for the two-column shape, not a card claim [2].

**Licence**: none stated - the repository carries no `license` tag and `cardData` has no `license` field; ungated (`"gated": false`, `"private": false`) [4]. The one catch: the upstream corpus this cleans, `code-search-net/code_search_net`, tags itself `license:other` and its own card says each example inherits its own GitHub repository's licence, unresolved per example, and not included in the data [5]; nothing on this repository resolves that for the cleaned copy either.

**Shape**: 1,914,013 rows, one config (`default`), 18 splits (6 languages times train/test/validation), 2 string columns (`doc`, `code`) [2][6].

**Hold out**: `test.*` (92,321 rows across the six languages) - the screening note names these as CodeSearchNet's retrieval benchmark; MTEB's `CodeSearchNetRetrieval` task is built from the same upstream corpus and also organizes its evaluation data as per-language "test" splits [3], though no source confirms row-for-row identity between the two. `validation.*` (82,221 rows) is the corpus's own model-selection split; no source shows it reused as a published benchmark, so it is not flagged as contamination risk, only as non-training data. Train on `train.*` (1,739,471 rows).

**Origin**: republished by Jina AI (`jinaai`); code and docstrings are human-written open-source GitHub content collected and filtered automatically by the original CodeSearchNet pipeline, not model-generated [5]. Hub API on the check date: `downloads` 286, `likes` 1 [4].

**Trained-on-by**: none found - a Hugging Face Hub model search filtered on `dataset:jinaai/code_search_net_clean` returns no models, and Jina AI's own `jina-embeddings-v2-base-code` model card names its training data as "more than 150 millions of coding question answer and docstring source code pairs" from unnamed sources, without naming this repository [7][8].

**Introduced by**: the underlying corpus by [1] (Husain et al.); this specific cleaned re-release has no accompanying paper or blog - its own dataset card is the Hugging Face auto-generated placeholder [2].

## Shape

Rows per split, from the repository's own `dataset_info` metadata (`cardData.dataset_info`) [2], matching the live datasets-server size endpoint [6]:

| split | go | java | javascript | php | python | ruby | row total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 311,516 | 417,492 | 109,037 | 466,044 | 389,480 | 45,902 | 1,739,471 |
| test | 14,079 | 24,246 | 5,706 | 25,139 | 21,066 | 2,085 | 92,321 |
| validation | 14,051 | 14,179 | 6,995 | 22,931 | 21,955 | 2,110 | 82,221 |

Grand total 1,914,013 rows, matching the datasets-server `/size` endpoint's `num_rows` [6]. One config, `default`, two columns: `doc` (string) and `code` (string), identical across all 18 splits [2][9].

Byte sizes: the repository's own `dataset_info.download_size` is 632,494,450 bytes, which matches the datasets-server `/size` endpoint's `num_bytes_original_files` and `num_bytes_parquet_files` exactly [2][6]. The repository's declared decoded size is 1,753,788,357 bytes [2], while the datasets-server's own live measurement of decoded size is 1,691,668,055 bytes [6][9] - the two disagree by about 3.5%; both are reported here rather than picking one. No source states token or sequence-length statistics for this dataset.

## Quality

- The dataset card itself states nothing beyond the boilerplate "More Information Needed" placeholder - no stated deduplication method, no stated quality metric, no stated reason the row counts are lower than the original corpus's [2].
- The corpus this repository cleans, `code-search-net/code_search_net`, documents its own filtering pipeline: functions without documentation are dropped, docstrings are truncated to the first paragraph, docstrings shorter than three tokens are dropped, functions shorter than three lines are dropped, functions whose name contains "test" are dropped, constructors and standard extension methods (e.g. `__str__`, `toString`) are dropped, and duplicate or near-duplicate functions are removed to keep one version [5]. This repository's row counts are lower than that corpus's own `all`-config train split (1,739,471 here versus 1,880,853 there) [5], consistent with further deduplication, but no source states what, if anything, was done beyond the original pipeline.
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this specific repository.

## Load it

Train on `train.*`, hold out `test.*`, and pin the revision this card's numbers were read at:

```python
import datasets

REV = "fbb1f7b22b8724c020b8a068f65463a56f2932eb"
train_py = datasets.load_dataset("jinaai/code_search_net_clean", revision=REV, split="train.python")  # 389,480 rows
test_py = datasets.load_dataset("jinaai/code_search_net_clean", revision=REV, split="test.python")    # 21,066 rows - hold out
```

**Trap**: the repository's file tree carries 7 parquet files that the current README's `configs.data_files` glob does not reference - one `data/test-00000-of-00001-f0d44a926cae704e.parquet` and six `data/train-0000{0..5}-of-00006-*.parquet` shards, left over from an earlier flat train/test layout before the repository was reorganized into per-language splits [10]. `load_dataset` with no `data_files` argument correctly ignores them because the active config only globs `data/train.<lang>-*`, `data/test.<lang>-*` and `data/validation.<lang>-*` [2]. A script that globs `data/*.parquet` directly, bypassing `load_dataset`, will pull these stale files in too and double-count or mismatch rows against the eighteen declared splits.

## Neighbors

- `code-search-net/code_search_net` - the original, unclean corpus this repository derives from, with a richer per-function schema (`repository_name`, `func_code_string`, `func_documentation_string`, `func_code_url`, and more) and larger row counts: its `all` config has 1,880,853 train / 100,529 test / 89,154 validation rows, all higher than this repository's per-language totals [5]. Prefer this repository over the original when only the docstring/code pair is needed and a smaller download is preferred; prefer the original when the extra metadata (repository name, file path, GitHub URL) is needed.
- `jinaai/codesearchnet-queries` - a sibling repository from the same org, one `test` split of 92,561 rows with columns `code` and `docs`, close in row count to this repository's `test.*` total (92,321) but not identical, and not documented as the same extraction [11].
- `jinaai/code_search_net_dedupe_only_annotated` - a much smaller sibling (4,374 rows across the six per-language splits) with a richer schema (`nwo`, `sha`, `path`, `docstring`, `score`, and more), a row count consistent with the origin paper's approximately 4,000 expert relevance annotations of likely results across 99 queries [1][12].
- `mteb/CodeSearchNetRetrieval` - a retrieval-benchmark repackaging built from the original `code-search-net/code_search_net` corpus (not from this repository), organized as per-language query/corpus/qrels splits of 1,000 rows each [3]; this is the benchmark the Hold out line above warns against training toward.
- the `CoIR-Retrieval/CodeSearchNet-*` family - a Hub search lists per-language `queries-corpus` and `qrels` repositories under this prefix for all six languages [14]; the one such card fetched for this row is a Hugging-Face auto-generated placeholder with no stated construction method, so no claim is made here about how the family relates to the original corpus's splits [15].

## A row

One config and one schema serve all 18 splits, confirmed identical (`doc`: string, `code`: string) via the datasets-server `/info` endpoint [9]. From `config="default"`, `split="train.python"`, `row_idx=0` (datasets-server `/first-rows`) [13]:

```json
{
  "doc": "Display slug with level by language.",
  "code": "def show_slug_with_level(context, page, lang=None, fallback=True):\n    if not lang:\n        lang = context.get('lang', pages_settings.PAGE_DEFAULT_LANGUAGE)\n\n    page = get_page_from_string_or_id(page, lang)\n    if not page:\n        return ''\n\n    return {'content': page.slug_with_level(lang)}"
}
```

## Where it came from

Republished by Jina AI. The underlying data is the CodeSearchNet Corpus: functions and their documentation collected from public, non-fork, popular open-source GitHub repositories (ranked by libraries.io usage), restricted to repositories whose licence permits redistribution, parsed with Treesitter, and paired with documentation extracted by a heuristic regular expression over Go, Java, JavaScript, PHP, Python and Ruby source [1][5]. The corpus was then filtered to keep only documented, non-trivial, non-test, non-boilerplate functions and deduplicated as described in Quality above [5]. This repository's own card states nothing about what further processing, if any, produced its lower row counts and its two-column `doc`/`code` shape [2].

## Sources

Every source below was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hugging Face Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Husain, Wu, Gazit, Allamanis, Brockschmidt, "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search", 2019. https://arxiv.org/abs/1909.09436 - the origin paper for the corpus; current title read from the live abs page. Fetched 2026-08-12.

[2] jinaai/code_search_net_clean dataset card (README, `cardData.dataset_info` and body). https://huggingface.co/datasets/jinaai/code_search_net_clean/raw/main/README.md - features, split names and row/byte counts, and the boilerplate card body. Fetched 2026-08-12.

[3] mteb/CodeSearchNetRetrieval dataset card. https://huggingface.co/datasets/mteb/CodeSearchNetRetrieval/raw/main/README.md - `source_datasets: code-search-net/code_search_net`, 1,000-row-per-language test-split sampling used for the MTEB retrieval task. Fetched 2026-08-12.

[4] Hugging Face Hub API record for jinaai/code_search_net_clean. https://huggingface.co/api/datasets/jinaai/code_search_net_clean?full=true - licence/tags, gate status, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-12.

[5] code-search-net/code_search_net dataset card (README). https://huggingface.co/datasets/code-search-net/code_search_net/raw/main/README.md - `license:other` tag, collection/filtering pipeline, per-example licensing caveat, `all`-config split row counts. Fetched 2026-08-12.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=jinaai%2Fcode_search_net_clean Fetched 2026-08-12.

[7] Hugging Face Hub model-search API filtered on this dataset. https://huggingface.co/api/models?filter=dataset:jinaai/code_search_net_clean - returns an empty list. Fetched 2026-08-12.

[8] jinaai/jina-embeddings-v2-base-code model card (README). https://huggingface.co/jinaai/jina-embeddings-v2-base-code/raw/main/README.md - stated training-data description, no mention of this dataset by name. Fetched 2026-08-12.

[9] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=jinaai%2Fcode_search_net_clean Fetched 2026-08-12.

[10] Hugging Face Hub tree API for this repository's `data/` directory. https://huggingface.co/api/datasets/jinaai/code_search_net_clean/tree/main/data - lists all 25 parquet files, including the 7 not matched by the current config's `data_files` globs. Fetched 2026-08-12.

[11] jinaai/codesearchnet-queries dataset card and size endpoint. https://huggingface.co/datasets/jinaai/codesearchnet-queries/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=jinaai%2Fcodesearchnet-queries Fetched 2026-08-12.

[12] jinaai/code_search_net_dedupe_only_annotated dataset card and size endpoint. https://huggingface.co/datasets/jinaai/code_search_net_dedupe_only_annotated/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=jinaai%2Fcode_search_net_dedupe_only_annotated Fetched 2026-08-12.

[13] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=jinaai%2Fcode_search_net_clean&config=default&split=train.python Fetched 2026-08-12.

[14] Hugging Face Hub dataset-search API, query "code_search_net". https://huggingface.co/api/datasets?search=code_search_net - lists the `CoIR-Retrieval/CodeSearchNet-{go,java,javascript,php,python,ruby}-queries-corpus` and matching `-qrels` repositories, establishing the family exists. Fetched 2026-08-12.

[15] CoIR-Retrieval/CodeSearchNet-python-queries-corpus dataset card (README), read as representative of the `CoIR-Retrieval/CodeSearchNet-*` family. https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet-python-queries-corpus/raw/main/README.md - a Hugging Face auto-generated placeholder with no stated construction method. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable, restricted to `train.*`: the dataset card itself states no intended use or restriction [2], but the corpus's own test-split reuse as a published retrieval benchmark [3] and the screening row's note both point to the same restriction - train on `train.*`, hold out `test.*`.

### The screening row

The row's own note: "CodeSearchNet functions with their GitHub docstrings, per-language; use the train.* splits, the test.* ones are CodeSearchNet's retrieval benchmark." The row carries no flag.
