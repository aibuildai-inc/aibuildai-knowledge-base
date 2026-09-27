# jhu-clsp/CLERC

CLERC: a legal case retrieval and retrieval-augmented analysis-generation benchmark from the Caselaw Access Project - query and passage collections, and a generation task where a model continues an opinion's analysis given the cases it cites.

**jhu-clsp/CLERC** from Johns Hopkins CLSP has a README that reads "README in progress" and a usage note: the data is "in folder according to the task and type (e.g. `generation` or `collection` for IR)" [1]. The repository holds IR collections (`collection/collection.doc.tsv.gz`, 8.2 GB; `collection/collection.passage.tsv.gz`, 9.2 GB), queries and qrels for test, training triples, and a `generation/` folder with `train.jsonl`, `test.jsonl` and `all.jsonl` [2]. It lives at https://huggingface.co/datasets/jhu-clsp/CLERC .

**Use it for**: training and evaluating legal case retrieval, or retrieval-augmented generation of legal analysis: given the text before a citation and the cited cases, write the next passage (`gold_text`) [3].

**Licence**: none on the card [4][1]. The texts come from the Caselaw Access Project. The one catch: no licence is stated on this repository.

**Shape**: `generation/test.jsonl` has 1,000 rows with fields `docid`, `previous_text`, `gold_text`, `citations`, `short_citations` [3]; `generation/train.jsonl` is 512,111,830 bytes and `all.jsonl` 5,349,346,981 [2]. The viewer auto-loads a `partial` 105,699-row `train` split with IR-style columns (`query_id`, `query`, `positive_passages`, `negative_passages`) [5][6].

**Hold out**: `generation/test.jsonl` and the `queries/test.*` and `qrels/*.test.*` files [2]. `generation/all.jsonl` contains everything - do not train on it.

**Origin**: U.S. opinions from the Caselaw Access Project, with citations extracted to form queries and targets [1][2]. Hub API at the check date: `downloads` 640, `downloadsAllTime` 40,415, `likes` 11 [4].

**Trained-on-by**: no model on the Hub declares this dataset through its dataset tag [7].

**Introduced by**: no paper is linked from the card, which reads "README in progress" [1].

## Shape

| split | rows |
| --- | --- |
| `train` | 105,699 |
| total | 105,699 |

What the viewer auto-loads (a partial `train`) [6]:

| column | dtype |
| --- | --- |
| `query_id` | string |
| `query` | string |
| `positive_passages` | list<struct<docid: string, title: string, text: string>> |
| `negative_passages` | list<struct<docid: string, title: string, text: string>> |

## Quality

- The viewer's automatic `train` split mixes whatever files it resolved first into an IR schema [5]; it is not the generation task and not a documented split.

## Load it

Load a specific file, as the card instructs, pinned:

```python
import datasets

REV = "ef042f8ab436f78704f17faa0a866d1b2b862f6f"  # main at the check date
gen_test = datasets.load_dataset("jhu-clsp/CLERC", data_files={"test": "generation/test.jsonl"}, revision=REV)["test"]   # 1,000 rows - hold out
gen_train = datasets.load_dataset("jhu-clsp/CLERC", data_files={"train": "generation/train.jsonl"}, revision=REV)["train"]
```

**Trap**: a plain `load_dataset("jhu-clsp/CLERC")` returns the viewer's auto-detected split, not any task [5]. Always pass `data_files`.

## A row

From the viewer's auto-loaded `train` (IR schema), `row_idx=0` (datasets-server `/first-rows`) [8], truncated:

```json
{
  "query_id": "247774",
  "query": "Since Greene, this court has rejected similar charges of misconduct where the government supplied counterfeit credit cards to detect which merchants would accept them. See Citro, 842 F.2d at 1153. In a case where an FBI agent bribed a state senator,  [...]"
}
```

## Where it came from

Released by Johns Hopkins CLSP in May 2024 [4]; the collections are built from the Caselaw Access Project [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] jhu-clsp/CLERC dataset card (README). https://huggingface.co/datasets/jhu-clsp/CLERC/raw/main/README.md. Fetched 2026-09-23.

[2] Repository file tree for jhu-clsp/CLERC. https://huggingface.co/api/datasets/jhu-clsp/CLERC/tree/main?recursive=true. Fetched 2026-09-23.

[3] jhu-clsp/CLERC file `generation/test.jsonl`, downloaded - 1,000 rows counted and fields read. Fetched 2026-09-23.

[4] Hugging Face Hub API record for jhu-clsp/CLERC. https://huggingface.co/api/datasets/jhu-clsp/CLERC?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=jhu-clsp%2FCLERC - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=jhu-clsp%2FCLERC - column schema; live, no revision parameter. Fetched 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:jhu-clsp/CLERC&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=jhu-clsp%2FCLERC&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable for legal retrieval and citation-grounded generation, loaded file by file. No licence and no documentation yet.

### The screening row

The row's own note: "CLERC retrieval + generation; README in progress; no licence." The row carries the flag `no-licence`.
