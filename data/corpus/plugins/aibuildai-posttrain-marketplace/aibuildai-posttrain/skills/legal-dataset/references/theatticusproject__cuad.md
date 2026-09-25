# theatticusproject/cuad

The raw CUAD v1 release - 510 contract PDFs and text files, the master JSON and CSV labels, and the datasheet - which the dataset viewer serves as 84,325 meaningless lines of text.

**theatticusproject/cuad** is the file release of CUAD v1 from The Atticus Project: a `CUAD_v1/` directory holding `CUAD_v1.json`, full-contract PDFs and text files grouped by agreement type, and the dataset's README and datasheet [1]. Its own README is 27 bytes [2]. The QA-formatted version is `theatticusproject/cuad-qa`; the paper is Hendrycks et al. [3]. It lives at https://huggingface.co/datasets/theatticusproject/cuad .

**Use it for**: the raw contracts and labels, when you want to build your own clause-extraction format or use full contracts as continued-pretraining text. For ready QA rows use `theatticusproject/cuad-qa`.

**Licence**: CC BY 4.0 in the card metadata [4]; the CUAD README inside the release carries the full terms [1].

**Shape**: 743 files under `CUAD_v1/` [1]. The viewer auto-loads every text file line by line and reports one `train` split of 84,325 rows with a single `text` column [5][6] - lines, not examples.

**Hold out**: the release does not mark a split; `cuad-qa`'s split uses 408 training and 102 test contracts [7]. LegalBench's `cuad_*` tasks are built from these contracts [8]; do not train on them if you report those tasks.

**Origin**: EDGAR contracts; labels by The Atticus Project annotators [3]. Hub API at the check date: `downloads` 7,078, `downloadsAllTime` 65,424, `likes` 50 [4].

**Trained-on-by**: the Hub's dataset tag lists contract models such as `litillabs/litil-contract-extractor-1.7b` (41 downloads) and `Mjolnirslams/roberta-base-squad2-cuad` (27) [9].

**Introduced by**: [3] (Hendrycks et al.).

## Shape

| split | rows |
| --- | --- |
| `train` | 84,325 |
| total | 84,325 |

What the viewer serves [6]:

| column | dtype |
| --- | --- |
| `text` | string |

`CUAD_v1/CUAD_v1.json` is 40,128,638 bytes and holds all 510 contracts with 20,910 questions [1][10].

## Quality

- Row 0 as served is a line of "=" characters [11]; the viewer's rows are lines of whatever text files it found, in file order.

## Load it

Download the files; do not use `load_dataset` on this repository:

```python
from huggingface_hub import hf_hub_download
import json

REV = "a3c393f5d103fd0c516374e4fdff676c8176dcb1"  # main at the check date
f = hf_hub_download("theatticusproject/cuad", "CUAD_v1/CUAD_v1.json", repo_type="dataset", revision=REV)
cuad = json.load(open(f))  # SQuAD-style: 510 contracts, 20,910 questions
```

**Trap**: `load_dataset("theatticusproject/cuad")` succeeds and returns 84,325 rows of single lines from the text files [5][11] - it looks like data and trains on nothing coherent.

## Neighbors

- `theatticusproject/cuad-qa` - the QA-formatted train/test release.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [11]:

```json
{
  "text": "================================================="
}
```

## Where it came from

Released by The Atticus Project in January 2023 on the Hub [4]; the same data is at https://github.com/TheAtticusProject/cuad [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Repository file tree for theatticusproject/cuad. https://huggingface.co/api/datasets/theatticusproject/cuad/tree/main?recursive=true. Fetched 2026-09-23.

[2] theatticusproject/cuad dataset card (README). https://huggingface.co/datasets/theatticusproject/cuad/raw/main/README.md. Fetched 2026-09-23.

[3] Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review", arXiv:2103.06268, 2021. https://arxiv.org/abs/2103.06268 - current title read from the live abs page. Fetched 2026-09-23.

[4] Hugging Face Hub API record for theatticusproject/cuad. https://huggingface.co/api/datasets/theatticusproject/cuad?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=theatticusproject%2Fcuad - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=theatticusproject%2Fcuad - column schema; live, no revision parameter. Fetched 2026-09-23.

[7] The CUAD archive https://github.com/TheAtticusProject/cuad/raw/main/data.zip - `train_separate_questions.json` (408 contracts) and `test.json` (102). Fetched 2026-09-23.

[8] This skill's own check by n-gram containment against the benchmark's test items (word 8-grams, or character 15-grams for Chinese), with self, words-sorted and positive controls, at the pinned revision. Run 2026-09-23.

[9] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:theatticusproject/cuad&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[10] theatticusproject/cuad file `CUAD_v1/CUAD_v1.json`, downloaded - contracts and questions counted by this skill. Fetched 2026-09-23.

[11] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=theatticusproject%2Fcuad&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as the raw CUAD release; ignore what `load_dataset` and the viewer return and read the files.

### The screening row

The row's own note: "raw CUAD v1 files; viewer rows are text lines." The row carries no flag.
