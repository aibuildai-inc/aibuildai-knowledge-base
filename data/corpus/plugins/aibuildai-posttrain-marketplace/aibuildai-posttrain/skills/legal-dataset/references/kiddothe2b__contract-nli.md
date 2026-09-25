# kiddothe2b/contract-nli

ContractNLI: non-disclosure agreements labelled against 17 hypotheses as entailment, contradiction or neutral, in two views - the relevant clause (`contractnli_a`) or the whole NDA (`contractnli_b`).

**kiddothe2b/contract-nli** packages ContractNLI, introduced by Koreeda and Manning as "A Dataset for Document-level Natural Language Inference for Contracts" [1]. The loading script defines two configs: in Task A "the input consists of the relevant part of the document w.r.t. to the hypothesis", and in Task B "the input consists of the full document" [2]. Each row is a `premise`, a `hypothesis` (row 0: "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.") and a three-way `label` [3]. The README is 33 bytes [4]. It lives at https://huggingface.co/datasets/kiddothe2b/contract-nli .

**Use it for**: SFT for contract NLI - deciding whether an NDA entails, contradicts or is silent on a standard obligation. Task A trains clause-level judgement; Task B trains finding the relevant clause in a long document.

**Licence**: CC BY-NC-SA 4.0 in the card metadata [5]. The one catch: non-commercial and share-alike.

**Shape**: 20,107 rows: `contractnli_a` 6,819 / 978 / 1,991 and `contractnli_b` 7,191 / 1,037 / 2,091 (`train` / `validation` / `test`) [6]; three columns [3].

**Hold out**: both `test` splits. Measured inside `contractnli_a`: 116 of 1,991 `test` premises (5.83%) appear verbatim in `train`, 72 (3.62%) with the same hypothesis too [7]. LegalBench's `contract_nli_*` tasks are built from these NDAs and part of them sits in `train` [8]; do not train on it if you report those tasks.

**Origin**: NDAs collected from public sources, annotated by the authors, per the paper [1]. Hub API at the check date: `downloads` 527, `downloadsAllTime` 19,379, `likes` 19 [5].

**Trained-on-by**: the Hub's dataset tag lists `tasksource/deberta-small-long-nli` (9,646 downloads) and `tasksource/deberta-base-long-nli` (3,053) [9].

**Introduced by**: [1] (Koreeda and Manning).

## Shape

Rows per config and split (datasets-server `/size`) [6]:

| config | split | rows |
| --- | --- | --- |
| `contractnli_a` | `train` | 6,819 |
| `contractnli_a` | `test` | 1,991 |
| `contractnli_a` | `validation` | 978 |
| `contractnli_b` | `train` | 7,191 |
| `contractnli_b` | `test` | 2,091 |
| `contractnli_b` | `validation` | 1,037 |
| all 2 configs | `train` 14,010 / `test` 4,082 / `validation` 2,015 | 20,107 |

Columns, shared by both configs [3]:

| column | dtype |
| --- | --- |
| `premise` | string |
| `hypothesis` | string |
| `label` | class_label[contradiction, entailment, neutral] |

Task B premises are whole NDAs: `contractnli_b` is 80,681,943 bytes in memory for 7,191 `train` rows against 5,177,216 for Task A's 6,819 [6].

## Quality

- Task B has only 422 distinct premises in `train` - each NDA is paired with all 17 hypotheses [7].
- The same NDAs are in LegalBench, NVIDIA's Nemotron legal set (as a rebuild script) and LawInstruct (whose ContractNLI file has the same 14,010 training rows) [8].

## Load it

Pick a config; the script needs remote code:

```python
import datasets

REV = "059927b91122a6827e7dbb4f296f6da8f5dcee1c"  # main at the check date
train = datasets.load_dataset("kiddothe2b/contract-nli", "contractnli_a", revision=REV, split="train", trust_remote_code=True)  # 6,819
test = datasets.load_dataset("kiddothe2b/contract-nli", "contractnli_a", revision=REV, split="test", trust_remote_code=True)    # 1,991 - hold out
```

**Trap**: the script's homepage URL is written `https://stanfordnlp.github.io/ contract- nli/`, with spaces [2]; code or cards that copy it get a dead link. The project page is https://stanfordnlp.github.io/contract-nli/ .

## Neighbors

- `nguha/legalbench` - its `contract_nli_*` tasks are built from ContractNLI.
- `lawinstruct/lawinstruct` - `ContractNLI-contract_nli-train-0.jsonl.xz` holds the same training rows with translated instructions.

## A row

From `config="contractnli_a"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [10], truncated:

```json
{
  "premise": "2.3 Provided that the Recipient has a written agreement with the following persons or entities requiring them to treat the Confidential Information in accordance with this Agreement, the Recipient may disclose the Confidential Information to: 2.3.1  Any other party with the Discloser’s prior written [...]",
  "hypothesis": "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.",
  "label": 2
}
```

## Where it came from

ContractNLI was built by Yuta Koreeda and Christopher Manning [1]; this Hub packaging by user kiddothe2b downloads `contract_nli.zip` and `contract_nli_long.zip` from the repository [2][11].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Koreeda and Manning, "ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts", arXiv:2110.01799, 2021. https://arxiv.org/abs/2110.01799 - current title read from the live abs page. Fetched 2026-09-23.

[2] kiddothe2b/contract-nli repository file `contract-nli.py`. https://huggingface.co/datasets/kiddothe2b/contract-nli/raw/main/contract-nli.py. Fetched 2026-09-23.

[3] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=kiddothe2b%2Fcontract-nli - column schema; live, no revision parameter. Fetched 2026-09-23.

[4] kiddothe2b/contract-nli dataset card (README). https://huggingface.co/datasets/kiddothe2b/contract-nli/raw/main/README.md. Fetched 2026-09-23.

[5] Hugging Face Hub API record for kiddothe2b/contract-nli. https://huggingface.co/api/datasets/kiddothe2b/contract-nli?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=kiddothe2b%2Fcontract-nli - takes no revision parameter; a live figure. Fetched 2026-09-23.

[7] This skill's own exact split-leakage count at the pinned revision. Fetched 2026-09-23.

[8] This skill's own check by n-gram containment against the benchmark's test items (word 8-grams, or character 15-grams for Chinese), with self, words-sorted and positive controls, at the pinned revision. Run 2026-09-23.

[9] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:kiddothe2b/contract-nli&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[10] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=kiddothe2b%2Fcontract-nli&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[11] Repository file tree for kiddothe2b/contract-nli. https://huggingface.co/api/datasets/kiddothe2b/contract-nli/tree/main?recursive=true. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable for non-commercial contract-NLI SFT: train on `train`, hold out `test`; decontaminate against LegalBench `contract_nli_*` before reporting it.

### The screening row

The row's own note: "ContractNLI NDAs, two views; NC-SA." The row carries no flag.
