# ricdomolm/lawma-instructions

553,419 training and 1,000 validation instructions rendering the Lawma legal classification tasks as prompt text with a one-letter answer - the SFT-ready form of `ricdomolm/lawma-tasks`.

**ricdomolm/lawma-instructions** is the instruction-formatted companion to the Lawma tasks [1][2]: each row is a `task` name, an `instruction` that contains the codebook question, the court opinion and lettered choices ending in "Answer:", and an `output` letter [3]. Its README carries only the `dataset_info` front matter and no prose [4]. It lives at https://huggingface.co/datasets/ricdomolm/lawma-instructions .

**Use it for**: direct SFT on legal classification: the rows are already prompt/answer pairs, so a completion-only trainer can take `instruction` as the prompt and `output` as the target. Pair it with held-out Lawma `test` splits or CaselawQA for evaluation.

**Licence**: none on the card - no `license` field and no licence text [5][4]. Ungated [5]. The one catch: no licence is stated; the parent `ricdomolm/lawma-tasks` is MIT [6], which is the best evidence of the author's intent.

**Shape**: 554,419 rows: `train` 553,419 / `validation` 1,000 [7]; three columns (`task`, `output`, `instruction`) [8].

**Hold out**: `validation` (1,000 rows). No source says how this split relates to the parent's `val` and `test` splits; the name and the parent's design suggest it is drawn from training-side data, but that is not stated. Screen against `ricdomolm/caselawqa-8k` before a scored run.

**Origin**: the same expert-coded Supreme Court Database and Songer Court of Appeals labels and U.S. opinions as the parent, formatted by the Lawma authors [2]. Hub API at the check date: `downloads` 223, `downloadsAllTime` 910, `likes` 0 [5].

**Trained-on-by**: the Lawma paper's models are trained on the Lawma tasks [2]; no model on the Hub declares this repository through its dataset tag [9].

**Introduced by**: [2] (Dominguez-Olmedo et al.), via the parent tasks.

## Shape

| split | rows |
| --- | --- |
| `train` | 553,419 |
| `validation` | 1,000 |
| total | 554,419 |

One config, `default` [8]:

| column | dtype |
| --- | --- |
| `task` | string |
| `output` | string |
| `instruction` | string |

The sampled rows are mostly Songer tasks: of the 11 `train` rows the viewer returned, 9 are `songer_*` and 2 `sc_*` [3], consistent with the parent's roughly 4-to-1 ratio of Songer to Supreme Court training rows [10].

## Quality

- Answers are single letters, and each carries a leading space - row 0's `output` is `" F"` [3] - matching the "Answer:" that ends every instruction.
- Instructions embed entire opinions, so rows are long: the `train` split is 5,948,194,381 bytes of Parquet for 553,419 rows [7].
- No source states a measured duplicate rate, or how the 553,419 training rows relate to the parent's 484,786 `train` rows.

## Load it

Train on `train`, hold out `validation`, pin the revision:

```python
import datasets

REV = "d35a9164bc698ba948c51a61b08c63a886eb3605"  # main at the check date
train = datasets.load_dataset("ricdomolm/lawma-instructions", revision=REV, split="train", streaming=True)  # 553,419 rows
val = datasets.load_dataset("ricdomolm/lawma-instructions", revision=REV, split="validation")             # 1,000 rows - hold out
```

**Trap**: the leading space in `output` is part of the target [3]. A template that inserts its own space after "Answer:" produces "Answer:  F" with two spaces and teaches the model a token sequence it will never see at evaluation. Concatenate `instruction + output` exactly as stored.

## Neighbors

- `ricdomolm/lawma-tasks` - the raw tasks with `opinion`, `choices` and index `answer`, and the `test` splits to hold out.
- `ricdomolm/lawma-instructions_llama3_8k`, `_llama3_16k`, `_gemma2_8k`, `_pythia_2k` and others - per-tokenizer length-truncated variants found by the Hub search [11]; not screened here.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3], with `instruction` truncated:

```json
{
  "task": "songer_circuit",
  "output": " F",
  "instruction": "What follows is an opinion from a United States Court of Appeals. Your task is to identify the circuit of the court that decided the case.\n\nRobert MORRIS et al., Appellants, v. WERNER-CONTINENTAL, INC., et al., Appellees.\nNo. 71-2044.\nUnited States Court of Appeals, Sixth Circuit.\nSept. 20, 1972.\nStanley H. Sidicane, Nashville, Tenn., on brief for appellants.\nGeorge W. Weber, Jr., Cincinnati, Ohio [...]"
}
```

## Where it came from

Released by Ricardo Dominguez-Olmedo alongside the Lawma tasks; the code that renders instructions is in the Lawma repository, https://github.com/socialfoundations/lawma [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] ricdomolm/lawma-tasks dataset card (README). https://huggingface.co/datasets/ricdomolm/lawma-tasks/raw/main/README.md. Fetched 2026-09-23.

[2] Dominguez-Olmedo et al., "Lawma: The Power of Specialization for Legal Annotation", arXiv:2407.16615, 2024. https://arxiv.org/abs/2407.16615 - current title read from the live abs page. Fetched 2026-09-23.

[3] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=ricdomolm%2Flawma-instructions&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[4] ricdomolm/lawma-instructions dataset card (README). https://huggingface.co/datasets/ricdomolm/lawma-instructions/raw/main/README.md. Fetched 2026-09-23.

[5] Hugging Face Hub API record for ricdomolm/lawma-instructions. https://huggingface.co/api/datasets/ricdomolm/lawma-instructions?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[6] Hugging Face Hub API record for ricdomolm/lawma-tasks. https://huggingface.co/api/datasets/ricdomolm/lawma-tasks?full=true - `license: mit`. Fetched 2026-09-23.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=ricdomolm%2Flawma-instructions - takes no revision parameter; a live figure. Fetched 2026-09-23.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=ricdomolm%2Flawma-instructions - column schema; live, no revision parameter. Fetched 2026-09-23.

[9] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:ricdomolm/lawma-instructions&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[10] datasets-server size endpoint for ricdomolm/lawma-tasks. https://datasets-server.huggingface.co/size?dataset=ricdomolm%2Flawma-tasks. Fetched 2026-09-23.

[11] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=lawma&sort=downloads. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as SFT data for legal classification in instruction form. No licence is stated on this repository; the MIT parent is the fallback. Screen against CaselawQA before reporting it.

### The screening row

The row's own note: "SFT-ready Lawma instructions; no licence field; split provenance unstated." The row carries the flag `no-licence`.
