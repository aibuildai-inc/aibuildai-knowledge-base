# nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions

4,394 Indian-law instruction rows built from only 268 distinct instructions, where the input is often just a case citation and the model must supply the case from memory.

**nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions** has no card text [1]; its five columns are `instruction`, `input`, `output`, a `prompt` and a `text` field that renders all three into the Alpaca template ("Below is an instruction that describes a task, paired with an input ...") [2][3]. The sampled rows are about Indian cases, for example *Central Inland Water Transport Corporation Ltd. vs Brojo Nath Ganguly* [3]. It lives at https://huggingface.co/datasets/nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions .

**Use it for**: a small Indian-law SFT set, with care: many rows ask the model to analyse a case given only its citation, which trains recall of case content that the prompt does not supply.

**Licence**: Apache 2.0 in the card metadata [4]; the card says nothing else [1]. The one catch: the outputs' authorship is unstated.

**Shape**: 4,394 rows in one `train` split [5]; five columns [2].

**Hold out**: no split is set aside.

**Origin**: unknown; the outputs read as model-written. Hub API at the check date: `downloads` 81, `downloadsAllTime` 3,440, `likes` 29 [4].

**Trained-on-by**: the Hub's dataset tag lists `sartajbhuvaji/Legal-Llama-3.2-3B-Instruct` (10 downloads) [6].

**Introduced by**: no paper and no card [1].

## Shape

| split | rows |
| --- | --- |
| `train` | 4,394 |
| total | 4,394 |

One config, `default` [2]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |
| `prompt` | string |
| `text` | string |

## Quality

- Measured on the full split: only 268 distinct `instruction` values across 4,394 rows (93.9% repeats); the most common instruction appears 1,083 times [7].
- Measured: 167 rows (3.8%) repeat another row's instruction, input and output exactly [7].
- Row 0's `input` is only a citation ("Central Inland Water Transport Corporation Ltd. vs Brojo Nath Ganguly & Anr., 1986 AIR 1571, 1986 SCR (2) 278") and its output states what the Supreme Court held [3] - a recall task with no grounding text.

## Load it

Pin the revision; train on `instruction`/`input`/`output`, not on `text`:

```python
import datasets

REV = "cfadbf50741c89ea05a93183210bfe57ae345cfd"  # main at the check date
ds = datasets.load_dataset("nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions", revision=REV, split="train")  # 4,394 rows
```

**Trap**: `text` already contains the Alpaca template with the answer after "### Response:" and no newline before it [3]; training a chat template on `text` wraps one template inside another. Use the three raw columns.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3], truncated:

```json
{
  "instruction": "Analyze and explain the legal reasoning behind the judgment in the given case.",
  "input": "Central Inland Water Transport Corporation Ltd. vs Brojo Nath Ganguly & Anr., 1986 AIR 1571, 1986 SCR (2) 278",
  "output": "The Supreme Court in this case applied a broad interpretation of the term 'State' under Article 12 of the Constitution. The court reasoned that a government company undertaking public functions qualifies as 'State' based on factors like government co [...]"
}
```

## Where it came from

Uploaded by user nisaar in July 2023 [4]; no source or method is stated [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions dataset card (README). https://huggingface.co/datasets/nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions/raw/main/README.md. Fetched 2026-09-23.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=nisaar%2FLLAMA2_Legal_Dataset_4.4k_Instructions - column schema; live, no revision parameter. Fetched 2026-09-23.

[3] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=nisaar%2FLLAMA2_Legal_Dataset_4.4k_Instructions&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[4] Hugging Face Hub API record for nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions. https://huggingface.co/api/datasets/nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=nisaar%2FLLAMA2_Legal_Dataset_4.4k_Instructions - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[7] This skill's own duplication count on the full `train` split at the pinned revision. Run 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable only as a small, deduplicated supplement for Indian law. Low instruction diversity and citation-only inputs make it a recall exercise with hallucination risk.

### The screening row

The row's own note: "Indian-law Alpaca rows; 268 distinct instructions; inputs are often bare citations." The row carries the flag `low-diversity`.
