# AdaptLLM/law-tasks

The three legal evaluation sets from AdaptLLM - CaseHOLD, SCOTUS and UNFAIR-ToS, 5,216 test prompts already rendered with few-shot examples - test-only.

**AdaptLLM/law-tasks** "contains the evaluation datasets" for "Adapting Large Language Models to Domains via Reading Comprehension" by Cheng et al. [1][2]. Its three configs are `CaseHOLD` (3,549), `SCOTUS` (60) and `UNFAIR_ToS` (1,607), each a single `test` split [3]. It lives at https://huggingface.co/datasets/AdaptLLM/law-tasks .

**Test-only evaluation data. Its `UNFAIR_ToS` config has exactly the 1,607 rows of LexGLUE's `unfair_tos` test split, and CaseHOLD and UNFAIR-ToS test items are measured inside the NVIDIA Nemotron legal set and Pile of Law's `tos` config.**

**Use it for**: reproducing AdaptLLM's legal evaluation. Prompts are pre-rendered; `UNFAIR_ToS` inputs include several solved few-shot examples before the query [4].

**Licence**: none on the card [5]. The one catch: no licence; the upstream LexGLUE is CC BY 4.0.

**Shape**: 5,216 rows: `CaseHOLD` 3,549; `SCOTUS` 60; `UNFAIR_ToS` 1,607 (all `test`) [3].

**Hold out**: all of it. `UNFAIR_ToS` has 1,607 rows, the size of LexGLUE's `unfair_tos` test split [3][6].

**Origin**: legal benchmark items reformatted into prompts by the AdaptLLM authors; the card names the tasks but not their source splits [1]. Hub API at the check date: `downloads` 246, `downloadsAllTime` 14,939, `likes` 37 [5].

**Trained-on-by**: an evaluation set; no model declares it [7].

**Introduced by**: [2] (Cheng et al.).

## Shape

Rows per config (datasets-server `/size`) [3]:

| config | split | rows |
| --- | --- | --- |
| `CaseHOLD` | `test` | 3,549 |
| `SCOTUS` | `test` | 60 |
| `UNFAIR_ToS` | `test` | 1,607 |
| all 3 configs | `test` 5,216 | 5,216 |

Columns of `SCOTUS` [8]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `input` | string |
| `options` | list<string> |
| `gold_index` | int64 |

## Quality

- `SCOTUS` has 60 rows [3]; a score on it moves in steps of 1.7 points.
- `CaseHOLD` stores the prompt as `input_options`, one full prompt per candidate, while the other configs have `input` and `options` [8].

## Load it

Evaluate on `test`:

```python
import datasets

REV = "9833f318db11941509f2b56a29b423757f8acbd9"  # main at the check date
unfair = datasets.load_dataset("AdaptLLM/law-tasks", "UNFAIR_ToS", revision=REV, split="test")  # 1,607 rows
```

**Trap**: `UNFAIR_ToS` `gold_index` is a list (multi-label), while `SCOTUS` `gold_index` is an integer [8]; one scoring function for both fails.

## A row

From `config="SCOTUS"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [4], truncated:

```json
{
  "id": 0,
  "input": "Analyze the following opinion from the Supreme Court of USA (SCOTUS): 503 U.S. 257\n112 S.Ct. 1151\n117 L.Ed.2d 400\nPFZ PROPERTIES, INC., Petitioner,v.Rene Alberto RODRIGUEZ, et al.\nNo. 91-122.\nMarch 9, 1992.\n\nOn Writ of Certiorari to the United States Court of Appeals for the First Circuit.\nCase below, 928 F.2d 28.\nPER CURIAM.\n\n\n1\nThe writ of certiorari is dismissed as improvidently granted.\n\n\nWhat [...]",
  "options": [
    "Criminal Procedure",
    "Civil Rights",
    "First Amendment",
    "Due Process",
    "Privacy",
    "Attorneys",
    "..."
  ],
  "gold_index": 8
}
```

## Where it came from

Released by the AdaptLLM authors with their benchmarking code at https://github.com/microsoft/LMOps [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] AdaptLLM/law-tasks dataset card (README). https://huggingface.co/datasets/AdaptLLM/law-tasks/raw/main/README.md. Fetched 2026-09-23.

[2] Cheng et al., "Adapting Large Language Models to Domains via Reading Comprehension", arXiv:2309.09530, 2023. https://arxiv.org/abs/2309.09530 - current title read from the live abs page. Fetched 2026-09-23.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=AdaptLLM%2Flaw-tasks - takes no revision parameter; a live figure. Fetched 2026-09-23.

[4] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=AdaptLLM%2Flaw-tasks&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[5] Hugging Face Hub API record for AdaptLLM/law-tasks. https://huggingface.co/api/datasets/AdaptLLM/law-tasks?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[6] datasets-server size endpoint for coastalcph/lex_glue. https://datasets-server.huggingface.co/size?dataset=coastalcph%2Flex_glue - `unfair_tos` `test` 1,607. Fetched 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:AdaptLLM/law-tasks&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=AdaptLLM%2Flaw-tasks - column schema; live, no revision parameter. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

An evaluation set from the CaseHOLD, SCOTUS and UNFAIR-ToS tasks: hold out, and decontaminate any mix containing the Nemotron legal set or Pile of Law's `tos` config before reporting it.

### The screening row

The row's own note: "AdaptLLM law eval prompts from LexGLUE test." The row carries no flag.
