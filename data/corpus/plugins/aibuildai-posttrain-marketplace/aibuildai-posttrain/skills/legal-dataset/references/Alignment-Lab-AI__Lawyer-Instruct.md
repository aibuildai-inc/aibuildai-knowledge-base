# Alignment-Lab-AI/Lawyer-Instruct

9,241 instruction/output pairs described as legal dialogue reformatted from a "LawyerChat" dataset - but most sampled rows are generic reasoning chatter, not legal content.

**Alignment-Lab-AI/Lawyer-Instruct** is described by its card as "a conversational dataset primarily in English, reformatted from the original LawyerChat dataset", reshaped into instruction, input and output fields, and "generated in part by dang/futures" [1]. The card's example is a tax question with a lawyer-style answer [1]. It lives at https://huggingface.co/datasets/Alignment-Lab-AI/Lawyer-Instruct .

**Of the first 100 rows, 57 contain no legal keyword at all; many are variations of "This problem seems like it requires ... generating reasoning traces and task-specific actions in an interleaved manner".**

**Use it for**: nothing without filtering. If used, keep only rows that pass a legal-topic filter and read them first; the unfiltered set teaches meta-commentary about reasoning techniques rather than legal answers.

**Licence**: Apache 2.0 in the card metadata [2]. The card names no licence for the "LawyerChat" source it reformats [1]. The one catch: the upstream source and its terms are unstated.

**Shape**: 9,241 rows in one `train` split [3]; three columns (`instruction`, `input`, `output`) [4].

**Hold out**: no split is set aside; none of the rows are drawn from a benchmark in this skill as far as any source states.

**Origin**: unknown - "generated in part by dang/futures" is the card's only statement [1]; the sampled rows read as model-generated. Hub API at the check date: `downloads` 207, `downloadsAllTime` 4,523, `likes` 19 [2].

**Trained-on-by**: the Hub's dataset tag lists `reaperdoesntknow/Qwen3-0.6B-Distilled-30B-A3B-Thinking-SFT` (4,818 downloads) [5].

**Introduced by**: no paper - the dataset card [1].

## Shape

| split | rows |
| --- | --- |
| `train` | 9,241 |
| total | 9,241 |

One config, `default` [4]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |

## Quality

- Topicality, measured on the 100 rows the viewer returned: 43 contain at least one of a list of legal keywords (law, court, contract, tenant, lawsuit, ...), 57 contain none [6]. Rows 0, 3, 4 and 9 are all variations on "generating reasoning traces and task-specific actions" [7].
- Measured duplication on the full split: 50 rows (0.54%) repeat another row's instruction and output; 190 (2.06%) repeat an instruction [8].
- The card's example row (a question about unpaid taxes) does not match the first rows actually served [1][7].

## Load it

Load, then filter before any use:

```python
import datasets, re

REV = "1dd073c3f7e27633dcafb18ed16467a32f65b515"  # main at the check date
ds = datasets.load_dataset("Alignment-Lab-AI/Lawyer-Instruct", revision=REV, split="train")  # 9,241 rows
LEGAL = re.compile(r"\b(law|legal|court|attorney|lawyer|contract|lawsuit|statute|tenant|landlord|liab)", re.I)
ds = ds.filter(lambda r: bool(LEGAL.search(r["instruction"] + " " + r["output"])))
```

**Trap**: the `input` column is empty in all 100 sampled rows [7]; an Alpaca template that always renders an "### Input:" block trains on thousands of empty input sections.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7] - an off-topic row, shown because it is the first one served:

```json
{
  "instruction": "This problem seems like it requires a lot of dynamic reasoning and incorporating external information. I suggest using the technique of generating reasoning traces and task-specific actions in an interleaved manner. This will allow us to create and adjust plans while also interacting with external sources for additional information.",
  "output": "I agree that we need to be able to explore multiple reasoning paths. The technique of treating the problem as a search over a tree structure could be helpful. We can decompose the problem into intermediate steps and use a search algorithm to find the solution.",
  "input": ""
}
```

## Where it came from

Released by Alignment-Lab-AI in July 2023 [2]; the card credits "dang/futures" with generating part of it and names "LawyerChat" as the source [1]. Neither is linked.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Alignment-Lab-AI/Lawyer-Instruct dataset card (README). https://huggingface.co/datasets/Alignment-Lab-AI/Lawyer-Instruct/raw/main/README.md. Fetched 2026-09-23.

[2] Hugging Face Hub API record for Alignment-Lab-AI/Lawyer-Instruct. https://huggingface.co/api/datasets/Alignment-Lab-AI/Lawyer-Instruct?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Alignment-Lab-AI%2FLawyer-Instruct - takes no revision parameter; a live figure. Fetched 2026-09-23.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Alignment-Lab-AI%2FLawyer-Instruct - column schema; live, no revision parameter. Fetched 2026-09-23.

[5] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:Alignment-Lab-AI/Lawyer-Instruct&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[6] This skill's topicality check on the 100 rows served by the datasets-server first-rows endpoint for `train`: a row counts as legal if instruction or output matches a legal-keyword regex; run on the check date. Fetched 2026-09-23.

[7] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=Alignment-Lab-AI%2FLawyer-Instruct&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[8] This skill's duplication measurement on the full `train` split, `references/contamination.md` (duplication table). Run 2026-09-23.

## Appendix: screening record

### Screening verdict

Not recommended as-is: most sampled rows are off-topic meta-reasoning text and the upstream source is unstated. Usable only after a topic filter and a manual read.

### The screening row

The row's own note: "called legal dialogue; 57/100 sampled rows have no legal content." The row carries the flag `off-topic`.
