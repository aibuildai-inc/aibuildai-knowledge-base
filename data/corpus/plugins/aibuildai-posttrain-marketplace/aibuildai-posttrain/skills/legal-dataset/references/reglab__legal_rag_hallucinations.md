# reglab/legal_rag_hallucinations

The public half of Stanford RegLab's legal-research-tool hallucination study: 400 responses from commercial legal AI tools to 100 queries, each labelled Accurate, Incomplete or Hallucination - plus the 100 queries alone.

**reglab/legal_rag_hallucinations** releases "the queries and raw model outputs" analysed in "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools" by Magesh et al. [1][2]. "This public dataset contains 50% of the data analyzed in our work: 400 responses to 100 different queries"; the other half is withheld "to guard against potential model memorization" [1]. It lives at https://huggingface.co/datasets/reglab/legal_rag_hallucinations .

**Use it for**: evaluating hallucination detection, or studying failure modes of legal RAG products. Too small to train on, and its responses are other systems' outputs.

**Licence**: CC BY 4.0 in the card metadata [3]; the card's licence section applies CC BY 4.0 to "The questions in the dataset" and argues fair use for 10 historical bar-exam prep questions [1]. The one catch: the responses are commercial tools' outputs, which the licence statement does not mention.

**Shape**: the viewer merges two files into one 500-row `train` split with eight columns [4][5]: `dataset.csv` (400 labelled responses) and `questions.csv` (100 questions) [1][6].

**Hold out**: all of it; the authors hold back the other half themselves [1].

**Origin**: queries written by the researchers; responses from legal AI research tools; labels by the researchers following the paper's protocol [1]. Hub API at the check date: `downloads` 68, `downloadsAllTime` 807, `likes` 1 [3].

**Trained-on-by**: an evaluation set; no model declares it [7].

**Introduced by**: [2] (Magesh et al.).

## Shape

| split | rows |
| --- | --- |
| `train` | 500 |
| total | 500 |

The merged schema the viewer serves [5]:

| column | dtype |
| --- | --- |
| `Question ID` | string |
| `Question Category` | string |
| `Model` | string |
| `Question` | string |
| `Response` | string |
| `Correctness` | string |
| `Groundedness` | string |
| `Label` | string |

## Quality

- All 80 rows the viewer returned are `Model` "Westlaw" [8]; the file is ordered by tool, so a head sample is not representative.
- In those rows the `Label` split is 34 Accurate, 32 Hallucination, 14 Incomplete [8].

## Load it

Read the two CSV files separately:

```python
import datasets

REV = "6364efad0dede984e9f4aeb1ca07cffb7fa85a77"  # main at the check date
resp = datasets.load_dataset("reglab/legal_rag_hallucinations", data_files="dataset.csv", revision=REV, split="train")    # 400 rows
qs = datasets.load_dataset("reglab/legal_rag_hallucinations", data_files="questions.csv", revision=REV, split="train")    # 100 rows
```

**Trap**: a plain `load_dataset` concatenates `dataset.csv` and `questions.csv` into one 500-row split [4][6]; the 100 question-only rows have no `Model`, `Response` or `Label` and silently lower any rate computed over the split.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [8], truncated:

```json
{
  "Question ID": "scalr-1",
  "Question Category": "SCALR",
  "Model": "Westlaw",
  "Question": "Does the inevitable discovery doctrine create a per se exception to the exclusionary rule for evidence seized after a Fourth Amendment \"knock and announce\" violation or is evidence subject to suppression after such violations?",
  "Response": "The application of the inevitable discovery doctrine to \"knock and announce\" violations under the Fourth Amendment is not uniform across jurisdictions (\nLee v. State, 139 Md.App. 79 (2001))[1], (\nPeople v. Hoag, 83 Cal.App.4th 1198 (2000))[2]. Some courts, such as the Supreme Court of Michigan in Pe [...]",
  "Correctness": "Incorrect",
  "Groundedness": null,
  "Label": "Hallucination"
}
```

## Where it came from

Released by Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher Manning and Daniel E. Ho [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] reglab/legal_rag_hallucinations dataset card (README). https://huggingface.co/datasets/reglab/legal_rag_hallucinations/raw/main/README.md. Fetched 2026-09-23.

[2] Magesh et al., "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools", arXiv:2405.20362, 2024. https://arxiv.org/abs/2405.20362 - current title read from the live abs page. Fetched 2026-09-23.

[3] Hugging Face Hub API record for reglab/legal_rag_hallucinations. https://huggingface.co/api/datasets/reglab/legal_rag_hallucinations?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=reglab%2Flegal_rag_hallucinations - takes no revision parameter; a live figure. Fetched 2026-09-23.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=reglab%2Flegal_rag_hallucinations - column schema; live, no revision parameter. Fetched 2026-09-23.

[6] Repository file tree for reglab/legal_rag_hallucinations. https://huggingface.co/api/datasets/reglab/legal_rag_hallucinations/tree/main?recursive=true. Fetched 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:reglab/legal_rag_hallucinations&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=reglab%2Flegal_rag_hallucinations&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

An evaluation set for legal hallucination labelling: hold out; load its two files separately.

### The screening row

The row's own note: "public half of the legal RAG hallucination study." The row carries no flag.
