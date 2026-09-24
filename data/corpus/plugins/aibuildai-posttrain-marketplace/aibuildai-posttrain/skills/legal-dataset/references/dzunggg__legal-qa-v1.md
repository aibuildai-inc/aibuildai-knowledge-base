# dzunggg/legal-qa-v1

3,742 U.S. lay legal questions with long answers, each prefixed "Q:" and "A:" - no card, no licence, and no stated source.

**dzunggg/legal-qa-v1** has an empty README [1] and two string columns, `question` and `answer` [2]. The questions read like consumer posts on a lawyer Q&A site (a patient discharged from a pain-management office, a houseboat moved without notice) and the answers open with the state they apply to ("In Kentucky, ...") [3]. Nothing on the repository says where either was taken from or who wrote the answers. It lives at https://huggingface.co/datasets/dzunggg/legal-qa-v1 .

**Use it for**: a small SFT set of U.S. consumer legal questions, only if its provenance can be established - its format is convenient, its origin unknown.

**Licence**: none - no licence field and an empty card [4][1]. Ungated [4]. The one catch: with no licence and no source, reuse rights are unknown.

**Shape**: 3,742 rows in one `train` split [5]; two columns [2].

**Hold out**: no split is set aside.

**Origin**: unknown [1]. Hub API at the check date: `downloads` 127, `downloadsAllTime` 4,788, `likes` 11 [4].

**Trained-on-by**: the Hub's dataset tag lists `perctrix/llama3-8b-Lawyer` (0 downloads, 8 likes) [6].

**Introduced by**: no paper and no card [1].

## Shape

| split | rows |
| --- | --- |
| `train` | 3,742 |
| total | 3,742 |

One config, `default` [2]:

| column | dtype |
| --- | --- |
| `question` | string |
| `answer` | string |

## Quality

- Every one of the 100 sampled `question` values starts with "Q:" and the answers with "A:" [3].
- Measured duplication on the full split: 10 questions (0.27%) repeat; 1 full question-answer pair repeats [7].

## Load it

Pin the revision and strip the prefixes:

```python
import datasets

REV = "6280beb74faf5b4dfd1f63adbf7d18908b377b93"  # main at the check date
ds = datasets.load_dataset("dzunggg/legal-qa-v1", revision=REV, split="train")  # 3,742 rows
ds = ds.map(lambda r: {"question": r["question"].removeprefix("Q:").strip(), "answer": r["answer"].removeprefix("A:").strip()})
```

**Trap**: the "Q:" and "A:" prefixes are inside the text [3]; left in, the model learns to begin every answer with "A:".

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3], truncated:

```json
{
  "question": "Q: I was wondering if a pain management office is acting illegally/did an illegal action.. I was discharged as a patient from a pain management office after them telling me that a previous pain management specialist I saw administered a steroid shot wrong and I told them in the portal that I spoke t [...]",
  "answer": "A:In Kentucky, your situation raises questions about patient rights and medical records access. If you were discharged from a pain management office and subsequently lost access to your patient portal, it's important to understand your rights regarding medical records. Under the Health Insurance Por [...]"
}
```

## Where it came from

Uploaded by user dzunggg in January 2024 [4]; no source is named anywhere on the repository [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] dzunggg/legal-qa-v1 dataset card (README). https://huggingface.co/datasets/dzunggg/legal-qa-v1/raw/main/README.md. Fetched 2026-09-23.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=dzunggg%2Flegal-qa-v1 - column schema; live, no revision parameter. Fetched 2026-09-23.

[3] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=dzunggg%2Flegal-qa-v1&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[4] Hugging Face Hub API record for dzunggg/legal-qa-v1. https://huggingface.co/api/datasets/dzunggg/legal-qa-v1?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=dzunggg%2Flegal-qa-v1 - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:dzunggg/legal-qa-v1&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[7] This skill's duplication measurement on the full `train` split, `references/contamination.md` (duplication table). Run 2026-09-23.

## Appendix: screening record

### Screening verdict

Not recommended until its source is known: no licence, no card, and answers of unknown authorship.

### The screening row

The row's own note: "US consumer legal Q&A, no card or licence." The row carries the flag `no-licence`.
