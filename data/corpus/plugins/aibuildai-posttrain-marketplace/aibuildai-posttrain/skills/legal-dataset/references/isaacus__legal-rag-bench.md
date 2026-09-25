# isaacus/legal-rag-bench

Legal RAG Bench: 100 expert-written questions on Victorian (Australian) criminal law with answers and the relevant passage, over a 4,876-passage corpus from the Judicial College of Victoria's Criminal Charge Book.

**isaacus/legal-rag-bench** is "a reasoning-intensive benchmark for assessing the end-to-end, real-world performance of production-grade legal RAG systems" by Isaacus [1], described in "Legal RAG Bench: an end-to-end benchmark for legal RAG" [2]. It pairs "4,876 passages sampled from the Judicial College of Victoria's Criminal Charge Book" with "100 complex, handwritten questions demanding expert-level knowledge of Victorian criminal law and procedure", labelling both the answer and the relevant passage [1]. It lives at https://huggingface.co/datasets/isaacus/legal-rag-bench .

**Use it for**: evaluating legal RAG end to end, separating retrieval errors from generation errors because the gold passage is labelled [1].

**Licence**: CC BY-NC-SA 4.0 [3]. The one catch: non-commercial and share-alike.

**Shape**: two configs, each a single `test` split: `corpus` 4,876 passages, `qa` 100 questions [4].

**Hold out**: all of it.

**Origin**: passages from the Criminal Charge Book; questions and answers hand-written by experts [1]. Hub API at the check date: `downloads` 814, `downloadsAllTime` 6,560, `likes` 25 [3].

**Trained-on-by**: an evaluation set; no model declares it [5]. The card reports Isaacus's own Kanon 2 Embedder as making the fewest errors [1].

**Introduced by**: [2] (Isaacus).

## Shape

Rows per config (datasets-server `/size`) [4]:

| config | split | rows |
| --- | --- | --- |
| `corpus` | `test` | 4,876 |
| `qa` | `test` | 100 |
| all 2 configs | `test` 4,976 | 4,976 |

Columns of `qa` [6]:

| column | dtype |
| --- | --- |
| `id` | int64 |
| `question` | string |
| `answer` | string |
| `relevant_passage_id` | string |

## Quality

- 100 questions: a single question moves accuracy by one point.
- The benchmark's publisher also sells the embedder it reports as best [1]; the labels are public, so results can be checked independently.

## Load it

Evaluate on `test`:

```python
import datasets

REV = "db0b31dc6d195ce9916897e1ac5e4e6209736c8a"  # main at the check date
corpus = datasets.load_dataset("isaacus/legal-rag-bench", "corpus", revision=REV, split="test")  # 4,876 passages
qa = datasets.load_dataset("isaacus/legal-rag-bench", "qa", revision=REV, split="test")          # 100 questions
```

**Trap**: the `corpus` config is the default [3]; `load_dataset("isaacus/legal-rag-bench")` returns passages, not questions.

## A row

From `config="qa"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [7]:

```json
{
  "id": 1,
  "question": "Bob and Ted are close friends. Ted is on trial for drug offences, and Bob has been selected as a juror in Ted’s case. Is the judge required to excuse Bob from serving on the jury?",
  "answer": "No. While the bench book instructs judges to inform members of the jury panel that they can excuse themselves if they know the accused, this is not mandatory. Instead, the court may excuse a potential juror if they are satisfied that the person will not be able to consider the case impartially.",
  "relevant_passage_id": "1.2-c2-s2"
}
```

## Where it came from

Built by Isaacus from the Judicial College of Victoria's Criminal Charge Book [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] isaacus/legal-rag-bench dataset card (README). https://huggingface.co/datasets/isaacus/legal-rag-bench/raw/main/README.md. Fetched 2026-09-23.

[2] Butler and Butler, "Legal RAG Bench: an end-to-end benchmark for legal RAG", arXiv:2603.01710, 2026. https://arxiv.org/abs/2603.01710 - current title read from the live abs page. Fetched 2026-09-23.

[3] Hugging Face Hub API record for isaacus/legal-rag-bench. https://huggingface.co/api/datasets/isaacus/legal-rag-bench?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=isaacus%2Flegal-rag-bench - takes no revision parameter; a live figure. Fetched 2026-09-23.

[5] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:isaacus/legal-rag-bench&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=isaacus%2Flegal-rag-bench - column schema; live, no revision parameter. Fetched 2026-09-23.

[7] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=isaacus%2Flegal-rag-bench&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

An evaluation set for legal RAG: hold out. Small (100 questions), NC-SA.

### The screening row

The row's own note: "Victorian criminal-law RAG benchmark, 100 questions." The row carries no flag.
