# isaacus/open-australian-legal-qa

2,124 Australian legal question-answer pairs synthesised by `gpt-4` from chunks of the Open Australian Legal Corpus, each carrying the source chunk and citation - a small grounded SFT or RAG set.

**isaacus/open-australian-legal-qa** is "the first open dataset of Australian legal questions and answers", "Comprised of 2,124 questions and answers synthesised by `gpt-4` from the Open Australian Legal Corpus" [1]. The card documents its method in full: 2,124 documents were sampled (excluding bills), split into chunks of up to 384 `gpt-4` tokens, one chunk per document was kept, and `gpt-4` was prompted at temperature 0 to write a decontextualised question and an answer "extracted from the snippet" that cites the source document by full name [1]. It lives at https://huggingface.co/datasets/isaacus/open-australian-legal-qa .

**Use it for**: SFT or RAG training for Australian law where the answer should name its source: every row keeps the chunk it came from in `source.text` alongside `question` and `answer` [2]. Small - 2,124 rows - so it suits a targeted mix or an evaluation of grounded answering, not a whole SFT stage.

**Licence**: `other`, the same custom licence as the parent corpus (`license_link` points at the corpus's `LICENCE.md`) [3][1]. Answers were written by `gpt-4` [1], so OpenAI's terms on using model outputs apply on top. The one catch: questions from Federal Court and High Court decisions inherit those courts' non-commercial terms, as in the parent corpus.

**Shape**: 2,124 rows in one `train` split [4]; five columns, one of them a `source` struct holding the document's metadata and the chunk text [2].

**Hold out**: no split is set aside; carve your own. The questions are synthetic and not drawn from any benchmark in this skill.

**Origin**: questions and answers written by `gpt-4`; source text from Australian legislation and court decisions [1]. Hub API at the check date: `downloads` 573, `downloadsAllTime` 18,843, `likes` 23 [3].

**Trained-on-by**: the Hub's dataset tag lists two embedding models, `bugBug04S/legal-embed-modernbert-v2` (117 downloads) and `Yac1n3/bge-base-legal-matryoshka` (34) [5].

**Introduced by**: no paper - the dataset card, which carries a DOI, 10.57967/hf/1479 [3][1].

## Shape

| split | rows |
| --- | --- |
| `train` | 2,124 |
| total | 2,124 |

One config, `default` [2]:

| column | dtype |
| --- | --- |
| `question` | string |
| `answer` | string |
| `text` | string |
| `prompt` | string |
| `source` | struct<version_id: string, type: string, jurisdiction: string, source: string, citation: string, url: string, text: string> |

## Quality

- Answers are constrained by the prompt to be "extracted from the snippet" and to name the source document [1]; row 0's answer cites *Nasr v NRMA Insurance [2006] NSWSC 1018* by name [6].
- Measured duplication: 2 of 2,124 questions repeat another row's question after normalisation (0.09%) [7].
- No correctness check against the source text is reported; the card says malformed `gpt-4` responses were discarded, nothing more [1].

## Load it

One split; pin the revision and split it yourself:

```python
import datasets

REV = "a2178afc2cff8af50fe4365c7d6f9cecf91dbe4b"  # main at the check date
qa = datasets.load_dataset("isaacus/open-australian-legal-qa", revision=REV, split="train")  # 2,124 rows
parts = qa.train_test_split(test_size=0.1, seed=0)
```

**Trap**: the `text` column is `question` and `answer` already joined as "Question: ...\nAnswer: ..." [1][6]. Feeding `text` as the prompt leaks the answer; build the prompt from `question` (and `source.text` for RAG) and use `answer` as the target.

## Neighbors

- `isaacus/open-australian-legal-corpus` - the parent corpus the chunks come from.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [6], with long fields truncated:

```json
{
  "question": "In the case of Nasr v NRMA Insurance [2006] NSWSC 1018, why was the plaintiff's appeal lodged out of time?",
  "answer": "In Nasr v NRMA Insurance [2006] NSWSC 1018, the plaintiff's appeal was lodged out of time because the summons was filed on 8 June 2006, seven months after the decision of the Local Court was made on 4 October 2005. No explanation was provided for thi [...]",
  "text": "Question: In the case of Nasr v NRMA Insurance [2006] NSWSC 1018, why was the plaintiff's appeal lodged out of time?\nAnswer: In Nasr v NRMA Insurance [2006] NSWSC 1018, the plaintiff's appeal was lodged out of time because the summons was filed on 8  [...]",
  "source": {
    "version_id": "nsw_caselaw:549fc6183004262463bb648a",
    "type": "decision",
    "jurisdiction": "new_south_wales",
    "source": "nsw_caselaw",
    "citation": "Nasr v NRMA Insurance [2006] NSWSC 1018",
    "url": "https://www.caselaw.nsw.gov.au/decision/549fc6183004262463bb648a",
    "text": " 3 The plaintiff claims that he was overseas when the Local Court struck out his case against the NRMA and they (the NRMA) rejected payment of his claim for his car after it was burnt on 6 July 2004. There are no grounds of appeal in his summons but  [...]"
  }
}
```

## Where it came from

Built by Umar Butler, now under Isaacus, by prompting `gpt-4` over one sampled chunk from each of 2,124 corpus documents; the card reproduces the full prompt, the decoding settings (`temperature` 0, `max_tokens` 768) and the parsing regex [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] isaacus/open-australian-legal-qa dataset card (README). https://huggingface.co/datasets/isaacus/open-australian-legal-qa/raw/main/README.md. Fetched 2026-09-23.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=isaacus%2Fopen-australian-legal-qa - column schema; live, no revision parameter. Fetched 2026-09-23.

[3] Hugging Face Hub API record for isaacus/open-australian-legal-qa. https://huggingface.co/api/datasets/isaacus/open-australian-legal-qa?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=isaacus%2Fopen-australian-legal-qa - takes no revision parameter; a live figure. Fetched 2026-09-23.

[5] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:isaacus/open-australian-legal-qa&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[6] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=isaacus%2Fopen-australian-legal-qa&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[7] This skill's own duplication count on the full `train` split at the pinned revision - normalised exact match on `question`. Run 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as a small grounded SFT or RAG set for Australian law. Model-written answers under a custom licence; inherit the parent corpus's non-commercial terms for case-law-derived rows.

### The screening row

The row's own note: "gpt-4-synthesised Australian legal QA with source chunks; 2,124 rows." The row carries no flag.
