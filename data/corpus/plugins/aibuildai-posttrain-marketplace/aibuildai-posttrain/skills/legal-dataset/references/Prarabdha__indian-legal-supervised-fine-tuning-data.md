# Prarabdha/indian-legal-supervised-fine-tuning-data

6,055,371 context-question-response triples from Indian judgments, statutes and commentary - the largest SFT set here, described as multilingual and human-curated, though every sampled row is English.

**Prarabdha/indian-legal-supervised-fine-tuning-data** is titled on its card "LegalBrain Indic Legal Corpus", "A large-scale multilingual Indian legal dataset" in `(context, question, response)` form [1]. The card lists Supreme Court and High Court judgments, Law Commission reports, textbooks, legal news and Q&A portals as sources, and describes OCR with Tesseract, MinHash deduplication, and an Argilla workflow to "Fix hallucinations" and "Ensure all responses strictly reference context" [1]. It lives at https://huggingface.co/datasets/Prarabdha/indian-legal-supervised-fine-tuning-data .

**Use it for**: context-grounded SFT for Indian law: the `context` is a passage, the `question` is about it, the `response` answers from it [2][3]. Sample before trusting the scale.

**Licence**: Apache 2.0 in the card metadata [4]. The card says "No proprietary or licensed content was used" [1] but also lists "Public legal textbooks & commentaries" among its sources, and no licence for those is shown. The one catch: the licence of the source passages is not established.

**Shape**: 6,055,371 rows in one `train` split, fully indexed [5]; three columns [2]. 9,101,140,222 bytes of Parquet [5].

**Hold out**: no split is set aside.

**Origin**: passages from Indian legal sources; questions and responses produced by "a prompting pipeline" and then curated, per the card [1]. Hub API at the check date: `downloads` 524, `downloadsAllTime` 3,622, `likes` 8 [4].

**Trained-on-by**: the Hub's dataset tag lists `goasty/Qwen3-4B-Indian-Law` (19 downloads) [6].

**Introduced by**: no paper - the dataset card [1].

## Shape

| split | rows |
| --- | --- |
| `train` | 6,055,371 |
| total | 6,055,371 |

One config, `default` [2]:

| column | dtype |
| --- | --- |
| `context` | string |
| `question` | string |
| `response` | string |

## Quality

- The card lists Hindi, Marathi, Bengali, Tamil, Telugu, Odia and others among its languages [1][4]; none of the 72 rows the viewer served contains any Indic-script character [7].
- Contexts show OCR artefacts - row 0's passage has stray page letters ("D", "C") and "escala\ntion" [3].
- Human curation of 6 million rows is claimed but not quantified [1]; the sampled responses are short, most under 300 characters [3].
- No duplicate rate was measured here: the 9 GB split was not downloaded.

## Load it

Stream it; pin the revision:

```python
import datasets

REV = "05e30cb9536ae6f0d7570c490eb95d73a5117552"  # main at the check date
ds = datasets.load_dataset("Prarabdha/indian-legal-supervised-fine-tuning-data", revision=REV, split="train", streaming=True)
```

**Trap**: the repository name says Indian legal SFT data but the card's title is "LegalBrain Indic Legal Corpus" [1], and a `CyCrawwler/LegalBrain-Indic-Legal-Corpus` repository also exists on the Hub [8]; check which one a plan names before loading.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [3], truncated:

```json
{
  "context": "the judgment of the High Court is set aside subject to the observations D\nmade in our opinions. No orders as to costs.\nH.K. Appeals allowed.\nA DELHI DEVELOPMENT AUTHORITY\nv.\nKANWAR KUMAR MEHTA AND ORS. ETC.\nSEPTEMBER\nB K. RAMASWAMY AND G.B. PATTANAIK, JJ.\nUrban Development\nLand allotted by Delhi Dev [...]",
  "question": "Can you determine if the High Court's decision to calculate interest on the escalation charges equitably was justified?",
  "response": "The High Court's decision to calculate interest on the escalation charges on the principle of equity was held to be justified. The Delhi Development Authority was directed to adjust the interest on the deposits from the date of the original draw till the communication of the letters of allotment."
}
```

## Where it came from

Uploaded by user Prarabdha in March 2025 and last modified October 2025 [4]; the card describes the collection and curation pipeline but links no code [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Prarabdha/indian-legal-supervised-fine-tuning-data dataset card (README). https://huggingface.co/datasets/Prarabdha/indian-legal-supervised-fine-tuning-data/raw/main/README.md. Fetched 2026-09-23.

[2] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Prarabdha%2Findian-legal-supervised-fine-tuning-data - column schema; live, no revision parameter. Fetched 2026-09-23.

[3] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=Prarabdha%2Findian-legal-supervised-fine-tuning-data&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[4] Hugging Face Hub API record for Prarabdha/indian-legal-supervised-fine-tuning-data. https://huggingface.co/api/datasets/Prarabdha/indian-legal-supervised-fine-tuning-data?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Prarabdha%2Findian-legal-supervised-fine-tuning-data - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:Prarabdha/indian-legal-supervised-fine-tuning-data&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[7] This skill's check of the 72 `train` rows served by the datasets-server first-rows endpoint for Devanagari-to-Sinhala script characters (U+0900-U+0DFF); run on the check date. Fetched 2026-09-23.

[8] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=legal&sort=downloads - lists `CyCrawwler/LegalBrain-Indic-Legal-Corpus`. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable with sampling first: a very large English Indian-law context-QA set whose card overstates its multilingual coverage and does not establish the licence of textbook-derived passages.

### The screening row

The row's own note: "6M Indian legal context-QA rows; claims multilingual, sample is English." The row carries no flag.
