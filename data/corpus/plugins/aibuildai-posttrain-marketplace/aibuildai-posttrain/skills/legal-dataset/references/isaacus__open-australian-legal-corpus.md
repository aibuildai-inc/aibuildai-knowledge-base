# isaacus/open-australian-legal-corpus

Every in-force statute and regulation of seven Australian jurisdictions plus about 189,000 court decisions, as plain text with citation, jurisdiction, and document-type fields - the only large open corpus of Australian law.

**isaacus/open-australian-legal-corpus** is a corpus of Australian legislative and judicial documents built by Umar Butler and now maintained by Isaacus [1][2]. Its statistics section reports 232,560 documents, 69,539,311 lines and 1,470,936,197 tokens, of which 189,216 are court and tribunal decisions and the rest primary legislation, secondary legislation and bills [1]. The same card's opening paragraph gives a different total, 229,122 texts, so the two headline numbers on one card disagree [1]. It lives at https://huggingface.co/datasets/isaacus/open-australian-legal-corpus .

**Most of the decisions are not licensed for commercial use: the card's licence file restricts Federal Court and High Court judgments to personal, non-commercial, or within-organisation use, and requires NSW Caselaw decisions to be kept from search-engine indexing.**

**Use it for**: continued pretraining or retrieval corpora for Australian law - one document per row with `citation`, `jurisdiction`, `type`, and `date` fields that make it easy to filter to legislation only, which is the commercially permissive part. Not question-answer data; the sibling `isaacus/open-australian-legal-qa` is the SFT-shaped derivative.

**Licence**: `other`, pointing at a custom `LICENCE.md` [3]. That file licenses the collection under CC BY 4.0 but passes through each source's own terms [4]: legislation is broadly reusable, while Federal Court of Australia and High Court of Australia decisions may be reproduced only unaltered and "for personal, non-commercial use or use within an organisation", and NSW Caselaw decisions must exclude "external robots from indexing" [4]. The card's own summary says the licences "in most cases, allow for both non-commercial and commercial usage" [1]; by document count the restricted courts are 71,845 of 232,560 documents [1].

**Shape**: 232,560 documents declared on the card [1]; the viewer indexes a `partial` 118,346 rows of an estimated 146,678 in a single split named `corpus` (not `train`) [5]. Ten columns [6].

**Hold out**: nothing inside the repository. Its QA sibling `isaacus/open-australian-legal-qa` was synthesised from this corpus's documents [7], and the evaluation set `isaacus/legal-rag-bench` draws on a Victorian bench book that this corpus does not include (Victoria is absent from its source list) [1].

**Origin**: statutes and judgments written by Australian parliaments and courts, scraped from nine official sources [1]. Hub API at the check date: `downloads` 20,145, `downloadsAllTime` 280,986, `likes` 98 [3].

**Trained-on-by**: the Hub's dataset tag lists one model, `Abbasgamer1/legalMind`, with 0 downloads [8]; no widely adopted model naming this corpus was found.

**Introduced by**: no paper - the dataset card and Umar Butler's article "How I built the largest open database of Australian law", linked from the card [1]. The repository carries a DOI, 10.57967/hf/2833 [3].

## Shape

Rows the viewer indexed, not the corpus (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `corpus` | 118,346 |
| total | 118,346 |

One config, `corpus` [6]:

| column | dtype |
| --- | --- |
| `version_id` | string |
| `type` | string |
| `jurisdiction` | string |
| `source` | string |
| `mime` | string |
| `date` | string |
| `citation` | string |
| `url` | string |
| `when_scraped` | string |
| `text` | string |

Documents by source, from the card's own table [1]: NSW Caselaw 117,371; Federal Court of Australia 63,749; Federal Register of Legislation 32,356; High Court of Australia 8,096; Queensland 3,306; Tasmania 2,552; NSW Legislation 2,216; Western Australia 1,564; South Australia 1,350. The repository stores the corpus as one 9,401,179,433-byte `corpus.jsonl` [9].

## Quality

- The card reports that 91.60% of documents were extracted from HTML, 6.47% from PDFs, 1.17% from Word documents and 0.76% from RTFs [1]; PDF-sourced text is where extraction noise concentrates.
- Release 7.1.0 (2025-03-10) fixed Unicode encoding errors, removed control characters, and fixed newlines added or removed during cleaning [2]. The legislation was scraped as at 10 March 2025 [4], so later amendments are missing.
- Of the 19 rows the viewer returned, 17 are legislation and 2 are decisions; the viewer's partial copy is not a random sample of the corpus [10].
- `partial` is true, so no whole-corpus duplicate or length figure can be read from the viewer [5].

## Load it

The only split is `corpus`. Stream it and pin the revision:

```python
import datasets

REV = "ef45e3fec41a960919a31149eee6dab9aa39f725"  # main at the check date
corpus = datasets.load_dataset("isaacus/open-australian-legal-corpus", revision=REV, split="corpus", streaming=True)
# keep only legislation, the commercially permissive part
legislation = corpus.filter(lambda d: d["type"] != "decision")
```

**Trap**: the split is named `corpus`, not `train`; `split="train"` fails [6]. And the `type` value that marks case law is `decision` [1]: filtering on the `source` names instead misses nothing today but breaks as soon as a new court is added, because each release can add sources [2].

## Neighbors

- `isaacus/open-australian-legal-qa` - 2,124 question-answer pairs synthesised by `gpt-4` from documents in this corpus, under the same licence [7].
- `isaacus/legal-rag-bench` - an evaluation set on Victorian criminal law, not drawn from this corpus.

## A row

From `config="corpus"`, `split="corpus"`, `row_idx=0` (datasets-server `/first-rows`) [10], with `text` truncated:

```json
{
  "version_id": "tasmanian_legislation:2008-10-08/sr-2008-119",
  "type": "secondary_legislation",
  "jurisdiction": "tasmania",
  "source": "tasmanian_legislation",
  "mime": "text/html",
  "date": "2008-10-08 00:00:00",
  "citation": "Proclamation under the Commonwealth Powers (De Facto Relationships) Act 2006 (Tas)",
  "url": "https://www.legislation.tas.gov.au/view/whole/html/inforce/current/sr-2008-119",
  "when_scraped": "2024-09-13T22:44:32.436265+10:00",
  "text": "Proclamation under the Commonwealth Powers (De Facto Relationships) Act 2006\n\nI, the Governor in and over the State of Tasmania and its Dependencies in the Commonwealth of Australia, acting with the advice of the Executive Council, by this my proclamation made under section 2 of the Commonwealth Powers (De Facto Relationships) Act 2006 fix 8 Octobe [...]"
}
```

## Where it came from

Built by Umar Butler with the Open Australian Legal Corpus Creator, a scraper for nine official Australian legal databases, and transferred to Isaacus in release 7.1.0 [2][1]. Legislation covers the Commonwealth, New South Wales, Queensland, Western Australia, South Australia, Tasmania and Norfolk Island; decisions come from the High Court, the Federal Court and NSW Caselaw [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] isaacus/open-australian-legal-corpus dataset card (README). https://huggingface.co/datasets/isaacus/open-australian-legal-corpus/raw/main/README.md. Fetched 2026-09-23.

[2] isaacus/open-australian-legal-corpus repository file `CHANGELOG.md`. https://huggingface.co/datasets/isaacus/open-australian-legal-corpus/raw/main/CHANGELOG.md - release history and the 7.1.0 transfer to Isaacus. Fetched 2026-09-23.

[3] Hugging Face Hub API record for isaacus/open-australian-legal-corpus. https://huggingface.co/api/datasets/isaacus/open-australian-legal-corpus?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] isaacus/open-australian-legal-corpus repository file `LICENCE.md`. https://huggingface.co/datasets/isaacus/open-australian-legal-corpus/raw/main/LICENCE.md - the per-source terms quoted in the card. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=isaacus%2Fopen-australian-legal-corpus - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=isaacus%2Fopen-australian-legal-corpus - column schema; live, no revision parameter. Fetched 2026-09-23.

[7] isaacus/open-australian-legal-qa dataset card (README). https://huggingface.co/datasets/isaacus/open-australian-legal-qa/raw/main/README.md - the gpt-4 synthesis statement and the shared licence. Fetched 2026-09-23.

[8] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:isaacus/open-australian-legal-corpus&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[9] Repository file tree for isaacus/open-australian-legal-corpus. https://huggingface.co/api/datasets/isaacus/open-australian-legal-corpus/tree/main?recursive=true. Fetched 2026-09-23.

[10] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=isaacus%2Fopen-australian-legal-corpus&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as continued-pretraining or retrieval text for Australian law. Commercial users should filter to legislation: most decisions carry non-commercial or no-indexing terms that the card's summary understates. Not auditable as a whole through the viewer, because `partial` is true.

### The screening row

The row's own note: "the Australian legal corpus; licence is per-source and most case law is non-commercial." The row carries the flag `licence-per-source`.
