# open-agreements/legal-practice-library

119 plain-English explainers of U.S. state (and a few foreign) law on non-competes and consumer data privacy, each citing statutes and cases and dated by when the law was checked - plus about 1,600 more practice-guide, checklist and case-excerpt files in the repository that the viewer does not serve. Small, current, and none of it marked as human-reviewed.

**open-agreements/legal-practice-library** is "a clean, source-cited snapshot of the OpenAgreements practice-guide corpus: plain-English explainers of US state (and select international) law, currently covering non-compete / restrictive-covenant law and consumer data-privacy law", "re-synced from the upstream source on every content change" [1]. It lives at https://huggingface.co/datasets/open-agreements/legal-practice-library .

**Reference text, not SFT pairs: each row is one jurisdiction's guide as a markdown document. Every served row has `human_reviewed_at: null`, so treat the guides as unreviewed until the canonical page says otherwise.**

**Use it for**: grounding or continued-pretraining text for employment and privacy work - state-by-state non-compete enforceability and state privacy statutes. Its question-and-table layout also suits building retrieval-grounded Q&A.

**Licence**: CC BY 4.0 in the card metadata and body, "attribute to openagreements.org" [1][2]. The catch: the repository carries separate `LICENSE` and `NOTICE` files for `checklists/`, `practice-guides/`, `legal-practice-library/` and `surveys/` [3]; read them before using files outside `data.jsonl`.

**Shape**: 119 rows, one config, split `train`, 10 columns [4]. Measured: 68 `non-compete` and 51 `privacy` guides; 107 U.S., 9 Australian, and one each for India, the Philippines and Singapore [5].

**Hold out**: nothing; no evaluation split.

**Origin**: written and maintained by openagreements.org against primary law, "with machine-verifiable source citations" and a last-reviewed date [1]. All 119 served rows record `human_reviewed_at: null` [5]; the card does not say who or what drafts the guides. Hub API at the check date: `downloads` 2,926, `downloadsAllTime` 8,037, `likes` 0 [2].

**Trained-on-by**: the Hub's model search by dataset tag returns no model [6].

**Introduced by**: [1] (dataset card; no paper).

## Shape

Columns [4]: `topic`, `slug`, `jurisdiction`, `country_code`, `canonical_url`, `license`, `last_reviewed`, `snapshot_as_of`, `stale`, `markdown`.

Repository files outside the viewer [3]: 1,765 files in total, including `legal-practice-library/case-excerpts/` (656), `checklists/` (non-compete, venture financing with NVCA-form checklists, founder separation), and guides on invention assignment, wage and hour, privacy policies and AI vendors. Snapshot as of 2026-09-23 [1].

## Quality

- Duplication, measured on `data.jsonl` [5]: no exact, cleaned or MinHash (Jaccard at least 0.8) duplicates among the 119 guides.
- `stale` is false for all 119 rows [5]; the `last_reviewed` field records the law-checked date, not a human review.
- The repository is re-synced "on every content change" [1]: pin the revision, or two runs will train on different law.

## Load it

```python
import datasets

REV = "e6c1bdbe1df9016cb20c843f30d1e9edff449eab"  # main at the check date
guides = datasets.load_dataset("open-agreements/legal-practice-library", revision=REV, split="train")   # 119 rows
```

**Trap**: the viewer and `load_dataset` return only the 119 guides in `data.jsonl`; the checklists, case excerpts and other practice areas are markdown and JSON files in the repository tree, under their own licence files. Download them with `huggingface_hub.snapshot_download` if you want them.

## Neighbors

- `ymoslem/Law-StackExchange` - human Q&A on many of the same employment and privacy questions, across jurisdictions.
- `reglab/housing_qa` - statute-grounded state-law questions in another area; an evaluation set.

## A row

From `split="train"`, `row_idx=0` (datasets-server `/first-rows`), `markdown` shortened [7]:

```json
{
  "topic": "non-compete",
  "slug": "alabama",
  "jurisdiction": "Alabama",
  "country_code": "US",
  "canonical_url": "https://openagreements.org/practice-guides/non-compete/us/alabama",
  "license": "CC BY 4.0",
  "last_reviewed": "2026-06-03T00:00:00",
  "snapshot_as_of": "2026-09-23T00:00:00",
  "stale": false,
  "markdown": "---\njurisdiction: \"Alabama\"\n...\nhuman_reviewed_at: null\n...\n# Non-Competes in Alabama\n\nA question-by-question summary of Alabama non-compete law under the 2016 Restrictive Covenant Act (Ala. Code § 8-1-190 et seq.) ..."
}
```

## Where it came from

Published and maintained by openagreements.org; source commit `faafc1edb61b9e172ef55420cb3e1eea62ee002c` per the card [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-24; that date covers every number, quote, and row above unless a line says it was measured. Hub repositories are mutable, which is why Load it pins the revision.

[1] Dataset card (README). https://huggingface.co/datasets/open-agreements/legal-practice-library/raw/e6c1bdbe1df9016cb20c843f30d1e9edff449eab/README.md. Fetched 2026-09-24.

[2] Hugging Face Hub API record. https://huggingface.co/api/datasets/open-agreements/legal-practice-library?full=true - licence field, `sha`, `lastModified` 2026-09-24; `downloads`, `downloadsAllTime`, `likes` through the `expand[]` variant. Fetched 2026-09-24.

[3] Repository tree. https://huggingface.co/api/datasets/open-agreements/legal-practice-library/tree/e6c1bdbe1df9016cb20c843f30d1e9edff449eab?recursive=true - 1,765 files. Fetched 2026-09-24.

[4] datasets-server size and info endpoints. https://datasets-server.huggingface.co/size?dataset=open-agreements%2Flegal-practice-library - 119 rows, `partial` false. Fetched 2026-09-24.

[5] This skill's own count of `data.jsonl` at the pinned revision: topics, countries, `stale`, `human_reviewed_at`, and exact, cleaned and MinHash duplication. Run 2026-09-24.

[6] Hub model search by dataset tag. https://huggingface.co/api/models?filter=dataset:open-agreements/legal-practice-library&sort=downloads - empty. Fetched 2026-09-24.

[7] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=open-agreements%2Flegal-practice-library&config=default&split=train. Fetched 2026-09-24.

## Appendix: screening record

### Screening verdict

Passed as grounding and pretraining text for employment and privacy work. Stage 0: readable, not gated, `partial` false. Stage 1: document text, not pairs. Stage 2: CC BY 4.0 in field and body; per-directory licence files not read. Stage 3: no duplicates in the 119 guides. Stage 5 not run. Authorship unstated and no guide marked human-reviewed.

### The screening row

The row's own note: "OpenAgreements practice guides, non-compete and privacy by state; 119 served rows, ~1,600 more files in the tree; none marked human-reviewed." The row carries the flag `unreviewed`.
