# stindardlogic/legal-reasoning-dpo-100k

Sold as 100,000 synthetic U.S. legal DPO pairs; measured, it is 16 distinct prompt/chosen/rejected triples, each repeated 6,250 times under a fresh UUID.

**stindardlogic/legal-reasoning-dpo-100k** describes itself as "A synthetic Direct Preference Optimization (DPO) dataset of 100,000 legal reasoning conversations with chosen (high-quality) and rejected (poor-quality) response pairs" across 13 categories of U.S. law [1]. Its chosen answers "Apply the correct legal framework by name" and give actionable guidance; its rejected answers show "Immediate deflection to 'consult an attorney'" and generic principles [1]. It lives at https://huggingface.co/datasets/stindardlogic/legal-reasoning-dpo-100k .

**Measured on the full split, the 100,000 rows contain 16 unique prompts, 16 unique chosen answers and 16 unique rejected answers; the largest group repeats 6,250 times. Every row still carries its own UUID, so deduplicating on `id` removes nothing.**

**Use it for**: nothing at its stated scale. The 16 distinct pairs are well-written examples of the intended contrast (named statute, concrete next steps, versus a generic deflection) and can serve as a handful of hand-checked preference exemplars; training DPO on all 100,000 rows repeats each pair 6,250 times.

**Licence**: Apache 2.0 in the card metadata [2]. The card calls the data synthetic but does not name the generating model [1]. The one catch: the generator's own terms are unknown.

**Shape**: 100,000 rows in one `train` split [3]; five columns (`prompt`, `chosen`, `rejected`, `metadata`, `id`) [4]. The Parquet file is 15,987,635 bytes for 355,768,750 bytes in memory, a 22x ratio that is itself a sign of repetition [3].

**Hold out**: no split is set aside, and none can be: any split of 16 repeated triples puts the same pairs on both sides. Harvey LAB: its data file was committed on 2026-07-22, after LAB went public, and was measured: none of LAB's 2,010 rubrics, 2,010 instructions or 48,687 documents is contained (`references/contamination.md`, "Harvey LAB").

**Origin**: model-generated, per the card's own "synthetic" description [1]; the generating model is not named. Hub API at the check date: `downloads` 59, `downloadsAllTime` 152, `likes` 0; created 2026-07-22 [2].

**Trained-on-by**: no model on the Hub declares this dataset through its dataset tag [5].

**Introduced by**: no paper - the dataset card [1].

## Shape

| split | rows |
| --- | --- |
| `train` | 100,000 |
| total | 100,000 |

One config, `default` [4]:

| column | dtype |
| --- | --- |
| `prompt` | string |
| `chosen` | string |
| `rejected` | string |
| `metadata` | struct<category: string, context: string> |
| `id` | string |

## Quality

- Measured duplication on all 100,000 rows: 16 unique normalised prompts (99.98% repeats), 16 unique chosen answers, 16 unique rejected answers, 16 unique whole rows; largest group 6,250 rows [6].
- In the 56 rows the viewer served, 16 distinct prompts already appear, spread over 13 `metadata.category` values led by `contract_interpretation` (13 rows) [7].
- The chosen answers cite statutes by section, for example "15 U.S.C. § 1692c(a)(3)" for the FDCPA workplace-contact rule in row 0 [7]. With only 16 prompts, no source checks these citations.

## Load it

If you use it, deduplicate on content, not `id`:

```python
import datasets

REV = "31db58e87aad732d3be9e71797bef989d42b3929"  # main at the check date
ds = datasets.load_dataset("stindardlogic/legal-reasoning-dpo-100k", revision=REV, split="train")  # 100,000 rows
seen, keep = set(), []
for i, r in enumerate(ds):
    k = (r["prompt"], r["chosen"], r["rejected"])
    if k not in seen:
        seen.add(k); keep.append(i)
ds = ds.select(keep)  # 16 rows
```

**Trap**: `id` is a unique UUID on every row - rows 0 and 1 have different ids [7] - even though the content repeats. Any pipeline that deduplicates or splits by `id` keeps all 100,000 rows and puts identical pairs in train and test.

## A row

From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], truncated:

```json
{
  "prompt": "I keep getting calls from a debt collector about a debt that I don't recognize. They're calling me at work and my family members. What are my rights?",
  "chosen": "What you're describing sounds like multiple violations of the Fair Debt Collection Practices Act (FDCPA) — and you have significant rights and potential remedies.\n\n**FDCPA Violations in Your Description:**\n\n**Calling at work**: Under 15 U.S.C. § 1692c(a)(3), a debt collector cannot contact you at work if they have reason to know your employer disap [...]",
  "rejected": "I'm sorry to hear you're experiencing unwanted debt collection calls. You have rights under the Fair Debt Collection Practices Act (FDCPA).\n\nDebt collectors have rules they must follow, including restrictions on when and how they contact you. They generally cannot call at inconvenient times, contact you at work if you've told them not to, or harass [...]",
  "metadata": {
    "category": "consumer_protection",
    "context": "debt collection and consumer rights"
  },
  "id": "c98936aa-0489-4353-abad-cac4614230bd"
}
```

## Where it came from

Uploaded by the `stindardlogic` account on 2026-07-22 [2]; the card documents the intended contrast between chosen and rejected answers and the 13 categories, but not the generation process [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] stindardlogic/legal-reasoning-dpo-100k dataset card (README). https://huggingface.co/datasets/stindardlogic/legal-reasoning-dpo-100k/raw/main/README.md. Fetched 2026-09-23.

[2] Hugging Face Hub API record for stindardlogic/legal-reasoning-dpo-100k. https://huggingface.co/api/datasets/stindardlogic/legal-reasoning-dpo-100k?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=stindardlogic%2Flegal-reasoning-dpo-100k - takes no revision parameter; a live figure. Fetched 2026-09-23.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=stindardlogic%2Flegal-reasoning-dpo-100k - column schema; live, no revision parameter. Fetched 2026-09-23.

[5] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:stindardlogic/legal-reasoning-dpo-100k&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[6] This skill's duplication measurement on the full `train` split, `references/contamination.md` (duplication table) - normalised exact match on prompt, chosen, rejected, and the whole row. Run 2026-09-23.

[7] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=stindardlogic%2Flegal-reasoning-dpo-100k&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Not usable at its stated scale: 16 distinct triples repeated 6,250 times each. At most, the 16 unique pairs are exemplars of the intended legal-helpfulness contrast.

### The screening row

The row's own note: "claims 100K legal DPO pairs; measured 16 unique rows." The row carries the flag `massive-duplication`.
