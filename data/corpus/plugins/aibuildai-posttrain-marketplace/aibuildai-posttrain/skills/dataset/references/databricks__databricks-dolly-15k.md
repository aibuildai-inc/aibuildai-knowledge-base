# databricks/databricks-dolly-15k

15,011 human-written instruction/response records, each tagged with one of eight task categories, generated entirely by Databricks employees - the original Dolly release.

**databricks-dolly-15k** is a corpus of instruction-following records that Databricks built by having its own employees write prompt/response pairs across eight behavioral categories - seven of them (brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization) taken from the InstructGPT paper [1], plus one added open-ended free-form category - so that open models could be instruction-tuned without relying on closed-model outputs [2]. The card states contributors were told to avoid consulting any web source except Wikipedia and were explicitly told not to use generative AI to write instructions or responses [2]. It lives at https://huggingface.co/datasets/databricks/databricks-dolly-15k .

**Use it for**: single-turn SFT on instruction/response pairs - the SFT method card. The four columns (`instruction`, `context`, `category`, `response`) are not a chat-template format; `context` is often empty and, when present, is reference text the model should read before answering, so a template needs to fold `instruction` and `context` into one user turn before training. No usage-shape restriction: the card states the data may be used for any purpose, academic or commercial, under its license [2].

**Licence**: CC-BY-SA 3.0 (`cardData.license` and the repo's `license:cc-by-sa-3.0` tag agree) [3][2]. Ungated (`"gated": false`) [3]. The one catch: the card notes some of the reference text passed to annotators came from Wikipedia, so downstream copies also carry Wikipedia's own CC-BY-SA attribution obligations, and the `context` field may still contain bracketed Wikipedia citation markers like `[42]` that the card recommends stripping before use [2].

**Shape**: 15,011 rows, one split (`train`), one config (`default`), four string columns [4][5].

**Hold out**: nothing found. No source - the dataset card, the screening row, or the neighbors checked below - flags an overlap with any evaluation set.

**Origin**: built and released by Databricks, Inc.; every instruction and response is written by a Databricks employee, not model-generated [2]. Hub API at the check date: 43,155 downloads, 999,554 all-time downloads, 1,072 likes [3][6].

**Trained-on-by**: Databricks' own `dolly-v2-12b` (and the smaller 7b/3b variants in the same family), instruction-tuned from Pythia base models on this dataset - the model's README states it "is trained on ~15k instruction/response fine tuning records `databricks-dolly-15k`" [7].

**Introduced by**: no paper - the dataset card [2] (plus the introducing blog post, "Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM", cited in the card's own BibTeX entry [2]).

## Shape

Rows and split (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 15,011 |
| total | 15,011 |

One config, `default`, with four columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `context` | string |
| `response` | string |
| `category` | string |

Sizes (datasets-server `/size`) [4]: 13,085,339 bytes as the original JSON download, 7,747,823 bytes as Parquet, 12,195,589 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

The card describes eight instruction categories [2]; of the first 100 served rows, all eight appear: `open_qa`, `closed_qa`, `classification`, `information_extraction`, `general_qa`, `brainstorming`, `summarization`, and `creative_writing` [8].

## Quality

- Every response was written by a Databricks employee under brief, non-rigorous task guidelines; the card itself calls these guidelines "succinct by design" and warns this trades completion rate for strict rubric compliance ("Caveat emptor") [2].
- The card states it knows of no private personal identifiers or sensitive information in the data, since the reference text is drawn from public Wikipedia content [2].
- The card's own stated limitations: Wikipedia is itself a crowdsourced source and may carry its own bias, factual errors, and topical skew; some annotators may not be native English speakers; and annotator demographics reflect the Databricks employee population rather than a general population [2]. No source states a measured contamination rate or duplicate rate for this dataset.
- Of the first 100 served rows, 69 have an empty `context` field; the rest carry a Wikipedia-sourced reference passage the instruction depends on [8]. This count covers only those 100 rows at offset 0 of `train` and does not describe the remaining 14,911 rows.

## Load it

Pin the revision this card's numbers were read at (the shortlist's own commit, matching the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-06-30) [3]:

```python
import datasets

REV = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"  # main at the check date
train = datasets.load_dataset("databricks/databricks-dolly-15k", revision=REV, split="train")  # 15,011 rows
```

**Trap**: `context` is an empty string, not null, when there is no reference passage - a template that concatenates `instruction` and `context` unconditionally will inject a trailing blank line into every row without one (69 of the first 100 rows in this split) [8]. The `response` field is plain text with no chat markup to strip.

## Neighbors

This corpus prefers the original repository over its re-releases and mirrors; every row count and column list below was read live at the check date [9][10]. All numbers below come from the same underlying record set unless a specific edit is described.

- `aisquared/databricks-dolly-15k` - a straight mirror of this repository's README and same four columns (`instruction`, `context`, `response`, `category`), also under CC-BY-SA 3.0; its own card states "this dataset was not originally created by AI Squared" and reproduces the original README verbatim [10]. It reports 15,015 rows against 15,011 here [9] - a difference this card cannot explain from either README, so treat the original as canonical.
- `HuggingFaceH4/databricks_dolly_15k` - the same content renamed to `input`/`output` columns (from `context`/`response`), reformatted for a different training-code convention, tagged CC-BY 3.0 rather than CC-BY-SA 3.0 in its own `cardData` [11]; it likewise reports 15,015 rows [9].
- `argilla/databricks-dolly-15k-curated-en` - not a copy but an open annotation task: its own card is a guideline document asking Argilla annotators to review and correct each `instruction`/`context`/`response`/`category` quadruple, storing both `original-*` and annotator-edited `new-*` fields (9 columns, 15,015 rows) [9][12]. It is a curation project built on this dataset, not a drop-in replacement.
- Machine-translated and reformatted derivatives exist for other languages and instruction-tuning formats (for example `kunishou/databricks-dolly-15k-ja`, `silk-road/chinese-dolly-15k`, `llm-wizard/dolly-15k-instruction-alpaca-format`); none was fetched in full here, so no row-level comparison is made for them.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [8]:

```json
{
  "instruction": "When did Virgin Australia start operating?",
  "context": "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route. It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.",
  "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",
  "category": "closed_qa"
}
```

`context` is empty in rows that need no reference passage (for example `classification` or `open_qa` rows); when present, as here, the `response` is meant to be derivable from `context` alone [2][8].

## Where it came from

Built and released by Databricks, Inc. Databricks employees were invited to write prompt/response pairs across eight instruction categories: seven from the InstructGPT paper's taxonomy (brainstorming, classification, closed QA, generation, information extraction, open QA, summarization) plus an added open-ended category [2]. For categories needing a reference passage (closed QA, summarization, information extraction), contributors selected the passage themselves from Wikipedia, with no guidance on how to choose it [2]. Partway through collection, contributors were also given the option to answer questions other contributors had posed, rephrasing the question and only answering ones they were confident they could answer correctly [2]. Contributors were instructed not to use any web source besides Wikipedia and not to use generative AI to produce instructions or responses [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Ouyang et al., "Training language models to follow instructions with human feedback", 2022. https://arxiv.org/abs/2203.02155 - the InstructGPT paper whose category taxonomy this dataset's card follows; current title read from the live abs page. Fetched 2026-08-11.

[2] databricks/databricks-dolly-15k dataset card (README). https://huggingface.co/datasets/databricks/databricks-dolly-15k/raw/main/README.md - summary, intended uses, collection process, annotator guidelines, known limitations, license/attribution, citation. Fetched 2026-08-11.

[3] Hugging Face Hub API record for databricks/databricks-dolly-15k. https://huggingface.co/api/datasets/databricks/databricks-dolly-15k?full=true - license, gate status, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=databricks%2Fdatabricks-dolly-15k Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=databricks%2Fdatabricks-dolly-15k Fetched 2026-08-11.

[6] Hugging Face Hub API record for databricks/databricks-dolly-15k with `expand[]=downloadsAllTime`. https://huggingface.co/api/datasets/databricks/databricks-dolly-15k?expand%5B%5D=downloadsAllTime - `downloadsAllTime` figure. Fetched 2026-08-11.

[7] databrickslabs/dolly GitHub repository README. https://raw.githubusercontent.com/databrickslabs/dolly/master/README.md - states `dolly-v2-12b` is trained on this dataset; the Hub API for the model repository itself returned an authentication error at fetch time, so this GitHub mirror of the same project is cited instead. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=databricks%2Fdatabricks-dolly-15k&config=default&split=train - sampled row, category distribution and empty-`context` count over the first 100 rows at offset 0. Fetched 2026-08-11.

[9] datasets-server size endpoint, one call per neighbor: `aisquared/databricks-dolly-15k`, `HuggingFaceH4/databricks_dolly_15k`, `argilla/databricks-dolly-15k-curated-en`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[10] aisquared/databricks-dolly-15k dataset card (README). https://huggingface.co/datasets/aisquared/databricks-dolly-15k/raw/main/README.md - states the dataset was not created by AI Squared and reproduces the original README. Fetched 2026-08-11.

[11] HuggingFaceH4/databricks_dolly_15k dataset card (README), including its YAML `cardData`. https://huggingface.co/datasets/HuggingFaceH4/databricks_dolly_15k/raw/main/README.md - column names (`input`/`output`) and declared license (`cc-by-3.0`). Fetched 2026-08-11.

[12] argilla/databricks-dolly-15k-curated-en dataset card (README). https://huggingface.co/datasets/argilla/databricks-dolly-15k-curated-en/raw/main/README.md - describes the curation/annotation guidelines and field list. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as single-turn SFT data: train on `train`, with no hold-out required. The screening row's note identifies this repository as the canonical original among four Dolly-15k-shaped repositories the corpus considered, and nothing in the card or its sources flags a contamination risk.

### The screening row

The row's own note: "the original Dolly; the canonical copy of the four in this range". The row carries no flag.
