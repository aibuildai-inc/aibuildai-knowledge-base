# pile-of-law/pile-of-law

A 256GB English-language corpus of U.S. and some international legal and administrative text - opinions, dockets, statutes, regulations, contracts, bills, hearings and r/legaladvice posts - in 45 source configs, for legal-domain pretraining.

**pile-of-law/pile-of-law** was built by Henderson et al. at Stanford and introduced in "Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset" [1]. The card describes "a large corpus of legal and administrative data" meant both to study how legal norms handle privacy and toxicity filtering and "to collect a dataset that can be used in the future for pretraining legal-domain language models" [2]. The loading script defines 45 source configs plus `all` [3], and the data sits in 119 xz-compressed JSONL files totalling about 44 GB compressed [4]. It lives at https://huggingface.co/datasets/pile-of-law/pile-of-law .

**The card says the authors "do not ... filter out any data from downstream tasks" - so this corpus carries the raw sources of several legal benchmarks, including the CLAUDETTE terms-of-service corpus behind LexGLUE `unfair_tos` and LegalBench `unfair_tos`.**

**Use it for**: continued pretraining on English legal text, config by config. Pick sources for the target: `courtlistener_opinions` and `state_codes` for U.S. doctrine, `edgar` and `atticus_contracts` for contract language, `r_legaladvice` for lay questions. Not SFT data: rows are raw documents with no instruction or answer.

**Licence**: CC BY-NC-SA 4.0 (`license: cc-by-nc-sa-4.0` in both the card and the Hub tag) [5][2]. The card adds that "individual sources may have other licenses" and asks users to "not re-host any data in a way that can be indexed by search engines" [2]. The one catch: non-commercial and share-alike, so a commercial model cannot train on it under this grant.

**Shape**: 45 source configs plus an `all` config, each with `train` and `validation` splits cut 75%/25% [2][3]; four columns (`text`, `created_timestamp`, `downloaded_timestamp`, `url`) [2]. The dataset viewer is disabled for this repository, so no row count can be read from it [6].

**Hold out**: the `validation` split is a random 25% the authors "do not use ... for any downstream tasks" [2]. Of external benchmarks: the `tos` config is the CLAUDETTE corpus (its rows' `url` is `http://claudette.eui.eu/ToS.zip`) [7], the raw source of LexGLUE `unfair_tos` [8]; `r_legaladvice` is r/legaladvice, the forum the LearnedHands questions in LegalBench come from [9]; `atticus_contracts` and `edgar` are EDGAR contracts like those CUAD and MAUD annotate. Screen against LegalBench and LexGLUE before any scored run - `references/contamination.md` has the measured overlaps. Harvey LAB: three data files were committed after LAB went public on 2026-05-06 (`courtlisteneropinions` 5 and 9, `courtlistenerdocketentries` validation 0, 3.16 GB); they are court records and were not measured against LAB. Every other file predates LAB (`references/contamination.md`, "Harvey LAB").

**Origin**: documents written by courts, agencies, legislatures, parties and, in `r_legaladvice`, members of the public; scraped by the authors from the sources listed in the card [2]. Hub API at the check date: `downloads` 7,659, `downloadsAllTime` 540,746, `likes` 285 [5].

**Trained-on-by**: the Hub's dataset tag lists BSC's Salamandra family (`BSC-LT/salamandra-7b-instruct`, 31,351 downloads; `BSC-LT/salamandra-2b-instruct`, 2,740) and `BSC-LT/ALIA-40b` (1,052) among models declaring it, plus the authors' own `pile-of-law/legalbert-large-1.7M-2` (474) [10].

**Introduced by**: [1] (Henderson et al.).

## Shape

The viewer is disabled for this repository [6]. Row counts measured by downloading three small `train` files and counting lines [7]:

| config | `train` rows (measured) | compressed file |
| --- | --- | --- |
| `r_legaladvice` | 109,740 | 61,491,876 bytes |
| `tos` | 37 | 280,156 bytes |
| `exam_outlines` | 12 | 363,852 bytes |

The large configs are split across many files: `courtlistener_opinions` spans 16 `train` shards and 6 `validation` shards, and `atticus_contracts` 5 `train` shards of about 1 GB each [3][4]. All 119 data files total about 44 GB compressed [4]; the paper's title gives the uncompressed size as 256GB [1].

## Quality

- The card makes "no representation that the legal information provided here is accurate" and notes the dataset "was edited on 7/17 to remove some data" after a request [2].
- `created_timestamp` "may be inaccurate": for CourtListener opinions it is the upload time to CourtListener, not the decision date [2].
- The card says documents "may contain personal and sensitive information", previously filtered by the publishing agencies [2]. The `exam_outlines` config's first row carries a student's email address and phone number in its first lines [7] - scrub before training.
- The config is named `r_legaladvice`, but its files are named `train.r_legaldvice.jsonl.xz` and `validation.r_legaldvice.jsonl.xz` (letters swapped) [3][4]; code that builds file paths from config names misses them.
- No source states a measured duplicate rate.

## Load it

There is no parquet copy, so `load_dataset` runs the repository's script. Load one config at a time and pin the revision:

```python
import datasets

REV = "2e96169e7e4b43f8ea36230515ebb44b27423b94"  # main at the check date
opinions = datasets.load_dataset("pile-of-law/pile-of-law", "courtlistener_opinions", revision=REV,
                                 split="train", streaming=True, trust_remote_code=True)
```

**Trap**: the loading script wraps each file in a bare `try: ... except: print("Error reading file:", filepath)` [3], so a truncated or corrupted download prints one line and the split silently comes back short. Count rows against a reference after loading, and on `datasets` 4.x, which removed script loading, read the `data/*.jsonl.xz` files directly instead.

## Neighbors

- `joelniklaus/Multi_Legal_Pile` - its English configs download Pile of Law files at load time, `train` and `validation` alike, into one `train` split; see that card.
- `common-pile/caselaw_access_project` - U.S. opinions under a public-domain tag rather than CC BY-NC-SA 4.0, the licence-clean alternative to `courtlistener_opinions`.
- `lamblamb/pile_of_law_subset` and `SKIML-ICL/pile-of-law` - community re-uploads found by the Hub search; they were not screened here.

## A row

The viewer serves no rows, so this is line 1 of `data/train.r_legaldvice.jsonl.xz`, downloaded and decompressed [7], with `text` truncated:

```json
{
  "url": "https://www.reddit.com/r/legaladvice/comments/59cv5x/landlord_broke_lease_agreement_what_are_my_rights/",
  "text": "Title: Landlord broke lease agreement, what are my rights? (Chicago, IL)\nQuestion:Our landlord has been promising us a washer/dryer unit since we moved in (July 2015). When we resigned the lease August 2016, we wrote into the lease that an in-unit washer and dryer would be installed by September 30th 2016. [...]",
  "downloaded_timestamp": "11-09-2021",
  "created_timestamp": "10-25-2016"
}
```

## Where it came from

Assembled by Peter Henderson, Mark Krass, Lucia Zheng, Neel Guha, Christopher Manning, Dan Jurafsky and Daniel Ho at Stanford [1]. Collection and processing code is at https://github.com/Breakend/PileOfLaw ; the card says the authors did not normalize the data [2]. The corpus follows DMCA notice-and-takedown, and the card records one takedown edit [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Henderson et al., "Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset", arXiv:2207.00220, 2022. https://arxiv.org/abs/2207.00220 - current title read from the live abs page. Fetched 2026-09-23.

[2] pile-of-law/pile-of-law dataset card (README). https://huggingface.co/datasets/pile-of-law/pile-of-law/raw/main/README.md. Fetched 2026-09-23.

[3] pile-of-law/pile-of-law repository file `pile-of-law.py`. https://huggingface.co/datasets/pile-of-law/pile-of-law/raw/main/pile-of-law.py - the 45 `_DATA_URL` source keys, the `all` config, the 75/25 file lists, and the bare `except`. Fetched 2026-09-23.

[4] Repository file tree for pile-of-law/pile-of-law. https://huggingface.co/api/datasets/pile-of-law/pile-of-law/tree/main?recursive=true. Fetched 2026-09-23.

[5] Hugging Face Hub API record for pile-of-law/pile-of-law. https://huggingface.co/api/datasets/pile-of-law/pile-of-law?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[6] datasets-server size, info and splits endpoints for pile-of-law/pile-of-law. https://datasets-server.huggingface.co/size?dataset=pile-of-law%2Fpile-of-law - each answered with the error quoted in the card instead of a size. Fetched 2026-09-23.

[7] pile-of-law/pile-of-law files `data/train.r_legaldvice.jsonl.xz`, `data/train.tos.jsonl.xz` and `data/train.examoutlines.jsonl.xz`, downloaded from https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/data/ and read line by line - row counts and first rows. Fetched 2026-09-23.

[8] Lippi et al., "CLAUDETTE: an Automated Detector of Potentially Unfair Clauses in Online Terms of Service", arXiv:1805.01217, 2018. https://arxiv.org/abs/1805.01217 - current title read from the live abs page. Fetched 2026-09-23.

[9] nguha/legalbench dataset card (README). https://huggingface.co/datasets/nguha/legalbench/raw/main/README.md - "Several tasks have been derived from the LearnedHands corpus, which consists of public posts on /r/LegalAdvice". Fetched 2026-09-23.

[10] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:pile-of-law/pile-of-law&sort=downloads - live list, unpinned. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as non-commercial continued-pretraining text, config by config. Not decontaminated by its authors, and it carries the raw sources of LexGLUE `unfair_tos` and LegalBench's LearnedHands tasks, so screen it against any legal benchmark you will report. Row counts for most configs are not auditable without downloading them, because the viewer is disabled.

### The screening row

The row's own note: "the reference English legal pretraining corpus; NC-SA, viewer disabled, and explicitly not decontaminated." The row carries the flag `not-decontaminated`.
