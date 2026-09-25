# nguha/legalbench

LegalBench: 162 English legal reasoning tasks from 40 contributors - issue spotting, rule recall, rule application, interpretation, rhetorical understanding - with a few-shot `train` of 856 rows and a `test` of 90,894. The legal benchmark to hold out, and the one most of this skill's training sets overlap.

**nguha/legalbench** is LegalBench, "an ongoing open science effort to collaboratively curate tasks for evaluating legal reasoning in English large language models", currently "162 tasks gathered from 40 contributors" [1], introduced by Guha et al. [2]. Tasks span binary and multi-class classification, extraction, generation and entailment over statutes, opinions and contracts, and "primarily" American law [1]. Following RAFT, each task's `train` split holds only a few labelled examples for prompting [1]. It lives at https://huggingface.co/datasets/nguha/legalbench .

**This is an evaluation set: hold out every `test` split. Several LegalBench task families are built from public datasets whose training splits are in this skill - CUAD, MAUD, ContractNLI, CLAUDETTE (UNFAIR-ToS), LearnedHands, OPP-115, PrivacyQA, SARA, GLOBALCIT. Do not train on a family's source if you report that family.**

**Use it for**: evaluation of legal reasoning; the `train` splits (1 to 10 rows per task) are few-shot exemplars, not training data [3].

**Licence**: CC BY 4.0 in the card metadata [4], but the card says "LegalBench tasks are subject to different licenses. Please see the paper for a description of the licenses" [1]. The one catch: per-task licences differ, and the paper, not the card, lists them.

**Shape**: 91,750 rows over 162 configs: `train` 856 / `test` 90,894 [3]. Most tasks have `index`, `text`, `answer`; some add fields such as `citation` or `question` [5].

**Hold out**: all of it. Public training sets that contain LegalBench test items [6]: `lawinstruct/lawinstruct` and Nemotron's `GlobalCit` config (`international_citizenship_questions`); `theatticusproject/maud` `train` (`maud_*`); the CUAD training contracts (`cuad_*`); Pile of Law's `tos` config (`unfair_tos`) and `r_legaladvice` config (`learned_hands_*`); ContractNLI `train` (`contract_nli_*`).

**Origin**: mixed: the card says tasks come from three sources - existing public datasets, datasets "previously constructed by legal professionals but never released", and tasks written for LegalBench by its authors - "drawn from 36 distinct corpora" [1]. It says data "has either been synthetically generated, or derived from an already public source" [1]. Hub API at the check date: `downloads` 15,446, `downloadsAllTime` 1,752,357, `likes` 188 [4].

**Trained-on-by**: an evaluation set. The Hub's dataset tag lists many small per-task models (for example the `davidschulte/ESM_nguha__legalbench_*` series) and `litillabs/litil-legal-intake-4b` (28 downloads) [7]. NVIDIA reports a proxy-LegalBench score for Nemotron 3 Nano ablations [8].

**Introduced by**: [2] (Guha et al.).

## Shape

Per-split totals and the first tasks (datasets-server `/size`) [3]:

| config | split | rows |
| --- | --- | --- |
| `abercrombie` | `train` | 5 |
| `abercrombie` | `test` | 95 |
| `canada_tax_court_outcomes` | `train` | 6 |
| `canada_tax_court_outcomes` | `test` | 244 |
| `citation_prediction_classification` | `train` | 2 |
| `citation_prediction_classification` | `test` | 108 |
| `citation_prediction_open` | `train` | 2 |
| `citation_prediction_open` | `test` | 53 |
| `consumer_contracts_qa` | `train` | 4 |
| `consumer_contracts_qa` | `test` | 396 |
| `contract_nli_confidentiality_of_agreement` | `train` | 8 |
| `contract_nli_confidentiality_of_agreement` | `test` | 82 |
| `contract_nli_explicit_identification` | `train` | 8 |
| `contract_nli_explicit_identification` | `test` | 109 |
| ... 308 more rows | | |
| all 162 configs | `train` 856 / `test` 90,894 | 91,750 |

Columns of `abercrombie`, the default config [5]:

| column | dtype |
| --- | --- |
| `index` | int64 |
| `answer` | string |
| `text` | string |

Task families by test items (items with at least 8 tokens, as measured): `cuad` 17,980; `privacy_policy` 15,258; `learned_hands` 11,109; `international_citizenship_questions` 9,306; `opp115` 9,187; `maud` 4,598; `supply_chain_disclosure` 3,787; `unfair_tos` 3,614; `ssla` 3,235; `overruling` 2,227; `definition` 2,018; `contract_nli` 1,927; `diversity` 1,800 [6]. A handful of families dominate the item count, so a pooled average weights them heavily.

## Quality

- `pretty_name` in the card metadata is "LegalBench (Staging)" [4]; `nguha/legalbench-staging` is a separate repository [9]. Load `nguha/legalbench`.
- Several tasks derive from r/LegalAdvice posts (LearnedHands), which the card notes "may discuss sensitive issues" [1].
- The card's example instance uses a `label` and `idx` field [1]; the served columns are `answer` and `index` [5].

## Load it

Load a task by config name and evaluate on `test`; use `train` for few-shot prompts only:

```python
import datasets

REV = "daec8237410aa23e3faf4bc41ad8b3a7e1696826"  # main at the check date
shots = datasets.load_dataset("nguha/legalbench", "hearsay", revision=REV, split="train")   # few-shot exemplars
test = datasets.load_dataset("nguha/legalbench", "hearsay", revision=REV, split="test")    # 94 rows
```

**Trap**: a single average over all 162 tasks is dominated by a few large families (CUAD, privacy policy, LearnedHands) [6]; those are also families built from public training data. Report per-family scores, and drop any family whose source is in your training mix.

## Neighbors

- `nguha/legalbench-staging` and `mteb/legalbench_*` - staging and retrieval repackagings found by the Hub search [9]; not screened here.
- `theatticusproject/cuad-qa`, `theatticusproject/maud`, `kiddothe2b/contract-nli`, `coastalcph/lex_glue` - sources of LegalBench task families.

## A row

From `config="abercrombie"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [10]:

```json
{
  "index": 0,
  "answer": "generic",
  "text": "The mark “Salt” for packages of sodium chloride."
}
```

## Where it came from

Curated by Neel Guha and 40 contributors from Stanford's Hazy Research lab and across law and computer science; homepage https://hazyresearch.stanford.edu/legalbench/ , code https://github.com/HazyResearch/legalbench/ [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] nguha/legalbench dataset card (README). https://huggingface.co/datasets/nguha/legalbench/raw/main/README.md. Fetched 2026-09-23.

[2] Guha et al., "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models", arXiv:2308.11462, 2023. https://arxiv.org/abs/2308.11462 - current title read from the live abs page. Fetched 2026-09-23.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=nguha%2Flegalbench - takes no revision parameter; a live figure. Fetched 2026-09-23.

[4] Hugging Face Hub API record for nguha/legalbench. https://huggingface.co/api/datasets/nguha/legalbench?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=nguha%2Flegalbench - column schema; live, no revision parameter. Fetched 2026-09-23.

[6] This skill's own check by n-gram containment against the benchmark's test items (word 8-grams, or character 15-grams for Chinese), with self, words-sorted and positive controls, at the pinned revision. Run 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:nguha/legalbench&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] nvidia/Nemotron-Pretraining-Legal-v1 dataset card (README). https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Legal-v1/raw/main/README.md - "boosted a proxy LegalBench average accuracy from 64.6 to 74.7". Fetched 2026-09-23.

[9] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=LegalBench&sort=downloads. Fetched 2026-09-23.

[10] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=nguha%2Flegalbench&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

An evaluation set: hold out every `test` split. Report per task family, and exclude families whose source datasets appear in the training mix.

### The screening row

The row's own note: "LegalBench; the primary English legal benchmark; many source datasets are public training data." The row carries no flag.
