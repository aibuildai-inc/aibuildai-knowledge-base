# nvidia/Nemotron-Pretraining-Legal-v1

9.6 million rows of legal pretraining data from NVIDIA - regulations, judicial-ethics opinions, 5.4 million Qwen3-written case-law summaries, and synthetic task data modelled on LegalBench's own task families - two of whose configs hold each other's content.

**nvidia/Nemotron-Pretraining-Legal-v1** is part of NVIDIA's Nemotron pretraining collection and "contains a collection of synthetic datasets intended to improve the legal capabilities of LLMs" [1]. The card reports that "adding these datasets to Nemotron 3 Nano pretraining boosted a proxy LegalBench average accuracy from 64.6 to 74.7" [1]. Its subsets are HTML-extracted regulations and opinions, LLM-cleaned case-law summaries, reformatted CaseHOLD and ContractNLI, and synthetic question sets whose names track LegalBench tasks: definition classification, diversity jurisdiction, function of decision, GlobalCit citizenship questions, CUAD clause types, and terms-of-service clauses [1]. It lives at https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Legal-v1 .

**The config named `Nemotron-Pretraining-Legal-CaseHOLD` holds 5.4 million case-law summaries, and the config named `Nemotron-Pretraining-Legal-Case-Law-Summary` holds the CaseHOLD multiple-choice questions - including every LexGLUE `case_hold` test item. Its GlobalCit config holds 8,895 of LegalBench's 9,306 `international_citizenship_questions` test items.**

**Use it for**: continued pretraining for legal knowledge, config by config: `eCFR`, `California-Code-Of-Regulations` and `NYCourts-Judicial-Ethics-Opinions` are extracted primary sources with no model in the loop [1]. Treat every config whose name matches a benchmark as benchmark-adjacent, and never report CaseHOLD, LexGLUE `case_hold`, or LegalBench citizenship questions after training on this set without decontamination.

**Licence**: CC BY 4.0 (`license: cc-by-4.0`), "ready for commercial use" [2][1]. Every sampled row's own `license` field reads `cc-by-4.0` [3]. Synthetic rows were written by Qwen3-235B-A22B-Instruct-2507 [1]. The one catch: two configs (`ToS-Clause-Understanding`, `ToSDR-QA`) ship with `<CLAUSE>` and `<DOCUMENT>` placeholders whose text must be fetched from other datasets under their own terms, and `Contract-NLI` is not shipped at all - it must be rebuilt with `convert_contract_nli.py` [1][4].

**Shape**: 9,616,568 rows across 13 configs, each a single `train` split, fully indexed [5]; four columns (`text`, `license`, `metadata`, `uuid`) [6].

**Hold out**: nothing inside the repository, but it carries benchmark items [7]: the config named `Case-Law-Summary` contains the CaseHOLD and LexGLUE `case_hold` test items, `GlobalCit` contains LegalBench's `international_citizenship_questions`, and `LegalBench-CUAD-v2` part of LegalBench's `cuad_*` items. Drop those configs if you report those benchmarks. The proxy-LegalBench gain the card reports was measured on a model trained with this data [1].

**Origin**: mixed: regulations and opinions extracted from official sites; summaries and synthetic questions written by Qwen3-235B-A22B-Instruct-2507; CaseHOLD and ContractNLI reformatted from the originals [1]. Hub API at the check date: `downloads` 1,048, `downloadsAllTime` 4,600, `likes` 25; created 2026-06-03 [2].

**Trained-on-by**: the card names Nemotron 3 Nano pretraining ablations and the Nemotron 3 family [1]. The Hub's dataset tag lists `kogai/laneformer-2b-it` (190 downloads) [8].

**Introduced by**: the NVIDIA Nemotron 3 Ultra technical report, which the card asks users to cite [1].

## Shape

Rows per config (datasets-server `/size`) [5]:

| config | split | rows |
| --- | --- | --- |
| `Nemotron-Pretraining-Legal-California-Code-Of-Regulations` | `train` | 57,523 |
| `Nemotron-Pretraining-Legal-Case-Law-Summary` | `train` | 53,137 |
| `Nemotron-Pretraining-Legal-CaseHOLD` | `train` | 5,449,347 |
| `Nemotron-Pretraining-Legal-Definition-Classification` | `train` | 10,200 |
| `Nemotron-Pretraining-Legal-Diversity-Jurisdiction` | `train` | 6,480 |
| `Nemotron-Pretraining-Legal-Function-Of-Decision` | `train` | 70,039 |
| `Nemotron-Pretraining-Legal-GlobalCit` | `train` | 88,898 |
| `Nemotron-Pretraining-Legal-LegalBench-CUAD-v2` | `train` | 460,031 |
| `Nemotron-Pretraining-Legal-NYCourts-Judicial-Ethics-Opinions` | `train` | 5,511 |
| `Nemotron-Pretraining-Legal-ToS-Clause-Understanding` | `train` | 6,831 |
| `Nemotron-Pretraining-Legal-ToSDR-QA` | `train` | 7,569 |
| `Nemotron-Pretraining-Legal-eCFR` | `train` | 35,173 |
| `Nemotron-Pretraining-Legal-eCFR-QA` | `train` | 3,365,829 |
| all 13 configs | `train` 9,616,568 | 9,616,568 |

All configs share one schema [6]:

| column | dtype |
| --- | --- |
| `text` | large_string |
| `license` | large_string |
| `metadata` | struct<category: large_string, models_used: large_string> |
| `uuid` | large_string |

Token counts from the card's own table [1]: `Case-Law-Summary` 4,026.9M, `eCFR-QA` 593.2M, `eCFR` 131.5M, `LegalBench-CUAD-v2` 53.4M, `California-Code-Of-Regulations` 34.9M, `CaseHOLD` 29.3M, and under 25M each for the rest. Those two largest-and-smallest entries are the swapped pair: the directory the viewer calls `CaseHOLD` holds 5,449,347 rows in 6.04 GB of Parquet, and the one called `Case-Law-Summary` 53,137 rows in 35 MB [5].

## Quality

- The config-name swap, verified on every row the viewer served: all 86 sampled rows of `Case-Law-Summary` open with the CaseHOLD instruction "select the correct holding statement from the options", and none of the 50 sampled rows of `CaseHOLD` does; the latter are narrative case summaries (row 0 summarises *United States v. Veon*, the same opinion that is row 0 of the Common Pile case-law corpus) [3].
- `models_used` is recorded per row: `none` for the extracted configs and `Qwen3-235B-A22B-Instruct-2507` for the synthetic ones - including `Case-Law-Summary`, whose CaseHOLD questions were reformatted with that model [3].
- The card names the source of each subset but states no decontamination against any benchmark [1].

## Load it

Load configs by name, remembering the swap, and pin the revision:

```python
import datasets

REV = "3d91d58a5c0c46fe9944300ec46719f97a385b13"  # main at the check date
NV = "nvidia/Nemotron-Pretraining-Legal-v1"
ecfr = datasets.load_dataset(NV, "Nemotron-Pretraining-Legal-eCFR", revision=REV, split="train")           # 35,173 rows
summaries = datasets.load_dataset(NV, "Nemotron-Pretraining-Legal-CaseHOLD", revision=REV, split="train",
                                  streaming=True)   # the 5.4M case-law SUMMARIES live under this name
casehold_mcq = datasets.load_dataset(NV, "Nemotron-Pretraining-Legal-Case-Law-Summary", revision=REV,
                                     split="train")  # 53,137 CaseHOLD questions - contains LexGLUE case_hold test
```

**Trap**: the directory names are the wrong way round for the two largest-impact configs [3][5]. A mix that "excludes CaseHOLD to stay clean" by dropping `Nemotron-Pretraining-Legal-CaseHOLD` drops the harmless summaries and keeps the CaseHOLD questions, test items included. The per-row `metadata.category` carries the same swapped name (the CaseHOLD question in row 0 of `Case-Law-Summary` is labelled `Nemotron-Pretraining-Legal-Case-Law-Summary`) [3], so it cannot fix the swap either: select by content, and read a few rows of each config before mixing.

## Neighbors

- `common-pile/caselaw_access_project_filtered` - the source of the case-law summaries and of the synthetic definition and function-of-decision questions [1].
- `casehold/casehold` and `coastalcph/lex_glue` - the CaseHOLD originals.

## A row

Two rows, both `row_idx=0` of `split="train"` (datasets-server `/first-rows`) [3]: first from `Nemotron-Pretraining-Legal-Definition-Classification`, then from the config named `Nemotron-Pretraining-Legal-Case-Law-Summary`, which shows it holds a CaseHOLD question (truncated):

```json
{
  "text": "Identify if following text from a judicial opinion defines a term.\nText: A motion for peremptory instruction challenges the sufficiency of the evidence, and not its admissibility.\nAnswer: The passage defines \"a motion for peremptory instruction\" by specifying that it challenges the sufficiency of the evidence, not its admissibility, thereby clarifying the term's legal meaning.",
  "license": "cc-by-4.0",
  "metadata": {
    "category": "Nemotron-Pretraining-Legal-Definition-Classification",
    "models_used": "Qwen3-235B-A22B-Instruct-2507"
  },
  "uuid": "d8f08729-9a56-41de-bc7a-e51ca7a47251"
}
```

```json
{
  "text": "You will be provided with a citing text from a judicial decision where the holding of the cited case has been replaced with a blank. Your task is to select the correct holding statement from the options.\n\nCiting text: Drapeau’s cohorts, the cohort would be a “victim” of making the bomb. Further, firebombs are inherently dangerous. There is no peaceful purpose for making a bomb. Felony offenses tha [...]",
  "license": "cc-by-4.0",
  "metadata": {
    "category": "Nemotron-Pretraining-Legal-Case-Law-Summary",
    "models_used": "Qwen3-235B-A22B-Instruct-2507"
  },
  "uuid": "f4866fea-d8ce-4ef7-9521-49c8cb05a06c"
}
```

## Where it came from

Built by NVIDIA for Nemotron 3 pretraining and released 2026-06-03 [2]. The card describes each subset's source and construction, the Qwen3 model used for synthesis, and the `convert_contract_nli.py` script for the unshipped ContractNLI subset [1][4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] nvidia/Nemotron-Pretraining-Legal-v1 dataset card (README). https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Legal-v1/raw/main/README.md. Fetched 2026-09-23.

[2] Hugging Face Hub API record for nvidia/Nemotron-Pretraining-Legal-v1. https://huggingface.co/api/datasets/nvidia/Nemotron-Pretraining-Legal-v1?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[3] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FNemotron-Pretraining-Legal-v1&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

[4] Repository file tree for nvidia/Nemotron-Pretraining-Legal-v1. https://huggingface.co/api/datasets/nvidia/Nemotron-Pretraining-Legal-v1/tree/main?recursive=true. Fetched 2026-09-23.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=nvidia%2FNemotron-Pretraining-Legal-v1 - takes no revision parameter; a live figure. Fetched 2026-09-23.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=nvidia%2FNemotron-Pretraining-Legal-v1 - column schema; live, no revision parameter. Fetched 2026-09-23.

[7] This skill's own check by n-gram containment against the benchmark's test items (word 8-grams, or character 15-grams for Chinese), with self, words-sorted and positive controls, at the pinned revision. Run 2026-09-23.

[8] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:nvidia/Nemotron-Pretraining-Legal-v1&sort=downloads - live list, unpinned. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as continued-pretraining data, with care. Its configs track legal benchmarks closely, two config names are swapped, and measured overlap covers all of LexGLUE `case_hold` test and 96% of LegalBench's citizenship questions. Decontaminate before reporting CaseHOLD, LexGLUE or LegalBench.

### The screening row

The row's own note: "NVIDIA synthetic legal pretraining; configs mirror LegalBench tasks; CaseHOLD/Case-Law-Summary names swapped." The row carries the flag `contains-benchmark-items`.
