# openmed-community/MedReason-Stenographic

31,535 medical question-answer pairs, each carrying a full natural-language chain-of-thought alongside a compressed symbolic reasoning trace built from the original transcript.

**openmed-community/MedReason-Stenographic** re-expresses the medical QA reasoning chains of UCSC-VLAA/MedReason [1] - itself introduced by "MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs" [2] - into a symbolic "stenographic" notation, using MiniMax M2.1 to first regenerate a full chain-of-thought for each source question and then compress that chain-of-thought into the symbolic trace [3]. Each row therefore carries both the full reasoning (`thinking`) and its compressed form (`reasoning`), so the row serves reasoning-trace SFT in either the verbose or the compressed target. **Rows are not unique per source question: the dataset card's own statistics table counts 31,535 total samples against only 10,550 unique source IDs, meaning most source questions recur roughly three times in the file** [3]; a sample read confirmed this below, and found the recurring rows to be a mix of byte-identical repeats and independently reworded regenerations. It lives at https://huggingface.co/datasets/openmed-community/MedReason-Stenographic .

**Use it for**: reasoning-trace SFT - either the compressed symbolic `reasoning` field or the full `thinking` chain-of-thought as the training target for `query`/`answer` - mapped to a single-turn instruction/response format; see the SFT method card. Because most source questions appear two or three times - some as byte-identical repeats, some independently reworded - deduplicate or sample by `id_in_dataset` before training if per-question weighting matters.

**Licence**: Apache-2.0 (`license: apache-2.0` in the card's YAML and in the repo's `license` tag), ungated [4]. No catch beyond the licence itself: the README's only other licence text is the closing "Apache 2.0" line, restating the same grant [3].

**Shape**: 31,535 rows in one config, one split (`train`), five columns [5][6].

**Hold out**: nothing named as an evaluation set by any source read for this card - the dataset card states no benchmark or eval split, and no evaluation-overlap flag is recorded elsewhere in the material seen. The repeated source IDs above are a per-question duplication risk within `train`, not an eval-holdout requirement.

**Origin**: built by the `openmed-community` organization (repository author `openmed-community`, previously hosted under `MaziyarPanahi`, whose account now redirects to this repository) [4]; the QA content originates in UCSC-VLAA/MedReason [1] and the `reasoning`/`thinking`/`answer` fields are LLM generations from MiniMax M2.1 [3]. Hub API at the check date: 103 downloads, 826 all-time downloads, 56 likes [4].

**Trained-on-by**: none found - a Hub model search for "MedReason-Stenographic" returned no results at the check date [7].

**Introduced by**: no paper - the dataset card [3]; the source dataset it transforms is introduced by [2].

## Shape

Rows and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 31,535 |

One config, `default`, with five columns (datasets-server `/info`) [6]:

| column | dtype |
| --- | --- |
| `query` | string |
| `reasoning` | string |
| `answer` | string |
| `thinking` | string |
| `id_in_dataset` | int64 |

The dataset card gives per-field average character counts, not token counts: query 157.4, reasoning 384.2, answer 35.2, thinking 5,548.4 chars [3]. No source states a token-length statistic.

## Quality

- The dataset card's own statistics table reports 31,535 total samples but only 10,550 unique source IDs (`id_in_dataset` values), summing to 42,085 in the shortlist's row-count table because that table adds two different quantities - a duplication the card's prose does not otherwise flag [3]. A read of the served rows confirms it directly: of 100 rows read at offset 31,435-31,534 (the last 100 rows of `train`), 91 `id_in_dataset` values were unique and 9 repeated. Text was compared across each repeated pair: five of the nine (ids 7133, 7134, 7138, 7143, 7150) were byte-identical, and the remaining four (ids 7131, 7137, 7145, 7149) were reworded, in one case substantially (id 7145's query reads as a full clinical vignette at offset 31,520 and as a two-sentence summary of the same case at offset 31,531) - so repeats mix exact duplicates and independent MiniMax M2.1 regenerations of the same source question, not one or the other uniformly [8].
- The card reports data-cleanliness counts directly: of 31,535 rows, 89 (0.3%) have an empty `query`, 9 (<0.1%) have an empty `answer`, 0 have an empty `reasoning` (the card states these were filtered out), and "All samples have thinking field populated" [3].
- The card gives usage frequency for each symbol in the stenographic lexicon across the 31,535 rows, e.g. `→` in 99.5%, `∴` in 98.2%, `<H≈` in 87.5%, down to `○` in 5.1% [3]; no independent verification of these figures was performed for this card.
- No source states an annotator-agreement figure or a measured medical-accuracy rate for the regenerated reasoning or answers.

## Load it

```python
import datasets

REV = "7a242e42a2bfdac47c152936407831aa7fe7fc1b"  # main at the check date
ds = datasets.load_dataset("openmed-community/MedReason-Stenographic", revision=REV, split="train")  # 31,535 rows
```

**Trap**: the card's own base-model tag (`base_model: MiniMax/MiniMax-M1` in the YAML front matter) names a different model from the one stated in the card's prose ("generated using MiniMax M2.1"); the prose is the more specific claim and is repeated three times in the body, so this card follows the prose [3][4]. Also note `id_in_dataset` is not a unique row key within this dataset (see Quality above) - do not use it as a deduplication key without also comparing `query` text.

## Neighbors

No sibling, cleaned, or binarized re-release of this specific stenographic transform was found; the repository itself is the only place this transform is served, having moved from the account `MaziyarPanahi` to `openmed-community` (the old author path now redirects to this repository) [4]. The direct upstream, UCSC-VLAA/MedReason, is the un-transformed source and is not a substitute: it serves 32,682 rows with `question`/`answer`/`reasoning`/`options`/`dataset_name`/`id_in_dataset` columns, natural-language reasoning only, and no `thinking` field [1][9].

## A row

One config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [10]:

```json
{
  "query": "Most sensitive test for H pylori",
  "reasoning": "Detect(Task:RAG) ● → Domain(Gastroenterology) ● → Pathogen(H.pylori) ● → DiagnosticTests(SensitivityComparison) ● → UreaBreathTest(Sensitivity:95-98%) ● → FecalAntigen(Sensitivity:90-95%) ● → BiopsyUrease(Sensitivity:90-95%) ● → Serology(Sensitivity:80-90%) ● → DistractorAnalysis ※ → ComparativeAssessment ◐ → NonInvasiveGoldStandard ● → <H≈0.4> → ∴",
  "answer": "D. Urea breath test",
  "thinking": "The user is asking a medical question about the most sensitive test for H. pylori (Helicobacter pylori). This is a knowledge/retrieval question with multiple choice options. [...] The urea breath test is widely considered the most sensitive test for detecting active H. pylori infection. [...] Let me construct the reasoning trace following the symbolic lexicon.",
  "id_in_dataset": 7134
}
```

## Where it came from

The `openmed-community` organization built this dataset by taking the question/answer/reasoning triples of UCSC-VLAA/MedReason [1] and, for each, prompting MiniMax M2.1 to generate a full natural-language chain-of-thought (stored in `thinking`), then compressing that chain-of-thought into a symbolic "stenographic" trace (stored in `reasoning`) under a documented lexicon the card calls the "Grand Unified Reasoning Protocol (v4.1)" [3]. MedReason itself was built by UCSC-VLAA from clinical QA pairs drawn from seven medical datasets, converted into reasoning chains via a structured medical knowledge graph, as described in its introducing paper [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. That pin, however, does not extend to the datasets-server endpoints used for the Shape table, the Quality duplication counts, and the sampled row [5][6][8][10]: a direct check found that datasets-server's `/info` endpoint returns HTTP 200 with the same live payload whether or not a `revision` parameter is supplied, even an invalid one, so it does not accept a pin at all. Those numbers therefore describe the `main` branch as read on the check date, not the commit `7a242e42a2bfdac47c152936407831aa7fe7fc1b` named in Load it; since the repository's Hub API record shows no modification since its creation on 2026-01-09 and `sha` for `main` still equals that commit as of the check date, the two are believed to agree, but a re-run of `load_dataset(..., revision=REV)` against a later `main` push would not be guaranteed to reproduce these counts.

[1] UCSC-VLAA/MedReason dataset card (README). https://huggingface.co/datasets/UCSC-VLAA/MedReason/raw/main/README.md - source dataset identity, row count, composition, citation. Fetched 2026-08-12.

[2] Wu et al., "MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs", 2025. https://arxiv.org/abs/2504.00993 - the origin paper for UCSC-VLAA/MedReason; current title read from the live abs page. Fetched 2026-08-12.

[3] openmed-community/MedReason-Stenographic dataset card (README). https://huggingface.co/datasets/openmed-community/MedReason-Stenographic/raw/main/README.md - generation process, statistics tables, symbol lexicon, licence text, field descriptions. Fetched 2026-08-12.

[4] Hugging Face Hub API record for openmed-community/MedReason-Stenographic. https://huggingface.co/api/datasets/openmed-community/MedReason-Stenographic?full=true - licence, gate, `sha`, `downloads`, `likes`, `cardData.base_model`, siblings; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant; the `MaziyarPanahi/MedReason-Stenographic` path was checked directly and returns an HTTP redirect to this same API record, confirming the account move. Fetched 2026-08-12.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=openmed-community%2FMedReason-Stenographic Fetched 2026-08-12.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=openmed-community%2FMedReason-Stenographic Fetched 2026-08-12.

[7] Hugging Face Hub model search API. https://huggingface.co/api/models?search=MedReason-Stenographic - returned an empty result list. Fetched 2026-08-12.

[8] datasets-server rows endpoint, two calls: offset 0 length 100, and offset 31,435 length 100 (the dataset's last 100 rows, given 31,535 total). https://datasets-server.huggingface.co/rows?dataset=openmed-community%2FMedReason-Stenographic&config=default&split=train&offset=<n>&length=100 Fetched 2026-08-12.

[9] datasets-server info and size endpoints for UCSC-VLAA/MedReason. https://datasets-server.huggingface.co/info?dataset=UCSC-VLAA%2FMedReason and https://datasets-server.huggingface.co/size?dataset=UCSC-VLAA%2FMedReason - column names/types and row count for the neighbor comparison. Fetched 2026-08-12.

[10] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=openmed-community%2FMedReason-Stenographic&config=default&split=train Fetched 2026-08-12.

[11] The corpus screening row for `openmed-community/MedReason-Stenographic`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data (compressed symbolic or full chain-of-thought). No usage-shape restriction is stated by the dataset itself, no evaluation-overlap flag was found in any source read, and the licence is a clean Apache-2.0 grant [3][4]. The one caveat a trainer should carry forward is the row-level duplication established above: roughly a third of `id_in_dataset` values are unique, so naive training over all 31,535 rows over-weights the ~10,550 distinct source questions that were regenerated multiple times.

### The screening row

The row's own note [11]: "UCSC-VLAA/MedReason medical QA re-expressed as compressed symbolic reasoning traces by MiniMax M2.1." The row carries no flag.
