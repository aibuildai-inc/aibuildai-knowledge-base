# mlabonne/FineTome-100k

100,000 multi-turn instruction-following conversations in ShareGPT-style format (`conversations` list of `from`/`value` turns), each carrying its originating-subset name and an educational-value score.

**mlabonne/FineTome-100k** is a single-repository re-filtering of [arcee-ai/The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome) (with The-Tome's `arcee-ai/qwen2-72b-magpie-en` subset excluded), re-scored with the [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) and reduced to 100,000 rows; the card does not state the selection rule used to reach that count [1]. It has no separate origin paper; the builder made it to accompany the blog post "Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth" [1][2], and its own card gives no other stated task restriction. **Two of its nine source subsets carry competition mathematics: 42,540 rows are drawn from `infini-instruct-top-500k` and 16,854 from `ultrainteract_trajectories_sharegpt`, and the corpus screening flag for this dataset names both as carrying MATH-style AMC/AIME problems, an eval-overlap risk for math-benchmark scoring runs that this card cannot rule out with the sources it could open (see Quality and the appendix).** It lives at https://huggingface.co/datasets/mlabonne/FineTome-100k .

**Use it for**: multi-turn instruction/reasoning SFT - the SFT method card - with the eval-overlap risk above in mind before scoring on AIME- or AMC-style math benchmarks. The `conversations` column is ShareGPT-style (`from`/`value` turns, roles `"human"`/`"gpt"`), not the OpenAI `role`/`content` shape a chat-template collator expects; convert before use (see Load it).

**Licence**: not stated on this dataset's own card - `cardData` carries no `license` key and the repo tags carry no `license:` tag [3]. The upstream `arcee-ai/The-Tome` repo is tagged `license:mit` [4], but that tag does not appear on FineTome-100k's own card, so no licence claim is made here beyond "not stated." The repo is ungated (`"gated": false`, `"private": false`) [3].

**Shape**: 100,000 rows, one config (`default`), one split (`train`), three columns (`conversations`, `source`, `score`) [3][5][6].

**Hold out**: no split is defined to hold out (train-only release) [3][6]. The contamination risk is the math-subset overlap named above, not a held-out split; Quality below gives the exact per-subset row counts read from the served data, and the appendix records the screening verdict.

**Origin**: built by mlabonne from arcee-ai/The-Tome, a nine-dataset blend curated by Arcee AI [1][4]; row-level filtering used the HuggingFaceFW/fineweb-edu-classifier, an automated classifier, not a human process [1][7]. Hub API at the check date: `downloads` 15,735, `downloadsAllTime` 315,750, `likes` 276 [3][8].

**Trained-on-by**: the builder's own accompanying blog post fine-tunes a Llama 3.1 8B model on this dataset with Unsloth/QLoRA and walks through the full run [2]. A Hub dataset-id search for "FineTome" at the check date returns 83 dataset repos and a model-id search returns 325 model repos, the large majority small community QLoRA fine-tunes and GGUF/AWQ re-quantizations of Llama-3.2 1B/3B and similar checkpoints (e.g. `NotASI/FineTome-Llama3.2-1B-0929`, `mradermacher/FineTome-Llama3.2-3B-1002-GGUF`), plus one larger example, `axolotl-ai-co/finetome-llama-3.1-70b` [9]. No adoption by a named, widely-used model family beyond the builder's own example was found.

**Introduced by**: no paper - the dataset card [1] - plus the introducing blog post [2].

## Shape

Splits and rows (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 100,000 |

Columns (shortlist manifest, confirmed by datasets-server `/info`) [3][6]:

| column | dtype |
| --- | --- |
| `conversations` | list\<struct\<from: string, value: string\>\> |
| `source` | string |
| `score` | float64 |

Byte sizes: `download_size` (Parquet on disk) 116,531,415 bytes; `num_bytes_memory` (decoded) 244,477,300 bytes, both read live from datasets-server `/size` at the check date [5]. The README's own `cardData` records a slightly different decoded figure, 239,650,960.7474458 bytes, from when the card was written [1]; the two are close but not identical, and the live figure is the one this card otherwise uses.

Turn-count statistics over the full 100,000-row `train` split, from datasets-server `/statistics` (`conversations` treated as a list column, one count per row) [10]: minimum 2 turns, maximum 62, mean 2.927, median 2.0. No source states a token-length statistic for this release; none is given here.

The `score` column (fineweb-edu-classifier logit, not the clamped 0-5 integer that classifier's own model card also documents) [7] ranges 3.73571 to 5.21262 over the full split, mean 3.9651, median 3.9125 [10] - consistent with a "keep only the highest-scoring rows" filter, though no source states the exact cutoff used.

The `source` column names which of The-Tome's nine subsets each row came from, with exact full-split counts from datasets-server `/statistics` [10]:

| `source` value | rows |
| --- | --- |
| `infini-instruct-top-500k` | 42,540 |
| `WebInstructSub_axolotl` | 23,747 |
| `ultrainteract_trajectories_sharegpt` | 16,854 |
| `self-instruct-sharegpt` | 5,223 |
| `glaive-function-calling-v2-sharegpt` | 3,375 |
| `reasoning-sharegpt` | 3,318 |
| `airoboros-3.2` | 2,046 |
| `systemchat-2.0-sharegpt` | 1,593 |
| `financial-instructions-cleaned-2` | 1,304 |

These nine values and counts match the corpus screening flag for this dataset exactly [11].

## Quality

- The filtering signal is automated, not human-reviewed: the `score` column is the output of the fineweb-edu-classifier, a sequence-classification model trained on 450,000 Llama3-70B-generated educational-value annotations of web text [7]; The-Tome's own upstream curation additionally applied a custom instruction-following reranker to two of its nine subsets and averaged its score with the fineweb-edu-classifier score before FineTome re-scored and re-cut the pool [1][4].
- No source states a measured duplicate rate or contamination rate against any specific eval set for this release.
- The eval-overlap risk named in the opening paragraph rests on two things this card could verify and one it could not. Verified: this dataset's own `source` column shows 42,540 rows from `infini-instruct-top-500k` and 16,854 from `ultrainteract_trajectories_sharegpt` (table above) [10], and The-Tome's card states `infini-instruct-top-500k` is built from `BAAI/Infinity-Instruct` and `ultrainteract_trajectories_sharegpt` from `cognitivecomputations/ultrainteract_trajectories_sharegpt` [4]. Partially verified: the upstream `openbmb/UltraInteract_sft` dataset (the non-ShareGPT original of the UltraInteract line) states its own `dataset` column includes a `MATH` subset of 70,671 rows out of 288,579 total (24.5%) [12], so a MATH-derived competition-math component in that lineage is confirmed at the family level, though this card could not confirm what share of FineTome-100k's own 16,854 UltraInteract rows are specifically MATH-sourced. Not verified: `BAAI/Infinity-Instruct` is a gated repository this card could not open without authentication, so whether it in turn contains `MathInstruct` (the claim in the screening flag) is reported here only as the screening row's own words, not as an independently confirmed fact [11][13].

## Load it

```python
import datasets

REV = "c2343c1372ff31f51aa21248db18bffa3193efdb"  # main at the check date
ds = datasets.load_dataset("mlabonne/FineTome-100k", revision=REV, split="train")  # 100,000 rows
```

**Trap**: `conversations` is ShareGPT-style - a list of `{"from": ..., "value": ...}` dicts with `from` values `"human"`/`"gpt"` (and occasionally `"system"`) - not the OpenAI-style `{"role": ..., "content": ...}` list most chat-template collators expect. Map `from`→`role` (`"human"`→`"user"`, `"gpt"`→`"assistant"`, `"system"`→`"system"`) and `value`→`content` before applying a tokenizer chat template.

## Neighbors

All row counts below were read live from datasets-server `/size` at the check date [14].

- `arcee-ai/The-Tome` - the direct upstream pool this dataset re-filters, 1.75M rows across the same nine subsets plus `arcee-ai/qwen2-72b-magpie-en`, which FineTome-100k excludes; its own card states the 1.75M total and lists all nine plus the excluded tenth subset [4]. Prefer FineTome-100k over re-deriving from The-Tome unless a larger or differently-filtered pool is specifically needed.
- `mlabonne/FineTome-Alpaca-100k` - same builder, same 100,000 rows and the same `source`/`score` columns, reformatted from `conversations` into flat `instruction`/`output` fields for Alpaca-style trainers [15].
- `argilla-warehouse/FineTome-CLAIR` - a derived preference dataset built with the CLAIR revision pipeline (`distilabel`), 77,831 rows total (73,939 train / 3,892 test), adding `chosen`/`rejected` message-list columns alongside the original `source`/`score`; a preference-pair dataset, not a same-shape SFT successor [16].
- `anakin87/FineTome-single-turn-dedup` - 83,341 rows, the first turn only of each conversation with MinHash deduplication and reformatted to OpenAI-style `role`/`content` [17].

## A row

One config, one split, so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [18], with both turn values truncated:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Explain what boolean operators are, what they do, and provide examples of how they can be used in programming. [...] Add the constraint that the test taker must write code that handles cases where truthiness and falsiness are implemented differently across different programming languages."
    },
    {
      "from": "gpt",
      "value": "Boolean operators are logical operators used in programming to manipulate boolean values. [...] Short-circuit evaluation is a behavior where the second operand of a logical operator is not evaluated if the result can be determined based on the value of the first operand. [...]"
    }
  ],
  "source": "infini-instruct-top-500k",
  "score": 5.212620735168457
}
```

## Where it came from

Built by mlabonne. The upstream pool is `arcee-ai/The-Tome`, Arcee AI's 1.75M-row curated instruction blend of nine publicly available datasets - `arcee-ai/infini-instruct-top-500k` (from `BAAI/Infinity-Instruct`), `TIGER-Lab/WebInstructSub`, `jondurbin/airoboros-3.2`, a glaive function-calling set, `arcee-ai/reasoning-sharegpt`, `arcee-ai/self-instruct-sharegpt`, `cognitivecomputations/ultrainteract_trajectories_sharegpt`, `cognitivecomputations/SystemChat-2.0`, and `arcee-ai/qwen2-72b-magpie-en` [4]. The-Tome's own curation applied an instruction-following reranker to two of the nine subsets, scored two of them with the fineweb-edu classifier, and averaged the two scores [4]. FineTome-100k drops the `qwen2-72b-magpie-en` subset and re-scores the rest with the fineweb-edu classifier alone, ending at 100,000 rows; neither this card nor The-Tome's card states the rule used to cut the pool to that count [1][4]. The corpus screening note for this dataset describes the underlying content as "mostly model-generated instruction sets" [11]; the specific generating models vary by subset and are not all named on any card this report opened.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11. Hugging Face Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; datasets-server endpoints that take no revision parameter (as noted per source below) are live and cover only the check-date state.

[1] mlabonne/FineTome-100k dataset card (README). https://huggingface.co/datasets/mlabonne/FineTome-100k/raw/main/README.md - dataset description, upstream relation, classifier used, blog link, `cardData` byte figures. Fetched 2026-08-11.

[2] mlabonne, "Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth". https://huggingface.co/blog/mlabonne/sft-llama3 - the introducing blog post; confirms a Llama 3.1 8B fine-tune on this dataset with Unsloth/QLoRA. Fetched 2026-08-11.

[3] Hugging Face Hub API record for mlabonne/FineTome-100k. https://huggingface.co/api/datasets/mlabonne/FineTome-100k?full=true - `sha`, `gated`, `private`, `cardData`, `downloads`, `likes`, tags. Fetched 2026-08-11.

[4] arcee-ai/The-Tome dataset card (README). https://huggingface.co/datasets/arcee-ai/The-Tome/raw/main/README.md - total row count, nine (plus excluded tenth) source subsets and their upstream names, curation process, `license:mit` tag. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=mlabonne%2FFineTome-100k - row counts, byte sizes. Live, no revision parameter. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=mlabonne%2FFineTome-100k - column names and dtypes, config list. Live, no revision parameter. Fetched 2026-08-11.

[7] HuggingFaceFW/fineweb-edu-classifier model card (README). https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/raw/main/README.md - classifier purpose, training data (450k Llama3-70B annotations), 0-5 score scale and raw-logit vs. clamped-integer distinction. Fetched 2026-08-11.

[8] Hugging Face Hub API record for mlabonne/FineTome-100k with `downloadsAllTime` expanded. https://huggingface.co/api/datasets/mlabonne/FineTome-100k?expand[]=downloadsAllTime Fetched 2026-08-11.

[9] Hugging Face Hub API model search for "FineTome". https://huggingface.co/api/models?search=FineTome - live search, no revision parameter; result count and listed model ids as of the check date. Fetched 2026-08-11.

[10] datasets-server statistics endpoint. https://datasets-server.huggingface.co/statistics?dataset=mlabonne%2FFineTome-100k&config=default&split=train - full-split `conversations` turn-count histogram, `score` histogram, `source` value frequencies. Live, no revision parameter. Fetched 2026-08-11.

[11] The corpus screening row for `mlabonne/FineTome-100k`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[12] openbmb/UltraInteract_sft dataset card and statistics. README: https://huggingface.co/datasets/openbmb/UltraInteract_sft/raw/main/README.md ; statistics: https://datasets-server.huggingface.co/statistics?dataset=openbmb%2FUltraInteract_sft&config=default&split=train - confirms a `MATH` subset (70,671 of 288,579 rows) within the UltraInteract lineage. Fetched 2026-08-11.

[13] Hugging Face Hub API record for BAAI/Infinity-Instruct. https://huggingface.co/api/datasets/BAAI/Infinity-Instruct?full=true - confirms the repo is gated (`"gated": "auto"`); its README and datasets-server `/info` both returned an authentication-required error and could not be read. Fetched 2026-08-11.

[14] datasets-server size endpoint, one call per neighbor, for every neighbor row count above: `arcee-ai/The-Tome`, `mlabonne/FineTome-Alpaca-100k`, `argilla-warehouse/FineTome-CLAIR`, `anakin87/FineTome-single-turn-dedup`. https://datasets-server.huggingface.co/size?dataset=<id> - live, no revision parameter. Fetched 2026-08-11.

[15] mlabonne/FineTome-Alpaca-100k dataset card (README). https://huggingface.co/datasets/mlabonne/FineTome-Alpaca-100k/raw/main/README.md - column schema (Alpaca-style `instruction`/`output` plus `source`/`score`), row count. Fetched 2026-08-11.

[16] argilla-warehouse/FineTome-CLAIR dataset card (README). https://huggingface.co/datasets/argilla-warehouse/FineTome-CLAIR/raw/main/README.md - CLAIR/distilabel pipeline, `chosen`/`rejected` columns, split sizes. Fetched 2026-08-11.

[17] anakin87/FineTome-single-turn-dedup dataset card (README). https://huggingface.co/datasets/anakin87/FineTome-single-turn-dedup/raw/main/README.md - states it is a transformed version of mlabonne/FineTome-100k: first-turn extraction, MinHash dedup, ShareGPT-to-OpenAI format conversion. Fetched 2026-08-11.

[18] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=mlabonne%2FFineTome-100k&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for SFT, with a named eval-overlap risk rather than a clean bill of health. The dataset's own card states no usage restriction and defines no split to hold out [1][3], so there is nothing to withhold structurally; but the screening row flags a rule-risk against `aime2025` because two of the nine `source` subsets - `infini-instruct-top-500k` (42,540 rows) and `ultrainteract_trajectories_sharegpt` (16,854 rows) - draw from lineages that carry MATH-style competition problems, a fact this card could partly confirm (UltraInteract's MATH subset) and could not fully confirm (Infinity-Instruct's contents, gated) [10][11][12][13]. A team scoring on AIME/AMC-style benchmarks should treat this dataset as unscreened for that overlap rather than assume it is clean.

### The screening row

The row's own note [11]: "re-filtered subset of arcee-ai/The-Tome, itself a blend of mostly model-generated instruction sets." The row's flag [11]: "rule-risk: aime2025 - Source counts: 42,540 infini-instruct rows (Infinity-Instruct contains MathInstruct) and 16,854 UltraInteract rows, both carrying MATH AMC/AIME problems."
