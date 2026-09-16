# allenai/tulu-3-sft-mixture

939,343 supervised fine-tuning conversations - multi-turn `messages` chat transcripts pooled from 19 upstream sources of human-written and model-generated data - the full SFT mixture behind the Tulu 3 model family.

**allenai/tulu-3-sft-mixture** is the supervised fine-tuning data mix built by Ai2 (Allen Institute for AI) and introduced in "Tulu 3: Pushing Frontiers in Open Language Model Post-Training" [1]. Its own dataset card lists it as combining human, expert, and machine-generated instruction data from named upstream sets - CoCoNot, FLAN v2, No Robots, OpenAssistant Guanaco (oasst1), several Tulu-3 "persona" math/code/instruction-following sets, NuminaMath-TIR, WildGuardMix, WildJailbreak, a small hard-coded set, Aya, WildChat, TableGPT, SciRIFF, and Evol CodeAlpaca - merged into one chat-format SFT corpus used to train the Tulu 3 series of models [2]. Each row is a full assistant dialogue under a `messages` field of `{role, content}` turns, the shape SFT trainers consume directly.

**The collection license is mixed: ODC-BY-1.0 at the collection level, but several named subsets carry their own separate terms and the card states some portions are non-commercial - No Robots is CC-BY-NC-4.0 [2]. Separately, one component subset in the served data, `ai2-adapt-dev/numinamath_tir_math_decontaminated` (64,312 rows, mapped from NuminaMath-TIR), draws from a parent pool (NuminaMath-CoT) that itself contains an `amc_aime` source (4,072 rows) and a `synthetic_amc` source (62,111 rows) [3]; this dataset's own decontamination scripts by default screen a training set against "the entire Tulu 3 evaluation suite" [4], a suite fixed as of this repository's last modification on 2024-12-02 [5], so it could not have screened against any AIME-2025-dated benchmark. Decontaminate against current AMC/AIME-style competition-math benchmarks before a scored run - see Quality below.**

**Use it for**: SFT chat training - the row shape is a `messages` list of `{role, content}` turns, which is the standard multi-turn chat format most SFT trainers (e.g. TRL's `SFTTrainer`) consume without conversion; maps to the SFT method card. Restricted by the mixed-license and AMC/AIME-overlap points above.

**Licence**: `odc-by` (Open Data Commons Attribution License v1.0) at the collection level [2][5], ungated, not private [5]. Catch: several named subsets carry separate licenses, and the card names No Robots specifically as CC-BY-NC-4.0, i.e. non-commercial [2].

**Shape**: 939,343 rows, one config (`default`), one split (`train`), three columns (`id`, `messages`, `source`) [6][7].

**Hold out**: no internal eval split exists to hold out (the repository serves only `train`) [6]. The risk to hold out against is external: the NuminaMath-TIR-derived subset's parent pool contains AMC/AIME items [3], and this dataset's decontamination targeted the Tulu 3 evaluation suite as it stood at the 2024-12-02 release [4][5] - not any AIME-2025-dated benchmark. See the bold paragraph above and Quality below.

**Origin**: built by Ai2; content mix is machine-generated, crowdsourced, and expert-generated per the repository's own annotation-creator tags [5]. Hub downloads 36,301, downloadsAllTime 302,809, likes 255, as of the check date [5].

**Trained-on-by**: the Tulu 3 series - Llama-3.1-Tulu-3-8B-SFT and Llama-3.1-Tulu-3-70B-SFT, and the downstream Tulu 3 DPO, RLVR, and reward-model checkpoints built on those SFT models - per the dataset card's own model-family table [2]. A separate, near-identical mixture (`allenai/tulu-3-sft-olmo-2-mixture`, see Neighbors) was used to train the OLMo 2 SFT models; that is a different repository, not this one [8].

**Introduced by**: [1] (Lambert et al.), and the dataset card itself for the mixture's own composition [2].

## Shape

Rows and split, from the datasets-server size endpoint [6]:

| split | rows |
| --- | --- |
| `train` | 939,343 |

One config, `default`, three columns (datasets-server info endpoint) [7]:

| column | dtype |
| --- | --- |
| `id` | string |
| `messages` | list<struct<content: string, role: string>> |
| `source` | string |

Sizes: 1,412,954,868 bytes of original Parquet download, matching in both the live datasets-server size endpoint and the repository's own pinned `dataset_info` metadata [6][2]. The two sources disagree on decoded-in-memory size: the repository's pinned card metadata states `dataset_size` 2,914,250,826.56 bytes, while the live datasets-server size endpoint reports 1,516,664,147 bytes of memory - roughly half - for the same 939,343 rows at the same commit [2][6]. No source explains the gap; both values are reported here rather than resolved. No source states sequence-length or token statistics for this release.

The dataset card's own prose lists 18 named subsets and states a total of 939,344 samples [2], one row more than the 939,343 rows actually served [6]. Neither figure matches the sum of the 18 listed subset counts (889,344). Reading the served `source` column's full value distribution (a population count over all 939,343 rows, not a sample) via the datasets-server statistics endpoint [9] finds 19 distinct source values, not 18: one source present in the served data, `ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k` (50,000 rows), does not appear anywhere in the card's list of named subsets [2][9]. Adding it to the 18 listed counts reconciles the total to 939,344, one row above the actual 939,343 - the statistics endpoint also shows `ai2-adapt-dev/oasst1_converted` at 7,131 rows against the card's stated 7,132 for OpenAssistant Guanaco, accounting for that single-row gap [2][9].

## Quality

- The repository's own tags mark the collection as `annotations_creators: crowdsourced, expert-generated, machine-generated`, spanning all three across its subsets, without a per-subset breakdown in the card itself [5].
- Several `source` values carry a `_decontaminated` suffix in the served data - `numinamath_tir_math_decontaminated`, `tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k`, `tulu_v3.9_wildjailbreak_decontaminated_50k`, and `evol_codealpaca_heval_decontaminated` [9] - indicating those four subsets (64,312 + 50,000 + 50,000 + 107,276 = 271,588 rows) were screened before being merged into this mixture. The open-instruct decontamination tooling's own documentation states its default query set is "the entire Tulu 3 evaluation suite" [4], which is fixed as of this repository's 2024-12-02 last-modified date [5] and therefore does not cover any AIME-2025-dated benchmark.
- No source states a measured duplicate rate or annotator-agreement figure for this release; none is invented here.
- The one undocumented `source` value and the 939,344-vs-939,343 count mismatch, both above, are the only source-stated quality signals beyond composition and decontamination scope.

## Load it

Single config, single split; pin the revision this card's numbers were read at (matches the shortlist's commit and the live Hub API `sha` at the check date) [5]:

```python
import datasets

REV = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"  # main at the check date
ds = datasets.load_dataset("allenai/tulu-3-sft-mixture", revision=REV, split="train")  # 939,343 rows
```

**Trap**: the `source` column names an internal, `ai2-adapt-dev/`-prefixed processing id (e.g. `ai2-adapt-dev/oasst1_converted`), not the human-readable subset name the card's prose uses (e.g. "OpenAssistant Guanaco") or the `source_datasets` tag value (e.g. `OpenAssistant/oasst1`) [2][9]; joining this dataset's `source` values against the card's named-subset list or against the `source_datasets` tags requires the id-to-name mapping given in Quality above, not a string match.

## Neighbors

- `allenai/tulu-3-sft-olmo-2-mixture` - a separate repository with the same 939,344 samples reported in its own pinned card metadata and the same named subsets, except it swaps the 240-row "Tulu 3 Hardcoded" set for an "OLMo 2 Hardcoded" set and adds a fourth `dataset` column; used to train the OLMo 2 SFT models, not the Tulu 3 models this repository trained [8].
- `allenai/tulu-3-sft-mixture-0225` - 866,137 rows live [10], built from `allenai/tulu-3-sft-mixture-filter-datecutoff` by a documented `update_subsets.py` command that removes the `personahub_math_v5_regen_149960` and `tulu-3-sft-personas-math-grade` sources and adds filtered replacements [11] - a pruned, revised mix, not a copy of this release.
- `allenai/tulu-3-sft-mixture-filter-datecutoff` - 938,487 rows live [10]; its card carries only pinned `dataset_info` metadata with no descriptive body, so no source states what "datecutoff" filters beyond the repository name itself.
- `allenai/tulu-3-sft-olmo-2-mixture-0225` and `allenai/tulu-3-sft-olmo-2-mixture-filter-datecutoff` - the OLMo 2 counterparts of the two variants above, at 866,138 and 938,488 rows live respectively [10].

Mixing any of these into training data alongside this release risks duplicating rows, since the `-0225` and `-filter-datecutoff` variants are documented edits of the same base mixture rather than independent corpora [11].

## A row

One served shape (`config="default"`, `split="train"`). From `row_idx=0` (datasets-server first-rows endpoint) [12], with the assistant's Terraform code block truncated:

```json
{
  "id": "oasst1_5921",
  "source": "ai2-adapt-dev/oasst1_converted",
  "messages": [
    {
      "role": "user",
      "content": "Create a snippet of Terraform HCL code that create an AWS autoscaling group, and an ALB in front to expose an application to internet."
    },
    {
      "role": "assistant",
      "content": "Sure, here's an example Terraform HCL code that creates an AWS Autoscaling Group and an Application Load Balancer to expose an application to the internet:\n``` \n# Configure the AWS provider\nprovider \"aws\" {\n  region = \"us-east-1\"\n}\n\n# Create a security group to allow traffic to the ALB\nresource \"aws_security_group\" \"alb_sg\" {\n  name_prefix = \"alb_sg\"\n  ingress {\n    from_port = 80\n    to_port = 80 [...]"
    }
  ]
}
```

Of the first 100 served rows, all carry `source: "ai2-adapt-dev/oasst1_converted"` and 2 to 4 messages per row [12].

## Where it came from

Built by Ai2 as the merged supervised fine-tuning stage of the Tulu 3 post-training recipe [1]. The card describes the mixture as drawing from 18 named upstream sets - some collected by third parties and re-released (CoCoNot, FLAN v2, No Robots, OpenAssistant Guanaco, Aya, TableGPT, NuminaMath-TIR, Evol CodeAlpaca, WildChat), and some built by Ai2 itself for Tulu 3 (the Tulu-3 persona math/code/algebra/instruction-following sets, WildGuardMix, WildJailbreak, and a small hard-coded set) [2]. The served `source` column additionally names a 19th component, an `open_math_2_gsm8k`-derived 50,000-row subset, not listed in the card's own prose (see Shape) [9]. Several of Ai2's own subsets carry a `_decontaminated` suffix in their internal ids, indicating they were screened against the Tulu 3 evaluation suite before being merged in [4][9].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision, matching the shortlist's `commit` field.

[1] Lambert et al., "Tulu 3: Pushing Frontiers in Open Language Model Post-Training", 2024. https://arxiv.org/abs/2411.15124 - the origin paper; current title read from the live abs page. Fetched 2026-08-11.

[2] allenai/tulu-3-sft-mixture on the Hub: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture ; card text read from the raw README, https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/raw/main/README.md - subset list and counts, license text, model-family table, dataset structure. Fetched 2026-08-11.

[3] AI-MO/NuminaMath-CoT dataset card (README). https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/raw/main/README.md - source breakdown table, including `amc_aime` and `synthetic_amc` rows; NuminaMath-TIR's own card states it draws from this pool. Fetched 2026-08-11.

[4] allenai/open-instruct decontamination tooling README (GitHub). https://raw.githubusercontent.com/allenai/open-instruct/main/decontamination/README.md - states the default query set for decontamination search is the entire Tulu 3 evaluation suite. A `main`-branch file, unpinned and mutable. Fetched 2026-08-11.

[5] Hugging Face Hub API record for allenai/tulu-3-sft-mixture. https://huggingface.co/api/datasets/allenai/tulu-3-sft-mixture?full=true - license, gate/private status, `sha`, `downloads`, `likes`, `lastModified`, annotation-creator tags; `downloadsAllTime` read via the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=allenai%2Ftulu-3-sft-mixture Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=allenai%2Ftulu-3-sft-mixture Fetched 2026-08-11.

[8] allenai/tulu-3-sft-olmo-2-mixture dataset card (README). https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture/raw/main/README.md - subset list, row count, model-family table for OLMo 2. Fetched 2026-08-11.

[9] datasets-server statistics endpoint. https://datasets-server.huggingface.co/statistics?dataset=allenai%2Ftulu-3-sft-mixture&config=default&split=train - `source` column value frequencies over the full 939,343 rows. Fetched 2026-08-11.

[10] datasets-server size endpoint, one call per neighbor: `allenai/tulu-3-sft-mixture-0225`, `allenai/tulu-3-sft-mixture-filter-datecutoff`, `allenai/tulu-3-sft-olmo-2-mixture-0225`, `allenai/tulu-3-sft-olmo-2-mixture-filter-datecutoff`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[11] allenai/tulu-3-sft-mixture-0225 dataset card (README). https://huggingface.co/datasets/allenai/tulu-3-sft-mixture-0225/raw/main/README.md - the `update_subsets.py` command documenting how this variant was derived from `allenai/tulu-3-sft-mixture-filter-datecutoff`. Fetched 2026-08-11.

[12] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=allenai%2Ftulu-3-sft-mixture&config=default&split=train Fetched 2026-08-11.

[13] The corpus screening row for `allenai/tulu-3-sft-mixture`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT training data, with two caveats already established above: hold licensing in mind for the non-commercial subset (No Robots, CC-BY-NC-4.0) [2], and decontaminate against current AMC/AIME-dated competition-math benchmarks before a scored run, since this repository's own decontamination pass predates them [3][4][5]. The screening row's flag names this second point directly.

### The screening row

The row's own note [13]: "939k Tulu-3 SFT mix: human sets (OASST, no_robots, Aya, WildChat, SciRIFF) plus GPT-4 persona-synthetic math/code/IF subsets." Its flag: "rule-risk: aime2025 - Card lists AI-MO/NuminaMath-TIR at 64,312 prompts; the NuminaMath pool carries AMC/AIME competition items."
