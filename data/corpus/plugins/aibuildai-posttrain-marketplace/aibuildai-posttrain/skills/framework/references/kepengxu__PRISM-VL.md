# PRISM-VL

A single research paper's code release for measurement-grounded vision-language training: it is not an independent post-training library, but a frozen, project-customized snapshot of Alibaba's ms-swift 3.12.0 shipped alongside one Qwen3-VL LoRA recipe, its data, and its eval harness.

PRISM-VL's own README describes it as "a research release for asking a simple but under-tested question: when the RGB image has already lost sensor evidence, can a vision-language model reason better from measurement-domain observations?" [1]. It is authored by Kepeng Xu, Li Xu, Gang He, and Wenxin Yu, and backs the arXiv paper "Allegory of the Cave: Measurement-Grounded Vision-Language Learning" [2][1]. There is no separately maintained trainer API: the repository bundles a full copy of the `swift` package (ms-swift, Alibaba's post-training library, version `3.12.0` per its own `swift/version.py` [3]) under `swift/`, patches it with project-specific arguments (`camera_metadata_residual_enabled` and related fields for injecting camera metadata into the ViT), and drives it entirely through the `swift sft` and `swift deploy` CLIs [4][5]. It lives at https://github.com/kepengxu/PRISM-VL [6].

**When to pick it**: pick PRISM-VL only to reproduce this paper's specific result - RAW-derived "Meas.-XYZ" input plus camera metadata used to fine-tune Qwen3-VL 2B/4B/8B with LoRA - using its packaged benchmark (MeasL-Bench-V1), training corpus (MeasL-150K-V1), and released adapters [1]. The paper's own headline comparison is the deciding number: PRISM-VL-8B reaches 0.6120 BLEU, 0.4571 ROUGE-L, and 82.66% LLM-Judge accuracy, improving over the RGB Qwen3-VL-8B baseline by +0.1074 BLEU, +0.1071 ROUGE-L, and +4.46 percentage points [2]. It is not a general-purpose choice among post-training libraries: its own quick start runs exactly one method (SFT/LoRA) through one bundled, version-pinned copy of ms-swift, so for GRPO/DPO/PPO/KTO or any other post-training workflow the underlying, actively-maintained library is upstream ms-swift itself (cross-reference; not covered here), not this frozen fork [3][4].

**Methods it ships**: the training configs and launch scripts PRISM-VL actually publishes run one method - SFT with LoRA (`train_type: lora`) - for all three released model sizes [4]. The vendored `swift/trainers/rlhf_trainer/__init__.py` additionally exposes `CPOTrainer`, `DPOTrainer`, `GRPOTrainer`, `KTOTrainer`, `ORPOTrainer`, `PPOTrainer`, `RewardTrainer`, and `GKDTrainer` classes; this file is byte-identical to the same path in upstream ms-swift at the `v3.12.0` tag [7][19]. None of these classes appear in PRISM-VL's own configs, launch scripts, or README, so they are present in the code but not exercised by this release's documented workflow. Method definitions and math belong to those methods' own cards, not here.

**Scale it handles**: as shipped, PRISM-VL's own launch script (`configs/qwen3_vl_150k_llmmeta_vit_proxy/train_prsimvl_2b.sh`) runs `swift sft` through `NPROC_PER_NODE` (default 2) and `CUDA_VISIBLE_DEVICES` (default `0,1`) for local multi-GPU data-parallel training on one machine; its packaged SFT config comments out the DeepSpeed setting with the note "Disabled for single GPU training to avoid MPI requirement," so the release documents single-machine, 1-2 GPU runs, not multi-node [8][4]. The vendored ms-swift package carries DeepSpeed ZeRO (`zero0`...`zero3_offload`) and FSDP2 options that could scale further [9], but PRISM-VL publishes no multi-node or multi-GPU-beyond-two benchmark of its own.

**Install**: `bash install_editable.sh`, which does `pip install -e libs/datasets-3.6.0`, `pip install -e libs/qwen-vl-utils`, `pip install -e libs/transformers-4.57.3`, then `pip install -e .` (the vendored `swift` package itself) [5]. There is no tagged release or GitHub Release for this repository - the GitHub Releases list is empty, and the repository's commit history (as read from the GitHub API) is four commits, all messaged "first commit," pushed within about eight hours of each other on 2026-05-27 (06:18:52, 06:36:12, 06:57:52, and 14:27:12 UTC) [10][11]; the shortlist commit `c73957b05fdc41cdfc3eb97c91a5e68251ef0681` is the last of those four, i.e. the current head of `main`, so it is not ahead of any separate release. `setup.py` sets `python_requires='>=3.8.0'` [12]. Code license is Apache-2.0 (repository `LICENSE`) [13], but the release notes state the packaged datasets (MeasL-Bench-V1, MeasL-150K-V1) are "CC BY-NC 4.0 for non-commercial research and education" - a different license from the code [1]. `requirements/framework.txt` (pulled in by `requirements.txt`) pins `transformers>=4.33,<4.58`, `datasets>=3.0,<4.0`, `trl>=0.15,<0.25`, `peft>=0.11,<0.19`, `gradio>=3.40.0,<6.0`, and `modelscope>=1.23` [14]; the editable install step overrides the transformers/datasets floor with exact local copies, `transformers-4.57.3` and `datasets-3.6.0` [5]. No CUDA or GPU hardware minimum is stated in the README, `requirements.txt`, `requirements/framework.txt`, or `install_editable.sh` [1][14][5]; the packaged SFT configs set `attn_impl: flash_attn`, which implies a flash-attention-capable GPU is needed to run the recipe as configured, though this is a config choice, not a documented platform floor [4].

**Maintained by**: Kepeng Xu and co-authors (Li Xu, Gang He, Wenxin Yu), as a research-group artifact tied to one paper, not an ongoing framework [2][1]. GitHub activity spans 2026-05-26 (repo created) to 2026-05-27 (four "first commit" pushes, last at 14:27:12 UTC) [11]; there are no GitHub Releases and no evidence of ongoing maintenance beyond this snapshot [10][11].

## Quick start

PRISM-VL's own quick start, quoted/adapted from its README [1]:

```bash
git clone <repo-url> PRSIMVL
cd PRSIMVL
bash install_editable.sh
```

Dry-run the packaged evaluation without launching inference:

```bash
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh --dry-run
```

Run one released LoRA adapter on the default Meas.-XYZ benchmark split:

```bash
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
```

Start a demo inference service and ask a question, from `inference/README.md` [15]:

```bash
CUDA_VISIBLE_DEVICES=0 swift deploy \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --adapters exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct/v8-20260421-133546/checkpoint-95000
```

```bash
python inference/ask_service.py \
  --image inference/demo_data/images/demo1_pole_color.png \
  --question "This is a linear Image with Metadata: ISO: 250, Exposure Time: 1/640, Aperture: f/9. What is the color of the vertical pole visible through the windshield?"
```

Training uses the packaged launch scripts directly: `bash configs/qwen3_vl_150k_llmmeta_vit_proxy/train_prsimvl_2b.sh` (also `_4b.sh`, `_8b.sh`) [1]. Both the released LoRA adapters and the training/benchmark data must be pulled from Hugging Face separately (`kepeng/PRSIMVL-LoRA-V1`, `kepeng/MeasL-150K-V1`, `kepeng/MeasL-Bench-V1`) and placed under `exps/`, `training_data/`, and `eval_data/` before these commands run against real data [1].

## Start it

- One machine, one or two GPUs is the form the release documents: `train_prsimvl_2b.sh` sets `NPROC_PER_NODE=2` and `CUDA_VISIBLE_DEVICES=0,1` by default and calls `swift sft --config <yaml>` [8]. Its packaged SFT config comments out `deepspeed: zero0` with the note "Disabled for single GPU training to avoid MPI requirement," so multi-node or larger multi-GPU deployment is not demonstrated by this release even though the vendored ms-swift supports DeepSpeed ZeRO and FSDP2 [4][9].
- Effective batch size in the shipped 2B config: `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 4`, on up to 2 GPUs by default [4].
- Configuration is a YAML file passed to `swift sft --config`; the packaged 2B config sets `train_type: lora`, `lora_rank: 8`, `lora_alpha: 32`, `target_modules: all-linear`, `torch_dtype: bfloat16`, `gradient_checkpointing: true`, and the project-added `camera_metadata_residual_enabled: true` with `camera_metadata_residual_num_layers: 2` [4]. This is PRISM-VL's own config on top of ms-swift's argument surface, not a library-wide default change - ms-swift's own CLI-parameter docs show `gradient_checkpointing` already defaults to `True` and learning rate already defaults to `1e-4` for LoRA, so the packaged config mostly confirms upstream defaults rather than overriding them [9][4].
- Out-of-memory first aid is not written up as its own section in this repo; the shipped config's own commented guidance is to keep `per_device_train_batch_size: 1` and rely on `gradient_accumulation_steps` and `gradient_checkpointing: true`, and to drop `deepspeed` only if avoiding an MPI dependency on a single machine [4]. Beyond the packaged config, memory knobs are whatever the vendored ms-swift CLI documents (not searched here beyond what config fields PRISM-VL itself sets).

## Watch it

Mechanics only - what a metric means for SFT is on that method's own card, not here.

- Logging is enabled through ms-swift's `report_to` argument, which the vendored library defaults to `'tensorboard'` (not off, unlike a library that logs nowhere by default) [9]; PRISM-VL's own packaged config does not override `report_to`, so it inherits that TensorBoard default, with `logging_steps: 5` set explicitly in the 2B SFT config [4][9].
- PRISM-VL publishes no metric-name list of its own; the metric names available are whatever the vendored ms-swift 3.12.0 SFT trainer emits, which was not enumerated from this repo's own docs (it ships no `docs/` for the `swift/` package) and was not fully enumerated from the one upstream page fetched for this card [9] either - a full per-trainer metric list belongs on ms-swift's own card, not here.
- Sample-level logging: ms-swift's `log_completions` flag (default `False`) logs generated text when paired with `--report_to wandb/swanlab`, and writes a `completions.jsonl` under the checkpoint directory when no such tracker is set - this is the vendored library's behavior, not something PRISM-VL's own SFT config turns on [9].
- Evaluation during training: PRISM-VL's packaged config sets `val_dataset: [MEASL/BENCH_RAW_V1]`, `eval_steps: 5000`, and `per_device_eval_batch_size: 2` [4]; this evaluates on held-out loss/metrics during the `swift sft` run, separate from the `eval/run_infer_and_eval.sh` benchmark wrapper described below, which runs full inference-plus-scoring against MeasL-Bench-V1 after or outside training [16][4].
- Stopping-rule honesty: no early-stopping or reward-threshold field appears in PRISM-VL's own SFT config; it sets a fixed `max_steps: 100000` and relies on `save_steps`/`save_total_limit` for checkpoint retention rather than any stopping rule [4]. No PRISM-VL-published health limit or stopping threshold was found in the README or the training/eval READMEs read for this card.

## Save it

- The packaged config sets `output_dir: exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct`, `save_steps: 5000`, `save_total_limit: 20`, `save_only_model: false` [4]. Under ms-swift's own default `add_version: True` behavior, each run adds a `<version>-<timestamp>` subdirectory before the numbered `checkpoint-<step>/` folders [9], matching the released paths PRISM-VL documents, e.g. `exps/.../v8-20260421-133546/checkpoint-95000` [1].
- `save_only_model: false` (the config's explicit setting) means PRISM-VL's own training saves optimizer state and random-seed state alongside weights during a run. Upstream ms-swift's own docs describe `save_only_model` as saving only model weights, excluding optimizer and random-seed states, to reduce time and space overhead in full-parameter training [9][4].
- The three checkpoints PRISM-VL actually redistributes on Hugging Face are pruned for inference only: the release manifest states "Only inference-required adapter files are included. Optimizer state, scheduler state, RNG state, trainer state, and training argument binaries are excluded" [17]. That means the downloaded `exps/.../checkpoint-*` directories are LoRA adapters usable for inference/merge, not resumable training checkpoints, even though a fresh `swift sft` run of your own would save the fuller state per the config above.
- Because training uses `train_type: lora`, what is saved per checkpoint is a LoRA adapter (small, incremental weights), not a full model; ms-swift's own parameter docs distinguish `--model` ("the directory path of the complete weights... for example `model.safetensors`") from `--adapters` ("a list of incremental adapter weight directory paths... for example `adapter_model.safetensors`") [9]. PRISM-VL's own demo inference command loads a base model plus the adapter directory together (`swift deploy --model Qwen/Qwen3-VL-2B-Instruct --adapters exps/.../checkpoint-95000`), confirming an adapter checkpoint on its own is not a loadable full model [15].
- Resume: PRISM-VL's own config leaves `resume_from_checkpoint` commented out with the note "set only when resuming a previous run" [4]; ms-swift's own docs specify passing `--resume_from_checkpoint <checkpoint_dir>` to resume model, optimizer, and RNG state, or `--resume_only_model` to load weights only [9].
- Loader handoff: whether an evaluator can load a released PRISM-VL checkpoint directly depends on pairing the adapter directory with the matching base model (`Qwen/Qwen3-VL-2B/4B/8B-Instruct`) exactly as the demo inference command does [15]; this card does not restate a general loader contract beyond what PRISM-VL's own demo shows.

## Find it in the docs

PRISM-VL itself publishes no separate documentation site - its "docs" are the README plus the per-folder README files in the repository (`eval/README.md`, `inference/README.md`, `training_data/README.md`, `eval_data/README.md`) [1][16][15]. There is no version-tagged docs address pattern to give, because there is no GitHub Release or docs build for this repository - only the single `main` branch at the commit read for this card [10][11].

- For the vendored training engine's own argument reference (what every `swift sft` config field does, `swift deploy` options, DeepSpeed/FSDP2 settings), the live source is upstream ms-swift's documentation at `https://swift.readthedocs.io/en/latest/`, which is NOT version-pinned to the 3.12.0 snapshot PRISM-VL vendors - it tracks ms-swift's current `main`, so a field's default or behavior there can have moved past what PRISM-VL's frozen `swift/` copy actually does [9][18]. To check the vendored copy's own behavior instead of the live docs, read the source directly at this card's commit, e.g. `https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/swift/trainers/`.
- Runnable references inside this repo beyond the README: `eval/run_infer_and_eval.sh --dry-run` prints the exact commands for all three released checkpoints without running inference, which doubles as a smoke test of the install [16]; `inference/demo_data/images/` ships three small images for the service demo [15].
- No community tutorial layer, curated or otherwise, is published by this repository - it is a single-paper release, not a library with a tutorials page.
- No official MCP endpoint is published for this repository.

Trap: none found in maintainer replies to closed issues, because the repository's issue tracker was not part of the pages fetched for this card and no in-repo document records one.

Honest boundary: this release demonstrates exactly one method (SFT/LoRA) on exactly one model family (Qwen3-VL, 2B/4B/8B) for exactly one dataset pairing (MeasL-150K-V1 training, MeasL-Bench-V1 evaluation) [1][4]; it documents no multi-node scaling, no online RL method, and no stopping rule of its own, and its docs layer stops at its own README files - there is no separate maintained documentation site for PRISM-VL itself.

## Sources

[1] PRISM-VL README (definition, release contents, install/quick-start/training/eval commands, licensing note for datasets, headline results). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/README.md. Fetched 2026-08-12.

[2] arXiv abstract page for "Allegory of the Cave: Measurement-Grounded Vision-Language Learning" (authors, dates, abstract). https://arxiv.org/abs/2605.11727. Fetched 2026-08-12.

[3] Vendored ms-swift version file. https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/swift/version.py. Fetched 2026-08-12.

[4] Packaged SFT config for the 2B model (train_type, LoRA fields, camera-metadata fields, batch/accumulation, output_dir, save/eval fields, deepspeed comment, resume_from_checkpoint comment). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/configs/qwen3_vl_150k_llmmeta_vit_proxy/sft_qwen3_vl_2b_prsimvl_v1.yaml. Fetched 2026-08-12.

[5] `install_editable.sh` (editable install of vendored transformers/datasets/qwen-vl-utils/swift). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/install_editable.sh. Fetched 2026-08-12.

[6] PRISM-VL GitHub repository (item home). https://github.com/kepengxu/PRISM-VL. Fetched 2026-08-12.

[7] Vendored `swift/trainers/rlhf_trainer/__init__.py` (trainer class list). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/swift/trainers/rlhf_trainer/__init__.py. Fetched 2026-08-12.

[8] Launch script for 2B training (`NPROC_PER_NODE`, `CUDA_VISIBLE_DEVICES` defaults, `swift sft --config` call). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/configs/qwen3_vl_150k_llmmeta_vit_proxy/train_prsimvl_2b.sh. Fetched 2026-08-12.

[9] Upstream ms-swift Command-line-parameters docs, read at the `v3.12.0` tag that matches PRISM-VL's vendored `swift/version.py` (deepspeed/FSDP2 options, report_to default, log_completions, save_only_model contract, resume_from_checkpoint/resume_only_model, adapters vs model, checkpoint versioning via add_version, gradient_checkpointing and learning-rate defaults). https://raw.githubusercontent.com/modelscope/ms-swift/v3.12.0/docs/source_en/Instruction/Command-line-parameters.md. Fetched 2026-08-12. This is the pinned `v3.12.0` build, distinct from the live `https://swift.readthedocs.io/en/latest/` referenced in "Find it in the docs," which tracks ms-swift's current `main` and is not pinned to 3.12.0.

[10] PRISM-VL GitHub Releases list (empty - no tagged releases). https://api.github.com/repos/kepengxu/PRISM-VL/releases. Fetched 2026-08-12.

[11] PRISM-VL repository metadata and commit history (created_at, pushed_at, four commits all messaged "first commit," license, description). https://api.github.com/repos/kepengxu/PRISM-VL and https://api.github.com/repos/kepengxu/PRISM-VL/commits. Fetched 2026-08-12.

[12] Vendored `setup.py` (`python_requires='>=3.8.0'`, Alibaba copyright header). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/setup.py. Fetched 2026-08-12.

[13] Repository `LICENSE` file (Apache License 2.0 text). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/LICENSE. Fetched 2026-08-12.

[14] `requirements/framework.txt`, pulled in by `requirements.txt` (dependency floors/caps: transformers, datasets, trl, peft, gradio, modelscope). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/requirements/framework.txt and https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/requirements.txt. Fetched 2026-08-12.

[15] `inference/README.md` (service demo commands, adapter loading with base model, demo images). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/inference/README.md. Fetched 2026-08-12.

[16] `eval/README.md` (benchmark wrapper behavior, dry-run command, data contract). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/eval/README.md. Fetched 2026-08-12.

[17] `RELEASE_MANIFEST.md` (pruned-checkpoint contract: "Only inference-required adapter files are included..."). https://github.com/kepengxu/PRISM-VL/blob/c73957b05fdc41cdfc3eb97c91a5e68251ef0681/RELEASE_MANIFEST.md. Fetched 2026-08-12.

[18] Upstream ms-swift README (documentation site link, confirming `https://swift.readthedocs.io/en/latest/` as the live docs address). https://raw.githubusercontent.com/modelscope/ms-swift/v3.12.0/README.md. Fetched 2026-08-12.

[19] Upstream ms-swift `swift/trainers/rlhf_trainer/__init__.py` at the `v3.12.0` tag, fetched to confirm it is byte-identical to PRISM-VL's vendored copy [7]. https://raw.githubusercontent.com/modelscope/ms-swift/v3.12.0/swift/trainers/rlhf_trainer/__init__.py. Fetched 2026-08-12.
