# InternVL

A research codebase for OpenGVLab's InternVL family of open-source vision-language models: clone the repo, run its `torchrun` shell scripts for supervised fine-tuning or preference optimization on a released InternVL checkpoint - there is no pip package or trainer-class API.

InternVL's README frames the project as an open-source multimodal (vision-language) model family that has iterated through InternVL, InternVL2, InternVL2.5, InternVL3, and InternVL3.5; several of its dated news entries compare against named closed-source models - InternVL2 and InternVL2.5 as matching the performance of GPT-4o, InternVL-Chat-V1.5 as approaching (not matching) GPT-4V and Gemini Pro - while the InternVL3 and InternVL3.5 entries instead claim state-of-the-art only among open-source MLLMs, without a closed-source comparison [1]. It is built by OpenGVLab, and the repository is copyrighted to OpenGVLab in its licence file [3]. It lives at https://github.com/OpenGVLab/InternVL [2]. The codebase's post-training surface is two plain scripts: `internvl_chat/internvl/train/internvl_chat_finetune.py`, a Hugging Face `Trainer` subclass driven by dataclass `ModelArguments`/`DataTrainingArguments` and launched with `torchrun` [4], and `internvl_chat/internvl/train/trainer_dpo.py`, whose `MultimodalDPOTrainer` directly subclasses `trl.DPOTrainer` for preference optimization [5].

**When to pick it**: pick InternVL only when the goal is to post-train an InternVL-family checkpoint itself (SFT or the paper's own Mixed Preference Optimization recipe) - the repo ships no generic multi-model trainer, no PyPI package, and its main codebase pins `transformers==4.37.2` [6], an old floor that a chooser already running a newer transformers stack for another library will need a separate environment to satisfy. It is not a comparison point on API breadth or scale-out ergonomics against trainer-class libraries like trl or verl (cross-reference; not covered here) - InternVL's own docs make no such comparison, and none is measured here.

**Methods it ships**: SFT, called "2nd fine-tuning" in the docs, launched via `sh internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_full.sh` and driven by `internvl_chat_finetune.py` [7]. Preference optimization is "Mixed Preference Optimization" (MPO): `internvl_chat/internvl/train/internvl_chat_mpo.py` calls `MultimodalDPOTrainer(DPOTrainer)` from `trainer_dpo.py`, which imports `from trl import DPOTrainer` directly [5]; the docs state plainly "Please use trl==0.10.1 to ensure the model works normally" [8]. MPO mixes DPO loss types by name and weight - the shipped script sets `--loss_type sigmoid,bco_pair --sigmoid_loss_weight 0.8 --bco_pair_loss_weight 0.2` plus `--rpo_alpha 1` (an NLL regularization term) [9]. LoRA fine-tuning is a flag on the same SFT entry point (`use_backbone_lora`, `use_llm_lora` in `ModelArguments`) [4], not a separate method. A newer, separate codebase, `internvl_chat_gpt_oss/`, holds SFT training for the GPT-OSS-20B-A4B-based InternVL3.5 variant and needs its own conda environment package [10]. The root README's 2025/08/30 entry also names "CascadeRL", a two-stage offline-then-online RL recipe whose online stage runs through a different repository (`Weiyun1025/verl-internvl`, on verl) [1] - not investigated further here since it is outside this repo.

**Scale it handles**: the finetune docs give per-model-size GPU counts, current as of their "Last updated on May 29, 2025" date: InternVL3-1B/2B need 8x 32G/40G GPUs full-parameter or 2x for LoRA; 8B/9B/14B need 8x A100 80G full or 2x for LoRA; 38B needs 16x A100 80G full (via SLURM) or 2x for LoRA; 78B needs 32x A100 80G full (via SLURM) or 8x for LoRA [11]. Multi-GPU, single node runs through `torchrun --nnodes=1 --nproc_per_node=${GPUS}`, as in the shipped SFT script [7]; multi-node MPO runs the same `internvl_chat_mpo.py` under a SLURM `srun` launcher with `--nodes`/`--ntasks-per-node` [9]. Sharding is DeepSpeed ZeRO, selected by config file path (`--deepspeed zero_stage1_config.json` for the SFT default, `zero_stage3_config_100b_1e8.json` for the shipped 8B MPO run) [7][9]; the repo does not document FSDP as an alternative in the pages read here. This is mechanism only: the finetune docs page gives GPU-count requirements per model size but no throughput, wall-clock, or tokens/samples-per-second numbers for any of these configurations, and no such benchmark was found elsewhere in the pages read for this card [11].

**Install**: no PyPI package exists - `pip install internvl` returns nothing on PyPI (checked 2026-08-10) - so the install is `git clone https://github.com/OpenGVLab/InternVL.git`, then `conda create -n internvl python=3.9 && conda activate internvl`, then `pip install -r requirements.txt`, which chains to `requirements/internvl_chat.txt` (plus `streamlit_demo.txt`, `classification.txt`, `segmentation.txt`; `clip_benchmark.txt` is opt-in) [12]. `requirements/internvl_chat.txt`, read at the pinned commit `2410d1dbf208f0e799459aff9376e5747dbf41a2`, hard-pins `transformers==4.37.2`, `tokenizers==0.15.1`, `peft==0.10.0`, `numpy==1.26.4`, `timm==0.9.12`, `sentencepiece==0.1.99`, `einops==0.6.1`, `bitsandbytes==0.42.0`, and caps `accelerate<1`; it floors `torch>=2` and `deepspeed>=0.13.5` without an upper bound [6]. `transformers==4.37.2` is the collision risk: it is an exact pin against a fast-moving package a reader is likely already running at a newer version for other work. MPO additionally needs `pip install trl==0.10.1` per the docs [8]. The repository's own MIT licence file names OpenGVLab as copyright holder [3], matching the GitHub API's reported licence [13], though `internvl_chat/pyproject.toml` separately carries an Apache-2.0 classifier for that subpackage [21] - the two disagree and this card defers to the repository-level MIT licence file. No CUDA or GPU-driver minimum is stated in the installation page read here; `flash-attn==2.3.6` is offered as an optional extra install [12]. GitHub Releases are stale: the newest tag is `v1.5.0`, published 2024-05-08 [14], while the repository has continued receiving commits through 2025-09-22 [15] - so the release history under-represents current code, and any read of a source file below is dated by its own commit, not by a release tag.

**Maintained by**: OpenGVLab; the repository shows 10,120 stars as a live count read 2026-08-10 (not a ranking signal) [13]. Active development continues past the last release: the pinned commit's own message is "Merge pull request #1165 ... Fix multi-round conversation template for GPT-OSS" [15], and the README's 2025/08/26 and 2025/08/30 news entries announce InternVL3.5 and the CascadeRL open-sourcing [1].

## Quick start

There is no post-training "hello world" one-liner in the docs read for this card; the smallest real unit is the shipped shell script plus its dataset JSON. The SFT launch script, `internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_full.sh`, read verbatim at the pinned commit, is a complete, runnable form [7]:

```bash
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

OUTPUT_DIR='work_dirs/internvl_chat_v3/internvl3_8b_dynamic_res_2nd_finetune_full'

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "OpenGVLab/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/internvl_1_2_finetune_custom.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard"
```

(the script also sets `set -x`, exports `PYTHONPATH`, `MASTER_PORT`, `TF_CPP_MIN_LOG_LEVEL`, and `LAUNCHER`, makes `OUTPUT_DIR` with a `mkdir -p` guard, and pipes its output through a trailing `tee` for log capture; this block omits all of that shell scaffolding but keeps every `torchrun`/training CLI flag as shipped [7].) `--do_train True` is what actually triggers `trainer.train()` inside `internvl_chat_finetune.py` - `transformers.TrainingArguments.do_train` defaults to `False`, so a copy of this command missing that flag would run to completion without training anything [4][7]. `--meta_path` points at a JSON registry of named datasets (root path, annotation-file path, `repeat_time`); the finetune docs show the schema and a `huggingface-cli download` command for each supported InternVL3 checkpoint size [11]. The MPO (DPO-family) launch script is the same shape, calling `internvl/train/internvl_chat_mpo.py` under `srun` instead of bare `torchrun`, with the MMPR preference dataset in place of `--meta_path` [9]. Both are shell scripts, not a Python trainer API - there is no importable one-liner like `Trainer(...).train()` documented for InternVL itself.

## Start it

- Single GPU: set `GPUS=1` and drop `torchrun`'s multi-process flags to run the same script above on one device; the docs do not show this form explicitly, but the script's `GPUS` variable is designed to be overridden [7].
- Multiple GPUs, one node: `torchrun --nnodes=1 --nproc_per_node=${GPUS} ...`, as shipped [7].
- Multiple nodes: the shipped MPO script wraps the same Python entry point in a SLURM `srun` call with `--nodes=$((GPUS / GPUS_PER_NODE))` and `--ntasks-per-node=${GPUS_PER_NODE})` [9]; no non-SLURM multi-node launcher is documented in the pages read for this card.
- Effective batch size is computed by the script itself: `GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))`, so `BATCH_SIZE` (target effective batch, e.g. 128 for the SFT script, 256 for the MPO script) stays fixed while `GRADIENT_ACC` is derived when you change `GPUS` or `PER_DEVICE_BATCH_SIZE` [7][9].
- Sharding is selected by pointing `--deepspeed` at one of the repo's ZeRO config JSON files; the shipped ZeRO-3 config sets `stage3_gather_16bit_weights_on_model_save: true`, which is what makes `trainer.save_model()` produce full, unsharded weights rather than per-rank shards [16].
- LoRA is a `ModelArguments` flag rather than a separate script: `--use_backbone_lora <r>` and/or `--use_llm_lora <r>` wrap the vision backbone and/or LLM in a LoRA adapter inside `internvl_chat_finetune.py`, with `lora_alpha` fixed at `2*r` in the code [4].
- Config surface is two dataclasses (`ModelArguments`, `DataTrainingArguments`) plus the standard Hugging Face `TrainingArguments`, since `internvl_chat_finetune.py` uses `transformers.HfArgumentParser` over all three, not a custom Config class [4]. Because the pinned `transformers==4.37.2` predates the newer `eval_strategy` naming, the shipped scripts pass the older `--evaluation_strategy` flag [7][9] - a silent version-specific naming trap if you swap in a newer transformers.
- Out-of-memory first aid is not written up as a checklist on any docs page read for this card; the mechanism available is: lower `--per_device_train_batch_size` and let the script's own arithmetic raise `--gradient_accumulation_steps` to hold the target batch size, or switch `--deepspeed` to a higher ZeRO stage / offload config, or add `--use_backbone_lora`/`--use_llm_lora` to train an adapter instead of full parameters [4][7].

## Watch it

This section covers only the mechanics of what InternVL turns on and where it goes; what a healthy value looks like for SFT loss or the DPO/MPO reward margin belongs on those methods' own cards, not here.

- Logging is off by default and opt-in per run: the shipped scripts pass `--report_to "tensorboard"` [7][9], the same `TrainingArguments` field used across the Hugging Face ecosystem; nothing is logged anywhere if this flag is omitted.
- Because both trainers are plain (or lightly subclassed) Hugging Face `Trainer`/`trl.DPOTrainer` instances rather than InternVL-specific loggers, the metric names that appear are whatever those base classes emit (loss, learning rate, grad norm for SFT; DPO's reward/margin family for MPO) - no InternVL-specific metrics page was found among the docs read for this card, and no InternVL-authored table of MPO's logged metric names was located either.
- `--logging_steps 1` in both shipped scripts logs every step [7][9]; `evaluation_strategy` is set to `"no"` in both, so neither shipped script evaluates during training out of the box [7][9].
- No sample-level generation logging (dumping decoded completions during training) is documented on the pages read for this card.
- No InternVL-published stopping rule, threshold, or patience value was found on the finetune or preference-optimization docs pages read here [11][8]; those pages state fine-tuning and MPO procedure and hyperparameters but no early-stopping mechanism.

## Save it

- Checkpoints land under `--output_dir` (e.g. `work_dirs/internvl_chat_v3/internvl3_8b_dynamic_res_2nd_finetune_full`), governed by the standard Hugging Face `TrainingArguments` fields `--save_strategy`, `--save_steps`, and `--save_total_limit` (the shipped SFT script saves every 200 steps, keeping only the latest) [7].
- With DeepSpeed ZeRO-3, the shipped config's `stage3_gather_16bit_weights_on_model_save: true` is what makes the saved checkpoint a normal, unsharded Hugging Face directory rather than per-rank shards [16]; the docs read for this card do not spell out the file-by-file contents of that directory beyond this.
- LoRA runs save only the adapter through the standard `use_backbone_lora`/`use_llm_lora` path in `internvl_chat_finetune.py` [4]; the repo ships a separate script, `tools/merge_lora.py`, to merge a saved LoRA adapter into the base model and produce a standalone checkpoint - an adapter checkpoint on its own is not a full model until this merge step runs.
- No InternVL-specific `push_to_hub`/resume documentation was found on the pages read for this card beyond the standard Hugging Face `Trainer` save/resume behavior inherited by both entry points; readers relying on `resume_from_checkpoint` should verify against the `transformers` version actually pinned (4.37.2) rather than the latest `transformers` docs.
- Whether an evaluator can load a saved InternVL checkpoint directly depends on whether it is a merged full model or a bare LoRA adapter directory (see above) - that loader contract is outside this card's scope.

## Find it in the docs

- Address pattern: `https://internvl.readthedocs.io/en/latest/<generation>/<page>.html`. There is no per-version URL substitution - checked 2026-08-10, `/en/v2.5.0/` 404s - the site publishes a single "latest" build organized by model-generation subfolders (`internvl3.0/`, `internvl2.5/`, `internvl2.0/`, `internvl1.5/`, `internvl1.2/`, `internvl1.1/`, `internvl1.0/`) rather than by release tag [17]. Because of this, always confirm which generation subfolder you are reading, not which "version" you think you are on.
- Question-to-slug map: fine-tuning a custom dataset -> `<generation>/finetune.html` (read at `internvl3.0/finetune.html` for this card) [11]; preference optimization / MPO -> `<generation>/preference_optimization.html` [8]; installation -> `get_started/installation.html` [12]; FAQs -> `tutorials/faqs.html` [18]; local chat demo -> `get_started/local_chat_demo.html` (fetched but not read in detail for this card).
- Runnable references beyond the docs: the `internvl_chat/shell/` tree in the repo holds the actual launch scripts referenced above, organized by generation and method (e.g. `internvl3.0/2nd_finetune/`, `internvl3.0/mpo/`) [7][9]; `internvl_chat/tools/merge_lora.py` is the LoRA-merge utility named above.
- Community/tutorial layer, curated door first: the docs' own tutorials section hosts "Enhancing InternVL2 on COCO Caption Using LoRA Fine-Tuning" as an official worked example [19] - prefer this over unofficial blog posts, and check any external post's pinned InternVL/transformers versions against your own before trusting its numbers, since `transformers==4.37.2` is an old, specific pin that community posts may not match.
- A trap from a maintainer reply: in GitHub issue #350 (LoRA/full-parameter finetuning errors on InternVL2-26B), OpenGVLab collaborator `G-z-w` twice redirected reporters to the (then-new) `internvl2.0/finetune.html#start-2nd-fine-tuning` page as the fix, on 2024-07-31 [20] - a sign that finetuning issues are frequently docs-drift problems (an older tutorial no longer matching the current entry point) rather than code bugs, so check the generation-specific finetune page for your model before filing an issue.
- Honest boundary: this card found no InternVL-specific hyperparameter-search, RLHF-beyond-MPO (e.g. no PPO/GRPO trainer), or non-SLURM multi-node launcher documented in the pages read; MPO's own docs require pinning to the old `trl==0.10.1` [8], so this repo is not a path to newer trl-only DPO features.

## Sources

All pages are unpinned "latest" readthedocs builds unless a commit is named; all fetched 2026-08-10 except where a docs page states its own "Last updated" date. Ecosystem tools named in passing (transformers, trl, DeepSpeed, PEFT/LoRA, SLURM, VLMEvalKit, verl) are deliberately not enumerated as references beyond the specific files cited.

[1] InternVL root README. https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/README.md. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[2] InternVL GitHub repository (item home). https://github.com/OpenGVLab/InternVL. Fetched 2026-08-10.

[3] InternVL LICENSE file. https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/LICENSE. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[4] `internvl_chat/internvl/train/internvl_chat_finetune.py`. https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/internvl_chat_finetune.py. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[5] `internvl_chat/internvl/train/trainer_dpo.py`. https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/trainer_dpo.py. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[6] `requirements/internvl_chat.txt`. https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/requirements/internvl_chat.txt. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[7] SFT launch script, `internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_full.sh`. https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_full.sh. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[8] InternVL3.0 "Mixed Preference Optimization" docs page. https://internvl.readthedocs.io/en/latest/internvl3.0/preference_optimization.html. Page states "Last updated on May 29, 2025." Fetched 2026-08-10.

[9] MPO launch script, `internvl_chat/shell/internvl3.0/mpo/internvl3_8b_mpo.sh`. https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/shell/internvl3.0/mpo/internvl3_8b_mpo.sh. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[10] `internvl_chat_gpt_oss/README.md`. https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat_gpt_oss/README.md. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[11] InternVL3.0 "Fine-tune on a Custom Dataset" docs page. https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html. Page states "Last updated on May 29, 2025." Fetched 2026-08-10.

[12] InternVL "Installation" docs page. https://internvl.readthedocs.io/en/latest/get_started/installation.html. Fetched 2026-08-10.

[13] GitHub API repository metadata for OpenGVLab/InternVL. https://api.github.com/repos/OpenGVLab/InternVL. Live endpoint, no revision parameter; fetched 2026-08-10 (stargazers_count and licence read at fetch time).

[14] GitHub API releases list for OpenGVLab/InternVL. https://api.github.com/repos/OpenGVLab/InternVL/releases. Live endpoint, no revision parameter; fetched 2026-08-10.

[15] GitHub API commit metadata for commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. https://api.github.com/repos/OpenGVLab/InternVL/commits/2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[16] ZeRO-3 DeepSpeed config, `internvl_chat/zero_stage3_config_100b_1e8.json` (fetched under the file name `zero_stage3_config.json` in scratch). https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/zero_stage3_config_100b_1e8.json. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.

[17] InternVL readthedocs landing page (version/build listing). https://internvl.readthedocs.io/en/latest/. Fetched 2026-08-10.

[18] InternVL FAQs docs page. https://internvl.readthedocs.io/en/latest/tutorials/faqs.html. Fetched 2026-08-10.

[19] "Enhancing InternVL2 on COCO Caption Using LoRA Fine-Tuning" docs tutorial page. https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html. Fetched 2026-08-10.

[20] GitHub issue #350 comments, OpenGVLab/InternVL. https://github.com/OpenGVLab/InternVL/issues/350 (comments fetched via the GitHub API at https://api.github.com/repos/OpenGVLab/InternVL/issues/350/comments, since the issue-object endpoint itself 404s for this issue while its comments endpoint resolves). Fetched 2026-08-10.

[21] `internvl_chat/pyproject.toml`. https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/pyproject.toml. Read at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2. Fetched 2026-08-10.
