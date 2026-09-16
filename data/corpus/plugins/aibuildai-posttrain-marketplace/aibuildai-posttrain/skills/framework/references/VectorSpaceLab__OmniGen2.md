# OmniGen2

Repository: https://github.com/VectorSpaceLab/OmniGen2

A single model's own repo, not a general framework: clone it to run, fine-tune, or GRPO-fine-tune the OmniGen2 image-generation/editing model specifically, with a reward server it ships for the RL leg.

**OmniGen2** is described by its own README as "a powerful and efficient generative model" with two decoupled decoding pathways for text and image, covering visual understanding, text-to-image generation, instruction-guided image editing, and in-context generation [1]. It is built and maintained by VectorSpaceLab, the authors of the accompanying technical report "OmniGen2: Towards Instruction-Aligned Multimodal Generation" [2][1]. The repository is a monolithic training-and-inference codebase around one architecture: `inference.py` / `app.py` run the released checkpoint, `train.py` plus `docs/FINETUNE.md` do full or LoRA supervised fine-tuning, and a nested `OmniGen2-RL/` subproject (with its own `train.py`) does online RL fine-tuning of the image-editing model against a reward server [1][3].

**When to pick it**: pick this repo only if the object you want to train IS OmniGen2 (or a fine-tune of it) - it is not a library you point at an arbitrary model, unlike trl or verl (cross-reference; not covered here). Within that scope, use `docs/FINETUNE.md`'s full/LoRA path for supervised fine-tuning on your own edit/T2I data, and `OmniGen2-RL/` for online reinforcement fine-tuning of the image-editing capability against a vision-language reward model called EditScore, using a FlowGRPO-style clipped-ratio policy-gradient loss with group-normalized advantages [3][4]. The RL leg was added on 2025-09-30, later than the base model and SFT path, which shipped 2025-06-16 and 2025-06-30 respectively [1].

**Methods it ships**: two, both scoped to the OmniGen2 architecture only, with no formal taxonomy page - this list is this card's own grouping of the repo's training entry points. (1) Supervised full-parameter or LoRA fine-tuning via the root `train.py`, configured from `options/ft.yml` or `options/ft_lora.yml`, documented in `docs/FINETUNE.md` [5]. (2) Online RL fine-tuning via `OmniGen2-RL/train.py`, configured from `OmniGen2-RL/options/*.yml`; its loss function `compute_single_step_ppo_loss` in `OmniGen2-RL/omnigen2/grpo/utils.py` implements a clipped surrogate ("Calculate two types of loss: original and clipped" / "Take maximum value (more conservative loss)" [4]) over per-step flow-matching log-probabilities, with advantages computed by `process_grpo_rewards` as within-prompt-group, reward-standardized values [4]. The reward comes from a separately launched EditScore reward server, reached over HTTP through `OmniGen2-RL/omnigen2/grpo/reward_client_edit.py`'s `RewardClient` class (the shortlist's "REWARD" hit is this client/server pair, not a distinct trainer) [6]. EditScore itself - the reward model - is a separate repository the OmniGen2 README points to for its own quickstart [1]. The shipped single-machine RL option file trains LoRA by default, not the full model: it sets `lora_ft: true` with `lora_rank: 32`, `lora_alpha: 64`, and its own conversion script writes a `transformer_lora/` directory unconditionally - a full-parameter RL run is not the template's out-of-the-box behavior and would require changing `lora_ft` to `false` in a copy of the option file [8][15].

**Scale it handles**: single GPU up to multi-node, all launched through Hugging Face Accelerate with PyTorch FSDP, but the two training paths wrap and checkpoint differently: SFT's `scripts/train/ft.sh` uses `--fsdp_sharding_strategy HYBRID_SHARD_ZERO2` with `--fsdp_state_dict_type FULL_STATE_DICT` (one merged file per save), while RL's `omnigen2_edit_rl_single_machine_editscore7b.sh` uses `--fsdp_sharding_strategy HYBRID_SHARD` with `--fsdp_state_dict_type SHARDED_STATE_DICT` (per-rank shards) - both wrap on `--fsdp_transformer_layer_cls_to_wrap OmniGen2TransformerBlock` [7][13]. Each launcher (`scripts/train/ft.sh` for SFT, `OmniGen2-RL/scripts/train/omnigen2_edit_rl_single_machine_editscore7b.sh` for RL) takes `--rank`, `--master_addr`, `--master_port`, `--world_size` to add machines - single-node needs no flags [5][7][13]. Multi-machine RL is documented as mechanism only: the repo ships a 4-machine, 8-GPU-per-machine option file and script (`omnigen2_edit_rl_4machine_editscore7b_avg4.yml`/`.sh`), differing from the single-machine file only in `global_batch_size` (576 vs 144) and `num_unique_prompts_per_sampling` (48 vs 12) - no published multi-node throughput or reward-curve benchmark accompanies it [8]. The EditScore reward server that RL training queries runs as its own multi-machine proxy/worker stack, launched separately from training GPUs via `reward_server/start_multi_machines.sh` [6].

**Install**: no PyPI package - clone the repo and `pip install -r requirements.txt` at the commit/branch you check out; the README additionally pins `pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124` before that, and an optional `pip install flash-attn==2.7.4.post1 --no-build-isolation` for speed, noting flash-attn is not required [1]. `requirements.txt` at this card's commit pins `torch==2.6.0`, `torchvision==0.21.0`, and `transformers==4.51.3`, with `accelerate`, `diffusers`, `wandb`, `omegaconf` unpinned [9]. The RL subproject's own `OmniGen2-RL/requirements.txt` adds `editscore`, `flash_attn`, `flask`, `diffusers`, `wandb`, `omegaconf`, also unpinned [10]. No `setup.py`/`pyproject.toml` exists, so there is no declared Python floor; the README's own setup instructions use `conda create -n omnigen2 python=3.11` as its recommended interpreter, not a stated minimum [1]. Licence is Apache-2.0 [11]. For inference the README states a hardware floor of an NVIDIA RTX 3090 or "approximately 17GB of VRAM," reducible via `enable_model_cpu_offload` or `enable_sequential_cpu_offload`; no comparable VRAM floor is published for training (SFT or RL) [1].

**Maintained by**: VectorSpaceLab; 4,112 GitHub stars as a raw count, not a ranking signal [17]. The repo is actively pushed - most recent push 2026-03-20 [17] - and the RL subproject was added later than the base release: a maintainer with COLLABORATOR association confirmed on 2025-09-30, in reply to a 2025-07-12 issue asking for an RL roadmap, that "the RL version has been updated... powered by our new reward model" [12].

## Quick start

Base model, from the README's recommended setup, verbatim structure [1]:

```bash
git clone git@github.com:VectorSpaceLab/OmniGen2.git
cd OmniGen2
conda create -n omnigen2 python=3.11
conda activate omnigen2
pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation   # optional
```

Then run one of the ready example scripts, each a complete program end to end [1]:

```bash
bash example_t2i.sh          # text-to-image
bash example_edit.sh         # instruction-guided image editing
bash example_in_context_generation.sh
bash example_understanding.sh
```

RL fine-tuning quick start, from `OmniGen2-RL/README.md` [3]:

```bash
cd OmniGen2-RL
pip install -r requirements.txt
python scripts/misc/extract_bin_from_pipe.py   # prepare the base checkpoint
bash reward_server/start_multi_machines.sh --model_name=editscore_7B --config_path=reward_server/server_configs/editscore_7B.yml
bash scripts/train/omnigen2_edit_rl_single_machine_editscore7b.sh
```

## Start it

- **SFT (root `train.py`)**: single node needs no distributed flags - `bash scripts/train/ft.sh`; multi-node passes `--rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --world_size=$WORLD_SIZE` to the same script, which itself calls `accelerate launch --use_fsdp --fsdp_sharding_strategy HYBRID_SHARD_ZERO2 --fsdp_transformer_layer_cls_to_wrap OmniGen2TransformerBlock --fsdp_state_dict_type FULL_STATE_DICT train.py --config options/ft.yml` [5][7]. LoRA fine-tuning uses the sibling `scripts/train/ft_lora.sh` with `options/ft_lora.yml` [5].
- **RL (`OmniGen2-RL/train.py`)**: the reward server must be running first - `bash reward_server/start_multi_machines.sh --model_name=<name> --config_path=<server yml>` - then launch training with `bash scripts/train/omnigen2_edit_rl_single_machine_editscore7b.sh` (single machine) or the `_4machine_` variant with the same `--rank/--master_addr/--master_port/--world_size` flags [3][13]. Both launchers call `accelerate launch ... --use_fsdp --fsdp_sharding_strategy HYBRID_SHARD --fsdp_transformer_layer_cls_to_wrap OmniGen2TransformerBlock --fsdp_state_dict_type SHARDED_STATE_DICT train.py --config options/<name>.yml` [13].
- **Effective-batch arithmetic (RL)**: at the single-machine template's values, `train.global_batch_size` (144) = `train.rl.num_unique_prompts_per_sampling` (12) x `train.rl.num_images_per_prompt` (12) - the images generated per sampling phase before a policy update; `train.batch_size` (18) = `train.rl.batch_size_per_forward` (9) x `gradient_accumulation_steps` (1) x `train.rl.num_update_steps_per_sampling` (2) - the images actually processed per forward across the gradient updates drawn from that one sampling phase [3][8]. Setting `num_update_steps_per_sampling > 1` reuses one batch of generations for more than one gradient step, which the RL README calls off-policy RL for better sample efficiency [3].
- **Config surface**: SFT/LoRA templates are `options/ft.yml` / `options/ft_lora.yml`; RL templates are `OmniGen2-RL/options/omnigen2_edit_rl_*.yml` [5][8]. The RL template's own defaults - `mixed_precision: 'bf16'` and `gradient_checkpointing: true` - are the values actually run, i.e. a bf16-capable GPU is assumed by the shipped config, not merely supported [8].
- **Generation-side layout (RL only)**: reward scoring is a separate service, not a training-process flag - `server_type: vlm` in the RL option file selects the EditScore server's serving mode, and the reward proxy fans requests out to one or more reward-server processes on their own machines/GPUs, entirely apart from the training GPUs [8][6].
- **OOM first aid**: for inference, the README's `enable_model_cpu_offload` ("Reduces VRAM usage by nearly 50% with a negligible impact on speed") and `enable_sequential_cpu_offload` ("Minimizes VRAM usage to less than 3GB, but at the cost of significantly slower performance") [1]. For RL training, no dedicated OOM guide exists; the reachable knobs are `train.batch_size_per_forward`, `train.rl.num_images_per_prompt`, and the data-side `max_output_pixels`/`max_input_pixels`/`max_side_length` fields, all present on the option template but not framed by the docs as an OOM checklist [8].

## Watch it

Mechanics only - what a given metric means for GRPO health lives on that method's own card, not here.

- **Enable it**: the RL option template sets `logger.log_with: [wandb, tensorboard]`; this is passed straight into `Accelerator(log_with=...)` and `accelerator.init_trackers("OmniGen2-RL", ...)` in `OmniGen2-RL/train.py`, so logging is on by default in the shipped template, but turning it off (`log_with: ~`, commented as an alternative in the same file) leaves the run with no tracked record beyond the console progress bar [8][14].
- **RL metric names**, read directly from where `OmniGen2-RL/train.py` builds its `logs` dict and calls `accelerator.log(logs, step=global_step)` [14]: `policy_loss`, `policy_loss_unclipped`, `policy_loss_clipped`, `policy_clip_frac`, `approx_kl`, `ratio`, `ratio_positive`, `ratio_negative`, `ratio_large_than_1`, `ratio_small_than_1`, `num_positive`, `num_negative`, `advantages` (plus `_std`, `_min`, `_max`), `kl_loss` (only logged when the config's `train.rl.kl_loss_weight` is non-zero - the shipped template sets it to 0.04), `loss`, `grad_norm`, `lr`, and the reward-side `rewards_min`, `rewards_max`, `rewards_mean`, `rewards_std`, `zero_std_ratio` [14][8]. `zero_std_ratio` is the deciding number for whether the GRPO-style step is producing any learning signal at all: it is the fraction of prompt groups in the batch whose sampled rewards all came back identical, which zeroes that group's advantage in `process_grpo_rewards`'s `(reward - mean) / (std + 1e-8)` computation [14][4].
- **SFT logging**: the root `train.py` was not opened for this card; no metric-name list is given here for the SFT path (search only covered the RL trainer file [14] and `docs/FINETUNE.md` [5], neither of which enumerates SFT's logged metrics).
- **Sample-level logging (RL)**: `val.train_visualization_interval` (default 5 steps) and `val.num_train_visualization_samples` (default 3) in the option template control a routine that saves, directly under `output_dir` as files rather than through the tracker, the input image(s), the generated output image, and a per-sample text file recording `instruction`, `reward`, `advantages`, `reasoning` (the EditScore server's returned justification text), and `meta_data` [8][14].
- **Evaluation during training**: none found. Neither `OmniGen2-RL/train.py` nor `OmniGen2-RL/README.md` defines an in-loop eval dataset or eval-step cadence; evaluation is a separate offline step after training, run on GEdit-Bench against a checkpoint that has already been converted to Hugging Face format [14][3].
- **Stopping**: none found. The RL option template, `OmniGen2-RL/README.md`, and `OmniGen2-RL/train.py` publish no early-stopping field, reward threshold, or patience value; `train.max_train_steps` (1000 in the template) is the only run-length control read in these files [8][3][14].

## Save it

- **RL checkpoints**: `accelerator.save_state(save_path)` runs every `logger.checkpointing_steps` (50 in the template) and at the end of training, writing to `experiments/<experiment_name>/checkpoint-<global_step>/`; under the FSDP `SHARDED_STATE_DICT` mode this directory contains a `pytorch_model_fsdp_0/` subdirectory of sharded weights, as referenced by the shipped conversion script's `--model_path $model_path/pytorch_model_fsdp_0` argument - this raw checkpoint is NOT directly loadable for inference [14][15]. `logger.checkpoints_total_limit` (unset/`~` in the template, meaning unlimited) controls retention; when set, the oldest `checkpoint-*` directories are deleted with `shutil.rmtree` once the count is exceeded [8][14].
- **Conversion is mandatory before inference or evaluation on either path, but the number of steps differs because the two launchers save in different FSDP formats** (see Scale it handles). SFT's `FULL_STATE_DICT` save already writes one `pytorch_model_fsdp.bin` per checkpoint, so `docs/FINETUNE.md` documents a single step: run `convert_ckpt_to_hf_format.py --config_path ... --model_path .../pytorch_model_fsdp.bin --save_path .../transformer` (or `.../transformer_lora` for a LoRA run) directly [5]. RL's `SHARDED_STATE_DICT` save instead leaves per-rank shards under `pytorch_model_fsdp_0/`, so it needs a shard-merge step first: `scripts/misc/convert_dist_ckpt_to_ckpt.py` merges those shards into one `pytorch_model_fsdp.bin`, then `scripts/misc/convert_ckpt_to_hf_format.py` turns that `.bin` into a `transformer_lora/` directory - the RL side wraps both steps in `bash OmniGen2-RL/scripts/misc/convert_dist_ckpt_to_hf_format.sh [EXPERIMENT_NAME] [STEP_NUMBER]` [15]. The root repository's own file listing has `convert_ckpt_to_hf_format.py` but no `convert_dist_ckpt_to_ckpt.py` - the shard-merge step exists only under `OmniGen2-RL/scripts/misc/` [16].
- **LoRA is not adapter-only here**: `docs/FINETUNE.md` states plainly, "Currently, when training with LoRA, the script saves the entire model's parameters (including the frozen base model weights) in the checkpoint. This is due to a limitation in easily extracting only the LoRA-related parameters when using FSDP. The conversion script in the next step will correctly handle this" [5] - so the raw FSDP checkpoint is a full-size dump regardless of LoRA, and the `transformer_lora/` directory the conversion script produces is the practical artifact to keep.
- **Resume**: setting `resume_from_checkpoint: latest` (the template's default) makes `OmniGen2-RL/train.py` scan `output_dir` for `checkpoint-*` directories and resume from the highest step; a specific path can be given instead of `latest` [8][14].
- **Reload for inference**: `inference.py --model_path "OmniGen2/OmniGen2" --transformer_path <converted transformer dir>` for a full fine-tune, or `--transformer_lora_path <converted transformer_lora dir>` for a LoRA fine-tune - in both cases the base pretrained model is still loaded via `--model_path` and the converted directory supplies only the trained weights layered on top, so `transformer_lora/` alone is not a standalone model [5]. Whether a downstream evaluator can load the converted directory directly is that evaluator's own loader contract, not something this card's sources state.

## Find it in the docs

There is no separate hosted documentation site or versioned docs domain for this repo - "the docs" are markdown files inside the GitHub repository itself, read at a branch or commit.

- **Address pattern**: `https://raw.githubusercontent.com/VectorSpaceLab/OmniGen2/<ref>/<path>` for raw file content, or `https://github.com/VectorSpaceLab/OmniGen2/blob/<ref>/<path>` to view in the browser, where `<ref>` is a branch name (`main`) or a full commit SHA - verified working by fetching `README.md` at this card's pinned commit `18e6f9d5271b517fcb32e999f10df943ae9b8f20` [1].
- **File map**: root `README.md` covers install, inference, Gradio, resource requirements [1]; `docs/FINETUNE.md` covers SFT/LoRA fine-tuning end to end [5]; `OmniGen2-RL/README.md` covers RL fine-tuning end to end, including the reward-server setup [3]. There is no separate FAQ or troubleshooting page among these.
- **Runnable references beyond the docs**: the ready `example_*.sh` scripts and `example.ipynb` at the repo root for inference [1]; `evaluation/GEdit-Bench/` for the Best-of-N and post-RL evaluation scripts referenced by `OmniGen2-RL/README.md` [3]; `omnicontext/` for the OmniContext in-context-generation benchmark's own evaluation code [1].
- **Sibling project**: the reward model used for RL, EditScore, lives in its own repository, `https://github.com/VectorSpaceLab/EditScore`, which the OmniGen2 README points readers to directly for the reward model's own quickstart [1] - this card does not cover that repo.
- **Community layer**: the README's own "Community Efforts" section is the curated door here, listing ComfyUI integrations (the official ComfyUI examples page, and two independent GitHub ports) and a DFloat11 lossless-compression fork, with an explicit caveat: "Currently, we have not confirmed whether there are no bugs. Please try to use the our official demo as much as possible" [1]. No separate practitioner-blog curation page was found for this repo.
- **MCP**: none found. No official Model Context Protocol endpoint for these docs is referenced anywhere in the README or subproject READMEs read for this card.
- **A maintainer-confirmed trap, in the section where it bites**: RL fine-tuning was not available at release and had to be explicitly requested - a COLLABORATOR-association maintainer replied to issue #100 ("Any RL plan?") on 2025-07-12 with "Sure, we'll do that indeed," and confirmed on 2025-09-30 that "the RL version has been updated... powered by our new reward model" - so any card, benchmark, or workflow built against this repo before 2025-09-30 would not have had the RL path described above at all [12].

## Sources

All GitHub file reads are pinned to commit `18e6f9d5271b517fcb32e999f10df943ae9b8f20` (this card's screening commit, which is also the repository's newest push at fetch time - there is no separate tagged release to resolve against, since the repository publishes none, confirmed by an empty releases list [17]). Repo metadata (stars, license, push date) is read live from the GitHub API on 2026-08-11 and is a snapshot, not a trend [17]. ComfyUI, DFloat11, and EditScore are named only as pointers from the README's own links and are deliberately not fetched or enumerated as separate sources.

[1] OmniGen2 README at commit 18e6f9d5. https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/README.md. Fetched 2026-08-11.

[2] arXiv abstract page, "OmniGen2: Towards Instruction-Aligned Multimodal Generation" (arXiv:2506.18871). https://arxiv.org/abs/2506.18871. Fetched 2026-08-11.

[3] OmniGen2-RL README at commit 18e6f9d5. https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/OmniGen2-RL/README.md. Fetched 2026-08-11.

[4] OmniGen2-RL GRPO utilities (`process_grpo_rewards`, `compute_single_step_ppo_loss`) at commit 18e6f9d5. https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/OmniGen2-RL/omnigen2/grpo/utils.py. Fetched 2026-08-11.

[5] docs/FINETUNE.md at commit 18e6f9d5. https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/docs/FINETUNE.md. Fetched 2026-08-11.

[6] OmniGen2-RL reward client (`reward_client_edit.py`) and reward-server directory listing at commit 18e6f9d5. https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/OmniGen2-RL/omnigen2/grpo/reward_client_edit.py ; https://github.com/VectorSpaceLab/OmniGen2/tree/18e6f9d5271b517fcb32e999f10df943ae9b8f20/OmniGen2-RL/reward_server. Fetched 2026-08-11.

[7] SFT/LoRA launch script `scripts/train/ft.sh` at commit 18e6f9d5, read directly for its `accelerate launch` flags. https://raw.githubusercontent.com/VectorSpaceLab/OmniGen2/18e6f9d5271b517fcb32e999f10df943ae9b8f20/scripts/train/ft.sh. Fetched 2026-08-11.

[8] OmniGen2-RL option templates at commit 18e6f9d5 (`omnigen2_edit_rl_single_machine_editscore7b.yml`, `omnigen2_edit_rl_4machine_editscore7b_avg4.yml`) and reward-server config (`reward_server/server_configs/editscore_7B.yml`). https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/OmniGen2-RL/options/omnigen2_edit_rl_single_machine_editscore7b.yml . Fetched 2026-08-11.

[9] requirements.txt (root) at commit 18e6f9d5. https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/requirements.txt. Fetched 2026-08-11.

[10] OmniGen2-RL/requirements.txt at commit 18e6f9d5. https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/OmniGen2-RL/requirements.txt. Fetched 2026-08-11.

[11] LICENSE at commit 18e6f9d5 (Apache License 2.0). https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/LICENSE. Fetched 2026-08-11.

[12] GitHub issue #100, "Any RL plan?" (maintainer replies from a COLLABORATOR-association account). https://github.com/VectorSpaceLab/OmniGen2/issues/100. Fetched 2026-08-11.

[13] OmniGen2-RL train launch script at commit 18e6f9d5 (`scripts/train/omnigen2_edit_rl_single_machine_editscore7b.sh`). https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/OmniGen2-RL/scripts/train/omnigen2_edit_rl_single_machine_editscore7b.sh. Fetched 2026-08-11.

[14] OmniGen2-RL trainer at commit 18e6f9d5 (`train.py`: logging dict, checkpoint save/resume/retention logic, sample-visualization routine). https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/OmniGen2-RL/train.py. Fetched 2026-08-11.

[15] OmniGen2-RL checkpoint-conversion script at commit 18e6f9d5 (`scripts/misc/convert_dist_ckpt_to_hf_format.sh`). https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/OmniGen2-RL/scripts/misc/convert_dist_ckpt_to_hf_format.sh. Fetched 2026-08-11.

[16] Root repository directory listing (GitHub Contents API) at commit 18e6f9d5. https://api.github.com/repos/VectorSpaceLab/OmniGen2/contents/?ref=18e6f9d5271b517fcb32e999f10df943ae9b8f20. Fetched 2026-08-11.

[17] GitHub repository API for VectorSpaceLab/OmniGen2 (stars, license, push date) and the repository's releases API (confirms zero published releases), both live/unpinned endpoints. https://api.github.com/repos/VectorSpaceLab/OmniGen2 ; https://api.github.com/repos/VectorSpaceLab/OmniGen2/releases. Fetched 2026-08-11.
