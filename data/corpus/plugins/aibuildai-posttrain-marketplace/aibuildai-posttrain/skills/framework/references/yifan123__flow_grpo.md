# flow_grpo (Flow-GRPO)

The official research codebase for Flow-GRPO: a single accelerate-launched training loop for applying GRPO, DPO, SFT, and GRPO-Guard to flow-matching image/video generators (SD3.5, FLUX, Qwen-Image, Wan2.1, Bagel).

**flow_grpo** is described by its own GitHub metadata as "[NeurIPS 2025] An official implementation of Flow-GRPO: Training Flow Matching Models via Online RL" [1]. It is maintained by the paper's authors, led by repository owner yifan123 (Jie Liu et al.) [1][2]. The API is script-first, not a Python package with trainer classes: each method has its own `scripts/train_*.py` entry point that takes an `ml_collections` config path of the form `config/<method>.py:<config_name>` and is launched with `accelerate launch` [3]. It lives at https://github.com/yifan123/flow_grpo [1].

**When to pick it**: pick it if you specifically need to reproduce or extend Flow-GRPO, Flow-DPO, or GRPO-Guard on flow-matching image/video backbones (SD3.5-medium, FLUX.1-dev, FLUX.1-Kontext-dev, Qwen-Image, Qwen-Image-Edit, Wan2.1, Bagel-7B) with the paper's own reward-model zoo (GenEval, OCR, PickScore, DeQA, ImageReward, UnifiedReward, aesthetic, JPEG compressibility) [4]. It is not a general post-training library: there is no Hub-native trainer-class API, no PyPI package, and no single unified `Trainer` — GRPO, DPO/OnlineDPO, and SFT/OnlineSFT are separate scripts sharing a `config.train.algorithm` switch and a common `stat_tracking` module [5]. For LLM-oriented GRPO/DPO/SFT with a trainer-class API and Hub integration, weigh trl instead (cross-reference; not covered here).

**Methods it ships** [4][5][6]:
- GRPO: `config/grpo.py`, run through `scripts/train_sd3.py` (and per-backbone variants `train_flux.py`, `train_qwenimage.py`, `train_wan2_1.py`, `train_bagel.py`, `train_sd3_fast.py` for the Flow-GRPO-Fast variant).
- GRPO-Guard: `config/grpo_guard.py`, run through `scripts/train_sd3_GRPO_Guard.py`; adds RatioNorm and Gradient Reweight on top of GRPO to counter an importance-ratio bias the README shows is below 1 and grows at low-noise steps, from a separate paper (arXiv 2510.22319) [4].
- DPO / Online DPO: `config/dpo.py`, run through `scripts/train_sd3_dpo.py`; the same config sets `config.train.algorithm = 'dpo'`, and switching `config.train.ref_update_step` to a small value (e.g. 40) turns it into Online DPO [6].
- SFT / Online SFT: `config/sft.py`, structurally identical to the DPO config but with `config.train.algorithm = 'sft'` [7]. The shipped launcher `scripts/single_node/sft.sh` points at `scripts/train_sd3_rwr.py`, which does not exist in this commit (confirmed 404 fetching that raw path); the working entry point is `scripts/train_sd3_sft.py` with the same `config/sft.py:<name>` argument [8][9].
- None of these are marked experimental in the README; there is no separate stable/experimental taxonomy page — the changelog at the top of the README is the closest thing to a living index of what is newly supported [4].

**Scale it handles**: single GPU (`accelerate launch --num_processes=1 ...`) up to multi-node, all through Hugging Face Accelerate; ready-made launcher configs sit in `scripts/accelerate_configs/`: `multi_gpu.yaml` (single machine, `num_processes: 8`, mixed_precision fp16), `multi_node.yaml` (`num_machines: 2`, `num_processes: 16`, static rendezvous with a literal `main_process_ip`), plus `deepspeed_zero1.yaml`, `deepspeed_zero2.yaml`, and `fsdp.yaml` for sharding [10]. LoRA is the default (`config.use_lora = True` in `config/base.py`); full-parameter training is opt-in per config (only one shipped GRPO config, `pickscore_bagel`, sets `config.use_lora = False`), and the README states full-parameter Bagel-7B training needs at least 8×80GB GPUs, with a LoRA fallback config (`config/grpo.py:pickscore_bagel_lora`) offered for OOM [4]. Multi-node scripts (e.g. `scripts/multi_node/sd3.sh`) are shown with 4 nodes as mechanism, not benchmarked in the README; the "reward curve" figures in the README are the closest thing to a published reference run, and they compare Flow-GRPO-Fast against Flow-GRPO on PickScore/GenEval, not a scale benchmark [4].

**Install**: no PyPI package. `git clone https://github.com/yifan123/flow_grpo.git && cd flow_grpo && pip install -e .`; `setup.py` (read at commit 879042c) pins `torch==2.6.0`, `torchvision==0.21.0`, `transformers==4.40.0`, `accelerate==1.4.0`, `diffusers==0.33.1`, `deepspeed==0.16.4`, `peft==0.10.0`, `wandb==0.18.7`, `python_requires=">=3.10"` [11]. MIT licence [1]. There is no release/tag on this repository — `git tag`/`git release` are both empty via the GitHub API, so the given commit `879042cf5707f8b90daa98d147d7deac2317c5da` (2026-05-07T08:23:21Z) is simply the repository's newest push, read live, not a pinned release [1][12]. The pinned `diffusers==0.33.1` is too old for Qwen-Image; the repo owner confirmed in a closed issue that Qwen-Image requires upgrading diffusers, and the README separately instructs installing diffusers from its GitHub main branch (`pip install git+https://github.com/huggingface/diffusers.git`) before training FLUX.1-Kontext-dev, Qwen-Image, or Qwen-Image-Edit [4][13]. Only the FLUX.1-Kontext-dev block adds a warning that PEFT may also need upgrading after that diffusers upgrade; the Qwen-Image and Qwen-Image-Edit blocks give the same diffusers-install line without repeating that warning [4]. No CUDA or GPU-driver minimum is stated anywhere in `setup.py` or the README; the Bagel path additionally needs `transformers>=4.44.0` and `flash-attn==2.7.4.post1`, overriding the base `transformers==4.40.0` pin [4][11].

**Maintained by**: yifan123 (Jie Liu) and co-authors of the Flow-GRPO (arXiv 2505.05470, NeurIPS 2025) and GRPO-Guard (arXiv 2510.22319) papers [1][14][15]. 2465 GitHub stars, not a ranking signal [1]. The GitHub API records the repository as created 2025-05-08 [1]; the README's own changelog then shows entries from 2025-05-15 through 2026-05-07, most recently announcing that Flow-GRPO is now supported inside the separate `verl-omni` framework, and before that (2025-11-04) adding GRPO-Guard and Bagel-7B support [4].

## Quick start

From the README's own quickstart [4]:

```bash
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo
conda create -n flow_grpo python=3.10.16
pip install -e .
```

Pre-download the base and reward models named in the README before training — e.g. `stabilityai/stable-diffusion-3.5-medium`, `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `yuvalkirstain/PickScore_v1` — then start single-GPU GRPO training on SD3.5-medium with the OCR reward:

```bash
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 --main_process_port 29501 \
  scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu
```

This is the literal command inside the shipped `scripts/single_node/grpo.sh` [16]. The equivalent DPO and SFT single-node launchers are `bash scripts/single_node/dpo.sh` (`scripts/train_sd3_dpo.py --config config/dpo.py:geneval_sd3`) and `bash scripts/single_node/sft.sh`, though the latter's script path is broken as noted above and should be pointed at `scripts/train_sd3_sft.py` instead [8][9].

## Start it

- One GPU: `accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 ... scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu` [16].
- Several GPUs on one machine: the same launch line with a higher `--num_processes`, still against `scripts/accelerate_configs/multi_gpu.yaml` (which itself declares `num_processes: 8`, `mixed_precision: fp16`) [10][16].
- Multiple nodes: `bash scripts/multi_node/sd3.sh <rank>` on each node, which wraps `accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml --num_machines <N> --num_processes <N*gpus> --machine_rank <rank> --main_process_ip <ip> --main_process_port <port> scripts/train_sd3.py ...`; equivalent per-backbone scripts exist for FLUX, FLUX-Kontext, Qwen-Image, Qwen-Image-Edit, Bagel, and GRPO-Guard [4][10].
- Sharding for models that don't fit on one card in full-parameter mode: `scripts/accelerate_configs/deepspeed_zero1.yaml`, `deepspeed_zero2.yaml`, or `fsdp.yaml`, named at those repo paths; the base config additionally exposes `config.fsdp_optimizer_offload` and `config.activation_checkpointing` flags [10][17].
- Effective batch arithmetic is the README's own stated invariant: `config.sample.train_batch_size * num_gpu / config.sample.num_image_per_prompt * config.sample.num_batches_per_epoch = 48` (i.e. `group_number=48`, `group_size=24` in the shipped SD3.5 GRPO configs), and `config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2`; several configs `assert` this evenness at import time [4][5].
- Config surface is per-method `ml_collections.ConfigDict` (`config/base.py` plus method files that mutate a `base.get_config()` instance), not a `TrainingArguments` subclass. Defaults the library sets (not silent, but easy to miss): `config.mixed_precision = "fp16"` and `config.use_lora = True` in `config/base.py`, while `config.train.cfg = True` (classifier-free guidance during training) is on by default — several GRPO-Fast configs explicitly flip it off (`config.train.cfg = False`) for speed [17].
- OOM first aid: the README's FAQ recommends fp16 over bf16 where the base model supports it (smaller log-prob gap between sampling and training, at the cost of Flux/Wan needing bf16 because fp16 inference breaks their outputs); for Bagel-7B, the documented fallback from full-parameter (needs ≥8×80GB GPUs) to fitting on less hardware is the shipped `config/grpo.py:pickscore_bagel_lora` LoRA config [4]. There is no separate generation engine (e.g. vLLM) — sampling and training run in the same process — so generation-side OOM knobs are the same `sample.train_batch_size` / `sample.num_image_per_prompt` fields used for the batch-arithmetic invariant above [5].

## Watch it

This section is mechanics only; what a healthy reward curve, KL, or clip-fraction shape looks like for GRPO or DPO is not restated here.

- Logging is wandb-only and unconditional: `scripts/train_sd3.py` calls `wandb.init(project="flow_grpo")` on the main process with no `report_to`-style flag to disable it or change the project name from inside the config — a run with wandb unavailable or unauthenticated has no built-in fallback logging destination in this script [18].
- GRPO per-optimizer-step metrics logged by `scripts/train_sd3.py` (read directly from the `info` dict built in its training loop, commit 879042c): `approx_kl`, `clipfrac`, `clipfrac_gt_one`, `clipfrac_lt_one`, `policy_loss`, `kl_loss` (only logged when `config.train.beta > 0`), `loss`, plus `epoch` and `inner_epoch` [18].
- GRPO per-rollout-epoch metrics logged in the same file: `reward_<name>` for every key in `config.reward_fn` (excluding any `_accuracy`/`_strict_accuracy` sub-keys) is logged unconditionally; `group_size`, `trained_prompt_num`, `zero_std_ratio`, and `reward_std_mean` are logged together, but only inside a single `if config.per_prompt_stat_tracking:` block; `actual_batch_size` is logged unconditionally, later in the same epoch [18].
- Sample-level logging: every 10 epochs, the training loop in `scripts/train_sd3.py` logs up to 15 `wandb.Image` training samples under the key `"images"`, each captioned with the prompt and its single averaged reward (`avg`); a separate `eval()` function logs an `eval_images` block during evaluation whose captions break the reward out per reward-function name, alongside the `eval_reward_<name>` scalars [18].
- Evaluation during training is driven by `config.eval_freq` (epochs between evaluation passes) alongside `config.save_freq` for checkpointing, both declared in `config/base.py`; there is no separate held-out `eval_dataset` field — evaluation reads from the same `config.dataset` directory's test split at `config.sample.eval_num_steps` / `config.sample.eval_guidance_scale` [17].
- `scripts/train_sd3_dpo.py` (the DPO/Online-DPO entry point for `config/dpo.py`) and `scripts/train_sd3_sft.py` (the working SFT entry point for `config/sft.py`, see Methods it ships) each log their own `wandb.log` calls at comparable points in their loops, but neither was enumerated metric-by-metric for this card; only the GRPO script's metric list above was read line-by-line [6][9].
- No stopping-rule or reward-threshold field is published anywhere read for this card: `config/base.py` declares no early-stopping or patience field and defaults `config.num_epochs` to `100000` [17]; a comment in `config/grpo_guard.py` on top of that default confirms the intent is manual stopping, noting the large epoch count is set intentionally because "Training will be manually stopped once sufficient" [21].

## Save it

- `save_ckpt()` in `scripts/train_sd3.py` writes to `<config.save_dir>/checkpoints/checkpoint-<global_step>/lora/`, and only calls `unwrap_model(transformer, accelerator).save_pretrained(...)` on that LoRA-adapter subdirectory — with the default `config.use_lora = True`, what lands on disk is a PEFT adapter directory (e.g. `adapter_config.json` + adapter weights), NOT a full model directory [18].
- Save cadence and retention: `config.save_freq` (epochs between checkpoints, default 20 in `config/base.py`, overridden per config e.g. 30/40/60) and `config.num_checkpoint_limit` (default 5, passed to Accelerate's `ProjectConfiguration(total_limit=...)`, which prunes older checkpoints automatically) [17][18].
- There is no `resume_from_checkpoint`-style call implemented: `config.resume_from` is declared in `config/base.py` but `scripts/train_sd3.py` never reads it, and no `accelerator.load_state` / `resume` logic exists in that script at this commit — despite the field's presence, resuming a run from a saved checkpoint is not wired up in the code read for this card [17][18].
- Full-parameter runs (`config.use_lora = False`, e.g. the Bagel path) still go through the same `save_ckpt()` function, which unconditionally writes to a `lora/` subdirectory regardless of `use_lora` — the naming does not change for full-parameter saves in the code read [18].
- Reload: `config.train.lora_path`, if set, is passed to `PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)` at startup to resume training from a saved adapter (not a training-state resume — optimizer/scheduler state is not restored this way) [17][19].
- Loader handoff: an adapter directory saved here is a PEFT adapter, not a full diffusion model — the loading library's contract (pairing it with the base pipeline via `PeftModel`/`AutoPeftModel`, or merging) governs whether a downstream evaluator can load it directly; that contract is not this card's to state.

## Find it in the docs

There is no hosted documentation site for this project — the README (`README.md` at the repo root) and the config/script source are the only documentation surface [1][4].

- Address pattern: `https://github.com/yifan123/flow_grpo` for the repo home; `https://raw.githubusercontent.com/yifan123/flow_grpo/<ref>/<path>` for any file, where `<ref>` is a branch (`main`) or a full commit SHA — there are no release tags to substitute in (confirmed empty via the GitHub releases and tags APIs) [1][12].
- Question-to-file map: available training methods and their config entry points -> `config/grpo.py`, `config/dpo.py`, `config/sft.py`, `config/grpo_guard.py` (each file's `def <name>():` functions are the runnable configs, selected as `config/<method>.py:<name>` on the command line) [5]. Reward-model options and how to combine them with weights -> the "Multi Reward Training" section of the README, backed by `flow_grpo/rewards.py` [4]. Per-backbone launch commands (SD3.5, FLUX, FLUX-Kontext, Qwen-Image, Qwen-Image-Edit, Wan2.1, Bagel) -> the collapsible `<details>` blocks under "Start Training" in the README [4]. Extending to a new backbone -> the README's "How to Support Other Models" section, which names the three files to adapt (`flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`, `scripts/train_sd3.py`, `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`) and a verification recipe using `scripts/demo/sd3_sde_demo.py` [4].
- Runnable references beyond the README: `scripts/demo/` for SDE-sampling sanity checks; `dataset/` holds the repo's own smoke-test data layout (`dataset/geneval`, `dataset/pickscore`, `dataset/pickscore_sfw`, `dataset/ocr`, `dataset/drawbench`, `dataset/counting_edit`) referenced directly by the shipped configs' `config.dataset` paths [5][20].
- Community layer: the README itself is the only curated pointer — it links out to a companion `reward-server` repository (same author) for GenEval and DeQA reward servers, and to a separate `verl-omni` project for a verl-style training framework wrapping Flow-GRPO, both linked directly from the README rather than from any separate curated tutorials page [4].
- No official MCP endpoint is documented for this project.
- Traps found in closed issues (maintainer replies only): a user reported that `diffusers==0.33.1` (the version pinned in `setup.py`) does not support Qwen-Image; the repository owner (association: OWNER) replied "使用qwen image请更新diffusers" ("to use Qwen-Image, please update diffusers"), confirming the pin is stale for that backbone (issue #214, closed 2026-05-07) [13]. This matches the README's own separate instruction to install diffusers from its GitHub main branch before training FLUX.1-Kontext-dev, Qwen-Image, or Qwen-Image-Edit [4].
- Honest boundary: this is a research codebase for flow-matching image/video generators, not a language-model post-training library — it ships no LLM-specific method (PPO, KTO, ORPO, etc.) and no Hub-native model/dataset loading convenience; resume-from-checkpoint is declared in the config schema but not implemented in the GRPO training script read for this card, so long unattended multi-day runs should assume no automatic resume on crash.

## Sources

[1] flow_grpo GitHub repository metadata (description, license, stars, pushed_at, created_at). https://api.github.com/repos/yifan123/flow_grpo. Fetched 2026-08-11.

[2] flow_grpo README acknowledgement and citation blocks (author list). https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/README.md. Fetched 2026-08-11.

[3] flow_grpo config-loading convention, read from `config/grpo.py`, `config/dpo.py`, `config/sft.py` (each ends with `def get_config(name): return globals()[name]()`, invoked via `--config <file>:<name>`). Fetched 2026-08-11 at commit 879042c.

[4] flow_grpo README. https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/README.md. Fetched 2026-08-11.

[5] flow_grpo `config/grpo.py` (GRPO configs, batch/reward/method fields) at commit 879042c. https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/config/grpo.py. Fetched 2026-08-11.

[6] flow_grpo `scripts/train_sd3_dpo.py` and `config/dpo.py` (algorithm switch, `ref_update_step`) at commit 879042c. https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3_dpo.py and .../config/dpo.py. Fetched 2026-08-11.

[7] flow_grpo `config/sft.py` at commit 879042c. https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/config/sft.py. Fetched 2026-08-11.

[8] flow_grpo `scripts/single_node/sft.sh` at commit 879042c (references `scripts/train_sd3_rwr.py`). https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/single_node/sft.sh. Fetched 2026-08-11.

[9] Confirmation that `scripts/train_sd3_rwr.py` does not exist at commit 879042c (raw fetch returns "404: Not Found") and that `scripts/train_sd3_sft.py` does exist at that path. https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3_rwr.py and .../scripts/train_sd3_sft.py. Fetched 2026-08-11.

[10] flow_grpo `scripts/accelerate_configs/` directory (`multi_gpu.yaml`, `multi_node.yaml`, `deepspeed_zero1.yaml`, `deepspeed_zero2.yaml`, `fsdp.yaml`) at commit 879042c. https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/accelerate_configs/multi_gpu.yaml and .../multi_node.yaml. Fetched 2026-08-11.

[11] flow_grpo `setup.py` at commit 879042c (dependency pins, `python_requires`). https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/setup.py. Fetched 2026-08-11.

[12] flow_grpo GitHub releases and tags APIs, both returning an empty list (no releases or tags exist on this repository). https://api.github.com/repos/yifan123/flow_grpo/releases and https://api.github.com/repos/yifan123/flow_grpo/tags. Fetched 2026-08-11.

[13] flow_grpo closed issue #214 ("环境依赖问题" / "Environment dependency issue"), maintainer (yifan123, association OWNER) reply confirming diffusers must be updated for Qwen-Image, closed 2026-05-07. https://api.github.com/repos/yifan123/flow_grpo/issues/214 and .../issues/214/comments. Fetched 2026-08-11.

[14] Flow-GRPO paper abstract page. https://arxiv.org/abs/2505.05470 (cited via the README's own badge link, not separately fetched for method content — method math is not restated on this card). Referenced 2026-08-11.

[15] GRPO-Guard paper abstract page. https://arxiv.org/abs/2510.22319 (cited via the README's own badge link, not separately fetched for method content). Referenced 2026-08-11.

[16] flow_grpo `scripts/single_node/grpo.sh` and `scripts/single_node/dpo.sh` at commit 879042c. https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/single_node/grpo.sh and .../scripts/single_node/dpo.sh. Fetched 2026-08-11.

[17] flow_grpo `config/base.py` at commit 879042c (all `config.*` defaults: `mixed_precision`, `use_lora`, `save_freq`, `eval_freq`, `num_checkpoint_limit`, `resume_from`, `num_epochs`, `train.cfg`, `fsdp_optimizer_offload`, `activation_checkpointing`). https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/config/base.py. Fetched 2026-08-11.

[18] flow_grpo `scripts/train_sd3.py` at commit 879042c (`wandb.init`, `save_ckpt`, per-step and per-epoch `wandb.log` calls, absence of resume logic). https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3.py. Fetched 2026-08-11.

[19] flow_grpo `scripts/train_sd3.py` LoRA setup block (`config.train.lora_path`, `PeftModel.from_pretrained`) at commit 879042c, same file as [18]. Fetched 2026-08-11.

[20] flow_grpo `dataset/` directory listing at commit 879042c. https://api.github.com/repos/yifan123/flow_grpo/contents/dataset?ref=879042cf5707f8b90daa98d147d7deac2317c5da. Fetched 2026-08-11.

[21] flow_grpo `config/grpo_guard.py` at commit 879042c (the "training will be manually stopped once sufficient" comment above `config.num_epochs`, and the `general_ocr_sd3_rationorm` / `pickscore_hps_sd3_ratio_norm` / `geneval_sd3_rationorm` GRPO-Guard configs). https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/config/grpo_guard.py. Fetched 2026-08-11.
