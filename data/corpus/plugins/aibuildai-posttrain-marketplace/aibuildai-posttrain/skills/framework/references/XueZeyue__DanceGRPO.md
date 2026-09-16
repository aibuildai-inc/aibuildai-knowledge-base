# DanceGRPO

The official research-code release for one paper: GRPO adapted to diffusion and rectified-flow visual generation, shipped as seven per-backbone training scripts (eight counting a separate FLUX LoRA variant), not a general multi-method framework.

DanceGRPO is described by its own paper as "a framework that addresses" the instability of prior RL fine-tuning methods for visual generation "through an innovative adaptation of Group Relative Policy Optimization (GRPO) for visual generation tasks," covering diffusion models and rectified flows across "three key tasks and four foundation models" [1]. The repository's README calls it "the official implementation for [the paper]," built by extending FastVideo, an existing video-diffusion training codebase whose code it reuses directly - the checkpoint-saving module itself carries a FastVideo copyright header [2][3]. It is maintained by the paper's authors (Zeyue Xue and ten coauthors, ByteDance-affiliated work) [1][2]. The API is not a trainer class: each of the seven supported base models (Stable Diffusion, FLUX, HunyuanVideo, SkyReels-I2V, Qwen-Image, Qwen-Image-Edit, Wan2.1) has its own standalone training script under `fastvideo/`, launched directly with `torchrun` [2]. It lives at https://github.com/XueZeyue/DanceGRPO [2].

**When to pick it**: to reproduce or extend the DanceGRPO paper's own reported results, or as a working reference for wiring GRPO into a diffusion or rectified-flow image/video generator - not for general-purpose post-training, since it ships exactly one method family (GRPO for visual generation) with no LLM trainers, no DPO/SFT/PPO, and no comparative benchmark against other RL post-training libraries in its own docs. Choose the FLUX, Stable Diffusion, HunyuanVideo, or SkyReels-I2V scripts if you want the codebase's most exercised path: the README plots reward-curve images for exactly these four backbones, while Qwen-Image and Qwen-Image-Edit are documented only with prose reward deltas and no plotted curve [2].

**Methods it ships**: GRPO adapted for visual generation is the only training method; the paper positions it against DDPO and DPOK as the RL baselines it improves on, and reports outperforming them "by up to 181%" across the HPS-v2.1, CLIP Score, VideoAlign, and GenEval benchmarks, which is the paper's own headline result and the number that shows the method works at all [1]. What varies across the eight scripts is the base model and reward model, not the RL algorithm: `train_grpo_sd.py`, `train_grpo_flux.py` (plus a LoRA variant `train_grpo_flux_lora.py`), `train_grpo_hunyuan.py`, `train_grpo_skyreels_i2v.py`, `train_grpo_qwenimage.py`, `train_grpo_qwenimage_edit.py`, `train_grpo_wan_2_1.py` [4]. Supported reward models, read from the FLUX script's own CLI flags and the README, are HPS-v2.1 (`--use_hpsv2`) and PickScore (`--use_pickscore`) [5]; the HunyuanVideo launch script additionally exposes VideoAlign for video motion/quality (`--use_videoalign`, with separate `vq_coef`/`mq_coef` weights) [10], and the Stable Diffusion script defaults to a `jpeg_compressibility` reward function inherited from its DDPO-style config [6]. No method is marked experimental or stable in the docs - there is no such taxonomy here, only "Key Features" listing the seven supported base models [2].

**Scale it handles**: single GPU through multi-node, launched with `torchrun` in every script - no Accelerate or Ray launcher layer for the GRPO trainers (the Stable Diffusion script separately wraps Hugging Face's `accelerate` library internally for its own DDPO-style training loop, but is still started via plain `torchrun`) [8][9]. Sharding options are FSDP (`--fsdp_sharding_startegy full`, the FLUX script's own flag name, misspelled in the source) and sequence parallelism (`--sp_size`) [8]. A published 2-node/16-GPU FLUX example and a 4-node/32-GPU HunyuanVideo example exist [8][10], but the 2-node script's own comment hedges multi-node reliability: "we don't use the original pytorch torchrun in our internal environment, so I just follow the official example of pytorch. Please adapt the torchrun scripts into your own environment" - so multi-node here is a documented pattern, not a benchmarked guarantee [8].

**Install**: no PyPI package and no GitHub Releases or tags exist for this repository (checked 2026-08-12) - the only install path is cloning the repo and running its setup script at a chosen commit; this card pins commit `15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344` (2025-10-16) [11][12]. `pyproject.toml` at that commit declares the installable package name as `fastvideo`, version `1.2.0`, Python `>=3.8`, Apache-2.0 [3]. The documented install sequence is `./env_setup.sh fastvideo`, which pins `torch==2.5.0` (via the `cu121` PyTorch wheel index) and `flash-attn==2.7.0.post2` before `pip install -e .` [13]. `pyproject.toml`'s own dependency list separately pins `transformers==4.46.1`, `diffusers==0.32.0`, `accelerate==1.0.1`, `peft==0.13.2`, `wandb==0.18.5`, and `huggingface_hub==0.26.1`, among ~50 other entries in its `dependencies` array; nearly all use exact `==` pins, though three (`watch`, `gpustat`, `loguru`) carry no version specifier at all [3]. This is inconsistent even within the repo, and the inconsistency sits in the card's own quoted install path: `env_setup.sh`'s last line, run immediately after `pip install -e .`, reinstalls `pydantic==1.10.9`, `huggingface_hub==0.24.0`, and `protobuf==3.20.0`, silently overriding the higher `pydantic==2.9.2`, `huggingface_hub==0.26.1`, and `protobuf==5.28.3` versions `pyproject.toml` itself declares - so a reader who runs the documented install command ends up with the older three versions, not the ones in `pyproject.toml` [13]. The HunyuanVideo launch script separately reinstalls `huggingface_hub==0.24.0` and `transformers==4.46.1` again at launch time [10]. No CUDA or GPU minimum is stated anywhere; the `cu121` wheel index in `env_setup.sh` is the only hardware signal [13].

**Maintained by**: the paper's authors, contact given as `xuezeyue@connect.hku.hk` in the README [2]; the repository is not a fork [14] and its most recent push, an update to `README.md`, is 2025-10-16 [15]. The README's own Updates log shows activity from the 2025-05-12 arXiv release through additions of Qwen-Image-Edit and Wan2.1 support on 2025-10-16 [2].

## Quick start

There is no pip-installable quickstart; the smallest complete run is clone-then-launch, quoted from the README and the repo's own launch script [2][8]:

```bash
git clone https://github.com/XueZeyue/DanceGRPO
cd DanceGRPO
./env_setup.sh fastvideo
```

Then, for a single-node FLUX GRPO run (adapted from the repo's 8-GPU FLUX script, which is a `torchrun` invocation of `fastvideo/train_grpo_flux.py` with a pretrained FLUX.1-dev checkpoint and a precomputed embeddings JSON as inputs) [16]:

```bash
torchrun --nproc_per_node=8 fastvideo/train_grpo_flux.py \
    --pretrained_model_name_or_path data/flux --vae_model_path data/flux \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --train_batch_size 2 --gradient_accumulation_steps 12 \
    --learning_rate 1e-5 --mixed_precision bf16 --use_hpsv2 --num_generations 12 \
    --output_dir data/outputs/grpo
```

There is no `pip install`-then-`import`-then-`.train()` path; every run is a shell-invoked training script, and dataset preparation (turning raw images/videos + captions into the `videos2caption.json` embeddings file the scripts expect) is a separate preprocessing step documented on its own page [17].

## Start it

- One GPU: drop `--nproc_per_node` to 1 or run the training script directly; no script in the repo demonstrates this, but every trainer takes the same CLI whether launched on one device or many [8].
- Several GPUs, one node: `torchrun --nproc_per_node=N fastvideo/train_grpo_<model>.py <args>`, as in `scripts/finetune/finetune_flux_grpo_8gpus.sh` [16] and the Wan2.1, Stable Diffusion, and HunyuanVideo single-node equivalents under `scripts/finetune/` [4].
- Multi-node: `torchrun --nnodes=N --nproc_per_node=8 --node_rank=<r> --master_addr=<ip> --master_port=<port> fastvideo/train_grpo_<model>.py <args>` on every node, shown for FLUX (2 nodes) [8] and HunyuanVideo (4 nodes, adding `--use_videoalign --bestofn 8 --vq_coef 1.0 --mq_coef 0.0`) [10]; the FLUX script's own comment warns this pattern was not exercised in the authors' internal cluster environment and "please adapt the torchrun scripts into your own environment" [8].
- LoRA is a separate script and config for FLUX only: `train_grpo_flux_lora.py`, launched via `finetune_flux_grpo_8gpus_lora.sh`, adding `--lora_alpha 256 --lora_rank 128` and using a learning rate of `3e-4` versus `1e-5` for full fine-tuning [18].
- Effective batch size follows the FLUX script's own formula: `total_batch_size = world_size * gradient_accumulation_steps / sp_size * train_sp_batch_size`; the Stable Diffusion config's comments work an example concretely - on an 8-GPU DGX node with `gradient_accumulation_steps=8`, `sample.batch_size=4`, `num_generations=16`: "8 * 16 * 4 = 512 samples per epoch" and "(16 * 4) / (4 * 8) = 2 gradient updates per epoch" [6][8].
- Config surface differs by trainer: the seven non-SD scripts take flat `argparse` CLI flags (as in the Quick start example above) [16]; only Stable Diffusion uses an `ml_collections` config object (`fastvideo/config_sd/base.py` for defaults, `fastvideo/config_sd/dgx.py` for the `compressibility`/`hpsv2`/`hpsv3` variants actually launched, e.g. `--config fastvideo/config_sd/dgx.py:hpsv2`) [6][19]. Mixed precision is `bf16` by default across the published launch scripts (a bf16-capable GPU is the implicit hardware assumption); the FLUX script's own `--mixed_precision` flag defaults to `None` if left unset [6][8].
- Out-of-memory first aid, drawn from the README's own troubleshooting notes and the reward-collapse fix: reduce sampling steps, resolution, or the timestep-fraction used for training, or adopt a mixed ODE/SDE sampler as in MixGRPO to cut memory and compute [2]. If training diverges into a reward collapse "similar to" the one discussed in the project's issue tracker, the README's fix is to reduce `--max_grad_norm` [2].

## Watch it

This section is mechanics only; what a healthy or unhealthy reward or loss curve looks like is method-level and not restated here.

- Logging differs by trainer and is not uniform across the repository. The FLUX script calls `wandb.init(project="flux", config=args)` directly and logs only `train_loss`, `learning_rate`, `step_time`, `avg_step_time`, and `grad_norm` to Weights & Biases at each optimizer step - reward is NOT sent to wandb; it is only printed to stdout and appended as a bare mean value to a local `./reward.txt` file, one line per step [20]. The Stable Diffusion script logs a materially different, wider set through `accelerate`'s `log_with="wandb"` tracker (`accelerator.init_trackers(project_name="grpo-sd", ...)`), including `reward`, `reward_mean`, `reward_std`, `epoch`, and `inner_epoch`, plus whatever scalar keys the per-step `info` dict accumulates [19]. A reader who only watches wandb for the FLUX trainer will see no reward curve at all.
- Six of the finetune shell scripts under `scripts/finetune/` were read for this card (FLUX 2-node, FLUX 8-GPU, FLUX 8-GPU LoRA, HunyuanVideo, Stable Diffusion, Wan2.1); all six set both `WANDB_DISABLED=true` and `WANDB_MODE=online` as environment variables in the same file - a contradictory pair as shipped; which one wins depends on wandb's own precedence rules, not anything DanceGRPO controls, so a reader should not assume logging is on (or off) from the scripts alone and should check their run's own wandb status. The Qwen-Image, Qwen-Image-Edit, and SkyReels-I2V scripts were not read and are not covered by this observation [8].
- HunyuanVideo's launch script additionally sets `--log_validation --tracker_project_name grpo` and `--validation_steps 100000000`, which is large enough that validation effectively never triggers during a normal run [10].
- No sample-level (image/video) logging call is present in the read portions of the FLUX or Stable Diffusion training loops beyond the scalar reward keys above; the FLUX script prints `reward`, `ratio`, `advantage`, and `final loss` to stdout for debugging on rank-0-mod-8 processes, which is not wandb logging, while the Stable Diffusion script separately prints `final advantage`, `hps_advantage`, and `final loss` on rank-0 - neither is a wandb call [20][19].
- No published stopping-rule threshold was found: the README's guidance on divergence is qualitative ("if you experience a reward collapse... reduce the max_grad_norm") with no numeric threshold or patience value given [2].

## Save it

- FLUX (and the other argparse-based trainers using the shared `fastvideo/utils/checkpoint.py` module) write a plain checkpoint every `--checkpointing_steps` steps (default 500) to `<output_dir>/checkpoint-<step>-<epoch>/`, containing only `diffusion_pytorch_model.safetensors` and `config.json` - no optimizer, scheduler, or RNG state is written by this path [7]. This directory layout is independently confirmed on a real published Hub artifact, `xzyhku/flux_hpsv2.1_dancegrpo`, whose file list is `checkpoint-300-0/{config.json,diffusion_pytorch_model.safetensors}` and, with `--use_ema` on, a parallel `checkpoint-ema-300-0/` pair from a separate `save_ema_checkpoint` function defined in the FLUX training script itself (not the shared `checkpoint.py` module) [21][20].
- `--resume_from_checkpoint` is documented in the FLUX script's own `argparse` help text as accepting a checkpoint path or `"latest"`, but the code path behind it is dead: `if args.resume_from_checkpoint: assert NotImplementedError(...)`. Because an exception instance is truthy in Python, this `assert` never fails - it is a silent no-op, not a raised error - so setting the flag neither crashes the run nor performs any resume; training simply starts from scratch as if the flag were never set, despite the flag existing and being documented [22].
- Stable Diffusion's `ml_collections` config carries a separate, apparently functional-looking `config.resume_from` field that calls `accelerator.load_state(...)` at startup, and the config's own comment describes it as resuming "from a checkpoint... either an exact checkpoint directory... or a directory containing checkpoints, in which case the latest one will be used" [23]. But the script's own training loop never calls the matching `accelerator.save_state()` - it instead hand-writes only `diffusion_pytorch_model.safetensors` and `config.json` to a hardcoded `./my_checkpoints/checkpoint_epoch_<epoch>/` path every `config.save_freq` epochs, ignoring the accelerate `ProjectConfiguration`'s own checkpoint directory and never producing the optimizer/RNG state `load_state` expects to find - so `config.resume_from` as shipped has no matching save to resume from [19]. Across both trainers, resuming a killed run from a checkpoint is not a working, tested path in this codebase as read.
- LoRA has its own save/resume pair, separate from the full-model path: `save_lora_checkpoint` writes to `lora-checkpoint-<step>-<epoch>/`, with the adapter weights via `pipeline.save_lora_weights()`, a `lora_optimizer.pt`, and a `lora_config.json` recording `{"step": ..., "lora_params": {"lora_rank": ..., "lora_alpha": ..., "target_modules": ...}}`; `resume_lora_optimizer` reads that same pair back [7]. A `lora-checkpoint-*/` directory is an adapter, not a full model - it must be paired with the base FLUX checkpoint to produce a usable model, matching the general PEFT adapter contract.
- No `push_to_hub()` or equivalent call exists in the read training scripts; the one published Hub checkpoint, `xzyhku/flux_hpsv2.1_dancegrpo`, appears to have been uploaded manually from a local `checkpoint-300-0/` directory [21]. Whether an evaluator can load a saved `checkpoint-<step>-<epoch>/` directory directly depends on the loader matching this repo's `diffusion_pytorch_model.safetensors` + `config.json` layout - it is a raw diffusers-style UNet/transformer weights directory, not a full pipeline snapshot, so the loader must reassemble the rest of the pipeline (text encoder, VAE, scheduler) itself.

## Find it in the docs

There is no separate hosted docs site; the README (`https://github.com/XueZeyue/DanceGRPO/blob/main/README.md`) is the top-level entry point, and `docs/data_preprocess.md` is the one other prose doc page in the repo, covering dataset preprocessing only [2][17].

- Project page (paper-level materials, not code docs): `https://dancegrpo.github.io/` - linked from the README's badges [2].
- Paper: `https://arxiv.org/abs/2505.07818` [1].
- Per-model launch scripts are the practical reference beyond the README: `scripts/finetune/finetune_<model>_grpo*.sh` (eight scripts) and `scripts/preprocess/` for the matching data-preparation commands - read the shell script for the model you want directly rather than searching for a docs page, since none exists beyond the README [4].
- `fastvideo/README.md` carries FLUX-specific operational notes not in the top-level README: rollout batch size is fixed at 1 due to unexplained probability-output inconsistencies at larger batch sizes; a strong SFT warm-start is reported to suppress GRPO exploration, so fewer SFT iterations are recommended for custom models; and EMA is recommended to improve visual quality since extended training alone does not, attributed to limits of the HPS-v2.1 reward model [24].
- No official curated tutorial page, blog list, or MCP endpoint exists for this repository; the closest thing to a community layer is the four curated-list mentions that surfaced this repo in sourcing (not independently verified here) and the base framework it credits, FastVideo, at `https://github.com/hao-ai-lab/FastVideo`, referenced only because DanceGRPO's checkpoint-saving code is copied from it [2][7].
- Honest boundary: this repo has no SFT, DPO, or PPO trainer, no LLM support, and no packaged install (no PyPI, no release tags) [3][11][12]; two of its own checkpoint/resume paths (FLUX's `--resume_from_checkpoint`, Stable Diffusion's `config.resume_from`) are non-functional as shipped at this commit, so treat mid-run resume as unsupported rather than debug it as a configuration error.

## Sources

Everything below is read at commit `15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344` unless marked as a live/unpinned fetch; all fetches dated 2026-08-12.

[1] arXiv abstract page, "DanceGRPO: Unleashing GRPO on Visual Generation," 2505.07818. https://arxiv.org/abs/2505.07818. Live/unpinned, fetched 2026-08-12.

[2] Repository README.md. https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/README.md.

[3] pyproject.toml. https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/pyproject.toml.

[4] Repository file tree (GitHub API recursive tree at the pinned commit), confirming the eight `fastvideo/train_grpo_*.py` scripts and the `scripts/finetune/` and `scripts/preprocess/` directories. https://api.github.com/repos/XueZeyue/DanceGRPO/git/trees/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344?recursive=1.

[5] `fastvideo/train_grpo_flux.py`, CLI flag definitions (`--use_hpsv2`, `--use_pickscore`). https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/train_grpo_flux.py.

[6] `fastvideo/config_sd/dgx.py` and `fastvideo/config_sd/base.py` (default `reward_fn = "jpeg_compressibility"`, `hpsv2()` config variant, batch-size arithmetic comments). https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/config_sd/dgx.py and .../fastvideo/config_sd/base.py.

[7] `fastvideo/utils/checkpoint.py` (checkpoint directory layout for full-model, EMA, and LoRA saves; FastVideo copyright header). https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/utils/checkpoint.py.

[8] `scripts/finetune/finetune_flux_grpo.sh` (2-node launch form, torchrun hedge comment, WANDB env var pair). https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/scripts/finetune/finetune_flux_grpo.sh.

[9] `fastvideo/train_grpo_sd.py`, imports (`from accelerate import Accelerator`) confirming accelerate is used internally by the SD script despite torchrun launch. https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/train_grpo_sd.py.

[10] `scripts/finetune/finetune_hunyuan_grpo.sh` (4-node launch form, `--use_videoalign`, `--validation_steps`, launch-time pip reinstalls). https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/scripts/finetune/finetune_hunyuan_grpo.sh.

[11] GitHub Releases API for the repository, returning an empty list. https://api.github.com/repos/XueZeyue/DanceGRPO/releases. Live/unpinned, fetched 2026-08-12.

[12] GitHub Tags API for the repository, returning an empty list. https://api.github.com/repos/XueZeyue/DanceGRPO/tags. Live/unpinned, fetched 2026-08-12.

[13] `env_setup.sh`. https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/env_setup.sh.

[14] GitHub repository API (`fork: false`). https://api.github.com/repos/XueZeyue/DanceGRPO. Live/unpinned, fetched 2026-08-12.

[15] GitHub commit API for the pinned SHA, confirming commit date and message. https://api.github.com/repos/XueZeyue/DanceGRPO/commits/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344. Live/unpinned, fetched 2026-08-12.

[16] `scripts/finetune/finetune_flux_grpo_8gpus.sh` (single-node 8-GPU launch form). https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/scripts/finetune/finetune_flux_grpo_8gpus.sh.

[17] `docs/data_preprocess.md`. https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/docs/data_preprocess.md.

[18] `scripts/finetune/finetune_flux_grpo_8gpus_lora.sh`. https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/scripts/finetune/finetune_flux_grpo_8gpus_lora.sh.

[19] `fastvideo/train_grpo_sd.py`, full training loop (accelerate tracker init, `accelerator.log` calls, hand-written checkpoint save block, absence of a matching `accelerator.save_state()` call). https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/train_grpo_sd.py.

[20] `fastvideo/train_grpo_flux.py`, wandb init and logging calls, and the `reward.txt` file-append block. https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/train_grpo_flux.py.

[21] Hugging Face Hub model metadata API for `xzyhku/flux_hpsv2.1_dancegrpo`, listing its file siblings (`checkpoint-300-0/`, `checkpoint-ema-300-0/`, each with `config.json` and `diffusion_pytorch_model.safetensors`). https://huggingface.co/api/models/xzyhku/flux_hpsv2.1_dancegrpo. Live/unpinned, fetched 2026-08-12. This model is the checkpoint the DanceGRPO README itself links as its trained FLUX release, per the README (not per this API page) [2].

[22] `fastvideo/train_grpo_flux.py`, `--resume_from_checkpoint` argparse help text and the `assert NotImplementedError(...)` code path. https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/train_grpo_flux.py.

[23] `fastvideo/config_sd/base.py`, `config.resume_from` field and its comment. https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/config_sd/base.py.

[24] `fastvideo/README.md` ("More Discussion on FLUX" notes). https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/README.md.

Not independently verified in this card: the Hugging Face account identifier `xuezeyue/dancegrpo` named in the sourcing shortlist does not resolve on the Hub (checked 2026-08-12); the working, matching Hub artifact is `xzyhku/flux_hpsv2.1_dancegrpo`, cited above, which the README itself links as its trained FLUX checkpoint [2][21]. GitHub issue #55, referenced by the README's reward-collapse note, could not be fetched (GitHub API rate limit) and is cited here only through the README's own paraphrase, not as an independently read maintainer reply.
