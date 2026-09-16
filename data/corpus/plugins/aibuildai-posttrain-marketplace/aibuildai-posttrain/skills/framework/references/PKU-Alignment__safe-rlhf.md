# safe-rlhf

An academic PKU-Alignment research codebase for constrained-optimization RLHF (Safe RLHF / PPO-Lagrangian) plus plain PPO, DPO, reward and cost model training; single-GPU to multi-node only through DeepSpeed, and only installable from source.

**safe-rlhf** (repository title "Constrained Value-Aligned LLM via Safe RLHF" [1]) is a training pipeline that covers "a complete pipeline from Supervised Fine-Tuning (SFT) to preference model training to RLHF alignment training" [1], built and maintained by the PKU-Alignment Team [1][2]. Its API is script-first: each stage is a `safe_rlhf.<package>` module invoked as `deepspeed --module safe_rlhf.<package> <flags>` (or through a wrapping `scripts/*.sh` launcher), reading CLI flags via argparse rather than a class-based trainer object exposed to user code [1][3]. It lives at https://github.com/PKU-Alignment/safe-rlhf [2].

**When to pick it**: pick it specifically to reproduce or extend Safe RLHF's constrained PPO-Lagrangian method - training a reward-maximizing policy under a learned cost constraint via a dual (Lagrange-multiplier) update - since that mechanism, implemented in `safe_rlhf/algorithms/ppo_lag/trainer.py`, is this codebase's reason to exist [1][4]. Do not pick it for a maintained, pip-installable library with a stable API or LoRA/PEFT support: there is no functional PyPI package under this name (see Install), no GitHub releases or tags exist, only branch commits [5][6], and the repository's own Future Plans checklist still shows "Support memory-efficient training, such as LoRA, PEFT, etc." as an unchecked, not-yet-done item [1].

**Methods it ships**: four RL/preference algorithms under `safe_rlhf/algorithms/` - `dpo`, `ppo` (plain PPO, no cost), `ppo_lag` (Safe RLHF's constrained PPO-Lagrangian, the flagship method), and `ppo_reward_shaping` (an alternate variant that folds the cost into the reward instead of a dual update) [3]; plus reward-model and cost-model training under `safe_rlhf/values/reward/` and `safe_rlhf/values/cost/` [3], and SFT under `safe_rlhf/finetune/` with two interchangeable backends, `deepspeed.py` and `huggingface.py` [3]. Each is launched as its own `--module` target (e.g. `--module safe_rlhf.algorithms.ppo_lag`) [7]; there is no experimental/stable split published for these - all four algorithms and both value-model trainers ship as the same first-class CLI entry points [3][7]. Method-level math and the defining papers are out of scope for this card.

**Scale it handles**: single GPU up to multi-node, exclusively through DeepSpeed as the launcher (`deepspeed --module safe_rlhf.<package> ...`, with `--hostfile` for multi-node) [7]; ZeRO stage is a `--zero_stage` flag (the PPO-Lag reference script defaults it to 3) [7], and `--offload {none,parameter,optimizer,all}` moves optimizer and/or parameter state to CPU via DeepSpeed ZeRO-Offload when memory is tight [1][8]. The README states all four training stages were tested on 8x NVIDIA A800-80GB GPUs and points multi-node users to DeepSpeed's own multi-node docs with a 4-node/8-GPU-each hostfile example, but publishes no multi-node throughput or scaling benchmark of its own [1][9] - the mechanism is documented, the numbers are not.

**Install**: no functional `pip install safe-rlhf` exists - the PyPI package of that exact name is an unrelated placeholder ("A small example package", version 0.0.1.dev0, last and only upload 2023-05-03, predating this repository's real commit history) [10]. The only real install path is from source: `git clone https://github.com/PKU-Alignment/safe-rlhf.git && cd safe-rlhf`, then either `conda env create --file conda-recipe.yaml` (native) or `make docker-run` (containerized) [1]. `pyproject.toml` states `requires-python >= '3.8'` and dependency floors with no upper bound on any of them: `torch>=1.13`, `transformers>=4.37`, `tokenizers>=0.13.3`, plus unpinned `datasets`, `accelerate`, `deepspeed`, `numpy`, `scipy`, `sentencepiece`, `wandb`, `tensorboard`, `optree`, `matplotlib`, `tqdm`, `rich` [11]; `requirements.txt` repeats the same floors [12]. `conda-recipe.yaml`, the environment file the README's install command actually uses, instead pins `python = 3.11` exactly and `pytorch::pytorch >= 2.0`, and is the only source in this repository naming a CUDA floor: it sets `nvidia/label/cuda-11.8.0::cuda-toolkit = 11.8` and instructs `CONDA_OVERRIDE_CUDA=11.8 conda env create --file conda-recipe.yaml` on systems where the host driver is older [13] - these conda pins are narrower than the pyproject/requirements floors above, so which one actually governs your environment depends on which install path you take. Licence is Apache-2.0 [2][14]. There are no GitHub releases or tags to pin against [5][6]; the version string is a synthetic dev tag (`0.0.1dev0`, `__release__ = False`) computed at import time from `git describe`, not a real release number [15].

**Maintained by**: the PKU-Alignment Team [1]; about 1,612 GitHub stars (not a ranking signal) [2][5]. The repository is not archived [5]. The pinned commit for this card, `e8cca16665ef2340ac92c6514f05519310251581`, is the `main` branch HEAD as of this reading and its own commit date is 2024-06-13 ("docs(README.md): release PKU-SafeRLHF datasets (#178)") [16][17]; a separately-reported repository `pushed_at` timestamp of 2025-11-24 belongs to an automated Dependabot branch, not to `main`, so it does not mark a content update to the code this card describes [5][17].

## Quick start

The README's four-stage pipeline, run in order from a cloned checkout, each a complete shell command rather than pseudo-code [1]:

```bash
bash scripts/sft.sh \
    --model_name_or_path <your-model-name-or-checkpoint-path> \
    --output_dir output/sft
```

```bash
bash scripts/reward-model.sh \
    --model_name_or_path output/sft \
    --output_dir output/rm
bash scripts/cost-model.sh \
    --model_name_or_path output/sft \
    --output_dir output/cm
```

```bash
bash scripts/ppo-lag.sh \
    --actor_model_name_or_path output/sft \
    --reward_model_name_or_path output/rm \
    --cost_model_name_or_path output/cm \
    --output_dir output/ppo-lag
```

`scripts/ppo-lag.sh` itself wraps a full `deepspeed --module safe_rlhf.algorithms.ppo_lag` invocation with defaults for the actor model (`PKU-Alignment/alpaca-7b-reproduced`), training data (`PKU-SafeRLHF/train`, plus `alpaca` for the PTX auxiliary loss), and every Safe RLHF hyperparameter below [7].

## Start it

- One GPU: run any `scripts/*.sh` launcher as-is, or call `deepspeed --module safe_rlhf.<package> <flags>` directly - DeepSpeed is the only supported launcher for every stage, there is no non-DeepSpeed single-process path documented [1][7].
- Multiple GPUs / multi-node: pass `--hostfile <path>` to the wrapping script (forwarded to `deepspeed`); the README gives a 4-node x 8-GPU hostfile example and points to DeepSpeed's own multi-node resource-configuration docs for the rest [1][9]. `ppo-lag.sh` also auto-picks a free `--master_port` and, before launching, forces `WANDB_MODE=offline` whenever `WANDB_API_KEY` is unset [7].
- Effective batch size for PPO-Lag is `--per_device_train_batch_size` (16 by default) x GPU count x `--gradient_accumulation_steps` (1 by default) [7].
- Config surface is pure argparse per stage, not a shared dataclass; `ppo-lag.sh`'s own defaults are the clearest documented reference point: `--zero_stage 3 --offload none --bf16 True --tf32 True --actor_lr 1e-5 --critic_lr 5e-6 --kl_coeff 0.01 --clip_range_ratio 0.2 --clip_range_score 50.0 --clip_range_value 5.0 --ptx_coeff 16.0`, plus the Safe-RLHF-specific Lagrangian knobs `--threshold 0.0 --lambda_init 1.0 --lambda_lr 0.1 --lambda_max 5.0 --lambda_update_delay_steps 0 --episode_cost_window_size 128` [7]. `--bf16 True` as the shipped default is a silent hardware assumption: a closed issue confirms fp16 risks loss-scale overflow and bf16 requires an Ampere-or-newer GPU, recommending fp32 on pre-Ampere cards such as V100 (issue #21, replies by XuehaiPan and calico-1226, both MEMBER, 2023) [18].
- Out-of-memory first aid: raise `--offload` from `none` to `parameter`, `optimizer`, or `all` to move DeepSpeed ZeRO state to CPU [1][7][8]; the README also names lowering `--per_device_train_batch_size` and the number of GPUs used as machine-specific knobs to retune per the comment in every `scripts/*.sh` file [1].

## Watch it

Mechanics only - what a given metric means for judging whether the run is healthy belongs on the Safe RLHF / PPO method card, not here.

- **Enable it**: `--log_type` is `wandb` by default for every RL, value-model, and SFT-via-deepspeed launch script's argparse (`choices=['wandb','tensorboard']`; there is no `'none'` choice exposed at this level) - runs log to Weights & Biases unless you pass `--log_type tensorboard`, and `ppo-lag.sh` falls back to `WANDB_MODE=offline` automatically when no API key is set rather than disabling logging [7][19]. The underlying `Logger` class itself defaults its `log_type` parameter to the string `'none'`, but that default is never reached through these scripts' CLI, since each stage's argparse always supplies `wandb` or `tensorboard` explicitly [20].
- **RL trainer metrics** (`safe_rlhf/algorithms/ppo_lag/trainer.py`, read at this commit): `train/lambda` (the Lagrange multiplier itself), `train/episode_cost`, `train/cost_critic_loss`, `train/cost_advantage`, `train/cost_return`, `train/cost_value`, `train/cost_critic_lr`, plus the shared PPO fields (actor/critic losses, reward terms, KL, learning rates); eval adds `eval/reward` and `eval/cost` [4]. Plain `ppo` (no cost) logs `train/actor_loss`, `train/reward_critic_loss`, `train/reward*`, `train/kl_divergence`, `train/actor_lr`, `train/reward_critic_lr`, `train/mean_generated_length`, `train/max_generated_length`, and `eval/reward` [21]. `ppo_reward_shaping` logs a `train/shaped_reward*` family plus `train/episode_cost` and `train/cost` [22]. `dpo` logs `train/loss`, `train/reward`, `train/better_sample_reward`, `train/worse_sample_reward`, `train/reward_accuracy`, `train/reward_margin`, `train/lr` [23].
- **Value-model metrics**: the reward trainer (`safe_rlhf/values/reward/trainer.py`, the file this card's shortlist row names) logs `train/loss`, `train/accuracy`, `train/lr` and `eval/accuracy`, `eval/reward_mean`, `eval/reward_std` [24]. The cost trainer adds a sign-accuracy variant: `train/loss`, `train/accuracy`, `train/accuracy_sign`, `train/lr` and `eval/accuracy`, `eval/accuracy_sign`, `eval/cost_mean`, `eval/cost_std` [25].
- **SFT** logs one custom field, `train/epoch` (current epoch as a fraction of total), alongside whatever `self.eval()` returns, which for the base `TrainerBase.eval()` is an empty dict unless a subclass overrides it [26][20].
- **Sample-level generation logging** does happen, on the main process only, during RL evaluation: `RLTrainer.eval()` decodes each prompt and its generated continuation with the tokenizer and passes them, alongside every per-sample eval score, to `self.logger.print_table(title='Evaluating...', columns=['Prompt', 'Generated', *scores.keys()], rows=rows, max_num_rows=5)`, capped at 5 rows [33]. A health-limit/stopping-rule surface (early stopping, patience, a published KL or cost threshold beyond `--threshold`/`--lambda_max` above) was not found in `safe_rlhf/trainers/rl_trainer.py` or the four algorithm `trainer.py`/`main.py` files read for this card [33][4][21][22][23][27]; no separate tuning/FAQ doc page exists to check further since the project's docs site is dead (see Find it in the docs).
- **Evaluation during training**: `--need_eval` (off by default) turns it on; `--eval_strategy` is `'epoch'` or `'steps'` (default `'epoch'`); `--eval_interval` (default 1000000, i.e. effectively never under the `'steps'` strategy unless set); `--eval_split_ratio` (default `None`) carves an eval split out of the training data when set; `--per_device_eval_batch_size` sizes the eval batch [3][27].

## Save it

- `TrainerBase.save()` always writes the Hugging Face config (`config.json`) and tokenizer files via `save_pretrained`/`to_json_file` from the main process, then branches on weight format: if `--save_16bit` is set, `model.save_16bit_model(output_dir)`; otherwise, if the DeepSpeed ZeRO stage is >= 2, it calls `model.save_checkpoint(output_dir)` (DeepSpeed's own sharded checkpoint format) and then runs `zero_to_fp32.py` in that directory to convert it into the standard Hugging Face `WEIGHTS_NAME` file; for ZeRO stage < 2 it calls `model_to_save.save_pretrained(output_dir, is_main_process=True)` directly [28]. No retention flag (e.g. a save-count limit) or an option to drop optimizer state was found in the base save path read for this card - `save()` here is a full-checkpoint write, not a periodic-with-pruning scheme [28].
- The library ships no PEFT/LoRA adapter path (see When to pick it) [1], so every save under the mechanics above is a full model directory, never an adapter-only save.
- No explicit `resume_from_checkpoint`-style call was found in the files read (`trainers/base.py`, `trainers/rl_trainer.py`); DeepSpeed's own `load_checkpoint` machinery underlies `model.save_checkpoint`, but this card did not read a codepath that exercises it, so resume mechanics here are unverified beyond what DeepSpeed's own docs describe generically [28][33].
- Loader handoff: the ZeRO-stage->=2 path leaves a standard Hugging Face weights file after `zero_to_fp32.py` runs, and the ZeRO-stage<2 / `--save_16bit` paths already write via `save_pretrained` / `save_16bit_model`, so in all three cases the output directory is a normal Hugging Face model directory an external evaluator can load with `from_pretrained` - whether the specific loader you plan to use accepts it is that loader's contract, not this card's [28].

## Find it in the docs

There is no live documentation site to navigate: the `Documentation` URL declared in `pyproject.toml`, https://safe-rlhf.readthedocs.io/, returns HTTP 404 "Project not found" both at the root and at `/en/latest/` as of this reading, and the repository's own `docs/source/` directory contains only a spelling wordlist, no Sphinx source [29][11]. The README on the repository's GitHub page is therefore the primary documentation, and it is what this card was written from [1].

- Look things up in the source tree directly: each pipeline stage is a subpackage under `safe_rlhf/` - `algorithms/{dpo,ppo,ppo_lag,ppo_reward_shaping}/`, `values/{reward,cost}/`, `finetune/` - and each subpackage's `main.py` is its full argparse surface (every flag, default, and help string for that stage) while `trainer.py` is its training loop and logged-metrics dict [30].
- Runnable references: the `scripts/` directory holds one launcher shell script per stage (`sft.sh`, `sft-deepspeed.sh`, `sft-huggingface.sh`, `reward-model.sh`, `cost-model.sh`, `dpo.sh`, `ppo.sh`, `ppo-lag.sh`, `ppo-reward-shaping.sh`), each a complete, runnable default configuration [30]. The README also documents evaluation tooling - a chatbot arena comparison script, a BIG-bench harness (cloned separately from `google/BIG-bench`), and a GPT-4-based evaluator - as further runnable references beyond training itself [1].
- No curated community-tutorials page, official blog series, or MCP documentation endpoint was found for this project in the sources read for this card; the README's own citations point to the Safe RLHF and BeaverTails papers, not to third-party tutorials [1].
- Traps found in closed, maintainer-answered GitHub issues: DeepSpeed's FusedAdam CUDA-extension build can fail without a properly set `CUDA_HOME`, and clearing the DeepSpeed build cache before retrying resolves it (issue #38, reply by XuehaiPan, MEMBER, 2023) [31]; a CUDA device-side-assert during training traces to a tokenizer/embedding-size mismatch specifically caused by using the broken `decapoda-research/llama-7b-hf` community mirror instead of the official Meta LLaMA weights (issue #9, replies by XuehaiPan and calico-1226, both MEMBER, 2023) [32]; and the DeepSpeed loss-scale-overflow warning under fp16 is expected and safe to ignore, with bf16 (Ampere+ GPUs, e.g. A100) or fp32 (pre-Ampere, e.g. V100) recommended instead, and bf16/fp16 stated as mutually exclusive (issue #21, replies by XuehaiPan and calico-1226, both MEMBER, 2023) [18].
- Honest boundary: no LoRA/PEFT support is shipped as of this commit (an unchecked Future Plans item) [1]; there is no non-DeepSpeed launch path documented for any stage [1][7]; and there is no pip-installable release, so anyone who needs an importable package rather than a cloned script tree should look elsewhere.

## Sources

All GitHub-hosted files are read at commit `e8cca16665ef2340ac92c6514f05519310251581` (the `main` branch HEAD used as this card's pin) unless otherwise noted; all pages/API responses were fetched 2026-08-12.

[1] safe-rlhf README, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/README.md

[2] safe-rlhf GitHub repository (item home). https://github.com/PKU-Alignment/safe-rlhf

[3] Recursive git tree at commit e8cca16665ef2340ac92c6514f05519310251581 (package/module layout). https://api.github.com/repos/PKU-Alignment/safe-rlhf/git/trees/e8cca16665ef2340ac92c6514f05519310251581?recursive=1

[4] `safe_rlhf/algorithms/ppo_lag/trainer.py`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/algorithms/ppo_lag/trainer.py

[5] GitHub repository API metadata (license, archived flag, stars, pushed_at, created_at, default_branch). https://api.github.com/repos/PKU-Alignment/safe-rlhf

[6] GitHub Releases and Tags API responses, both empty. https://api.github.com/repos/PKU-Alignment/safe-rlhf/releases and https://api.github.com/repos/PKU-Alignment/safe-rlhf/tags

[7] `scripts/ppo-lag.sh`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/scripts/ppo-lag.sh

[8] `safe_rlhf/configs/deepspeed_config.py`, at commit e8cca16665ef2340ac92c6514f05519310251581 (`get_deepspeed_train_config`: `--offload` values `parameter`/`all` set `train_config['zero_optimization']['offload_param']['device'] = 'cpu'`, and `optimizer`/`all` set `offload_optimizer`'s device to `'cpu'`). https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/configs/deepspeed_config.py

[9] README section on multi-node training via DeepSpeed hostfile (same document as [1], cited separately for this specific claim).

[10] PyPI JSON API for the package "safe-rlhf" (unrelated placeholder package, not this library). https://pypi.org/pypi/safe-rlhf/json

[11] `pyproject.toml`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/pyproject.toml

[12] `requirements.txt`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/requirements.txt

[13] `conda-recipe.yaml`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/conda-recipe.yaml

[14] GitHub repository API `license` field (same response as [5], cited separately for this specific claim).

[15] `safe_rlhf/version.py`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/version.py

[16] GitHub Commits API for commit e8cca16665ef2340ac92c6514f05519310251581 (commit date, message). https://api.github.com/repos/PKU-Alignment/safe-rlhf/commits/e8cca16665ef2340ac92c6514f05519310251581

[17] GitHub Branches API (confirms `main`'s HEAD equals the pinned commit; `pushed_at` on [5] belongs to a Dependabot branch, not `main`). https://api.github.com/repos/PKU-Alignment/safe-rlhf/branches

[18] Closed GitHub issue #21, replies by XuehaiPan and calico-1226 (both MEMBER), 2023 (fp16 overflow warning, bf16/fp32 hardware guidance). https://api.github.com/repos/PKU-Alignment/safe-rlhf/issues/21/comments

[19] `safe_rlhf/algorithms/ppo_lag/main.py` and `safe_rlhf/values/reward/main.py`, at commit e8cca16665ef2340ac92c6514f05519310251581 (`--log_type` argparse default and choices). https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/algorithms/ppo_lag/main.py and .../safe_rlhf/values/reward/main.py

[20] `safe_rlhf/logger.py` and `safe_rlhf/trainers/base.py`, at commit e8cca16665ef2340ac92c6514f05519310251581 (`Logger` class `'none'` default, `TrainerBase.eval()` default). https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/logger.py and .../safe_rlhf/trainers/base.py

[21] `safe_rlhf/algorithms/ppo/trainer.py`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/algorithms/ppo/trainer.py

[22] `safe_rlhf/algorithms/ppo_reward_shaping/trainer.py`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/algorithms/ppo_reward_shaping/trainer.py

[23] `safe_rlhf/algorithms/dpo/trainer.py`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/algorithms/dpo/trainer.py

[24] `safe_rlhf/values/reward/trainer.py`, at commit e8cca16665ef2340ac92c6514f05519310251581 (the file this card's shortlist row names). https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/values/reward/trainer.py

[25] `safe_rlhf/values/cost/trainer.py`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/values/cost/trainer.py

[26] `safe_rlhf/trainers/supervised_trainer.py`, at commit e8cca16665ef2340ac92c6514f05519310251581. https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/trainers/supervised_trainer.py

[27] `safe_rlhf/algorithms/ppo_lag/main.py`, at commit e8cca16665ef2340ac92c6514f05519310251581 (evaluation-strategy argparse group: `--need_eval`, `--eval_strategy`, `--eval_interval`, `--eval_split_ratio`, `--per_device_eval_batch_size`). https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/algorithms/ppo_lag/main.py

[28] `safe_rlhf/trainers/base.py`, at commit e8cca16665ef2340ac92c6514f05519310251581 (`TrainerBase.save()` full branching). https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/trainers/base.py

[29] safe-rlhf Read the Docs site, both checked live: root and `/en/latest/`, both HTTP 404 "Project not found". https://safe-rlhf.readthedocs.io/ and https://safe-rlhf.readthedocs.io/en/latest/

[30] Recursive git tree at commit e8cca16665ef2340ac92c6514f05519310251581 (same response as [3], cited separately for the `scripts/` directory listing).

[31] Closed GitHub issue #38, reply by XuehaiPan (MEMBER), 2023 (DeepSpeed FusedAdam CUDA build failure and fix). https://api.github.com/repos/PKU-Alignment/safe-rlhf/issues/38/comments

[32] Closed GitHub issue #9, replies by XuehaiPan and calico-1226 (both MEMBER), 2023 (CUDA device-side assert traced to a broken LLaMA mirror). https://api.github.com/repos/PKU-Alignment/safe-rlhf/issues/9/comments

[33] `safe_rlhf/trainers/rl_trainer.py`, at commit e8cca16665ef2340ac92c6514f05519310251581 (eval-time sample-level generation logging via `print_table`; checkpoint-save call; no resume or stopping-rule surface found). https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/e8cca16665ef2340ac92c6514f05519310251581/safe_rlhf/trainers/rl_trainer.py
