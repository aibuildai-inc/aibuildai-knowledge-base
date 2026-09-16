# rl-swarm

Gensyn's decentralized RL application: clone it and launch a Docker (or shell-script) node that joins a public swarm to train a fixed CodeZero code-generation task with GRPO, not a library you import into your own training loop.

RL Swarm describes itself as "a peer-to-peer system for reinforcement learning" that lets you "train models collaboratively with others in the swarm, leveraging their collective intelligence," runnable on a consumer laptop or a cloud GPU, and connectable to the Gensyn Testnet for an on-chain identity that tracks progress [1]. It is built and maintained by Gensyn (GitHub org `gensyn-ai`) [2]. There is no importable Python API: a node is started by cloning the repo and running either `docker-compose run ... swarm-cpu|swarm-gpu` or `./run_rl_swarm.sh`, both of which shell out to a Hydra-configured launcher (`python -m code_gen_exp.runner.swarm_launcher`) that builds a `SwarmGameManager` and starts training [1][3]. The repo lives at https://github.com/gensyn-ai/rl-swarm [2].

**When to pick it**: pick this only to run a node in Gensyn's own decentralized CodeZero swarm - training a fixed, maintainer-defined code-generation task under GRPO (Solver role) coordinated over a peer-to-peer network and an on-chain smart contract - not as a general post-training library for your own model/dataset/method, which is the trl or verl use case instead (cross-reference; not covered here). The fetched README is internally inconsistent on whether an official swarm is currently live: a note near the top states "there are no official swarms running right now" and points interested users to a community-owned swarm or a separate prediction-market product instead, while a few lines later the same document states "Currently, we are running CodeZero on the Gensyn Testnet" - this card cannot resolve which statement is current and reports both [1]. Training itself runs on a fixed model pool the coordinator assigns (large-GPU users get a larger model, everyone else a smaller one) rather than a model you choose freely [1][4].

**Methods it ships**: two methods run inside one CodeZero round, one per role, neither exposed as a general-purpose trainer class. The **Solver** role subclasses GRPO: `code_gen_exp/src/trainer.py`'s `GRPOTrainerModule` inherits directly from the separate `gensyn-genrl` PyPI package's `genrl.trainer.grpo_trainer.GRPOLanguageTrainerModule` and adds only a judge-based `evaluate()` hook - the loss computation, clipping, and optimizer step live in `genrl`, not in this repo [5][6]. The **Proposer** role runs its own from-scratch PPO implementation in `code_gen_exp/src/proposer.py`: a `PPOConfig` dataclass (`clip_range: float = 0.2`, `ppo_epochs: int = 2`, `learning_rate: float = 1e-5`) drives a clipped-ratio update loop in `Proposer.train()`, and a `VllmConfig` dataclass controls an optional vLLM generation engine for that role [7]. A third role, **Evaluator**, runs a frozen (non-trained) model [4]. This split is fixed by the shipped `code_gen_exp/config/code-gen-swarm.yaml`, not a menu the operator picks from [8].

**Scale it handles**: one process per peer machine - there is no multi-GPU or multi-node launcher in this repo; scale is horizontal across independent peers joining the same swarm over a Hivemind DHT and an on-chain `SwarmCoordinator`, not data/tensor parallelism within one training job [1][9]. The README states hardware requirements vary with "model size and the accelerator platform," and documents two supported paths: an arm64/x86 CPU node with a minimum of 32GB RAM, or a CUDA GPU from an explicit list (RTX 3090, RTX 4090, RTX 5090, A100, H100) [4]. No throughput or round-time benchmark is published for either path.

**Install**: not pip-installable - there is no root-level `requirements.txt` or `pyproject.toml` [10]; the supported install is `git clone https://github.com/gensyn-ai/rl-swarm` followed by `docker-compose run --rm --build -Pit swarm-cpu` (or `swarm-gpu`), or, for the shell-script path, `python3 -m venv .venv && source .venv/bin/activate && ./run_rl_swarm.sh`, which itself runs `pip install -r code_gen_exp/requirements.txt` [1][3]. The pinned commit `9c95410b1` (2026-01-05, HEAD of `main`) is ahead of the latest tagged release, `v0.7.0` "CodeZero" (published 2025-11-12) [11][12]; the files this card cites for dependency pins and config (`code_gen_exp/requirements.txt`, `docker-compose.yaml`) are byte-identical between the pinned commit and the `v0.7.0` tag, confirmed by diff, so those specific pins hold at the release too [13]. `code_gen_exp/requirements.txt` pins `gensyn-genrl==0.3.0`, `ollama`, and `hivemind` from a pinned Gensyn fork commit (`git+https://github.com/gensyn-ai/hivemind@639c964a`) [13]; `gensyn-genrl` 0.3.0 on PyPI requires Python `>=3.10` and states no license in its metadata [14]. The README states a Python floor of `>=3.10,<=3.13` [4]. Licence: MIT (repo `LICENSE.TXT`, copyright Gensyn) [15]. Hardware/CUDA floor: the GPU Docker path's own `docker-compose.yaml` comment requires "NVIDIA Drivers version >=525.60.13" plus the nvidia-container-toolkit, and builds on base image `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`; the CPU path needs no CUDA but requires >=32GB RAM [16][4]. A separate `ollama/ollama:0.11.10` container is a hard runtime dependency for the reward/judge model, not an optional extra [16].

**Maintained by**: Gensyn (GitHub org `gensyn-ai`); repo carries 1,680 stars as of the fetched snapshot (a point-in-time count, not a trend) [2]. Actively maintained: the latest tagged release, v0.7.0 "CodeZero" (2025-11-12), was authored by a Gensyn engineer (`jcd496`) and the HEAD commit used for this card postdates it (2026-01-05, author `sirbonneville`) [12][17].

## Quick start

There is no code-level quick start - the smallest complete run is the CLI launch sequence from the README [1][3]:

```sh
git clone https://github.com/gensyn-ai/rl-swarm
cd rl-swarm
docker-compose run --rm --build -Pit swarm-cpu
```

or, on a supported GPU:

```sh
docker-compose run --rm --build -Pit swarm-gpu
```

or the shell-script (non-Docker) path:

```sh
python3 -m venv .venv
source .venv/bin/activate
./run_rl_swarm.sh
```

Each path opens a browser window at `http://localhost:3000` for an Alchemy modal login; after login, an optional Hugging Face token prompt, and an optional AI Prediction Market opt-in, the README's own "Initial peering and training" section states that "from this stage onward your device will begin training" [1]. There is no separate "train.py"-style entry point to call directly with your own dataset; the launcher always runs `python -m code_gen_exp.runner.swarm_launcher --config-path "$ROOT/code_gen_exp/config" --config-name "code-gen-swarm.yaml"` [3].

## Start it

- One machine is the unit of scale: each `docker-compose run` or `run_rl_swarm.sh` invocation starts one peer node with one model. There is no `accelerate launch`-style multi-GPU or multi-node form in this repo; multiple peers scale the swarm by each independently running the same launch command [1][9].
- Effective batch size, from the shipped `code_gen_exp/config/code-gen-swarm.yaml`: `training.num_generations: 2`, and the Solver's `data_manager` sets `batch_size: 2`, `local_batch_size: 1`, `proposer_batch_size: 1` - i.e. 1 local sample and 1 proposer-sourced sample per round, 2 generations sampled per prompt [8].
- Config surface: a Hydra YAML, `code_gen_exp/config/code-gen-swarm.yaml`, is what the launcher actually reads - `run_rl_swarm.sh`'s final command passes `--config-path "$ROOT/code_gen_exp/config" --config-name "code-gen-swarm.yaml"` explicitly [3]. The same script separately copies that file to `$ROOT/configs/code-gen-swarm.yaml` on first run (Docker volume-mounts this path so it survives container rebuilds) and, only if the `GENSYN_RESET_CONFIG` env var is set, backs up and resets it when it drifts from the shipped default [3][16]; nothing in `run_rl_swarm.sh` copies `$ROOT/configs/code-gen-swarm.yaml` back into `$ROOT/code_gen_exp/config`, so this card found no evidence that editing the `$ROOT/configs` copy affects the launcher, which reads `$ROOT/code_gen_exp/config/code-gen-swarm.yaml` directly - to change a run, edit that file. Notable defaults the config sets (not "changed from a base," since there is no base trainer config here): `training.dtype: 'float32'` (no bf16/fp16 default - a CPU-friendly but memory-heavier choice), `training.seed: 42`, `training.max_new_tokens: 256`, GRPO `epsilon: 0.2` / `epsilon_high: 0.28`, and `log_with: wandb` [8].
- Model choice is not free: `MODEL_NAME` env var overrides it, but by default a `gpu_model_choice` resolver in the config picks from `default_large_model_pool` (`deepseek-ai/deepseek-coder-1.3b-instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`) on GPU or `default_small_model_pool` (`Qwen/Qwen2.5-Coder-0.5B-Instruct`) on CPU [8].
- Out-of-memory first aid is documented only for the CPU/Mac path, in the README's Troubleshooting section: set `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before relaunching [4]. No GPU-side OOM knob (equivalent to a generation-engine memory-utilization setting) is documented in the README or the config file read for this card.

## Watch it

This section is mechanics only - what a metric shape means for GRPO or PPO training health lives on those methods' own cards, not here.

- **Enable it**: the shipped config sets `log_with: wandb` on the Solver's `GRPOTrainerModule` (inherited from `genrl`'s `LoggerMixin`), which is on by default in this repo's own config, unlike a library that logs nowhere until a tracker is set [8][18]. `LoggerMixin.init_tracker` starts a wandb run with `wandb.init(project="genrl", dir=logging_dir, mode="offline")` - i.e. runs are logged locally in offline mode by default, not streamed to a wandb.ai project automatically [18].
- **Metric names actually emitted by the Solver's `train()` loop**, read directly from `genrl`'s `grpo_trainer.py` (the class `code_gen_exp/src/trainer.py`'s `GRPOTrainerModule` subclasses; fetched at the `gensyn-genrl` `v0.3.0` tag, ahead of nothing here since this is the pinned dependency version) [6]: per training step, `train/loss` (mean loss across minibatches) and `train/rewards` (mean of that step's reward tensor) are logged via `self.log(metrics, global_step)` [6]. Internally, `compute_loss` also tracks per-batch `kl` (only appended when `beta != 0.0`), `clip_ratio`, and `loss` into a `self._metrics[mode]` dict, but these three are accumulated in memory, not shown to be pushed through the same `self.log` call inside the code read for this card [6].
- The Proposer's own PPO loop (`code_gen_exp/src/proposer.py`) emits plain Python `logging` calls at 11 points in the file (e.g. `logger.info(f"Using Vllm for inference: {self._vllm_available}")`, difficulty-level and performance-diff messages, proposal-parse failures, `logger.info(f'proposals: {len(proposal)}')`) - these are terminal/log-file messages via the standard `logging` module, not metrics pushed to a tracker (no wandb/tensorboard call was found in this file), so they land in `swarm.log` rather than as named metrics [7].
- **Sample-level logging of generations** and **evaluation-during-training fields**: not found in `manager.py`, `trainer.py`, or the shipped `code-gen-swarm.yaml` read for this card - the Solver's `evaluate()` method (in `code_gen_exp/src/trainer.py`) sends one generated answer per round to an external judge service (`eval.judge_base_url: https://codezero-judge.gensyn.ai`) rather than logging samples locally [5][8].
- **Logs on disk**: the README's Troubleshooting section names the `/logs` directory contents directly - `yarn.log` for the modal-login server, `swarm.log` as "the main log file for the RL Swarm application," and a `wandb/` directory (including a `debug.log`) that is "only available if you log_with wandb" [4].
- **Published health limit / stopping rule**: `SwarmGameManager.agent_block()`, in `code_gen_exp/src/manager.py`, sets `self.train_timeout = 60 * 60 * 24 * 31` (one month in seconds) and logs the literal string `"Training timed out!"` if a full month passes without the on-chain round advancing past the node's current round; this is a wall-clock ceiling on waiting for the swarm, not a reward- or loss-based stopping rule [9]. No other stopping threshold, patience, or reward-based early-stop was found in `manager.py`, `trainer.py`, or the shipped config read for this card.

## Save it

- rl-swarm's own code defines exactly one persistence path: pushing the trained Solver model to the Hugging Face Hub. `SwarmGameManager._save_to_hf()`, called after every round (`_hook_after_round_advanced`) and at the end of the run (`_hook_after_game`), calls `self.trainer.model.push_to_hub(repo_id=..., token=self.hf_token, commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}", tags=["rl-swarm","genrl-swarm","grpo","gensyn", f"I am {self.animal_name}"])`, gated on both an HF token being supplied at launch and `self.state.round % self.hf_push_frequency == 0` [9]. The shipped config sets `training.hf_push_frequency: 1` - a push attempt every round [8]. The repo name is built as `{username}/{model_name}-Gensyn-Swarm-{animal_name}` [9]. If the push fails, `manager.py` only logs an exception and points the operator at Hugging Face's own upload docs - it does not retry or fall back to a local save [9].
- No local checkpoint-directory logic (numbered `checkpoint-<step>/` folders or similar) exists in `code_gen_exp/src/manager.py` or `code_gen_exp/src/trainer.py`, the two files that own the Solver's training loop in this repo [9][5].
- A local save/load contract does exist, but it lives one layer down, in the separate `gensyn-genrl` dependency's `GRPOLanguageTrainerModule` (base class of rl-swarm's `GRPOTrainerModule`), read at the pinned `v0.3.0` tag: `save(self, save_dir)` calls `self.trainer.save_model(save_dir)` (a full model save, via the underlying HF `Trainer`) and then separately `torch.save({"metrics": ..., "total_train_tokens": ..., "generation_config": ...}, os.path.join(save_dir, "trainer_state.pt"))`; `load(cls, load_dir)` reverses this with `AutoModelForCausalLM.from_pretrained(load_dir)` plus reloading `trainer_state.pt` [6]. Nothing found in `code_gen_exp` calls this `save`/`load` pair, so this card cannot confirm rl-swarm's launcher itself ever exercises it - only that the class it runs inherits it.
- No adapter/PEFT saving path (LoRA or similar) was found in any file read for this card; every save discussed above is a full model.
- Loader handoff: a Hub push from `_save_to_hf()` is a standard full HF model repo, directly loadable with `AutoModelForCausalLM.from_pretrained(repo_id)` by any downstream evaluator - this card did not verify that claim against a loader's own contract beyond that it is an ordinary `push_to_hub()` call [9].

## Find it in the docs

The docs for this repo split across the README itself and a separate hosted docs site; there is no per-version docs URL pattern to teach, since the README lives at whatever commit you check out.

- Primary reference: the repo's own `README.md` on GitHub, which is also where "Requirements," "Instructions," "Environment Overview (CodeZero)," "Identity management," and "Troubleshooting" all live as named sections a reader can jump to directly [4].
- Hosted docs: `https://docs.gensyn.ai/testnet/rl-swarm` is the top-level page the README's own "Learn More" section links to; the release notes for `v0.7.0` point to a specific sub-page, `https://docs.gensyn.ai/testnet/rl-swarm/how-it-works/codezero`, for the CodeZero mechanics, and that exact URL returned HTTP 200 when checked [4][12][19].
- Question-to-section map, from the README's own headings [4]: hardware/Python floors -> "Requirements"; on-chain peer identity, multi-node-same-account rules -> "Identity management"; browser login flow -> "Login"; Hugging Face upload -> "Huggingface"; opting out of the betting side-experiment -> "AI Prediction Market"; log file locations, viem/npm fixes, MacBook OOM fix, VM port-forwarding, multi-GPU-one-machine setup -> "Troubleshooting".
- Runnable references beyond the README: the maintainers point advanced users at the separate `gensyn-ai/genrl` repo's `getting_started.ipynb` notebook for experimenting with the underlying training library directly [1].
- Traps found only in the README's own Troubleshooting section (an official, maintainer-authored source, used here in place of a closed-issue search that returned only generic contributor replies asking for logs - issues #218 and #352 - and one non-maintainer comment on #537, none of which stated a specific new trap): a `viem` package version conflict on the login screen, fixed by pinning `"viem": "2.25.0"` in `modal-login/package.json` or running `yarn upgrade && yarn add next@latest && yarn add viem@latest`; and running multiple GPUs on one machine is supported only by manually isolating each GPU, installing the repo once per GPU, and exposing each peer on a different port [4].
- Honest boundary: this is not a library for training your own model on your own data - the task (CodeZero code generation), reward/judge model, and model pool are fixed by the shipped config. Whether an official Gensyn-run swarm is currently joinable is unclear from the README itself, which contains both a "no official swarms running right now" note and a separate "Currently, we are running CodeZero on the Gensyn Testnet" statement [1][4].

## Sources

[1] rl-swarm README, top section and "Instructions" (self-description, Docker/shell-script launch commands, login flow, GenRL attribution, no-swarms-running notice). https://raw.githubusercontent.com/gensyn-ai/rl-swarm/9c95410b1ac0d0a6005513c276b6b84f6db13bcd/README.md. Fetched 2026-08-12.

[2] gensyn-ai/rl-swarm GitHub repository API (owner, description, stargazers_count, license, pushed_at). https://api.github.com/repos/gensyn-ai/rl-swarm. Fetched 2026-08-12.

[3] rl-swarm `run_rl_swarm.sh` (shell-script launch path, config-copy/reset logic, final launcher invocation). https://raw.githubusercontent.com/gensyn-ai/rl-swarm/9c95410b1ac0d0a6005513c276b6b84f6db13bcd/run_rl_swarm.sh. Fetched 2026-08-12.

[4] rl-swarm README, "Requirements," "Environment Overview (CodeZero)," "Identity management," and "Troubleshooting" sections. Same URL as [1]. Fetched 2026-08-12.

[5] rl-swarm `code_gen_exp/src/trainer.py` (Solver's `GRPOTrainerModule`, its judge-based `evaluate()` hook). https://raw.githubusercontent.com/gensyn-ai/rl-swarm/9c95410b1ac0d0a6005513c276b6b84f6db13bcd/code_gen_exp/src/trainer.py. Fetched 2026-08-12.

[6] `gensyn-genrl` `src/genrl/trainer/grpo_trainer.py` at the pinned dependency tag (GRPOLanguageTrainerModule's `train`/`step`/`compute_loss` metrics, `save`/`load` methods). https://raw.githubusercontent.com/gensyn-ai/genrl/v0.3.0/src/genrl/trainer/grpo_trainer.py. Fetched 2026-08-12.

[7] rl-swarm `code_gen_exp/src/proposer.py` (Proposer's `PPOConfig`, `VllmConfig`, and clipped-ratio training loop). https://raw.githubusercontent.com/gensyn-ai/rl-swarm/9c95410b1ac0d0a6005513c276b6b84f6db13bcd/code_gen_exp/src/proposer.py. Fetched 2026-08-12.

[8] rl-swarm shipped config, `code_gen_exp/config/code-gen-swarm.yaml` (training/reward/game_manager/data_manager/proposer sections, GRPO epsilon values, model pools). https://raw.githubusercontent.com/gensyn-ai/rl-swarm/9c95410b1ac0d0a6005513c276b6b84f6db13bcd/code_gen_exp/config/code-gen-swarm.yaml. Fetched 2026-08-12.

[9] rl-swarm `code_gen_exp/src/manager.py` (`SwarmGameManager`: `_save_to_hf`, `agent_block` timeout, round-advance hooks). https://raw.githubusercontent.com/gensyn-ai/rl-swarm/9c95410b1ac0d0a6005513c276b6b84f6db13bcd/code_gen_exp/src/manager.py. Fetched 2026-08-12.

[10] rl-swarm repository root directory listing (confirms absence of a root `requirements.txt`/`pyproject.toml`). https://api.github.com/repos/gensyn-ai/rl-swarm/contents/?ref=9c95410b1ac0d0a6005513c276b6b84f6db13bcd. Fetched 2026-08-12.

[11] gensyn-ai/rl-swarm commits API (confirms `9c95410b1ac0d0a6005513c276b6b84f6db13bcd` is HEAD of `main`, dated 2026-01-05). https://api.github.com/repos/gensyn-ai/rl-swarm/commits/main. Fetched 2026-08-12.

[12] gensyn-ai/rl-swarm releases API (latest release `v0.7.0` "CodeZero," published 2025-11-12, author `jcd496`, body linking the CodeZero docs page). https://api.github.com/repos/gensyn-ai/rl-swarm/releases. Fetched 2026-08-12.

[13] rl-swarm `code_gen_exp/requirements.txt` and `docker-compose.yaml`, fetched at both the pinned commit and the `v0.7.0` tag and diffed byte-for-byte identical. https://raw.githubusercontent.com/gensyn-ai/rl-swarm/9c95410b1ac0d0a6005513c276b6b84f6db13bcd/code_gen_exp/requirements.txt and .../v0.7.0/code_gen_exp/requirements.txt (and the equivalent `docker-compose.yaml` pair). Fetched 2026-08-12.

[14] `gensyn-genrl` PyPI JSON API (version 0.3.0: `requires_python`, `license` field). https://pypi.org/pypi/gensyn-genrl/0.3.0/json. Fetched 2026-08-12.

[15] rl-swarm `LICENSE.TXT` (MIT License, copyright Gensyn). https://raw.githubusercontent.com/gensyn-ai/rl-swarm/9c95410b1ac0d0a6005513c276b6b84f6db13bcd/LICENSE.TXT. Fetched 2026-08-12.

[16] rl-swarm `docker-compose.yaml` (GPU service's NVIDIA driver-version comment and base image; `ollama/ollama:0.11.10` service image). Same file as cited in [13]. Fetched 2026-08-12.

[17] gensyn-ai/rl-swarm commit detail for `9c95410b1ac0d0a6005513c276b6b84f6db13bcd` (author `sirbonneville`, date). https://api.github.com/repos/gensyn-ai/rl-swarm/commits/9c95410b1ac0d0a6005513c276b6b84f6db13bcd. Fetched 2026-08-12.

[18] `gensyn-genrl` `src/genrl/logging_utils/ml_logger.py` at the pinned dependency tag (`LoggerMixin.init_tracker`'s offline wandb.init call). https://raw.githubusercontent.com/gensyn-ai/genrl/v0.3.0/src/genrl/logging_utils/ml_logger.py. Fetched 2026-08-12.

[19] Gensyn docs page for CodeZero (checked reachable via a status-code-only request; content not otherwise cited in this card). https://docs.gensyn.ai/testnet/rl-swarm/how-it-works/codezero. Fetched 2026-08-12 (HTTP 200 confirmed).
