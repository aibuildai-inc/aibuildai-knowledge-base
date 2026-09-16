# OpenManus-RL

A research repo that wires ALFWorld, WebShop, and other agent environments into a forked verl PPO trainer - active, but its main branch is stale and its own quickstart install command is broken.

OpenManus-RL is described on its own repository as "A live stream development of RL tunning for LLM agents" [1], and its README frames it the same way: "an open-source initiative collaboratively led by Ulab-UIUC and MetaGPT", extending the separate OpenManus agent project and exploring RL-based tuning inspired by DeepSeek-R1 and QwQ-32B [2]. It ships no trainer-class API of its own; instead it vendors a fork of Bytedance's verl as a git submodule and adds agent-environment glue (ALFWorld, WebShop, a generic "tool_use" environment) plus shell scripts that call verl's `python3 -m verl.trainer.main_ppo` with environment-specific config overrides [2][3]. It lives at https://github.com/OpenManus/OpenManus-RL [1].

**When to pick it**: pick this only if you specifically want the agent-environment reward glue (ALFWorld subgoal/distance reward shaping, an LLM-judge reward for a tool-use environment) already wired to a verl PPO trainer - not as a general-purpose post-training library [3][4]. Both shipped example launch scripts contain a literal bash bug (a `# TODO` comment breaks a line-continuation) and one ships with empty data paths, so neither runs unedited [5][6]. GitHub reports a push as recent as 2026-05-05, but that single push is an automated dependency-bump commit on a `dependabot/*` branch, not human activity; `main`, which is what this card is pinned to, and the repo's other named/personal branches are all dated 2025 (see Maintained by for the exact dates) [1][7]. For a maintained general framework, weigh verl or verl-agent directly instead (cross-reference; not covered here).

**Methods it ships**: only PPO is shipped as a runnable script, for ALFWorld and WebShop, via verl's `main_ppo` entry point with per-environment config overrides [5][6]. The README's "Post-Training Strategies" and "Method" sections also name GRPO, DPO, and PRM as directions the project explores [2], and a "Training of Agent Reward Model" section says the project trains standalone reward models [2] - but no GRPO/DPO training script and no reward-model-training script exist anywhere in the tree at the pinned commit; the only "reward" code present computes reward signals for PPO rollouts, not a trained reward model [8]. Concretely: `openmanus_rl/environments/env_package/alfworld/alfworld/env/reward.py` implements a hardcoded, per-action dense reward (e.g. a `(prev_distance - curr_distance) * 0.2` shaping term for navigation, plus a fixed positive/negative reward on subgoal completion) [9]; `openmanus_rl/environments/env_package/tool_use/reward_manager.py` implements an LLM-judge reward that extracts an `<answer>` tag from the agent's response and scores it against ground truth using a GPT-4o-mini call [10]. Method math and defining papers for PPO live on that method's own card, not here.

**Scale it handles**: this repo adds no launcher of its own - it calls verl's Ray-based `python3 -m verl.trainer.main_ppo` directly and sets verl's own `trainer.n_gpus_per_node` / `trainer.nnodes` config keys [5][6]; distributed execution (Ray worker groups, FSDP sharding) is documented at the architecture level in the repo's own docs page but the mechanics belong to the pinned verl submodule, not to this repo [11]. Both shipped example scripts hard-code single-node layouts: `train_alfworld.sh` sets `trainer.n_gpus_per_node=4`, `trainer.nnodes=1`; `train_webshop.sh` sets `trainer.n_gpus_per_node=2`, `trainer.nnodes=1` (both also set `CUDA_VISIBLE_DEVICES="1,2,3,4"`, which only lists which 4 device IDs are visible to the process, not how many verl actually schedules onto - that count is the `n_gpus_per_node` value) [5][6]; no multi-node example or benchmark is published anywhere in this repo.

**Install**: not on PyPI - a lookup of both `openmanus-rl` and `openmanus_rl` on PyPI returned "Not Found" (checked 2026-08-11) [12][13]. The README's documented command, `git clone --recursive` then `pip install -e .[vllm]` [2], does not work as written: the repo's own root `pyproject.toml` is an unedited copy of verl's own file - it declares `name = "verl"`, pins `transformers<4.48` and `vllm<=0.6.3`, and defines only a `test` extras group, no `vllm` extra [14]; a separate `setup.py`, marked in its own header as "the fallback installation script when pyproject.toml does not work," does define a `vllm` extra (`vllm<=0.8.5`, `tensordict<=0.6.2`) [15], but pip uses the PEP 621 `[project]` table in `pyproject.toml` when present, so the `.[vllm]` extra the README tells you to install is not resolvable. A maintainer (`realtmxi`, repository collaborator) answered this exact confusion in a closed issue by telling the reporter to instead run `pip install -r requirements.txt` [16]. That file pins `transformers==4.51.1` and leaves `vllm` commented out, `tensordict<=0.6.2` [17] - a third, different pin set from both `pyproject.toml` and `setup.py`. Apache-2.0 license [18][1]. No CUDA or hardware floor is stated in any of `pyproject.toml`, `setup.py`, or `requirements.txt`. The `verl` submodule itself is pinned (via `.gitmodules`) to a fork, `realtmxi/verl`, branch `main`, resolving at the screened commit to `46aaa8969...` [19] - not upstream `volcengine/verl`.

**Maintained by**: named contributors from Ulab-UIUC and MetaGPT, listed in the README as Kunlun Zhu, Muxin Tian, Zijia Liu, Yingxuan Yang, Jiayi Zhang, Xinbing Liang, Weijia Zhang, Haofei Yu, Cheng Qian, and Bowen Jin [2]. `pushed_at` reports 2026-05-05 [1], but `main`'s own tip, which is the commit this card is pinned to, is dated 2025-09-21 [20], and the repo's other 42 non-`main` branches are dated 2025 - including all six named `dev_05xx` (`dev_0511`'s tip, the latest of them, is 2025-05-12) [7]. Only one GitHub Release exists: a prerelease tagged `version`, "OpenManus 0.0.2", published 2025-05-09, whose note says "Now the OpenManus can be successfully trained using train_ppo.sh and save checkpoints. More testing on different environments and rl algorithms coming soon" [21] - note `train_ppo.sh` does not exist in the tree at the pinned commit (see Find it in the docs). The one branch that matches `pushed_at` exactly is `dependabot/pip/gymnasium-1.1.1`, an automated dependency-bump commit that updates gymnasium from 0.29.1 to 1.1.1, dated 2026-05-05T06:03:26Z [22].

## Quick start

The README's own "Quick Start" sequence, condensed to the commands it gives verbatim [2]:

```bash
# Clone with submodules
git clone --recursive https://github.com/OpenManus/OpenManus-RL.git

# Environment
conda create -n openmanus-rl python=3.10 -y
conda activate openmanus-rl
pip3 install torch torchvision
pip install -e .[vllm]        # README's documented command - see Install for why this fails as written;
                               # a maintainer instead directs `pip install -r requirements.txt` [16]
pip3 install flash-attn --no-build-isolation
pip install wandb

# ALFWorld environment
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
alfworld-download -f

# Train (after downloading the OpenManus-RL dataset from Hugging Face)
bash scripts/ppo_train/train_alfworld.sh
```

There is no minimal Python snippet (no `import openmanus_rl; trainer.train()` form) in the README or docs read for this card - the only documented entry point is the shell script above, which itself needs the edits described in Start it before it runs [2][5].

## Start it

- One node only: neither shipped script wraps verl's launcher in anything (no Accelerate, no explicit `torchrun`) - it is a direct `python3 -m verl.trainer.main_ppo <hydra overrides>` call [5][6]. GPU count is set by verl's own `trainer.n_gpus_per_node`/`trainer.nnodes` keys, not by `CUDA_VISIBLE_DEVICES`; both shipped scripts hard-code 1 node, but `train_alfworld.sh` sets `n_gpus_per_node=4` and `train_webshop.sh` sets `n_gpus_per_node=2` [5][6].
- Effective batch: the ALFWorld script sets `data.train_batch_size=128`, `actor_rollout_ref.actor.ppo_mini_batch_size=256`, `ppo_micro_batch_size_per_gpu=4`; the WebShop script sets `data.train_batch_size=128`, `ppo_mini_batch_size=64`, `ppo_micro_batch_size_per_gpu=8` [5][6] - these are literal values in the scripts, with no batch-size arithmetic helper provided by this repo.
- Config surface: both scripts pass Hydra-style dot-path overrides straight to `verl.trainer.main_ppo` (`algorithm.*`, `data.*`, `actor_rollout_ref.*`, `critic.*`, `trainer.*`); the keys and their defaults are defined in the pinned verl submodule's own `ppo_trainer.yaml`, not in this repo - this repo's contribution is the specific override values plus the `env.*` keys (`env.env_name`, `env.seed`, `env.max_steps`) that select and size the agent environment [5][6][19].
- Non-obvious defaults this repo sets: `actor_rollout_ref.rollout.gpu_memory_utilization=0.3` (ALFWorld) / `0.6` (WebShop) for the colocated vLLM generation share [5][6]; and, in both scripts, `trainer.save_freq=-1` - per the pinned verl submodule's own config comment this is the "Save frequency (by iteration) for model checkpoints" and -1 is verl's own default, meaning as shipped, neither script ever writes a checkpoint [5][6][19].
- Both shipped scripts carry the same bug: a data-path line ends `\ # TODO: change to the correct path` - the trailing comment sits after the line-continuing backslash, which breaks the bash multi-line command; `train_webshop.sh` additionally ships both `data.train_files=` and `data.val_files=` empty [5][6]. Both need hand-editing before they will run.
- OOM: no OpenManus-RL-specific out-of-memory guidance is published in the README or the two `docs/` pages read for this card; the only tunable knobs visible are the verl actor/critic micro-batch sizes and `gpu_memory_utilization` already shown in the shipped scripts [2][11][23].

## Watch it

This section covers only the logging mechanics this repo controls; what a metric means for PPO belongs to PPO's own method card, and the metric set itself belongs to the pinned verl submodule (cross-reference), not restated here.

- Enable it: both shipped scripts set `trainer.logger=['console','wandb']`, with `WANDB_API_KEY` left empty and `WANDB_BASE_URL` pointed at a non-default proxy (`https://api.bandw.top`) - both need to be filled in/edited before wandb logging will actually work [5][6].
- The repo's own architecture doc walks the reward path conceptually - trajectories collected by `OpenManusAgent.run_llm_loop`, combined via a `RewardComposer` into `GoalReward`, `LengthPenalty`, and `FormatReward` components, then allocated to tokens by one of `last_token` / `uniform_positive` / `discounted` strategies - but it gives no table of the metric names actually written to the logger [11]. The development guide separately says `run_llm_loop` "Handles visualization if enabled" but names no config key or output location for that visualization [23].
- Evaluation during training: both scripts set `trainer.val_before_train=True` and `trainer.test_freq=5`, verl config keys [5][6].
- Stopping-rule: none published. Neither `docs/README.md` nor `docs/DEVELOPMENT_GUIDE_EN.md`, the two documentation pages read for this card, states a threshold, patience value, or stopping criterion; training is fixed-epoch (`trainer.total_epochs=150` in the ALFWorld script) [5][11][23].

## Save it

- With `trainer.save_freq=-1` left as shipped, no checkpoint is ever written; set it to a positive iteration count to enable saving [5][6][19].
- When enabled, the checkpoint directory is wherever the pinned verl submodule's own `default_local_dir` template resolves - `checkpoints/${trainer.project_name}/${trainer.experiment_name}` - which neither shipped script overrides [19].
- Resume is governed by the pinned verl submodule's own `resume_mode` key (`auto` / `disable` / `resume_path`); neither shipped script sets it, so verl's own default (`auto`, resume from the last checkpoint if one exists) applies [19].
- No adapter/PEFT save path is documented in the README or the two `docs/` pages read for this card; `peft` appears in `requirements.txt` as a dependency, but no script under `scripts/` invokes it [17][5][6].
- Loader handoff: not documented on this repo's own pages - whatever format the underlying verl checkpoint writes is what a loader must read, and that contract belongs to the pinned verl submodule, not to OpenManus-RL.

## Find it in the docs

There is no hosted docs site. All narrative documentation is two files inside the repository itself, linked from the README's own "Documentation" list [2]: `docs/README.md` (a "Training Process Overview" walking the PPO training loop and reward composition) [11] and `docs/DEVELOPMENT_GUIDE_EN.md` (a component-by-component developer guide, with Chinese translations of both alongside them) [23]. Address pattern for either: `https://raw.githubusercontent.com/OpenManus/OpenManus-RL/<ref>/docs/<file>`, or browse at `https://github.com/OpenManus/OpenManus-RL/blob/main/docs/<file>`.

- The development guide is stale relative to the tree at the pinned commit: it walks through `train_ppo.sh` and `train_grpo.sh` at the repo root, and its own worked examples reference a maintainer's personal path (`/home/kunlunz2/github_repos/OpenManus-RL/...`) [23] - neither script exists anywhere in the tree at commit `c66d4930...`; the only shipped training scripts are `scripts/ppo_train/train_alfworld.sh` and `scripts/ppo_train/train_webshop.sh` [5][6]. Follow the actual `scripts/` tree over the development guide's file names.
- Runnable references beyond the docs: ready-made parquet datasets ship directly under `data/<env>/{train,val,test}.parquet` for eleven environments - `babyai`, `gaia`, `maze`, `movie`, `sciworld`, `sqlgym`, `textcraft`, `todo`, `weather`, `webshop`, `wordle` [24]. ALFWorld is not among them even though it has a dedicated PPO script; that script's own `data.train_files`/`data.val_files` are the broken placeholders described above, so it does not point at any of the shipped parquet files by default [5][24].
- Community layer: none curated by the project - no tutorials page and no MCP endpoint is referenced from the README or the two `docs/` pages read for this card.
- Trap, from a closed issue: a user asked which of the repo's several dependency files to follow to get `train_ppo.sh` running; the reply, from repository collaborator `realtmxi` (2025-05-23), was simply `pip install -r requirements.txt` [16] - i.e., treat `requirements.txt` as the working install path over the broken `pip install -e .[vllm]` in the README and over `pyproject.toml`/`setup.py`.
- Honest boundary: the README's own "Method" and "Post-Training Strategies" sections describe GRPO, DPO, PRM, and standalone reward-model training as project directions [2], but as of the pinned commit only PPO has a runnable training script, for ALFWorld and WebShop only [5][6][8].

## Sources

[1] OpenManus-RL GitHub repository metadata (description, license, `pushed_at`, star count). https://api.github.com/repos/OpenManus/OpenManus-RL. Fetched 2026-08-11.

[2] OpenManus-RL README, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/README.md. Fetched 2026-08-11.

[3] OpenManus-RL `.gitmodules` (verl submodule pointer), at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/.gitmodules. Fetched 2026-08-11.

[4] OpenManus-RL repository tree, at commit c66d493. https://api.github.com/repos/OpenManus/OpenManus-RL/git/trees/c66d493022bc4c58e875ba73e8f6a07dd5d499a2?recursive=1. Fetched 2026-08-11.

[5] `scripts/ppo_train/train_alfworld.sh`, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/scripts/ppo_train/train_alfworld.sh. Fetched 2026-08-11.

[6] `scripts/ppo_train/train_webshop.sh`, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/scripts/ppo_train/train_webshop.sh. Fetched 2026-08-11.

[7] OpenManus-RL branches list. https://api.github.com/repos/OpenManus/OpenManus-RL/branches?per_page=100. Fetched 2026-08-11.

[8] Repository tree search for `reward`-named files, at commit c66d493 (from [4]); confirms only `alfworld/env/reward.py`, `tool_use/reward_manager.py`, and `openmanus_rl/reward_manager/episode.py` exist, no reward-model-training script.

[9] `openmanus_rl/environments/env_package/alfworld/alfworld/env/reward.py`, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/openmanus_rl/environments/env_package/alfworld/alfworld/env/reward.py. Fetched 2026-08-11.

[10] `openmanus_rl/environments/env_package/tool_use/reward_manager.py`, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/openmanus_rl/environments/env_package/tool_use/reward_manager.py. Fetched 2026-08-11.

[11] `docs/README.md` ("OpenManus Model Training Overview"), at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/docs/README.md. Fetched 2026-08-11.

[12] PyPI lookup, `openmanus-rl`. https://pypi.org/pypi/openmanus-rl/json. Fetched 2026-08-11 (returns "Not Found").

[13] PyPI lookup, `openmanus_rl`. https://pypi.org/pypi/openmanus_rl/json. Fetched 2026-08-11 (returns "Not Found").

[14] `pyproject.toml`, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/pyproject.toml. Fetched 2026-08-11.

[15] `setup.py`, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/setup.py. Fetched 2026-08-11.

[16] Closed issue #44 and its comment thread. https://api.github.com/repos/OpenManus/OpenManus-RL/issues/44 and https://api.github.com/repos/OpenManus/OpenManus-RL/issues/44/comments. Fetched 2026-08-11.

[17] `requirements.txt`, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/requirements.txt. Fetched 2026-08-11.

[18] `LICENSE`, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/LICENSE. Fetched 2026-08-11.

[19] `verl/trainer/config/ppo_trainer.yaml`, from the pinned `realtmxi/verl` submodule commit `46aaa89691c4e82636847edfa5e1974b8829a2fa` (the gitlink recorded for `verl` in [4]). https://raw.githubusercontent.com/realtmxi/verl/46aaa89691c4e82636847edfa5e1974b8829a2fa/verl/trainer/config/ppo_trainer.yaml. Fetched 2026-08-11. This submodule commit is a dependency pin read alongside the OpenManus-RL commit, not itself the object this card screens.

[20] OpenManus-RL `main` branch metadata (confirms `main`'s tip commit and date match the screened commit c66d493). https://api.github.com/repos/OpenManus/OpenManus-RL/branches/main. Fetched 2026-08-11.

[21] OpenManus-RL GitHub Releases (the "OpenManus 0.0.2" prerelease, tag `version`). https://api.github.com/repos/OpenManus/OpenManus-RL/releases. Fetched 2026-08-11.

[22] Commit detail for `dependabot/pip/gymnasium-1.1.1` branch tip. https://api.github.com/repos/OpenManus/OpenManus-RL/commits/3a7540bf2245e94c9b6993fff6117c5578ec5c6b. Fetched 2026-08-11.

[23] `docs/DEVELOPMENT_GUIDE_EN.md`, at commit c66d493. https://raw.githubusercontent.com/OpenManus/OpenManus-RL/c66d493022bc4c58e875ba73e8f6a07dd5d499a2/docs/DEVELOPMENT_GUIDE_EN.md. Fetched 2026-08-11.

[24] Repository tree search under `data/`, at commit c66d493 (from [4]); lists the eleven environments with shipped parquet files.
