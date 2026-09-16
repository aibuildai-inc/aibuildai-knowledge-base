# XPolicyLab

Repository: https://github.com/XPolicyLab/XPolicyLab

A shared serving-and-evaluation layer for 42 vendored robot manipulation policies, not a training library in its own right - each policy's install, data prep, and training scripts live under its own `policy/<POLICY>/` and are whatever that policy's upstream authors wrote.

XPolicyLab describes itself as "the shared layer between policy code and evaluation environments", handling "serving, observation/action contracts, and eval wiring" while each model's "dependencies, checkpoints, and training recipes" stay under `policy/<POLICY>/` [1]. It is a community project led by MMLab@HKU and THU, with Tianxing Chen as project lead and a broader author list credited on the accompanying arXiv paper [1][2]. The top-level API is a websocket client/server split: a policy server wraps a per-policy `Model` class (`__init__`, `update_obs`, `get_action`, `reset`, plus batched variants) and an environment client drives it through `eval.sh` or a split-machine server/client pair [1].

**When to pick it**: pick XPolicyLab when you need one standardized serving/evaluation harness across many pre-existing robot manipulation policies (VLA, world-action, imitation-learning, memory-augmented) feeding the RoboDojo or RoboTwin benchmarks [1] - not when you need a general post-training trainer. It ships no training-method implementation of its own; the one RL trainer found in the tree is vendored inside a single policy adapter (below), so it is not a substitute for trl, verl, or similar libraries for writing new post-training runs. This card does not compare it against those on training throughput or method coverage - no such comparison is published.

**Methods it ships**: none, as a library-level API - `pyproject.toml` packages only `client_server` and `utils` and explicitly excludes `policy*` from the installable package [3]. Inside the vendored `policy/Dexbotic_DM0/dexbotic/` tree, one adapter carries an RL trainer: its `GRPOConfig` class docstring states the advantage calculation is "based on SimpleVLA-RL implementation", computing group-normalized outcome-reward advantages, and the accompanying `GRPOTrainer` class (whose own docstring says only that it is a "GRPO trainer implementation following SimpleVLA-RL") applies a PPO-style clipped policy loss with `clip_ratio_low=0.2` and `clip_ratio_high=0.28` as its function defaults [4]. That is the only training-method code confirmed inside this repository; the shortlist row's DPO and SFT evidence paths point to files in a different repository (`nvidia-cosmos/cosmos-rl`), not to anything inside XPolicyLab, so this card makes no claim that XPolicyLab ships DPO or SFT. Of the two policy READMEs read for this card, `demo_policy` ships a `train.sh` stub that only demonstrates checkpoint-naming conventions and performs no real training, while Dexbotic_DM0 documents a real `train.sh` entrypoint that calls into its own vendored trainer (the GRPO code above); this card read only these two of the 42 policy READMEs, so it makes no claim about the training-script pattern across the rest [5][6].

**Scale it handles**: the repo documents scale for evaluation serving, not for the vendored trainers. `eval.sh` runs the policy server and environment client on one machine; `setup_eval_policy_server.sh` / `setup_eval_env_client.sh` split them across a GPU machine and a simulator/robot machine over websocket, with the server bound to `0.0.0.0` and the client given the server's real IP [1]. Per-policy `train.sh` scripts take a comma-separated `gpu_id` list with `NUM_GPUS` inferred from it (documented this way for the Dexbotic_DM0 adapter) [6], but no multi-node training launcher or sharding option is documented at the XPolicyLab level - any such mechanism belongs to whichever upstream trainer a given policy vendors.

**Install**: `pip install -e .` after `git clone https://github.com/XPolicyLab/XPolicyLab.git` [1]. There is no PyPI package and no tagged GitHub release, so a fresh clone always delivers whatever is on the default branch at clone time; this field reports the live `main` branch fetched 2026-08-12 at commit `8b74924181eb877bb43a5da0ec2593740e0d4ebf` [7]. `pyproject.toml` there sets `requires-python = ">=3.10"`, version `0.0.1`, and Apache-2.0 licence (confirmed by the repository's own `LICENSE` file) [3][8]. Dependencies pinned there are all lower-bound only, no upper caps: `numpy>=1.23`, `pyyaml>=6`, `opencv-python-headless>=4.8`, `h5py>=3.8`, `websockets>=14.0`, `msgpack>=1.0.8`, `msgpack-numpy>=0.4.8`, `pydantic>=2.5` [3]. No deep-learning core (torch, transformers) is pinned at this level - torch and any model-specific stack are installed per policy by that policy's own `install.sh`. No CUDA or hardware minimum is stated anywhere in `pyproject.toml` or the README; hardware requirements flow entirely through each policy's own install script [3][1]. The shortlist instead pins an earlier commit, `72b3a52f2b069220088ef824ade5890885858379` (2026-07-29); `pyproject.toml` read at that exact commit is identical to the live fetch above except one floor - `websockets>=13.0` there, not `>=14.0` - so a reader targeting the shortlist's pinned snapshot specifically should expect that older websockets floor and nothing else different in this dependency list [9].

**Maintained by**: the XPolicyLab community, MMLab@HKU and THU, led by Tianxing Chen [1]. The repository shows sustained activity through 2026-08-11 - the newest push at the time of this reading reverts an in-progress policy submission (PR #93, "Revert '[policy] CSU-AI-0: add CSU-AI-0 adaptor'") [7], and the project has an accompanying arXiv paper, "XPolicyLab: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment" (arXiv:2608.09892, cs.RO) [2].

## Quick start

From the README's own quick-start block [1]:

```bash
mkdir demo_env
cd demo_env
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab
pip install -e .
```

Fetch a small bundled demo dataset (no simulator needed) and run the offline wiring check against the reference `demo_policy` adapter, which returns zero actions and does not train or load a real checkpoint [5]:

```bash
bash scripts/RoboDojo/download_robodojo_data.sh demo

export EVAL_ENV_TYPE=debug
cd policy/demo_policy
bash install.sh
bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 base base
```

The same template (`install.sh`, then `process_data.sh`, `train.sh`, `eval.sh`) applies to any real adapter by swapping in `policy/<POLICY>` and that policy's argument values [1].

## Start it

- One machine: `bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env_conda_env>` starts the policy server and environment client together and cleans up afterward [1].
- Split machines: `bash setup_eval_policy_server.sh ... 0.0.0.0` on the GPU machine, then `bash setup_eval_env_client.sh ... <policy_server_port> <policy_server_ip>` on the simulator/robot machine, connecting over websocket [1].
- Training a specific policy goes through that policy's own `train.sh`, e.g. for Dexbotic_DM0: `bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>`, where `gpu_id` is comma-separated for multi-GPU and `NUM_GPUS` is inferred from it; the effective batch there is `global_batch = DM0_BATCH_SIZE x NUM_GPUS x DM0_GRAD_ACCUM`, with `DM0_GRAD_ACCUM` auto-derived if unset [6]. This arithmetic and these env vars are specific to the Dexbotic_DM0 adapter, not a library-wide config surface - XPolicyLab has no shared training Config class.
- No library-level default is CHANGED from a base framework (there is no base training framework here to change from); precision, batch, and other training defaults are set per policy inside that policy's own scripts.
- Out-of-memory first aid is not published at the XPolicyLab level; it would live in the individual policy's README (e.g. Dexbotic_DM0's `DM0_BATCH_SIZE` / `DM0_GRAD_ACCUM` knobs) [6], not in a shared XPolicyLab surface.

## Watch it

XPolicyLab itself does not train, so it does not define a metrics surface for a training run; watching a run means watching whatever the individual policy's vendored trainer logs, on that policy's own terms. For evaluation, the live README (fetched 2026-08-12) documents a transport-level signal rather than a metric-level one: the client raises `ServerRestartedError` if a reconnect lands on a different server process, a `timeout` error means the call may still be running on the server (treat it as fatal, do not retry), and the full traceback of a model failure is logged on the policy-server side, not surfaced to the client; `deploy.yml` timeout keys (`request_timeout_s` default `120.0`, `max_connect_attempts` default `180`, `max_connect_seconds` default `900.0`, and others) are the closest thing to a published operational limit, and they govern connection/serving health, not training convergence [1]. This entire transport-details passage is absent from the README as it read at the shortlist's pinned commit (`72b3a52f...`, 2026-07-29) - it was added to the repository after that snapshot, so a reader working strictly from the pinned commit will not find it there [9]. No stopping-rule, threshold, or training-metric list is published anywhere in the README, CONTRIBUTING.md, or the Dexbotic_DM0 adapter's own README read for this card [1][6]; a reader who needs training-time metrics must open the specific policy's own training code, e.g. `policy/Dexbotic_DM0/dexbotic/dexbotic/exp/rl/rl_base.py` for its GRPO trainer [4].

## Save it

Checkpoint layout is a naming convention, not a file-format contract, because XPolicyLab does not write the checkpoint files itself: `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` is the standard directory name that lets `eval.sh` find a run without extra bookkeeping [1]. At eval time `ckpt_name` can be the short training nickname (auto-combined into that directory name), the full run-directory name, or an explicit path; some adapters instead read explicit keys from `deploy.yml` (`checkpoint_path`, `model_path`) [1]. What actually lands inside that directory - the weight file names, whether optimizer state is retained, adapter-only vs. merged saves - is set entirely by the individual policy's own training code and is not documented at the XPolicyLab level; for Dexbotic_DM0 specifically, no retention flag or file inventory is given beyond the directory name itself [6]. There is no XPolicyLab-level `save_model` / `push_to_hub` / `resume_from_checkpoint` call - `model.py`'s `__init__(model_cfg)` is the only load path the framework defines, and it is each adapter's job to point that at a checkpoint [1]. Whether a saved checkpoint can be loaded by an evaluator is therefore not this framework's contract; it is the contract of whatever trainer wrote the checkpoint.

## Find it in the docs

There is no versioned docs site; the README on the default branch (`main`) is the primary reference, alongside `CONTRIBUTING.md` for the adapter contract and one README per policy under `policy/<POLICY>/README.md` [1]. Fetched 2026-08-12 at commit `8b74924181eb877bb43a5da0ec2593740e0d4ebf` (the repository's newest push at that time) [7].

- Project website: `https://xpolicylab.github.io/`, which the README lists as carrying the full contributor list across every integrated policy [1].
- Leaderboards it feeds: RoboDojo (`https://robodojo-benchmark.com/LeaderBoard`) and RoboTwin (`https://robotwin-platform.github.io/leaderboard`) [1].
- Per-policy lookup: every entry in the README's "Integrated Policies" catalog links to `policy/<POLICY>/README.md`, which is the source of truth for that model's install, data format, training entrypoint, and checkpoint layout [1]. `policy/demo_policy/README.md` is the minimal reference adapter to read first [5].
- Contribution rules: `CONTRIBUTING.md` documents the full standard for a new adapter PR - required files, the `Model` contract, `deploy.yml` keys, and script conventions [1].
- Runnable references: `scripts/create_policy.sh <POLICY_NAME>` scaffolds a new adapter; `scripts/RoboDojo/download_robodojo_data.sh demo` fetches a small known-good HuggingFace demo bundle (10 episodes) plus larger `hdf5`, `lerobot_v3.0`, `lerobot_v2.1`, and `real` (real-robot HDF5) exports for offline development [1].
- Coding-agent skills: the repo ships two Agent Skills under `.agents/skills` (symlinked from `.cursor/skills` and `.claude/skills`) - `xpolicylab-model-integration` to build an adapter and `xpolicylab-adapter-check` to audit one against `CONTRIBUTING.md`; `AGENTS.md` carries the always-on rules for any coding agent working in the repo [1]. No official MCP endpoint for the docs is named anywhere read for this card.
- No community-tutorial curation page (e.g. a blog roundup) exists in the README beyond the project website and the two integrated benchmarks; this card names no third-party blogs because none are curated by the project itself.

**Boundary a reader would hit**: this is an evaluation and serving harness, not a post-training method library - a reader looking for a shared trainer, a shared metrics surface, or a shared checkpoint format across policies will not find one; each of the 42 `policy/<POLICY>/` adapters is free to vendor its own upstream training code (or, like `demo_policy`, ship a non-functional training stub) [1][5]. Only one training method (GRPO) was confirmed present anywhere in this repository's own tree, inside a single vendored adapter, not as a library feature [4]. No GitHub issue with a maintainer reply documenting a specific breakage was found in the pages read for this card.

## Sources

[1] XPolicyLab README. https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/main/README.md. Fetched 2026-08-12 (unpinned `main` branch content).

[2] XPolicyLab arXiv abstract page. https://arxiv.org/abs/2608.09892. Fetched 2026-08-12.

[3] XPolicyLab `pyproject.toml`. https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/main/pyproject.toml. Fetched 2026-08-12 (unpinned `main` branch content).

[4] XPolicyLab vendored GRPO trainer, `policy/Dexbotic_DM0/dexbotic/dexbotic/exp/rl/rl_base.py`. https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/main/policy/Dexbotic_DM0/dexbotic/dexbotic/exp/rl/rl_base.py. Fetched 2026-08-12 (unpinned `main` branch content).

[5] `policy/demo_policy/README.md`. https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/main/policy/demo_policy/README.md. Fetched 2026-08-12 (unpinned `main` branch content).

[6] `policy/Dexbotic_DM0/README.md`. https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/main/policy/Dexbotic_DM0/README.md. Fetched 2026-08-12 (unpinned `main` branch content).

[7] XPolicyLab GitHub repository and commit history (repository metadata, licence, star count, pinned-commit resolution, newest-push commit). https://github.com/XPolicyLab/XPolicyLab and https://api.github.com/repos/XPolicyLab/XPolicyLab/commits. Fetched 2026-08-12.

[8] XPolicyLab `LICENSE` file. https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/main/LICENSE. Fetched 2026-08-12 (unpinned `main` branch content; confirmed identical to the LICENSE file read at the pinned commit in [9]).

[9] XPolicyLab README.md, pyproject.toml, and LICENSE read at the shortlist's pinned commit `72b3a52f2b069220088ef824ade5890885858379`. https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/72b3a52f2b069220088ef824ade5890885858379/README.md, https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/72b3a52f2b069220088ef824ade5890885858379/pyproject.toml, and https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/72b3a52f2b069220088ef824ade5890885858379/LICENSE. Fetched 2026-08-12 (pinned-commit content, verified against the commit API response in [7]).
