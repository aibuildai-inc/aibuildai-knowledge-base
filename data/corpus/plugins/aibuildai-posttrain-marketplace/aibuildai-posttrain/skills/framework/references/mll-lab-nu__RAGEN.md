# RAGEN

A research codebase for training LLM agents with multi-turn RL in stochastic environments, built around verl - clone and pin a commit, there is no packaged release.

**RAGEN** ("Reasoning AGENT") is described in its own README as "a flexible RL framework for training reasoning agents" that also ships "diagnostics to understand *how* agent RL training works" [1]. It is built by a group at Northwestern University's mll-lab (Manling Li's lab) and collaborators, credited in the README as the RAGEN Team and a named contributor list of eighteen people [1]. Its API shape is a single Hydra-configured entrypoint, `train.py`, that reads a YAML config naming an environment and algorithm, builds a Ray-distributed PPO/GRPO loop on top of the verl library (vendored as a git submodule), and runs `trainer.fit()` [3][1]. It lives at https://github.com/mll-lab-nu/RAGEN [2].

**When to pick it**: pick RAGEN when the goal is reproducing or extending its own StarPO multi-turn-agent research - its ten built-in interactive environments (Sokoban, FrozenLake, WebShop, DeepCoder, SearchQA, Lean, Bandit, Countdown, MetaMathQA, Sudoku) and its reasoning-collapse diagnostics (mutual-information and conditional-entropy proxy metrics, SNR-Adaptive rollout filtering) are not offered by general post-training libraries [1]. It is not a general-purpose trainer: there is no PyPI package, no GitHub Release, and no git tag in this repository - confirmed by empty JSON arrays `[]` from both the `/releases` and `/tags` GitHub API endpoints [4][5] - so every install is a clone pinned to a commit, unlike trl or verl which ship versioned releases (cross-reference; not covered here). For general single- or multi-node PPO/GRPO training without RAGEN's specific environments, weigh the verl card instead.

**Methods it ships**: RAGEN's own algorithmic layer is **StarPO** (State-Thinking-Actions-Reward Policy Optimization), described in the README as "a unified RL framework for training multi-turn, trajectory-level agents", which supports two update rules: PPO with token-level advantage estimation via a value function, and GRPO with a trajectory-level normalized reward [1]. `config/base.yaml` sets `algorithm.adv_estimator: gae` for PPO or `grpo` for GRPO, and separately sets `algorithm.gamma: 1.0` and `algorithm.lam: 1.0` (the GAE discount and lambda) [6]. A third estimator, REMAX, is wired into the advantage-estimation branch but is explicitly unsupported at this commit: the trainer code calls `logger.log("[NotImplemented] REMAX implementation is not tested yet in RAGEN. Exiting.")` and exits when it is selected [7]. RAGEN-2 adds two training-time mechanisms on top of PPO/GRPO: reasoning-collapse diagnostics built from mutual-information and conditional-entropy proxy metrics [21], and SNR-Adaptive (rollout) Filtering, which discards low within-group reward-variance groups before the policy-gradient update [22]. A user reported on issue #128 that Sokoban and FrozenLake already compute process rewards but RAGEN's GRPO advantage implementation does not use them; the maintainer (COLLABORATOR ZihanWang314, 2025-11-10) replied that process reward with bi-level advantage estimation is instead supported in PPO by a separate follow-up project, VAGEN, and that "we currently do not support process reward with GRPO" in RAGEN itself [8]. All of this is read at the pinned commit `20daedc4`; there is no separate "taxonomy" page to recheck since there is no released version.

**Scale it handles**: single GPU up to multi-node via Ray, the distributed-execution library verl uses under the hood. `train.py`'s `TaskRunner` sizes a `ResourcePoolManager` from `config.trainer.n_gpus_per_node * config.trainer.nnodes`, and only the `'fsdp'` strategy is implemented for the actor and critic workers - selecting any other strategy raises `NotImplementedError` in `TaskRunner.run()` [3]. `base.yaml` defaults to `trainer.n_gpus_per_node: 1` and pins training to `system.CUDA_VISIBLE_DEVICES: "0"`, i.e. single-GPU unless both are raised together (`train.py`'s own config validator asserts `len(CUDA_VISIBLE_DEVICES.split(','))  == n_gpus_per_node`) [6][3]. No multi-node benchmark or example config was found in the files read for this card - the mechanism (Ray resource pools, `nnodes`) is documented, a published multi-node run is not.

**Install**: `git clone https://github.com/mll-lab-nu/RAGEN.git && cd RAGEN && conda create -n ragen python=3.12 -y && conda activate ragen && bash scripts/setup_ragen.sh` [1]. There is no PyPI package and no tagged/released version to pin - `setup.py` at the pinned commit declares only `version='0.1'`, and both the GitHub Releases and Tags API endpoints return empty lists for this repository [9][4][5]. Licence is MIT, copyright "RAGEN Team" [10]. Python floor is 3.12, set by the setup script's `conda create -n ragen python=3.12` [11]. `setup.py`'s `install_requires` names `vllm==0.8.2`, `tensordict>=0.8.0,<0.9.0`, `ray>=2.10`, and `faiss-cpu==1.11.0` among its base dependencies [9], but the install script runs `pip install -e . --no-deps` for RAGEN itself, so none of those pins are actually installed by that step; the deep-learning core (PyTorch, vLLM) instead comes from `verl`'s own `scripts/install_vllm_sglang_mcore.sh`, run inside the pinned verl v0.6.1 submodule immediately after [11]. The setup script separately pins `numpy==1.26.4` and reinstalls `"setuptools<70.0.0"` with a comment explaining that vllm may upgrade setuptools, breaking `pkg_resources` for `gym_sokoban` [11]. `.gitmodules` pins `verl` to `volcengine/verl.git` (no tag is stated in `.gitmodules` itself; the version is asserted as v0.6.1 by the setup script's own step label "Installing verl dependencies for v0.6.1" [11][12]). Hardware: the setup script's header comment states the environment is "Verified on NVIDIA H100, H200, and B200" - no other CUDA/driver minimum is stated in the files read for this card [11].

**Maintained by**: the RAGEN Team / mll-lab-nu on GitHub, led by contributors including Zihan Wang, Kangrui Wang, and Manling Li [1]. The repository's most recent push at the time of this card was 2026-07-24, about three months after the commit this card is pinned to (2026-04-14) [13][14]. Two papers document the project: the original RAGEN paper (arXiv 2504.20073, dated 2025-04-24 by arXiv's own citation metadata) and RAGEN-2 (arXiv 2604.06268, dated 2026-04-07) [15][16]. The README's own news log dates the RAGEN-2 release itself to 2026-03-12 [1].

## Quick start

From the README's Getting Started section [1]:

```bash
git clone https://github.com/mll-lab-nu/RAGEN.git
cd RAGEN
conda create -n ragen python=3.12 -y && conda activate ragen
bash scripts/setup_ragen.sh
```

Train (no filter, default), on the built-in Sokoban config:

```bash
python train.py --config-name _2_sokoban
```

Train with SNR-Adaptive Filtering (RAGEN-2, Top-p):

```bash
python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout_filter_strategy=top_p \
  actor_rollout_ref.rollout.rollout_filter_value=0.9
```

Evaluate:

```bash
python -m ragen.llm_agent.agent_proxy --config-name _2_sokoban
```

All four commands are quoted from the README as-is [1].

## Start it

- One process, one GPU is the default: `base.yaml` sets `trainer.n_gpus_per_node: 1` and `system.CUDA_VISIBLE_DEVICES: "0"`, so the quick-start commands above already run on a single GPU [6].
- More than one GPU is reached by raising both `trainer.n_gpus_per_node` and the matching `system.CUDA_VISIBLE_DEVICES` list together - `train.py`'s own config validator asserts they have equal length, and Ray's `ResourcePoolManager` in `TaskRunner.run()` sizes its pool as `n_gpus_per_node * nnodes` [3]. No ready-made multi-GPU launcher script or config template beyond the per-environment configs (`config/_2_sokoban.yaml`, etc., each a thin override of `base.yaml`) was found among the files read for this card [17][6].
- Effective batch size: `train.py`'s `add_dependency_and_validate_config()` sets `config.data.train_batch_size = es_manager.train.env_groups * es_manager.train.group_size` and asserts `ppo_mini_batch_size` is divisible by `actor.ppo_micro_batch_size_per_gpu * n_gpus_per_node` - scaling GPU count means retuning `ppo_micro_batch_size_per_gpu` or `ppo_mini_batch_size` to keep that divisibility [3]. `base.yaml` defaults: `es_manager.train.env_groups: 8`, `group_size: 16` (128 rollouts/step), `ppo_mini_batch_size: 32`, `micro_batch_size_per_gpu: 1` [6].
- Generation is a separate layout choice inside the same process: `actor_rollout_ref.rollout.name: vllm` with `gpu_memory_utilization: 0.3` (the vLLM engine's share of each training GPU, since rollout and training colocate on the same card by default) and `tensor_model_parallel_size: 1` [6].
- Config surface: everything above lives in Hydra YAML under `config/`, composed from `base.yaml` plus per-environment overrides (e.g. `config/_2_sokoban.yaml` adds only `trainer.experiment_name: sokoban-main` and a Hydra search path) [6][17]. `base.yaml`'s own `defaults:` list names `ppo_trainer` and `envs`; `train.py`'s `__main__` block injects two extra Hydra `--config-dir` search paths at startup, one of which points at `verl/verl/trainer/config` inside the vendored verl submodule - so `ppo_trainer.yaml`'s defaults are supplied by verl, not by a file in RAGEN's own `config/` directory [3]. `base.yaml` itself sets `actor_rollout_ref.rollout.name: vllm` and adds the `rollout_filter_*` filtering knobs on top of whatever verl's own `ppo_trainer.yaml` defaults are - this card did not fetch verl's `ppo_trainer.yaml` at the pinned submodule commit, so it does not claim which of these RAGEN sets to a non-default value versus which verl already defaults to [6].
- Out-of-memory first aid is documented on the evaluation-side rollout config, not a dedicated training OOM section: `docs/eval.md`'s Troubleshooting block recommends lowering `actor_rollout_ref.rollout.max_model_len`, `response_length`, and `gpu_memory_utilization` [18]. The same fields exist under the training config (`base.yaml`'s `actor_rollout_ref.rollout.max_model_len: 3600`, `response_length: 400`) and are the load-bearing generation-memory knobs during training too [6].

## Watch it

This section is the mechanics only - what a metric means for judging a run (healthy reward-variance shapes, entropy trends) is method-level and not restated here.

- **Enable it**: `trainer.logger` in `base.yaml` defaults to `['console', 'wandb']`, so a default run already logs to Weights & Biases as well as stdout; no separate flag is needed to turn W&B on [6]. Sample generations are logged via a `GenerationsLogger` instance whose `.log()` call is gated by `trainer.generations_to_log_to_wandb.val` (default `20`) [7]. A maintainer reply (COLLABORATOR ZihanWang314, 2025-05-06, issue #84) directs users to find generated validation trajectories "in wandb, under the val/generations metric" [19].
- **Standard training metrics**: the files read for this card do not include a page enumerating every verl-inherited PPO/GRPO scalar (e.g. reward, KL, entropy) by name; RAGEN delegates that logging to `verl.utils.tracking.Tracking`, called once per step in `RayAgentTrainer.fit()` [7]. A maintainer reply (COLLABORATOR ZihanWang314, 2025-08-18, issue #121) names one environment-specific validation metric directly: `val-env/WebShop/success_purchase`, and reports the average reward with StarPO-S + PPO on WebShop as "around 0.53" - a maintainer-supplied figure the same reply says will be updated "in the upcoming revisions", not a fixed number from the published paper [20]. This provisional issue reply is the only quantitative reference figure found for this card: `docs/experiment_main_table.md`, the README-linked reproduction doc for RAGEN's own main results table, was read in full (122 lines) and documents only the three shell scripts that reproduce the paper's algorithm/model-size/model-type comparisons - their CLI flags, output log paths, and the `filter`/`nofilter` config mapping - with no numeric results table anywhere in the file [23].
- **Collapse-diagnostic metrics** (RAGEN-2-specific, fully enumerated in `docs/reference_mutual_information_metrics.md` Section 9 [21]): namespaced under `collapse_first_turn_sample/<suffix>` and `collapse_trajectory_sample/<suffix>` (a third namespace, `collapse_turn_sample/`, exists in code but is disabled by default), each carrying suffixes such as `mi_estimate`, `mi_upper_bound`, `conditional_entropy_est`, `reasoning_entropy_est`, `retrieval_accuracy` (with `@2`/`@4`/`@8` variants), `retrieval_above_chance`, and variance-normalized forms (`mi_zscore`, `marginal_std_ema`); plus directly-logged coverage/timing keys `collapse/valid_thinking_rate`, `collapse/first_turn_num_total`, `collapse/first_turn_num_valid`, `collapse/first_turn_valid_rate`, `timing_s/collapse_multi_turn_step`, `timing_s/collapse_first_turn_step` [21]. These are computed at the cadence set by `collapse_detection.compute_freq` (default `5` steps) [6].
- **Evaluate during training**: `trainer.test_freq` (default `10`) and `trainer.val_before_train` (default `True`) in `base.yaml` control in-loop validation cadence; separately, `python -m ragen.llm_agent.agent_proxy --config-name <name>` runs a standalone evaluation pass that prints per-environment metrics to the terminal, documented examples being `{env}/success`, `{env}/num_actions`, `{env}/pass@k`, and `episodic_return` [18][6].
- **Rollout filtering has its own emitted signal**: `rollout_filter_metric` (default `reward_variance`) is computed per group each step, and `rollout_filter_empty_stop_steps` (default `5`) triggers an early stop after that many consecutive steps filter every group out - `docs/guide_rollout_filtering.md` names this state `empty_after_filter` [22].
- **Stopping**: no RL-specific early-stopping threshold or patience value was found in `docs/guide_rollout_filtering.md`, `docs/eval.md`, or `config/base.yaml` - the only stop-like field found is `rollout_filter_empty_stop_steps`, which halts training when filtering has emptied every group for that many consecutive steps, not a reward- or loss-based stopping rule [22][6].

## Save it

- Checkpoints land under `{config.trainer.default_local_dir}/global_step_{N}/`, with `actor/` and (when a critic is used, i.e. PPO) `critic/` subdirectories, written by `RayAgentTrainer._save_checkpoint()` via calls to `self.actor_rollout_wg.save_checkpoint(...)` and `self.critic_wg.save_checkpoint(...)` on the Ray worker groups - the actual weight-file writing is delegated to verl's checkpoint manager, not reimplemented in RAGEN [7].
- A plain-text tracker file, `latest_checkpointed_iteration.txt`, is written at `{default_local_dir}/latest_checkpointed_iteration.txt` containing the most recent `global_steps` integer, used to find the latest checkpoint on resume [7].
- Retention: `trainer.max_actor_ckpt_to_keep` and `trainer.max_critic_ckpt_to_keep` both default to `1` in `base.yaml`, i.e. only the single most recent actor/critic checkpoint is kept unless raised [6]. The files read for this card do not show a RAGEN-level flag equivalent to trl's `save_only_model` that drops optimizer state on save; checkpoint retention/format is inherited from verl's `save_checkpoint` call and was not independently re-verified here.
- Cadence: `trainer.save_freq` defaults to `100` steps in `base.yaml`; `RayAgentTrainer.fit()` calls `_save_checkpoint()` when `save_freq > 0` and the step is a multiple of it, or on the last step [7][6].
- Resume: `RayAgentTrainer.fit()` calls `self._load_checkpoint()` before the training loop begins; the mechanism reads `latest_checkpointed_iteration.txt` under `default_local_dir` to locate the checkpoint to resume from (the exact resume-path override flag, if any, was not confirmed in the code read for this card) [7].
- Evaluation output is a separate artifact from a training checkpoint: `python -m ragen.llm_agent.agent_proxy --config-name eval` writes either a pickled `DataProto` object (`output.format: pkl`, the default) or an OpenAI-message-format JSONL file (`output.format: jsonl`), each trajectory line carrying `custom_id`, `messages`, and a `metadata` block with `success`, `total_reward`, `num_turns`, `entropy`, and `n_tokens` [18]. `scripts/convert_to_jsonl.py` converts an existing `.pkl` to `.jsonl` after the fact [18].
- Whether an evaluator can load a saved `global_step_N/actor/` directory directly depends on verl's checkpoint format, since RAGEN delegates the write to verl's `save_checkpoint`; this card does not independently confirm the loader contract - read this skill's shared loading-the-result reference, and treat verl's own checkpoint-format documentation as authoritative before writing a loader against a RAGEN checkpoint.

## Find it in the docs

There is no versioned or hosted docs site current for this commit - the README itself says the readthedocs build "is now outdated" [1] - so "the docs" here means the repository's own `docs/` folder and README, read at the pinned commit, plus GitHub's code-search UI over the same tree.

- Item home: https://github.com/mll-lab-nu/RAGEN [2].
- Raw-file address pattern for anything in `docs/` or `config/`: `https://raw.githubusercontent.com/mll-lab-nu/RAGEN/<commit-or-branch>/<path>` - every source cited on this card was fetched this way at the pinned commit `20daedc4` [1][21][18][22].
- The README's own Documentation section is the closest thing to a page-slug map, listing: `docs/eval.md` (evaluation and output formats), `docs/guide_rollout_filtering.md` (SNR-Adaptive filtering), `docs/reference_mutual_information_metrics.md` (collapse-diagnostic metric reference), and environment-config guidance pointing at `config/envs.yaml` directly rather than a docs page [1]. That same Documentation section also links per-experiment reproduction notes (`docs/experiment_main_table.md`, `docs/experiment_intervention_sweep.md`, `docs/experiment_frozen_lake_slipper_sweep.md`, `docs/experiment_sokoban_gradient_analysis.md`, `docs/experiment_search.md`, `docs/experiment_deepcoder.md`, `docs/experiment_webshop_release.md`) [1]. Separately, a legacy `docs/readme_v1.md` for the pre-RAGEN-2 README is linked from an intro blockquote near the top of the README, outside the Documentation section [1].
- Runnable references beyond the docs: a directory listing of `config/` at the pinned commit shows 19 files - one Hydra config per built-in environment (`_1_bandit.yaml` through `_10_rubikscube.yaml`), plus `base.yaml`, `envs.yaml`, `webshop_full.yaml`, `eval.yaml`, `eval_webshop.yaml`, `evaluate_api_llm.yaml`, and `stream.yaml` [24]; `scripts/runs/` holds shell scripts that reproduce specific paper experiments (entropy sweeps, the filtering ablation, per-environment main-table runs) [25]. `config/envs.yaml` at the pinned commit lists 23 `custom_envs` entries (including `Alfworld`, `AlfworldOOD`, `game_2048`, and `rubikscube`, none of which the README's ten-environment list names), so the environment count actually configured in this file exceeds what the README's prose states - read narrowly as a fact about this one file at this one commit, not a claim about which of the 23 tags are genuinely new environments versus variants [26][1].
- Community layer: none is curated by the project itself - the README's "Awesome Work Powered or Inspired by RAGEN" section links downstream projects (ROLL, VAGEN, Search-R1, ZeroSearch, Agent-R1, OpenManus-RL, MetaSpatial, s3) rather than tutorials about using RAGEN itself, so there is no equivalent to trl's community-tutorials page to point at [1].
- No official MCP endpoint for these docs was found in the files read for this card.
- Traps found only in closed-issue maintainer replies: issue #128 (process reward unsupported in GRPO, quoted above) [8]; issue #84 (validation trajectories live in the `val/generations` W&B metric, quoted above) [19]; issue #121 (the WebShop success metric name and the maintainer's provisional ~0.53 average-reward figure, quoted above) [20]; issue #132, where COLLABORATOR ZihanWang314 replied on 2025-09-09 that a "loss mask and response mask" warning some users hit did not reproduce on the `main` branch, and the reporter confirmed on 2025-09-10 that using the latest `main` resolved it [27].

## Sources

All files were fetched at the pinned commit `20daedc47558e000f7de912b060646bf2e8026bd` on 2026-08-11 unless a GitHub API endpoint or issue thread is named, which reflects live repository state at the same fetch date. Ray, verl, vLLM, and Hydra are named only as the infrastructure RAGEN's own files describe and are not separately cited.

[1] RAGEN README.md, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/README.md. Fetched 2026-08-11.

[2] mll-lab-nu/RAGEN GitHub repository. https://github.com/mll-lab-nu/RAGEN. Fetched 2026-08-11.

[3] RAGEN train.py, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/train.py. Fetched 2026-08-11.

[4] GitHub API, mll-lab-nu/RAGEN releases (empty list). https://api.github.com/repos/mll-lab-nu/RAGEN/releases. Fetched 2026-08-11.

[5] GitHub API, mll-lab-nu/RAGEN tags (empty list). https://api.github.com/repos/mll-lab-nu/RAGEN/tags. Fetched 2026-08-11.

[6] RAGEN config/base.yaml, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/config/base.yaml. Fetched 2026-08-11.

[7] RAGEN ragen/trainer/agent_trainer.py, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/ragen/trainer/agent_trainer.py. Fetched 2026-08-11.

[8] mll-lab-nu/RAGEN issue #128, "How to use process reward in GRPO" (closed), maintainer reply. https://github.com/mll-lab-nu/RAGEN/issues/128. Fetched 2026-08-11.

[9] RAGEN setup.py, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/setup.py. Fetched 2026-08-11.

[10] RAGEN LICENSE, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/LICENSE. Fetched 2026-08-11.

[11] RAGEN scripts/setup_ragen.sh, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/scripts/setup_ragen.sh. Fetched 2026-08-11.

[12] RAGEN .gitmodules, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/.gitmodules. Fetched 2026-08-11.

[13] GitHub API, mll-lab-nu/RAGEN repository metadata (pushed_at, stargazers_count, license, description). https://api.github.com/repos/mll-lab-nu/RAGEN. Fetched 2026-08-11.

[14] GitHub API, mll-lab-nu/RAGEN commit 20daedc4 metadata (author, commit date). https://api.github.com/repos/mll-lab-nu/RAGEN/commits/20daedc47558e000f7de912b060646bf2e8026bd. Fetched 2026-08-11.

[15] arXiv abstract page, 2504.20073, "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning". https://arxiv.org/abs/2504.20073. Fetched 2026-08-11.

[16] arXiv abstract page, 2604.06268, "RAGEN-2: Reasoning Collapse in Agentic RL". https://arxiv.org/abs/2604.06268. Fetched 2026-08-11.

[17] RAGEN config/_2_sokoban.yaml, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/config/_2_sokoban.yaml. Fetched 2026-08-11.

[18] RAGEN docs/eval.md, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/docs/eval.md. Fetched 2026-08-11.

[19] mll-lab-nu/RAGEN issue #84, "How to save generated trajectories (messages) in each validation round?" (closed), maintainer reply. https://github.com/mll-lab-nu/RAGEN/issues/84. Fetched 2026-08-11.

[20] mll-lab-nu/RAGEN issue #121, "webshop parameters" (closed), maintainer reply. https://github.com/mll-lab-nu/RAGEN/issues/121. Fetched 2026-08-11.

[21] RAGEN docs/reference_mutual_information_metrics.md, Section 9, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/docs/reference_mutual_information_metrics.md. Fetched 2026-08-11.

[22] RAGEN docs/guide_rollout_filtering.md, at commit 20daedc4. https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/docs/guide_rollout_filtering.md. Fetched 2026-08-11.

[23] RAGEN docs/experiment_main_table.md, at commit 20daedc4 (122 lines, read in full: reproduction-script documentation, no numeric results table). https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/docs/experiment_main_table.md. Fetched 2026-08-11.

[24] GitHub API, mll-lab-nu/RAGEN directory listing for config/, at commit 20daedc4 (19 files). https://api.github.com/repos/mll-lab-nu/RAGEN/contents/config?ref=20daedc47558e000f7de912b060646bf2e8026bd. Fetched 2026-08-11.

[25] GitHub API, mll-lab-nu/RAGEN tree listing for scripts/runs/, at commit 20daedc4. https://api.github.com/repos/mll-lab-nu/RAGEN/contents/scripts/runs?ref=20daedc47558e000f7de912b060646bf2e8026bd. Fetched 2026-08-11.

[26] RAGEN config/envs.yaml, at commit 20daedc4 (top-level `custom_envs` keys). https://raw.githubusercontent.com/mll-lab-nu/RAGEN/20daedc47558e000f7de912b060646bf2e8026bd/config/envs.yaml. Fetched 2026-08-11.

[27] mll-lab-nu/RAGEN issue #132, "warning about loss mask and response mask" (closed), maintainer reply and reporter confirmation. https://github.com/mll-lab-nu/RAGEN/issues/132. Fetched 2026-08-11.
