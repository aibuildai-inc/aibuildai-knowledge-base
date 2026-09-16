# Agent-R1

A research framework layered on top of verl for training multi-step, tool-using LLM agents with RL, built around a step-level (not token-level) trajectory representation - clone-and-run, no pip package.

Agent-R1 is "an open-source framework for training powerful language agents with end-to-end reinforcement learning," letting a user "build custom agent workflows, define interactive environments and tools, and train multi-step agents in a unified RL pipeline" [1]. It is built by a team at the State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China (USTC), and is described in the technical report "Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning" [2][1]. Its API surface is a Hydra-configured launch script per task and algorithm (`examples/<task>/run_<algo>.sh`) that runs `python -m agent_r1.trainer.main_agent_ppo`, itself a thin wrapper that plugs an `AgentFlow`/environment pair into verl's Ray-based `RayPPOTrainer` [3][4]. It lives at https://github.com/AgentR1/Agent-R1 [28].

**When to pick it**: pick Agent-R1 when the task is genuinely multi-step (tool calls, environment feedback, multiple turns) and you want the environment/tool logic factored out from the RL trainer into reusable `AgentEnv`/`BaseTool` layers, and when you are already prepared to pin a specific `verl` version - the current version requires `verl==0.7.0` exactly, not a range [3][5]. Do not pick it for single-turn SFT/DPO-style post-training (it has no such trainer), and note that there is no PyPI package and no GitHub Release object (only a single git tag, `v0.1.0`, which the README calls "the first official release of the refactored architecture," at a commit earlier than this card's pinned commit) [3][6][7].

**Methods it ships**: Agent-R1 does not add new trainer classes; it adds an agent/environment layer on top of verl's existing PPO-family trainer, and each RL algorithm is selected through Hydra config overrides in a per-task launch script, not through a distinct Python trainer class [3][8]. The repository's own tutorial page lists seven algorithm scripts: PPO (`run_ppo.sh`, actor-critic baseline), GRPO (`run_grpo.sh`, used as the common script base for several variants), StepPO (`run_steppo.sh`, the framework's own step-aligned policy optimization, arXiv 2604.18401), RLOO (`run_rloo.sh`, leave-one-out baseline), REINFORCE (`run_reinforce.sh`, described as "step-level critic-free"), GSPO (`run_gspo.sh`) and GiGPO (`run_gigpo.sh`), the last two both marked "GRPO-family algorithm variant exposed through script overrides" [8]. Not every task ships every script; GSM8K intentionally ships only the StepPO single-turn and tool-calling sanity-check scripts [8]. Separately, `agent_r1/reward_loop/reward_loop.py` defines an async Ray-actor reward-computation adapter (`RewardLoopWorker`) that wraps verl's reward-manager registry to support rule-based, discriminative-reward-model (disrm), and generative-reward-model (genrm, which must go through a user-supplied custom reward function) scoring paths - this is infrastructure for computing the reward signal, not itself a trainable RL method [4]. A newer feature, Online Policy Distillation (OPD), was announced 2026-07-21 but lives only on the separate `opd` branch, not on `main` at the pinned commit [3]. The method taxonomy is scattered across the README and the `tutorials/recipes-and-algorithms` docs page rather than one page - check both if you need the current list [3][8].

**Scale it handles**: single GPU up to multi-node, entirely through the launch configuration Agent-R1 inherits from verl - the same Ray resource-pool and worker-per-GPU pattern, set via `trainer.n_gpus_per_node` and `trainer.nnodes` in the Hydra overrides, with no Agent-R1-specific launcher of its own [9][4]. Generation is a separate async engine layer: `actor_rollout_ref.rollout.mode=async` with `agent.num_workers` (default 8) sizing the AgentFlow worker pool, and `agent.max_steps` (default 10) capping the multi-turn rollout length per trajectory [9]. The docs' own results page shows RL training clearing a non-trivial bar: on GSM8K, the untrained ReAct baseline scores 53.1% versus 83.3% for GRPO, and on ALFWorld-seen success rate ReAct scores 7.14% versus 81.29% for GRPO, with the page stating "all four RL methods outperform the training-free ReAct baseline across these settings" [10]. No multi-node benchmark for Agent-R1's own agent workloads was found in the README, docs pages, or config files read for this card; a maintainer's issue reply gives a single-GPU anecdote instead, that a 0.5B model trains on one 40GB A100 [10][11].

**Install**: there is no `pip install agent-r1` and no GitHub Release object - `pyproject.toml` at the pinned commit contains only `[tool.ruff]` and `[tool.mypy]` linting/typechecking sections, no `[project]` table or version string [6]. A single git tag, `v0.1.0`, exists at commit `bc406f2313ee1c3cda7afcf3185848999c83b40c`, which the README's own changelog dated 2026.03.23 calls "the first official release of the refactored architecture" - that tagged commit is earlier than, and different from, the commit b124aa46... this card pins, so claims sourced from files at the pinned commit are ahead of that release and are not covered by the tag [3][7]. The README's own present-tense setup instruction is: "You only need to clone this repository; there is no separate Agent-R1 installation step" - set up the same environment `verl` itself documents, and "the current version requires `verl==0.7.0`" [3]. `verl` is not currently vendored as a git submodule: no `verl` directory or `.gitmodules` file exists in the repository tree at the pinned commit, and the README's only submodule language is a 2025.03.18 changelog entry ("`verl` is moved to a git submodule") that predates the pinned commit by over a year; a closed issue from that earlier period (#25, closed 2025-04-11) shows the setup docs at the time omitted `git submodule update --init --recursive`, since fixed, but this no longer describes how `verl` is obtained at the pinned commit [12][3]. No Python-version floor or CUDA/hardware minimum is stated anywhere in the README or docs read for this card; the repository's own license is MIT [3][13].

**Maintained by**: a team at USTC's State Key Laboratory of Cognitive Intelligence, with `0russwest0` (the repository's former name/account) active as a collaborator answering issues [3][11]. Signs of life: 1,601 GitHub stars as of this reading (not a ranking signal) [13], and the repository's own changelog shows pushes as recent as 2026-07-21 (OPD announcement) and 2026-05-30 (a revised technical report) [3].

## Quick start

Both quick-start paths are explicitly framed by the README as sanity checks of the training stack, not full benchmark runs [3]. Stage 1, a single-turn check:

```bash
python3 -m recipes.gsm8k.data_preprocess.process_gsm8k --local_save_dir ~/data/gsm8k
bash examples/gsm8k/run_steppo.sh
```

Stage 2, the minimal multi-turn tool-calling example (`ToolEnv` + `BaseTool`, with the recipe-local `calc_gsm8k_reward` tool):

```bash
python3 -m recipes.gsm8k.data_preprocess.process_gsm8k_tool --local_save_dir ~/data/gsm8k_tool
bash examples/gsm8k/run_steppo_tool.sh
```

Both scripts are complete Hydra-driven launchers, not pseudo-code: `run_steppo.sh` sets `BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'`, `algorithm.adv_estimator=gae`, and runs on 2 GPUs; `run_steppo_tool.sh` uses `Qwen/Qwen3-4B-Instruct-2507`, adds `actor_rollout_ref.rollout.agent.agent_flow_config_path=...` and `default_agent_flow=gsm8k_tool`, `max_steps=5`, with smaller batch sizes than the single-step script [14][15]. Processed data for these and other recipes is published on ModelScope (`Melmaphother/Agent-R1-data`), pulled with either `pip install modelscope` + the ModelScope CLI, or `git clone https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data.git` [3].

## Start it

- One GPU: run any `examples/<task>/run_<algo>.sh` script as-is with GPU count set to 1 in the script's env vars, per the pattern a maintainer confirmed can train a 0.5B model on a single 40GB A100, with `VLLM_ATTENTION_BACKEND=XFORMERS` and `CUDA_VISIBLE_DEVICES=0` [11].
- Multiple GPUs/nodes: the same scripts set `trainer.n_gpus_per_node` and `trainer.nnodes` as Hydra overrides, inherited unchanged from verl's Ray resource-pool mechanism; a closed issue shows a wrong `n_gpus_per_node` value (larger than the machine's GPU count) as the source of a Ray "no available node types" autoscaler error, fixed by lowering it to match the box [4][11]. There is no Agent-R1-specific config template beyond the per-recipe `examples/<task>/run_<algo>.sh` scripts themselves [8].
- Generation-layout knobs live under the `agent` block of the rollout config (`agent_r1/config/config.py`'s `AgentFlowConfig`, merged into verl's rollout config): `agent.num_workers` (default 8, the AgentFlow worker pool size), `agent.max_steps` (default 10, the per-trajectory turn cap), `agent.default_agent_flow` (default `"single_step_agent"`), and `agent.agent_flow_config_path` for pointing at a task-specific flow config [9]. The class explicitly also accepts verl's legacy `agent_loop`-named fields for backward compatibility, but ignores them [9].
- Effective batch size follows the standard PPO-family arithmetic (per-device batch x GPUs x gradient-accumulation), configured through the same `data.train_batch_size` / `actor_rollout_ref.actor.ppo_mini_batch_size`-style Hydra keys verl uses; Agent-R1 does not change this arithmetic, only adds the per-step multi-turn dimension on top via `trajectory_uids` grouping [4].
- Out-of-memory first aid: a closed bug report shows sequences overflowing `max_length` even with `filter_overlong_prompts` and `truncation="left"` set; the maintainer's diagnosis was that this was a bug in the pinned `verl` version, already fixed upstream, and at the time (2025-03-29) recommended reinstalling `verl` from the repository's own submodule checkout rather than a separate pip install to avoid version mismatches [16]. That submodule-checkout advice predates the pinned commit's `verl` setup by over a year and does not apply here - at the pinned commit there is no submodule to check out; instead, follow verl's own installation guide and confirm the environment ends up with `verl==0.7.0`, per the Install field above [5]. The durable takeaway from the issue is simply to match your installed `verl` version against the pinned floor if you hit this overflow. A separate closed bug (fixed via a merged PR) shows the tool-calling rollout comparing prompt length against `max_response_length` instead of `max_prompt_length` when deciding whether a prompt is too long to run, in `agent_r1/llm_agent/generation.py` - check this comparison if prompts are being dropped unexpectedly on an older checkout [17].

## Watch it

This section covers the logging mechanics only - what a given metric means for a specific algorithm (GRPO, PPO, RLOO, ...) is on that method's own card, not here.

- **Enable it**: logging backends are set via `trainer.logger` in the Hydra config, consumed by verl's own `Tracking` utility inside Agent-R1's `RayAgentTrainer.fit()`; the same trainer logs a table of validation-sample generations to "wandb or swanlab" per its own source comment [4]. Agent-R1 does not add a separate logging entry point of its own; sample-level validation logging and metric logging both run through this same verl-inherited path [4].
- **Metric names, trajectory level (Agent-R1's own addition)**: `agent_r1/trainer/ppo/metric_utils.py` wraps verl's per-sequence `compute_data_metrics` and, when a batch carries `trajectory_uids` (i.e., multi-step rollouts), adds trajectory-level aggregates by summing each step's tensors before computing mean/max/min: `prompt_length/trajectory/{mean,max,min}`, `response_length/trajectory/{mean,max,min}`, `num_steps/{mean,max,min}`, `critic/score/trajectory/{mean,max,min}`, `critic/rewards/trajectory/{mean,max,min}` [18]. The underlying per-sequence (per-step) metrics these are built on come from verl's own `compute_data_metrics` and are not restated here - see verl's own metric documentation for that list [18].
- **Sample-level generations**: validation-time samples are logged as a table via the `validation_generations_logger`, to whichever backend `trainer.logger` names [4].
- **Evaluation during training**: cadence is `trainer.test_freq`, inherited unchanged from verl - the trainer runs validation when `is_last_step or self.global_steps % self.config.trainer.test_freq == 0` [4].
- **A specific past crash and its apparent fix**: a closed issue reported that validation with the file logging backend crashed with `TypeError: Object of type float32 is not JSON serializable` on `verl==0.7.0`, attributing the root cause to `verl`'s validation-metrics path rather than Agent-R1's own code [19]. The `RayAgentTrainer.fit()` source at the pinned commit wraps both validation and training metrics in a `make_json_safe()` call before logging, consistent with that failure mode being handled, though this card cannot confirm from the issue thread alone whether that fix landed before or after the report [4][19].
- **Stopping**: no RL-specific stopping rule, threshold, or patience value is published in the README, the `getting-started`, `core-concepts`, or `tutorials` docs pages, or the config files read for this card; training runs for a fixed `trainer.total_epochs`/step budget inherited from verl, with save/eval cadence as the only tunable stopping-adjacent knobs found [3][8][9].

## Save it

- Checkpoint save/load behavior is inherited unchanged from verl's `RayPPOTrainer`: `RayAgentTrainer` (in `agent_r1/trainer/ppo/ray_trainer.py`) calls `self._save_checkpoint()` and `self._load_checkpoint()` without overriding either method in this file, and cadence is `trainer.save_freq`, checked as `is_last_step or self.global_steps % self.config.trainer.save_freq == 0` [4].
- What a save keeps is controlled by `agent_r1/config/config.py`'s `CheckpointConfig`, a dataclass mirroring verl's own: `save_contents` defaults to `["model", "optimizer", "extra"]`, and the class's own docstring marks this as "what to include in saved checkpoints" - dropping `"optimizer"` from that list reduces checkpoint size but removes the ability to resume training state, since `load_contents` defaults to the same list as `save_contents` [20].
- Agent-R1 inherits the checkpoint contract unchanged from verl's `RayPPOTrainer`, since `RayAgentTrainer` calls but does not override `_save_checkpoint`/`_load_checkpoint` [4]. In verl v0.7.0's `ray_trainer.py`, `_save_checkpoint` writes under `trainer.default_local_dir` as `global_step_{N}/actor/` (the weights) and, when a critic role is used, a sibling `global_step_{N}/<critic-role-name>/`, plus a `global_step_{N}/data.pt` (dataloader state); it also writes a top-level `latest_checkpointed_iteration.txt` marker recording `N`, skipped when `async_save=True` [27].
- Resume is controlled by `trainer.resume_mode`: `"disable"` skips loading; `"auto"` resumes from the folder `find_latest_ckpt_path` finds under `default_local_dir`, or trains from scratch if none exists; `"resume_path"` requires `trainer.resume_from_path` to be a path string containing the substring `"global_step_"`, from which `_load_checkpoint` parses the step number and reloads `actor/`, the critic subfolder (if present), and `data.pt` from that exact folder [27].
- Whether an evaluator can load a saved checkpoint directly follows from this same layout: the weights live under `global_step_{N}/actor/`, so a loader that knows verl's actor-checkpoint format can load that subfolder directly; this card does not restate verl's actor-checkpoint file format itself, which was not read for this card.

## Find it in the docs

The docs are the live source; this section teaches the lookup, not the content.

- Address pattern: `https://agentr1.github.io/Agent-R1/<page>/` (capital "Agent-R1", matching `mkdocs.yml`'s own `site_url`) - checked 2026-08-12, `getting-started/`, `core-concepts/step-level-mdp/`, and `experiments/` all return HTTP 200 [21][22][23]. The README's own "Documentation" links instead point to `https://agentr1.github.io/agent-r1` (lowercase), which is a DIFFERENT deployed site with different page titles and content from the mkdocs-material build described by `mkdocs.yml` - fetched 2026-08-12, the lowercase root's title is "Agent-R1 — A Unified Framework for Agentic RL" versus the capitalized root's "Agent-R1", and the lowercase Chinese docs page includes example script paths (e.g., `examples/run_qwen3-4b_gsm8k_tool.sh`) that do not match any path in the repository tree at the pinned commit [24][25][3]. Use the capitalized URL; it is the one built from this repository's own `docs/` folder [21][26].
- `docs/experiments.md` (the page carrying the results table and the two ablations below) is not linked from `mkdocs.yml`'s `nav:`, so it will not appear in the site's sidebar, but the page still builds and is reachable directly at `https://agentr1.github.io/Agent-R1/experiments/` [26][23].
- Page-slug recipes, all under `getting-started/`, `core-concepts/`, `tutorials/`: `index`, `installation-guide`, `quick-start` under `getting-started/`; `step-level-mdp`, `layered-abstractions` under `core-concepts/`; `agent-task`, `recipes-and-algorithms` under `tutorials/` [26]. A Chinese mirror exists at the same slugs under a `zh/` prefix [26].
- Question-to-slug map: "what algorithms does it support and where are the scripts" -> `tutorials/recipes-and-algorithms` [8]; "what is a step in Agent-R1's MDP" -> `core-concepts/step-level-mdp` [1]; "which abstraction layer should I use for my task" -> `core-concepts/layered-abstractions` [1]; "how do I run the smallest example" -> `getting-started/quick-start` [3]; "what verl version do I need" -> `getting-started/installation-guide` [5].
- Runnable references beyond the docs: the `examples/<task>/run_<algo>.sh` scripts and the `recipes/<task>/` tree in the GitHub repository itself; ModelScope's `Melmaphother/Agent-R1-data` dataset for known-good smoke-test data across GSM8K, HotpotQA, ALFWorld, and WebShop [8][3].
- Community layer: the README's own "Awesome Projects Using Agent-R1" section is the closest thing to a curated door - it lists TableMind, PaperScout, Cast-R1, and StepPO as downstream projects built on Agent-R1, each linking to its own arXiv abstract page, but none are tutorials or blog posts about using the framework itself [3]. No separate community-tutorials page or official MCP endpoint was found for this project.
- Honest boundary: Agent-R1 has no SFT/DPO/offline trainer of its own (it is RL-only, layered on verl's online PPO-family trainer) [3][8]; it has no pip package and no GitHub Release object, so version pinning means pinning a git commit of this repo together with a `verl==0.7.0` install, per the README's present-tense setup instructions [6][5][3]; and it publishes no multi-node benchmark for its own agent workloads, only a single-A100 anecdote from a maintainer's issue reply [11].

## Sources

[1] Agent-R1 docs index (definition, Step-level MDP and Layered Abstractions summaries). https://agentr1.github.io/Agent-R1/. Fetched 2026-08-12.

[2] Agent-R1 technical report abstract. https://arxiv.org/abs/2511.14460. Cited via the README's own link and citation block; not separately fetched. Fetched 2026-08-12 (README).

[3] Agent-R1 README.md at commit b124aa46534cbf2fb8bc8af11405774984c42ac7. https://raw.githubusercontent.com/AgentR1/Agent-R1/b124aa46534cbf2fb8bc8af11405774984c42ac7/README.md. Fetched 2026-08-12.

[4] Agent-R1 `agent_r1/trainer/ppo/ray_trainer.py` at commit b124aa46534cbf2fb8bc8af11405774984c42ac7. https://raw.githubusercontent.com/AgentR1/Agent-R1/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/trainer/ppo/ray_trainer.py. Fetched 2026-08-12.

[5] Agent-R1 docs, Installation Guide. https://agentr1.github.io/Agent-R1/getting-started/installation-guide/. Fetched 2026-08-12.

[6] Agent-R1 `pyproject.toml` at commit b124aa46534cbf2fb8bc8af11405774984c42ac7. https://raw.githubusercontent.com/AgentR1/Agent-R1/b124aa46534cbf2fb8bc8af11405774984c42ac7/pyproject.toml. Fetched 2026-08-12.

[7] GitHub API, releases list for AgentR1/Agent-R1 (empty array, confirming no releases). https://api.github.com/repos/AgentR1/Agent-R1/releases. Fetched 2026-08-12.

[8] Agent-R1 docs, Recipes and Algorithms. https://agentr1.github.io/Agent-R1/tutorials/recipes-and-algorithms/. Fetched 2026-08-12.

[9] Agent-R1 `agent_r1/config/config.py` at commit b124aa46534cbf2fb8bc8af11405774984c42ac7. https://raw.githubusercontent.com/AgentR1/Agent-R1/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/config/config.py. Fetched 2026-08-12.

[10] Agent-R1 docs, Experiments (main results table across GSM8K/HotpotQA/ALFWorld/WebShop, Qwen3-4B). https://agentr1.github.io/Agent-R1/experiments/. Fetched 2026-08-12.

[11] GitHub issue #4, "What kind of GPU will be able to run the RL training?", closed 2025-03-16, reply from `0russwest0` (association: COLLABORATOR). https://github.com/AgentR1/Agent-R1/issues/4. Fetched 2026-08-12.

[12] GitHub issue #25, "repo uses git submodules, and skips initializing them in setup", closed 2025-04-11, reply from `0russwest0` (association: COLLABORATOR). https://github.com/AgentR1/Agent-R1/issues/25. Fetched 2026-08-12.

[13] GitHub API, repository metadata for AgentR1/Agent-R1 (license, star count). https://api.github.com/repos/AgentR1/Agent-R1. Fetched 2026-08-12.

[14] Agent-R1 `examples/gsm8k/run_steppo.sh` at commit b124aa46534cbf2fb8bc8af11405774984c42ac7. https://raw.githubusercontent.com/AgentR1/Agent-R1/b124aa46534cbf2fb8bc8af11405774984c42ac7/examples/gsm8k/run_steppo.sh. Fetched 2026-08-12.

[15] Agent-R1 `examples/gsm8k/run_steppo_tool.sh` at commit b124aa46534cbf2fb8bc8af11405774984c42ac7. https://raw.githubusercontent.com/AgentR1/Agent-R1/b124aa46534cbf2fb8bc8af11405774984c42ac7/examples/gsm8k/run_steppo_tool.sh. Fetched 2026-08-12.

[16] GitHub issue #15, "NotImplementedError: sequence_length=5393 is larger than max_length=4096", closed 2025-03-29, reply from `0russwest0` (association: COLLABORATOR). https://github.com/AgentR1/Agent-R1/issues/15. Fetched 2026-08-12.

[17] GitHub issue #58, "[BUG] Wrong generation length limitation", closed 2025-07-23, reply from `0russwest0` (association: COLLABORATOR). https://github.com/AgentR1/Agent-R1/issues/58. Fetched 2026-08-12.

[18] Agent-R1 `agent_r1/trainer/ppo/metric_utils.py` at commit b124aa46534cbf2fb8bc8af11405774984c42ac7. https://raw.githubusercontent.com/AgentR1/Agent-R1/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/trainer/ppo/metric_utils.py. Fetched 2026-08-12.

[19] GitHub issue #83, "Validation crashes with file logger because NumPy scalars are not JSON serializable", closed 2026-04-20. https://github.com/AgentR1/Agent-R1/issues/83. Fetched 2026-08-12.

[20] Agent-R1 `agent_r1/config/config.py`, `CheckpointConfig` class at commit b124aa46534cbf2fb8bc8af11405774984c42ac7. https://raw.githubusercontent.com/AgentR1/Agent-R1/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/config/config.py. Fetched 2026-08-12.

[21] Live fetch of https://agentr1.github.io/Agent-R1/ (title "Agent-R1", HTTP 200). Fetched 2026-08-12.

[22] Live fetch of https://agentr1.github.io/Agent-R1/getting-started/ (title "Getting Started - Agent-R1", HTTP 200). Fetched 2026-08-12.

[23] Live fetch of https://agentr1.github.io/Agent-R1/experiments/ (HTTP 200, title "Experiments - Agent-R1"). Fetched 2026-08-12.

[24] Live fetch of https://agentr1.github.io/agent-r1 (lowercase; title "Agent-R1 — A Unified Framework for Agentic RL", HTTP 200, a different site from [21]). Fetched 2026-08-12.

[25] Live fetch of https://agentr1.github.io/agent-r1/docs/ (lowercase Chinese-titled docs page, HTTP 200, contains example script paths not present in the repository tree at the pinned commit). Fetched 2026-08-12.

[26] Agent-R1 `mkdocs.yml` at commit b124aa46534cbf2fb8bc8af11405774984c42ac7. https://raw.githubusercontent.com/AgentR1/Agent-R1/b124aa46534cbf2fb8bc8af11405774984c42ac7/mkdocs.yml. Fetched 2026-08-12.

[27] verl `verl/trainer/ppo/ray_trainer.py` at tag v0.7.0 (checkpoint save/resume mechanics: `_save_checkpoint`, `_load_checkpoint`, `resume_mode`, `find_latest_ckpt_path`). https://raw.githubusercontent.com/volcengine/verl/v0.7.0/verl/trainer/ppo/ray_trainer.py. Fetched 2026-08-12.

[28] Agent-R1 GitHub repository home page. https://github.com/AgentR1/Agent-R1. Fetched 2026-08-12.
