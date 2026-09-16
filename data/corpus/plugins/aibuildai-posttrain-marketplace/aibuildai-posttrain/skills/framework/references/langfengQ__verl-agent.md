# verl-agent

An NTU-maintained fork of ByteDance's verl, retooled for training LLM/VLM agents through multi-turn, long-horizon RL against real gym-style environments (ALFWorld, WebShop, Search, Sokoban, Gym Cards, AppWorld) - pick it over base verl when the environment loop, not the single-turn reward, is the point.

`verl-agent` is described in its own README as "an extension of veRL, specifically designed for training large language model (LLM) agents via reinforcement learning (RL)" [1]. It ships a step-independent multi-turn rollout mechanism so per-step input structure, history management, and memory modules are fully customizable, which the README frames as what makes it scale to long-horizon tasks such as ALFWorld episodes needing up to 50 steps [1]. Its Python package is still literally named `verl` (installed with `pip install -e .`) [2], and it is built and maintained by the GiGPO team at Nanyang Technological University (NTU), Singapore, as an extension of ByteDance's original verl codebase - both copyright lines are stated together in the repo's NOTICE file [3]. The API is verl's own: a Hydra/OmegaConf YAML config assembled by dotted-path CLI overrides, launched as `python3 -m verl.trainer.main_ppo algorithm.adv_estimator=... trainer.n_gpus_per_node=...` [4]. It lives at https://github.com/langfengQ/verl-agent [1].

**When to pick it**: you need an agent training loop - multi-turn, environment-in-the-loop rollouts with group-based credit assignment - rather than the single-turn RLHF/RLVR loop base verl and trl both assume; the README's own contrast is "unlike prior approaches that simply concatenate full interaction histories" it decouples per-step inputs from history [1]. Its headline algorithm, GiGPO, is a critic-free, two-level (episode + step) grouping scheme that the README states keeps the same GPU memory footprint and rollout cost as GRPO while improving credit assignment on repeated states [1] (method-level math is not restated here - see the GiGPO methodology card). The README's own results table gives the reference numbers: GiGPO on ALFWorld reaches an 86.7% success rate with Qwen2.5-1.5B-Instruct and 90.8% with Qwen2.5-7B-Instruct, and on WebShop 67.4% (1.5B) and 75.2% (7B) [1]. On the Search-R1/Tool-Calling benchmark suite, the README's own comparison table shows GiGPO averaging 42.1 (Qwen2.5-3B-Instruct) and 47.2 (Qwen2.5-7B-Instruct) across seven QA datasets, against a Search-R1 baseline averaging 32.5 and 38.5 at the same two model sizes - a roughly 9-point gap in the framework's own published comparison [1]. If you don't need one of its six built-in agent environments and just want single-turn RLHF, weigh base verl or trl instead (cross-reference; not covered here).

**Methods it ships** [1][5]: the README's Quick Feature Summary lists GiGPO (its own novel algorithm, arXiv 2505.10978, NeurIPS 2025), GRPO, PPO, DAPO, GSPO, RLOO, and REINFORCE++ as supported RL algorithms, each with its own runnable script under `examples/` (`gigpo_trainer/`, `grpo_trainer/`, `ppo_trainer/`, `dapo_trainer/`, `rloo_trainer/`, `gigpo_dynamic_trainer/` for GiGPO with DAPO's dynamic sampling and clip-higher) [1]. The News section additionally documents two newer recipe additions, GraphGPO (ICML 2026) and HGPO (ICLR 2026), each under `recipe/` [1]. A generic (inherited, not agent-specific) SFT trainer exists under `examples/sft/` [6]. The repo tree also still contains a `recipe/sppo/` directory, but `recipe/README.md` (unmodified from base verl) only describes the `recipe/` folder in general as "representative extensions to verl for specific end-to-end RL training recipes" and then lists a set of unrelated third-party "Awesome work using verl" projects (none of them SPPO, prime, r1, or spin) - it never mentions or documents sppo at all [7], and neither does the main README's own method list [1][5]. The shortlist's SPPO mention is this undocumented recipe folder, inherited unmodified from base verl and not named anywhere in verl-agent's own README as one of its methods.

**Scale it handles**: single GPU up to multi-GPU/multi-node, through the same Ray single-controller launcher as base verl; a SLURM+Ray multi-node template ships at `examples/slurm/ray_on_slurm.slurm`, starting a head node with `ray start --head` and workers with `ray start --address`, then launching `python3 -m verl.trainer.main_ppo` with `trainer.n_gpus_per_node`/`trainer.nnodes` read from SLURM environment variables [8]. verl-agent itself publishes no multi-node benchmark or agent-specific multi-node guide; asked directly in a closed issue, the maintainer (OWNER) pointed users to base verl's own hosted multinode tutorial instead of anything in this repo [9]. Environment-side parallelism is a separate axis: environments run as parallelized Gym workers under Ray by default, and a maintainer reply in a closed issue on CPU-limited ALFWorld training says to switch that worker pool from Ray to plain multiprocessing, pointing at a specific line in `agent_system/environments/env_package/alfworld/envs.py` - a code edit, not a config flag [10].

**Install**: no PyPI package; the README's own recipe is `conda create -n verl-agent python==3.12 -y && conda activate verl-agent && pip3 install vllm==0.11.0 && pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir && pip install -e .` [2]. This delivers whatever is at the current `master` HEAD, commit `20bd331bdbc9026a5668e11362178e10ab7400c8` (pushed 2026-06-09) [11] - the repository's only tagged release is the older `v0.1.0` (2025-12-11) at commit `080965f74fb2d6390cc750cdd07928551f22c2aa` [12][13], so a reader who checks out that tag gets materially different pins from a fresh clone: at HEAD, `setup.py` floors `ray[default]>=2.41.0,<=2.50.0`, `tensordict>=0.8.0,<=0.10.0,!=0.9.0`, `transformers<=4.57.3`, and its `vllm` extra pins `vllm>=0.8.5,<=0.11.0`; at v0.1.0 the same fields read `ray[default]>=2.41.0,<2.50.0`, `tensordict<=0.6.2`, `transformers<=4.51.1`, and `vllm<=0.8.5` [14][15]. At that same HEAD commit, the repo's separate `requirements.txt` ("the full set of dependencies for development" [16]) pins `transformers==4.51.1` and `tensordict<=0.6.2` - both narrower than and inconsistent with `setup.py`'s own floors at the identical commit [16][14], so a reader following `requirements.txt` instead of `setup.py` lands on a different transformers/tensordict pair. `pyproject.toml` sets `python_requires>=3.8` [17], but the README's own install commands only ever target Python 3.12 (or 3.10 for the WebShop environment, which needs a second, dedicated `verl-agent-webshop` conda environment) [2]. Licence is Apache-2.0 [12]. No CUDA/hardware minimum is stated anywhere in the README or setup files; hardware requirements flow through vllm and flash-attn, both pinned by exact version in the install command above [2].

**Maintained by**: the GiGPO team at Nanyang Technological University (NTU), Singapore, building on ByteDance's verl [3]; actively developed - the README's own News section records GraphGPO's ICML 2026 acceptance (2026-05), HGPO's ICLR 2026 acceptance (2026-02), and a multi-agent spin-off project Dr. MAS (2026-02) [1]; the repository's default branch is `master`, last pushed 2026-06-09 [11].

## Quick start

The README's own GiGPO/ALFWorld example is the smallest complete run once ALFWorld is installed (`pip install alfworld && alfworld-download -f` [2]) and a training-data stub is prepared:

```bash
python3 -m examples.data_preprocess.prepare --mode 'text' --train_data_size 16 --val_data_size 128

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.rollout.name=vllm \
    env.env_name=alfworld/AlfredTWEnv \
    env.rollout.n=8 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1
```
This is the (trimmed) content of the repo's own `examples/gigpo_trainer/run_alfworld.sh`, runnable as-is via `bash examples/gigpo_trainer/run_alfworld.sh` [4]. Equivalent one-line launchers exist per method and environment, e.g. `bash examples/grpo_trainer/run_webshop.sh`, `bash examples/ppo_trainer/run_alfworld.sh`, `bash examples/rloo_trainer/run_webshop.sh`, `bash examples/dapo_trainer/run_alfworld.sh` [1]. A LoRA variant of the same GiGPO/ALFWorld script (`examples/gigpo_trainer/run_alfworld_lora.sh`) and a prompt-only GPT-4o baseline agent (`examples/prompt_agent/run_gpt4o_agent.sh`) are also provided [1].

## Start it

- One process, one GPU: run any `examples/<method>_trainer/run_<env>.sh` script directly; the ALFWorld GiGPO example above uses `trainer.n_gpus_per_node=2, trainer.nnodes=1` [4].
- Multiple GPUs/nodes go through Ray, inherited from base verl: for a single multi-GPU node, raise `trainer.n_gpus_per_node`; for multiple nodes, the repo's own template is `examples/slurm/ray_on_slurm.slurm`, which starts a Ray head with `ray start --head --port=6379` on one SLURM node and workers with `ray start --address` on the rest, then runs `python3 -m verl.trainer.main_ppo trainer.n_gpus_per_node=$SLURM_GPUS_PER_NODE trainer.nnodes=$SLURM_NNODES ...` [8]. This script is a generic base-verl PPO/gsm8k template, not agent-specific, and verl-agent publishes no agent-specific multi-node example or doc of its own [9].
- Group size for group-based methods (GRPO, GiGPO, DAPO) is set independently on two axes: `env.rollout.n`, documented in the shipped config as "the group number of envs (for GRPO and GiGPO)" and defaulting to 1 [18], and `actor_rollout_ref.rollout.n`, the LLM sampling count per prompt, defaulting to 1 in the same config [18]. The ALFWorld GiGPO example only sets the former, `env.rollout.n=$group_size` with `group_size=8`, and leaves `actor_rollout_ref.rollout.n` unset (so it stays at its default of 1) [4].
- Generation runs through vLLM as a separate engine, selected with `actor_rollout_ref.rollout.name=vllm`; the ALFWorld example sets `tensor_model_parallel_size=2` and `gpu_memory_utilization=0.6`, leaving the remaining share of each GPU for the FSDP-sharded actor/ref weights on the same devices [4].
- Config surface: the same Hydra `ppo_trainer.yaml` as base verl, with `algorithm.gamma=1.0`, `algorithm.lam=1.0`, `algorithm.adv_estimator=gae`, and PPO `clip_ratio=0.2` as the shipped defaults [18] - the ALFWorld GiGPO example overrides `algorithm.gamma=0.95` and adds GiGPO-only knobs `algorithm.gigpo.step_advantage_w=1.0` and `algorithm.gigpo.mode=mean_std_norm` [4]. verl-agent adds one env-side block absent from base verl: `env.env_name`, `env.seed`, `env.max_steps`, `env.rollout.n`, and `env.resources_per_worker.num_cpus` (the CPU budget per parallel environment worker, 0.1 in the ALFWorld example) [4][18].
- Out-of-memory first aid: the ALFWorld example itself documents the CPU-side knob inline (`num_cpus_per_env_worker`, "the CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value") [4]; on the GPU side, lower `actor_rollout_ref.rollout.gpu_memory_utilization` (0.6 in the example) and `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` (32 in the example) [4]. The README's own prose carries no OOM or memory-tuning section: grepping the full fetched README text for "oom", "out of memory", "gpu_memory_utilization", and "micro_batch" returns zero matches for any of the four terms (checked 2026-08-11) [1] - the only knobs documented anywhere in this repo for memory/CPU pressure are the inline comment above and the example scripts' own flag values [4]. A closed-issue reply from the maintainer is the only other documented resource-limit fix: on CPU-constrained ALFWorld training, switch the environment worker pool from Ray to multiprocessing by editing `agent_system/environments/env_package/alfworld/envs.py` around line 79 - a code change, not a config flag [10].

## Watch it

Mechanics only - what a metric's shape means for a given algorithm lives on that method's own card.

- **Enable it**: `trainer.logger=['console','wandb']` in the config (used verbatim in the ALFWorld example) [4]; wandb is the only tracker demonstrated in the shipped scripts, and console is the fallback if no tracker is set - the README does not document other backends for this fork [1][4].
- **Base-verl metrics** (inherited unmodified): actor-side losses and health such as `actor/pg_loss`, `actor/kl_loss`, `actor/kl_coef`, `actor/ppo_kl`, `actor/pg_clipfrac`, `actor/pg_clipfrac_lower`, `actor/grad_norm`, read from `verl/workers/actor/dp_actor.py` at commit `20bd331b` [19]; plus the generic `critic/*`, `response_length/*`, `prompt_length/*`, `perf/*`, `timing_s/*`, and `best@N`/`worst@N`/`maj@N` families from `verl/trainer/ppo/metric_utils.py` at the same commit [20].
- **verl-agent's own additions**, sourced from the same `metric_utils.py` and from `agent_system/environments/env_manager.py` at commit `20bd331b` [20][21]: `episode/reward/{mean,max,min}`, `episode/length/{mean,max,min}`, `episode/tool_call_count/mean`, and a per-run, dynamically-keyed success metric - `episode/success_rate` in general, with per-environment variants such as `episode/webshop_success_rate` when the key name embeds the environment [21]. This `episode/*` family is verl-agent's most direct signal that an agent is actually completing tasks, and is the concrete thing base verl's own metrics do not give you.
- **A same-quantity caveat**: WebShop's environment manager also accumulates a second, separate score under the literal dictionary key `'webshop_task_score (not success_rate)'` - the code's own key name flags that this graded task score is not the same quantity as the binary `episode/success_rate` metric, so do not average or compare them as one number [21].
- **Sample-level logging of generations** and **evaluation-during-training** fields are inherited from base verl's Hydra config surface (`trainer.val_before_train`, `trainer.test_freq`, used in the ALFWorld example as `trainer.val_before_train=True` and `trainer.test_freq=5` [4]); verl-agent's README does not document any generation-transcript logging switch beyond these base-verl trainer fields, and none was found in the config file read for this card [18].
- **Health limit**: the maintainer states, in a closed issue reply, that "a valid RL training should begin with a non-zero success rate," and to check the starting point before assuming the run is broken [22] - this is a maintainer-stated heuristic, not a published numeric threshold; no other stopping-rule or threshold value was found in the README or `ppo_trainer.yaml` (checked 2026-08-11) [1][18].

## Save it

- Checkpoints land under `default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}` by default, one `global_step_<N>/` subdirectory per save (the repo's own `scripts/model_merger.py` docstring gives a worked path, `checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor`) [18][23].
- Cadence is `trainer.save_freq` (-1, i.e. off, by default; the ALFWorld example leaves it at -1) [4][18]; resume mode is `trainer.resume_mode: auto` by default (`disable` or an explicit `resume_path` are the other settings) [18].
- What lands in each `global_step_<N>/actor/` directory depends on `actor.checkpoint.contents`, default `['model', 'optimizer', 'extra']`: the shipped config comment reads "with 'hf_model' you can save whole model as hf format, now only use sharded model checkpoint to save space" [18]. Reading the FSDP checkpoint manager code confirms this at commit `20bd331b`: the default writer produces per-rank sharded files (`model_world_size_{W}_rank_{R}.pt`, `optim_world_size_{W}_rank_{R}.pt`, `extra_state_world_size_{W}_rank_{R}.pt`), and only writes a real `huggingface/` subdirectory in `save_pretrained` format when `'hf_model'` is added to `checkpoint.contents` [24]. Without that opt-in, the default checkpoint is NOT directly loadable with `from_pretrained` - it is a resumable sharded checkpoint, not a portable model.
- To get a loadable model out of a default (non-`hf_model`) checkpoint after the fact, the repo ships `scripts/model_merger.py`, run as `python scripts/model_merger.py merge --backend fsdp --local_dir checkpoints/<project>/<experiment>/global_step_<N>/actor --target_dir /path/to/merged_hf_model`; it also supports `--backend megatron` with a `--tie-word-embedding` flag, and its own docstring points to base verl's hosted docs for the full conversion contract [23].
- Resume: set `trainer.resume_mode=auto` (or point `resume_from_path` at a specific `global_step_<N>` directory) and rerun the same `main_ppo` command; this is base-verl mechanism, unmodified in this fork, and the loadable-checkpoint contract above governs it identically [18][24].
- Whether an evaluator can load what you saved is the loader's contract, not this framework's: a default sharded checkpoint needs `model_merger.py` first; an `hf_model`-content checkpoint or a merged output directory is a normal HuggingFace model directory.

## Find it in the docs

verl-agent ships a `docs/` Sphinx tree and a working `.readthedocs.yaml` [25], but there is no live hosted docs site for this fork: `https://verl-agent.readthedocs.io/` returns 404 (checked 2026-08-11) [26], and the docs source itself is unmodified base-verl content - `docs/index.rst` still opens "Welcome to verl's documentation!" [27] and `recipe/README.md` still describes the generic base-verl recipe layout [7]. Treat the README as the primary maintained documentation for this repo.

- The README (fetched at commit `20bd331b` via `raw.githubusercontent.com/langfengQ/verl-agent/master/README.md`) is organized with an in-file table of contents; its own section anchors are the fastest lookup: Installation -> per-environment subsections (`#1-alfworld`, `#2-webshop`, `#3-search`, `#4-sokoban`, `#5-gym-cards`, `#6-appworld-experimental`); Run Examples -> RL Training (one subsection per algorithm); FAQ -> memory customization, data preparation, prompt customization, adding new environments [1].
- For anything that is base-verl mechanism rather than agent-specific (multi-node setup, checkpoint conversion, Ray internals), the maintainer's own closed-issue replies point to base verl's live, hosted docs at `https://verl.readthedocs.io/en/latest/` (verified live, 2026-08-11) rather than to anything in this repo [9][23].
- Runnable references beyond the README: the `examples/` tree (`gigpo_trainer/`, `grpo_trainer/`, `ppo_trainer/`, `rloo_trainer/`, `dapo_trainer/`, `gigpo_dynamic_trainer/`, `sft/`, `search/`, `slurm/`, `ray/tutorial.ipynb`, `prompt_agent/`) [1]; released GiGPO checkpoints and the arXiv paper are collected on the Hub at `https://huggingface.co/collections/langfeng01/verl-agent-684970e8f51babe2a6d98554`, linked directly from the README's own badge row - `langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld` and `langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop` [1][28].
- No official MCP endpoint for these docs was found; none is named in the README or the docs tree.

Traps, from maintainer replies in closed issues:
- WebShop needs only the second, dedicated `verl-agent-webshop` conda environment for training, not the main `verl-agent` one; a user asked why two environments exist after following the install steps, and the maintainer (OWNER) clarified only `verl-agent-webshop` is needed at training time (issue #101, reply 2025-07-24) [22].
- CPU-limited ALFWorld runs should switch the environment worker pool from Ray to multiprocessing by editing `agent_system/environments/env_package/alfworld/envs.py`; the maintainer (OWNER) gave this as the fix rather than any config flag (issue #204, reply 2025-11-30) [10].
- No verl-agent-specific multi-node walkthrough exists; asked directly, the maintainer (OWNER) pointed to base verl's own hosted tutorial instead (issue #144, reply 2025-08-19) [9].

Honest boundary: verl-agent documents six built-in environments (ALFWorld, WebShop, Search/Tool Calling, Sokoban, Gym Cards, AppWorld) and marks AppWorld itself "Experimental" in its own table of contents [1]; it publishes no multi-node benchmark of its own [9], and beyond the inline script comments and the maintainer's issue-thread advice above, no dedicated memory/OOM-tuning guide was found in the README (checked 2026-08-11) [1].

## Sources

Method names (GiGPO, GRPO, PPO, DAPO, GSPO, RLOO, REINFORCE++, GraphGPO, HGPO) are deliberately cited to nothing here; their defining papers and math live on their own methodology cards. All GitHub file/API reads are pinned to commit `20bd331bdbc9026a5668e11362178e10ab7400c8` (current `master` HEAD, pushed 2026-06-09) unless a different commit or tag is named; this commit is ahead of the repository's only tagged release, `v0.1.0` (2025-12-11, commit `080965f74fb2d6390cc750cdd07928551f22c2aa`), and the Install field discloses where the two disagree. All fetches were made 2026-08-11 unless noted.

[1] verl-agent README, raw at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/README.md. Fetched 2026-08-11.

[2] verl-agent README, Installation section (same file as [1]).

[3] verl-agent NOTICE file. https://raw.githubusercontent.com/langfengQ/verl-agent/master/Notice.txt. Fetched 2026-08-11.

[4] verl-agent GiGPO/ALFWorld example script. https://raw.githubusercontent.com/langfengQ/verl-agent/master/examples/gigpo_trainer/run_alfworld.sh. Fetched 2026-08-11.

[5] verl-agent README, Quick Feature Summary table (same file as [1]).

[6] verl-agent `examples/sft/` directory listing. https://api.github.com/repos/langfengQ/verl-agent/contents/examples/sft. Fetched 2026-08-11.

[7] verl-agent `recipe/README.md`, unmodified from base verl. https://raw.githubusercontent.com/langfengQ/verl-agent/master/recipe/README.md. Fetched 2026-08-11.

[8] verl-agent SLURM+Ray multi-node template. https://raw.githubusercontent.com/langfengQ/verl-agent/master/examples/slurm/ray_on_slurm.slurm. Fetched 2026-08-11.

[9] GitHub issue #144, "Support multi-node training," maintainer (OWNER) reply, 2025-08-19. https://github.com/langfengQ/verl-agent/issues/144. Fetched 2026-08-11.

[10] GitHub issue #204, "Training ALFWorld with small number of CPUs," maintainer (OWNER) reply, 2025-11-30. https://github.com/langfengQ/verl-agent/issues/204. Fetched 2026-08-11.

[11] GitHub API, repo default branch and last-push date. https://api.github.com/repos/langfengQ/verl-agent. Fetched 2026-08-11.

[12] GitHub API, releases (v0.1.0, 2025-12-11) and licence. https://api.github.com/repos/langfengQ/verl-agent/releases and https://api.github.com/repos/langfengQ/verl-agent/license. Fetched 2026-08-11.

[13] GitHub API, tags (v0.1.0 -> commit `080965f7`). https://api.github.com/repos/langfengQ/verl-agent/tags. Fetched 2026-08-11.

[14] verl-agent `setup.py` at commit `20bd331b` (current HEAD). https://raw.githubusercontent.com/langfengQ/verl-agent/master/setup.py. Fetched 2026-08-11.

[15] verl-agent `setup.py` at tag `v0.1.0`. https://raw.githubusercontent.com/langfengQ/verl-agent/v0.1.0/setup.py. Fetched 2026-08-11.

[16] verl-agent `requirements.txt` at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/requirements.txt. Fetched 2026-08-11.

[17] verl-agent `pyproject.toml` at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/pyproject.toml. Fetched 2026-08-11.

[18] verl-agent `verl/trainer/config/ppo_trainer.yaml` at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/verl/trainer/config/ppo_trainer.yaml. Fetched 2026-08-11.

[19] verl-agent `verl/workers/actor/dp_actor.py` at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/verl/workers/actor/dp_actor.py. Fetched 2026-08-11.

[20] verl-agent `verl/trainer/ppo/metric_utils.py` at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/verl/trainer/ppo/metric_utils.py. Fetched 2026-08-11.

[21] verl-agent `agent_system/environments/env_manager.py` at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/agent_system/environments/env_manager.py. Fetched 2026-08-11.

[22] GitHub issue #101, "Webshop success rate remains 0," maintainer (OWNER) replies, 2025-07-09 and 2025-07-24. https://github.com/langfengQ/verl-agent/issues/101. Fetched 2026-08-11.

[23] verl-agent `scripts/model_merger.py` docstring, at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/scripts/model_merger.py. Fetched 2026-08-11.

[24] verl-agent `verl/utils/checkpoint/fsdp_checkpoint_manager.py` at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/verl/utils/checkpoint/fsdp_checkpoint_manager.py. Fetched 2026-08-11.

[25] verl-agent `.readthedocs.yaml` at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/.readthedocs.yaml. Fetched 2026-08-11.

[26] `https://verl-agent.readthedocs.io/` liveness check (404). Fetched 2026-08-11.

[27] verl-agent `docs/index.rst` at commit `20bd331b`. https://raw.githubusercontent.com/langfengQ/verl-agent/master/docs/index.rst. Fetched 2026-08-11.

[28] Hugging Face collection "verl-agent" (owner `langfeng01`), linked from the README's badge row. https://huggingface.co/collections/langfeng01/verl-agent-684970e8f51babe2a6d98554. Fetched 2026-08-11.
