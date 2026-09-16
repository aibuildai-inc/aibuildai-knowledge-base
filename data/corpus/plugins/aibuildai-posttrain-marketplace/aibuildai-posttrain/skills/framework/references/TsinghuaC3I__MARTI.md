# MARTI

A Tsinghua/Shanghai AI Lab multi-agent RL framework built on top of OpenRLHF: one Ray-orchestrated job trains a graph of interacting agent policies, not a single model.

MARTI is described by its own README as "A Framework for LLM-based Multi-Agent Reinforced Training and Inference" [1]. It is developed and maintained by Tsinghua University's C3I group and Shanghai AI Lab, with project leads Kaiyan Zhang and Biqing Qi and a listed Core Contributors group [1]. Its shape: a graph-based multi-agent "workflow" (e.g. debate, chain-of-agents, mixture-of-agents, or a third-party framework like AutoGen/CAMEL) supplies per-agent prompts and rollouts to a centralized reward-allocation step, whose outputs feed a Ray-distributed PPO-family trainer that updates each agent's policy weights, all launched from one `python3 -m marti.cli.multi_agent_train_ppo_ray` CLI call carrying JSON-string agent/workflow/reward configs as flags [1][2][3]. It lives at https://github.com/TsinghuaC3I/MARTI [4], pinned at commit `a2fe2c7b9ec46cf24769c90575c51d847f41d04e` [4].

**When to pick it**: pick MARTI when the object being trained is a graph of two or more interacting agent policies with a shared, centrally-computed reward (debate, chain-of-agents, mixture-of-agents, review) and you want that multi-agent loop built on a Ray/vLLM/DeepSpeed PPO-style trainer rather than assembled from a single-agent trainer plus custom orchestration code; the README states it is "developed primarily based on" OpenRLHF [1][5], so single-agent post-training on the same stack is OpenRLHF's job, not this repo's differentiator. Its newer MARS² mode adds tree-search-augmented RL for code generation with a GSPO sequence-level loss and truncated-importance-sampling (TIS) correction for vLLM/policy sampling mismatch [1] — pick that mode specifically when the task is code generation under a Pass@1/Pass@N metric: on the README's own LiveCodeBench (LCB) results, single-agent MCTS beats a vanilla-GRPO baseline by up to 4.6 points Pass@1 (up to 5.1 points Pass@1(MCTS)), and multi-agent MCTS on Qwen3-8B adds 8.0 points over the base model, 4.4 points over vanilla GRPO, and 2.9 points over the single-agent MCTS peak [1].

**Methods it ships**: single-agent trainer classes for SFT, DPO, KTO, PRM, RM, and KD (`marti/trainer/{sft,dpo,kto,prm,rm,kd}_trainer.py`), plus a single-agent PPO-family Ray trainer (`marti/trainer/ppo_trainer.py`, with an async variant `ppo_trainer_async.py`) and a multi-agent PPO-family Ray trainer (`marti/trainer/multi_agent_ppotrainer.py`), all present in the tree at the pinned commit [4]. Within the PPO-family trainers, `--advantage_estimator` selects among GAE (PPO), `reinforce`, `rloo`, `reinforce_baseline`, `group_norm` (used for GRPO-style updates), and `dr_grpo` [6]; the README additionally lists PPO, GRPO, REINFORCE++, and TTRL among "diverse RL algorithms" it supports [1] — these are advantage-estimator/reward-allocation modes of the one PPO-family trainer rather than separate trainer files, and TTRL specifically is toggled via a `use_ttrl` flag inside the `--reward_alloc` JSON config [7]. `--policy_loss_type` chooses `ppo` (token-level) or `gspo` (sequence-level, the MARS² default) [6][8]. There is no separate methods index page outside this source tree to recheck against; the taxonomy is read directly from `marti/trainer/` and the CLI argparse definitions at the pinned commit [4][6].

**Scale it handles**: single GPU up to a documented multi-node cluster, launched through Ray, which the CLI attaches to via a bare `ray.init()` call with no address (starts or joins a local Ray instance) [6]; node/GPU placement per role is set with `--{ref,actor,critic,reward}_num_nodes` and `--{ref,actor,critic,reward}_num_gpus_per_node` [6]. Training-side sharding is DeepSpeed ZeRO via `--zero_stage` (default 2) [6]; generation runs through vLLM, either colocated on the training GPUs (`--colocate_all_models`) or on separate engines sized by `--vllm_num_engines` and `--vllm_tensor_parallel_size` [6][9]. A published multi-agent run used a 3-node x 8xA800-80GB cluster for MARTI-v1 training, one full node per agent, per the docs' own Experiments page [21], and the README reports a 3-node x 8xH200 cluster for MARTI-v2, also one full node per agent [10] — a documented multi-node benchmark, but the repository ships no committed multi-node cluster-start script (no SLURM/`ray start` script found in the tree at the pinned commit); a reader must write their own Ray-cluster bootstrap before the `--*_num_nodes` flags take effect [4].

**Install**: `git clone https://github.com/TsinghuaC3I/MARTI.git && cd MARTI && pip install -r requirements.txt` [1]; the repo carries no GitHub Releases or tags (both endpoints return an empty list) [11][12], so there is no versioned release to pin against — only the raw pinned commit `a2fe2c7b9ec46cf24769c90575c51d847f41d04e` exists, and `version.txt` at that commit reads `2.0` but is not attached to any git tag [13]. `setup.py` at the same commit sets `python_requires=">=3.10"` and defines extras `vllm` (`vllm==0.8.5.post1`) and `vllm_latest` (`vllm>0.8.5.post1`) [14]; licence is MIT [15]. Load-bearing pins in `requirements.txt` at the pinned commit: `transformers==4.57.0`, `deepspeed==0.18.0`, `ray[default]==2.48.0`, `flash-attn==2.8.3`; `torch`, `accelerate`, `datasets`, `peft`, `bitsandbytes`, `wandb`, and `tensorboard` are listed unpinned [16]. Neither `requirements.txt` nor `setup.py` states a CUDA or GPU-architecture minimum; `flash-attn==2.8.3` and the vLLM extra imply a CUDA-capable GPU is required, but no explicit hardware floor is written anywhere in these files [16][14].

**Maintained by**: Tsinghua University's C3I group and Shanghai AI Lab, per the README's Acknowledge and Contact sections [1]; the repository was created 2025-05-10 and last pushed 2026-04-14 [4], with the pinned commit itself a merged PR titled "Fix stale openrlhf references in single-agent examples" [17] — a dated, concrete sign of active maintenance, though (see the trap noted in Start it) that same PR title implies the equivalent multi-agent example scripts were left unfixed.

## Quick start

MARTI has no minimal single-file "hello world" script in its own docs; its quickstart is entirely shell-script invocations over full training configs. The README's installation block, quoted in full [1]:

```bash
git clone https://github.com/TsinghuaC3I/MARTI.git
cd MARTI

pip install -r requirements.txt
```

The README's smallest run forms are `bash` calls to committed example scripts, each requiring the reader to first set `ROOT_DIR`/`MODEL_DIR` path variables inside the script [1]:

```bash
# Single-agent MCTS training (MARS2), ~8x80G GPUs
bash examples/mars2/run_train_single_mcts.sh

# Asynchronous single-agent math RL
bash examples/single-agent/run_train_math_async.sh

# Multi-agent debate, ~8x80G GPUs per agent
bash examples/multi-agent/run_train_mad.sh
```

Under the hood, `run_train_math_async.sh` (single-agent) invokes the CLI directly; this is the actual runnable command form, reproduced with its own default single-node/2-GPU layout [2]:

```bash
python3 -m marti.cli.multi_agent_train_ppo_ray \
    --default_agent "$DEFAULT_AGENT" \
    --agents "$AGENT0" \
    --workflow_args "$WORKFLOW_ARGS" \
    --workflow_func_path marti/agent_workflows/single_mathworkflow.py \
    --actor_num_nodes 1 --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 --vllm_tensor_parallel_size 2 \
    --colocate_all_models --zero_stage 3 --bf16 \
    --advantage_estimator group_norm --policy_loss_type gspo \
    --train_batch_size 32 --micro_train_batch_size 1 \
    --prompt_data ${PROMPT_DATA} --eval_dataset ${PROMPT_DATA}
```

## Start it

- Single-GPU/single-node runs use the same `marti.cli.multi_agent_train_ppo_ray` (multi-agent-capable, works with one `--agents` entry) or the single-agent `marti.cli.train_ppo_ray`, `train_sft`, `train_dpo`, `train_kto`, `train_kd`, `train_prm`, `train_rm` CLIs, all under `marti/cli/` [4].
- Multi-GPU/multi-node scales through Ray: `--{ref,actor,critic,reward}_num_nodes` and matching `--*_num_gpus_per_node` flags set per-role placement, and `--colocate_all_models` puts vLLM generation on the same GPUs as training instead of separate `--vllm_num_engines` [6]. The CLI's own `ray.init()` call takes no cluster address, so it attaches to (or starts) a local Ray instance [6]; no committed script in this repo starts a multi-node Ray cluster, so multi-node runs require the operator to `ray start` the cluster themselves before launching the CLI (see Scale it handles).
- Effective train batch is `train_batch_size` (global) divided across `micro_train_batch_size` (per-GPU) with the rest absorbed by gradient accumulation internal to the DeepSpeed strategy; the example scripts pair `train_batch_size 32` with `micro_train_batch_size 1` on 2 actor GPUs [2].
- Generation layout for online (PPO-family) training is a config choice up front: `--colocate_all_models` shares the training GPUs with vLLM (both example scripts above use it, with `--vllm_gpu_memory_utilization 0.6` reserving 60% of a colocated card for generation), or omit it and size dedicated vLLM engines with `--vllm_num_engines`/`--vllm_tensor_parallel_size` [2][3][6].
- Config surface is plain argparse (`marti/cli/*.py`), not a config-file system; the docs' Workflows Integration page separately describes the workflow layer itself (task rounds, default agent settings) as "a configuration-driven approach using Hydra" [18], but no `hydra` import or Hydra YAML file exists anywhere in the pinned-commit tree — in the actual example scripts, workflow/agent/tool/reward-allocation settings are passed as inline JSON strings to `--workflow_args`, `--agents`, `--tools_config`, and `--reward_alloc` CLI flags, not as Hydra config files [4][2][3].
- ZeRO stage: default 2 [6], but both example scripts above override it to `--zero_stage 3` for their multi-GPU actor layout [2][3] — a silent-default trap only if a reader trusts the CLI default rather than the example.
- Out-of-memory first aid: no OOM guidance is published anywhere in this repo. Searched at the pinned commit: README.md and all four `docs/*.md` pages return zero matches for "oom", "out of memory", or "memory" as a tuning topic [1][18][19][20][21]. The only memory-relevant knob documented by name is `--vllm_gpu_memory_utilization` (default 0.95 in the CLI's own argparse definition [6], overridden down to 0.6 in both example scripts above [2][3]) and `--vllm_enable_sleep`/`--deepspeed_enable_sleep`, both of which the example scripts turn on without commentary on what they trade off [2][3].

## Watch it

This section is mechanics only; what a logged number means for a given method (PPO/GRPO/DPO/etc.) belongs on that method's own card.

- Enable it with `--use_wandb <api_key>` (also accepts `True` to read from env) or `--use_tensorboard <log_dir>`; the multi-agent trainer only opens TensorBoard when wandb is unset (`if self.strategy.args.use_tensorboard and self._wandb is None`), so wandb silently wins if both are passed [22]. Neither example script above enables either by default — the wandb lines in `run_train_mad.sh` are present but commented out — so a run produces no persisted metrics until one of these flags is set explicitly [3].
- Single-agent PPO trainer (`marti/trainer/ray/ppo_actor.py`) logs a `short_status` dict built from the training-step `status` dict with these keys, read directly from the source at the pinned commit: `act_loss` (policy loss), `reward`, `return`, `gen_len` (response length), `tot_len` (total length), `kl` (KL, normalized per response length), `act_lr` (actor learning rate), and `ent_loss` when entropy regularization is active [23].
- Multi-agent trainer (`marti/trainer/multi_agent_ppotrainer.py`) namespaces every scalar per agent as `train/{agent_key}/{metric}`, on both the wandb and TensorBoard code paths; evaluation metrics log as `eval/{metric}` [24]. The TensorBoard path additionally writes each scalar a second time as `agent_{agent_idx}_step/train/{metric}`, keyed on that agent's own step counter so agents trained at different cadences get separate x-axes — this second form is written only in the TensorBoard branch, not the wandb branch, so it is absent whenever wandb is active (the common case per the enablement note above, since wandb silently wins when both are configured) [24].
- Sample-level logging of generations is supported: when wandb is active, generated text and its reward are appended to a running `wandb.Table` and logged as `train/{agent_key}/generated_samples`; the TensorBoard path logs the same text via `add_text` under the same key [24].
- Evaluation during training is scheduled by `--eval_steps`; both example scripts above set `--eval_steps 20` against a held-out `--eval_dataset`/`--eval_split` pair [2][3][6].
- Stopping-rule search, run at the pinned commit: `marti/cli/train_ppo_ray.py` and `marti/trainer/multi_agent_ppotrainer.py` were grepped for early-stopping, patience, and reward-threshold arguments; none of the CLI's argparse definitions define one, and none of the four `docs/*.md` pages mention a stopping rule [6][18][19][20][21]. `--num_episodes` and `--max_samples` bound a run by episode/sample count, not by a converged-metric threshold [6]. No RL-specific stopping rule or threshold is published; shape (metric names above) is published, threshold is not.

## Save it

- `--save_path` (default `./ckpt`) is where the final HF-format model is written; `--ckpt_path` (default `./ckpt/checkpoints_ppo_ray`) is where periodic training checkpoints go; `--save_steps` (default -1, meaning save only at the end) sets checkpoint cadence, with `--max_ckpt_num` (default 3) and `--max_ckpt_mem` bounding how many/how large checkpoints are kept on disk [25].
- Two distinct save paths exist and are not interchangeable. `save_ckpt()` (DeepSpeed's native `model.save_checkpoint`) is called on every checkpoint step regardless of flags and writes a DeepSpeed engine checkpoint that includes optimizer and scheduler state, and is resumable via DeepSpeed's own `load_checkpoint` [26]. A separate HF-format snapshot — a `from_pretrained`-loadable directory — is written only when `--save_hf_ckpt` is passed on the command line; both example scripts above pass it [2][3][25]. Without `--save_hf_ckpt`, only the resumable-but-not-directly-loadable DeepSpeed checkpoint exists until `save_model()` is called at the end of training.
- `--load_checkpoint` (a boolean flag, both example scripts pass it) resumes training from the checkpoint under `--ckpt_path` using DeepSpeed's `load_checkpoint`, restoring optimizer and scheduler state along with the weights [25][26].
- PEFT/LoRA models are saved specially in `deepspeed_strategy.py`'s `save_model()`: under ZeRO stage 3 or tensor-parallel training the adapter is gathered and written as `adapter_model.bin` rather than the standard `adapter_model.safetensors` PEFT normally writes, so a loader expecting the PEFT-standard filename needs to account for this MARTI-specific naming [27]. As with any LoRA adapter, an adapter checkpoint is not a full model directory and needs the base model to reload.
- Whether an external evaluator can load a MARTI checkpoint directly depends on which of the two save paths produced it: the `--save_hf_ckpt` directory (or the final `save_model()` output at `--save_path`) is `from_pretrained`-loadable; the DeepSpeed-native `--ckpt_path` checkpoint is resumable inside MARTI/DeepSpeed but is not a directly loadable HF model directory.

## Find it in the docs

This repository has no separate hosted documentation site; "the docs" are the four Markdown pages under `docs/` in the GitHub repo itself, linked from the README's Documentation section [1], plus one external technical report:

- `docs/1-Overview-Of-MARTI.md` — the Multi-Agent World / Centralized Reward Models / Agent Policy Trainer module breakdown [19].
- `docs/2-Workflows-Integration.md` — the workflow-graph and per-agent config description, including the Hydra-language claim discussed under Start it [18].
- `docs/3-Reward-And-Training.md` — reward-shaping/credit-assignment formulas [20].
- `docs/4-Experiments-Of-MARTI.md` — the MARTI-v1 experiment write-up, cluster sizes, and benchmark figures [21].
- The MARS²/MARTI-v2 technical report lives outside the repo, at the arXiv abs page linked from the README as "MARTI-v2 Technical Details": http://arxiv.org/abs/2602.07848 [1]. Method-level math from that report belongs on a separate method card, not here.
- Runnable references beyond the docs: the `examples/` tree holds one script per workflow — `examples/mars2/{run_train_single_mcts.sh,run_train_multi_mcts.sh}`, `examples/single-agent/{run_train_code_async.sh,run_train_math_async.sh}`, `examples/multi-agent/{run_train_chain.sh,run_train_mad.sh,run_train_mathchat.sh}`, `examples/reviewrl/run_train_reviewrl_async.sh`, and `examples/multi-turn-tool/{run_train_search_r1.sh,run_train_tir.sh}` — each a complete, runnable CLI invocation rather than a doc snippet [4][1].
- No official curated community-tutorials page or MCP endpoint was found for this repository; the README's only external pointers are the arXiv report and the OpenReview submission page for the ICLR 2026 MARTI paper (`https://openreview.net/forum?id=E7jZqo0A50`) [1], both citation targets rather than tutorials.
- Trap, found directly in source rather than in a closed issue (no matching GitHub issue exists — a search for "openrlhf.cli" against this repo's issues returned zero results) [28]: at the pinned commit, the multi-agent example scripts `run_train_mad.sh`, `run_train_chain.sh`, and `run_train_reviewrl_async.sh` still invoke `python3 -m openrlhf.cli.multi_agent_train_ppo_ray` [3] — a module that does not exist in this package (`openrlhf` is not in `requirements.txt` and ships no `openrlhf.cli` namespace) [16][4] — while the single-agent example scripts (e.g. `run_train_math_async.sh`) correctly call `python3 -m marti.cli.multi_agent_train_ppo_ray` [2]. The pinned commit's own message, "Fix stale openrlhf references in single-agent examples," confirms the single-agent scripts were just fixed and implies the multi-agent scripts were left with the stale, broken path [17]. Anyone running the multi-agent example scripts as committed at this pin should replace `openrlhf.cli` with `marti.cli` first.
- Honest boundary: this repo's methods list is single- and multi-agent SFT/DPO/KTO/PRM/RM/KD/PPO-family trainers built on Ray+DeepSpeed+vLLM [4][6]; it documents no non-Ray single-process training path, no non-DeepSpeed sharding backend (no FSDP), and no CPU-only path — `flash-attn` and the vLLM extra in its dependency files imply a CUDA GPU is required throughout [16][14].

## Sources

[1] MARTI README at commit a2fe2c7b9ec46cf24769c90575c51d847f41d04e. https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/README.md. Fetched 2026-08-12.

[2] `examples/single-agent/run_train_math_async.sh` at the pinned commit. https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/examples/single-agent/run_train_math_async.sh. Fetched 2026-08-12.

[3] `examples/multi-agent/run_train_mad.sh` at the pinned commit. https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/examples/multi-agent/run_train_mad.sh. Fetched 2026-08-12.

[4] MARTI GitHub repository (full recursive tree at the pinned commit via the GitHub Trees API). https://github.com/TsinghuaC3I/MARTI ; tree read via https://api.github.com/repos/TsinghuaC3I/MARTI/git/trees/a2fe2c7b9ec46cf24769c90575c51d847f41d04e?recursive=1. Fetched 2026-08-12.

[5] MARTI README, Acknowledge section (built primarily on OpenRLHF). Same source as [1].

[6] `marti/cli/train_ppo_ray.py` at the pinned commit (argparse definitions: advantage estimator/policy loss choices, Ray node/GPU placement flags, zero_stage, batch-size flags, vLLM flags, ray.init call). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/marti/cli/train_ppo_ray.py. Fetched 2026-08-12.

[7] `marti/cli/multi_agent_train_ppo_ray.py` at the pinned commit (`--reward_alloc` default includes `use_ttrl`). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/marti/cli/multi_agent_train_ppo_ray.py. Fetched 2026-08-12.

[8] `run_train_math_async.sh`/`run_train_mad.sh` (`--policy_loss_type gspo` usage). Same sources as [2] and [3].

[9] `run_train_math_async.sh`/`run_train_mad.sh` (`--vllm_num_engines`, `--vllm_tensor_parallel_size`, `--colocate_all_models`, `--vllm_gpu_memory_utilization`). Same sources as [2] and [3].


[10] MARTI README, Experimental Results / MARTI-v2 Training Details (3-node x 8xH200 cluster). Same source as [1].

[11] MARTI GitHub Releases API (empty list). https://api.github.com/repos/TsinghuaC3I/MARTI/releases. Fetched 2026-08-12.

[12] MARTI GitHub Tags API (empty list). https://api.github.com/repos/TsinghuaC3I/MARTI/tags. Fetched 2026-08-12.

[13] `version.txt` at the pinned commit (content "2.0"). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/version.txt. Fetched 2026-08-12.

[14] `setup.py` at the pinned commit (python_requires, vllm/vllm_latest extras). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/setup.py. Fetched 2026-08-12.

[15] MARTI LICENSE file at the pinned commit (MIT). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/LICENSE. Fetched 2026-08-12.

[16] `requirements.txt` at the pinned commit (transformers, deepspeed, ray, flash-attn pins; unpinned torch/accelerate/datasets/peft/bitsandbytes/wandb/tensorboard). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/requirements.txt. Fetched 2026-08-12.

[17] GitHub API, pinned-commit detail (commit message: "Fix stale openrlhf references in single-agent examples"; commit date 2026-04-14). https://api.github.com/repos/TsinghuaC3I/MARTI/commits/a2fe2c7b9ec46cf24769c90575c51d847f41d04e. Fetched 2026-08-12.

[18] `docs/2-Workflows-Integration.md` at the pinned commit (Hydra configuration claim). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/docs/2-Workflows-Integration.md. Fetched 2026-08-12.

[19] `docs/1-Overview-Of-MARTI.md` at the pinned commit. https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/docs/1-Overview-Of-MARTI.md. Fetched 2026-08-12.

[20] `docs/3-Reward-And-Training.md` at the pinned commit. https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/docs/3-Reward-And-Training.md. Fetched 2026-08-12.

[21] `docs/4-Experiments-Of-MARTI.md` at the pinned commit. https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/docs/4-Experiments-Of-MARTI.md. Fetched 2026-08-12.

[22] `marti/trainer/multi_agent_ppotrainer.py` at the pinned commit (wandb/TensorBoard initialization logic). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/marti/trainer/multi_agent_ppotrainer.py. Fetched 2026-08-12.

[23] `marti/trainer/ray/ppo_actor.py` at the pinned commit (short_status logged-metric keys). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/marti/trainer/ray/ppo_actor.py. Fetched 2026-08-12.

[24] `marti/trainer/multi_agent_ppotrainer.py` at the pinned commit (per-agent metric namespacing, sample-table logging, eval logging). Same source as [22].

[25] `marti/cli/train_ppo_ray.py` at the pinned commit (save_path, ckpt_path, save_steps, save_hf_ckpt, max_ckpt_num, max_ckpt_mem, load_checkpoint flags). Same source as [6].

[26] `marti/utils/deepspeed/deepspeed.py` at the pinned commit (`save_ckpt`/`load_ckpt` DeepSpeed-native checkpoint contract, `save_model` HF-format save). https://raw.githubusercontent.com/TsinghuaC3I/MARTI/a2fe2c7b9ec46cf24769c90575c51d847f41d04e/marti/utils/deepspeed/deepspeed.py. Fetched 2026-08-12.

[27] `marti/utils/deepspeed/deepspeed.py` at the pinned commit (PEFT/LoRA `adapter_model.bin` save path under ZeRO-3/tensor-parallel). Same source as [26].

[28] GitHub Issues search for "openrlhf.cli" scoped to this repository (zero results). https://api.github.com/search/issues?q=repo:TsinghuaC3I/MARTI+openrlhf.cli. Fetched 2026-08-12.
