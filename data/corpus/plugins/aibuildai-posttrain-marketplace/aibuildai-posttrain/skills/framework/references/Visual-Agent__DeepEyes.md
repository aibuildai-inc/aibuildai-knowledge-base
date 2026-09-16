# DeepEyes

A research-code fork of verl that adds an agentic, tool-using rollout loop for training vision-language models to "think with images" - install it as verl, then flip one config switch on.

DeepEyes is the reproduction code for the paper "DeepEyes: Incentivizing 'Thinking with Images' via Reinforcement Learning", which trains a model "end-to-end with reinforcement learning without requiring pre-collected reasoning data for cold-start supervised fine-tuning" so that it "learns to strategically ground its reasoning in visual information" [1]. The paper's own results give the reference numbers this training recipe is meant to reproduce: a 7B model trained this way reaches 90.1% accuracy on V* (a gain of 18.9 percentage points over the untrained Qwen2.5-VL-7B backbone) and improves HR-Bench-4K and HR-Bench-8K by 6.3 and 7.3 percentage points respectively [1]. It is maintained by the Visual-Agent GitHub organization and the paper's authors (Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, Xing Yu) [1][2]. Structurally it is a frozen fork of ByteDance Seed's verl - its own package still installs and identifies itself as `verl` [3] - onto which DeepEyes overlays an `agent:` config block and a pluggable tool-environment registry: a sample's `env_name` field selects a `ToolBase` subclass (e.g. an image-zoom tool) that the rollout loop calls mid-generation before handing control back to the policy [2]. It lives at https://github.com/Visual-Agent/DeepEyes [2].

**When to pick it**: pick DeepEyes specifically to reproduce or extend its paper's image-grounding agentic-RL recipe (Qwen2.5-VL policy, LLM-as-judge reward, per-sample tool routing) on top of an already-familiar verl workflow; its own README frames it as reproduction code for that one paper, not a general-purpose framework [2]. For post-training methods and scale outside that use case, prefer upstream verl itself or another framework card in this deck (cross-reference; not covered here) - DeepEyes' verl base was last synced with upstream `main` on April 23, 2025, so it trails current verl by many months [2]. A newer, broader sibling project, DeepEyesV2 (code execution and search added, arXiv:2511.05271), is a separate repository and is out of scope for this card [2].

**Methods it ships**: DeepEyes' own contribution is the `agent:` block in `verl/trainer/config/ppo_trainer.yaml` - `activate_agent` (default `False`, which the README states makes the run "identical to the original version of verl"), `max_turns`, `concurrent_workers`, `tool_name_key: env_name`, and `vl_model_path` [2][4] - plus a registry of `ToolBase` tool/environment plugins imported in `verl/workers/agent/__init__.py`: five versioned `visual_toolbox` image-zoom tools, two RAG-engine variants, three `vl_agent` visual-agent variants, an MM search engine, and a FrozenLake tool [5]. The paper's own reward is an LLM-as-judge implemented in `verl/utils/reward_score/vl_agent.py`, which calls an external OpenAI-compatible endpoint (`LLM_AS_A_JUDGE_BASE`) and reads the served model's id dynamically from that endpoint's `/models` list rather than hardcoding a name [6]; the README specifies the judge deployed in that role as Qwen-2.5-72B-Instruct [2]. This plugin layer is method-agnostic: it is exercised in the README's own quickstart with GRPO-style KL-loss settings, but the algorithm is still selected through inherited verl config, which implements PPO/GAE, GRPO (with a Dr.GRPO variant), REINFORCE++ (plain and baseline), RLOO, and ReMax in `verl/trainer/ppo/core_algos.py` [7] - none of that algorithm code was written for DeepEyes. The shortlist row's DAPO (`recipe/dapo/src/dapo_ray_trainer.py`) and REWARD (`recipe/r1/reward_score.py`) pointers are likewise stock verl recipes carried over unmodified from the upstream fork, not DeepEyes originals [8][9]; treat them as inherited surface, not part of this project's method.

**Scale it handles**: single GPU up to multi-node, unchanged from base verl's FSDP or Megatron-LM (v0.11) training backends and vLLM/SGLang rollout backends [10]; multi-node here means a manually started Ray cluster (`ray start --head`, then `ray start --address=<head>` on each worker, jobs submitted with `ray job submit`) [11], which is also how the LLM-judge's separate vLLM server and the training cluster are wired together. DeepEyes' README publishes a documented benchmark recommendation, not a mechanism-only claim: "no less than 32 GPUs (4 nodes x 8 GPUs) for 7B training, and no less than 64 GPUs (8 nodes x 8 GPUs) for 32B training," each node with "no less than 1200GB CPU RAM" because the V* and ArxivQA image datasets are high resolution [2].

**Install**: `pip install -e .` from a clone, then `bash scripts/install_deepeyes.sh` for DeepEyes-specific extras [2]. There is no PyPI package, and the repository has no GitHub releases or tags [12][13]; the version string in `verl/version/version` at the pinned commit reads `0.2.0.dev` [30], and with no release or tag to resolve, the only reproducible pin available is the commit SHA itself. This card pins `11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1`, which was the tip of `main` at fetch time (committed 2025-11-14) [14][15]. `pyproject.toml` at that commit states `requires-python = ">=3.8"` [16]; the in-repo doc `docs/start/install.rst` (unchanged from upstream verl, not re-verified for this fork) separately gives a Python floor of 3.9 and a CUDA floor of 12.1 [10] - the two disagree, and neither figure is DeepEyes-specific. Licence is Apache-2.0 [17]. Load-bearing pins at this commit: `setup.py`'s `vllm` extra caps `vllm<=0.8.3` and `tensordict<=0.6.2` [3], but DeepEyes' own `scripts/install_deepeyes.sh` overrides this with the exact pins `vllm==0.8.2` and `tensordict==0.6.2` [18] - install the script after the base extras, in that order, or the two can fight. Base `install_requires` in `setup.py` pulls in `transformers`, `accelerate`, `peft`, and `wandb` with no version constraint, plus `ray[default]>=2.10` and `pyarrow>=15.0.0` as the only other floors, and `tensordict<=0.6.2` as the only other cap; torch is not listed at all and arrives however the environment provides it [3]. No CUDA/hardware minimum is stated in DeepEyes' own files; the 12.1 figure above is inherited install.rst text, not re-verified against this fork [10].

**Maintained by**: the Visual-Agent GitHub organization and the DeepEyes paper authors [1][2]; not archived [15]. The repository's most recent commit at fetch time, `11d20c6` (committed 2025-11-14), is a merged pull request that adds the DeepEyesV2 pointer to the README [14][2]; the GitHub API separately reports the repository's last push as 2025-11-20, a few days later, but no commit fetched for this card is dated to confirm what that later push changed [15]. Star and fork counts are not used here to rank the project.

## Quick start

DeepEyes' own quickstart is not a single-command, single-GPU run: it assumes a separate multi-GPU vLLM judge server plus a Ray training cluster. From the README, the full sequence is [2]:

```bash
# Environment setup
pip install -e .
bash scripts/install_deepeyes.sh

# Step 1: serve the LLM judge (Qwen2.5-72B-Instruct)
huggingface-cli download --resume-download https://huggingface.co/Qwen/Qwen2.5-72B-Instruct \
    --local-dir /path/to/your/local/filedir --local-dir-use-symlinks False
vllm serve /path/to/your/local/filedir \
    --port 18901 --gpu-memory-utilization 0.8 --max-model-len 32768 \
    --tensor-parallel-size 8 --served-model-name "judge" \
    --trust-remote-code --disable-log-requests

# Step 2: build a Ray cluster and download the training data
# https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k

# Step 3: launch training
wandb login
export LLM_AS_A_JUDGE_BASE="http://your.vllm.machine.ip:18901/v1"
export WORLD_SIZE=8
bash examples/agent/final_merged_v1v8_thinklite.sh       # 7B config
bash examples/agent/final_merged_v1v8_thinklite_32b.sh   # 32B config
```

For a genuinely single-GPU, self-contained run, use the inherited base-verl quickstart instead (frozen in this fork at `docs/start/quickstart.rst`, requiring "GPU with at least 24 GB HBM"): it downloads the GSM8K dataset and a small model, then trains PPO with `python3 -m verl.trainer.main_ppo` and about a dozen `data.*`/`actor_rollout_ref.*`/`trainer.*` overrides, none of which touch the `agent:` block [19]. This path exercises inherited verl, not DeepEyes' agentic layer.

## Start it

- Single GPU: run the inherited verl quickstart above with `activate_agent` left at its default `False` [4][19].
- Agentic training is multi-node by construction in the published recipe: `examples/agent/final_merged_v1v8_thinklite.sh` sets `WORLD_SIZE` nodes x 8 GPUs, `agent.activate_agent=True`, `agent.max_turns`, `agent.concurrent_workers`, and points `env.env_name`-tagged samples at `visual_toolbox_v2`; it also sets `actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra']`, which is a non-default addition (base config omits `hf_model`, see Save it) [20].
- The cluster is built manually with Ray, not through a bundled launcher script: `ray start --head --dashboard-host=0.0.0.0` on the head node, `ray start --address=<head-address>` on each worker, then `ray job submit --address="http://127.0.0.1:8265" --runtime-env=verl/trainer/runtime_env.yaml -- python3 -m verl.trainer.main_ppo ...` with `trainer.n_gpus_per_node` and `trainer.nnodes` set to match [11].
- The `agent:` block is the config surface DeepEyes adds on top of stock verl's `ppo_trainer.yaml`; every other field (batch sizes, optimizer, FSDP/Megatron wrap policy, KL settings) is unmodified verl [4]. `single_response_max_tokens` (default 32768) is documented by the README as needing to be set to half or one third of `max_response_length` to avoid the left padding problem [2] - a DeepEyes-specific tuning trap, not a verl default concern.
- Effective batch size arithmetic is unchanged from verl: `data.train_batch_size` (global) versus each worker's `ppo_micro_batch_size_per_gpu` / dynamic-batching `ppo_max_token_len_per_gpu`, as seen in `examples/split_placement/config/ppo_trainer_split.yaml` [21].
- Out-of-memory first aid is the inherited verl guidance: the quickstart page recommends `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1` and `critic.ppo_micro_batch_size_per_gpu=1` for GPUs under 32GB HBM [19]; DeepEyes' README separately warns that its own image datasets can exhaust host RAM, recommending "no less than 1200GB CPU RAM" per node [2] - this is a host-memory floor, not a GPU-memory knob.
- A closed-issue trap on the tool/answer boundary: contributor JaaackHongggg explained in issue #58 (2025-06-19) that if a model's output contains both `<tool_call>` and `<answer>` in the same turn, DeepEyes truncates on `</tool_call>` and, when `<answer>` is also present, extracts the `<answer>` content and exits without executing the tool call - a scripted precedence rule, not a bug to work around [22].

## Watch it

This section is mechanics only; what a metric value means for GRPO, PPO, or RLOO training health lives on those methods' own cards.

- **Enable it**: the README states training scripts use "both wandb and RL Logging Board... to visualize the training dynamics" [2]. Logging is configured through inherited verl's `Tracking` class (`verl/utils/tracking.py`), whose `supported_backend` list is `["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "rl_logging_board"]`; `Tracking` takes an explicit `default_backend` list (sourced from `trainer.logger`) and activates each backend named in it, with no implicit fallback in the class itself [23]. The shipped default in `ppo_trainer.yaml` is `logger: ['console', 'wandb']` [4], and the 7B launch script overrides it to `['console','wandb','rl_logging_board']` [20].
- **DeepEyes-specific backend**: `rl_logging_board` is this fork's own addition, wired to `RLLoggingBoardLogger` (`verl/utils/rl_logging_board_utils.py`); its `.log()` call requires a `tokenizer` keyword argument and raises `ValueError("Please provide a tokenizer.")` if omitted [24]. It writes per-sample generation records - decoded prompt and response text, a per-token list of decoded response tokens, logprobs, token-level rewards, the scalar reward, and optional `ground_truth`, `values`, and reward-model output fields - to a `rollout_data_rank0.jsonl` file under `<rl_logging_board_dir>/<project_name>/<experiment_name>/`, which is this fork's sample-level generation logging [24].
- **Metric names**: inherited verl's `verl/trainer/ppo/metric_utils.py` emits `critic/score/*`, `critic/rewards/*`, `critic/advantages/*`, `critic/returns/*`, and (when `use_critic`) `critic/values/*` and `critic/vf_explained_var`, plus `response_length/*`, `prompt_length/*`, `timing_s/*`, `timing_per_token_ms/*`, `perf/total_num_tokens`, `perf/time_per_step`, and `perf/throughput` [25]. DeepEyes does not add its own scalar metric names on top of this list; its logging addition is the sample-level JSONL above, not new scalars [24][25].
- **Evaluation during training**: not covered by this card's reading of the training config; evaluation is documented as a separate offline step (see below), not an in-loop `eval_dataset` field specific to this fork [26].
- **Stopping**: no DeepEyes- or verl-specific stopping rule, threshold, or patience field was found in `verl/trainer/config/ppo_trainer.yaml` or the metric/tracking files read for this card; shapes are published (the metric names above), thresholds are not [4][25].
- **A reward-computation trap surfaced in issue #83** (opened 2025-07-07, closed 2025-07-09): multiple non-maintainer users report that `visual_toolbox_v2`'s per-tool-call reward appears to always be zero and that only the final `<answer>`-bearing turn receives a nonzero reward, which GRPO then broadcasts back across all tokens of the response; no maintainer account replied in the thread as read for this card, so treat this as an open, user-reported behavior rather than a confirmed fix or an acknowledged bug [27].

## Save it

- Checkpoint mechanics are unmodified verl. FSDP checkpoints write per-rank sharded files - `model_world_size_{ws}_rank_{r}.pt`, `optim_world_size_{ws}_rank_{r}.pt`, `extra_state_world_size_{ws}_rank_{r}.pt` - under `checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_steps_${i}/{actor,critic}/`, controlled by `trainer.default_local_dir` and `trainer.save_freq` [28][4].
- `checkpoint.contents` selects what a checkpoint saves; the docs state plainly that "checkpoint.contents field has no effect to FSDP checkpoint except hf_model, the other 3 fields are binded together to save and load. We recommend to include model, optimizer and extra all" [28]. The shipped default in `ppo_trainer.yaml` is `contents: ['model', 'optimizer', 'extra']` - no `hf_model` - meaning the default run saves only the sharded training checkpoint, not full HF-format weights [4]; the published 7B training script explicitly overrides this to `['model', 'hf_model', 'optimizer', 'extra']` [20].
- Because the default omits `hf_model`, a checkpoint saved with default settings cannot be loaded directly with `from_pretrained`; it must be converted first with `scripts/model_merger.py` (`--backend fsdp|megatron`, `--hf_model_path`, `--local_dir`, `--target_dir`, optional `--hf_upload_path`) [28]. This is exactly the trap reported in closed issue #52 ("The trained checkpoints cannot be loaded", opened 2025-06-16): the reporter's own follow-up comment (author association NONE, i.e. not a maintainer, 2025-06-19) states the fix was to run `scripts/model_merger.py` to merge the weights - no maintainer account is recorded as replying in that thread [29].
- `trainer.resume_mode: auto` (default) finds the last checkpoint under `default_local_dir` and resumes from it; `disable` starts from scratch; `resume_path` resumes from `trainer.resume_from_path` explicitly [4].
- Loader handoff: an unconverted FSDP-sharded checkpoint directory is not a full model and is not loadable by a plain `from_pretrained` call; only the `scripts/model_merger.py` output (or a run that included `hf_model` in `checkpoint.contents`) is [28].

## Find it in the docs

DeepEyes ships no separate documentation site of its own; "documentation" here means the repository's README and its frozen copy of verl's Sphinx docs tree.

- The README (https://github.com/Visual-Agent/DeepEyes/blob/main/README.md) is the primary and only DeepEyes-specific source: Quick Start, the `agent:` config block and field-by-field explanation, "Training on Customized Datasets" (the `env_name` mechanism), "Training with Customized Tools" (subclassing `ToolBase`), and the "Using latest VeRL code" note giving the April 23, 2025 upstream sync date [2].
- The rest of the repo's `docs/` tree is upstream verl's Sphinx source, carried over unmodified by this fork and not re-verified against current upstream verl - treat page content there (install requirements, checkpoint layout, multinode setup) as accurate to the April 2025 verl snapshot this fork is based on, not to current verl [2][10].
- Evaluation is a separate, DeepEyes-specific guide at `eval/EVALUATION.md`: `eval_vstar.py` runs model evaluation through a vLLM server with automatic bounding-box processing, `judge_result.py` scores results using a Qwen2.5 72B judge served over vLLM [26] (the README separately names the same role as Qwen-2.5-72B-Instruct [2]), and `watch_demo.ipynb` visualizes trajectories [26].
- Runnable references beyond the README: the `examples/agent/` directory holds the 7B/32B launch scripts and variants for RAG and FrozenLake tool environments; the paired dataset is `ChenShawn/DeepEyes-Datasets-47k` and the released model is `ChenShawn/DeepEyes-7B` on the Hugging Face Hub, both linked from the README and both outside the `visual-agent` Hub org [2].
- No official curated-tutorials page or MCP endpoint is published by this project; the paper's project homepage (linked from the README) is a case-study page, not a docs/lookup surface [2].
- Honest boundary: this is single-paper reproduction code layered on a verl fork frozen at April 2025, with no releases, no PyPI package, and no CI-verified compatibility matrix published for it; anyone needing current verl features or a maintained general framework should use upstream verl instead [2][12][13].

## Sources

Method names (PPO, GRPO, RLOO, REINFORCE++, ReMax, DAPO) are deliberately cited to nothing here; their defining papers live on the methodology cards. Ecosystem tools reached only in passing (Ray, vLLM, wandb, RL Logging Board, FSDP, Megatron-LM, SGLang) are not enumerated as separate references.

[1] DeepEyes paper, arXiv:2505.14362: abstract page https://arxiv.org/abs/2505.14362, and full-text HTML (V*/HR-Bench result figures) https://arxiv.org/html/2505.14362. Both fetched 2026-08-12.

[2] DeepEyes README at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1. https://raw.githubusercontent.com/Visual-Agent/DeepEyes/main/README.md. Fetched 2026-08-12.

[3] DeepEyes `setup.py` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (package name `verl`, install_requires, vllm/tensordict extras, upstream `url`). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/setup.py. Fetched 2026-08-12.

[4] DeepEyes `verl/trainer/config/ppo_trainer.yaml` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (the `agent:` block, `trainer.resume_mode`, `checkpoint.contents` default). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/verl/trainer/config/ppo_trainer.yaml. Fetched 2026-08-12.

[5] DeepEyes `verl/workers/agent/__init__.py` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (registered tool/environment classes). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/verl/workers/agent/__init__.py. Fetched 2026-08-12.

[6] DeepEyes `verl/utils/reward_score/vl_agent.py` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (LLM-as-judge reward implementation). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/verl/utils/reward_score/vl_agent.py. Fetched 2026-08-12.

[7] DeepEyes `verl/trainer/ppo/core_algos.py` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (advantage estimators: GAE, GRPO, REINFORCE++, RLOO, ReMax). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/verl/trainer/ppo/core_algos.py. Fetched 2026-08-12.

[8] DeepEyes `recipe/dapo/src/dapo_ray_trainer.py` and `recipe/dapo/src/config/dapo_trainer.yaml` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (stock verl DAPO recipe, unmodified). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/recipe/dapo/src/dapo_ray_trainer.py. Fetched 2026-08-12.

[9] DeepEyes `recipe/r1/reward_score.py` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (stock verl R1-recipe reward dispatcher, unmodified). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/recipe/r1/reward_score.py. Fetched 2026-08-12.

[10] DeepEyes `docs/start/install.rst` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (inherited verl install doc: Python/CUDA floors, backend choices, Docker images). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/docs/start/install.rst. Fetched 2026-08-12.

[11] DeepEyes `docs/start/multinode.rst` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (manual Ray cluster setup and job submission). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/docs/start/multinode.rst. Fetched 2026-08-12.

[12] GitHub API, releases list for Visual-Agent/DeepEyes (empty). https://api.github.com/repos/Visual-Agent/DeepEyes/releases. Fetched 2026-08-12.

[13] GitHub API, tags list for Visual-Agent/DeepEyes (empty). https://api.github.com/repos/Visual-Agent/DeepEyes/tags. Fetched 2026-08-12.

[14] GitHub API, commit detail for 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (committer date 2025-11-14). https://api.github.com/repos/Visual-Agent/DeepEyes/commits/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1. Fetched 2026-08-12.

[15] GitHub API, repository metadata for Visual-Agent/DeepEyes (pushed_at, archived, created_at, license). https://api.github.com/repos/Visual-Agent/DeepEyes. Fetched 2026-08-12.

[16] DeepEyes `pyproject.toml` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (`requires-python = ">=3.8"`). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/pyproject.toml. Fetched 2026-08-12.

[17] DeepEyes `LICENSE` file at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (Apache-2.0 full text). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/LICENSE. Fetched 2026-08-12.

[18] DeepEyes `scripts/install_deepeyes.sh` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (`vllm==0.8.2`, `tensordict==0.6.2`, and other DeepEyes-specific pins). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/scripts/install_deepeyes.sh. Fetched 2026-08-12.

[19] DeepEyes `docs/start/quickstart.rst` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (inherited base-verl single-GPU PPO/GSM8K quickstart, OOM first aid). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/docs/start/quickstart.rst. Fetched 2026-08-12.

[20] DeepEyes `examples/agent/final_merged_v1v8_thinklite.sh` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (7B training launch script: agent/trainer/checkpoint overrides). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/examples/agent/final_merged_v1v8_thinklite.sh. Fetched 2026-08-12.

[21] DeepEyes `examples/split_placement/config/ppo_trainer_split.yaml` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (inherited verl split-placement PPO config; the shortlist row's PPO pointer). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/examples/split_placement/config/ppo_trainer_split.yaml. Fetched 2026-08-12.

[22] GitHub issue #58, "How to handle <tool_call> with <answer> in reward calculation", closed; comment by JaaackHongggg (author association CONTRIBUTOR), 2025-06-19. https://api.github.com/repos/Visual-Agent/DeepEyes/issues/58/comments. Fetched 2026-08-12.

[23] DeepEyes `verl/utils/tracking.py` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (`Tracking` class, `supported_backend` list, `default_backend` activation with no implicit fallback). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/verl/utils/tracking.py. Fetched 2026-08-12.

[24] DeepEyes `verl/utils/rl_logging_board_utils.py` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (`RLLoggingBoardLogger`, tokenizer requirement, JSONL output). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/verl/utils/rl_logging_board_utils.py. Fetched 2026-08-12.

[25] DeepEyes `verl/trainer/ppo/metric_utils.py` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (inherited verl metric names). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/verl/trainer/ppo/metric_utils.py. Fetched 2026-08-12.

[26] DeepEyes `eval/EVALUATION.md` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (evaluation scripts and workflow). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/eval/EVALUATION.md. Fetched 2026-08-12.

[27] GitHub issue #83, "`env_reward` does not match the formula in the paper!", closed; comments by 2018hahazhufeng, mengzchen, FolSpark (all author association NONE), 2025-07-09 to 2025-07-11. https://api.github.com/repos/Visual-Agent/DeepEyes/issues/83/comments. Fetched 2026-08-12.

[28] DeepEyes `docs/advance/checkpoint.rst` at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (checkpoint.contents contract, FSDP/Megatron directory layout, `scripts/model_merger.py` usage). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/docs/advance/checkpoint.rst. Fetched 2026-08-12.

[29] GitHub issue #52, "The trained checkpoints cannot be loaded", closed; comment by Daisy-Zhang (author association NONE), 2025-06-19. https://api.github.com/repos/Visual-Agent/DeepEyes/issues/52/comments. Fetched 2026-08-12.

[30] DeepEyes `verl/version/version` file at commit 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1 (`0.2.0.dev`). https://raw.githubusercontent.com/Visual-Agent/DeepEyes/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/verl/version/version. Fetched 2026-08-12.
