# OpenRLHF

A Ray + vLLM RLHF/agentic-RL framework: one hierarchical CLI drives PPO-family RL, plus SFT/RM/DPO, from single GPU to multi-node clusters.

OpenRLHF combines a Ray-orchestrated, vLLM-accelerated distributed architecture with a single agent-based execution pipeline that separates the RL algorithm from single-turn or multi-turn rollout mode [1]. It is built and maintained by the OpenRLHF Organization on GitHub [2], with the DeepSpeed-native `MuonWithAuxAdam` optimizer and the hierarchical CLI shipped as of the 0.10.2 line [1]. Its API surface is a set of CLI entry points (`openrlhf.cli.train_ppo_ray`, `train_sft`, `train_rm`, `train_dpo`) launched via `ray job submit` or `deepspeed --module`, configured entirely through dotted-prefix flags (`--actor.*`, `--ds.*`, `--vllm.*`, ...) rather than a Python trainer-class API [1]. It lives at https://github.com/OpenRLHF/OpenRLHF [2].

**When to pick it**: online RLVR/agentic RL at cluster scale, when the workload needs Ray to place and time-slice Actor/Critic/Reward/Reference/vLLM across GPUs and you want algorithm choice (PPO, REINFORCE++, GRPO, RLOO, ...) to be a one-flag switch rather than a different trainer class per method [1]. Contrast with trl, whose per-method trainer classes launch through Accelerate on a single machine up to multi-node without Ray's placement layer, and which additionally ships KTO, offline preference methods (CPO/ORPO/BCO), and distillation that OpenRLHF's own docs say have been removed from its upstream (see Methods it ships) [3].

**Methods it ships**: RL trainer (`train_ppo_ray`, single entry point) with the estimator switched by `--algo.advantage.estimator`: `gae` (PPO, full critic, default), `reinforce` (REINFORCE++, critic-free), `reinforce_baseline` (REINFORCE++-baseline, mean-reward baseline, no per-prompt std), `rloo` (leave-one-out baseline), `group_norm` (GRPO, per-group mean/std normalization plus KL loss), `dr_grpo` (Dr. GRPO, GRPO without the local mean/std normalization); `rloo`, `reinforce_baseline`, and `group_norm` all require `--rollout.n_samples_per_prompt > 1` [4]. Separate non-RL trainers: SFT (`train_sft`), Reward Model (`train_rm`), DPO (`train_dpo`) [5]. The docs state plainly that "earlier versions of OpenRLHF shipped with KTO, PRM, Knowledge Distillation, and batch_inference-based iterative workflows (rejection sampling, iterative DPO, conditional SFT)" and that "these modules have been removed from the upstream codebase" — pin an older release to get them [5]. Reading v0.10.2 vs v0.10.4 release notes was out of scope for this card; the removal note is dated only to the current live docs page (fetched 2026-08-10), not to a specific release. VLM RLHF (image inputs through the same agent pipeline, e.g. Qwen3.5, auto-detected via the HF config's `vision_config` field) and multi-turn agent RL are additive capabilities on the RL trainer, not separate CLI entry points; VLM training does not support `--ds.packing_samples` and is documented as untested with `ForConditionalGeneration`-style architectures [4].

**Scale it handles**: single GPU (`deepspeed --module openrlhf.cli.train_sft ...` with no Ray) up to multi-node Ray clusters launched with `ray start --head` on one node and `ray start --address <head>` on the rest, or under SLURM via a job script that resolves node hostnames with `scontrol show hostnames` and starts Ray per node [6]. Sharding is DeepSpeed ZeRO (`--ds.zero_stage`), with ZeRO-3 training directly from HuggingFace checkpoints with no conversion step [1]. GPU placement across roles is either full colocation (`--train.colocate_all` with vLLM/DeepSpeed sleep-mode time-slicing — the "Hybrid Engine") or a distributed mode with separate GPU groups per role, which the Performance Tuning page calls "a fallback for very large models or mixed-hardware clusters where colocation isn't viable" [7]. Async training (`--train.async_enable`) overlaps rollout and training through a bounded queue, and Partial Rollout (`--train.partial_rollout_enable`) additionally overlaps weight sync with generation [8]. RingAttention sequence parallelism (`--ds.ring_attn_size`) is the documented long-context mechanism; the dedicated Sequence Parallelism page states the mechanism and its flags but publishes no benchmark numbers for reachable context length or multi-node throughput [9].

**Install**: `pip install openrlhf` (core) or `pip install openrlhf[vllm]` for the vLLM extra; latest PyPI release is 0.10.4, uploaded 2026-06-08 [10]. Python floor is `>=3.10`, no declared licence classifier in the PyPI metadata (the repository's LICENSE file is Apache-2.0, per the shortlist) [10][2]. The v0.10.4 git tag resolves to commit `ad1796e6`, distinct from and older than the repository's newest push at commit `bc71bb19464aca306b33080b2d2bb45d154e2f49` (2026-07-14) — dependency floors below are read at the `ad1796e6` release commit, not the newer push [11]. `requirements.txt` at that commit pins `deepspeed==0.19.1`, `transformers==5.7.0`, `ray[default]==2.55.0`, `flash-attn==2.8.3`, `grpcio>=1.74.0`, `huggingface_hub>=1.0.0`; torch itself is unpinned in that file, arriving transitively [12]. `setup.py` at the same commit defines extras `vllm` (`vllm==0.22.1`), `vllm_latest` (`vllm>0.22.1`), `ring` (`ring_flash_attn`), and `liger` (`liger_kernel`) [13]. The Muon optimizer additionally requires DeepSpeed >= 0.18.2 per the docs' install note [1]. No CUDA/hardware minimum is stated in `setup.py` or the install docs beyond an `Environment :: GPU :: NVIDIA CUDA` classifier and a recommendation to install inside an NVIDIA PyTorch container [13][1].

**Maintained by**: the OpenRLHF Organization on GitHub, not archived, with a push on 2026-07-14 and a v0.10.4 release published 2026-06-08 [2][14]. The README documents the project accepting an NVIDIA-backed "Molt" training backend as a news item, and the docs' own What's-New-style highlights list the 0.10.2 hierarchical CLI and Muon optimizer as recent additions [15][1].

## Quick start

Smallest complete RL run from the docs' own first-run walkthrough — REINFORCE++-baseline on math reasoning with a Python reward function, no reward-model training required, on 4 GPUs [1]:

```bash
# 1) start ray on the head node
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4

# 2) submit the training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor.model_name_or_path Qwen/Qwen3-4B-Thinking-2507 \
   --reward.remote_url examples/python/math_reward_func.py \
   --data.prompt_dataset zhuzilin/dapo-math-17k \
   --data.input_key prompt \
   --data.label_key label \
   --data.apply_chat_template \
   --ds.packing_samples \
   --ref.num_nodes 1 --ref.num_gpus_per_node 4 \
   --actor.num_nodes 1 --actor.num_gpus_per_node 4 \
   --vllm.num_engines 2 --vllm.tensor_parallel_size 2 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.7 \
   --vllm.enable_sleep --ds.enable_sleep \
   --vllm.sync_backend nccl --vllm.enforce_eager \
   --algo.advantage.estimator reinforce_baseline \
   --algo.kl.use_loss --algo.kl.estimator k2 --algo.kl.init_coef 1e-5 \
   --rollout.batch_size 128 --rollout.n_samples_per_prompt 8 \
   --train.batch_size 1024 \
   --data.max_len 8192 --rollout.max_new_tokens 4096 \
   --ds.zero_stage 3 --ds.param_dtype bf16 \
   --actor.gradient_checkpointing_enable \
   --actor.adam.lr 5e-7 \
   --ckpt.output_dir ./exp/Qwen3-4B-Thinking
```

Smallest SFT run, no Ray involved, from the non-RL trainer page [5]:

```bash
deepspeed --module openrlhf.cli.train_sft \
   --model.model_name_or_path meta-llama/Meta-Llama-3-8B \
   --data.dataset Open-Orca/OpenOrca \
   --data.input_key question \
   --data.output_key response \
   --data.input_template $'User: {}\nAssistant: ' \
   --data.max_samples 500000 \
   --data.max_len 2048 \
   --ds.packing_samples \
   --train.batch_size 256 \
   --train.micro_batch_size 8 \
   --train.max_epochs 1 \
   --adam.lr 5e-6 \
   --ds.zero_stage 2 \
   --ds.param_dtype bf16 \
   --ds.attn_implementation flash_attention_2 \
   --model.gradient_checkpointing_enable \
   --ckpt.output_dir ./checkpoint/llama3-8b-sft \
   --ckpt.save_steps -1 \
   --logger.logging_steps 1 \
   --eval.steps -1 \
   --logger.wandb.key {wandb_token}
```

## Start it

- One GPU, no Ray: the SFT/RM/DPO trainers launch directly through `deepspeed --module openrlhf.cli.train_sft|train_rm|train_dpo ...` [5].
- Several GPUs, one node, RL: start Ray (`ray start --head --num-gpus N`) then `ray job submit ... -- python3 -m openrlhf.cli.train_ppo_ray ...` as in the Quick start recipe [1]. GPU placement is set per role with `--actor.num_nodes`/`--actor.num_gpus_per_node` and the equivalents for `--ref.*`, `--critic.*`, `--reward.*`, `--vllm.num_engines`/`--vllm.tensor_parallel_size` [1][8].
- Multi-node / SLURM: a job script resolves node hostnames with `scontrol show hostnames "$SLURM_JOB_NODELIST"`, starts the Ray head on the first node with `ray start --head --node-ip-address=$ip --port=$port --block`, and starts Ray workers on the rest with `ray start --address $ip_head --block`, before submitting the same `ray job submit` command; a parallel non-Ray SLURM path launches `deepspeed`/`torchrun` directly with `--nnodes $SLURM_NNODES --node_rank $SLURM_PROCID` for SFT/RM/DPO [6].
- Generation layout for RL is a start-time choice: Hybrid Engine (`--train.colocate_all --vllm.enable_sleep --ds.enable_sleep`) time-slices vLLM and DeepSpeed on the same GPUs via memory sleep mode and weight sync over NCCL (`--vllm.sync_backend nccl`); the alternative is a distributed layout with separate GPU groups per role, which the Performance Tuning page calls a fallback for very large models or mixed hardware [7]. `--train.async_enable` is documented as incompatible with `--vllm.enable_sleep` [16].
- Effective RL batch: the docs give `train.batch_size = rollout.batch_size * rollout.n_samples_per_prompt` as "a common choice," not an enforced constraint [7].
- Config surface is entirely CLI flags under named dotted sections (`--ds.*`, `--vllm.*`, `--rollout.*`, `--data.*`, `--train.*`, `--eval.*`, `--ckpt.*`, `--logger.*`, `--algo.*`, `--actor.*`, `--critic.*`, `--ref.*`, `--reward.*`) introduced as the 0.10.2 hierarchical CLI; old flat flags (`--pretrain`, `--zero_stage`, `--vllm_num_engines`, `--learning_rate`, ...) no longer parse and argparse errors out on them [17][18]. The docs do not describe this hierarchical CLI as changing any prior numeric default (e.g., precision) — it is a flag-naming migration, not a defaults change, per the pages read for this card [1][19].
- Out-of-memory first aid, priority order from the tuning guide: enable `--ds.packing_samples` and `--actor.gradient_checkpointing_enable`; reduce `--train.micro_batch_size`/`--rollout.micro_batch_size`; lower `--vllm.gpu_memory_utilization` (e.g. 0.6 -> 0.5 -> 0.4); enable `--ds.adam_offload` and raise `--ds.zero_stage` (2 -> 3); disable colocation and move to distributed mode [7]. `--ds.adam_offload` is documented as incompatible with `--actor.optim muon`/`--critic.optim muon`, since DeepSpeed's Muon keeps optimizer state on GPU [7][19].

## Watch it

Mechanics only — what a metric's shape means for a given RL algorithm lives on that method's card, not here.

- **Enable it**: `--logger.wandb.key {token-or-True}` (Wandb; `True` reuses a prior `wandb login`) with `--logger.wandb.org`/`.group`/`.project`/`.run_name`, or `--logger.tensorboard_dir {logdir}` for TensorBoard; `--logger.logging_steps` sets log cadence [20]. No flag enables logging by default: the trainer only constructs a `WandbLogger`/`TensorboardLogger` when `--logger.wandb.key`/`--logger.tensorboard_dir` is set, leaving both `None` otherwise, so a run keeps no external record beyond the console progress bar unless one of these two flags is set [21].
- **PPO-trainer metric names**, read from the `train_ppo_ray` module's own PPO actor and trainer classes at the `ad1796e6` release commit, since the docs pages fetched for this card do not enumerate the full field list: `policy_loss` (surfaced to the progress bar and logs as `act_loss`), `reward`, `return`, `response_length` (`gen_len`), `total_length` (`tot_len`), `kl`, `actor_lr` (`act_lr`), `actor_grad_norm` (`grad_norm`), and, when entropy regularization is on, `entropy_loss` (`ent_loss`) [22]. The trainer additionally logs `timing/make_experience`, `timing/ppo_train`, `timing/broadcast`, `timing/generation`, `timing/step_total`, `generated_samples`, and, when dynamic filtering is enabled, `dynamic_filtering_pass_rate` [21]. These are field names read directly out of `openrlhf/trainer/ppo_actor.py` and `openrlhf/trainer/ppo_trainer.py` at commit `ad1796e6`, not off a docs page — treat them as ahead of any doc page that lists a shorter set [22][21].
- **Sample-level logging**: not documented on any docs page read for this card, but present in code at the `ad1796e6` release commit: each PPO training step decodes the first sequence of the first experience in the batch, pairs it with its scalar reward, logs it to the console as `Sample: [...]`, and also stores it under the `generated_samples` key of the same `status` dict that is passed unfiltered into `WandbLogger.log_train`/`TensorboardLogger.log_train` [22][21]. Those two classes explicitly branch on the `generated_samples` key: the Wandb path appends it as a row (`global_step`, text, reward) to a growing `wandb.Table` logged under `train/generated_samples`, and the TensorBoard path formats it as `Sample:\n{text}\n\nReward: {reward}` and calls `add_text` under the same tag — so one decoded sample (not the full batch) does reach both dashboards each `--logger.logging_steps` interval whenever a logger backend is enabled, with no flag to disable or resize it [23].
- **Evaluate during training**: `--eval.dataset` sets the eval dataset path (paired with `--data.prompt_dataset` for PPO reward-function runs), `--eval.steps` sets the cadence in training steps (`-1` disables it), `--eval.split` picks the eval split, and for PPO `--eval.temperature`/`--eval.n_samples_per_prompt` set the eval-rollout sampling parameters; `--eval.dataset`/`--eval.steps` together are what best-checkpoint tracking needs to have a metric to compare against (see Save it) [19].
- **Stopping**: search run 2026-08-10 over the three pages most likely to carry a threshold — Performance Tuning, Troubleshooting, and Common CLI Options — found no early-stopping flag, patience value, or reward-threshold field on any of them; Performance Tuning states "start from a known-good recipe ... and adjust one knob at a time" but names no stopping rule [7][19][18]. Shapes for individual signals (kl, reward, grad_norm) are published as field names above; no threshold on any of them is published.

## Save it

- OpenRLHF saves four kinds of state at each checkpoint: DeepSpeed-format sharded model weights; optimizer and scheduler state; dataset progress via a resumable `DistributedSampler`; and, when `--ckpt.save_hf` is set, an additional HuggingFace-format model export [24].
- `--ckpt.path` is the DeepSpeed-checkpoint directory, written every `--ckpt.save_steps` global steps (`-1` = never; for PPO these are model-update steps, not mini-batches); `--ckpt.max_num`/`--ckpt.max_mem` cap retained checkpoint count/size [24].
- `--ckpt.disable_ds` "skip[s] DeepSpeed checkpoints to save disk — training progress is no longer recoverable (only HF-format models are kept)" [24].
- `--ckpt.load_enable` resumes from `--ckpt.path`, and "gracefully falls back to training-from-scratch if the checkpoint directory exists but contains no valid checkpoint" [24].
- PPO-only: `--ckpt.best_metric_key` selects the eval metric used for best-checkpoint saving (empty string auto-detects the first `pass1` metric, `none` disables it); best-checkpoint tracking requires `--eval.dataset`/`--eval.steps` to be set, and the best checkpoint is written to a separate path from the latest checkpoint so the two don't collide; `--train.enable_ema` saves an additional Exponential-Moving-Average copy of the policy alongside the regular model (`--train.ema_beta`, default 0.992); `--critic.save_value_network` additionally saves the critic [24].
- `--ckpt.output_dir` is always written at the end of training as the final HuggingFace-format model save path, independent of the periodic `--ckpt.path` checkpoints [24].
- Changing ZeRO stage or world size between runs requires converting the DeepSpeed checkpoint to Universal format first (`examples/scripts/ckpt_ds_zero_to_universal.sh`), then resuming with `--ds.use_universal_ckpt` [24].
- LoRA/QLoRA changes what a save IS: SFT, RM, and DPO all support `--ds.lora.rank > 0` (plus `--ds.load_in_4bit` for QLoRA) — Ray + vLLM PPO does not support LoRA at all — and when it is enabled, only the adapter weights are saved, not a full model directory [5]. Reload pairs the adapter with the base model via `openrlhf.cli.lora_combiner --model_path <base> --lora_path <adapter-dir> --output_path <merged-dir>`, which merges the two into a single deployable model [5][19].
- Loader handoff: a `--ckpt.output_dir` or `--ckpt.save_hf` export is HuggingFace-format and directly loadable by any HF-compatible evaluator; a bare `--ckpt.path` DeepSpeed checkpoint (with `--ckpt.disable_ds` unset) is not a HuggingFace model directory and needs `--ckpt.save_hf` or the Universal-checkpoint conversion script to become one [24].

## Find it in the docs

- Address pattern: `https://openrlhf.readthedocs.io/en/<slug>/<page>.html`. Checked 2026-08-10: `en/latest/quick_start.html` returns 200; `en/v0.10.4/quick_start.html` and `en/stable/quick_start.html` both return 404 [25]. Read the Docs' own versions API for this project lists five branch-based builds (`main`, `latest`, and three `hijkzzz-patch-*` branches) and marks only `latest` as active — there is no tag-pinned or "stable" build at all, so every fetch of this docs site is against a live, unpinned branch build, and the page itself renders a stale "0.10.2" version banner even though the current PyPI release is 0.10.4 [25]. Treat any docs claim as current-as-of-fetch-date against the `latest` branch, not against any specific release.
- Page slugs read for this card: `quick_start`, `hybrid_engine`, `async_training`, `performance`, `checkpoint`, `common_options`, `non_rl` (the SFT/RM/DPO page), `multi-node`, `troubleshooting`, and the docs `index` [1][8][16][7][24][19][5][6][18].
- Question-to-slug map: install/CLI migration -> `common_options`; RL algorithm/agent/VLM setup -> not fetched separately for this card, likely under an `agent_training`-named guide referenced from the index; SFT/RM/DPO -> `non_rl`; checkpoint/resume -> `checkpoint`; OOM/argparse/Muon/NCCL errors -> `troubleshooting`; throughput/memory knobs -> `performance`; SLURM/multi-machine -> `multi-node`.
- Runnable references beyond the docs: the repository's `examples/scripts/` tree, which the Hybrid Engine page says the Quick-start RL recipe was adapted from (`train_reinforce_baseline_hybrid_engine.sh`, `train_prorlv2_math_hybrid_engine.sh`) [8]; `examples/python/math_reward_func.py` and `examples/python/reward_func.py` as canonical custom-reward-function references [1]. Known-good smoke-test datasets named in the docs' own runnable examples: `zhuzilin/dapo-math-17k` (RL), `Open-Orca/OpenOrca` (SFT), `OpenRLHF/preference_dataset_mixture2_and_safe_pku` (RM) [1][5].
- Community layer, curated door first: the docs index's own "Resources" list is the curated door, and it names four items — the GitHub repository, a technical report, a slide deck, and one practitioner post, the vLLM-team blog post "Accelerating RLHF with vLLM" — with no dedicated tutorials/blog-aggregation page beyond that short list [1]. No author or date is attached to the vLLM-blog link on the index page itself, so a reader following it should check the post's own byline and version references against their installed 0.10.4 before trusting any numbers in it. The README separately links a Google Slides deck and a ResearchGate technical report as "Learn More" resources, and advertises an NVIDIA-authored "Molt" backend integration as a news item; these are the project's own links, not third-party curation, and are not evaluated further here [15].
- No official MCP endpoint for the docs was found in the pages fetched for this card; the README links a third-party DeepWiki badge (`deepwiki.com/OpenRLHF/OpenRLHF`) for AI-assisted Q&A over the repo, which is not an MCP server and is not verified further here [15].
- Traps, from maintainer/contributor replies in closed issues only: (1) issue #1222 — upgrading `transformers` past 5.5.4 broke the `ring_flash_attn` integration's import of `is_flash_attn_greater_or_equal_2_10`, which `transformers` had moved to a new module path; a COLLABORATOR reply (hijkzzz, 2026-04-19) initially redirected the fix to the upstream `ring-flash-attn` repository, but OpenRLHF shipped its own compatibility shim shortly after — a follow-up comment (allen-dc, 2026-04-21) thanks the maintainers for the quick fix and refines it, and the shim is present in code at the `ad1796e6` release commit, in `openrlhf/models/ring_attn_utils.py`, with a docstring citing issue #1222 by number [26][27]. (2) Issue #1258, closed 2026-07-10 — a CONTRIBUTOR (Functionhx) traced a bug where, for Qwen3.5's hybrid decoder architecture under ZeRO-3, the `set_z3_leaf_modules` call silently drops gradients on roughly 390 of 417 inner per-layer parameters, so a ZeRO-3 RL or SFT run on this architecture trains under 5% of the model without erroring; comparing the v0.10.4 release commit against the repository's current push commit confirms the fix (an added `detect_hybrid=False` argument in `openrlhf/models/actor.py`) landed only after the v0.10.4 release — it is present at push commit `bc71bb19` but not at the `ad1796e6` release commit, so anyone on the pip-installed 0.10.4 release is still exposed [28][11].
- Honest boundary: the removed-trainer note above (KTO, PRM, Knowledge Distillation, batch_inference-based iterative workflows) is the clearest documented boundary — none of those are available on a current install, only on an older pinned release [5]. No hardware/CUDA minimum is stated anywhere in the install docs or `setup.py` beyond a generic NVIDIA-CUDA classifier (see Install) [13][1].

## Sources

[1] OpenRLHF docs, Quick Start / index highlights. https://openrlhf.readthedocs.io/en/latest/quick_start.html and https://openrlhf.readthedocs.io/en/latest/index.html. Live `latest`-branch build, fetched 2026-08-10, page banner reads "0.10.2".

[2] OpenRLHF GitHub repository (API). https://api.github.com/repos/OpenRLHF/OpenRLHF. Fetched 2026-08-10. Repository page: https://github.com/OpenRLHF/OpenRLHF.

[3] trl documentation index (cross-reference only, for the "when to pick it" contrast; not otherwise used in this card). https://huggingface.co/docs/trl/main/en/index. Fetched 2026-08-10 (reused from the trl card's own fetch).

[4] OpenRLHF docs, RL Training Guide (advantage-estimator table and algorithm-specific requirements; Vision-Language Model RLHF section). https://openrlhf.readthedocs.io/en/latest/agent_training.html. Fetched 2026-08-10.

[5] OpenRLHF docs, Supervised & Preference Training (SFT / RM / DPO). https://openrlhf.readthedocs.io/en/latest/non_rl.html. Fetched 2026-08-10.

[6] OpenRLHF docs, Multi-node Training. https://openrlhf.readthedocs.io/en/latest/multi-node.html. Fetched 2026-08-10.

[7] OpenRLHF docs, Performance Tuning. https://openrlhf.readthedocs.io/en/latest/performance.html. Fetched 2026-08-10.

[8] OpenRLHF docs, Hybrid Engine. https://openrlhf.readthedocs.io/en/latest/hybrid_engine.html. Fetched 2026-08-10.

[9] OpenRLHF docs, Sequence Parallelism (RingAttention). https://openrlhf.readthedocs.io/en/latest/sequence_parallelism.html. Fetched 2026-08-10.

[10] OpenRLHF on PyPI (JSON API). https://pypi.org/pypi/openrlhf/json. Fetched 2026-08-10.

[11] Git ref resolution for tag v0.10.4 (API). https://api.github.com/repos/OpenRLHF/OpenRLHF/git/refs/tags/v0.10.4 — resolves to commit `ad1796e6`; compared against `openrlhf/models/actor.py` at that commit versus at push commit `bc71bb19464aca306b33080b2d2bb45d154e2f49`. Fetched 2026-08-10.

[12] `requirements.txt` at commit `ad1796e6` (the v0.10.4 release tag). https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/ad1796e6/requirements.txt. Fetched 2026-08-10.

[13] `setup.py` at commit `ad1796e6` (the v0.10.4 release tag). https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/ad1796e6/setup.py. Fetched 2026-08-10.

[14] OpenRLHF GitHub releases list (API). https://api.github.com/repos/OpenRLHF/OpenRLHF/releases. Fetched 2026-08-10.

[15] OpenRLHF README. https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/bc71bb19464aca306b33080b2d2bb45d154e2f49/README.md — read at the repository's newest push commit, ahead of the v0.10.4 release. Fetched 2026-08-10.

[16] OpenRLHF docs, Async Training & Partial Rollout. https://openrlhf.readthedocs.io/en/latest/async_training.html. Fetched 2026-08-10.

[17] OpenRLHF docs, index highlights (hierarchical CLI section list). https://openrlhf.readthedocs.io/en/latest/index.html. Fetched 2026-08-10.

[18] OpenRLHF docs, Troubleshooting. https://openrlhf.readthedocs.io/en/latest/troubleshooting.html. Fetched 2026-08-10.

[19] OpenRLHF docs, Common CLI Options. https://openrlhf.readthedocs.io/en/latest/common_options.html. Fetched 2026-08-10.

[20] OpenRLHF docs, Common CLI Options, Logging section. https://openrlhf.readthedocs.io/en/latest/common_options.html#logging. Fetched 2026-08-10.

[21] `openrlhf/trainer/ppo_trainer.py` at commit `ad1796e6` (the v0.10.4 release tag) — `timing/*`, `generated_samples`, `dynamic_filtering_pass_rate` fields. https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/ad1796e6/openrlhf/trainer/ppo_trainer.py. Fetched 2026-08-10.

[22] `openrlhf/trainer/ppo_actor.py` at commit `ad1796e6` (the v0.10.4 release tag) — logged metric field names (`policy_loss`, `reward`, `return`, `response_length`, `total_length`, `kl`, `actor_lr`, `actor_grad_norm`, `entropy_loss`). https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/ad1796e6/openrlhf/trainer/ppo_actor.py. Fetched 2026-08-10.

[23] `openrlhf/utils/logging_utils.py` at commit `ad1796e6` (the v0.10.4 release tag) — `WandbLogger.log_train`/`TensorboardLogger.log_train` branching on the `generated_samples` key. https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/ad1796e6/openrlhf/utils/logging_utils.py. Fetched 2026-08-10.

[24] OpenRLHF docs, Checkpointing. https://openrlhf.readthedocs.io/en/latest/checkpoint.html. Fetched 2026-08-10.

[25] Read the Docs versions API for the `openrlhf` project. https://readthedocs.org/api/v3/projects/openrlhf/versions/, cross-checked against direct fetches of `en/latest/quick_start.html` (200), `en/v0.10.4/quick_start.html` (404), and `en/stable/quick_start.html` (404). Fetched 2026-08-10.

[26] GitHub issue #1222, "Transformer version update breaks `ring_flash_attn`" (closed). https://github.com/OpenRLHF/OpenRLHF/issues/1222. Fetched 2026-08-10.

[27] `openrlhf/models/ring_attn_utils.py` at commit `ad1796e6` (the v0.10.4 release tag) — `patch_transformers_for_ring_flash_attn` docstring citing issue #1222. https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/ad1796e6/openrlhf/models/ring_attn_utils.py. Fetched 2026-08-10.

[28] GitHub issue #1258, "Qwen3.5-9B and above trained under zero3 have most weights frozen" (closed). https://github.com/OpenRLHF/OpenRLHF/issues/1258. Fetched 2026-08-10.
