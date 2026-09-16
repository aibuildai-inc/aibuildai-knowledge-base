# SkyRL

A Ray-orchestrated, modular RL post-training library from Berkeley: separate Trainer/Generator/Environment components, FSDP or Megatron for training, vLLM for generation, config-driven Hydra-style launch.

SkyRL's own repository description is "SkyRL: A Modular Full-stack RL Library for LLMs" [1]. It separates training into a Trainer and a Generator, where the Generator is further split into an InferenceEngine and an Environment, all coordinated by a single Controller [2]. It is built by the Berkeley Sky Computing Lab together with Anyscale, with compute support credited to Databricks, NVIDIA, Lambda Labs, AMD, AWS, Modal, and Daytona [3]. The API is config-driven: a training run is a YAML/Hydra-style config passed to a `main_base` entrypoint invoked through `uv run` [4]. It lives at https://github.com/NovaSky-AI/SkyRL [1].

**When to pick it**: an RL post-training stack built specifically around Ray placement groups and a strict separation of the training backend (FSDP or Megatron) from the vLLM generation engine, useful when you want fine control over GPU placement/colocation and parallelism dimensions (including Megatron's 5-way tensor/pipeline/context/expert/data split) [5][2]. This card does not compare SkyRL's throughput or scale against sibling libraries such as trl or verl - no measured comparison across libraries was found in the sources read for this card.

**Methods it ships**: the repository tree confirms two concrete trainer entry points, `examples/train/algorithms/dapo/main_dapo.py` for DAPO and `skyrl/train/sft_trainer.py` for SFT [6]. The docs' Algorithms section documents DAPO as GRPO plus four components - Clip-Higher (`trainer.algorithm.eps_clip_high` set separately from `eps_clip_low`), Dynamic Sampling (`trainer.algorithm.dynamic_sampling.type="filter"` with `max_sample_batches`), Token-Level Policy Gradient Loss (`trainer.algorithm.loss_reduction="token_mean"`, and the docs state this is SkyRL's default setting), and Overlong Reward Shaping (`generator.apply_overlong_filtering=true`, or a custom `DAPOTrainer` subclass) [7]. DAPO and Off-Policy Correction are both stated to be supported on the `fsdp` and `megatron` backends [7][8]. The docs' own reproduction runs (reproducible at commit `8263149145f2455b75c082f3280d344b8a554f5d`) show DAPO without Dynamic Sampling reaching AIME24 Pass@32/Mean@32 of 0.766/0.381 for Qwen-2.5-32B on FSDP (2x8xH100, 260 steps) and 0.733/0.4375 for Qwen3-30B-A3B-Base on Megatron with tp=4, ep=8 (2x8xH100, 120 steps); a LoRA variant (rank 128, alpha 128) of the same Qwen3-30B-A3B-Base/Megatron run scores higher, 0.8/0.433, on 8xH100 in 165 steps [9]. A separate GSM8K quickstart reproduction reaches 0.796 eval accuracy for Qwen2.5-1.5B-Instruct on FSDP, 4xH100, 140 steps [9]. Off-Policy Correction is a separate documented algorithm page covering Truncated Importance Sampling (TIS, `tis_ratio_type` set to `token` or `sequence`), geometric-mean sequence masking, token masking (icepop, via `token_mask_is_threshold_low/high`), outlier-based sequence masking, and MoE router replay (R3); the page's own recommendation is to start with geometric sequence masking [8]. The generated config reference additionally lists `policy_loss_type` options `regular`, `sapo`, `clip_cov`, `kl_cov`, `cispo`, and `dppo`, each with its own config block; `use_kl_in_reward` defaults to `False` but `use_kl_loss` defaults to `True` with `kl_loss_coef=0.001`, so a KL loss against the reference model (`policy_loss + kl * kl_loss_coef`) is applied by default unless a run explicitly disables it [10]. SFT is a separate `skyrl.train.sft_trainer` path with its own trainer class, distinct from the RL trainer [11][6]. LoRA (parameter-efficient adapter training) is supported for policy and critic on both `fsdp` and `megatron` backends via a `lora:` config block (`rank`, `alpha`, `dropout`, `target_modules`) [12]. DAPO and off-policy correction are presented as algorithm guides rather than as marked-experimental or marked-stable trainer classes, with no separate taxonomy page grouping SkyRL's methods by stability [7][8].

**Scale it handles**: single GPU up to multi-node, launched through Ray (`ray start --head` to initialize a cluster, then the training entrypoint is run through `uv run` with Ray's uv-runtime-env integration) [13][4]. FSDP is the default sharding backend; Megatron is a second backend selected with `trainer.strategy=megatron` and exposes five parallelism dimensions - tensor, pipeline, context, expert, and expert-tensor - with the sizing constraint `world_size % (pipeline_model_parallel_size * expert_model_parallel_size * expert_tensor_parallel_size) == 0` [5]. Both backends and generation-engine placement are set through `trainer.placement.*` config, including `colocate_all=true` to co-locate training and inference actors on the same GPUs [4]. The Megatron page publishes a Search-R1 (4K max context) FSDP-vs-Megatron comparison table, averaged over the first 10 steps at a train batch size of 512: on dense models the two backends are close (Qwen2.5-3B-Instruct, 8xH100, 48s vs. 42s training time; Qwen2.5-7B-Instruct, 8xH100, 93s vs. 100s), but at Qwen3-30B-A3B (a MoE model) on 4x8xH100 with Megatron configured DP=4/TP=2/EP=8, the page states plainly that "the FSDP backend was unable to complete a training step due to memory constraints," so Megatron is the only backend of the two that runs this model at all [5]. A dedicated troubleshooting script, `scripts/multi_node_nccl_test.py`, is provided to verify multi-node NCCL connectivity before a real run [14].

**Install**: at commit `f5bc3b78dfddfb352870d5d7430cd226e5785838`, the commit the `skyrl-v0.3.0` GitHub release tag resolves to (published 2026-07-17) [15][16], the root `pyproject.toml` declares `name = "skyrl"`, `requires-python = ">=3.11"`, Apache-2.0 licence (confirmed separately by the GitHub API) [17][1]. PyPI's `skyrl` package currently reports the same version, 0.3.0 [18]. Base dependencies pin `transformers>=5.6.1,<=5.8.0` (a hard upper bound) and `peft==0.18.1`, with no torch pin in the base install [17]. torch and vLLM are pulled in only through extras: `pip install "skyrl[fsdp]"` adds `torch==2.11.0` and `vllm==0.23.0` (both `sys_platform == 'linux'` only); `pip install "skyrl[megatron]"` adds the same `torch==2.11.0` and `vllm==0.23.0` plus `transformer-engine[pytorch]==2.11.0`; both extras also pull in the `skyrl-train` extra, which adds `ray==2.56.0`, `skyrl-gym==0.4.0`, `flash-attn==2.8.3`, `hydra-core==1.3.2`, and `wandb` (unpinned) [17]. There is a separate, legacy standalone PyPI package `skyrl-train` at version 0.3.1, distinct from the unified `skyrl` package and not the one this Install field describes [19]. The docs' own installation page requires CUDA 12.8 and `uv` as prerequisites, and recommends Ray 2.56.0 with Python 3.12 on existing clusters while stating support for Ray `>=2.44.0` except the known-buggy 2.47.0/2.47.1 [13] - note the docs' 3.12 recommendation is narrower than the `pyproject.toml`'s `>=3.11` floor read above. The row's screening commit, `7bc2524025e22c3feb2399745b6859f2b81490c0` (2026-07-31), is 14 days ahead of this release commit and is not what `pip install skyrl` currently delivers [20][15].

**Maintained by**: Berkeley Sky Computing Lab and Anyscale [3]; the repository shows 7 GitHub releases, the newest tagged `skyrl-v0.3.0` on 2026-07-17 [16], and a README News section with dated entries running from 2025-05-06 through 2026-02-17 [3]; the repository's most recent push (the row's screening commit) is dated 2026-07-31 [20].

## Quick start

The docs' "Quick Start: GRPO on GSM8K" page gives a complete two-step run [4]. Prepare data:

```bash
uv run --isolated examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
```

Launch training (abridged from the full command, which sets model, backend, placement, and logging together) [4]:

```bash
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$HOME/data/gsm8k/train.parquet']" \
  data.val_data="['$HOME/data/gsm8k/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=10 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.num_engines=4 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  environment.env_class=gsm8k \
  trainer.logger="wandb"
```

The docs also give an equivalent packaged form: `export WANDB_API_KEY=...` then `bash examples/train/gsm8k/run_gsm8k.sh` [4].

## Start it

- One GPU: the quickstart command above with `generator.inference_engine.num_engines=1`, `tensor_parallel_size=1`, and `trainer.placement.policy_num_gpus_per_node=1`; the docs require `num_engines * tensor_parallel_size` to equal the total policy GPU count [4].
- More GPUs, one node: a Ray cluster is required before launch - `ray start --head` initializes it, and the same `uv run --isolated --extra <backend>` command is then dispatched through Ray's uv-runtime-env integration [13][4].
- More than one node: confirm connectivity first with `scripts/multi_node_nccl_test.py` (`uv run --isolated --env-file .env scripts/multi_node_nccl_test.py --num-nodes 2`) before a real run [14]. Placement-group creation timeouts on autoscaling (e.g. KubeRay) clusters can be raised past the 180-second default with the `SKYRL_RAY_PG_TIMEOUT_IN_S` environment variable [14].
- Generation layout: `generator.inference_engine.backend=vllm` with `num_engines` and `tensor_parallel_size` set so their product equals the policy GPU count; `trainer.placement.colocate_all=true` shares GPUs between training and inference actors instead of using separate GPU pools [4].
- Megatron backend adds a five-way parallelism config block (`tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size`, `expert_model_parallel_size`, `expert_tensor_parallel_size`) under `trainer.policy.megatron_config`, plus pass-through kwargs to Megatron's optimizer/DDP/transformer config objects [5]. Sizing must satisfy `model_size = pp_size * tp_size * cp_size`, `dp_size = world_size / model_size`, and `world_size % (pp_size * ep_size * etp_size) == 0` [5].
- Effective batch arithmetic: `train_batch_size` (default 1024) is the number of prompts pulled per dataloader step, and `policy_mini_batch_size` (default 256) is the global number of prompts per optimizer step, so `train_batch_size / policy_mini_batch_size` optimizer steps run per training batch - the docs' own worked example is "with `train_batch_size=4` and `policy_mini_batch_size=2` there are 2 optimizer steps (model updates) per training batch"; the per-worker mini-batch size is `policy_mini_batch_size * generator.n_samples_per_prompt / number of DP ranks` [10].
- Config surface: every trainer config is Hydra-style YAML overridden on the command line, documented field-by-field in the generated API reference at `/docs/api-ref/skyrl/config` - the docs' own commit history states this page is now the single source of truth for config fields, replacing a since-removed handwritten `configuration/config.mdx` page that had drifted from the code [21][20]. One default the library sets that a newcomer should know before running: `use_kl_loss` defaults to `True` with `kl_loss_coef=0.001`, so a KL loss against the reference model is applied unless explicitly turned off, even though the sibling `use_kl_in_reward` flag defaults to `False` [10].
- Out-of-memory first aid: a hard `Aborted`/`SIGABRT` crash with free VRAM still showing is diagnosed by the docs as CUDA allocator fragmentation, not real capacity exhaustion, and is called out as worst under `trainer.placement.colocate_all=true` because the trainer repeatedly offloads/reloads GPU memory while the inference engine sleeps and wakes in the same address space [22]. The fix is PyTorch's expandable-segments allocator, controlled by two independent flags because trainer and vLLM are separate processes: `trainer.use_expandable_segments` (default on) for policy/critic/reference workers, and `generator.inference_engine.use_expandable_segments` (default off) for the vLLM engine, safe to enable with vLLM sleep mode on vLLM >= 0.20.1 but a hard error with sleep mode on older vLLM [22]. Megatron's `optimizer_config_kwargs.use_precision_aware_optimizer=true` is separately flagged as a setting that "can cause checkpointing to fail", with the docs recommending it be left `false` [5].

## Watch it

This section covers mechanics only - which signals SkyRL emits and how to turn logging on; what a signal means for a given algorithm (e.g. reward or KL shape) lives on that method's own card, not here.

- **Enable it**: `trainer.logger` selects the backend and defaults to `"wandb"`; accepted values are `"wandb"`, `"mlflow"`, `"swanlab"`, `"tensorboard"`, or `"console"` [10]. Backend dispatch is implemented in `skyrl/train/utils/tracking.py`'s `Tracking` class, whose header comment states it is adapted from veRL's `tracking.py` [23]. `console` prints locally with no external record; picking any of the other four is what produces a persisted run.
- **Training-progress metric names** (read from `skyrl/train/trainer.py` at the row's screening commit `7bc2524025e22c3feb2399745b6859f2b81490c0`, since no single docs page enumerates the RL-training metric list): the trainer builds one `log_payload` dict per step from `self.all_metrics`, `self.all_timings` (prefixed `timing/`), and scraped vLLM metrics, then calls `self.tracker.log(log_payload, step=self.global_step, commit=True)` [24]. Key namespaces present in the source include `critic/*`, `generate/*` (e.g. `batch_num_seq`, `num_seq_after_merge`), `loss/*` (e.g. `avg_final_rewards`, `avg_kl`, `kl_loss_coef`), `policy/*` (e.g. `rollout_train_logprobs_abs_diff_mean`), `reward/*` (e.g. `avg_pass_at_n`, `avg_raw_reward`, `num_zero_variance_filtered`), `timing/*`, `trainer/*` (`epoch`, `global_step`, `tokens_per_second_per_gpu`), and `trajectories/train` [24].
- **SFT metric names**, from `skyrl/train/sft_trainer.py` at the same commit: `train/loss`, `train/grad_norm`, `train/epoch`, `train/global_step`, `train/tokens_per_second`, `train/tokens_per_second_per_gpu`, `train/actual_num_tokens`, `train/batch_padded_seq_len`, `train/total_tokens_processed`, plus `eval/{k}` and `timing/{k}` (including `timing/save_checkpoint` and `timing/save_hf_model`) [11]. This is a source-code reading, not a docs enumeration, and is distinct from the RL trainer's metric set above.
- **vLLM engine metrics forwarded to wandb**: on by default (`generator.inference_engine.enable_ray_prometheus_stats: false` to turn off); a `VLLMMetricsScraper`, instantiated in the trainer constructor at `trainer.py:139-140` [24], scrapes every Ray node's Prometheus endpoint once per step and merges eight keys into the same log payload as training metrics - `vllm/num_requests_running`, `vllm/num_requests_waiting`, `vllm/kv_cache_usage_perc`, `vllm/generation_throughput_tok_s`, `vllm/prompt_throughput_tok_s`, `vllm/prefix_cache_hit_rate`, `vllm/ttft_seconds_avg`, `vllm/tpot_seconds_avg` - each with its own source metric type and aggregation rule documented on the page [25]. This curated set is explicitly only a subset: every metric vLLM exports remains queryable directly via Prometheus/PromQL against Ray's per-node metrics endpoints, with names sanitized (`:` becomes `_`, prefixed `ray_`) [25].
- **Infrastructure logs are routed separately from stdout**: the console/stdout stream shows only config, dataset, step, reward, and metric summaries; vLLM startup, KV-cache, and weight-sync logs go to `{trainer.log_path}/infra-YYMMDD_HHMMSS.log` (default `/tmp/skyrl-logs`), overridable to stdout via `SKYRL_DUMP_INFRA_LOG_TO_STDOUT` [26]. The page's own Known Limitations note that Ray system messages are not captured, all actors share one log file (causing interleaving), and repeated messages are not deduplicated [26].
- **Sample-level generation logging**: `skyrl/train/utils/tracking.py` defines a `log_samples_to_table` function on the `Tracking` class (source-code evidence, not yet cross-checked against a docs page describing its trigger condition or config key) [23].
- **Evaluation during training**: `trainer.eval_before_train`, `trainer.eval_interval`, and `trainer.eval_batch_size` are used directly in the quickstart command to run periodic evaluation [4].
- **Stopping**: SkyRL publishes a step-count stopping mechanism - `self.cfg.trainer.max_training_steps` triggers a "Reached max_training_steps=..., stopping early" log line in `trainer.py` at the row's screening commit [24] - but this is a fixed step cap, not a reward- or KL-based stopping rule. No RL-specific stopping threshold (e.g. a reward-plateau or KL-divergence limit) was found in the troubleshooting page [22], the checkpointing/logging pages [26][25][21], or the DAPO/off-policy-correction algorithm pages [7][8] read for this card.

## Save it

- **FSDP checkpoint layout**: `{ckpt_path}/global_step_{N}/policy/` contains `fsdp_config.json`, a `huggingface/` subdirectory with `config.json`, `tokenizer_config.json`, and `generation_config.json`, plus `model_state.pt`, `optimizer_state.pt`, and `lr_scheduler_state.pt`; a `critic/` directory with the same shape exists if a critic is enabled; `data.pt` (dataloader state) and `trainer_state.pt` sit alongside them, and `{ckpt_path}/latest_ckpt_global_step.txt` records the newest step [27].
- **Megatron checkpoint layout** uses Megatron's `dist_checkpointing` library for parallel, resharding-capable writes: `{ckpt_path}/global_step_{N}/policy/` holds `metadata.json`, a `huggingface/` subdirectory, and per-rank `.distcp` files [27].
- **What is saved**: policy and critic (if enabled) each get parameters, optimizer state, and LR-scheduler state; the reference model is explicitly not checkpointed - it is recreated from the policy model on resume [27].
- **Retention and cadence**: `ckpt_interval` (default 10) sets save frequency in steps; `ckpt_path` (default `~/ckpts/`, also accepts `s3://` or GCS paths) sets the base directory; `max_ckpts_to_keep` (default `-1`, keep all) can be set to a positive N to auto-delete older checkpoints and save disk space, without documented effect on resumability [27].
- **Resume**: `resume_mode` (default `"latest"`) controls resumption - `"none"`/`null` starts from scratch, `"latest"` resumes the most recent checkpoint automatically, and `"from_path"` resumes from a specific `global_step_N` directory named in `resume_path` (default `null`) [27].
- **HuggingFace safetensors export is a separate, optional artifact from the resumable checkpoint**: `hf_save_interval` (default `-1`, disabled) saves the policy model in HF format every N steps to `{export_path}/global_step_{N}/policy/` (`export_path` default `~/exports/`) [27]. This export does not itself carry the retention-vs-resumability tradeoff the spec warns about - the docs describe no equivalent of an "optimizer-state-dropping" retention flag for the resumable checkpoint path (the closest analogue, `max_ckpts_to_keep`, discards whole old checkpoint directories rather than trimming what is saved inside each one).
- **Distributed HF export (Megatron only)**: by default the Megatron-to-HuggingFace export writes entirely from rank 0, which stalls on multi-hundred-GB checkpoints; `trainer.policy.megatron_config.hf_export_config.distributed_save` (default `false`) fans the write across ranks, with `save_every_n_ranks` (default 1) as the rank stride - and the docs warn this requires `trainer.export_path` to be a shared filesystem visible to all ranks, because Megatron-Bridge builds `model.safetensors.index.json` from what rank 0 alone can see, producing an incomplete index otherwise [27].
- **LoRA saves an adapter, not a full model**: `lora_sync_path` (config example shows `/tmp/skyrl_lora_sync`) is "Directory path where LoRA adapters are saved and synchronized between training and inference engines" [12]; the docs read for this card do not spell out the adapter's on-disk file names or give a merge-to-full-model call, so a reader relying only on this card should treat the LoRA output directory as an adapter (small, requires the base model to use) rather than a directly loadable full checkpoint, consistent with the HF export format the trainer otherwise produces [12][27].
- **Loader handoff**: whether an evaluator can load a saved checkpoint or export directly is the loader's contract, not documented on these pages; the HF-format export under `export_path` is the artifact most likely to load with a standard `from_pretrained`-style call, while the FSDP/Megatron `ckpt_path` directory is SkyRL's own resumable format and not shown to be loadable outside SkyRL in the pages read here.

## Find it in the docs

The docs are the live source; this section is the lookup, not a mirror.

- Address pattern: `https://docs.skyrl.ai/docs/<category>/<slug>`, a nested Next.js/Fumadocs site - top-level guesses like `docs.skyrl.ai/docs/checkpointing` 404; the real slugs are nested under categories such as `checkpointing-logging/checkpointing`, `checkpointing-logging/logging`, `checkpointing-logging/vllm-metrics`, `troubleshooting/troubleshooting`, `algorithms/dapo`, `algorithms/off_policy_correction`, `getting-started/installation`, `getting-started/quickstart`, `getting-started/overview`, and `api-ref/skyrl/config` - checked 2026-08-11 [4][13][27][26][25][22][7][8][2][21].
- The row's screening commit (`7bc2524025e22c3feb2399745b6859f2b81490c0`, 2026-07-31) removed the old handwritten `docs/configuration/config.mdx` page and moved `configuration/placement.mdx` into a Tutorials section, stating the generated `/docs/api-ref/skyrl/config` page (built from the `skyrl/train/config/config.py` dataclasses) is now the single source of truth for config fields [20]. Both old slugs 404 live; the current homes are `api-ref/skyrl/config` and `tutorials/placement` [21][28].
- The config API reference (`api-ref/skyrl/config`) is the place to look up any individual field's default and docstring - it is large (thousands of lines) and organized by dataclass (`AlgorithmConfig`, `TrainerConfig`, `MegatronConfig`, `GeneratorConfig`, ...); searching within the fetched page for a field name (e.g. `enforce_eager`, `kl_loss_coef`, `policy_loss_type`) is faster than browsing [21].
- Runnable references beyond the docs: the `examples/` tree in the GitHub repository, including `examples/train/gsm8k/` (quickstart dataset script and `run_gsm8k.sh`), `examples/train/algorithms/dapo/main_dapo.py` (the DAPO entry point named in this row's own evidence), and `scripts/multi_node_nccl_test.py` for multi-node smoke testing [4][14][1].
- Community/curated layer: the docs site's own top-level navigation includes a "Recipes" section, `docs/recipes/overview`, distinct from "Examples" and "Tutorials" [2][9]; it collects the project's own end-to-end reproduction runs (GSM8K, DAPO, SkyRL-SQL, SearchR1) with model, hardware, step count, reproduction commit, and a WandB report link for each - this is SkyRL's own curated door and the place with published reference numbers, and should be checked before searching for third-party blog posts [9].
- No official MCP endpoint for querying these docs was found in the pages read for this card.

Honest boundary: no page read for this card states a hardware floor below CUDA 12.8, and the installation page's own Requirements section names only CUDA 12.8 and `uv` - no CPU-only or non-NVIDIA path is documented [13]. Two Ray versions, 2.47.0 and 2.47.1, are explicitly called out as not recommended due to a known uv+Ray integration bug [13]. `use_precision_aware_optimizer=true` under the Megatron backend is documented as capable of breaking checkpointing and is recommended left off [5].

## Sources
[1] NovaSky-AI/SkyRL GitHub repository API record. https://api.github.com/repos/NovaSky-AI/SkyRL. Fetched 2026-08-11.

[2] SkyRL System Overview. https://docs.skyrl.ai/docs/getting-started/overview. Fetched 2026-08-11.

[3] SkyRL root README. https://raw.githubusercontent.com/NovaSky-AI/SkyRL/main/README.md. Fetched 2026-08-11.

[4] SkyRL Quick Start: GRPO on GSM8K. https://docs.skyrl.ai/docs/getting-started/quickstart. Fetched 2026-08-11.

[5] SkyRL Megatron Backend for 5D Parallelism (examples page). https://docs.skyrl.ai/docs/examples/megatron. Fetched 2026-08-11.

[6] NovaSky-AI/SkyRL repository file tree at the skyrl-v0.3.0 tag, via jsDelivr's GitHub metadata API. https://data.jsdelivr.com/v1/packages/gh/NovaSky-AI/SkyRL@skyrl-v0.3.0?structure=tree. Fetched 2026-08-11.

[7] SkyRL DAPO algorithm guide. https://docs.skyrl.ai/docs/algorithms/dapo. Fetched 2026-08-11.

[8] SkyRL Off Policy Correction guide. https://docs.skyrl.ai/docs/algorithms/off_policy_correction. Fetched 2026-08-11.

[9] SkyRL Recipes overview (E2E reproduction runs: GSM8K, DAPO, SkyRL-SQL, SearchR1, with hardware, steps, commit, and WandB report links). https://docs.skyrl.ai/docs/recipes/overview. Fetched 2026-08-11.

[10] SkyRL generated config API reference (AlgorithmConfig section: policy_loss_type, use_kl_in_reward, use_kl_loss, logger default). https://docs.skyrl.ai/docs/api-ref/skyrl/config. Fetched 2026-08-11.

[11] `skyrl/train/sft_trainer.py`, read at commit 7bc2524025e22c3feb2399745b6859f2b81490c0. https://raw.githubusercontent.com/NovaSky-AI/SkyRL/7bc2524025e22c3feb2399745b6859f2b81490c0/skyrl/train/sft_trainer.py. Fetched 2026-08-11.

[12] SkyRL LoRA Training guide. https://docs.skyrl.ai/docs/examples/lora. Fetched 2026-08-11.

[13] SkyRL Installation page. https://docs.skyrl.ai/docs/getting-started/installation. Fetched 2026-08-11.

[14] SkyRL Troubleshooting page (Placement Group Timeouts, Multi-node Training sections). https://docs.skyrl.ai/docs/troubleshooting/troubleshooting. Fetched 2026-08-11.

[15] NovaSky-AI/SkyRL git ref for tag skyrl-v0.3.0 (resolves to commit f5bc3b78dfddfb352870d5d7430cd226e5785838). https://api.github.com/repos/NovaSky-AI/SkyRL/git/refs/tags/skyrl-v0.3.0. Fetched 2026-08-11.

[16] NovaSky-AI/SkyRL releases list (skyrl-v0.3.0 published 2026-07-17T01:39:33Z). https://api.github.com/repos/NovaSky-AI/SkyRL/releases. Fetched 2026-08-11.

[17] Root `pyproject.toml`, read at the skyrl-v0.3.0 release commit f5bc3b78dfddfb352870d5d7430cd226e5785838. https://raw.githubusercontent.com/NovaSky-AI/SkyRL/f5bc3b78dfddfb352870d5d7430cd226e5785838/pyproject.toml. Fetched 2026-08-11.

[18] `skyrl` package on PyPI. https://pypi.org/pypi/skyrl/json. Fetched 2026-08-11.

[19] `skyrl-train` package on PyPI (legacy standalone package, version 0.3.1). https://pypi.org/pypi/skyrl-train/json. Fetched 2026-08-11.

[20] NovaSky-AI/SkyRL commit 7bc2524025e22c3feb2399745b6859f2b81490c0 ("[chore][docs] Remove outdated configuration.mdx; make the API reference the single source of truth"). https://api.github.com/repos/NovaSky-AI/SkyRL/commits/7bc2524025e22c3feb2399745b6859f2b81490c0. Fetched 2026-08-11.

[21] SkyRL generated config API reference, general structure and page identity. https://docs.skyrl.ai/docs/api-ref/skyrl/config. Fetched 2026-08-11.

[22] SkyRL Troubleshooting page (Hard Aborted/SIGABRT crash section: expandable-segments flags). https://docs.skyrl.ai/docs/troubleshooting/troubleshooting. Fetched 2026-08-11.

[23] `skyrl/train/utils/tracking.py`, read at commit 7bc2524025e22c3feb2399745b6859f2b81490c0. https://raw.githubusercontent.com/NovaSky-AI/SkyRL/7bc2524025e22c3feb2399745b6859f2b81490c0/skyrl/train/utils/tracking.py. Fetched 2026-08-11.

[24] `skyrl/train/trainer.py`, read at commit 7bc2524025e22c3feb2399745b6859f2b81490c0. https://raw.githubusercontent.com/NovaSky-AI/SkyRL/7bc2524025e22c3feb2399745b6859f2b81490c0/skyrl/train/trainer.py. Fetched 2026-08-11.

[25] SkyRL vLLM Engine Metrics page. https://docs.skyrl.ai/docs/checkpointing-logging/vllm-metrics. Fetched 2026-08-11.

[26] SkyRL Logging page. https://docs.skyrl.ai/docs/checkpointing-logging/logging. Fetched 2026-08-11.

[27] SkyRL Checkpointing page. https://docs.skyrl.ai/docs/checkpointing-logging/checkpointing. Fetched 2026-08-11.

[28] SkyRL Model Placement and Colocation tutorial (moved-to location of the former configuration/placement.mdx page). https://docs.skyrl.ai/docs/tutorials/placement. Fetched 2026-08-11.
