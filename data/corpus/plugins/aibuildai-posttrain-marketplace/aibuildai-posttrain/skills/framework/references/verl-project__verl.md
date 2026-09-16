# verl

A Ray-orchestrated RL post-training library for LLMs: single-controller Python code drives FSDP/Megatron-LM training and vLLM/SGLang rollout workers across a cluster.

**verl** (Volcano Engine Reinforcement Learning for LLMs) is "a RL training library initiated by ByteDance Seed team and maintained by the verl community" [1], and it is the open-source implementation of the HybridFlow paper, "A Flexible and Efficient RLHF Framework" [2][1]. It was migrated in January 2026 from the `volcengine` GitHub organization to the current `verl-project` organization [1]. Its API is a single Ray-driven training script (`python -m verl.trainer.main_ppo ...`) that takes Hydra-style dot-path config overrides for data, actor/critic/rollout/reward models, and trainer settings, and dispatches the actual FSDP or Megatron training steps and vLLM/SGLang generation steps to distributed Ray workers [3]. It lives at https://github.com/verl-project/verl [4].

**When to pick it**: RL post-training that needs to scale across multiple nodes with an explicit cluster scheduler - verl's own docs describe four built-in multi-node launch paths (manual Ray cluster, SkyPilot, Slurm-via-Ray, and the dstack container orchestrator) and support FSDP, FSDP2, and Megatron-LM as training backends, with vLLM or SGLang as separate rollout engines [3][5][6]. Pick it when your workflow is already Ray-based or you need Megatron-scale sharding (671B-parameter models are named as a supported scale in its own docs) [1]; for a Hub-native, single-trainer-class workflow that starts on one machine and only later scales through Accelerate, weigh the trl card instead (cross-reference; not covered here).

**Methods it ships**: the main repo's own Algorithms section documents PPO and GRPO in full, with GRPO set as `algorithm.adv_estimator=grpo` [7][8]. The PPO page's `algorithm.adv_estimator` field also accepts `gae`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, and `rloo` as advantage-estimator choices within the same trainer, so REINFORCE++ and RLOO are reachable without a separate trainer class [7]. Beyond these, the docs index lists further algorithms as "Recipe:" pages - DAPO, SPIN, SPPO, and an Entropy Mechanism recipe - alongside separate pages for GPG (Group Policy Gradient), Rollout Correction, Optimal Token Baseline (OTB), Divergence PPO (DPPO), On-Policy Distillation (OPD), and Direct Reward Optimization [9]. The README's feature list additionally names GSPO, PRIME, DrGRPO, and KL_Cov & Clip_Cov, each linked out to the separate `verl-recipe` GitHub repository rather than shipped in the main `verl` package [1]. This algorithms taxonomy is on the docs sidebar and moves; recheck the live index at [9].

**Scale it handles**: single GPU up to multi-node clusters. Training backends are FSDP, FSDP2, or Megatron-LM (Megatron-LM v0.13.1 is the version currently supported); the install docs frame FSDP/FSDP2 as the research/prototyping path and Megatron-LM as the scalability path [5]. Rollout runs through vLLM or SGLang as separate inference engines, or HF Transformers for simple cases [5]. Multi-node launch has four documented forms: a manual Ray cluster (`ray start --head`, `ray start --address=...`, then `ray job submit`) [10]; SkyPilot, with ready two-node YAML templates for PPO and GRPO under `examples/tutorial/skypilot/` [10]; Slurm, via Ray's own Slurm tutorial, with a verified GSM8K example and a `examples/tutorial/slurm/ray_on_slurm.slurm` template [10][11]; and dstack, an open-source container orchestrator, where a Ray cluster is defined as a `dstack` task YAML (`dstack init`, then `dstack apply -f ray-cluster.dstack.yml`) and jobs are submitted to the forwarded Ray dashboard port with `ray job submit` [10]. The README states support up to 671B-parameter models with expert parallelism and hundreds of GPUs as a documented mechanism; no benchmark run at that scale was found in the pages read for this card [1].

**Install**: `pip install verl`; version 0.8.0 released 2026-06-01 [12]. Resolving the `v0.8.0` tag gives commit `7aed6b230776f963fa09509c10d9c3a767d1102c` [13]; the repository's newest push as of this card is a later, unreleased commit `98be7e979fc8869e64160455fbd6b8db53267c5e` (2026-08-10), which is ahead of the v0.8.0 release and not what this Install field describes [14]. Python >= 3.10; Apache-2.0 [12]. Base install pins `numpy<2.0.0`, `pyarrow>=19.0.0`, `ray[default]>=2.41.0`, and `tensordict!=0.9.0,<=0.10.0,>=0.8.0`; torch is not in the base dependency list and arrives through an extra or the training backend [15][16]. The `vllm` extra (`pip install verl[vllm]`) pins `vllm<=0.12.0,>=0.8.5`; the `sglang` extra pins `sglang[openai,srt]==0.5.8` together with `torch==2.9.1`; the `trl` extra pins `trl<=0.9.6`; the `mcore` extra adds `mbridge` (Megatron-Bridge, for HF-format checkpoint export from Megatron); the `trtllm` extra pins `tensorrt-llm>=1.2.0rc6` [15][16]. The install docs (read live, not pinned to v0.8.0) state hardware/software minimums of Python >= 3.10 and CUDA >= 12.8, and separately note cuDNN >= 9.10.0 as a prerequisite for a from-source build [5]; neither `setup.py` nor the PyPI metadata for the v0.8.0 release itself states a CUDA floor [15][16].

**Maintained by**: the verl community, originating from ByteDance's Seed team [1]; the project moved to the `verl-project` GitHub organization in January 2026 and the README's own dated news feed lists activity through May 2026, including a VeRL-Omni pre-release and a GTC26 presentation [1]; the v0.8.0 release (2026-06-01) is the fifth dated release in the tags read for this card, following v0.7.1 (2026-03-16), v0.7.0 (2026-01-05), v0.6.1 (2025-11-14), and v0.6.0 (2025-10-15) [17].

## Quick start

The quickstart page's smallest complete run is PPO on GSM8K [3]. Prerequisites: verl and its dependencies installed, and a GPU with at least 24 GB HBM [3]. Prepare data and a model:

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
```

Then launch training:

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log
```

This is a `python -m` invocation, not a separate CLI wrapper; every setting is a Hydra dot-path override on the config tree [3].

## Start it

- One process, one GPU is the quickstart command above, with `trainer.n_gpus_per_node=1 trainer.nnodes=1` [3].
- Multiple GPUs and multi-node runs go through Ray, in one of four documented forms: (1) manually start a Ray cluster (`ray start --head --dashboard-host=0.0.0.0` on the head node, `ray start --address=<address>` on each worker, verify with `ray status`) and submit with `ray job submit --address="http://127.0.0.1:8265" --runtime-env=verl/trainer/runtime_env.yaml --no-wait -- python3 -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 trainer.nnodes=2 ...`; (2) SkyPilot, using the ready templates `examples/tutorial/skypilot/verl-ppo.yaml` and `verl-grpo.yaml`, launched with `sky launch -c verl --secret WANDB_API_KEY verl-cluster.yml`; (3) Slurm, via Ray's own Slurm tutorial and the repo's `examples/tutorial/slurm/ray_on_slurm.slurm` template, submitted with `sbatch`; (4) dstack, a container orchestrator that manages a Ray cluster without Kubernetes or Slurm - define a fleet, describe a Ray cluster task in a `.dstack.yml` file, run `dstack apply -f ray-cluster.dstack.yml` (which forwards the Ray dashboard to `localhost:8265`), then `ray job submit` against that address [10]. Agent Loop / tool-use training has no preconfigured SkyPilot task yet [10].
- Batch-size arithmetic follows one rule stated on the config page: `data.train_batch_size` and `actor_rollout_ref.actor.ppo_mini_batch_size` are global quantities, normalized across all workers/GPUs from a single-controller perspective, while every `*_micro_batch_size_per_gpu` field is a local, per-GPU allocation, conceptually similar to gradient accumulation steps [18][19]. The deprecated global `ppo_micro_batch_size` field still exists but per-GPU fields are the current path [18].
- The config surface is Hydra YAML organized under `data`, `actor_rollout_ref` (actor/rollout/ref sub-trees), `critic`, `algorithm`, and `trainer` [18]. verl's own default in the algorithm block is `gamma: 1.0` and `lam: 1.0` (the GAE bias-variance trade-off term), with `adv_estimator: gae`, `use_kl_in_reward: False`, and `kl_ctrl: {type: fixed, kl_coef: 0.005, horizon: 10000, target_kl: 0.1}` [18]. `rollout.gpu_memory_utilization` means different things per engine: for vLLM v0.7.0+ it is a fraction of TOTAL GPU memory, for SGLang it maps to `mem_fraction_static`, a fraction of FREE GPU memory reserved for weights plus KV cache - a value tuned for one engine can OOM on the other [18][19].
- Out-of-memory first aid, from the performance-tuning guide [19]: keep `gpu_memory_utilization` between 0.5 and 0.7 if actor parameters/optimizer state are not offloaded, since too high a value causes OOM on the rollout side; enable `actor_rollout_ref.model.enable_gradient_checkpointing=True` (and the same on `critic.model`) to allow larger micro-batches; enable `actor_rollout_ref.model.enable_activation_offload=True` for further headroom (FSDP backend only, as of this reading); or switch to dynamic batching with `use_dynamic_bsz=True` and tune `ppo_max_token_len_per_gpu` instead of a fixed micro-batch size. For long sequences (>32k tokens), set `ulysses_sequence_parallel_size>1` and expect to lower the per-GPU token/micro-batch limits further to avoid OOM [19].

## Watch it

This section covers the logging mechanics only; what a metric's value means for a given method lives on that method's own card.

- **Enable it**: `trainer.logger` accepts `console`, `wandb`, `swanlab`, `mlflow`, `tensorboard`, `trackio`, or `rl_insight`, and can be a list, e.g. the config page's own default `['console', 'wandb']` [18]. Without a tracker entry beyond `console`, a run leaves no persisted metrics record - only terminal output.
- **PPO run metric names**, from the quickstart's own sample log lines [3]: timing (`timing/gen`, `timing/ref`, `timing/values`, `timing/adv`, `timing/update_critic`, `timing/update_actor`); actor (`actor/reward_kl_penalty`, `actor/reward_kl_penalty_coeff`, `actor/entropy_loss`, `actor/pg_loss`, `actor/pg_clipfrac`, `actor/ppo_kl`, `actor/grad_norm`, `actor/lr`); critic (`critic/vf_loss`, `critic/vf_clipfrac`, `critic/vpred_mean`, `critic/grad_norm`, `critic/lr`, and the `critic/score/*`, `critic/rewards/*`, `critic/advantages/*`, `critic/returns/*`, `critic/values/*` mean/max/min families); and `response_length/*`, `prompt_length/*` mean/max/min. During validation, the quickstart page states the score is logged as `val/test_score/openai/gsm8k`, computed every `trainer.test_freq` steps [3].
- **Sample-level logging**: `trainer.log_val_generations` sets how many generations are logged during validation, default 0 [18].
- **Evaluation during training**: `trainer.val_before_train` (run validation before training starts) and `trainer.test_freq` (validation cadence in iterations) are the trainer-level fields; `data.val_files` supplies the held-out set [18][3].
- **A documented health limit** comes from the FAQ, not the metrics page: `actor_rollout_ref.rollout.calculate_log_probs=True` adds a `training/rollout_probs_diff_mean` metric that flags precision mismatch between the inference and training engines; the FAQ states this value "should be below 0.005" under normal conditions, and a reading above 0.01 indicates a precision issue in the inference engine, most often on non-Hopper GPUs (A100, L20, B200) with long sequences [20]. The FAQ names a specific vLLM flash-attention bug as one cause and gives `actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True` as the workaround pending an upstream fix [20].
- **Stopping**: no RL-specific stopping rule, threshold, or patience field was found across the performance-tuning guide [19], the FAQ [20], and the config page's trainer/algorithm sections [18] - the search that would carry one, per this card's own health-limit reading above. Shapes (the metric families above) are published; stopping thresholds beyond the precision-diagnostic value above are not.

## Save it

- Checkpoints land under `trainer.default_local_dir`, which defaults to `checkpoints/${trainer.project_name}/${trainer.experiment_name}`, in per-step subdirectories named `global_steps_${i}` (also referenced as `global_step_N` in tool usage examples), each holding `actor/` and `critic/` role directories plus a root `latest_checkpointed_iteration.txt` [21][18].
- **FSDP layout**: each role directory has a `huggingface/` subfolder (present when `hf_model` is in `checkpoint.save_contents`) and `fsdp_config.json`, alongside per-rank shard files `model_world_size_{W}_rank_{R}.pt`, `optim_world_size_{W}_rank_{R}.pt`, and `extra_state_world_size_{W}_rank_{R}.pt`; the docs state that for FSDP, `model`, `optimizer`, and `extra` in `save_contents` are bound together for save/load and recommend including all three [21].
- **Megatron layout (schema v2)**: each role directory carries a `ckpt_contents.json` manifest mapping every saved content to its on-disk path and format, `transformer_config.json` when `extra` is saved, and `model/`, `optimizer/`, `extra/` subdirectories; `model/huggingface/` holds HF-format weights (written via the `mbridge` bridge, requiring `use_mbridge=True`) and `model/dist_ckpt/` holds native Megatron shards (written when `use_dist_checkpointing=True`); optimizer and RNG/scheduler state always go through `dist_checkpointing` into `optimizer/dist_ckpt/` and `extra/dist_ckpt/` [21]. Pre-v2 checkpoints (a single root `dist_ckpt/` plus a root `huggingface/`) are rejected outright at load time by the current loader; migrate with `python scripts/migrate_megatron_checkpoint_layout.py --checkpoint /path/to/global_step_N/actor` (or `--checkpoint-root ... --all-steps`), which hardlinks old shards into the new layout without duplicating disk usage [21].
- **`checkpoint.save_lora_only=True` trades full state for size**: only LoRA/adapter-tagged parameters are kept, cutting a 27B model's checkpoint from about 54 GiB to about 150 MiB; the loader auto-detects an adapter-only checkpoint by checking whether every saved key is an adapter key, and loads it with `strict=False`, while a full checkpoint still loads with `strict=True` [21]. A LoRA-only save is not a full model on its own.
- **Convert to a loadable HF model**: `python -m verl.model_merger merge --backend {fsdp,megatron} --local_dir <checkpoint dir> --target_dir <output dir>` merges sharded FSDP or Megatron checkpoints into a standard HuggingFace directory, with `--hf_upload_path` to push straight to the Hub; for pre-`fsdp_config.json` checkpoints from older verl versions, use the legacy `verl/scripts/legacy_model_merger.py` instead [21]. Megatron merges can run distributed via `torchrun --nproc_per_node ... -m verl.model_merger merge --backend megatron ...` for very large models [21].
- **Resume**: `trainer.resume_mode` is `disable`, `auto` (default - resumes from the latest checkpoint under `default_local_dir`), or `resume_path` (resumes from the path in `trainer.resume_from_path`) [18].
- Whether an evaluator can load the result directly depends on the format: an `mbridge`-produced `model/huggingface/` tree or a `model_merger`-merged directory loads with standard HF `from_pretrained`; a raw FSDP/Megatron shard directory or a LoRA-only checkpoint does not, and needs the merge step or an adapter-aware loader first [21].

## Find it in the docs

The docs are the live source; this section teaches the lookup pattern, not the content.

- Address pattern: `https://verl.readthedocs.io/en/<version>/<page>.html`. Checked 2026-08-10: `/en/latest/start/install.html` loads (HTTP 200); `/en/v0.8.0/start/install.html` does not (HTTP 404) - the docs site publishes no version-pinned tree reachable this way, so every page fetched for this card is the unpinned `latest` build, read on 2026-08-10 [22][5].
- Page-slug groups, from the sidebar TOC [9]: `start/` for installation, quickstart, and multinode training; `examples/config.html` for the full config reference (grep this one page for any specific key); `algo/` for method pages (`algo/ppo.html`, `algo/grpo.html`, `algo/baseline.html` for reproduced scores, and further `algo/` pages for recipes); `perf/perf_tuning.html` for the performance-tuning guide; `advance/checkpoint.html` for the checkpoint/resume contract; `faq/faq.html` for known errors and workarounds.
- Question-to-slug map: "what does this metric mean" -> the relevant `algo/` page or `perf/perf_tuning.html`, not a single dedicated metrics page (verl has no page equivalent to trl's `logging` page; metric names are documented inline on the quickstart and algorithm pages) [3][7][8]. "why did I OOM" -> `perf/perf_tuning.html` [19]. "how do I resume / convert a checkpoint" -> `advance/checkpoint.html` [21]. "what score should I expect" -> `algo/baseline.html`, which tabulates reproduced test scores per model/method/hardware combination, e.g. `google/gemma-2-2b-it`: 23.9 (base HF checkpoint) -> 52.06 (SFT) -> 64.02 (SFT + PPO) on GSM8K, and `Qwen/Qwen2.5-0.5B-Instruct`: 49.6 (base) -> 56.7 (PPO) [23].
- Runnable references beyond the docs: `examples/data_preprocess/gsm8k.py` for the quickstart dataset [3]; `examples/tutorial/skypilot/` for ready multi-node configs [10]; `examples/tutorial/slurm/ray_on_slurm.slurm` for Slurm [10]; a separate `recipe/` directory in the main repo holds recipe implementations, each pinning its own required verl commit or tag via a `REQUIRED_VERL.txt` file [1].
- Community layer, curated door first: the README's own "Blogs from the community" section is the closest verl has to a curated tutorials page, linking posts such as a multi-turn tokenization deep dive and a verl-x-SGLang multi-turn code walkthrough (both on the community `Awesome-ML-SYS-Tutorial` repo), and an AMD ROCm integration writeup on `rocm.blogs.amd.com` [1]. Unlike trl's curated page, this list gives no author byline and no publish date for any entry [1]; check each post's own page for its date and the verl version it targets before trusting it against your install. No separate, docs-hosted curated-tutorials page (of the shape trl's `community_tutorials` slug) was found in the pages read for this card.
- No official MCP endpoint for the verl docs was found in the sources read for this card.
- Traps found in the docs' own FAQ (not from external issue threads - none were fetched for this card): a `pip install tensordict==0.6.2` failure on linux-arm64 with no matching wheel, worked around by building tensordict from source at tag `v0.6.2` [20]; a Triton `compile_module_from_src` compilation error, worked around by setting `use_torch_compile` per the config page to disable JIT-compiled fused kernels [20]; and the precision-mismatch trap already covered under Watch it [20].
- Honest boundaries: SGLang support is explicitly called "under extensive development" on the install page, distinct from the more settled vLLM path [5]. Two verl pages disagree on AMD ROCm coverage rather than agreeing: the README states ROCm (MI300X/MI325X/MI355X) supports FSDP, FSDP2, and Megatron as trainer backends with vLLM as the validated inference engine and SGLang support "in progress" [1], while the install page's own AMD section states ROCm currently supports FSDP as the training engine with both vLLM and SGLang as inference engines, and that Megatron support is future work [5]. Which statement is current was not resolved from the pages read for this card; treat ROCm's Megatron and SGLang status as unsettled and confirm against the linked AMD ROCm quick-start guide before relying on either combination [1][5].

## Sources

All pages marked "Fetched 2026-08-10" are unpinned `main`/`latest` live docs or API responses read on that date, except the v0.8.0 release artifacts, which are pinned to the `v0.8.0` git tag as stated. Method names (PPO, GRPO, GSPO, DAPO, ...) are deliberately cited to nothing beyond the pages that name them; their defining papers live on the methodology cards.

[1] verl README, pinned at the `v0.8.0` tag. https://raw.githubusercontent.com/verl-project/verl/v0.8.0/README.md. Fetched 2026-08-10.

[2] HybridFlow: A Flexible and Efficient RLHF Framework, arXiv abstract page. https://arxiv.org/abs/2409.19256. Fetched 2026-08-10.

[3] verl Quickstart: PPO training on GSM8K dataset. https://verl.readthedocs.io/en/latest/start/quickstart.html. Fetched 2026-08-10.

[4] verl GitHub repository. https://github.com/verl-project/verl. Fetched 2026-08-10.

[5] verl Installation docs. https://verl.readthedocs.io/en/latest/start/install.html. Fetched 2026-08-10.

[6] verl documentation index (sidebar naming FSDP/FSDP2/Megatron-LM as training backends and vLLM/SGLang as rollout engines). https://verl.readthedocs.io/en/latest/index.html. Fetched 2026-08-10.

[7] verl Proximal Policy Optimization (PPO) docs. https://verl.readthedocs.io/en/latest/algo/ppo.html. Fetched 2026-08-10.

[8] verl Group Relative Policy Optimization (GRPO) docs. https://verl.readthedocs.io/en/latest/algo/grpo.html. Fetched 2026-08-10.

[9] verl documentation sidebar table of contents (as rendered on the Algorithm Baselines page). https://verl.readthedocs.io/en/latest/algo/baseline.html. Fetched 2026-08-10.

[10] verl Multinode Training docs. https://verl.readthedocs.io/en/latest/start/multinode.html. Fetched 2026-08-10.

[11] verl FAQ, "How to use verl on a Slurm-managed cluster?". https://verl.readthedocs.io/en/latest/faq/faq.html. Fetched 2026-08-10.

[12] verl on PyPI. https://pypi.org/pypi/verl/json. Fetched 2026-08-10.

[13] GitHub tags API for verl-project/verl (resolves the `v0.8.0` tag to commit `7aed6b230776f963fa09509c10d9c3a767d1102c`). https://api.github.com/repos/verl-project/verl/tags. Fetched 2026-08-10.

[14] GitHub commits API for verl-project/verl, `main` branch HEAD. https://api.github.com/repos/verl-project/verl/commits/main. Fetched 2026-08-10.

[15] verl `setup.py`, pinned at the `v0.8.0` tag. https://raw.githubusercontent.com/verl-project/verl/v0.8.0/setup.py. Fetched 2026-08-10.

[16] verl PyPI JSON metadata for release 0.8.0 (`requires_dist`, matching the pinned setup.py). https://pypi.org/pypi/verl/json. Fetched 2026-08-10.

[17] GitHub releases API for verl-project/verl (dated release list). https://api.github.com/repos/verl-project/verl/releases. Fetched 2026-08-10.

[18] verl Config Explanation docs (data, actor, rollout, algorithm, and trainer config blocks and their defaults). https://verl.readthedocs.io/en/latest/examples/config.html. Fetched 2026-08-10.

[19] verl Performance Tuning Guide. https://verl.readthedocs.io/en/latest/perf/perf_tuning.html. Fetched 2026-08-10.

[20] verl Frequently Asked Questions. https://verl.readthedocs.io/en/latest/faq/faq.html. Fetched 2026-08-10.

[21] verl "Using Checkpoints to Support Fault Tolerance Training" docs. https://verl.readthedocs.io/en/latest/advance/checkpoint.html. Fetched 2026-08-10.

[22] Direct fetch of `https://verl.readthedocs.io/en/v0.8.0/start/install.html` (HTTP 404, confirming no version-pinned docs tree) versus `https://verl.readthedocs.io/en/latest/start/install.html` (HTTP 200). Checked 2026-08-10.

[23] verl Algorithm Baselines docs (reproduced test scores by model/method/hardware). https://verl.readthedocs.io/en/latest/algo/baseline.html. Fetched 2026-08-10.
