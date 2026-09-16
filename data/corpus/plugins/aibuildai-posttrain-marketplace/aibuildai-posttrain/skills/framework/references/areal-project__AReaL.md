# AReaL

A microservice-based asynchronous RL training system for LLM reasoning and agents: decouple generation from training, then scale each with its own scheduler and sharding backend.

**AReaL** is described in its own README as "a reinforcement learning (RL) infrastructure designed to bridge foundation model training with modern agent-based applications" [1]. It was originally built by researchers and engineers from Tsinghua IIIS and the AReaL Team at Ant Group [1], and its "major contributors are from the AReaL Team" at IIIS Tsinghua and Ant Group [1]. Its API shape is config-driven: a Python training script (e.g. `examples/math/gsm8k_rl.py`) reads a YAML config plus dot-path command-line overrides (`actor.path=...`, `rollout.backend=...`) and is launched under a chosen `scheduler.type` (`local`, `ray`, or `slurm`) that runs AReaL's independent training, inference, agent, and weight-update services [1][2]. It lives at https://github.com/areal-project/AReaL [1].

**When to pick it**: pick AReaL when the run needs the generation and training steps decoupled and independently scaled through named services, with off-policy rollout data explicitly controlled through a lag parameter rather than always synchronized to the latest policy - the README states that setting `max_head_offpolicyness=0` reduces any of its RL algorithms to a synchronous version, meaning the same trainers cover both regimes [1][3]. It ships both a single-controller launch path (a script driven by `scheduler.type=local|ray|slurm`) and a legacy SPMD path via a dedicated `torchrun`-based launcher, with the docs recommending single-controller for most cases [4]. AReaL's own changelog reports a within-project reference number for that design choice: its "boba²" (v0.3) release is reported to achieve a 2.77x speedup while delivering training performance comparable to or better than a synchronous system [5]; this is AReaL's own comparison of its asynchronous mode against a synchronous baseline, not a cross-framework benchmark against sibling RL libraries, and no such cross-framework throughput or scale comparison was found in the sources read for this card.

**Methods it ships**: the README's Support Matrix table lists GRPO, GSPO, PPO, DAPO, LitePPO, Dr.GRPO, REINFORCE++, RLOO, SAPO, IcePop, KPop, M2PO, SFT, and Distillation with a GSM8K-scale example YAML/script each; the table's remaining two rows, DPO and RLHF Reward Modeling, instead link an Anthropic HH-RLHF-based example [3]. Of these, GRPO, GSPO, PPO, DAPO, LitePPO, Dr.GRPO, RLOO, SAPO, IcePop, and KPop share one documentation page (`docs/en/algorithms/grpo_series.md`); M2PO, DPO, and Distillation each have a separate page (`docs/algorithms/m2po.md`, `docs/en/algorithms/dpo.md`, `docs/en/algorithms/distillation.md`); REINFORCE++, RLHF Reward Modeling, and SFT have no linked documentation page in that table [3]. This table is the live taxonomy source and it changes with releases (the README's own changelog shows KPop added 2026/06/17) [3][5], so recheck it at https://github.com/areal-project/AReaL#-support-matrix before relying on a name here. The README's Models table separately shows which of these run under which training backend, e.g. Qwen3-MoE is supported on Megatron, FSDP, and Archon, while Gemma 3 is FSDP-only [3].

**Scale it handles**: single GPU with `scheduler.type=local`, up to multi-node with `scheduler.type=ray` or `scheduler.type=slurm`, launched by direct execution of the training script (the recommended single-controller mode) or, for backward compatibility, by a dedicated `torchrun`-based SPMD launcher that sets `AREAL_SPMD_MODE=1` [4]. Sharding is chosen per role through an allocation string `<backend>:<dims>` where dims combine data (`d`), tensor (`t`), context (`c`), and, for Megatron and the experimental PyTorch-native Archon engine only, pipeline (`p`) and expert (`e`) parallelism; FSDP supports only `d`, `t`, `c` [6]. The docs give worked example commands for 4-node Ray (12 GPUs generation, 4 training) and 16-node/128-GPU Slurm launches [2], but neither that page nor any other fetched page publishes a scaling benchmark number for those configurations - the multi-node mechanism is documented, not measured.

**Install**: the README's own quick-start is `git clone https://github.com/areal-project/AReaL && cd AReaL && pip install uv && uv sync --extra cuda` (SGLang is the default inference backend this way); switching to vLLM requires swapping in a separate lockfile pair (`cp pyproject.vllm.toml pyproject.toml && cp uv.vllm.lock uv.lock`) because the installation docs state that SGLang and vLLM pin torch and torchao versions that are incompatible with each other, so the two backends are kept in separate dependency files [7]. The package is also published on PyPI as `areal`, version 1.0.4, uploaded 2026-06-10, licensed Apache-2.0, Python floor `>=3.11,<3.13` [8]. Reading the v1.0.4 release tag's own `pyproject.toml` (resolved to commit 37d6c6400e99a05fa3409d6a067762a44df40d3b) [9]: the base install pins `torch>=2.9.1,<2.11` (platform-conditioned), `transformers>=5.0.0,<=5.3.0`, and `peft<=0.18.1`; the `sglang` extra pins `sglang[tracing]==0.5.10.post1` exactly; the `megatron` extra pins `megatron-core==0.17.0` exactly and additionally requires `python_version>=3.12`, narrower than the package's own 3.11 floor [9]. The separate vLLM variant at the same tag pins a different, incompatible core: `torch>=2.10.0,<2.11`, `torchao==0.16.0`, and `vllm==0.19.1` [10]. The screening commit 6c0f9521e1f01d66864665644f83f9e2015ce327 is ahead of this release - its `pyproject.toml` already declares `version = "2.0.0"` - so any install-field claim here is read from the v1.0.4 tag and its resolved commit, not the screening commit [11]. Neither the installation docs page nor the pyproject files state a CUDA floor as an install-time constraint, but the docs' own extensively-tested hardware table names CUDA 12.8 and NVIDIA driver 550.127.08 [7].

**Maintained by**: the AReaL Team at Ant Group together with Tsinghua IIIS [1]; the README's changelog records a 2026/07/01 "AReaL 2.0" release refactoring the project into its current microservice architecture, and a 2026/06/17 entry adding the KPop algorithm, so both project and docs are dated as recently active [5][3].

## Quick start

The README's own single-node quick start needs no manual dataset or model download - the training script fetches `openai/gsm8k` and `Qwen/Qwen2-1.5B-Instruct` itself [1]:

```bash
git clone https://github.com/areal-project/AReaL
cd AReaL
pip install uv
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
uv sync --extra cuda
```

```bash
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local
```

For a Ray cluster, the README's own form updates the YAML paths to shared storage first, then runs [1]:

```bash
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
  cluster.n_nodes=2 cluster.n_gpus_per_node=8 \
  cluster.fileroot=/path/to/nfs \
  scheduler.type=ray
```

## Start it

- One GPU: run the script above with `scheduler.type=local`; naming `experiment_name`/`trial_name` on the command line is required by the quickstart's own example invocation [2].
- Several GPUs, one node: add explicit per-role allocation strings, e.g. `rollout.backend=sglang:d2p1t1 actor.backend=fsdp:d2p1t1 cluster.n_nodes=1 cluster.n_gpus_per_node=4` [2]. This is the generation-layout choice this async method adds: `rollout.backend` sizes the inference engine's GPU share separately from `actor.backend`'s training share, and the docs require the total GPUs implied by both strings to equal `cluster.n_nodes * cluster.n_gpus_per_node` [2].
- Multiple nodes: switch `scheduler.type` to `ray` or `slurm` and scale both backend strings and `cluster.n_nodes`/`cluster.n_gpus_per_node` together; the docs' own examples show a 4-node Ray run (`sglang:d12p1t1` / `fsdp:d4p1t1`) and a 16-node Slurm run (`sglang:d96p1t1` / `fsdp:d32p1t1`), and warn that Ray/Slurm allocate GPUs at node granularity, so generation and training GPU counts must each be an integer multiple of `cluster.n_gpus_per_node` [2].
- Legacy SPMD mode launches the same script through a dedicated launcher instead of a scheduler string, e.g. `python3 -m areal.infra.launcher.local examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml`, which spawns workers via `torchrun` and sets `AREAL_SPMD_MODE=1`; the docs mark this mode as maintained for backward compatibility only [4].
- All configuration fields live in `areal/api/cli_args.py`; options already in the YAML are overridden by dot-path (`actor.path=Qwen/Qwen3-1.7B`), and options that exist in `cli_args.py` but not the YAML need a `+` prefix (`+sglang.attention_backend=triton`) [2]. Effective batch size is set by `train_dataset.batch_size` directly (default 1, read from the dataclass at the screening commit) [12][2] rather than a per-device-times-accumulation product; the quickstart's own customization example sets `train_dataset.batch_size=1024` [2].
- The base `TrainEngineConfig` at the screening commit 6c0f9521e1f01d66864665644f83f9e2015ce327 defaults `dtype="bfloat16"` for forward/backward compute and `gradient_checkpointing=False`, while `optimizer_dtype` stays `"float32"` by default to keep fp32 master weights matching DeepSpeed ZeRO-3/Megatron precision-aware-optimizer behavior [12] - the bf16 compute default is a silent assumption of bf16-capable hardware.
- Out-of-memory first aid: the docs' dedicated OOM page gives separate remedies for generation and training. On the generation side it lists reducing `max_concurrent_rollouts`, raising tensor parallelism for the rollout backend, and lowering SGLang's `mem_fraction_static` [13]. On the training side, the default fp32 master weights plus fp32 AdamW state cost roughly 12N GB for an N-billion-parameter model; switching `optimizer_dtype: bfloat16` together with `optimizer.type: adam_bf16` brings that down to about 8N GB (a roughly one-third reduction, not a halving) via an AnyPrecisionAdamW implementation with Kahan-summation compensation, and the training section separately documents per-layer optimizer streaming (`actor.fsdp.per_layer_optim_step: true`) for FSDP CPU-offload setups [13]. The page states explicitly: "Do not use optimizer.type: adam together with optimizer_dtype: bfloat16 — torch.optim.AdamW will silently create bf16 optimizer states and the late-stage convergence will plateau ~3× higher than fp32 master weights (see issue #1292)" [13].

## Watch it

This section covers only the logging mechanics; what a metric value means for a given method belongs on that method's own card.

- **Enable it**: logging is configured under `stats_logger` in the YAML, with per-backend sub-blocks `wandb` (`mode: "online"|"offline"|"disabled"`), `swanlab` (`mode: "online"|"local"|"disabled"`), and `tensorboard` (`path: ...`, `null` to disable) [14]. A `StatsLogger` instance is managed automatically by the trainer and runs only on rank 0 to avoid duplicate writes, and it is called once per training step from the trainer's own step-export routine [14]. AReaL publishes no metric-name list for `wandb`/`swanlab`/`tensorboard` on this page beyond the illustrated code path - the concrete metric keys reaching those backends are whatever the actor's, rollout's, and eval-rollout's `export_stats()` produce, and the page does not enumerate them [14].
- **Mechanism**: `areal.utils.stats_tracker` exposes two logging paradigms - a streaming paradigm for rollout workers, where each async workflow calls `stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward, ...)` independently and the controller later computes weighted averages across workers, and a batch paradigm for training engines, which logs whole-batch tensors with denominator masks reduced across data-parallel ranks at export time [14]. Both paradigms support named/scoped trackers so rollout and training metrics stay in separate namespaces [14]. The example scalar names shown on this page are `reward`, `num_turns`, `max_tokens` (rollout side, user-added) and `learning_rate` (training side, illustrative default-tracker call); this is a code-example list, not a documented complete metric catalogue [14].
- **Sample-level logging of generations**: not documented on the fetched metrics-tracking or logging-adjacent pages read for this card; not confirmed present or absent beyond that.
- **Evaluation during training**: the Evaluation tutorial states that AReaL "provides distributed inference for your trained model, not a complete evaluation pipeline with dataset retrieval and metrics computation," and its quick start runs a separate script (`examples/math/gsm8k_eval.py --config ... actor.path=/path/to/checkpoint`) against a saved checkpoint through the same scheduler/backend controller infrastructure used for training, rather than an in-loop eval hook exposed as a trainer config field [15]. It notes evaluation needs no checkpoint-format conversion because AReaL already saves HuggingFace-compatible checkpoints [15].
- **Stopping-rule honesty**: searching `areal/api/cli_args.py` at the screening commit for early-stopping, patience, or plateau fields found none - the only threshold-named fields are `m2_threshold` (an M2PO algorithm parameter) and an unrelated action-on-metric-exceeds-threshold field for behavior-policy divergence, neither an early-stopping rule [12]. A search of the fetched OOM, checkpointing, and metrics-tracking docs pages for "early stop", "patience", or "plateau" returned no match outside the OOM page's unrelated convergence-plateau warning quoted above [13][14][16]. No RL early-stopping threshold is published in the sources read for this card.

## Save it

- Two independent, automatically-invoked mechanisms exist, per the checkpointing reference's own comparison table: the **Saver** exports HuggingFace-format weights (safetensors + `config.json`) for evaluation or publishing and does not include optimizer state, while the **RecoverHandler** writes a backend-native Distributed Checkpoint (DCP) format that does include optimizer, RNG, and dataloader state, for fault-tolerant resume [16].
- Saver output lands at `{fileroot}/checkpoints/{user}/{experiment_name}/{trial_name}/default/epoch{E}epochstep{S}globalstep{G}/`, containing `config.json`, one or more `model*.safetensors` shards, and tokenizer files; it is configured via `config.saver.mode` (`"auto"` by default, `"sync"`, or `"async"` - async is Archon-engine only and falls back to sync with a warning on other engines) and `freq_epochs`/`freq_steps`/`freq_secs`, any of which can trigger a save [16].
- RecoverHandler output lands under `{fileroot}/checkpoints/{user}/{experiment_name}/{trial_name}/default/recover_checkpoint/` as sharded `*.distcp` files plus a `recover_info/` metadata directory (`step_info.json`, `dataloader_info.pkl`, etc.); it is configured via `config.recover.mode` (`"on"`/`"auto"` to enable, `"off"`/`"disabled"` to disable), the same three frequency fields, and `retries` (default 3) [16]. **The retention contract here is the opposite of a retention flag that silently drops state**: the docs state RecoverHandler "Overwrites previous checkpoint. Only one copy exists at a time," so only the single latest DCP checkpoint is ever recoverable, while the Saver instead keeps every export as its own directory [16].
- Load a Saver checkpoint with standard HuggingFace APIs directly: `AutoModelForCausalLM.from_pretrained("/path/to/checkpoint/epoch0epochstep99globalstep99")` [16] - this directory is a full model and is loader-compatible out of the box.
- Resume training from a RecoverHandler checkpoint by setting `recover.mode: on` (or `auto`) with a matching `freq_steps`/`freq_epochs`/`freq_secs`; on restart, `RecoverHandler.load()` restores dataloader, saver, and evaluator state and training continues from the recorded `global_step`, retrying up to `retries` times on failure - the docs' own best-practice note requires "identical parallelism configuration, experiment name, and trial name" for the DCP checkpoint to be valid [16].
- LoRA changes what a save contains: the LoRA reference page states that with LoRA enabled, "only the LoRA adapters need to be saved and shipped" rather than full model weights [17], but neither that page nor the checkpointing reference documents the on-disk adapter file layout, a merge call, or whether `Saver`'s HuggingFace-format export writes adapter-only or merged weights when `use_lora=true` - this is a gap in the docs read for this card, not a confirmed behavior either way.
- Loader handoff: a Saver checkpoint directory is directly `from_pretrained`-loadable per the docs' own example [16]; whether an external evaluator can load a LoRA-mode save the same way is unconfirmed for the reason above and should be checked on disk before relying on it.

## Find it in the docs

- Address pattern: `https://areal-project.github.io/AReaL/en/<slug>.html`, confirmed working for `<slug>` values `intro`, `tutorial/quickstart`, `tutorial/installation`, `tutorial/eval`, `reference/checkpointing`, `reference/metrics_tracking`, `reference/alloc_mode`, `reference/lora`, `algorithms/async`, `algorithms/grpo_series`, `best_practices/handling_oom`, and `best_practices/cli_guide` - each of these pages was fetched directly at this pattern for this card [1][2][6][7][13][14][15][16][17][18][19][20][21]. `<slug>` groups pages under `tutorial/`, `reference/`, `algorithms/`, and `best_practices/` rather than sitting flat, so a guess like `en/installation.html` (no `tutorial/` prefix) will not match; the correct nesting is discoverable from the sidebar navigation on any fetched page [7][18].
- Page-slug recipes: tutorials sit under `tutorial/`, reference material under `reference/`, algorithm pages under `algorithms/`, and operational guides under `best_practices/` [18]. The Overview/landing page itself (`en/intro.html`) carries almost no prose beyond a welcome line and the full sidebar - use it only for the nav tree, not content [18].
- Question-to-slug map: "how do I size GPUs for a run" -> `reference/alloc_mode` (the backend-string dimension table and worked GPU-count examples) [6]; "what do I log and where" -> `reference/metrics_tracking` [14]; "how do checkpoints work" -> `reference/checkpointing` [16]; "how do I run LoRA" -> `reference/lora` [17]; "how do I control on/off-policy lag" -> `algorithms/async` [20]; "how do I avoid OOM" -> `best_practices/handling_oom` [13]; "how do I drive training/inference/agent processes from the CLI" -> `best_practices/cli_guide`, which documents three subcommand groups, `areal train run --config <path> --driver <module.path>:<func>`, `areal inf run --service ... --model-path ...` (an OpenAI-compatible local inference service under `~/.areal/inf/`), and `areal agent run --agent my_package.my_agent.MyAgent --num-pairs N` (a session-based agent service under `~/.areal/agent/`, where `--agent` takes a plain dot-path import string for the agent class rather than the colon-separated form used by `--driver`) [19].
- Runnable references beyond the docs: the repo's `examples/` tree covers math/reasoning (GSM8K with GRPO/PPO/DAPO/REINFORCE/RLOO/LitePPO/Dr.GRPO/GSPO), agentic RL (general agent workflows, an end-to-end coding-agent example, a search agent, tool-integrated reasoning), vision-language models, and alignment/reward-modeling tasks, each linked from the README's own Examples tables [3]. The quickstart's own smoke-test path is `examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml`, which self-downloads `openai/gsm8k` and `Qwen/Qwen2-1.5B-Instruct` [1].
- Community layer: the README does not curate a separate community-tutorials page the way some sibling libraries do; instead it names a WeChat group and a GitHub Discussions board as its community channels [1], and separately links a Community Repository (`github.com/areal-project/community`) for meeting materials [1]. No independent, dated practitioner blog is curated by AReaL's own pages in the sources read for this card.
- No official MCP endpoint for querying these docs was found in the pages read for this card.
- Honest boundaries and traps: the LoRA support matrix itself states Archon supports no LoRA combination (vLLM or SGLang) and Megatron only pairs LoRA with vLLM, not SGLang [17]. The org was formerly `inclusionAI/AReaL` on GitHub with docs at `inclusionai.github.io/AReaL`; the v1.0.4 release tag's own `pyproject.toml` still lists `project.urls` under the old `inclusionAI` org even though the shortlisted repo has since moved to `areal-project/AReaL` [9] - a reader following the v1.0.4 package metadata's URLs by hand would land on the old, possibly-redirecting org.

## Sources

All docs pages are unpinned, mutable builds read on the dates given; the screening-commit code citations are explicitly noted as ahead of the shipped 1.0.4 release. GitHub's REST API (`api.github.com`) could not be queried for this card - repository metadata, stars, and tag/release listings returned rate-limit errors - so no claim here is attributed to that API; tag-to-commit resolution instead used `git ls-remote`.

[1] AReaL README at commit 6c0f9521e1f01d66864665644f83f9e2015ce327. https://raw.githubusercontent.com/areal-project/AReaL/6c0f9521e1f01d66864665644f83f9e2015ce327/README.md. Fetched 2026-08-10.

[2] AReaL Quickstart tutorial. https://areal-project.github.io/AReaL/en/tutorial/quickstart.html. Fetched 2026-08-10.

[3] AReaL README, Support Matrix and Examples tables (same fetch as [1]). https://raw.githubusercontent.com/areal-project/AReaL/6c0f9521e1f01d66864665644f83f9e2015ce327/README.md. Fetched 2026-08-10.

[4] AReaL Quickstart tutorial, "Legacy: SPMD Mode with Dedicated Launchers" section (same page as [2]). https://areal-project.github.io/AReaL/en/tutorial/quickstart.html. Fetched 2026-08-10.

[5] AReaL README, News/changelog section (same fetch as [1]). https://raw.githubusercontent.com/areal-project/AReaL/6c0f9521e1f01d66864665644f83f9e2015ce327/README.md. Fetched 2026-08-10.

[6] AReaL Allocation Mode reference. https://areal-project.github.io/AReaL/en/reference/alloc_mode.html. Fetched 2026-08-10.

[7] AReaL Installation tutorial (hardware/software requirements table, uv sync commands, vLLM lockfile swap). https://areal-project.github.io/AReaL/en/tutorial/installation.html. Fetched 2026-08-10.

[8] areal package on PyPI (JSON API). https://pypi.org/pypi/areal/json. Fetched 2026-08-10.

[9] AReaL pyproject.toml at git tag v1.0.4, resolved via `git ls-remote` to commit 37d6c6400e99a05fa3409d6a067762a44df40d3b. https://raw.githubusercontent.com/areal-project/AReaL/v1.0.4/pyproject.toml. Fetched 2026-08-10.

[10] AReaL pyproject.vllm.toml at git tag v1.0.4 (same resolved commit as [9]). https://raw.githubusercontent.com/areal-project/AReaL/v1.0.4/pyproject.vllm.toml. Fetched 2026-08-10.

[11] AReaL pyproject.toml at the screening commit 6c0f9521e1f01d66864665644f83f9e2015ce327 (shows in-repo version 2.0.0, ahead of the published 1.0.4 release). https://raw.githubusercontent.com/areal-project/AReaL/6c0f9521e1f01d66864665644f83f9e2015ce327/pyproject.toml. Fetched 2026-08-10.

[12] AReaL areal/api/cli_args.py at the screening commit 6c0f9521e1f01d66864665644f83f9e2015ce327 (TrainEngineConfig, InferenceEngineConfig, _DatasetConfig defaults; searched for early-stopping/patience/threshold fields). https://raw.githubusercontent.com/areal-project/AReaL/6c0f9521e1f01d66864665644f83f9e2015ce327/areal/api/cli_args.py. Fetched 2026-08-10.

[13] AReaL Handling OOM Issues best-practices page. https://areal-project.github.io/AReaL/en/best_practices/handling_oom.html. Fetched 2026-08-10.

[14] AReaL Metrics Tracking reference. https://areal-project.github.io/AReaL/en/reference/metrics_tracking.html. Fetched 2026-08-10.

[15] AReaL Evaluation tutorial. https://areal-project.github.io/AReaL/en/tutorial/eval.html. Fetched 2026-08-10.

[16] AReaL Checkpointing reference. https://areal-project.github.io/AReaL/en/reference/checkpointing.html. Fetched 2026-08-10.

[17] AReaL LoRA Reference. https://areal-project.github.io/AReaL/en/reference/lora.html. Fetched 2026-08-10.

[18] AReaL Overview/landing page. https://areal-project.github.io/AReaL/en/intro.html. Fetched 2026-08-10.

[19] AReaL CLI Guide best-practices page (`areal train`, `areal inf`, `areal agent` subcommands). https://areal-project.github.io/AReaL/en/best_practices/cli_guide.html. Fetched 2026-08-10.

[20] AReaL Asynchronous RL algorithms page. https://areal-project.github.io/AReaL/en/algorithms/async.html. Fetched 2026-08-10.

[21] AReaL GRPO-series algorithms page. https://areal-project.github.io/AReaL/en/algorithms/grpo_series.html. Fetched 2026-08-10.
