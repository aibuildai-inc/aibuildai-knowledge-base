# miles

An enterprise-facing RL post-training framework, forked from slime, that pairs SGLang rollout with a production Megatron-LM training backend and adds unified low-precision, weight-sync, and fault-tolerance machinery for MoE-scale jobs.

**Miles** is "a high-performance, enterprise-ready reinforcement learning (RL) framework specifically optimized for Large-Scale model Post-Training," built as a fork of slime that couples SGLang for rollout with Megatron-LM for training [1][2]. It is maintained by the radixark organization on GitHub [3], and its API is bash launch scripts that assemble named argument-group arrays (`MODEL_ARGS`, `CKPT_ARGS`, `ROLLOUT_ARGS`, `GRPO_ARGS`, ...) and submit `train.py` (or `train_async.py`) as a Ray job [4][5]. It lives at https://github.com/radixark/miles [3].

**When to pick it**: production-shaped RL post-training at MoE / trillion-parameter scale on NVIDIA Hopper or Blackwell (with AMD MI300X-class ROCm support), when you want a Megatron-LM training backend with unified low-precision (FP8/MXFP8) rollout-training and rank-level fault tolerance built in, rather than trl's single-machine-first Accelerate/DeepSpeed story or a research-first framework without those production features [1][6]. It is a fork of and co-evolves with slime [2] (cross-reference; not covered here) - pick slime instead if you want the upstream, non-enterprise-branded project. The default training backend, Megatron-LM, is the production path; an in-tree FSDP2 backend exists but the docs mark it explicitly experimental with no TP/PP/CP/EP, so it is not a substitute for Megatron at MoE or multi-rack scale [7].

**Methods it ships**: the RL objective is selected with `--advantage-estimator`, whose documented enum is `grpo` (default), `gspo`, `ppo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, and `on_policy_distillation` [8]. SFT is a separate rollout path (`miles/rollout/sft_rollout.py` per the shortlist's own tree probe). Built-in reward types are selected with `--rm-type`: `math`, `dapo`, `deepscaler`, `f1`, `gpqa`, `ifbench`, `remote_rm`, `random` [9]; a DAPO-style dynamic-sampling group filter is available via `--dynamic-sampling-filter-path` [9]. Two precision-and-routing stability methods sit alongside the objective: Rollout Routing Replay (R3, `--use-rollout-routing-replay`), which captures inference-side MoE expert routing and replays it during training [9], and Truncated Importance Sampling (`--use-tis`) for residual train/inference precision drift [8][9]. LoRA is supported for both SFT and RL recipes but only on the Megatron backend, only under `--colocate`, and only as a single adapter per run - the launcher asserts at startup if `--target-modules` is unset whenever `--lora-rank > 0` [10]. Two further advanced features are grouped alongside LoRA and FP8 on the docs' own Advanced Features index: INT4 (W4A16) quantization-aware training "for fitting large models on a single 8-GPU node" at `/advanced/int4-qat`, and speculative decoding - draft-plus-target speculative rollout with online MTP-SFT for the draft - at `/advanced/speculative-decoding`; neither page's body was fetched for this card, so their mechanics are not detailed here beyond this index description [34]. This card does not restate any method's math; those definitions belong on each method's own methodology card.

**Scale it handles**: single GPU (`--actor-num-nodes 1 --actor-num-gpus-per-node 8` defaults [11]) up to multi-node, with generation and training either colocated on the same GPUs (`--colocate`, which implicitly turns on `--offload-train`, `--offload-rollout`, and defaults `--sglang-cuda-graph-backend-prefill=disabled` [11]) or disaggregated across separate GPU pools. Sharding on the Megatron backend spans tensor, pipeline, context, expert, and expert-tensor parallelism (TP/PP/CP/EP/ETP), though the docs warn only a subset of TP x PP x CP x EP x ETP combinations is actually supported and to change one dimension at a time from a tested recipe [12]. Weight sync after each update uses NCCL broadcast by default or point-to-point RDMA via the Mooncake transfer engine under `--update-weight-transfer-mode p2p` (p2p cannot combine with `--colocate`) [4]. Multi-node needs a high-bandwidth interconnect - InfiniBand, RoCEv2, or Slingshot at 200+ GB/s per node; single-node jobs run fine over NVLink alone [13]. Unified low-precision rollout-plus-training is documented with a maturity ladder and mechanism per format, not a published throughput benchmark in the pages read for this card: FP8 block-wise (128x128, FP32 scales) is "Generally available" on Hopper and Blackwell, tested on Qwen3-4B, Qwen3-30B-A3B, and DeepSeek-V3/R1; MXFP8 (1x32, UE8M0 scales) is "Beta," Blackwell-only, tested on Qwen3-30B-A3B; NVFP4 (E2M1, MoE-experts-only) is "Experimental," with the full unified recipe still in development [6].

**Install**: no PyPI package exists for radixark/miles - `pypi.org/project/Miles` is an unrelated package by a different author and must not be conflated with this project. There are no git tags and no GitHub Releases in the repository, so the closest analog to a pinned release is `docker pull radixark/miles:latest` (an explicitly rolling, unpinned tag; Docker Hub lists 114 tags under this repository, and `latest` was last pushed 2026-08-10) or installing from source at a specific commit [3][14]. From source: `git clone https://github.com/radixark/miles.git && cd miles && pip install -r requirements.txt && pip install -e . --no-deps`; the docs warn that Miles pins patched versions of SGLang and Megatron-LM and that installing them yourself at the wrong commit is the most common source of bug reports [15]. At the pinned commit `d9e3c3b394c9c50746bb0c9174b7e1afcbb03d96` (2026-07-31), `setup.py` reports `version="0.2.1"`, `python_requires=">=3.10"`, and Apache-2.0 licensing (matching the shortlist row) [16][17]. `requirements.txt` at that commit pins no torch, SGLang, or Megatron version - those ship only through the Docker image as patched forks - but does hard-pin `transformers==5.12.1` ("Required by HF-native weight conversion; also avoids SGLang's qwen3_asr collision in 5.13") and `polars==1.42.1`, plus extras `fsdp` (`torch>=2.0`), `mlflow` (`mlflow>=2.0`), and `dashboard` (`fastapi>=0.135`, `uvicorn>=0.41`, `prometheus_client>=0.24`) declared in `setup.py` [17][18]. No CUDA/torch hardware minimum is stated in the installation docs beyond the hardware table: NVIDIA H100/H200 is "Production (CI guarded)," B100/B200 "Production," A100 "Supported — FP8 features disabled," and AMD MI300X/MI325/MI350X/MI355X "Supported via ROCm" [13].

**Maintained by**: the radixark GitHub organization [3]. The docs index's own "Latest updates" list, read as a dated changelog rather than a single snapshot, shows six entries across four months: 2026/02 (a completed CLI argument reference), 2026/01 (INT4 W4A16 QAT; unified VLM/LLM multi-turn rollout), 2025/12 (Rollout Routing Replay for MoE), and 2025/11 (unified FP8 pipeline reaching general availability; speculative decoding with online MTP-SFT) [34]. The repository's own last-push timestamp, a live snapshot rather than something pinned to the reviewed commit, was 2026-07-31 per the shortlist row and 2026-08-11 as read from the live API during this card's research - both after the docs index's newest dated entry, consistent with ongoing activity [3].

## Quick start

The quickstart page's five-step Qwen3-4B recipe, quoted/transcribed from its own commands [4]:

```bash
# 1. On the host - start the container
docker pull radixark/miles:latest
docker run --rm \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -it radixark/miles:latest /bin/bash

# inside the container, run the latest main
cd /root/miles && git pull && pip install -e . --no-deps

# 2. Download model and data
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17K --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024

# 3. Convert to Megatron torch_dist format
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-4B \
   --save /root/Qwen3-4B_torch_dist

# 4. Launch training
bash scripts/run-qwen3-4B.sh
```

The launched run is colocated: four SGLang engines (2 GPUs each) and the Megatron trainer share the same 8 GPUs, alternating generation and training phases, with `--rm-type deepscaler` as a rule-based verifier reward and `--kl-loss-coef 0.00` (KL term off) in this recipe [4]. The README's terser one-liner form of the same launch is `python train.py --advantage-estimator grpo --model-name qwen3-30b-a3b --hf-checkpoint /path/to/qwen3-30b-a3b-hf --rollout-batch-size 512 --n-samples-per-prompt 8` [2] - the two pages give different quick-start snippets for different models; this card carries both because they diverge in shape (five-step recipe vs. a single command) as well as model.

## Start it

- One node, one shared GPU pool: `bash scripts/run-qwen3-4B.sh` as above, with `--colocate` set inside the script [4].
- Multiple GPUs/nodes go through Ray: the launch script starts a local Ray cluster and submits `train.py` as a Ray job (`ray job submit --address=auto -- python3 train.py ...`) [4][19]. Cluster topology is set with `--actor-num-nodes` (default 1) and `--actor-num-gpus-per-node` (default 8); `--rollout-num-gpus` is ignored under `--colocate` [11]. Argument-group templates for a given concern live at fixed repo paths, e.g. architecture constants source from `scripts/models/<family>.sh` into `MODEL_ARGS` [20].
- Effective batch arithmetic is the four-knob invariant: `rollout_batch_size x n_samples_per_prompt = global_batch_size x num_steps_per_rollout`; Miles fills in whichever side is left unset and validates the rest, and the quick-start recipe instantiates it as 32 prompts x 8 samples = 256 = one optimizer step at global batch size 256 [4][8].
- Generation-layout choice for online methods: colocated (`--colocate`, sharing GPUs, implicitly enabling `--offload-train`/`--offload-rollout` and defaulting `--sglang-cuda-graph-backend-prefill=disabled` [11]) versus disaggregated, which instead chooses a weight-transfer transport with `--update-weight-transfer-mode`: `broadcast` (default, NCCL collective) or `p2p` (RDMA via the Mooncake transfer engine; incompatible with `--colocate`) [4].
- A second, orthogonal cadence choice is synchronous versus asynchronous rollout. By default the trainer blocks on `generate()` before `train_step()` fires, so per-iteration wall time is the sum of rollout and train time; asynchronous rollout instead runs a background worker that keeps `--rollout-batch-size` generations in flight and feeds a queue the trainer drains, dropping per-iteration wall time to roughly `max(rollout_time, train_time)`. It is enabled by launching `train_async.py` in place of `train.py` with an added `--rollout-function-path miles.rollout.fully_async_rollout.generate_rollout_fully_async` flag. The docs' own comparison table gives async higher per-iteration latency but up to roughly 2x the overall throughput of sync, recommending sync for strict on-policy training or debugging and async for rollout-bound jobs and long runs [35].
- Config surface: Megatron's entire argument parser is imported at launch (`from megatron.training.arguments import parse_args`), so every installed-Megatron flag works without Miles re-declaring it, and Miles threads its own flags in via an `extra_args_provider` [21]. Any flag accepted by `python -m sglang.launch_server` is accepted with a `--sglang-` prefix, and two flags are set BY Miles rather than the user: `--tp-size` (from `--rollout-num-gpus-per-engine`) and `--model-path` (from `--hf-checkpoint`) [22]. `--lr` defaults to `1e-6` - the docs flag post-training as sensitive to large updates and note recipes typically stay near that value, a changed-from-general-purpose-training default worth knowing before raising it [23].
- Out-of-memory first aid: with `--use-dynamic-batch-size` on (recommended for varlen workloads), `max_tokens_per_gpu` caps tokens per GPU per micro-batch; the FAQ's safe starting point is `max_tokens_per_gpu = rollout_max_response_len / cp_size`, then increase until OOM and back off about 10% - if still OOM with a small value, the individual samples are too long and `--context-parallel-size N` should be raised to spread one sample across N ranks [24]. On the generation side, colocation memory is tight because Megatron reserves VRAM during init before handing off to SGLang; the fix is dropping `--sglang-mem-fraction-static` to 0.8 or lower [25].

## Watch it

This section covers mechanics only - which signals emit and how to turn them on. What each signal means for GRPO/PPO/etc. lives on that method's own methodology card, not here.

- **Enable it**: Miles emits a structured per-rollout row to stdout by default (illustrative shape from the docs: `[trainer] iter 12/3000 | loss=0.412 reward=0.61 kl=0.018 | rollout=18.4s train=22.1s p2p=2.1s (total 42.6s) | grad_norm=0.93 lr=1.0e-06`) [26]. Weights & Biases is opt-in via `--use-wandb --wandb-project <name> --wandb-group <name>`, with `WANDB_API_KEY` supplied through Ray's `env_vars` rather than baked into the script [26].
- **Metric namespaces**: when wandb is on, metrics land under `train/`, `rollout/`, `perf/`, `multi_turn/`, and `passrate/` namespaces [26]. The quick-start page names two concrete metrics used during a first run: `rollout/raw_reward` (mean reward of freshly scored responses - "the number to watch") and the pair `perf/rollout_time` vs `perf/actor_train_time` to see whether generation or training is the bottleneck [4].
- **What to watch table** (from the monitoring page - eight signals with a healthy pattern and a red flag each) [26]: `loss` (slow decay over hundreds of iterations vs. a spike-to-crash within one iteration); `raw_reward` (trending up with healthy variance vs. saturating at a single value - collapse); `kl_loss` (bounded, drifts up over time vs. a sudden jump indicating divergence from the reference - only logged when `--use-kl-loss` is set); `train_rollout_logprob_abs_diff` (stable and small, "much less than 1.0," vs. climbing without bound, signaling train/inference precision drift); `entropy_loss` (slowly decreasing vs. falling to near zero too fast - mode collapse); `grad_norm` (below `clip_grad`, default 1.0, vs. repeatedly hitting the clip threshold); `rollout_time` / `train_time` (roughly balanced vs. one much greater than the other - resource imbalance); `train/pg_clipfrac` (below 0.2 is healthy; above 0.5 means the policy is moving fast and the guidance is to drop the learning rate).
- **Sample-level logging**: a custom rollout logger can replace the default one to push per-sample data to internal systems, wired through `--custom-rollout-log-function-path <module>.<fn>`, a function taking `(rollout_id, args, samples, extra, rollout_time)` and returning a bool that controls whether default logging also runs [26].
- **Evaluation during training**: `--eval-prompt-data` names one or more eval sets, `--eval-interval` sets the rollout cadence, `--n-samples-per-eval-prompt` sets the eval group size, and `--eval-max-response-len`/`--eval-temperature`/`--eval-top-p` each inherit from the rollout-side value if left unset [27]. The quick-start recipe evaluates on AIME-2024 every 20 rollouts via `--eval-interval 20` [4].
- **Profiling**: `nvidia-smi dmon -s u` for a quick GPU-utilization check, `nsys profile` for CUDA-level profiling, `py-spy dump --pid <ray worker>` for Python-side stalls, and `ray timeline` for Ray task scheduling; a built-in PyTorch profiler is wired in separately per backend (`--profile-target train_overall|train_actor|train_log_probs` on Megatron; `--use-pytorch-profiler --profile-step-start --profile-step-end --memory-snapshot-path --tensorboard-dir` additionally on FSDP) [28].
- **Stopping-rule honesty**: no RL-specific early-stopping threshold or patience field is published. This is a 2026-08-11 reading of the three pages that would carry one: the CLI reference's Complete-reference pass lists every Miles flag with type and default and contains no stopping-rule or patience field in its RL-algorithm, optimizer, or logging sections [8][9]; the monitoring page's health-limit content is the `train/pg_clipfrac`/"drop LR" guidance quoted above, which is advice, not an enforced threshold [26]; the FAQ's closest health-limit-adjacent guidance is `--no-check-for-nan-in-loss-and-grad` to skip NaN/Inf steps temporarily and investigate the underlying cause, again not a stopping rule [29]. Shapes are published (the "What to watch" table); thresholds beyond `clip_grad` (1.0) and the 0.2/0.5 `pg_clipfrac` bands are not.

## Save it

- Checkpoint format is Megatron's `torch_dist`: parallelism-agnostic `.distcp` files, so TP/PP/EP can change without reconversion. A checkpoint directory looks like `/ckpt/latest_checkpointed_iteration.txt`, `iter_0000100/_0_0.distcp` (and siblings), `iter_0000200/`, and so on; the docs state plainly: "Always pass the parent directory to `--load`, not a specific iteration" - the loader reads `latest_checkpointed_iteration.txt` to pick the step [30].
- Cadence is `--save-interval` (rollouts between saves); the quick-start recipe uses `--save-interval 20`, writing to `/root/Qwen3-4B_miles/` [4][27]. No checkpoint-retention-limiting flag (an analog to trl's `save_total_limit`) was found anywhere in the Complete-reference pass of the CLI reference, which lists every Miles flag with type and default [9].
- On-demand save: `--save-trigger-sentinel <path>` forces a checkpoint outside the normal `--save-interval` cadence; touching the sentinel path and waiting for it to disappear is the documented pattern (`touch /path/to/save_now && until [ ! -e /path/to/save_now ]; do sleep 5; done`). The contract, quoted: "A request fired at any moment during an iteration is consumed at that iteration's save point... the checkpoint is written with `force_sync=True`... and only then is the sentinel file deleted. 'File gone' means 'checkpoint durable on disk.' If the job crashes mid-save, the sentinel survives and the request stays pending for the next run. Requires `--save` to be set." [30]
- Resume: set `--load` to the same directory `--save` was writing to and relaunch; the FAQ states this as the entire procedure ("That's it") [24][31]. The quick-start recipe relies on the same mechanism for crash recovery: relaunching the script resumes from the last checkpoint because `--load` points at the save directory [4].
- LoRA changes what a save IS: only the Megatron backend has a LoRA path, only under `--colocate`, and only one set of `--lora-*` arguments is honored per run - training multiple adapters in parallel within one `train.py` run is not implemented [10]. `--target-modules` is required whenever `--lora-rank > 0` with no auto-detection; the launcher asserts at startup if it is missing [10]. `--megatron-to-hf-mode bridge` is required for the LoRA path (the default `raw` converter does not understand LoRA layers) [10]. Adapters trained by Miles load directly into SGLang for rollout with no separate merge step [10]; for MoE models the default SGLang LoRA backend silently drops expert adapters at inference time and logs "Current LoRA backend does not support LoRA on MoE layers; skipping MoE layer" unless `--sglang-lora-backend triton` is set [10] - a trap that bites exactly at the save/serve boundary, not at save time itself.
- The docs read for this card do not enumerate a full checkpoint file inventory beyond the `.distcp`/`latest_checkpointed_iteration.txt` layout shown above [30]; whether an evaluator can load a saved `torch_dist` checkpoint directly is the loader's contract, not documented on the pages read here - confirm against the target evaluation tool's own loader before assuming direct compatibility.

## Find it in the docs

The docs are the live source; this card teaches the lookup, not the content.

- Address pattern: the docs site is `https://miles.radixark.com/docs`, and both `https://miles.radixark.com/docs` and `https://miles.radixark.com/docs/getting-started/quick-start` were confirmed to return HTTP 200 on a 2026-08-11 fetch; no version-tag path segment was found in any URL cited from this site, so pages appear to be served unversioned/live rather than per-release [4][32].
- Page-slug recipes: getting-started pages live at `/getting-started/<slug>` (`quick-start`, `installation`); user-guide pages at `/user-guide/<slug>` (`usage`, `cli-reference`, `monitoring`, `concepts`, `argument-groups`, `dashboard`, `training-script-walkthrough`); advanced pages at `/advanced/<slug>` (`lora`, `fault-tolerance`, `fp8-low-precision`); developer pages at `/developer/<slug>` (`architecture`, `experimental-features`, `debug`) [4][6][8][9][10][11][12][21][26][30].
- Question-to-slug map: batch-sizing math and cluster-topology defaults -> `/user-guide/cli-reference` [8][11]; what to watch and how logging is enabled -> `/user-guide/monitoring` [26]; Megatron parallelism, checkpoint format, and hooks -> `/user-guide/usage` [21][30]; LoRA -> `/advanced/lora` [10]; low-precision recipes -> `/advanced/fp8-low-precision` [6]; rank recovery and P2P timeouts -> `/advanced/fault-tolerance` [33]; the FSDP experimental backend -> `/developer/experimental-features` [7]; common first-week errors -> `/faq` [24][29]; sync-vs-async rollout cadence -> `/user-guide/training-script-walkthrough` [35]; the full advanced-features menu, including INT4 QAT (`/advanced/int4-qat`) and speculative decoding (`/advanced/speculative-decoding`), neither fetched in full for this card -> `/advanced/index` [34].
- Runnable references beyond the docs: the `examples/` tree, including `examples/lora/` for the three canonical LoRA launchers and `examples/infra_features/low_precision/` for the FP8 recipes [10][6]; `scripts/models/<family>.sh` for per-architecture `MODEL_ARGS`, and `scripts/run-qwen3-4B.sh` / `scripts/run_qwen3_30b_a3b.py` as named reference recipes [4][6]. Known-good smoke-test datasets named by the quick-start page: `BytedTsinghua-SIA/DAPO-Math-17K` for training prompts and `zhuzilin/aime-2024` for evaluation-only [4].
- Community layer: no curated community-tutorials page was found among the docs pages read for this card (no page analogous to trl's `community_tutorials` slug appeared in the site's navigation as read); FAQ instead directs stuck users to the Miles channel of the SGLang Slack (`https://slack.sglang.ai`) or to opening a GitHub issue [24].
- No official MCP endpoint for querying these docs was found in any page read for this card.
- Documented trap, stated where it bites: for MoE LoRA, the default SGLang LoRA backend silently skips expert-layer adapters at inference and requires `--sglang-lora-backend triton` instead - this is a docs-stated behavior, not a maintainer issue reply, since no closed-issue search was performed for this card [10]. A second documented trap: under `--update-weight-transfer-mode p2p`, a failed transfer is logged but "there is no automatic retry or automatic broadcast-mode fallback in the source today," per the fault-tolerance page's read of `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py` at the pinned commit [33].
- Honest boundary: the FSDP training backend has no TP/PP/CP/EP - it runs as plain data parallel only and is explicitly marked experimental, not for production runs [7]; LoRA is unavailable on that FSDP backend and, on the Megatron backend, unavailable in disaggregated (PD) rollout mode - `NotImplementedError` is raised at weight-sync time if attempted [10].

## Sources

Method names (GRPO, PPO, GSPO, REINFORCE++, DAPO, LoRA, ...) are deliberately cited to nothing here; their defining papers belong on the methodology cards. All docs pages below are unversioned/live pages read on 2026-08-11 unless a commit is named; code-path claims carry the pinned commit `d9e3c3b394c9c50746bb0c9174b7e1afcbb03d96` (2026-07-31) named in the shortlist row. `repo.json`'s live snapshot (`pushed_at: 2026-08-11T23:04:26Z`) is newer than that pinned commit and is cited only for stars/forks/license/description, not for code-path claims.

[1] miles docs index page. https://miles.radixark.com/docs. Fetched 2026-08-11 (saved as docs_raw/docs_index.md).

[2] miles GitHub README. https://github.com/radixark/miles/blob/main/README.md (saved as readme.md). Fetched 2026-08-11 at commit d9e3c3b394c9c50746bb0c9174b7e1afcbb03d96.

[3] miles GitHub repository. https://github.com/radixark/miles. Fetched 2026-08-11 (repo.json - live snapshot: stars, forks, license, pushed_at).

[4] miles Quick Start docs page. https://miles.radixark.com/docs/getting-started/quick-start. Fetched 2026-08-11 (saved as docs_raw/docs_getting-started_quick-start.md).

[5] miles Argument Groups docs page. https://miles.radixark.com/docs/user-guide/argument-groups. Fetched 2026-08-11 (saved as docs_raw/docs_user-guide_argument-groups.md).

[6] miles Low Precision RL docs page. https://miles.radixark.com/docs/advanced/fp8-low-precision. Fetched 2026-08-11 (saved as docs_raw/docs_advanced_fp8-low-precision.md).

[7] miles Experimental Features docs page. https://miles.radixark.com/docs/developer/experimental-features. Fetched 2026-08-11 (saved as docs_raw/docs_developer_experimental-features.md).

[8] miles CLI Reference docs page, Essentials pass (RL algorithm / batch sizing sections). https://miles.radixark.com/docs/user-guide/cli-reference. Fetched 2026-08-11 (saved as docs_raw/docs_user-guide_cli-reference.md).

[9] miles CLI Reference docs page, Complete reference pass (RL algorithm / reward and filters / logging sections). Same URL and file as [8].

[10] miles LoRA Training and Serving docs page. https://miles.radixark.com/docs/advanced/lora. Fetched 2026-08-11 (saved as docs_raw/docs_advanced_lora.md).

[11] miles CLI Reference docs page, Cluster topology and Complete-reference Cluster sections. Same URL and file as [8].

[12] miles Training Backend (usage) docs page, Parallelism compatibility section. https://miles.radixark.com/docs/user-guide/usage. Fetched 2026-08-11 (saved as docs_raw/docs_user-guide_usage.md).

[13] miles Installation docs page, Hardware requirements table. https://miles.radixark.com/docs/getting-started/installation. Fetched 2026-08-11 (saved as docs_raw/docs_getting-started_installation.md).

[14] Docker Hub tags listing for radixark/miles. https://hub.docker.com/r/radixark/miles/tags (queried via Docker Hub API; saved as dockerhub_tags.json). Fetched 2026-08-11.

[15] miles Installation docs page, Method 2: From source, and its Warning box. Same URL and file as [13].

[16] miles repository commit metadata for d9e3c3b394c9c50746bb0c9174b7e1afcbb03d96. https://github.com/radixark/miles/commit/d9e3c3b394c9c50746bb0c9174b7e1afcbb03d96 (queried via GitHub API; saved as commit.json). Fetched 2026-08-11.

[17] miles setup.py at the pinned commit. https://raw.githubusercontent.com/radixark/miles/d9e3c3b394c9c50746bb0c9174b7e1afcbb03d96/setup.py (saved as setup.py). Fetched 2026-08-11.

[18] miles requirements.txt at the pinned commit. https://raw.githubusercontent.com/radixark/miles/d9e3c3b394c9c50746bb0c9174b7e1afcbb03d96/requirements.txt (saved as requirements.txt). Fetched 2026-08-11.

[19] miles Monitoring & Logging docs page, Enabling wandb section (ray job submit form). Same URL and file as [26].

[20] miles Argument Groups docs page, MODEL_ARGS section. Same URL and file as [5].

[21] miles Training Backend (usage) docs page, Parameter discovery and Hooks sections. Same URL and file as [12].

[22] miles Training Backend (usage) docs page, SGLang as the inference engine / Passthrough convention section. Same URL and file as [12].

[23] miles CLI Reference docs page, Essentials Optimizer section (`--lr` default and note). Same URL and file as [8].

[24] miles FAQ docs page, "I'm OOM during training" and "How do I resume training?" accordions. https://miles.radixark.com/docs/faq. Fetched 2026-08-11 (saved as docs_raw/docs_faq.md).

[25] miles Training Backend (usage) docs page, Colocation memory subsection. Same URL and file as [12].

[26] miles Monitoring & Logging docs page. https://miles.radixark.com/docs/user-guide/monitoring. Fetched 2026-08-11 (saved as docs_raw/docs_user-guide_monitoring.md).

[27] miles CLI Reference docs page, Complete-reference Eval and Model-and-checkpoints sections. Same URL and file as [8].

[28] miles Monitoring & Logging docs page, Profiling section. Same URL and file as [26].

[29] miles FAQ docs page, "Gradient is NaN / Inf" accordion. Same URL and file as [24].

[30] miles Training Backend (usage) docs page, Checkpoint format and On-demand save sections. Same URL and file as [12].

[31] miles FAQ docs page, "How do I resume training?" accordion. Same URL and file as [24].

[32] Live-fetch verification of the docs index and quick-start pages returning HTTP 200 (saved as live_index.html, live_quickstart.html). Fetched 2026-08-11.

[33] miles Fault Tolerance docs page. https://miles.radixark.com/docs/advanced/fault-tolerance. Fetched 2026-08-11 (saved as docs_raw/docs_advanced_fault-tolerance.md).

[34] miles docs index page, Latest updates section, and miles Advanced Features index page (CardGroup listing INT4 QAT and Speculative Decoding). https://miles.radixark.com/docs and https://miles.radixark.com/docs/advanced/index. Fetched 2026-08-11 (saved as docs_raw/docs_index.md and docs_raw/docs_advanced_index.md).

[35] miles Training Script Walkthrough docs page, Synchronous vs. asynchronous rollout section. https://miles.radixark.com/docs/user-guide/training-script-walkthrough. Fetched 2026-08-11 (saved as docs_raw/docs_user-guide_training-script-walkthrough.md).
