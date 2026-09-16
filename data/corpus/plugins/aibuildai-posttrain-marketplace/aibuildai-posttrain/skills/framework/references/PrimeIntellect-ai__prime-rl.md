# prime-rl

Prime Intellect's async RL trainer: three cooperating processes (inference, orchestrator, trainer) that scale from one consumer GPU to 1000+-GPU SLURM clusters without changing the algorithm code.

`prime-rl` is "a framework for large-scale, asynchronous reinforcement learning of large language models" that is "designed to be easy to use and hackable, yet capable of training 1T+-parameter MoE models on 1000+ GPU clusters" [1]. It is built and maintained by Prime Intellect [2][3]. Its API is TOML-plus-CLI, not a Python trainer class: a run is one `[[orchestrator.train.source]]`-per-environment TOML config (composed with `pydantic-config`) launched through one of a handful of entrypoint scripts (`rl`, `sft`, `trainer`, `orchestrator`, `inference`) that each read the config, validate it, and start the corresponding process(es) [4][1].

**When to pick it**: agentic and multi-turn online RL against Prime Intellect's `verifiers` environments (tool use, sandboxes, long-horizon rollouts), when you need MoE-scale trainer parallelism (expert parallelism with DeepEP kernels, context parallelism for long sequences) and SLURM-native multi-node deployment out of the box; weigh trl or verl instead if you want a Hub-native trainer-class API or you are not building on the `verifiers`/Environments Hub ecosystem (cross-reference; not covered here). The repo's live GitHub description reads "Agentic RL Training at Scale" [3], while the v0.8.0-release README and `pyproject.toml` both still read "Async RL Training at Scale" [4][5] — the two have not been kept in sync, so the async architecture (below) is the release-documented framing this card uses.

**Methods it ships**: algorithms are configured, not imported — set `[orchestrator.algo].type` per run or per environment [6]. The shipped types are `grpo` (default, standard group-relative RL, no frozen model) [6]; `max_rl` (MaxRL, arXiv:2602.02710, centers the group reward by its mean rather than its standard deviation) [6]; `rae` (SPIRAL's self-play advantage estimator, arXiv:2506.24119, an EMA per-agent baseline for multi-agent envs) [6]; `hierarchical_grpo` (separate group baselines for proposer and solver roles in proposer-solver environments) [6]; `opd` (on-policy distillation, per-token reverse KL against a frozen `teacher` endpoint, credited to a Thinking Machines blog post) [6]; `sft` (hard distillation: a frozen model's rollouts trained with cross-entropy) [6]; `opsd` (SDFT, arXiv:2601.19897, self-distills the live policy against its own demonstration-conditioned rollout, no frozen model) [6]; and `echo` (GRPO plus a weighted cross-entropy loss on environment-provided observation tokens, selected by message role) [6]. SFT is a separate entrypoint (`uv run sft`) built on the same trainer, not an `orchestrator.algo` type [4]. The docs give no experimental/stable split for these types; each type's class defaults are described as its "vetted setting" [6]. The live page for this taxonomy is the algorithms doc [6].

**Scale it handles**: single GPU up to 1000+ GPUs on SLURM, documented at that ceiling in the project's own framing [1]. One GPU runs the SFT or RL quick-start directly; two GPUs (one inference, one trainer) run the smallest full RL stack via `uv run rl @ <config>.toml` [7]. Multi-GPU/multi-node scaling goes through FSDP2 sharding (`dp_replicate`, `reshard_after_forward`, optional `fsdp_cpu_offload`), expert parallelism for MoE families (`ep="auto"` by default, with a `torch` all-to-all backend by default and a faster `deepep` backend requiring pre-built H100/H200 CUDA-12.9 binaries via `uv sync --all-extras`) [8][9], and context parallelism for long sequences (`cp_style` ulysses by default, ring required for GLM-5) [8]. SLURM multi-node deployment uses Jinja2 sbatch templates and a `[deployment]` block (`single_node` or `multi_node`), with a hard shared-filesystem requirement: "The prime-rl checkout and its `uv` venv must live on a shared filesystem" [8]. The advanced examples directory documents runs from 32 to 2048 GPUs (e.g. Qwen3-30B-A3B, GLM-4.5-Air, a high-throughput GLM-5 P/D-disaggregated run), but the docs read for this card publish no controlled throughput/MFU benchmark table at any of these scales — only the mechanism and the example configs [4].

**Install**: `pip install prime-rl` is not the documented path; setup is `curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash`, which clones the repo, initializes the `verifiers`/`renderers`/`research-environments` submodules, installs `uv`, and runs `uv sync --all-extras` [1]. The latest tagged release is v0.8.0, published 2026-08-07 at commit `60bc29547a8824ad1de7b9af8d265e2b27b2a72d` [10][11] — 17 commits ahead of this card's screening commit `bbb90a1b4132c351cbe8b0ed1fa808dde99f0318` (2026-07-30) [12], the reverse of the usual case: the release is newer than the screening commit, so every version/pin claim below is read at the v0.8.0 tag, not the screening commit. At that tag, `pyproject.toml` sets `requires-python = "~=3.12.0"` and pins `torch>=2.9.0`, `transformers==5.6.2`, `vllm>=0.26.0` [5]. `uv` itself is floored at `required-version = ">=0.11.1"`, with the comment that older `uv` silently bypasses the `exclude-newer = "7 days"` rolling dependency-resolution cooldown the project relies on as a security policy [5]. Licence is Apache-2.0, per the README's license section pointing at the repo's `LICENSE` file [4]. Hardware: at least one NVIDIA GPU from RTX 3090/4090/5090, A100, H100, H200, or B200 [1]; the `[tool.uv].environments` field restricts resolution to linux x86_64/aarch64 [5]. Optional extras at v0.8.0 include `disagg` (deep-ep, deep-gemm, nixl, nixl-cu12, vllm-router, for P/D-disaggregated inference), `flash-attn`, `flash-attn-3`, `flash-attn-cute`, and `quack`, all five bundled together under the `all` extra; `gpt-oss` is a separate extra not included in `all` [5].

**Maintained by**: Prime Intellect [2][3]; about 1889 GitHub stars as of the live fetch, not archived [3]; actively developed, with a v0.8.0 release on 2026-08-07 and the newest commit on the default branch dated 2026-08-11 [10][3].

## Quick start

Both quick starts are real shipped configs; the RL command is quoted from the docs' own Quick Run section [7], the SFT command is quoted from the example's own README [13], and each `[section]` shown is the config file's own content, read at the v0.8.0 release commit [14][15].

SFT (`examples/basic/reverse-text/sft.toml`, one GPU):

```bash
uv run sft @ examples/basic/reverse-text/sft.toml \
  --wandb.project ... \
  --wandb.name ...
```

```toml
max_steps = 100

[ckpt] # Checkpoint at the end of training

[model]
name = "PrimeIntellect/Qwen3-0.6B"

[data]
name = "willcb/R1-reverse-wikipedia-paragraphs-v1-1000"
seq_len = 4096
batch_size = 32

[optim]
lr = 2e-5
```

RL, the full stack (`examples/basic/reverse-text/rl.toml`, two GPUs — one inference, one trainer) [7]:

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
```

```toml
max_steps = 20
seq_len = 2048

[model]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

[wandb]
project = "reverse-text"
name = "reverse-text"

[orchestrator]
batch_size = 128
group_size = 16

[orchestrator.train.sampling]
max_completion_tokens = 128

[[orchestrator.train.source]]
name = "reverse-text"

[orchestrator.train.source.env.taskset]
id = "reverse-text-v1"

[orchestrator.train.source.env.agent.harness]
id = "null"

[orchestrator.train.source.env.agent.runtime]
type = "subprocess"

[trainer.optim]
lr = 3e-6

[ckpt] # Checkpoint at the end of training

[inference]
```

The `rl` entrypoint splits this one file into per-process sub-configs, picks GPU 0 for inference and GPU 1 for the trainer, launches all three processes, and tees stdout into `outputs/logs/{trainer,orchestrator,inference}.log`; the trainer logs `step 1` within a minute, and after 20 steps final HF-compatible weights land at `outputs/weights/step_20` [7].

## Start it

- **One GPU**: the SFT quick start above runs standalone; a single-GPU RL run is also supported for debugging, with production RL typically one inference node plus one or more trainer nodes [1].
- **Several GPUs, one node or SLURM**: launched through the same `rl`/`sft`/`trainer`/`orchestrator`/`inference` entrypoints, composed from multiple TOML files with `@` (left-to-right, deep-merged) or per-section (`--trainer @ trainer.toml --orchestrator @ orch.toml`) [16]. SLURM multi-node runs use sbatch templates and a `[deployment]` discriminated union (`single_node` vs `multi_node`) [8]; ready SLURM-scale reference configs live under `examples/advanced/` (e.g. `examples/advanced/qwen3-30b-a3b/`, `examples/advanced/glm-5.2/`) [4].
- **Config surface and precedence**: values come from Pydantic defaults, then `@`-composed TOML files, then CLI flags in dotted kebab-case (`--trainer.optim.lr 1e-5`); later sources win [16]. Discriminated unions (loss, advantage, optimizer, scheduler, weight-broadcast transport, `[deployment]`, `[orchestrator.algo]`) are selected by a `type` field, defaulting to the class's own default variant if `type` is omitted [16]. `--dry-run --output-dir <dir>` writes the fully resolved per-process TOMLs without launching anything — the documented way to bisect a misbehaving config [16].
- **Effective batch**: RL batching is set on the orchestrator (`orchestrator.batch_size`, `orchestrator.group_size`) rather than a per-device/accumulation product; the quick-start config above uses `batch_size = 128`, `group_size = 16` [15]. The docs' own rule of thumb is a batch-size floor of 64 (128-512 for ablations, 1024+ for production) and a group-size floor of 8 (16-32 common), because a group where every rollout succeeds or fails uniformly collapses its advantage to zero [17].
- **Generation-layout knob this online method adds**: RL is a 3-process topology, not a single trainer process — the orchestrator drives rollouts against a separate vLLM-backed inference server/fleet over ZMQ by default, and weight broadcast from trainer to inference uses NCCL by default, falling back to filesystem transport automatically when LoRA is enabled or no inference server is configured [1].
- **Changed defaults to flag**: multi-tenant training (`trainer.max_concurrent_runs > 1`, for serving many concurrent LoRA tenants off one trainer + inference deployment) is incompatible with `optim_cpu_offload` [8][18]. `optim_cpu_offload` itself defaults to on [8] — a CPU-offload default that assumes host memory headroom, worth checking before assuming a GPU-only memory budget. With the `deepep` expert-parallelism backend, gradient clipping is not supported and `optim.max_norm` is set to `None` automatically [9] — a silent behavior change versus the `torch` EP backend.
- **OOM first aid**: the docs read for this card publish no single OOM checklist page; the load-bearing knobs are activation checkpointing/offloading (full mode is the default, offloading defaults on with `max_inflight_activations=5`, documented at roughly 30-40% peak-memory reduction for about 3-5% throughput loss) [8], `optim_cpu_offload` (default on, mutually exclusive with `fsdp_cpu_offload`) [8], and, on the inference side, `--inference.gpu-memory-utilization` (used at 0.7 in the project's own small-scale smoke test) and `--inference.model.max-model-len` [19].

## Watch it

This section is mechanics only — what a metric means for GRPO or another algorithm's health lives on that method's own card, not here.

- **Enable it**: Weights & Biases logging is off unless `[wandb]` (top-level, for `rl` runs) or `[trainer.wandb]` (standalone) is set; the quick-start RL config above sets `[wandb] project = "reverse-text"` [15]. In a full `rl` run the trainer and orchestrator write into a single shared W&B run — this requires W&B SDK `>=0.19.9` and is incompatible with `wandb.offline = true` [17]. A per-project "overview" saved W&B view is auto-created and re-versioned (e.g. `overview-v2`) if the set of environments changes [17]. Without `[wandb]`/`[trainer.wandb]` set, a run produces no dashboard record — only the log files below.
- **Metric names, RL**: `docs/training.md`'s Important Metrics table groups them as Progress (`reward/{all,env}/mean`, `seq_len/{all,env}/mean`, `is_truncated/{all,env}/mean`, `num_turns/{all,env}/mean`, `empty_rollouts`, `errored_rollouts` — flagged unhealthy above roughly 5% — and `eval/{env}/{avg@k,pass@k}`), Stability (`mismatch_kl/{all,env}/{mean,std,max}`, `entropy/{all,env}/mean`, `masked_advantage_{positive,negative}/mean`, `optim/grad_norm`), and Performance (`time/wait_for_batch`, which signals the orchestrator is the bottleneck, and `time/wait_for_ckpt`, which signals the trainer is) [17].
- **Metric names, SFT**: `loss/mean`, `val/loss`, `progress/epoch`, `progress/num_samples`, `progress/num_tokens`, `progress/<subset>/ratio_{samples,tokens}`, `optim/grad_norm`, `optim/lr`, `optim/zero_grad_ratio`, plus MoE-only `max_vio/mean` and `routing_confidence/mean`; a shared performance table (`perf/throughput`, `perf/throughput_per_gpu`, `perf/mfu`, `perf/peak_memory`, `time/step`, `time/forward_backward`, `time/save_ckpt`) is logged for both SFT and RL trainers [17].
- **Sample-level logging**: sample generations (prompts/completions with rewards and advantages) are logged as W&B tables by default every 10 steps [17]; for VLM runs the prompt text is captured but the images themselves are not logged to monitors [20].
- **Evaluation during training**: `[[orchestrator.eval.source]]` entries mirror `[[orchestrator.train.source]]` (same `env.taskset`/`env.agent` shape) and report `eval/{env}/{avg@k,pass@k}` per the Progress metrics above [16][17].
- **Platform monitoring**: `--orchestrator.prime-monitor` sends metrics to Prime Intellect's own Prime Lab platform, gated behind a `PRIME_API_KEY` and described as internal/allowlisted-team tooling in the docs — not a generally available feature [17].
- **Stopping**: no published stopping-rule, threshold, or patience value was found on the Training guide or the Algorithms guide, the two pages that would carry one — both read in full [17][6]. The only quantitative threshold either page publishes is the `empty_rollouts`/`errored_rollouts` "above ~5%" health smell noted above, which is a data-quality flag, not a stopping rule.

## Save it

- **Checkpoint layout**: trainer state (FSDP-sharded distributed checkpoint plus optimizer, scheduler, and progress) is written under `<output_dir>/checkpoints/step_N/trainer/`; the orchestrator writes its own progress and per-environment state alongside it; the inference process is stateless and saves nothing [17]. Separately, HF-compatible weight snapshots for serving land at `<output_dir>/weights/step_N/` — the quick-start RL run above ends with weights at `outputs/weights/step_20` [7].
- **Off by default**: checkpointing is disabled unless `[ckpt]` is set — an empty `[ckpt]` table (as in both quick-start configs above) enables it with default settings; `--no-ckpt` or the TOML string `"None"` disables an enabled sub-config [16][15][14].
- **Retention**: `ckpt.interval`, `ckpt.keep-last`, and `ckpt.keep-interval` control cadence and how many checkpoints are retained [17]. `ckpt.weights-only` writes the cheaper HF weight snapshot without the full trainer state needed to resume [17].
- **Resume**: `--ckpt.resume-step <N>` resumes from a specific step, or `-1` for the latest checkpoint [17].
- **LoRA adapters**: with `[model.lora]` set, the default RL weight-broadcast transport automatically falls back from NCCL to the filesystem, since NCCL weight broadcast is not supported with LoRA [21]. `[ckpt.weights] save_adapter_separately = true` saves the raw LoRA adapter alongside the merged HF weights [21] — without that flag only the merged full-model weights are written, so an adapter-only save is opt-in, not the default. LoRA pairs with multi-tenant training (`trainer.max_concurrent_runs > 1`, the `MultiRunManager` singleton), where each tenant keeps its own adapter, optimizer, scheduler, checkpoints, and progress while sharing one backbone and one vLLM server — the topology behind Prime Intellect's hosted Lab platform [18].
- **Loader handoff**: the docs read for this card describe the weight snapshots at `<output_dir>/weights/step_N/` as HF-compatible [7], meaning a standard `from_pretrained`-style loader should accept them directly; whether a merged-vs-adapter-only save loads with a given evaluator is that loader's contract and was not independently verified here.

## Find it in the docs

- **Address pattern**: the docs are plain Markdown files served from the GitHub repo tree, not a versioned hosted docs site — e.g. `https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/training.md`, or the raw form `https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/training.md`; swap `main` for a release tag (e.g. `v0.8.0`) to pin a version, verified by fetching the release-tag raw README used throughout this card [1][5].
- **Page-slug recipes**: the docs directory (`docs/`) has one page per topic — `overview.md`, `configuration.md`, `training.md`, `inference.md`, `scaling.md`, `algorithms.md`, `advanced.md`, `development.md` — each linked with a one-line description from the overview page's own Documentation index [1].
- **Question-to-slug map**: TOML composition/CLI syntax -> `configuration.md` [16]; launching and reading RL/SFT runs -> `training.md` [17]; the vLLM-backed serving layer, routers, and P/D disaggregation -> `inference.md` [22]; FSDP/EP/CP/SLURM scaling knobs -> `scaling.md` [8]; algorithm math, loss/advantage/filter plugins, and multi-turn trajectory merging -> `algorithms.md` [6]; custom modeling, multimodal (VLM), LoRA, multi-tenant, and disaggregated inference -> `advanced.md` [20][18][23]; test suite and adding a new model architecture -> `development.md` [24].
- **Runnable references beyond the docs**: the `examples/` tree is called out as the canonical, kept-up-to-date reference (the rest of the repo's TOMLs under `configs/` are CI/debug-internal and may drift) [16]. Basic examples (1-8 GPUs) include Reverse Text (single-turn SFT+RL, one consumer GPU, minutes), Wordle (multi-turn, 2-4 H100s), Alphabet Sort (multi-turn LoRA RL with no SFT warmup, one H100), Wiki Search (multi-turn tool use), and Hendrycks Sanity (algorithm-ablation sanity check) [16]. Advanced examples (32-2048 GPUs, SLURM) include Qwen3-30B-A3B, GLM-4.5-Air, Nemotron-3-Super, MiniMax-M2.5, INTELLECT-3.1, and a high-throughput GLM-5/GLM-5.2 P/D-disaggregated deployment [16]. `scripts/mini_moe.py` builds a small (~0.5B-parameter) test model per architecture for fast iteration on modeling code, and the smoke-test SFT run is documented to drop loss from roughly 12 to roughly 2.5 as a warm-up sanity check [24].
- **Community layer**: the docs pages read for this card carry no dedicated curated-tutorials page (no `community_tutorials`-style index was found in `overview.md`, `configuration.md`, `training.md`, or `advanced.md`); the closest official pointer is the per-example READMEs under `examples/basic/*/README.md`, which the configuration page calls out as each carrying "the full launch story" [16].
- **Boundary the reader will hit**: disaggregated prefill/decode inference requires building NIXL against UCX 1.19.x from source — the docs state plainly that the pip-wheel NIXL's bundled UCX "segfaults on the prefill→decode KV transfer ... reproduced on vLLM 0.22 and 0.23, with/without mooncake, with/without llm-d", so the source build is required, not optional, and must be re-run after every `uv sync` because the lockfile pins the wheel [23]. VLM (multimodal) training is custom-implementation-only — `get_model` rejects any model without a registered custom PrimeRL VLM class — and currently only Qwen3.5 dense and Qwen3.5-MoE are registered [24][20]; VLM training also mandates bfloat16 (the trainer config validator refuses other dtypes) and freezes the vision encoder by default, which is incompatible with LoRA if the encoder is unfrozen [20]. No maintainer-confirmed trap from a closed GitHub issue is cited here: the two closed issues checked for this card (#2515, a validation-loop FSDP deadlock report with no confirmed resolution in the maintainer reply, and #1713, an orchestrator event-loop hang whose maintainer-cited fix in PR #2609 predates the v0.8.0 release by roughly two and a half months) did not meet the bar of a still-live, reader-relevant trap as of this release.

## Sources

All pages are unpinned `main`-branch docs read live on 2026-08-11 unless a release tag or commit is named. Ecosystem/dependency packages named in passing (`verifiers`, `renderers`, `research-environments`, `pydantic-config`, FSDP2, DeepEP, vLLM, uv) are reached through the sources below and are not separately enumerated.

[1] prime-rl docs overview. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/overview.md. Fetched 2026-08-11.

[2] prime-rl on PyPI-equivalent packaging metadata / project description, `pyproject.toml` at the v0.8.0 tag. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/v0.8.0/pyproject.toml. Fetched 2026-08-11.

[3] prime-rl GitHub repository (live). https://github.com/PrimeIntellect-ai/prime-rl. Fetched 2026-08-11 via the GitHub API (`repos/PrimeIntellect-ai/prime-rl`).

[4] prime-rl README at the v0.8.0 release commit `60bc29547a8824ad1de7b9af8d265e2b27b2a72d`. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/v0.8.0/README.md. Fetched 2026-08-11.

[5] prime-rl `pyproject.toml` at the v0.8.0 release commit `60bc29547a8824ad1de7b9af8d265e2b27b2a72d`. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/v0.8.0/pyproject.toml. Fetched 2026-08-11.

[6] prime-rl docs, Algorithms. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/algorithms.md. Fetched 2026-08-11.

[7] prime-rl docs overview, Quick Run section. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/overview.md. Fetched 2026-08-11.

[8] prime-rl docs, Scaling. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/scaling.md. Fetched 2026-08-11.

[9] prime-rl docs, Advanced, Expert Parallelism Backends section. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/advanced.md. Fetched 2026-08-11.

[10] GitHub Releases API for prime-rl (v0.8.0, published 2026-08-07). https://api.github.com/repos/PrimeIntellect-ai/prime-rl/releases. Fetched 2026-08-11.

[11] GitHub Git Refs API resolving the v0.8.0 tag to commit `60bc29547a8824ad1de7b9af8d265e2b27b2a72d`. https://api.github.com/repos/PrimeIntellect-ai/prime-rl/git/refs/tags/v0.8.0. Fetched 2026-08-11.

[12] GitHub Compare API, v0.8.0...bbb90a1b4132c351cbe8b0ed1fa808dde99f0318 (screening commit is 17 commits behind the v0.8.0 release commit). https://api.github.com/repos/PrimeIntellect-ai/prime-rl/compare/v0.8.0...bbb90a1b4132c351cbe8b0ed1fa808dde99f0318. Fetched 2026-08-11.

[13] `examples/basic/reverse-text/README.md`, SFT section, at the v0.8.0 release commit `60bc29547a8824ad1de7b9af8d265e2b27b2a72d`. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/v0.8.0/examples/basic/reverse-text/README.md. Fetched 2026-08-11.

[14] `examples/basic/reverse-text/sft.toml` at the v0.8.0 release commit `60bc29547a8824ad1de7b9af8d265e2b27b2a72d`. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/v0.8.0/examples/basic/reverse-text/sft.toml. Fetched 2026-08-11.

[15] `examples/basic/reverse-text/rl.toml` at the v0.8.0 release commit `60bc29547a8824ad1de7b9af8d265e2b27b2a72d`. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/v0.8.0/examples/basic/reverse-text/rl.toml. Fetched 2026-08-11.

[16] prime-rl docs, Configuration. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/configuration.md. Fetched 2026-08-11.

[17] prime-rl docs, Training (full page: Rules of Thumb, Important Metrics, W&B logging and platform-monitoring sections, checkpointing and resume). https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/training.md. Fetched 2026-08-11.

[18] prime-rl docs, Advanced, Multi-Tenant Training section. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/advanced.md. Fetched 2026-08-11.

[19] prime-rl docs, Development, Run the Smoke Test section (inference GPU-memory and max-model-len flags used in the full RL smoke test). https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/development.md. Fetched 2026-08-11.

[20] prime-rl docs, Advanced, Multimodal Training section. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/advanced.md. Fetched 2026-08-11.

[21] prime-rl docs, Advanced, LoRA Training section. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/advanced.md. Fetched 2026-08-11.

[22] prime-rl docs, Inference. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/inference.md. Fetched 2026-08-11.

[23] prime-rl docs, Advanced, Disaggregated Prefill/Decode Inference section. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/advanced.md. Fetched 2026-08-11.

[24] prime-rl docs, Development. https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/development.md. Fetched 2026-08-11.
