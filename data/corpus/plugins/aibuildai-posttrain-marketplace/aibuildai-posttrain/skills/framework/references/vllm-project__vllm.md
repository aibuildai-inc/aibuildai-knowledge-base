# vLLM

The generation engine, not a trainer: vLLM serves the rollouts that other post-training libraries (TRL, verl, and peers) train against, and this card covers it in that role.

vLLM describes itself as "a fast and easy-to-use library for LLM inference and serving" [1]. It was originally built in the Sky Computing Lab at UC Berkeley and is now maintained by "a diverse community of many dozens of academic institutions and companies from over 2000 contributors" [1]. Its API is a Python `LLM` class for offline batched generation plus a `vllm serve` command that exposes an OpenAI-compatible HTTP server [2]; it ships no `Trainer` classes and runs no training loop of its own. It lives at https://github.com/vllm-project/vllm.

**When to pick it**: not a competitor to TRL or verl - it is the generation backend both of them, and nine other named RL libraries (Cosmos-RL, ms-swift, NeMo-RL, Open Instruct, OpenRLHF, PipelineRL, Prime-RL, SkyRL, Unsloth), call out to for rollout generation during online RL [3]. In this corpus's own trace audit it was seen training in 197 of 1198 sampled runs; the row that reports this figure attributes it to `methods_seen` pointing at `examples/rl/rlhf_async_new_apis.py`, but the row itself does not define what role vLLM played in those runs [4]. Pick it when you need fast rollout generation to plug into an RL post-training loop, or when you need standalone high-throughput LLM serving; look elsewhere if you want a library that also runs the optimizer step.

**Methods it ships**: none - no SFT, DPO, or GRPO trainer lives in this repository. What it ships instead is the RL-generation plumbing that method-owning libraries call: a pluggable weight-transfer system with NCCL (separate training/inference GPUs), IPC (colocated on one GPU), and sparse_nccl backends, driven through a four-phase protocol (`init_weight_transfer_engine`, `start_weight_update`, `update_weights`, `finish_weight_update`) [5]; a pause/resume API (`pause_generation`, `resume_generation`) for updating weights mid-flight without losing in-flight requests during asynchronous, overlapped generation-and-training loops [6]; and Sleep Mode, which offloads model weights to CPU RAM and/or discards the KV cache to free GPU memory between generation and training steps on a colocated GPU, with two levels (level 1 keeps weights in CPU RAM for waking the same model; level 2 discards everything, meant for RLHF weight updates) [7]. The repo's `examples/rl/` directory holds ten files at the commit this corpus screened it at (e3be89673db6143c1f9c8689d853b9c7c7a5eb29, four days ahead of the v0.26.0 release this card otherwise pins to); eight are `rlhf_*.py` scripts covering these combinations (NCCL, IPC, HTTP+NCCL, HTTP+IPC, FSDP+expert-parallel variants, sparse NCCL), and the remaining two (`routed_experts_e2e.py`, `skip_loading_weights_in_engine_init.py`) are not RLHF-specific [8].

**Scale it handles**: single GPU with no distributed inference needed when the model fits; single-node multi-GPU via tensor parallelism (`tensor_parallel_size`); multi-node via tensor parallelism combined with pipeline parallelism (`pipeline_parallel_size` set to the node count), launched through the `vllm serve` CLI or the `LLM`/`AsyncLLMEngine` Python classes - no separate launcher script is documented [9]. Data-parallel deployment for MoE models is documented as a distinct mechanism combining data-parallel attention with expert- or tensor-parallel MoE layers [9]; for async RL specifically, vLLM's internal Ray-based load balancer (`data_parallel_backend="ray"`) handles pause/resume across all data-parallel ranks with a single call, while an external load balancer in front of independent vLLM instances requires calling each engine individually [6]. No multi-node throughput benchmark was found on the pages read for this card - the mechanism is documented, a number is not.

**Install**: `pip install vllm`, version 0.26.0, uploaded to PyPI 2026-07-25 [10] (GitHub release tagged the same version 2026-07-27, at commit 568afb3a13806beb53bb2e6bd518269357b237c0 [11]); Python `>=3.10,<3.15` and Apache-2.0 license, both read from `pyproject.toml` at the `v0.26.0` tag [12]. The screening commit e3be89673db6143c1f9c8689d853b9c7c7a5eb29 (pushed 2026-07-31) is four days ahead of this release and is not what `pip install vllm` delivers. Load-bearing pins, read from `requirements/common.txt` and `requirements/cuda.txt` at the same tag: `torch==2.11.0` exactly (plus matching `torchaudio==2.11.0`, `torchvision==0.26.0`) for the CUDA build, and `transformers>=5.5.3` as a floor only [13][14]. The GPU install page states pre-compiled wheels bundle CUDA 12.9 binaries and require Linux, Python 3.10-3.13, and GPU compute capability 7.5 or higher (T4, RTX20xx, A100, L4, H100, B200, etc.) [15].

**Maintained by**: the vLLM project, a community effort with no single corporate owner, originally started at UC Berkeley's Sky Computing Lab [1]; the repository's most recent push at fetch time was 2026-08-07 and the latest tagged release (v0.26.0) shipped 2026-07-27 [11][16], both signs of active, ongoing development.

## Quick start

Both forms are quoted from the quickstart page, using real Hugging Face model IDs [2]:

Offline batched inference:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Online serving, an OpenAI-API-compatible server on `http://localhost:8000` by default:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

## Start it

- One GPU: the `LLM(model=...)` form above, as-is.
- Single node, multiple GPUs: set `tensor_parallel_size` to the GPU count (e.g. 4 for a 4-GPU node), either in `LLM(...)` or as `vllm serve ... --tensor-parallel-size 4` [9].
- Multi-node: combine `tensor_parallel_size` (GPUs per node) with `pipeline_parallel_size` (node count) - e.g. `tensor_parallel_size=8, pipeline_parallel_size=2` for two 8-GPU nodes [9]. For uneven GPU counts across a single node, the docs recommend pipeline parallelism (`tensor_parallel_size=1`, `pipeline_parallel_size=<GPU count>`) instead of tensor parallelism [9].
- RL-specific generation layout: pick a weight-transfer backend via `WeightTransferConfig(backend=...)` - `"nccl"` (default) for separate training and inference GPUs, `"ipc"` for training and inference colocated on the same GPU, or `"sparse_nccl"` for sparse flat-index weight patches at TP=1/PP=1 [5]. On a colocated GPU, `enable_sleep_mode=True` lets the trainer offload vLLM's weights and/or KV cache between generation and training steps to make room for the training step, and `llm.wake_up(tags=["weights"])` can restore only the weights first if only a partial reload is needed [7].
- vLLM has no gradient accumulation or effective-batch arithmetic of its own - it does not train. Its throughput-sizing number is the "Maximum concurrency" figure vLLM logs at startup, derived from GPU KV cache size divided by the model's `max_model_len`; the docs say to add GPUs or nodes if that concurrency is below your throughput requirement [9].
- Config surface: `LLM`/`AsyncLLMEngine` constructor arguments and the `EngineArgs`/`AsyncEngineArgs` dataclasses they wrap (`tensor_parallel_size`, `pipeline_parallel_size`, `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, `enable_sleep_mode`, `weight_transfer_config`, `compilation_config`, among others) [5][7][17]. There is no upstream training-arguments base class for vLLM to diverge from, since it is not a trainer; the one default worth flagging for an RL setup is that `gpu_memory_utilization` (not itself given a documented default value on the pages read for this card) trades KV-cache headroom against the training process sharing the same GPU - too high a value starves the colocated trainer of memory.
- Out-of-memory first aid, from the "Conserving Memory" page [17]: split the model across GPUs with `tensor_parallel_size`; use a statically or dynamically quantized model; lower `max_model_len` and `max_num_seqs` to shrink the KV cache and batch size; reduce or disable CUDA graph capture via `compilation_config` or `enforce_eager=True`; and, for multi-modal models, shrink `mm_processor_cache_gb` (default 4 GiB) if CPU RAM is the constraint. On the preemption side, the docs' own warning reads: "Increase `gpu_memory_utilization` or `tensor_parallel_size` to provide more KV cache memory" when requests are being preempted for lack of KV cache space [18].

## Watch it

Mechanics only - what a metric means for a specific post-training method (reward, KL, entropy, ...) is not vLLM's concern, since vLLM does not compute those signals; it lives on that method's own trainer.

- **Enable it**: vLLM's OpenAI-compatible server exposes Prometheus metrics on its `/metrics` HTTP endpoint by default whenever the server is running - no separate tracker or flag is needed to turn logging on [19].
- **Metric names** (from the live "Production Metrics" page, general metrics group) [19]: counters `vllm:corrupted_requests`, `vllm:external_prefix_cache_hits`, `vllm:external_prefix_cache_queries`, `vllm:generation_tokens`, `vllm:mm_cache_hits`, `vllm:mm_cache_queries`, `vllm:num_preemptions`, `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries`, `vllm:prompt_tokens`, `vllm:prompt_tokens_by_source`, `vllm:prompt_tokens_cached`, `vllm:request_success`; gauges `vllm:engine_sleep_state` (quoted directly since the polarity is easy to invert: "awake = 0 means engine is sleeping; awake = 1 means engine is awake; weights_offloaded = 1 means sleep level 1; discard_all = 1 means sleep level 2" [19] - directly relevant to RLHF colocated runs), `vllm:kv_cache_usage_perc`, `vllm:lora_requests_info`, `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_requests_waiting_by_reason`; and a histogram `vllm:e2e_request_latency_seconds`, among further histograms the page lists below the excerpt read here [19]. Read the live page for the full set - this card excerpts the general-metrics group, not every histogram documented.
- **Sample-level logging of generations**: not found on the metrics or optimization/tuning pages read for this card; vLLM's `/metrics` endpoint is numeric/Prometheus-shaped, and text-level generation logging (if any) was not located in this search.
- **Evaluation during training**: not applicable - vLLM does not run an evaluation loop; a caller invokes `llm.generate()` or the HTTP API on whatever prompts it wants scored, and any eval cadence is the calling trainer's concern.
- **Health limit, quoted**: vLLM does not publish a numeric threshold, but it does publish the shape of one failure mode and its fix. When KV cache space runs out, "vLLM can preempt requests to free up KV cache space for other requests," logging a warning ("Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space...") and recomputing preempted requests once space frees up; the documented remedy is "Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory" [18]. No stopping-rule or patience value applies, since there is no training loop to stop.

## Save it

vLLM does not produce trained checkpoints, so "saving" here means what a caller must know to move weights correctly between training and generation:

- vLLM loads weights from a Hugging Face model ID or local path when constructing `LLM(model=...)` - it does not write model checkpoints to disk itself as an ordinary part of generation [2].
- The weight-transfer contract, not a filesystem contract, is what carries updated weights into a running vLLM instance during RL training: `init_weight_transfer_engine` establishes the channel once, `start_weight_update` / `update_weights` (may be called more than once, e.g. for chunked transfers) / `finish_weight_update` push a specific update, over the NCCL or IPC backend chosen at start time [5]. The same operations are exposed as HTTP endpoints (`/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`, `/finish_weight_update`, plus `/pause` and `/resume` to safely quiesce in-flight requests around the sync) when running vLLM as a server [5].
- Sleep Mode's level distinction matters for what survives a sleep/wake cycle: level 1 "will offload the model weights and discard the KV cache... The model weights are backed up in CPU memory," good for waking the same model back up; level 2 "will discard both the model weights and the KV cache," which the docs say is the level meant for "RLHF weight update," i.e. loading a different or updated model afterward [7].
- No adapter-vs-merged-model distinction was found on the pages read for this card - vLLM serves whatever full model or already-merged checkpoint it is pointed at; PEFT-adapter merge/reload semantics belong to the trainer that produced the adapter, not to vLLM.
- Loader handoff: because vLLM only ever loads a standard Hugging Face model directory or repo ID, whether a given post-training run's output can be loaded here is entirely the training library's contract, not vLLM's - if that library saved a full, mergeable model, `LLM(model="<path or repo id>")` loads it directly.

**Trap**: a closed issue reports that vLLM's FlashInfer attention backend keeps a stateful `block_table_arange` tensor tagged into the Sleep Mode KV-cache memory pool; after `llm.sleep()` discards it and `llm.wake_up()` runs, the tensor is not recreated, producing silently wrong (not crashing) generation output, and the reporter states this "will cause bad rollout outputs in VERL using vllm + flashinfer" - i.e. it corrupts RLHF rollouts without an error. A vLLM collaborator (`robertgshaw2-redhat`, association COLLABORATOR) engaged on the thread the same day it was filed, 2025-12-19, questioning the allocator placement that causes it; the issue was closed 2026-07-04 [20]. If you run Sleep Mode with the FlashInfer attention backend in a colocated RL setup, verify generation quality after a wake-up rather than assuming a silent success.

## Find it in the docs

The docs are the live source; this section is the lookup recipe, not a mirror of the content.

- Address pattern: `https://docs.vllm.ai/en/<version>/<slug>/` where `<version>` is `latest` or a release tag in the form `v<X.Y.Z>` - verified 2026-08-07: both `https://docs.vllm.ai/en/latest/getting_started/installation/gpu/` and `https://docs.vllm.ai/en/v0.26.0/getting_started/installation/gpu/` return 200 (the bare `.html` form 302-redirects to the trailing-slash form) [9][15].
- Key slugs read for this card: `training/rlhf`, `training/weight_transfer`, `training/async_rl`, `features/sleep_mode`, `usage/metrics`, `serving/parallelism_scaling`, `configuration/conserving_memory`, `configuration/optimization`, `getting_started/quickstart`, `getting_started/installation/gpu` [2][3][5][6][7][9][15][17][18][19].
- Question-to-slug map: "how do I integrate vLLM into an RL training loop" -> `training/rlhf` (the page that names the eleven RL libraries and links to the weight-transfer and async-RL pages) [3]; "how do I sync weights from my trainer" -> `training/weight_transfer`; "how do I overlap generation and training" -> `training/async_rl`; "how do I free GPU memory between generation and training steps" -> `features/sleep_mode`; "what metrics does the server expose" -> `usage/metrics`; "I'm running out of memory" -> `configuration/conserving_memory`, and separately `configuration/optimization` for the preemption warning and startup-time tuning.
- Runnable references beyond the docs: the `examples/rl/` tree in the repository ships eight `rlhf_*.py` scripts (out of ten files total) pairing each weight-transfer backend and colocation choice with a runnable RLHF loop against real Qwen3 models, read at commit e3be89673db6143c1f9c8689d853b9c7c7a5eb29, four days ahead of the v0.26.0 release cited elsewhere in this card [8].
- Community layer, curated door first: the `training/rlhf` docs page itself curates two notebooks for GRPO with vLLM - "Efficient Online Training with GRPO and vLLM in TRL" (Hugging Face cookbook) and "Qwen-3 4B GRPO using Unsloth + vLLM" (an Unsloth Colab notebook) [3]. Follow the trl and Unsloth cards' own community-layer sections for further practitioner content; this card does not duplicate their curation.
- No official MCP endpoint for querying vLLM's docs was found on the pages read for this card.

**Honest boundary**: vLLM ships no post-training method of its own - it does not run SFT, DPO, GRPO, or any other trainer, and picking it alone gets you an inference engine, not a post-training pipeline; it must be paired with a trainer such as trl or verl. It documents no multi-node throughput benchmark, only the mechanism, on the pages read for this card [9].

## Sources

[1] vLLM GitHub repository README, `v0.26.0` tag. https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/README.md. Fetched 2026-08-07.

[2] vLLM Quickstart docs, `v0.26.0`. https://docs.vllm.ai/en/v0.26.0/getting_started/quickstart/. Fetched 2026-08-07.

[3] vLLM "Reinforcement Learning from Human Feedback" docs page, `v0.26.0`. https://docs.vllm.ai/en/v0.26.0/training/rlhf/. Fetched 2026-08-07; page's own last-update date shown as March 18, 2026.

[4] This corpus's own audit of public post-training run traces, as given in the shortlist row for this card (not an external source): "trained in 197 of 1198 runs," with `methods_seen` attributing this to `examples/rl/rlhf_async_new_apis.py`. The row does not state or define what role vLLM played in those runs.

[5] vLLM "Weight Transfer" docs page, `v0.26.0`. https://docs.vllm.ai/en/v0.26.0/training/weight_transfer/. Fetched 2026-08-07.

[6] vLLM "Async Reinforcement Learning" docs page, `v0.26.0`. https://docs.vllm.ai/en/v0.26.0/training/async_rl/. Fetched 2026-08-07.

[7] vLLM "Sleep Mode" docs page, `v0.26.0`. https://docs.vllm.ai/en/v0.26.0/features/sleep_mode/. Fetched 2026-08-07.

[8] vLLM GitHub repository contents API for the `examples/rl` directory, at commit `e3be89673db6143c1f9c8689d853b9c7c7a5eb29`. https://api.github.com/repos/vllm-project/vllm/contents/examples/rl?ref=e3be89673db6143c1f9c8689d853b9c7c7a5eb29. Fetched 2026-08-07.

[9] vLLM "Parallelism and Scaling" docs page, `v0.26.0`. https://docs.vllm.ai/en/v0.26.0/serving/parallelism_scaling/. Fetched 2026-08-07.

[10] vLLM PyPI JSON API. https://pypi.org/pypi/vllm/json. Fetched 2026-08-07.

[11] vLLM GitHub repository, releases and tag-ref APIs (`api.github.com/repos/vllm-project/vllm/releases/latest`, `.../git/ref/tags/v0.26.0`, `.../git/commits/568afb3a...`). https://github.com/vllm-project/vllm. Fetched 2026-08-07.

[12] `pyproject.toml` at the `v0.26.0` tag. https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/pyproject.toml. Fetched 2026-08-07.

[13] `requirements/common.txt` at the `v0.26.0` tag. https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/requirements/common.txt. Fetched 2026-08-07.

[14] `requirements/cuda.txt` at the `v0.26.0` tag. https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/requirements/cuda.txt. Fetched 2026-08-07.

[15] vLLM GPU installation docs page, `v0.26.0`. https://docs.vllm.ai/en/v0.26.0/getting_started/installation/gpu/. Fetched 2026-08-07.

[16] vLLM GitHub repository metadata API (`pushed_at`, `stargazers_count`, `archived`, `license`). https://api.github.com/repos/vllm-project/vllm. Fetched 2026-08-07.

[17] vLLM "Conserving Memory" docs page, `v0.26.0`. https://docs.vllm.ai/en/v0.26.0/configuration/conserving_memory/. Fetched 2026-08-07.

[18] vLLM "Optimization and Tuning" docs page, `v0.26.0` (preemption warning and remedy). https://docs.vllm.ai/en/v0.26.0/configuration/optimization/. Fetched 2026-08-07.

[19] vLLM "Production Metrics" docs page, `v0.26.0`. https://docs.vllm.ai/en/v0.26.0/usage/metrics/. Fetched 2026-08-07.

[20] vLLM GitHub issue #31016, "[Bug]: FlashInfer Incompatible with Sleep Mode," closed. https://github.com/vllm-project/vllm/issues/31016. Fetched 2026-08-07 via the GitHub issues and issue-comments APIs.

Ecosystem libraries named only as callers of vLLM (TRL, verl, Unsloth, and the other RL libraries listed in [3]) are covered on their own cards and not re-cited here beyond [3].
