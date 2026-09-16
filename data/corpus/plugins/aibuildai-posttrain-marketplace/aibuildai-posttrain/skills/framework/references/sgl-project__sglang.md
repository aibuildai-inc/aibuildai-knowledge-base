# sglang

A high-throughput LLM/VLM serving engine that post-training pipelines run as the rollout and reward-scoring backend, not as a trainer in its own right.

**SGLang** is "a high-performance serving framework for large language models and multimodal models" [1]. It is built and maintained by the SGLang project, hosted under the non-profit organization LMSYS [1]. Its API is an OpenAI-compatible HTTP server plus a native `/generate` endpoint: you launch `python3 -m sglang.launch_server --model-path <hub-id>` and then call it with an OpenAI SDK client or raw HTTP requests [2]. The repository's own README states it is used as "a proven rollout backend used for training many frontier models," with adoption named for AReaL, Miles, slime, Tunix, and verl [1] - it is the generation engine those post-training frameworks call out to, not a training loop of its own; the shortlist's own `trains_what` field for this repository is null, confirming no method it trains a model with. It lives at https://github.com/sgl-project/sglang [3].

**When to pick it**: pick sglang as the ROLLOUT/GENERATION backend behind an external RL framework (verl, slime, AReaL, Miles, Tunix, or a custom loop), when you need RadixAttention prefix caching, prefill-decode disaggregation, and an explicit sleep/wake + weight-refit API built for the train-generate loop [1][4]; it ships no trainer class of its own, so for SFT/DPO/GRPO/PPO training code look at trl or verl instead (cross-reference cards; not covered here). It also doubles as a reward-model SCORING server for post-training pipelines that need synchronous reward calls [5].

**Methods it ships**: none, as trainers - sglang has no SFT/DPO/GRPO/PPO trainer class; the shortlist's `methods_seen` field records exactly one hit, `REWARD`, pointing at `examples/runtime/reward_model.py` [6], and that script is a client that sends completions to an already-running sglang server's `/classify` endpoint and reads back scalar scores - it does not train a reward model, it serves one for inference-time scoring [6][5]. A reward model is launched like any other model, with `--is-embedding` (and `--trust-remote-code` for some checkpoints) added to `launch_server`; the docs' example is `python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-Math-RM-72B --is-embedding --tp-size=4` and list Llama-3.1/Skywork, Gemma-2-27B/Skywork, InternLM2, and Qwen2.5-Math/Sequence reward checkpoints as supported [5]. The RL-facing surface that actually matters to a post-training pipeline is documented separately as "SGLang for RL" - sleep/wake memory release, three weight-refit paths (disk, in-process tensor, distributed NCCL group), generation pause/continue, and deterministic inference [4] (detailed in Start it and Watch it below).

**Scale it handles**: single GPU up to multi-node, with tensor/pipeline/expert/data parallelism (TP/PP/EP/DP) as core features [1]. Multi-node is launched with `--dist-init-addr`, `--nnodes`, and `--node-rank`, shown in the docs on a 2-node TP=16 Llama-3.1-405B example and a SLURM `sbatch`/`srun` template [7]; the server-arguments reference separately warns that multi-node TP setups can deadlock and directs readers to `--disable-cuda-graph` as the workaround, so this is a documented rough edge rather than a routine path [8]. Load balancing across many sglang server replicas (including split reward-model workers) goes through "SGLang Model Gateway," a router the docs describe as deployed in GLM-4.5+ training runs [4][9]. None of these pages publish a throughput or latency benchmark for the multi-node or gateway paths; the mechanism is documented, the number is not.

**Install**: the docs' Method 1 is `pip install --upgrade pip`, `pip install uv`, then `uv pip install --prerelease=allow sglang` [10]. The shortlist's screening commit (`3c5f1157...`) is the repository's newest push, not a release, so it is the wrong commit for dependency pins; resolving the `v0.5.17` tag (released 2026-08-08 [11]) through the GitHub API gives release commit `2948168...` [12], and that commit's `pyproject.toml` is what the pins below come from [13]. Python floor is `>=3.10`; licence Apache-2.0 [13]. The core deep-learning pin is `torch==2.11.0` (torchaudio pinned to match; torchvision unpinned) [13] - the install guide's separate CUDA-12 path force-reinstalls `torch==2.13.0` from the cu129 wheel index instead, so the torch version actually running depends on which install path you took [10]. Other load-bearing `==`/bounded pins at that commit: `transformers==5.12.1`, `flashinfer_python[cu13]==0.6.15.post1`, `sglang-kernel==0.4.5`, `xgrammar==0.2.1`, `kernels>=0.14.1,<0.15`, `llguidance>=1.7.6,<2.0.0` [13]. Extras: `sglang[vllm]`-equivalent is not offered - instead `pip install sglang[all]` pulls the `diffusion`, `http2`, and `tracing` extras; `sglang[ray]` pins `ray[default]>=2.55.1`; `sglang[test]` pins `peft>=0.18.0` [13]. Between the screening commit and this release commit, two extras drifted: `helion` moved 0.2.6 to 1.4 and `sgl-deep-gemm` moved 0.1.5 to 0.1.5.post1 - a concrete case of the screening push being ahead of, and different from, the tagged release. The quickstart page states the hardware floor plainly: "NVIDIA GPU with CUDA support (sm80 and above, e.g., A10, A100, L4, L40S, H100)", Linux recommended [2]. Docker images are also published (`lmsysorg/sglang:latest` / `:latest-runtime`), with the install guide's own warning that `latest` and `dev` tags are mutable and a pinned tag such as `lmsysorg/sglang:v0.5.16` should be used instead [10].

**Maintained by**: the SGLang project under LMSYS [1]; GitHub reports about 31k stars as of 2026-08-10, a raw count not a ranking claim [14]. Release cadence is active: v0.5.17 (2026-08-08), v0.5.16 (2026-07-25), v0.5.15.post1 (2026-07-14), v0.5.15 (2026-07-10), v0.5.14 (2026-06-26) [11].

## Quick start

Smallest complete server-plus-client run, from the docs [2]:

```bash
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 30000
```

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

The launched server also exposes interactive API docs at `/docs`, `/redoc`, and `/openapi.json` [2].

## Start it

- One GPU: the launch command above, as-is [2].
- More GPUs on one node: add `--tp <n>` for tensor parallelism (`--enable-p2p-check` if P2P access needs an explicit check), or launch through `python3 -m sglang_router.launch_server --dp <n>` for data parallelism [8].
- Multiple nodes: pass the same `--dist-init-addr <addr>:<port> --nnodes <N> --node-rank <rank>` to every node, shown for a 2-node TP=16 405B model; a SLURM template resolves the head-node address with `scontrol show hostname` and polls for the port before launching `srun` [7]. The server-arguments reference separately flags that multi-node TP can deadlock and to try `--disable-cuda-graph` if it does [8].
- Generation-layout choice for RL rollouts: run the reward/policy generation server standalone (disaggregated - training and rollout on separate GPUs, weights synced by refit calls) or co-located; three refit paths exist for handing new weights to a running server: `POST /update_weights_from_disk` (reloads from a path, with `abort_all_requests`, `is_async`, `torch_empty_cache`, `keep_pause`, `recapture_cuda_graph`, `flush_cache` fields), `POST /update_weights_from_tensor` (in-process, requires training and rollout to share GPU memory, for co-located setups) and the NCCL-based `POST /init_weights_update_group` / distributed-broadcast path for disaggregated setups [4].
- Before a refit call, memory can be released and restored: `--enable-memory-saver` at launch enables `POST /release_memory_occupation` and `POST /resume_memory_occupation`, both taking a `tags` field to release/restore weights and/or KV cache selectively [4].
- Config surface is CLI flags to `launch_server`, collected in one server-arguments reference page rather than a single config object [8]; the page explicitly marks `--enable-torch-compile` / `torch.compile` support as "out of maintenance and might cause error" - a documented rough edge, not a recommended default [8].
- Effective throughput/batch knobs: `--mem-fraction-static` sets the fraction of GPU memory reserved for the static model + KV cache pool, defaulting, per the docs, to "(GPU memory - reserved memory) / GPU memory, defaulting to 0.88 if GPU memory cannot be detected" [8]; `--max-running-requests`, `--max-queued-requests`, `--max-total-tokens`, and `--chunked-prefill-size` bound concurrency and prefill chunk size [8].
- Out-of-memory first aid, from the FAQ [15]: for prefill OOM, lower `--chunked-prefill-size` to 4096 or 2048; for decode OOM, lower `--max-running-requests`; for general OOM, decrease `--mem-fraction-static` (e.g. to 0.8 or 0.7); for OOM on long-prompt input logprobs, set `logprob_start_len`.
- Pausing generation mid-flight (useful when weights are about to change): `POST /pause_generation` with mode `abort`, `retract`, or `in_place`, and `POST /continue_generation` to resume [4].

## Watch it

Mechanics only - what a given metric or signal means for a specific RL method lives on that method's card, not here.

- **Enable it**: pass `--enable-metrics` at launch to expose a Prometheus `/metrics` endpoint; request-content logging is a separate opt-in, `--log-requests` (with `--log-request-level` for verbosity) - off by default, so a server run with neither flag records nothing beyond stdout [16][17].
- **Metric names** (from the production-metrics page [16]): counters `sglang:prompt_tokens_total`, `sglang:generation_tokens_total`; gauges `sglang:token_usage`, `sglang:cache_hit_rate`, `sglang:num_running_reqs`, `sglang:num_used_tokens`, `sglang:gen_throughput`, `sglang:num_queue_reqs`, `sglang:spec_num_steps`, `sglang:spec_num_draft_tokens`; histograms `sglang:time_to_first_token_seconds`, `sglang:e2e_request_latency_seconds`, `sglang:time_per_output_token_seconds`, `sglang:func_latency_seconds`. A separate MFU-related counter trio - `sglang:estimated_flops_per_gpu_total`, `sglang:estimated_read_bytes_per_gpu_total`, `sglang:estimated_write_bytes_per_gpu_total` - is exported only when both `--enable-metrics` and `--enable-mfu-metrics` are set [16].
- **Request tracing**: `--dump-requests-folder` and `--dump-requests-threshold` write request dumps that can be replayed with `scripts/playground/replay_request_dump.py`; `--crash-dump-folder` writes a pickle plus CUDA device coredumps on crash, and the docs note explicitly that this "does not configure OS process core dumps" [17].
- **Sample-level logging of generations**: not a distinct opt-in flag on this page - `--log-requests` at the appropriate `--log-request-level` logs request/response text to the server log [17]; there is no dedicated completions-sampling config comparable to a trainer's `log_completions`, because sglang is a server, not a trainer.
- **Determinism for RL**: `--enable-deterministic-inference` makes repeated identical requests produce identical logprobs by using batch-invariant kernels, and the docs frame this explicitly for RL - "Ensures consistent logprobs across runs, reducing stochastic noise and making RL training more stable, reproducible, and debuggable" [18]. It is supported only with the FlashInfer, FlashAttention-3, and Triton attention backends, and combinability with CUDA graphs, chunked prefill, radix cache, and non-greedy sampling varies by backend per a compatibility table [18]. This is a documented mechanism, not a guarantee with no exceptions: closed issue #22819 records multiple reports (none from a maintainer-tier MEMBER/OWNER/COLLABORATOR account, so this is reported here as a community-observed caveat, not a confirmed maintainer trap) of divergence with `--enable-deterministic-inference` combined with radix cache and FA3 that disappears when radix cache is disabled - worth testing your own combination before trusting bit-exact logprobs [19].
- **Evaluation-during-training / health limits**: sglang has no eval-during-training field of its own (there is no training loop to evaluate mid-run); a search of the server-arguments reference [8] and the observability page [17] for a stopping-rule, health-limit, or threshold value beyond the OOM knobs above turns up none - this section, like the OOM first-aid above, is the load-bearing "no limit is published" finding for this library, searched 2026-08-10.

## Save it

sglang does not train, so there is no training checkpoint to save; "saving" here means getting weights INTO a running server and, separately, moving fast-changing weights from an external trainer to it.

- Standard load: `--model-path <hub-id-or-local-path>` at launch loads a full model directory the normal way (safetensors + config, as produced by any Hugging Face-format save) [2].
- Fast distributed weight loading for large models uses the external `checkpoint-engine` package (`pip install 'checkpoint-engine[p2p]'`), with a `--wait-for-initial-weights` server flag and Broadcast/P2P/All transfer modes - this is a loading-speed integration, not a checkpoint format [20].
- Live weight refit, for handing an RL trainer's updated weights to an already-running rollout server without restarting it: `POST /update_weights_from_disk` (reload from a path), `POST /update_weights_from_tensor` (in-process tensors, serialized with `MultiprocessingSerializer.serialize(...)`, for co-located training+rollout - CPU offload of those tensors breaks the update path), or the NCCL-based `/init_weights_update_group` + broadcast path for disaggregated training and rollout, where training workers gather weights on rank 0 and broadcast to the rollout group [4].
- No adapter/full-model save distinction applies here - sglang serves whatever full model or LoRA-merged model it is pointed at (`--enable-lora`, default off, for serving multiple LoRA adapters at inference time) [21]; it is not where a trained adapter gets written.
- Loader handoff: because a running sglang server exposes the same OpenAI-compatible and native `/generate` HTTP surface used in the quick start, any evaluator that already talks HTTP to sglang can load results directly - there is no separate "sglang checkpoint" format for an evaluator to parse.

## Find it in the docs

The docs are unversioned - fetched 2026-08-10, `https://docs.sglang.io/docs/get-started/install.md` returns the current page directly, while a per-release form (`.../docs/v0.5.16/get-started/install.md`) 404s, so there is no way to pin a docs snapshot to an install version other than checking the page's own last-updated date against your installed version.

- Address pattern: `https://docs.sglang.io/docs/<section>/<slug>` renders in a browser; appending `.md` (`https://docs.sglang.io/docs/<section>/<slug>.md`) returns the raw markdown - useful for curl/grep workflows [2][4]. The full page index is published at `https://docs.sglang.io/llms.txt` [22].
- Key slugs: install -> `get-started/install`; first server run -> `get-started/quickstart`; sending requests -> `basic_usage/send_request`; RL/rollout integration -> `advanced_features/sglang_for_rl`; reward-model serving -> `supported-models/reward_models` [5]; determinism -> `advanced_features/deterministic_inference`; router/gateway -> `advanced_features/sgl_model_gateway`; every CLI flag -> `references/server_arguments`; OOM and other troubleshooting -> the FAQ page [15]; production observability -> `observability/production_metrics` and the general observability page at `advanced_features/observability` [16][17]; multi-node walkthroughs -> `references/multi_node_deployment` [7].
- Runnable references beyond the docs: the repository's `examples/` tree, including `examples/runtime/reward_model.py`, the script the shortlist's method-detection matched [6].
- Community layer: the README curates a "Blog" link (`lmsys.org/blog`) with release-announcement posts (v0.2 through the GB300 long-context post) but no separate curated tutorials page comparable to trl's `community_tutorials`; no such page was found in the `llms.txt` index [1][22].
- Official MCP endpoint: `https://docs.sglang.io/mcp` responds to a JSON-RPC `initialize` call identifying itself as "SGLang Documentation" and describing itself as providing "search and retrieval tools for the SGLang Documentation site," read-only except for a `submit_feedback` tool for reporting doc problems [23]. This is distinct from the "MCP Integration" section on the Model Gateway page [9], which is the gateway's own tool-calling passthrough for served models, not a docs-search server.
- Documented boundary: multi-node tensor parallelism is flagged by the maintainers' own reference page as a deadlock risk requiring `--disable-cuda-graph` as a workaround, and `torch.compile` support is marked "out of maintenance" - both stated plainly on the server-arguments page rather than discovered by trial and error [8].

## Sources

[1] sglang README, `README.md` at commit `3c5f1157`. https://github.com/sgl-project/sglang/blob/main/README.md. Fetched 2026-08-10.

[2] sglang Quickstart. https://docs.sglang.io/docs/get-started/quickstart.md. Fetched 2026-08-10.

[3] sglang GitHub repository (home page). https://github.com/sgl-project/sglang. Fetched 2026-08-10.

[4] sglang for RL. https://docs.sglang.io/docs/advanced_features/sglang_for_rl.md. Fetched 2026-08-10.

[5] sglang reward-models guide. https://docs.sglang.io/docs/supported-models/reward_models.md. Fetched 2026-08-10.

[6] `examples/runtime/reward_model.py` at commit `3c5f1157`. https://github.com/sgl-project/sglang/blob/3c5f1157/examples/runtime/reward_model.py. Fetched 2026-08-10.

[7] sglang multi-node deployment reference. https://docs.sglang.io/docs/references/multi_node_deployment.md. Fetched 2026-08-10.

[8] sglang server arguments reference (common launch commands, memory/scheduling flags, multi-node deadlock note, torch.compile maintenance note, LoRA flag). https://docs.sglang.io/docs/references/server_arguments.md. Fetched 2026-08-10.

[9] sglang Model Gateway page. https://docs.sglang.io/docs/advanced_features/sgl_model_gateway.md. Fetched 2026-08-10.

[10] sglang install guide (pip/uv methods, CUDA-12 torch reinstall, Docker tag mutability warning). https://docs.sglang.io/docs/get-started/install.md. Fetched 2026-08-10.

[11] sglang GitHub releases list. https://api.github.com/repos/sgl-project/sglang/releases. Fetched 2026-08-10.

[12] sglang `v0.5.17` tag resolved to its commit via the GitHub Git Data API: https://api.github.com/repos/sgl-project/sglang/git/refs/tags/v0.5.17 (tag object `b6a09f38...`), dereferenced at https://api.github.com/repos/sgl-project/sglang/git/tags/b6a09f38fcc5e96574324b4acc19d421c539cfc6 (commit `2948168...`). Fetched 2026-08-10.

[13] `pyproject.toml` at the `v0.5.17` release commit `2948168`. https://raw.githubusercontent.com/sgl-project/sglang/2948168/pyproject.toml. Fetched 2026-08-10.

[14] sglang GitHub repository metadata (star count, not used as a ranking). https://api.github.com/repos/sgl-project/sglang. Fetched 2026-08-10.

[15] sglang FAQ / troubleshooting page (OOM first aid). https://docs.sglang.io/docs/references/faq.md. Fetched 2026-08-10.

[16] sglang production metrics page (Prometheus metric names). https://docs.sglang.io/docs/observability/production_metrics.md. Fetched 2026-08-10.

[17] sglang observability page (logging, request dump/replay, crash dump). https://docs.sglang.io/docs/advanced_features/observability.md. Fetched 2026-08-10.

[18] sglang deterministic inference page. https://docs.sglang.io/docs/advanced_features/deterministic_inference.md. Fetched 2026-08-10.

[19] sglang GitHub issue #22819, "[Bug] KV cache corruption at radix cache block boundary with `--enable-deterministic-inference` (prefix_len == block_size)" (closed; commenters CONTRIBUTOR/NONE association, not maintainer-confirmed). https://github.com/sgl-project/sglang/issues/22819. Fetched 2026-08-10.

[20] sglang checkpoint-engine integration page. https://docs.sglang.io/docs/advanced_features/checkpoint_engine.md. Fetched 2026-08-10.

[21] sglang server arguments reference, LoRA section. https://docs.sglang.io/docs/references/server_arguments.md. Fetched 2026-08-10.

[22] sglang documentation index. https://docs.sglang.io/llms.txt. Fetched 2026-08-10.

[23] sglang Documentation MCP endpoint (JSON-RPC `initialize` handshake, self-described as "search and retrieval tools for the SGLang Documentation site"). https://docs.sglang.io/mcp. Queried 2026-08-10.
