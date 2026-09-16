# sonar

An inference/serving engine, not a trainer: it runs the rollout and production-serving side of the stack, and exposes a documented weight-transfer API an external RL training loop can use to push updated policy weights into a running server.

Sonar "is an inference engine for Hugging Face-compatible language and multimodal models," providing "continuous batching, paged KV-cache management, optimized kernels, quantization, speculative decoding, and distributed serving," and it states it "is based on vLLM," adding "additional model and quantization formats, sampling methods, kernels, platforms, and deployment features" [1]. It is built by dphnAI as a continuation of the Aphrodite Engine project, whose Python package is still named `aphrodite-engine` and whose declared author is PygmalionAI [2][3]; the README states it serves production workloads for the Dolphin Inference Network and PygmalionAI [1]. The API shape is a single CLI entry point, `aphrodite serve <model>`, which starts an OpenAI-compatible HTTP server; there is no trainer class, no `.train()` call, and no loss function anywhere in the public surface read for this card [1][4]. It lives at https://github.com/dphnAI/sonar [2].

**When to pick it**: pick Sonar when you need to serve a Hugging Face-compatible model - for production traffic, or as the generation/rollout backend behind an online RL training loop run by a separate library - and you want one process per replica with continuous batching, paged KV-cache, quantization, and speculative decoding [1]. Do not pick it to run SFT, DPO, GRPO, or any other post-training method: it ships no trainer, no optimizer step, and no checkpoint-writing training loop of its own (contrast with trl or verl, covered on their own cards; not restated here). Its documented tie to post-training is narrow and one-directional: a `--weight-transfer-config` server flag whose only stated purpose is "the configurations for weight transfer during RL training" [5], letting an external process pause generation and push new weights into an already-running Sonar server.

**Methods it ships**: none. Sonar ships no SFT/DPO/GRPO/PPO-style trainer. What it ships instead is a serving-side RL support surface: the `--weight-transfer-config` CLI flag [5], plus a set of HTTP endpoints read directly from `aphrodite/entrypoints/serve/dev/rlhf/api_router.py` at commit `9434928` - `/pause`, `/resume`, `/abort_requests`, `/is_paused`, `/init_weight_transfer_engine`, `/start_weight_update`, `/start_draft_weight_update`, `/update_weights`, `/finish_weight_update`, `/update_weight_version`, `/weight_info`, and `/get_world_size` [6]. These endpoints sit under a `dev` route namespace in the source tree and are not documented anywhere on the public docs site beyond the one `--weight-transfer-config` flag description [5][6] - treat them as an internal, undocumented integration point rather than a stable public API. The same file is present unchanged (still 246 lines) at the `v0.23.0` release tag, six commits behind the commit this card reads [6][7][8].

**Scale it handles**: single GPU (`aphrodite serve MODEL`) up to multi-node, all through the `aphrodite serve` CLI itself - Sonar starts one multiprocessing executor per host and states plainly "You do not need a Ray cluster for multi-node tensor or pipeline parallelism" [9]. One node uses `--tensor-parallel-size`; multiple nodes add `--nnodes`, `--node-rank`, `--master-addr`/`--master-port`, and `--headless` on every non-zero-rank node, with total world size = tensor_parallel_size x pipeline_parallel_size x data_parallel_size and `--nnodes` required to divide it [9]. The multi-node CUDA path selects the `mp` distributed-executor backend automatically and the docs say explicitly not to select `ray` for this launch method [9]. Data-parallel serving and expert parallelism are documented features [9], but the distributed-deployment and benchmarking pages read for this card give mechanism and a `aphrodite bench` measurement tool only - no published multi-node throughput numbers [9][10].

**Install**: `pip install aphrodite-engine` (the package name stays `aphrodite-engine`, the CLI stays `aphrodite`) [3]; latest release `v0.23.0`, published 2026-07-31, resolving to commit `795396a3a5` [7][8]; licence AGPL-3.0 [2][3]. At that release commit, `requires-python` is `>=3.10,<3.15` [8], and the CUDA install docs state the published wheel targets Linux x86-64, Python 3.10-3.13, and "NVIDIA GPUs with compute capability 8.0 or newer" [4] - no CUDA/driver version floor is stated beyond that GPU-architecture requirement. Load-bearing pins from `requirements/cuda.txt` at the release commit: `torch==2.13.0` (exact, no range - the training/deep-learning core is hard-pinned, not just floored), `torchaudio==2.11.0`, `torchvision==0.28.0`, `numba==0.65.0`, `flashinfer-python==0.6.15.post1` (from a separate `--extra-index-url`) [8]. The row's screening commit, `9434928`, is the repository's newest push and sits six commits ahead of the `v0.23.0` release tag per GitHub's compare API [7]; code-level claims sourced at that commit (the RL dev-server endpoints above) are ahead of what `pip install aphrodite-engine` currently delivers, though the same endpoint file is unchanged between the two [6][8].

**Maintained by**: dphnAI, continuing the Aphrodite Engine project (`aphrodite-engine/aphrodite-engine`, authored by PygmalionAI per the package metadata) [2][3]; the `v0.22.0` and `v0.21.0` changelogs are authored by AlpinDale, and while the `v0.21.0` changelog still references pull requests against the upstream `aphrodite-engine/aphrodite-engine` repository, the two most recent releases, `v0.23.0` and `v0.22.0`, reference only `dphnAI/sonar` pull requests [11]. About 1.8k GitHub stars, not used here as a ranking signal [2]. Releases are frequent - `v0.23.0` (2026-07-31), `v0.22.0` (2026-07-19), `v0.21.0` (2026-05-02) [11] - and the repository's last push at time of reading was 2026-08-10, one week after the last tagged release [2].

## Quick start

Both forms are quoted from Sonar's own quickstart page [12]. Serve a model:

```bash
aphrodite serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3
```

The server listens on `http://127.0.0.1:2242` by default and exposes OpenAI-compatible APIs, health checks, metrics, and an OpenAPI schema [1]. Send a request with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:2242/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

Use `--api-key` "when clients outside a trusted network can reach the server" [12]; without it, the server accepts unauthenticated requests, since `--api-key` only "require[s] one of these keys to be presented" when it is set [5].

## Start it

- One GPU, one process: `aphrodite serve MODEL` as above.
- One node, several GPUs: add `--tensor-parallel-size N` for a fast local interconnect [9].
- Multiple nodes: run the same `aphrodite serve` command on every host, varying only `--node-rank`; node 0 starts the API server, every other node adds `--headless` and runs model workers only. World size is `tensor_parallel_size x pipeline_parallel_size x data_parallel_size`, and `--nnodes` must divide it evenly; every node must see the same model files, Sonar commit, command options, and environment [9]. Sonar auto-selects the `mp` distributed-executor backend for multi-node CUDA launches and its docs say not to set `ray` for this path [9]. `aphrodite run` is kept only as a compatibility alias for `aphrodite serve` [9].
- RL generation layout: `--weight-transfer-config` configures weight transfer for an external RL training loop [5]; the underlying dev-only HTTP surface (`/pause`, `/resume`, `/init_weight_transfer_engine`, `/update_weights`, `/finish_weight_update`, ...) is source-level only, not documented on the public docs site [6].
- There is no gradient-accumulation or per-device-batch arithmetic to tune, because Sonar does not train; the analogous "effective capacity" arithmetic is the world-size product above plus request concurrency, governed by `--max-num-seqs` and the scheduler flags on the server-arguments page [9][5].
- Config surface: every setting is a CLI flag to `aphrodite serve`, catalogued on the server-arguments reference [5]. Two defaults worth flagging: `--optimization-level` defaults to `2` ("`-O2` is used by default," trading some startup time for performance) [5], and `--performance-mode` defaults to `balanced` rather than favoring either latency or throughput [5] - both are silent choices a reader should reconsider for a latency- or throughput-critical deployment.
- Out-of-memory first aid, from the troubleshooting page [13]: at startup, reduce `--max-num-seqs`, set a smaller `--max-model-len` (or `--max-model-len auto`), or lower `--gpu-memory-utilization` (fraction of GPU memory for the model executor, default `0.92`, per-instance) [5][13]. During requests, check KV-cache preemptions and request length, reduce concurrency or max context, and test FP8 KV cache where supported; an unusually large multimodal input can spike encoder memory outside the normal text-only path, so set per-prompt modality limits [13].

## Watch it

Mechanics only - what any of these signals should look like for a specific post-training method is out of scope here and lives on that method's own card; Sonar itself runs no training loop to have a method-level signal in the first place.

- Health: `curl --fail http://localhost:2242/health`; data-parallel supervisor deployments also expose `/ready` and `/readyz` [14].
- Metrics: `curl http://localhost:2242/metrics` (Prometheus format). The observability page groups the exposed metrics as: running and waiting requests; time to first token; inter-token latency; end-to-end request latency; prompt and generation throughput; KV-cache occupancy and preemption; prefix-cache hits; and speculative draft/acceptance counters [14] - it gives these as named groups, not individual metric-name strings, so read `/metrics` directly for the exact names in your build.
- Logs: set `APHRODITE_LOGGING_LEVEL=DEBUG` for a bounded diagnostic run; debug output "can be large and can contain workload details," and prompt logging must follow the deployment's own privacy policy [14].
- Traces: set `--otlp-traces-endpoint` to export OpenTelemetry traces to a collector; trace sampling reduces overhead, and the docs recommend preserving full traces for errors plus a small sample of successes [14].
- Request correlation: `--enable-request-id-headers` returns an `X-Request-Id` header for cross-service correlation [14].
- Sample-level and evaluation-during-training logging do not apply - Sonar serves requests, it does not run an evaluation loop or log training samples.
- Stopping-rule / health-limit search: read to answer this section were the observability page [14], the troubleshooting page [13], and the server-arguments reference [5]. None publishes an RL- or training-specific stopping threshold, since Sonar runs no training loop; the closest published guidance is operational, not statistical - "Alert on sustained queue growth and tail latency. A short GPU utilization spike does not require an alert" [14].

## Save it

Sonar does not save trained checkpoints, because it does not train. Its documented on-disk behavior is entirely about loading: it "accepts a Hugging Face repository name or a local model directory," downloads and caches Hugging Face model files under the standard Hugging Face cache (redirectable with `HF_HOME` or `--download-dir`), and can pin a specific `--revision`/`--tokenizer-revision` for reproducibility [15]. `--load-format` picks the loader for the checkpoint format on disk, defaulting to `auto`, which "selects a compatible loader from the checkpoint" [15]. The only "save"-shaped operation documented anywhere read for this card is in-memory, not to disk: the weight-transfer endpoints (`/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`, `/finish_weight_update`, `/update_weight_version`) let an external process push a new set of weights into an already-running server for RL rollout generation [6][5]. There is no adapter-vs-merged-model distinction to report and no evaluator-loader handoff to describe, because Sonar never produces a new checkpoint directory for another tool to load - it only ever loads one.

## Find it in the docs

The docs are the live source; this section is the lookup, not a mirror.

- Address pattern: `https://sonar.dphn.ai/<section>/<page>/` - there is no version-tag segment in the URL (verified 2026-08-11: the site has no `/v<X.Y.Z>/` path form; it documents the current default branch). Sections seen in the sitemap: `getting-started/`, `deployment/`, `features/`, `guides/`, `reference/`, `serving/`, plus a standalone `troubleshooting/` page [16].
- Question-to-slug map, from the sitemap [16]: install -> `getting-started/installation/`; first run -> `getting-started/quickstart/`; upgrading -> `getting-started/upgrading/`; which models/quantization are supported -> `reference/models/` and `reference/quantization/` (the README states both are "generated from the current source tree" [1]); every CLI flag -> `reference/server-arguments/`; multi-node -> `deployment/distributed/`; tuning for latency/throughput -> `deployment/optimization/`; parallelism strategy -> `deployment/parallelism/`; production topology, replicas, prefill/decode disaggregation -> `deployment/production/`; where model files live -> `deployment/model-storage/`; auth/TLS -> `deployment/security/`; metrics/logs/traces -> `features/observability/`; benchmarking -> `deployment/benchmarking/`; OpenAI-compatible API details -> `serving/openai/`; other APIs (Anthropic, pooling, scoring, reranking, transcription, Kobold) -> `serving/other-apis/`; error messages -> `troubleshooting/` [1][16].
- Runnable references beyond the docs: the repository's own `examples/` are not enumerated in this card - none of the docs pages read pointed at a specific example script - but the quickstart page uses `Qwen/Qwen3-0.6B` as its smoke-test model [12], and the `aphrodite bench` CLI (`latency`, `throughput`, `serve`, `startup`, `perf`, `mm-processor`, `sweep` subcommands) is the documented way to get a repeatable local measurement [10].
- Community layer: no curated community-tutorials page was found in the site's sitemap, and the GitHub repository's landing page carries no Discord or blog link among the elements read for this card [16][2]. Treat the docs site and the `examples/`/`aphrodite bench --help` output as the primary references until a curated community page turns up.
- MCP: no official MCP endpoint for the Sonar docs was found in the pages read for this card.
- Traps: no maintainer reply in a closed GitHub issue was read for this card, so none is reported here - a future update should search `dphnAI/sonar` issues specifically before adding one.
- Honest boundary: Sonar will not run on native Windows ("Use WSL 2 to run Sonar on Windows. Native Windows builds are not supported" [4]), and the published CUDA wheel requires compute capability 8.0 or newer [4]; below that, or for post-training itself (SFT/DPO/GRPO/PPO/any trainer), this is the wrong library - it has none.

## Sources

All pages fetched 2026-08-11 unless a commit or release is named. `main`-branch/live pages are unpinned and can move; commit- and release-pinned claims name their commit explicitly.

[1] sonar README at commit `94349283951fd16e7003b38d8505f31276543a9d`. https://github.com/dphnAI/sonar/blob/94349283951fd16e7003b38d8505f31276543a9d/README.md

[2] dphnAI/sonar GitHub repository (API): license, stars, pushed_at, description, archived status. https://api.github.com/repos/dphnAI/sonar

[3] sonar `pyproject.toml` at commit `94349283951fd16e7003b38d8505f31276543a9d` (package name `aphrodite-engine`, author PygmalionAI, Homepage URL, CLI entry point).

[4] Sonar installation docs (unpinned live docs, no version marker on the page; read as rendered 2026-08-11). https://sonar.dphn.ai/getting-started/installation/

[5] Sonar server-arguments reference; the README ([1]) states this reference is generated from the current source tree rather than pinned to a release (unpinned live docs, no version marker on the page; read as rendered 2026-08-11). https://sonar.dphn.ai/reference/server-arguments/

[6] `aphrodite/entrypoints/serve/dev/rlhf/api_router.py` at commit `94349283951fd16e7003b38d8505f31276543a9d`. https://github.com/dphnAI/sonar/blob/94349283951fd16e7003b38d8505f31276543a9d/aphrodite/entrypoints/serve/dev/rlhf/api_router.py

[7] GitHub compare API, `v0.23.0...94349283951fd16e7003b38d8505f31276543a9d` (ahead_by 6). https://api.github.com/repos/dphnAI/sonar/compare/v0.23.0...94349283951fd16e7003b38d8505f31276543a9d

[8] sonar `pyproject.toml` and `requirements/cuda.txt` at the `v0.23.0` release commit `795396a3a527ba449018c4cd1ed9e96d017eee99`; the same `api_router.py` re-fetched at this commit to confirm it is unchanged (246 lines both times).

[9] Sonar distributed-deployment docs (unpinned live docs, no version marker on the page; read as rendered 2026-08-11). https://sonar.dphn.ai/deployment/distributed/

[10] Sonar benchmarking docs (unpinned live docs, no version marker on the page; read as rendered 2026-08-11). https://sonar.dphn.ai/deployment/benchmarking/

[11] dphnAI/sonar GitHub releases (API): tag names, publish dates, changelog bodies and PR authorship. https://api.github.com/repos/dphnAI/sonar/releases

[12] Sonar quickstart page (unpinned live docs, no version marker on the page; read as rendered 2026-08-11). https://sonar.dphn.ai/getting-started/quickstart/

[13] Sonar troubleshooting docs (unpinned live docs, no version marker on the page; read as rendered 2026-08-11). https://sonar.dphn.ai/troubleshooting/

[14] Sonar observability docs (unpinned live docs, no version marker on the page; read as rendered 2026-08-11). https://sonar.dphn.ai/features/observability/

[15] Sonar model-loading-and-storage docs (unpinned live docs, no version marker on the page; read as rendered 2026-08-11). https://sonar.dphn.ai/deployment/model-storage/

[16] Sonar docs sitemap (unpinned live page, read 2026-08-11). https://sonar.dphn.ai/sitemap-0.xml

PyPI's `aphrodite-engine` project page (https://pypi.org/pypi/aphrodite-engine/json) was fetched to cross-check the release version (0.23.0), licence, and author against [2], [3], and [8], and is folded into those citations rather than listed separately. Ecosystem tools named only in passing - vLLM (the base Sonar forks [1]), Hugging Face, Prometheus, OpenTelemetry, NCCL/Gloo - are not separately cited beyond the pages above that name them.
