# TensorRT-LLM

NVIDIA's inference and serving engine for LLMs on NVIDIA GPUs - not a trainer, but the rollout/generation backend a post-training loop can plug into through a prototype Ray path.

TensorRT-LLM "provides users with an easy-to-use Python API to define Large Language Models (LLMs) and supports state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs," and also ships "components to create Python and C++ runtimes that orchestrate the inference execution" [1]. It is built and maintained by NVIDIA [1][2]. The API shape is offline batch generation (`LLM(model=...).generate(prompts, sampling_params)`) or an OpenAI-compatible HTTP server started with the `trtllm-serve` CLI [3]. It lives at https://github.com/NVIDIA/TensorRT-LLM [2].

**When to pick it**: not as a post-training trainer - it has none - but as the generation/rollout engine an externally-orchestrated RLHF loop calls into for fast sampling on NVIDIA GPUs, when that loop already drives training elsewhere (e.g. an FSDP process) and only needs a fast decoder. The one piece of RLHF-specific code in the repo, `tensorrt_llm/llmapi/rlhf_utils.py`'s `WorkerExtension` class, is injected into an `LLM()` via `ray_worker_extension_cls` under `orchestrator_type="ray"`, and its `update_weights` method "receives shared memory handles from another process (typically FSDP training)... and loads them into the TensorRT-LLM model" [4]. The Ray orchestrator that carries this is explicitly a prototype: "This project is under active development and currently in a prototype stage... there are currently no guarantees regarding functionality, stability, or reliability," and "MPI remains the default in TensorRT-LLM" [5]. Direct framework-level integration with an RL trainer - "Integration with RLHF frameworks, such as Verl and NVIDIA NeMo-RL" - is listed only on that page's Roadmap, not as shipped functionality [5]. This card does not compare TensorRT-LLM against trainer libraries such as trl or verl; it is not a substitute for either.

**Methods it ships**: none. TensorRT-LLM has no SFT, DPO, GRPO, PPO, or reward-model trainer classes; its only RLHF-adjacent surface is the `WorkerExtension` weight-update path described above, which is plumbing for an external trainer to push weights into this engine, not a training method [4].

**Scale it handles**: single GPU up to multi-node. Within one process, `LLM(model=..., tensor_parallel_size=2)` turns on tensor parallelism, with commented-out `pipeline_parallel_size`, `moe_expert_parallel_size`, and `moe_tensor_parallel_size` fields shown alongside it in the same distributed-generation example for pipeline and MoE-expert parallelism [6]. Multi-node runs go through Slurm via the `trtllm-llmapi-launch` wrapper script, launched with `srun`/`sbatch` against a container image, with the docs marking this feature "experimental and may not work on all systems" and requiring that total MPI processes (nodes times tasks-per-node) equal the target `tensor_parallel_size` [7]. MPI (via mpi4py) is the default multi-GPU/multi-node orchestrator; Ray is an alternative, prototype-stage orchestrator selected with `orchestrator_type="ray"` [5]. For serving at scale, disaggregated serving splits the context (prefill) and generation (decode) phases onto separate GPU pools to relieve the compute-characteristic mismatch between them, versus aggregated serving where both phases share one GPU [8]. None of these pages publishes a throughput or latency benchmark for the scale claims above - they document mechanism only [6][7][8].

**Install**: `pip3 install tensorrt_llm`; version 1.2.1, uploaded to PyPI 2026-04-20 [9]; Python `>=3.10,<4` [9][10]; the project's own `setup.py` declares `license="Apache License 2.0"` [10], while the GitHub API reports the repository license as `NOASSERTION` because the bundled `LICENSE` file mixes the project's Apache-2.0 grant with third-party license texts (e.g. a BSD block for causal-conv1d) [11]. At the v1.2.1 tag (commit `376f7e1bd8ed543f75014309e3fd4b237e9b0e73`, 2026-04-16, close to the PyPI upload date) [12][13], `requirements.txt` hard-pins `transformers==4.57.3`, `datasets==3.1.0` (pinned below its normal floor, with a comment citing a datasets GitHub issue), `flashinfer-python==0.6.4`, `xgrammar==0.1.32`, `llguidance==0.7.29`, and `triton==3.5.1`, and bounds `torch>=2.9.1,<=2.10.0a0` and `nvidia-nccl-cu13>=2.27.7,<=2.28.9` [14]. The pinned v1.2.1 install page adds a CUDA Toolkit 13.1 requirement, a `pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130` step, and a required `libopenmpi-dev` system package, plus a stated wheel/container mismatch: "The TensorRT LLM wheel on PyPI is built with PyTorch 2.9.1. This version may be incompatible with the NVIDIA NGC PyTorch 25.12 container, which uses a more recent PyTorch build from the main branch" [15]. The shortlist's screening commit, `0a0de1c3eddb0c93cff9d22ef3786cd68120af3f` (2026-07-31), is the repository's newest push at screening time, not this release [16]; it is cited below only for the `rlhf_utils.py` source claim.

**Maintained by**: NVIDIA [1][2]; repository pushed as recently as 2026-08-10 [11], and the PyPI record shows a 1.2.1 release on 2026-04-20 [9] following earlier non-prerelease GitHub Releases for v1.1.0 (2025-12-19) and v1.0.0 (2025-09-24) [17].

## Quick start

Offline batch generation, quoted from the pinned v1.2.1 quick-start guide [3]:

```python
from tensorrt_llm import LLM, SamplingParams

def main():
    # Model could accept HF model name, a path to local HF model,
    # or Model Optimizer's quantized checkpoints like nvidia/Llama-3.1-8B-Instruct-FP8 on HF.
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    for output in llm.generate(prompts, sampling_params):
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

if __name__ == '__main__':
    main()
```

The CLI equivalent starts an OpenAI-compatible server directly from a model id: `trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"`, then a client sends a normal `POST` to `v1/chat/completions` [3]. The same guide also shows `trtllm-serve "nvidia/Qwen3-8B-FP8"`, noting the reader should "Ensure your GPU supports FP8 quantization before running" [3].

## Start it

- One process, one GPU is the base form: either quick-start snippet above, unmodified.
- Multiple GPUs on one node: pass `tensor_parallel_size` (and, commented in the same example, `pipeline_parallel_size`, `moe_expert_parallel_size`, `moe_tensor_parallel_size`) to `LLM()` - the docs' distributed-generation example shows this running as a single Python process, with no separate launcher invoked [6].
- Multi-node goes through Slurm: the repo's `trtllm-llmapi-launch` wrapper (installed as a console script) runs an existing LLM-API script (the docs' template wraps `examples/llm-api/quickstart_advanced.py`) under `srun`, inside a container image, with the docs' own constraint that total MPI processes across nodes must equal the target `tensor_parallel_size`; the docs mark this path "experimental and may not work on all systems" [7].
- Disaggregated serving is a separate, serving-time layout choice, not a batch-size knob: `trtllm-serve disaggregated -c disagg_config.yaml` starts an orchestration server that routes client requests between separately-launched `context_servers` and `generation_servers` groups (each its own `trtllm-serve ... pytorch --config ...` process), because the context (prefill) and generation (decode) phases have different compute characteristics and benefit from being run on separate GPU pools rather than sharing GPUs in aggregated (in-flight-batched) serving [8].
- No batch-size/gradient-accumulation arithmetic applies here - there is no training loop; `--max_batch_size` (default 2048) and `--max_num_tokens` (default 8192) on `trtllm-build` bound how many requests and tokens an engine can schedule at inference time, not a training effective batch size [18].
- Launch first-aid from the LLM API introduction's Tips and Troubleshooting section (pinned v1.2.1) [19]: single-node multi-GPU does not need `mpirun`; a Slurm hang is worked around with `mpirun -n 1 --oversubscribe --allow-run-as-root ...`; constructing `LLM()` at module scope under mpi4py can trigger a recursive-spawn bug, fixed by wrapping construction in a function guarded by `if __name__ == "__main__":`; a process that "cannot quit after generation" is fixed the same way, or by using `with LLM(...) as llm:`; and a Docker container run with `--net=host` can hang unless started with `--ipc=host` or the `OMPI_MCA_btl_tcp_if_include=lo` / `OMPI_MCA_oob_tcp_if_include=lo` environment variables set.
- Out-of-memory first aid is not published as a dedicated troubleshooting list on the pages read for this card; the closest lever is disaggregated serving's separation of context and generation phases onto different GPU pools, and `trtllm-build`'s `--max_batch_size`/`--max_num_tokens`/`--max_seq_len` flags, which bound engine memory at build time [18][8].

## Watch it

The RLHF-specific weight-update path (`WorkerExtension.update_weights`) is checkpoint-safe by construction rather than logged: it "uses the control_action_decorator to ensure all active requests are finished before updating weights" [4], but neither this file nor the Ray orchestrator page documents a metric, log line, or health check for that update - what happens on the training side that drives it is out of scope for this card.

What TensorRT-LLM does publish is serving-runtime telemetry, not a training curve: `trtllm-serve` exposes a `/metrics` HTTP endpoint, gated by setting `enable_iter_perf_stats: true` in a YAML file passed via `trtllm-serve ... --config config.yaml`. The page's own example output shows the full field set: top-level `gpuMemUsage`, `iter`, `iterLatencyMS`, and `numActiveRequests`, plus a `kvCacheStats` object with `allocNewBlocks`, `allocTotalBlocks`, `cacheHitRate`, `freeNumBlocks`, `maxNumBlocks`, `missedBlocks`, `reusedBlocks`, `tokensPerBlock`, and `usedNumBlocks` [20]. The same page notes the PyTorch backend's metrics endpoint is in beta and is not as comprehensive as the TensorRT backend's, with some fields such as CPU memory usage not yet available for the PyTorch backend [20]. No RL-specific stopping rule, reward curve, or training-loss metric is published anywhere in these docs, because TensorRT-LLM runs no training loop to log one.

## Save it

- `trtllm-build` compiles a TensorRT-LLM checkpoint into a deployable engine; its `--output_dir` flag is "the directory path to save the serialized engine files and engine config file," defaulting to `engine_outputs` [18]. This is a build artifact, not a training checkpoint - it never holds optimizer state, because there is no optimizer.
- Loading weights into TensorRT-LLM (the direction this library actually performs) goes through a plugin architecture of Checkpoint Loaders, Config Loaders, Weight Loaders, and Weight Mappers that ingest HuggingFace-format checkpoints [21]; nothing on that page or elsewhere read for this card documents an inverse "export back to HuggingFace format" path.
- The RLHF weight-update mechanism (`update_weights`) takes shared-memory IPC handles from an external process "typically FSDP training" and loads them directly into the running engine's weights in memory - it does not write anything to disk, and is not a save/checkpoint operation at all [4].
- Whether a downstream evaluator can load a `trtllm-build` engine or a raw checkpoint directory is a question for that evaluator's own loader contract; this card does not assert one, since it was not read for this card.

## Find it in the docs

The docs are versioned Sphinx/GitHub Pages, live and separate from this card.

- Address pattern: `https://nvidia.github.io/TensorRT-LLM/<version>/<page>.html`, where `<version>` is `latest` or an exact release string such as `1.2.1` or `1.2.0` - verified against the version list published at `_static/switcher.json` and by direct fetch, checked 2026-08-10 [22]. The unpinned `latest` build tracked ahead of the pinned release during this check (its version switcher named itself `1.3.0rc23`, and its own install page's torch pin, `torch==2.9.0`, already differed from 1.2.1's pinned `torch==2.9.1` [15][23]), so swap in the release tag before trusting a value.
- Page-slug recipes seen while researching this card: `quick-start-guide.html`; `installation/linux.html`; `llm-api/introduction.html`; `commands/trtllm-serve/trtllm-serve.html`; `commands/trtllm-build.html`; `features/ray-orchestrator.html`; `features/checkpoint-loading.html`; `features/disagg-serving.html`; `examples/llm_mgmn_trtllm_bench.html` and the sibling multi-node/multi-GPU example pages under `examples/`.
- Question-to-slug map: install/CUDA/torch versions -> `installation/linux.html`; multi-GPU/tensor-parallel args -> `examples/llm_inference_distributed.html` and the `examples/` multi-node pages; runtime metrics -> `commands/trtllm-serve/trtllm-serve.html`'s `/metrics` section; RLHF/Ray weight updates -> `features/ray-orchestrator.html` and `tensorrt_llm/llmapi/rlhf_utils.py` in the source tree [4][5].
- The GitHub Releases API (`/repos/NVIDIA/TensorRT-LLM/releases`) does not list every published version: it surfaces mostly `-rc` prereleases and misses v1.2.0/v1.2.1 entirely, even though both exist as ordinary git tags and PyPI carries 1.2.1 as the current release - the PyPI JSON API is the authoritative source for what `pip install` actually delivers, and a plain tag lookup (`git/ref/tags/<tag>`) resolves the release commit when the Releases API is silent [9][17][24].
- No official MCP endpoint for these docs was found or fetched for this card.
- No dedicated community-tutorials or curated-blog page (comparable to trl's `community_tutorials` slug) was found on the pages read for this card; NVIDIA publishes model-specific "Deployment Guide" pages (e.g. for DeepSeek R1, Llama 3.3 70B) linked from the disaggregated-serving page's sidebar, which are first-party deployment recipes rather than third-party community posts [8].

Honest boundary: TensorRT-LLM ships no post-training method and no training loop of any kind - every method-related claim in this card is about a single prototype code path (`rlhf_utils.WorkerExtension`) for receiving weights from an external trainer, not about training performed here [4]. The Ray orchestrator that carries that path is explicitly unstable ("no guarantees regarding functionality, stability, or reliability") and direct integration with named RLHF frameworks is roadmap-only, not shipped [5]. No maintainer-confirmed GitHub issue trap was located within the pages read for this card.

## Sources

Every page below is the pinned v1.2.1 docs build unless noted, fetched 2026-08-10 except where an earlier fetch date differs; the v1.2.1 tag resolves to commit `376f7e1bd8ed543f75014309e3fd4b237e9b0e73` (2026-04-16), distinct from the shortlist's screening commit `0a0de1c3eddb0c93cff9d22ef3786cd68120af3f` (2026-07-31), which is cited only where explicitly named.

[1] TensorRT-LLM GitHub repository description, via the GitHub Repos API. https://api.github.com/repos/NVIDIA/TensorRT-LLM. Fetched 2026-08-10.

[2] TensorRT-LLM GitHub repository. https://github.com/NVIDIA/TensorRT-LLM. Fetched 2026-08-10.

[3] Quick Start Guide, pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/quick-start-guide.html. Fetched 2026-08-10.

[4] `tensorrt_llm/llmapi/rlhf_utils.py`, at commit `0a0de1c3eddb0c93cff9d22ef3786cd68120af3f`. https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/0a0de1c3eddb0c93cff9d22ef3786cd68120af3f/tensorrt_llm/llmapi/rlhf_utils.py. Fetched 2026-08-10.

[5] Ray Orchestrator (Prototype), pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/features/ray-orchestrator.html. Fetched 2026-08-10.

[6] Distributed LLM Generation example, pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/examples/llm_inference_distributed.html. Fetched 2026-08-10.

[7] Multi-node/multi-GPU Slurm example script docs, pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/examples/llm_mgmn_trtllm_bench.html. Fetched 2026-08-10.

[8] Disaggregated Serving, pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/features/disagg-serving.html. Fetched 2026-08-10.

[9] `tensorrt_llm` on PyPI (JSON API). https://pypi.org/pypi/tensorrt_llm/json. Fetched 2026-08-10.

[10] `setup.py` at the v1.2.1 tag (commit `376f7e1bd8ed543f75014309e3fd4b237e9b0e73`). https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v1.2.1/setup.py. Fetched 2026-08-10.

[11] TensorRT-LLM GitHub repository metadata (license `NOASSERTION`, `pushed_at`), via the GitHub Repos API. https://api.github.com/repos/NVIDIA/TensorRT-LLM. Fetched 2026-08-10.

[12] Git tag reference for `v1.2.1`, via the GitHub Git Refs API. https://api.github.com/repos/NVIDIA/TensorRT-LLM/git/ref/tags/v1.2.1. Fetched 2026-08-10.

[13] Commit metadata for `376f7e1bd8ed543f75014309e3fd4b237e9b0e73`, via the GitHub Commits API. https://api.github.com/repos/NVIDIA/TensorRT-LLM/commits/376f7e1bd8ed543f75014309e3fd4b237e9b0e73. Fetched 2026-08-10.

[14] `requirements.txt` at the v1.2.1 tag. https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v1.2.1/requirements.txt. Fetched 2026-08-10.

[15] Installing on Linux, pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/installation/linux.html. Fetched 2026-08-10.

[16] Commit metadata for the shortlist's screening commit `0a0de1c3eddb0c93cff9d22ef3786cd68120af3f`, via the GitHub Commits API. https://api.github.com/repos/NVIDIA/TensorRT-LLM/commits/0a0de1c3eddb0c93cff9d22ef3786cd68120af3f. Fetched 2026-08-10.

[17] TensorRT-LLM GitHub Releases, via the GitHub Releases API. https://api.github.com/repos/NVIDIA/TensorRT-LLM/releases. Fetched 2026-08-10.

[18] `trtllm-build` command reference, pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/commands/trtllm-build.html. Fetched 2026-08-10.

[19] LLM API Introduction, Tips and Troubleshooting section, pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/llm-api/introduction.html. Fetched 2026-08-10.

[20] `trtllm-serve` command reference, Metrics Endpoint section, pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/commands/trtllm-serve/trtllm-serve.html. Fetched 2026-08-10.

[21] Checkpoint Loading, pinned v1.2.1. https://nvidia.github.io/TensorRT-LLM/1.2.1/features/checkpoint-loading.html. Fetched 2026-08-10.

[22] Docs version switcher manifest. https://nvidia.github.io/TensorRT-LLM/_static/switcher.json. Fetched 2026-08-10.

[23] Installation guide, live unpinned `latest` build (`version_match` `1.3.0rc23` at fetch time), used only to demonstrate that the unpinned build already diverges from the pinned v1.2.1 install page. https://nvidia.github.io/TensorRT-LLM/latest/installation/installation-guide.html. Fetched 2026-08-10.

[24] Git tag reference for `v1.2.0`, via the GitHub Git Refs API (used only to confirm the tag exists outside the Releases API listing). https://api.github.com/repos/NVIDIA/TensorRT-LLM/git/ref/tags/v1.2.0. Fetched 2026-08-10.
