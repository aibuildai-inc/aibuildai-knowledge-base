# agent-lightning

Microsoft's agent-training harness: wrap an existing agent with a decorator, keep its code unchanged, and hand its rollouts to a pluggable training algorithm (APO out of the box, or RL through a VERL bridge).

**Agent-lightning** aims "to provide a structured way to train your agents", built on the idea that "Your agent continues to run as usual" while a central store records its execution and an algorithm learns from it [1][2]. It is built and maintained by Microsoft [3][4]. The API centers on a `@agl.rollout`-decorated function (or a `LitAgent` subclass) that any agent framework - LangChain, the OpenAI Agent SDK, AutoGen, CrewAI, Microsoft Agent Framework, or plain OpenAI-Python calls - can wrap, plus an `agl.Trainer` that takes an `algorithm`, an agent, and train/val datasets and runs `.fit()` [2][5]. It lives at https://github.com/microsoft/agent-lightning [4].

**When to pick it**: you have an agent already written in some framework and want to improve it - via prompt optimization or RL - without rewriting its control flow; the library's own selling point is "ZERO CODE CHANGE (almost)" and framework-agnostic wrapping [2]. Its built-in RL path is not a training algorithm of its own: the `VERL` class is a subclass of `agentlightning.Algorithm` that hands the decomposed rollouts to the separate `verl` PyPI package's PPO/GRPO runner, with overrides merged with verl's packaged defaults via Hydra [7]. If you are choosing a training core rather than an agent-wrapping layer, weigh the `verl` or `trl` cards instead (cross-reference; not covered here).

**Methods it ships**: the docs' own "Algorithm Zoo" table lists exactly two built-in `Algorithm` classes - APO ("Automatic Prompt Optimization (APO) algorithm using textual gradients and beam search") and VERL ("Reinforcement Learning with VERL framework") [6]. `VERL` optimizes a resource named `main_llm`, forwards a Hydra-style `config` dict of overrides "merged with VERL's packaged defaults" into the third-party verl package's `RayPPOTrainer`, and supports GRPO through `config["algorithm"]["adv_estimator"] = "grpo"` the same way the standalone verl framework does [6][7]; the shortlist's GRPO example, `contrib/recipes/envs/config_verl/alfworld/grpo.yaml`, is exactly this override style [7]. APO optimizes a `PromptTemplate` resource; its docs give a shortcut `agl.APO(async_openai_client, ...)` and state it is "currently scoped to optimize a single prompt template" [24]. SFT is not in the Algorithm Zoo: the shortlisted `examples/unsloth/sft_algorithm.py` is a standalone example script (`async def sft_algorithm(*, store)`), not a subclass of `agentlightning.Algorithm`, built around Unsloth and vLLM rather than a library-shipped SFT trainer [9]. Rewards are attached to spans through `agentlightning.emit_reward`; the older `@reward` decorator in `agentlightning/emitter/reward.py` is documented in its own docstring as "Deprecated: ... Use `emit_reward` instead" [10]. This taxonomy is a single doc page and can move; recheck the live "Algorithm Zoo" index [6].

**Scale it handles**: any of the built-in algorithms scale as far as their backend does - APO's parallelism is `n_runners` agent workers on one machine [8]; VERL's actual GPU/node layout (`trainer.n_gpus_per_node`, `trainer.nnodes`) is verl-framework config forwarded through the `config` dict, and the wrapper's own `AgentLightningTrainer` subclasses verl's `RayPPOTrainer`, so multi-GPU/multi-node scaling is Ray's, not something agent-lightning adds [7]. What agent-lightning itself scales is the coordination layer: the default `InMemoryLightningStore` is single-process Python data structures ("fast startup with zero external dependencies—ideal for local development, CI, and unit tests"), while `MongoLightningStore` is "persistent storage suitable for production deployments and multi-process safe via database-level atomicity", with an explicit `partition_id` for running several trainers against one database [11]. The v0.3.0 changelog publishes a store throughput benchmark across runner counts: at 1000 concurrent runners the in-memory store rose from 3.36 to 14.60 rollouts/sec between v0.2.2 and v0.3.0, and the Mongo-backed store reached 50.05 rollouts/sec at the same scale - the release's own claimed "up to a 15x increase in throughput" [12].

**Install**: `pip install --upgrade agentlightning`; PyPI shows the latest published release as 0.3.0, uploaded 2025-12-24, with `requires_python = ">=3.10"` and no license classifier set in that metadata (the repository ships an MIT license file) [13][14][3]. The v0.3.0 tag's own `pyproject.toml` pins no deep-learning core in the base package - `torch` is not a base dependency - and the base install pulls `agentops>=0.4.13`, `opentelemetry-api/-sdk/-exporter-otlp>=1.35`, `litellm[proxy]>=1.74`, and `pydantic>=2.11` [15]. The `[verl]` extra adds `verl>=0.5.0` and `vllm>=0.8.4`, but the docs recommend against installing it automatically, instead giving a manual sequence pinned to `torch==2.8.0`+`torchvision==0.23.0` (CUDA 12.8 wheels), `vllm==0.10.2`, and `verl==0.5.0` "to avoid version conflicts" [16][15]. The `[apo]` extra adds `poml`, and its docs warn it "also depends on the OpenAI Python SDK, version 2.0 or newer" [16]. The repo head commit read for this card (f0a77cfa, 2026-07-16) carries `version = "0.3.1"` in `pyproject.toml`, i.e. unreleased development past the 0.3.0 tag - the dependency floors quoted above are read at the v0.3.0 tag itself (commit 3b5d733861cf313fc09821a23240bbdf3cb2ee5b), not the head commit [17][15]. The install docs state Linux only (Ubuntu 22.04+ recommended; macOS/Windows outside WSL2 unsupported) and that a GPU is optional - "you only need CUDA-capable hardware if you plan to fine-tune model weights" [16].

**Maintained by**: Microsoft [3][4]; the repository's maintainer guide documents a formal bump-first release and tagging workflow (version bump, stable branch, tagged release publishes to PyPI and docs) [18], and the v0.3.0 changelog (2025-12-24) lists concrete recent work: a MongoDB-backed store, a Tinker RL backend, an Azure OpenAI backend, and a preview web dashboard [12].

## Quick start

The docs' first full working example trains a prompt with APO on a tool-calling "room selector" agent, quoted from the tutorial [8]:

```python
import agentlightning as agl
from openai import AsyncOpenAI

@agl.rollout
def room_selector(task: RoomSelectionTask, prompt_template: agl.PromptTemplate) -> float:
    # ... agent logic using the prompt_template ...
    reward = room_selection_grader(client, final_message, task["expected_choice"])
    return reward

openai = AsyncOpenAI()
algo = agl.APO(openai)

trainer = agl.Trainer(
    algorithm=algo,
    n_runners=8,                 # Run 8 agents in parallel to try out the prompts
    initial_resources={"prompt_template": prompt_template_baseline()},
    adapter=agl.TraceToMessages(),
)
dataset_train, dataset_val = ...
trainer.fit(agent=room_selector, train_dataset=dataset_train, val_dataset=dataset_val)
```

`TraceToMessages` "requires openai >= 1.100.0" [8]. Running this with `val_batch_size=10, gradient_batch_size=4, beam_width=2, branch_factor=2, beam_rounds=2` on a 29-sample validation set, the docs report validation accuracy rising from 0.569 (baseline) to 0.721 after two beam rounds, in about 10 minutes with 8 runners [8]. The VERL path is the same `Trainer.fit`, but with a `VERL` algorithm instance built from a Hydra-override `config` dict instead of `APO` [7].

## Start it

- One process, one machine is the base form for APO: the script above, run directly; `n_runners` controls how many agent rollouts run concurrently in-process [8].
- For VERL, GPU/node layout is verl-framework config passed through `VERL(config={...})`: `actor_rollout_ref.rollout.tensor_model_parallel_size` for tensor parallelism, `trainer.n_gpus_per_node` and `trainer.nnodes` for the cluster shape, all merged with verl's own Hydra defaults [7]. The shortlisted recipe `contrib/recipes/envs/config_verl/alfworld/grpo.yaml` is a concrete 2-GPU, 1-node GRPO template using this same schema (`variables.NUM_GPUS: 2`, `trainer.nnodes: 1`, `trainer.save_freq: 100`) [19].
- Effective batch arithmetic for the VERL path follows verl's own fields inside the same config: `data.train_batch_size` (global) versus `actor.ppo_mini_batch_size` and `actor.ppo_micro_batch_size_per_gpu` per step; the ALFWorld recipe sets `train_batch_size: 32`, `ppo_mini_batch_size: 32` (`MINI_BATCH_SIZE`), `ppo_micro_batch_size_per_gpu: 16` (`PER_GPU_BATCH_SIZE`) [19].
- Config surface: `Trainer` itself is a small shell (`algorithm`, `n_runners`, `strategy`, `store`, `adapter`, `initial_resources`) [5]; the VERL algorithm's real config surface is verl's, reached only through the `config` dict - this card does not restate verl's own defaults (see the `verl` card for that).
- Generation layout: VERL launches the rollout engine itself (`actor_rollout_ref.rollout.name: "vllm"`, with `gpu_memory_utilization` controlling how much of each GPU vLLM's colocated engine may claim - the ALFWorld recipe sets it to 0.6) [19][7].
- Trajectory-level aggregation is an experimental VERL-wrapper feature that merges a whole multi-turn rollout into one masked training sample instead of one sample per turn, enabled via `config["agentlightning"]["trace_aggregator"] = {"level": "trajectory", "trajectory_max_prompt_length": ..., "trajectory_max_response_length": ...}`, with a `debug=True` + `mismatch_log_dir` option to inspect retokenization mismatches [7].
- Out-of-memory first aid is not published by agent-lightning's own docs as a distinct section; the pages read for this card (VERL algorithm reference [7], the installation guide [16], and the serving-LLM deep dive [20]) give no agent-lightning-specific OOM checklist - the applicable knobs (rollout `gpu_memory_utilization`, FSDP `param_offload`/`optimizer_offload`, micro-batch sizes) are verl's own, visible only inside the forwarded `config` dict [7][19].

## Watch it

This section is mechanics only; what a metric means for a specific method lives on that method's card.

- Tracing is on by default through the `AgentOpsTracer` (OpenTelemetry-based); every LLM call, tool call, and `emit_reward` call becomes a structured span recorded in the `LightningStore`, ordered by "a monotonic sequence id per (rollout_id, attempt_id)" [11].
- No agent-lightning-specific scalar-metric-name list was found on the pages read for this card (the store internals page [11], the VERL algorithm reference [7], and the changelog [12]); VERL-path scalar logging instead follows verl's own `trainer.logger` field - the ALFWorld recipe sets `trainer.logger: [console, wandb]` - so metric names for the RL run are verl's, documented on the `verl` card, not agent-lightning's [19].
- Sample-level visibility: the store records full request/response objects per span, including token IDs, not just chat text - "spans in Agent-lightning always store both chat messages and token IDs (actually the full request and response objects)" [20]. The v0.3.0 changelog lists a preview web Dashboard "for inspecting and debugging Agent-lightning experiments" as the human-facing view of this store data, alongside a preview OTel semantic-convention spec "specifically designed for Agent-optimization areas" [12].
- Rollout/attempt health is a state machine, not a metric: valid `RolloutStatus` values are `queuing, preparing, running, succeeded, failed, requeuing, cancelled`, and a watchdog moves an attempt to `timeout` when `now - start_time > timeout_seconds` or to `unresponsive` when `now - last_heartbeat > unresponsive_seconds` [11]. Both thresholds live on `RolloutConfig`, and the type reference states each defaults to `None`, meaning "no timeout" / "no unresponsive timeout" by default [25]; the store guide's own worked example sets them explicitly - `RolloutConfig(timeout_seconds=600, unresponsive_seconds=120, max_attempts=3, retry_condition=["failed", "timeout"])` - and attaches that config when calling `store.enqueue_rollout(input, config=cfg)` [11].
- Evaluation during training: the `Trainer.fit(train_dataset=..., val_dataset=...)` signature accepts a validation set directly [8]; on the VERL path, cadence is verl's own `trainer.test_freq` field inside the forwarded config (the ALFWorld recipe sets `test_freq: 5` against `total_epochs: 200`) [19].
- Stopping-rule honesty: no agent-lightning-published early-stopping threshold or patience value was found in the pages read for this card (store [11], VERL reference [7], APO tutorial [8], serving-LLM guide [20]); APO's own stopping shape is the `beam_rounds` count set by the caller (2 in the quickstart example), not a monitored threshold [8].

## Save it

- What persists depends entirely on which `LightningStore` backend the `Trainer` uses, and the default is not persistent: `Trainer._make_store` defaults to `InMemoryLightningStore` whenever no store is passed in, and that store keeps rollouts, spans, and resources in "Python data structures" with "zero external dependencies" - i.e., everything is lost when the process exits [21][11]. Durable storage requires opting in to `MongoLightningStore(mongo_uri=..., database_name=..., partition_id=...)`, described as "persistent storage suitable for production deployments and multi-process safe via database-level atomicity"; it needs the `mongo` extra (`pip install agentlightning[mongo]`) [11].
- What actually gets learned is method-specific and is not a agent-lightning-owned checkpoint file in either built-in algorithm: APO's output is a resource - an updated `PromptTemplate` string stored as a versioned resource in the `LightningStore` - not a model checkpoint at all; no save/export API for resources was found on the pages read for this card (the store page [11] and the tutorial [8]) beyond their being versioned inside whichever store backend is active. VERL's output is model weights, but the save/reload/resume contract for those weights is verl's own trainer's (`trainer.save_freq` in the config dict, e.g. `save_freq: 100` and `save_freq: 64` in the two recipes read for this card), not documented as an agent-lightning API on the pages read [19][7] - see the `verl` card for what a verl checkpoint directory contains and how to resume it.
- Loader handoff: because neither built-in algorithm's saved artifact is produced or documented by agent-lightning itself (a prompt-template resource for APO, a verl checkpoint for VERL), whether an evaluator can load the result is that artifact's own contract, not agent-lightning's - for the VERL path, follow this skill's shared `references/loading-the-result.md` using the verl card's checkpoint-directory description.

## Find it in the docs

The docs are the live source; this card teaches the lookup, not the content.

- Address pattern: `https://microsoft.github.io/agent-lightning/<version>/<page>/`, built with mkdocs-material and versioned with `mike`; `<version>` is `latest` (development docs, built from `main`) or a release tag such as `0.3.0`, with `stable` an alias for the newest release - `versions.json` (fetched at the docs root) lists `latest`, `0.3.0` (aliased `stable`), `0.2.2`, `0.2.1`, `0.2.0`, `0.1.2` [22][6]. Every page read for this card carries a visible banner reading "You are viewing development documentation" when on `latest`, confirming `latest` and `stable` diverge in content, not just version number [6][8][11].
- Page-slug recipes, read from each page's own `<link rel="canonical">` tag rather than guessed from the nav or the README (the README's own installation link points at a stale `stable/tutorials/installation/` path that still resolves, but the canonical tag confirms the current slug) [2]: installation is `tutorials/installation/`; the Algorithm Zoo pages are `algorithm-zoo/`, `algorithm-zoo/apo/`, `algorithm-zoo/verl/`; the quickstart tutorial is `how-to/train-first-agent/`; writing a custom algorithm is `how-to/write-first-algorithm/`; the store deep dive is `deep-dive/store/`; LLM serving is `deep-dive/serving-llm/`; the maintainer guide is `community/maintainers/` [6][24][7][8][23][11][20][18][16].
- Question-to-slug map: "which algorithms ship?" -> Algorithm Zoo Overview [6]; "how do I wrap my agent?" -> Train the First Agent [8]; "how do I write a custom algorithm?" -> Write the First Algorithm [23]; "what does the store guarantee under concurrency?" -> Understanding Store [11]; "how do I get token IDs / avoid retokenization drift?" -> Serving LLMs [20]; "what changed recently?" -> Changelog [12].
- Runnable references beyond the docs: the repo's `examples/` tree (README confirms `./examples` as the companion to the docs site) [2], including the shortlisted `examples/unsloth/sft_algorithm.py` SFT recipe [9] and a `contrib/recipes/` tree of environment-specific VERL configs such as the ALFWorld GRPO example read for this card [19]. The README's own "Community Projects" section names `Youtu-Agent`, which reports RL training verified up to 128 GPUs on a modified branch of this repository, and `AgentFlow`, a multi-agent framework built on top of it [2].
- Community layer: the README curates its own "Articles" list with dates and venues rather than a separate tutorials page - among them a 2025-10-22 vLLM blog post ("No More Retokenization Drift: Returning Token IDs via the OpenAI Compatible API Matters in Agent RL"), a 2025-08-05 arXiv paper introducing the project (arXiv:2508.03680), and dated Medium posts by a named core contributor (Yuge Zhang) on SQL-agent RL and Tinker integration [2]. No separate official MCP endpoint for querying these docs was found on the pages read for this card.
- Honest boundary: agent-lightning does not implement its own RL optimizer - the VERL path is a wrapper whose docs state plainly, "Advanced customisation currently requires copying the VERL source and modifying it directly. Native hooks for overriding training behaviour will land in a future release" [7]; and "At present, VERL does not expose fine-grained control over its reward propagation or credit assignment mechanisms. Users requiring customized reward shaping or trajectory decomposition are advised to clone and modify the VERL source implementation directly" [7].

## Sources

Method names (APO, GRPO, PPO, SFT) are deliberately cited to nothing here; their defining papers live on their own methodology cards. Fields inside the VERL wrapper's `config` dict that mirror verl-framework options (`train_batch_size`, `ppo_mini_batch_size`, `n_gpus_per_node`, etc.) are read from agent-lightning's own docs/recipe files, not from the separate `verl` PyPI package's documentation, which is out of scope for this card. All docs pages are `latest` (development) builds unless a version is named, fetched 2026-08-10.

[1] agent-lightning docs, Train the First Agent with Agent-lightning (opening description of the framework's goal). https://microsoft.github.io/agent-lightning/latest/how-to/train-first-agent/. Fetched 2026-08-10.

[2] agent-lightning GitHub README (raw). https://raw.githubusercontent.com/microsoft/agent-lightning/f0a77cfad71e6222a3edb7dfc7a0f611bd231364/README.md. Fetched 2026-08-10.

[3] agent-lightning on PyPI. https://pypi.org/pypi/agentlightning/json. Fetched 2026-08-10.

[4] agent-lightning GitHub repository (API metadata: full_name, html_url, license, archived, pushed_at, stargazers_count). https://api.github.com/repos/microsoft/agent-lightning. Fetched 2026-08-10.

[5] agent-lightning source, `agentlightning/trainer/trainer.py` at commit f0a77cfad71e6222a3edb7dfc7a0f611bd231364 (Trainer class fields: n_runners, strategy, store, adapter). https://raw.githubusercontent.com/microsoft/agent-lightning/f0a77cfad71e6222a3edb7dfc7a0f611bd231364/agentlightning/trainer/trainer.py. Fetched 2026-08-10.

[6] agent-lightning docs, Algorithm Zoo Overview. https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/. Fetched 2026-08-10.

[7] agent-lightning docs, VERL (Algorithm Zoo). https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/verl/. Fetched 2026-08-10.

[8] agent-lightning docs, Train the First Agent with Agent-lightning (quickstart code, hyperparameters, and reported validation-accuracy results). https://microsoft.github.io/agent-lightning/latest/how-to/train-first-agent/. Fetched 2026-08-10.

[9] agent-lightning source, `examples/unsloth/sft_algorithm.py` at commit f0a77cfad71e6222a3edb7dfc7a0f611bd231364 (function signatures showing `sft_algorithm` is a standalone script, not an `Algorithm` subclass). https://raw.githubusercontent.com/microsoft/agent-lightning/f0a77cfad71e6222a3edb7dfc7a0f611bd231364/examples/unsloth/sft_algorithm.py. Fetched 2026-08-10.

[10] agent-lightning source, `agentlightning/emitter/reward.py` at commit f0a77cfad71e6222a3edb7dfc7a0f611bd231364 (the `reward` decorator's deprecation docstring in favor of `emit_reward`). https://raw.githubusercontent.com/microsoft/agent-lightning/f0a77cfad71e6222a3edb7dfc7a0f611bd231364/agentlightning/emitter/reward.py. Fetched 2026-08-10.

[11] agent-lightning docs, Understanding Store (store contents, status transitions, InMemoryLightningStore and MongoLightningStore descriptions). https://microsoft.github.io/agent-lightning/latest/deep-dive/store/. Fetched 2026-08-10.

[12] agent-lightning docs, Changelog (v0.3.0 highlights and store throughput benchmark table). https://microsoft.github.io/agent-lightning/latest/changelog/. Fetched 2026-08-10.

[13] agent-lightning on PyPI, release metadata (version 0.3.0, upload_time, requires_python). https://pypi.org/pypi/agentlightning/json. Fetched 2026-08-10.

[14] agent-lightning GitHub repository LICENSE file (MIT). https://raw.githubusercontent.com/microsoft/agent-lightning/f0a77cfad71e6222a3edb7dfc7a0f611bd231364/LICENSE. Fetched 2026-08-10.

[15] agent-lightning `pyproject.toml` at the v0.3.0 tag, commit 3b5d733861cf313fc09821a23240bbdf3cb2ee5b (base dependencies, `[verl]`/`[apo]`/`[mongo]` extras). https://raw.githubusercontent.com/microsoft/agent-lightning/v0.3.0/pyproject.toml. Fetched 2026-08-10.

[16] agent-lightning docs, Installation Guide (platform/hardware requirements, `[apo]`/`[verl]` extras, manual VERL install sequence). https://microsoft.github.io/agent-lightning/latest/tutorials/installation/. Fetched 2026-08-10.

[17] agent-lightning source, `pyproject.toml` at commit f0a77cfad71e6222a3edb7dfc7a0f611bd231364 (repo-head version string "0.3.1", ahead of the 0.3.0 PyPI release). https://raw.githubusercontent.com/microsoft/agent-lightning/f0a77cfad71e6222a3edb7dfc7a0f611bd231364/pyproject.toml. Fetched 2026-08-10.

[18] agent-lightning docs, Maintainer Guide (bump-first release workflow, tagging publishes to PyPI and docs). https://microsoft.github.io/agent-lightning/latest/community/maintainers/. Fetched 2026-08-10.

[19] agent-lightning source, `contrib/recipes/envs/config_verl/alfworld/grpo.yaml` at commit f0a77cfad71e6222a3edb7dfc7a0f611bd231364 (GRPO config: GPU count, batch sizes, save/test frequency, logger backend). https://raw.githubusercontent.com/microsoft/agent-lightning/f0a77cfad71e6222a3edb7dfc7a0f611bd231364/contrib/recipes/envs/config_verl/alfworld/grpo.yaml. Fetched 2026-08-10.

[20] agent-lightning docs, Serving LLMs under Agent-lightning (token-ID handling, span content, `agl vllm` CLI mention). https://microsoft.github.io/agent-lightning/latest/deep-dive/serving-llm/. Fetched 2026-08-10.

[21] agent-lightning source, `agentlightning/trainer/trainer.py` at commit f0a77cfad71e6222a3edb7dfc7a0f611bd231364 (`_make_store` defaulting to `InMemoryLightningStore`). https://raw.githubusercontent.com/microsoft/agent-lightning/f0a77cfad71e6222a3edb7dfc7a0f611bd231364/agentlightning/trainer/trainer.py. Fetched 2026-08-10.

[22] agent-lightning docs site version manifest (mike-generated `versions.json`: `latest`, `0.3.0` aliased `stable`, `0.2.2`, `0.2.1`, `0.2.0`, `0.1.2`). https://microsoft.github.io/agent-lightning/versions.json. Fetched 2026-08-10.

[23] agent-lightning docs, Write the First Algorithm. https://microsoft.github.io/agent-lightning/latest/how-to/write-first-algorithm/. Fetched 2026-08-10.

[24] agent-lightning docs, APO (Algorithm Zoo) (the `agl.APO(...)` shortcut and the "currently scoped to optimize a single prompt template" limitation). https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/apo/. Fetched 2026-08-10.

[25] agent-lightning docs, API Reference: Types (`RolloutConfig` field defaults, including `timeout_seconds = None` and `unresponsive_seconds = None`). https://microsoft.github.io/agent-lightning/latest/reference/types/. Fetched 2026-08-10.
