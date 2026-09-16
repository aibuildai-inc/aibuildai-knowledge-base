# OpenEnv

Hugging Face's Gymnasium-style environment framework: it does not train models — it packages the isolated, containerized environment a trainer talks to over `reset()` / `step()` / `state()`.

**OpenEnv** is described in its own README as "an e2e framework for creating, deploying and using isolated execution environments for agentic RL training, built using Gymnasium style simple APIs" [1]. It is built and maintained by Hugging Face [2], with the README itself warning that the project "is currently in an experimental stage" and that reader should "expect bugs, incomplete features, and APIs that may change in future versions" [1]. It lives at https://github.com/huggingface/OpenEnv [3]. Its shape is client-server: an `EnvClient` subclass (e.g. `EchoEnv`) on the caller's side opens a WebSocket to a FastAPI `Environment` server, usually running inside a Docker container, and exchanges typed `Action`/`Observation` objects through `reset()`, `step(action)`, and `state()` [1][4].

**When to pick it**: pick OpenEnv when you need a standardized, containerized environment server to plug into an RL trainer's rollout loop — its own docs list TRL's `GRPOTrainer(environment_factory=...)` as "the native path" for this, with OpenReward and Harbor named as interchangeable alternatives at the same `environment_factory` contract [5]. OpenEnv itself ships no trainer and no training method (see "Methods it ships" below); a card choosing between trainers (trl, verl, torchforge, ...) is a separate decision this card does not make. Skip it if you only need in-process Python tool functions with no process boundary — the docs say plainly that MCP-style tool serving "earns its complexity when the tool surface has to exist as a process boundary", and a local function needs none of this [6].

**Methods it ships**: none. OpenEnv contains no trainer class and no RL algorithm implementation; grep of the pinned commit's source shows no PPO/GRPO/DPO trainer code. The one GRPO example in the repo, `examples/grpo_blackjack/`, is torchforge's (Meta's PyTorch agentic-RL framework) own GRPO loop, wired to consume OpenEnv's `OpenSpielEnv` — the loss function, advantage computation, and metric logging in that file are torchforge's, not OpenEnv's [7]. What OpenEnv does ship, at the pinned commit 024eedc90305cc8bd7a5b44f44d1b987102e957b (2026-07-31) [8], is: a `Rubric` reward-composition module (`openenv.core.rubrics`) with `WeightedSum`, `Gate`, `Sequential`, LLM-judge rubrics, and `TrajectoryRubric`/`ExponentialDiscountingTrajectoryRubric` for delayed, per-step-discounted rewards [9]; an `AutoEnv`/`AutoAction` auto-discovery layer [10]; an `openenv` CLI (`init`, `import`, `push`, `serve`, plus `build`/`validate` documented on the packaging guide) [1][11]; and MCP (Model Context Protocol) tool serving over a `/mcp` endpoint, which the docs mark as still in flight — RFC 003 proposing it as the standard agent-facing interface is "still In Review", and only `echo_env`, `finqa_env`, and a local-wrapper `calendar_env` are MCP-backed today, with the rest (`textarena_env`/Wordle, `openspiel_env`, `chess_env`, `browsergym_env`, and most others) using custom, non-MCP action types instead [12].

**Scale it handles**: this is environment/container scale, not multi-GPU model-training scale — OpenEnv runs no model itself. One environment instance is one Docker container behind a FastAPI server; `LocalDockerProvider` and `UVProvider` (a container-free local process) are the single-machine paths, and `DockerSwarmProvider`, `DaytonaProvider`, `ACASandboxProvider`, and `ModalProvider` are the documented cloud/cluster paths, all implementing a common `ContainerProvider` contract so switching providers is "a one-line change"; `KubernetesProvider` exists only as a placeholder, marked "Not yet implemented" [13]. Concurrency within one running server is capped: by default "OpenEnv servers allow only 1 concurrent session", and scaling that up for parallel RL rollouts requires the environment to declare `SUPPORTS_CONCURRENT_SESSIONS: bool = True` and the server to set `max_concurrent_envs` on `create_app(...)`, sized to at least the trainer's `generation_batch_size` [14]. Running many environments in parallel is documented mechanism with a runnable example (`examples/daytona_tbench2_concurrent.py`, `asyncio.gather` over per-environment `DaytonaProvider` instances) rather than a published throughput or latency benchmark [13].

**Install**: `pip install openenv`; this installs "the full OpenEnv runtime: the environment server, the client, the openenv CLI, the web interface, and MCP support" [10]. Latest release is v0.4.1, published 2026-07-03 [15]; Python floor is `>=3.10`; license is BSD-3-Clause [16]. Core dependency floors at the v0.4.1 tag: `fastapi>=0.116.0`, `pydantic>=2.0.0`, `uvicorn>=0.24.0`, `requests>=2.25.0`, `typer>=0.9.0`, `huggingface_hub>=0.20.0`, `openai>=2.7.2`, `websockets>=15.0.1`, `fastmcp>=3.0.0`, `gradio>=4.0.0`, `httpx>=0.28.1`; no torch or numpy pin in the core package — those live in individual environments' own `pyproject.toml` files [16]. Optional extras: `daytona`, `aca` (pinned `azure-containerapps-sandbox>=0.1.0b2,<0.2.0` as a capped preview SDK), `modal`, and `inspect` (for the Inspect AI evaluation harness) [10][16]. The pinned screening commit 024eedc90305cc8bd7a5b44f44d1b987102e957b (pushed 2026-07-31) is ahead of this release: its `pyproject.toml` reports version `0.4.2.dev0`, a development version newer than what `pip install openenv` currently delivers [8]; the dependency floors themselves are identical between the two files [8][16]. Neither the getting-started page nor the release's `pyproject.toml` states a CUDA or GPU minimum — OpenEnv's own server has no model-execution hardware requirement; any GPU need comes from the trainer or the individual environment's own dependencies [10][16].

**Maintained by**: Hugging Face; the repository shows 2,494 GitHub stars and was created 2025-10-01, with a push recorded 2026-08-11 [2]. It is under active RFC-driven development: seven numbered RFCs are tracked in the README, including RFC 003 (MCP support) and RFC 010 ("Env-token World Modeling (ECHO)") [1].

## Quick start

The README's smallest complete example, connecting to a hosted environment rather than a local one [1]:

```bash
pip install openenv
pip install git+https://huggingface.co/spaces/openenv/echo_env
```

```python
import asyncio
from echo_env import CallToolAction, EchoEnv

async def main():
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        result = await client.reset()
        print(result.observation.echoed_message)  # "Echo environment ready!"

        result = await client.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": "Hello, World!"},
            )
        )
        print(result.observation.result)  # "Hello, World!"
        print(result.reward)

asyncio.run(main())
```

A synchronous form wraps the same client with `.sync()` for scripts and notebooks [1][10]. The docs' `AutoEnv` shortcut resolves a client and its action type by name alone, accepting `"echo"`, `"echo-env"`, or `"echo_env"` interchangeably [10]:

```python
from openenv import AutoAction, AutoEnv

env = AutoEnv.from_env("echo")
EchoAction = AutoAction.from_env("echo")
with env.sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello, OpenEnv!"))
```

## Start it

- **Connect to a hosted environment (simplest)**: point an `EnvClient` at a Hugging Face Space URL, as in the quick start above; no local setup is required [1][10].
- **Run it via Docker**: `EchoEnv.from_docker_image("registry.hf.space/openenv-echo-env:latest")` pulls and starts the container automatically; running the same image by hand with `docker run -it -p 7860:7860 --platform=linux/amd64 registry.hf.space/openenv-echo-env:latest` requires connecting the client to port 7860 [10].
- **Run a local server without Docker**: `uv pip install -e .` inside the environment directory, then `uv run server --host 0.0.0.0 --port 8000`, and connect with `EnvClient(base_url="http://localhost:8000")` [10].
- **Run in a cloud sandbox**: pass a `provider=` (`DaytonaProvider`, `ACASandboxProvider`, `ModalProvider`) to the client; `DaytonaProvider` and `ModalProvider` build their image from a Dockerfile path via `image_from_dockerfile(...)`. `LocalDockerProvider` and `DockerSwarmProvider` instead require calling `provider.start_container(image)` and `provider.wait_for_ready(base_url, timeout_s=180)` explicitly before connecting the client [13].
- **Scaffold a new environment**: `openenv init my_env` generates the client/server/model/Dockerfile skeleton; `openenv build` builds the Docker image (`--tag`, `--build-arg`, `--no-cache` flags available); `openenv validate --verbose` checks required files and entrypoints and exits non-zero on failure, so it can gate CI [11].
- **Deploy it**: `openenv push` deploys to a Hugging Face Space by default (auto-enabling the web UI); `openenv push --repo-id my-org/my-env` targets a specific Space; `openenv push --registry ghcr.io/my-org --tag my-env:latest` pushes to a container registry instead (web interface disabled by default in that path); `--env-var/-e` sets public Space variables and `--secret` sets private ones, which only apply on direct Space pushes, not `--registry` [11].
- **Reusing one server for multiple sessions**: after a client connects, `await client.new_session()` opens another independent session against the same running server; child sessions close when their parent closes, and `new_session()` can fail if the server has already hit `MAX_CONCURRENT_ENVS` [13].
- **Concurrency for training**: default server concurrency is 1 session; enabling parallel rollouts (as TRL's `environment_factory` path requires — it opens one WebSocket per generation) means declaring `SUPPORTS_CONCURRENT_SESSIONS: bool = True` in the environment and passing `max_concurrent_envs=<N>` to `create_app(...)` on the server side, where `N` should be at least `per_device_train_batch_size × gradient_accumulation_steps` [14].
- **Out-of-memory / capacity first aid**: for a shared Hugging Face Space, duplicate it to your own account before training, since shared Spaces "may not support" the N-simultaneous-WebSocket load a trainer opens [14]; there is no separate OpenEnv-side generation-memory knob to tune — GPU memory for the model lives with the trainer (e.g., TRL's `vllm_gpu_memory_utilization`), not with OpenEnv [14].

## Watch it

The mechanics only — what a reward signal or shaping choice MEANS for training belongs to the consuming trainer's own methods.

- **Reward computation happens inside the environment, not the trainer.** The contract is that `Environment.step()` computes the reward: an environment declares `self.rubric` in its constructor, calls `self._reset_rubric()` from `reset()`, and `self._apply_rubric(action, observation)` (or the async form) from `step()` [17][9].
- **Component-level introspection**: a composed `Rubric` exposes `env.rubric.named_rubrics()`, returning each named component's `last_score`; the docs recommend this specifically so a drop in the combined reward can be traced to which component caused it [9].
- **Per-environment-function reward logging in a training loop is the trainer's job, not OpenEnv's.** TRL's multi-environment example logs one metric per reward function (`train/reward_func_0`, `train/reward_func_1`, ...) and recommends watching those over the combined `train/reward`, which "alternates between environments and can appear noisy" when one model is trained across several OpenEnv environments at once [18].
- **Sample-level inspection**: the built-in web interface gives a two-pane, real-time view (agent interaction on one side, state observation on the other) with an action-history log; it is disabled by default for local development and turned on with `ENABLE_WEB_INTERFACE=true`, or explicitly via `create_web_interface_app(env, YourAction, YourObservation)`, served at `/web` [1].
- **Health check**: every environment's Dockerfile carries a container-level health check, `HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1`, which is a liveness probe on the server process, not an RL training-health or stopping-rule signal [19].
- **No RL-specific stopping rule or reward threshold is published by OpenEnv itself.** The reward-design guide states design principles (start with a sparse signal, shape carefully, prefer `WeightedSum`/`Gate` composition over dense hand-crafted shaping) and common pitfalls (reward hacking, sparse rewards that never fire, conflicting signals) but names no numeric threshold or stopping criterion [20]; the only numeric capacity limit found across the runtime-providers, concepts, and TRL-integration pages is the session-concurrency cap `MAX_CONCURRENT_ENVS` described under "Start it" above, which governs how many simultaneous episodes a server accepts, not when training should stop [13][14].

## Save it

OpenEnv saves environment code and container images, not trained model weights — a run's checkpoints are the consuming trainer's responsibility (e.g., TRL's own checkpoint contract, documented on its own card) and are not covered here.

- **What `openenv push` writes**: it validates the local `openenv.yaml`, injects Hugging Face Space frontmatter as needed, and uploads the prepared bundle either to a Hugging Face Space (default) or to a container registry via `--registry` [11].
- **Environment manifest**: every environment carries an `openenv.yaml` naming its client class, action class, observation class, and `default_image`; this file, not a checkpoint directory, is what identifies a saved environment version [17].
- **Reload / connect to a saved environment**: `MyEnv.from_docker_image("my-env:latest")` loads a built image locally; `MyEnv.from_hub("my-org/my-env")` connects to a pushed Hugging Face Space; `MyEnv(base_url="http://localhost:8000")` connects to a locally running server — all three return the same typed client, so "loading" a saved OpenEnv environment means starting or connecting to its server, never deserializing weights into memory [11].
- **What this is NOT**: an `openenv push`-produced Space or registry image contains the environment's server code and Docker image only; it holds no trained model parameters, so an evaluator cannot load a policy from it the way it would load a trainer's checkpoint — the model being trained against the environment is saved separately by that trainer.

## Find it in the docs

The docs are the live source; this section teaches the lookup, not the content.

- Address pattern: `https://huggingface.co/docs/openenv/<slug>` for the default (`main`) docs, or `https://huggingface.co/docs/openenv/v0.4.1/<slug>` pinned to the latest release — checked 2026-08-11: `/docs/openenv/v0.4.1/index` returns the same title as the `main` index [22][1]. `<slug>` mixes flat names (`getting-started`, `index`) and nested ones (`guides/concepts`, `guides/rewards`, `guides/runtime-providers`, `getting_started/environment-builder`, `tutorials/mcp-environment`); several plausible flat guesses (`cli-reference`, `container-providers`, `core-api`, `core-concepts`) 404 — the nested forms above are the ones that resolve [23][9][13][11][12].
- Question-to-slug map: environment lifecycle and the `Action`/`Observation`/`StepResult`/`Rubric` vocabulary → `guides/concepts` [17]; reward design and the `Rubric` composition API → `guides/rewards` [9]; picking a container/cloud runtime → `guides/runtime-providers` [13]; building and packaging a new environment end to end (CLI `build`/`validate`/`push`, Dockerfile template, GitHub Actions build matrix) → `getting_started/environment-builder` [11]; MCP tool serving, its current adoption state, and the `/ws` vs `/mcp` boundary → `tutorials/mcp-environment` [12]; the pre-built environment catalog → `environments`, itself flagged as possibly stale in favor of the `openenv` Hugging Face organization page and Spaces tagged `agent-environment` [24].
- Runnable references beyond the docs: the repo's `examples/` tree, including `examples/grpo_blackjack/` (torchforge GRPO on `OpenSpielEnv`) [7], `examples/daytona_tbench2_concurrent.py` (N-sandbox concurrency demo) [13], and a Colab-hosted end-to-end tutorial linked from the README [1]. `envs/README.md` in the repo is a complete step-by-step guide to building a new environment (models, FastAPI server via `create_fastapi_app`, Dockerfile, client) [21].
- Trainer-side integration docs live outside this repo: TRL's own OpenEnv integration guide (`https://huggingface.co/docs/trl/openenv`) documents the `environment_factory` contract, the multi-turn tool-calling loop, and a full Wordle GRPO worked example with concrete `GRPOConfig` values [5][18].
- No official MCP query endpoint or curated community-tutorials page for OpenEnv itself was found in the pages fetched for this card (searched the docs index, getting-started, concepts, rewards, runtime-providers, environment-builder, and MCP tutorial pages); the closest curated pointer is the `openenv` Hugging Face organization page and its tagged Spaces, named directly on the environments-catalog page rather than through a separate community-tutorials index [24].
- **Documented trap**: shared Hugging Face Spaces "may not support" the concurrent WebSocket load a trainer's `environment_factory` opens (one connection per generation); TRL's own docs recommend duplicating the Space before training rather than training against a shared one [14]. **Honest boundary**: OpenEnv's Kubernetes runtime is explicitly unimplemented (placeholder class only) [13], and MCP tool serving covers only a handful of the catalog's environments today, with most environments still on custom, non-MCP action types [12].

## Sources

[1] OpenEnv README at pinned commit 024eedc90305cc8bd7a5b44f44d1b987102e957b. https://raw.githubusercontent.com/huggingface/OpenEnv/024eedc90305cc8bd7a5b44f44d1b987102e957b/README.md. Fetched 2026-08-11.

[2] OpenEnv GitHub repository API. https://api.github.com/repos/huggingface/OpenEnv. Fetched 2026-08-11.

[3] OpenEnv GitHub repository. https://github.com/huggingface/OpenEnv. Fetched 2026-08-11.

[4] OpenEnv README architecture diagram (component overview), same source as [1].

[5] trl OpenEnv integration guide. https://huggingface.co/docs/trl/openenv. Fetched 2026-08-11.

[6] OpenEnv MCP tutorial. https://huggingface.co/docs/openenv/tutorials/mcp-environment. Fetched 2026-08-11.

[7] OpenEnv grpo_blackjack example at pinned commit. https://raw.githubusercontent.com/huggingface/OpenEnv/024eedc90305cc8bd7a5b44f44d1b987102e957b/examples/grpo_blackjack/grpo_utils.py. Fetched 2026-08-11.

[8] OpenEnv pyproject.toml at pinned commit 024eedc90305cc8bd7a5b44f44d1b987102e957b. https://raw.githubusercontent.com/huggingface/OpenEnv/024eedc90305cc8bd7a5b44f44d1b987102e957b/pyproject.toml. Fetched 2026-08-11.

[9] OpenEnv reward-design guide. https://huggingface.co/docs/openenv/guides/rewards. Fetched 2026-08-11.

[10] OpenEnv getting-started page. https://huggingface.co/docs/openenv/getting-started. Fetched 2026-08-11.

[11] OpenEnv environment-builder ("Packaging & Deploying") guide. https://huggingface.co/docs/openenv/getting_started/environment-builder. Fetched 2026-08-11.

[12] OpenEnv MCP tutorial, same source as [6].

[13] OpenEnv runtime-providers guide. https://huggingface.co/docs/openenv/guides/runtime-providers. Fetched 2026-08-11.

[14] trl OpenEnv integration guide, server-concurrency section, same source as [5].

[15] OpenEnv GitHub Releases API. https://api.github.com/repos/huggingface/OpenEnv/releases. Fetched 2026-08-11.

[16] OpenEnv pyproject.toml at the v0.4.1 release tag. https://raw.githubusercontent.com/huggingface/OpenEnv/v0.4.1/pyproject.toml. Fetched 2026-08-11.

[17] OpenEnv Concepts guide. https://huggingface.co/docs/openenv/guides/concepts. Fetched 2026-08-11.

[18] trl OpenEnv integration guide, multi-environment training section, same source as [5].

[19] OpenEnv envs/README.md ("Building Your Own Environment") at pinned commit, Dockerfile template. https://raw.githubusercontent.com/huggingface/OpenEnv/024eedc90305cc8bd7a5b44f44d1b987102e957b/envs/README.md. Fetched 2026-08-11.

[20] OpenEnv reward-design guide, "Common Pitfalls" section, same source as [9].

[21] OpenEnv envs/README.md, same source as [19].

[22] OpenEnv docs index, v0.4.1-pinned form. https://huggingface.co/docs/openenv/v0.4.1/index. Fetched 2026-08-11.

[23] OpenEnv docs index (main). https://huggingface.co/docs/openenv/index. Fetched 2026-08-11.

[24] OpenEnv environments catalog page. https://huggingface.co/docs/openenv/environments. Fetched 2026-08-11.
