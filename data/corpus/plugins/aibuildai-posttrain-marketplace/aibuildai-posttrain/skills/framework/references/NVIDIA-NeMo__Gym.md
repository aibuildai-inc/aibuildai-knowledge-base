# NeMo Gym

NVIDIA's environment and verifier library for evaluating and post-training agents - it runs rollouts and scores them, then hands the scored data to an external RL or SFT framework of your choice.

NeMo Gym "is a library for evaluating and improving models and agents using environments," providing infrastructure to develop environments, run evaluation and training at scale, and a library of pre-built benchmarks and training environments [1]. It is built by NVIDIA under the NeMo umbrella - the README describes NeMo Gym as "a component of NVIDIA NeMo, a GPU-accelerated platform for training generative AI models and optimizing AI agents" [1] - and it is maintained in the `NVIDIA-NeMo/Gym` GitHub organization repository [2]. Its API shape is a local-server model driven by a `gym` CLI: `gym env start` launches a resources server (task data plus a `/verify` endpoint), an agent server (the harness that talks to the model), and a model server (policy inference), and `gym eval run` then drives an agent through tasks and writes scored rollouts to disk [1][3]. NeMo Gym does not itself implement RL or SFT training loops: "Train with the RL framework of your choice" is one of its stated capabilities, and the rollouts/rewards it produces are consumed by an external trainer such as NeMo RL, verl, or Unsloth [1]. It lives at https://github.com/NVIDIA-NeMo/Gym [2].

**When to pick it**: you need reproducible, stateful evaluation of agents (tool calling, code execution, sandboxes) across a shared library of environments and verifiers, or you want to generate scored rollouts to feed into training, and you are willing to run that training through a separate framework - NeMo Gym explicitly is not a trainer: its own docs say "If you're scoring model outputs with a stateless check and don't need scale or training, a script is probably sufficient" [1], and it defers all weight updates to NeMo RL, verl, or Unsloth [4]. This makes it a different kind of choice than trainer-class libraries such as trl or verl (cross-reference; not covered here) - pick NeMo Gym for the environment/verifier/rollout-collection layer, and pick a training library separately for the optimizer step. The README itself flags the project as "currently in early development," warning readers to "expect evolving APIs, incomplete documentation, and occasional bugs" [1].

**Methods it ships**: NeMo Gym does not ship RL or SFT algorithm implementations; its own docs describe GRPO, DAPO, and GSPO only as background concepts for readers deciding when to train, explicitly deferring implementation to an external "Training Library" [5]. What it ships is the reward/verification layer: every resources server implements a `verify()` method returning a scalar `reward` field (the shortlist row's own `methods_seen` pointer, `nemo_gym/reward_profile.py`, is not that verifier but the `gym eval profile` implementation - a `RewardProfiler` class that computes per-task pass/majority statistics from collected rollouts, at the v0.5.0 tag [6]) [7]. For multi-objective verification it ships `BaseMultiRewardVerifyResponse`, a response type that carries a scalar reward alongside a `reward_components` dict and per-objective top-level fields, documented as feeding "GDPO and other multi-objective trainers" while the scalar still feeds single-reward consumers like GRPO [7]. Around that verifier layer it ships an environment hub: the README's Available Environments table lists 164 rows (a row is one environment/harness-config pairing, so some environments appear more than once), of which 5 rows carry no domain label; of the labeled rows the largest domains are agent (38), knowledge (30), math (27), coding (22), and other (19), with instruction_following (12), games (5), rlhf (3), and safety (3) rounding out the table, by direct count of that table as of this reading [1]. A curated 76-benchmark subset with pre-wired agents is exposed through `gym list benchmarks` [8]. It also ships rollout-to-training-data conversion for offline SFT/DPO, marked experimental: "This tutorial is experimental and may contain bugs" [9].

**Scale it handles**: single machine up to many concurrent rollouts within one Gym invocation - `gym eval run` takes a `--concurrency` flag capping simultaneous samples, and Ray (`ray[default]>=2.56.1`) is a non-optional, auto-installed dependency that backs this rollout-level scaling [10][11]; the README states no GPU is required to run NeMo Gym itself, only for the model or resources-server workloads it drives [1]. Multi-node scale is documented, but it belongs to the external training framework, not to Gym: the NeMo RL GRPO tutorial's multi-node step launches training via Slurm from the NeMo RL repo, with Gym only supplying the environment config, and shows no Gym-side node count knob [12]. For isolating untrusted code execution, the README's v0.5.0 changelog names seven sandbox providers - "Docker, Daytona, ECS Fargate, Enroot, and OpenShell join OpenSandbox and Apptainer" [1] - though the live Sandbox docs page, fetched the same day, documents dedicated setup pages for only four of the seven (OpenSandbox, Apptainer, Docker, ECS Fargate) [13], so treat Daytona, Enroot, and OpenShell as named but not yet walked-through in the docs.

**Install**: `pip install nemo-gym`, or clone and `uv sync` for the repo; version 0.5.0 released 2026-08-07 per the GitHub release [14], tag `v0.5.0` resolving to commit `6e50df7a444262536b558d1c039e662bfcc8dbf7` [15]; requires Python >=3.13.14 [16][1]; Apache-2.0 [16]. Note a docs/package mismatch: the live Installation page says "Python 3.12 is required" [17], while the README and the PyPI package metadata both pin >=3.13.14 [1][16] - trust the package metadata for what `pip install` actually enforces. Read at the v0.5.0 tag, `pyproject.toml` pins no deep-learning core (no `torch` dependency in the base install) [18]; the two load-bearing pins likely to collide with a reader's existing environment are the fast-moving-SDK upper bounds `openai<=2.7.2` and `anthropic<=0.109.2` (the file's own comments date these updates to 2026-02-17 and 2026-06-16 respectively) [18], and the `vllm` extra (`pip install nemo-gym[vllm]`), exactly pinned to `vllm==0.24.0` with `flashinfer-python==0.6.12`, matched to that vllm version specifically so pre-compiled CUDA kernels are used instead of triggering JIT compilation on the first generation step [18]. `ray[default]>=2.56.1` is a base (non-extra) dependency [18]. No CUDA or hardware minimum is stated by the README or `pyproject.toml`; the README's own Requirements table says a GPU is "Not required for NeMo Gym library operation" and may only be needed for the resources servers or models you point it at [1].

**Maintained by**: NVIDIA, in the `NVIDIA-NeMo` GitHub organization [2]; about 1,104 GitHub stars as of this reading [2]. Actively developed: the repo's most recent push (2026-08-12) is five days after the v0.5.0 release (2026-08-07) [2][14], and the README's News section documents a release cadence of roughly one minor version per month through 2026 (v0.3.0 on 06/04, v0.4.0 on 07/01, v0.5.0 on 08/06) [1].

## Quick start

The smallest complete run, quoted from the README's own Quick Start section [1]. Install and configure a model:

```bash
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym
uv venv --python 3.13.14 && source .venv/bin/activate
uv sync
```

```yaml
# env.yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: <your-openai-api-key>
policy_model_name: gpt-4.1-2025-04-14
```

Start the three coordinating servers for a built-in multiple-choice environment, then run the agent on a handful of tasks:

```bash
gym env start \
    --resources-server mcqa \
    --model-type openai_model
```

```bash
source .venv/bin/activate
gym eval run --no-serve \
    --agent mcqa_simple_agent \
    --input resources_servers/mcqa/data/example.jsonl \
    --output results/mcqa_rollouts.jsonl \
    --limit 5 \
    --num-repeats 1
```

This prints a progress bar and then key metrics such as `"mean/reward": 0.8` and `"pass@1/accuracy": 80.0`, and writes rollouts plus `results/mcqa_rollouts_aggregate_metrics.json` [1].

## Start it

- **Servers**: `gym env start --resources-server <name> --model-type <type>` launches the resources, agent, and model servers for a chosen environment on one machine; `gym env status` reports each server's health, PID, and uptime, ending in a summary line like `3 servers found (3 healthy, 0 unhealthy)` [19].
- **Running an evaluation**: `gym eval run` is the main command - by default it starts the required servers itself, or add `--no-serve` to run against servers already started with `gym env start` [3]. Common flags: `--benchmark NAME` or `--resources-server NAME` to pick the environment, `--model-type NAME`, `--input`/`--output` JSONL paths, `--limit`, `--num-repeats` (rollouts per task, an int applied to every task or a dict keyed by `agent_ref.name` for per-agent counts, e.g. `'{simple_agent: 32, swe_agent: 1}'`), and `--concurrency` for the maximum concurrent samples [3]. Generation parameters `--temperature`, `--top-p`, and `--max-output-tokens` have dedicated flags; any other `responses_create_params` field is set with a raw override `++responses_create_params.<field>=value` - note the merge is shallow, so overriding a nested field like `++responses_create_params.reasoning.effort=low` replaces that entire nested dict rather than merging into it [3].
- **Config surface**: server configs are YAML under `<server_type>/<implementation>/configs/*.yaml` and are checked into the repo; `env.yaml` at the repository root holds secrets and machine-specific values and is gitignored [20]. Every server config declares a `server_id`, `server_type` (one of `responses_api_models`, `resources_servers`, `responses_api_agents`), and `implementation`; a resources server config additionally declares a `domain` from a fixed set (math, coding, agent, knowledge, instruction_following, long_context, safety, games, translation, e2e, rlhf, other) and a `verified` flag (default `false`) meaning it has "Passed reward profiling and training checks" [20]. Relative paths (config files, datasets, prompt configs, the `--<component>` selectors) resolve against an ordered list of roots - extra roots from `NEMO_GYM_EXTRA_ROOTS` or `--search-dir` first, then the current working directory, then the Gym install root - so a same-named component in your own root shadows a built-in one [20]. No changed-precision or hardware default is documented on this config page; Gym itself sets no training-time precision, since precision belongs to the external trainer.
- **Model backends**: the model server supports OpenAI, Azure OpenAI, hosted inference providers (Fireworks, Together.ai, OpenRouter, etc.), a self-managed vLLM server, a Gym-managed "Local vLLM" server, and a "Local vLLM Proxy" for serving multiple request-time configs from one deployment; backends marked "training" in the docs are the ones that return token IDs and log probabilities, a requirement for RL training [21].
- **Throughput/OOM first aid**: lower `--concurrency` on `gym eval run` to reduce simultaneous in-flight samples against the model and resources servers [3]. Beyond that flag, the CLI and configuration references read for this card publish no dedicated out-of-memory guidance - GPU memory pressure during generation is a property of whichever model server backend (e.g. vLLM) you point Gym at, not of Gym's own process.
- **Interrupted runs**: `gym eval run --resume` restarts the same command after a crash and skips rows already completed, matched by `(task_index, rollout_index)` against a `<output-stem>_materialized_inputs.jsonl` sidecar written on the first run; retriable failures land in a `<stem>_failures.jsonl` sidecar so partial progress survives a crash; if the config, schema, or data changes between runs, delete both sidecar files to start fresh [22].

## Watch it

This section covers the mechanics only - what a metric value means for a given training algorithm (GRPO, DPO, ...) lives on that method's own card, not here.

- **Where results land**: `gym eval run` writes rollouts to your `--output` JSONL and, after collection, computes aggregate metrics by POSTing stripped verify responses to each agent's `/aggregate_metrics` endpoint, writing the result to `<output>_aggregate_metrics.json` [23]. This file is a JSON array with one entry per agent, each carrying `agent_metrics` (mean/max/min/median/std computed over every numeric verify-response field, e.g. `mean/reward`, `max/reward`, `min/reward`, `median/reward`, `std/reward`), `key_metrics` (by default every `mean/*` entry), and `group_level_metrics` (the same per-task, one entry per task) [23]. A resources server can add fields to this by overriding two hooks: `compute_metrics(tasks)` for dataset-wide statistics such as pass@k, and `get_key_metrics(agent_metrics)` to choose which fields surface in `key_metrics` [23].
- **Per-task reward profiling**: `gym eval profile --inputs <materialized_inputs.jsonl> --rollouts <rollouts.jsonl>` computes per-task average reward, standard deviation, min/max, and pass rate from rollouts collected with `--num-repeats` greater than 1, useful for filtering tasks by difficulty or variance before training [24]. This is the command implemented by `nemo_gym/reward_profile.py`'s `RewardProfiler` class, at the v0.5.0 tag [6].
- **Multi-objective verification**: a resources server can return `reward_components` (an `{objective: score}` dict for multi-objective trainers like GDPO) alongside the scalar `reward` that single-reward consumers such as GRPO read directly; aggregate metrics only profiles top-level numeric fields on the response, so an objective nested only inside `reward_components` will not appear in `agent_metrics` unless it is also declared as a top-level field [7].
- **No tracker integration is documented** in the CLI, quickstart, or configuration pages read for this card - metrics land in the JSON files above, not in a dashboard like W&B or TensorBoard, unless the external training framework you feed the rollouts into adds that logging itself. As one example of that boundary: the NeMo RL GRPO multi-node tutorial logs to Weights & Biases via `WANDB_API_KEY` and `logger.wandb.project`, and reports metrics named `train:reward_mean` and `val:accuracy` - these are NeMo RL's own training-loop metrics, not something NeMo Gym itself emits [12].
- **Recomputing rewards without re-running inference**: `gym eval reverify` replays existing rollouts through a resources server's `/verify` endpoint using stored data, useful after fixing a verifier bug. It first checks the server's `GET /reverify_mode`: `stateless` (safe - "the verifier is a pure function of (request body, server config)"), `unsupported` (unsafe - "the verifier reads per-rollout session state that is no longer present"), or `unknown` (the undeclared default, "Treated as potentially unsafe") [25]. `unsupported` or `unknown` servers abort the command unless `--force` is passed, in which case the output file is prefixed `unsafe_` [25].
- **Sample-level generation logging**: this card's CLI/configuration reading found a model-call capture feature - "Record per-rollout model requests, responses, token usage, latency, and correlation evidence" - listed on the model-server backends page as an "observability" capability, but did not locate the specific config field that turns it on within the pages fetched for this card [21].
- **Stopping rule**: none is published in the CLI reference, configuration reference, or model-server pages read for this card (searched 2026-08-12) - `gym eval run` and `gym eval profile` report statistics but neither page names an early-stopping threshold, patience value, or automatic abort condition tied to a reward curve.

## Save it

- **Rollouts and metrics on disk**: a `gym eval run` invocation writes three files next to your `--output` path: the rollouts JSONL itself, a `<stem>_materialized_inputs.jsonl` sidecar (the fully expanded input rows), and a `<stem>_aggregate_metrics.json` summary [22][23]. There is no model checkpoint directory here - NeMo Gym does not train, so it never writes model weights.
- **Sharded runs**: `gym eval aggregate --input-glob '<pattern>' --output <path>` merges rollout shards (produced by running `gym eval run --no-serve` with `+disable_aggregation=true` per shard) into one rollouts file and recomputes aggregate metrics over the global union [26].
- **Turning rollouts into training data**: the offline-training tutorial (marked "experimental and may contain bugs") converts collected rollouts into SFT data (`{"messages": [...], "quality_score": ...}` records - full conversations) or DPO data (`{"prompt", "chosen", "rejected", "quality_difference"}` preference pairs), after a quality-filtering step that drops low-reward or malformed rollouts; that filtering and formatting is example Python in the tutorial, not a built-in CLI subcommand [9].
- **Datasets to/from the Hub**: `gym dataset upload --name <name> --input <rollouts.jsonl> +resource_config_path=<config>` pushes a JSONL to a Hugging Face dataset repo (naming convention `{hf_organization}/{hf_dataset_prefix}-{domain}-{resources_server}`, credentials read from `env.yaml`), with `--create-pr` to open a pull request instead of pushing directly when you lack write access; `gym dataset download --repo-id <repo> --output-dir <dir>` pulls a dataset back down [27].
- **No adapter or merged-model save exists in Gym itself**: since Gym does not train, PEFT/LoRA adapter saving, merging, and reload are properties of whichever training framework (NeMo RL, verl, Unsloth) you route the rollouts to - check that framework's own card for the adapter-vs-merged-model contract before assuming a "saved model" from this pipeline is directly loadable.
- **Loader handoff**: what NeMo Gym itself hands off to an evaluator or trainer is scored rollout data (JSONL) and aggregate-metrics JSON, not a model artifact - whether a downstream evaluator can load the *trained* result is entirely the training framework's loader contract, out of scope for this card.

## Find it in the docs

The docs are the live source; this section teaches the lookup and does not mirror the content.

- **Address pattern**: `https://docs.nvidia.com/nemo/gym/<page>` - checked 2026-08-12: unlike some Fern-hosted docs sites, this one does not take a working version segment for individual pages; every page resolves as a bare `/nemo/gym/<section>/<page>` slug, e.g. `/nemo/gym/get-started/installation`, `/nemo/gym/reference/cli-commands`, `/nemo/gym/tutorials/training-tutorials/nemo-rl-grpo` [3]. The `/main/` prefix behaves inconsistently by depth: the bare site root `/nemo/gym/main/` 307-redirects to `/nemo/gym/about` (the "main" segment is dropped), but `/nemo/gym/main/<slug>` for an actual subpage instead 301-redirects to the same `/main/`-prefixed path with a trailing slash added, which then serves 200 directly - so drop `/main/` for the root but do not assume it is stripped everywhere [28]. Appending `.md` to any page URL returns that page's Markdown source, and the site publishes an `/llms.txt` index at its root for AI-agent-oriented lookup [28].
- **Page-slug recipes**: About/overview pages sit under `about/` (e.g. `about/ecosystem` for the framework-integration map [4]); getting-started pages under `get-started/` (`installation`, `quickstart`); the full CLI surface is one page, `reference/cli-commands` [3]; the configuration syntax is `reference/configuration` [20]; concept explainers sit under `about/concepts/` (`training`, `environments`) [5]; training-framework walkthroughs sit under `tutorials/training-tutorials/` with one slug per framework (`nemo-rl-grpo`, `verl`, `unsloth`, `offline-training-w-rollouts`) [12][29][9].
- **Question-to-slug map**: "what CLI flags exist" -> `reference/cli-commands` (also carries the full legacy `ng_*` -> `gym ...` command-mapping table for anyone migrating an older script) [3]; "how do I configure a server" -> `reference/configuration`; "which model backend should I use" -> `model-server`; "how do I add multiple reward components" -> `build-verifiers/multi-reward-verification` [7]; "what sandbox providers exist" -> `infrastructure/sandbox` (documents OpenSandbox, Apptainer, Docker, and ECS Fargate providers individually; Daytona, Enroot, and OpenShell are named in the README's changelog but had no dedicated docs subpage as of this reading) [13][1].
- **Runnable references beyond the docs**: the repo's `resources_servers/`, `responses_api_agents/`, and `responses_api_models/` directories hold every built-in environment, agent harness, and model backend with their own configs and example data (the quickstart's `resources_servers/mcqa/data/example.jsonl` is one) [1][2]; `docker/install_codec_deps.sh` restores codec-bearing packages (`opencv-python-headless`, `torchvision`, `torchaudio`) that the NeMo-Gym container drops by default to avoid shipping royalty-bearing binaries, for anyone running VLM or audio/video benchmarks in-container [1].
- **Community layer**: this card found no curated community-tutorials page analogous to trl's or a similarly structured index during this reading; the closest official curation is the README's own Ecosystem section, which names specific integration points rather than blog posts - environment libraries (Aviary, Harbor, OpenEnv, Reasoning Gym, Verifiers), training frameworks (NeMo RL, Unsloth, VeRL), and agent harnesses (OpenHands, Mini SWE Agent, LangGraph) [4]. No practitioner-blog layer was located in the pages fetched for this card.
- **MCP**: no official MCP endpoint for NeMo Gym's own docs was found in the pages read for this card. NVIDIA's docs site groups NeMo Gym under the general `docs.nvidia.com` domain rather than a dedicated developer-docs MCP server; this is a "not found in what was read" statement, not a confirmed absence.
- **Traps**: no maintainer reply in a closed GitHub issue was consulted for this card - that search was not performed in this reading, so no issue-sourced trap is reported here. The one honest boundary this card can state from the docs themselves: the project is self-described as "early development" with "evolving APIs, incomplete documentation, and occasional bugs" [1], and the RL Framework Compatibility reference page's own version table stops at Gym v0.4.0 while the current release is v0.5.0 [30] - treat any framework-pairing guidance there as pinned to that older Gym version until the table is updated.

## Sources

Every claim above is a 2026-08-12 reading unless a specific fetch date is given inline; pages under `docs.nvidia.com/nemo/gym/` are unpinned, mutable docs read on that date. Method names (SFT, DPO, GRPO, DAPO, GSPO) are deliberately cited to nothing here beyond the concept page that mentions them in passing [5]; their defining papers live on the methodology cards, not this one. Training-framework specifics for NeMo RL, verl, and Unsloth belong on those frameworks' own cards and are cited here only where NeMo Gym's own tutorial pages describe the hand-off.

[1] NeMo Gym GitHub repository README. https://github.com/NVIDIA-NeMo/Gym (raw README fetched at https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/main/README.md). Fetched 2026-08-12.

[2] NeMo Gym GitHub repository (API metadata: stars, license, pushed_at, created_at). https://github.com/NVIDIA-NeMo/Gym via https://api.github.com/repos/NVIDIA-NeMo/Gym. Fetched 2026-08-12.

[3] NeMo Gym docs, CLI Commands reference. https://docs.nvidia.com/nemo/gym/reference/cli-commands. Fetched 2026-08-12.

[4] NeMo Gym docs, About/Ecosystem page. https://docs.nvidia.com/nemo/gym/about/ecosystem. Fetched 2026-08-12.

[5] NeMo Gym docs, About/Concepts: Training page. https://docs.nvidia.com/nemo/gym/about/concepts/training. Fetched 2026-08-12.

[6] `nemo_gym/reward_profile.py` at the v0.5.0 tag, commit 6e50df7a444262536b558d1c039e662bfcc8dbf7. https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/v0.5.0/nemo_gym/reward_profile.py. Fetched 2026-08-12.

[7] NeMo Gym docs, Build Verifiers: Multi-Reward Verification page. https://docs.nvidia.com/nemo/gym/build-verifiers/multi-reward-verification. Fetched 2026-08-12.

[8] NeMo Gym docs, Quickstart page (`gym list benchmarks` output showing "Available benchmarks in NeMo Gym (76)"). https://docs.nvidia.com/nemo/gym/get-started/quickstart. Fetched 2026-08-12.

[9] NeMo Gym docs, Tutorials: Offline Training with Rollouts page. https://docs.nvidia.com/nemo/gym/tutorials/training-tutorials/offline-training-w-rollouts. Fetched 2026-08-12.

[10] NeMo Gym docs, CLI Commands reference (`--concurrency` flag on `gym eval run`). https://docs.nvidia.com/nemo/gym/reference/cli-commands. Fetched 2026-08-12.

[11] NeMo Gym `pyproject.toml` at the v0.5.0 tag (Ray dependency pin). https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/v0.5.0/pyproject.toml. Fetched 2026-08-12.

[12] NeMo Gym docs, Tutorials: NeMo RL GRPO, Multi-Node Training page. https://docs.nvidia.com/nemo/gym/tutorials/training-tutorials/nemo-rl-grpo/multi-node-training. Fetched 2026-08-12.

[13] NeMo Gym docs, Infrastructure: Sandbox API page. https://docs.nvidia.com/nemo/gym/infrastructure/sandbox. Fetched 2026-08-12.

[14] NeMo Gym GitHub Releases (v0.5.0, published_at 2026-08-07T00:37:57Z). https://github.com/NVIDIA-NeMo/Gym/releases via https://api.github.com/repos/NVIDIA-NeMo/Gym/releases. Fetched 2026-08-12.

[15] NeMo Gym GitHub tags (v0.5.0 -> commit 6e50df7a444262536b558d1c039e662bfcc8dbf7). https://github.com/NVIDIA-NeMo/Gym/tags via https://api.github.com/repos/NVIDIA-NeMo/Gym/tags. Fetched 2026-08-12.

[16] `nemo-gym` on PyPI (JSON API: version 0.5.0, requires_python >=3.13.14, license). https://pypi.org/project/nemo-gym/ via https://pypi.org/pypi/nemo-gym/json. Fetched 2026-08-12.

[17] NeMo Gym docs, Installation page. https://docs.nvidia.com/nemo/gym/get-started/installation. Fetched 2026-08-12.

[18] NeMo Gym `pyproject.toml` at the v0.5.0 tag (dependency pins, extras, dated comments). https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/v0.5.0/pyproject.toml. Fetched 2026-08-12.

[19] NeMo Gym docs, CLI Commands reference (`gym env status` output format). https://docs.nvidia.com/nemo/gym/reference/cli-commands. Fetched 2026-08-12.

[20] NeMo Gym docs, Reference: Configuration page. https://docs.nvidia.com/nemo/gym/reference/configuration. Fetched 2026-08-12.

[21] NeMo Gym docs, Model Server page (backend list, training-capable backends, model-call capture). https://docs.nvidia.com/nemo/gym/model-server. Fetched 2026-08-12.

[22] NeMo Gym docs, CLI Commands reference (Resume interrupted runs section). https://docs.nvidia.com/nemo/gym/reference/cli-commands. Fetched 2026-08-12.

[23] NeMo Gym docs, Aggregate Metrics reference page. https://docs.nvidia.com/nemo/gym/evaluation/aggregate-metrics. Fetched 2026-08-12.

[24] NeMo Gym docs, CLI Commands reference (`gym eval profile` section). https://docs.nvidia.com/nemo/gym/reference/cli-commands. Fetched 2026-08-12.

[25] NeMo Gym docs, CLI Commands reference (`gym eval reverify` section and `/reverify_mode` table). https://docs.nvidia.com/nemo/gym/reference/cli-commands. Fetched 2026-08-12.

[26] NeMo Gym docs, CLI Commands reference (`gym eval aggregate` section). https://docs.nvidia.com/nemo/gym/reference/cli-commands. Fetched 2026-08-12.

[27] NeMo Gym docs, FAQ reference page (Hugging Face dataset upload/download how-tos). https://docs.nvidia.com/nemo/gym/reference/faq. Fetched 2026-08-12.

[28] NeMo Gym docs, About page (redirect target confirming the bare `/nemo/gym/<slug>` URL pattern; `.md` suffix and `/llms.txt` index). https://docs.nvidia.com/nemo/gym/about. Fetched 2026-08-12.

[29] NeMo Gym docs, Tutorials: Training with VeRL page. https://docs.nvidia.com/nemo/gym/tutorials/training-tutorials/verl. Fetched 2026-08-12.

[30] NeMo Gym docs, Reference: RL Framework Compatibility page. https://docs.nvidia.com/nemo/gym/reference/rl-framework-compatibility. Fetched 2026-08-12.
