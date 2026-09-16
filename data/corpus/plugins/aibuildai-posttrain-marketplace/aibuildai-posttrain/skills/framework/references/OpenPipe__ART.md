# ART (Agent Reinforcement Trainer)

A client/backend RL library for training multi-step LLM agents with GRPO, splitting a thin agent-side client from a GPU-side training-and-inference backend that can run locally or on OpenPipe's managed autoscaling service.

**ART** (Agent Reinforcement Trainer) is "an open-source training framework for teaching agentic LLMs to improve performance and reliability through experience" [1]. It is built and maintained by OpenPipe, published under the GitHub organization OpenPipe [2]. The API separates concerns into two objects: an `art.TrainableModel` client that an agent's own code calls for inference and that gathers scored `Trajectory` objects, and a backend (`LocalBackend` or `ServerlessBackend`) that runs a vLLM server for generation and drives GRPO weight updates, swapping the two while alternating between inference and training [3]. It lives at https://github.com/OpenPipe/ART [2].

**When to pick it**: pick ART when the task is a multi-step Python agent (tool calls, multi-turn conversation) and the priority is a small amount of glue code around GRPO, with the option to run everything on one local GPU or to hand training off to OpenPipe's managed W&B Training service without provisioning infrastructure [4][5]. It is narrower than trl or verl in method coverage - GRPO is the only method with dedicated documentation pages and quick-start code, SFT is documented for distillation and warmup, and PPO/GSPO exist only as internal flags with no docs-site walkthrough (see Methods below) [3][6]. The FAQ states the task should already succeed at least 30% of the time with an open model, needs a consistent quantifiable reward, and must be safely re-runnable many times, since GRPO trains on repeated rollouts of the same scenario [7].

**Methods it ships**: GRPO is the documented, primary method - the quick-start, client, and backend pages all build their examples around `backend.train(model, trajectory_groups, ...)` driving GRPO updates [1][3]. SFT is a second documented method (`train_sft_from_file`, `model.train_sft`) for distillation, style transfer, and warm-starting before RL - both backends support it through the same API, and a run can switch from SFT to RL because both phases update the same LoRA adapter [6]. RULER (`art.rewards.ruler_score_group`) is a reward-generation utility, not a training method: it uses an LLM-as-judge to rank trajectories relative to each other and feeds those scores into GRPO, and the docs' own launch-announcement figure caption reports it slightly outperforming hand-crafted rewards on three of four evaluated tasks [8]. Two more knobs live only in the internal `art.dev.TrainConfig` TypedDict read from source at commit `901a9e26`: `ppo: bool` and `importance_sampling_level: Literal["token","sequence","average","geometric_average"]` - the latter is GRPO's PPO-style clipped surrogate switched to a per-token vs. sequence-level importance ratio [9]. Setting `importance_sampling_level="sequence"` is ART's documented, explicitly experimental GSPO variant; the docs state the GSPO algorithm itself was introduced by the Qwen team to train models including Qwen3-235B-A22B-Instruct-2507, and warn that ART's own GSPO support has "the API and behavior may change in future releases" [10]. The `ppo` flag has no corresponding docs page, quick-start, or worked example anywhere in the fetched docs site or README - it is source-only.

**Scale it handles**: single GPU is the base case for both backends. `LocalBackend` runs vLLM plus Unsloth (or torchtune) on the same machine as the agent [3]; a `PipelineTrainer`-based dedicated mode splits training and inference onto separate GPUs via `InternalModelConfig(trainer_gpu_ids=[...], inference_gpu_ids=[...])`, and the docs state a shared `LocalBackend` "still pauses inference during training, so ART rejects that configuration for `PipelineTrainer`" [11]. `ServerlessBackend` instead sends inference and training requests to OpenPipe's autoscaling W&B Training cluster and needs no local GPU at all, at the cost of being restricted to two supported base models (see Find it in the docs) [4][12]. The source tree also carries an undocumented `src/art/megatron/` package (tensor/pipeline/expert/context-parallel `MegatronTopologyConfig`, streaming weight offload) and the PyPI `megatron` extra pins `megatron-core`, `megatron-bridge`, `transformer-engine`, and `apex` - read at commit `901a9e26`, this is a real, shipped multi-GPU sharding path with no page on the public docs site describing its setup or a published benchmark [13][14]. No documented page states a multi-node launch form.

**Install**: `pip install openpipe-art` for the client; `pip install openpipe-art[backend]` adds local training/inference dependencies; a bare `pip install openpipe-art` plus `ServerlessBackend` needs no extra [4]. Version 0.5.18, uploaded to PyPI 2026-05-23 [15]; there is no matching `v0.5.18` git tag (the newest tag is `v0.5.17`, from 2026-03-13) - the pinned pyproject.toml below was read at commit `ccf5b1bc6f41` ("Bump version to 0.5.18"), whose `pyproject.toml` content is byte-identical to PyPI's own published `requires_dist` for 0.5.18 [16][17]. Python floor `>=3.11`; license Apache-2.0, from the repository's LICENSE file [18][2]. Base package pins are unconstrained ranges (`openai>=2.14.0`, `litellm>=1.71.1,<=1.82.0`, `weave>=0.52.24`, `polars>=1.26.0`); the `backend` extra hard-pins the deep-learning core with `==`: `torch==2.10.0`, `transformers==5.2.0`, `trl==0.20.0`, `accelerate==1.7.0`, `unsloth==2026.3.3`, `unsloth-zoo==2026.3.1`, `wandb==0.25.0`, `torchao==0.16.0`, plus `peft>=0.14.0`, `bitsandbytes>=0.45.2`, and `gql<4` [16]. Neither the installation page nor the release's `pyproject.toml` states a CUDA or GPU-model minimum; the installation page says only that `LocalBackend` needs "any machine with a GPU" [4], and the backend page separately describes the case where an agent is "already set up on a machine equipped with an advanced GPU" as one reason to pick `LocalBackend`, again with no version or memory floor given [11]. Note the shortlist row's screening commit `901a9e2615455c1ce952f1935fa3f56ea5c0aecf` (2026-07-29) is itself ahead of this pinned release commit `ccf5b1bc6f41` (2026-05-22) - the screening commit reflects the repository's newest push, not a tagged release, and its own `pyproject.toml` already shows different pins (`torch==2.11.0`, `wandb==0.28.0`, `requires-python>=3.12`), so code read at the screening commit is not what `pip install openpipe-art` currently delivers [20][13].

**Maintained by**: OpenPipe, a GitHub Organization (not an individual) [21][2]. The repository shows 10,574 stars (not a ranking signal) and was last pushed 2026-08-08, five months after the last tagged GitHub Release (v0.5.17, 2026-03-13) and about two and a half months after the untagged 0.5.18 PyPI upload - both push activity and PyPI activity postdate the last formal release, evidence of ongoing but between-release development [22][15][23].

## Quick start

The client/backend split means there is no single-file CLI quick start; the docs' own quick start instead runs a hosted Colab notebook against `ServerlessBackend` [24]. The smallest complete runnable pattern is assembled from the installation and client pages [4][25]:

```python
import art
from art.local.backend import LocalBackend
# or: from art.serverless.backend import ServerlessBackend

backend = LocalBackend()

model = art.TrainableModel(
    name="agent-001",
    project="my-agentic-task",
    base_model="OpenPipe/Qwen3-14B-Instruct",
)
await model.register(backend)

async def rollout(model: art.Model, scenario) -> art.Trajectory:
    openai_client = model.openai_client()
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": "..."},
                               {"role": "user", "content": "..."}]
    )
    chat_completion = await openai_client.chat.completions.create(
        messages=trajectory.messages(), model=model.get_inference_name()
    )
    trajectory.messages_and_choices.append(chat_completion.choices[0])
    trajectory.reward = ...  # your scoring logic
    return trajectory

for _ in range(await model.get_step(), 50):
    train_groups = await art.gather_trajectory_groups(
        (art.TrajectoryGroup(rollout(model, scenario) for _ in range(8))
         for scenario in scenarios),
        pbar_desc="gather",
    )
    result = await backend.train(model, train_groups, learning_rate=1e-5)
    await model.log(train_groups, metrics=result.metrics, step=result.step, split="train")
```

This is the docs' own client-page pattern: 50 training steps, 8 rollout attempts per scenario per step [25]. There is no separate CLI entry point documented - training always runs as Python code that imports `art` [25][4].

## Start it

- One GPU, one process: instantiate `LocalBackend()` (needs the `backend` extra installed) or `ServerlessBackend()` (needs only a W&B API key, as an env var `WANDB_API_KEY` or an init argument) and call `model.register(backend)` [4][11].
- Dedicated-mode multi-GPU on one node: pass `_internal_config=InternalModelConfig(trainer_gpu_ids=[0], inference_gpu_ids=[1])` to `TrainableModel` when using `PipelineTrainer` with `LocalBackend`; this is the only documented multi-GPU launch form, and it is single-node only in the fetched docs [11]. There is no Accelerate/DeepSpeed/FSDP-style launcher script or config-template directory documented for ART itself.
- Effective batch size for a GRPO step is `len(scenarios) x num_generations_per_scenario` (8 in the quick-start pattern above) [25]; `TrainSFTConfig.batch_size` defaults to `"auto"` for `train_sft_from_file` and can be overridden per call [6][26].
- The public `TrainConfig` surface is small: `learning_rate` (default `5e-6`), `kl_penalty_coef` (default `0.0`), `kl_penalty_source` (`"current_learner"` or `"sample"`), `grad_accumulation_sequences`, `optimizer_save_interval` (default `5`), read from `src/art/types.py` at commit `901a9e26` [27]. A much larger internal-only `art.dev.TrainConfig` TypedDict adds clip-epsilon, advantage-balancing, importance-sampling, KL-reference-staleness, and MoE-routing-replay fields with no docs-site page describing them [9].
- Generation is not a separately launched engine in the documented flow - `LocalBackend` starts its own vLLM server internally and reloads each new LoRA into it after a training step, and inference is paused while a training step runs on a shared backend [1][11].
- Out-of-memory first aid: the fetched docs (quick-start, installation, backend, and tracking-metrics pages) publish no explicit OOM knob list or troubleshooting section for ART itself; the closest documented lever is `LocalBackend(gpu_cost_per_hour_usd=...)`, which only affects cost logging, not memory [28]. `TrainableModel` can be warm-started from an existing SFT LoRA adapter directory to shorten the RL run instead of running longer from scratch, which the docs frame partly as a way to "stabilize early training" for small models rather than as OOM relief [25].

## Watch it

This section is mechanics only; what a given metric shape means for GRPO or SFT training health belongs on those methods' own cards, not here.

- **Enable it**: every `model.log(...)` call writes a row to `history.jsonl` in the run directory; it additionally goes to W&B if W&B logging is set up, which the docs describe only as "if W&B logging is enabled" without naming the exact trigger check in the fetched pages [28]. `ServerlessBackend` requires a W&B API key outright, so its runs are always W&B-logged [4].
- **Metric names**, from the tracking-metrics page's own table [28]: reward group `train/reward`, `train/reward_std_dev`, `train/exception_rate`, `val/reward`; loss group `loss/train`, `loss/entropy`, `loss/kl_div`, `loss/grad_norm`, `loss/learning_rate`; data group `data/step_num_scenarios`, `data/step_num_trajectories`, `data/step_num_groups_submitted`, `data/step_num_groups_trainable`; time group `time/wall_clock_sec`, `time/step_wall_s`, `time/step_trainer_s`; and `costs/gpu` on `LocalBackend` when GPU pricing is known. When available it also derives cumulative fields (`time/cum/trainer_s`, `data/cum/num_unique_scenarios`, `costs/cum/all`), cost rollups (`costs/train`, `costs/eval`, `costs/all`), and throughput fields (`throughput/avg_trainer_tok_per_s`, `throughput/avg_actor_tok_per_s`) [28].
- **Task-specific and step-level metrics**: attach `trajectory.metrics["name"] = value` inside the rollout function to get it auto-averaged under `train/name`; use `model.metrics_builder()` with `.measure(...)` and `.add_data(...)` for actor-side timing or token counts ART cannot infer on its own [28].
- **Sample-level logging**: the fetched docs do not document a dedicated flag for logging sample completions/generations the way trl's `log_completions` does; W&B Weave tracing of completions is named as one of the four W&B services the quick start wires up, but the exact opt-in mechanism for that trace logging is not described on the pages fetched here [24].
- **Judge/API cost tracking**: `@track_api_cost(source=..., provider=..., model_name=...)` wraps a function returning a provider response with token usage, logged under `costs/train/<source>` or `costs/eval/<source>` depending on which `model.metrics_builder(...)` context is active; ART prices OpenAI and Anthropic responses automatically and needs `register_model_pricing(...)` for anything else [28].
- **Evaluation during training**: the fetched docs show no dedicated `eval_dataset`/`eval_steps` config surface analogous to trl's; `val/reward` appears as a logged metric name, and `builder.add_data(step_eval_s=...)` is the documented way to fold externally-run eval timing into the same metrics stream, implying evaluation is orchestrated by the caller's own code rather than a backend-managed eval loop [28].
- **Stopping**: no page fetched here (quick-start, client, backend, tracking-metrics, FAQ) publishes an early-stopping threshold, patience field, or reward-plateau rule; the training loop's stopping condition in every documented example is a fixed step count set by the caller (`for _ in range(await model.get_step(), 50)`) [25][6].

## Save it

- Checkpoints land under `LocalBackend`'s `path` argument (default `./.art`) as `{project}/models/{model_name}/checkpoints/{step:04d}/`, confirmed from `src/art/local/checkpoints.py` at commit `901a9e26`; that same file's `migrate_checkpoints_to_new_structure` function documents that this nested `checkpoints/` layout replaced an older flat `{project}/models/{model_name}/{step}` layout, so older on-disk runs may need migration [29]. On `ServerlessBackend`, checkpoints are stored as W&B Artifacts instead of local files [6].
- Every ART checkpoint is a LoRA adapter, not a full model - the checkpoint-deletion page states each is "80-150MB" (about 120MB in its own worked example, versus a 6GB unmanaged total over 50 steps) [30], and the SFT page states plainly that RL and SFT "train the same LoRA adapter," which is why a run can switch from SFT to RL without a separate export step [6].
- Retention: `model.delete_checkpoints()` keeps only the most recent checkpoint plus the best one by `val/reward` (or by any metric passed as `best_checkpoint_metric=`, e.g. `"train/reward"`), and permanently deletes the rest - the docs warn "once checkpoints are deleted, they generally cannot be recovered" [30]. There is no documented flag that saves weights-only vs. weights-plus-optimizer-state (unlike trl's `save_only_model`); the fetched pages give no contract for what optimizer/scheduler state a checkpoint keeps.
- Resume: the training-loop pattern in every fetched example resumes automatically by starting its step range at `await model.get_step()` rather than 0, so re-running the same script against an already-registered model continues from its last completed step [25][30].
- Forking (marked experimental): `backend._experimental_fork_checkpoint(model, from_model=..., not_after_step=..., from_s3_bucket=(optional), verbose=...)` copies another model's checkpoint - locally or pulled from S3 - as a new model's starting point, using `<=` semantics for `not_after_step`; the docs state "only checkpoint files are copied - training logs and trajectories are not included in the fork" [31].
- Loading the result: initializing a fresh `TrainableModel` with `base_model` pointed at an on-disk adapter directory (containing `adapter_config.json` and `adapter_model.bin`/`.safetensors`) resumes RL from those weights, per the client page's documented pattern for warm-starting from an SFT adapter [25][7]; this confirms an ART checkpoint directory is a standard PEFT-style adapter, not a merged full model, and an evaluator needs the same base model plus a PEFT-aware loader (e.g. `PeftModel.from_pretrained`) rather than a plain `from_pretrained` on the checkpoint directory alone - the fetched pages document no ART-side merge-and-unload helper.

## Find it in the docs

The docs site (art.openpipe.ai, built on Mintlify) is the live source; this section is the lookup, not a mirror.

- Address pattern: `https://art.openpipe.ai/<page>` for the rendered page, or `https://art.openpipe.ai/<page>.md` for its raw Markdown - verified 2026-08-10 by fetching `getting-started/quick-start.md` directly; the rendered HTML root otherwise returns an empty shell to a plain HTTP client because the site is JS-rendered [24]. There is no version-tag segment in the URL - this is a single unversioned live site, not per-release docs, so a docs claim here is pinned only to the fetch date given, not to any package version.
- The complete page index is published at `https://art.openpipe.ai/llms.txt`, listing all 22 pages by title, slug, and one-line description - fetch this first instead of guessing slugs [32].
- Question-to-slug map, from that index [32]: "how do I start training" -> `getting-started/quick-start` and `getting-started/installation-setup`; "what's the client API" -> `fundamentals/art-client`; "local vs. managed training" -> `fundamentals/art-backend`; "what metrics does it log" -> `features/tracking-metrics`; "how do I do SFT/distillation" -> `fundamentals/sft-training`; "what models are supported" -> `resources/models`; "reward function without labeled data" -> `fundamentals/ruler`; "delete old checkpoints" -> `features/checkpoint-deletion`; "restart from a checkpoint" -> `features/checkpoint-forking`; "GSPO" -> `experimental/gspo`; "terminology" -> `resources/glossary`.
- Runnable references beyond the docs: the quick start itself is a Colab notebook (2048-game agent trained with `ServerlessBackend`) linked from the docs and the GitHub README, plus a Summarizer tutorial page (`tutorials/summarizer`) and an Open Deep Research tutorial (`tutorials/open-deep-research`) [24][32]; the repository's `examples/` tree (e.g. `examples/hn_title_generator/reference_grpo_trainer.py`, the file the shortlist row's method detection matched on) holds further worked scripts [33].
- Community/integration layer: the docs curate two integration guides directly - LangGraph (`integrations/langgraph-integration`) for building agents with LangGraph's graph framework, and OpenEnv (`integrations/openenv-integration`), which the docs state works with ART "without any special adapters or configuration" because ART is unopinionated about environment shape [34][35]. No separate curated community-blog page was found among the 22 indexed pages, unlike trl's `community_tutorials` page - this docs site's own list is the only curated layer discovered.
- No official MCP endpoint for querying these docs was found on the pages fetched; none is named on the docs index or README.
- Traps found in maintainer-facing docs (none from third-party issue threads were fetched for this card): the README states plainly that Gemma 3 "does not appear to be supported for the time being" [36], and the Supported Models page separately states that a LoRA-incompatible model architecture "still may work with our full-fine-tuning backend" even when it cannot use ART's LoRA-based path [19]. Multi-turn use of the Qwen 3 family needs the `additional_histories` feature to work around that family's chat template stripping `<think>` tokens from prior turns [19].
- Honest boundary: `ServerlessBackend` supports exactly two base models today (`OpenPipe/Qwen3-14B-Instruct` and `Qwen/Qwen3-30B-A3B`), with the docs stating more are being added on request; `LocalBackend` is far broader (most vLLM/HF-transformers-compatible causal LMs, anything Unsloth supports) [19]. The FAQ states ART is built on GRPO, which needs many parallel rollouts of the same scenario, so it is an explicit poor fit for agents whose actions are not safely repeatable in the real world [7].

## Sources

All docs-site pages were fetched 2026-08-10 from the live, unversioned art.openpipe.ai site (no version-tag URL segment exists for this docs site, so these citations are pinned only to that fetch date, not to a package release). GitHub source citations carry the commit read; PyPI/release citations carry the version and upload date. Ecosystem tools named only in passing (vLLM, Unsloth, torchtune, trl, W&B/Weave/Artifacts, LangGraph, OpenEnv) are reached through the cited pages' own links and are not separately enumerated. GRPO, PPO, SFT, and GSPO are deliberately cited to nothing here: their defining papers belong on the methodology cards.

[1] ART docs index ("ART Docs" / about page). https://art.openpipe.ai/getting-started/about. Fetched 2026-08-10.

[2] OpenPipe/ART GitHub repository (metadata via API: owner, license, description, URL). https://github.com/OpenPipe/ART. Repository API response fetched 2026-08-10.

[3] ART Training Loop page. https://art.openpipe.ai/fundamentals/training-loop. Fetched 2026-08-10.

[4] ART Installation + Setup page. https://art.openpipe.ai/getting-started/installation-setup. Fetched 2026-08-10.

[5] ART README, "W&B Training: Serverless RL" section. https://raw.githubusercontent.com/OpenPipe/ART/main/README.md. Fetched 2026-08-10.

[6] ART SFT Training page. https://art.openpipe.ai/fundamentals/sft-training. Fetched 2026-08-10.

[7] ART FAQ page. https://art.openpipe.ai/getting-started/faq. Fetched 2026-08-10.

[8] ART RULER page. https://art.openpipe.ai/fundamentals/ruler. Fetched 2026-08-10.

[9] `art.dev.train.TrainConfig` / `TrainSFTConfig` source, commit 901a9e2615455c1ce952f1935fa3f56ea5c0aecf. https://raw.githubusercontent.com/OpenPipe/ART/901a9e2615455c1ce952f1935fa3f56ea5c0aecf/src/art/dev/train.py. Fetched 2026-08-10.

[10] ART GSPO (experimental) page. https://art.openpipe.ai/experimental/gspo. Fetched 2026-08-10.

[11] ART Backend page (`LocalBackend`/`ServerlessBackend`, `PipelineTrainer` dedicated mode). https://art.openpipe.ai/fundamentals/art-backend. Fetched 2026-08-10.

[12] ART Supported Models page (`ServerlessBackend` model restriction). https://art.openpipe.ai/resources/models. Fetched 2026-08-10. (Same source as [19]; cited separately here for the scale claim.)

[13] `pyproject.toml` at the `main` branch HEAD (ahead of the pinned 0.5.18 release; shows the `megatron` extra and current, different pins). https://raw.githubusercontent.com/OpenPipe/ART/main/pyproject.toml. Fetched 2026-08-10.

[14] `openpipe-art` PyPI JSON API, `megatron` extra's `requires_dist` entries for version 0.5.18. https://pypi.org/pypi/openpipe-art/json. Fetched 2026-08-10.

[15] `openpipe-art` PyPI JSON API, version and upload_time for 0.5.18. https://pypi.org/pypi/openpipe-art/json. Fetched 2026-08-10.

[16] `pyproject.toml` at commit ccf5b1bc6f41 ("Bump version to 0.5.18 (#694)"), the commit that set the released version's dependency pins. https://raw.githubusercontent.com/OpenPipe/ART/ccf5b1bc6f41/pyproject.toml. Fetched 2026-08-10.

[17] GitHub commit-history search on `pyproject.toml`, used to locate commit ccf5b1bc6f41 and confirm no `v0.5.18` git tag exists (`tags` API, 100 entries, highest is `v0.5.17`). https://api.github.com/repos/OpenPipe/ART/commits?path=pyproject.toml and https://api.github.com/repos/OpenPipe/ART/tags?per_page=100. Fetched 2026-08-10.

[18] ART README, License section (Apache-2.0, referencing the repository's LICENSE file). https://raw.githubusercontent.com/OpenPipe/ART/main/README.md. Fetched 2026-08-10.

[19] ART Supported Models page. https://art.openpipe.ai/resources/models. Fetched 2026-08-10.

[20] `pyproject.toml` at the shortlist row's screening commit 901a9e2615455c1ce952f1935fa3f56ea5c0aecf, i.e. the repository's newest push, not a tagged release; commit metadata (author, date, message "Add Llama 3 Megatron model support (#783)") confirmed via the GitHub commits API. https://raw.githubusercontent.com/OpenPipe/ART/901a9e2615455c1ce952f1935fa3f56ea5c0aecf/pyproject.toml and https://api.github.com/repos/OpenPipe/ART/commits/901a9e2615455c1ce952f1935fa3f56ea5c0aecf. Fetched 2026-08-10.

[21] ART README, Citation section (author list, OpenPipe attribution). https://raw.githubusercontent.com/OpenPipe/ART/main/README.md. Fetched 2026-08-10.

[22] GitHub repository API (`stargazers_count`, `pushed_at`, `created_at`). https://api.github.com/repos/OpenPipe/ART. Fetched 2026-08-10.

[23] GitHub Releases API, latest tagged release. https://api.github.com/repos/OpenPipe/ART/releases/latest. Fetched 2026-08-10.

[24] ART Quick Start page. https://art.openpipe.ai/getting-started/quick-start. Fetched 2026-08-10.

[25] ART Client page. https://art.openpipe.ai/fundamentals/art-client. Fetched 2026-08-10.

[26] `art.types.TrainSFTConfig` source, commit 901a9e2615455c1ce952f1935fa3f56ea5c0aecf. https://raw.githubusercontent.com/OpenPipe/ART/901a9e2615455c1ce952f1935fa3f56ea5c0aecf/src/art/types.py. Fetched 2026-08-10.

[27] `art.types.TrainConfig` source (public config defaults), commit 901a9e2615455c1ce952f1935fa3f56ea5c0aecf. https://raw.githubusercontent.com/OpenPipe/ART/901a9e2615455c1ce952f1935fa3f56ea5c0aecf/src/art/types.py. Fetched 2026-08-10.

[28] ART Tracking Metrics page. https://art.openpipe.ai/features/tracking-metrics. Fetched 2026-08-10.

[29] `src/art/local/checkpoints.py` source (checkpoint directory layout and the old-structure migration function), commit 901a9e2615455c1ce952f1935fa3f56ea5c0aecf. https://raw.githubusercontent.com/OpenPipe/ART/901a9e2615455c1ce952f1935fa3f56ea5c0aecf/src/art/local/checkpoints.py. Fetched 2026-08-10.

[30] ART Deleting Checkpoints page. https://art.openpipe.ai/features/checkpoint-deletion. Fetched 2026-08-10.

[31] ART Checkpoint Forking page (experimental feature). https://art.openpipe.ai/features/checkpoint-forking. Fetched 2026-08-10.

[32] ART docs page index. https://art.openpipe.ai/llms.txt. Fetched 2026-08-10.

[33] `examples/hn_title_generator/reference_grpo_trainer.py`, the file matched by the shortlist row's method detection. https://github.com/OpenPipe/ART/blob/901a9e2615455c1ce952f1935fa3f56ea5c0aecf/examples/hn_title_generator/reference_grpo_trainer.py (existence confirmed via the repository tree API at that commit; file contents not separately fetched for this card). Fetched 2026-08-10.

[34] ART LangGraph integration page. https://art.openpipe.ai/integrations/langgraph-integration. Fetched 2026-08-10.

[35] ART OpenEnv integration page. https://art.openpipe.ai/integrations/openenv-integration. Fetched 2026-08-10.

[36] ART README, Supported Models section (Gemma 3 note). https://raw.githubusercontent.com/OpenPipe/ART/main/README.md. Fetched 2026-08-10.
