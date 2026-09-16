# rllm-org/rllm (rLLM)

A CLI-first agent-training framework: bring an existing coding-agent harness, wrap it with a rollout decorator, and switch between a distributed verl backend and a single-machine tinker backend with one flag.

rLLM is described in its own README as an open-source framework for training language agents with reinforcement learning, letting a user "bring any harness, run it in any sandbox, and switch training backends with one flag" [1]. It is built and maintained by the rLLM Team under the `rllm-org` GitHub organization [1][2], and it packages three layers behind one CLI and one Python decorator API: an agent-harness/sandbox layer (`rllm eval`), a reward/evaluator layer (`evaluate(task, episode)` functions), and a training layer built on a `UnifiedTrainer` that dispatches to a chosen backend (`AgentTrainer(..., backend="verl"|"tinker"|"fireworks")`) [1][3][4]. It lives at https://github.com/rllm-org/rllm [2]. Note that a same-named, separately maintained repository `agentica-project/rllm` also exists; this card covers only `rllm-org/rllm`, the claimant this pipeline selected — stars are not used for that selection and are not repeated here as a ranking signal.

**When to pick it**: pick rLLM when the unit of RL is an existing CLI coding/terminal-agent harness (Claude Code, Codex, Terminus-2, mini-swe-agent, opencode, or a Harbor-compatible task directory) or a custom agent wrapped with `@rllm.rollout`, and you want one training script that can run on a single machine via the tinker backend or scale out to multi-GPU/multi-node via the verl backend without rewriting the agent [1][5]. Weigh this against trl (Hugging Face's per-method trainer-class library, no built-in CLI-harness/sandbox layer) and against verl itself (rLLM's own distributed backend, used directly rather than through rLLM's harness/reward wrapping) — both are cross-references, not covered here.

**Methods it ships**: the README lists GRPO, REINFORCE, RLOO, SFT, and on-policy distillation as supported training methods [1]. The shortlist row's own `methods_seen` field carries only one path genuinely rooted in this repository — REWARD at `rllm/rewards/reward_fn.py` — and the row is otherwise not a clean picture of this repository's own layout: its PPO entry points at `rllm/trainer/verl/agent_ppo_trainer.py` inside the separate `agentica-project/rllm` repository, and its SFT entry points at `rllm/rllm/trainer/agent_sft_trainer.py` inside `Osilly/Vision-DeepResearch`, a third, unrelated repository; neither of those two paths was found inside `rllm-org/rllm` at commit 75926c1 (the PPO-relevant file that repository actually has at that path is `rllm/trainer/verl/train_agent_ppo.py`). This card reports the row's literal values without asserting they describe `rllm-org/rllm`'s own code. On the unified verl backend, only four advantage estimators ship out of the box — `grpo`, `reinforce`, `reinforce_plus_plus_baseline`, `rloo` — and verl's other estimators (`gae`, `remax`, `gpg`, and others) are "not available out of the box", with the corresponding `algorithm.gamma`/`algorithm.lam` knobs stated as "no longer wired through" on this path; a custom estimator can be registered instead [6]. The tinker backend adds a `distill` estimator for knowledge distillation from a teacher model [7]. Reward functions follow one plain-function convention across both eval and RL — a module exporting `evaluate(task, episode) -> EvalOutput` — so the same grader powers benchmark scoring and RL reward with no separate code path [8]. Method definitions and math are not restated here; they belong on each method's own card.

**Scale it handles**: single GPU up to multi-node, through two backends with materially different reach. The verl backend runs on Ray-managed worker groups (actor/critic/rollout/trainer roles) and documents multi-node scaling via `ray.init(address="auto")` on the head node and `ray.init(address="ray://<head-ip>:10001")` on worker nodes, plus a Hydra config tree (`rllm/trainer/config/agent_ppo_trainer.yaml`) for batch-size and parallelism overrides [9]; no published multi-node throughput benchmark accompanies this — only the mechanism. The tinker backend is single-machine and LoRA-native by default (LoRA rank 32, configurable attention/MLP/embedding training flags), delegating actual training compute to the remote Tinker service rather than local multi-GPU sharding [7]. A third backend, fireworks, delegates similarly to a managed Fireworks AI deployment and requires Python >= 3.11 [10]. Two documentation pages disagree on how fireworks is reached: the fireworks backend page shows it driven through the same exported `AgentTrainer` used by verl and tinker (`from rllm.trainer import AgentTrainer`, `backend="fireworks"`) [10], while the trainer capability-matrix page states plainly that "fireworks exists only on the legacy class" (`rllm.trainer.agent_trainer.AgentTrainer`, not the unified one imported by the CLI and launchers) and that the unified trainer "silently no-ops on an unrecognized backend" [4] — a reader should verify which `AgentTrainer` a fireworks example actually imports before relying on either page.

**Install**: `uv pip install "rllm[verl] @ git+https://github.com/rllm-org/rllm.git"` (or `rllm[tinker]`, `rllm[fireworks]`) — rLLM is not published on PyPI; the PyPI JSON API for the name `rllm` returns "Not Found" [11], so a git install is the only path. The shortlist row's pinned commit `75926c15e58fa29e4183d01292d10462d2047be9` (2026-07-31) is the repository's newest push, not a release [12]; the actual latest GitHub release is `v0.3.0-pre`, tagged at commit `0956764...` and published 2026-04-30 — about three months earlier [13]. Reading the release's own `pyproject.toml` at that tag: `requires-python = ">=3.10"`, with the installation docs stating `>=3.11` is required only for the tinker backend [14][5] — but the README, at both the release tag and HEAD, states flatly that "rLLM requires `Python >= 3.11`" with no verl/tinker carve-out, directly conflicting with the pyproject/installation-docs floor this card otherwise reports [1]; a reader on Python 3.10 installing only the verl extra should expect this contradiction and verify against their own environment rather than trust either page alone. The licence is Apache-2.0 (the LICENSE file at that tag is genuine Apache License 2.0 text, though the same `pyproject.toml`'s classifier list mis-labels it "MIT License" — a documented-vs-actual mismatch worth flagging, not resolving) [14], and the `verl` extra pins `verl==0.7.1`, `vllm==0.17.0`, `flash-attn==2.8.1`, `torch>=2.10.0`, `transformers>=4.55.0,<5.0.0` (a hard upper-bound cap at `<5.0.0`), `ray` (unpinned) [14]. The release's `pyproject.toml` carries no `fireworks` extra at all, so the `>=3.11` Python floor the fireworks backend page states is not reachable from this release — it is a HEAD-only, HEAD-documented requirement [10][15]. At the row's own HEAD commit (75926c1, ahead of the release), the `verl` extra has moved to `verl==0.8.0`, `vllm==0.22.1`, `flash-attn==2.8.3`, torch floor unchanged, `transformers` has moved to `>=5.5.3` with the `<5.0.0` upper cap dropped entirely — a full major-version floor increase and the removal of the release's hard cap, the single largest collision-relevant pin change between the two — and a `fireworks` extra now exists (with its own `>=3.11` Python requirement) that the v0.3.0-pre release does not have at all [15][10] — any claim sourced from HEAD is explicitly ahead of what `pip install` from the release tag would deliver. No CUDA or hardware floor is stated on the installation page or in either pyproject.toml; the page only says to verify CUDA/PyTorch compatibility yourself [5].

**Maintained by**: the rLLM Team, GitHub org `rllm-org` [1][2]; the repository was pushed to as recently as 2026-08-09 and is not archived [2], and the docs site's own community-projects list under `projects/` (DeepScaleR, DeepCoder, DeepSWE and others) is presented as an ongoing line of published results rather than a single snapshot [16].

## Quick start

Smallest complete forms, from the README and CLI docs:

CLI, no code:
```bash
rllm model setup
rllm eval gsm8k
rllm train gsm8k
```
[17]

Python decorator API — a rollout plus an evaluator handed to the CLI/trainer:
```python
# my_flow.py
from openai import OpenAI
import rllm
from rllm.types import AgentConfig, Episode, Task, Trajectory

@rllm.rollout
def solve(task: Task, config: AgentConfig) -> Episode:
    client = OpenAI(base_url=config.base_url, api_key="EMPTY")
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": task.instruction}],
    )
    answer = response.choices[0].message.content or ""
    return Episode(
        trajectories=[Trajectory(name="solver", steps=[])],
        artifacts={"answer": answer},
    )
```
[1]

```python
# my_evaluator.py
import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode

@rllm.evaluator
def score(task: dict, episode: Episode) -> EvalOutput:
    answer = str(episode.artifacts.get("answer", ""))
```
[1] (README quickstart snippet is truncated at this point in the file as read).

A full Hydra-driven training script, from the distributed-training guide, mirroring `cookbooks/math/train.py` [9]:
```python
import hydra
from omegaconf import DictConfig

from math_flow import math_flow
from math_eval import math_evaluator
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("gsm8k", "train")
    test_dataset = DatasetRegistry.load_dataset("gsm8k", "test")
    trainer = AgentTrainer(
        backend="verl",
        agent_flow=math_flow,
        evaluator=math_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

## Start it

- One machine, no GPU sharding: the tinker backend is the CLI default — `rllm train gsm8k --model Qwen/Qwen3-8B` starts "the unified training pipeline with the tinker backend, using GRPO for advantage computation and LoRA for efficient fine-tuning", tunable with `--batch-size`, `--group-size`, `--lr`, `--max-steps` [17].
- Multi-GPU/multi-node goes through the verl backend's Ray cluster: `ray.init(address="auto")` on the head node, `ray.init(address="ray://<head-ip>:10001")` on worker nodes, then the same training script; Ray distributes workers automatically per the docs, with no published multi-node benchmark accompanying that claim [9]. The Ray dashboard defaults to `http://localhost:8265` when started with `ray start --head --dashboard-host=0.0.0.0` [9].
- Config surface is Hydra, layered from `agent_ppo_trainer.yaml` (verl) or the equivalent tinker/unified YAML, overridable as a Python dict, a `key=value` string list, or CLI dotted overrides (`data.train_batch_size=256 rllm.workflow.n_parallel_tasks=512`) [9][18]. Two config fields document real self-contradictions rather than a single default, and a hand-built config should set both explicitly instead of relying on the shipped value: `trainer.val_before_train` — the configuration reference's own `base.yaml` table lists default `True` [18], while the capability-matrix page states "the code falls back to `True` if the key is absent, but the shipped `base.yaml` sets `false`" — on a normal CLI/Hydra run the yaml's `false` wins [19]; and `algorithm.norm_adv_by_std_in_grpo` — the shared `base.yaml` table lists default `True` [18], while the tinker backend's own `ParamField` for the same key lists default `false` [7]. A third drift to note without resolving: `verl` itself is cited as three different pinned versions across the docs and source — `v0.6.1` in the installation page's editable-Megatron-install snippet and in the verl backend page's own quoted dependency block, `verl==0.7.1` in the v0.3.0-pre release's `pyproject.toml`, and `verl==0.8.0` in the HEAD `pyproject.toml` [5][6][14][15]. A fourth trap sits in the tinker backend's own worked full-config example: it sets `lam: 0.95` under `algorithm`, but neither the tinker backend page nor the shared configuration reference documents an `algorithm.lam` field anywhere — the only GAE-lambda-adjacent field either page documents is `algorithm.gamma` (default `1.0`) [7][18]. A reader copying that example gets a real, undocumented, non-default value with no `ParamField` entry to explain what it does or whether it is honored on this backend.
- rLLM changes no documented precision default from its own base config on the pages read for this card (no bf16/fp16 flag surfaced in the ParamField tables read); GPU memory is governed by verl's own FSDP/vLLM knobs, which the verl backend page's Performance Tips section calls out directly: use `rollout.mode=async`, tune `train_batch_size` for GPU utilization, "Enable FSDP" for models > 7B, and tune vLLM tensor-parallel size and max tokens [6].
- Tinker's own OOM/first-aid path is server-side: because generation and training run on the remote Tinker service rather than local GPUs, there is no local generation-engine memory knob equivalent to vLLM's `gpu_memory_utilization` documented on the tinker backend page read for this card [7].

## Watch it

The mechanics only — what a metric means for a given method lives on that method's card, not here.

- **Enable it**: set `trainer.logger` to a list of backends; the verl backend page's example is `logger: ["console", "wandb", "tensorboard"]` alongside `project_name`/`experiment_name` [6]. Adding `"ui"` to that list streams runs to rLLM's own hosted or self-hosted dashboard via a `UILogger` backend registered in `rllm/utils/tracking.py` [20].
- **verl-backend metric names**, the verl page's own "Key Metrics" list: `actor/entropy` (policy entropy), `actor/loss` (actor policy loss), `actor/ppo_ratio_mean` (PPO clipping ratio), `critic/full-score/mean` (average trajectory reward), `val/test_score/*` (validation accuracy by data source), `training/global_step` [6]. No equivalent named metric list was found on the tinker backend page read for this card — its "Monitoring" content covers only logger configuration, not metric names [7].
- **Sample-level logging**: the base config exposes `trainer.log_episodes` (off by default) and `trainer.episode_log_dir` for local episode logs [9]. rLLM's own hosted/self-hosted UI (`rllm-org/rllm-ui`) goes further — described as "Think of wandb dedicated to rLLM, with powerful features such as episode/trajectory search, observability AI agent and more" — and stores sessions, metrics, episodes, trajectories, and logs in a database (SQLite by default, or Postgres) so individual generations are searchable after the run, not just scalar curves [20].
- **Evaluate during training**: `trainer.test_freq` (default `5`) and `trainer.val_before_train` control validation cadence, and `trainer.val_only` runs validation with no training step, per the shared configuration reference's `base.yaml` table — but see the `val_before_train` drift noted in Start it [18][19].
- **Stopping**: none of the pages read for this card (the verl backend page, the tinker backend page, and the shared configuration reference) publish an RL-specific early-stopping threshold or patience value; the search covered the verl "Key Configuration Options" and "Performance Tips" sections, the tinker "Configuration" and "Troubleshooting" sections, and the full trainer/algorithm tables on the configuration reference page, and found only cadence fields (`test_freq`, `save_freq`) and dead/no-op flags noted in the gotchas below — no stopping rule.
- **A documented trap**: the capability-matrix page's "Gotchas worth memorizing" section states that `kl_beta`, `eps_clip`/`eps_clip_high`, `loss_agg_mode`, `rollout_correction`/TIS, and `mask_truncated_samples` "are honored on verl and silently ignored on tinker" despite living in backend-agnostic config where they "look settable everywhere" [4]. The same section states `filter_token_mismatch` ships `True` in `base.yaml` but "no rLLM-side Python reads it" on either unified backend — a dead knob [4].

## Save it

- **verl backend**: checkpoints save automatically under `{trainer.default_local_dir}/checkpoints/` at a frequency controlled by `trainer.save_freq` (documented default `100`, described as "steps" in the ParamField table on this page) [6], and training resumes from the latest checkpoint automatically if one is present; a specific checkpoint is loaded manually by passing `trainer.default_local_dir=/path/to/checkpoint/dir` [6]. The shared configuration reference's own `base.yaml` table gives `save_freq` a different default (`20`) and describes the unit as "epochs" rather than steps — a genuine cross-page disagreement on both the number and its unit, not just wording, and a reader should check which config file is actually loaded rather than trust either page's prose [18].
- **tinker backend**: checkpoints save under `trainer.default_local_dir` (default `/tmp/rllm-tinker-checkpoints`) at `trainer.save_freq` steps, but the actual weights live on the remote Tinker service — resuming uses a Tinker-hosted identifier, `trainer.resume_from_tinker_id=tinker://uuid/weights/000060`, rather than a local file path [7].
- **LoRA and what a checkpoint IS**: the tinker backend enables LoRA by default (rank 32); its docs warn to set `model.train_unembed=false` "for Fireworks AI compatibility when deploying LoRA adapters" [7], which implies a tinker-trained checkpoint is an adapter, not a merged full model, when LoRA is on — but neither the tinker backend page nor the checkpointing section read for this card states the on-disk adapter file layout or gives a merge/reload call; a reader relying on this card alone should treat the tinker checkpoint format as unconfirmed beyond "an identifier on the Tinker service" and verify against the live checkpointing section before depending on it.
- **Whether an evaluator can load what you saved is the loader's contract, not rLLM's** — this card does not resolve it; check the loading contract before the first save, especially for the tinker LoRA case above.

## Find it in the docs

The docs are the live Mintlify source at https://docs.rllm-project.com/; this section teaches the lookup, not the content.

- Address pattern: `https://docs.rllm-project.com/<slug>`, fetchable as raw Markdown by appending `.md` (e.g. `https://docs.rllm-project.com/installation.md`) — confirmed working for every page fetched for this card [5][6][7][9][17][18]. The site publishes no version-tag path segment among the pages checked; it reads as an always-latest ("main") docs build, so a page's stated default can drift from what an older release actually ships (see the `verl==` and `val_before_train` drifts above).
- A machine-readable page index lives at `https://docs.rllm-project.com/llms.txt`, listing every doc slug with a one-line description — the fastest way to find a page without guessing a slug [21].
- Page-slug recipes: each training backend is `backends/<name>` (`backends/verl`, `backends/tinker`, `backends/fireworks`), compared side by side at `backends/comparison`; the config system is `training/configuration`, and the per-dimension capability warnings live at `training/capability-matrix`; reward-function conventions are at `datasets/reward-functions`; the CLI reference is `core-concepts/cli-and-ui`; the hosted/self-hosted dashboard is `experimental/ui`.
- Question-to-slug map: "which advantage estimators exist" -> `backends/verl.md`'s "Supported advantage estimators" section [6]; "why did my config knob do nothing on tinker" -> `training/capability-matrix.md`'s "Gotchas worth memorizing" accordion [4]; "how do I add my own benchmark" -> `datasets/byo-dataset` (listed in the index but not read for this card) [21]; "how do I write a reward function" -> `datasets/reward-functions.md` [8].
- Runnable references beyond the docs: the `cookbooks/` tree in the repository (e.g. `cookbooks/math/train.py`, `cookbooks/math/train_verl.sh`) is the source the distributed-training guide itself quotes from [9]; `examples/countdown/unified_trainer/` is the fireworks/unified-trainer worked example [10]. Known-good smoke-test datasets are surfaced through `rllm dataset list --all` and `rllm dataset inspect gsm8k -n 3` in the CLI quickstart, with `gsm8k` used as the walkthrough benchmark throughout the docs [17].
- Community layer: the docs site curates a dedicated "Community Projects" set of pages under `projects/` (DeepScaleR, DeepCoder, DeepSWE, and others, per the site's own page index) [21] — this is the curated door; separately, a `cookbooks/` set (Math, Deepcoder, FinQA, FrozenLake, Geo3K and others) holds runnable end-to-end examples rather than write-ups of published results, and is not part of the `projects/` set despite the similar subject matter [21]. No additional third-party blog layer was identified in the pages read for this card, so none is cited here beyond that official curation.
- No official MCP endpoint for the docs was found among the pages read for this card.
- **Documented cross-repository inconsistency**: the tinker backend page's "Basic Usage" links its worked cookbook example to a path under the sibling, separately-maintained `agentica-project/rllm` repository rather than under `rllm-org/rllm` itself [7] — worth a second look before following that link, since it points outside the repository this card covers.

## Sources

Every statement above is a 2026-08-10 reading of the docs pages and repository state listed below, except where a specific commit or release tag is named. Method names (GRPO, REINFORCE, RLOO, SFT) are deliberately cited to nothing here — their defining papers live on the methodology cards.

[1] rllm-org/rllm README at HEAD commit 75926c15e58fa29e4183d01292d10462d2047be9. https://raw.githubusercontent.com/rllm-org/rllm/75926c15e58fa29e4183d01292d10462d2047be9/README.md. Fetched 2026-08-10.

[2] rllm-org/rllm GitHub repository (API metadata: default branch, pushed_at, license, archived, topics). https://api.github.com/repos/rllm-org/rllm. Fetched 2026-08-10.

[3] rLLM documentation index / llms.txt page list. https://docs.rllm-project.com/llms.txt. Fetched 2026-08-10.

[4] rLLM trainer capability matrix. https://docs.rllm-project.com/training/capability-matrix.md. Fetched 2026-08-10.

[5] rLLM installation guide. https://docs.rllm-project.com/installation.md. Fetched 2026-08-10.

[6] rLLM verl backend guide (advantage estimators, ParamFields, checkpointing, Key Metrics, Performance Tips). https://docs.rllm-project.com/backends/verl.md. Fetched 2026-08-10.

[7] rLLM tinker backend guide (ParamFields, LoRA, checkpointing, example config with undocumented `lam: 0.95`). https://docs.rllm-project.com/backends/tinker.md. Fetched 2026-08-10.

[8] rLLM reward-functions guide (`evaluate(task, episode)` convention). https://docs.rllm-project.com/datasets/reward-functions.md. Fetched 2026-08-10.

[9] rLLM distributed-training guide (Ray multi-node, Hydra config overrides, Ray dashboard). https://docs.rllm-project.com/guides/distributed-training.md. Fetched 2026-08-10.

[10] rLLM fireworks backend guide (unified `AgentTrainer` example with `backend="fireworks"`). https://docs.rllm-project.com/backends/fireworks.md. Fetched 2026-08-10.

[11] PyPI JSON API lookup for package name `rllm`. https://pypi.org/pypi/rllm/json. Fetched 2026-08-10 (returned `{"message": "Not Found"}`).

[12] GitHub commit lookup for 75926c15e58fa29e4183d01292d10462d2047be9. https://api.github.com/repos/rllm-org/rllm/commits/75926c15e58fa29e4183d01292d10462d2047be9. Fetched 2026-08-10.

[13] rllm-org/rllm GitHub releases list (v0.3.0-pre published 2026-04-30, tag commit 0956764...). https://api.github.com/repos/rllm-org/rllm/releases and https://api.github.com/repos/rllm-org/rllm/tags. Fetched 2026-08-10.

[14] rllm-org/rllm pyproject.toml at the v0.3.0-pre release tag (Python floor, licence classifier, verl/vllm/flash-attn/torch pins). https://raw.githubusercontent.com/rllm-org/rllm/v0.3.0-pre/pyproject.toml. Fetched 2026-08-10.

[15] rllm-org/rllm pyproject.toml at HEAD commit 75926c15e58fa29e4183d01292d10462d2047be9 (updated verl/vllm/flash-attn pins, added `fireworks` extra). https://raw.githubusercontent.com/rllm-org/rllm/75926c15e58fa29e4183d01292d10462d2047be9/pyproject.toml. Fetched 2026-08-10.

[16] rLLM community-projects pages (DeepScaleR, DeepCoder, DeepSWE and others, listed under `projects/` in the docs index). https://docs.rllm-project.com/projects/deep-scaler.md and sibling `projects/*` pages listed in [3]. Fetched 2026-08-10.

[17] rLLM CLI quickstart. https://docs.rllm-project.com/quickstart-cli.md. Fetched 2026-08-10.

[18] rLLM unified configuration reference (`base.yaml` field tables, bidirectional config-sync table). https://docs.rllm-project.com/training/configuration.md. Fetched 2026-08-10.

[19] rLLM trainer capability matrix, "val_before_train default disagrees with itself" gotcha. https://docs.rllm-project.com/training/capability-matrix.md. Fetched 2026-08-10 (same page as [4]; cited separately for a specific accordion entry).

[20] rLLM UI guide (`UILogger`, `rllm-org/rllm-ui`, database and dashboard description). https://docs.rllm-project.com/experimental/ui.md. Fetched 2026-08-10.

[21] rLLM documentation index (page-slug map, community-projects listing). https://docs.rllm-project.com/llms.txt. Fetched 2026-08-10 (same page as [3]; cited separately for the navigation and community-curation claims).
