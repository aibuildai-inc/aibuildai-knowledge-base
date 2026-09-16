# UniRL

Tencent-Hunyuan's Ray/Hydra RL post-training framework for unified multimodal models — one trainer loop, pluggable rollout engines, run through flat recipe YAMLs.

UniRL "applies one RL post-training loop — generate samples, score them, compute advantages, update the policy, and sync weights back to rollout workers — across multimodal model families" [1]. It is built and maintained by Tencent-Hunyuan [2][1], with 24 listed authors led by Haonan Wang in the repo's citation block [1]. Each entrypoint (`train_diffusion`, `train_ar`, `train_pe`, `train_unified_model`, plus agentic variants) loads a self-contained Hydra recipe naming a model, algorithm, rollout engine, placement, reward, and weight-sync setup by `_target_` dotpath, then hands off to a matching `<Domain>Trainer` that runs the rollout-reward-advantage-train loop over a Ray `DevicePool` [1][3]. The repository lives at https://github.com/Tencent-Hunyuan/UniRL [2].

**When to pick it**: multimodal RL post-training — diffusion/video, vision-language and text-only AR, a prompt-enhancer pipeline, and unified AR+diffusion models (HunyuanImage3, Bagel) — under one Hydra-recipe and Ray-worker runtime, and specifically when you want the team's own trust-region algorithms (Flow-DPPO, DRPO, CPPO) with their own tutorials [1]. This card does not compare its scale or throughput against sibling frameworks (trl, verl); no such measurement was found in the sources read.

**Methods it ships**: three team-proposed algorithms, each with its own tutorial folder and runnable recipe — Flow-DPPO (`FlowDPPO/`), DRPO (`DRPO/`), and CPPO (`CPPO/`) — plus "standard reference algorithms": (LLM's)GRPO, DiffusionNFT, DanceGRPO, and MixGRPO, the latter two documented as FlowGRPO recipe variants (a different SDE strategy or index scheduler) rather than separate classes [1][4]. The algorithms-module README additionally documents FlowDPPO, DRPO, and GRPO/FlowGRPO as `StageAlgorithm` implementations living in `unirl/algorithms/` [4], and the module's file tree at the pinned commit also contains `gspo.py`, `dppo.py`, `cppo.py`, and `bagel_flow_unigrpo.py`, none of which is named in the top-level README's algorithm tables or the algorithms README's narrative sections [4]. `gspo.py`'s own header names GSPO (arXiv:2507.18071) and states its implementation is independent, not a port of external code; `dppo.py`'s header names DPPO (arXiv:2602.04879) and describes it as the AR-only trust region that CPPO and DRPO extend (source read at commit 683a59b) [5][6]. Every algorithm is a `StageAlgorithm`: a loss combine over `stage.replay`'s `new_logp`, a frozen pi_old anchor, and advantages, plus two declared knobs, `requires_ema_rollout` and `supports_multi_update`, that reconfigure the sampler and train stack around it [4]. SFT is a separate, non-RL algorithm file at `unirl/algorithms/sft.py` [7], reached through its own `train_sft` entrypoint [8].

**Scale it handles**: single GPU up to multi-node, launched through bash wrappers (`examples/run_experiment_single_node.sh`, `examples/run_experiment_multinode_taiji.sh`) or a bare `python -m unirl.train_<domain> --config-name=... num_devices=8` invocation, all placed by a Ray `DevicePool` [8]. Three deployment modes trade off GPU layout for the rollout engine: train-side sampling (training workers generate directly, no weight sync), separate rollout (disjoint GPU pools, sync required), and colocated rollout (shared GPU bundles with offload/onload, sync required) [3]. Multi-node requires `save_dir`/`load_dir` and model/data paths to sit on storage mounted on every node, and a mooncake-backed recipe variant needs its metadata server started first [9][8]; no published multi-node throughput benchmark was found in the sources read — the multi-node launcher and storage contract are documented mechanism, not a measured number.

**Install**: no GitHub release or tag exists for this repository (`releases` and `tags` API endpoints both return empty lists, checked 2026-08-12) [10][11], so there is no version to resolve to a release commit; `unirl.__version__` is hardcoded `"0.1.0"` in both `pyproject.toml`'s dynamic-version hook and `setup.py`, unchanged since the repo's 2026-06-08 creation [12][13]. All install claims below are read at the pinned commit 683a59b3cb1f04bea3d93cc79a79cd2552769902 itself, since that commit is what a source install would deliver — there is no separate release tag to diverge from it. `pyproject.toml` is the canonical dependency source (`setup.py` says so of itself, calling itself legacy/compat-only) [13][12]; it requires Python `>=3.12,<3.14` and pins a base `diffusers>=0.38.0`, `transformers>=5.6,<5.7`, `peft>=0.20` (torch is not in the base dependency list) [12]. Two mutually exclusive engine extras each pin a matching CUDA-build torch: `pip install -e ".[sglang]"` pulls `sglang[diffusion]==0.5.12.post1` with `torch==2.11.0+cu130`, and `pip install -e ".[vllm]"` pulls `vllm==0.20.0` with `torch==2.11.0+cu129` [12]; the two extras are declared conflicting under `[tool.uv]` and use custom PyTorch CUDA indexes [12]. `setup.py` and the flat `requirements.txt` are a looser, legacy path: unpinned `torch>=2.1`, `sglang[diffusion]==0.5.12.post1` as a hardcoded base dependency rather than an optional extra, and lower floors `diffusers>=0.37.0` / `peft>=0.14.0` — the pyproject.toml comment explains the 0.38 diffusers floor is needed for LTX-2 support (the Qwen-Image RoPE fix it also mentions was already covered at the 0.37 floor), so following the older `setup.py`/`requirements.txt` path can install a diffusers version documented as missing LTX-2 support [13][14][12]. `INSTALL.md` documents two separate venvs by CUDA/glibc floor: vllm-omni needs CUDA 12.9 and glibc>=2.28, sglang needs CUDA 13.0 and glibc>=2.34 [15]. The repository's own `LICENSE` file states "UniRL is licensed under the Apache-2.0" [16], but GitHub's repository API classifies the license as `{"key": "other", "name": "Other", "spdx_id": "NOASSERTION"}` [10], matching the shortlist row's recorded licence value of `NOASSERTION`.

**Maintained by**: Tencent-Hunyuan [2]; the pinned commit (2026-07-31) is a lint/review sweep by lead author Haonan Wang, GitHub-verified [17]; the repository's own News section records three algorithm releases in 2026-06 (DRPO, Flow-DPPO, CPPO) [1]; GitHub's live repository page reports 895 stargazers as of 2026-08-12, a live count not usable for cross-framework ranking [10].

## Quick start

The top-level README's Getting Started block, after installing per `INSTALL.md` [15][1]:

```bash
# compose-check, then launch a single-node example
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside --cfg job --resolve
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
```

The `examples/README.md` launch guide gives the domain-entrypoint pairing and one CLI form per domain [8]:

```bash
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
ENTRY=train_ar bash examples/run_experiment_single_node.sh ar/qwen_vl_grpo_geo3k_mc_4x8
ENTRY=train_sft bash examples/run_experiment_single_node.sh sft/validation/qwen3_agent_sft_lora
ENTRY=train_pe  bash examples/run_experiment_single_node.sh pe/pe_trainside_pickscore
ENTRY=train_agentic bash examples/run_experiment_single_node.sh deep_research/deep_research_search_judge
```

There is no separate Python-API quick start in the sources read: recipes are launched only through the entrypoint modules and the bash launchers above, never instantiated from a trainer class directly in the docs [8][1].

## Start it

- One GPU: run an entrypoint directly, e.g. `python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside num_devices=8` [8] (the `num_devices` override also covers a single device).
- Several GPUs / multi-node: the bash launchers `examples/run_experiment_single_node.sh` (single node) and `examples/run_experiment_multinode_taiji.sh` (multi-node) take the domain-qualified recipe name as their first argument and forward any extra arguments as Hydra overrides; `ENTRY` selects the entrypoint (default `train_diffusion`) [8]. Env vars carry cluster paths and W&B identity through to the recipe (`PRETRAINED_MODEL`, `DATA_PATH`, `EVAL_DATA_PATH`, `SFT_DATA`, `SFT_EVAL_DATA`, `REPORT_TO_WANDB`, `WANDB_PROJECT`, `WANDB_ENTITY`) rather than being hardcoded in YAML [15][8]. A mooncake-backed recipe (`*_tq_mooncake`) needs `bash examples/mooncake_master.sh start` run on the head node first [8].
- Generation layout is a recipe-level choice, not a runtime flag: the rollout engine and an optional `sync:` block set one of three deployment modes — train-side sampling (no sync), separate rollout (disjoint GPU pools, sync required), or colocated rollout (shared bundles with offload/onload, sync required) [3]. A recipe name's trailing `topology` segment (`colocate`/`separate`, or a sync transport like `nccl`/`tensor`/`ipc`) documents which mode it selects [8].
- Config surface: every recipe is one flat Hydra YAML with no config groups and no `defaults:` list, wired entirely by `_target_` dotpaths, and must start with `# @package _global_` on line 1 or Hydra silently nests the whole recipe under a bucket key [19][8]. Per-dataclass `__post_init__` validation runs today via a shared `require(...)` helper and a `validate_precision_type` field validator that accepts several aliases per precision (`bf16`/`bfloat16`, `fp16`/…, `fp32`/…) and returns a canonical form, but every call site invokes it as a bare statement and discards that return value, so the raw alias string is what actually stays in `cfg`, not a normalized one; cross-component validators for engine/sync/offload/layout contracts exist in `unirl/config/validation.py` but have no live call site yet, so a recipe that violates one of those contracts (e.g. a `sync:` block on a train-side engine) is not rejected at config time — only the trainer's own inline `layout=separate requires a dedicated engine` check fires [19]. Effective-batch arithmetic and default precision are not centrally documented outside individual recipes; `unirl/train/README.md` instead flags a silent default: `master_dtype` defaults to `None`, so the optimizer master dtype follows `param_dtype` — a bf16-loaded base keeps a bf16 optimizer master with no warning, which the doc says can round away small AdamW steps; `master_dtype: fp32` must be set explicitly for an fp32 master [20].
- OOM first aid: no OOM-specific troubleshooting section was found in the sources read (`unirl/train/README.md`, `unirl/trainer/README.md`, `unirl/config/README.md`, `INSTALL.md`); the closest documented levers are `num_updates_per_batch` (splits a rollout into N disjoint optimizer mini-batches instead of one large batch, gated by an algorithm's `supports_multi_update` flag) and the colocated-rollout mode's offload/onload of the rollout engine [9][3]. `unirl/train/README.md` warns that `fsdp_wrap` silently wraps nothing (leaving the model unsharded and un-cast) when no block class is discovered for a model, and recommends passing `block_class_names` explicitly in the recipe [20].

## Watch it

This section is mechanics only — what a metric means for a given algorithm (e.g. `ratio_mean`, `clip_fraction`) is method-card territory, not covered here.

- **Enable it**: a trainer builds a live W&B logger only when both `logging.report_to_wandb` is true and `logging.project_name` is set on the recipe; otherwise `self.wandb_logger` is a disabled null-object and trainer code calls it unconditionally with no guards, so a recipe with neither key produces no external run record (source read at commit 683a59b3) [21]. Other read fields: `run_name`, `entity` (falling back to the `WANDB_ENTITY` env var), `tags`, `logging_dir`, and a media-logging trio `log_media`/`media_max_items`/`media_log_interval` [21].
- **Metric names and namespaces** (`unirl/utils/wandb_logger.py`, read at commit 683a59b3): `log_rollout_step` is the single per-rollout entry point every trainer calls, and it fans out into three namespaces — `rollout/*` (reward/advantage and, for AR, response-length distribution stats, computed by `compute_rollout_sample_metrics`), `train/*` (optimizer scalars plus per-algorithm metrics, one point per optimizer update when `num_updates_per_batch>1`), and `perf/step_time_s` plus `perf/<phase>_time_s` for phase wall-clocks such as generate/weight_sync/reward/train [22]. `log_step`'s own docstring lists typical `train/` keys: `loss`, `policy_loss`, `kl_loss`, `approx_kl`, `clip_fraction`, `ratio_mean`/`ratio_std`, `grad_norm`, `lr` [22]. `unirl/algorithms/base.py` (read at the same commit) is where several of these keys are literally produced: `ratio_mean` and `clip_fraction` from a PPO-style clipped-ratio helper, `k3_mean` from an unbiased KL estimator, and `rollout_replay_logp_absdiff_mean` from an AR replay-vs-rollout log-prob absolute-difference check that the algorithms-module Gotchas section names as the diagnostic to watch when an AR sampling-temperature mismatch silently biases every ratio [23][4]. Multi-track runs (e.g. Prompt-Enhancer's paired AR+diffusion tracks) namespace these as `<track>/<key>` on a shared `train/step` axis [22].
- **Sample-level logging**: `log_generated_media` uploads decoded previews (images/video) at the same step as the surrounding metrics, gated by `should_log_media`, which fires when `logging.log_media` is on and `rollout_id % media_log_interval == 0` [22].
- **Evaluate during training**: `eval_interval=0` disables evaluation (the default); trainers with evaluation support run a baseline before training and then re-evaluate after every `eval_interval` completed rollouts, logged via `log_eval` under `eval/*` [24][22]. `ARTrainer` reports mean reward over a bounded batch, exposed as avg@k accuracy for binary evaluators; `DiffusionTrainer`, `PETrainer`, and `UnifiedModelTrainer` report image reward, optionally scored by an `eval_rewards` suite; agentic evaluation is not implemented — barrier and partial agentic trainers raise if evaluation is enabled, and async agentic trainers force it off regardless of the recipe value [24].
- **Health limits published**: `unirl/train/README.md`'s Gotchas quote the one hard-failure contract found in the sources read: `optimizer_step` "silently *skips* (does not crash) on a non-finite grad norm" and zeroes grads, so "a flat loss curve with a logged warning means grads went non-finite" [20]. No numeric early-stopping threshold, patience value, or reward-curve stopping rule is published in `unirl/trainer/README.md`, `unirl/train/README.md`, or `unirl/algorithms/README.md` (searched 2026-08-12) — the only threshold-shaped field found across those three pages is `eval_interval`, which paces evaluation cadence, not a stopping criterion [24][20][4].
- **Docs-site drift warning applies to this section too**: the external documentation site's Evaluation page states in its status callout that "there is no automatic periodic evaluation loop in the training driver yet," and heads a later section "Eval Plumbing (currently inert)" [18], directly contradicting the pinned commit's own `unirl/trainer/README.md`, which describes an implemented, present-tense evaluation cadence as summarized above [24]. Trust the pinned-commit repository README over that docs-site page; see "Find it in the docs" for the full staleness finding.

## Save it

- Checkpoints land at `<save_dir>/checkpoint-<step>/` [9]. The default backend, `checkpoint_format=torch`, gathers full state to distributed rank 0 and writes one `checkpoint.pt` per checkpoint directory; setting a backend's `fsdp_cfg.checkpoint_format=dcp` instead writes reshardable per-rank DCP shards plus a `metadata.pt`, and load auto-detects whichever format is on disk [9]. Prompt-Enhancer checkpoints use one subdirectory per trained side, `checkpoint-<step>/diffusion/` and (unless `freeze_llm=true`) `checkpoint-<step>/ar/`, with a shared `trainer_state.json` at the common parent [9].
- A checkpoint bundles model state, optimizer and scheduler state, step counters, and (when LoRA is active) the recorded `lora_config` (rank/alpha/target_modules/exclude_modules) that export tooling later reads [9]. `save_mode` controls what "model state" means and trades resumability for size: `save_mode=full` saves the whole model state; `save_mode=adapter` saves LoRA keys only, with the frozen base reloading from the pretrained snapshot on resume; `save_mode=auto` (the default) picks LoRA-only when LoRA is active, otherwise full [9]. Async and partial-agentic checkpoints restore the trainable model, optimizer, scheduler, and counters, but not runtime-only rollout state — in-flight generations, buffers, carried trajectories, and environment episodes restart empty on resume [9].
- Top-level Hydra keys drive saving, read by the entrypoints: `save_interval` (default `0`, disables saving; saves every N rollouts and on the last), `save_dir` (default `./checkpoints`), `save_mode` (default `auto`, but `adapter` for ReFL and `full` for Async AR), `load_dir` (unset by default; a checkpoint dir to restore and resume from) [9]. Recipes may not expose these keys, in which case they are appended with Hydra's `+` override syntax [9].
- Multi-node: `save_dir`/`load_dir` must be on storage mounted on every node; a rank that cannot see the checkpoint fails fast on every rank at load time, rather than stranding the others until an NCCL timeout [9].
- Full train-resume-export-share lifecycle, quoted from `unirl/trainer/README.md` [9]:

```bash
# 1. Train, saving LoRA-only checkpoints every 200 rollouts
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside \
    num_rollouts=500 \
    +save_interval=200 +save_dir=/ckpts/sd3_run +save_mode=adapter

# 2. Resume (after a preemption, or to extend the budget).
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside \
    num_rollouts=1000 \
    +load_dir=/ckpts/sd3_run/checkpoint-400 \
    +save_interval=200 +save_dir=/ckpts/sd3_run +save_mode=adapter

# 3a. Export a merged model
python -m unirl.tools.export_full \
    --checkpoint /ckpts/sd3_run/checkpoint-1000 \
    --base stabilityai/stable-diffusion-3.5-medium --subfolder transformer \
    --output /ckpts/sd3_run/hf-1000

# 3b. Or export a PEFT adapter artifact
python -m unirl.tools.export_adapter \
    --checkpoint /ckpts/sd3_run/checkpoint-1000 \
    --base stabilityai/stable-diffusion-3.5-medium \
    --output /ckpts/sd3_run/adapter-1000

# 4. Share
hf upload <user>/<repo> /ckpts/sd3_run/hf-1000
```

- `load_dir` restores model/optimizer/scheduler plus the optimizer-step counter and resumes the loop from the saved step; the W&B run continues too, since the driver-written `trainer_state.json` at the checkpoint root carries the run id and `train/` step axis [9].
- Adapter vs. full model: the README states plainly that "the checkpoint directory is a raw training artifact (PEFT-injected names and optimizer state), not a release artifact" [9]. An adapter-mode save is NOT a full model — `export_full` folds the recorded LoRA delta into the base weights and writes a standard `save_pretrained` folder (`--library`, `--subfolder` flags for AR vs. diffusion targets), while `export_adapter` extracts a single adapter into a PEFT adapter folder (`adapter_model.safetensors` + `adapter_config.json`) for LoRA-aware loaders [9]. NFT runs can export the EMA shadow adapter specifically with `--adapter old` [9]. Both exporters auto-detect torch-format vs. complete DCP checkpoints and reject an incomplete asynchronous DCP directory rather than partially exporting it [9].
- Loader handoff: an `export_full` output is a standard Hugging Face `save_pretrained` folder, loadable directly with `from_pretrained` per the README's own reload snippet (`AutoModel.from_pretrained("<user>/<repo>", ...)`) [9]; a raw `checkpoint-<step>/` directory or an `export_adapter` output is not — the former needs the export step above, and the latter needs pairing with the base model through a PEFT-aware loader.

## Find it in the docs

- The repository's own module-level READMEs, read at the pinned commit, are the more trustworthy source for current behavior than the external documentation site: `README.md` (top level), `unirl/README.md` (architecture map), `unirl/algorithms/README.md`, `unirl/train/README.md`, `unirl/trainer/README.md`, `unirl/config/README.md`, `INSTALL.md`, and `examples/README.md` [1][3][4][20][9][19][15][8].
- **The external docs site is stale relative to the pinned commit, and this is load-bearing** — its own `llms.txt` site map states "built from commit: unknown" [25], and reading several of its pages against the pinned-commit repo READMEs surfaces concrete drift: (a) the docs site's Evaluation page states in its status callout that "there is no automatic periodic evaluation loop in the training driver yet," and separately heads a later section "Eval Plumbing (currently inert)" [18], directly contradicted by `unirl/trainer/README.md`'s implemented, present-tense Evaluation cadence section quoted above [24]; (b) the docs site's own "Authoritative Runtime Entry" guidance and other pages use the entrypoint name `train_vlm` throughout, where the pinned commit's entrypoint is `train_ar` [25][1]; (c) the docs site's Installation page states the SGLang stack uses "torch 2.9.1+cu129," while the pinned `pyproject.toml` pins `torch==2.11.0+cu130` for the sglang extra [26][12]; (d) the docs site's Overview page references a class path `unirl.algorithms.diffusion_grpo.DiffusionGRPO` that does not exist in the pinned commit's `unirl/algorithms/` tree [27][4]; (e) the docs site's multi-node page names a recipe `vlm/qwen_vl_argrpo_geo3k_mc_sglang_4x8` [28] and its first-run page names `vlm/qwen_vl_argrpo_geo3k_mc_4x8` [29], both using an `argrpo` algorithm segment and a `vlm/` domain folder that do not exist in the pinned commit's nested example tree, whose matching recipe is `ar/qwen_vl_grpo_geo3k_mc_4x8` under the `ar/` domain folder [8]. Given this, prefer the in-repo READMEs for anything behavioral; use the docs site mainly for its page-address pattern below.
- Address pattern: `https://unirl-project.github.io/unirl/en/docs/<slug>`, a Fumadocs-style Next.js site; a machine-readable markdown mirror sits at `https://unirl-project.github.io/unirl/md/<slug>/index.md`, and a single-file corpus at `https://unirl-project.github.io/unirl/llms-full.txt` — both pointers are given by the site's own `llms.txt` [25].
- In-repo lookup recipe: architecture and module map -> `unirl/README.md`; algorithm contract and per-algorithm gotchas -> `unirl/algorithms/README.md`; train-stack internals, FSDP wrap gotchas, and the profiler env vars -> `unirl/train/README.md`; checkpointing, evaluation cadence, and trainer-level gotchas -> `unirl/trainer/README.md`; recipe-composition and cross-component validation gotchas -> `unirl/config/README.md`; install and hardware floors -> `INSTALL.md`; launch CLI forms and the recipe-naming schema -> `examples/README.md` [3][4][20][9][19][15][8].
- Runnable references beyond the docs: the `examples/` tree holds one self-contained Hydra recipe per experiment, grouped by domain (`diffusion/`, `ar/`, `sft/`, `pe/`, `unified_model/`) or agentic workflow (`alfworld/`, `deep_research/`), each entrypoint's own default recipe named as "a safe place to start" (e.g. `diffusion/sd3/sd3_trainside`, `ar/qwen_vl_grpo_geo3k_mc_4x8`) [8]. The three team-proposed algorithms each additionally ship a step-by-step tutorial folder at the repo root (`FlowDPPO/`, `DRPO/`, `CPPO/`) [1].
- Community layer: no curated community-tutorials page was found on the docs site's own site map (`llms.txt`) in the sources read [25]; the top-level README instead points readers to a WeChat group QR code and to opening a GitHub issue for questions, bug reports, or feature requests [1].
- No official MCP endpoint for querying these docs was found in the sources read.
- Traps found in maintainer-authored docs (not GitHub issue threads — no closed-issue trap discussion was fetched for this card): `unrl/config/README.md`'s own Gotchas section states cross-component config validators (engine/sync/offload/layout contracts) have no live call site today, so a recipe that violates one is not rejected at compose time [19]; `unirl/algorithms/README.md`'s Gotchas section states `num_updates_per_batch > 1` on DiffusionNFT raises in `TrainStack.__init__` because DiffusionNFT keeps `supports_multi_update = False` [4]; `unirl/train/README.md`'s Gotchas section states `fsdp_wrap` can silently wrap nothing, leaving a model unsharded and un-cast, when no block class is discovered for it [20]. Honest boundary: agentic evaluation is explicitly not implemented for any agentic trainer variant, per `unirl/trainer/README.md`'s own Evaluation cadence section [24].

## Sources

All GitHub pages, raw file reads, and API calls are at the pinned commit 683a59b3cb1f04bea3d93cc79a79cd2552769902 unless marked otherwise; all fetches performed 2026-08-12. Method names (GRPO, GSPO, DPPO, DRPO, CPPO, Flow-DPPO, DiffusionNFT) are deliberately cited to nothing here beyond the repo's own module docs — their defining papers live on the methodology cards. Ecosystem tools named in passing (vLLM, SGLang, slime, verl, Ray, Hydra, FSDP2, DCP, mooncake) are reached through the cited README links and deliberately not enumerated separately.

[1] UniRL top-level README, raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/README.md. Fetched 2026-08-12.

[2] UniRL GitHub repository. https://github.com/Tencent-Hunyuan/UniRL. Fetched 2026-08-12.

[3] `unirl/README.md` (architecture map, deployment modes, runtime data flow), raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/README.md. Fetched 2026-08-12.

[4] `unirl/algorithms/README.md` (StageAlgorithm contract, algorithm list, Gotchas), raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/algorithms/README.md. Fetched 2026-08-12.

[5] `unirl/algorithms/gspo.py` header/docstring, raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/algorithms/gspo.py. Fetched 2026-08-12.

[6] `unirl/algorithms/dppo.py` header/docstring, raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/algorithms/dppo.py. Fetched 2026-08-12.

[7] `unirl/algorithms/sft.py`, raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/algorithms/sft.py. Fetched 2026-08-12.

[8] `examples/README.md` (domains/entrypoints, running-a-recipe CLI forms, recipe-naming schema), raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/examples/README.md. Fetched 2026-08-12.

[9] `unirl/trainer/README.md` (Checkpointing, Export to Hugging Face format, Evaluation cadence, Gotchas), raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/trainer/README.md. Fetched 2026-08-12.

[10] GitHub repository API for Tencent-Hunyuan/UniRL (license classification, stargazers_count, pushed_at). https://api.github.com/repos/Tencent-Hunyuan/UniRL. Fetched 2026-08-12.

[11] GitHub tags and releases API for Tencent-Hunyuan/UniRL (both empty). https://api.github.com/repos/Tencent-Hunyuan/UniRL/tags and https://api.github.com/repos/Tencent-Hunyuan/UniRL/releases. Fetched 2026-08-12.

[12] `pyproject.toml`, raw at commit 683a59b3 (dependency floors, engine extras, CUDA index pins). https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/pyproject.toml. Fetched 2026-08-12.

[13] `setup.py`, raw at commit 683a59b3 (legacy/compat install path, version="0.1.0"). https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/setup.py. Fetched 2026-08-12.

[14] `requirements.txt`, raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/requirements.txt. Fetched 2026-08-12.

[15] `INSTALL.md`, raw at commit 683a59b3 (engine venvs, CUDA/glibc floors, env vars). https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/INSTALL.md. Fetched 2026-08-12.

[16] `LICENSE`, raw at commit 683a59b3. https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/LICENSE. Fetched 2026-08-12.

[17] GitHub commit API for 683a59b3cb1f04bea3d93cc79a79cd2552769902 (author, date, verification status). https://api.github.com/repos/Tencent-Hunyuan/UniRL/commits/683a59b3cb1f04bea3d93cc79a79cd2552769902. Fetched 2026-08-12.

[18] External docs site Evaluation page, "Evaluation" section of the full-corpus export. https://unirl-project.github.io/unirl/llms-full.txt. Fetched 2026-08-12; unpinned live page whose own site map states it is built from an unknown commit [25].

[19] `unirl/config/README.md`, raw at commit 683a59b3 (Hydra flat-recipe flow, validator Gotchas). https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/config/README.md. Fetched 2026-08-12.

[20] `unirl/train/README.md`, raw at commit 683a59b3 (FSDP wrap, master_dtype default, Gotchas, profiler env vars). https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/train/readme.md. Fetched 2026-08-12.

[21] `unirl/trainer/base.py`, raw at commit 683a59b3 (`_init_wandb` config fields). https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/trainer/base.py. Fetched 2026-08-12.

[22] `unirl/utils/wandb_logger.py`, raw at commit 683a59b3 (`log_rollout_step`, `log_step`, `log_rollout`, `log_perf`, `log_eval`, `log_generated_media`, `should_log_media`). https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/utils/wandb_logger.py. Fetched 2026-08-12.

[23] `unirl/algorithms/base.py`, raw at commit 683a59b3 (`ratio_mean`, `clip_fraction`, `k3_mean`, `rollout_replay_logp_absdiff_mean` metric-key definitions). https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/683a59b3cb1f04bea3d93cc79a79cd2552769902/unirl/algorithms/base.py. Fetched 2026-08-12.

[24] `unirl/trainer/README.md`, Evaluation cadence section — cited again here specifically for the evaluation-cadence claims contradicted by [18]. Same URL as [9]. Fetched 2026-08-12.

[25] External docs site `llms.txt` site map ("built from commit: unknown", page inventory, address patterns). https://unirl-project.github.io/unirl/llms.txt. Fetched 2026-08-12.

[26] External docs site Installation page, "Installation" section of the full-corpus export. https://unirl-project.github.io/unirl/llms-full.txt. Fetched 2026-08-12; unpinned, same staleness caveat as [18].

[27] External docs site Overview page, "Overview" section of the full-corpus export. https://unirl-project.github.io/unirl/llms-full.txt. Fetched 2026-08-12; unpinned, same staleness caveat as [18].

[28] External docs site Multi-node page, "Multinode Runs" section of the full-corpus export. https://unirl-project.github.io/unirl/llms-full.txt. Fetched 2026-08-12; unpinned, same staleness caveat as [18].

[29] External docs site First Run page, "First Run" section of the full-corpus export. https://unirl-project.github.io/unirl/llms-full.txt. Fetched 2026-08-12; unpinned, same staleness caveat as [18].
