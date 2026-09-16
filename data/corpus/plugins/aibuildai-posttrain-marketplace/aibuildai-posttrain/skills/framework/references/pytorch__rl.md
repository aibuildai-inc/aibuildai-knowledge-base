# torchrl

A general-purpose PyTorch reinforcement-learning toolkit that ships a GRPO/SFT stack for LLM post-training as one specialized workflow among many, launched through Hydra configs and Ray.

**TorchRL** is "a PyTorch-native toolkit for reinforcement learning, decision making, robotics, and simulation," built as "a collection of composable pieces for building RL systems while keeping the code close to the PyTorch programming model" rather than a single algorithm or a narrow benchmark suite [1]. It lives at https://github.com/pytorch/rl [3], is copyrighted to Meta Platforms, Inc. and affiliates [15], and is maintained under the name "torchrl contributors" [2]. Its LLM post-training path is a set of loss classes (`GRPOLoss`, `SFTLoss`, ...) under `torchrl.objectives.llm`, paired with a `ChatEnv` environment, vLLM/SGLang generation wrappers, and Ray-based collectors, driven from Hydra-configured training scripts rather than a single trainer class [4].

**When to pick it**: pick it when you are already building on TorchRL's environment/collector/replay-buffer primitives, or specifically want GRPO/DAPO/CISPO or SFT losses assembled from those primitives with Ray-orchestrated sync or async generation; the README itself frames LLM post-training as an "additional specialized workflow" layered on a general RL toolkit, not the library's primary purpose [1]. The GRPO reference scripts assume at least 3 CUDA GPUs (one each for training, vLLM inference, and the reference model) [5], a materially higher floor than single-GPU trainer-class libraries; weigh that against a Hub-native, single-trainer-class library such as trl (cross-reference; not covered here) if you want to start on one machine.

**Methods it ships**: for LLM post-training, `torchrl.objectives.llm` exposes `GRPOLoss` ("GRPO loss."), `CISPOLoss` ("CISPO (Clipped Importance Sampling Policy Optimization)."), `DAPO` ("DAPO (Clip-Higher over GRPO)."), and `SFTLoss` ("Supervised fine-tuning loss."), plus their paired `*LossOutput` classes and an `MCAdvantage` ("Monte-Carlo advantage computation engine.") advantage estimator [4]. These import from the top-level `torchrl.objectives.llm` package, confirmed by reading `torchrl/objectives/llm/__init__.py` at commit fa08a50 [6]. The reference GRPO recipes live under `sota-implementations/grpo/` and `sota-implementations/expert-iteration/`, which the README calls its maintained, benchmarked implementations, distinct from the `examples/` folder, which it calls smaller and less actively maintained [1] — the shortlist's REWARD/RLHF pointer (`examples/rlhf/models/reward.py`, a GPT-2 reward-model wrapper around `GPT2RewardModel`) lands in that less-maintained folder, and its own README references "the current release (v0.3)" as not yet benchmarking it, a stale marker against the current 0.13.3 release [7][8]. The shortlist's PPO pointer (`sota-implementations/gail/ppo_utils.py`) is classic-control PPO utility code used inside the GAIL recipe, not an LLM-RLHF PPO trainer [9]. TorchRL also ships a large catalogue of non-LLM RL algorithms (PPO, SAC, TD3, DDPG, DQN, CQL, Dreamer, and more) under `sota-implementations/`, which this card does not enumerate since they are outside LLM post-training [1]. Beyond LLM post-training, TorchRL also carries a separate, generic `torchrl.trainers.Trainer` hook-based class for classic RL loops; the GRPO/expert-iteration scripts do not use it, and it is not part of the LLM stack described here [10].

**Scale it handles**: single multi-GPU node via `train_model.num_devices` / `ref_model.num_devices` / `inference_model.num_devices` overrides, which the GRPO README says "automatically handles device allocation" and "works correctly in both sync and async modes" [5]; multi-node is documented as mechanism through two provided SLURM scripts, `grpo-sync-multi-node.sbatch` and `grpo-async-multi-node.sbatch`, which launch Ray-orchestrated training across nodes, with no published multi-node benchmark accompanying them [5]. Sharding/memory options on the training model are LoRA (`train_model.lora.enabled`, with rank/alpha/dropout fields) and 4-bit quantization toggles; the example configs also expose `train_model`/`ref_model`/`inference_model` `num_devices` splits so each role can be placed on its own GPU(s), and generation itself is a separate vLLM or SGLang process rather than colocated with the training model [11][12]. FSDP-related parameters (`fsdp`, `fsdp_config`) appear in the GRPO model-building code, but their config surface and reachable scale are not documented on a page read for this card and are not restated here beyond noting they exist in `grpo_utils.py` at commit fa08a50 (ahead of the pinned 0.13.3 release) [12].

**Install**: `pip install torchrl`; a CUDA build such as `pip install "torchrl==0.13.0+cu128" --extra-index-url https://download.pytorch.org/whl/cu128` is offered for GPU wheels [1]. Latest PyPI release is 0.13.3, uploaded 2026-07-14 [13], whose git tag `v0.13.3` resolves to commit `662ba6750d67da1007a2010162f09e11631343cd` (tagged 2026-07-14T15:31:34Z) [14] — this is the commit this Install field is pinned to, and it is older than the shortlist's screening commit `fa08a50974d69f22891bb54b06dca22cdfc79d5f` (the repository's newest push, 2026-08-05). Python >= 3.10; MIT License [2][15]. Core dependency floors from `pyproject.toml` at the v0.13.3 tag: `torch>=2.1.0` and `tensordict>=0.13.0,<0.14.0` (the latter an upper-bounded pin on TorchRL's own companion tensor library) [14]. The LLM path is opt-in via extras: `torchrl[llm]` (transformers, accelerate, datasets, sentencepiece, ...), `torchrl[llm-vllm]` (adds `vllm>=0.17` on Linux), `torchrl[llm-sglang]` (adds `sglang[all]` on Linux), and a bundling `torchrl[grpo]` extra that adds peft, wandb, ray, and (recommended but optional) flash-attn, bitsandbytes, xformers [14]. Neither the README nor the pinned `pyproject.toml` states a CUDA or hardware-driver minimum; the GRPO recipe's own hardware line (3 CUDA GPUs) is a workflow requirement, not a package-level floor [1][5][14].

**Maintained by**: the "torchrl contributors" group under the PyTorch org, contact listed as vmoens@fb.com in the package metadata [14]; recent GitHub releases show roughly monthly cadence, most recently v0.13.3 (2026-07-14), v0.13.2 (2026-06-17), v0.13.1 (2026-06-08), and v0.13.0 (2026-06-05) [16].

## Quick start

The snippet below is quoted verbatim from the docs' "Quick Example" section for the LLM API, "Using vLLM backend" [4]. It builds the collector but stops short of a training loop, since the same page's LLM objectives (`GRPOLoss`, `SFTLoss`) are documented separately in its API tables, not wired into this example [4]:

```python
from torchrl.modules.llm import vLLMWrapper, AsyncVLLM
from torchrl.envs.llm import ChatEnv
from torchrl.collectors.llm import LLMCollector

# Create vLLM engine
engine = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-7B", num_replicas=2)
policy = vLLMWrapper(engine, input_mode="history")

# Create environment
env = ChatEnv(tokenizer=tokenizer)

# Create collector
collector = LLMCollector(env, policy, dialog_turns_per_batch=256)
```

The same page also quotes an SGLang-backend form of the identical example, swapping in `SGLangWrapper`/`AsyncSGLang.from_pretrained("Qwen/Qwen2.5-7B", tp_size=2)` for the engine and wrapper lines, with `ChatEnv` and `LLMCollector` unchanged [4].

The reference GRPO recipe reduces to a single Hydra-driven CLI invocation, quoted from its README: `python sota-implementations/grpo/grpo-sync.py mode=sync train_model.num_devices=2 ref_model.num_devices=2 inference_model.num_devices=2` for the synchronous script, or `grpo-async.py` with `mode=async` for the (README-recommended) asynchronous one [5].

## Start it

- One node, N GPUs: run `grpo-sync.py` or `grpo-async.py` directly with `train_model.num_devices=`, `ref_model.num_devices=`, and `inference_model.num_devices=` set to partition GPUs across the three roles; the README states this "automatically handles device allocation," "works correctly in both sync and async modes," and "prevents device conflicts between model components" [5].
- Multi-node: `sbatch sota-implementations/grpo/grpo-sync-multi-node.sbatch` or `grpo-async-multi-node.sbatch`, SLURM job scripts that launch Ray-orchestrated GRPO across nodes [5].
- Generation layout: the reference configs always place vLLM/SGLang inference on its own device group (`inference_model.num_devices`) separate from the training and reference models — the example config sets `train_model.num_devices: 1`, `inference_model.num_devices: 1`, `ref_model.num_devices: 1` with `inference_model.backend: "vllm"` — so, unlike a colocated single-GPU mode, generation is always a distinct process/device pool in this recipe [11].
- Sync vs async is the primary scale/throughput choice: sync runs collection and optimization sequentially per the README's three-nested-loop pseudocode and updates weights before every data collection; async runs collection in the background against a larger replay buffer and updates weights at a configured interval, which the README states offers "better performance," "more efficient GPU utilization," and "better throughput" — but gives no published benchmark numbers for that comparison [5].
- Effective batch / step arithmetic (GSM8K example config, real numbers): `train.dialog_turns_per_batch: 32`, `train.optim_batch_size: 32`, `train.gradient_accumulation_steps: 8`, `train.total_dialog_turns: 100_000`; sync mode requires `buffer_size = dialog_turns_per_batch` per the README, while async allows a larger buffer [5][11].
- Config surface: Hydra YAML (`config/grpo_gsm8k.yaml` default, `config/grpo_ifeval.yaml` for IFEval), overridable on the CLI (`optimizer.lr=2e-5`) and sweepable with `--multirun` (`optimizer.lr=1e-4,1e-5,1e-6`) [5]. The example config's precision default is `torch_dtype: bfloat16` for all three model roles, with `train.mixed_precision: true` — a bf16-capable GPU is the silent hardware assumption carried by this default, not stated as a requirement elsewhere [11].
- Memory-relevant defaults worth flagging: LoRA is on by default in the example config (`train_model.lora.enabled: true`, r=8, alpha=16, dropout=0.1, mirrored on `ref_model`) and gradient checkpointing is on for the training model (`train_model.gradient_checkpointing: true`) but off for the reference model, since it never backpropagates [11].
- Out-of-memory first aid, quoted/paraphrased from the README's own debugging section [5]: for vLLM inference OOM, reduce `inference_model.gpu_memory_utilization=FRACTION` or `env.num_envs=N`; if KL scoring runs on the batch, reduce `env.num_envs=N`; for training OOM, reduce `train.optim_batch_size`.

## Watch it

This section covers the mechanics only; what each signal means for GRPO (healthy KL ranges, clip-fraction interpretation, etc.) is not restated here and belongs on the method's own card.

- **Enable it**: the GRPO scripts construct a `WandbLogger` unconditionally at startup (`wandb_logger = WandbLogger(project="grpo-sync", exp_name=...)` in `grpo-sync.py`) — there is no config flag in the read source to disable it or swap backends, so a run logs to Weights & Biases by default with no built-in "log nowhere" mode observed in this script [17]. Logging cadence is `train.logging_frequency` (10 in the example config) [11].
- **Metric names actually logged**, read directly from the `log_training_metrics()` function in `grpo_utils.py` at commit fa08a50 (ahead of the pinned 0.13.3 release) — this is more precise than the README's metric list, which only names categories ("Reward," "Advantage," "KL penalty," ...) without the logged key strings [18][5]: `training/loss_objective`, `training/clip_fraction`, `training/ESS`, `training/entropy_loss`, `training/kl_approx_to_inference`, `training/kl_to_inference`, `training/loss_kl_to_inference`, `training/grad_norm`, `training/gradient_steps`, `training/optim_steps`; `inference/policy_version`, `inference/batch_policy_version`, `inference/batch_policy_age`, `inference/staleness_mean`, `inference/staleness_max`; `buffer/write_count`, `buffer/reward_mean`, `buffer/seq_length_mean`, `buffer/step_count_mean`, `buffer/data_read_count`; `throughput/gradient_steps_per_second`, `throughput/optim_steps_per_second`, `throughput/gradient_steps_per_write`, `throughput/optim_steps_per_write`; and, only when `train.use_kl_to_ref` is true, `training/kl_to_ref`, `training/loss_kl_to_ref`, and `buffer/kl_penalty_to_ref_mean` [18].
- **Two distinct KL terms, not one knob**: the README distinguishes KL[ref || policy] (canonical for LLM post-training, penalizing drift from the frozen pre-trained reference, coefficient `train.kl_to_ref_coeff`) from KL[policy || inference] (canonical PPO-style, penalizing drift from the policy that generated the data, coefficient `train.kl_to_inference_coeff`) — these are logged separately (`training/kl_to_ref` vs `training/kl_to_inference`) and are not the same quantity [5].
- Sample-level generation logging and a documented evaluation-during-training path were not found in the README, `grpo_utils.py`, or the example configs read for this card; none of the fields inspected (`grpo_gsm8k.yaml`, `grpo_readme.md`, `grpo_utils.py`) name an eval dataset or eval cadence field for the GRPO recipe [5][11][18].
- **Stopping**: no RL-specific stopping rule, threshold, or patience value is published for GRPO. This is the search that says so, read 2026-08-11: the GRPO README's Monitoring and Debugging sections name metrics and OOM knobs but no threshold [5]; `grpo_gsm8k.yaml` has no early-stopping or patience field [11]; the Knowledge Base index page lists "Things to consider when debugging RL" among its linked pages [19], and that linked page itself — the library's actual runtime-tuning/debugging content — has no match for "early stop," "patience," "threshold," or "health" in its rendered text [23]. Shapes (which metrics exist) are published; thresholds for when to stop are not.

## Save it

- The GRPO README documents checkpoints under `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/checkpoint_*.pt`, saved every `train.checkpoint_frequency` steps and stated to contain model state, optimizer state, gradient-scaler state, and the full config [5]. Reading the actual scripts at commit fa08a50 contradicts this: in `grpo-sync.py`, the entire checkpoint-saving block is commented out with the comment "Checkpointing disabled to prevent disk space issues," covering the `torch.save({"step":..., "model_state_dict":..., "optimizer_state_dict":..., "scaler_state_dict":..., "config":...}, checkpoint_dir / f"checkpoint_{global_step:04d}.pt")` call; `grpo-async.py` creates the checkpoint directory (`checkpoint_dir.mkdir(...)`) but contains no checkpoint-saving code anywhere in the file [20]. The same pattern repeats in `sota-implementations/expert-iteration/expert-iteration-sync.py`, which has an identical commented-out save block, against an `expert-iteration/README.md` that makes the same "automatic checkpointing" claim as the GRPO README [21]. At the screening commit, no run of either reference GRPO script actually writes a checkpoint file — a reader relying on the README's claim would get an empty `checkpoints/` directory.
- These SOTA recipes therefore do not currently give a save/reload/resume path to verify; TorchRL's separate, generic `torchrl.trainers.Trainer` class (not used by the GRPO scripts) documents its own checkpointing via a `CKPT_BACKEND` environment variable selecting `torch`, `torchsnapshot`, or `memmap` backends, described on its docs page as one of the class's "Checkpointing support" key features [10] — this is the classic-RL trainer's mechanism, not the LLM GRPO path, and is named here only to avoid conflating the two.
- No loader-handoff claim is made for the GRPO checkpoint path in this card, since no checkpoint file is actually produced by the read scripts at the screening commit; do not assume an evaluator can load a GRPO run's output without first confirming (by reading the current script) whether the checkpoint block has been re-enabled.

## Find it in the docs

The docs are the live source; this section is the lookup, not a mirror.

- Address pattern: `https://docs.pytorch.org/rl/<version>/<path>`, where `<version>` is `stable`, `main`, or a short release series like `0.13` — all three forms return HTTP 200 when fetched directly (checked 2026-08-02) [1]. The LLM API reference page is `reference/llms.html`; the generic RL trainer page is `reference/trainers.html`; the knowledge base is `reference/knowledge_base.html` [4][10][19].
- Question-to-page map: full LLM objectives/environments/collectors API -> `reference/llms.html` [4]; classic RL `Trainer` hook system and checkpoint backends -> `reference/trainers.html` [10]; install/debugging/cluster tips -> `reference/knowledge_base.html` [19]; runnable end-to-end recipes -> the `sota-implementations/` and `examples/` trees in the GitHub repository, including the GRPO recipe's own README and configs used throughout this card [1][5].
- Runnable references beyond the docs: `sota-implementations/grpo/` (config files, sync/async scripts, SLURM multi-node scripts) and `sota-implementations/expert-iteration/` are the maintained recipes; `examples/rlhf/` is the older, less-maintained reward-model example the shortlist also points at [1][7].
- Community layer, curated door first: the docs' own tutorials index curates "LLM Wrappers in TorchRL" alongside general RL tutorials (PPO, DDPG, DQN, replay buffers, custom environments) — checked 2026-08-11, that index lists 24 first-party tutorial titles with no separate practitioner-blog curation page found alongside it [22]. No external practitioner blogs are named here because none were found curated by TorchRL's own docs (unlike trl's `community_tutorials` page); if you find one, check its pinned TorchRL version against yours before trusting it.
- No official MCP endpoint for querying TorchRL's docs was found in the pages read for this card.
- Honest boundary: the GRPO reference recipe assumes at least 3 CUDA GPUs and has no documented single-GPU colocated mode in the configs read here [5][11]; its checkpoint-saving code is disabled at the screening commit despite the README's claim otherwise, so treat "automatic checkpointing" as not currently true for this recipe until you verify the current script state [5][20][21]. No GitHub issue with a maintainer reply about either of these points was found in the pages fetched for this card, so no issue-number trap is cited here.

## Sources

All pages fetched from the repository at commit `fa08a50974d69f22891bb54b06dca22cdfc79d5f` (the shortlist's screening commit, the repository's newest push as of 2026-08-05) unless a release tag is named; docs pages are `stable`-version live pages read on the dates given. Ecosystem tools named in passing (Ray, Hydra, vLLM, SGLang, SLURM, Weights & Biases, PEFT/LoRA, FSDP) are not separately cited beyond the sources that name them above.

[1] torchrl README at commit fa08a50. https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/README.md. Fetched 2026-08-02.

[2] torchrl on PyPI. https://pypi.org/project/torchrl/. Fetched 2026-08-02.

[3] torchrl GitHub repository. https://github.com/pytorch/rl. Fetched 2026-08-02.

[4] torchrl LLM API reference docs, showing "torchrl 0.13 documentation" at the `stable` URL. https://docs.pytorch.org/rl/stable/reference/llms.html. Fetched 2026-08-02.

[5] sota-implementations/grpo/README.md at commit fa08a50. https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/sota-implementations/grpo/README.md. Fetched 2026-08-02.

[6] torchrl/objectives/llm/__init__.py at commit fa08a50. https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/torchrl/objectives/llm/__init__.py. Fetched 2026-08-02.

[7] examples/rlhf/models/reward.py at commit fa08a50 (the shortlist's REWARD/RLHF methods_seen file). https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/examples/rlhf/models/reward.py. Fetched 2026-08-02.

[8] examples/rlhf/README.md at commit fa08a50 (the "current release (v0.3)" benchmarking note). https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/examples/rlhf/README.md. Fetched 2026-08-02.

[9] sota-implementations/gail/ppo_utils.py at commit fa08a50 (the shortlist's PPO methods_seen file). https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/sota-implementations/gail/ppo_utils.py. Fetched 2026-08-02.

[10] torchrl generic Trainer reference docs, showing "torchrl 0.13 documentation" at the `stable` URL. https://docs.pytorch.org/rl/stable/reference/trainers.html. Fetched 2026-08-02.

[11] sota-implementations/grpo/config/grpo_gsm8k.yaml at commit fa08a50 (the default reference config). https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/sota-implementations/grpo/config/grpo_gsm8k.yaml. Fetched 2026-08-11.

[12] sota-implementations/grpo/grpo_utils.py at commit fa08a50 (model-building code: LoRA/quantization/FSDP parameters). https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/sota-implementations/grpo/grpo_utils.py. Fetched 2026-08-02.

[13] PyPI API JSON for torchrl (release date). https://pypi.org/pypi/torchrl/json. Fetched 2026-08-02.

[14] pyproject.toml at the v0.13.3 git tag (dependency floors, extras, maintainer contact). https://raw.githubusercontent.com/pytorch/rl/v0.13.3/pyproject.toml. Fetched 2026-08-02.

[15] LICENSE file at commit fa08a50. https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/LICENSE. Fetched 2026-08-02.

[16] GitHub Releases API for pytorch/rl. https://api.github.com/repos/pytorch/rl/releases. Fetched 2026-08-02.

[17] sota-implementations/grpo/grpo-sync.py at commit fa08a50 (WandbLogger instantiation, no toggle found). https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/sota-implementations/grpo/grpo-sync.py. Fetched 2026-08-02.

[18] sota-implementations/grpo/grpo_utils.py at commit fa08a50, `log_training_metrics()` function (exact logged metric keys). Same URL as [12]. Fetched 2026-08-02.

[19] torchrl Knowledge Base docs page (runtime/debugging/tuning reference searched for a stopping rule), showing "torchrl 0.13 documentation" at the `stable` URL. https://docs.pytorch.org/rl/stable/reference/knowledge_base.html. Fetched 2026-08-11.

[20] sota-implementations/grpo/grpo-sync.py and grpo-async.py at commit fa08a50 (checkpoint-saving code: commented out in sync, absent in async). Same URL pattern as [17], plus https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/sota-implementations/grpo/grpo-async.py. Fetched 2026-08-02.

[21] sota-implementations/expert-iteration/README.md and expert-iteration-sync.py at commit fa08a50 (same "automatic checkpointing" claim and same disabled save block). https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/sota-implementations/expert-iteration/README.md and https://raw.githubusercontent.com/pytorch/rl/fa08a50974d69f22891bb54b06dca22cdfc79d5f/sota-implementations/expert-iteration/expert-iteration-sync.py. Fetched 2026-08-02.

[22] torchrl tutorials index page (curated first-party tutorial list), showing "torchrl 0.13 documentation" at the `stable` URL. https://docs.pytorch.org/rl/stable/tutorials/index.html. Fetched 2026-08-11.

[23] torchrl "Things to consider when debugging RL" knowledge-base content page, linked from the Knowledge Base index [19] and searched directly for a stopping rule, showing "torchrl 0.13 documentation" at the `stable` URL. https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/DEBUGGING_RL.html. Fetched 2026-08-11.
