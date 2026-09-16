# EasyDeL

A JAX/Flax post-training library with a trl-shaped trainer-class API, aimed at TPU pods and NVIDIA GPUs rather than the PyTorch/Accelerate stack.

**EasyDeL** describes itself as "an open-source framework designed to enhance and streamline the training process of machine learning models, with a primary focus on Jax/Flax," built on Flax NNX and providing "convenient and effective solutions for training and serving Flax/Jax models on TPU/GPU at scale" [1]. It is built and maintained by Erfan Zare Chavoshi [2][3], and its API is trainer-class shaped like trl's: a model is loaded with `AutoEasyDeLModelForCausalLM.from_pretrained(...)`, wrapped in a per-method trainer (`SFTTrainer`, `GRPOTrainer`, `DPOTrainer`, ...) constructed from a model, a method-specific `*Config` (a dataclass, not a transformers `TrainingArguments` subclass), and a dataset, then run with `.train()` [3]. It lives at https://github.com/erfanzar/EasyDeL [3].

**When to pick it**: JAX/Flax-native post-training when the target hardware is a TPU pod or an NVIDIA GPU and Pallas/Triton kernels matter, with a trl-like trainer-per-method API rather than trl's own PyTorch/Accelerate stack or verl's Ray-orchestrated actor model - weigh against trl when your models, datasets, and team tooling are already PyTorch/Hub-centric, since EasyDeL has no Hub-push mechanism (below) and its PyPI package classifiers list "Development Status :: 3 - Alpha" [2].

**Methods it ships**: the README's own count is "16 Specialized Trainers" [3], grouped as supervised (`SFTTrainer`, general-purpose `Trainer`), preference optimization (`DPOTrainer`, `CPOTrainer` with 5 loss variants, `ORPOTrainer`, `KTOTrainer`, `BCOTrainer`, `XPOTrainer`), reinforcement learning (`GRPOTrainer`, `GSPOTrainer`, `GFPOTrainer`, `NashMDTrainer`), distillation (`DistillationTrainer`, `GKDTrainer`), reward modeling (`RewardTrainer`), and distributed (`RayDistributedTrainer`) [3]. This list undercounts the source tree at the pinned commit: `easydel/trainers/` also contains `proximal_policy_optimization_trainer`, `rlvr_trainer`, `self_distillation_policy_optimization`, `seq_kd_trainer`, `sparse_distillation_trainer`, `embedding_trainer`, `on_policy_distillation_trainer`, and `agentic_moshpit` - eight further trainer directories the README's "16 Specialized Trainers" section does not name [5]. DAPO is not a separate trainer here: it is one of five string values (`"grpo"`, `"bnpo"`, `"dr_grpo"`, `"dapo"`, `"cispo"`) that `GRPOConfig.loss_type` accepts, and `"dapo"` is that field's own default [6]. Recheck the README's trainer list at run time, since it is the only place the method taxonomy is stated and it already understates the source.

**Scale it handles**: single GPU or TPU core direct execution; multi-host and multi-slice TPU through Ray plus EasyDeL's own `eformer` cluster-management utility, launched with the `eopod` CLI wrapping `python -m eformer.escale.tpexec.tpu_patcher` (flags for TPU version, slice count, and internal IPs, spelled out per-slice for a named topology such as "2x v4-64") - the install docs state plainly "For GPUs, manual configuration is required, but TPUs can leverage eformer" [7]. `RayDistributedTrainer` is not a general worker-placement launcher in the trl-Accelerate or verl-Ray sense: its docs (reachable only under the unpinned `/en/latest/` build - the same page 404s under the pinned `/en/stable/` build, so treat this one claim as unpinned) describe it as "a specialized training orchestrator designed for distributed training with Ray. It provides a lightweight wrapper that manages model configuration, scaling, and initialization while delegating the actual training logic to EasyDeL's core trainers," with dimension-based model scaling (`scaling_index`) as its primary feature [8]. No published multi-node benchmark accompanies either mechanism in the pages read for this card - both are documented as mechanism only [7][8].

**Install**: `uv pip install "easydel[cuda]"` for NVIDIA GPUs or `easydel[tpu]` for TPUs (plain `pip install easydel` omits the accelerator backend) [3]; version 0.3.0, released 2026-07-27 [9]; Python `>=3.11,<3.14`; Apache-2.0 [10]. At the v0.3.0 tag, the base dependency list pins `ray[default]==2.54.0`, `eformer==0.0.99.12`, `ejkernel==0.0.78`, `flax==0.12.3`, and `transformers~=5.5.0` exactly, while `jax`/`jaxlib` carry only a floor (`>=0.9.2`, no upper bound) [10]. The `[cuda]` extra additionally pins `torch==2.8.0` and `jax[cuda13]>=0.9.2`; the `[tpu]` extra pins `jax[tpu]>=0.9.2` with no torch; a separate `[torch]` extra also pins `torch==2.8.0` [10]. No CUDA driver or GPU-generation minimum is stated in `pyproject.toml`, the installation page, or the README read for this card - the only guidance, from the README, is "Choose `[cuda]` for NVIDIA GPUs with Triton kernels, or `[tpu]` for Google TPUs with Pallas kernels" [3].

**Maintained by**: Erfan Zare Chavoshi [2]; the GitHub repository shows 370 stars (not a ranking signal) and was last pushed 2026-08-04 [4]; the v0.3.0 release was published 2026-07-27, and the pinned commit `090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246` matches that release's tag exactly [9][11].

## Quick start

Both snippets are quoted verbatim, in full, from the README's training-example sections [3]:

```python
import easydel as ed
from transformers import AutoTokenizer
from datasets import load_dataset
import jax.numpy as jnp

# Load model with configuration
model_id = "Qwen/Qwen3-VL-8B-Thinking"

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
    partition_axis=ed.PartitionAxis(),
)

# Configure trainer
trainer = ed.SFTTrainer(
    model=model,
    arguments=ed.SFTConfig(
        max_length=2048,
        dataset_text_field="text",
        add_special_tokens=False,
        packing=False,
        total_batch_size=32,
        eval_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        weight_decay=0.01,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        save_directory="./checkpoints",
        report_steps=10,
        progress_bar_type="tqdm",
    ),
    train_dataset=load_dataset("timdettmers/openassistant-guanaco", split="train"),
    processing_class=AutoTokenizer.from_pretrained(model_id),
)

# Train
trainer.train()

# Save
model.save_pretrained("./my-finetuned-model")
```

```python
import easydel as ed
from transformers import AutoTokenizer
import jax.numpy as jnp

# Load model with configuration
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    backend=ed.EasyDeLBackends.GPU,
    platform=ed.EasyDeLPlatforms.TRITON,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    ),
    partition_axis=ed.PartitionAxis(),
)

# GRPO: Generate multiple completions and learn from relative rewards
trainer = ed.GRPOTrainer(
    model=model,
    arguments=ed.GRPOConfig(
        num_generations=4,  # Generate 4 completions per prompt
        max_prompt_length=2048,
        max_completion_length=1024,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        beta=0.04,
        total_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        num_train_epochs=2,
        ref_model_sync_steps=128,
        save_steps=1000,
        report_steps=20,
    ),
    train_dataset=your_prompts_dataset,
    processing_class=AutoTokenizer.from_pretrained(model_id),
    reward_funcs=your_custom_reward_fn,  # Custom reward logic
)

trainer.train()
```

The SFT trainer docs page also gives an equivalent command-line form, quoted here in full: `python -m easydel.scripts.finetune.sft --repo_id meta-llama/Llama-3.1-8B-Instruct --dataset_name trl-lib/Capybara --dataset_split "train" --dataset_text_field messages --attn_mechanism vanilla --max_sequence_length 2048 --packing True --total_batch_size 16 --learning_rate 2e-5 --learning_rate_end 5e-6 --num_train_epochs 3 --do_last_save --save_steps 1000 --use_wandb` [12].

## Start it

- One process, one accelerator (GPU or single TPU core/host) is the base form: the scripts above, as-is, once `easydel[cuda]` or `easydel[tpu]` is installed [3].
- Multi-host/multi-slice TPU goes through Ray plus `eformer`, launched with the `eopod` CLI. For a 2x v4-64 setup: `eopod run "python -m eformer.escale.tpexec.tpu_patcher --tpu-version TPU-VERSION --tpu-slice TPU-SLICES --num-slices NUM_SLICES --internal-ips INTERNAL_IP1-SLICE1,INTERNAL_IP2-SLICE1,INTERNAL_IP3-SLICE1,INTERNAL_IP4-SLICE1,INTERNAL_IP1-SLICE2,INTERNAL_IP2-SLICE2,INTERNAL_IP3-SLICE2,INTERNAL_IP4-SLICE2 --self-job"`. For a v4-256 setup: `eopod run "python -m eformer.escale.tpexec.tpu_patcher --tpu-version v4 --tpu-slice 256 --num-slices 1 --internal-ips <comma-separated-TPU-IPs> --self-job"`. Once Ray is configured, `eformer.escale.tpexec` replaces `eopod` for running distributed code [7]. GPUs are not covered by this mechanism - the same page states GPU distribution needs manual configuration [7].
- Hardware and precision are set on the MODEL, not the trainer Config: `dtype`, `param_dtype`, `precision`, `platform` (Triton/Pallas/JAX), and `sharding_axis_dims` are all arguments to `AutoEasyDeLModelForCausalLM.from_pretrained(...)`, and gradient checkpointing (`EasyDeLGradientCheckPointers`) is set through that call's `config_kwargs` - `training_configurations.py`, the file defining the base `TrainingArguments` dataclass at the pinned commit, carries none of these fields [13][3]. This is a structural split from trl, where the Config class carries hardware settings too, not a changed default to flag.
- Effective batch size is `total_batch_size` combined with `gradient_accumulation_steps` on the trainer Config, e.g. `total_batch_size=32, gradient_accumulation_steps=4` in the SFT example above [3].
- GRPO's generation for rollouts runs through a separate engine by default: `use_esurge_generation` defaults to `True` ("Whether to use eSurge engine for preview generation instead of compiled functions"), with `esurge_hbm_utilization` (default `0.45`, "HBM memory utilization target for eSurge engine (0.0-1.0)") as its memory knob [13].
- No dedicated out-of-memory guide exists in the pages searched for this card - the README, the SFT and GRPO trainer docs pages, the install page, and the base-trainer docs page were grepped for "out of memory", "oom", "memory error", and "reduce...batch" with zero matches [3][12][15][7][14]. The closest documented lever is lowering `esurge_hbm_utilization` on the generation engine, by analogy to trl's `vllm_gpu_memory_utilization`, plus the standard `total_batch_size`/`gradient_accumulation_steps` trade shown above [13].

## Watch it

This section covers only the logging mechanics; what a metric means for a given method belongs on that method's own card.

- **Enable it**: `use_wandb` on `TrainingArguments` defaults to `True` - WandB logging is opt-out, not opt-in; if `wandb` is not installed the trainer warns and returns `None` rather than crashing [13]. Cadence is set by `log_steps` (default 10, "Log metrics every X steps") and `report_steps` (default 5, "Report metrics every X steps") [13].
- **GRPO metric names are code-sourced**: the trainer docs page for GRPO carries no dedicated metrics list (checked 2026-08-12 by grepping the page for "metric", "wandb", and "logg", which returned only configuration-example and navigation text) [15], so these names come from the pinned-commit source instead. Rollout-level, from the `metrics_dict` construction in `grpo_trainer.py`: `reward_mean`, `reward_std`, `completion_length`, `grouped_comp_time`, `rewarding_time`, `token_logps_time`, `generation_time`, `preprocessing_time`, `frac_reward_zero_std`, plus one entry per reward function keyed by the bare function name itself (no `rewards/` prefix) [16]. Train-step-level, from `_fn.py`'s `other_metrics`: `mean_entropy`, `advantages`, `mean_kl` and `ref_per_token_logps` when `beta != 0`, and either the `clip_ratio/low_mean` / `clip_ratio/high_mean` / `clip_ratio/region_mean` family (for the `grpo`, `bnpo`, `dr_grpo`, and `dapo` loss types) or `cispo_clip_ratio` (for the `cispo` loss type) [17].
- **Sample-level logging**: rollout generations reach WandB only when three booleans are all true - `use_wandb`, `can_log_metrics`, and `log_training_generations_to_wandb` (all default `True`) - per the guard clause in `_log_training_generations_to_wandb` [18]. Separately, `generation_preview_print` (default `False`) prints preview generations to the terminal, and `generation_log_to_wandb` (default `True`) logs those preview generations, distinct from rollout generations, to WandB [13].
- **Evaluate during training**: `do_eval`, `eval_batch_size` (falls back to `total_batch_size` when unset), and `evaluation_steps` (default `None`, meaning disabled) drive in-loop evaluation on an `eval_dataset` [13]. A second, benchmark-suite path runs `lm_eval` (an optional extra, `pip install easydel[lm_eval]`) on a schedule set by `benchmark_interval` (default `None`, "Run configured lm-eval benchmark suites every X training steps (disabled when None)") against a `benchmarks` list of `BenchmarkConfig` entries [13][10].
- **Stopping**: no early-stopping, patience, or threshold field exists anywhere in `training_configurations.py`, the 2150-line file defining `TrainingArguments` at the pinned commit - confirmed by grepping it for "early_stop", "patience", and "stopping" with zero matches - nor in `grpo_config.py` [13][6]. The GRPO docs page's tuning advice ("Tips for Effective GRPO Training") is qualitative only: reward design, 4-8 generations, a beta range of 0.01-0.05 [15].

## Save it

- A checkpoint directory (default `save_directory="EasyDeL-Checkpoints"`) holds `config.json`, model parameters, and - only when `save_optimizer_state` (default `True`) is left on - optimizer state in TensorStore format, plus `metadata.json` recording step, timestamp, and whether optimizer state and a `_resume_model/` subdirectory are present [13][19]. `_save_state` additionally writes a JSON dump of the `TrainingArguments` and an auto-generated README to the directory root before calling `state.save_state(...)` [20].
- Turning `save_optimizer_state` off is the retention flag that costs resumability: it shrinks the checkpoint but the optimizer state needed to resume training is not written [13].
- LoRA adapters are merged into the base model at save time by default (`merge_lora_before_save=True`); the unmerged, resumable copy is kept under `_resume_model/` inside the same checkpoint directory, and `load_state`'s docstring confirms that when checkpoint metadata declares `_resume_model/` as the load source, the model and config are loaded from there so training can resume, while the merged model at the checkpoint root remains directly loadable for inference [13][19].
- Resume is on by default: `resume_if_possible` defaults to `True` [13]. The explicit call form is `EasyDeLState.load_state(...)`, whose keyword arguments include `device`, `dtype`/`param_dtype` (default `bfloat16`), `precision`, `sharding_axis_dims`, `sharding_axis_names`, `auto_shard_model`, `quantization_config`, and `apply_quantization`; the docstring's own examples cover a basic load, a sharded 8-device load, a CPU-only inference load with `auto_shard_model=False`, and a quantized load [19]. Optimizer-state load failures are logged as info, not raised as errors, so a weights-only load can still proceed [19].
- There is no `push_to_hub` method anywhere in `base_trainer.py` or `base_state.py` at the pinned commit (confirmed by grep, zero matches) - EasyDeL has no Hub-push mechanism [21][19]. The only HF-format export is `save_pretrained`, which calls `hf_model.save_pretrained(save_directory, **torch_save_pretrained_kwargs)` to produce a torch-loadable directory [21].
- Loader handoff: a saved checkpoint directory (weights plus `config.json`) is directly loadable by `EasyDeLState.load_state`, and the torch-exported `save_pretrained` output is loadable by the standard transformers `from_pretrained`; a `_resume_model/` subdirectory on its own is not a full model - it exists to restore the pre-merge training graph, not for standalone inference [19][21].

## Find it in the docs

The docs are the live source; the pointers below teach the lookup rather than mirroring the content.

- Address pattern: `https://easydel.readthedocs.io/en/<version>/<page>.html`. The pinned commit `090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246` matches the `v0.3.0` git tag exactly [11], but ReadTheDocs never completed a build for that tag - its own versions API marks `v0.3.0` as `"built": false` [22], and fetching `https://easydel.readthedocs.io/en/v0.3.0/install.html` on 2026-08-12 returned HTTP 404 [23]. The same versions API lists `stable` as an alias pointing at the identical commit with `"built": true` [22], and `https://easydel.readthedocs.io/en/stable/install.html` returned HTTP 200 with real content on 2026-08-12 [24]; this is the pinned-docs form this card uses. A byte-level comparison of the `install`, `trainers/sft`, and `trainers/base_trainer` pages fetched under `/en/stable/` against the same pages fetched under `/en/latest/` (which tracks `main`) found real content divergences, not just template noise: `install.html` gives the TPU launch command a different Python module path between builds (`eformer.escale.tpexec.tpu_patcher` under `/stable/`, `eformer.executor.tpu_patch_ray` under `/latest/`); `trainers/sft.html`'s two `SFTConfig` examples carry a `chars_per_token=3.6` line under `/stable/` that is absent under `/latest/`; `trainers/base_trainer.html`'s `sharding_array` example differs by one tuple dimension and `/latest/` adds two collate-function examples not present under `/stable/`; `trainers/ray_distributed_trainer.html` exists under `/latest/` but returns HTTP 404 under `/stable/` [24]. Every claim in this card sourced from a docs page therefore cites the `/en/stable/` build specifically, except the `RayDistributedTrainer` quote in "Scale it handles," which has no `/en/stable/` copy to cite and is flagged there as unpinned.
- Page slugs seen for this card: `install.html` at the top level; trainer pages live under `trainers/`, e.g. `trainers/sft.html`, `trainers/grpo.html`, `trainers/dpo.html`, `trainers/orpo.html`, `trainers/reward.html`, `trainers/base_trainer.html`, `trainers/trainer_protocol.html`, `trainers/ray_distributed_trainer.html` [1].
- One page is stale relative to the pinned-commit source: `trainers/base_trainer.html`'s `TrainingArguments(...)` example uses field names - `per_device_train_batch_size`, `per_device_eval_batch_size`, `sharding_array`, `evaluation_strategy`, `output_dir`, `gradient_checkpointing`, `logging_steps`, `max_steps`, `use_fast_kernels` - that do not exist anywhere in `training_configurations.py` at the pinned commit; the real names are `total_batch_size`, `eval_batch_size` (on the trainer Config, not `TrainingArguments`), `save_directory`, `log_steps`, `evaluation_steps`, `save_steps`, `save_total_limit` [14][13]. This staleness is specific to that page - `trainers/sft.html`'s own `SFTConfig` example matches the pinned-commit field names exactly: the SFT-specific fields it shows (`dataset_text_field`, `packing`, `dataset_num_proc`, `dataset_batch_size`, `eval_packing`, `num_of_sequences`) are all defined in `sft_config.py`, and the base fields it shows (`total_batch_size`, `learning_rate`, `num_train_epochs`, `save_directory`, `use_wandb`, `warmup_steps`, `weight_decay`) are all defined in `training_configurations.py` [12][13][25].
- Runnable references beyond the docs: the `easydel.scripts.finetune.*` CLI module family (SFT shown above) [12], and the repository's own examples reachable from the GitHub tree [4].
- No official community-tutorials curation page and no MCP endpoint for the docs were found: the docs index's table of contents lists API reference, training guides, and a "Community-Driven Development" feature bullet describing future collaboration plans, but no tutorials-listing page or MCP server entry [1].

## Sources

Method names (SFT, DPO, GRPO, KTO, CPO, ORPO, BCO, XPO, GSPO, GFPO, Nash-MD, distillation, reward modeling) are deliberately cited to nothing here; their defining papers belong on the methodology cards. All page fetches are dated 2026-08-12 unless a different date is given. All docs claims tied to the pinned commit are sourced from the `/en/stable/` build, except the single `RayDistributedTrainer` quote in "Scale it handles," which is sourced from the unpinned `/en/latest/` build and flagged there, since that page 404s under `/en/stable/`. A byte-level comparison found `/en/stable/` and `/en/latest/` genuinely differ on every page compared - see "Find it in the docs" for specifics - so `/en/latest/` content is never treated as a stand-in for the pin.

[1] EasyDeL documentation index, pinned build. https://easydel.readthedocs.io/en/stable/index.html. Fetched 2026-08-12.

[2] EasyDeL PyPI package metadata (author; classifiers, including "Development Status :: 3 - Alpha"). https://pypi.org/pypi/easydel/json. Fetched 2026-08-12.

[3] EasyDeL GitHub repository README. https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/README.md. Fetched 2026-08-12, at the pinned commit.

[4] EasyDeL GitHub repository API record (stars, pushed_at, description). https://api.github.com/repos/erfanzar/EasyDeL. Fetched 2026-08-12.

[5] EasyDeL repository tree at the pinned commit (trainer subdirectories under `easydel/trainers/`). https://api.github.com/repos/erfanzar/EasyDeL/git/trees/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246?recursive=1. Fetched 2026-08-12, at the pinned commit.

[6] EasyDeL `grpo_config.py` at the pinned commit (`loss_type` field and its default; `num_generations`/`num_return_sequences` alias reconciliation). https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/easydel/trainers/group_relative_policy_optimization/grpo_config.py. Fetched 2026-08-12, at the pinned commit.

[7] EasyDeL install/Ray documentation page, pinned build (multi-host/multi-slice TPU setup via `eopod` and `eformer`). https://easydel.readthedocs.io/en/stable/install.html. Fetched 2026-08-12.

[8] EasyDeL RayDistributedTrainer documentation page, unpinned `/en/latest/` build only - this page returns HTTP 404 under the pinned `/en/stable/` build. https://easydel.readthedocs.io/en/latest/trainers/ray_distributed_trainer.html. Fetched 2026-08-12.

[9] EasyDeL GitHub releases API (v0.3.0 `published_at`). https://api.github.com/repos/erfanzar/EasyDeL/releases. Fetched 2026-08-12.

[10] EasyDeL `pyproject.toml` at the v0.3.0 tag (Python floor, licence, dependency pins, optional-dependency extras). https://raw.githubusercontent.com/erfanzar/EasyDeL/v0.3.0/pyproject.toml. Fetched 2026-08-12, at the v0.3.0 tag, which resolves to the pinned commit.

[11] EasyDeL GitHub tag reference for v0.3.0 (tag object sha, matching the pinned commit). https://api.github.com/repos/erfanzar/EasyDeL/git/refs/tags/v0.3.0. Fetched 2026-08-12.

[12] EasyDeL SFT trainer documentation page, pinned build (`SFTConfig` field names, CLI command form). https://easydel.readthedocs.io/en/stable/trainers/sft.html. Fetched 2026-08-12.

[13] EasyDeL `training_configurations.py` at the pinned commit (`TrainingArguments` dataclass: `use_wandb`, `log_steps`, `report_steps`, `save_directory`, `save_optimizer_state`, `merge_lora_before_save`, `resume_if_possible`, `do_eval`, `eval_batch_size`, `evaluation_steps`, `benchmark_interval`, `benchmarks`, `use_esurge_generation`, `esurge_hbm_utilization`, `generation_preview_print`, `generation_log_to_wandb`, `log_training_generations_to_wandb`; confirmed absence of hardware/precision fields and of any stopping-rule field). https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/easydel/trainers/training_configurations.py. Fetched 2026-08-12, at the pinned commit.

[14] EasyDeL base trainer documentation page, pinned build (`TrainingArguments` example with stale field names). https://easydel.readthedocs.io/en/stable/trainers/base_trainer.html. Fetched 2026-08-12.

[15] EasyDeL GRPO trainer documentation page, pinned build (no metrics list; qualitative tuning tips). https://easydel.readthedocs.io/en/stable/trainers/grpo.html. Fetched 2026-08-12.

[16] EasyDeL `grpo_trainer.py` at the pinned commit (rollout-level `metrics_dict` construction). https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py. Fetched 2026-08-12, at the pinned commit.

[17] EasyDeL `_fn.py` at the pinned commit (train-step-level `other_metrics`, loss-type-conditional clip-ratio metrics). https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/easydel/trainers/group_relative_policy_optimization/_fn.py. Fetched 2026-08-12, at the pinned commit.

[18] EasyDeL `base_trainer.py` at the pinned commit (`_log_training_generations_to_wandb` guard clause). https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/easydel/trainers/base_trainer.py. Fetched 2026-08-12, at the pinned commit.

[19] EasyDeL `base_state.py` at the pinned commit (`EasyDeLState.save_state`/`load_state` contract and docstrings). https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/easydel/infra/base_state.py. Fetched 2026-08-12, at the pinned commit.

[20] EasyDeL `base_trainer.py` at the pinned commit (`_save_state` method: TrainingArguments JSON and README written to the checkpoint directory root). https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/easydel/trainers/base_trainer.py. Fetched 2026-08-12, at the pinned commit.

[21] EasyDeL `base_trainer.py` at the pinned commit (`save_pretrained`/`_save_to_torch`; confirmed absence of any `push_to_hub` method). https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/easydel/trainers/base_trainer.py. Fetched 2026-08-12, at the pinned commit.

[22] ReadTheDocs versions API for EasyDeL (`v0.3.0` marked `"built": false`; `stable` marked `"built": true` and pointing at the same commit). https://readthedocs.org/api/v3/projects/easydel/versions/. Fetched 2026-08-12.

[23] EasyDeL docs page under the `v0.3.0` version slug, returning HTTP 404. https://easydel.readthedocs.io/en/v0.3.0/install.html. Fetched 2026-08-12.

[24] EasyDeL docs pages under the `stable` version slug (`install.html`, `trainers/sft.html`, `trainers/grpo.html`, `trainers/base_trainer.html`, `trainers/ray_distributed_trainer.html`), compared byte-for-byte against the same pages under `/en/latest/`; `install.html` returned HTTP 200 with content that differs from `/en/latest/` in the TPU launch command's module path, as do the other three existing pages in narrower ways, and `trainers/ray_distributed_trainer.html` returns HTTP 404 under `/en/stable/` though it exists under `/en/latest/`. https://easydel.readthedocs.io/en/stable/install.html. Fetched 2026-08-12.

[25] EasyDeL `sft_config.py` at the pinned commit (`SFTConfig` dataclass fields: `dataset_text_field`, `packing`, `dataset_num_proc`, `dataset_batch_size`, `eval_packing`, `num_of_sequences`). https://raw.githubusercontent.com/erfanzar/EasyDeL/090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py. Fetched 2026-08-12, at the pinned commit.
