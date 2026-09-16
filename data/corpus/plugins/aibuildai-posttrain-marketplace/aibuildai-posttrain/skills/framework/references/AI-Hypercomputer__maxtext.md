# maxtext

Google's reference JAX LLM training library for Google Cloud TPUs and GPUs, with post-training (SFT, DPO, GRPO, GSPO) built on Tunix.

MaxText describes itself as "a high performance, highly scalable, open-source LLM library and reference implementation written in pure Python/JAX and targeting Google Cloud TPUs and GPUs for training" [1]. It is built and maintained by Google, under the `AI-Hypercomputer` GitHub organization, and its stack is "Flax (neural networks), Tunix (post-training), Orbax (checkpointing), Optax (optimization), and Grain (dataloading)" [1]; post-training is a set of standalone trainer scripts (`maxtext.trainers.post_train.{sft,dpo,rl,distillation}`) that each wrap a Tunix trainer class and are launched as `python3 -m <module> key=value ...` against a YAML base config [2][3]. It lives at https://github.com/AI-Hypercomputer/maxtext [1].

**When to pick it**: pick MaxText specifically for TPU-first pre-training and post-training in pure JAX/Flax, where checkpoints and sharding stay in the Orbax/JAX world throughout; the README states pre-training scales "up to tens of thousands of chips" and post-training covers "Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO, ...) and Group Sequence Policy Optimization (GSPO, ...)" [1]. This is a narrower method menu than a PyTorch-first, Hub-native library like trl — pick trl instead if you need broader method coverage or a Hub-native, PyTorch-first workflow (cross-reference; not covered here, and no cross-framework method count is asserted here since that comparison belongs on the card deck).

**Methods it ships**: SFT (`maxtext.trainers.post_train.sft.train_sft`) [2], DPO (`maxtext.trainers.post_train.dpo.train_dpo`, which also supports ORPO per the docs tutorial's title "Preference Optimization (DPO & ORPO) on Single-Host TPUs") [4], RL via GRPO and GSPO (`maxtext.trainers.post_train.rl.train_rl`, selected with the config key `loss_algo: 'grpo'` or `'gspo-token'`) [3][5], and knowledge distillation (`maxtext.trainers.post_train.distillation`, present as a directory with a `train_distill.py` script and its own README, not read in detail for this card) [6]. GRPO also has an async rollout mode, `AgenticGRPOLearner`, enabled with `rl.use_agentic_rollout=True` [3]. RL is generation-heavy: MaxText documents that "we rely on the vLLM library" for response generation during GRPO/GSPO [3]. The shortlist row's own evidence field points at `src/maxtext/experimental/rl/grpo_trainer.py`; that file still exists in the repository but sits under `experimental/`, and both the README and the current RL tutorial instead document `src/maxtext/trainers/post_train/rl/train_rl.py` as the entry point [1][3][7] — treat the `experimental/rl` path as legacy.

**Scale it handles**: single-host TPU VM (e.g. v6e-8, v5p-8) up to multi-host, per the RL tutorial's own worked example and its separate "Reinforcement Learning on Multi-Host TPUs" page [3]. GPU support is narrower than TPU support: the install docs scope the `cuda12` extra to "pre-training and decoding on GPUs" only, while post-training runs through the separate `tpu-post-train` extra, documented as TPU-only and also the extra used "for running vllm_decode on TPUs" [8]; the `cuda12` extra's own PyPI dependency list omits `google-tunix`, `accelerate`, `peft`, and `ray` [10], and SFT's own trainer imports `tunix.sft.metrics_logger`, `peft_trainer`, and `profiler` directly from Tunix [2] — so SFT (and the other Tunix-based post-training methods) is not installable through the `cuda12` extra; post-training on this library is TPU-only as documented. Multi-host launchers are Google Kubernetes Engine via Cluster Toolkit (`gcluster`) or XPK, and Google's Pathways orchestration layer; the RL config exposes `use_pathways: true` and comments the fraction of chips split between trainer and sampler roles (`trainer_devices_fraction: 0.5`, `sampler_devices_fraction: 0.5`) [5][9]. No published multi-node throughput benchmark was found in the sources read for this card; the multi-host mechanism is documented, not benchmarked here.

**Install**: `UV_TORCH_BACKEND=cpu uv pip install maxtext[tpu-post-train]==0.2.3 --resolution=lowest`, then run the post-install console script `install_tpu_post_train_extra_deps` [8]; version 0.2.3 was published 2026-06-12 [10] as tag `maxtext-v0.2.3`, which resolves to commit `3f36aef23439dd5875b48cbf294bbcba1996a726` [11]. Requires Python >=3.12, and the docs state MaxText "is only tested on Linux during releases" [8]; license is Apache-2.0 [1]. `pyproject.toml` at that commit declares zero unconditional base dependencies (`dependencies = []`) — every package is gated behind an extra [12]. The `tpu-post-train` extra's resolved requirements file at that same commit pins `torch==2.11.0+cpu` and `torchvision==0.26.0+cpu` (hard pins, both CPU wheels), and floors `transformers>=5.11.0`, `flax>=0.12.4`, `jax>=0.10.1`, `jaxlib>=0.10.1`, `ray>=2.55.1`, `accelerate>=1.13.0`, `peft>=0.19.1`, `google-tunix>=0.1.3`, `datasets>=5.0.0` [13] — this exactly matches what PyPI's own JSON metadata reports for version 0.2.3 [10]. vLLM is notably NOT a pinned dependency anywhere in that requirements file (zero matches for "vllm"); instead, the `install_tpu_post_train_extra_deps` script sets `VLLM_TARGET_DEVICE=tpu` and `UV_TORCH_BACKEND=cpu` and runs `uv pip install {repo_root}/maxtext/integration/vllm --no-deps`, installing vLLM from a MaxText-vendored local directory rather than a versioned PyPI release [14]. No CUDA/hardware minimum is stated on the install page beyond the per-extra choice of `tpu`, `cuda12`, or `tpu-post-train` [8]. Note on pinning: the shortlist row's screening commit is `f26b2547956db2055f1517eddd35ea13d78caed4` (2026-08-05), and the repository's actual newest push at card-writing time is a later commit still (2026-08-11) — both are well ahead of the `maxtext-v0.2.3` release that the install line above actually delivers, so the dependency floors in this field are read at the release commit, not at the screening commit [11][15][16].

**Maintained by**: Google, under the `AI-Hypercomputer` GitHub organization [1]; not archived, with the repository's most recent push recorded in the shortlist row as 2026-08-05 and a later push observed directly at 2026-08-11, i.e. actively developed past the last tagged release [15][16]. The README's dated news section records ongoing changes, e.g. an April 14, 2026 entry announcing removal of legacy `MaxText.*` post-training shims [1].

## Quick start

There is no single-file "hello world" script; MaxText's smallest complete runs are the tutorial CLI invocations below, each against a real HF-hosted model checkpoint and a real dataset [2][3].

SFT, from the SFT tutorial (env vars `MODEL`, `BASE_OUTPUT_DIRECTORY`, `RUN_NAME`, `STEPS`, `PER_DEVICE_BATCH_SIZE`, `DATASET_NAME`, `TRAIN_SPLIT`, `TRAIN_DATA_COLUMNS` set beforehand) [2]:

```bash
python3 -m maxtext.trainers.post_train.sft.train_sft \
    run_name=${RUN_NAME?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL?} \
    load_parameters_path=${MAXTEXT_CKPT_PATH?} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE?} \
    steps=${STEPS?} \
    hf_path=${DATASET_NAME?} \
    train_split=${TRAIN_SPLIT?} \
    train_data_columns=${TRAIN_DATA_COLUMNS?} \
    profiler=xplane
```

GRPO, from the RL tutorial, fine-tuning Llama3.1-8B-IT on GSM8K on a single TPU host [3]:

```bash
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  chips_per_vm=${CHIPS_PER_VM?}
```

`MAXTEXT_CKPT_PATH` must be a MaxText/Orbax-format checkpoint, either an existing one or a Hugging Face checkpoint converted with MaxText's own converter [3].

## Start it

- One TPU VM, one process is the base form: the commands above, as-is, on a single-host TPU VM such as v6e-8 or v5p-8 [3].
- Multi-host runs go through GKE via Cluster Toolkit (`gcluster`) or XPK, or through Google's Pathways layer, per the docs' separate "Run MaxText" pages for each launcher and the RL config's `use_pathways: true` flag [5][9]; the RL tutorial has a dedicated "Reinforcement Learning on Multi-Host TPUs" page for this case, not read in full for this card [3].
- GRPO/GSPO split available chips between a trainer role and a sampler (generation) role: the RL config's `trainer_devices_fraction` and `sampler_devices_fraction` each default to 0.5, with `num_trainer_slices`/`num_samplers_slices` at -1 (auto) and `rollout_tensor_parallelism`/`rollout_expert_parallelism` at 1 [5].
- Batch arithmetic for RL: `batch_size` (default 1) and `num_batches` (default 4) set the outer training loop shape; `train_micro_batch_size` and `rollout_micro_batch_size` default to -1 (auto-derived) [5]. GRPO additionally multiplies each prompt by `num_generations` (default 2) samples per rollout, and `num_iterations` (default 1) controls PPO-style reuse of each rollout batch [5].
- Switch method by selecting `loss_algo: 'grpo'` or `loss_algo: 'gspo-token'` in `rl.yml`, or by passing `loss_algo=gspo-token` on the command line [3][5].
- Config surface: `src/maxtext/configs/post_train/{sft,dpo,rl,distillation}.yml`, each layered under MaxText's shared `base.yml` and overridable on the CLI as `key=value` [9][5]. The RL config sets `weight_dtype: 'bfloat16'` and turns on activation offloading for `decoder_layer_input`, `query_proj`, `key_proj`, `value_proj` by default — a deliberate memory/precision choice for the trainer+sampler split, not a generic default [5].
- Out-of-memory first aid documented specifically for the RL/vLLM generation side: `hbm_utilization_vllm` (default 0.72) caps the fraction of TPU HBM vLLM may use for generation, and `swap_space_vllm_gb` (default 2) sets vLLM's CPU swap space [5]. On the training side, `remat_policy: 'custom'` plus the offload flags above are the RL config's own memory-saving defaults [5]; no separate library-published OOM checklist beyond these config fields was found in the sources read for this card.

## Watch it

This section covers only the mechanics of what gets logged and where; what a GRPO/GSPO/DPO/SFT signal means for training health is method-specific and not restated here.

- Metrics logging is wired through Tunix, not a pluggable `report_to` list: `train_rl.py` constructs a `tunix.sft.metrics_logger.MetricsLoggerOptions(log_dir=trainer_config.tensorboard_dir, flush_every_n_steps=trainer_config.log_period)` and passes it into the Tunix trainer, i.e. metrics land in the run's `tensorboard_dir` as TensorBoard event files, flushed every `log_period` steps (default 20 in `rl.yml`) [3][5]. The exact per-step scalar names (loss, KL, reward, clip-ratio, entropy, or similar) are defined inside Tunix's own `GRPOLearner`/`DPOTrainer` classes; MaxText's own docs pages read for this card do not enumerate them, and the general "Understand logs and metrics" guide covers only the pre-training trainer's log line (`completed step, seconds, TFLOP/s/device, Tokens/s/device, total_weights, loss`) and TFLOP/MFU/throughput formulas, not RL-specific fields [17].
- A separate optional performance-metrics path exists: setting `enable_tunix_perf_metrics: true` (default `false` in `rl.yml`) turns on `tunix.perf.metrics.PerfMetricsConfig`, with a custom export function built from the run's cluster config; `train_rl.py` logs a warning and skips this if the `tunix.perf` modules are unavailable [5][3].
- MaxText adds its own lighter intermediate-evaluation logging on top of Tunix's, specifically because Tunix's built-in evaluation is impractical for frequent monitoring: the RL hooks module's own docstring states that Tunix's `eval_every_n_steps` "is silently dead unless an `eval_ds` is passed to `trainer.train()`", and that even then Tunix's default evaluation "re-runs the full GRPO rollout (`num_generations` sampled per prompt), which is ~3hr/eval and impractical for trajectory monitoring" [18]. MaxText's `RLTrainingHooks` instead fires every `eval_interval` outer steps (default 10 in `rl.yml`) and logs a single line: `Intermediate Eval (step=<n>): corr=<c>, total=<t>, accuracy=<a>%, partial_accuracy=<p>%, format_accuracy=<f>%, mean_reward=<m>` [18][5]. `accuracy` is the percentage of samples with an exactly correct final numeric answer, `partial_accuracy` allows answers within a tolerance, and `format_accuracy` checks the reasoning/answer tag format, per the evaluation module's own docstring [19].
- Sample-level generations are visible in this same eval path: `evaluate_rl.py` logs the evaluation question, the list of acceptable answers, and the raw sampled responses for each evaluated prompt via `max_logging.log` [19].
- Evaluation-during-training fields on the RL config: `eval_interval` (default 10 outer steps), `num_test_batches` (default 5), `eval_batch_size` (-1, auto), `num_eval_passes` (default 1), and `eval_sampling_strategy` choosing among three named presets (`greedy`, `standard`, `liberal`, each with its own temperature/top-k/top-p) [5].
- Stopping: no RL-specific stopping rule, threshold, or patience field is published. Search performed 2026-08-11 over `rl.yml` (the RL config surface) [5], `train_rl.py`, `utils_rl.py`, and `evaluate_rl.py` (the RL trainer and eval implementation) [3][19][20]: the only threshold-named field found, `gradient_clipping_threshold`, clips the gradient global norm for optimizer stability and is not a stopping rule; `stop_strings` is a vLLM generation stop-string list, not a training stopping condition [5][3]. No early-stopping callback equivalent to a generic trainer's patience/threshold pair was found in these files.

## Save it

- Checkpoint format follows two independent axes documented on the Checkpoints reference page: with vs without training state (optimizer state), and stacked vs unstacked layer weights (stacking is required for `jax.lax.scan`-compiled models) [21]. "For saving and resuming training, MaxText uses Stacked Training Checkpoints by default" — weights plus optimizer state, stacked; "we treat Stacked Inference Checkpoints as the default format for checkpoint conversion" — weights only, stacked [21].
- The `scan_layers` flag controls stacking (`true` recommended for training, `false` for inference or heterogeneous-layer models); MaxText auto-loads `scan_layers` from a checkpoint's saved metadata on resume via `load_parameters_path` unless it is explicitly overridden, and raises a `ValueError` if an explicit override conflicts with the checkpoint's actual format [21].
- Save-during-training flags, per the same reference page: `base_output_directory` (the GCS bucket root), `enable_checkpointing` (bool), `async_checkpointing` (overlaps checkpoint writes with training), `checkpoint_period` (step interval between saves) [21]; the RL config's own values are `enable_checkpointing: true`, `async_checkpointing: false`, `checkpoint_period: 50`, `max_num_checkpoints_to_keep: 10` (retention count, not a state-dropping flag) [5]. `train_rl.py` wires `checkpoint_period` and `max_num_checkpoints_to_keep` directly into an Orbax `ocp.CheckpointManagerOptions(save_interval_steps=..., max_to_keep=...)` [3]. No flag equivalent to trl's `save_only_model` (one that keeps weights but silently drops optimizer state on an otherwise-full save) was found on the Checkpoints reference page or in `rl.yml`; the only route to a params-only checkpoint documented is the separate Stacked/Unstacked Inference Checkpoint format itself, produced by checkpoint conversion, not by a retention flag on a training-mode save [21].
- Emergency checkpointing (local-disk snapshots for fast recovery from preemption) is a separate mechanism: `enable_emergency_checkpoint`, `enable_autocheckpoint` (save on SIGTERM), `local_checkpoint_directory`, `local_checkpoint_period` [21]; a dedicated "Checkpointing" how-to guide with GCS-bucket-based, emergency, multi-tier, and conversion-utility subpages exists but was not read in detail for this card [21].
- SFT's own tutorial states the practical save location directly: "Your fine-tuned model checkpoints will be saved here: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`" [2].
- Resume by pointing `load_parameters_path` (or the training-state equivalent for a full-state resume) at a saved checkpoint directory; the automatic `scan_layers` resolution described above is what makes this safe when the run's explicit config doesn't restate it [21].
- LoRA/PEFT-adapter saving exists as its own tutorial pages ("LoRA Fine-tuning on single-host TPUs", "Native LoRA on single-host TPUs") that were not read for this card; do not assume MaxText's adapter save/reload contract mirrors any other library's without checking those pages directly.
- Loader handoff: whether a full Stacked Training Checkpoint or a converted Stacked/Unstacked Inference Checkpoint loads directly into a given evaluator is that evaluator's contract, not MaxText's — check the target loader's own checkpoint-format expectations before assuming interchangeability with the four MaxText formats above.

## Find it in the docs

The docs live at ReadTheDocs; this card teaches the lookup, not a mirror of the content.

- Address pattern: `https://maxtext.readthedocs.io/en/latest/<section>/<page>.html`, e.g. `https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl.html` and `https://maxtext.readthedocs.io/en/latest/reference/core_concepts/checkpoints.html`, both fetched and confirmed loading for this card [3][21]. `<section>/<page>` mirrors the GitHub source path under `docs/` with `.md` swapped for `.html` — this is more reliable than guessing a human-readable slug: three guessed slugs for the logs-and-metrics page (`how-to-guides/monitoring-debugging/...`, `guides/monitor/...`, `how_to_guides/...`) all 404ed, while the GitHub Contents API path `docs/guides/monitoring_and_debugging/understand_logs_and_metrics.md` resolved directly to the working page [17].
- Question-to-page map, read off the site's own left-nav sidebar [3]: install -> "Install MaxText"; first run -> "Getting Started"; SFT -> "SFT on single-host TPUs" / "SFT on multi-host TPUs"; preference optimization -> "Preference Optimization (DPO & ORPO) on Single-Host TPUs"; RL -> "Reinforcement Learning on single-host TPUs" / "...on Multi-Host TPUs" (plus two model-specific multi-host RL tutorials, for Qwen3-30b-a3b-base and GPT-OSS 20B); distillation -> "Knowledge distillation"; LoRA -> three separate tutorials (single-host, "Native LoRA", multi-host); checkpoint mechanics -> Reference documentation > Core concepts > "Checkpoints"; logging/metrics/debugging -> How-to guides > Monitoring and debugging (eight subpages, including "Understand logs and metrics", "Troubleshooting: Megascale hangs", "Profiling with XProf", "ML Goodput measurement") [3][17].
- Runnable references beyond the docs: the GitHub repo's own `src/maxtext/trainers/post_train/{sft,dpo,rl,distillation}` scripts each carry a real example CLI invocation in their module docstring (SFT's cites the `HuggingFaceH4/ultrachat_200k` dataset; DPO's cites `Anthropic/hh-rlhf`) [2][22][3]; the RL config's comments list five datasets it is built to support: `openai/gsm8k`, `nvidia/OpenMathInstruct-2`, `nvidia/OpenMathReasoning`, `open-r1/OpenR1-Math-220k`, `bethgelab/CuratedThoughts` [5]. `tests/post_training/integration/` and `tests/post_training/unit/` hold correctness tests for SFT, DPO, RL, and distillation, useful as further working examples [23].
- Community layer: not identified for this card. No curated community-tutorials page equivalent to trl's `community_tutorials` was found in the sidebar sections read (Getting Started, Tutorials, Run MaxText, How-to guides, Reference documentation, How to Contribute) [3]; recurring third-party blogs were not searched for this card.
- No official MCP endpoint for querying MaxText's docs was found in the sources read for this card.
- Honest boundary: MaxText's RL/GRPO path is documented and tested only for TPU generation via vLLM ("Currently, this option should also be used for running vllm_decode on TPUs") [8]; the `cuda12` extra exists for GPU installs but the RL tutorial gives no GPU-specific generation instructions, so treat RL-on-GPU as unverified from the docs read here. No maintainer-confirmed GitHub issue traps are cited in this card — a closed-issue search was not performed for this pass.

## Sources

All GitHub pages are read at the commits named inline; ReadTheDocs pages are the unpinned `latest` build, fetched 2026-08-11 unless noted. Ecosystem tools reached only by name through MaxText's own docs (Flax, Orbax, Optax, Grain, Pathways, Cluster Toolkit, XPK, math_verify) are not separately enumerated as references.

[1] maxtext README, main branch. https://github.com/AI-Hypercomputer/maxtext (raw: https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/README.md). Fetched 2026-08-11.

[2] SFT training script docstring and hooks, main branch. https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/trainers/post_train/sft/train_sft.py and .../sft/hooks.py. Fetched 2026-08-11.

[3] "Reinforcement Learning on single-host TPUs" tutorial. https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl.html. Fetched 2026-08-11. Also: `train_rl.py`, main branch. https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/trainers/post_train/rl/train_rl.py. Fetched 2026-08-11.

[4] Docs left-nav sidebar (page title "Preference Optimization (DPO & ORPO) on Single-Host TPUs"), captured from [3]. Fetched 2026-08-11.

[5] `rl.yml`, main branch. https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/configs/post_train/rl.yml. Fetched 2026-08-11.

[6] GitHub Contents API listing of `src/maxtext/trainers/post_train/distillation`. https://api.github.com/repos/AI-Hypercomputer/maxtext/contents/src/maxtext/trainers/post_train/distillation. Fetched 2026-08-11.

[7] GitHub Contents API listing of `src/maxtext/experimental/rl`. https://api.github.com/repos/AI-Hypercomputer/maxtext/contents/src/maxtext/experimental/rl. Fetched 2026-08-11.

[8] "Install MaxText" docs page. https://maxtext.readthedocs.io/en/latest/tutorials/install_maxtext.html. Fetched 2026-08-11.

[9] "Getting Started" docs page. https://maxtext.readthedocs.io/en/latest/getting_started.html. Fetched 2026-08-11.

[10] maxtext on PyPI, JSON API. https://pypi.org/pypi/maxtext/json. Fetched 2026-08-11.

[11] GitHub API resolution of tag `maxtext-v0.2.3` to its commit. https://api.github.com/repos/AI-Hypercomputer/maxtext/git/refs/tags/maxtext-v0.2.3. Fetched 2026-08-11.

[12] `pyproject.toml` at commit `3f36aef23439dd5875b48cbf294bbcba1996a726` (the `maxtext-v0.2.3` release). https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/3f36aef23439dd5875b48cbf294bbcba1996a726/pyproject.toml. Fetched 2026-08-11.

[13] `tpu-post-train-requirements.txt` at commit `3f36aef23439dd5875b48cbf294bbcba1996a726`. https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/3f36aef23439dd5875b48cbf294bbcba1996a726/src/dependencies/requirements/generated_requirements/tpu-post-train-requirements.txt. Fetched 2026-08-11.

[14] `install_post_train_extra_deps.py` at commit `3f36aef23439dd5875b48cbf294bbcba1996a726`. https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/3f36aef23439dd5875b48cbf294bbcba1996a726/src/dependencies/scripts/install_post_train_extra_deps.py. Fetched 2026-08-11.

[15] Shortlist row's screening commit, GitHub API. https://api.github.com/repos/AI-Hypercomputer/maxtext/commits/f26b2547956db2055f1517eddd35ea13d78caed4. Fetched 2026-08-11.

[16] GitHub API, repository's newest commit on the default branch. https://api.github.com/repos/AI-Hypercomputer/maxtext/commits/main. Fetched 2026-08-11.

[17] "Understand logs and metrics" how-to guide. https://maxtext.readthedocs.io/en/latest/guides/monitoring_and_debugging/understand_logs_and_metrics.html. Fetched 2026-08-11.

[18] `hooks.py` (RLTrainingHooks docstring and intermediate-eval log line), main branch. https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/trainers/post_train/rl/hooks.py. Fetched 2026-08-11.

[19] `evaluate_rl.py`, main branch. https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/trainers/post_train/rl/evaluate_rl.py. Fetched 2026-08-11.

[20] `utils_rl.py`, main branch. https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/trainers/post_train/rl/utils_rl.py. Fetched 2026-08-11.

[21] "Checkpoints" core-concepts reference page. https://maxtext.readthedocs.io/en/latest/reference/core_concepts/checkpoints.html. Fetched 2026-08-11.

[22] `train_dpo.py` docstring, main branch. https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/trainers/post_train/dpo/train_dpo.py. Fetched 2026-08-11.

[23] GitHub Contents API listing of `tests/post_training/integration` and `tests/post_training/unit`. https://api.github.com/repos/AI-Hypercomputer/maxtext/contents/tests/post_training/integration and .../tests/post_training/unit. Fetched 2026-08-11.
