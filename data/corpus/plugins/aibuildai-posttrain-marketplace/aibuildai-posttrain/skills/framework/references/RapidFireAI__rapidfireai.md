# rapidfireai

A hyperparallel experiment-execution wrapper around Hugging Face TRL that runs and interactively controls many SFT/DPO/GRPO configs concurrently, one data shard at a time, on shared GPUs.

RapidFire AI's own docs describe it as "a new AI experiment execution framework that transforms your LLM pipeline customization from slow, sequential processes into rapid, intelligent workflows with hyperparallelized execution, dynamic real-time experiment control, and automatic backend optimization" [1]. It is built and maintained by RapidFire AI Inc. [2], and lives at https://github.com/RapidFireAI/rapidfireai [3]. Its API is not its own trainer math: an `Experiment` object is constructed in `"fit"` or `"evals"` mode, and `experiment.run_fit(param_config, create_model_fn, train_dataset, eval_dataset, num_chunks, ...)` launches one or many hyperparameter configs, each config an `RFSFTConfig`/`RFDPOConfig`/`RFGRPOConfig` that the docs describe collectively as "thin wrappers around the corresponding APIs of Hugging Face's TRL libraries," and individually (e.g. "This is a wrapper around SFTConfig in HF TRL") as preserving that trainer's semantics [4][5]. The README separately frames the package as built for "agentic RAG, context engineering, fine-tuning, and post-training of LLMs and other DL models" [6].

**When to pick it**: you are already fine-tuning or post-training with Hugging Face TRL (SFT/DPO/GRPO) on one machine (single GPU up to several GPUs) and want to compare many hyperparameter configs side by side, with the ability to stop, resume, or clone-and-modify individual runs while they train, rather than launching them sequentially or as separate jobs [1][6]. Its own docs describe "cross-GPU model partitioning (within a machine)" via FSDP, not multi-node training, so it is not documented as a cluster-scale, multi-node distributed-training engine [5]. It is also not exclusively a post-training library — the same package and CLI cover RAG/context-engineering evaluation as a separate `"evals"` mode with its own config and data-sharding surface, which this card does not cover in depth [7][1].

**Methods it ships**: `RFSFTConfig`, `RFDPOConfig`, and `RFGRPOConfig` — each documented as "a wrapper around" TRL's `SFTConfig`, `DPOConfig`, and `GRPOConfig` respectively, adding only the ability to give any individual knob a `List(...)` or `Range(...)` of values so one dictionary expands into a config group, and otherwise preserving that TRL trainer's semantics [5]. `RFGridSearch(configs, trainer_type)` and `RFRandomSearch(configs, trainer_type, num_runs, seed=42)` turn a knob dictionary into a config group — grid search rejects any `Range()`-valued knob, random search IID-samples `List()`/`Range()`/fixed values [8]. `RFLoraConfig` is documented the same way, as a wrapper around PEFT's `LoraConfig` with the same List/Range extension [9]. Method definitions, math, and training-signal semantics are not restated here — the SFT/DPO/GRPO cards carry them; this card covers only what RapidFire changes about running them.

**Scale it handles**: single GPU up to several GPUs on one machine. GPU allocation for `"fit"` mode is per-config via the `num_gpus` argument of `run_fit()` (default 1) or an `RFModelConfig`, not an experiment-wide auto-allocation — that auto-allocation (one actor per GPU, or a CPU-count tier) applies only in `"evals"` mode [10]. For a model too large for one GPU, `RFSFTConfig` and `RFDPOConfig` take `fsdp` / `fsdp_config` dict arguments mirroring standard HF FSDP settings for cross-GPU partitioning within a machine [5]; `RFGRPOConfig` does not yet support this — "out-of-the-box support for FSDP for GRPO is still in the works" [5][11]. DeepSpeed-based multi-GPU partitioning is planned but not available as of this writing [11]. No multi-node launch path is documented for `"fit"` mode. RapidFire's own concurrency mechanism is shard-based scheduling within a `run_fit()` call: the training/eval data is split into `num_chunks` shards (recommended at least 4), and all configs in the launched group are scheduled one shard at a time so early results appear for every config together rather than one config finishing before the next starts; the docs report this swap/adapter-cache overhead as "minimal, less than 5% of the runtime, as per our measurements," a self-reported figure with no comparison baseline published [12][10].

**Install**: `pip install rapidfireai`; PyPI version 0.16.1, released 2026-06-25 (GitHub tag `v0.16.1`, commit `beca2857e3e2d757148adbb01bed171905b7d17a`); requires Python >=3.12; Apache-2.0 [2][13]. The base package pulls in none of torch/transformers/trl/peft/vllm/ray — those arrive only when you run `rapidfireai init`, which shells out to `uv pip install -r <requirements file>` [14]. For fine-tuning/post-training, `rapidfireai init --train` on a non-Colab machine installs from `setup/fit/requirements-local.txt` at this release: `trl==0.21.0` (exact pin, load-bearing), `transformers>=4.56.1`, `peft>=0.17.0,<0.19.0`, `ray<=2.49.0`, `datasets>=3.6.0`, `mlflow>=3.11.1`, plus a CUDA-detected torch/torchvision/torchaudio wheel selection (e.g. CUDA 12.9+ installs torch 2.8.0 on the `cu129` wheel index; CPU-only or undetected CUDA falls back to CPU wheels); `cli.py`'s `install_packages()` function only appends the matching `vllm==<version>` package (versions 0.7.3-0.11.0 depending on detected CUDA) when running in `"evals"` mode, so `--train`, which sets `evals=False`, does not install vLLM [15][16]. This differs from the `local_fit` extra pinned in `pyproject.toml` at the same release (`ray==2.44.1` exact, `transformers>=4.56.1,<5.0.0`) — that extra is reachable via `pip install rapidfireai[local_fit]` directly, but `rapidfireai init --train` is the documented path and installs the `requirements-local.txt` set above, not the pyproject extra [17][14]. README prerequisites state an NVIDIA GPU with 7.x or 8.x Compute Capability, NVIDIA CUDA Toolkit 11.8+, Python 3.12.x, and PyTorch 2.8.0+ "with corresponding forward compatible prebuilt CUDA binaries" as the hardware floor [1].

**Maintained by**: RapidFire AI Inc. [2]; the GitHub repository shows 168 stars (not a ranking signal) and its most recent push was 2026-08-03, weeks after the 0.16.1 release used for the pins above [3]. The docs sidebar labels the current build "RapidFire AI 0.16.1", and the Known Issues and Coming Soon pages list active, dated work in progress: FSDP support for GRPO (FSDP already covers SFT and DPO) is "coming soon", DeepSpeed-based multi-GPU model partitioning is separately planned "soon", and the IC Ops page has its own "Coming Soon: Templated Automation of IC Ops" section [11].

## Quick start

From the docs' `create_model_fn` reference page, quoted from its SFT-tutorial example, this is the mandatory model-construction function every `run_fit()` call needs; it dispatches on the model type given in the config and returns the `(model, tokenizer)` pair `run_fit` requires [18]:

```python
def sample_create_model(model_config):
    """Function to create model object for any given config; must return tuple of (model, tokenizer)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
    model_name = model_config["model_name"]
    model_type = model_config["model_type"]
    model_kwargs = model_config["model_kwargs"]
    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    elif model_type == "seq2seq_lm":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    elif model_type == "masked_lm":
        model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
    elif model_type == "custom":
        # Handle custom model loading logic, e.g., loading your own checkpoints
        # model = ...
        pass
    else:
        # Default to causal LM
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return (model, tokenizer)
```

The docs' Experiment page gives the matching launch form — construct an `Experiment` in `"fit"` mode and pass this function to `run_fit`, alongside the config group, datasets, and chunk count [4]:

```python
experiment = Experiment(experiment_name="exp1-chatqa", mode="fit")
experiment.run_fit(
    config_group,
    sample_create_model,
    train_dataset,
    eval_dataset,
    num_chunks=4,
    seed=42,
)
# Started 4 worker processes successfully ...
```

A single-config `RFSFTConfig`, from the same trainer-configs page — this is what one entry of `config_group` above is built from [5]:

```python
RFSFTConfig(
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=25,
    fp16=True,
    save_strategy="epoch",
)
```

CLI forms for install and first run, from the README's Get Started block [1]:

```bash
pip install rapidfireai
hf auth login --token YOUR_TOKEN
rapidfireai init --train
rapidfireai start
```

## Start it

- One machine, one or more GPUs: `rapidfireai init --train && rapidfireai start` brings up the dispatcher, dashboard, and (outside Colab) a Ray backend; your training script then calls `Experiment(..., mode="fit")` and `run_fit(...)` from a notebook or process on that same machine [1][10]. There is no separate multi-node launcher command documented for `"fit"` mode — `run_fit`'s only concurrency and placement controls are `num_chunks` (shard count) and `num_gpus` (GPUs per config) [10].
- Concurrency is expressed as `num_chunks`, not as a launcher flag: `run_fit(..., num_chunks=4, ...)` splits the datasets into 4 shards and schedules every config in `param_config` on one shard at a time, so all configs report partial results together; the docs recommend at least 4 chunks, trading a "relatively minor" swap overhead for earlier cross-config comparison [10][12].
- Effective batch size is the underlying TRL/`transformers` arithmetic per config — `per_device_train_batch_size x gradient_accumulation_steps` (times device count under FSDP) — since `RFSFTConfig`/`RFDPOConfig`/`RFGRPOConfig` "preserve all semantics" of the wrapped TRL config; RapidFire does not change this arithmetic [5].
- The Config surface is the wrapped TRL config's full field set plus the List()/Range() multi-config extension; RapidFire's docs do not claim to change any TRL default — the `fp16=True` and other values seen in the quick-start example are the tutorial's own choices, not library-imposed defaults [5]. Cross-GPU model fit for a single large model is set on the config itself: `RFSFTConfig(fsdp="full_shard auto_wrap", fsdp_config={...})` and the same for `RFDPOConfig`; `RFGRPOConfig` has no such argument yet — GRPO's policy and reference model must both fit on one GPU [5][11].
- Multiple hyperparameter combinations are produced with `List([...])`/`Range(start, end, dtype=...)` on individual knobs inside one `RFSFTConfig`/`RFDPOConfig`/`RFGRPOConfig`, or by passing a knob dictionary through `RFGridSearch(configs, trainer_type="SFT")` (no `Range()` allowed) or `RFRandomSearch(configs, trainer_type="SFT", num_runs=N, seed=42)` (IID sampling over List/Range/fixed values) to `run_fit`'s `param_config` argument [8][5].
- Out-of-memory first aid is not separately documented by RapidFire; because every fit-mode trainer is a thin TRL wrapper, the applicable knobs are TRL's own (lower `per_device_train_batch_size`, raise `gradient_accumulation_steps`, use `fsdp`/`fsdp_config` for SFT/DPO) — this card does not restate TRL's own OOM guidance [5].
- A documented operational trap for live control: Clone-Modify, the IC Op that clones a running config with edits, is not validated at submission time — "RapidFire AI does not currently block these edits at submission time; invalid edits cause the run to fail when the model, LoRA adapter, or retrieval index is loaded, or produces a run that is not comparable to its parent" [19]. Warm-starting a clone from its parent's weights additionally requires an identical neural architecture (base model, dtype, quantization, LoRA rank and `target_modules`); editing those on a warm clone errors out at load time, while optimization knobs (learning rate, scheduler, batch size, gradient accumulation, `max_grad_norm`), LoRA's `lora_alpha`/`lora_dropout`, and method knobs (DPO `beta`; GRPO `num_generations`, reward weighting, KL coefficient) are documented as safe to edit on a warm clone [19].

## Watch it

This section is the mechanics only; what a metric shape means for SFT/DPO/GRPO is on that method's own card, not here.

- **Enable it**: the dashboard is on by default. `rapidfireai start` serves a fork of MLflow on port 8852 (`RF_MLFLOW_PORT`), on by default on local/cloud machines and off by default on Colab, where an in-notebook table is used instead; TensorBoard and Trackio are additional, opt-in backends toggled with `rapidfireai start --tracking-backends [mlflow|tensorboard|trackio]` or the `RF_MLFLOW_ENABLED`/`RF_TENSORBOARD_ENABLED`/`RF_TRACKIO_ENABLED` environment variables [20]. The dashboard's "Chart" tab plots "loss on the training set and evaluation set, as well [as] all named metrics returned in your `compute_metrics()` function in the trainer config" [20] — RapidFire does not publish a fixed per-trainer metric-name list the way TRL's own docs do; what you see beyond train/eval loss is exactly whatever keys your own `compute_metrics_fn` returns [20][21].
- **Metric names**: RapidFire itself surfaces only `loss` (train and eval) as a named metric; everything else on the Chart view is whatever your `compute_metrics_fn(eval_preds) -> Dict[str, float]` returns, e.g. a tutorial example returning ROUGE/BLEU scores via the `evaluate` library [21]. `get_results()` returns a DataFrame with "run ID, step number, loss, and one column per metric plot displayed on the dashboard" — the same set, read back programmatically [22].
- A documented, non-obvious axis-semantics trap: on the Chart view, the minibatch-level "Step" axis is "absolute number of minibatches seen by that run," so runs with different `batch_size` will not line up until the end, and the epoch-level "Step" axis is likewise "absolute number of epochs seen by that run" and will not line up across runs with different epoch counts — the docs state this is "not a bug but the expected correct behavior" [20].
- Two other message logs run alongside the metrics tabs: an "Experiment Log" tab records every API operation, and an "Interactive Control Log" tab records every IC Op; the full experiment log is also written locally to a file named `rapidfire.log` [20].
- **Sample-level logging**: not documented by RapidFire's own pages read for this card; generation/sample inspection is a TRL-level feature (e.g. GRPO's `log_completions`) that RapidFire's thin-wrapper configs would pass through unchanged, but no RapidFire-specific sample-logging surface is named on the dashboard, trainer-configs, or fit-functions pages [5][20].
- **Evaluate during training**: `run_fit()` takes an `eval_dataset` argument directly (not optional-by-name like TRL's own `eval_strategy="no"` default), and the wrapped TRL config's own `eval_strategy`/`eval_steps` fields (e.g. `eval_strategy="steps", eval_steps=25` in the quick-start example) control cadence, since RapidFire preserves TRL's config semantics [10][5].
- **Stopping**: no RL- or training-specific stopping rule, threshold, or patience value is published by RapidFire. The pages searched for this card — Trainer Configs [5], the Dashboard page [20], the IC Ops page [19], and Known Issues [11] — name no automatic stopping mechanism; the only "stop" is the manual IC Op, which a human (or, per the roadmap, a future semi-automated policy "in the near future") must trigger explicitly [11][19].

## Save it

- Checkpoints land under `<experiment_path>/<experiment_name>/runs/<run_id>/checkpoints/`, with three fixed subpaths defined in `DataPath`: `initial_checkpoint/`, `final_checkpoint/`, and `intermediate_checkpoints/checkpoint-<completed_steps>/` — one folder per chunk save, named after the HF Trainer convention, with a legacy unsuffixed `checkpoint/` folder used only as a fallback when `completed_steps` is 0 [23]. Both the screening commit and the 0.16.1 release commit of this file are byte-identical, so this layout is current as of the release [23].
- Inside a checkpoint folder, `save_checkpoint_to_disk()` writes the model via `trainer.model.save_pretrained(checkpoint_path)` (or, for PEFT+FSDP, a gathered adapter state dict saved as `adapter_model.bin` plus the adapter config via `peft_config[...].save_pretrained(...)`), plus `trainer_state.json`, and — only when present — `optimizer.pt` and `scheduler.pt`, and (if the trainer exposes RNG state) `rng_state.pth` [24]. When RapidFire trains with a LoRA adapter, the checkpoint holds only the trained adapter weights, not the base model — "when you use LoRA adapters, RapidFire AI saves only the trained adapters in the checkpoints of the runs, not the base models," a design the docs also cite as a reason base-model `.cache` folders are safe to delete between experiments [25]. An adapter-only checkpoint is therefore not a loadable full model on its own; it must be paired with the base model to reload, following PEFT's own adapter-plus-base-model reload contract (not restated here — see the PEFT/TRL cards for that contract).
- Retention: `save_strategy`/`save_steps`/`save_total_limit`/`save_only_model` are the wrapped TRL config's own fields, preserved unchanged by `RFSFTConfig`/`RFDPOConfig`/`RFGRPOConfig`; RapidFire's docs do not add or override this contract, so its behavior (including what `save_only_model=True` drops) is TRL's, not restated here [5].
- Programmatic access: `experiment.get_runs_info()` returns a DataFrame with one row per run, including `run_id`, `status`, `completed_steps`, `total_steps`, `warm_started_from`, and the run's full `config` dictionary, for fit-mode experiments only [26]. `experiment.get_log_file_path(log_type)` returns the path to either the main experiment log or the training log [26].
- Whether an evaluator can load a saved checkpoint directly is a loader question, not a RapidFire one: a full-model checkpoint is a standard `save_pretrained()` directory and loads with `from_pretrained()`; a LoRA checkpoint is an adapter directory and needs the base model plus a PEFT loader to reconstitute a usable model [24][25].

## Find it in the docs

The docs are the live source; this section teaches the lookup, not the content.

- Address pattern: `https://oss-docs.rapidfire.ai/en/latest/<page>.html` — checked 2026-08-12, the version shown on every fetched page is "RapidFire AI 0.16.1," matching the current PyPI release; no alternate version-tag path (e.g. `/en/0.16.1/`) was tested [1]. `<page>` slugs seen while researching this card: `overview`, `sftrft` (a navigation stub whose real content is split across `trainers`, `models`, and `fitfunctions`), `configs` (Multi-Config Specification), `trainers` (`RFSFTConfig`/`RFDPOConfig`/`RFGRPOConfig`), `models` (`RFLoraConfig`/`RFModelConfig`), `fitfunctions` (`create_model_fn`/`compute_metrics_fn`/`sample_formatting_fn`/`reward_function`), `experiment` (the `Experiment` class), `dashboard`, `icops`, `issues`, `troubleshooting`, `difference`, `tutorials`, `glossary` [1].
- Question-to-slug map: "how do I run many configs at once" -> `configs`; "what does my SFT/DPO/GRPO config accept" -> `trainers`; "what does my model/LoRA config accept" -> `models`; "what functions do I have to write" -> `fitfunctions`; "how do I start/stop/resume/clone a run live" -> `icops`; "what's on the dashboard" -> `dashboard`; "why is my run stuck / port in use / ImportError between experiments" -> `troubleshooting` and `issues` [1].
- Runnable references beyond the docs: the release commit's file tree lists complete notebooks under `tutorial_notebooks/fine-tuning/` and `tutorial_notebooks/post-training/`, including `rf-tutorial-sft-chatqa-fsdp-large.ipynb` and `-fsdp-lite.ipynb`, `rf-tutorial-dpo-alignment.ipynb`, and `rf-tutorial-grpo-mathreasoning.ipynb`, plus a Colab SFT/TensorBoard notebook and a Colab RAG (FiQA) notebook under `tutorial_notebooks/rag-contexteng/` [27]. The README separately confirms these Colab notebooks and the general `./tutorial_notebooks/[fine-tuning | post-training]` layout without naming individual files [6]. `rapidfireai init --train && rapidfireai init` (whichever mode) also copies these tutorial notebooks into the working directory via the CLI's `copy_tutorial_notebooks()` step [14].
- Community layer: the README points to the project's Discord for feature requests, and the docs' own "Example Use Case Tutorials" page is the only curated door to worked examples found in the pages read for this card [6]. No official MCP endpoint for querying these docs was found in the pages read for this card.
- Honest boundary, stated where it bites: `"fit"` mode has no documented multi-node launch path, and `RFGRPOConfig` has no FSDP option yet, so a GRPO run whose policy and reference models do not both fit on one GPU is out of scope until DeepSpeed or FSDP support for GRPO ships [11][5]. IC Ops themselves only queue and apply at the next chunk boundary of the affected run, not immediately, and different runs reach their own chunk boundary at different times [19]. The `evaluate`/reward-function/formatting-function surface documented in `fitfunctions` is unrelated to RAG evaluation config, which this card does not cover [7][21].

## Sources

Read 2026-08-12 unless noted. `main`/`latest`-labeled docs pages are the live, mutable Sphinx site at `oss-docs.rapidfire.ai`; source-code citations name the commit read. Method names (SFT, DPO, GRPO) are deliberately cited to nothing here — their defining papers live on the methodology cards.

[1] RapidFire AI docs, "Overview" and site navigation (page slugs, version banner "RapidFire AI 0.16.1"). https://oss-docs.rapidfire.ai/en/latest/overview.html. Fetched 2026-08-12.

[2] rapidfireai on PyPI. https://pypi.org/project/rapidfireai/. Fetched 2026-08-12.

[3] rapidfireai GitHub repository (stars, latest push). https://github.com/RapidFireAI/rapidfireai. Fetched 2026-08-12 (repo API snapshot).

[4] RapidFire AI docs, "API: Experiment" (`Experiment` constructor, `run_fit`/`run_evals` signatures and mode semantics). https://oss-docs.rapidfire.ai/en/latest/experiment.html. Fetched 2026-08-12.

[5] RapidFire AI docs, "API: Trainer Configs" (`RFSFTConfig`/`RFDPOConfig`/`RFGRPOConfig` as thin TRL wrappers, FSDP config examples, GRPO-FSDP gap). https://oss-docs.rapidfire.ai/en/latest/trainers.html. Fetched 2026-08-12.

[6] rapidfireai README.md at release tag v0.16.1, commit `beca2857e3e2d757148adbb01bed171905b7d17a` (quick-start CLI block, tutorial-notebook paths, Documentation link). https://raw.githubusercontent.com/RapidFireAI/rapidfireai/beca2857e3e2d757148adbb01bed171905b7d17a/README.md. Fetched 2026-08-12.

[7] RapidFire AI docs, "APIs for RAG and Context Engineering" section (named in the site navigation as the evals-mode counterpart to fit mode; not read in depth for this card). https://oss-docs.rapidfire.ai/en/latest/ragcontexteng.html. Fetched 2026-08-12.

[8] RapidFire AI docs, "API: Multi-Config Specification" (`List`, `Range`, `RFGridSearch`, `RFRandomSearch`). https://oss-docs.rapidfire.ai/en/latest/configs.html. Fetched 2026-08-12.

[9] RapidFire AI docs, "API: LoRA and Model Configs" (`RFLoraConfig` as thin PEFT wrapper, `RFModelConfig` fields). https://oss-docs.rapidfire.ai/en/latest/models.html. Fetched 2026-08-12.

[10] RapidFire AI docs, "API: Experiment", Resource Allocation Defaults and Override Semantics / Run Fit sections (`num_gpus`, `num_chunks`, eval_dataset, evals-mode-only auto-allocation). https://oss-docs.rapidfire.ai/en/latest/experiment.html. Fetched 2026-08-12.

[11] RapidFire AI docs, "Known Issues and Updates Coming Soon" (Multi-GPU Model Support: GRPO FSDP gap, DeepSpeed roadmap; ImportError trap and fix; storage-recovery/adapter-only-save note; Semi-Automated IC Ops roadmap). https://oss-docs.rapidfire.ai/en/latest/issues.html. Fetched 2026-08-12.

[12] RapidFire AI docs, "What Makes RapidFire AI Different?" (adaptive execution engine, chunk/shard scheduling, self-reported "less than 5% of the runtime" swap-overhead figure). https://oss-docs.rapidfire.ai/en/latest/difference.html. Fetched 2026-08-12.

[13] rapidfireai GitHub releases and tags (v0.16.1 published 2026-06-25, tag commit `beca2857e3e2d757148adbb01bed171905b7d17a`). https://github.com/RapidFireAI/rapidfireai/releases and https://github.com/RapidFireAI/rapidfireai/tags. Fetched 2026-08-12.

[14] rapidfireai `cli.py` at release commit `beca2857e3e2d757148adbb01bed171905b7d17a` (`install_packages`/`run_init` logic: `uv pip install -r setup/fit/requirements-local.txt` for `init --train` on non-Colab, CUDA-version-dependent torch/vLLM wheel selection, `copy_tutorial_notebooks`). https://raw.githubusercontent.com/RapidFireAI/rapidfireai/beca2857e3e2d757148adbb01bed171905b7d17a/rapidfireai/cli.py. Fetched 2026-08-12.

[15] rapidfireai `setup/fit/requirements-local.txt` at release commit `beca2857e3e2d757148adbb01bed171905b7d17a` (the exact pins installed by `rapidfireai init --train` on a non-Colab machine). https://raw.githubusercontent.com/RapidFireAI/rapidfireai/beca2857e3e2d757148adbb01bed171905b7d17a/setup/fit/requirements-local.txt. Fetched 2026-08-12.

[16] rapidfireai `cli.py` at release commit `beca2857e3e2d757148adbb01bed171905b7d17a`, CUDA-version-to-wheel table (torch/torchvision/torchaudio/vLLM versions selected by detected CUDA major/minor). Same URL as [14]. Fetched 2026-08-12.

[17] rapidfireai `pyproject.toml` at release commit `beca2857e3e2d757148adbb01bed171905b7d17a` (`[project.optional-dependencies]` `local_fit` extra pins, differing from `requirements-local.txt`: `ray==2.44.1`, `transformers>=4.56.1,<5.0.0`). https://raw.githubusercontent.com/RapidFireAI/rapidfireai/beca2857e3e2d757148adbb01bed171905b7d17a/pyproject.toml. Fetched 2026-08-12.

[18] RapidFire AI docs, "API: User-Provided Functions for Run Fit" (`create_model_fn` example from the SFT tutorial notebook). https://oss-docs.rapidfire.ai/en/latest/fitfunctions.html. Fetched 2026-08-12.

[19] RapidFire AI docs, "Dashboard: Interactive Control (IC) Ops" (Stop/Resume/Clone-Modify/Delete semantics, chunk-boundary queuing, unenforced Clone-Modify correctness requirements, editable-vs-architecture-breaking knob lists for warm clones). https://oss-docs.rapidfire.ai/en/latest/icops.html. Fetched 2026-08-12.

[20] RapidFire AI docs, "Metrics Dashboard" (MLflow-fork default, tracking-backends flag and env vars, port/env-var defaults, 4 tabs, Chart-view metric set and axis-semantics note, `rapidfire.log`). https://oss-docs.rapidfire.ai/en/latest/dashboard.html. Fetched 2026-08-12.

[21] RapidFire AI docs, "API: User-Provided Functions for Run Fit", `compute_metrics_fn` section (signature, ROUGE/BLEU tutorial example via the `evaluate` library). Same URL as [18]. Fetched 2026-08-12.

[22] RapidFire AI docs, "API: Experiment", `get_results()` section (returned DataFrame columns). Same URL as [4]. Fetched 2026-08-12.

[23] rapidfireai `rapidfireai/fit/utils/datapaths.py` at release commit `beca2857e3e2d757148adbb01bed171905b7d17a` (`DataPath` classmethods defining the checkpoint directory layout; confirmed byte-identical to the screening-commit version of this file). https://raw.githubusercontent.com/RapidFireAI/rapidfireai/beca2857e3e2d757148adbb01bed171905b7d17a/rapidfireai/fit/utils/datapaths.py. Fetched 2026-08-12.

[24] rapidfireai `rapidfireai/fit/utils/checkpoint_utils.py` at release commit `beca2857e3e2d757148adbb01bed171905b7d17a`, `save_checkpoint_to_disk()` (model/optimizer/scheduler/RNG-state save calls and file names; confirmed byte-identical to the screening-commit version of this file). https://raw.githubusercontent.com/RapidFireAI/rapidfireai/beca2857e3e2d757148adbb01bed171905b7d17a/rapidfireai/fit/utils/checkpoint_utils.py. Fetched 2026-08-12.

[25] RapidFire AI docs, "Known Issues and Updates Coming Soon", Recovering Storage Space section (LoRA adapters save only trained adapter weights, not base models). Same URL as [11]. Fetched 2026-08-12.

[26] RapidFire AI docs, "API: Experiment", `get_runs_info()` and `get_log_file_path()` sections. Same URL as [4]. Fetched 2026-08-12.

[27] rapidfireai repository file tree (Git Trees API) at release commit `beca2857e3e2d757148adbb01bed171905b7d17a`, `tutorial_notebooks/` paths. https://api.github.com/repos/RapidFireAI/rapidfireai/git/trees/beca2857e3e2d757148adbb01bed171905b7d17a?recursive=1. Fetched 2026-08-12.
