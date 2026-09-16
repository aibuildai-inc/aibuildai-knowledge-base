# mlx-tune

An Unsloth-shaped fine-tuning library for Apple Silicon: swap one import to move an SFT/DPO/GRPO script from CUDA+Unsloth to a Mac's unified memory, running on Apple's MLX framework.

**mlx-tune** ships "SFT, DPO, GRPO, Vision, TTS, STT, Embedding, and OCR fine-tuning" natively on MLX, behind an "Unsloth-compatible API" [1]. It is a solo, community project by GitHub user ARahim3, explicitly "not affiliated with Unsloth AI or Apple" [1][2], originally released under the name `unsloth-mlx` and renamed to avoid implying an official Unsloth tie [2]. The API mirrors Unsloth/TRL: `FastLanguageModel.from_pretrained(...)` loads a model, `get_peft_model(...)` attaches LoRA, and a per-method `Trainer`/`Config` pair (`SFTTrainer`/`SFTConfig`, `DPOTrainer`/`DPOConfig`, ...) runs `.train()` [1][3]. It lives at https://github.com/ARahim3/mlx-tune [2].

**When to pick it**: fine-tuning on a Mac with Apple Silicon (the docs list M1 through M5) when you want to prototype with an Unsloth-shaped script before moving to CUDA. The README frames the project explicitly as not a replacement for or competitor to Unsloth, but a bridge for Mac users to prototype locally with fast iteration and then move to cloud NVIDIA GPUs with the original Unsloth for production training [1]. Its own comparison table names the tradeoff: mlx-tune runs on the MLX framework against Unsloth's Triton kernels, unified memory (up to 512GB) against limited VRAM, and its "Best For" cell reads "Local dev, large models" against Unsloth's "Production training" [1]. There is no multi-GPU or multi-node launcher anywhere in the docs or source read for this card - Apple Silicon's unified-memory model has no analogue to a multi-card cluster, so scale here means a bigger single Mac, not more machines.

**Methods it ships**: the project status table marks every one of the following "Stable" as of the v0.6.0 release [2]: SFT, DPO, ORPO, GRPO, KTO, SimPO, Continued Pre-Training (CPT), embedding fine-tuning, MoE fine-tuning, vision (VLM) fine-tuning, OCR fine-tuning, JEPA-family fine-tuning, and audio fine-tuning covering TTS/STT/ASR models (Orpheus, Whisper, Parakeet, Voxtral, Qwen3-ASR, Canary, and others) [1][2]. All are imported directly from the top-level `mlx_tune` package (`from mlx_tune import SFTTrainer, SFTConfig`, `from mlx_tune import DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig, ...`) - there is no experimental sub-namespace like some sibling libraries use [1]. The Contributing section flags two shipped-but-limited paths as open work: audio training currently runs at `batch_size=1` only, and RL training (the DPO/ORPO/GRPO/KTO/SimPO family) currently runs single-sample rather than batched [1]. The taxonomy is the "Supported Training Methods" table on the docs site and in the README, and it moves as JEPA and audio methods are recent additions per the v0.6.0 changelog banner - recheck the live table [1][4].

**Scale it handles**: single Mac only, from the docs' stated floor of 8GB unified RAM (0.5-1B models at 4-bit) up to 64GB+ (13B+ models, or 8-bit precision) [4]. No launcher, no distributed training, and no published multi-node or multi-GPU benchmark exist for this library - MLX's unified-memory design does not target that shape of scale. Within one machine, the docs' own Performance page gives self-reported numbers from the maintainer's own runs on an M4 Pro (48GB), explicitly calling them "rough indicators, not benchmarks" and "starting expectations" rather than a formal suite [5]. The table's own deciding numbers: SFT on Qwen3-0.6B-bf16 at context 1024 runs about 2.2 iterations/sec (16 LoRA layers, no checkpointing, peak 2.6GB) but drops to about 0.28 it/s at context 4096 with full LoRA (peak 18GB); DPO on the 4-bit Qwen3.5-0.8B-MLX model at context 512 runs about 540ms/sample at batch size 1 (sharing the prompt forward pass between chosen and rejected) versus about 810ms/sample at batch size 2, where that sharing no longer applies; GRPO on Qwen3-0.6B-bf16 at context 4096 with 4 generations of 256 tokens runs about 0.11 it/s at 5.3GB peak [5]. The same page's rule of thumb for scaling down: halving available memory roughly halves the context length you can comfortably train at for the same model class, and below 16GB it recommends staying at 4-bit models in the 0.5-1B range at context 1024 or shorter [5].

**Install**: `pip install mlx-tune` (or `uv pip install mlx-tune`); PyPI's latest release is v0.6.0, published 2026-06-23, which the GitHub `v0.6.0` tag resolves to commit `9690fe1fed19c72f0810902b9c188d9a2625eb5b` [6][7] - an exact match to this card's screening commit, so no push-vs-release gap applies here. `pyproject.toml` at that tag states `requires-python = ">=3.9"`, license Apache-2.0, and classifies the project "Development Status :: 3 - Alpha" [8]. Core pins from the same file: `mlx>=0.31.0`, `mlx-lm>=0.31.0`, `transformers>=4.36.0`, `datasets>=2.14.0,<4.0.0` (the one upper bound in the core set), `huggingface-hub>=0.20.0`, `tokenizers>=0.15.0`, `numpy>=1.23.0` [8]; torch is not a dependency at all - training runs on MLX's own array backend, not PyTorch. The `train` extra (`pip install "mlx-tune[train]"`) adds `trl>=0.7.0`, `peft>=0.7.0`, `wandb>=0.15.0`; the `audio` extra adds `mlx-audio>=0.4.2`, `soundfile>=0.12.0`, `librosa>=0.10.0` [8]. A separate root `requirements.txt` in the repo lists looser, unpinned-upper-bound versions of the same core packages and predates the `pyproject.toml` extras split - it is not what `pip install mlx-tune` resolves against, so treat `pyproject.toml` as authoritative [8][9]. Hardware floor: Apple Silicon (M1-M5) running macOS 13.0+, per the docs' own Requirements section - there is no CUDA path [4].

## Quick start

The README's own Quick Start section, reproduced verbatim [1]:

```python
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import load_dataset

# Load any HuggingFace model (1B model for quick start)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# Load a dataset (or create your own)
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")

# Train with SFTTrainer (same API as TRL!)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        max_steps=50,
    ),
)
trainer.train()

# Save (same API as Unsloth!)
model.save_pretrained("lora_model")  # Adapters only
model.save_pretrained_merged("merged", tokenizer)  # Full model (16-bit)
model.save_pretrained_gguf("model", tokenizer)  # GGUF (see note below)
```

`get_peft_model` is a class method reached through `FastLanguageModel.get_peft_model(...)`, not a standalone top-level import - grepping `mlx_tune`'s exports finds no bare `get_peft_model` name [3]. Install first with `pip install mlx-tune` (`uv pip install mlx-tune`, or `uv pip install "mlx-tune[audio]"` for the audio extra) [4]. There is no CLI entry point - `pyproject.toml` defines no `[project.scripts]` section, so every workflow is a Python script [8].

## Start it

- The only launch form is one process on one Mac: run the Python script directly (`python train.py`). No `accelerate launch`-style wrapper, no multi-process config template, and no distributed backend exist in the docs or source read for this card.
- Effective batch size is `per_device_train_batch_size` times whatever gradient-accumulation the specific trainer exposes; the Performance page's own note calls out DPO/ORPO at batch size 1 as a distinct, slower-per-sample regime worth watching separately from larger batches [5].
- Config surface: each method has its own `Config` dataclass (`SFTConfig`, `DPOConfig`, `GRPOConfig`, `ORPOConfig`, `KTOConfig`, `SimPOConfig`) [1][3]. Read at commit `9690fe1fed19c72f0810902b9c188d9a2625eb5b`: every RL Config defaults `save_steps=100`, and `logging_steps` defaults to 10 for DPO, ORPO, KTO, and SimPO but to 1 for GRPO - the library's own choice, not a documented rationale [10]. `SFTConfig` defaults `output_dir="./outputs"`, `logging_steps=10`, `save_steps=100`, and `grad_checkpoint=False` [11].
- Gradient checkpointing is off by default across the library and is a real memory/speed knob, not a formality: the Performance page states it roughly halves activation memory at the cost of about 2x backward-pass time, and documents a past bug where the setting silently did nothing for every trainer except SFT - now fixed, and "respected by every trainer" as of the version read [5].
- Out-of-memory first aid comes from the docs' Troubleshooting page's dedicated "Out of Memory" section: a RAM-to-model-size table (8GB->0.5-1B at 4-bit, 16GB->1-3B at 4-bit, 32GB->up to 7B at 4-bit, 48GB->7-13B at 4-bit, 64GB+->13B+ or 8-bit), plus turning on gradient checkpointing via a one-line code example on the model object [12]. Generation is not a separate engine here - MLX/mlx-lm shares the same process and memory pool as training, so there is no separate generation-side memory knob to tune independently.
- Two environment variables tune the always-on performance layer rather than being knobs you must set to get correctness: `MLX_TUNE_DISABLE_COMPILE=1` turns off the library's `@mx.compile` step wrapper (useful for debugging a shape error, at a speed cost), and `MLX_TUNE_BUCKET_SIZE=N` changes the length-bucket granularity mlx-tune uses so that `mx.compile`'s per-shape cache is reused across batches instead of recompiling for every new sequence length [5][12].

## Watch it

This section is mechanics only - what a metric means for a given method (DPO, GRPO, ...) lives on that method's own methodology card, not here.

- **There is no experiment-tracker integration in mlx-tune's own training loop.** Grepping the full source of `mlx_tune/sft_trainer.py` and `mlx_tune/rl_trainers.py` at commit `9690fe1fed19c72f0810902b9c188d9a2625eb5b` finds no `wandb`, `tensorboard`, or `report_to`-style call anywhere in either file - every trainer logs by plain `print()` at its `logging_steps` cadence, and nothing is written to disk or to a dashboard by default [10][11]. This holds even though `wandb>=0.15.0` is pinned as part of the `train` extra in `pyproject.toml` [8] - that pin is not wired to anything in the training loop as of this commit.
- **RL trainer print format** (DPO, ORPO, KTO, SimPO), read directly from the step loop: `Step {step+1}/{iters} | Loss: {avg_loss:.4f} | batch_size: {bs}` [10].
- **GRPO's print format** differs and includes the reward signal: `Step {step+1}/{iters} | Loss: {avg_loss:.4f} | Reward: {avg_rew:.3f}`, with a distinct branch printed as `... (skipped, equal rewards)` when a batch's reward standard deviation falls below `1e-8` and the update is skipped entirely [10].
- **SFT logging** is delegated to mlx-lm's own native trainer: `SFTTrainer` passes `steps_per_report=self.logging_steps` into `mlx_lm.tuner.trainer.train`/`TrainingArgs`, so the exact per-step fields printed during the training phase come from mlx-lm, not from mlx-tune's own code - mlx-tune's own prints around it cover setup (output dir, learning rate, iteration count, batch size, LoRA rank/alpha, grad-checkpoint flag) and end-of-phase status (dataset sizes loaded, adapter save path) [11].
- **Sample-level generation logging**: not found in either trainer file for this card - no method here prints or saves sampled completions during training, unlike GRPO implementations elsewhere that log generation text.
- **Evaluation during training**: `SFTTrainer` builds and saves a validation split alongside the training file, printed as "Created validation set (copied from train)" or "Prepared validation set" depending on whether one was supplied, and passes it through to mlx-lm's native trainer; no RL trainer (DPO/ORPO/GRPO/KTO/SimPO) in `rl_trainers.py` builds or evaluates on a held-out set - none of the five accepts an eval dataset argument [10][11].
- **Stopping**: no early-stopping, patience, or reward-threshold field was found in either trainer file, nor on the docs' Performance or Troubleshooting pages, which is where a knob like that would be documented alongside the other tuning parameters (env vars, gradient checkpointing) if it existed [5][12]. The library publishes fixed `iters`/`max_steps` counts as the only stopping mechanism read for this card.

## Save it

- LoRA adapters save via `model.save_pretrained("lora_model")`, which writes an adapter directory - NOT a full model - via the shared `_save_adapters_and_config()` helper used by every RL trainer at each `save_steps` interval and at the end of training [10]. An adapter directory alone cannot be loaded as a standalone model; it must be paired with the base model to be used.
- RL trainers checkpoint by overwriting the single `adapter_path` directory at every `save_steps` interval (read from the DPO trainer's save loop at commit `9690fe1fed19c72f0810902b9c188d9a2625eb5b`) [10] - this is not a series of numbered checkpoints like `checkpoint-500/`, `checkpoint-1000/`; only the latest adapter state survives on disk. No `resume_from_checkpoint` call or flag exists anywhere in `rl_trainers.py`.
- `SFTTrainer` delegates its own checkpoint and resume mechanics to mlx-lm's native `mlx_lm.tuner.trainer.train`/`TrainingArgs`, an external dependency outside mlx-tune's own source - this card does not assert mlx-lm's checkpoint/resume behavior, since that is out of scope for a claim sourced from mlx-tune's code [11].
- Merging LoRA into the base model for a standalone deployable model uses `model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")` (the default) or `save_method="merged_4bit"`, implemented by `save_model_hf_format()` in `trainer.py`. Its own docstring states the tradeoff directly: `merged_16bit` "dequantizes a quantized base and saves a full-precision merged model so the fine-tune is preserved exactly", while `merged_4bit` "keeps the base quantization and re-quantizes the fused weights (smaller on disk, but small LoRA deltas can be rounded away...)" [13]. The code comments this fusion step as "CRITICAL: Fuse LoRA adapters into base weights before saving. Without this, LoRA layers are saved as-is and won't load properly" [13]. This exact failure mode was a real, fixed bug: the Troubleshooting page documents that a merged model losing its fine-tune was issue #15, fixed in release 0.5.1, and explains the root cause matches the `merged_4bit` re-quantization risk named in the docstring above [12].
- GGUF export goes through `model.save_pretrained_gguf(...)`, implemented by `export_to_gguf()`, which shells out to `mlx_lm.fuse --export-gguf` [13]. The Troubleshooting page states that GGUF export from a quantized (4-bit) base model does not work, linking it as a known mlx-lm limitation via issue mlx-lm#353, and gives three workarounds - start from a non-quantized base model, pass `dequantize=True` before export, or skip GGUF entirely and use `save_pretrained_merged()` instead [12]. The README separately lists a second, related upstream issue, mlx-examples#1382, describing the same quantized-to-GGUF gap [1].
- `model.push_to_hub("username/my-model")` is documented in the README's Post-Training Workflow section and marked "Stable" in the project status table [1][2]; this card does not assert its internal wiring beyond that documented call form, since `push_to_hub` is not defined directly on the `MLXModelWrapper` class in `model.py` at the commit read - `model.py` delegates unknown attributes to the underlying MLX model object via `__getattr__` [3].
- Loader handoff: whether an external evaluator can load a saved adapter or merged model directly is that evaluator's own contract, not documented by mlx-tune itself for this card - a merged, non-quantized model directory is a standard Hugging Face-format model directory per `save_model_hf_format()`'s use of `mlx_lm.utils.save_model`/`save_config`, but an adapter-only save requires pairing with the base model exactly as it does for LoRA saves generally [13].

## Find it in the docs

The docs live at https://arahim3.github.io/mlx-tune/, the URL named in the repository's own homepage field [6]; fetching it returns an HTTP 301 redirect to a custom domain, `https://arahim.dev/mlx-tune/`, where the pages actually resolve with HTTP 200 (checked 2026-08-12) - use the `arahim.dev` URLs directly rather than following the redirect each time.

- Page slugs, all flat HTML under that domain: `index.html` (home, including the `#quick-start` anchor), `llm.html`, `vlm.html`, `ocr.html`, `audio.html`, `jepa.html`, `workflow.html`, `performance.html`, `examples.html`, `troubleshooting.html`, `api.html` [4]. There is no versioned URL path - the site serves whatever is on the `main` branch's `docs/` directory at deploy time, so it is an unpinned, moving source; the v0.6.0 badge on the homepage is the only version marker on the page itself [4].
- A trap for this recipe: `docs/quick_start.md` exists in the repository but is stale and not part of the live site - it uses a placeholder clone URL and documents an older CLI-based workflow (`mlx_lm.lora --model ... --train --data train.jsonl`) that predates the current `SFTTrainer`/`SFTConfig` API, and `quick_start.html` 404s on the live site (checked 2026-08-12) [14]. Use the homepage's `#quick-start` anchor or the README's own Quick Start section instead [1][4].
- Runnable references beyond the docs: the repository's `examples/` directory holds 60+ numbered example scripts, one per model/method combination, e.g. `04_simple_finetuning.py`, `09_rl_training_methods.py`, `21_dpo_preference_tuning.py`, `22_grpo_reasoning_training.py`, `27_embedding_finetuning.py`, `33_ocr_document_finetuning.py`, plus a demo notebook `mlx_tune_demo.ipynb` [15].
- Two known, docs-stated boundaries a reader should check before starting: fine-tuning a DeepSeek-OCR model requires `transformers<5.0`, which the README states as a hard version-pin trap with a full worked pip command for a "verified working environment" [1]; and the GGUF-from-quantized-model limitation above (mlx-lm#353 [12], mlx-examples#1382 [1]) applies to any base model whose path contains a quantization marker (the code's own check list is `4bit`, `8bit`, `3bit`, `2bit`, `-q4`, `-q8`, `int4`, `int8`, `bnb`) [13].
- No official curated community-tutorials page or MCP endpoint was found for this project - it is a small, solo-maintained repository without the tutorial-curation layer some larger libraries publish. The README's own "Why I Built This" note frames the project as a personal itch-scratch rather than a maintained ecosystem hub [1].
- Honest boundary: this is single-machine, Apple-Silicon-only software with no distributed training path, and RL training here is documented as running single-sample (not batched) as of this release, so throughput on preference-tuning methods (DPO/ORPO/GRPO/KTO/SimPO) will be slower per wall-clock than a batched CUDA implementation of the same method [1].

## Sources

All pages are unpinned, live docs read on 2026-08-12 unless a commit or release tag is named. Source-code claims are read at commit `9690fe1fed19c72f0810902b9c188d9a2625eb5b`, the commit the `v0.6.0` GitHub release tag resolves to, which is an exact match to this card's screening commit - no push-ahead-of-release gap applies. Method names (SFT, DPO, GRPO, ORPO, KTO, SimPO, JEPA, ...) are deliberately cited to nothing here; their defining papers live on the methodology cards.

[1] mlx-tune README. https://raw.githubusercontent.com/ARahim3/mlx-tune/9690fe1fed19c72f0810902b9c188d9a2625eb5b/README.md. Fetched 2026-08-12.

[2] mlx-tune GitHub repository (metadata: license, default branch, homepage, push/release timestamps, star count). https://github.com/ARahim3/mlx-tune. Fetched 2026-08-12.

[3] mlx-tune `mlx_tune/model.py` at commit 9690fe1fed19c72f0810902b9c188d9a2625eb5b (`FastLanguageModel`, `MLXModelWrapper`, `__getattr__` delegation). https://raw.githubusercontent.com/ARahim3/mlx-tune/9690fe1fed19c72f0810902b9c188d9a2625eb5b/mlx_tune/model.py. Fetched 2026-08-12.

[4] mlx-tune docs homepage (quickstart, installation, requirements, training-methods table, comparison table). https://arahim.dev/mlx-tune/index.html (redirected from https://arahim3.github.io/mlx-tune/). Fetched 2026-08-12.

[5] mlx-tune docs Performance page (M4 Pro 48GB indicative benchmark table, scaling rule of thumb, gradient-checkpointing tradeoff, env vars, "what's automatic" list). https://arahim.dev/mlx-tune/performance.html. Fetched 2026-08-12.

[6] mlx-tune GitHub API repository metadata (homepage field, default_branch, pushed_at). https://api.github.com/repos/ARahim3/mlx-tune. Fetched 2026-08-12.

[7] mlx-tune GitHub releases and tags API (v0.6.0 published_at and tag-to-commit resolution). https://api.github.com/repos/ARahim3/mlx-tune/releases and https://api.github.com/repos/ARahim3/mlx-tune/tags. Fetched 2026-08-12.

[8] mlx-tune `pyproject.toml` at the v0.6.0 tag (Python floor, license, classifiers, core and extras dependency pins). https://raw.githubusercontent.com/ARahim3/mlx-tune/v0.6.0/pyproject.toml. Fetched 2026-08-12.

[9] mlx-tune `requirements.txt` at commit 9690fe1fed19c72f0810902b9c188d9a2625eb5b (looser, unpinned-upper-bound legacy dependency list). https://raw.githubusercontent.com/ARahim3/mlx-tune/9690fe1fed19c72f0810902b9c188d9a2625eb5b/requirements.txt. Fetched 2026-08-12.

[10] mlx-tune `mlx_tune/rl_trainers.py` at commit 9690fe1fed19c72f0810902b9c188d9a2625eb5b (DPO/ORPO/GRPO/KTO/SimPO Config defaults, print-based step logging, checkpoint-overwrite save loop, `_save_adapters_and_config`). https://raw.githubusercontent.com/ARahim3/mlx-tune/9690fe1fed19c72f0810902b9c188d9a2625eb5b/mlx_tune/rl_trainers.py. Fetched 2026-08-12.

[11] mlx-tune `mlx_tune/sft_trainer.py` at commit 9690fe1fed19c72f0810902b9c188d9a2625eb5b (`SFTConfig` defaults, delegation to mlx-lm's native trainer, validation-set handling, setup/status prints). https://raw.githubusercontent.com/ARahim3/mlx-tune/9690fe1fed19c72f0810902b9c188d9a2625eb5b/mlx_tune/sft_trainer.py. Fetched 2026-08-12.

[12] mlx-tune docs Troubleshooting page (OOM RAM/model-size table, gradient-checkpointing example, issue #15 merged-model bug and 0.5.1 fix, GGUF-from-quantized limitation with mlx-lm#353/mlx-examples#1382 links and workarounds). https://arahim.dev/mlx-tune/troubleshooting.html. Fetched 2026-08-12.

[13] mlx-tune `mlx_tune/trainer.py` at commit 9690fe1fed19c72f0810902b9c188d9a2625eb5b (`save_model_hf_format` merged_16bit/merged_4bit docstring and fuse-before-save logic, `export_to_gguf` quantization check and `mlx_lm.fuse` subprocess call). https://raw.githubusercontent.com/ARahim3/mlx-tune/9690fe1fed19c72f0810902b9c188d9a2625eb5b/mlx_tune/trainer.py. Fetched 2026-08-12.

[14] mlx-tune `docs/quick_start.md` at commit 9690fe1fed19c72f0810902b9c188d9a2625eb5b (stale placeholder-URL, CLI-based quickstart, not linked from the live site). https://raw.githubusercontent.com/ARahim3/mlx-tune/9690fe1fed19c72f0810902b9c188d9a2625eb5b/docs/quick_start.md. Fetched 2026-08-12.

[15] mlx-tune `examples/` directory listing at commit 9690fe1fed19c72f0810902b9c188d9a2625eb5b. https://api.github.com/repos/ARahim3/mlx-tune/contents/examples. Fetched 2026-08-12.
