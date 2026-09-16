# mlx-vlm

An Apple-Silicon-first vision-language-model library: one `load`/`generate` pair for inference, one CLI script for LoRA/QLoRA/full fine-tuning and ORPO preference training.

**mlx-vlm** "is a package for inference and fine-tuning of Vision Language Models (VLMs) and Omni Models (VLMs with audio and video support) on your Mac using MLX" [1]. It is authored by Prince Canuma (`Blaizzy`) [2][3] and lives at https://github.com/Blaizzy/mlx-vlm [2]. The inference API is two calls - `load(model_path)` returns a model and processor, `generate(model, processor, prompt, image)` returns text [4] - and a separate top-level script, `mlx_vlm/lora.py`, drives training: it is invoked as `python lora.py --dataset <id> [options]` and is not exposed as a `python -m mlx_vlm` subcommand [5][6].

**When to pick it**: you are fine-tuning or running inference on a VLM on a Mac and want a single package that covers both, with a wide day-one model list (Qwen2/3/3.5-VL, LLaVA, Deepseek-VL(-V2), Mllama, and more, per the LoRA doc's supported-models note) [5] and built-in LoRA/QLoRA/full-finetune plus one preference method (ORPO); this card does not compare it against another card in this deck, since none of the sourced pages state such a comparison. `pyproject.toml` also lists an optional `mlx-vlm[cuda]` extra pulling in `mlx-cuda` as an alternate MLX backend [7], but the only documented CUDA-specific feature is inference-side activation quantization for `mxfp8`/`nvfp4` models [8]; none of the sources read here document or walk through fine-tuning on CUDA, so treat training as an Apple-Silicon-only, undemonstrated-elsewhere workflow. If your target is scaling one method (e.g. GRPO) across a Linux/Nvidia cluster, this library is out of scope - it ships no RL-style trainer, only SFT and ORPO [5][9].

**Methods it ships**: `mlx_vlm/lora.py --train-mode {sft,orpo}` (default `sft`) selects between two trainer modules [9]: `mlx_vlm/trainer/sft_trainer.py` (supervised fine-tuning, cross-entropy loss with optional completion-only masking via `--train-on-completions`) and `mlx_vlm/trainer/orpo_trainer.py` (Odds Ratio Preference Optimization, a preference method combining an SFT term with a log-odds-ratio preference term weighted by `--beta`, default 0.1) [10][11]. `--train-mode orpo` and its `--beta`/`--eps` flags are read directly from the CLI source [9]; the shipped fine-tuning doc, `LORA.MD`, documents only the SFT/LoRA path and does not mention ORPO, `--train-mode`, `--beta`, or `--eps` anywhere in its Arguments or Examples sections - so ORPO is wired into the code but currently undocumented in the library's own guide [5]. Fine-tuning mode is chosen independently of the method: `--full-finetune` unfreezes the language-model weights, the default path instead wraps the language model's linear layers in LoRA adapters (`--lora-rank`, `--lora-alpha`, `--lora-dropout`), and `--train-vision` additionally unfreezes the vision tower/projector modules for either mode [12][5]. Two model types are excluded from training outright: the code's `not_supported_for_training` set is `{"gemma3n", "qwen3_omni"}` [13]. This method list is training-side; the inference side additionally supports speculative decoding (DFlash and EAGLE-3 drafters) and distributed multi-machine generation, which are not fine-tuning methods and are not covered further here [14].

**Scale it handles**: fine-tuning is single-process, one Mac (or one CUDA machine via the `mlx-cuda` extra) - the LoRA doc's "Trainer Backend" list advertises gradient checkpointing and gradient accumulation for memory, not multi-device sharding, and no distributed-training launcher is documented for `lora.py` [5]. Distributed scaling is documented for inference only: `mlx.launch --hostfile <hosts.json> --backend jaccl -- mlx-vlm/examples/sharded_generate.py --model <repo> ...` shards the language model (not the vision tower) across networked Macs, recommended over Thunderbolt using the JACCL backend, with the docs citing a 1T-parameter model (Kimi-K2.6) as the scale example; no throughput/latency benchmark accompanies that example in the pages read [14][15]. Treat "trains on your Mac" as the scale ceiling for fine-tuning; there is no published multi-GPU or multi-node fine-tuning path.

**Install**: `pip install -U mlx-vlm` [16]; PyPI's current release is 0.6.10, uploaded 2026-08-04T15:26:26Z [17], matching GitHub release `v0.6.10` published 2026-08-04T15:25:34Z at commit `c2fe301bb3f21c45985a423d7832bb238987312d` [18]. Python floor `>=3.10`; license MIT [16][17]. At the `v0.6.10` tag, `requirements.txt` pins the load-bearing floors: `mlx>=0.32.0` (the deep-learning core), `transformers>=5.14.0`, `mlx-lm>=0.31.3`, `mlx-audio>=0.4.3`, plus `Pillow>=10.3.0`, `opencv-python>=4.12.0.88`, `fastapi>=0.95.1`, `uvicorn`, `numpy`, and others, all lower-bounded with no upper cap [19]. Optional extras from `pyproject.toml` at the same tag: `mlx-vlm[ui]` adds `gradio>=5.19.0` for the chat UI, `mlx-vlm[train]` adds `datasets>=2.19.1` for fine-tuning, and `mlx-vlm[cuda]` / `mlx-vlm[cpu]` add `mlx-cuda` / `mlx-cpu` respectively as alternate MLX backend packages, with no sourced page tying either extra to the fine-tuning path specifically [7]. No CUDA/GPU driver version or other hardware minimum is stated in the installation page, the README, or the pinned requirements file [16][1][19]. This card's shortlist row names commit `07bd830e7b46cd88ef90b859d5d4c145abff3dd5` (pushed 2026-07-31T18:42:43Z); a GitHub compare shows that commit is 30 commits BEHIND the `v0.6.10` tag (0 ahead, 30 behind) [20] - the opposite of the usual case, since here the release is newer than the row's commit. All source-code claims on this card that name a specific file (the trainer modules, `lora.py`, `utils.py`) were read at that row commit, then diffed byte-for-byte against the same files at the `v0.6.10` tag for `sft_trainer.py` and `orpo_trainer.py`, which are identical between the two [9][10][11][20].

**Maintained by**: Prince Canuma, sole listed author in the package metadata and docs site [16][2]. Repository not archived, last pushed 2026-08-09T01:55:23Z [21]; release cadence is roughly weekly to sub-weekly over the month read - v0.6.6 (2026-07-20), v0.6.7 (2026-07-23), v0.6.8 (2026-07-27), v0.6.9 (2026-08-03), v0.6.10 (2026-08-04) [18].

## Quick start

Inference, quoted in shape from the README's Python Script section [4]:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

CLI form of the same call, from the docs Usage page [22]:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg
```

Smallest complete LoRA fine-tuning run, from `LORA.MD`'s Basic-LoRA example [5]:

```bash
python lora.py \
    --model-path mlx-community/Qwen3-VL-2B-Instruct-bf16 \
    --dataset your-huggingface-dataset-id \
    --batch-size 2 \
    --epochs 2 \
    --learning-rate 2e-5 \
    --output-path ./qwen3-lora-adapter.safetensors
```

## Start it

- One process, one device is the only supported form for `lora.py` - there is no documented multi-GPU or multi-node launcher for fine-tuning [5].
- Fine-tuning mode is chosen entirely through flags on the same script: default (no flag) trains LoRA adapters on the language model; `--full-finetune` unfreezes the language model's own weights instead of adding adapters; `--train-vision` additionally unfreezes the vision tower and projector modules (named in code as `vision_model`, `vision_tower`, `mm_projector`, `multi_modal_projector`, `aligner`, `connector`, `vision_resampler`) for either mode; `--train-mode orpo` swaps the SFT loss for the ORPO preference loss and expects a chosen/rejected preference dataset instead of a plain instruction dataset [12][9][10].
- Effective batch is `--batch-size` x `--gradient-accumulation-steps` (default accumulation 1); the script also accepts `--epochs`, which recomputes `--iters` as `(len(dataset) // batch_size) * epochs` and overrides any `--iters` value passed [23][12].
- Config surface is entirely CLI flags on `lora.py`, not a YAML/JSON config object; there is one documented default that the CLI text disagrees with the code on: `LORA.MD`'s Training-Arguments table states `--val-batches` defaults to 25 and repeats 25 in every one of its programmatic-Python examples [5], but the actual `argparse` default in `mlx_vlm/lora.py` at the row commit is `4` [24] - pass `--val-batches` explicitly rather than trust either number blindly. Two defaults the library sets deliberately: `--learning-rate` for LoRA defaults to `2e-5` in the CLI (the underlying `TrainingArgs` dataclass instead defaults to `1e-5`, but that value only applies if a caller builds `TrainingArgs` directly and bypasses `lora.py`'s argparse) [24][25], and `--grad-checkpoint` is off by default, so gradient checkpointing must be opted into for large models [24].
- Out-of-memory first aid, from `LORA.MD`'s Memory-Optimization tips [5]: enable `--grad-checkpoint`; lower `--batch-size`; use `--gradient-accumulation-steps` to keep the effective batch size while lowering the per-step memory; for constrained devices, point `--model-path` at a quantized checkpoint (QLoRA) - the doc's QLoRA section states the resulting adapter trains in full precision on top of a quantized base and is smaller and faster to deploy [5]. There is no separate generation engine for training (no vLLM-style split); OOM knobs are all on the one process.

## Watch it

This section is mechanics only - what a metric means for a method (SFT loss shape, ORPO's accuracy/margin behavior) is outside this card's scope.

- **Enable it**: the LoRA doc's own Output section is the entire logging surface - there is no `report_to`/tracker integration (no W&B, TensorBoard, or Trackio hook) documented for `lora.py`; progress prints to stdout at the cadence set by `--steps-per-report` (default 10) [26][5]. A run with `--steps-per-report` left at default and stdout not captured leaves no persisted log.
- **What prints**, per `LORA.MD`'s Output section: current iteration and total iterations, the loss value at that reporting step, a running average loss, training speed in tokens/sec, and an estimated time remaining [5].
- **SFT internals** (`mlx_vlm/trainer/sft_trainer.py`, read at the row commit): the loss function is cross-entropy over the (optionally completion-masked, via `--train-on-completions`) token sequence; validation runs every `--steps-per-eval` (default 200) steps for `--val-batches` batches [10][24].
- **ORPO internals** (`mlx_vlm/trainer/orpo_trainer.py`): the training loop tracks and reports, in addition to the combined loss, `accuracies`, `margins`, `policy_chosen_logps`, `policy_rejected_logps`, `sft_term`, and `orpo_pref_term` as internal metrics inside `evaluate_orpo`/`train_orpo`, alongside `chosen_logits_mean`/`rejected_logits_mean`; these are computed in code but the same stdout-only Output section is what actually surfaces them during a run - `LORA.MD` does not enumerate the ORPO-specific fields since it does not document ORPO at all [11][5].
- **Sample-level generation logging**: not present - neither `LORA.MD` nor the two trainer files read log sample generations during training; the only qualitative check documented is running inference on a saved checkpoint yourself [5][10][11].
- **Evaluation during training**: `--steps-per-eval` (default 200) and `--val-batches` (documented default 25, actual CLI default 4 - see the discrepancy above) control validation cadence and sample count for both SFT and ORPO [5][24].
- **Stopping-rule search**: `LORA.MD` publishes no early-stopping flag, threshold, or patience value anywhere in its Arguments, Examples, or Training Tips sections, and `mlx_vlm/lora.py`'s argparse list (read at the row commit) defines no such flag either - training runs for the fixed `--iters` (or the epoch-derived iteration count) with no automatic stop [5][24].

## Save it

- Output is a single file, not a directory: `--output-path` (default `adapters.safetensors`) is passed straight through to `save_adapter()`, and if the path does not already end in `.safetensors`, `lora.py` appends `/adapters.safetensors` to it [23][27]. Alongside that file, `save_adapter()` writes `adapter_config.json` into the same parent directory whenever the model carries a `config.lora` attribute (i.e., LoRA/QLoRA runs); it always writes `model.trainable_parameters()` to the `--output-path` file itself, regardless of run mode [27].
- **What "adapters.safetensors" actually contains depends on the run mode**: `save_adapter()` always writes `model.trainable_parameters()`. For LoRA/QLoRA runs that is the small set of LoRA weight deltas; for `--full-finetune` runs, the unfrozen language-model weights (and vision weights too, if `--train-vision` was also set) are the trainable parameters, so the same-named `adapters.safetensors` file holds a full model-sized weight dump, not a small adapter, even though `adapter_config.json` is not written in that mode (no `config.lora` attribute exists for a full-finetune model) [27][12].
- Periodic checkpoints during a run are saved every `--steps-per-save` steps (default 100) as numbered snapshots named `{iteration:07d}_adapters.safetensors` alongside the final output file, per the trainer's save-step logic [10][11][5].
- Resume: pass `--adapter-path <path>`; `lora.py` calls `apply_lora_layers(model, adapter_path)` to reload it onto the base model before continuing training [12]. `apply_lora_layers()` treats `adapter_path` as a DIRECTORY, not a file: it runs `Path(adapter_path)`, raises `FileNotFoundError` if that path does not exist, then opens `adapter_path / "adapter_config.json"` and (inside the LoRA-layer application it calls) loads weights from the hardcoded name `adapter_path / "adapters.safetensors"` [28]. This is a real trap: `LORA.MD`'s own Resume-Training example passes `--adapter-path ./qwen3-lora-adapter.safetensors`, a custom-named `.safetensors` FILE, which does not match what the function expects - a directory containing `adapter_config.json` and a file literally named `adapters.safetensors` [5][28]. Resume only works cleanly when `--output-path` was left at its default (`adapters.safetensors` in some directory) or when the adapter files are manually placed to match that layout.
- Loader handoff: a saved `adapters.safetensors` (LoRA case) is NOT a standalone model - `mlx_vlm.load(path_or_hf_repo, adapter_path=...)` is required to pair it back with the original base model id/path; internally `load()` calls `apply_lora_layers(model, adapter_path)` then `model.eval()` [29]. A full-finetune output, despite sharing the same filename convention, is not documented as reloadable through the `adapter_path=` argument, since that path is built around `adapter_config.json`, which full-finetune runs never write [27][29] - treat a full-finetune save as requiring its own manual weight-loading path, which the sources read here do not document.

## Find it in the docs

The mkdocs site is the maintained reference, but it does not cover fine-tuning at all - the CLI Reference page there lists only `mlx_vlm.convert`, `mlx_vlm.generate`, `mlx_vlm.chat_ui`, and `mlx_vlm.server`; `lora.py` is absent from it [6]. The one authoritative fine-tuning source is the GitHub-hosted `LORA.MD` file, reached only from the README's own "LoRA & QLoRA" section, which is a one-line pointer: "To learn more about LoRA, please refer to the [LoRA.md] file" [30].

- Docs site address pattern: `https://Blaizzy.github.io/mlx-vlm/<page>.html` (built from `mkdocs.yml`'s `docs_dir: docs`, `site_url: https://Blaizzy.github.io/mlx-vlm`) [31]; checked 2026-08-10, `https://blaizzy.github.io/mlx-vlm/` and `.../usage/` both return HTTP 200. The site's own `mkdocs.yml` sets `extra.version.provider: mike` (a versioned-docs tool), but no working `versions.json` was found live (404) - so despite mike being configured, this is currently a single, unversioned, live build, dated only by fetch date, not by a version-tag URL [31].
- Nav pages, by slug from `mkdocs.yml`: `index` (one-paragraph description), `installation` (bare `pip install mlx-vlm`, no extras or pins shown), `cli_reference` (the four subcommands above), `examples`, `contributing`, `community_projects`, `report_issues`, `changelog` [31]. A `usage.md` file exists in the repo's `docs/` tree but is not linked in the nav [31][22].
- For fine-tuning specifics (dataset JSON formats per model family, the full CLI argument reference, worked examples for LoRA/QLoRA/full-finetune/resume), read `mlx_vlm/LORA.MD` directly on GitHub rather than the docs site [5].
- Runnable references beyond the docs: the repo's `mlx_vlm/models/<family>/README.md` files (linked from the main README's "Model-Specific Documentation" table) give per-model prompt formats and examples for models with non-obvious inputs (DeepSeek-OCR, Mllama, GLM-OCR, and others) [1]. The training doc's dataset examples reference user-supplied Hugging Face dataset ids rather than a named smoke-test dataset, so no specific known-good training dataset is named in the sources read here [5].
- Community layer: the mkdocs nav does carry a `community_projects` slug [31], but the page itself is a placeholder - its full text is "If you have a project built on top of MLX-VLM let us know! We plan to showcase community examples and links here", with no tutorials or links listed at the row commit [32]. No other curated community-tutorials page was found in the README during this reading, so beyond the model-specific README table above [1], there is nothing further to point to here.
- No official MCP endpoint for querying these docs was found in the pages read.

Honest boundary: fine-tuning excludes `gemma3n` and `qwen3_omni` model types outright (`not_supported_for_training`, checked in code, raises `ValueError` if hit) [13]; the only training methods shipped are SFT and ORPO - no PPO/GRPO/DPO/KTO-style trainer exists in the package as read [9]; and there is no documented multi-device fine-tuning path, only single-device (Apple Silicon or, via the `cuda` extra, a single CUDA machine) [5][7].

## Sources

[1] mlx-vlm README, top-of-file description and Installation section. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/README.md. Fetched 2026-08-10.

[2] mlx-vlm GitHub repository. https://github.com/Blaizzy/mlx-vlm. Fetched 2026-08-10.

[3] mkdocs.yml `site_author` field. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mkdocs.yml. Fetched 2026-08-10.

[4] mlx-vlm README, Python Script quickstart section. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/README.md. Fetched 2026-08-10.

[5] mlx_vlm/LORA.MD, fine-tuning documentation (Overview, Requirements, Supported Models, Arguments, Examples, Output, Training Tips, QLoRA and Quantization sections). https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/LORA.MD. Fetched 2026-08-10.

[6] mlx-vlm docs, CLI Reference page. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/docs/cli_reference.md. Fetched 2026-08-10.

[7] mlx-vlm `pyproject.toml` at the `v0.6.10` tag (optional-dependency extras). https://raw.githubusercontent.com/Blaizzy/mlx-vlm/v0.6.10/pyproject.toml. Fetched 2026-08-10.

[8] mlx-vlm README, "Activation Quantization (CUDA)" section. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/README.md. Fetched 2026-08-10.

[9] `mlx_vlm/lora.py` source, argparse definitions including `--train-mode`, `--beta`, `--eps`, and the mode dispatch in `main()`. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/lora.py. Fetched 2026-08-10.

[10] `mlx_vlm/trainer/sft_trainer.py` source. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/trainer/sft_trainer.py. Fetched 2026-08-10; identical, byte-for-byte, to the same file at the `v0.6.10` tag.

[11] `mlx_vlm/trainer/orpo_trainer.py` source. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/trainer/orpo_trainer.py. Fetched 2026-08-10; identical, byte-for-byte, to the same file at the `v0.6.10` tag.

[12] `mlx_vlm/lora.py` source, `setup_model_for_training()` function (LoRA vs. full-finetune vs. train-vision branching, `apply_lora_layers` resume call). https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/lora.py. Fetched 2026-08-10.

[13] `mlx_vlm/trainer/utils.py` source, `not_supported_for_training` set. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/trainer/utils.py. Fetched 2026-08-10.

[14] mlx-vlm docs, Usage page, Speculative Decoding and Distributed Inference sections. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/docs/usage.md. Fetched 2026-08-10.

[15] mlx-vlm README, Distributed Inference section. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/README.md. Fetched 2026-08-10.

[16] mlx-vlm docs, Installation page. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/docs/installation.md. Fetched 2026-08-10.

[17] mlx-vlm PyPI JSON API (current version, upload time, license, `requires_python`). https://pypi.org/pypi/mlx-vlm/json. Fetched 2026-08-10.

[18] GitHub Releases API for Blaizzy/mlx-vlm (tag names and `published_at` dates for v0.6.6 through v0.6.10). https://api.github.com/repos/Blaizzy/mlx-vlm/releases. Fetched 2026-08-10.

[19] `requirements.txt` at the `v0.6.10` tag. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/v0.6.10/requirements.txt. Fetched 2026-08-10.

[20] GitHub Compare API, `v0.6.10...07bd830e7b46cd88ef90b859d5d4c145abff3dd5` (status "behind", 0 ahead, 30 behind); and Git Refs API resolving the `v0.6.10` tag to commit `c2fe301bb3f21c45985a423d7832bb238987312d`. https://api.github.com/repos/Blaizzy/mlx-vlm/compare/v0.6.10...07bd830e7b46cd88ef90b859d5d4c145abff3dd5 and https://api.github.com/repos/Blaizzy/mlx-vlm/git/refs/tags/v0.6.10. Fetched 2026-08-10.

[21] GitHub Repos API for Blaizzy/mlx-vlm (`pushed_at`, `archived`, `stargazers_count`, `license`). https://api.github.com/repos/Blaizzy/mlx-vlm. Fetched 2026-08-10.

[22] mlx-vlm docs, Usage page, CLI generate example. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/docs/usage.md. Fetched 2026-08-10.

[23] `mlx_vlm/lora.py` source, `main()` function (`--output-path` normalization, `--epochs`-to-`--iters` computation). https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/lora.py. Fetched 2026-08-10.

[24] `mlx_vlm/lora.py` source, full argparse default list (`--val-batches`, `--learning-rate`, `--grad-checkpoint`, and all other CLI flags). https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/lora.py. Fetched 2026-08-10.

[25] `mlx_vlm/trainer/sft_trainer.py` source, `TrainingArgs` dataclass default for `learning_rate`. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/trainer/sft_trainer.py. Fetched 2026-08-10.

[26] `mlx_vlm/LORA.MD`, Output section. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/LORA.MD. Fetched 2026-08-10.

[27] `mlx_vlm/trainer/utils.py` source, `save_adapter()` function. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/trainer/utils.py. Fetched 2026-08-10.

[28] `mlx_vlm/trainer/utils.py` source, `apply_lora_layers()` function. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/trainer/utils.py. Fetched 2026-08-10.

[29] `mlx_vlm/utils.py` source, `load()` function signature and adapter-loading branch. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mlx_vlm/utils.py. Fetched 2026-08-10.

[30] mlx-vlm README, "LoRA & QLoRA" section. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/README.md. Fetched 2026-08-10.

[31] mlx-vlm `mkdocs.yml` (site URL, nav slugs, `mike` version provider setting); live docs pages at https://blaizzy.github.io/mlx-vlm/ and https://blaizzy.github.io/mlx-vlm/usage/ (HTTP 200) and https://blaizzy.github.io/mlx-vlm/versions.json (HTTP 404), checked 2026-08-10. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/mkdocs.yml. Fetched 2026-08-10.

[32] mlx-vlm docs, Community Projects page. https://raw.githubusercontent.com/Blaizzy/mlx-vlm/07bd830e7b46cd88ef90b859d5d4c145abff3dd5/docs/community_projects.md. Fetched 2026-08-10.
