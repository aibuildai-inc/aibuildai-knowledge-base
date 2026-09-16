# finetrainers

A Hugging Face-maintained library for LoRA and full-rank finetuning of diffusion (video and image generation) models, not LLM post-training - one CLI (`train.py`) driven entirely by flags, with two interchangeable distributed backends.

**finetrainers** describes itself as "a work-in-progress library to support (accessible) training of diffusion models and various commonly used training algorithms" [1]. It lives at https://github.com/huggingface/finetrainers [1] and was transferred there from an individual maintainer's account, `a-r-r-o-w/finetrainers`, at some unstated point - the PyPI package metadata for the latest release still lists `github.com/a-r-r-o-w/finetrainers` as its home page [2], and the package's own `setup.py` at the screening commit still names Aryan V S as author [3]. Every model is trained through one script, `train.py`, driven by CLI flags documented in `docs/args.md` (or `python train.py --help`) [4], with no per-method trainer class the way an LLM post-training library exposes one.

**When to pick it**: training or LoRA-finetuning a diffusion image/video model (LTX-Video, HunyuanVideo, CogVideoX, CogView4, Flux, Wan) with SFT or channel-concatenated conditional control, when you want a single-flag CLI over model-specific training scripts. It is not a fit for LLM post-training methods (SFT-for-language-models aside, there is no DPO/PPO/GRPO/reward-modeling here) - the SFT and Control trainers are the only two that exist [5][6]; a ControlNet trainer and a Distillation trainer are named in the model support matrix but implemented for zero of the six supported models [7]. The README calls `main` the development branch and recommends checking out the latest release tag, `v0.2.0`, for stable use, warning that `main` "is unstable at the moment and may use higher memory" than the tagged benchmark numbers below [1].

**Methods it ships** [5][6][7]:
- SFT Trainer (`--training_type lora` or `full-finetune`): LoRA is the only low-rank method, sourced from Hugging Face PEFT; full-finetune trains all weights [7]. Supported for all six models: LTX-Video, HunyuanVideo, CogVideoX, CogView4, Flux, Wan [7].
- Control Trainer (`--training_type control-lora` or `control-full-finetune`): adds a channel-concatenated conditioning input trained jointly with (or grafted onto) the base transformer, the same family of mechanism as CogVideoX-I2V, HunyuanVideo-I2V, and Alibaba's Fun Control models [6]. Supported only for CogView4 and Wan [7].
- ControlNet and Distillation columns appear in the support-matrix table but show no supported model [7].
- The taxonomy is model-by-model and can change; recheck the live support-matrix table at [7] rather than this summary.

**Scale it handles**: single GPU up to multi-node, through two parallel backends selected by `--parallel_backend` - `accelerate` (default) or `ptd`, a raw PyTorch-DTensor path launched with `torchrun` [8][4]. Both expose the same degrees: `dp_degree`/`dp_shards` (data-parallel replicas / FSDP2-style sharding, both default 1), `cp_degree` (context parallelism, requires PyTorch 2.6+ [9]), and `pp_degree` (pipeline parallelism, documented as "currently unsupported") [8][4]. DDP, FSDP2, and HSDP are the parallelization forms the docs list as supported; tensor parallelism is only a placeholder in the same table (commented out) [8]. The docs' own multi-backend page calls the whole multi-backend effort "completely experimental," states the DTensor (`ptd`) path is the one with stable support "following Accelerate," and marks the multi-node launch command for BOTH backends as a `TODO` with no example given - only single-node launch commands (up to 8 GPUs) are shown [8]. So multi-node has a documented mechanism (the same `dp_degree`/`dp_shards`/`cp_degree` flags apply across nodes) but no published launch example or benchmark.

**Install**: `pip install finetrainers`, version 0.2.0, released 2025-04-25 [10][11]; the README's own recommended path is instead `git clone` + `pip install -r requirements.txt`, checking out tag `v0.2.0` for the stable release [1]. Python floor: PyPI lists `requires_python: >=3.8.0` [10]; licence Apache-2.0 [10][3]. `requirements.txt` at the v0.2.0 tag pins `torch>=2.5.1`, `torchvision>=0.20.1`, `diffusers>=0.32.1`, `transformers>=4.45.2`, `accelerate` and `bitsandbytes` and `huggingface_hub` unpinned, `peft>=0.13.0`, `datasets>=3.3.2` - no upper bound on any of them [12]. The README separately recommends installing `diffusers` from its `main` branch rather than the pinned floor, "for the latest features and bugfixes" [1]. Hardware/CUDA: no explicit minimum is stated; the environment page instead says finetrainers "has only been widely tested with" one exact stack - PyTorch 2.5.1+cu124, Python 3.10.14, on A100-80GB GPUs - and separately requires PyTorch 2.6+ specifically for context parallelism [9]. The screening commit (`7e9257a`, the repository's newest push, from `main`) is ahead of this release: its `setup.py` reports version `0.2.0.dev0` against the tagged `0.2.0` [3][13], so code-level claims on this card sourced from that commit describe `main`, not what `pip install finetrainers` currently delivers.

**Maintained by**: Hugging Face, after transfer from an individual maintainer's GitHub account (former name `a-r-r-o-w/finetrainers`) [1][2]. The README's dated News section runs from 2024-12-18 to 2025-04-25, each entry naming a feature landing (Wan I2V, per-model attention providers, `torch.compile` support, Flux support) [1]; the repository's most recent push, per the GitHub API, is 2026-05-26 [13], over a year after the last dated News entry, so recent activity is not narrated in the README itself.

## Quick start

The README's quickstart is: clone the repo, `pip install -r requirements.txt`, install `diffusers` from source, then check out the stable tag and run one of the bundled example scripts [1]:

```bash
git fetch --all --tags
git checkout tags/v0.2.0
chmod +x ./examples/training/sft/ltx_video/crush_smol_lora/train.sh
./examples/training/sft/ltx_video/crush_smol_lora/train.sh
```

That script is a complete, runnable LoRA SFT run on LTX-Video: it sets `--model_name "ltx_video" --pretrained_model_name_or_path "a-r-r-o-w/LTX-Video-diffusers"`, `--training_type "lora" --rank 32 --lora_alpha 32`, `--optimizer "adamw" --lr 5e-5 --lr_scheduler "constant_with_warmup"`, `--gradient_checkpointing --checkpointing_steps 1000 --checkpointing_limit 2`, and `--report_to "wandb"` [14]. The README also names the compatible, ready-to-use datasets for a first run: the `Wild-Heart/Disney-VideoGeneration-Dataset` Hub dataset, the `bigdata-pw` org's video-dataset collection, and Hugging Face's own `finetrainers` Hub dataset collection [1].

## Start it

- One GPU: run `train.py` directly with the flags above.
- Several GPUs/nodes go through one of two launchers, selected by `--parallel_backend`:
  - `accelerate` (default): `accelerate launch --config_file accelerate_configs/uncompiled_<N>.yaml --gpu_ids <ids> train.py <args>`, with a ready config template per GPU count (`uncompiled_1.yaml`, `uncompiled_2.yaml`, ... in the repo's `accelerate_configs/` directory) plus a `deepspeed.yaml` template for DeepSpeed [8][15][16].
  - `ptd`: `torchrun --standalone --nnodes=1 --nproc_per_node=<N> --rdzv_backend c10d --rdzv_endpoint="localhost:0" train.py <args>` [8][14].
  - Multi-node launch commands for either backend are placeholders in the docs (`# TODO(aryan): Add slurm script`); no example is published [8].
- Effective batch: per-device `batch_size` (default 1) x device count x `gradient_accumulation_steps` (default 1) [4].
- Config surface is CLI flags grouped by area, under the banners `docs/args.md` itself uses: Parallel, Model, Dataset, Dataloader, Diffusion, Training, Optimizer, Validation, Miscellaneous, Torch Config, plus an Attention Provider subsection and separate SFT-training and Control-training blocks - documented at `docs/args.md` or `python train.py --help` [4]. The library deliberately does not expose a `mixed_precision` CLI flag - that lives only in the Accelerate config YAML - "because it can be confusing to have `transformer_dtype` and `mixed_precision` in the codebase"; `transformer_dtype` (default `bfloat16`) is what actually controls training precision, and the docs state it "will also most likely always have to be `torch.bfloat16`" because most supported models do not work well in FP16 [17]. `text_encoder_dtype` and `vae_dtype` likewise default to `bfloat16` [4].
- OOM first aid, from the memory-optimizations doc [18]: `--enable_precomputation` (this file's arg name; note `docs/models/optimization.md`, read at the same commit, instead calls it `--precompute_conditions` - the two docs pages have drifted [18][4]) precomputes and caches text/latent conditioning to disk so conditioning models need not stay resident; `--gradient_checkpointing` trades recompute for activation memory; `--layerwise_upcasting_modules transformer` stores weights in `float8_e4m3fn`/`float8_e5m2` while computing in `transformer_dtype`, halving weight memory; an 8-bit bitsandbytes optimizer (`adam-bnb-8bit`/`adamw-bnb-8bit`) halves optimizer-state memory; a DeepSpeed config via `accelerate_configs/deepspeed.yaml`; or skipping validation entirely. A stale `precomputed/` folder left over from an earlier run - especially one that mixed the `ptd` and `accelerate` backends - can silently corrupt a later run and was reported as a training crash in issue #348 (closed 2025-04-27); maintainer a-r-r-o-w (CONTRIBUTOR) traced it to leftover precomputation files not being cleaned up, landed a fix in PR #389, and confirmed that precomputed embeddings are not reusable across separate runs unless `--precomputation_reuse` is explicitly set [27]. The README's own VRAM table - measured on the older `v0.0.1` release branch, not `main`, at resolution 49x512x768, rank 128, FP8 weights with gradient checkpointing, no validation - reports LoRA-training minimums of 5 GB (LTX-Video), 32 GB (HunyuanVideo), 18 GB (CogVideoX-5b), and full-finetune minimums of 21 GB (LTX-Video), OOM (HunyuanVideo, i.e. it does not fit at this measured setting), 53 GB (CogVideoX-5b) [1].

## Watch it

Mechanics only - what a logged value means for training health is not covered here.

- Enable it with `--report_to`. The `docs/args.md` docstring and the `TrainingArgs` dataclass both say the default is `wandb` [4], but the actual `argparse` definition - the code path that runs - sets `default="none", choices=["none", "wandb"]`, so an unconfigured run logs nowhere by default; this docstring-vs-argparse mismatch exists in the source at both the screening commit and the tagged v0.2.0 release [26]. Passing anything outside `{"none", "wandb"}` is rejected by `argparse` before training starts; there is no TensorBoard or other backend, and `report_to="none"` is what returns the no-op tracker below, not an invalid choice [19][26]. With `report_to` empty, `initialize_trackers` returns a no-op `BaseTracker` that logs nothing [19]. Cadence is `--logging_steps`, default 1 [4].
- Logged scalar metric names, read directly from the SFT trainer's training loop at the screening commit (no dedicated logging-metrics doc page exists) [20]: `train/global_avg_loss`, `train/global_max_loss`, `train/grad_norm` (only when not `None`), `train/observed_data_samples`, plus five timing metrics from `tracker.timed(...)` calls: `timing/batch_preparation`, `timing/forward`, `timing/backward`, `timing/optimizer_step`, `timing/checkpoint` [20]. This is a `main`-branch (commit `7e9257a`) reading, ahead of the tagged `v0.2.0` release.
- Sample-level logging: during validation, generated images/videos are written to disk under `output_dir` as `validation-{step}-{rank}-{index}-{prompt_filename}-{timestamp}.{ext}` (or `final-...` for the end-of-training validation pass), and simultaneously logged to Weights & Biases under a top-level `"validation"` or `"final"` key holding nested `"images"`/`"videos"` lists of `wandb.Image`/`wandb.Video` objects captioned with the validation prompt [20].
- Evaluation during training: `--validation_dataset_file` points at a CSV/JSON/PARQUET/ARROW file with at least a `"caption"` column (optionally `"image_path"`/`"video_path"` and per-sample overrides like resolution or `guidance_scale`); `--validation_steps` (default 500) sets the cadence; `--enable_model_cpu_offload` offloads components to CPU during validation to save memory [4].
- Stopping: no early-stopping or reward/loss-threshold flag is documented anywhere in `docs/args.md` [4] - the only stopping controls are `--train_steps` (default 1000, a fixed step budget) and `--max_data_samples` (default 2**64), which stops training early once that many samples have been observed, before `train_steps` is reached [4]. No RL-style reward or health threshold is published, since neither trainer is an RL method.

## Save it

- Two separate things land on disk per checkpoint interval, both driven by `--checkpointing_steps` (default 500) and pruned by `--checkpointing_limit` [4][21][22]:
  - A full resumable state at `{output_dir}/finetrainers_step_{step}/`, written via Accelerate's `accelerator.save_state(..., safe_serialization=True)` plus a hand-rolled `states.pt` holding the custom training state (optimizer/scheduler/dataloader progress) - this is what `--resume_from_checkpoint` reads back with `accelerator.load_state(...)` [21].
  - A lightweight, directly usable weight snapshot fired by the same save hook: `{output_dir}/lora_weights/{step:06d}/` for LoRA runs (written through the model's diffusers pipeline, e.g. `LTXPipeline.save_lora_weights(...)` for LTX-Video) or `{output_dir}/model_weights/{step:06d}/` for full-finetune runs (`transformer.save_pretrained(...)`) [20][23]. This save path is confirmed from the LTX-Video model specification only, at the screening commit; the other five model specs were not individually checked and may implement the same base-class methods differently.
- Resume: `--resume_from_checkpoint <STEP_OR_LATEST>` takes an integer step or the string `"latest"` [4][22]. It only works if the parallel backend and every degree of parallelism (e.g. `dp_degree`, `dp_shards`) match the run that produced the checkpoint - the docs give the concrete failure case: switching a saved run's `--dp_degree 2 --dp_shards 1` to `--dp_degree 1 --dp_shards 2` on resume will not work [22].
- `--push_to_hub` uploads the entire `output_dir` to the Hub, but excludes the resumable `finetrainers_step_*` directories via an `ignore_patterns` filter - only the lightweight weight snapshots and validation artifacts are pushed [20].
- An adapter directory (`lora_weights/{step:06d}/`) is NOT a full model: it holds only the trained LoRA weights. Reload for inference by loading the base pipeline and calling `pipe.load_lora_weights(...)` - the README's own inference snippet for LTX-Video does exactly this against a Hub-pushed adapter repo [24].
- Whether an evaluator can load a saved checkpoint directly depends on which of the two directories it opens: a `lora_weights`/`model_weights` snapshot loads through the model's own diffusers pipeline as shown above; a `finetrainers_step_*` directory is Accelerate training state, not a deployable model, and is meant only for `--resume_from_checkpoint`.

## Find it in the docs

There is no separate hosted docs site for finetrainers - the docs are the `docs/` directory of the GitHub repository itself, rendered as plain Markdown, read at `raw.githubusercontent.com/huggingface/finetrainers/<ref>/docs/<page>.md` or browsed on GitHub at `github.com/huggingface/finetrainers/blob/<ref>/docs/<page>.md`; `<ref>` is a branch (`main`) or tag (`v0.2.0`) [1][4].

- The full CLI reference is one page, `docs/args.md`, grouped by argument category under its `## General` header (Parallel, Model, Dataset, Dataloader, Diffusion, Training, Optimizer, Validation, Miscellaneous, Torch Config, plus an Attention Provider subsection and separate SFT- and Control-specific blocks); the page itself says "for more information, please take a look at the `finetrainers/args.py` file," and carries an internal note about keeping the page auto-updated, so treat it as close to but not guaranteed identical to the code [4].
- Per-model pages live under `docs/models/<model>.md` (`ltx_video.md`, `hunyuan_video.md`, `cogvideox.md`, `cogview4.md`, `flux.md`, `wan.md`), each with an example command, an inference snippet, and memory numbers; the index page `docs/models/README.md` carries the SFT/Control/ControlNet/Distillation support matrix and the resume-compatibility and `mixed_precision` notes above [7]. Dataset formats and the precomputation system are documented at `docs/dataset/README.md`, which also names the external `huggingface/video-dataset-scripts` repo for dataset preparation and documents a specific failure mode: when `dp_shards > 1` or `tp_degree > 1`, the sample count must be a multiple of `dp_shards * tp_degree` or the run fails via an NCCL timeout - a known, undocumented-as-fixed limitation [25]. Parallel-backend mechanics live at `docs/parallel/README.md` [8].
- Runnable references beyond the docs: the `examples/training/` tree ships three complete Pika-Effects-replication scripts (LTX-Video, CogVideoX, Wan) [1][14], and `examples/inference/` for inference-only usage [8].
- Community layer: the README's own "Featured Projects" and "UIs built for finetrainers" lists are the closest thing to an official curation - both are one-line linked lists (no author/date/version metadata given), including two third-party UIs (`jbilcke-hf/VideoModelStudio`, `neph1/finetrainers-ui`) and community forks such as an LTX-Video image-to-video extension; treat any of these against your own installed version, since none carry a pinned finetrainers version in the README's listing [1].
- No official MCP endpoint for these docs was found in the pages fetched for this card.
- Honest boundary: this is a diffusion-model trainer, not an LLM post-training library - no DPO/PPO/GRPO/reward-modeling trainer exists here [5][6]. Multi-node launch is undocumented (`TODO` placeholders for both backends) [8]. Tensor and pipeline parallelism are listed but unsupported/commented-out, not working options [8][4]. Only `wandb` and no-op logging exist, defaulting to no-op unless `--report_to wandb` is set explicitly; there is no TensorBoard integration [19][26]. The precomputation-folder trap in issue #348 (above) is the maintainer-confirmed failure mode found for this card [27].

## Sources

[1] finetrainers README, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://github.com/huggingface/finetrainers/blob/7e9257aae285c2fd665b39faf5c4109762762438/README.md. Fetched 2026-08-12.

[2] finetrainers PyPI project metadata (JSON API). https://pypi.org/pypi/finetrainers/json. Fetched 2026-08-12.

[3] finetrainers `setup.py`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/setup.py. Fetched 2026-08-12.

[4] finetrainers CLI argument reference, `docs/args.md`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/docs/args.md. Fetched 2026-08-12.

[5] finetrainers `docs/trainer/sft_trainer.md`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/docs/trainer/sft_trainer.md. Fetched 2026-08-12.

[6] finetrainers `docs/trainer/control_trainer.md`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/docs/trainer/control_trainer.md. Fetched 2026-08-12.

[7] finetrainers `docs/models/README.md` (support matrix, resume-compatibility note, `mixed_precision` rationale), screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/docs/models/README.md. Fetched 2026-08-12.

[8] finetrainers `docs/parallel/README.md`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/docs/parallel/README.md. Fetched 2026-08-12.

[9] finetrainers `docs/environment.md`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/docs/environment.md. Fetched 2026-08-12.

[10] finetrainers PyPI project metadata (version, licence, Python floor, release dates). https://pypi.org/pypi/finetrainers/json. Fetched 2026-08-12.

[11] finetrainers GitHub release page, tag v0.2.0. https://github.com/huggingface/finetrainers/releases/tag/v0.2.0. Fetched 2026-08-12.

[12] finetrainers `requirements.txt` at tag v0.2.0. https://raw.githubusercontent.com/huggingface/finetrainers/v0.2.0/requirements.txt. Fetched 2026-08-12.

[13] finetrainers repository metadata, GitHub REST API. https://api.github.com/repos/huggingface/finetrainers. Fetched 2026-08-12.

[14] finetrainers LTX-Video Pika-Effects example launch script, `examples/training/sft/ltx_video/crush_smol_lora/train.sh`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/examples/training/sft/ltx_video/crush_smol_lora/train.sh. Fetched 2026-08-12.

[15] finetrainers `accelerate_configs/uncompiled_1.yaml`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/accelerate_configs/uncompiled_1.yaml. Fetched 2026-08-12.

[16] finetrainers `docs/models/optimization.md` (memory-optimization list, DeepSpeed config pointer), screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/docs/models/optimization.md. Fetched 2026-08-12.

[17] finetrainers `docs/models/README.md` (`mixed_precision` vs `transformer_dtype` section). Same URL as [7]. Fetched 2026-08-12.

[18] finetrainers `docs/models/optimization.md`. Same URL as [16]. Fetched 2026-08-12.

[19] finetrainers tracker module, `finetrainers/trackers.py`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/finetrainers/trackers.py. Fetched 2026-08-12.

[20] finetrainers SFT trainer training loop, `finetrainers/trainer/sft_trainer/trainer.py`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/finetrainers/trainer/sft_trainer/trainer.py. Fetched 2026-08-12.

[21] finetrainers Accelerate checkpointer, `finetrainers/parallel/accelerate.py`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/finetrainers/parallel/accelerate.py. Fetched 2026-08-12.

[22] finetrainers `docs/models/README.md` ("Resuming training" section). Same URL as [7]. Fetched 2026-08-12.

[23] finetrainers LTX-Video model specification, `finetrainers/models/ltx_video/base_specification.py`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/finetrainers/models/ltx_video/base_specification.py. Fetched 2026-08-12.

[24] finetrainers `docs/models/ltx_video.md` (inference snippet with `load_lora_weights`), screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/docs/models/ltx_video.md. Fetched 2026-08-12.

[25] finetrainers `docs/dataset/README.md`, screening commit 7e9257aae285c2fd665b39faf5c4109762762438. https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/docs/dataset/README.md. Fetched 2026-08-12.

[26] finetrainers `finetrainers/args.py`, the `argparse` definition of `--report_to`, read both at the screening commit and at the v0.2.0 release tag (identical at both). https://raw.githubusercontent.com/huggingface/finetrainers/7e9257aae285c2fd665b39faf5c4109762762438/finetrainers/args.py and https://raw.githubusercontent.com/huggingface/finetrainers/v0.2.0/finetrainers/args.py. Fetched 2026-08-12.

[27] finetrainers GitHub issue #348, "Exception: Error during training: Given groups=1, weight of size [3072, 16, 1, 2, 2]...", opened by neph1, closed 2025-04-27, with maintainer a-r-r-o-w's (CONTRIBUTOR) closing comment and PR #389 fix referenced in-thread. https://github.com/huggingface/finetrainers/issues/348. Fetched 2026-08-12.
