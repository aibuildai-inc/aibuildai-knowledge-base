# Ovis

The training codebase for Alibaba's Ovis multimodal LLM family: one supervised fine-tuning script wired to the model's own custom architecture, launched with `torchrun` and DeepSpeed - it fine-tunes Ovis checkpoints specifically, not a general model zoo.

Ovis (Open VISion) is described in its own README as "a novel Multimodal Large Language Model (MLLM) architecture, designed to structurally align visual and textual embeddings" [1]. It is built by Alibaba's Ovis team [1], and the repository lives on GitHub under the org ATH-MaaS after being renamed from AIDC-AI/Ovis - the old address now 301-redirects to the current one [2]. Its post-training surface is a single entry point, `ovis/train/train.py`, which loads an existing Ovis checkpoint with `trust_remote_code=True`, wraps it in a `transformers.Trainer`, and exposes fine-tuning through CLI flags rather than one trainer class per method [3]. It lives at https://github.com/ATH-MaaS/Ovis.

**When to pick it**: fine-tuning an existing Ovis checkpoint (Ovis2.5, Ovis2, Ovis1.6, ...) with supervised fine-tuning on your own conversation or caption data, using the model's own training loop - a custom multimodal data collator and module-selective learning rates - rather than a general multi-model trainer. The README itself names the alternative for broader model coverage: ms-swift, described there only as "a flexible training framework for LLMs" [1]. This repository ships no other post-training method: no DPO or RL trainer is present, even though a past release (Ovis1.6-Gemma2-9B) is described as having added DPO training after instruction-tuning - that DPO code is not in this repository [1].

**Methods it ships**: SFT only, through `ovis/train/train.py`, built directly on `transformers.Trainer` rather than a distinct trainer class per method [3]. The repository has exactly one training script, `scripts/run_ovis2_5_sft.sh` [4], and exactly one dataset-info file, `ovis/train/dataset/ovis2_5_sft_datainfo.json` [5]. Fine-grained control comes from `--train_modules`, which selects which submodules get unfrozen - `all`, `llm`, `visual_tokenizer`, `visual_tokenizer.head`, `visual_tokenizer.vit`, `visual_tokenizer.vit.last_block`, or `vte` - each addressable with its own learning rate via `module_name:lr` [3]. This is selective full-parameter fine-tuning, not PEFT/LoRA; no adapter code exists in this repository, and `model.save_pretrained` always writes full model weights [6].

**Scale it handles**: single GPU up to multi-GPU on one node, launched with `torchrun --nproc_per_node=N ovis/train/train.py`, with DeepSpeed sharding via `--deepspeed`; ZeRO stage 0/1/2/3 config templates ship at `scripts/zero_configs/zero{0,1,2,3}_cp.json` [7]. The only published launch example uses `--nproc_per_node=2` on a single node [4]; no multi-node launch form (hostfile, SLURM, or otherwise) appears in the README or the `scripts/` directory, so multi-node is undocumented here.

**Install**: no PyPI package and no GitHub release or tag exist for this repository - checked against the repository's release and tag Atom feeds, both of which return zero entries [8] - so install is from source at a commit: `git clone`, `pip install -r requirements.txt`, `pip install -e .` [1]. `requirements.txt` pins `torch==2.4.0`, `transformers==4.51.3`, `accelerate==1.1.0`, `deepspeed==0.15.4`, `tokenizers==0.21.1`, `pillow==10.3.0` [9]; the README states it has been tested with Python 3.10 [1]. Licence is Apache-2.0 [10]. `ovis/train/train.py` imports `flash_attn` directly, and the shipped SFT script passes `--attn_implementation flash_attention_2` [3][4], but `flash_attn` is not listed in `requirements.txt` [9] - it must be installed separately, and it requires a CUDA GPU that FlashAttention-2 supports. No other CUDA or hardware minimum is stated anywhere read for this card. `setup.py` names the package version `2.5.0`, but this string is not published to PyPI [11].

**Maintained by**: Alibaba's Ovis team [1], published under the GitHub org ATH-MaaS (908 followers, no org description) [12], after a rename from AIDC-AI/Ovis - the old GitHub address 301-redirects here [2]. The two most recent commits on `main`, both authored by the same GitHub user `runninglsy` (email `runninglsy@gmail.com` on the newer commit, `running.lsy@alibaba-inc.com` on the older one), are a 2026-07-15 commit that adds a technical-report PDF (`docs/OvisOCR2_Tech_Report.pdf`) and a 2025-09-22 commit that updates the README [13][14] - neither touches the training code, so recent activity is documentation, not evidence of code changes to the fine-tuning path.

## Quick start

The README gives no standalone Python fine-tuning snippet; the smallest complete run is the shipped shell script. Edit the checkpoint and dataset paths, then run [1][4]:

```bash
git clone git@github.com:AIDC-AI/Ovis.git
conda create -n ovis python=3.10 -y
conda activate ovis
cd Ovis
pip install -r requirements.txt
pip install -e .
```

```bash
bash scripts/run_ovis2_5_sft.sh
```

That script hardcodes `OVIS_CKPT_DIR="AIDC-AI/Ovis2.5-9B"` and `data_name="geometry3k_local"`, and ends with [4]:

```bash
torchrun --nproc_per_node=2 ovis/train/train.py $CMDARG
```

To point it at your own data, add an entry to a datainfo JSON file such as `ovis/train/dataset/ovis2_5_sft_datainfo.json`, whose one shipped entry has the shape [5]:

```json
{
    "geometry3k_local": {
        "meta_file": "path/to/geometry3k_local.json",
        "storage_type": "hybrid",
        "data_format": "conversation",
        "image_dir": "path/to/images/"
    }
}
```

## Start it

- One process, one GPU: run `ovis/train/train.py` directly with the `ModelArguments`/`TrainingArguments` CLI flags shown in `run_ovis2_5_sft.sh`, dropping `torchrun` [3][4].
- Multiple GPUs on one node: `torchrun --nproc_per_node=N ovis/train/train.py $CMDARG`, with sharding selected by `--deepspeed scripts/zero_configs/zero{0,1,2,3}_cp.json` (ZeRO stage 0 through 3) [4][7]. No multi-node launch form is documented in this repository.
- Effective batch size is `per_device_train_batch_size x devices x gradient_accumulation_steps`, the same as any `transformers.Trainer` run; the shipped script uses `per_device_train_batch_size=2` and `gradient_accumulation_steps=16` on 2 GPUs [4].
- Configuration is CLI flags parsed into `ModelArguments` and a `TrainingArguments` subclass of the `transformers.TrainingArguments` [12]. Two behaviors are set silently in `__post_init__`, not by the shipped script: `gradient_checkpointing_kwargs` is forced to `{"use_reentrant": False}` whenever `--gradient_checkpointing` is set, and `--stage < 3` forces `save_safetensors=False` (checkpoints below DeepSpeed ZeRO stage 3 are saved in the legacy PyTorch format instead) [12]. `TrainingArguments.__post_init__` also asserts `model_init_seed != seed` [12] - the two seeds must differ, or the run raises at startup.
- Out-of-memory first aid: no dedicated OOM guidance is published in the README or in `arguments.py`/`train.py` read for this card; the general `transformers.Trainer` levers apply - lower `per_device_train_batch_size`, raise `gradient_accumulation_steps`, move to a higher DeepSpeed ZeRO stage, or reduce `--single_image_max_pixels`/`--multiple_image_max_pixels`/`--multimodal_max_length`, which are Ovis-specific flags controlling image-token and sequence length that this codebase does expose [3][4][12].

## Watch it

This section is mechanics only - trl and verl carry the shapes for SFT loss curves on their own cards; nothing library-specific about the meaning of a given signal is claimed here.

- **Enable it**: the shipped script passes `--report_to none` [4]; `TrainingArguments` is a `transformers` subclass, so any backend `transformers` supports (`wandb`, `tensorboard`, ...) can be set via the same `report_to` flag, but the default in the provided example produces no external tracker record - only console output from the standard `transformers.Trainer` logging plus the callback below [4][12].
- **Metric names**: no Ovis-specific metric names or logged-metrics list is published; the loss, learning rate, and gradient-norm fields logged at `--logging_steps` come from stock `transformers.Trainer` logging, not from anything defined in this repository.
- **`MonitorCallback`**: a repository-specific callback that gathers the tensors returned by the model's `get_monitor_tensors()` method under DeepSpeed and prints their sum and full contents to stdout, at `on_step_begin` when `step % monitor_step == 0` or `step == 10`, and again at `on_epoch_end`; `monitor_step` defaults to 100 [12][15]. This is a debugging print, not a metric logged to any tracker.
- **A silent-failure trap, from a maintainer reply**: in closed issue #41, a repository collaborator (`runninglsy`) explained that the dataset's `invalidate_label` flag marks a sample corrupted (for example, an image download that failed over the network) by setting every label in that sample to `-100`, so the sample contributes no loss; the reply warns that if this flag is `True` most of the time, check your training data and code, because the run will otherwise train silently on near-empty labels (issue #41, closed 2025-01-25) [16].
- **Evaluate during training**: `TrainingArguments` inherits the standard `eval_strategy`/`eval_steps`/`per_device_eval_batch_size` fields from `transformers`, but the shipped script sets `--eval_strategy no` and passes no `eval_dataset` construction path in `train.py`'s `train()` function - only a `train_dataset` is built [3][4].
- **Stopping**: no RL-specific or SFT-specific stopping rule, threshold, or patience value is published anywhere read for this card - checked 2026-08-12 across the README [1], `train.py` [3], and `arguments.py` [12]; none of the three defines an early-stopping or divergence-threshold field beyond what stock `transformers.Trainer` callbacks would add.

## Save it

- `train.py` calls `trainer.save_state()` then `trainer.save_model()` once training finishes, writing the final model to `training_args.output_dir` [3]. Intermediate checkpoints land under the same `output_dir` in the standard `transformers.Trainer` numbered `checkpoint-<step>/` layout, cadence set by `--save_strategy`/`--save_steps`/`--save_total_limit` (the shipped script uses `save_steps 0.4` with `save_total_limit 10`) [4][12]. `save_steps` is `transformers.TrainingArguments` surface, not Ovis-specific: a float below 1 is read as a ratio of total training steps, and a save is always additionally performed at the very end of training regardless of that ratio [18] - so `save_steps 0.4` over one epoch yields checkpoints at roughly 40% and 80% of training plus the always-performed final save, not a fixed count per epoch.
- Ovis overrides `Ovis.save_pretrained` to additionally write the text tokenizer and the visual tokenizer's image processor into the same directory alongside the model weights, so a saved directory is self-contained for reloading, not just a weights file [6].
- **`save_safetensors` is stage-gated, not a flag you set**: `TrainingArguments.__post_init__` forces `save_safetensors=False` whenever `--stage < 3`, meaning checkpoints saved below DeepSpeed ZeRO stage 3 use the legacy `.bin` format instead of safetensors [12] - there is no adapter-only save path to worry about, since this codebase does not implement PEFT.
- Resume: `train.py` checks `pathlib.Path(training_args.output_dir).glob("checkpoint-*")` at startup and, if any checkpoint exists, calls `trainer.train(resume_from_checkpoint=True)`; otherwise it calls `trainer.train()` with no argument - resuming is automatic based on what is already in `output_dir`, not a separate flag you pass [3].
- Because this codebase performs full-parameter (or module-selective) fine-tuning and not PEFT, every checkpoint directory is a complete model. `train.py`'s own `load_model()` loads a checkpoint the same way it loads the starting pretrained model, with `Ovis.from_pretrained(<path>, trust_remote_code=True)` [3]; the README's separate vLLM serving example loads a checkpoint with `vllm serve <checkpoint_dir> --trust-remote-code` [1]. Whether a downstream evaluator's own loader accepts a raw `transformers`-style directory with custom modeling code is that loader's contract, not something this card can settle.

## Find it in the docs

There is no separate hosted documentation site for this repository - the README at the repository root is the documentation, and it is a single moving page, not versioned like the other cards in this deck.

- Address pattern: `https://github.com/ATH-MaaS/Ovis/blob/main/<path>` for any file, `https://github.com/ATH-MaaS/Ovis/tree/main/<path>` for any directory; there is no `<version>` segment to swap in, because no tag or release exists (checked against the repository's release and tag Atom feeds, both returning zero entries) [8]. Every claim on this card is pinned to commit `e1916bf3cb0ca123e71ab173ff4463df5c3a71b5` on `main`, read 2026-08-12 [1][3][4][5][6][7][9][10][11][12][15].
- The README's own table of contents is the map of the page: Model, Performance, Install, Inference, Model Fine-tuning, Citation, Team, License, Disclaimer [1]. The fine-tuning instructions - data format, datainfo JSON shape, and the pointer to `scripts/run_ovis2_5_sft.sh` - sit entirely under the "Model Fine-tuning" heading [1].
- Runnable references beyond the README: the training code itself under `ovis/train/` (dataset classes, arguments, callback, `train.py`) [3][12][15], the DeepSpeed config templates under `scripts/zero_configs/` [7], and the technical report on arXiv for architecture and training-recipe background - "Ovis2.5 Technical Report" [17].
- No community-tutorials page or MCP endpoint is published by this repository; the README's only pointer to a broader ecosystem is the one-line mention of ms-swift as an alternative fine-tuning framework, with no further detail given [1].
- Honest boundary: this repository trains Ovis checkpoints only, with SFT only, on a single node; it documents no DPO/RL trainer, no multi-node launch, no PEFT/adapter path, and no published OOM or stopping-rule guidance - each of those gaps is stated at the point above where it would otherwise be expected.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. This repository publishes no versioned docs site and no release or tag; every statement is pinned to commit `e1916bf3cb0ca123e71ab173ff4463df5c3a71b5` on the `main` branch, fetched 2026-08-12, except the org and issue-history checks noted individually. Method name (SFT) is deliberately cited to nothing here; its defining literature lives on the methodology card, not this one.

[1] ATH-MaaS/Ovis README.md, main branch at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5. https://github.com/ATH-MaaS/Ovis/blob/main/README.md. Fetched 2026-08-12.

[2] Redirect check: https://github.com/AIDC-AI/Ovis returns HTTP 301 to https://github.com/ATH-MaaS/Ovis. Checked 2026-08-12.

[3] ovis/train/train.py at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5. https://github.com/ATH-MaaS/Ovis/blob/main/ovis/train/train.py. Fetched 2026-08-12.

[4] scripts/run_ovis2_5_sft.sh at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5. https://github.com/ATH-MaaS/Ovis/blob/main/scripts/run_ovis2_5_sft.sh. Fetched 2026-08-12.

[5] ovis/train/dataset/ovis2_5_sft_datainfo.json at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5. https://github.com/ATH-MaaS/Ovis/blob/main/ovis/train/dataset/ovis2_5_sft_datainfo.json. Fetched 2026-08-12.

[6] ovis/model/modeling_ovis.py at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5 (save_pretrained override, AutoConfig/AutoModelForCausalLM registration). https://github.com/ATH-MaaS/Ovis/blob/main/ovis/model/modeling_ovis.py. Fetched 2026-08-12.

[7] scripts/zero_configs/ (zero0_cp.json, zero1_cp.json, zero2_cp.json, zero3_cp.json) at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5. https://github.com/ATH-MaaS/Ovis/tree/main/scripts/zero_configs. Fetched 2026-08-12.

[8] ATH-MaaS/Ovis release and tag Atom feeds, both with zero `<entry>` elements. https://github.com/ATH-MaaS/Ovis/releases.atom and https://github.com/ATH-MaaS/Ovis/tags.atom. Fetched 2026-08-12.

[9] requirements.txt at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5. https://github.com/ATH-MaaS/Ovis/blob/main/requirements.txt. Fetched 2026-08-12.

[10] LICENSE at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5. https://github.com/ATH-MaaS/Ovis/blob/main/LICENSE. Fetched 2026-08-12.

[11] setup.py at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5 (version string, not published to PyPI - checked against https://pypi.org/pypi/ovis/json, 2026-08-12). https://github.com/ATH-MaaS/Ovis/blob/main/setup.py. Fetched 2026-08-12.

[12] ovis/train/arguments.py at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5 (ModelArguments/TrainingArguments fields and __post_init__). https://github.com/ATH-MaaS/Ovis/blob/main/ovis/train/arguments.py. Fetched 2026-08-12.

[13] Commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5 (adds docs/OvisOCR2_Tech_Report.pdf, authored by runninglsy). https://github.com/ATH-MaaS/Ovis/commit/e1916bf3cb0ca123e71ab173ff4463df5c3a71b5. Fetched 2026-08-12.

[14] Commit 4de4946166ee40a2c3c6209b7ef57c50a7eaaf99 (README update, authored by runninglsy, running.lsy@alibaba-inc.com). https://github.com/ATH-MaaS/Ovis/commit/4de4946166ee40a2c3c6209b7ef57c50a7eaaf99. Fetched 2026-08-12.

[15] ovis/train/callback.py at commit e1916bf3cb0ca123e71ab173ff4463df5c3a71b5 (MonitorCallback). https://github.com/ATH-MaaS/Ovis/blob/main/ovis/train/callback.py. Fetched 2026-08-12.

[16] GitHub issue #41, "A few questions about `train.py`" (closed 2025-01-25), comment by runninglsy (repository collaborator). https://github.com/ATH-MaaS/Ovis/issues/41. Fetched 2026-08-12.

[17] "Ovis2.5 Technical Report", arXiv:2508.11737. https://arxiv.org/abs/2508.11737. Fetched 2026-08-12.

[18] `transformers` 4.51.3 (the version `requirements.txt` [9] pins) `training_args.py` docstring for `save_steps` and `save_strategy` (ratio interpretation below 1, always-performed final save). PyPI wheel `transformers-4.51.3-py3-none-any.whl`. Downloaded and inspected 2026-08-12.
