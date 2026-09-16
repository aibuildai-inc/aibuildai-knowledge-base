# alignment-handbook

Hugging Face's open reproduction recipes for aligning chat models — YAML configs and thin CLI scripts wrapped around trl's SFT, DPO, and ORPO trainers, not a trainer library of its own.

The Alignment Handbook describes itself as offering "robust recipes to continue pretraining and to align language models with human and AI preferences" [1]. It is built and maintained by Hugging Face, with a named author list spanning Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Shengyi Huang, Kashif Rasul, Alvaro Bartolome, Carlos M. Patiño, Alexander M. Rush, and Thomas Wolf [1]. Its API shape is a set of task scripts (`scripts/sft.py`, `scripts/dpo.py`, `scripts/orpo.py`) that each parse a YAML config plus CLI overrides into `trl.ModelConfig` and an `alignment`-package Config subclass, then hand them straight to the matching trl trainer [7][8]. It lives at https://github.com/huggingface/alignment-handbook [3].

**When to pick it**: you want the exact, published, reproducible recipe (YAML config plus launch command) behind a specific released model — Zephyr-7B-β, StarChat2, SmolLM/SmolLM2/SmolLM3 — and are willing to read and adapt someone else's SFT→DPO/ORPO pipeline rather than write your own trainer call; it is a config-and-script layer on top of trl, not a substitute for trl itself, and at this commit it ships only SFT, DPO, and ORPO as runnable scripts, not a general RLHF/RL framework [1][2][7]. The recipe's own README does not publish a target score, only stating that it changed hyperparameters from the original paper run to "achieve comparable performance to zephyr-7b-beta" [14]; the number that decides whether a rerun matches is on the released model's own Hub page: `HuggingFaceH4/zephyr-7b-beta` reports an MT-Bench score of 7.34 and an AlpacaEval win rate of 0.906 in its model-index metadata [22].

**Methods it ships**: three runnable trainer scripts at this commit — SFT (`scripts/sft.py`, wraps `trl.SFTTrainer`), DPO (`scripts/dpo.py`, wraps `trl.DPOTrainer`), ORPO (`scripts/orpo.py`, wraps `trl.ORPOTrainer`) [2][7][8][9]. The package import is `from alignment import ScriptArguments, SFTConfig, DPOConfig, ORPOConfig, get_dataset, get_model, get_tokenizer` [12]; each of `SFTConfig`/`DPOConfig`/`ORPOConfig` is a bare subclass of the matching `trl` Config that adds only one field, `chat_template` — the rest of the Config surface, and all method math and training-signal semantics, is trl's and belongs on trl's own cards, not here [10]. The README's Contents section additionally names continued pretraining, reward modeling, and rejection sampling as techniques the handbook aims to cover [1], but no standalone script for reward modeling or rejection sampling exists in `scripts/` at this commit, and continued pretraining is documented as available "only ... in the `gpt-nl` example recipe" [2] — a directory (`recipes/gpt2-nl`) that returns 404 in this repository at this commit [17]. Treat CPT, reward modeling, and rejection sampling as aspirational rather than shipped.

**Scale it handles**: single GPU with LoRA or QLoRA through the `ddp.yaml` Accelerate config and `--num_processes=1`, up through full-weight fine-tuning on multiple GPUs with DeepSpeed ZeRO-3 (`zero3.yaml`), and LoRA/QLoRA on multiple GPUs with either ZeRO-3 or FSDP — the FSDP+QLoRA config is named `fsdp+qlora.yaml` in the docs, though that exact path did not resolve at this commit, a possible doc/repo drift [2][13]. Multi-node runs go through a Slurm script, `recipes/launch.slurm`, launched with `sbatch`; the repo states its defaults are "optimised for the Hugging Face Compute Cluster and may require tweaking" elsewhere [2]. The only published scale reference is the project's own reproduction runs, not a benchmark: Zephyr-7B-β full fine-tuning is documented as tested on 8×80GB GPUs [14], and the SmolLM3-3B mid-training/SFT/DPO stages on the same 8×80GB setup [16].

**Install**: the README's own path is a git checkout, not a versioned PyPI release: create a venv, `uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126` (pinned exactly, "since this is hardware-dependent"), then `uv pip install .` from the checkout, then `uv pip install "flash-attn==2.7.4.post1" --no-build-isolation`, then `huggingface-cli login` and Git LFS [1]. `python_requires` is `>=3.10.9`, licence Apache-2.0 [3][4]. At the screening commit's `setup.py` (1de1fc9…, dated 2026-04-08, itself well ahead of any tag) the floors are `torch>=2.6.0`, `transformers>=4.53.3`, `accelerate>=1.9.0`, `trl>=0.19.1`, `peft>=0.16.0`, `deepspeed>=0.17.2`, `liger-kernel>=0.6.0`, `huggingface-hub>=0.33.4,<1.0`, `protobuf<=3.20.2`; the package version string is `0.4.0.dev0`, never tagged or published [4][12]. The project's only PyPI release and only git tag are both `v0.3.0`, uploaded 2024-09-19, pinning markedly looser/older floors — `torch>=2.1.2`, `transformers>=4.39.3`, `accelerate>=0.29.2`, `trl>=0.9.6`, `peft>=0.9.0`, `deepspeed>=0.14.4` [5][6][18] — so `pip install alignment-handbook` from PyPI installs a build roughly a year and a half behind the git checkout the README's own quickstart builds from; the README itself never mentions PyPI and only documents the from-source install above [1]. No CUDA or other hardware floor is stated anywhere in the docs read; the exact `torch==2.6.0` pin against the `cu126` wheel index implies a CUDA 12.6-class stack without saying so [1].

**Maintained by**: Hugging Face (org repo `huggingface/alignment-handbook`) [3]; about 5.7k GitHub stars, not a ranking signal [3]. The commit resolved by the screening pin (1de1fc9…, 2026-04-08) is a CI-hardening PR ("pin quality.yml actions to commit SHAs", #229); the substantive work before it is a config-loading fix (#223, 2025-09-08, requiring `--config` as a flag rather than a positional argument) and the SmolLM3-3B recipe plus a "data mixer, deps, and scripts" upgrade (#221, 2025-07-24), matching the README's own 2025-07-24 News entry [1][18].

## Quick start

Environment setup, quoted in form from the README [1]:

```shell
uv venv handbook --python 3.11 && source handbook/bin/activate && uv pip install --upgrade pip
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install .
uv pip install "flash-attn==2.7.4.post1" --no-build-isolation
huggingface-cli login
```

Smallest complete SFT run, from `scripts/sft.py`'s own usage docstring [8]:

```shell
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 true \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-SFT
```

Smallest complete DPO run, from `scripts/dpo.py`'s own usage docstring [7]:

```shell
python scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns
```

A full published-recipe reproduction runs the same scripts against a config file instead of flags, e.g. Zephyr-7B-β's two stages [14]. The reference result this reproduction is measured against — MT-Bench 7.34, AlpacaEval win rate 0.906 — is published on the resulting model's own Hub page, not in this repo [22]:

```shell
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py --config recipes/zephyr-7b-beta/sft/config_full.yaml
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/dpo.py --config recipes/zephyr-7b-beta/dpo/config_full.yaml
```

## Start it

- One GPU: LoRA or QLoRA through `--config_file recipes/accelerate_configs/ddp.yaml --num_processes=1`, appending `--load_in_4bit=true` for QLoRA or `false` for LoRA [2].
- Multiple GPUs: full-weight training through `zero3.yaml` (DeepSpeed ZeRO-3); LoRA/QLoRA through `zero3.yaml` or the FSDP config family named `fsdp+qlora.yaml` in the docs (see the scale note above on that path) [2][13].
- Multi-node: `sbatch --job-name=... --nodes=<n> recipes/launch.slurm {model_name} {task} {precision} {accelerator}`, tuned for Hugging Face's own cluster [2].
- Effective batch size is `per_device_train_batch_size × num_processes × gradient_accumulation_steps`; the docs explicitly tell you to scale per-device batch size or accumulation steps together with GPU count to hold the global batch constant when you change the GPU count [2]. Zephyr-7B-β's own full-SFT config uses `per_device_train_batch_size: 16`, `gradient_accumulation_steps: 1` on 8 GPUs (effective batch 128); its DPO stage uses `per_device_train_batch_size: 8`, `gradient_accumulation_steps: 2` (also effective batch 128) [15].
- Configuration is not this repo's own surface: `SFTConfig`/`DPOConfig`/`ORPOConfig` subclass `trl.SFTConfig`/`trl.DPOConfig`/`trl.ORPOConfig` and add only `chat_template`; every precision, checkpointing, and logging default therefore comes from trl and transformers `TrainingArguments`, not from alignment-handbook itself [10].
- Dataset mixtures are this repo's own feature: `ScriptArguments.dataset_mixture` lets a YAML config combine several Hub datasets, each with a per-dataset column selection and a weight used to subsample it, then optionally splits off a test set — implemented by `get_dataset()` in `src/alignment/data.py` [10][11].
- Out-of-memory first aid: no dedicated OOM section was found in the README or `scripts/README.md`; the documented levers are the same ones used to scale down — LoRA/QLoRA (`--load_in_4bit`), lowering `per_device_train_batch_size` while raising `gradient_accumulation_steps`, and flash-attention, which the Zephyr recipe README says "also allows you to drastically increase the batch size" [1][2][14].

## Watch it

This section is mechanics only — alignment-handbook adds no signal semantics of its own; what a metric means for a method lives on that method's card (SFT/DPO/ORPO), and trl's own trainer pages carry the exact metric name lists for those trainers.

- Enable it: logging goes through trl/transformers' `report_to`. `scripts/README.md` states plainly that "by default, all training metrics are logged with TensorBoard," and appending `--report_to=wandb` switches to Weights & Biases if you are logged in [2].
- Metric names: alignment-handbook defines none of its own. Each script calls `trainer.log_metrics(...)` / `trainer.save_metrics(...)` on the underlying trl trainer for both `train` and `eval` splits, so the metric names are exactly what `SFTTrainer`/`DPOTrainer`/`ORPOTrainer` log in trl — consult trl's own trainer pages for that list, not this card [7][8].
- Sample-level generation logging: not found in `scripts/dpo.py`, `scripts/sft.py`, `scripts/orpo.py`, or `scripts/README.md` — alignment-handbook does not add any generation-sampling logging of its own; if the wrapped trl trainer supports it, that support is trl's, not this repo's [2][7][8].
- Evaluation during training: every script passes `eval_dataset` and calls `trainer.evaluate()` only when `training_args.eval_strategy != "no"`, logging and saving eval metrics the same way as train metrics [7]. The Zephyr DPO recipe sets `eval_strategy: steps`, `eval_steps: 100` as one concrete, published example [15].
- Health limits / stopping rules: none published. Searched the README [1], `scripts/README.md` [2], and `src/alignment/configs.py` [10] — none define a stopping-rule, threshold, or patience field beyond the plain trl Config fields those classes subclass.

## Save it

- Every script follows the same sequence: after `trainer.train(...)`, it calls `trainer.log_metrics`/`trainer.save_metrics` for `train`, then `trainer.save_state()`; if `eval_strategy != "no"` it also calls `trainer.evaluate()` and logs/saves those metrics; finally it calls `trainer.save_model(training_args.output_dir)`, and `trainer.push_to_hub(...)` if `training_args.push_to_hub` is set [7]. This is exactly trl's save contract via `TrainingArguments` (`save_strategy`, `save_steps`, `save_total_limit`, `save_only_model`) — alignment-handbook adds or changes nothing about what is saved [10].
- A concrete retention example: Zephyr's DPO config sets `save_strategy: "steps"`, `save_steps: 100`, `save_total_limit: 1`, keeping only the single newest checkpoint on disk [15].
- Resume: all three scripts check `get_last_checkpoint(training_args.output_dir)` at start and resume from it automatically unless `training_args.resume_from_checkpoint` is already set to a specific path [7].
- LoRA/QLoRA: `model_args` feeds `get_peft_config(model_args)` directly into the trl trainer's `peft_config` argument [7][8]; alignment-handbook adds no adapter-saving or merge logic of its own, so whether the result is an adapter-only directory or a merged full model — and what an adapter directory is NOT — is trl's and PEFT's contract, not this repo's.
- Loader handoff: because this repo does not touch the save path, whether an evaluator can load a saved checkpoint directly is entirely the wrapped trl trainer's (and, for adapters, PEFT's) contract — check those cards, not this one, before assuming a checkpoint loads.

## Find it in the docs

There is no hosted docs site for this project: `https://huggingface.co/docs/alignment-handbook/index` returned 404, checked 2026-08-10 [19]. Everything lives in-repo as Markdown.

- Root `README.md` is the entry point: installation, the Contents list of intended techniques, links to the Hub model/dataset collection and the Zephyr technical report, and the project-structure map [1].
- `scripts/README.md` is the CLI/task reference: task names (`cpt`, `sft`, `dpo`, `orpo`), the full/LoRA/QLoRA/FSDP launch-command templates, the Slurm launch form, W&B logging, dataset-formatting conventions per task, and pointers to MT-Bench and AlpacaEval for evaluating the resulting chat models [2].
- Each `recipes/<model_name>/README.md` documents one specific reproduction, e.g. `recipes/zephyr-7b-beta/README.md` [14] and `recipes/smollm3/README.md` [16]; the recipe directories (`recipes/accelerate_configs/`, `recipes/launch.slurm`) are the runnable references beyond the prose [2][13].
- Known-good smoke datasets recur throughout: `HuggingFaceH4/ultrachat_200k` for SFT and `HuggingFaceH4/ultrafeedback_binarized` for DPO/ORPO in the recipe configs [1][2], and `trl-lib/Capybara` / `trl-lib/ultrafeedback_binarized` in the scripts' own usage docstrings [7][8].
- Community layer: the README links to a Hugging Face Hub collection of the Zephyr models and datasets, and to the Zephyr technical report on arXiv, as the canonical write-up behind the flagship recipe [1][21]; this repo does not curate a separate community-tutorials page the way trl does.
- No repo-specific MCP endpoint was found; Hugging Face's general docs-search MCP (`huggingface.co/mcp`) indexes Hub content broadly but was not verified against this repository specifically here.
- Maintainer-confirmed trap: loading a saved checkpoint with a plain `AutoModelForCausalLM.from_pretrained(path)` silently returns `float32` weights even though training ran in `bfloat16`, unless `torch_dtype` is passed explicitly at load time — confirmed by Hugging Face team member alvarobartt in issue #174, 2024-07-23 [20].
- Honest boundary: at this commit the repository ships runnable scripts and configs only for SFT, DPO, and ORPO; reward modeling and rejection sampling are named as goals in the README's Contents section with no corresponding script in `scripts/` [1][2], and continued pretraining is documented as available "only ... in the `gpt-nl` example recipe" [2] whose directory, `recipes/gpt2-nl`, returns 404 in this repository at this commit [17] — so CPT is effectively undocumented in practice despite being named in the README.

## Sources

All GitHub file citations are read at commit `1de1fc996972aa76b7d40c64c07b66dec8b6976a` unless a tag is named; all fetches were performed 2026-08-10.

[1] alignment-handbook README.md. https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/README.md. Fetched 2026-08-10.

[2] alignment-handbook scripts/README.md. https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/scripts/README.md. Fetched 2026-08-10.

[3] alignment-handbook GitHub repository (metadata: license, star count, pushed/created dates, archived status). https://github.com/huggingface/alignment-handbook. Fetched 2026-08-10 via the GitHub API.

[4] alignment-handbook setup.py at the screening commit (dependency floors, `python_requires`, version string `0.4.0.dev0`). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/setup.py. Fetched 2026-08-10.

[5] alignment-handbook setup.py at tag v0.3.0 (commit 6da26f58139211c2c3b94a67233d2462ee68196b) — older dependency floors. https://github.com/huggingface/alignment-handbook/blob/v0.3.0/setup.py. Fetched 2026-08-10.

[6] alignment-handbook on PyPI (latest published version 0.3.0, uploaded 2024-09-19). https://pypi.org/project/alignment-handbook/. Fetched 2026-08-10.

[7] alignment-handbook scripts/dpo.py at the screening commit (CLI usage docstring, DPOTrainer wrapping, checkpoint-resume, save/push logic). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/scripts/dpo.py. Fetched 2026-08-10.

[8] alignment-handbook scripts/sft.py at the screening commit (CLI usage docstring, SFTTrainer wrapping). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/scripts/sft.py. Fetched 2026-08-10.

[9] alignment-handbook scripts/orpo.py at the screening commit (CLI usage docstring, ORPOTrainer wrapping). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/scripts/orpo.py. Fetched 2026-08-10.

[10] alignment-handbook src/alignment/configs.py at the screening commit (ScriptArguments/dataset_mixture; SFTConfig/DPOConfig/ORPOConfig subclassing trl configs with only `chat_template` added). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/src/alignment/configs.py. Fetched 2026-08-10.

[11] alignment-handbook src/alignment/data.py at the screening commit (`get_dataset` mixture logic). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/src/alignment/data.py. Fetched 2026-08-10.

[12] alignment-handbook src/alignment/__init__.py at the screening commit (`__version__ = "0.4.0.dev0"`, exported symbols — no CPT config exported). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/src/alignment/__init__.py. Fetched 2026-08-10.

[13] alignment-handbook recipes/accelerate_configs/ (zero3.yaml, ddp.yaml, fsdp.yaml) at the screening commit — Accelerate launcher templates. https://github.com/huggingface/alignment-handbook/tree/1de1fc996972aa76b7d40c64c07b66dec8b6976a/recipes/accelerate_configs. Fetched 2026-08-10.

[14] alignment-handbook recipes/zephyr-7b-beta/README.md at the screening commit (full/QLoRA launch commands, 8×80GB GPU note, flash-attention batch-size note). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/recipes/zephyr-7b-beta/README.md. Fetched 2026-08-10.

[15] alignment-handbook recipes/zephyr-7b-beta/{sft,dpo}/config_full.yaml at the screening commit (concrete batch size, accumulation, and save-retention numbers). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/recipes/zephyr-7b-beta/sft/config_full.yaml and .../dpo/config_full.yaml. Fetched 2026-08-10.

[16] alignment-handbook recipes/smollm3/README.md at the screening commit (mid-training/SFT/DPO stages, 8×80GB GPU note). https://github.com/huggingface/alignment-handbook/blob/1de1fc996972aa76b7d40c64c07b66dec8b6976a/recipes/smollm3/README.md. Fetched 2026-08-10.

[17] alignment-handbook recipes/gpt2-nl/README.md at the screening commit — returns 404, confirming the recipe named in [2] does not exist in this repository at this commit. https://raw.githubusercontent.com/huggingface/alignment-handbook/1de1fc996972aa76b7d40c64c07b66dec8b6976a/recipes/gpt2-nl/README.md. Fetched 2026-08-10.

[18] GitHub API data for alignment-handbook: tags (v0.3.0 → commit 6da26f58139211c2c3b94a67233d2462ee68196b) and recent commit history on the branch resolved by the screening commit (PR #229, #223, #221 with dates). https://api.github.com/repos/huggingface/alignment-handbook/tags and https://api.github.com/repos/huggingface/alignment-handbook/commits. Fetched 2026-08-10.

[19] Attempted fetch of a hosted docs page for this project, returning 404 — evidence that no `huggingface.co/docs/alignment-handbook` site exists. https://huggingface.co/docs/alignment-handbook/index. Fetched 2026-08-10.

[20] alignment-handbook GitHub issue #174, comment by Hugging Face team member alvarobartt (torch_dtype fp32-on-reload trap). https://github.com/huggingface/alignment-handbook/issues/174. Fetched 2026-08-10.

[21] Zephyr technical report on arXiv, linked from the README as the write-up behind the flagship recipe; not restated here since method math belongs on method cards. https://arxiv.org/abs/2310.16944. Fetched 2026-08-10.

[22] `HuggingFaceH4/zephyr-7b-beta` Hub model page (the model the Zephyr recipe reproduces), model-index metadata: MT-Bench score 7.34, AlpacaEval win rate 0.906. https://huggingface.co/HuggingFaceH4/zephyr-7b-beta. Fetched 2026-08-10.

Ecosystem ‌tools referenced only in passing — Accelerate, DeepSpeed, FSDP, PEFT, Weights & Biases, TensorBoard, MT-Bench, AlpacaEval — are reached through [1] and [2]'s own links and are deliberately not enumerated as separate references. trl itself, which every script in this repo wraps, is deliberately not cited here beyond the wrapping relationship shown in [7]–[9]: its own methods, Config surface, and metric names belong on trl's own card.
