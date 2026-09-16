# LMFlow

A finetuning-and-alignment toolbox from Hong Kong University of Science and Technology's OptimalScale group: install from source, launch through Accelerate, and reach LISA, LoRA/QLoRA, RAFT, and two generations of DPO through one CLI-driven pipeline layer.

**LMFlow** is "an extensible, convenient, and efficient toolbox for finetuning large machine learning models, designed to be user-friendly, speedy and reliable, and accessible to the entire community" [1]. It is built and maintained by OptimalScale, and its accompanying paper credits Shizhe Diao, Rui Pan, Hanze Dong, Ka Shun Shum, Jipeng Zhang, Wei Xiong, and Tong Zhang [2]. Its API is pipeline-shaped: a `Pipeline` class (`Finetuner`, `Evaluator`, `Inferencer`, `RewardModelTuner`, an aligner) is fetched by name through `AutoPipeline.get_pipeline(...)` and driven by argument dataclasses (`ModelArguments`, `DatasetArguments`, a `TrainingArguments`-derived pipeline-arguments class) that the shipped shell scripts under `scripts/` populate and pass to `examples/*.py` entry points [3][4]. It lives at https://github.com/OptimalScale/LMFlow [5].

**When to pick it**: a single, script-driven toolbox for LLM finetuning and alignment that bundles memory-saving finetuning (LISA, LoRA, QLoRA) with a curated set of alignment methods (RAFT, DPO, iterative DPO) behind one CLI, when you are comfortable installing from source and picking exact extras yourself; the method menu is narrower and more RLHF-specific than trl's 16 trainers [6] — pick LMFlow when LISA or RAFT specifically is what you want, since neither ships in trl.

**Methods it ships**: Full-parameter, LISA (Layerwise Importance Sampled AdamW, a memory-saving alternative to LoRA), LoRA, and QLoRA finetuning via the `Finetuner` pipeline [1]; Reward Modeling via `RewardModelTuner`, a subclass that reuses `Finetuner`'s dataclass and Trainer plumbing with no additional fields (`class RewardModelTunerArguments(FinetunerArguments): pass`) [7]; RAFT (Reward rAnked FineTuning) via `raft_aligner.py`, conditionally registered only `if not is_package_version_at_least("transformers", "4.35.0")` - on any modern transformers install `AutoPipeline.get_pipeline("raft_aligner")` raises `NotImplementedError(f'Please install the necessary dependencies to use pipeline "{pipeline_name}"')`, naming the pipeline itself rather than the specific missing package, so RAFT is effectively unavailable outside an old-transformers environment [8]; DPO via `dpo_aligner.py`, a thin wrapper around trl's own `DPOTrainer` (`from trl import DPOTrainer`) [9]; DPOv2 via `dpov2_aligner.py`, which instead drives a custom in-repo `DPOv2Trainer` (not trl's), taking `beta`, `loss_type`, `max_prompt_length`, `max_length`, `mask_prompt`, and `len_penalty` [10]; and iterative DPO via `iterative_dpo_aligner.py`, registered only when trl is present together with vLLM or SGLang [8]. vLLM- and SGLang-backed inference pipelines (`vllm_inferencer`, `sglang_inferencer`) are likewise gated behind `is_vllm_available()` / `is_sglang_available()` [8]. No GRPO or RLOO trainer is present in the pipeline directory (`src/lmflow/pipeline/`, which lists only the pipelines named above) [3].

**Scale it handles**: single GPU up to single-node multi-GPU, launched through Hugging Face Accelerate with repo-provided config templates at `configs/accelerate_fsdp_config.yaml` (FSDP, `FULL_SHARD`, bf16, 8 processes) and `configs/accelerate_dsz3_config.yaml` (DeepSpeed ZeRO-3, bf16, 8 processes) [11][12]; both templates set `num_machines: 1`, and neither the README nor any docs page fetched for this card (home, finetuning, datasets, checkpoints, reward-modeling, examples index) mentions "multi-node" or `num_machines` beyond 1, so multi-node is undocumented mechanism at best, not a published path [1][13][14].

**Install**: no working PyPI release - `pip install lmflow` pulls the placeholder version `0.0.1.dev0` uploaded 2023-04-02, years stale [15]; the real install is git clone plus editable install: `git clone -b v1.0.0 https://github.com/OptimalScale/LMFlow.git && cd LMFlow && conda create -n lmflow python=3.9 -y && conda activate lmflow && conda install mpi4py && pip install -e .` [1]. Latest tagged release is v1.0.0 (2025-07-11), resolving to commit `a6b97aa3a2efbfa16c71c9628ad2506839bf4bf9` [16][17]; at that tag, `requirements.txt` pins `torch>=2.0.1`, `transformers>=4.31.0`, `accelerate>=0.27.2`, `peft>=0.10.0`, `datasets==2.14.6`, `evaluate==0.4.0`, `bitsandbytes>=0.40.0`, `cpm_kernels==1.0.11` [18], and `setup.py`'s extras pin `vllm>=0.4.3` and `trl==0.8.0` (an exact pin) [19]. `setup.py`, `requires_python` is `>=3.9`; licence is Apache-2.0 plus a commercial-use notice requiring a linked form for for-profit use [1][20]. **Pinned-vs-live note**: the repository's current main branch (read at commit `ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4`, pushed 2026-08-07, which is ahead of the v1.0.0 release) has since moved these same extras to `vllm>=0.8.0` and `trl>=0.11,<0.12` plus an added `rich` dependency (a code comment there notes trl's `DPOTrainer` lazy-imports `rich` without declaring it), and added a new `sglang` extra (`sglang`, plus `pybase64`, which a code comment says `sglang.utils` needs but does not declare) [21] - a `pip install -e .` from a `v1.0.0` checkout will NOT pick up these newer pins. No CUDA or hardware-driver minimum is stated on the install page; the README instead gives an estimated-total-memory table by parameter count and method, with no GPU-count breakdown - e.g. for a 7B model: 120GB for full bf16/fp16 finetuning, 16GB for LoRA, 10GB for QLoRA at `quant_bit=8`, and 6GB for QLoRA at `quant_bit=4` [1].

**Maintained by**: OptimalScale (HKUST). The latest tagged release, v1.0.0, was published 2025-07-11 [16]; the repository's most recent push was 2026-08-07 [5], over a year later, showing main-branch commits have continued well past that last tagged release (the setup.py drift documented under Install is one such post-release change) [21].

## Quick start

Full finetuning, from the README's Full Finetuning quickstart (a GPT-2 base-model example), quoted verbatim as a shell command [1]:

```sh
cd data && ./download.sh alpaca && cd -

bash ./scripts/run_finetune.sh \
  --model_name_or_path gpt2 \
  --dataset_path data/alpaca/train_conversation \
  --output_model_path output_models/finetuned_gpt2
```

LoRA finetuning, from `scripts/run_finetune_with_lora.sh`, which sets defaults and calls Accelerate directly [22]:

```bash
model_name_or_path=meta-llama/Llama-3.2-3B-Instruct
dataset_path=data/alpaca/train_conversation
output_dir=output_models/finetune_lora
lora_r=8
lora_alpha=32
lora_dropout=0.1

accelerate launch --config_file configs/accelerate_fsdp_config.yaml \
    examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --use_lora 1 \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout}
```

Reward modeling, from `scripts/run_reward_modeling.sh`, which also launches through Accelerate [23]:

```bash
model_name_or_path=google/gemma-2b-it
train_dataset_path=data/ultrafeedback-binarized-preferences-cleaned/train
eval_dataset_path=data/ultrafeedback-binarized-preferences-cleaned/train
output_dir=output_models/reward_modeling

accelerate launch --config_file configs/accelerate_fsdp_config.yaml \
    examples/reward_modeling.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${train_dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --eval_dataset_path ${eval_dataset_path} \
    --arch_type "text_regression"
```

## Start it

- One process, one GPU: run any of the scripts above unmodified; Accelerate detects a single device from its config file.
- Multiple GPUs, one node: the repo's own `configs/accelerate_fsdp_config.yaml` (FSDP, `fsdp_sharding_strategy: FULL_SHARD`, `fsdp_state_dict_type: FULL_STATE_DICT`, `fsdp_cpu_ram_efficient_loading: true`, `mixed_precision: bf16`, `num_processes: 8`) and `configs/accelerate_dsz3_config.yaml` (DeepSpeed ZeRO-3, `zero3_init_flag: true`, `zero3_save_16bit_model: true`, `mixed_precision: bf16`, `num_processes: 8`) are passed with `accelerate launch --config_file <yaml> examples/finetune.py <args>`; both ship with an inline comment that `distributed_type` should be set to `NO` for a single-GPU run [11][12]. No multi-node config template is shipped, and no page fetched for this card documents one [1][13][14].
- Effective batch size follows Accelerate/transformers convention: per-device batch x number of processes x `gradient_accumulation_steps`; the quickstart script sets `per_device_train_batch_size=1` and `gradient_accumulation_steps=1` as its baseline [24].
- Configuration is the pipeline's arguments dataclass; `FinetunerArguments` and `RaftAlignerArguments` both subclass transformers' `TrainingArguments` directly, so undocumented fields fall through to transformers' own defaults [7][25]. LMFlow's own added fields default as follows: `use_lora=False`, `use_qlora=False`, `use_dora=False`, `lora_r=8`, `lora_alpha=32`, `lora_dropout=0.1`, `save_aggregated_lora=False`, `use_flash_attention=False`, `torch_dtype=None` (no forced override; derived from the checkpoint unless set) [26]. LMFlow does change one default away from transformers' own: `mixed_precision` defaults to `"bf16"`, but only on the `EvaluatorArguments`/`InferencerArguments` side (inference and evaluation), not on `FinetunerArguments`, which carries no `mixed_precision` field of its own and instead inherits `TrainingArguments`' native fp32-unless-set behavior - so a training run's precision is whatever `--bf16`/`--fp16` you pass, while eval/inference silently assumes bf16-capable hardware unless overridden [27].
- Out-of-memory first aid: the README's hardware-requirement table is itself the OOM guide - move down the method column (Full to LoRA to QLoRA `quant_bit=8` to QLoRA `quant_bit=4`) for a given parameter count, e.g. a 7B model's estimated total memory drops from 120GB (full bf16/fp16) to 16GB (LoRA) to 10GB (QLoRA 8-bit) to 6GB (QLoRA 4-bit) [1]. `use_flash_attention` (default `False`) is a plain toggle on `ModelArguments` [26]; `FinetunerArguments` itself declares no LMFlow-specific `gradient_checkpointing` field and instead inherits transformers' own `gradient_checkpointing` (default `False`) via `TrainingArguments` - the only place LMFlow overrides that default is the separate, older `DPOAlignerArguments` dataclass, which sets `gradient_checkpointing=True` [28].

## Watch it

This section is mechanics only - what a metric or accuracy number MEANS for a method lives on that method's methodology card.

- **Enable it**: `FinetunerArguments` inherits `TrainingArguments` with no `report_to` override, so logging follows transformers' own `report_to` field and backend set (e.g. `wandb`, `tensorboard`) with no LMFlow-specific default [7]. The separate, older `DPOAlignerArguments` dataclass (used by `dpo_aligner.py`'s trl-`DPOTrainer` wrapper) sets its own default `report_to="wandb"`, with `logging_steps=10`, `save_steps=100`, `eval_steps=100`, `output_dir="./results"` [29]. The README also flags that finetuning scripts log to Weights & Biases by default and gives the flag to disable it [1].
- **Metric names**: LMFlow's `RewardModelTuner` uses a custom `RewardTrainer(transformers.Trainer)` (not trl's `RewardTrainer` class of the same name) whose `compute_metrics` reports a single `"accuracy"` field - the fraction of pairs where the chosen sequence scores higher than the rejected one [30]. Loss is `-nn.functional.logsigmoid(rewards_j - rewards_k).mean()`, matching the InstructGPT-style pairwise loss the reward-modeling docs page also states in prose [31][30]. Trainers built on trl (`dpo_aligner.py`) or on plain transformers `Trainer` (`Finetuner`, `RewardModelTuner`) otherwise surface whatever metric names those upstream libraries log; no LMFlow-specific metrics page was found beyond the reward-modeling accuracy field.
- **Sample-level logging of generations and evaluation-during-training fields**: not documented on any docs page fetched for this card (home, finetuning, datasets, checkpoints, reward-modeling, examples index) beyond the standard transformers `eval_dataset`/`eval_strategy` surface that `FinetunerArguments` inherits via `TrainingArguments` [1][13][14][7].
- **Stopping**: no LMFlow-specific early-stopping or reward-plateau rule is published in the README or any docs page read for this card; `FinetunerArguments` and `RaftAlignerArguments` inherit whatever transformers' own `TrainingArguments`/callback surface provides, and LMFlow adds nothing beyond that [7][25].
- **Deciding number** (reward modeling, from the docs' own eval-accuracy table on the Anthropic `Dahoas/full-hh-rlhf` set, 112K train / 12.5K test pairs [31]): a LLaMA-13B reward model reaches 84.55% pairwise eval accuracy after 2 epochs of SFT before reward-model training, 81.80% after 1 epoch of SFT, and only 71.64% with no SFT at all - a 13-point swing showing that the SFT step, not just the reward-model step, drives most of the accuracy; LLaMA-7B shows a similar gap: 79.52% pairwise accuracy in one row against 71.64% in a row the table labels "RM from LLaMA without SFT," though the table's own remark for the 79.52% row is blank, so it does not itself state that this run used SFT [31].

## Save it

- On a non-LoRA run, `Finetuner` calls `trainer.save_model()` at the end of training, which also saves the tokenizer for easy re-upload [24].
- On a LoRA run, saving depends on `save_aggregated_lora` (default `False`): if it is left `False`, `model.save(output_dir, save_aggregated_lora)` writes only the adapter, via the backend's `get_backend_model().save_pretrained(dir)` path - an adapter directory, NOT a full model; if `save_aggregated_lora=True`, the pipeline first calls `model.merge_lora_weights()` then saves the merged full model with `self.backend_model_full.to(dtype=save_dtype).save_pretrained(dir)`, logging the dtype used [24][32].
- Loading an adapter back requires pairing it with its base model: `HFModelMixin` loads `PeftModel.from_pretrained(self.backend_model, model_args.lora_model_path, ...)` only when `lora_model_path` is set separately from `model_name_or_path` - an adapter directory alone cannot be loaded with a plain `from_pretrained()` call [33]. The docs' checkpoints page shows this pairing in its own evaluate example: `--model_name_or_path ${llama-hf-path}/... --lora_model_path output_models/${llama-model-diff-path}` [34].
- After `save_model()`, `Finetuner` calls `trainer.log_metrics`, `trainer.save_metrics`, and `trainer.save_state`, then either `trainer.push_to_hub(**kwargs)` if `training_args.push_to_hub` is set, or `trainer.create_model_card(**kwargs)` otherwise [24].
- `DPOv2Aligner` saves differently from `Finetuner`: after training it calls `dpo_trainer.save_model(output_dir)` and then separately `dpo_trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint"))` [10].
- Resume: `Finetuner` resumes from `training_args.resume_from_checkpoint` if explicitly set, otherwise from the last checkpoint found in `output_dir`, passed as `trainer.train(resume_from_checkpoint=checkpoint)` [24].
- Whether an evaluator can load a saved checkpoint directly is the loader's contract: a non-LoRA or aggregated-LoRA save is a standard `from_pretrained`-loadable directory, per the `save_model()`/merged-save paths above [24][32]; a bare (non-aggregated) LoRA save is not, and needs the base-model pairing shown above [33][34].

## Find it in the docs

The docs are the live source; this card teaches the lookup, not the content.

- Address pattern: `https://optimalscale.github.io/LMFlow/en/latest/<section>/<page>.html` - confirmed by fetching `docs_home.html` (pagename `index`), `docs_finetuning.html` (pagename `examples/finetuning`), `docs_reward_modeling.html` (pagename `examples/reward_modeling`), `docs_checkpoints.html` (pagename `examples/checkpoints`), `docs_datasets.html` (pagename `examples/DATASETS`, capitalized - the README's own link to it uses `.../examples/DATASETS.html#conversation-template`) [13][31][34][35]. No canonical `<link>` tag was found on any of these five pages to confirm a version-pinned URL form; none was located [13][14][31][34][35].
- **Drift warning**: the docs home page's Installation section is stale relative to the current README - it still shows a plain `git clone` with no `-b <tag>` and a bare `pip install -e .` with no extras selection, and frames the project as "Task Tuning"/"Instruction Tuning" rather than the README's current LISA/LoRA/QLoRA/RAFT/DPO framing [13]. The finetuning, datasets, reward-modeling, and checkpoints pages read for this card do NOT show this staleness - the finetuning page's Full/LISA/LoRA quickstart commands match the README's current commands, and the reward-modeling and datasets pages carry current, non-duplicated content [14][31][35].
- Question-to-slug map: dataset JSON shapes -> `examples/DATASETS.html` (general `{"type": ..., "instances": [...]}` schema, plus a "conversation" ShareGPT-style sub-format with `conversation_id`, `system`, `tools`, `messages`) [35]; finetuning quickstart commands (Full/LISA/LoRA) -> `examples/finetuning.html` [14]; reward-model procedure and eval numbers -> `examples/reward_modeling.html` [31]; loading a raw LLaMA checkpoint and pairing it with a LoRA adapter -> `examples/checkpoints.html` [34]; the full page tree (Dataset, Checkpoints, Finetuning, Reward Modeling, RAFT, Inference, Evaluation/LMFlow Benchmark) is listed at `examples/index.html` [36].
- Runnable references beyond the docs: `cd data && ./download.sh all` (or `./download.sh <name>`, e.g. `alpaca`) fetches the example datasets the quickstart scripts expect, including `alpaca` and the reward-modeling script's `ultrafeedback-binarized-preferences-cleaned` [1][23]; the `scripts/` directory holds one shell script per pipeline (`run_finetune.sh`, `run_finetune_with_lora.sh`, `run_reward_modeling.sh`, `run_dpo_align.sh`, `run_iterative_dpo.sh`) as the runnable smoke tests [3].
- Community layer: no curated community-tutorials page was found among the docs pages fetched for this card (home, examples index, finetuning, datasets, checkpoints, reward-modeling); the top nav on the examples index lists only Blogs, Examples, API Reference, and About, none of which were fetched for this card [36].
- No official MCP endpoint for the docs was found or referenced on any page fetched for this card.
- **Honest boundary / trap**: RAFT is registered in `AutoPipeline` only when the installed transformers version is below 4.35.0 (`if not is_package_version_at_least("transformers", "4.35.0")`); on any current transformers install, requesting the `raft_aligner` pipeline raises `NotImplementedError` - this is read directly from `auto_pipeline.py` at commit `ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4` (ahead of the v1.0.0 release) rather than from a maintainer issue reply, so it is stated here as a code-level trap, not an issue-tracker one [8]. Multi-node training has no published guide or benchmark anywhere fetched for this card [1][13][14].

## Sources

All source-file citations are read at commit `a6b97aa3a2efbfa16c71c9628ad2506839bf4bf9` (the `v1.0.0` release tag) unless marked as read at `ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4` (main branch, pushed 2026-08-07, ahead of the release). Docs pages are unpinned, live pages read on 2026-08-10.

[1] LMFlow README (installation, hardware-requirement table, finetuning quickstart commands, WandB default, Supported Features, License, Citation). https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/README.md. Fetched 2026-08-10, read at commit a6b97aa.

[2] LMFlow paper (author list). https://arxiv.org/abs/2306.12420. Cited via README [1].

[3] `src/lmflow/pipeline/` directory listing (pipeline files). https://api.github.com/repos/OptimalScale/LMFlow/contents/src/lmflow/pipeline?ref=v1.0.0. Fetched 2026-08-10, read at commit a6b97aa.

[4] LMFlow `finetuner.py` pipeline-argument wiring (illustrative of the `AutoPipeline`/arguments-dataclass pattern; see also [24]). https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/src/lmflow/pipeline/finetuner.py. Fetched 2026-08-10, read at commit a6b97aa.

[5] LMFlow GitHub repository (stars, open issues, latest push date). https://api.github.com/repos/OptimalScale/LMFlow. Fetched 2026-08-10.

[6] trl documentation index (method count, cited only for the cross-framework method-count contrast). https://huggingface.co/docs/trl/main/en/index. Fetched 2026-08-10 (per prior trl card research).

[7] LMFlow `src/lmflow/args.py`, `FinetunerArguments`/`RewardModelTunerArguments` classes. https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/src/lmflow/args.py. Fetched 2026-08-10, read at commit a6b97aa.

[8] LMFlow `src/lmflow/pipeline/auto_pipeline.py` (conditional pipeline registration, RAFT version gate). https://raw.githubusercontent.com/OptimalScale/LMFlow/ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4/src/lmflow/pipeline/auto_pipeline.py. Fetched 2026-08-10, read at commit ea19450 (ahead of the v1.0.0 release).

[9] LMFlow `src/lmflow/pipeline/dpo_aligner.py` (trl `DPOTrainer` import). https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/src/lmflow/pipeline/dpo_aligner.py. Fetched 2026-08-10, read at commit a6b97aa.

[10] LMFlow `src/lmflow/pipeline/dpov2_aligner.py` (custom `DPOv2Trainer`, save calls). https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/src/lmflow/pipeline/dpov2_aligner.py. Fetched 2026-08-10, read at commit a6b97aa.

[11] LMFlow `configs/accelerate_fsdp_config.yaml`. https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/configs/accelerate_fsdp_config.yaml. Fetched 2026-08-10, read at commit a6b97aa.

[12] LMFlow `configs/accelerate_dsz3_config.yaml`. https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/configs/accelerate_dsz3_config.yaml. Fetched 2026-08-10, read at commit a6b97aa.

[13] LMFlow docs home page (stale install instructions, "Task Tuning" framing). https://optimalscale.github.io/LMFlow/en/latest/index.html. Fetched 2026-08-10.

[14] LMFlow docs finetuning page. https://optimalscale.github.io/LMFlow/en/latest/examples/finetuning.html. Fetched 2026-08-10.

[15] `lmflow` on PyPI (stale placeholder release). https://pypi.org/pypi/lmflow/json. Fetched 2026-08-10.

[16] LMFlow releases API (v1.0.0 release date). https://api.github.com/repos/OptimalScale/LMFlow/releases. Fetched 2026-08-10.

[17] LMFlow tags API (v1.0.0 tag -> commit sha). https://api.github.com/repos/OptimalScale/LMFlow/tags. Fetched 2026-08-10.

[18] LMFlow `requirements.txt` at the v1.0.0 tag. https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/requirements.txt. Fetched 2026-08-10, read at commit a6b97aa.

[19] LMFlow `setup.py` at the v1.0.0 tag (release-pinned extras: `vllm>=0.4.3`, `trl==0.8.0`). https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/setup.py. Fetched 2026-08-10, read at commit a6b97aa.

[20] LMFlow README license section (Apache-2.0, commercial-use form). Same source as [1].

[21] LMFlow `setup.py` on main branch (moved extras: `vllm>=0.8.0`, `trl>=0.11,<0.12` plus `rich`, new `sglang` extra). https://raw.githubusercontent.com/OptimalScale/LMFlow/ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4/setup.py. Fetched 2026-08-10, read at commit ea19450 (ahead of the v1.0.0 release).

[22] LMFlow `scripts/run_finetune_with_lora.sh`. https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/scripts/run_finetune_with_lora.sh. Fetched 2026-08-10, read at commit a6b97aa.

[23] LMFlow `scripts/run_reward_modeling.sh`. https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/scripts/run_reward_modeling.sh. Fetched 2026-08-10, read at commit a6b97aa.

[24] LMFlow `src/lmflow/pipeline/finetuner.py` (trainer_kwargs, save/resume/push logic). https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/src/lmflow/pipeline/finetuner.py. Fetched 2026-08-10, read at commit a6b97aa.

[25] LMFlow `src/lmflow/args.py`, `RaftAlignerArguments` class. Same source as [7].

[26] LMFlow `src/lmflow/args.py`, `ModelArguments` class (LoRA/QLoRA/flash-attention/torch_dtype defaults). Same source as [7].

[27] LMFlow `src/lmflow/args.py`, `EvaluatorArguments`/`InferencerArguments` classes (`mixed_precision="bf16"` default). Same source as [7].

[28] LMFlow `src/lmflow/args.py`, `gradient_checkpointing` field, default `True`, declared on `DPOAlignerArguments` (not on `FinetunerArguments`, which has no such field of its own). Same source as [7].

[29] LMFlow `src/lmflow/args.py`, `DPOAlignerArguments` class (`report_to="wandb"` default, logging/eval/save-step defaults). Same source as [7].

[30] LMFlow `src/lmflow/pipeline/utils/rm_trainer.py` (custom `RewardTrainer`, pairwise loss, `compute_metrics`). https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/src/lmflow/pipeline/utils/rm_trainer.py. Fetched 2026-08-10, read at commit a6b97aa.

[31] LMFlow docs reward-modeling page (InstructGPT-style procedure, dataset, eval-accuracy table). https://optimalscale.github.io/LMFlow/en/latest/examples/reward_modeling.html. Fetched 2026-08-10.

[32] LMFlow `src/lmflow/models/hf_decoder_model.py`, `save()`/`merge_lora_weights()` methods. https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/src/lmflow/models/hf_decoder_model.py. Fetched 2026-08-10, read at commit a6b97aa.

[33] LMFlow `src/lmflow/models/hf_model_mixin.py` (`PeftModel.from_pretrained` with separate `lora_model_path`). https://raw.githubusercontent.com/OptimalScale/LMFlow/v1.0.0/src/lmflow/models/hf_model_mixin.py. Fetched 2026-08-10, read at commit a6b97aa.

[34] LMFlow docs checkpoints page (base-model + `lora_model_path` evaluate example). https://optimalscale.github.io/LMFlow/en/latest/examples/checkpoints.html. Fetched 2026-08-10.

[35] LMFlow docs datasets page (`examples/DATASETS.html`, dataset JSON schema, conversation format). https://optimalscale.github.io/LMFlow/en/latest/examples/DATASETS.html. Fetched 2026-08-10.

[36] LMFlow docs examples-index page (full docs page tree, top nav). https://optimalscale.github.io/LMFlow/en/latest/examples/index.html. Fetched 2026-08-10.
