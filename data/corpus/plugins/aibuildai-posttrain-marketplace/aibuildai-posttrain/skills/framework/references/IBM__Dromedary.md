# Dromedary

The reference code release for the SELF-ALIGN paper: a fixed, non-reusable pipeline of scripts (prompt cleaning, self-align generation, QLoRA SFT) that reproduces one specific model, not a general post-training library.

Dromedary describes itself as "an open-source self-aligned language model trained with minimal human supervision" [1]. It is built and maintained by the paper's authors, with the corresponding NeurIPS 2023 (Spotlight) paper "Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision" [2]. There is no trainer-class or CLI API: the repo is a sequence of standalone Python scripts and shell wrappers, organized by pipeline stage under `training/step1_prompt_cleaning`, `training/step2_principle_driven_self_alignment`, and `training/step3_principle_engraving`, each invoked directly with `python` or `torchrun` [3]. It lives at https://github.com/IBM/Dromedary, published under the `IBM` GitHub organization [4].

**When to pick it**: only to reproduce or study the SELF-ALIGN / Dromedary-2 pipeline itself - it is a paper artifact, not a library meant for other training runs. The `main` branch holds the current two-stage Dromedary-2 pipeline (prompt curation + QLoRA SFT, called "Principle Engraving") [1][5]; the original four-stage v1 pipeline (adding Topic-Guided Red-Teaming Self-Instruct and a Verbose Cloning stage) lives only on the separate `dromedary_v1` branch [1]. The RLAIF stage of Dromedary-2 (SALMON) is not in this repo at all - it is a separate repository, `IBM/SALMON` [1]. For any other post-training workload, this is not a candidate; compare instead against a general-purpose library such as trl or verl (cross-reference; not covered here).

**Methods it ships**: one method, SFT, via QLoRA (4-bit-quantized LoRA fine-tuning) [6]. It is implemented directly in `training/finetune_qlora.py`, `training/qlora_model.py`, `training/qlora_utils.py`, and `training/data_utils_sft.py` - there is no package namespace or import path, since the code is a script tree, not an installed library [3][6]. There is no experimental/stable split, no method-taxonomy page to recheck, and no other trainer: this is the only training entry point in the repo.

**Scale it handles**: single or multi-GPU on one node via `torchrun --standalone --nnodes=1 --nproc-per-node=$GPUS_PER_NODE`, or multi-node under Slurm (`salloc --nodes N ... srun bash scripts/...`); the shipped Dromedary-2 70B SFT script targets 1 node x 8 A100-80GB GPUs [5]. There is no separate config-template layer (no DeepSpeed/FSDP config files in the repo) - sharding for the SFT stage comes only from 4-bit quantization plus LoRA, not from a distributed-training framework [6][5]. A repository contributor, replying in a closed issue about training time, stated that the 70B SFT stage finishes in about one day on 8 GPUs, and estimated about 4 days on 2 GPUs if `GRAD_ACCUMULATION` is raised from 4 to 16 to hold the global batch size constant (issue #12, CONTRIBUTOR, 2023-10-24) [7]. That is the only published run-time figure; there is no larger benchmark beyond it.

**Install**: no PyPI package - clone the repository and install its two script trees as local editable packages: `cd llama_dromedary && pip install -r requirements.txt && pip install -e .` for model-parallel LLaMA inference/generation, and `cd inference && pip install -r requirements.txt` for the chatbot demo [1]. The GitHub API's newest push (`0b86740e48ea1371752b185002b1c322118d4f23`, dated 2023-10-26) is the same commit as the tip of `main` used throughout this card - the repo has no tags or releases at all, so there is no versioned release to pin against, and every claim here is read at that commit [8]. `llama_dromedary/requirements.txt` pins nothing (`torch`, `fairscale`, `fire`, `sentencepiece`, all unversioned) [9]. `inference/requirements.txt` pins `peft==0.2.0` (an exact, load-bearing cap - peft 0.2.0 predates the LoRA/PEFT APIs `training/qlora_model.py` calls, such as `PeftModel.from_pretrained` with `adapter_name`, so installing this pin before running the training scripts will likely break them) alongside unversioned `transformers`, `accelerate`, `fire`, `gradio` [10]. The `training/` directory that runs QLoRA SFT ships no `requirements.txt` of its own; it imports `bitsandbytes`, `transformers`, `peft`, and `datasets` with no version stated anywhere in the repo [6]. No Python floor and no license file for `llama_dromedary/requirements.txt`'s dependencies is stated; the repo's own code is GPL-3.0 and the released Dromedary-2 synthetic data is CC BY-NC 4.0 [1][11]. The one hardware figure that is published applies to inference/evaluation, not to the training install itself: the inference and evaluation READMEs both state that "Dromedary is a 65B model, it requires a minimum of 130GB GPU memory to accommodate the entirety of its model weights within the GPU memory" [12][13].

**Maintained by**: the paper's authors, listed first-author Zhiqing Sun among eight [2], publishing under the GitHub organization `IBM` [4]; about 1,138 GitHub stars, not a ranking signal [4]. The repository's default-branch history stops at the 2023-10-26 commit read for this card; two later pushes on 2025-05-08 and 2025-09-18 are automated dependency-bot configuration branches (`renovate/configure`, `whitesource/configure`) that were never merged into `main`, so they are not evidence of ongoing development [14].

## Quick start

The repo has no minimal end-to-end quickstart snippet; the smallest complete run is the shipped SFT shell script, quoted in full from `training/step3_principle_engraving/scripts/finetune_dromedary2_70b_sft.sh` [5]:

```bash
# We use 1 x 8 = 80 A100-80GB GPUs
# salloc --nodes 8 --time 24:00:00 --gres=gpu:80g:8 srun bash scripts/finetune_dromedary2_70b_sft.sh
set -e
set -x

cd ..

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/your/model/dir"
export DATA_DIR="/your/data/dir"
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8

LEARNING_RATE=1e-4
BATCH_SIZE=4
GRAD_ACCUMULATION=4
NUM_EPOCHS=1
CKPT_STEPS=500

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SYNC_NODE_RANK + 1)) | tail -n 1)
export MASTER_PORT=9901

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_qlora.py \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path "/path/to/your/llama-2-70b-hf" \
    --learning_rate $LEARNING_RATE \
    --source_max_len 512 \
    --target_max_len 512 \
    --dataset "$DATA_DIR/llama2_70b_self_align_merged.json" \
    --dataset_format "dromedary" \
    --meta_prompt_pattern "../prompts/inference_prompts/dromedary_*prompt_distill.txt" \
    --double_quant True \
    --quant_type "nf4" \
    --bits 4 \
    --lora_r 64 \
    --output_dir "$MODEL_DIR/dromedary2-70b-qlora-sft" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $CKPT_STEPS \
    --save_total_limit 3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --add_eos_to_target False
```

`MODEL_DIR`, `DATA_DIR`, and `/path/to/your/llama-2-70b-hf` must be set by the user before this runs; there is no equivalent short Python snippet in the docs, since every documented entry point is a shell script wrapping `torchrun` or `python` [5][3]. Note that `--resume_from_training True` is set here even though, per Save it below, resuming from a checkpoint currently makes the script exit immediately - copying this script as-is and re-running it after any checkpoint has been written will fail.

## Start it

- One GPU: run `finetune_qlora.py` directly with `python` (or `torchrun --nproc-per-node=1`); the argument surface is unchanged.
- Multiple GPUs on one node: `torchrun --standalone --nnodes=1 --nproc-per-node=$GPUS_PER_NODE finetune_qlora.py ...`, as in the shipped script [5]. There is no launcher config-template file (no DeepSpeed/Accelerate YAML) in the repo - GPU count and node layout are set only through the shell variables and `torchrun` flags in the script itself [5].
- Multi-node: the shipped scripts wrap `torchrun` inside `salloc ... srun bash scripts/...` for Slurm clusters, with `MASTER_ADDR`/`MASTER_PORT` derived from `SLURM_JOB_NODELIST` [5][15].
- Effective batch = `per_device_train_batch_size` x GPU count x `gradient_accumulation_steps`; the shipped 70B config uses batch size 4, gradient accumulation 4, on 8 GPUs. The maintainer's issue-#12 reply confirms this arithmetic directly: to hold the same global batch size on 2 GPUs instead of 8, raise `GRAD_ACCUMULATION` from 4 to 16 [7].
- Config surface is the `TrainingArguments` subclass in `training/finetune_qlora.py`; defaults it sets that a reader should notice: `gradient_checkpointing=True`, `group_by_length=True`, `lr_scheduler_type="constant"` (the field's help text reads "Constant a bit better than cosine, and has advantage for analysis"), `double_quant=True`, `quant_type="nf4"`, `bits=4`, `optim="paged_adamw_32bit"`, `save_total_limit=40`, and `report_to="none"` (logging is off by default) [16]. The shipped 70B script overrides several of these defaults: it sets `--lr_scheduler_type "cosine"`, `--evaluation_strategy "no"`, `--save_strategy "steps"` with `--save_steps 500` and `--save_total_limit 3`, and `--report_to "tensorboard"` [5].
- Out-of-memory first aid: none is published as an explicit troubleshooting section; the mechanism available is the same knobs used to configure the run - lower `per_device_train_batch_size` and raise `gradient_accumulation_steps`, or reduce `lora_r`/`source_max_len`/`target_max_len` - and `max_memory_MB` (default 80000, i.e. an 80GB card) is read by the model-loading code but is commented out where it would be passed to `from_pretrained` (`# max_memory=max_memory` in `qlora_model.py`), so it currently has no effect [16][6].

## Watch it

This section is the mechanics only; there is no method-specific health-signal guidance for SFT loss shapes anywhere in this repo - that lives on the SFT method card.

- **Enable it**: `report_to` on the `TrainingArguments` subclass defaults to `"none"`, i.e. no logging backend is active unless set explicitly; the shipped 70B script sets `--report_to "tensorboard"` [16][5]. `logging_steps` defaults to 10; the shipped script overrides it to 5 [16][5].
- **Metric names**: the training script is a thin wrapper around the standard Hugging Face `Trainer` with no custom logging callback beyond checkpoint saving, so the logged scalars are whatever `Trainer.log_metrics` emits for a causal-LM SFT run (training loss, learning rate, etc.) - `finetune_qlora.py` calls `trainer.log_metrics("train", metrics)` and `trainer.save_metrics("train", metrics)` after training completes, and also writes a flat `metrics.json` in `output_dir` [17]. No metric-name table is published anywhere in this repo; the definitive list is whatever `transformers.Trainer` itself reports, which this card does not restate.
- **Sample-level logging of generations**: not present. Nothing in `training/` samples or logs example completions during SFT training.
- **Evaluation during training**: `evaluation_strategy` is a standard `Seq2SeqTrainingArguments` field but the shipped 70B script sets `--evaluation_strategy "no"`, i.e. the reference run does not evaluate during training at all [5]. Post-hoc evaluation is a separate step, run after training via the scripts in `mc_evaluation/` (multiple-choice HHH-Eval and TruthfulQA-MC) and via `inference/`, both requiring the model to be re-sharded for model-parallel inference first [12].
- **Stopping**: no early-stopping or reward-threshold mechanism is published anywhere in this repo (there is no RL stage here at all - only SFT); training runs for the fixed `num_train_epochs` set on the command line, with no patience or threshold field defined in `training/finetune_qlora.py`. Search performed 2026-08-12 over `training/finetune_qlora.py`, `training/qlora_utils.py`, and `training/README.md`; none define a stopping rule.

## Save it

- Checkpoints land under `output_dir` as `checkpoint-<global_step>/`, matching `transformers.trainer_utils.PREFIX_CHECKPOINT_DIR`; inside each, the LoRA adapter is written to a nested `adapter_model/` subdirectory by `kwargs["model"].save_pretrained(peft_model_path)` in the custom `SavePeftModelCallback` [18].
- That same callback actively deletes state on every save: it globs and removes any `pytorch_model*.bin` files and any `optimizer.pt` file from the checkpoint folder right after saving the adapter - "Saving PEFT checkpoint..." is printed, and the callback runs on every `on_save` and again at `on_train_end` (which also touches a `completed` marker file in `output_dir`) [18]. This means checkpoint directories never carry optimizer state, by design, regardless of any `save_only_model`-style flag.
- Because of that, the training script's own resume path is explicitly unsupported: `finetune_qlora.py` calls `get_last_checkpoint(output_dir)` to detect a prior checkpoint, and if one is found while `--resume_from_training True` is also set (as the shipped 70B script sets it), the script prints "Resuming from training not supported yet. Exiting." and calls `exit(1)` rather than resuming [17]. A code comment in the same file states plainly: "`resume_from_checkpoint` not supported for adapter checkpoints by HF. Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not." [17]. There is no working resume-from-checkpoint call form in this repo for a training run in progress.
- `checkpoint-<step>/adapter_model/` is a PEFT LoRA adapter directory, NOT a full model - it holds only the adapter weights, not the base LLaMA/LLaMA-2 weights. To use it, load the base model and apply the adapter, mirroring `training/qlora_model.py`'s own loading path: `PeftModel.from_pretrained(base_model, checkpoint_dir, adapter_name=..., is_trainable=...)` [6]. Separately, for the original Dromedary release, the README distributes weights only as LoRA "delta weights" against LLaMA to comply with the LLaMA license, to be added onto a user's own LLaMA weights, and again for inference the `inference/` scripts require converting to a model-parallel sharded checkpoint format via `utils/convert_hf_weights_to_llama_ckpt.py` before they can be loaded [1][13].
- Whether a downstream evaluator can load a saved checkpoint directly depends on which artifact you point it at: a raw `checkpoint-<step>/adapter_model/` directory needs the base model and PEFT to reconstruct the trained model; the model-parallel sharded format under `inference/` and `mc_evaluation/` needs its own conversion utility first. Neither is a drop-in Hugging Face `from_pretrained` full-model directory.

## Find it in the docs

There is no separate hosted documentation site; the docs ARE the repository's README files, read directly on GitHub.

- Top-level orientation: the repo root `README.md` [1]. Stage-by-stage training walkthrough: `training/README.md` [5]. Chatbot inference walkthrough and GPU-sharding recipes: `inference/README.md` [13]. Multiple-choice evaluation walkthrough (HHH-Eval, TruthfulQA-MC): `mc_evaluation/README.md` [12].
- Address pattern for any file at the pinned commit: `https://raw.githubusercontent.com/IBM/Dromedary/main/<path>` for the live `main` branch, or substitute the commit `0b86740e48ea1371752b185002b1c322118d4f23` for the exact version read here; the original four-stage pipeline is reachable only by substituting `dromedary_v1` for `main` in that same pattern [1].
- Runnable references beyond the docs: `training/dummy_data/vicuna_dummy_data.json`, used by `merge_and_fileter_self_align_with_dummy.py` in stage 2, is the repo's own smoke-test data [5][3]. The synthetic training data for the released models is hosted on the Hugging Face Hub, not in this repo, as `zhiqings/dromedary-65b-verbose-clone-v0` and `zhiqings/dromedary-2-70b-v2` [1].
- No official curated-tutorials page and no MCP endpoint exist for this project; the closest thing to a community layer is the repo's own GitHub Issues, which is also where its only maintainer replies live (see trap below).
- Trap, found in a closed issue: a user asked how many GPUs the original v1 pipeline needed. The maintainer (Edward-Sun, CONTRIBUTOR) replied that generating the stage-1/2 synthetic data used a cluster of 64 x 6 V100 GPUs, while the actual LoRA fine-tuning itself needed only about 16 x 6 V100 GPUs "or less" with gradient accumulation (issue #3, CONTRIBUTOR, 2023-05-16) [19]; `training/README.md`'s own launch command for that data-generation stage confirms the "64" as a node count, `salloc --nodes 64 ... --gres=gpu:32g:6 ...` [5]. Asked to clarify what "6 V100GPU" meant, the same maintainer replied that it meant six V100-32GB GPUs per node, i.e. 16 x 6 = 96 V100-32GB GPUs in total (about 3072 GB of VRAM in aggregate), because the reference run used a large batch size of 768 (issue #3, CONTRIBUTOR, 2023-05-22) [19]. Both figures describe the discontinued `dromedary_v1` data-generation and fine-tuning pipeline, not the QLoRA SFT stage on `main`, which is documented instead by the smaller 8xA100-80GB figure in the Scale field above.
- Honest boundary: this is not a reusable library. It has no installable package for the repo as a whole, no stable public API, no documented multi-node benchmark for the current `main`-branch SFT stage beyond the single maintainer estimate above, and its RLHF/RLAIF stage (SALMON) is intentionally not in this repository at all [1].

## Sources

[1] Dromedary repository root README. https://raw.githubusercontent.com/IBM/Dromedary/main/README.md. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[2] "Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision," arXiv:2305.03047 abstract page. https://arxiv.org/abs/2305.03047. Fetched 2026-08-12.

[3] GitHub API listing of `training/` directory contents. https://api.github.com/repos/IBM/Dromedary/contents/training. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[4] Dromedary GitHub repository. https://github.com/IBM/Dromedary. Fetched 2026-08-12 (GitHub repo metadata API, https://api.github.com/repos/IBM/Dromedary).

[5] Dromedary training-pipeline README and shipped SFT launch script. https://raw.githubusercontent.com/IBM/Dromedary/main/training/README.md and https://raw.githubusercontent.com/IBM/Dromedary/main/training/step3_principle_engraving/scripts/finetune_dromedary2_70b_sft.sh. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[6] `training/qlora_model.py` source (QLoRA/LoRA model construction, quantization config, adapter loading). https://raw.githubusercontent.com/IBM/Dromedary/main/training/qlora_model.py. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[7] Closed issue #12, "time taken to run dromedary training on llama2(7b) llama2(70b)," reply from Edward-Sun (CONTRIBUTOR). https://api.github.com/repos/IBM/Dromedary/issues/12/comments. Fetched 2026-08-12.

[8] GitHub API: latest commit on `main` and repository tags/releases (empty). https://api.github.com/repos/IBM/Dromedary/commits?per_page=1, https://api.github.com/repos/IBM/Dromedary/tags, https://api.github.com/repos/IBM/Dromedary/releases. Fetched 2026-08-12.

[9] `llama_dromedary/requirements.txt`. https://raw.githubusercontent.com/IBM/Dromedary/main/llama_dromedary/requirements.txt. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[10] `inference/requirements.txt`. https://raw.githubusercontent.com/IBM/Dromedary/main/inference/requirements.txt. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[11] `DATA_LICENSE` file (CC BY-NC 4.0). https://raw.githubusercontent.com/IBM/Dromedary/main/DATA_LICENSE. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23; also asserted by the code-license/data-license badges in [1].

[12] `mc_evaluation/README.md` (HHH-Eval and TruthfulQA-MC evaluation launch scripts, run only after training). Same fetch as [15].

[13] `inference/README.md` (chatbot demo, GPU-sharding conversion utilities). https://raw.githubusercontent.com/IBM/Dromedary/main/inference/README.md. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[14] GitHub API branch listing and per-branch latest-commit metadata for `renovate/configure` and `whitesource/configure`. https://api.github.com/repos/IBM/Dromedary/branches, https://api.github.com/repos/IBM/Dromedary/branches/renovate/configure, https://api.github.com/repos/IBM/Dromedary/branches/whitesource/configure. Fetched 2026-08-12.

[15] `mc_evaluation/README.md` (Slurm-wrapped `torchrun` launch pattern, shared across training and evaluation scripts). https://raw.githubusercontent.com/IBM/Dromedary/main/mc_evaluation/README.md. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[16] `training/finetune_qlora.py` source (`TrainingArguments` subclass and its defaults). https://raw.githubusercontent.com/IBM/Dromedary/main/training/finetune_qlora.py. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[17] `training/finetune_qlora.py` source (metrics/logging calls, checkpoint-detection and unsupported-resume exit path, and its code comment on optimizer/scheduler state). Same file and fetch as [16].

[18] `training/qlora_utils.py` source (`SavePeftModelCallback`, `get_last_checkpoint`). https://raw.githubusercontent.com/IBM/Dromedary/main/training/qlora_utils.py. Fetched 2026-08-12 at commit 0b86740e48ea1371752b185002b1c322118d4f23.

[19] Closed issue #3, "Time & Cost for this training," replies from Edward-Sun (CONTRIBUTOR). https://api.github.com/repos/IBM/Dromedary/issues/3/comments. Fetched 2026-08-12.
