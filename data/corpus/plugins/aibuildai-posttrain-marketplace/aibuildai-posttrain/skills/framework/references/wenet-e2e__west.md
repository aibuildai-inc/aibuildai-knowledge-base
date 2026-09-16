# west

A speech-domain post-training toolkit built on Hugging Face Transformers: `HfArgumentParser` + `Trainer` recipes for SFT, GRPO, and on-policy knowledge distillation of audio-LLMs, run through `torchrun`/DeepSpeed with no PyPI package.

**west** (WE Speech Toolkit) is described in its own paper as "a speech toolkit based on a large language model (LLM) for speech understanding, generation, and interaction" [1]. It is built and maintained by engineers from the WeNet community, led by Binbin Zhang, alongside co-authors from Northwestern Polytechnical University and industry labs [1][2]. Its API is task recipes, not a trainer library a caller imports generically: each task (`examples/aishell/asr`, `examples/grpo`, `examples/on_policy_distillation`, ...) ships a `run.sh` that calls a `west/bin/train*.py` script, which parses a `TrainingArguments` subclass with `HfArgumentParser` and hands a Hugging Face `Trainer` subclass (`MyTrainer`, `GRPOTrainer`, `KnowledgeDistillationTrainer`) a model, dataset, and (for GRPO/KD) reward functions [3][4][5]. It lives at https://github.com/wenet-e2e/west [2].

**When to pick it**: audio/speech LLM post-training (ASR, TTS, speech QA, speech interaction) where you want ready-made recipes over Qwen2-Audio / Qwen2.5-Omni and WeNet/WeSpeaker encoders, built directly on the `transformers.Trainer` API rather than a dedicated RL library such as trl or verl (cross-reference; not covered here) - its GRPO trainer is a from-scratch ~300-line `Trainer` subclass credited as based on R1-V, not an import of trl's `GRPOTrainer` [4]. Pick something else for text-only post-training or if you need trl's/verl's wider method menus and vLLM-accelerated training rollout: west's `GRPOTrainer` generates its on-policy rollouts through the training model's own `.generate()`, with no `--use_vllm`/`vllm_mode`-style flag in `west/trainer/grpo_trainer.py` or `west/bin/train_grpo.py` [4]; vLLM appears only downstream, in the separate `west/bin/decode_mmau.py` and `decode_mmsu.py` scripts used to evaluate a finished GRPO checkpoint on the MMAU/MMSU benchmarks [6].

**Methods it ships**: three, none imported from an external RL/post-training library - all are in-repo `Trainer` subclasses (method math and defining papers belong on each method's own card, not here):
- SFT: `west/bin/train.py`'s `MyTrainer(Trainer)`, used for the built-in TouchASU (speech understanding) and TouchTTS (speech synthesis) model families [3][7], plus TouchChat and TouchOmni, which the README lists as built-in speech-interaction and multimodal-interaction model families with no recipe link filled in [2]; the trainer manually logs `train_loss`/`train_accuracy` inside an overridden `training_step` [3].
- GRPO: `west.trainer.grpo_trainer.GRPOTrainer`, a `Trainer` subclass for Qwen2-Audio / Qwen2.5-Omni, imported at `from west.trainer.grpo_trainer import GRPOTrainer`; its own docstring calls it a "Simplified implementation for Audio Question Answering with custom reward functions" [4]. The GRPO README's own results table is the deciding number: Qwen2-Audio-7B goes from 56.9 to 67.2 on MMAU and 30.38 to 54.12 on MMSU with GRPO added, and both Qwen2.5-Omni-3B (69.8/59.1 -> 71.6/60.46) and Qwen2.5-Omni-7B (72.1/58.56 -> 73.4/65.38) show smaller but consistent gains [8].
- On-policy knowledge distillation: `west.trainer.kd_trainer.KnowledgeDistillationTrainer` (local teacher, loaded with DeepSpeed) and `RemoteKnowledgeDistillationTrainer` (teacher served behind a vLLM-compatible HTTP API) [5][9]. The on-policy-distillation README's own results tables carry the deciding numbers for the two tasks it targets: on Audio QA, distilling a GRPO-trained Qwen2.5-Omni-3B teacher into the Qwen2-Audio-7B + GRPO checkpoint moves MMAU from 67.2 to 67.9 while MMSU slips slightly from 54.12 to 53.30; on detailed audio captioning (evaluated by downstream QA over the generated captions), a distilled Qwen2.5-Omni-3B student scores 69.1-69.6 MMAU against a 65.6 un-distilled baseline and teacher scores of 72.3-72.8, and a distilled Qwen2.5-Omni-7B student reaches 70.1 against a 68.7 baseline [9].
- There is no published method-taxonomy page (no docs site) - this list is read directly from `west/trainer/` and `west/bin/` at the commit below and can grow as the repo does; recheck the directory listing at run time [10].

**Scale it handles**: single GPU up to single-node multi-GPU, launched with `torchrun --nproc_per_node=<n>` (the GRPO and ASR recipes both use this form; GRPO's `run.sh` computes `num_gpus` from `CUDA_VISIBLE_DEVICES` and passes `--nnodes=1 --node-rank=0`) [8][11]. Sharding is DeepSpeed ZeRO-1/ZeRO-2/ZeRO-3 via a `--deepspeed <config.json>` flag on every recipe, with ZeRO-3 additionally offloading optimizer and parameter state to CPU in the shipped config [8][11][12]. No multi-node launch form, SLURM template, or FSDP option is present in any example or script read for this card, and no scale benchmark (throughput, max model size) is published - only the DeepSpeed config files and the single-node `torchrun` invocations are documented mechanism [8][11][12].

**Install**: no PyPI package and no `setup.py`/`pyproject.toml` in the repository tree - install is `conda create -n west python=3.10 && conda activate west && pip install -r requirements.txt` from a repo clone [2]. There are no GitHub releases or tags (`releases` and `tags` API responses are both empty), so there is no version to pin beyond a commit; this card reads `requirements.txt` at the shortlisted commit b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb (2026-03-31), and it is byte-identical to the file on the `main` branch fetched 2026-08-12, so no drift was found between this pin and the live branch for this file [13][14]. That file pins `torch>=2.6.0`, `torchaudio>=2.6.0`, `transformers==4.52.3`, `deepspeed==0.17.6`, `peft==0.12.0`, `flash_attn==2.7.4.post1`, plus `git+https://github.com/wenet-e2e/wenet.git` and `git+https://github.com/wenet-e2e/wespeaker.git` as unpinned git dependencies [13]. **The GRPO and on-policy-distillation recipes carry a second, conflicting requirements file**: `examples/grpo/requirements.txt` (symlinked as `examples/on_policy_distillation/requirements.txt`) pins `transformers>=4.55.0` and adds `trl`, `vllm`, `deepspeed`, `wandb`, `tabulate`, `tensorboardX`, and `datasets==3.1.0` with no upper bound - a floor that directly conflicts with the root file's `transformers==4.52.3` pin, so installing both files into one environment is unsatisfiable as written and a GRPO/OPD user must resolve this manually [15]. `trl` is imported directly by both `west/trainer/grpo_trainer.py` (`unwrap_model_for_generation`, `prepare_deepspeed`, and `selective_log_softmax`) and `west/trainer/kd_trainer.py` (`unwrap_model_for_generation` and `prepare_deepspeed` only), and `vllm` is a hard import in `west/bin/decode_mmau.py`/`decode_mmsu.py`, so both are load-bearing dependencies for those recipes, not optional extras [4][5][6]. Licence is Apache-2.0 [2]. Python floor is 3.10 per the README's `conda create` line; no CUDA/hardware minimum is stated in the README, docs, or requirements file read for this card [2][10].

**Maintained by**: the WeNet community (GitHub org `wenet-e2e`), with Binbin Zhang as lead author of the accompanying paper [1]; 208 GitHub stars as of 2026-08-12 [2]. The repository was created 2025-06-13 and last pushed 2026-07-17, and issue #90 (a QLoRA inference bug), filed 2025-12-08, received a maintainer (`robin1001`, COLLABORATOR) reply on 2025-12-24, indicating the repository does receive maintainer triage on filed issues [2][16].

## Quick start

The README's own quickstart is install-only; there is no minimal "hello world" training snippet on the README or docs index - the smallest complete runs are each task recipe's `run.sh`. The two blocks below are trimmed excerpts of the `train` stage of each script, with unrelated flags (data-loader tuning, LR-schedule details, exit-on-error guards) omitted for length; the source files are the full, runnable scripts.

ASR SFT (`examples/aishell/asr/run.sh`, `stage=train` block), after preparing `data/train.jsonl` records of the form `{"wav": "...", "txt": "..."}` [17][11]:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus west/bin/train.py \
    --model_config_or_dir $model_conf \
    --data_path $data/train.jsonl \
    --output_dir $dir \
    --pack_size 8192 \
    --bf16 True \
    --max_steps $steps \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" --save_steps 100 --save_total_limit 100 \
    --learning_rate 3e-4 --lr_scheduler_type "cosine" \
    --deepspeed conf/ds_config_zero2.json \
    --accelerator_config conf/accelerator_config.json
```

GRPO (`examples/grpo/run.sh`, `stage=train` block), against a downloaded Qwen2.5-Omni-7B checkpoint and the `avqa_hf_dataset_path` Hub dataset [8]:
```bash
torchrun --nproc_per_node=${num_gpus} --nnodes=1 --node-rank=0 \
    --master_addr=127.0.0.1 --master_port=32778 \
    west/bin/train_grpo.py \
    --deepspeed ${deepspeed_config} \
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${dir} \
    --hf_dataset_path ${avqa_hf_dataset_path} \
    --run_name ${run_name} \
    --template ${prompt_template} --temperature 0.7 \
    --num_generations 4 --max_completion_length 1024 \
    --use_wandb true
```
There is no separate CLI entry point beyond these `west/bin/*.py` scripts - west is invoked as a script, not a console command [8][11].

## Start it

- One GPU: run either script above with `--nproc_per_node=1`, or drop `torchrun` for `python west/bin/train.py ...` (both scripts are ordinary argparse programs) [11].
- Multiple GPUs on one node: `torchrun --nproc_per_node=<n>`, the form both `examples/aishell/asr/run.sh` and `examples/grpo/run.sh` use; GRPO's script derives `<n>` from `CUDA_VISIBLE_DEVICES` [8][11]. No multi-node config template (SLURM, `--nnodes>1`) exists in the examples read for this card [8][11][12].
- DeepSpeed config selection is the sharding knob: `examples/grpo/conf/` ships `ds_zero1.json` (ZeRO-1, no offload) and `ds_zero3.json` (ZeRO-3 with CPU offload of optimizer and parameters); the GRPO README recommends starting with ZeRO-1 for speed, and "If you encounter OOM errors, switch to ZeRO-3" [12][8]. The ASR recipe instead ships `ds_config_zero2.json` (ZeRO-2) [11][10].
- Effective batch size is `per_device_train_batch_size x num_gpus x gradient_accumulation_steps`, the standard `transformers.Trainer` arithmetic; both example scripts set `train_batch_size`/`train_micro_batch_size_per_gpu` to `"auto"` in the DeepSpeed JSON, meaning DeepSpeed derives them from the `TrainingArguments` values rather than the JSON overriding them [12][8][11].
- GRPO's Config surface is a plain dataclass (`CustomTrainingArguments(TrainingArguments)`) in `west/bin/train_grpo.py`, not a separate `GRPOConfig` class; it defaults `bf16=True`, `save_only_model=True`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=2`, `num_generations=8`, `temperature=1.0`, `beta=0.04` (KL penalty) - the bf16-on-by-default choice assumes a bf16-capable GPU [4]. Generation runs through the training model's own `.generate()` inside `GRPOTrainer._rollout`, wrapped by trl's `unwrap_model_for_generation` for DeepSpeed compatibility; there is no vLLM path for training-time rollout, only for the separate post-training evaluation scripts `decode_mmau.py`/`decode_mmsu.py` (see Save it) [4][6].
- Out-of-memory first aid: none is written as an explicit troubleshooting note in the GRPO or ASR READMEs beyond the ZeRO-1-to-ZeRO-3 recommendation quoted above [8]; no generation-side memory knob (equivalent to trl's `vllm_gpu_memory_utilization`) exists for GRPO training rollout because that generation is not a separate engine there - a vLLM `LLM(...)` instance, with its own `tensor_parallel_size`/`max_num_seqs` arguments, only appears in the separate `decode_mmau.py`/`decode_mmsu.py` evaluation scripts, not in training [6].

## Watch it

This section is the mechanics only - what a logged reward or KL value means for GRPO or for on-policy distillation lives on those methods' own cards, not here.

- **Enable it**: `report_to` is set by each `west/bin/train_*.py` script's own `use_wandb` flag - `args.report_to = ["wandb"] if args.use_wandb == "true" else []` when the CLI does not pass `--report_to` explicitly - so a run with `--use_wandb` unset or `false` logs to no external tracker at all, only to the console via `logging_steps` (default 1 for GRPO/KD) [4][5]. The ASR SFT recipe instead passes `--report_to "tensorboard"` explicitly on the command line [11].
- **GRPO metric names**, read directly from `GRPOTrainer._log_metrics` in `west/trainer/grpo_trainer.py` (there is no separate metrics-reference page to point at): `completion_length`, `completion_length_min`, `completion_length_max`; one `rewards/<func_name>` per reward function passed to the trainer (e.g. `rewards/accuracy_reward`, `rewards/format_reward`); `reward` (weighted sum, mean over the batch); `reward_std`; `kl` (mean per-token KL against the frozen reference model, only meaningful when `beta > 0`) [4].
- **Knowledge-distillation metric names**, read from `KnowledgeDistillationTrainer` in `west/trainer/kd_trainer.py`: `completion_length`/`_min`/`_max`, `kl`, `loss`, and - only when reward functions are supplied for monitoring - `reward`, `reward_min`, `reward_max`, and per-function `rewards/<name>`, `rewards/<name>_min`, `rewards/<name>_max` [5]. The training script comments that these reward functions are "for monitoring (not used in loss, only for logging)" in the KD case [5].
- **SFT metric names**, read from `MyTrainer.training_step` in `west/bin/train.py`: `train_loss` and `train_accuracy` (token accuracy over non-`-100` labels), logged manually every `logging_steps * gradient_accumulation_steps` steps inside the training step, in addition to whatever the base `Trainer` logs by default (e.g. `loss`, `learning_rate`, `epoch`) [3].
- **Sample-level logging of generations**: no `log_completions`-equivalent flag was found in `GRPOTrainer`, `KnowledgeDistillationTrainer`, or the two training scripts; generated strings are used for reward computation and logging of aggregate reward/length statistics only, not persisted as text samples [4][5].
- **Evaluation during training**: both `train_grpo.py` and `train_knowledge_distillation.py` build an `eval_dataset` from the same `hf_dataset_path`'s `"validation"` split unconditionally and pass it to the trainer; evaluation cadence and behavior otherwise follow the base `transformers.Trainer`'s `eval_strategy`/`eval_steps` fields, which are not overridden by west's `CustomTrainingArguments` dataclasses [4][5].
- **Stopping**: no RL-specific stopping rule, reward threshold, or patience value is published in the GRPO README, the on-policy-distillation README, or in `west/trainer/grpo_trainer.py` / `west/trainer/kd_trainer.py` - searched 2026-08-12; `max_steps` (default 1000) and `num_train_epochs` (default 2) are the only stopping controls found, both plain step/epoch counts with no dependence on the reward or KL signal [4][5][8].

## Save it

- Checkpoints land under `--output_dir` as `checkpoint-<step>/` directories created by the base `transformers.Trainer`'s standard checkpointing; `west/bin/decode_mmau.py`/`decode_mmsu.py` and the GRPO `run.sh` reference them as `${dir}/checkpoint-${iter}` [8]. The exact file inventory inside a checkpoint (e.g. `model.safetensors` naming) was not enumerated by any file read for this card - inspect one on disk.
- `save_steps` (100 in both the GRPO and ASR recipes) and `save_total_limit` (100 in the ASR recipe; unset, so unbounded, in the GRPO recipe as read) control cadence and retention [8][11].
- **`save_only_model` trades resumability for size, and GRPO/KD default it to `True`**: `west/bin/train_grpo.py` and `west/bin/train_knowledge_distillation.py` both set `save_only_model: bool = field(default=True, ...)` in their `CustomTrainingArguments`, so unless overridden on the command line, a GRPO or KD checkpoint holds only model weights, not optimizer/scheduler state, and `trainer.train(resume_from_checkpoint=...)` on such a checkpoint would only be able to reload weights, not resume the optimizer trajectory [4][5]. The SFT script (`west/bin/train.py`) does not set `save_only_model`, so it inherits the base `Trainer` default (`False`) and its checkpoints keep optimizer state [3].
- **Resume**: `west/bin/train.py`'s `main()` checks `pathlib.Path(output_dir).glob("checkpoint-*")` and, if any exist, calls `trainer.train(resume_from_checkpoint=True)` (else a fresh `trainer.train()`), then always calls `trainer.save_state()` at the end [3]. No equivalent auto-resume check is present in `train_grpo.py` or `train_knowledge_distillation.py` - resuming those requires passing `resume_from_checkpoint` explicitly [4][5].
- **LoRA is handled inside the model, not the trainer**: TouchASU's `modeling_touch_asu.py` wraps its LLM with `peft.get_peft_model(self.llm, LoraConfig(**config.lora_config))` when `config.lora_config` is set, so the LoRA adapter is part of the model's own `state_dict` and its config is persisted in the model's own `config.json` [18]. `west/bin/decode.py` loads a saved checkpoint directly with `AutoModel.from_pretrained(model_config_or_dir)` and runs `.generate()` - no separate adapter-merge step is shown, because the on-disk checkpoint is the full wrapped model, not an adapter-only directory in the trl/PEFT sense [19]. Whether an *adapter-only* PEFT checkpoint (saved by a bare `PeftModel.save_pretrained()`) can be loaded this same way is not addressed by any file read for this card - inspect the checkpoint's config before assuming.
- Loader handoff: `decode.py`'s `AutoModel.from_pretrained(checkpoint_dir)` is the pattern shown for reloading a west checkpoint for inference [19]; whether an external evaluator harness can load it depends on that harness's own loader contract, which is out of scope here.

## Find it in the docs

There is no versioned docs site for west - "docs" means Markdown files inside the repository at whatever ref you check out, plus per-recipe `README.md` files; the address pattern is `https://github.com/wenet-e2e/west/blob/<ref>/<path>` (or `raw.githubusercontent.com/wenet-e2e/west/<ref>/<path>` for the raw text), where `<ref>` is a branch (`main`) or a full commit SHA - there are no release tags to substitute (`git tag`/`releases` are both empty as of 2026-08-12) [2][14].

- Top-level entry point: `docs/README.md`, which links out to `docs/data_format.md`, `docs/data_pack.md` (sequence packing), and `docs/data_extractor.md`, and marks "Model Design" as `TODO` - the docs are openly incomplete on the modeling side [10].
- Task-to-recipe map, from the root `README.md`'s own "Supported Tasks and Models" table: ASR -> `examples/aishell/asr`, TTS -> `examples/libritts/tts`, speech QA -> `examples/belle_1.4M_qa`; that table lists Speech Interaction and MultiModal Interaction (TouchChat/TouchOmni) with no recipe link filled in [2]. GRPO and on-policy distillation are not in that table at all - their recipes live at `examples/grpo` and `examples/on_policy_distillation` in the repository tree, found by directory listing rather than by the README's task table [10]. Each recipe directory's own `README.md` is the closest thing to a per-task manual, including its own results table and configuration-argument table (see the GRPO README's "Training Arguments" and "Prompt Templates" tables) [8].
- Runnable references beyond a single README: `west/bin/` holds every entry-point script (`train.py`, `train_grpo.py`, `train_knowledge_distillation.py`, `decode.py`, `decode_mmau.py`, `decode_mmsu.py`, `tts_flow_inference.py`); the tree listing at this commit shows three test files total: `test/test_sequence_pack.py`, `demo/goat_slm/test_goat_slm.py`, and `demo/osum_echat/test_osum_echat.py` [10]. Known-good smoke-test data for GRPO/KD is fetched by `run.sh --stage prepare`, which pulls `gijs/avqa-processed` and `yuantuo666/MMSU-full_5k_hf_format.v0` from the Hugging Face Hub and MMAU test-mini audio via `scripts/download_mmau_test.sh` [8].
- No official curated tutorials page or blog series was found linked from the README or `docs/`; the only external write-up pointed to is the project's own arXiv paper [1][2]. No official MCP endpoint for querying west's docs was found.
- A confirmed trap, from a closed issue with a maintainer reply: issue #90 reports `NotImplementedError: Cannot copy out of meta tensor; no data!` when loading a QLoRA-finetuned TouchASU checkpoint for inference; maintainer `robin1001` (repository collaborator) asked for the reporter's config and error log on 2025-12-24, and the reporter posted their `lora_config` plus a `quantization`/`load_in_4bit` block on 2026-01-01 - the issue is closed with no further maintainer reply recorded, so QLoRA (4-bit) inference load-back is an open sharp edge, not a documented supported path [16].
- Honest boundary: west documents no multi-node launch path and no FSDP option; its GRPO trainer's own rollout generation is not vLLM-accelerated (vLLM is used only downstream, for evaluating a finished checkpoint, not for training-time sampling) - all training-time mechanism found in this card is single-node, DeepSpeed-sharded, with generation through the training model's own `.generate()` [8][11][12][4][6].

## Sources

All GitHub pages and raw files are read at commit b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb unless a different ref is stated; the `main`-branch comparison for `requirements.txt` and the issue-tracker search are live reads dated 2026-08-12. `trl` and `vllm` are cited directly ([4], [6]) because they are hard, version-constrained imports for the GRPO/KD and decode paths, not passing mentions. Other ecosystem tools named in passing (Hugging Face `transformers`, `peft`, DeepSpeed, `torchrun`, WeNet, WeSpeaker) are reached through the files below and are deliberately not enumerated as separate references.

[1] WEST: LLM based Speech Toolkit for Speech Understanding, Generation, and Interaction (arXiv:2509.19902 abstract page). https://arxiv.org/abs/2509.19902. Fetched 2026-08-12.

[2] wenet-e2e/west GitHub repository (description, licence, star count, README, task table, install instructions, releases/tags emptiness). https://github.com/wenet-e2e/west. Repository metadata and README fetched 2026-08-12 at commit b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb.

[3] west/bin/train.py (SFT `MyTrainer`, manual `train_loss`/`train_accuracy` logging, checkpoint-resume check, `save_state`). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/west/bin/train.py. Fetched 2026-08-12.

[4] west/bin/train_grpo.py and west/trainer/grpo_trainer.py (GRPOTrainer implementation, CustomTrainingArguments defaults, logged metrics, rollout via `.generate()`, no vLLM path, `trl` import for `unwrap_model_for_generation`/`prepare_deepspeed`). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/west/bin/train_grpo.py and .../west/trainer/grpo_trainer.py. Fetched 2026-08-12.

[5] west/bin/train_knowledge_distillation.py and west/trainer/kd_trainer.py (KnowledgeDistillationTrainer / RemoteKnowledgeDistillationTrainer, CustomTrainingArguments defaults, logged metrics). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/west/bin/train_knowledge_distillation.py and .../west/trainer/kd_trainer.py. Fetched 2026-08-12.

[6] west/bin/decode_mmau.py and west/bin/decode_mmsu.py (vLLM-based decode scripts for scoring a finished GRPO checkpoint on the MMAU/MMSU benchmarks; `from vllm import LLM, SamplingParams`, `LLM(..., tensor_parallel_size=..., max_num_seqs=...)`). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/west/bin/decode_mmau.py and .../west/bin/decode_mmsu.py. Fetched 2026-08-12.

[7] west/models/touch_asu/, touch_tts/, touch_chat/ package listing (built-in model families used by the SFT recipe). https://github.com/wenet-e2e/west/tree/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/west/models. Tree listing fetched 2026-08-12.

[8] examples/grpo/README.md and examples/grpo/run.sh (GRPO quickstart, training-argument and prompt-template tables, results table, DeepSpeed config recommendation, launch command). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/examples/grpo/README.md and .../examples/grpo/run.sh. Fetched 2026-08-12.

[9] examples/on_policy_distillation/README.md (on-policy distillation flow, local vs. remote teacher modes, results table). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/examples/on_policy_distillation/README.md. Fetched 2026-08-12.

[10] docs/README.md and full repository tree at commit b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb (docs index, "Model Design: TODO", directory listing used to enumerate west/trainer and test/ contents). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/docs/README.md and https://api.github.com/repos/wenet-e2e/west/git/trees/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb?recursive=1. Fetched 2026-08-12.

[11] examples/aishell/asr/README.md and examples/aishell/asr/run.sh (ASR SFT quickstart, launch command, results table). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/examples/aishell/asr/README.md and .../examples/aishell/asr/run.sh. Fetched 2026-08-12.

[12] examples/grpo/conf/ds_zero1.json, ds_zero3.json, and examples/aishell/asr/conf/ds_config_zero2.json (DeepSpeed ZeRO stage and offload configuration). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/examples/grpo/conf/ds_zero1.json (and sibling paths). Fetched 2026-08-12.

[13] requirements.txt at commit b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb (dependency pins). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/requirements.txt. Fetched 2026-08-12.

[14] requirements.txt on the `main` branch, and GitHub releases/tags API responses (both empty), used to check drift and confirm no versioned release exists. https://raw.githubusercontent.com/wenet-e2e/west/main/requirements.txt, https://api.github.com/repos/wenet-e2e/west/releases, https://api.github.com/repos/wenet-e2e/west/tags. Fetched 2026-08-12.

[15] examples/grpo/requirements.txt, and examples/on_policy_distillation/requirements.txt (confirmed a symlink to the former by diffing the two fetched files). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/examples/grpo/requirements.txt and .../examples/on_policy_distillation/requirements.txt. Fetched 2026-08-12.

[16] wenet-e2e/west issue #90, "Inference Error After use QLoRA Fine-tuning in Touch-ASU" (closed, maintainer reply from `robin1001`, COLLABORATOR). https://github.com/wenet-e2e/west/issues/90. Fetched 2026-08-12.

[17] examples/aishell/asr/README.md data-format example (`{"wav": ..., "txt": ...}` JSONL). Same source as [11]. Fetched 2026-08-12.

[18] west/models/touch_asu/modeling_touch_asu.py (`peft.get_peft_model` wrapping inside the model class). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/west/models/touch_asu/modeling_touch_asu.py. Fetched 2026-08-12.

[19] west/bin/decode.py (`AutoModel.from_pretrained` checkpoint-loading pattern for inference). https://raw.githubusercontent.com/wenet-e2e/west/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/west/bin/decode.py. Fetched 2026-08-12.
