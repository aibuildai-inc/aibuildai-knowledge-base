# VLM-R1

https://github.com/om-ai-lab/VLM-R1

A research codebase, not a packaged library: a fork of Hugging Face's open-r1 that adds a vision-language GRPO trainer on top of Qwen2.5-VL and InternVL, run by editing shell scripts and cloning the repo rather than by `pip install`.

VLM-R1 describes itself as "a stable and generalizable R1-style Large Vision-Language Model" project that applies DeepSeek-R1's rule-based reward formulation, which the accompanying tech report identifies as the core of the R1 method, to vision-language tasks such as referring expression comprehension (REC) and open-vocabulary detection (OVD) [1][2]. It is built and maintained by Om AI Lab, described on its GitHub organization page as "Open Multimodal AGI Research" [3][1]. The API surface is not a pip package with trainer classes; it is a `git clone` of the repo followed by editing and running `torchrun`-launched Python scripts (`grpo_rec.py`, `grpo_jsonl.py`) that each wire a Qwen2.5-VL or InternVL model, a JSONL image+text dataset, and a set of reward functions into a vendored GRPO trainer class [4][5]. A third script, `sft.py`, is also present in the source tree but is an unmodified, text-only Hugging Face open-r1 script (its own docstring example fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` on `HuggingFaceH4/Bespoke-Stratos-17k`, no images) that the README never references; VLM-R1's own documented SFT path is external, through LLaMA-Factory (see Methods it ships) [6][7].

**When to pick it**: pick VLM-R1 specifically to reproduce or extend its own REC/OVD/GUI-defect-detection GRPO recipes on Qwen2.5-VL or InternVL, or to use its `VLMBaseModule` pattern for adding a new VLM to a GRPO loop [8][1]. It is not a general-purpose post-training library like trl or verl (cross-reference; not covered here): there is no pip package, no stable public API, and the codebase is a maintained fork rather than an actively evolving framework — its own contributors say the vLLM-accelerated trainer variant is not their recommended path (see Watch it) [9].

**Methods it ships**: the README groups its supported training modes under "Features" as full-parameter GRPO, GRPO with frozen vision modules, LoRA GRPO, multi-node GRPO, and multi-image-input GRPO [8]; separately it documents an SFT path that is not implemented in this repo but delegated to the external LLaMA-Factory project [10]. GRPO here is a hand-vendored `VLMGRPOTrainer(Trainer)` in `src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py`, not a subclass of trl's `GRPOTrainer` — it extends `transformers.Trainer` directly and reimplements the GRPO update, importing only trl's `GRPOConfig` dataclass and a few utility functions [11]. A second, non-default trainer class, `Qwen2VLGRPOVLLMTrainer` in `vllm_grpo_trainer.py`, adds vLLM-based generation, but is not the trainer any of the README's run scripts invoke [12][8]. There is no PPO, DPO, or other offline-preference method in this repo; SFT is out-of-repo (LLaMA-Factory) [10].

**Scale it handles**: single GPU up to multi-node, launched with `torchrun --nproc_per_node=<N> --nnodes=<M> ...`, with DeepSpeed ZeRO-2 or ZeRO-3 config files (`local_scripts/zero2.json`, `zero3.json`) selected per run script for sharding [13][14][15]. A documented multi-node path exists as a demo script (`run_scripts/multinode_training_demo.sh`) that generates a Docker Compose file per node and reads training args from a companion YAML (`run_scripts/multinode_training_args.yaml`), with node hostnames and IPs to be filled in by the user — this is unpublished-benchmark mechanism, not a measured multi-node result [15][16]. The alternative vLLM-generation trainer places the vLLM engine on a single additional GPU (`vllm_device` defaults to `"auto"`, resolved in code to `cuda:{num_training_processes}`, i.e. the next free GPU index) rather than sharding vLLM itself [12].

**Install**: no PyPI package; the documented path is `git clone` plus `bash setup.sh`, which runs `pip install -e ".[dev]"` from `src/open-r1-multimodal` (an editable install of the vendored, renamed-but-unrenamed `open-r1` package) followed by pinned extras (`wandb==0.18.3`, `flash-attn`, `qwen_vl_utils`, `torchvision`, and others) [17]. The README's own setup instructions create a `python=3.10` conda environment; the setup script's own comment header instead names `python=3.11` — the two documented Python versions disagree, and neither is enforced by a version check in the script itself [17][18]. At the newest pushed commit (`90052f478646e38bc67363862f699b7afca6b337`, 2026-07-07), `src/open-r1-multimodal/setup.py` pins `transformers==4.49.0`, `trl==0.17.0`, `vllm==0.6.6.post1`, `deepspeed==0.15.4`, `liger_kernel==0.5.2`, and floors `torch>=2.5.1`; `python_requires>=3.10.9` [18]. There is no tagged release newer than v0.2.1 (published 2025-04-15, commit `3c04653c2d228c00babffc17d2fa468ed7309382`) whose own `setup.py` pins `trl` unbound to an installed git checkout of trl's `main` branch instead of a fixed version and floors `transformers==4.49.0`, `vllm==0.6.6.post1`, `deepspeed==0.15.4` — so a reader following the v0.2.1 tag gets a moving trl, while the current push pins trl exactly; the commit read for the rest of this card is about 448 days (roughly 1.2 years) ahead of that release tag [19]. Apache-2.0 licence [3]. No CUDA or GPU minimum is stated anywhere in the README or setup script; `flash-attn` in the setup script implies an NVIDIA GPU is assumed, not documented as a floor [17].

**Maintained by**: Om AI Lab (GitHub org, ~877 followers) [20][3]; the repo's own README changelog lists dated updates through 2025-08-29 (an Ascend/xllm inference optimization) and shows continuous activity from the 2025-02-15 initial release through 2025-08-29, with the repository's most recent push recorded as 2026-07-07 [8][3].

## Quick start

The README's REC-GRPO quick start, as a sequence of documented steps, is not runnable as a single snippet — it requires downloading data first [7]:

```bash
conda create -n vlm-r1 python=3.10
conda activate vlm-r1
bash setup.sh
```

then, after downloading and unzipping the COCO train2014 images and the RefCOCO/+/g annotation JSONLs and editing `data_paths`/`image_folders` in the script [7]:

```bash
bash run_scripts/run_grpo_rec.sh
```

which under the hood runs [13]:

```bash
torchrun --nproc_per_node="8" --nnodes="1" --node_rank="0" \
    --master_addr="127.0.0.1" --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/Qwen2.5-VL-3B-GRPO-REC \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/rec.yaml \
    --image_root <your_image_root> \
    --max_prompt_length 1024 --num_generations 8 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --bf16 --torch_dtype bfloat16 --report_to wandb \
    --gradient_checkpointing false --attn_implementation flash_attention_2 \
    --num_train_epochs 2 --save_steps 100 --save_only_model true
```

For custom data, the README documents a JSONL record shape (`id`, `image` or list of `image` paths, `conversations`) consumed by `grpo_jsonl.py`, launched the same way with `--data_file_paths` and `--image_folders` (colon-separated for multiple files/folders) [21].

## Start it

- One GPU: drop `--nproc_per_node` to 1 and remove the multi-GPU DeepSpeed config, or keep DeepSpeed ZeRO with a single process — the run scripts do not special-case single-GPU.
- Multiple GPUs, one node: `torchrun --nproc_per_node=<N>` with a DeepSpeed config path via `--deepspeed`; `local_scripts/zero2.json` (used by the LoRA and "more params" example scripts) sets `zero_optimization.stage: 2` (shards optimizer state and gradients, not parameters, with `offload_optimizer.device: "none"`, i.e. no CPU offload), while `zero3.json` (used by the full-finetune script) sets stage 3 (also shards parameters) with `stage3_gather_16bit_weights_on_model_save: true` so a save gathers full fp16/bf16 weights [13][14][22].
- Multiple nodes: `run_scripts/multinode_training_demo.sh` takes node names, looks up their IPs from a hostname map hard-coded at the top of the script, and generates a Docker Compose file per invocation; training args are read from a companion YAML (`run_scripts/multinode_training_args.yaml`, matched by the `RUN_NAME` variable) rather than passed as flags [15][16].
- LoRA: `run_grpo_rec_lora.sh` adds `--use_peft true --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 --lora_task_type CAUSAL_LM --freeze_vision_modules true` on top of the base REC script, and drops `--gradient_checkpointing` to `true` [23].
- Generation-layout choice: the default trainer (`VLMGRPOTrainer`, what every README run script invokes) generates with the training model itself via `model.generate`, no separate engine [11]. The alternate `Qwen2VLGRPOVLLMTrainer` (not wired into any README run script) instead starts a vLLM `LLM()` on one dedicated GPU chosen by `--vllm_device` (default `"auto"`, resolved to the next free CUDA index after the training ranks) with `--vllm_gpu_memory_utilization` (default 0.9) controlling that GPU's memory share [12].
- Effective batch size is `per_device_train_batch_size x num_processes x gradient_accumulation_steps`; the README's OOM guidance is to reduce `per_device_train_batch_size` first [7].
- Config surface: GRPO-specific fields are a dataclass (`GRPOConfig`) layered on trl's own `GRPOConfig`/`TrainingArguments`, with per-repo defaults including `max_prompt_length=512`, `num_generations=8`, `max_completion_length=256`, `temperature=0.9`, `top_p=1.0`, `top_k=50`, `learning_rate=1e-6`, `beta=0.04` (KL coefficient against the reference model), `epsilon=0.2` (PPO-style clip range), `num_iterations=1` [24]. None of the README run scripts change the default precision — they all pass `--bf16 --torch_dtype bfloat16` explicitly [13][23][25].
- Out-of-memory first aid: the README's own note is "you can try to reduce the `per_device_train_batch_size`" [7]. A repository contributor (SZhanZ) gives the same short list in a closed issue: reduce `num_generations`, set `gradient_checkpointing` to `true`, or use LoRA, adding that 8xA100-80GB should handle a full-parameter 7B model but a 72B model does not fit on a single node (issue #7) [26]. Contributor replies on `gradient_checkpointing` disagree by case: in issue #36, a user (no maintainer affiliation) reports that with ZeRO-3 on a 3B model on 8xA800-80GB, disabling `gradient_checkpointing` caused OOM [9]; in a separate closed issue, contributor xrc10 instead recommends turning `gradient_checkpointing` off and using DeepSpeed ZeRO-3, stating that is the setting used in the team's own experiments (issue #74) [27]. A repository contributor also explains the `num_iterations` GRPO config field as a way to reduce seconds-per-step by reusing each generation round for more than one optimizer update, and separately recommends `num_generations=7` for OOM on an A100 (issue #77) [28]. A separate unresolved issue reports OOM training a 7B model on 8xH800-80GB even after setting `num_generations=2` and `per_device_train_batch_size=1`, with `--gradient_checkpointing true` alone reported as insufficient by a non-maintainer commenter (issue #93) [29].

## Watch it

This section covers only the logging mechanics; what a healthy KL, reward, or clip-fraction curve looks like for GRPO is on that method's card, not here.

- Logging is enabled the same way as any `transformers.Trainer`-based run: `--report_to wandb` (used by every README run script) turns on Weights & Biases; leaving `report_to` unset or `"none"` logs nowhere but stdout [13][23].
- `VLMGRPOTrainer` overrides `Trainer.log()` to merge in a running average of its own metrics dict before calling into the base logger, so these appear at each `logging_steps` interval: `completion_length`, `reward` (mean over the batch), `reward_std`, one `rewards/<name>` entry per reward function registered for the run, `kl` (only appended when `beta` is non-zero), and `clip_ratio` [30].
- `GRPOConfig` carries a `log_completions` field (inherited docstring: log a sample of prompt/completion pairs every `logging_steps` steps) [24], but grepping both trainer implementations for its name at the commit read for this card finds zero uses — the flag is defined and documented but not read anywhere in `grpo_trainer.py` or `vllm_grpo_trainer.py`, so setting it does nothing in this repo [11][12].
- Sample-level generation logging instead happens through the reward functions themselves, gated by two environment variables the run scripts export: `DEBUG_MODE=true` and `LOG_PATH=<path>`. When set, `grpo_jsonl.py`'s accuracy/format/length/repetition reward functions append plain-text records (image path, problem, model completion, ground-truth solution) to `<LOG_PATH>_accuracy.txt`-style files whenever a completion scores at or below a per-reward threshold — this is a debug trace to disk, not a wandb table [31].
- Evaluation during training is not wired into these scripts: the README's evaluation section is a separate offline step (`torchrun ... test_rec_r1.py` / `test_rec_baseline.py` in `src/eval`, run manually against a saved checkpoint after training), not an `eval_dataset`/`eval_steps` loop inside the GRPO run itself [32].
- No stopping-rule or threshold is published: the README's only OOM/tuning guidance is the single note quoted above, and no early-stopping field appears in `GRPOConfig`; this card's search covered the README's Setup/Training/Evaluation sections and `grpo_config.py`'s full field list [7][24].

## Save it

- Checkpointing is inherited, unmodified `transformers.Trainer` behavior — `VLMGRPOTrainer` does not override `_save` or `save_model` — so checkpoints land under `--output_dir` as `checkpoint-<global_step>/` directories on the same cadence and retention fields as any `TrainingArguments`-based run: `--save_steps` (100 in every README script) and `--save_total_limit` (set to 3 in the multi-node YAML example, unset — unlimited — in the single-node scripts) [13][16][11].
- Every README run script passes `--save_only_model true`, which (as `transformers.Trainer` documents for this flag generically) drops optimizer, scheduler, and RNG state from the checkpoint, shrinking it but making it unusable for `resume_from_checkpoint` — this repo's scripts do not demonstrate a resume call at all [13][23].
- With ZeRO-3 (`zero3.json`, used by the full-finetune script), `stage3_gather_16bit_weights_on_model_save: true` means a save gathers full-precision weights across shards rather than leaving them partitioned [22].
- LoRA runs (`--use_peft true`) wrap the model with `peft.get_peft_model` inside `VLMGRPOTrainer.__init__`; no call to `merge_and_unload` appears anywhere in the trainer, so a saved checkpoint under LoRA is an adapter-only directory, not a merged full model — consistent with the standard, unmodified `transformers`+`peft` `Trainer.save_model` contract this repo relies on rather than reimplements [33].
- Whether an evaluator can load a saved checkpoint directly is the loader's contract, not this repo's: a full-finetune `checkpoint-<step>/` directory is a standard Hugging Face model directory loadable with `from_pretrained`; a LoRA `checkpoint-<step>/` directory is an adapter and needs pairing with the base model id passed at training time (`--model_name_or_path`) before it is a usable model.

## Find it in the docs

There is no separate hosted docs site: the primary reference is the repository README itself, plus a handful of linked project-blog posts and the arXiv tech report.

- Repository: https://github.com/om-ai-lab/VLM-R1 — the README is the single source for setup, all run-script flags, dataset formats, and the model/checkpoint table [1].
- Adding a new VLM: `assets/add_new_model.md` documents eleven methods a `VLMBaseModule` subclass must implement (`get_vlm_key`, `get_model_class`, `post_model_init`, `is_embeds_input`, `get_processing_class`, `get_vision_modules_keywords`, `get_custom_multimodal_keywords`, `get_non_generate_params`, `get_custom_processing_keywords`, `prepare_prompt`, `prepare_model_inputs`) to plug a new VLM family into the existing GRPO trainer, pointing to `qwen_module.py` and `internvl_module.py` as worked examples [5].
- Tech report: "VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model", arXiv:2504.07615, submitted 2025-04-10 — covers the ablations behind the repo's design choices (reward hacking in object detection, an "OD aha moment", data-quality and model-scale effects) that the README itself only summarizes [2].
- Project blog: `om-ai-lab.github.io`, linked from the README changelog for specific findings, e.g. the 2025-03-20 OVDEval results post and the 2025-03-24 REC pixel-config correction post [8].
- Runnable references: `run_scripts/` at the repo root (and mirrored under `src/open-r1-multimodal/run_scripts/`) holds every documented recipe as a shell script — REC full fine-tune, REC LoRA, REC InternVL, multi-image GUI-defect, and a "more params" example exposing `--beta` and `--epsilon_high` not shown in the base script [13][23][25]. `local_scripts/zero2.json` and `zero3.json` are the DeepSpeed templates every script references by relative path [14][22].
- Known-good smoke-test data: the README points to Hugging Face dataset `omlab/VLM-R1` for the COCO/RefCOCO/RefCOCO+/RefCOCOg REC annotations and images, and separate zips for multi-image GUI-defect and LISA-grounding out-of-domain evaluation data [7][34].
- No official MCP endpoint or curated community-tutorials page is documented for this repository, unlike the `trl` or `transformers` docs sites (cross-reference; not covered here).
- Traps found in closed issues: see the vLLM-trainer and OOM notes under Start it (issues #7, #36, #74, #77, and #93), including the contributor disagreement on whether to enable or disable `gradient_checkpointing` under ZeRO-3 [9][29][26][27][28]. Issue #81 was also read and its replies carry no CONTRIBUTOR/OWNER association, so it is not cited as a maintainer-confirmed trap.
- Honest boundary: this is a single-purpose research fork, not a general post-training framework — it ships only GRPO (SFT is delegated entirely to the external LLaMA-Factory project) [10], has no PyPI release, no published multi-node benchmark, and its own contributors describe the vLLM-generation trainer as not clearly faster and difficult to reproduce results with, so it is not the vLLM-accelerated path a reader should expect from its presence in the source tree [9].

## Sources

Method names (GRPO, SFT, LoRA) are deliberately cited to nothing here; their defining papers live on the methodology cards. Ecosystem tools named in passing (DeepSpeed, trl, LLaMA-Factory, Qwen2.5-VL, InternVL, vLLM, wandb) are reached through the cited README and source files and are not separately enumerated. All GitHub source-file and API readings below were made at commit `90052f478646e38bc67363862f699b7afca6b337` (the repository's newest push as of the read) unless a release tag is explicitly named, on 2026-08-10; the v0.2.1 release comparison in Install is a separate reading of that tag's own `setup.py` at commit `3c04653c2d228c00babffc17d2fa468ed7309382`.

[1] om-ai-lab/VLM-R1 GitHub repository (README). https://github.com/om-ai-lab/VLM-R1. Fetched 2026-08-10.

[2] "VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model", arXiv:2504.07615. https://arxiv.org/abs/2504.07615. Fetched 2026-08-10.

[3] om-ai-lab GitHub organization API. https://api.github.com/orgs/om-ai-lab. Fetched 2026-08-10.

[4] VLM-R1 repository file tree (Git Trees API, recursive). https://api.github.com/repos/om-ai-lab/VLM-R1/git/trees/main?recursive=1. Fetched 2026-08-10.

[5] `assets/add_new_model.md`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/assets/add_new_model.md. Fetched 2026-08-10.

[6] `src/open-r1-multimodal/src/open_r1/sft.py` (vendored, unmodified Hugging Face open-r1 text-only SFT script; not referenced by readme.md). https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/src/open-r1-multimodal/src/open_r1/sft.py. Fetched 2026-08-10.

[7] VLM-R1 README, Setup and Training > REC > GRPO sections. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/README.md. Fetched 2026-08-10.

[8] VLM-R1 README, Features and Update sections. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/README.md. Fetched 2026-08-10.

[9] Issue #36, "3B model OOM and Training Efficiency" (closed), including comment by contributor xrc10. https://github.com/om-ai-lab/VLM-R1/issues/36. Fetched 2026-08-10.

[10] VLM-R1 README, SFT subsection (delegates to LLaMA-Factory). https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/README.md. Fetched 2026-08-10.

[11] `src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py` (`VLMGRPOTrainer` class, `Trainer` base, `log()` override, PEFT wrapping). https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py. Fetched 2026-08-10.

[12] `src/open-r1-multimodal/src/open_r1/trainer/vllm_grpo_trainer.py` (`Qwen2VLGRPOVLLMTrainer` class, `vllm_device` resolution). https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/src/open-r1-multimodal/src/open_r1/trainer/vllm_grpo_trainer.py. Fetched 2026-08-10.

[13] `run_scripts/run_grpo_rec.sh`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/run_scripts/run_grpo_rec.sh. Fetched 2026-08-10.

[14] `src/open-r1-multimodal/local_scripts/zero2.json`, referenced by `run_scripts/run_grpo_rec_lora.sh` and `run_scripts/run_grpo_rec_more_params.sh`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/src/open-r1-multimodal/local_scripts/zero2.json. Fetched 2026-08-10.

[15] `run_scripts/multinode_training_demo.sh`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/run_scripts/multinode_training_demo.sh. Fetched 2026-08-10.

[16] `run_scripts/multinode_training_args.yaml`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/run_scripts/multinode_training_args.yaml. Fetched 2026-08-10.

[17] `setup.sh`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/setup.sh. Fetched 2026-08-10.

[18] `src/open-r1-multimodal/setup.py` at commit 90052f478646e38bc67363862f699b7afca6b337 (newest push). https://raw.githubusercontent.com/om-ai-lab/VLM-R1/90052f478646e38bc67363862f699b7afca6b337/src/open-r1-multimodal/setup.py. Fetched 2026-08-10.

[19] `src/open-r1-multimodal/setup.py` at tag v0.2.1, commit 3c04653c2d228c00babffc17d2fa468ed7309382. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/v0.2.1/src/open-r1-multimodal/setup.py. Fetched 2026-08-10.

[20] om-ai-lab/VLM-R1 repository metadata API (stars, license, pushed_at, created_at). https://api.github.com/repos/om-ai-lab/VLM-R1. Fetched 2026-08-10.

[21] VLM-R1 README, "For your own data" section. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/README.md. Fetched 2026-08-10.

[22] `src/open-r1-multimodal/local_scripts/zero3.json`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/src/open-r1-multimodal/local_scripts/zero3.json. Fetched 2026-08-10.

[23] `run_scripts/run_grpo_rec_lora.sh`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/run_scripts/run_grpo_rec_lora.sh. Fetched 2026-08-10.

[24] `src/open-r1-multimodal/src/open_r1/trainer/grpo_config.py` (`GRPOConfig` field defaults, `log_completions` docstring). https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/src/open-r1-multimodal/src/open_r1/trainer/grpo_config.py. Fetched 2026-08-10.

[25] `run_scripts/run_grpo_rec_more_params.sh`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/run_scripts/run_grpo_rec_more_params.sh. Fetched 2026-08-10.

[26] Issue #7, "Larger Model" (closed), comment by contributor SZhanZ. https://github.com/om-ai-lab/VLM-R1/issues/7. Fetched 2026-08-10.

[27] Issue #74, "显存使用问题" [GPU-memory usage question] (closed), comment by contributor xrc10. https://github.com/om-ai-lab/VLM-R1/issues/74. Fetched 2026-08-10.

[28] Issue #77, "The training time is very long." (closed), comments by contributor xrc10. https://github.com/om-ai-lab/VLM-R1/issues/77. Fetched 2026-08-10.

[29] Issue #93, "How many GPUs are needed to train a 7B model" (closed). https://github.com/om-ai-lab/VLM-R1/issues/93. Fetched 2026-08-10.

[30] `src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py`, `_metrics` accumulation and `log()` override. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py. Fetched 2026-08-10.

[31] `src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`, reward-function `DEBUG_MODE`/`LOG_PATH` logging blocks. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py. Fetched 2026-08-10.

[32] VLM-R1 README, Evaluation section. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/README.md. Fetched 2026-08-10.

[33] `src/open-r1-multimodal/src/open_r1/grpo_rec.py`, `get_peft_config`/`peft_config` wiring into `VLMGRPOTrainer`. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/src/open-r1-multimodal/src/open_r1/grpo_rec.py. Fetched 2026-08-10.

[34] VLM-R1 README, Models table and dataset links. https://raw.githubusercontent.com/om-ai-lab/VLM-R1/main/README.md. Fetched 2026-08-10.
