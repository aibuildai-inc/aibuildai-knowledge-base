# MedicalGPT

A shell-script-driven, multi-stage post-training pipeline built on `transformers`/`peft`/`trl`, aimed at reproducing the full ChatGPT-style training recipe (PT to SFT to RM/RLHF/DPO/ORPO to standalone OPD distillation) on medical and general LLMs.

The GitHub API lists its description as "MedicalGPT: Training Your Own Medical GPT Model with ChatGPT Training Pipeline", naming "增量预训练(PT)、有监督微调(SFT)、RLHF、DPO、ORPO、GRPO" (continued pretraining, supervised fine-tuning, RLHF, DPO, ORPO, GRPO) [2]. Its README states the same pipeline in its own words: "training medical GPT model with ChatGPT training pipeline, implemantation of Pretraining, Supervised Finetuning, RLHF(Reward Modeling and Reinforcement Learning), DPO(Direct Preference Optimization) and standalone OPD" [1]. It is built and maintained by a single GitHub user, shibing624 [2][3]. It lives at https://github.com/shibing624/MedicalGPT [3]. The API is not a Python library with an importable trainer class; each stage is a standalone Python script under `training/` invoked through a matching `scripts/run_*.sh` shell wrapper that sets `CUDA_VISIBLE_DEVICES` and passes `argparse`/`HfArgumentParser` CLI flags, most stages wrapping a `trl` trainer class internally [4][5].

**When to pick it**: pick MedicalGPT when you want a runnable, opinionated reference pipeline that chains PT to SFT to preference-optimization to distillation as one directory of scripts, with medical-domain example data and five released Chinese-medical checkpoints already on the Hub, rather than a general-purpose trainer library you import into your own code [1]. It is a thin orchestration layer over `transformers`, `peft`, and `trl` (Hugging Face's post-training library, covered on its own card) — every online/preference method here except reward modeling directly instantiates a `trl` trainer class, so pick `trl` directly instead when you don't need MedicalGPT's medical data, its multi-stage shell-script wiring, or its LoRA-merge-between-stages workflow [4][5][6][7].

**Methods it ships**, matching the README's own stage grouping, where stage 3 itself lists RLHF (with RM and RL as its two named steps), DPO, ORPO, and OPD as four alternatives [1]:
- Stage 1, PT (Continue Pretraining, optional): `training/pretraining.py` via `scripts/run_pt.sh`. Reading the script itself (not just the README) shows it trains with a `SavePeftModelTrainer(Trainer)` subclass of plain `transformers.Trainer`, whose only override is `save_model` [8].
- Stage 2, SFT: `training/supervised_finetuning.py` via `scripts/run_sft.sh`, including an Agent/tool-call fine-tuning mode selected with `--tool_format {default,qwen,qwen3,qwen3.5,glm4,llama3,mistral}` [1].
- Stage 3, the README's own four alternatives, only one normally run per checkpoint lineage [1][9]:
  - RLHF, two steps: RM (Reward Modeling), `training/reward_modeling.py` via `scripts/run_rm.sh` — MedicalGPT's own `RewardTrainer(Trainer)` subclass over plain `transformers.Trainer` and `AutoModelForSequenceClassification` — not `trl`'s `RewardTrainer` — implementing the InstructGPT pairwise log-sigmoid loss, quoted from its own docstring: "Define how to compute the reward loss. Use the InstructGPT pairwise logloss" [10]. Then RL, `training/ppo_training.py` via `scripts/run_ppo.sh`: despite the file and script names, its own module docstring states it trains "using RLOO (REINFORCE Leave-One-Out, PPO alternative)", and it instantiates `trl.RLOOTrainer`/`RLOOConfig` directly, with a code comment noting "RLOO does not need a separate value model or ref model (unlike PPO)" [11].
  - DPO: `training/dpo_training.py` via `scripts/run_dpo.sh`, instantiating `trl.DPOTrainer`/`DPOConfig` directly [12].
  - ORPO: `training/orpo_training.py` via `scripts/run_orpo.sh`. There is no separate ORPOTrainer class in this repo; it imports `trl.DPOTrainer`/`DPOConfig` and sets `loss_type="orpo"` with `beta=args.orpo_beta`, i.e. ORPO is wired as a DPO loss variant [13].
  - OPD (On-Policy Distillation, added v2.7.0, 2026-04-20 per the release list): `training/opd_training.py` via `scripts/run_opd.sh`, a standalone student/teacher distillation pipeline built on `trl.experimental.gkd.GKDTrainer`/`GKDConfig` — the experimental import path `trl.experimental.gkd` is used verbatim in the source [15][16]. The v2.7.0 release notes state OPD v1 is a standalone training path that does not modify the existing PPO/GRPO main flow, and the repo's own training-parameters notes describe the same default: only the student is trained, the teacher stays frozen ("OPD建议默认只训练student，teacher保持冻结") [16][17].
- GRPO (added v2.4, 2025-04-18 per the README changelog): `training/grpo_training.py` via `scripts/run_grpo.sh`, instantiating `trl.GRPOTrainer`/`GRPOConfig` directly [1][14]. The README's prose stage-3 bullets (RLHF/DPO/ORPO/OPD) do not mention GRPO, but its project-structure tree comments `grpo_training.py` as "Stage 3: GRPO" and its architecture-directory table lists the stage-3 training flow as "PT→SFT→RM→PPO/DPO/ORPO/GRPO" — the README is internally inconsistent about whether GRPO is a fifth stage-3 alternative or a separate addition, and this card follows the tree comment in grouping it with stage 3 [1].
Method definitions, math, and training-signal semantics are not restated here; they live on each method's own methodology card and on `trl`'s card for the trainer classes MedicalGPT wraps.

**Scale it handles**: single GPU (drop `torchrun` and run the Python script directly, or set `--nproc_per_node 1`) up to multi-node, documented as raw `torchrun --nproc_per_node N --nnodes N --master_addr ADDR --master_port PORT --node_rank RANK`, run once per machine with a different `node_rank` — mechanism only, no published multi-node benchmark [17]. Sharding is via DeepSpeed ZeRO stage 2 or 3 config JSON files passed as `--deepspeed zero2.json`/`zero3.json`; the docs state ZeRO-3 is slower but shards parameters for lower memory, and ZeRO-3 specifically wraps MoE leaf modules (Mixtral, Qwen3MoE, Qwen3.5MoE) via `deepspeed.utils.set_z3_leaf_modules` in `grpo_training.py` [17][14]. One documented launcher gap: `scripts/run_rm.sh` carries the inline comment "reward model 训练暂不支持 torchrun 多卡训练" (reward-model training does not currently support multi-GPU `torchrun`) and its own wrapper uses plain `CUDA_VISIBLE_DEVICES=0,1 python3` instead [18]. QLoRA (`--qlora True --load_in_4bit True`, nf4) is documented as usable on RTX4090/A100/H100 with `--torch_dtype bfloat16 --optim paged_adamw_32bit` recommended for precision [17].

**Install**: `git clone https://github.com/shibing624/MedicalGPT.git && pip install -r requirements.txt --upgrade`; the README's own badge states Python 3.8+ [1]. Licence is Apache-2.0 per the repository's `LICENSE` file and GitHub metadata [2][19], but the repo's separate `DISCLAIMER` file states the project is for research only and "strictly prohibited from using them for any commercial purposes" [20] — an explicit tension between the code licence and the project's own usage terms; read the DISCLAIMER before any commercial use. At `requirements.txt` on the shortlist-pinned commit `ccc05f4b46442ecdefcc95d53aeedc9834d09dd6` (which this repository's commit history confirms is also the current `main` branch head, i.e. a moving target, not a tag) [21][22]: `transformers>=5.6.0`, `trl>=0.29.0`, `peft>=0.19.1`, `accelerate`, `datasets>=2.14.6`, `math-verify==0.5.2` — torch is not listed and arrives transitively [23]. There is no tagged release matching this commit: the latest tag is v2.7.0, published 2026-04-20, whose own `requirements.txt` pins a lower floor — `transformers>=5.1.0`, `peft>=0.14.0` — than the commit above, so anyone installing from the v2.7.0 release tag gets older dependency floors than anyone installing from `main` at the pinned commit [24][25]. No CUDA or hardware minimum is stated anywhere in the README or requirements files read for this card; FlashAttention-2 support is scoped to RTX3090/RTX4090/A100/H100 [1][17].

**Maintained by**: a single maintainer, GitHub user shibing624 [2]. Live-repo signs of activity as of this card's fetch: 5699 stars (not a ranking signal) [2], `pushed_at` 2026-06-03T03:39:57Z [2], and a dated README changelog running from v0.2 (2023-06-05) through v2.7.0 (2026-04-20, OPD added) [1].

## Quick start

There is no importable trainer API; the smallest complete run is the shell wrapper over each stage's script. The repo's shipped `scripts/run_sft.sh` and `scripts/run_dpo.sh` are reproduced here with their non-essential flags (batch-size tuning, LoRA rank, logging cadence, dtype, etc.) trimmed for readability — this is an abbreviated excerpt, not a verbatim copy; see [26] for the full flag list of each script [26]. SFT, core flags:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --train_file_dir ./data/sft \
    --validation_file_dir ./data/sft \
    --do_train --do_eval --use_peft True \
    --output_dir outputs-sft-qwen-v1 \
    --report_to tensorboard
```

DPO, core flags (no `torchrun`, single process over two visible GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 training/dpo_training.py \
    --model_name_or_path Qwen/Qwen3.5-2B \
    --train_file_dir ./data/reward \
    --do_train --do_eval --use_peft True \
    --output_dir outputs-dpo-qwen-v1 \
    --report_to tensorboard
```

The README also links two ready Colab notebooks that run a complete pipeline end to end: `run_training_dpo_pipeline.ipynb` (about 15 minutes) and `run_training_ppo_pipeline.ipynb`, i.e. the RLOO script above (about 20 minutes) [1].

## Start it

- One GPU: drop `torchrun` and call the Python script directly (`python3 training/supervised_finetuning.py ...`), or keep `torchrun` with `--nproc_per_node 1` [17].
- Multiple GPUs, one machine: the shipped `scripts/run_*.sh` wrappers set `CUDA_VISIBLE_DEVICES=0,1` and, for PT/SFT/GRPO/OPD, use `torchrun --nproc_per_node 2`; RM/RLOO("ppo")/DPO/ORPO instead run plain `python3` with no `torchrun`, i.e. those four scripts as shipped do not do multi-GPU data parallel training out of the box [18].
- Multiple machines: raw `torchrun --nproc_per_node 8 --nnodes 2 --master_addr <ip> --master_port 14545 --node_rank <0|1> training/supervised_finetuning.py ...`, run once per machine with the matching `node_rank`; no config template file is shipped for this, it is a documented command pattern only [17].
- DeepSpeed sharding is opt-in per run via `--deepspeed zero2.json` or `--deepspeed zero3.json`, config files shipped in `scripts/` [17][27].
- Effective batch size is `per_device_train_batch_size x visible GPUs x gradient_accumulation_steps`, standard `transformers.TrainingArguments` arithmetic; the shipped example scripts use different per-device/accumulation values per stage (e.g. SFT ships `--per_device_train_batch_size 2 --gradient_accumulation_steps 8`; GRPO ships `--per_device_train_batch_size 4 --gradient_accumulation_steps 1`) [28][14].
- Every stage's Config surface is that method's underlying class (`transformers.TrainingArguments` for PT/SFT/RM, `trl`'s `RLOOConfig`/`DPOConfig`/`GRPOConfig`/`GKDConfig` for the rest), extended with MedicalGPT-specific fields such as `--tool_format`, `--target_modules`, `--lora_rank`; MedicalGPT's example scripts default to `--use_peft True` (LoRA) rather than full fine-tuning, and the training-parameters notes state full-parameter LLaMA-7B training needs about 120GB VRAM versus about 13GB for LoRA [17][14][11]. GRPO's own example script sets `--bf16 True` and a low `--learning_rate 5.0e-7`, both worth checking against your hardware and base model rather than assumed [14].
- Out-of-memory first aid documented in the training-parameters notes: pass `--load_in_4bit True` or `--load_in_8bit True` for quantized training, or `--deepspeed zero3.json --fp16` when VRAM is tight; the notes state the `NotImplementedError: Cannot copy out of meta tensor; no data!` error in the FAQ is itself a single-GPU-OOM symptom (`device_map='auto'` falling back to CPU offload), fixed by training on multiple visible GPUs instead [17][29]. Generation-side OOM knobs for the online methods (GRPO's `--num_generations`, `--max_completion_length`; RLOO's `--max_completion_length`) are the standard way to shrink the rollout batch when a run OOMs during sampling [14][11].

## Watch it

This section is the mechanics only; what a metric MEANS for a given method lives on that method's own methodology card, and for the four stages that wrap `trl` trainers directly (RLOO, DPO, ORPO, GRPO), the metric names themselves are `trl`'s and are documented on `trl`'s card, not repeated here [11][12][13][14].

- Logging is enabled per run with `--report_to tensorboard` in every shipped example script; MedicalGPT does not publish its own default logging destination beyond what `transformers.TrainingArguments`/`trl`'s Config classes default to, and no example script in this repo enables a hosted tracker such as Weights & Biases [17][14][11][12][13].
- The repo's own save/output notes state `trainer_state.json`, written under each stage's `output_dir`, records the loss and learning-rate history, and that the `logs/` subdirectory holds TensorBoard event files, started with `tensorboard --logdir output_dir/logs --host 0.0.0.0 --port 8008` [17].
- PT, SFT, and RM (the three stages built on `transformers.Trainer` subclasses, e.g. PT's `SavePeftModelTrainer` and RM's `RewardTrainer`) log whatever `transformers.Trainer`'s standard `logging_steps`-driven loop reports (loss, learning rate, etc.); this card does not enumerate that list, since it is transformers' own surface, not MedicalGPT's [8][30][10].
- RLOO, DPO, ORPO, and GRPO log through `trl`'s trainer classes and use `trl`'s own metric names for those classes (`RLOOTrainer`/`DPOTrainer`/`GRPOTrainer`) — read `trl`'s card and its live docs pages, not this one, for the exact list per trainer [11][12][13][14].
- No library-published RL-specific stopping rule, threshold, or health limit was found for any MedicalGPT stage in the files read for this card (`training/*.py`, the wiki `training_params.md` page, `docs/FAQ.md`); every `--eval_steps`/`--save_steps`/`--max_steps` field seen is a fixed schedule, not an adaptive stopping condition [17][29][11][12][13][14].
- Evaluation during training is enabled per run with `--do_eval --eval_strategy steps --eval_steps N`, shipped in every example script (e.g. SFT: `--eval_steps 50`; DPO: `--eval_steps 20`) [28][12].

## Save it

- Every stage writes to its own `--output_dir`. The repo's own layout example shows: `adapter_config.json` and `adapter_model.bin` at the top level (a LoRA run, the default mode), numbered `checkpoint-<step>/` subdirectories each containing their own `adapter_config.json`/`adapter_model.bin`/`trainer_state.json`/`training_args.bin`, plus `train_results.txt`, `eval_results.txt`, `config.json`, tokenizer files, and a `logs/` directory of TensorBoard event files [17].
- Retention/cadence flags are the standard `transformers.TrainingArguments`-family `--save_steps`/`--save_strategy`/`--save_total_limit`, set per example script (e.g. RM ships `--save_total_limit 3`, SFT ships `--save_total_limit 13`) [28][18]; no MedicalGPT-specific flag that silently drops optimizer state was found in the files read for this card.
- Because `--use_peft True` is the shipped default, what lands on disk after each stage is a LoRA adapter, not a full model: `adapter_config.json` + `adapter_model.bin`, not a full model directory [17]. The repo's own notes state this explicitly and require merging the adapter into the base model before the next pipeline stage or before deployment: `python merge_peft_adapter.py --base_model <dir> --tokenizer_path <dir> --lora_model <lora_dir> --output_dir outputs-merged`, requiring `peft>=0.4.0`; the merged weights are then "可通过from_pretrained直接加载" (directly loadable via `from_pretrained`) [17]. Under the hood the merge script calls `peft.PeftModel.from_pretrained(base_model, lora_model_path)` then `.merge_and_unload()` and saves the result [31].
- The v2.7.0 release notes and the training-parameters notes both state OPD's LoRA output is saved and merged the same way as SFT/DPO ("OPD的LoRA输出与SFT/DPO相同，也可以用同样的方式合并和部署") — OPD saves only the student, since the teacher is frozen and never updated [16][17].
- Resume-from-checkpoint is documented as two different flags depending on training mode: for LoRA runs pass `--peft_path <old adapter dir>`; for full-parameter runs pass `--resume_from_checkpoint <old checkpoint dir>` [17].
- Loader handoff: an adapter-only `output_dir` is not directly loadable as a full model by a plain `from_pretrained()` call — it needs pairing with the base model (either at merge time via `merge_peft_adapter.py`, or at load time with a PEFT-aware loader); only a merged `outputs-merged` directory is a full model directly loadable with `from_pretrained` [17][31].

## Find it in the docs

The GitHub repository itself is the canonical documentation surface; there is no separate hosted docs site.

- Repo home: `https://github.com/shibing624/MedicalGPT` [2]. The bilingual `README.md` (Chinese) and `README_EN.md` (English) at repo root are the primary narrative docs, covering install, the pipeline table, hardware/VRAM sizing, supported base models, and dataset links [1].
- `docs/FAQ.md` in the repo holds maintainer-answered troubleshooting entries in a Q/A format; three entries were read for this card [29].
- The GitHub Wiki (`https://github.com/shibing624/MedicalGPT/wiki`, Chinese page titles, URL-encoded, e.g. `.../wiki/%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E` for "训练参数说明") mirrors and extends README content; that training-parameters page is the source for this card's launcher, save-layout, DeepSpeed, and multi-node sections [17].
- `DISCLAIMER` at repo root (also linked from the README) is the usage-restriction document, separate from the Apache-2.0 `LICENSE` file — read it before assuming the licence alone governs allowed use [20][2].
- Runnable references beyond the docs: the `scripts/` directory's `run_*.sh` files are working, parameterized examples for every stage [18]; `notebooks/` on the repo holds the two Colab pipeline notebooks named in Quick start [1]; `data/` ships small example datasets per stage (`data/pretrain`, `data/sft`, `data/reward`, `data/grpo`) that the example scripts point at directly, usable as smoke-test data [1][28][14].
- Known trap: in closed issue #107, asking whether ChatGLM/Baichuan can do RM and RL training, maintainer shibing624 (repository owner, author association OWNER) replied on 2023-07-23: "是，chatglm不是标准CausalLM" (yes, ChatGLM is not a standard CausalLM) [32]. `docs/FAQ.md` records the fuller, standing version of the same limitation: RM training needs `AutoModelForSequenceClassification`, which ChatGLM does not implement, and PPO training needs `AutoModelForCausalLMWithValueHead`, which ChatGLM also does not support — and the FAQ states the same applies to Baichuan for the same reason [29]. The FAQ's `AutoModelForCausalLMWithValueHead` requirement describes the trainer that predates the repo's later switch to `trl.RLOOTrainer` for the RL stage: the current `training/ppo_training.py` uses RLOO, whose own code comment states it "does not need a separate value model or ref model (unlike PPO)" [11], so this FAQ entry's stated cause for the RL-side limitation is stale for the current RLOO-based implementation even though the RM-side limitation (no `AutoModelForSequenceClassification`) still applies as written.
- No official MCP endpoint for these docs was found or searched for beyond this repo; none is claimed by the repository.

## Sources

All GitHub repo files (README, docs/FAQ.md, source under `training/`, etc.) are commit- or tag-pinned as stated per claim, since raw.githubusercontent.com takes a revision parameter; the GitHub Wiki genuinely has no revision parameter and its pages are cited as unpinned live pages read on the date given, and GitHub-API metadata endpoints (repo, issue, commit lookups) are likewise live and cited by fetch date only.

[1] MedicalGPT README.md, main branch. https://raw.githubusercontent.com/shibing624/MedicalGPT/main/README.md. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[2] MedicalGPT GitHub repository metadata (GitHub API). https://api.github.com/repos/shibing624/MedicalGPT. Fetched 2026-08-10.

[3] MedicalGPT repository home page. https://github.com/shibing624/MedicalGPT. Fetched 2026-08-10.

[4] trl GRPOTrainer / RLOOTrainer / DPOTrainer / GKDTrainer classes, as imported by MedicalGPT's training scripts (see [11][12][13][14][15] for the exact import lines) — see trl's own card for trl's documentation, not mirrored here.

[5] trl documentation index. https://huggingface.co/docs/trl/main/en/index (cited for the fact that trl is Hugging Face's post-training trainer-class library; see trl's own card for detail). Fetched 2026-08-10.

[6] MedicalGPT training/dpo_training.py, main branch (trl DPOTrainer/DPOConfig import). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/training/dpo_training.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[7] MedicalGPT training/grpo_training.py, main branch (trl GRPOTrainer/GRPOConfig import). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/training/grpo_training.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[8] MedicalGPT training/pretraining.py, main branch (`SavePeftModelTrainer(Trainer)` subclass, `save_model` override). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/training/pretraining.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[9] MedicalGPT README.md Features section (Stage 3 as four alternatives: RLHF(RM+RL)/DPO/ORPO/OPD). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/README.md. Fetched 2026-08-10.

[10] MedicalGPT training/reward_modeling.py (custom RewardTrainer class, `AutoModelForSequenceClassification`, InstructGPT pairwise logloss docstring), main branch. https://raw.githubusercontent.com/shibing624/MedicalGPT/main/training/reward_modeling.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[11] MedicalGPT training/ppo_training.py (module docstring naming RLOO, `trl.RLOOTrainer`/`RLOOConfig` import and instantiation), main branch. https://raw.githubusercontent.com/shibing624/MedicalGPT/main/training/ppo_training.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[12] MedicalGPT training/dpo_training.py (same file as [6]; cited separately for the DPOTrainer instantiation and DPOConfig fields). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/training/dpo_training.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[13] MedicalGPT training/orpo_training.py (`trl.DPOTrainer`/`DPOConfig` with `loss_type="orpo"`, `beta=args.orpo_beta`), main branch. https://raw.githubusercontent.com/shibing624/MedicalGPT/main/training/orpo_training.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[14] MedicalGPT training/grpo_training.py (same file as [7]; cited separately for GRPOTrainer instantiation, DeepSpeed ZeRO-3 MoE leaf-module handling, and example-script fields). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/training/grpo_training.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[15] MedicalGPT training/opd_training.py (`from trl.experimental.gkd import GKDConfig, GKDTrainer`), main branch. https://raw.githubusercontent.com/shibing624/MedicalGPT/main/training/opd_training.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[16] MedicalGPT GitHub Releases, v2.7.0 release body (OPD v1 standalone pipeline statement) and v2.5.0 release body. https://api.github.com/repos/shibing624/MedicalGPT/releases. Fetched 2026-08-10.

[17] MedicalGPT GitHub Wiki, training-parameters page ("训练参数说明" / training-parameters notes: launcher forms, save-directory layout, DeepSpeed guidance, multi-node command, LoRA merge workflow, resume-from-checkpoint flags, QLoRA/FlashAttention/OOM guidance). https://github.com/shibing624/MedicalGPT/wiki/%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E. Fetched 2026-08-10.

[18] MedicalGPT scripts/run_rm.sh (no-torchrun comment and plain CUDA_VISIBLE_DEVICES launch), main branch, and the sibling run_pt.sh/run_sft.sh/run_grpo.sh/run_opd.sh (torchrun launch) for contrast. https://raw.githubusercontent.com/shibing624/MedicalGPT/main/scripts/run_rm.sh. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[19] MedicalGPT LICENSE file, main branch. https://raw.githubusercontent.com/shibing624/MedicalGPT/main/LICENSE (existence and Apache-2.0 text confirmed via repository root listing and GitHub API license field, [2]). Fetched 2026-08-10.

[20] MedicalGPT DISCLAIMER file, main branch (commercial-use prohibition). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/DISCLAIMER. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[21] MedicalGPT commit history, main branch, most recent 5 commits (GitHub API), confirming commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6 is the current branch head. https://api.github.com/repos/shibing624/MedicalGPT/commits?sha=main. Fetched 2026-08-10.

[22] Shortlist row for shibing624/MedicalGPT (pinned commit field), supplied to this card-writing task.

[23] MedicalGPT requirements.txt, main branch. https://raw.githubusercontent.com/shibing624/MedicalGPT/main/requirements.txt. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[24] MedicalGPT requirements.txt, tag 2.7.0. https://raw.githubusercontent.com/shibing624/MedicalGPT/2.7.0/requirements.txt. Fetched 2026-08-10, at the v2.7.0 tag.

[25] MedicalGPT GitHub Releases, v2.7.0 entry (tag name, publish date 2026-04-20). https://api.github.com/repos/shibing624/MedicalGPT/releases. Fetched 2026-08-10.

[26] MedicalGPT scripts/ directory, main branch (run_*.sh wrappers as the quickstart forms). https://github.com/shibing624/MedicalGPT/tree/main/scripts. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[27] MedicalGPT scripts/zero2.json (DeepSpeed ZeRO stage 2 config file, existence confirmed). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/scripts/zero2.json. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[28] MedicalGPT scripts/run_sft.sh and scripts/run_dpo.sh (example-script batch-size, accumulation, and eval fields), main branch. https://raw.githubusercontent.com/shibing624/MedicalGPT/main/scripts/run_sft.sh. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[29] MedicalGPT docs/FAQ.md, main branch (three Q/A entries: single-GPU meta-tensor OOM error, ChatGLM/Baichuan LoRA-merge issue, ChatGLM/Baichuan RM+RL limitation). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/docs/FAQ.md. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[30] transformers.Trainer default logging surface, referenced only as the base class PT/SFT/RM run on top of; see trl's card, which documents this surface, rather than repeating it here.

[31] MedicalGPT tools/merge_peft_adapter.py, main branch (`PeftModel.from_pretrained` then `merge_and_unload` then save). https://raw.githubusercontent.com/shibing624/MedicalGPT/main/tools/merge_peft_adapter.py. Fetched 2026-08-10, at commit ccc05f4b46442ecdefcc95d53aeedc9834d09dd6.

[32] MedicalGPT GitHub issue #107 comments (GitHub API), maintainer shibing624 (association: OWNER) reply dated 2023-07-23T01:49:00Z. https://api.github.com/repos/shibing624/MedicalGPT/issues/107/comments. Fetched 2026-08-10.
