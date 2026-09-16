# align-anything

PKU-Alignment's multi-modal post-training library: one shared trainer/config pattern across eight input-output modality pairs, launched through DeepSpeed shell scripts rather than a Python trainer API.

Align-Anything "aims to align any modality large models (any-to-any models) with human intentions and values" [1]. It is built and maintained by the PKU-Alignment Team [1][2], and its shape is a set of per-modality, per-method trainer scripts (`align_anything/trainers/<modality>/<method>.py`) launched with `deepspeed --module`, each reading a default YAML config from `align_anything/configs/train/<modality>/<method>.yaml` that CLI flags override [3][4]. It lives at https://github.com/PKU-Alignment/align-anything [2].

**When to pick it**: alignment training that has to span modalities beyond text. The README's own algorithm table lists nine modality pairs sharing one trainer/config pattern - text-to-text, text+image-to-text, text+image-to-text+image, text+audio-to-text, text+video-to-text, text-to-image, text-to-video, text-to-audio, and text+video-to-action - though for the last three of those (text-to-image, text-to-video, text-to-audio) only SFT and DPO are checked off, with RM and PPO marked as under construction, and text+video-to-action has only SFT checked [1]. trl and verl cover text-to-text only (cross-reference; not covered here). Within text-to-text, align-anything also carries the widest method list found for that modality: SFT, RM, DPO, PPO, GRPO, KTO, ORPO, and SimPO trainer files all exist in the repo [6]. The cost is operational: there is no accelerate/Ray-style single launcher abstraction - every run is a `deepspeed --module` invocation naming the exact trainer module path [4], and, as detailed under Install below, the project ships no PyPI release and no GitHub release/tag, so pinning it means pinning a commit [7][8][9].

**Methods it ships** (grouped by modality, from README's own algorithm table [1] and the repository's trainer directory tree at the pinned commit [6]; method definitions and math live on each method's own card, not here): README's algorithm table names SFT, RM (reward model), DPO, PPO as the cross-modality baseline, but marks it complete only for the four text-and-image/audio/video-to-text pairs; for text-to-image, text-to-video, and text-to-audio only SFT and DPO are checked, with RM and PPO shown as under construction, and text+video-to-action has only SFT checked [1]. text_to_text additionally carries GRPO, KTO, ORPO, and SimPO trainer files, plus `multi_ppo.py` (multi-reward-model PPO), `ppo_remote_rm.py` (reward model served remotely rather than co-located) and `ppo_vllm.py` (vLLM-accelerated generation for PPO) [6]. GRPO and ORPO are not named in the README's algorithm table or the docs training-overview page - both are established here from the trainer file tree and their config files, not from any prose description [6]. The README's dated news log records GRPO support being added on 2025-02-19 and a vLLM-backed PPO speedup (about 150 to 22 minutes per iteration on their benchmark) on 2025-03-31 [1].

**Scale it handles**: single GPU up to single-node multi-GPU via `deepspeed --module <trainer> <args>`, with ZeRO stage selected by the `train_cfgs.ds_cfgs` YAML key pointing at a DeepSpeed JSON config (`ds_z0_config.json` through `ds_z3_offload_config.json`) [4][6]. Slurm is supported by wrapping the same shell script in `srun`, and the repository's own example (`scripts/slurm/slurm_llava_dpo.sh`) requests `--nodes=1 --gres=gpu:8`, so the published example is single-node - the docs read for this card show no multi-node worked example or hostfile guidance, so treat multi-node as undocumented rather than absent [1][10].

**Install**: no PyPI release exists for this project as the install path - a package literally named `align-anything` does exist on PyPI with `author`/`author_email` set to `PKU-Alignment`/`jmji@zju.edu.cn`, the same org name as the GitHub org, but its metadata is otherwise a placeholder: summary "A small example package", MIT-licensed (the repo is Apache-2.0), version 0.0.1b0, uploaded once on 2024-06-17, no `home_page` or `project_urls` set [11]. There are also no GitHub Releases and no git tags on this repository (both list endpoints return empty) [8][9], so the only way to pin the library is a commit; this card pins `3f9decc221be74b2052e712e3a32e155686ec6ec` [2]. The install path is `git clone` then `pip3 install -e .`, with `pip3 install vllm==0.7.2` added "to run ppo on vllm engine" [1]. `pyproject.toml` at that commit sets `requires-python >= 3.10` and declares Apache-2.0 [7]; its own version file reports an unreleased dev version, `0.0.1.dev0` [14]. The base dependency list pins nothing: `torch`, `torchvision`, `torchaudio`, `accelerate`, `deepspeed`, `wandb`, `peft` are all unpinned, and the one floor given is `transformers >= 4.50.0` [7]. That floor collides with one optional extra: `pip install -e .[minicpmv]` pins `timm==1.0.11` and `transformers==4.40.0`, below the base `>=4.50.0` requirement, so installing that extra downgrades transformers against the base spec; the sibling `.[minicpmo]` extra pins different packages (`vector-quantize-pytorch`, `vocos`, `decord`, `moviepy`) and carries no transformers pin at all [7]. A separate `.[ascend]` extra (Huawei NPU) pins `torch==2.1.0`, `torchvision==0.16.0`, `torchaudio==2.1.0`, `numpy==1.26.0`, `torch-npu==2.1.0.post10` [7], and its documented test environment is Python 3.10.6, CANN 8.0.rc3, aarch64, 8x Ascend-SNT9B [1]. No CUDA minimum is stated anywhere in the docs read for this card; the README's Nvidia path only recommends CUDA 12.2.0 because that is what the maintainers' own H800 cluster used, not a floor [1].

**Maintained by**: the PKU-Alignment Team [1][2]; the GitHub API reports 4,664 stargazers and a most recent push of 2025-11-27, both live counts as of the fetch date rather than a trend [2]. The README's dated news log shows activity into 2025-11-11, including added Qwen3/Qwen3-MoE support and integration of a separate `eval-anything` evaluation project into the repository [1].

## Quick start

The README's quickstart is the install sequence above, then a per-modality shell script rather than a Python snippet [1]:

```bash
git clone git@github.com:PKU-Alignment/align-anything.git
cd align-anything
conda create -n align-anything python==3.11
conda activate align-anything
pip3 install -e .
pip3 install vllm==0.7.2  # to run ppo on vllm engine

cd scripts
bash llava/llava_dpo.sh
```

`scripts/llava/llava_dpo.sh` at the pinned commit is the complete script that command runs - a real model and a real Hub dataset, no placeholders [15]:

```bash
MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf"
TRAIN_DATASETS="PKU-Alignment/align-anything"
TRAIN_TEMPLATE="AA_TI2T"
TRAIN_NAME="text-image-to-text"
TRAIN_SPLIT="train"
OUTPUT_DIR="../outputs/llava_dpo"
export WANDB_API_KEY=""
source ./setup.sh
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_image_to_text.dpo \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_split ${TRAIN_SPLIT} \
     --train_name ${TRAIN_NAME} \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 3 \
     --train_batch_size 1 \
     --epochs 2
```

There is no separate CLI entry point - `--module align_anything.trainers.<modality>.<method>` naming the exact trainer file is the invocation form for every run [3][15].

## Start it

- One GPU or several on one node: run the `deepspeed --module ...` command above as-is; `deepspeed` picks up all visible devices, and issue #159's maintainer reply shows restricting devices with `CUDA_VISIBLE_DEVICES=0,1,2,3` before the same command [16].
- Slurm: `sbatch scripts/slurm/slurm_llava_dpo.sh`, which sets `--nodes=1 --gres=gpu:8` and runs `srun llava_dpo.sh` - the same single-script pattern, not a distinct distributed entry point [10].
- `source ./setup.sh` before every launch is required by the scripts above: it derives a free `MASTER_PORT`, creates `OUTPUT_DIR`, and - the one silent default worth flagging - sets `WANDB_MODE=offline` whenever `WANDB_API_KEY` is unset, so an unconfigured run still logs, just not to the cloud [17].
- Configuration is six YAML categories per modality/method file (`align_anything/configs/train/<modality>/<method>.yaml`): `train_cfgs`, `data_cfgs`, `model_cfgs`, `logger_cfgs`, `lora_cfgs`, `bnb_cfgs`, each overridable from the CLI as `--<dotted_or_flat_key> <value>` the way the quickstart script does [18][4]. `train_cfgs.bf16` defaults to `True` across the SFT, DPO, GRPO, and PPO configs read for this card, a silent bf16-capable-GPU assumption [4][19][20]. Sharding is chosen with `train_cfgs.ds_cfgs`, a filename under `align_anything/configs/deepspeed/` (`ds_z0_config.json` through the `_offload` ZeRO-2/3 variants) [4][6].
- Effective batch size is `per_device_train_batch_size x num_gpus x gradient_accumulation_steps`, the DeepSpeed-standard product; PPO's config additionally separates `per_device_prompt_batch_size` (generation) from `per_device_train_batch_size` (optimization) [19].
- PPO-specific knobs live in `train_cfgs` of `ppo.yaml`: `kl_coeff: 0.02` (actor-vs-reference KL penalty coefficient), `clip_range_ratio: 0.2`, `clip_range_score: 50.0`, `clip_range_value: 5.0`, `gamma: 1.0`, `gae_lambda: 0.95`, `ptx_coeff: 16.0`, `update_iters: 1` [19]. GRPO's `grpo.yaml` carries its own, differently-scoped KL-like term, `beta: 0.01`, plus `num_generations: 10` [20] - these are not the same quantity as PPO's `kl_coeff` and are not compared here. `ppo_vllm.py` and `ppo_remote_rm.py` are separate trainer modules for vLLM-backed generation and a remote reward-model server respectively, launched the same `--module` way [6].
- Out-of-memory first aid: no dedicated OOM guide was found in the docs pages read for this card; the one documented data point is a maintainer reply on issue #159 (COLLABORATOR reply posted 2025-03-17, issue closed 2025-03-18) stating that fine-tuning Baichuan-M1-14B needs at least 4x A100-80G and that a single-GPU OOM at batch size 1 is expected for that model size, with a fix of adding `CUDA_VISIBLE_DEVICES` for more GPUs rather than any config flag [16]. Switching `train_cfgs.ds_cfgs` to one of the `_offload` DeepSpeed configs and lowering `per_device_train_batch_size` are the mechanism-level options exposed by the config surface itself [4][6].
- LoRA/QLoRA are config-only, not separate scripts: `lora_cfgs.use_lora` and `bnb_cfgs.use_bnb` toggle them on the same trainer, with `lora_cfgs.save_full_model` (default `True` in the read configs) deciding whether a save merges the adapter into the base model or keeps it separate - see Save it [19].

## Watch it

This section covers only the logging mechanics; what a logged value should look like for a given method lives on that method's own card.

- Enable it via `logger_cfgs.log_type`, `wandb` or `tensorboard` [18]; `wandb` is the default in every config read for this card [4][19][20]. As noted under Start it, `scripts/setup.sh` forces `WANDB_MODE=offline` when `WANDB_API_KEY` is empty, so an unconfigured run still produces a local wandb log directory rather than no record at all [17].
- No dedicated metrics-reference doc page was found; metric names below come from reading each trainer's own logging calls in `align_anything/trainers/text_to_text/*.py` at the pinned commit [21][22][23][24][25][26].
- **SFT** logs `train/loss`, `train/lr`, and periodically `train/epoch`; its `eval()` computes `eval/loss/<template>` when an `eval_dataset` is configured, or returns nothing if none is set [21].
- **RM** logs `train/loss`, `train/accuracy`, `train/lr`, `train/epoch`; its `eval()` computes preference accuracy over the eval set when configured [22].
- **DPO** logs `train/loss`, `train/reward`, `train/better_sample_reward`, `train/worse_sample_reward`, `train/reward_accuracy`, `train/reward_margin`, `train/lr`, `train/epoch` - but its `eval()` is an unimplemented stub that always returns `{}`, so setting `data_cfgs.eval_datasets` for DPO produces no eval metrics [23].
- **ORPO** subclasses `DPOTrainer` and only overrides the loss computation, not the training-step or logging code, so it emits the same `train/*` key set as DPO under the inherited `train_step` [24][23].
- **KTO** logs the same key set as DPO (`train/loss`, `train/reward`, `train/better_sample_reward`, `train/worse_sample_reward`, `train/reward_accuracy`, `train/reward_margin`, `train/lr`, `train/epoch`) [25].
- **GRPO** logs `train/loss` and `train/reward` only, from its `train_step` return dict [26].
- **PPO** logs the widest set: `train/actor_loss`, `train/reward_critic_loss`, `train/reward`, `train/reward_with_kl_penalty`, `train/reward_advantage`, `train/reward_return`, `train/reward_value`, `train/kl_divergence`, `train/actor_lr`, `train/reward_critic_lr`, `train/mean_generated_length`, `train/max_generated_length`, plus `train/ptx_loss` logged in a separate call when ptx training is enabled [27].
- Evaluation-during-training is a `data_cfgs.eval_datasets`/`eval_template` field plus `train_cfgs.eval_strategy` (`epoch` or `steps`) and `eval_interval`, present on every trainer's config, but the two trainer base classes handle it differently: `SupervisedTrainerBase` (the base for SFT, RM, DPO, and - by inheritance through DPO - KTO and ORPO) defaults `eval()` to an unconditional stub returning `{}`, which SFT and RM override with real evaluation but DPO does not, so DPO/KTO/ORPO produce no eval metrics regardless of `eval_datasets` [23][28]. `RLTrainerBase` (the base for GRPO and PPO, neither of which overrides `eval()`) instead runs real generation-based evaluation whenever `eval_dataloader` is configured, returning `{}` only when it is `None` [26][27][31].
- Sample-level generation logging: no `log_completions`-style flag was found in any of the six text-to-text trainer config files read for this card, and PPO's own `train_step` only logs generation-length statistics (`train/mean_generated_length`, `train/max_generated_length`), not the text itself [19][27]. Text does appear during evaluation, though: `RLTrainerBase.eval()`, inherited by both GRPO and PPO, decodes prompts and generations with the tokenizer and prints them to the console via the shared `Logger.print_table()` helper (a Rich-library table, up to 5 rows, titled "Evaluating...") whenever `eval_dataloader` is configured - a console print, not a value pushed to the `wandb`/`tensorboard` backend selected by `logger_cfgs.log_type` [26][27][31][32].
- Stopping: no published early-stopping threshold, patience value, or RL-specific stopping rule was found in the training-overview or training-configs docs pages, or in any of the six text-to-text YAML configs read [3][18][19][20]. Shapes (loss, reward, KL) are logged; no threshold that ends a run on them is documented.

## Save it

- Checkpoints land under `logger_cfgs.output_dir` as `slice_<step>/` directories, or `slice_end/` for the final save, written by the shared `save_transformers` method every trainer calls [28].
- Inside a `slice_*` directory: `config.json` and tokenizer files always; the model weights as `pytorch_model.bin` (via DeepSpeed's `save_16bit_model`) when the active ZeRO stage is 2 or higher, or via a plain `save_pretrained` otherwise [28]. This is the loadable Hugging Face directory - no separate export step is shown.
- `train_cfgs.save_checkpoint` (not the model-weight save) additionally triggers `model.save_checkpoint(output_dir)`, DeepSpeed's own engine checkpoint (optimizer, scheduler, RNG state) written into the same `slice_*` directory; if this flag is off, only the inference-ready weights are kept and training cannot resume from that directory [28].
- LoRA changes what gets written: with `lora_cfgs.use_lora=True` and `lora_cfgs.save_full_model=False`, the trainer saves adapter weights only via `model.save_pretrained(output_dir)` - an adapter directory, not a full model; with `save_full_model=True` (the default in every config read for this card) it instead calls `merge_and_unload()` before saving, producing a full merged model directory [19][28].
- Resume: set `train_cfgs.load_checkpoint=True` and point `model_cfgs.model_name_or_path` at a prior `slice_<step>` directory - the trainer parses the step number back out of that path name to set its resumed `global_step`, and calls DeepSpeed's `load_checkpoint` on the same directory, so resuming requires the optimizer/scheduler state from `save_checkpoint` having been written at save time [29].
- `logger_cfgs.save_total_limit` caps how many `slice_*` directories are kept, per the field's presence in every config read [19]; the exact rotation behavior (oldest-first deletion) was not directly read from the save code for this card.
- Loader handoff: a full-model `slice_*` directory (SFT/RM/DPO/KTO/GRPO/PPO with `save_full_model=True`, or any non-LoRA run) is a standard Hugging Face model directory an evaluator can load with `from_pretrained` directly; an adapter-only `slice_*` directory (`save_full_model=False`) is not, and needs the base model plus a PEFT-style adapter loader - this card did not verify which loader class align-anything itself uses for reloading an adapter-only save, so treat that path as unverified rather than direct-loadable.

## Find it in the docs

- The docs site is `https://align-anything.readthedocs.io/en/latest/` [5]. Read the Docs' own versions API lists exactly two versions for this project: slug `main` (`active: false`, `built: false`) and slug `latest` (`active: true`, `built: true`), and both resolve to the same underlying git ref, `identifier: "main"`, with a `vcs` URL of `github.com/PKU-Alignment/align-anything/tree/main/` - there is no version-tagged docs URL form the way some libraries offer `/en/v1.2.3/`, matching the absence of any GitHub release or tag [12][8][13]. Anything read from these pages is therefore a live-`main` reading, dated 2026-08-10 for this card, not a pinned-version reading.
- Page slugs seen from the site navigation: `training/overview`, `training/configs`, `training/dataset_custom`, `training/any_to_any`, `data/text_to_text`, `evaluation/overview`, `evaluation/evaluation_configurations`, `evaluation/example`, `tutorial/chameleon` [30]. `training/overview` restates the modality-x-method matrix and describes SFT/DPO/PPO as the "basic alignment algorithms" extended per modality; `training/configs` documents the six YAML categories and gives the CLI-override pattern [3][18].
- Runnable references beyond the docs: the `scripts/` tree in the GitHub repo holds one shell script per modality/method pair (e.g. `scripts/llava/llava_dpo.sh`) plus a parallel `scripts/slurm/` tree of Slurm-wrapped variants, and these are the scripts the README's quickstart runs directly, not illustrative snippets [1][15][10]. The quickstart dataset is the maintainers' own Hub dataset, `PKU-Alignment/align-anything` [15].
- Support and traps: the README points troubleshooting to a GitHub Discussions FAQ thread rather than the docs site [1]. The one concrete trap found in closed issues: issue #159, a single-GPU OOM fine-tuning Baichuan-M1-14B at batch size 1, closed 2025-03-18 after a COLLABORATOR reply the day before that this model needs at least 4 A100-80G GPUs and that the OOM is expected on one GPU, not a bug [16].
- No official MCP endpoint for these docs was found or searched for as part of this card.
- Honest boundary: as covered under Watch it, DPO's `eval()` is a stub that always returns `{}`, so eval-during-training silently produces no metrics for that trainer (and, by the same inherited stub, KTO and ORPO) regardless of `eval_datasets` configuration [23][28]. As covered under Install, there is no PyPI package and no GitHub release/tag to pin against, only a commit [11][8][9]. As covered under Scale it handles, the only published multi-GPU example (Slurm or plain `deepspeed`) is single-node.

## Sources

All GitHub pages are read at commit `3f9decc221be74b2052e712e3a32e155686ec6ec` unless a live API endpoint or the docs site is named; docs-site and PyPI/GitHub-API pages are unpinned live reads, fetched 2026-08-10.

[1] align-anything README. https://github.com/PKU-Alignment/align-anything/blob/3f9decc221be74b2052e712e3a32e155686ec6ec/README.md. Fetched 2026-08-10.

[2] align-anything GitHub repository (repo metadata via API). https://github.com/PKU-Alignment/align-anything ; https://api.github.com/repos/PKU-Alignment/align-anything. Fetched 2026-08-10.

[3] align-anything docs, training overview. https://align-anything.readthedocs.io/en/latest/training/overview.html. Fetched 2026-08-10.

[4] align-anything docs, training configurations. https://align-anything.readthedocs.io/en/latest/training/configs.html. Fetched 2026-08-10.

[5] align-anything docs index (readthedocs landing page and nav). https://align-anything.readthedocs.io/en/latest/. Fetched 2026-08-10.

[6] align-anything repository tree at the pinned commit (trainer and config file listing). https://api.github.com/repos/PKU-Alignment/align-anything/git/trees/3f9decc221be74b2052e712e3a32e155686ec6ec?recursive=1. Fetched 2026-08-10.

[7] align-anything `pyproject.toml` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/pyproject.toml. Fetched 2026-08-10.

[8] align-anything GitHub Releases (empty). https://api.github.com/repos/PKU-Alignment/align-anything/releases. Fetched 2026-08-10.

[9] align-anything git tags (empty). https://api.github.com/repos/PKU-Alignment/align-anything/tags. Fetched 2026-08-10.

[10] `scripts/slurm/slurm_llava_dpo.sh` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/scripts/slurm/slurm_llava_dpo.sh. Fetched 2026-08-10.

[11] PyPI JSON API for the `align-anything` package (placeholder metadata; author field matches the GitHub org name). https://pypi.org/pypi/align-anything/json. Fetched 2026-08-10.

[12] Read the Docs versions API for align-anything (lists the two versions, `main` and `latest`, their `active`/`built` flags, shared `identifier`, and `vcs` URL). https://readthedocs.org/api/v3/projects/align-anything/versions/. Fetched 2026-08-10.

[13] align-anything git tags (empty; same source as [9], cited separately for the pinned-vs-live claim). https://api.github.com/repos/PKU-Alignment/align-anything/tags. Fetched 2026-08-10.

[14] `align_anything/version.py` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/version.py. Fetched 2026-08-10.

[15] `scripts/llava/llava_dpo.sh` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/scripts/llava/llava_dpo.sh. Fetched 2026-08-10.

[16] Issue #159 and its comments (COLLABORATOR reply created 2025-03-17T11:19:24Z; issue `closed_at` 2025-03-18T03:35:24Z). https://api.github.com/repos/PKU-Alignment/align-anything/issues/159 ; https://api.github.com/repos/PKU-Alignment/align-anything/issues/159/comments. Fetched 2026-08-10.

[17] `scripts/setup.sh` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/scripts/setup.sh. Fetched 2026-08-10.

[18] align-anything docs, training configurations (same page as [4], cited separately for the CLI-override claim). https://align-anything.readthedocs.io/en/latest/training/configs.html. Fetched 2026-08-10.

[19] `align_anything/configs/train/text_to_text/ppo.yaml` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/configs/train/text_to_text/ppo.yaml. Fetched 2026-08-10.

[20] `align_anything/configs/train/text_to_text/grpo.yaml` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/configs/train/text_to_text/grpo.yaml. Fetched 2026-08-10.

[21] `align_anything/trainers/text_to_text/sft.py` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/text_to_text/sft.py. Fetched 2026-08-10.

[22] `align_anything/trainers/text_to_text/rm.py` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/text_to_text/rm.py. Fetched 2026-08-10.

[23] `align_anything/trainers/text_to_text/dpo.py` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/text_to_text/dpo.py. Fetched 2026-08-10.

[24] `align_anything/trainers/text_to_text/orpo.py` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/text_to_text/orpo.py. Fetched 2026-08-10.

[25] `align_anything/trainers/text_to_text/kto.py` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/text_to_text/kto.py. Fetched 2026-08-10.

[26] `align_anything/trainers/text_to_text/grpo.py` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/text_to_text/grpo.py. Fetched 2026-08-10.

[27] `align_anything/trainers/text_to_text/ppo.py` at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/text_to_text/ppo.py. Fetched 2026-08-10.

[28] `align_anything/trainers/base/supervised_trainer.py` (`save_transformers` method) at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/base/supervised_trainer.py. Fetched 2026-08-10.

[29] `align_anything/trainers/base/supervised_trainer.py` (`load_checkpoint`/`global_step` resume logic; same file as [28], cited separately for the resume claim). https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/base/supervised_trainer.py. Fetched 2026-08-10.

[30] align-anything docs site navigation (page-slug listing). https://align-anything.readthedocs.io/en/latest/. Fetched 2026-08-10.

[31] `align_anything/trainers/base/rl_trainer.py` (`RLTrainerBase.eval()`: real generation-based evaluation, decodes and displays prompts/generations, returns `{}` only when `eval_dataloader` is `None`) at the pinned commit; `GRPOTrainer`/`PPOTrainer` confirmed to subclass `RLTrainerBase` without overriding `eval()` via `align_anything/trainers/text_to_text/grpo.py` and `align_anything/trainers/text_to_text/ppo.py` (same files as [26]/[27]). https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/trainers/base/rl_trainer.py. Fetched 2026-08-10.

[32] `align_anything/utils/logger.py` (`Logger.print_table()`: renders a Rich-library console table, not a call into the `wandb`/`tensorboard` backends) at the pinned commit. https://raw.githubusercontent.com/PKU-Alignment/align-anything/3f9decc221be74b2052e712e3a32e155686ec6ec/align_anything/utils/logger.py. Fetched 2026-08-10.
