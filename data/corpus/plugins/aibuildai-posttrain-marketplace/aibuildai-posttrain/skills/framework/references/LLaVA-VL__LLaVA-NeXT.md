# LLaVA-NeXT

The research codebase for the LLaVA family of multimodal models, not a general-purpose post-training library: its DPO path is one legacy training script wired to a frozen, vendored copy of an old trl release, and that vendored copy's PPO, Reward, DDPO, and IterativeSFT trainers are never invoked by any script in the repository.

**LLaVA-NeXT** is described by its own README as "LLaVA-NeXT: Open Large Multimodal Models", the repository behind the LLaVA-OneVision, LLaVA-NeXT (image/video/interleave), and earlier LLaVA papers [1]. It is built and maintained by the LLaVA-VL organization [2]. Its own README states that "the training pipeline in this repository is now considered legacy" and points readers to a separate, actively developed successor, lmms-engine, for LLaVA-OneVision and LLaVA-OneVision-2 training [1]. The API is not a trainer-class library: a user edits or calls one of a fixed set of shell scripts under `scripts/train/` (e.g. `dpo.sh`, `finetune_ov.sh`) that each launch a Python training entry point (e.g. `llava/train/train_dpo.py`) with `torchrun` and a large set of CLI flags [3][4]. It lives at https://github.com/LLaVA-VL/LLaVA-NeXT [2].

**When to pick it**: only if you specifically want to reproduce or extend a LLaVA-family multimodal model's own legacy DPO recipe on this exact codebase; the maintainers' own README redirects general OneVision/OneVision-2 training work to lmms-engine instead [1]. Do not pick this repository as a general post-training library the way you would pick trl or verl: its "Methods it ships" below are read from a directory named `trl/` that the repository vendors, and extracting a full tarball of the repository at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483 and grepping every `.py` file outside `trl/` shows that only the DPO trainer in that directory is subclassed and reached by any of the repository's own scripts - PPOTrainer, RewardTrainer, DDPOTrainer, and IterativeSFTTrainer are defined in that vendored directory, and no file outside it references any of those four class names (a substring match on "PPOTrainer" elsewhere in the tree turns out to be verl's unrelated `RayPPOTrainer`, in the separate `llava-critic-r1/EasyR1/` subdirectory described below) [5][6][7][8][9][18].

**Methods it ships**: DPO is the one method with an end-to-end path: `llava/train/train_dpo.py` defines `LLaVADPOTrainer(DPOTrainer)`, importing `DPOTrainer` from the vendored `trl.trainer` package, and `scripts/train/dpo.sh` / `scripts/train/dpo_ov7b.sh` launch it with `torchrun` (commit bce12e479bc4dfee2b9c50c88137b01ff51bd483) [5][10][3][4]. DDPO (`trl/trainer/ddpo_trainer.py`), PPO (`trl/trainer/ppo_trainer.py`), plain SFT via `IterativeSFTTrainer` (`trl/trainer/iterative_sft_trainer.py`), and Reward modeling (`trl/trainer/reward_trainer.py`) exist as classes in this same vendored directory, each carrying its own 2022- or 2023-dated HuggingFace (and, for DDPO, DDPO-pytorch author Kevin Black) copyright header, and the package as a whole declares itself version `0.7.11.dev0` in `trl/__init__.py` (commit bce12e479bc4dfee2b9c50c88137b01ff51bd483) [6][7][8][9][11]. The full-tarball grep above shows the repository's own scripts and `llava/` package never call them, so treat them as frozen library code bundled for the DPO trainer's imports rather than as a supported feature of this repository [18]. Method math and training-signal semantics for DPO live on the DPO methodology card, not here. Separately, the repository also bundles a newer, self-contained `llava-critic-r1/EasyR1/` subdirectory added for a GRPO-trained critic model announced 2025-08-29, built on a vendored copy of verl/EasyR1 rather than the `trl/` directory described above; it is out of scope for this card because it was not part of the DPO-trainer import chain evidenced above and the README documents it separately [1].

**Scale it handles**: `dpo.sh` launches with `torchrun --nproc_per_node=... --nnodes=... --node_rank=... --master_addr=...`, i.e. multi-GPU, multi-node via torchrun's own rendezvous, plus DeepSpeed ZeRO-3 through `--deepspeed scripts/zero3.json` [3]. No benchmark of node count or throughput is published in the README or the training scripts read for this card; scale is documented only as these launch flags [3][1].

**Install**: clone the repo, then `pip install -e ".[train]"` inside a `python=3.10` conda environment, per the README's own installation steps (commit bce12e479bc4dfee2b9c50c88137b01ff51bd483) [1]. `pyproject.toml` at that commit pins `torch==2.1.2`, `torchvision==0.16.2`, `deepspeed==0.14.4`, `peft==0.4.0`, `accelerate>=0.29.1`, `tokenizers~=0.15.2`, `bitsandbytes==0.41.0`, `transformers` from a pinned HuggingFace git commit, and `datasets==2.16.1`, under the `train` extra; Python floor is `>=3.8` in the same file, though the README's own conda command uses 3.10 [1][12]. Licence is Apache-2.0, both from the shortlist's own field and from the `LICENSE` file at that commit [12][13]. `setuptools.packages.find` in the same file explicitly includes both `llava*` and `trl*`, meaning `pip install -e ".[train]"` installs the vendored `trl` package under the plain `import trl` name - this will collide with a separately installed `trl` package from PyPI if one is present in the same environment [12]. The repository has no GitHub Releases (checked 2026-08-10: the Releases API returns an empty list), so there is no version tag to resolve; the commit above is the repository's newest push at the time of this card, not a release, and no separate install artifact exists for it [14][2]. No CUDA or GPU hardware minimum is stated in the README or `pyproject.toml` read for this card [1][12].

**Maintained by**: the LLaVA-VL GitHub organization; the repository shows 4711 stars (not a ranking signal) and its most recent push was 2026-06-15 [2]. The README's own release notes still list a 2025-08-29 entry (LLaVA-Critic-R1) as its most recent dated announcement, alongside the standing notice that this repository's training pipeline itself is legacy in favor of lmms-engine [1].

## Quick start

There is no `pip install`-and-train quick start comparable to a trainer-class library's; the smallest complete path is the README's own install sequence, given there as two separate steps under an "Installation" heading, followed by running one of the shell scripts under `scripts/train/`. Step 1 clones the repository and changes into it [1]:

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
```

Step 2 creates the conda environment and installs the package [1]:

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```
[1]

`scripts/train/dpo.sh` is the smallest complete DPO run in the repository; its `torchrun` invocation and full flag list are quoted under Start it below [3].

## Start it

- One GPU: none of the scripts read for this card provide a single-process form - `dpo.sh` and `dpo_ov7b.sh` are both written as `torchrun` multi-process launches with `--nproc_per_node`, `--nnodes`, `--node_rank`, and `--master_addr` read from environment variables (e.g. `ARNOLD_WORKER_GPU`, `ARNOLD_WORKER_NUM`) [3][4]. Running on one GPU means setting these variables to 1 node / 1 process yourself; the scripts do not document that case.
- Sharding: `--deepspeed scripts/zero3.json` in `dpo.sh` selects DeepSpeed ZeRO-3; the config file's own contents were not read for this card [3].
- `dpo.sh`'s full flag set, quoted, is the DPO Config surface for this repository - `--dpo_alpha 1.0 --beta 0.1 --gamma 0`, `--per_device_train_batch_size 1`, `--gradient_accumulation_steps 16`, `--learning_rate 5e-7`, `--bf16 True`, `--gradient_checkpointing True`, `--attn_implementation sdpa` [3]. `train_dpo.py`'s own `TrainingArguments` dataclass sets the un-overridden defaults `dpo_alpha=1.0`, `beta=0.1`, `gamma=1.0`, `attn_implementation="flash_attention_2"`, `gradient_checkpointing=True` (commit bce12e479bc4dfee2b9c50c88137b01ff51bd483); the script overrides `gamma` to 0 and `attn_implementation` to `sdpa` [15]. None of these flags are documented in prose anywhere in the README or `docs/` tree read for this card - their meaning is only the field name and default in the source [15][1].
- Effective batch size: `per_device_train_batch_size` x GPU count x `gradient_accumulation_steps`; `dpo.sh` sets `per_device_train_batch_size=1` and `gradient_accumulation_steps=16`, so effective batch scales with GPU count alone [3].
- Out-of-memory first aid: none is published as such in `dpo.sh`, `dpo_ov7b.sh`, or the README read for this card; the same levers as any DeepSpeed-backed HuggingFace `Trainer` run apply (lower `per_device_train_batch_size`, raise `gradient_accumulation_steps`, or move to a higher ZeRO stage), but no LLaVA-NeXT-specific guidance for this was found [3][4][1].
- Precision default: `--bf16 True` is set explicitly in the script rather than left to a library default [3].

## Watch it

- Logging is enabled by `--report_to wandb` in `dpo.sh`; no other backend is demonstrated in the training scripts read for this card [3].
- Metric names come from the vendored `DPOTrainer.get_batch_loss_metrics`, which computes and stores (per train/eval split, prefixed `eval_` for eval) `losses/dpo`, `losses/sft`, `losses/total`, `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `logps/chosen`, `logps/rejected`, `ref_logps/chosen`, `ref_logps/rejected` (commit bce12e479bc4dfee2b9c50c88137b01ff51bd483) [16]. `losses/total` is `dpo_alpha` times the DPO loss plus an SFT loss term when `dpo_alpha`/`gamma` are non-zero, per the same file; the exact loss composition and what each of these signals means for a healthy run belongs on the DPO methodology card, not here [16].
- `train_dpo.py`'s `TrainingArguments` carries a `generate_during_eval` field, but `train()` hard-codes `generate_during_eval=False` when constructing `LLaVADPOTrainer`, overriding the CLI flag, so sample-level generation logging during evaluation is not reachable through this script as written (commit bce12e479bc4dfee2b9c50c88137b01ff51bd483) [15].
- Evaluation during training: `train()` passes `eval_dataset=None` to `LLaVADPOTrainer` unconditionally, and `dpo.sh` sets `--evaluation_strategy "no"`; no evaluation-during-training path is wired up in this script (commit bce12e479bc4dfee2b9c50c88137b01ff51bd483) [15][3].
- No stopping-rule, threshold, or patience field was found in `train_dpo.py`, `dpo.sh`, `dpo_ov7b.sh`, or the README's training-related sections read for this card; searched 2026-08-10 [15][3][4][1].

## Save it

- Checkpoints land under `--output_dir`; `dpo.sh` sets `--save_strategy "steps" --save_steps 3000 --save_total_limit 1` [3]. `train()` checks `pathlib.Path(training_args.output_dir).glob("checkpoint-*")` at startup and calls `trainer.train(resume_from_checkpoint=True)` if any exist, otherwise a plain `trainer.train()` (commit bce12e479bc4dfee2b9c50c88137b01ff51bd483) [15] - resume is automatic from whatever `output_dir` already contains, with no separate flag to name a specific checkpoint.
- At the end of training, `train()` branches on `training_args.lora_enable`: with LoRA, it extracts the PEFT adapter state via `get_peft_state_maybe_zero_3` and writes `model.save_pretrained(output_dir, state_dict=state_dict)` plus a separate `non_lora_trainables.bin` for whatever was trained outside the LoRA adapters (e.g. the multimodal projector); without LoRA, it calls the local `safe_save_model_for_hf_trainer(trainer, output_dir)` helper, which in turn calls `trainer.save_model(output_dir)` after gathering DeepSpeed ZeRO-3 shards to rank 0 (commit bce12e479bc4dfee2b9c50c88137b01ff51bd483) [15]. The script read for this card does not document what files land in the checkpoint or LoRA-adapter directories beyond the `non_lora_trainables.bin` name quoted above; inspect a run's output directory directly for the rest.
- The LoRA save path here writes a PEFT adapter plus a separate non-adapter state dict, not a merged full model; whether an evaluator can load the result directly depends on how it reassembles `non_lora_trainables.bin` with the base model and the adapter - this repository's own scripts do not perform that reassembly, so treat the loader contract as unresolved by this card and confirm it before relying on the saved output.

## Find it in the docs

- This is a GitHub-only project: there is no separate hosted docs site read for this card. The README at https://github.com/LLaVA-VL/LLaVA-NeXT is the entry point, and it links out to per-feature pages under `docs/` in the same repository (e.g. `docs/LLaVA-NeXT.md`, `docs/LLaVA_OneVision.md`, `docs/LLaVA-NeXT-Video.md`) for inference and evaluation, not for the DPO training path covered above [1].
- No `docs/` page dedicated to DPO or RLHF training was found by listing the `docs/` directory at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483; the only DPO-training documentation that exists is the shell scripts themselves, `scripts/train/dpo.sh` and `scripts/train/dpo_ov7b.sh` [17][3][4].
- Runnable references beyond the README: `scripts/train/` holds the other training launch scripts for non-DPO stages (`finetune_ov.sh`, `finetune_si.sh`, `pretrain_clip.sh`, `pretrain_siglip.sh`, and others), and `scripts/train/README.md` exists alongside them but was not read for this card [17].
- Community layer: the README does not curate a separate tutorials page; it instead points readers to the successor project lmms-engine for current training work and to `lmms-eval` for evaluation, so treat those two repositories, not community blog posts, as the maintainers' own pointers forward from this one [1].
- Trap, stated where it bites: the README's own top-of-file notice is the honest boundary for the whole training path covered by this card - "the training pipeline in this repository is now considered legacy" - so a reader hitting friction with the DPO script above has no maintainer support path within this repository and should evaluate lmms-engine instead [1].

## Sources

Every claim about `trl/`, `llava/train/train_dpo.py`, and the `scripts/train/*.sh` files is read at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483, the shortlist's pinned commit, which the GitHub API confirms is also the repository's newest push as of this reading (2026-08-10) - there is no separate release to pin instead, per [14]. The nested `llava-critic-r1/EasyR1/` subdirectory was located by listing the repository tree at the same commit but not read in depth, since it sits outside the vendored `trl/` import chain that this card's methods claims are scoped to.

[1] LLaVA-NeXT README. https://github.com/LLaVA-VL/LLaVA-NeXT (raw: https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/README.md). Fetched 2026-08-10.

[2] LLaVA-NeXT GitHub repository metadata. https://github.com/LLaVA-VL/LLaVA-NeXT (API: https://api.github.com/repos/LLaVA-VL/LLaVA-NeXT). Fetched 2026-08-10.

[3] `scripts/train/dpo.sh` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/scripts/train/dpo.sh. Fetched 2026-08-10.

[4] `scripts/train/dpo_ov7b.sh` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/scripts/train/dpo_ov7b.sh. Fetched 2026-08-10.

[5] `trl/trainer/dpo_trainer.py` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483 (vendored trl DPOTrainer, HuggingFace copyright header). https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/trl/trainer/dpo_trainer.py. Fetched 2026-08-10.

[6] `trl/trainer/ppo_trainer.py` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/trl/trainer/ppo_trainer.py. Fetched 2026-08-10.

[7] `trl/trainer/reward_trainer.py` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/trl/trainer/reward_trainer.py. Fetched 2026-08-10.

[8] `trl/trainer/ddpo_trainer.py` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/trl/trainer/ddpo_trainer.py. Fetched 2026-08-10.

[9] `trl/trainer/iterative_sft_trainer.py` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/trl/trainer/iterative_sft_trainer.py. Fetched 2026-08-10.

[10] `llava/train/llava_trainer.py` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483 (defines `LLaVADPOTrainer(DPOTrainer)`, imports `DPOTrainer` from `trl.trainer`). https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/llava/train/llava_trainer.py. Fetched 2026-08-10.

[11] `trl/__init__.py` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483 (`__version__ = "0.7.11.dev0"`, exports PPOTrainer, RewardTrainer, IterativeSFTTrainer, and conditionally DDPOTrainer). https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/trl/__init__.py. Fetched 2026-08-10.

[12] `pyproject.toml` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/pyproject.toml. Fetched 2026-08-10.

[13] `LICENSE` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/LICENSE. Fetched 2026-08-10.

[14] LLaVA-NeXT GitHub Releases (empty) and commit lookup for bce12e479bc4dfee2b9c50c88137b01ff51bd483. https://api.github.com/repos/LLaVA-VL/LLaVA-NeXT/releases and https://api.github.com/repos/LLaVA-VL/LLaVA-NeXT/commits/bce12e479bc4dfee2b9c50c88137b01ff51bd483. Fetched 2026-08-10.

[15] `llava/train/train_dpo.py` at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483 (TrainingArguments fields, `train()` entry point, save/resume logic). https://raw.githubusercontent.com/LLaVA-VL/LLaVA-NeXT/bce12e479bc4dfee2b9c50c88137b01ff51bd483/llava/train/train_dpo.py. Fetched 2026-08-10.

[16] `trl/trainer/dpo_trainer.py`, `get_batch_loss_metrics` method, at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. Same URL as [5]. Fetched 2026-08-10.

[17] Repository tree listing at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483 (`docs/` and `scripts/train/` directory contents). https://api.github.com/repos/LLaVA-VL/LLaVA-NeXT/git/trees/bce12e479bc4dfee2b9c50c88137b01ff51bd483?recursive=1. Fetched 2026-08-10.

[18] Full source tarball of the repository at commit bce12e479bc4dfee2b9c50c88137b01ff51bd483, extracted and grepped for `PPOTrainer`, `RewardTrainer`, `DDPOTrainer`, and `IterativeSFTTrainer` across every `.py` file outside `trl/` (162 non-vendored Python files; the only substring hits, both in `llava-critic-r1/EasyR1/verl/trainer/`, are verl's own unrelated `RayPPOTrainer` class). https://github.com/LLaVA-VL/LLaVA-NeXT/archive/bce12e479bc4dfee2b9c50c88137b01ff51bd483.tar.gz. Fetched 2026-08-10.
