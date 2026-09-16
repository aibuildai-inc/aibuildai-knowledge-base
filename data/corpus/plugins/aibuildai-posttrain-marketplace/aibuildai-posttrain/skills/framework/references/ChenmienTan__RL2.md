# RL2

A minimal, torchrun-only post-training library (SFT, reward modeling, DPO, PPO/GRPO) that colocates FSDP or Megatron training with an in-process SGLang generation engine on the same GPUs, without a Ray scheduling layer.

**RL2** is described by its own README as "a concise library of post-training for large language models", and its name expands to "Ray Less Reinforcement Learning" [1]. It is authored by Chenmien Tan together with nine other named co-authors listed in the repository's own citation entry [1]. There is no importable trainer-class API: each method is a Hydra-configured entry point (`RL2.trainer.sft`, `.dpo`, `.rm`, `.ppo`) launched with `torchrun`, reading data and an optional environment file [1]. It lives at https://github.com/ChenmienTan/RL2 [1].

**When to pick it**: pick it when you want the training and generation for online RL to run as one `torchrun` job per node - FSDP or Megatron for training, an in-process SGLang engine sharing the same GPUs for rollout - with no separate cluster scheduler, in contrast to Ray-orchestrated frameworks; the library names itself "Ray Less Reinforcement Learning" and describes itself as concise and without complicated abstractions, positioning itself against heavier, Ray-based training stacks [1]. As evidence the approach converges, the README's own GEM-benchmark chart plots RL2 against four other frameworks (oat, OpenRLHF, ROLL, verl) on two environments: all five reach a mean episode return near 1.0 on `game:GuessTheNumber` within roughly 100-120 training steps, and all five plateau in the 0.3-0.5 range on the harder `rg:LetterCounting` task, i.e. RL2's PPO trainer tracks the other frameworks rather than lagging them on this pair of environments [1].

**Methods it ships** [1]: one `PPOTrainer` covers PPO, GRPO, and REINFORCE-style variants through config, not separate classes - the default is Dr. GRPO (loss averaged per token, advantage not divided by its standard deviation); setting `kl.type=reward, kl.reward_estimator=k1, adv.estimator=gae` recovers the original PPO recipe, and `actor.avg_level=sequence, kl.type=loss, kl.loss_estimator=k3, adv.norm_var=true` recovers DeepSeek-style GRPO [1][2]. Separate trainers exist for SFT, DPO, and reward modeling (RM) [3][4][5]. `kl.type` also accepts a third, undocumented value, `"advantage"`, visible only in `RL2/utils/algorithms.py` at commit 9161ede - the README's hyperparameter section names only `reward` and `loss` [1][2]. There is no experimental/stable split in this repo, and no separate methods-index page beyond the README itself.

**Scale it handles**: single GPU up to multi-node, both launched with plain `torchrun` (`--nproc_per_node`, plus `--nnodes`/`--node_rank`/`--master_addr`/`--master_port` for multi-node) - there is no bundled launcher config directory analogous to Accelerate's [1]. The training backend is chosen by a Hydra override, `actor=fsdp` (default) or `actor=megatron`, with `ddp_size`/`tp_size`/`cp_size` (context parallelism via ring-flash-attention) on FSDP, and full 5-way (DP/CP/PP/TP/EP) parallelism on Megatron [6][7][8]. The only published cross-run evidence is the two-environment GEM chart above; no multi-node throughput or convergence benchmark is published in the README [1].

**Install**: `pip install rl-square`, version 0.0.2, uploaded to PyPI 2025-09-25; Python floor `>=3.10` [9][10]; the repository's own `LICENSE` file at commit 9161ede is the Apache License 2.0 [26]. This release is materially behind the git source: unpacking the sdist shows a flat `RL2/` package with FSDP support only - no Megatron backend, no `envs/` (GEM/agentic environments), no `examples/`, no Docker images - all of which exist in the current repository at commit 9161ede5dcd2700bc166a17b1041bd206a75b5d4, pushed 2026-05-20 per the commit's own patch metadata, eight months after the PyPI upload [11][12][17]. The PyPI 0.0.2 build pins `torch==2.8.0`, `sglang[all]==0.5.2`, `flash-attn==2.8.3`, `ring-flash-attn==0.1.8`, and additionally requires `peft` and `liger_kernel`, none of which the current git `requirements.txt` lists as extras [11]. The git source's own `requirements.txt` at commit 9161ede is looser and different: `torch`, `transformers`, and `flash-attn` are unpinned, `ring-flash-attn==0.1.8` is the only hard pin, and `peft`/`liger_kernel` are absent [12]. No CUDA or GPU minimum is stated in the README or either dependency manifest, but the Dockerfiles are explicit: the FSDP image starts `FROM nvidia/cuda:12.8.0-devel-ubuntu22.04` and installs `flash-attn==2.8.3` with `--no-build-isolation` [13], while the Megatron image instead builds atop `slimerl/slime:latest` and compiles Hopper flash-attention from a pinned git commit (`3ba6f82`), with no `==` pin and no explicit CUDA base image of its own [14]. There are no GitHub Releases and no git tags on this repository [15][16]; the PyPI upload is the only versioned artifact.

**Maintained by**: Chenmien Tan and co-authors [1]; the repository is not archived and its most recent push is dated 2026-05-20T08:28:37Z per the GitHub API [27], which the commit patch for that push shows is a README citation-list edit, not a code change [17]. No dedicated announcements page exists; the closest signal is the PyPI upload history (one prior release, `0.0.1.post1`, on 2025-09-12) [10].

## Quick start

RL2 has no Python quickstart snippet and the README's own "Launch" section gives only a generic template (`torchrun --nproc_per_node=<number of GPUs> -m RL2.trainer.ppo <args>`), not a runnable one-liner [1]. The smallest complete, runnable forms are the shipped scripts under `examples/`, reproduced here exactly as written [18]:

```bash
# SFT, examples/limo_sft.sh - fine-tunes on the LIMO reasoning dataset
torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.sft \
    data.train.path=Chenmien/LIMO \
    data.train.max_length=16384 \
    data.train.batch_size=32 \
    actor.model_name=Qwen/Qwen2.5-7B-Instruct \
    actor.cp_size=4 \
    actor.max_length_per_device=4096 \
    trainer.project=LIMO \
    trainer.experiment_name=qwen2.5-7b-inst \
    trainer.n_epochs=15
```

```bash
# PPO/GRPO, examples/orz_ppo.sh - trains against the OpenReasonerZero math dataset
torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    rollout.train.path=Chenmien/OpenReasonerZero \
    rollout.train.prompts_per_rollout=128 \
    rollout.train.responses_per_prompt=64 \
    rollout.train.sampling_params.max_new_tokens=8192 \
    rollout.test.path=Chenmien/OlympiadBench \
    rollout.env_path=envs/orz.py \
    actor.model_name=Qwen/Qwen2.5-7B \
    actor.cp_size=2 \
    actor.max_length_per_device=8192 \
    actor.freeze_steps=4 \
    critic.model_name=Qwen/Qwen2.5-7B \
    critic.cp_size=2 \
    critic.max_length_per_device=8192 \
    adv.estimator=gae \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen2.5-7b-ppo \
    trainer.total_steps=512 \
    trainer.test_freq=8 \
    trainer.save_freq=32
```

Note that `data.train.path` is only a valid key for SFT/DPO/RM, whose configs carry a top-level `data:` block [3][4][5]; the PPO config has no `data:` block at all - its training-data key is `rollout.train.path`, as used above [2][18].

## Start it

- One GPU: the same `torchrun -m RL2.trainer.<sft|dpo|rm|ppo>` commands above with `--nproc_per_node=1` [1].
- Multiple GPUs or nodes: no bundled config templates exist - scale by adding `torchrun` flags directly (`--nnodes`, `--node_rank`, `--master_addr`, `--master_port`) and raising `actor.ddp_size`/`tp_size`/`cp_size` (FSDP) or the Megatron parallelism dims in the config [1][6][7].
- Backend switch: Hydra's `defaults:` composition picks `actor: fsdp` unless overridden with `actor=megatron` on the command line [6][7].
- Effective batch: for PPO, `rollout.train.prompts_per_rollout x rollout.train.responses_per_prompt` sets the rollout batch, and `actor.update_per_rollout` sets how many gradient updates run per rollout (the PPO "epochs" knob); for SFT/DPO/RM, `data.train.batch_size` (128 by default) is the batch directly [2][3].
- Generation is always colocated, not a separate server/client choice: the SGLang engine launches in-process on the same GPU set as the actor, sized by `rollout.server_args.mem_fraction_static` (default 0.6) and `rollout.server_args.tp_size`; the launch code hardcodes `enable_memory_saver=True` on the SGLang `ServerArgs` and contains a `# TODO: support cross-node server` comment, so the rollout server itself is single-node only regardless of the training backend [19][2].
- Changed defaults worth flagging: `dtype: bfloat16` is the default precision for both the actor [6] and the critic [28] (a bf16-capable-GPU assumption baked into the config, not opt-in); `offload_optimizer` defaults to `true` for every trainer's FSDP actor [6], and PPO's `ppo.yaml` additionally sets `actor.offload_model: true`, inherited by the critic through `critic.offload_model: ${actor.offload_model}` - SFT, DPO, and RM never set `offload_model` in their configs [3][4][5], so it falls back to `False` there via the code-level default in `RL2/workers/fsdp/base.py` [25]; `enable_gradient_checkpointing` also defaults to `true` [6].
- OOM first aid: lower `actor.max_length_per_device` / `critic.max_length_per_device` (the per-GPU packed-sequence length bound) or raise `actor.cp_size`/`tp_size` to shard further [6][7]; on the generation side, lower `rollout.server_args.mem_fraction_static` to shrink the SGLang engine's memory share [2][19].

## Watch it

This section covers only the mechanics of what RL2 emits; what a metric shape means for a given method belongs on that method's own card.

- **Enable it**: `trainer.use_wandb` (default `true`) is the only logging backend; when set `false`, the `Trainer` base class replaces `wandb.log` with a no-op inside `__init__`, so there is no console-only or file fallback - disabling wandb means no persisted metrics at all [20]. Rank-0 also mirrors the same scalars to stdout via `tqdm`/`rank0_log` regardless of the wandb setting [21][20].
- **SFT** (`actor.sft_step`): `loss/train`, `loss/test`, `grad_norm` [22].
- **DPO** (`dpo_loss` + `actor.dpo_step`): `rewards/chosen/train`, `rewards/chosen/test`, `rewards/rejected/{train,test}`, `rewards/margin/{train,test}`, `accuracy/{train,test}`, `loss/{train,test}`, `grad_norm` [23][22].
- **RM** (`rm_loss`): `accuracy/{train,test}`, `loss/{train,test}`, `grad_norm` - note this is fewer fields than DPO: RM logs only the pairwise accuracy, not the chosen/rejected reward decomposition [23].
- **PPO** (`actor.ppo_update` and `compute_advantages`): `actor/entropy`, `actor/loss`, `actor/clip_ratio`, `actor/llm_old_approx_kl`, `actor/grad_norm`, and `actor/old_ref_approx_kl` (only emitted when `actor.kl.coef > 0`) [22][23]. The rollout side (`RL2/datasets/rl.py`, `RL2/workers/rollout.py`) logs `response_length/{train,test}`, `length_clip_ratio/{train,test}`, `turns/{train,test}`, `rewards/{train,test}`, and `dynamic_filtering_ratio/{train,test}` (the fraction of prompt groups dropped for having zero reward variance, when `rollout.train.dynamic_filtering=true`) [24][19]. Every backend also emits `timing/<name>` entries (`compute_logps`, `compute_advantages`, `update_actor`, `update_rollout`) from a `time_logger` decorator wrapping those methods [21][22].
- **Sample-level logging**: each rollout's `SampleGroup` is written to JSONL under the run's save directory once per step via `SampleGroup.save`, and the first completed sample of a rollout is printed in full (prompt, response, reward) to stdout via `SampleGroup.print` [24].
- **Evaluate during training**: PPO's `trainer.test_freq` (in steps) triggers an eval-only rollout pass; SFT and DPO/RM run their held-out test split once per epoch inside the training loop, unconditionally (no configurable eval cadence for these three) [2][3][4][5].
- **Stopping**: search run 2026-08-12 over `base.py`, `ppo.py`, `sft.py`, `dpo.py`, `rm.py`, and all four config YAMLs - no early-stopping, patience, or reward-threshold field is published anywhere. `trainer.total_steps` (PPO) and `trainer.n_epochs` (SFT/DPO/RM) are fixed training budgets set by the user, not adaptive stopping rules [20][2][3][4][5].

## Save it

- Layout: checkpoints land under `trainer.save_dir` (default `ckpts/${trainer.experiment_name}`), one subdirectory per step, `<save_dir>/step<N>/`; within it, each active worker (`actor/`, and `critic/` when using GAE-style PPO) gets a `model/` directory (a full Hugging Face checkpoint - tokenizer files plus safetensors, written through `save_pretrained`) and an `optimizer_scheduler/` directory (a `torch.distributed.checkpoint` shard); a sibling `trainer/` DCP shard holds the current step and dataloader position [20][25].
- No retention flag trims what a periodic checkpoint contains: `save_ckpt` always writes both the model and the `optimizer_scheduler` DCP shard together whenever `trainer.save_freq` triggers - the only retention control is `save_freq` itself (checkpointing less often), not a flag that drops optimizer state from an individual save [25].
- Final save (`Trainer.save_model`) writes only a full Hugging Face model directory (tokenizer plus safetensors) to `<save_dir>/actor` (and `/critic`), with no optimizer or scheduler state, since it is meant as a terminal artifact rather than a resume point; the `/latest` suffix (`<save_dir>/latest/actor`) is appended only when `trainer.save_freq` is set, which every shipped trainer config leaves `null` by default [20][2][3][4][5].
- Resume: set `trainer.load_ckpt_from` to a specific `step<N>` path, or the literal string `"latest"`, which the loader auto-resolves by globbing for the highest step under `save_dir`; `Trainer.load_ckpt` then repoints the actor/critic model path (and the rollout engine's `model_path`) at the checkpoint's `model/` directory and restores optimizer, scheduler, current step, and dataloader position [20].
- No PEFT/LoRA path exists: the README's own "Incoming Features" checklist still lists "Support Low-Rank Adaptation" as an unchecked, pending item, so every save is a full model directory, never an adapter [1].
- Loader handoff: every saved `model/` directory is a standard Hugging Face `save_pretrained` output (safetensors plus tokenizer files) and is directly loadable with `AutoModelForCausalLM.from_pretrained` by any evaluator - no RL2-specific unpacking step is required.

## Find it in the docs

There is no separate documentation site - the README at https://github.com/ChenmienTan/RL2#readme is the only prose documentation, and the Hydra YAML files under `RL2/trainer/config/` are the authoritative source for every default value; this card's Watch-it section, read directly from source at commit 9161ede, is the closest thing to a metrics inventory that exists today [1][2][3][4][5].

- README section map: "Data Preperation" documents the SFT/RM/DPO/PPO JSON data shapes; "Environments" documents the async `step()` contract that a `rollout.env_path` file must implement for PPO (the README's prose names this option `actor.rollout.env_path`, but every shipped example script and the PPO config YAML both use the shorter key `rollout.env_path` - the config file, not the prose, is the one that is loaded) [1][2][18]; "Hyper-Parameters" documents `ddp_size`/`tp_size`, `max_length_per_device`/`max_new_tokens`, and the config recipes for recovering OpenAI-style PPO or DeepSeek-style GRPO from the default Dr. GRPO settings [1].
- Runnable references: `examples/*.sh` (nine ready `torchrun` launch scripts spanning SFT, DPO, RM, and PPO/REINFORCE, including GEM and search-agent environments) [18]; `envs/*.py` (countdown, ORZ, SearchR1, GEM, deep-research) as environment implementations to copy from [1]. The datasets referenced by the example scripts - `Chenmien/OpenReasonerZero`, `Chenmien/LIMO`, `Chenmien/UltraFeedback`, `Chenmien/SkyworkRM` - are Hugging Face Hub datasets under the author's own namespace and serve as known-good smoke-test data [18].
- Community layer: RL2 curates no separate tutorials page; the README instead links five of the authors' own Weights & Biases run reports (covering OpenThoughts, SkyworkRM, UltraFeedback, TinyZero/Countdown, LetterCounting, and SearchR1 runs) as worked examples - these are training dashboards, not written guides, so read them as reference runs rather than tutorials [1].
- No official MCP endpoint for these docs is published.
- Honest boundaries: LoRA/PEFT is not supported yet (explicitly unchecked in the README's feature list) [1]; the PyPI release lags the Megatron backend and the GEM environment integration by roughly eight months, so anyone needing either must install from the git source, not from `pip install rl-square` [9][11][17]; no maintainer-reply trap from a closed GitHub issue is cited here because none was fetched for this card - stated plainly rather than invented.

## Sources

All repository-code claims are read at commit 9161ede5dcd2700bc166a17b1041bd206a75b5d4, fetched 2026-08-12, except where a different commit or a PyPI build is named. The metrics search in "Watch it" was run 2026-08-12.

[1] RL2 README at commit 9161ede. https://raw.githubusercontent.com/ChenmienTan/RL2/9161ede5dcd2700bc166a17b1041bd206a75b5d4/README.md. Fetched 2026-08-12.

[2] RL2 PPO trainer config, `RL2/trainer/config/ppo.yaml`, and actor configs `RL2/trainer/config/actor/fsdp.yaml` / `actor/megatron.yaml`, at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[3] RL2 SFT trainer, `RL2/trainer/sft.py` and `RL2/trainer/config/sft.yaml`, at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[4] RL2 DPO trainer, `RL2/trainer/dpo.py` and `RL2/trainer/config/dpo.yaml`, at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[5] RL2 RM trainer, `RL2/trainer/rm.py` and `RL2/trainer/config/rm.yaml`, at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[6] RL2 FSDP actor config, `RL2/trainer/config/actor/fsdp.yaml`, at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[7] RL2 Megatron actor config, `RL2/trainer/config/actor/megatron.yaml`, and `RL2/workers/megatron/base.py`, at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[8] RL2 Megatron worker files, `RL2/workers/megatron/actor.py` and `RL2/workers/megatron/critic.py`, at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[9] rl-square on PyPI (package metadata: version 0.0.2, upload date, license, Python floor). https://pypi.org/pypi/rl-square/json. Fetched 2026-08-12.

[10] rl-square PyPI release history (prior release 0.0.1.post1). https://pypi.org/pypi/rl-square/json. Fetched 2026-08-12.

[11] rl-square 0.0.2 sdist contents (`RL2/`, `PKG-INFO`, `requirements.txt` inside the tarball) - the actual shipped PyPI package. https://files.pythonhosted.org/packages/source/r/rl-square/rl_square-0.0.2.tar.gz. Fetched and extracted 2026-08-12.

[12] RL2 `requirements.txt` and `setup.py` at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[13] RL2 `docker/Dockerfile` (FSDP image) at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[14] RL2 `docker/Dockerfile-megatron` at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[15] RL2 GitHub Releases listing (empty). https://api.github.com/repos/ChenmienTan/RL2/releases. Fetched 2026-08-12.

[16] RL2 GitHub tags listing (empty). https://api.github.com/repos/ChenmienTan/RL2/tags. Fetched 2026-08-12.

[17] RL2 commit 9161ede patch (date, author, subject: a README citation-list edit). https://github.com/ChenmienTan/RL2/commit/9161ede5dcd2700bc166a17b1041bd206a75b5d4.patch. Fetched 2026-08-12.

[18] RL2 `examples/orz_ppo.sh`, `examples/limo_sft.sh`, `examples/ultrafeedback_dpo.sh`, `examples/skywork_rm.sh` at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[19] RL2 `RL2/workers/rollout.py` at commit 9161ede (SGLang server launch, `mem_fraction_static`, `dynamic_filtering_ratio`, weight-sync bucketing). Read from the commit tarball. Fetched 2026-08-12.

[20] RL2 `RL2/trainer/base.py` (`Trainer` base class: wandb init/no-op, checkpoint save/load/resume, `save_model`) at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[21] RL2 `RL2/utils/logging.py` (`time_logger`, `gather_and_log`, `gather_and_reduce`, `rank0_log`) at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[22] RL2 `RL2/workers/fsdp/actor.py` (`sft_step`, `dpo_step`, `ppo_update`, `update_rollout` metric keys) at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[23] RL2 `RL2/utils/algorithms.py` (`dpo_loss`, `rm_loss`, `compute_advantages`, `actor_ppo_loss`) at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[24] RL2 `RL2/datasets/rl.py` (`Sample`, `SampleGroup.save`, `SampleGroup.print`, rollout metric keys) at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[25] RL2 `RL2/workers/fsdp/base.py` (checkpoint save/load contract: `_get_model_state_dict`, `_get_ckpt`, `save_ckpt`, `load_ckpt`, `save_model`) at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.

[26] RL2 `LICENSE` file at commit 9161ede (Apache License 2.0 text). Read from the commit tarball. Fetched 2026-08-12.

[27] RL2 GitHub repository API metadata (`archived`, `pushed_at`). https://api.github.com/repos/ChenmienTan/RL2. Fetched 2026-08-12.

[28] RL2 critic FSDP config, `RL2/trainer/config/critic/fsdp.yaml`, at commit 9161ede. Read from the commit tarball. Fetched 2026-08-12.
