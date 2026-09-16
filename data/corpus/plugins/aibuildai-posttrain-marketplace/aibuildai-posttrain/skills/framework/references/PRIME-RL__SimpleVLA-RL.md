# SimpleVLA-RL

A veRL-derived RL post-training framework for Vision-Language-Action (VLA) robot policies, driven by a binary task-success reward instead of a learned reward model.

**SimpleVLA-RL** describes itself as "an efficient RL framework for VLA that improves long-horizon planning under data scarcity" [1]. It is built by the PRIME-RL organization (Haozhan Li and Ning Ding are named as contacts) on top of ByteDance's veRL, reusing veRL's FSDP-based distributed-training and Ray orchestration stack and adding VLA-specific rollout, model, and dataset code [1][2]. The API surface is a single Hydra-configured entry point, `python -m verl.trainer.main_ppo`, invoked through example shell scripts that override dozens of config fields for a chosen VLA model and benchmark [3][4]. It lives at https://github.com/PRIME-RL/SimpleVLA-RL [2].

**When to pick it**: pick this over a general-purpose LLM post-training library (trl, verl) when the policy is a VLA model (OpenVLA or OpenVLA-OFT) being trained against a simulated manipulation benchmark, because the environment rollout, action-chunk handling, and binary success-reward wiring are already built for that loop; the README documents support for two VLA backbones (OpenVLA, OpenVLA-OFT) and three benchmarks (LIBERO, RoboTwin 1.0/2.0), and states that other models "may need to fine-tune them yourself" first [1]. It is not a general text-LLM RL library: the repository's own Getting Started guide never points a user at its bundled SFT trainer for the VLA workflow, instead directing them to download pre-trained SFT VLA checkpoints from a Hugging Face collection [1]. The README's own Main Results give the deciding comparison against the SFT baseline this RL loop is meant to improve: on OpenVLA-OFT with only one demonstration trajectory per task for cold-start SFT, RL training raises LIBERO-Long success from 17.3 to 91.7, a 74.4-point (430.1%) gain, and with fuller SFT data the RL-trained model reaches 97.6 on LIBERO-Long, which the README describes as a new state of the art [1].

**Methods it ships**: the only trainer loop this repository documents running end-to-end is an online policy-gradient RL loop launched via `main_ppo.py` / `RayTrainer.fit()` [5][6]; despite the "ppo" naming, both bundled quickstart scripts set `algorithm.adv_estimator=grpo` [3][4], while the shipped Hydra default is `algorithm.adv_estimator: gae` (i.e. PPO with a critic) [7] - GRPO and PPO are separate methods with separate method cards, and this library's own examples pick GRPO, not PPO, so read the "Start it" arithmetic in that light. The reward is a rule-based, binary (0/1) task-completion score read out of the simulator (`RobRewardManager.verify()` reads `data.batch['complete']`), not a trained reward model [5]. A `reward_model.enable` / `PRIMERewardModelWorker` code path exists in `main_ppo.py`, but it is off by default (`reward_model.rm_coef: 0`); if it were enabled, `ray_trainer.py`'s training loop unconditionally hits `if self.use_rm: print("Not implement yet"); raise ValueError` - it is present in the source but not a working, run-tested path in this codebase [6][7]. `verl/trainer/fsdp_sft_trainer.py`, a generic text SFT trainer inherited from veRL (`AutoModelForCausalLM`, not a VLA model class), is present in the tree but is not referenced anywhere in the documented VLA workflow [8][1]. There is no version-tagged methods index to recheck; this reading is of the single `main` branch at the pinned commit below.

**Scale it handles**: single-node up to 2-node multi-node, both through Ray, no other launcher; the shipped runtime-env file (`align.json`) configures Ray's per-worker environment variables (NCCL, PYTORCH_CUDA_ALLOC_CONF, WANDB_API_KEY, etc.) [9]. The only worker strategy exercised by the documented examples is FSDP (`actor_rollout_ref.actor.strategy: fsdp`); a `megatron` strategy branch exists in `main_ppo.py` and a matching `verl/trainer/config/ppo_megatron_trainer.yaml` file exists in the tree, but that YAML's defaults are still the generic veRL LLM config (`model.path: ~/models/deepseek-llm-7b-chat`, gsm8k data files) - it is inherited scaffolding, not a documented or VLA-adapted scale path [10][3][4]. The README's only published scale numbers are a tested single-node run (1 node x 8 A800 80GB GPUs) and a tested multi-node run (2 nodes x 8 A800 80GB GPUs), with no larger configuration benchmarked [1].

**Install**: there is no package release. GitHub Releases for this repository is empty, and its one tag, `v1-stable-backup`, points at commit `00e3ae3` from 2025-09-26 - about three months (102 days) older than the newest push - so no tag is a meaningful install pin either; every claim in this card is read at the newest-push commit `7c51662df27b586f9e8a1ab35fcf849f2b8852f9` (2026-01-06) [11][12][13]. Installation is git-clone-only per `SETUP.md`: create `conda create -n simplevla python==3.10`, `pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124`, then `git clone -b v0.2.x https://github.com/volcengine/verl.git && cd verl && pip3 install -e .` as a sibling directory, with the guide's own warning: "We recommend veRL version 0.2 or 0.3. Latest versions may have library conflicts" [14]. After veRL, clone and `pip install -e` OpenVLA-OFT plus flash-attn, then either LIBERO or RoboTwin2.0 (the latter needs `bash copy_overwrite_robotwin2.sh <robotwin_path> <simplevlarl_path>`) [14]. No `setup.py`, `pyproject.toml`, or `requirements.txt` exists in this repository's own tree at the pinned commit - there is no `pip install simplevla-rl` [13]. Licence: MIT, copyright PRIME-RL 2025 [15]. No CUDA/driver minimum is documented as a hard requirement; the README instead reports the driver/CUDA combination its own tests happened to use - driver 470.161.03, CUDA 12.4 - and marks that line "(Not necessary)" [1].

**Maintained by**: PRIME-RL on GitHub, about 1,800 stars (not a ranking signal) [2]. The README's own News section shows continuing activity: a 2026-01-01 entry announcing a follow-on real-world RL result, a 2025-10-01 entry adding RoboTwin2.0 support, and a 2025-09-12 entry for the arXiv paper release [1]; the repository's newest push is 2026-01-06 [12].

## Quick start

The README's own Getting Started sequence, quoted at the step level [1]: (1) follow `SETUP.md` to build the conda environment; (2) obtain an SFT VLA checkpoint - for OpenVLA-OFT, download from the "SimpleVLA-RL Collection" on Hugging Face (e.g. a `libero-10 traj1/trajall SFT` model), since "other models... may need to fine-tune them yourself"; (3) set `WANDB_API_KEY` in `align.json`, then edit `SFT_MODEL_PATH`, `CKPT_PATH`, `DATASET_NAME`, `NUM_GPUS`, `NUM_NODES` in `examples/run_openvla_oft_rl_libero.sh` (or `_twin2.sh`) and run it; (4) to evaluate instead of train, set `trainer.val_only=True` in the same script and rerun it. The training command itself, reduced to its invocation form from the LIBERO example script [3]:

```bash
export WANDB_API_KEY='YOUR WANDB KEY'
export ROBOT_PLATFORM=LIBERO
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=libero_10 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=openvla-oft \
    algorithm.adv_estimator=grpo \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.default_local_dir=$CKPT_PATH/SimpleVLA-RL/$EXPERIMENT_NAME
```

There is no separate `trl sft`-style CLI wrapper; every run goes through this one Hydra entry point with different overrides [3][4].

## Start it

- One node: `NUM_GPUS=8`, `NUM_NODES=1` in the example script, unchanged otherwise; Ray is initialized in-process from `main_ppo.py` (`ray.init(...)`) rather than through a separate launcher CLI [5].
- Multiple nodes: set `NUM_NODES=2` (the README's only tested multi-node value) with the same `NUM_GPUS=8` per node, giving 16 total A800 GPUs in the tested configuration; there is no documented launcher beyond starting the Ray job with these Hydra overrides and a shared `ALIGN_PATH` runtime-env file [1][9].
- Batch arithmetic, from the LIBERO example and a maintainer's own worked explanation in a closed issue (zhan72, COLLABORATOR, 2025-06-06): rollout collects `data.train_batch_size` prompts times `data.n_samples` trajectories each - with `train_batch_size=64`, `n_samples=8` that is 512 trajectories per RL step - and the RL update runs in `ppo_mini_batch_size`-sized chunks, so `ppo_mini_batch_size=128` means 4 update passes per step (512/128) [3][16].
- Config surface is the Hydra `ppo_trainer.yaml` tree (`data.*`, `actor_rollout_ref.*`, `algorithm.*`, `trainer.*`) [7], and the two bundled examples change several of its defaults for VLA training: `algorithm.adv_estimator` from the default `gae` to `grpo`; `algorithm.kl_ctrl.kl_coef` from `0.001` to `0.00` (no KL penalty against the reference policy); `actor_rollout_ref.actor.entropy_coeff` from `0.001` to `0.` (no entropy bonus); `actor_rollout_ref.rollout.name` from the default `vllm` to `hf` (Hugging Face generate, not vLLM) [3][4][7]. The default config's own `val_only: True` is a trap for a first run - a script launched with an unedited `trainer.val_only` setting would only evaluate, never train; the example scripts explicitly set `trainer.val_only=False` to override this [3][7].
- Out-of-memory first aid is not written up as a dedicated section anywhere searched (README, SETUP.md, `ppo_trainer.yaml` comments); the fields present in the config that a reader would use for it are the FSDP offload flags the examples already set (`actor.fsdp_config.grad_offload=True`, `optimizer_offload=True`) and `actor_rollout_ref.rollout.gpu_memory_utilization` (0.9 in the examples), which caps the share of GPU memory the HF/vLLM rollout engine is allowed to use on a colocated GPU [3][4][7].

## Watch it

This section is the mechanics of what the library emits, not what a healthy curve looks like for GRPO or PPO - that lives on those methods' cards.

- **Enable it**: `trainer.logger` accepts a list drawn from `verl.utils.tracking.Tracking`'s `supported_backend`, which is only `['wandb', 'console']` - there is no TensorBoard backend in this code path [17]. Both bundled examples set `trainer.logger=['console','wandb']` and `trainer.wandb_mode=online` [3][4]. The console backend is not stdout-only: it writes through `LocalLogger(log_dir=local_dir)` where `local_dir` is `trainer.default_local_dir`, so console logs land on disk under the same directory as checkpoints [17][6].
- **Metric names**, read from `compute_data_metrics()` and `_validate()` in `ray_trainer.py` [6]: training-side `critic/score/mean|max|min`, `critic/rewards/mean|max|min`, `critic/advantages/mean|max|min`, `critic/returns/mean|max|min`, plus `train_reward/*` from the reward function and per-stage `timing/gen`, `timing/verify`, `timing/adv`, `timing/update_actor`, `timing/testing` wall-clock entries; when `algorithm.kl_ctrl` applies a KL penalty, `apply_kl_penalty()` additionally logs `critic/kl` and `critic/kl_coeff` [6]. Validation logs `test_reward/*`, `format_acc/*`, `acc_wformat/*`, `test_score/<data_source>` and `test_score/all` [6]. There is no separate published metrics-reference page for this repository (unlike trl's per-trainer docs site) - this list is read directly from `ray_trainer.py` at the pinned commit, and a reader who needs the exact keys for a modified fork should re-grep that file rather than trust this list against a different commit [6].
- **Reward composition to watch**: the binary task-success score (`RobRewardManager.verify()`, reading `data.batch['complete']`) is scaled by `verifier.reward_coef` (Hydra default 5) before being added to the reward tensor logged as `reward_all` / `train_reward/reward_all`; the disabled reward-model path (`reward_model.rm_coef`, default 0) is dead weight at these defaults [5][7].
- **Sample-level generations**: the README's Key Implementations map describes a "VLA rollout implementation" file that handles "environment creation, multi-environment parallel rendering, VLA action generation, environment interaction, video saving, trajectory and 0/1 reward collection" during rollout, but no config field for turning that on or off, nor a docs page describing it, was found in the files read for this card [1].
- **Evaluation during training**: `trainer.val_before_train` (the bundled examples set it `True`) and `trainer.test_freq` (both the LIBERO and the RoboTwin2.0 example set it to `4`) gate periodic `_validate()` calls that emit the `test_*` metrics above [3][4][6].
- **Stopping**: no early-stopping, patience, or reward-threshold field was found in `ray_trainer.py`, `ppo_trainer.yaml`, the README, or `SETUP.md` - the search covered all four for "early stop", "patience", "threshold", and "stop" - so shapes are published (the metric names above) but no stopping threshold is; training runs for `trainer.total_epochs` (both examples set `100`) unless killed manually [6][7][1][14].

## Save it

- Checkpoint cadence and location are Hydra fields: `trainer.save_freq` (default `-1`, meaning never; the LIBERO example sets `25`, the RoboTwin2.0 example sets `20`) writes under `trainer.default_local_dir`, laid out as `<default_local_dir>/actor/global_step_<N>/` (and a matching `critic/` subdirectory if a critic is in use) [6][7][3][4].
- The save call, `RobActorRolloutRefWorker.save_checkpoint()` in `fsdp_workers.py`, branches on whether LoRA (PEFT) is active [18]:
  - Without LoRA: it gathers the full FSDP state dict with `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`, then calls `self.actor_module.save_pretrained(local_path, state_dict=state_dict)` and `self.tokenizer.save_pretrained(local_path)` - a standard Hugging Face model directory, no optimizer or scheduler state included [18].
  - With LoRA: it writes the adapter alone to a `lora_adapter/` subdirectory via `save_pretrained(..., safe_serialization=True)`, and separately reloads the base model, merges the adapter with `PeftModel.from_pretrained(...).merge_and_unload()`, and writes that full merged model at the checkpoint root - so a LoRA checkpoint here contains both an adapter-only directory and a full merged-model directory side by side [18]. A closed issue (#36, "LoRA support?", opened 2025-08-30) reports an FSDP prefetching error when training with LoRA on this codebase; no maintainer (COLLABORATOR/OWNER) reply appears on that issue, so this is reported here as an open, unconfirmed report rather than a documented trap [19].
- No optimizer or scheduler state is saved in `save_checkpoint()` at any point in this function, and no `resume_from_checkpoint`-style call or step-resume logic was found anywhere in `ray_trainer.py` - the local `global_steps` variable inside `fit()` is only ever initialized to `0`, with no code path that reads a step counter back from a saved checkpoint [6][18]. Treat every checkpoint here as a fresh-inference artifact, not a training-resume point.
- Loader handoff: a non-LoRA checkpoint directory is a standard Hugging Face model directory (`save_pretrained`/`tokenizer.save_pretrained` output) and loads with the matching `from_pretrained` call for the VLA model class in use; a LoRA run's merged-model subdirectory is likewise a full model directory, while its sibling `lora_adapter/` directory is an adapter only and needs the base model to reload, per the same PEFT contract used elsewhere in the ecosystem [18].

## Find it in the docs

This repository ships no separate documentation site - the README and `SETUP.md` at the repository root are the whole of the documentation; there is no version-tag URL pattern to give, because there is no release (see Install) [1][14].

- Two files carry everything: `README.md` (project description, News, Highlights, Model/Environment support, a "Key Implementations" map of the source tree, Getting Started, Main Results, Acknowledgement, Roadmap, Contact) and `SETUP.md` (the full install guide) [1][14].
- The README's own Key Implementations section is the closest thing to an API map: it names `main_ppo.py` (entry point, `RobRewardManager`), `ray_trainer.py` (`RayTrainer`, the training loop, advantage computation), `fsdp_workers.py` (VLA model/optimizer init, `generate_sequences`, `compute_entropy`, `update_actor`), a `dp_rob.py`-style module (loss computation, `compute_log_prob`), the VLA rollout/environment-interaction script, and the dataset construction code [1].
- Runnable references beyond the two root docs: the `examples/` directory holds the two complete quickstart scripts used above, plus `overwrite_vla_ckpt_utils.sh` and (for RoboTwin2.0) `copy_overwrite_robotwin2.sh`, referenced directly from `SETUP.md` and the README [1][3][4][14]. The Hugging Face collection linked from the README ("SimpleVLA-RL Collection", under the `Haozhan72` account) is the source of known-good SFT starting checkpoints for LIBERO and RoboTwin2.0 tasks [1].
- Community layer: the repository curates none itself (no tutorials page); the README instead links out to a single announcement thread (Twitter/X) and a WeChat group QR code for support, and its Acknowledgement section credits veRL, OpenVLA-OFT, RoboTwin2.0, and PRIME as the projects it is built on [1]. There is no MCP endpoint for this repository's own docs.
- Traps found in closed GitHub issues, maintainer replies only (author `zhan72`, association COLLABORATOR in every case) [16][20][21]:
  - Minimum compute (#8, opened 2025-06-05): the maintainer recommends at minimum an 8x A100 or 8x A800 server, warning that fewer GPUs "will noticeably reduce training speed" (translated from the original reply); they report roughly 45 minutes per RL step on 8x A800 at `batchsize=64`, and roughly 23 minutes per step on 16x A800 [16].
  - `unnorm_key`/dataset-statistics path error (#11, opened 2025-06-06): the maintainer traces the root cause to `SFT_MODEL_PATH` being a Hugging Face Hub id rather than a local path - `dataset_statistics_path = os.path.join(local_path, "dataset_statistics.json")` resolves incorrectly in that case - and gives downloading the model locally first as one fix [20].
  - Heterogeneous GPU placement (#18, opened 2025-06-17): the maintainer confirms that using one GPU set for rollout and a different set for updates is feasible in principle but not implemented; the current loop performs rollout and update sequentially on the same GPU set [21].
  - Reference training time (#20, opened 2025-06-18): the maintainer states their published LIBERO-Long traj1-SFT result was trained with RL on 2x8xA800 GPUs for approximately 97 hours to reach 230 steps [22].

## Sources

All files are read at commit `7c51662df27b586f9e8a1ab35fcf849f2b8852f9` (the repository's newest push, 2026-01-06) unless otherwise noted; the repository has no GitHub Releases and its one tag (`v1-stable-backup`, commit `00e3ae3d`) is dated 2025-09-26, about three months (102 days) older, so it is not used as an install pin. GitHub API responses and issue threads were fetched 2026-08-11.

[1] SimpleVLA-RL README.md at commit 7c51662d. https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/README.md. Fetched 2026-08-11.

[2] SimpleVLA-RL GitHub repository metadata. https://api.github.com/repos/PRIME-RL/SimpleVLA-RL (repo page: https://github.com/PRIME-RL/SimpleVLA-RL). Fetched 2026-08-11.

[3] examples/run_openvla_oft_rl_libero.sh at commit 7c51662d. https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/examples/run_openvla_oft_rl_libero.sh. Fetched 2026-08-11.

[4] examples/run_openvla_oft_rl_twin2.sh at commit 7c51662d. https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/examples/run_openvla_oft_rl_twin2.sh. Fetched 2026-08-11.

[5] verl/trainer/main_ppo.py at commit 7c51662d (RobRewardManager, reward composition, worker-class selection). https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/verl/trainer/main_ppo.py. Fetched 2026-08-11.

[6] verl/trainer/ppo/ray_trainer.py at commit 7c51662d (RayTrainer.fit, logging, checkpoint save logic, dead reward-model branch, global_steps initialization). https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/verl/trainer/ppo/ray_trainer.py. Fetched 2026-08-11.

[7] verl/trainer/config/ppo_trainer.yaml at commit 7c51662d (Hydra config defaults). https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/verl/trainer/config/ppo_trainer.yaml. Fetched 2026-08-11.

[8] verl/trainer/fsdp_sft_trainer.py at commit 7c51662d. https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/verl/trainer/fsdp_sft_trainer.py. Fetched 2026-08-11.

[9] align.json at commit 7c51662d (Ray runtime-env template). https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/align.json. Fetched 2026-08-11.

[10] verl/trainer/config/ppo_megatron_trainer.yaml at commit 7c51662d. https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/verl/trainer/config/ppo_megatron_trainer.yaml. Fetched 2026-08-11.

[11] GitHub Releases list for PRIME-RL/SimpleVLA-RL (empty). https://api.github.com/repos/PRIME-RL/SimpleVLA-RL/releases. Fetched 2026-08-11.

[12] GitHub commit metadata for 7c51662d (author, date, confirms newest-push identity). https://api.github.com/repos/PRIME-RL/SimpleVLA-RL/commits/7c51662df27b586f9e8a1ab35fcf849f2b8852f9. Fetched 2026-08-11.

[13] GitHub recursive tree at commit 7c51662d (confirms absence of setup.py/pyproject.toml/requirements.txt). https://api.github.com/repos/PRIME-RL/SimpleVLA-RL/git/trees/7c51662df27b586f9e8a1ab35fcf849f2b8852f9?recursive=1. Fetched 2026-08-11.

[14] SETUP.md at commit 7c51662d. https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/SETUP.md. Fetched 2026-08-11.

[15] LICENSE at commit 7c51662d. https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/LICENSE. Fetched 2026-08-11.

[16] GitHub issue #8 "计算资源" and its comments (maintainer zhan72, COLLABORATOR). https://api.github.com/repos/PRIME-RL/SimpleVLA-RL/issues/8 and /issues/8/comments. Fetched 2026-08-11.

[17] verl/utils/tracking.py at commit 7c51662d (Tracking class, supported_backend list, console logger local_dir behavior). https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/verl/utils/tracking.py. Fetched 2026-08-11.

[18] verl/workers/fsdp_workers.py at commit 7c51662d (RobActorRolloutRefWorker.save_checkpoint, LoRA vs full-model save paths). https://raw.githubusercontent.com/PRIME-RL/SimpleVLA-RL/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/verl/workers/fsdp_workers.py. Fetched 2026-08-11.

[19] GitHub issue #36 "LoRA support?" and its comments (no maintainer reply present). https://api.github.com/repos/PRIME-RL/SimpleVLA-RL/issues/36 and /issues/36/comments. Fetched 2026-08-11.

[20] GitHub issue #11 "unnorm_key Assertion Error" and its comments (maintainer zhan72, COLLABORATOR). https://api.github.com/repos/PRIME-RL/SimpleVLA-RL/issues/11 and /issues/11/comments. Fetched 2026-08-11.

[21] GitHub issue #18 "GPU Type Support" and its comments (maintainer zhan72, COLLABORATOR). https://api.github.com/repos/PRIME-RL/SimpleVLA-RL/issues/18 and /issues/18/comments. Fetched 2026-08-11.

[22] GitHub issue #20 "How many hours of training?" and its comments (maintainer zhan72, COLLABORATOR). https://api.github.com/repos/PRIME-RL/SimpleVLA-RL/issues/20 and /issues/20/comments. Fetched 2026-08-11.

Method names (PPO, GRPO) and the reward-shaping term "verifier" are deliberately cited to nothing here beyond the code that uses them; their defining papers and semantics live on the methodology cards. veRL/verl (the base framework this repository forks), OpenVLA, OpenVLA-OFT, LIBERO, and RoboTwin are named in passing and reached only through the README's own links [1]; they are not separately fetched or cited here.
