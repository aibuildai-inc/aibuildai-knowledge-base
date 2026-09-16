# simpleRL-reason

A research-paper code release, not a maintained library: a vendored, patched snapshot of an early verl (v0.1) plus one shell script and a rule-based math reward, built to reproduce a single GRPO recipe for zero-RL reasoning training.

**simpleRL-reason** is described by its own README as "a simple reinforcement learning recipe to improve models' reasoning abilities" using "only rule-based reward and GSM8K/Math datasets" [1]. It is built and maintained by the NLP Group at HKUST, credited in the repo's MIT licence [2], as the code release for the paper "SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild" [1][3]. Its API is not its own: the repo vendors a `verl/` directory carrying package name `verl`, version `0.1`, and homepage `https://github.com/volcengine/verl` in its own `pyproject.toml` [4], and the README states the RL code is "implemented based on Verl" [1] - so training is driven the same way as upstream verl, through Hydra config overrides passed to `python -m verl.trainer.main_ppo`, wrapped here in one launch script, `train_grpo_math_tune_ray.sh`, that fills in the recipe's specific hyperparameters [5].

**When to pick it**: pick this repo only to reproduce the SimpleRL-Zoo paper's exact GRPO-on-math recipe (rule-based reward, GSM8K/MATH-derived data, one of ten named base models) [1][3] - not as a general post-training library. For a maintained, actively-developed engine with the same Ray+vLLM+FSDP/Megatron shape, use upstream verl instead (cross-reference; not covered here): this fork's own `verl/version/version` file still reads `0.1` [4], and the training paths this card describes (`main_ppo.py`, `train_grpo_math_tune_ray.sh`) last changed in March 2025, months before the repo's most recent push in December 2025, which only edited `README.md` (see Install and Maintained by) [6][7].

**Methods it ships**: one entrypoint, `verl/trainer/main_ppo.py`, dispatched by the Hydra key `algorithm.adv_estimator` - the paper's GRPO recipe sets `adv_estimator=grpo` and is what the launch script and README quick start use [5]; the same entrypoint's config file defaults to `adv_estimator=gae` (i.e. PPO) with a value-model critic [8]. A separate `fsdp_sft_trainer.py` runs supervised fine-tuning [9]. Reward-model scoring workers exist under `verl/workers/reward_model/` (FSDP and Megatron backends) [10], but they are not the recipe's path: `main_ppo.py`'s own reward dispatch routes any data source whose name contains `"simplelr"` to a rule-based math-equivalence checker (`hf_math_verify` or, on Python 3.9, a bundled Qwen-Math grader - see Find it in the docs) instead of a reward model [11]. This is not a taxonomy that "moves" in the sense of an actively developed library - there is no live methods page; the list above is read directly from the vendored source at the pinned commit.

**Scale it handles**: single GPU up to the paper's own runs - single H/A100-80G GPU for the 0.5B model, 2x8 H100-80G GPUs for 7B/14B models (about 15 hours for 100 steps on 8K examples), and 8x8 H100-80G GPUs for the 32B model (about 1.5 days) [1]; these are the paper's own published reference runs, not a documented ceiling. Multi-node is a documented mechanism: the script starts a Ray head node and lets additional nodes join with `ray start --address ... --num-gpus 8`, and `trainer.nnodes` sets the node count passed to `verl.trainer.main_ppo` [1][5]. Sharding is FSDP only via the vendored `verl` (`actor_rollout_ref.actor.fsdp_config`, etc. [8]); the README states the environment setup provided "only support[s] custom environment setup and FSDP training", and Megatron requires following upstream verl's own docs instead [1].

**Install**: no PyPI package - `pip install -e .` from a git checkout of this MIT-licensed repo, against Python >=3.8 per the vendored `pyproject.toml`'s `requires-python` [4]; there is no GitHub release or tag (`releases` API returns an empty list [12]), so "the release" is the commit itself, read here at `cf1c7858bfc145a27aa733e75d930d34f0d318a6` (2025-08-03). That commit's own diff touches only `README.md` [7]; the vendored training code this card cites was last changed earlier and separately - `verl/trainer/main_ppo.py` at sha `2204dd1` (2025-03-24) and `train_grpo_math_tune_ray.sh` at sha `a89537d` (2025-03-29), per each file's own commit history [6] - so the pinned commit is a README-only snapshot several months after the last change to the training paths this card describes, not itself a training-code commit. `requirements.txt`, pinned at the pinned commit, hard-caps the deep-learning core the release needs: `transformers<4.48`, `vllm<=0.6.3`, `ray[default]==2.10.0`, plus `hydra-core==1.4.0.dev1` and `omegaconf==2.4.0.dev3`; torch is not listed there but the README's own install line pins `torch==2.4.0` against CUDA 12.4 (`--index-url .../cu124`) [1][13]. `math-verify[antlr4_11_0]==0.6.0` is also pinned in `requirements.txt` [13], but a collaborator's reply on an open issue states it needs Python 3.10+, and that on Python 3.9 the repo instead uses a bundled Qwen-Math equivalence checker for the same rule-based reward [14] - so the Python floor a reader actually needs depends on which reward path they take, not just the stated `>=3.8`.

**Maintained by**: the HKUST NLP Group [2], credited in the MIT `LICENSE` file to "NLP Group @ HKUST" copyright 2025 [2]; last push 2025-12-23 (a `README.md` edit, per GitHub's commits API for that date - not separately re-fetched here beyond the repo API's `pushed_at` field [12]); the paper behind the code was accepted at the Second Conference on Language Modeling, per the README's own citation block [1]. As of 2026-08-11 the repo has 33 open issues per the GitHub search API [15]; among a sample of 13 open issues checked for replies, several carry a reply from collaborator Zeng-WH (e.g. issue #80, 2025-04-22, and issue #90, 2025-07-31) [16][15], and several others in that same sample have no reply at all - see Find it in the docs for the specific traps sourced from this activity.

## Quick start

There is no pip-installable quick start; the smallest complete path is the README's own install-then-download-then-run sequence [1]:

```bash
conda create -n verl python==3.9
conda activate verl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
```

```bash
wget https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_qwen_level3to5/train.parquet
wget https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_qwen_level3to5/test.parquet
```

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
bash train_grpo_math_tune_ray.sh --model_name Qwen-2.5-7B --max_response_length 8192 \
  --train_batch_size 1024 --rollout_n 8 --kl_loss_coef 0.0001 --entropy_coeffient 0.001 \
  --rollout_gpu_memory_util 0.75 --rollout_tp 2 --save_freq 5
```

There is no separate CLI beyond this script and the underlying `python -m verl.trainer.main_ppo` Hydra invocation it wraps [1][5].

## Start it

- One GPU: run the same `train_grpo_math_tune_ray.sh` path with `ray start --head --num-gpus 1` and a model sized to fit, per the README's stated single-H/A100-80G floor for the 0.5B model [1].
- Several GPUs/nodes go through Ray, not Accelerate or torchrun: start the head node with `ray start --head --node-ip-address 0.0.0.0 --num-gpus 8`, join additional nodes with `ray start --address {MASTER-NODE-ADDRESS}:6379 --num-gpus 8`, then submit the job from the master node; `--model_name`, `--kl_loss_coef`, `--entropy_coeffient`, `--rollout_gpu_memory_util`, and `--rollout_tp` (rollout tensor-parallel size) are the script's own CLI flags, translated internally to `trainer.nnodes`, `actor_rollout_ref.*`, and `algorithm.*` Hydra overrides on `main_ppo.py` [1][5].
- Effective batch: the script's `TRAIN_BATCH_SIZE` (data-loader batch, default 256, README example uses 1024) is split into `PPO_MINI_BATCH_SIZE` update chunks and further into `PPO_MICRO_BATCH_SIZE` per-GPU forward/backward chunks (default 2); `ROLLOUT_N` (default 8) is the number of GRPO samples generated per prompt [5].
- Generation is a separate layout choice inherited from vendored verl: `actor_rollout_ref.rollout.name=vllm` with `gpu_memory_utilization` (script flag `--rollout_gpu_memory_util`, e.g. 0.6-0.75) and `tensor_model_parallel_size` (script flag `--rollout_tp`) sharing the same GPUs as training [5][8].
- Config surface: the vendored `verl/trainer/config/ppo_trainer.yaml` sets `actor_rollout_ref.model.enable_gradient_checkpointing: True` and `actor_rollout_ref.rollout.dtype: bfloat16` by default [8] - a bf16-capable GPU is assumed; the launch script additionally forces `actor_rollout_ref.actor.use_kl_loss=True` for the GRPO recipe, which the base config otherwise defaults to `False` [5][8].
- Out-of-memory first aid is not written up as a tuning guide anywhere in this repo; the closest documented fix is a maintainer's reply in a closed issue about actor processes dying from CPU out-of-memory during rollout generation: set `actor_rollout_ref.rollout.free_cache_engine=False` to mitigate it (issue #84, reply by collaborator Zeng-WH, 2025-05-07) [14].

## Watch it

- Logging backend: the launch script hard-codes `trainer.logger=['console','wandb']` [5]; the vendored `Tracking` class also supports `mlflow` and `swanlab` as `trainer.logger` values, and needs no external tracker only if `logger` is set to `['console']` alone [17].
- Metric names below come from the parts of the vendored `verl/trainer/ppo/ray_trainer.py` read for this card (there is no separate metrics doc page in this repo), and are a partial list, not the full metric surface - the same file also merges in actor-loss, critic-loss, and validation metrics (`actor_output_metrics`, `critic_output_metrics`, `val_metrics`), plus a fourth batch, `global_balance_stats`, produced by `log_seqlen_unbalance` and imported from `verl.utils.seqlen_balancing` - all from other modules this card did not read [18]. From `compute_data_metrics`: `critic/score/{mean,max,min}`, `critic/rewards/{mean,max,min}`, `critic/advantages/{mean,max,min}`, `critic/returns/{mean,max,min}`, and (when a critic is in use) `critic/values/{mean,max,min}` and `critic/vf_explained_var`; `critic/kl` and `critic/kl_coeff` from the KL controller; `response_length/{mean,max,min,clip_ratio}`, `prompt_length/{mean,max,min,clip_ratio}`, and `filtered_response_length/{mean,max,min,clip_ratio}` [18]. From `compute_timing_metrics`, a separate function in the same file: `timing_s/{name}` (wall-clock seconds per named stage) and `timing_per_token_ms/{name}` (milliseconds per token for stages with a token count) [18]. What healthy shapes for these look like is not this card's job - that belongs on the GRPO/PPO method cards, not here.
- Sample-level logging: the reward manager in `main_ppo.py` accepts a `num_examine` count of decoded response batches to print to console per step, but this repo publishes no config flag or docs page describing that surface further than the source itself [11].
- Evaluation-during-training: the same `data.val_files` / `data.val_batch_size` fields as upstream verl's PPO config, with `trainer.test_freq` controlling cadence (`-1` disables it in the base config; the README's evaluation is instead a separate offline script, `eval_math_nodes.sh`, run after training) [8][1].
- Stopping-rule honesty: no early-stopping or reward-threshold field is defined anywhere in the files read for this card (`ray_trainer.py`, `ppo_trainer.yaml`, `train_grpo_math_tune_ray.sh`) [5][8][18]; training runs for `trainer.total_epochs` (default 20 in the script) with no other stop condition.

## Save it

- Layout, from `ray_trainer.py`'s `_save_checkpoint` and the FSDP checkpoint manager, both read at this commit: each save writes `default_local_dir/global_step_<N>/actor/` (and `/critic/` when a critic is present) [18][19]. Inside `actor/`, every rank writes its own `model_world_size_<W>_rank_<r>.pt` (sharded FSDP weights), `optim_world_size_<W>_rank_<r>.pt`, and `extra_state_world_size_<W>_rank_<r>.pt` (LR scheduler + RNG state); rank 0 additionally writes a `huggingface/` subfolder holding the merged full-precision weights via `save_pretrained` plus the tokenizer and config - that subfolder, not the sharded `.pt` files, is what a plain `from_pretrained()` can load [19].
- Retention: `trainer.remove_previous_ckpt` defaults to `True` in the base config [8] and is passed straight into `save_checkpoint(..., remove_previous_ckpt=...)`, which deletes the prior `global_step_<N>/` directory after each new save - only the latest checkpoint survives locally unless this flag is explicitly overridden [8][19].
- Resume: `trainer.resume_mode` is `auto` by default, meaning `_load_checkpoint` searches `default_local_dir` for the newest `global_step_<N>` folder and resumes from it if found, or starts from scratch if none exists; setting `resume_mode` to a literal path containing `global_step_` resumes from that folder specifically, and `resume_mode: disable` always starts fresh [18]. The SFT trainer instead reads `trainer.resume_path` directly, with no auto-discovery logic shown in the section of `fsdp_sft_trainer.py` read for this card [20].
- SFT saves are plain Hugging Face format only: `save_checkpoint` in `fsdp_sft_trainer.py` writes `default_local_dir/global_step_<N>/` via `model.save_pretrained(path, state_dict=state_dict)` plus `tokenizer.save_pretrained(path)`, with no separate sharded-state files [20].
- LoRA is a config option on the SFT trainer (`model.lora_rank`, default 0 = disabled; `lora_alpha`, `target_modules`) [9], but the section of `fsdp_sft_trainer.py` read for this card only shows a single `save_pretrained` call and does not show adapter-only-versus-merged save branching - verify on disk before assuming either behavior.
- Loader handoff: the PPO/GRPO `huggingface/` subfolder and the SFT `global_step_<N>/` directory are both plain Hugging Face model directories loadable with `AutoModelForCausalLM.from_pretrained`; the sharded `actor/*.pt` files are not directly loadable outside this repo's own FSDP checkpoint manager.

## Find it in the docs

This repo has no hosted documentation site - the README is the whole of the docs, and the vendored `verl/` source is the only other reference [1].

- Everything the README does not cover is answered by reading the vendored source directly: config surfaces live in `verl/trainer/config/*.yaml` (`ppo_trainer.yaml`, `ppo_megatron_trainer.yaml`, `sft_trainer.yaml`) [8][9], the reward dispatch lives in `verl/trainer/main_ppo.py` [11], and the rule-based reward implementations live in `verl/utils/reward_score/` (`simplelr_math.py`, `hf_math_verify.py`, the bundled `qwen_math_eval_toolkit/` for the Python 3.9 path) [11][14].
- The repo has a `v0` branch/tag holding "an old version of this repo... with our early results and codebase using OpenRLHF and PPO", linked directly from the README - useful only for historical comparison, not as an alternative install path [1].
- Runnable references beyond the README: `eval_math_nodes.sh` for the offline evaluation pipeline against AIME24, AMC23, MATH500, OlympiadBench, GSM8K, and Minerva Math, and `launch_gradio.sh` for a Gradio tool that visualizes per-step responses [1]. The paper's own training data is published as `hkust-nlp/SimpleRL-Zoo-Data` on the Hub, with per-difficulty-level, per-prompt-format subfolders (e.g. `simplelr_qwen_level3to5/`) [1].
- Traps found from maintainer (collaborator) replies only: math-verify-based rule reward requires Python 3.10+; on Python 3.9 the repo instead uses the bundled Qwen-Math grader for the same "simplelr"-tagged data (issue #77, open, reply by collaborator Zeng-WH, 2025-04-22) [14]; actor processes dying from CPU out-of-memory during rollout generation is mitigated by `actor_rollout_ref.rollout.free_cache_engine=False` (issue #84, closed, reply by collaborator Zeng-WH, 2025-05-07) [14].
- Honest boundary: this repo documents FSDP training only for its own install path, explicitly deferring Megatron support to upstream verl's docs [1]; it has no PyPI package, no versioned release, and (per the files read for this card) no CI or test-suite entry point beyond a `tests/` directory not inspected here.

## Sources

All GitHub pages and files are read at commit `cf1c7858bfc145a27aa733e75d930d34f0d318a6` unless otherwise noted; the repo-level `pushed_at` field and issue listing are live-endpoint reads dated 2026-08-11.

[1] simpleRL-reason README. https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/README.md. Fetched 2026-08-11.

[2] simpleRL-reason LICENSE (MIT, copyright NLP Group @ HKUST). https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/LICENSE. Fetched 2026-08-11.

[3] simpleRL-reason GitHub repository (item home). https://github.com/hkust-nlp/simpleRL-reason. Fetched 2026-08-11.

[4] simpleRL-reason vendored verl build metadata (package name, version 0.1, upstream homepage, Python floor). https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/pyproject.toml and https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/verl/version/version. Fetched 2026-08-11.

[5] simpleRL-reason GRPO launch script. https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/train_grpo_math_tune_ray.sh. Fetched 2026-08-11.

[6] GitHub per-path commit history for the training paths this card cites. https://api.github.com/repos/hkust-nlp/simpleRL-reason/commits?path=verl/trainer/main_ppo.py and https://api.github.com/repos/hkust-nlp/simpleRL-reason/commits?path=train_grpo_math_tune_ray.sh. Fetched 2026-08-11.

[7] GitHub commits API for the pinned commit (its own file diff, showing only README.md changed). https://api.github.com/repos/hkust-nlp/simpleRL-reason/commits/cf1c7858bfc145a27aa733e75d930d34f0d318a6. Fetched 2026-08-11.

[8] vendored verl PPO trainer config (algorithm.adv_estimator default, actor/rollout defaults, checkpoint/resume fields). https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/verl/trainer/config/ppo_trainer.yaml. Fetched 2026-08-11.

[9] vendored verl SFT trainer config (LoRA fields, trainer.resume_path). https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/verl/trainer/config/sft_trainer.yaml. Fetched 2026-08-11.

[10] simpleRL-reason repository tree at this commit (confirms `verl/workers/reward_model/megatron/reward_model.py` and `verl/trainer/fsdp_sft_trainer.py` paths). https://api.github.com/repos/hkust-nlp/simpleRL-reason/git/trees/cf1c7858bfc145a27aa733e75d930d34f0d318a6?recursive=1. Fetched 2026-08-11.

[11] vendored verl PPO/GRPO entrypoint and reward dispatch. https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/verl/trainer/main_ppo.py. Fetched 2026-08-11.

[12] GitHub repository API for simpleRL-reason (pushed_at, stargazers_count, archived, license, no releases). https://api.github.com/repos/hkust-nlp/simpleRL-reason and https://api.github.com/repos/hkust-nlp/simpleRL-reason/releases. Fetched 2026-08-11.

[13] simpleRL-reason pinned Python dependencies. https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/requirements.txt. Fetched 2026-08-11.

[14] Maintainer(collaborator)-answered issues on simpleRL-reason: #77 (open, Python 3.10+ requirement for math_verify, reply by collaborator Zeng-WH) and #84 (closed, CPU OOM during rollout generation, reply by collaborator Zeng-WH). https://api.github.com/repos/hkust-nlp/simpleRL-reason/issues/77, https://api.github.com/repos/hkust-nlp/simpleRL-reason/issues/77/comments, https://api.github.com/repos/hkust-nlp/simpleRL-reason/issues/84, https://api.github.com/repos/hkust-nlp/simpleRL-reason/issues/84/comments. Fetched 2026-08-11.

[15] GitHub search API for open issues on simpleRL-reason (total count), and comments on issues #80 and #90 (collaborator Zeng-WH replies). https://api.github.com/search/issues?q=repo:hkust-nlp/simpleRL-reason+type:issue+state:open, https://api.github.com/repos/hkust-nlp/simpleRL-reason/issues/80/comments, https://api.github.com/repos/hkust-nlp/simpleRL-reason/issues/90/comments. Fetched 2026-08-11.

[16] GitHub issues listing for simpleRL-reason (open/closed state and titles, live endpoint). https://api.github.com/repos/hkust-nlp/simpleRL-reason/issues?state=all&per_page=30. Fetched 2026-08-11.

[17] vendored verl tracking/logging backends. https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/verl/utils/tracking.py. Fetched 2026-08-11.

[18] vendored verl Ray PPO trainer (metric names, checkpoint save/load logic, resume_mode handling). https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/verl/trainer/ppo/ray_trainer.py. Fetched 2026-08-11.

[19] vendored verl FSDP checkpoint manager (sharded and Hugging Face save layout, remove_previous_ckpt behavior). https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/verl/utils/checkpoint/fsdp_checkpoint_manager.py. Fetched 2026-08-11.

[20] vendored verl FSDP SFT trainer (save_checkpoint, resume_path field). https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/cf1c7858bfc145a27aa733e75d930d34f0d318a6/verl/trainer/fsdp_sft_trainer.py. Fetched 2026-08-11.
