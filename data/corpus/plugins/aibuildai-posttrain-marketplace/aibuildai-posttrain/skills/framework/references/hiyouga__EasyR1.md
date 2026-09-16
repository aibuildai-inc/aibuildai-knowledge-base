# EasyR1

A single-repo, Ray-and-vLLM RL post-training framework for both language and vision-language models, forked from verl and driven entirely through Hydra-style CLI overrides on one YAML config.

See https://github.com/hiyouga/EasyR1.

EasyR1 is described in its own README as "an Efficient, Scalable, Multi-Modality RL Training Framework" and states plainly that "this project is a clean fork of the original veRL project to support vision language models" [1]. Its README credits seven core contributors led by Yaowei Zheng (hiyouga) in the repository's citation block, with no organizational affiliation stated there [1]; its installed Python package is literally named `verl` - `setup.py` sets `name="verl"` and reads the version from `verl/__init__.py` [2]. The API is a single CLI entry point, `python3 -m verl.trainer.main`, which takes a base YAML config plus dotted-key overrides (`data.train_files=...`, `worker.actor.model.model_path=...`) and runs one Ray-orchestrated PPO-style training job [1].

**When to pick it**: a vision-language-capable GRPO/DAPO-family RL trainer, when your workload is one Ray cluster training a single model with vLLM colocated on the same GPUs (the README calls this its HybirdEngine design [1]), and you do not also need SFT or plain-inference scripts - the README explicitly declines to ship those and redirects such needs to LlamaFactory [1]. Whether EasyR1 is a better fit than upstream verl or a Hub-native trainer-class library such as trl is not settled by any source read for this card (cross-reference to those cards; not covered here).

**Methods it ships**: the README's own "Supported algorithms" list is GRPO, DAPO, Reinforce++, ReMax, RLOO, GSPO, and CISPO [1]; a SAPO example script also ships under `examples/` though it is not named in the README algorithm list [3]. Mechanically these split into two knobs read from `verl/trainer/core_algos.py` at commit b44b311: an `algorithm.adv_estimator` choice of `grpo`, `grpo_passk`, `rloo`, `reinforce_plus_plus`, `remax`, or `gae` (PPO-with-critic) [4], and a `worker.actor.loss_type` choice of `default`, `gspo`, `gspo_token`, `cispo`, or `sapo` that reshapes the policy-loss clipping [4]; DAPO is reached by setting `algorithm.online_filtering=True` together with the default GRPO estimator, and its group-filtering plus a token-clip-range widened to `[0.2, 0.28]` are shown directly in `examples/qwen2_5_vl_7b_geo3k_dapo.sh` [5]. LoRA is a `worker.actor.model.lora.rank>0` toggle documented as new in the README feature list [1]. There is no separate methods index page; the algorithm list lives only in the README, so recheck it there [1].

**Scale it handles**: single GPU (set `trainer.n_gpus_per_node=1`, as the LoRA example script does [6]) up to multi-node, launched by Ray rather than a wrapper CLI - the README's multi-node recipe is `ray start --head`, then `ray start --address=<head_ip>:6379` on each worker, then running the training script only on the head node [1]. Sharding is FSDP (`worker.actor.fsdp.enable_full_shard`, `enable_cpu_offload`) with rollout generation done in-process by vLLm at a configurable `tensor_parallel_size` [7]; the README links out to a separate baselines file, and that file's Performance Baselines table (no algorithm column - the runs behind it are unspecified) publishes an 8xH100, 7B-model run reaching 23.2% actor MFU as a documented benchmark, not just mechanism [8]. Ulysses sequence parallelism exists (`worker.actor.ulysses_size`) but the README's "Known bugs" section states vision-language models are not yet compatible with it [1].

The same baselines file's Algorithm Baselines table is the deciding number for whether the shipped methods work at all: on Qwen2.5-VL-7B-Instruct/Geometry3k, test accuracy rises from a shared 0.37 baseline to 0.48 with GRPO (+0.11), 0.50 with DAPO (+0.13), 0.48 with GSPO (+0.11), 0.50 with CISPO (+0.13), and 0.54 with SAPO (+0.17) [8].

**Install**: no PyPI package - the README's own steps are `git clone https://github.com/hiyouga/EasyR1.git`, `cd EasyR1`, `pip install -e .` [1]. Latest tagged release is v0.3.2, published 2025-09-18, resolving to commit a23fb5b [9]; Python floor is >=3.9 per `setup.py` [2]; licence Apache-2.0 [10]. At the v0.3.2 tag, `requirements.txt` pins `transformers>=4.54.0,<=4.56.2`, `vllm>=0.8.0`, plus `flash-attn>=2.4.3`, `peft`, `ray[default]`, `wandb` unpinned upward [11]; torch is not listed and arrives transitively through vllm/transformers. The screening commit b44b311 (pushed 2026-07-30, about ten months after the v0.3.2 release) is well ahead of that release: `verl/__init__.py` there already reads `0.3.3.dev0`, and its `requirements.txt` has loosened the transformers cap to `<5.0.0` while keeping `vllm>=0.8.0` [12] - so a `pip install -e .` from the default branch today pulls a materially newer transformers ceiling than the tagged v0.3.2 release documented in `baselines.md`. No CUDA/hardware floor is stated in the install instructions themselves; the README instead recommends a pinned Docker image, `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`, which names CUDA 12.9 and vLLM 0.11.0 [1].

**Maintained by**: Yaowei Zheng and the six other core contributors named in the citation block [1]; about 5,100 GitHub stars, not a ranking signal [13]. The repository shows an active push on 2026-07-30 [13] and a dated third release, v0.3.2 "RL Baselines", on 2025-09-18 [9].

## Quick start

The README's own three-step tutorial trains Qwen2.5-VL-7B with GRPO on the Geometry3K dataset [1]:

```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
```

```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

That script is a complete, real-model run - not pseudo-code [6]:

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=8
```

The README's LoRA variant is `bash examples/qwen3_vl_4b_geo3k_grpo_lora.sh`, and merging a saved sharded checkpoint back to Hugging Face format is `python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor` [1].

## Start it

- One GPU: set `trainer.n_gpus_per_node=1` on the same CLI form, as the shipped LoRA script `examples/qwen3_4b_math_grpo_lora.sh` does (it also drops `worker.rollout.tensor_parallel_size` to 1) [6].
- Several GPUs on one node: raise `trainer.n_gpus_per_node`; every example script under `examples/` is this same `python3 -m verl.trainer.main config=examples/config.yaml <overrides>` form with no separate launcher wrapper [5][6].
- Multi-node goes through Ray directly, not a config template: `ray start --head --port=6379 --dashboard-host=0.0.0.0` on the head node, `ray start --address=<head_node_ip>:6379` on each worker, `ray status` to check the pool, then the training script runs only on the head node; set `trainer.nnodes` accordingly [1].
- Effective batch arithmetic: `data.rollout_batch_size` (default 512) prompts are sampled per step, each generating `worker.rollout.n` (default 5) responses, forming the GRPO group; `worker.actor.global_batch_size` (default 128, EasyR1's name for verl's `ppo_mini_batch_size`) is the gradient-update mini-batch, further split into `worker.actor.micro_batch_size_per_device_for_update` (default 1) and `..._for_experience` (default 2) micro-batches per device for memory [7].
- Config surface: one YAML, `examples/config.yaml`, with `data`, `algorithm`, `worker.{actor,rollout,ref,reward}`, and `trainer` top-level blocks, overridden by dotted CLI keys [7]. EasyR1's own default in that template sets `worker.rollout.n=5` group samples per prompt, while the underlying `RolloutConfig` dataclass in `verl/workers/rollout/config.py` itself defaults `n` to 1 - the shipped config is what every example script actually runs, so `n=5` is the value a newcomer gets unless they override it [7][14]. `worker.actor.fsdp.enable_cpu_offload` and `worker.rollout.gpu_memory_utilization` (default 0.6) both trade GPU memory for CPU memory or against generation-engine headroom respectively [7].
- Out-of-memory first aid, from the README's FAQ [1]: for the vLLM `cumem_allocator` CUDA OOM message, "Reduce the `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`" [1]. A maintainer reply on closed issue #552 (FSDP OOM training Qwen3-VL-32B on 8xA800-80GB) gave the fix as reducing `data.max_prompt_length`, confirmed working by the reporter [15].

## Watch it

This section is mechanics only; what a metric shape means for a given algorithm lives on that method's own card.

- Logging is enabled by `trainer.logger`, a list of backend names; the shipped `examples/config.yaml` defaults it to `["file", "wandb"]`, so a run logs to a local JSONL file and Weights & Biases without any extra flag - unlike libraries that log nowhere until a tracker is set [7][16]. Supported backend names are `console`, `file`, `mlflow`, `swanlab`, `tensorboard`, `wandb`, dispatched by a `Tracker` class in `verl/utils/logger/logger.py` [16]; the `file` backend writes `experiment_config.json`, `experiment_log.jsonl`, and `generations.log` under `trainer.save_checkpoint_path` [16].
- Metric names, read directly from `verl/trainer/metrics.py` and `verl/trainer/core_algos.py` at commit b44b311: length metrics `response_length/{mean,max,min,clip_ratio}` and `prompt_length/{mean,max,min,clip_ratio}`; reward/return metrics `critic/{score,rewards,advantages,returns}/{mean,max,min}`, plus `critic/values/*` and `critic/vf_explained_var` only when a critic is used; timing metrics `timing_s/<stage>` and `timing_per_token_ms/<stage>`; throughput metrics `perf/{total_num_tokens,time_per_step,throughput}` [17]. The policy-update step (`verl/workers/actor/dp_actor.py`) logs `actor/pg_loss`, and, from `compute_policy_loss`'s returned metrics dict, `actor/ppo_kl`, `actor/entropy_loss`, and `actor/pg_clipfrac_higher` / `actor/pg_clipfrac_lower` (the two directions of PPO-style ratio clipping); `actor/kl_loss` and `actor/kl_coef` are logged only when `algorithm.use_kl_loss` is true [18][19].
- Sample-level logging of generations is separate from scalar metrics: `Tracker.log_generation` writes validation samples through an `AggregateGenerationsLogger`, gated by `trainer.val_generations_to_log` (default 3 in `examples/config.yaml`) [7][16].
- Evaluation during training is controlled by `trainer.val_freq` (default 5 steps, -1 disables), `trainer.val_before_train` (default true), and `trainer.val_only`; validation uses `worker.rollout.val_override_config` to swap in different sampling settings (the shipped config sets `temperature: 0.6, top_p: 0.95, n: 1` for eval versus `temperature: 1.0, top_p: 1.0, n: 5` for training) [7].
- No published stopping-rule or health-limit threshold was found for this card: `examples/config.yaml` (checked at commit b44b311) carries cadence and retry knobs - `trainer.max_try_make_batch` (default 20, -1 for no limit), consumed in `verl/trainer/ray_trainer.py` to bound retries when DAPO-style online filtering discards a whole rollout batch - but no early-stopping patience or reward-threshold field; the README's FAQ section, the only tuning-guidance prose EasyR1 publishes outside the config comments, likewise gives no such limit [1][7][22].

## Save it

- Checkpoints land under `trainer.save_checkpoint_path/global_step_<N>/`, with an `actor/` subfolder and, if a critic is used, a `critic/` subfolder, written by `FSDPCheckpointManager.save_checkpoint` in `verl/utils/checkpoint/fsdp_checkpoint_manager.py` at commit b44b311 [20]. Inside `actor/`, every rank writes its own `model_world_size_<W>_rank_<R>.pt` (sharded model state) and, unless `save_model_only` is set, `optim_world_size_<W>_rank_<R>.pt` and `extra_state_world_size_<W>_rank_<R>.pt` (optimizer, LR scheduler, RNG state); rank 0 additionally writes a `huggingface/` subfolder holding just the config, generation config, and tokenizer/processor files (not the weights) [20].
- `trainer.save_model_only` (default false in `examples/config.yaml`) trades resumability for size the same way as `save_only_model` does in transformers-based trainers: with it set, `save_checkpoint` writes only the model shard, `torch.save(model_state_dict, model_path)`, and never writes the optimizer, scheduler, or RNG-state files needed to resume [20][7].
- LoRA runs save an adapter, not a merged model: `FSDPCheckpointManager.save_checkpoint` detects a `PeftModel` and additionally writes `lora_adapter/adapter_model.safetensors` and `lora_adapter/adapter_config.json` under the same `global_step_<N>/actor/` folder [20]. Neither `lora_adapter/` nor the `huggingface/` folder next to it is a directly loadable full model - `scripts/model_merger.py` is the tool that reads `lora_adapter/adapter_config.json`, loads the named base model, and merges the adapter into a dense Hugging Face checkpoint written to disk [21].
- Retention: `trainer.save_freq` (default 5 steps, -1 disables saving) and `trainer.save_limit` (default 3, -1 for unlimited) are read by `remove_obsolete_ckpt`, which also preserves the best-validation-score checkpoint even if it falls outside the retention window, per the tracker file below [7][22].
- A `checkpoint_tracker.json` file at `trainer.save_checkpoint_path` records `best_global_step`, `best_val_reward_score`, `last_global_step`, and `last_actor_path` after every save [22][20].
- Resume: `trainer.load_checkpoint_path` set to a specific `global_step_<N>` directory resumes from it; leaving it unset with `trainer.find_last_checkpoint=true` (the default) resumes from the path recorded in `checkpoint_tracker.json`; `_load_checkpoint` also restores the training dataloader's position from a `dataloader.pt` file saved alongside the checkpoint, and continues from scratch with a printed warning if that file is missing [22].
- Loader handoff: only the `huggingface/` folder (config/tokenizer only) or a merged output of `scripts/model_merger.py` is a standard Hugging Face model directory that an external evaluator's `from_pretrained` can load directly; a raw `global_step_<N>/actor/` shard folder or a `lora_adapter/` folder is not - consult this skill's shared loader-contract reference before the first save.

## Find it in the docs

EasyR1 publishes no separate documentation site - the README at the repository root is the whole of its own documentation, and the docs address pattern is simply `https://github.com/hiyouga/EasyR1/blob/<ref>/README.md`, where `<ref>` is `main` or a tag like `v0.3.2` (verified: `README.md` renders at both `main` and the `v0.3.2` tag) [1][9].

- For anything below README depth - config field meanings, loss-type math, checkpoint file layout - the source is the answer: `examples/config.yaml` for every tunable default with inline comments [7], `verl/trainer/core_algos.py` for the advantage estimators and policy-loss variants [4], and `verl/utils/checkpoint/fsdp_checkpoint_manager.py` for the save/load contract [20]. EasyR1's README explicitly hands off to upstream verl's own hosted docs for one topic it does not duplicate: multi-node Ray training and the Ray debugger, at `https://verl.readthedocs.io/en/latest/start/multinode.html` [1].
- Runnable references beyond the README: the `examples/` tree holds one shell script per model/algorithm combination (e.g. `qwen2_5_vl_7b_geo3k_{grpo,dapo,gspo,cispo,sapo,reinforce}.sh`) [5], plus `examples/baselines/` reproducing two R1-V project baselines (CLEVR-70k counting, GeoQA-8k) [1]. Known-good smoke-test datasets are the ones the quickstart itself uses: `hiyouga/geometry3k` (image-text), `hiyouga/math12k` (text), plus `hiyouga/journeybench-multi-image-vqa` and `hiyouga/rl-mixed-dataset` named under "Custom Dataset" [1].
- `assets/baselines.md` is the closest thing to a results/community page: it publishes the reference accuracy and throughput numbers used in the quick-facts field above, generated with the pinned Docker image `hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0` at EasyR1 v0.3.2, and states its own hyperparameter table entries are all default-config values except the ones listed [8].
- Beyond the repository: the README does not curate a tutorials page or list recurring third-party blogs the way some sibling frameworks do, so there is no official curated-community layer to point to here. No official MCP endpoint for EasyR1's own docs was found - there is no separate docs site for one to query.
- Documented boundary: EasyR1 will not run supervised fine-tuning or plain inference - the README says outright, "We will not provide scripts for supervised fine-tuning and inference in this project. If you have such requirements, we recommend using LlamaFactory" [1]. Vision-language models are documented as incompatible with Ulysses sequence parallelism under "Known bugs" [1]. A maintainer confirmed on closed issue #552 that large peak CPU/GPU memory during full-parameter FSDP training of a 32B VLM on 8xA800-80GB was expected behavior, with `data.max_prompt_length` reduction as the practical lever when it OOMs [15].

## Sources

All GitHub source files are cited at commit b44b311b669bf1fd1aa2fc36f2251482ba33cb16 unless a tag is named; all pages fetched 2026-08-10.

[1] EasyR1 README. https://github.com/hiyouga/EasyR1 (raw: https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/README.md). Fetched 2026-08-10.

[2] EasyR1 `setup.py` at commit b44b311. https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/setup.py. Fetched 2026-08-10.

[3] EasyR1 `examples/qwen2_5_vl_7b_geo3k_sapo.sh` at commit b44b311. https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/examples/qwen2_5_vl_7b_geo3k_sapo.sh. Fetched 2026-08-10.

[4] EasyR1 `verl/trainer/core_algos.py` at commit b44b311 (advantage estimators, `compute_policy_loss` loss-type branches and returned metrics). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/verl/trainer/core_algos.py. Fetched 2026-08-10.

[5] EasyR1 `examples/qwen2_5_vl_7b_geo3k_dapo.sh` at commit b44b311. https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/examples/qwen2_5_vl_7b_geo3k_dapo.sh. Fetched 2026-08-10.

[6] EasyR1 `examples/qwen3_4b_math_grpo_lora.sh` and `examples/qwen2_5_vl_7b_geo3k_grpo.sh` at commit b44b311. https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/examples/qwen3_4b_math_grpo_lora.sh ; https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/examples/qwen2_5_vl_7b_geo3k_grpo.sh. Fetched 2026-08-10.

[7] EasyR1 `examples/config.yaml` at commit b44b311 (full default config surface, including `trainer.logger`, `trainer.val_generations_to_log`, `trainer.save_model_only`, `trainer.save_freq`/`save_limit`, `worker.rollout.val_override_config`). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/examples/config.yaml. Fetched 2026-08-10.

[8] EasyR1 `assets/baselines.md` at commit b44b311 (algorithm and performance baseline tables, pinned Docker image, EasyR1 version). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/assets/baselines.md. Fetched 2026-08-10.

[9] EasyR1 v0.3.2 release and tag (published 2025-09-18, resolves to commit a23fb5b06be61d0eaa5819430533bf11136609cf). https://github.com/hiyouga/EasyR1/releases ; https://api.github.com/repos/hiyouga/EasyR1/commits/v0.3.2. Fetched 2026-08-10.

[10] GitHub repository API metadata for hiyouga/EasyR1 (licence field: Apache-2.0). https://api.github.com/repos/hiyouga/EasyR1. Fetched 2026-08-10.

[11] EasyR1 `requirements.txt` at the v0.3.2 tag (`transformers>=4.54.0,<=4.56.2`, `vllm>=0.8.0`). https://raw.githubusercontent.com/hiyouga/EasyR1/v0.3.2/requirements.txt. Fetched 2026-08-10.

[12] EasyR1 `verl/__init__.py` and `requirements.txt` at commit b44b311 (version `0.3.3.dev0`; `transformers>=4.54.0,<5.0.0`). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/verl/__init__.py ; https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/requirements.txt. Fetched 2026-08-10.

[13] GitHub repository API metadata for hiyouga/EasyR1 (star count not used for ranking; `pushed_at` 2026-07-30). https://api.github.com/repos/hiyouga/EasyR1. Fetched 2026-08-10.

[14] EasyR1 `verl/workers/rollout/config.py` at commit b44b311 (`RolloutConfig.n` dataclass default of 1). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/verl/workers/rollout/config.py. Fetched 2026-08-10.

[15] EasyR1 GitHub issue #552, "[Request for Help] FSDP mode OOM when training Qwen3-VL-32B-Instruct in 8 * A800 80G," closed; maintainer reply by hiyouga (repository owner), 2025-11-03. https://github.com/hiyouga/EasyR1/issues/552. Fetched 2026-08-10.

[16] EasyR1 `verl/utils/logger/logger.py` at commit b44b311 (`Tracker`, `LOGGERS` backend map, `FileLogger` output files, `log_generation`). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/verl/utils/logger/logger.py. Fetched 2026-08-10.

[17] EasyR1 `verl/trainer/metrics.py` at commit b44b311 (`compute_length_metrics`, `compute_data_metrics`, `compute_timing_metrics`, `compute_throughout_metrics`). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/verl/trainer/metrics.py. Fetched 2026-08-10.

[18] EasyR1 `verl/workers/actor/dp_actor.py` at commit b44b311 (`actor/pg_loss`, `actor/kl_loss`, `actor/kl_coef` logging). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/verl/workers/actor/dp_actor.py. Fetched 2026-08-10.

[19] EasyR1 `verl/trainer/core_algos.py`, `compute_policy_loss` metrics dict (`ppo_kl`, `entropy_loss`, `pg_clipfrac_higher`, `pg_clipfrac_lower`) — same source as [4].

[20] EasyR1 `verl/utils/checkpoint/fsdp_checkpoint_manager.py` at commit b44b311 (checkpoint directory layout, `save_model_only` contract, LoRA adapter save path). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/verl/utils/checkpoint/fsdp_checkpoint_manager.py. Fetched 2026-08-10.

[21] EasyR1 `scripts/model_merger.py` at commit b44b311 (LoRA-adapter-into-base merge and Hugging Face save). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/scripts/model_merger.py. Fetched 2026-08-10.

[22] EasyR1 `verl/utils/checkpoint/checkpoint_manager.py` and `verl/trainer/ray_trainer.py` at commit b44b311 (`CHECKPOINT_TRACKER` filename, `find_latest_ckpt`, `remove_obsolete_ckpt`, `_save_checkpoint`/`_load_checkpoint` including dataloader-state resume). https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/verl/utils/checkpoint/checkpoint_manager.py ; https://raw.githubusercontent.com/hiyouga/EasyR1/b44b311b669bf1fd1aa2fc36f2251482ba33cb16/verl/trainer/ray_trainer.py. Fetched 2026-08-10.
