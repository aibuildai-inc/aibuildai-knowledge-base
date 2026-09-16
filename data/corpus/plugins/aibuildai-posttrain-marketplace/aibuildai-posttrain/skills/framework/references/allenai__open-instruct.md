# open-instruct

AllenAI's research codebase for post-training: SFT, DPO, reward modeling, and Ray+vLLM online RL (GRPO), with recommended paths built on OLMo-core's FSDP training infrastructure - install with `uv sync`, no PyPI package.

**open-instruct** describes itself as "an open effort on instruction-tuning and post-training popular pretrained language models on publicly available datasets" [1]. It is built and maintained by the Allen Institute for AI (AI2), where GitHub itself labels the repository "AllenAI's post-training codebase" [2]. Each method is a standalone `open_instruct/<method>.py` training script taking a `--model_name_or_path` and a `--mixer_list` of Hub or local datasets, run directly with `torchrun` or through AI2's `mason.py` cluster launcher [3][4]. It lives at https://github.com/allenai/open-instruct [2].

**When to pick it**: reproducing or extending AI2's own Tulu/OLMo post-training recipes (SFT, DPO, reward modeling, GRPO with verifiable rewards), when you want the exact scripts and hyperparameters behind published Tulu 3 and Olmo 3 checkpoints [1][5][6]. It is explicitly a research codebase - the README itself warns it "does not guarantee backward compatibility" [3] - unlike trl or verl, which are built as reusable libraries for outside adoption (cross-reference; not covered here). Every method has a "recommended" OLMo-core-based script and an older "legacy" DeepSpeed/Accelerate script kept for reproducibility of past runs [5][6][7]; pick the recommended one unless you need to exactly rerun a historical launch command.

**Methods it ships**: SFT (`open_instruct/olmo_core_finetune.py`, recommended for OLMo/OLMoE/Qwen3, vs. legacy `open_instruct/finetune.py`) [5]; DPO (`open_instruct/dpo.py`, recommended, vs. legacy `open_instruct/dpo_tune_cache.py`) [6], which the docs page also lists as supporting `dpo_norm`, `simpo`, and `wpo` loss-type variants via `--loss_type` [6]; Reward Modeling (`open_instruct/reward_modeling.py`, the only variant) [8]; and GRPO for RLVR (`open_instruct/grpo.py`, OLMo-core + Ray + vLLM, vs. the DeepSpeed-based, packing-optimized `open_instruct/grpo_fast.py`) [7]. `open_instruct/finetune.py` carries a `DeprecationWarning` at its own entry point pointing readers to the OLMo-core SFT script instead [9]. This taxonomy is the docs site's own "Training" nav section and it can change; recheck the live nav at [10].

**Scale it handles**: SFT, DPO, and GRPO's OLMo-core paths all use Hybrid Sharded Data Parallelism (HSDP) via PyTorch's DTensor-based FSDP2, auto-detecting shard/replica degree from node count so all-gather traffic stays inside a node and only the lighter all-reduce crosses nodes [11]. SFT auto-selects FSDP for one node and HSDP for multiple, with no manual sharding flags; DPO and GRPO expose `--fsdp_shard_degree`/`--fsdp_num_replicas` for manual control, and DPO additionally supports tensor parallelism (`--tensor_parallel_degree`, recommended for 32B+ models) - DPO's own docs state context parallelism is "not yet supported" [11]. GRPO also has a `--single_gpu_mode` that collocates vLLM and the training actor on one GPU with no sharding [12][11]. Launch is via `torchrun` for single/multi-node SFT and DPO, or Beaker+Ray for GRPO; published Olmo 3 GRPO runs range from a single debug GPU up to 28 nodes / 224 GPUs (32B "Think" model) [12] - this is a documented scale point, not an independent benchmark.

**Install**: no PyPI package - `pip install open-instruct` 404s on PyPI [13]. Clone the repo and run `uv sync` [3]; the pinned `pyproject.toml` at the latest release tag v0.3.0 (2026-06-11, commit `26de99604002d6a7b25a2c5f2ee880e388a21dca`) requires Python `==3.12.*`, Apache-2.0 licensed, and pins `torch>=2.10.0`, `transformers>=5.4.0`, `vllm>=0.19.1`, `deepspeed>=0.18.3`, `ray[default]>=2.49.2`, `peft>=0.13.2`, `wandb==0.23.1` (hard pin) [14]. No CUDA/hardware floor is stated in that file, but `flash-attn>=2.8.3` and platform-specific `flash-attn-3`/`flash-attn-4` wheel URLs are pinned as direct dependencies on Linux x86_64 [14]. The repo's newest push (commit `b18cfc57c07c37f6a90d8b86e4c2b24934944b50`, ahead of v0.3.0) has since restructured GPU dependencies (torch, vllm, flash-attn) into `uv` dependency-groups named `cuda12`/`cuda13`; `default-groups = ["dev", "cuda12"]` in that `pyproject.toml` means a plain `uv sync` still pulls the cuda12 GPU packages by default, and only switching to `cuda13` (or opting out) needs an explicit `--group` flag [15]; the docs' own installation page is stale against both of these and still describes Python 3.10 with `pip install torch==2.5.1 ... --index-url .../cu121` and `flash-attn==2.7.2.post1` [16] - follow `pyproject.toml`/the README's `uv sync`, not that page.

**Maintained by**: AI2 (Allen Institute for AI); Apache-2.0, about 3,827 GitHub stars, not archived, with a push on 2026-08-11 [2] and three tagged releases (v0.1.0, v0.2.0, v0.3.0, latest 2026-06-11) [17]. The docs' Models section and README News list ongoing model releases through Olmo 3 (dated 2025-11-20 in-docs) [1][10].

## Quick start

From the README's own Training section [3]:

```bash
# train an 8B tulu3 model using 8 GPU
bash scripts/train/tulu3/finetune_8b.sh
```

```bash
# train an 8B tulu3 model using 8 GPU
bash scripts/train/tulu3/dpo_8b.sh
```

```bash
# Single-GPU smoke test on Beaker (small model, fast).
./scripts/train/build_image_and_launch.sh scripts/train/debug/single_gpu_on_beaker.sh
```

For SFT, the docs give the OLMo-core `torchrun` form directly [5]:

```bash
torchrun --nproc_per_node=8 open_instruct/olmo_core_finetune.py \
    --model_name_or_path allenai/OLMo-2-0325-32B-DPO \
    --mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
    --max_seq_length 4096 \
    --learning_rate 8e-5 \
    --num_epochs 3
```

## Start it

- One process, one GPU: run any of the debug scripts directly, e.g. `bash scripts/train/debug/finetune.sh` (legacy SFT) or, via the Beaker image builder, `./scripts/train/build_image_and_launch.sh scripts/train/debug/single_gpu_grpo.sh` for GRPO [5][12].
- Multi-GPU/multi-node SFT and DPO launch with `torchrun`; the parallelism strategy (FSDP vs. HSDP) is chosen automatically from node count for SFT, or set through `--fsdp_shard_degree`/`--fsdp_num_replicas` for DPO and GRPO [11]. GRPO instead runs through Ray actors, launched at AI2 via `scripts/train/build_image_and_launch.sh`, which builds a Beaker image from the current commit and runs the named script - published GRPO configs range 1 to 28 nodes [12].
- Effective batch size follows `--per_device_train_batch_size x devices x --gradient_accumulation_steps`; GRPO's own knobs additionally multiply by `--num_unique_prompts_rollout` (default 16) x `--num_samples_per_prompt_rollout` (default 4) at the rollout stage [12].
- Config surface is per-script CLI flags via a dataclass parser, not a shared class: DPO's OLMo-core script defaults `--compile_model=True` and `--activation_memory_budget=1.0` (full recompute disabled unless lowered below 1.0) [6]; SFT defaults `--learning_rate=8e-5`, `--num_epochs=3`, `--max_seq_length=4096` and, like DPO, `--compile_model=True` by default [5] - torch.compile-by-default is a silent assumption that the model architecture is compile-compatible, worth checking before a first run on an unusual architecture.
- Out-of-memory first aid: for the OLMo-core trainers, lower `--activation_memory_budget` below 1.0 to enable budget-mode activation checkpointing, or increase `--fsdp_shard_degree` to shard further within a node [11][6]; DPO also exposes `--sync_each_batch` to reduce memory during gradient accumulation [6]. On the generation side, GRPO's `--vllm_gpu_memory_utilization` (default 0.9) sets vLLM's share of GPU memory, and `--single_gpu_mode` collocates the actor and vLLM engine on the same card for small-scale runs [12].

## Watch it

This section is the mechanics only; what a metric shape means for a given method lives on that method's methodology card.

- **Enable it**: every script exposes `--with_tracking` (Weights & Biases, default `False`) plus `--wandb_project` (default `"open_instruct_internal"`) and `--wandb_entity`; with no tracker set, a run keeps no external record beyond stdout and local files [5][6]. GRPO and DPO can additionally push alerts to Slack: set the `SLACK_WEBHOOK_URL` environment variable and pass `--send_slack_alerts`; on AI2's Beaker cluster, `mason.py` auto-injects a per-user `{username}_SLACK_WEBHOOK_URL` secret the same way it injects `HF_TOKEN`/`WANDB_API_KEY` [18].
- **Legacy SFT (`finetune.py`) metric names** [19]: `learning_rate`, `train_loss`, `total_tokens`, `per_device_tps`, `total_tokens_including_padding`, `per_device_tps_including_padding`, logged every `--logging_steps`.
- **Legacy DPO (`dpo_tune_cache.py`) metric names** [20]: `training_step`, `learning_rate`, `epoch`, `train_loss`, `logps/chosen`, `logps/rejected`, plus `rewards/chosen`, `rewards/rejected`, `rewards/average`, `rewards/accuracy`, `rewards/margin` for the `dpo`/`dpo_norm` loss types, and `aux_loss` when load-balancing loss is enabled (OLMoE). The docs do not publish a separate metrics list for the recommended OLMo-core `dpo.py` or `olmo_core_finetune.py` scripts on their algorithm pages [5][6] - only the legacy scripts' pages enumerate metric names, so treat the OLMo-core scripts' exact logged-metric set as unconfirmed by docs and check a run's own W&B config.
- **`grpo_fast.py` metric names** - the docs' training-metrics list sits entirely under the `grpo_fast.py` section, with no separate metrics section documented for `grpo.py` (the two scripts share config classes and flags per the docs, but that shared-flags statement does not extend to metrics) [7]: `episode`, `lr`, `epoch`; the `objective/*` family (`kl`, `scores`, `rlhf_reward`, `non_score_reward`, `entropy`, `loss`, `kl2`, `kl3`, `scores_mean`, `reward_std`, `verifiable_correct_rate`); `loss/policy_avg`; the `policy/*` family (`approxkl_avg`, `clipfrac_avg`, `entropy_avg`); `time/from_scratch`, `time/training`; the `val/*` family (`sequence_lengths`, `num_stop_token_ids`, `ratio`, `ratio_var`, `stop_token_rate`, `format_scores` - only when `--add_r1_style_format_reward` is on); plus two `grpo_fast.py`-specific fields, `other/real_batch_size_ratio`, the fraction of samples in a batch that still produce a nonzero gradient (a batch shrinks because a group with all-correct or all-incorrect rollouts has zero reward-std, producing a zero-divide-guarded advantage of exactly 0), and `other/packed_ratio`, the fraction of forward passes needed after sequence packing versus without it [7]. Treat `grpo.py`'s exact logged-metric set as unconfirmed by docs.
- **Reward modeling metric names** [8]: `episode`, `epoch`, `train/rm/accuracy`, `train/rm/loss`, `train/rm/chosen_rewards`, `train/rm/rejected_rewards`, `train/rm/reward_margin`, `train/rm/lr`, plus an `eval/rm/*` mirror of the same fields.
- **Evaluation during training**: DPO and SFT's recommended scripts expose `--try_launch_beaker_eval_jobs` (DPO default `True`) to launch AI2's internal Beaker eval jobs after training, plus `--oe_eval_tasks` to name them [6] - this is AI2-cluster-specific and not a generic held-out-loss eval loop.
- **Stopping**: none of the algorithm docs pages read for this card ([5], [6], [7], [8]) publish an early-stopping threshold, patience value, or other RL-specific stopping rule; GRPO's only duration-shaped fields are `--total_episodes` (default 100000) and `--num_epochs` (default 1), which cap a run rather than stop it on a signal [7].

## Save it

- Full-precision saves from the Accelerate-based scripts (`finetune.py`, `dpo_tune_cache.py`) are written by a shared `save_with_accelerate()` helper, which calls `unwrapped_model.save_pretrained(..., safe_serialization=False)` with an explicit code comment that safetensors is not used "for now" - checkpoints on disk are legacy sharded `.bin` files (e.g. `pytorch_model-00001-of-00004.bin` plus `pytorch_model.bin.index.json`, `config.json`, tokenizer files), confirmed by the directory listings on the trained-model-location docs page [21][22]. Periodic checkpoints during training are written with `accelerator.save_state()` into `step_N`/`epoch_N` directories under `--output_dir`, cadence set by `--checkpointing_steps`; `--resume_from_checkpoint <path>` (or `True` for the latest) resumes model, optimizer, and scheduler state from one of these directories [22].
- LoRA runs on the legacy scripts save only the PEFT adapter via `PeftModel.save_pretrained()`, not a merged full model [21]; a separate `open_instruct/merge_lora.py` script merges an adapter (including QLoRA dequantization through bitsandbytes) into a full checkpoint afterward [23].
- The recommended OLMo-core scripts (`olmo_core_finetune.py`, `dpo.py`) save through OLMo-core's own native distributed checkpointer (`CheckpointerConfig`, `save_folder=--output_dir`), cadence set by `--checkpointing_steps` (default 500) with a separate, more frequent `--ephemeral_save_interval` (default 500, must be <= `checkpointing_steps`) and `--keep_last_n_checkpoints` (default 3) controlling retention [24][11]. This is an OLMo-core distributed-checkpoint format, not directly an HF `from_pretrained`-loadable directory - the repo ships `scripts/train/convert_olmo_core_to_hf.py`, which loads the OLMo-core checkpoint's state dict and calls `olmo_core_utils.save_state_dict_as_hf()` to convert it into a standard HF `save_pretrained()` directory before it can be loaded with `transformers` [24].
- Trained models are additionally uploaded to up to four places for redundancy when run on AI2's cluster: the Hugging Face Hub (a revision under `allenai/open_instruct_dev`), Google Cloud Storage (`gs://ai2-llm/post-training/...`), an AI2 Beaker dataset, and ephemeral local/NFS storage (`mason.py` overwrites `--output_dir` to `/weka/oe-adapt-default/allennlp/deletable_checkpoint/$beaker_user/`) - all four hold the same `.bin`-sharded HF-format directory [22].
- Loader handoff: a full (non-LoRA) legacy-script checkpoint directory is directly `from_pretrained`-loadable by `transformers` (non-safetensors `.bin` shards); an OLMo-core checkpoint directory is not, and must go through `convert_olmo_core_to_hf.py` first; a LoRA output directory is an adapter only and needs `merge_lora.py` or a PEFT loader paired with the base model, not a standalone `from_pretrained` call.

## Find it in the docs

The docs are the live source; this card teaches the lookup, not the content itself.

- Address pattern: `https://allenai.github.io/open-instruct/<page>/` - unversioned (no version-tag path segment), built from the `main` branch via mkdocs-material; verified 2026-08-11 that `https://allenai.github.io/open-instruct/algorithms/grpo/` loads (HTTP 200) [25]. There is no `main`/tag switcher like trl's docs - what you see is whatever was last deployed from `main`.
- Page-slug recipe: the nav groups pages under `Get Started` (`get_started/installation`, `get_started/ai2_internal_setup`), `Models` (`olmo3`, `olmo2`, `tulu3`, `tulu1_tulu2`), `Training` (`algorithms/dataset_transformation`, `algorithms/trained_model_location`, `algorithms/finetune`, `algorithms/dpo`, `algorithms/grpo`, `algorithms/olmo_core_sharding`, `algorithms/tool_training`, `algorithms/reward_modeling`, `verify-tokenization`, `slack-alerts`), and `Not Maintained` (`algorithms/synthetic_preference_dataset`) [10]. Each algorithm page is the full story for that method: variants, debug-script table, key-flags table, and (for the legacy scripts) a training-metrics list.
- Question-to-page map: dataset shape and mixing -> `algorithms/dataset_transformation`; where a finished model ends up and how to fetch it -> `algorithms/trained_model_location` [22]; sharding/parallelism knobs shared by SFT/DPO/GRPO -> `algorithms/olmo_core_sharding` [11]; Slack notifications -> `slack-alerts` [18]; AI2-internal cluster setup -> `get_started/ai2_internal_setup`.
- Runnable references beyond the docs: the `scripts/train/debug/` tree (single-GPU and multi-node smoke tests for each method, launched via `bash` locally or `./scripts/train/build_image_and_launch.sh` on Beaker) and the `scripts/train/tulu3/` and `scripts/train/olmo3/` trees, which are the exact scripts behind the published Tulu 3 and Olmo 3 checkpoints [3][12][6].
- No community-tutorials page exists in the docs nav read for this card [10]; the closest curated pointer is the README's own Acknowledgements section, which names the codebases open-instruct's algorithms were adapted from (Hugging Face TRL and Eric Mitchell's DPO code for preference tuning; OpenAI's `lm-human-preferences`/`summarize-from-feedback` and vwxyzjn's reproduction for the PPO code; OpenRLHF for the Ray+vLLM distributed pattern used to scale PPO/RLVR to 70B) [3]. No official MCP endpoint for these docs was found in the pages read for this card.
- Trap: the docs' own `get_started/installation` page is stale against the current `pyproject.toml` - it documents a Python 3.10 / `torch==2.5.1` / `flash-attn==2.7.2.post1` pip-based setup and a `uv sync --extra compile` form, while the release and push-commit `pyproject.toml` require Python 3.12, `torch>=2.10.0`, and route flash-attn through direct dependency pins rather than an `extra` [16][14][15] - trust `pyproject.toml` and the README's plain `uv sync`, not that page.
- Honest boundary: open-instruct is explicitly framed by its own README as a research codebase without a backward-compatibility guarantee [3]; several algorithm variants documented on its own pages are themselves marked legacy in favor of the OLMo-core scripts (`finetune.py` carries a runtime `DeprecationWarning` pointing at the OLMo-core SFT script) [9], and one whole page (`algorithms/synthetic_preference_dataset`) is filed under the docs nav's own "Not Maintained" section [10].

## Sources

[1] open-instruct docs index (Overview). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/docs/index.md. Fetched 2026-08-11 at commit b18cfc5.

[2] open-instruct GitHub repository metadata (description, license, stars, pushed_at, archived). https://github.com/allenai/open-instruct. Fetched via GitHub API 2026-08-11.

[3] open-instruct README.md. https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/README.md. Fetched 2026-08-11 at commit b18cfc5.

[4] open_instruct/utils.py (Beaker/mason helper functions referenced for the launcher). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/open_instruct/utils.py. Fetched 2026-08-11 at commit b18cfc5.

[5] docs/algorithms/finetune.md (SFT variants, key flags, legacy metrics). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/docs/algorithms/finetune.md. Fetched 2026-08-11 at commit b18cfc5.

[6] docs/algorithms/dpo.md (DPO variants, key flags, legacy metrics). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/docs/algorithms/dpo.md. Fetched 2026-08-11 at commit b18cfc5.

[7] docs/algorithms/grpo.md (GRPO variants, key flags, scripts table, training metrics). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/docs/algorithms/grpo.md. Fetched 2026-08-11 at commit b18cfc5.

[8] docs/algorithms/reward_modeling.md. https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/docs/algorithms/reward_modeling.md. Fetched 2026-08-11 at commit b18cfc5.

[9] open_instruct/finetune.py (`__main__` DeprecationWarning). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/open_instruct/finetune.py. Fetched 2026-08-11 at commit b18cfc5.

[10] mkdocs.yml (docs site nav). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/mkdocs.yml. Fetched 2026-08-11 at commit b18cfc5.

[11] docs/algorithms/olmo_core_sharding.md (HSDP mechanism, per-algorithm parallelism flags). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/docs/algorithms/olmo_core_sharding.md. Fetched 2026-08-11 at commit b18cfc5.

[12] docs/algorithms/grpo.md, Olmo 3 Scripts table and `--single_gpu_mode`/`--vllm_gpu_memory_utilization` flags. Same source as [7].

[13] PyPI lookup for `open-instruct` (404 Not Found). https://pypi.org/pypi/open-instruct/json. Fetched 2026-08-11.

[14] pyproject.toml at release tag v0.3.0, resolved to commit `26de99604002d6a7b25a2c5f2ee880e388a21dca` via the GitHub API tag ref. https://raw.githubusercontent.com/allenai/open-instruct/26de99604002d6a7b25a2c5f2ee880e388a21dca/pyproject.toml. Fetched 2026-08-11.

[15] pyproject.toml at commit b18cfc57c07c37f6a90d8b86e4c2b24934944b50 (the shortlist's pinned push commit, ahead of the v0.3.0 release; `cuda12`/`cuda13` dependency-group restructuring) and requirements.txt at the same commit (`uv export ... --group cuda12` in its autogenerated header). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/pyproject.toml and .../requirements.txt. Fetched 2026-08-11.

[16] docs/get_started/installation.md. https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/docs/get_started/installation.md. Fetched 2026-08-11 at commit b18cfc5.

[17] GitHub Releases API listing for allenai/open-instruct (v0.1.0, v0.2.0, v0.3.0 with dates). https://api.github.com/repos/allenai/open-instruct/releases. Fetched 2026-08-11.

[18] docs/slack-alerts.md. https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/docs/slack-alerts.md. Fetched 2026-08-11 at commit b18cfc5.

[19] docs/algorithms/finetune.md, Training Metrics section (legacy `finetune.py`). Same source as [5].

[20] docs/algorithms/dpo.md, Training Metrics section (legacy `dpo_tune_cache.py`). Same source as [6].

[21] open_instruct/model_utils.py, `save_with_accelerate()` function (safetensors-disabled save path, LoRA adapter-only save path). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/open_instruct/model_utils.py. Fetched 2026-08-11 at commit b18cfc5.

[22] docs/algorithms/trained_model_location.md (four upload destinations, checkpoint directory file listings, `mason.py` local-storage overwrite path). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/docs/algorithms/trained_model_location.md. Fetched 2026-08-11 at commit b18cfc5.

[23] open_instruct/merge_lora.py (LoRA/QLoRA adapter merge script). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/open_instruct/merge_lora.py. Fetched 2026-08-11 at commit b18cfc5.

[24] open_instruct/olmo_core_finetune.py and open_instruct/olmo_core_utils.py (OLMo-core `CheckpointConfig` defaults, `save_state_dict_as_hf`) and scripts/train/convert_olmo_core_to_hf.py (the HF conversion script). https://raw.githubusercontent.com/allenai/open-instruct/b18cfc57c07c37f6a90d8b86e4c2b24934944b50/open_instruct/olmo_core_finetune.py, .../olmo_core_utils.py, and .../scripts/train/convert_olmo_core_to_hf.py. Fetched 2026-08-11 at commit b18cfc5.

[25] Live docs site reachability check for the unversioned address pattern. https://allenai.github.io/open-instruct/algorithms/grpo/ (HTTP 200) and https://allenai.github.io/open-instruct/. Fetched 2026-08-11.

Ecosystem tools named in passing (Ray, vLLM, DeepSpeed, Accelerate, PEFT/LoRA, bitsandbytes, Beaker, OLMo-core, W&B) are reached through [1]-[12]'s own text and are deliberately not enumerated as separate references. Method names (SFT, DPO, GRPO, PPO) are deliberately cited to nothing here; their defining papers live on the methodology cards, not this one. The four open-instruct research papers (Tulu 1/2/3, Unpacking DPO and PPO) are named only as context for "who maintains it" via [3] and are not independently fetched, since no method math from them is restated on this card.
