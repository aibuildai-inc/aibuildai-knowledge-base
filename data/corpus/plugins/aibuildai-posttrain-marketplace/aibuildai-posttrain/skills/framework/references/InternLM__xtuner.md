# xtuner

An FSDP-native training engine purpose-built for ultra-large MoE models - start a Qwen3 fine-tune on one GPU, scale the same `Trainer` to hundred-billion-parameter MoE without hand-tuned 3D parallelism.

**XTuner V1** is described in its own README as "a next-generation LLM training engine specifically designed for ultra-large-scale MoE models" [1]. It is built and maintained by the InternLM GitHub organization [2][3]. Its API is a single `Trainer` class (and its Pydantic `TrainerConfig`) that takes a model config, a dataset path, an optimizer config and an LR config, and runs `.fit()`; the same shape is exposed through CLI entry points under `xtuner/v1/train/cli/` (`sft.py` for SFT, `rl.py` for RL) launched with `torchrun` [4][30]. It lives at https://github.com/InternLM/xtuner [3].

**When to pick it**: dense or MoE post-training when the model is large enough that FSDP sharding plus optional tensor/expert/sequence parallelism is the concern - the README's stated selling point is training MoE models "above 200B scale" on FSDP without full 3D parallelism, and up to roughly 1T parameters with expert parallelism [1]. XTuner V1 is a from-scratch rewrite (announced September 2025) of the pre-existing XTuner project; the older, PyPI-published `xtuner` (LoRA/QLoRA-oriented, config-file style fine-tuning of dense LLMs and VLMs) is a separate, legacy code path still in the same repository, not this card's subject [1][6]. Method coverage is narrow next to trl or verl: only SFT/pretrain and GRPO are implemented today, with several RL methods "coming soon" [1] - pick XTuner V1 for large-MoE FSDP training or GRPO, not for a wide menu of offline/preference methods.

**Methods it ships**: the README's own Algorithm section lists, as implemented: multimodal pre-training, multimodal supervised fine-tuning, and GRPO (citing arXiv:2402.03300); as "Coming Soon": MPO (arXiv:2411.10442), DAPO (arXiv:2503.14476), and multi-turn agentic RL [1]. RL is explicitly marked Beta: the GRPO quick-start page opens with "XTuner's RL (Reinforcement Learning) functionality is currently in Beta version" [7]. Method-level math and training-signal semantics are not restated here; they belong on each method's own card. The README also tracks inference-engine integration for RL rollout as a checklist: LMDeploy is checked, vLLM and SGLang are unchecked (not yet integrated) [1].

**Scale it handles**: single GPU up to many-GPU single-node runs are demonstrated with `torchrun --nproc-per-node <N>` in both the SFT and GRPO quick starts [4][5]; parallelism dials are `FSDPConfig.tp_size` (tensor parallel) and `FSDPConfig.ep_size` (expert parallel), plus `TrainerConfig.sp_size` (DeepSpeed-Ulysses-style sequence parallel) - the trainer computes data-parallel size as `world_size / (tp_size * sp_size)` and raises if the global batch size does not divide evenly across it [8][9]. `FSDPConfig.hsdp_sharding_size` enables hybrid-sharded data parallel, but only when `ep_size == 1` [9]. No multi-node launch example (SLURM, `--nnodes`, rendezvous config) was found in the docs pages fetched for this card, so multi-node is stated here only as standard `torchrun` mechanism, not as a documented, benchmarked recipe [4][5]. The scale claims - 200B-parameter MoE trained without expert parallelism, 600B-parameter MoE needing only intra-node EP, 200B MoE at 64k sequence length without sequence parallelism, and MoE training up to about 1T parameters, plus FSDP throughput on MoE above 200B said to surpass "traditional 3D parallel schemes," and Ascend A3 Supernode efficiency said to exceed NVIDIA H800 - are all read from the README's Key Features prose; no benchmark table or reproducible script backing these specific numbers was fetched for this card, so treat them as the vendor's own claim rather than a measured one [1].

**Install**: only `git clone https://github.com/InternLM/xtuner.git && cd xtuner && pip install -e .` installs the code this card describes [10]. `pip install xtuner` instead installs the last PyPI release, version 0.2.0 (uploaded 2025-07-11), which is the pre-V1 legacy package, not XTuner V1 - PyPI has not published a V1 release [11]. The newest git tag is `v1.0.1` (published 2026-05-15, commit `91e30dd1636d591648ca1fb24954f4b1e280dc26`); at that tag's `pyproject.toml`: Python `>=3.10`, license Apache-2.0, and load-bearing pins `torch>=2.6.0`, `transformers==5.2.0` (exact pin, easy to collide with an existing environment), `mmengine==0.11.0rc2`, `bitsandbytes==0.45.0`, `datasets<4.0.0`, `peft>=0.14.0` [12]. RL adds dependencies through one of two install paths that do not list the same packages: `pip install -e '.[rl]'` pulls the `rl` extra defined in `pyproject.toml` at the `v1.0.1` tag - `ray[default]`, `httpx`, `fastapi`, `uvicorn`, `mathruler`, `pylatexenc`, all unpinned [12] - while `pip install -r requirements/rl.txt` at the same tag's commit installs only `ray[default]` and `httpx` [13]. Either way, RL rollout also needs a separately installed inference engine (LMDeploy is the one documented) [10]. No CUDA version floor is stated in the fetched install docs; they require only an NVIDIA driver newer than `550.127.08` and recommend installing `flash-attn` (and `flash-attn-3` for RL) [10]. MoE training additionally recommends InternLM's own `GroupedGEMM`, and FP8 MoE training additionally requires `AdaptiveGEMM`, both installed from GitHub via `pip install git+...` rather than pinned in `pyproject.toml` [10].

**Maintained by**: the InternLM GitHub organization [2][3]; the repository shows 5,175 GitHub stars (not a ranking signal) [3]; it is actively pushed - the pinned commit's own push timestamp is 2026-07-31, and the repository's HEAD had already moved past that pin to 2026-08-10 by the time this card was written, with `v1.0.1` (2026-05-15) as the newest tagged release [2][14][15].

## Quick start

Two complete, quoted quick starts from the docs [4][5]:

SFT (dense LLM), after `huggingface-cli download Qwen/Qwen3-8B --local-dir <model_path>` and preparing an OpenAI-format `jsonl` dataset [4]:

```bash
torchrun --nproc-per-node 8 xtuner/v1/train/cli/sft.py --load-from <model_path> --chat_template qwen3 --dataset <dataset_path> --total-step 100 --work-dir <target_work_directory>
```

GRPO (Beta RL), after downloading a model the same way and converting GSM8K with the repo's own `xtuner/v1/utils/convert_gsm8k.py` script (a ready example dataset also ships at `tests/resource/gsm8k_train_example_data.jsonl`) [5], the docs page's own command is:

```bash
XTUNER_USE_FA3=1 XTUNER_USE_LMDEPLOY=1 python xtuner/v1/train/cli/grpo.py --model-path <model_path> --data-path tests/resource/gsm8k_train_example_data.jsonl
```

This command does not run at the commit this card pins: `xtuner/v1/train/cli/grpo.py` is absent from that commit's tree, and the file it names was replaced by `xtuner/v1/train/cli/rl.py` in the repository's own history well before the pin - the docs page has drifted out of sync with the code [5][29][30]. `rl.py` as it exists at the pinned commit takes only a `--config <path>` argument (no `--model-path`/`--data-path`), building its `Trainer` from a Python config file the way the SFT `--config` path does [30][17]; this card cannot show a concrete, runnable GRPO config file because none was fetched, so treat the RL quick start as unverified against the pinned commit and re-check the live docs' current CLI path before running it.

The Trainer tutorial's own "equivalent minimal Python program" is itself incomplete in the source: it defines `dataset_cfg = []` and then a bare, unfinished `dataloader_cfg =` line before passing `dataloader_cfg=dataloader_cfg` into `Trainer(...)` - the tutorial page never assigns a value to that variable, so the snippet as published does not run [16]. This card does not repair or complete it; treat the Trainer-class form as illustrative of the constructor's field names, not as copy-paste-runnable code, and use the `torchrun ... sft.py` form above for an actually runnable quick start.

## Start it

- One GPU or many GPUs on one node both use the same `torchrun --nproc-per-node <N>` form shown above; no multi-node launch template (SLURM script, `--nnodes`/rendezvous config) was found in the docs pages fetched for this card [4][5].
- A CLI training entry accepts either command-line flags or a Python config file passed as `--config <path>`, and the two are mutually exclusive; the config file builds a `TrainerConfig` from Python objects (`Qwen3Dense8BConfig`, `AdamWConfig`, `LRConfig`, ...) rather than YAML [17].
- Parallelism is set on `FSDPConfig` (`tp_size`, `ep_size`, `reshard_after_forward`, `hsdp_sharding_size`) and on `TrainerConfig.sp_size`; the trainer derives data-parallel size as `world_size / (tp_size * sp_size)` and requires `global_batch_size` to divide evenly across it, then sets `micro_batch_size = global_batch_size / dp_size` - there is no separate gradient-accumulation-steps knob in this arithmetic [8][9].
- XTuner V1 changes two defaults from a plain PyTorch/FSDP2 setup: `FSDPConfig.param_dtype` and `reduce_dtype` both default to `torch.bfloat16` (a silent bf16-capable-GPU assumption), and `FSDPConfig.torch_compile` defaults to `True` [9].
- OOM first aid on the training side: the SFT quick-start page's own tip is `--fsdp-config.cpu-offload`, which flips `FSDPConfig.cpu_offload` (default `False`) to move sharded state to CPU [4][9]; gradient checkpointing is tunable via `FSDPConfig.recompute_ratio` (default `1.0`, i.e. full recompute) [9]. No generation-side (rollout) memory-tuning knob was found in the RL docs pages fetched for this card; `RolloutConfig`/`DataflowConfig` fields documented there govern rollout concurrency (`rollout_max_batch_size_per_instance`, `DataflowConfig.max_concurrent`), not memory [18].

## Watch it

Mechanics only - what a given metric means for GRPO specifically lives on that method's card, not here.

- **Enable it**: `TrainerConfig.exp_tracker` selects the backend and only accepts `"jsonl"` (default) or `"tensorboard"`; no built-in Weights & Biases integration was found in the trainer source read for this card [8]. Output goes under the run's `log_dir` at `exp_tracking/rank<N>/` [8].
- **SFT/pretrain metric names**, read from the trainer's per-step logging call (`xtuner/v1/train/trainer.py`, pinned commit `4d7e23d61e6b27467be7366e97d1e3540da73cd9`): `lr`, `time/data_time`, `time/step_time`, `time/train_time`, `time/eta_seconds`, `runtime_info/text_tokens`, `runtime_info/seqlen_tokens`, `runtime_info/approximate_total_consumed_tokens`, `runtime_info/tgs` (tokens/GPU/second), `runtime_info/seqlen_tgs`, `runtime_info/exp_tgs`, `runtime_info/efficient_attn_ratio`, `runtime_info/img_efficient_attn_ratio`, `memory/max_memory_GB`, `memory/reserved_memory_GB`, `grad_norm`, plus a `loss/<name>` entry for every key the model returns in its loss dict [19]. The console log line additionally prints `tgs`, `seqlen_tgs`, `exp_tgs` and an ETA string [19].
- **GRPO** adds a policy-gradient loss config with defaults `cliprange_high=0.2`, `cliprange_low=0.2`, `loss_type="vanilla"`, `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type="low_var_kl"` [20]; this card does not enumerate GRPO's own logged scalar names because the RL trainer/logging pages fetched for this card did not render a metrics list distinct from the SFT one above - check `docs/en/rl/tutorial/rl_grpo_trainer.md` on the live docs site directly before a GRPO run.
- **Sample-level logging**: no `log_completions`-style flag or generation-sample logger was found in the RL docs and source read for this card.
- **Evaluation during training**: no `eval_dataset`/`eval_steps` fields were found in `TrainerConfig` (`xtuner/v1/train/trainer.py`) as read at the pinned commit; evaluation-during-training is not confirmed as a built-in feature by the sources fetched for this card [8].
- **Health check and stopping**: `TrainerConfig.check_health_interval` (default `None`) periodically calls a `check_health()` function and raises `RuntimeError("Health check failed, exit training")` if it fails, at the pinned commit [21]; the specific condition `check_health()` tests was not read for this card. A separate debug hook can be registered that logs a warning whenever a module's forward output contains NaN [21]. No RL reward-curve or KL-based automatic stopping threshold was found in the sources fetched for this card.

## Save it

- Two distinct checkpoint formats exist at the pinned commit (`xtuner/v1/engine/train_engine.py`, `xtuner/v1/model/base.py`): `save_dcp`/`async_save_dcp` write a PyTorch Distributed Checkpoint (model plus, by default, optimizer state) under `<exp_dir>/checkpoints/ckpt-step-<N>/`, intended for resuming training; `save_hf`/`async_save_hf` write a standard Hugging Face directory (`config.json`, sharded `.safetensors` files, `model.safetensors.index.json`) under `<exp_dir>/hf-<N>/`, intended for deployment/evaluation and containing no optimizer state [22][23][24].
- Cadence and retention on `TrainerConfig`: `checkpoint_interval`/`checkpoint_maxkeep` govern the DCP checkpoints; `hf_interval`/`hf_max_keep` govern the HF exports separately. Whether HF export is possible at all is gated by `model_cfg.hf_config is not None or <model loaded from an HF path>`, not by `hf_max_keep`; if that gate fails, `hf_interval`/`hf_max_keep`/`async_hf_export` must all be left unset [8][22]. When `hf_max_keep` is exceeded, the oldest HF export directories are deleted, and a symlink named `hf-latest` under the run's experiment directory is repointed to the newest export [22].
- Resume: `TrainerConfig.auto_resume=True` (or `load_checkpoint_cfg.checkpoint_path` for an explicit path) resumes from a DCP checkpoint; `LoadCheckpointConfig` fields `load_optimizer_states`, `load_optimizer_args`, `load_dataset`, `load_scheduler` (all default `True`) control what gets restored [8]. A working directory also carries a `.xtuner` metadata file that the SFT quick-start page tells the reader to inspect after a run [4][8].
- A saved `hf-<N>/` directory is a full, standard HF model directory, not an adapter - the source read for this card found no LoRA/PEFT-adapter-only save path in XTuner V1's `save_hf` [23][24]. Whether an evaluator can load an `hf-<N>/` checkpoint directly is that evaluator's own loader contract; this card only confirms the on-disk shape is `from_pretrained`-compatible HF format [23].

## Find it in the docs

- Address pattern: `https://xtuner.readthedocs.io/en/<version>/<page>.html` (Chinese docs are the same host under `/zh-cn/`). Checked 2026-08-10: `/en/latest/` and `/zh-cn/latest/` both load; `/en/v1.0.1/` returns 404 - ReadTheDocs is publishing this project only as an unpinned `latest` build, not a per-tag docs snapshot, so a docs page read today may describe code ahead of whatever release you installed [25]. `pyproject.toml`'s own `Documentation` URL points only at the Chinese `latest` build [12].
- Page-slug recipe: Getting Started pages live at `get_started/<page>` (`installation`, `sft`, `mllm_sft`, `grpo`); tutorials live at `pretrain_sft/tutorial/<page>` and `rl/tutorial/<page>`; deeper material is under `pretrain_sft/advanced_tutorial/` and `rl/advanced_tutorial/` (e.g. `float8.md`, `profile.md`, `loss.md`, `model.md`, `efficiency.md`); a from-Megatron tuning guide sits at `benchmark/megatron_moe_benchmark` - note this page benchmarks Megatron/Pai-Megatron-Patch, not XTuner's own FSDP path, so do not read its numbers as XTuner benchmarks [26]. A parallel `legacy_index` toctree entry documents the pre-V1 XTuner [27].
- Question-to-slug map: dataset format -> `get_started/sft` (SFT section "Prepare Dataset") and `rl/tutorial/...` for the RL record shape (adds `reward_model.ground_truth`, `data_source`, `ability`) [4][5]; command-line vs. config-file training -> `pretrain_sft/tutorial/config`; hand-building a `Trainer` in Python -> `pretrain_sft/tutorial/llm_trainer` [17][16].
- Runnable references beyond the docs: the `tests/resource/gsm8k_train_example_data.jsonl` file in the repo is the ready-made GRPO smoke-test dataset named directly in the GRPO quick start [5]; a `xtuner/v1/utils/convert_gsm8k.py` script converts the Hugging Face `gsm8k` dataset into the RL record format [5].
- No community-tutorials curation page (of the kind trl publishes) was found among the docs pages fetched for this card; the closest official pointer to worked examples is the repo's own `examples/` tree entries seen in the file listing (e.g. `examples/v1/config/sft_intern_s1_tiny_config.py`) [28].
- No official MCP endpoint for these docs was found in the sources fetched for this card.
- Honest boundary: as of the pinned commit, RL rollout only integrates LMDeploy as inference engine - vLLM and SGLang are unchecked items in the README's own roadmap [1]; RL itself is labeled Beta on its own quick-start page [7]; and only SFT/pretrain and GRPO are marked implemented, with DPO-family and other preference methods absent from both the implemented and "coming soon" lists [1].

## Sources

All pages are `main`/`latest`-branch docs or live pages unless a tag or commit is named; page fetches are dated 2026-08-10. Method papers named in passing (GRPO, MPO, DAPO) are cited to the README that lists them, not to the papers themselves, since method math belongs on each method's own card.

[1] xtuner README, pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9. https://github.com/InternLM/xtuner/blob/4d7e23d61e6b27467be7366e97d1e3540da73cd9/README.md. Fetched 2026-08-10.

[2] xtuner GitHub repository metadata (GitHub REST API). https://api.github.com/repos/InternLM/xtuner. Fetched 2026-08-10.

[3] xtuner GitHub repository. https://github.com/InternLM/xtuner. Fetched 2026-08-10.

[4] XTuner SFT quick start. https://xtuner.readthedocs.io/en/latest/get_started/sft.html. Fetched 2026-08-10.

[5] XTuner GRPO quick start (Beta). https://xtuner.readthedocs.io/en/latest/get_started/grpo.html. Fetched 2026-08-10.

[6] xtuner GitHub releases list (GitHub REST API) - shows the pre-V1 release history (v0.1.x) alongside V1's v1.0.0rc0/v1.0.1. https://api.github.com/repos/InternLM/xtuner/releases. Fetched 2026-08-10.

[7] XTuner GRPO quick start Beta notice - same page as [5].

[8] `xtuner/v1/train/trainer.py`, `TrainerConfig` and `LoadCheckpointConfig` class definitions, pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9. https://github.com/InternLM/xtuner/blob/4d7e23d61e6b27467be7366e97d1e3540da73cd9/xtuner/v1/train/trainer.py. Fetched 2026-08-10.

[9] `xtuner/v1/config/fsdp.py`, `FSDPConfig` class definition, pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9. https://raw.githubusercontent.com/InternLM/xtuner/4d7e23d61e6b27467be7366e97d1e3540da73cd9/xtuner/v1/config/fsdp.py. Fetched 2026-08-10.

[10] XTuner installation guide. https://xtuner.readthedocs.io/en/latest/get_started/installation.html. Fetched 2026-08-10.

[11] xtuner on PyPI (JSON API) - `info.version` 0.2.0, uploaded 2025-07-11. https://pypi.org/pypi/xtuner/json. Fetched 2026-08-10.

[12] `pyproject.toml` at git tag `v1.0.1` (commit 91e30dd1636d591648ca1fb24954f4b1e280dc26), the newest tagged release as of this card. https://raw.githubusercontent.com/InternLM/xtuner/v1.0.1/pyproject.toml. Fetched 2026-08-10.

[13] `requirements/rl.txt` at commit 91e30dd1636d591648ca1fb24954f4b1e280dc26. https://raw.githubusercontent.com/InternLM/xtuner/91e30dd1636d591648ca1fb24954f4b1e280dc26/requirements/rl.txt. Fetched 2026-08-10.

[14] xtuner GitHub repository metadata `pushed_at` field - same source as [2].

[15] xtuner GitHub tags list (GitHub REST API), showing `v1.0.1` -> commit 91e30dd1636d591648ca1fb24954f4b1e280dc26. https://api.github.com/repos/InternLM/xtuner/tags. Fetched 2026-08-10.

[16] XTuner "Fine-tuning Large Models with Trainer" tutorial. https://xtuner.readthedocs.io/en/latest/pretrain_sft/tutorial/llm_trainer.html. Fetched 2026-08-10.

[17] XTuner "Training Configuration" tutorial (CLI flags vs. Python config file). https://xtuner.readthedocs.io/en/latest/pretrain_sft/tutorial/config.html. Fetched 2026-08-10.

[18] XTuner RL rollout-concurrency tuning guide (`RolloutConfig`, `DataflowConfig` fields). https://xtuner.readthedocs.io/en/latest/rl/advanced_tutorial/efficiency.html. Fetched 2026-08-10.

[19] `xtuner/v1/train/trainer.py`, `_log_step` method (console log line and `log_scalars` dict), pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9 - same file as [8].

[20] XTuner "Customize GRPO Training" tutorial (`GRPOLossConfig` defaults). https://xtuner.readthedocs.io/en/latest/rl/tutorial/rl_grpo_trainer.html. Fetched 2026-08-10.

[21] `xtuner/v1/train/trainer.py`, `_maybe_check_health` and `_register_debug_hook` methods, pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9 - same file as [8].

[22] `xtuner/v1/train/trainer.py`, `_maybe_save_hf`/`_finalize_dcp_save` methods and checkpoint-directory properties, pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9 - same file as [8].

[23] `xtuner/v1/model/base.py`, `save_hf` method (HF directory contents: `config.json`, sharded safetensors, `model.safetensors.index.json`), pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9. https://raw.githubusercontent.com/InternLM/xtuner/4d7e23d61e6b27467be7366e97d1e3540da73cd9/xtuner/v1/model/base.py. Fetched 2026-08-10.

[24] `xtuner/v1/engine/train_engine.py`, `save_hf`/`save_dcp`/`async_save_hf`/`async_save_dcp` methods, pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9. https://raw.githubusercontent.com/InternLM/xtuner/4d7e23d61e6b27467be7366e97d1e3540da73cd9/xtuner/v1/engine/train_engine.py. Fetched 2026-08-10.

[25] ReadTheDocs version-URL check: `https://xtuner.readthedocs.io/en/latest/` (200), `https://xtuner.readthedocs.io/zh-cn/latest/` (200), `https://xtuner.readthedocs.io/en/v1.0.1/` (404). Checked 2026-08-10.

[26] XTuner Megatron MoE benchmark and tuning guide (benchmarks Pai-Megatron-Patch, not XTuner's own FSDP path). https://xtuner.readthedocs.io/en/latest/benchmark/megatron_moe_benchmark.html. Fetched 2026-08-10.

[27] XTuner docs master table of contents (`docs/en/index.rst`), pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9 - lists the `get_started`, `pretrain_sft`, `rl`, `benchmark` and `legacy` toctrees. https://raw.githubusercontent.com/InternLM/xtuner/4d7e23d61e6b27467be7366e97d1e3540da73cd9/docs/en/index.rst. Fetched 2026-08-10.

[28] xtuner GitHub repository file tree (GitHub Trees API), pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9. https://api.github.com/repos/InternLM/xtuner/git/trees/4d7e23d61e6b27467be7366e97d1e3540da73cd9?recursive=1. Fetched 2026-08-10.

[29] GitHub commit history for `xtuner/v1/train/cli/grpo.py` (GitHub REST API), showing commit f535ab5 ("[Refactor] Refactor RL entrypoint (#1156)", 2025-10-24) as the last commit touching that path before it was removed, months before the July 2026 pin. https://api.github.com/repos/InternLM/xtuner/commits?path=xtuner/v1/train/cli/grpo.py. Fetched 2026-08-10.

[30] `xtuner/v1/train/cli/rl.py`, the RL CLI entrypoint that exists at the pinned commit in place of `grpo.py`, pinned commit 4d7e23d61e6b27467be7366e97d1e3540da73cd9. https://raw.githubusercontent.com/InternLM/xtuner/4d7e23d61e6b27467be7366e97d1e3540da73cd9/xtuner/v1/train/cli/rl.py. Fetched 2026-08-10.
