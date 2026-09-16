# NeMo RL

NVIDIA's Ray-orchestrated post-training library: one YAML config per method, DTensor or Megatron Core for training, vLLM/SGLang/Megatron-native for generation, scaling from one GPU to hundreds of nodes.

**NeMo RL**'s README titles itself "NeMo RL: A Scalable and Efficient Post-Training Library" [1], and its packaging metadata repeats that title and extends it: "NeMo RL: A Scalable and Efficient Post-Training Library for Models Ranging from 1 GPU to 1000s, and from Tiny to >100B Parameters" [2]. It is built and maintained by NVIDIA under the NVIDIA-NeMo GitHub organization, with named maintainers Yi-Fu Wu, Terry Kong, Yuki Huang, and an NVIDIA nemo-toolkit contact in its release metadata [2]. Its API is YAML-driven: each method is a runnable example script (`examples/run_grpo.py`, `examples/run_sft.py`, ...) paired with a default config file, launched through `uv run` and overridden on the command line with dotted keys [3]. It lives at https://github.com/NVIDIA-NeMo/RL [4].

**When to pick it**: post-training when you want NVIDIA's own Megatron Core parallelisms (TP/PP/CP/SP/EP/FSDP) for large or long-context models, or a PyTorch-native DTensor path for smaller experiments, with generation offloaded to vLLM, SGLang, or a weight-conversion-free Megatron-native inference path [5][6]. Ray, a cluster scheduler, places every actor (training workers, generation workers) and is required infrastructure even for a single GPU [4][6]. Its method menu includes methods not covered on the trl card (GSPO, DAPO, CISPO, GDPO, on-policy and cross-tokenizer off-policy distillation) [7]; unlike trl's Accelerate-launched single-machine-first model, NeMo RL's SLURM/Kubernetes multi-node path is a first-class, documented launch form, not a secondary scaling story (cross-reference to the trl card; not covered here).

**Methods it ships** [7][8]: GRPO, GSPO, DAPO, CISPO, GDPO (Group reward-Decoupled Normalization Policy Optimization, added 2026-03-12) and PPO for online RL; SFT (with LoRA) and DPO for offline post-training; RM (reward modeling); On-policy Distillation and X-Token (cross-tokenizer) Off-Policy Distillation; Multi-Turn RL and Async RL as training modes layered on GRPO. The README's own Features table uses a flat two-state legend, an "Available now" mark versus a "Coming in v0.7" mark, with no distinction of how recently an "Available now" item shipped; under that legend, X-Token Off-Policy Distillation, GDPO, LoRA for GRPO/DPO, SGLang inference, speculative decoding, and Muon optimizer support are all marked "Available now", while "Improved Native Performance", "Improved Large MoE Performance", "Resiliency", and on-policy cross-tokenizer distillation are marked "Coming in v0.7" [7]. Recheck the live Features table before depending on any "Available now" item, since it moves. LoRA for SFT, GRPO, and DPO is supported on both the DTensor and Megatron Core backends as of the 2026-02-04 README entry [7]. The taxonomy moves; the live page for it is the README's Features section and the docs' `about/algorithms/index.html` page [7][9].

**Scale it handles**: single GPU (`uv run python examples/run_grpo.py`) up to multi-node SLURM clusters launched via `sbatch` against the repo's `ray.sub` script, with `cluster.num_nodes` and `cluster.gpus_per_node` set on the command line and a documented `--gres=gpu:4` variant for 4-GPU-per-node GB200 systems; Kubernetes is also documented as a cluster target [10][11]. Two training-side sharding options: DTensor (PyTorch-native FSDP2, TP, SP, CP, PP) and Megatron Core (6D parallelism: TP/PP/CP/SP/EP/FSDP) [5]. The README's own worked example — GRPO on Qwen2.5-32B across 32 nodes at 16k sequence length, combining vLLM tensor-parallel=4 for generation with DTensor tensor-parallel=8, sequence-parallel, and activation-checkpointing for training — is a documented mechanism and command form, not itself a published throughput number [12], but the README's v0.6.0 News entry links to a separate performance-summary page that does publish reference-run numbers on DGX-H100 (BF16, Megatron Core backend): on-policy GRPO on DeepSeek V3 at 512 GPUs reaches 7.24 tokens/sec/GPU with a 111 s total step time (the 256-GPU on-policy row on the same table reaches 12.1 tokens/sec/GPU at 134 s), and on-policy GRPO on Qwen3-235B at 256 GPUs reaches 37.4 tokens/sec/GPU with a 312 s total step time; the same page also publishes H100 FP8 and GB200 BF16 tables [33].

**Install**: `git clone --recursive` + `uv venv` (via the `uv` package/environment manager, not raw pip); no PyPI package name is used in the README, only the Git clone [10]. Resolving the shortlist row's pinned commit (`e08fc27...`, the repository's newest push, not a release) to its nearest release: the latest tagged release is v0.7.0, released 2026-07-29, at commit `81aa43dda4765b0429cf31dab44441e4e4383911` [13][14] — the install and dependency facts below are read at that release tag, which is several days behind the shortlist's pinned push commit. `requires-python = ">=3.13.13,<3.14"`; licence "Apache 2.0" in `pyproject.toml` (Apache-2.0 per GitHub's own detection) [2][4]. Core pins at v0.7.0's `pyproject.toml`: `torch==2.11.0`, `torchvision==0.26.0`, `ray[default]>=2.55.1`, `transformers>=5.5.0,<5.9.0` (comment: "Floor raised to 5.5.0 for Gemma 4 support"), `wandb>=0.28.0`, `datasets>=4.0.0`, `accelerate>=0.26`, `sympy>=1.14.0`, `pillow>=12.3.0` (comment: "Address CVE") [2]. Optional-dependency extras load different, sometimes conflicting, generation-stack pins: `[vllm]` pins `vllm==0.20.0` and `flashinfer-python==0.6.8.post1`; `[sglang]` pins `sglang==0.5.12.post1` and a different `flashinfer-python==0.6.11.post1`, plus its own `transformers==5.6.0`; `[mcore]` pins `transformer-engine` at tag `release_v2.15`, `transformers>=5.8.1,<5.9.0`, and `megatron-core`/`megatron-bridge` [2]. No CUDA or GPU-driver minimum is stated in the README or `pyproject.toml`; the README instead calls CUDA 13 the "current primary" version when giving cuDNN install commands, which hedges rather than pins a floor [10][15]. The pre-built NGC container `nvcr.io/nvidia/nemo-rl:latest` (with tagged releases, e.g. `:v0.7.0`) bundles CUDA, cuDNN, vLLM, and SGLang and is the README's own recommended path to skip bare-metal dependency setup [10].

**Maintained by**: NVIDIA, under the NVIDIA-NeMo GitHub organization; 1,894 GitHub stars as a live count, not a ranking signal [4]. Actively developed: the GitHub Releases API lists v0.7.0 as published 2026-07-29T04:31:13Z [13], and the README's News section separately lists dated entries through 2026-06-12 (Minimax-M3 day-0 support), with v0.6.0 (2026-04-30), v0.5.0 (2026-01-31), and v0.4.0 (2025-12-01) as prior release announcements [7].

## Quick start

Smallest complete runs, quoted in form (not literally) from the README's own Quick Start and per-method sections — each launches from a clone of the repo [10][16][17][18]:

```sh
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl
uv venv
uv run python examples/run_grpo.py
```

This runs GRPO on `Qwen/Qwen2.5-1.5B` against the OpenMathInstruct-2 math dataset on a single GPU, using the default `examples/configs/grpo_math_1B.yaml` [16].

```sh
uv run python examples/run_sft.py
```

This fine-tunes `Llama3.2-1B` on the SQuAD dataset on a single GPU [17].

```sh
uv run python examples/run_dpo.py
```

This trains `Llama3.2-1B-Instruct` on the HelpSteer3 preference dataset on a single GPU [18].

The Docker path is the README's own recommended default: `docker pull nvcr.io/nvidia/nemo-rl:latest`, then run the container with GPU, HF cache, and W&B key mounts, and inside it run the same `uv run python examples/run_grpo.py` [10].

## Start it

- One GPU is the base form for every method above; no separate single-GPU launcher is needed beyond `uv run`.
- More GPUs on one node: pass `cluster.gpus_per_node=8` (or any count) as a command-line override, e.g. `uv run python examples/run_grpo.py cluster.gpus_per_node=8` [16].
- Multi-node goes through SLURM: set `NUM_ACTOR_NODES`, build a `COMMAND` string that itself is a `uv run ./examples/run_<method>.py --config <path> cluster.num_nodes=N cluster.gpus_per_node=8 ...` invocation, then `sbatch --nodes=$NUM_ACTOR_NODES --gres=gpu:8 ray.sub` — `ray.sub` is the repo's own SLURM submission script, documented per-method (GRPO, SFT, DPO, RM) with the same shape each time; the README notes `--gres=gpu:4` for GB200's 4-GPUs-per-node systems [11][19][20]. Kubernetes is documented separately via `docs/cluster.md` (referred to in the README as "Set Up Clusters") but not detailed with launch commands in the README itself [21].
- GRPO's generation layer is a separate config block, not the launcher: `policy.generation` names the backend (`vllm` in the default 1B config) and, when colocating generation on the training GPUs, `colocated.enabled: true` shares them; `vllm_cfg.gpu_memory_utilization` (0.6 in the default config) sets the generation share of a colocated card [22]. The 32-node Qwen2.5-32B example combines `policy.generation.vllm_cfg.tensor_parallel_size=4` for generation with `policy.dtensor_cfg.tensor_parallel_size=8`, `sequence_parallel=True`, and `activation_checkpointing=True` for training, all as command-line overrides on the same config [12].
- Effective batch arithmetic is explicit in the config, not implicit: the default GRPO config sets `policy.train_global_batch_size: 512` against `policy.train_micro_batch_size: 4`, so scaling GPU count means retuning these two (and `cluster.gpus_per_node`) together to hold the effective batch constant [23].
- Config surface: every method reads a YAML under `examples/configs/` (e.g. `grpo_math_1B.yaml`, `sft.yaml`, `dpo.yaml`), overridable key-by-key on the command line with dotted paths (`policy.model_name=...`) [16][17][18]. One default the library sets rather than inherits: the GRPO example config's `policy.precision` is `"bfloat16"`, a silent bf16-GPU assumption baked into the shipped default rather than left to a framework floor [23].
- Out-of-memory first aid, from the README's own Tips and Tricks: set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`, either globally at launch (`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 uv run python examples/run_dpo.py ...`) or permanently in the YAML under `policy.dtensor_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF`; the README attributes this fragmentation risk specifically to "models without support for FlashAttention2" [24]. A second, unrelated first-aid entry in the same section: a `ModuleNotFoundError: No module named 'megatron'` means the git submodules were not initialized — fix with `git submodule update --init --recursive` followed by `NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py ...` to force a venv rebuild [24].

## Watch it

Mechanics only — what a metric MEANS for GRPO or PPO training health lives on those methods' cards; this section says only how NeMo RL turns logging on and what it names each signal.

- **Enable it**: the default GRPO config disables every tracker (`logger.wandb_enabled: false`, `tensorboard_enabled: false`, `mlflow_enabled: false`, `swanlab_enabled: false`) — a run with no `logger.*_enabled=True` override produces no external log at all beyond the `log_dir: "logs"` local directory [23]. GPU utilization polling is a separate opt-in: `logger.monitor_gpus: true` (default in the example config) with `gpu_monitoring.collection_interval: 10` and `flush_interval: 10` seconds, sampled by Ray node polling [23][25].
- **GRPO metric names**, from the docs' GRPO guide Metrics section [9]: `token_mult_prob_error` (Multiplicative Token Probability Error, flagged as concerning if it trends upward past roughly 1-2%); `gen_kl_error`, `policy_kl_error`, `js_divergence_error` (the KL Divergence Error family) — the guide states a quantitative threshold: "Ideally, all KL divergence metrics should be close to 0, with values below 1e-3 considered acceptable. Investigate any metric that shows spikes above this threshold." [9]; `sampling_importance_ratio`, expected to hover around 1; `approx_entropy`, watched for entropy collapse [9].
- **PPO logs the same metric set as GRPO plus critic-specific fields**, per the docs' PPO guide, which states PPO "logs all the same metrics as GRPO" [26]: `critic/loss`, `critic/grad_norm`, `critic/values_mean`, `critic/values_min`, `critic/values_max`, `critic/returns_mean`, and `critic/explained_var`, where a value near 1.0 indicates a well-fit value function [26].
- **DPO and RM** report validation metrics as class fields in the library's API reference rather than in a prose metrics section: `DPOValMetrics` lists `accuracy`, `loss`, `preference_loss`, `sft_loss`, `rewards_chosen_mean`, `rewards_rejected_mean`, `global_valid_seqs`, `global_valid_toks`, `num_valid_samples`; `RMValMetrics` lists `accuracy`, `loss`, `rewards_chosen_mean`, `rewards_rejected_mean`, `num_valid_samples` [27]. These are read from the API reference's class-attribute listing, not from a rendered "how to read this" guide page the way GRPO's and PPO's metrics are, so treat them as field names to log rather than as health-threshold guidance.
- **Sample-level logging**: the GRPO/PPO logger design supports logging validation samples; the example config's `logger.num_val_samples_to_print` field, set via a command-line override in the README's own GRPO example (`logger.num_val_samples_to_print=10`), controls how many generated validation samples are printed [16][23].
- **Evaluation during training**: GRPO's config exposes `val_period`, `val_start_at`, `val_at_start`, `val_at_end`, `max_val_samples`, and `val_batch_size` directly in the `grpo:` block of the example config [23]; a separate standalone evaluation path (`examples/run_eval.py` against `examples/configs/evals/eval.yaml`) runs Pass@k-style benchmarks (e.g., MATH-500) on a converted Hugging Face checkpoint after training [28].
- **Stopping**: no RL-specific stopping rule, patience, or reward threshold is published in the GRPO guide's Metrics section [9], the PPO guide's Metrics section [26], or the README's Tips and Tricks page [24] — searched 2026-08-11. `grpo.max_num_steps` and `grpo.max_num_epochs` in the example config are hard iteration caps, not adaptive stopping criteria [23].

## Save it

- Checkpoint cadence and retention are set in the config's `checkpointing:` block: `checkpoint_dir` (`"results/grpo"` in the default GRPO config), `save_period` (steps between saves; 10 in the default config), `keep_top_k` (3), `metric_name` and `higher_is_better` for selecting the best checkpoint, `model_save_format: "safetensors"`, and two retention flags with a documented cost: `save_consolidated: false` and `save_optimizer: true` [23]. NeMo RL writes two distinct checkpoint formats: "Torch distributed" (the default, used to save intermediate checkpoints during training) and "Hugging Face format"; the design doc states HF-format checkpoints "save only the model weights, ignoring the optimizer states", and recommends using Torch distributed for intermediate saves and HF format only at the end of training — because dropping optimizer state (as HF format and `save_optimizer: false` both do) prevents resuming training from that checkpoint [29].
- Converting a Torch-distributed checkpoint to Hugging Face format for evaluation or sharing: `uv run python examples/converters/convert_dcp_to_hf.py --config results/grpo/step_170/config.yaml --dcp-ckpt-path results/grpo/step_170/policy/weights/ --hf-ckpt-path results/grpo/hf` [30]. For a Megatron-backend checkpoint, the analogous converter requires the `mcore` extra: `uv run --extra mcore python examples/converters/convert_megatron_to_hf.py --config results/grpo/step_170/config.yaml --megatron-ckpt-path results/grpo/step_170/policy/weights/iter_0000000 --hf-ckpt-path results/grpo/hf` [30].
- LoRA on the Megatron backend saves an adapter, not a full model: `convert_lora_to_hf.py` either merges the adapter into the base model to produce a standalone full Hugging Face checkpoint (`--base-ckpt`, `--adapter-ckpt`, `--hf-model-name`, `--hf-ckpt-path`) [30], or, with `--adapter-only`, exports just the LoRA weights in Hugging Face PEFT format, keeping the base model separate [31]. Even in adapter-only mode, the converter still requires `--base-ckpt` to reconstruct the Megatron model and apply the LoRA modules before export, since the adapter directory alone is not a loadable model [31].
- Resume: the docs' checkpointing design page documents the Torch-distributed-vs-HF-format save distinction above as the basis for resumability, but the exact resume CLI flag/call form is not stated on that page or in the README beyond the converter and checkpoint-directory paths already given [29][30].
- Loader handoff: `examples/run_eval.py generation.model_name=$PWD/results/grpo/hf` loads a converted Hugging Face checkpoint directly for evaluation [28] — so the HF-format conversion step above is the loader's contract for this library: a Torch-distributed checkpoint is not directly loadable by `run_eval.py` without first running one of the `convert_*_to_hf.py` scripts.

## Find it in the docs

The docs are the live source; this section teaches the lookup, not the content.

- Address pattern: `https://docs.nvidia.com/nemo/rl/<version>/<page>.html`. `<version>` is `latest` or a bare release number (`0.7.0`); checked 2026-08-11: `/nemo/rl/0.7.0/index.html` loads, `/nemo/rl/v0.7.0/index.html` (v-prefixed) 404s. `latest` currently resolves to 0.7.0, confirmed via the page's embedded `DOCUMENTATION_OPTIONS.theme_switcher_version_match` value.
- Page-slug recipes, from the docs' own top-level site map [32]: per-method conceptual pages live under `about/algorithms/<method>.html` (`grpo`, `ppo`, `dpo`, `sft`, `rm`, `cispo`, `dapo`, `mopd`, `on-policy-distillation`); per-method how-to guides with metrics and usage notes live under `guides/<method>.html` (`grpo.html`, `ppo.html`, `sft.html`, `dpo.html`, `rm.html`, plus `async-grpo.html`, `lora.html`, `xtoken-off-policy-distillation.html`); architecture and internals live under `design-docs/` (`checkpointing.html`, `logger.html`, `training-backends.html`, `generation.html`, `env-vars.html`, `uv.html`); setup pages are `about/installation.html`, `about/quick-start.html`, `about/tips-and-tricks.html`, `cluster.html`, `docker.html`.
- Question-to-slug map: "what does this metric mean" -> `guides/grpo.html#metrics` or `guides/ppo.html#metrics` (these two pages have drifted apart — PPO's page adds a critic-metrics table and states outright that it reuses GRPO's set, rather than repeating GRPO's definitions) [9][26]. "How do I save/convert a checkpoint" -> `design-docs/checkpointing.html` [29]. "How do I set up a cluster" -> `cluster.html`. "What's the CUDA/Python floor" -> `about/installation.html` [15].
- Runnable references beyond the docs: the `examples/` tree of the GitHub repo, including per-scale recipe YAMLs under `examples/configs/recipes/llm/` (e.g. `grpo-qwen3.5-9b-1n8g-megatron.yaml`) named directly in the README's News entries [7][4]. The GRPO smoke test (`examples/configs/grpo_smoke.yaml`, GSM8K, capped at 10 optimizer steps) is the README's own named install-verification recipe [16].
- Community layer: no curated community-tutorials page or blog-post index was found in the docs site map fetched for this card (`docs_index.html`) [32]; the README's own "News" section links to some external write-ups (a Google Cloud blog on GCP RL scaling, a GitHub Discussions post on weight-transfer optimization), but these are dated project announcements, not a maintained curated-tutorials page, so treat them as historical pointers rather than an ongoing community layer [7].
- No official MCP endpoint for querying these docs was found in the docs site map or README fetched for this card [32][4].
- Trap, from the README's own Tips and Tricks (not an issue-tracker reply, but the maintainers' own documented failure mode): forgetting `--recursive` on clone produces `ModuleNotFoundError: No module named 'megatron'`, fixed by `git submodule update --init --recursive` plus a forced venv rebuild [24].
- Honest boundary: the README's own Features list marks "Improved Native Performance", "Improved Large MoE Performance", "Resiliency" (fault tolerance / auto-scaling), and on-policy cross-tokenizer distillation as not yet shipped ("Coming in v0.7") as of the version fetched for this card [7]; no CUDA/driver floor is published anywhere in the README or `pyproject.toml`, only the "current primary" CUDA 13 hedge in the cuDNN install instructions [10][15].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. README-sourced claims are read at the shortlist row's pinned commit `e08fc276e41b9f4fef69a0b91f6f384ad1964f3d`; the Install field's package-metadata claims are read at the v0.7.0 release tag (commit `81aa43dda4765b0429cf31dab44441e4e4383911`), which is several days behind that pinned commit — flagged in the Install field itself. Docs pages are `latest`-version live pages unless noted; all fetches for this card are dated 2026-08-11, except the stopping-rule search, also 2026-08-11. Method names (GRPO, PPO, DPO, SFT, RM, GSPO, DAPO, CISPO, GDPO, distillation) are deliberately cited to nothing here: their defining papers live on the methodology cards. Ecosystem tools named in passing (Ray, vLLM, SGLang, Megatron-LM, Megatron-Bridge, SLURM, Kubernetes, NGC) are reached through the sources below and are not separately enumerated.

[1] NVIDIA-NeMo/RL README, title (H1). https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[2] NVIDIA-NeMo/RL `pyproject.toml` at the v0.7.0 release tag. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/v0.7.0/pyproject.toml. Fetched 2026-08-11.

[3] NVIDIA-NeMo/RL README, Quick Start / Bare-Metal Quick Start section (example script launch pattern). https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[4] NVIDIA-NeMo/RL GitHub repository page and metadata (stars, licence, archived status, organization). https://github.com/NVIDIA-NeMo/RL and https://api.github.com/repos/NVIDIA-NeMo/RL. Fetched 2026-08-11.

[5] NVIDIA-NeMo/RL README, Training Backends and Generation Backends sections. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[6] NVIDIA-NeMo/RL README, Overview section ("Efficient resource management using Ray"). https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[7] NVIDIA-NeMo/RL README, Features and News sections. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[8] NVIDIA-NeMo/RL example config files for each method (`examples/configs/grpo_math_1B.yaml`, `dpo.yaml`, `sft.yaml`, `ppo_math_1B.yaml`) at the shortlist commit. Fetched 2026-08-11.

[9] NeMo RL docs, GRPO guide, Metrics section. https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html. Fetched 2026-08-11.

[10] NVIDIA-NeMo/RL README, Quick Start (Docker) and Prerequisites sections. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[11] NVIDIA-NeMo/RL README, GRPO Multi-node / SFT Multi-node / DPO Multi-node / RM Multi-node sections (shared `sbatch`/`ray.sub` launch form). https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[12] NVIDIA-NeMo/RL README, GRPO Qwen2.5-32B section. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[13] NVIDIA-NeMo/RL GitHub Releases API (v0.7.0 release date). https://api.github.com/repos/NVIDIA-NeMo/RL/releases. Fetched 2026-08-11.

[14] NVIDIA-NeMo/RL GitHub Tags API (v0.7.0 tag resolved to commit `81aa43dda4765b0429cf31dab44441e4e4383911`). https://api.github.com/repos/NVIDIA-NeMo/RL/tags. Fetched 2026-08-11.

[15] NeMo RL docs, Installation and Prerequisites page ("current primary" CUDA 13 language; Python 3.13 venv paths). https://docs.nvidia.com/nemo/rl/latest/about/installation.html. Fetched 2026-08-11.

[16] NVIDIA-NeMo/RL README, GRPO / GRPO Single Node section. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[17] NVIDIA-NeMo/RL README, Supervised Fine-Tuning (SFT) / SFT Single Node section. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[18] NVIDIA-NeMo/RL README, DPO / DPO Single Node section. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[19] NVIDIA-NeMo/RL README, GRPO Multi-node section. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[20] NVIDIA-NeMo/RL README, RM / RM Single Node / RM Multi-node section. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[21] NVIDIA-NeMo/RL README, Set Up Clusters section (pointer to `docs/cluster.md`). https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[22] NVIDIA-NeMo/RL `examples/configs/grpo_math_1B.yaml` at the shortlist commit, `generation` and `colocated` blocks. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/examples/configs/grpo_math_1B.yaml. Fetched 2026-08-11.

[23] NVIDIA-NeMo/RL `examples/configs/grpo_math_1B.yaml` at the shortlist commit, `grpo`, `policy`, `checkpointing`, `logger`, and `cluster` blocks. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/examples/configs/grpo_math_1B.yaml. Fetched 2026-08-11.

[24] NVIDIA-NeMo/RL README, Tips and Tricks section. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[25] NeMo RL docs, Logger design doc (GPU monitoring via Ray node polling). https://docs.nvidia.com/nemo/rl/latest/design-docs/logger.html. Fetched 2026-08-11.

[26] NeMo RL docs, PPO guide, Metrics section. https://docs.nvidia.com/nemo/rl/latest/guides/ppo.html. Fetched 2026-08-11.

[27] NeMo RL docs, API reference, `nemo_rl.algorithms.dpo.DPOValMetrics` and `nemo_rl.algorithms.rm.RMValMetrics` class-attribute listings, via the docs site index. https://docs.nvidia.com/nemo/rl/latest/index.html. Fetched 2026-08-11.

[28] NVIDIA-NeMo/RL README, Evaluation / Run Evaluation section. https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[29] NeMo RL docs, Checkpointing design doc (Torch distributed vs. Hugging Face format contract). https://docs.nvidia.com/nemo/rl/latest/design-docs/checkpointing.html. Fetched 2026-08-11.

[30] NVIDIA-NeMo/RL README, Evaluation / Convert Model Format section (converter CLI forms). https://raw.githubusercontent.com/NVIDIA-NeMo/RL/e08fc276e41b9f4fef69a0b91f6f384ad1964f3d/README.md. Fetched 2026-08-11.

[31] NeMo RL docs, Checkpointing design doc, "Option B — Adapter-only (PEFT format)" section. https://docs.nvidia.com/nemo/rl/latest/design-docs/checkpointing.html. Fetched 2026-08-11.

[32] NeMo RL docs site index / table of contents (page-slug map). https://docs.nvidia.com/nemo/rl/latest/index.html. Fetched 2026-08-11.

[33] NeMo RL docs, Performance Summary page (H100 BF16, H100 FP8, and GB200 BF16 reference-run benchmark tables). https://docs.nvidia.com/nemo/rl/latest/about/performance-summary.html. Fetched 2026-08-11.
