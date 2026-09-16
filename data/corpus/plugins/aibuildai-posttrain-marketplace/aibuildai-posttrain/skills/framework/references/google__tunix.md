# tunix

Google's JAX-native post-training library ("Tune-in-JAX"): one config-driven trainer surface for SFT, preference optimization, and RL, built on Flax NNX and Orbax, with vLLM/SGLang-JAX rollout for online methods.

**Tunix (Tune-in-JAX)** "is a JAX based library designed to streamline the post-training of Large Language Models" [1]. It is built and maintained by Tunix Developers at Google [1][2], and it sits "as an intermediate layer" between core JAX libraries (Flax, Optax, Orbax) and optimized model implementations such as MaxText and MaxDiffusion [1]: a training run is a `PeftTrainer` (SFT) or an `RLLearner`/`rl.Trainer` (RL/agentic), constructed from a `ModelConfig`, a JAX sharding `Mesh`, an optimizer, and a dataset, then driven with `.train()` [3]. It lives at https://github.com/google/tunix [4].

**When to pick it**: post-training on TPUs (or GPU/CPU) inside a JAX/Flax stack, when you want native `vLLM`/`SGLang-JAX` rollout for online RL and Pathways-based multi-host scale-out, rather than a PyTorch-ecosystem trainer like trl or a Ray-orchestrated cluster like verl (cross-reference; not covered here). Its docs describe the project as still evolving ("Current Status: V2 Release... Tunix is under active development") [1], and PyPI classifies it "Development Status :: 3 - Alpha" [5], so expect API and config-surface churn.

**Methods it ships** [1][6]: SFT & preference - Full-weights fine-tuning, PEFT/LoRA, DPO, ORPO; RL - PPO, GRPO, GSPO-Token, DAPO, Dr.GRPO; Agentic RL - multi-turn tool use, asynchronous rollout, trajectory batching/grouping built around `TrajectoryCollectEngine`/`RolloutOrchestrator`/`GroupQueueManager` [7]. The algorithms page is the live grouping to recheck [6]. The shortlist row's own methods-detection cites ORPO to a specific test file, `tests/sft/dpo/orpo_trainer_test.py` at the screening commit `f1922757f84909bd1b931ec879e5248d4e9400c4` (a 17,828-byte file, confirmed present at that commit) [8]. Extending the algorithm set is a documented pattern: subclass `AlgorithmConfig`/`RLLearner`, register custom losses with `@register_policy_loss_fn`, and implement `AbstractRewardManager` for custom reward logic [9].

**Scale it handles**: single TPU/GPU/CPU host up to multi-node; multi-host scale-out goes through Pathways on GKE via the `xpk` CLI (`xpk cluster create-pathways`, a Docker image build, then `xpk workload create-pathways`), which the docs describe as "a transparent change that simply requires you to submit your job through Pathways instead of running directly on a VM" [10]. The README claims this "can scale up to thousands of devices" [1], but no published multi-node benchmark accompanies that claim in any doc read for this card - mechanism is documented, throughput is not. Sharding is a `jax.sharding.Mesh` with named axes (commonly `fsdp`/`tp`) set per role; roles can share one mesh (collocated) or use distinct meshes (disaggregated), see Start it [11].

**Install**: for TPU, `pip install "google-tunix[prod]"`; for GPU, `pip install google-tunix` then `pip install -U "jax[cuda13]"`; for CPU, `pip install google-tunix "jax[cpu]"` [12]. Version 0.1.7 uploaded to PyPI 2026-06-11T20:34:49, resolved at git tag `v0.1.7` = commit `ce63a9fd65f02c4c398e74f000f98371c1575eb6` (committer date 2026-06-09T22:27:21Z) [13][14] - this is earlier than the shortlist's screening commit `f1922757...` (2026-07-31T22:39:26Z), so the screening commit is ahead of what `pip install` actually delivers; the fields below are read at the release tag, not the screening commit. Python >=3.11; licence Apache-2.0 [5][13]. Core dependencies at v0.1.7 include `flax>=0.11.1`, `orbax-checkpoint>=0.12.0`, `qwix>=0.1.6`, `google-metrax>=0.2.3`; torch is not a dependency. Two load-bearing pins carry inline reasons in the release's own `pyproject.toml`: `safetensors<0.8` ("higher versions breaks with jax >=0.9") and the `prod` extra's `jax[tpu]>=0.6.0,!=0.7.2` ("Jax 0.7.2 has performance regression on OSS") [13]. Other extras at this release: `dev` (empty - the comment says to manually install vLLM and `tpu-inference`, which "depends on jax[tpu]==0.7.2"), `test` (`pytest`, `pytest-xdist`, `gcsfs`, `chex`), `cli` (`tensorflow`), `docs` (Sphinx toolchain) [13]. No CUDA/hardware minimum is stated in this file or the installation page beyond routing GPU users to `jax[cuda13]` [12][13]. Note a pinned-vs-live gap: the live metrics docs describe an opt-in `otel` extra (`pip install "google-tunix[otel]"`) [15], but no `otel` extra exists in the v0.1.7 release's `pyproject.toml` [13] - that feature postdates this release.

**Maintained by**: Tunix Developers, Google [1][2]; per the shortlist row's screening snapshot the repository is not archived, last pushed 2026-07-31T22:39:32Z, licensed Apache-2.0, with 9 Hub models attributed to it as the only repository of that name in the index (star count is a raw, non-ranking field per the shortlist and is not used here for comparison). The README's dated News entries show active work: "[2026/04] Gemma4 models are supported in Tunix!"; "[2026/01] Tunix model now supports efficient kernel execution (splash attn, GMM MoE)."; "[2025/12] Agentic RL Training has been released, with efficient support of multi-turn agent-env interaction, tool usage, async rollout, etc." [16].

## Quick start

The docs' own smallest complete walkthrough is titled "Quick start: GRPO" but its actual code is an SFT/PEFT run on Gemma 3 270M, not GRPO training - reproduced here as the docs give it [17]:

Load the model:
```python
from huggingface_hub import snapshot_download
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib

MESH = [(1, 1), ("fsdp", "tp")]
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
model_path = snapshot_download(repo_id=model_id, ignore_patterns=["*.pth"])
config = gemma_lib.ModelConfig.gemma3_270m()
with mesh:
    model = params_safetensors_lib.create_model_from_safe_tensors(model_path, config, mesh)
```
The docs note Gemma 3 bypasses the usual `AutoModel` loader here "since Gemma 3 isn't supported for now" by that unified API [17].

Load data and run full fine-tuning:
```python
from tunix.generate import tokenizer_adapter
from tunix.examples.data import translation_dataset as data_lib
from tunix.sft import peft_trainer

tokenizer = tokenizer_adapter.Tokenizer("./tokenizer_gemma.model")
train_ds, val_ds = data_lib.create_datasets(
    'mtnt/en-fr', global_batch_size=64, max_target_length=256,
    num_train_epochs=3, tokenizer=tokenizer,
)
trainer = peft_trainer.PeftTrainer(
    model=model, optimizer=optax.adamw(learning_rate=1e-4),
    mesh=mesh, model_input_fn=input_fn,
)
trainer.train(train_ds=train_ds, num_steps=100, eval_ds=val_ds, eval_steps=20)
```
LoRA/QLoRA fine-tuning swaps in `qwix.LoraProvider(module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj", rank=RANK, alpha=ALPHA)` (QLoRA adds `weight_qtype="nf4"`) and `qwix.apply_lora_to_model(model, lora_provider, **model_input)` before training the same way [17]; the Performance Considerations page gives a concrete instance of the same call with `rank=16, alpha=2.0` [11]. There is no separate single-command CLI quickstart in the docs read for this card; the CLI form is covered in Start it.

## Start it

- **One host/one accelerator**: run the quickstart script above directly, or a CLI shell script under `examples/rl/` or `examples/sft/` (e.g. `run_gemma_2b.sh` for PEFT on MTNT, `run_gemma_7b.sh` for GRPO on GSM8K) [18].
- **The Tunix CLI** is "the default choice" for launching jobs; it composes configuration from three tiers, lowest to highest priority: a `base_config.yaml` base config file, an `override_config_file` argument, and individual CLI `key=value` arguments [19]. CLI entry points are `peft_main.py`, `grpo_main.py`, and `ppo_main.py`, each reading configuration processed by `config.py` [18]. Multi-GPU/TPU on one host uses the same CLI with a wider `mesh.shape`; environment setup is scripted via `scripts/setup_cli_tpu_single_host.sh` or `scripts/setup_cli_gpu_single_host.sh` [18].
- **Multi-node** goes through Pathways on GKE: install `xpk` (`pip install xpk`), create a Pathways cluster (`xpk cluster create-pathways --cluster $CLUSTER_NAME --num-slices=1 --tpu-type=$TPU_TYPE ...`), build the base Docker image (`bash ./build_docker.sh`), then submit with `xpk workload create-pathways --cluster=$CLUSTER_NAME --command="... source your-script-to-launch-job.sh" --num-slices=1 --tpu-type=$TPU_TYPE --base-docker-image docker.io/library/tunix_base_image` [10].
- **Sharding mesh**: `mesh_config.shape` (e.g. `"(2,2)"`) and `mesh_config.axis_names` (e.g. `"('fsdp','tp')"`) define the hardware mesh for `model_config`, and `actor_model_config`/`reference_model_config`/`rollout_model_config` typically inherit from it in RL [20].
- **Collocated vs. disaggregated rollout**: collocated sets one shared mesh for every role in `ClusterConfig(role_to_mesh={Role.ACTOR: mesh, Role.REFERENCE: mesh, Role.ROLLOUT: mesh}, ...)`, and "the cluster will finish rollout generation before shifting to inference and training" sequentially; collocated mode supports `host_offloading`/`offload_to_cpu`, which "saves HBM by moving non-active models to CPU RAM." Disaggregated mode splits devices into distinct meshes per role and runs roles concurrently - "the preferred mode for large-scale runs where maximizing global throughput is more critical" [21].
- **Effective-batch arithmetic is three-tiered** [21]: Global Batch Size is "the total amount of data processed in one full RL iteration," the product of dataloader batch size and generations per prompt (e.g. 256 prompts x 4 generations = 1024) - for PPO it equals the dataloader's batch size directly, and it is derived, not directly tunable. Mini Batch Size "determines how frequently the model is updated within a global step" (a 1024 global batch with a 256 mini-batch updates 4 times per global step); it must evenly divide the global batch and defaults to it if unset. Micro Batch Size has three separate knobs: `rollout_micro_batch_size` (deprecated under the newer/agentic rollout path, which "automatically decomposes batches into individual prompts"), `compute_logps_micro_batch_size` (recommended 2-4x the trainer's micro-batch since it is forward-only), and `train_micro_batch_size` (enables automatic gradient accumulation; steps per mini-batch = mini_batch_size / train_micro_batch_size). Tuning tip from the docs: set `compute_logps_micro_batch_size` to 2x `train_micro_batch_size` (try 4x for throughput, back off to 2x on OOM) [21].
- **Config surface** groups into `model_config` (model id/source/path, `lora_config`, `mesh`), `tokenizer_config`, dataset fields (`dataset_name`, `batch_size`, `max_target_length`, `num_train_epochs`), `optimizer_config` (`opt_type`, `learning_rate`, `schedule_type`, `max_grad_norm`), `training_config` (`max_steps`, `eval_every_n_steps`, `gradient_accumulation_steps`, `checkpointing_options`, `metrics_logging_options`, `data_sharding_axis`), and method-specific `grpo_config`/`ppo_config` blocks (`num_generations`, `beta`, `epsilon`, `temperature` for GRPO; `num_iterations`, `lambda`, `gamma`, `beta`, `epsilon` for PPO) [20]. No default-precision field (bf16/fp32) is documented on the Launching Jobs config page read for this card, so a silent-default claim cannot be made from this source.
- **OOM first aid**: for vLLM-backed generation, lower `rollout_vllm_hbm_utilization` (example default shown is 0.2) and reduce `max_prompt_length`/`max_tokens_to_generate` [22]. For SGLang-Jax-backed generation, lower `rollout_sglang_jax_mem_fraction_static` (example 0.2) [23]. A vLLM data-parallel note: "If you set `data_parallel_size > 1`, Tunix sets `NEW_MODEL_DESIGN=1` for vLLM. Ensure your rollout mesh size matches `tensor_parallel_size * data_parallel_size`" [22]. LoRA with vLLM rollout and CLI support for the vLLM rollout engine are both explicitly marked "WIP" in the docs [22]. SGLang-Jax rollout "does not support data-parallel for rollout yet," and its sampler "does not populate token-level logprobs" - compute logprobs via the trainer model if your algorithm needs them [23]. Fault tolerance against OOM more generally comes from `InflightThrottler`, which limits concurrently scheduled TPU computations via `max_inflight_computations` in `TrainingConfig` [24].

## Watch it

This section is mechanics only - what a given metric means for a specific method lives on that method's card, not here.

- **Enable it**: logging is configured through `metrics_logging_options` in `training_config` (SFT, via `peft_trainer.TrainingConfig`) or `RLTrainingConfig` (RL, inside `ClusterConfig`), covering project name, run name, and flush frequency [25][26]. Two backends are "Enabled by default in external environments": Weights & Biases (if `wandb` is installed; `project_name` defaults to `"tunix"`, `run_name` defaults to a timestamp) and TensorBoard (`log_dir`, `flush_every_n_steps` default 100) [26]. A third, OpenTelemetry, is opt-in and experimental: "Off by default. With the flag unset, behavior is identical to previous releases," requires `pip install "google-tunix[otel]"` (a live-docs-only extra - see Install), and Tunix "never configures exporters or shuts down providers"; it maps stable instrument names under a `tunix.*` namespace (e.g. `loss` -> `tunix.training.loss`) and ships `tunix.sft.otel_wandb.WandbMetricsExporter` to forward OTel gauges into `wandb.log` [26].
- **Common metrics** (SFT and RL alike): `loss`, `perplexity` (= exp(loss)), `learning_rate`, `step_time_sec`, `steps_per_sec`, `tflops_per_step` [26].
- **DPO and ORPO metrics**: `rewards/chosen`, `rewards/rejected`, `rewards/margin` (= rewards/chosen - rewards/rejected), `rewards/accuracy`, `log_probs/chosen`, `log_probs/rejected`; ORPO adds `odds_ratio`, `sft_loss` (the NLL component of the ORPO loss), and `or_loss` (the odds-ratio preference-loss component) [26] - directly relevant to this row's ORPO detection [8].
- **RL (PPO/GRPO) metrics**: rewards/scores - `rewards/sum`, `rewards/mean|max|min`, `score/mean|max|min`, `reward_kl_penalty`, per-reward-function `rewards/<name>`; policy/value - `advantages/mean|max|min`, `returns/mean|max|min`, `values/mean|max|min`, `pg_clipfrac`, `vf_clipfrac`, `loss/entropy`; generation/data - `prompts`, `completions`, `completions/mean_length|max_length|min_length`, `trajectory_ids`, `actor_dequeue_time` [26].
- **Sample-level logging**: `AsyncTrajectoryLogger` "logs the trajectories including prompts, responses, etc." to CSV asynchronously without blocking the training loop; it is "enabled in `agentic_grpo_learner` by default" once a `log_dir` is set on `cluster_config.training_config.metrics_logging_options` (a local path or a GCS path), and logged data is queryable as a pandas DataFrame [17].
- **Evaluation during training**: `eval_every_n_steps` in `training_config` controls cadence [20]; SFT's `trainer.train()` also takes `eval_ds`/`eval_steps` directly, as shown in Quick start [17].
- **Performance metric tracing**: an additional v1/v2 tracing layer exports to Perfetto, but the docs state it is "currently only supported for the GRPO main entry point" [26] - do not expect it for other trainers.
- **Stopping-rule honesty**: no RL stopping rule, early-stopping callback, or reward threshold is published anywhere searched for this card - metrics.txt, performance.txt, reliability.txt, and launching.txt. The only threshold-shaped field found is `max_grad_norm` in `optimizer_config`, described as a gradient-clipping threshold "especially in RL," which is unrelated to a stopping condition [20].

## Save it

- Checkpointing is Orbax-backed and shared by both trainers: SFT uses `PeftTrainer` directly, RL uses `rl.Trainer` (a `PeftTrainer` subclass) inside `RLLearner`, so "both SFT and RL share the same checkpointing mechanism" [27]. It is off unless you turn it on: "By default, checkpointing is disabled if `checkpoint_root_directory` is not specified" in `SFTConfig`/`RLConfig`; once set, Tunix "automatically saves checkpoints and resumes training from the most recent one if interrupted, restoring model weights, optimizer state, and training step count" [27].
- Saved content: "saving model parameters (supporting full state or only LoRA parameters for PEFT) and optimizer state" [27].
- `checkpointing_options` sub-fields and their defaults: a save-decision policy (`FixedIntervalPolicy` or the default `ContinuousCheckpointingPolicy(minimum_interval_secs=180)`, which "saves every 180 seconds"); a preservation policy (default `LatestN(n=3)`, which "keeps the latest 3 checkpoints"); a step-name format (default `ocp.path.step.standard_name_format()`, "simple integer step names"); and async-write controls, `enable_async_checkpointing` (default `True`, recommended to stay on "to prevent the main thread from blocking") and `timeout_secs` (default 1200) [27]. No flag equivalent to a weights-only "discard optimizer state" retention switch was found in this section or in the Launching Jobs config page's `checkpointing_options` listing (`max_to_keep`, `save_interval_steps`, `enable_async_checkpointing`, `timeout_secs`) [20][27] - retention here trims checkpoint *count*, not what each checkpoint contains.
- No explicit `save_model`/`push_to_hub`/`resume_from_checkpoint`-style call signatures are given on the docs pages read for this card; resumption is described as automatic once `checkpoint_root_directory` is set and training is restarted, picking up "the most recent" checkpoint [27]. Determinism during a resumed or fresh run is controlled by `rng_seed`/`random_seed` (model init) and `data_shuffle_seed` (RL data shuffling) [27].
- PEFT/LoRA changes what is saved, per the checkpoint-support statement above: only LoRA parameters can be persisted instead of full model state [27]. No dedicated adapter-directory file layout (analogous to trl's `adapter_config.json`/`adapter_model.safetensors`) is documented in the pages read for this card - inspect a checkpoint directory on disk to confirm its layout before assuming a format.
- Loader handoff: none of the docs pages read for this card state whether a saved checkpoint directory is directly loadable by an external evaluator (e.g., via `AutoModel.from_pretrained` or a raw Orbax restore) - the Models page describes `AutoModel.from_pretrained` loading from Hugging Face, Kaggle, GCS, or MaxText sources, not explicitly from a Tunix-written checkpoint directory [28]; confirm the loader contract from a checkpoint before treating this as a full-model directory.

## Find it in the docs

Docs are the live source here; this section is the lookup, not a mirror.

- Address pattern: `https://tunix.readthedocs.io/en/<version>/<page>.html`. Checked 2026-08-11: `en/latest/index.html` and `en/stable/index.html` both load (200), but the release-tag form `en/v0.1.7/index.html` does not - it 404s. This differs from some other libraries in this corpus (e.g. trl) whose docs do accept a `v<X.Y.Z>` tag; for Tunix, pin your reading to `stable` or note that you read `latest` (unpinned, mutable) [29].
- Page slugs, from the left-nav index [1][30]: `quickstart`, `design`, `agentic_rl`, `performance`, `reliability`, `launching`, `rollout`, `algorithms`, `models`, `metrics`, `examples`, `talks`, `contributing`, `code_of_conduct` (exact slugs confirmed by the fetched pages in scratch; verify the precise `.html` filename by fetching before citing a new one).
- Question-to-page map: "how do I launch a job / what CLI flags exist" -> Launching Jobs [19][20]; "what does vLLM/SGLang rollout look like, what do I do on OOM" -> Rollout [22][23]; "what metrics get logged" -> Metrics [26]; "how do checkpoints work / will my run resume" -> Reliability [27]; "what's the batch-size math / collocated vs disaggregated" -> Performance Considerations [21]; "what models are supported and how do I name one" -> Models [28]; "what algorithms exist and how do I add one" -> Algorithms [9].
- Runnable references beyond the docs: the repo's `examples/` tree, including Colab notebooks (`qlora_gemma.ipynb`, `grpo_gemma.ipynb`, `dpo_gemma.ipynb`, `logit_distillation.ipynb`) and script directories (`rl/grpo/gsm8k/`, `rl/grpo/gsm8k/verl_compatible/`, `deepscaler/`, `sft/mtnt/`, `model_load/`, `agentic/`) [31]; CLI smoke-test scripts under `examples/sft/` and `examples/rl/` train Gemma/Llama/Qwen variants on the MTNT translation dataset (SFT) and GSM8K (GRPO, PPO) [18].
- Community layer: the docs' own left-nav "Example gallery" curates a small tutorial set (Tuning, DPO Demo with math (gsm8k), GRPO Demo, Knowledge Distillation Gemma 7B->2B, LoRA & QLoRA Demo, PEFT of Llama 3.1-8B with LoRA/QLoRA on NVIDIA GPUs, VLM fine-tuning with DPO) [1]. A separate "Talks and Announcements" page curates external posts and talks in two dated tables: Announcements and Blogs (e.g. 2025-09-30 "Introducing Tunix: A JAX-Native Library," the official launch post; 2025-11-14 a Kaggle hackathon announcement; 2025-12-11 a blog on fine-tuning Gemma 3 for mobile; 2025-12-16 a blog on Tunix for LLM-agent post-training) and Talks (e.g. 2025-09-24 a JAX DevLab lightning talk; 2025-09-30 a Google for Developers introductory talk; 2025-12-11 a JAX/OpenXLA DevLab deep dive with SFT/GRPO/PPO demos) [32]. As with any community layer, check a post's or talk's date and pinned Tunix version against the version you are running.
- No official MCP endpoint for querying Tunix's docs was found or cited in this card.
- Boundary/traps found in the docs themselves (not from closed GitHub issues - none were fetched for this card, so no issue-numbered trap is reported here): LoRA + vLLM rollout is explicitly "WIP" [22]; CLI support for the vLLM rollout engine is explicitly "WIP" [22]; SGLang-Jax rollout does not support data-parallel and its sampler does not populate token-level logprobs [23]; Gemma 3 is not loadable via the unified `AutoModel` API and needs the family-specific loader shown in Quick start [17]; the "Internal" `ModelSource` "is not supported in OSS version" [28].

## Sources

All pages are `latest`/unpinned live docs read on 2026-08-11 unless a commit or release tag is named. Method names (SFT, DPO, ORPO, PPO, GRPO, GSPO-Token, DAPO, Dr.GRPO) are deliberately cited to nothing here; their defining papers live on the methodology cards, not this library card.

[1] Tunix documentation index. https://tunix.readthedocs.io/en/latest/index.html. Fetched 2026-08-11.

[2] tunix GitHub repository, About/description metadata. https://github.com/google/tunix. Fetched 2026-08-11.

[3] Tunix Quick Start guide (trainer construction pattern). https://tunix.readthedocs.io/en/latest/quickstart.html. Fetched 2026-08-11.

[4] tunix GitHub repository (item home). https://github.com/google/tunix. Fetched 2026-08-11.

[5] google-tunix on PyPI (JSON API: Python floor, licence, classifiers, upload date). https://pypi.org/pypi/google-tunix/json. Fetched 2026-08-11.

[6] Tunix Algorithms page, Supported Algorithms section. https://tunix.readthedocs.io/en/latest/algorithms.html. Fetched 2026-08-11.

[7] Tunix Agentic RL page (TrajectoryCollectEngine, RolloutOrchestrator, GroupQueueManager). https://tunix.readthedocs.io/en/latest/agentic_rl.html. Fetched 2026-08-11.

[8] GitHub Contents API for tests/sft/dpo/orpo_trainer_test.py at commit f1922757f84909bd1b931ec879e5248d4e9400c4. https://api.github.com/repos/google/tunix/contents/tests/sft/dpo/orpo_trainer_test.py?ref=f1922757f84909bd1b931ec879e5248d4e9400c4. Fetched 2026-08-11.

[9] Tunix Algorithms page, Add a New RL Algorithm section. https://tunix.readthedocs.io/en/latest/algorithms.html. Fetched 2026-08-11.

[10] Tunix Quick Start guide, Multi-Node Training via Pathways in GKE section. https://tunix.readthedocs.io/en/latest/quickstart.html. Fetched 2026-08-11.

[11] Tunix Performance Considerations page, PEFT with LoRA and Collocated vs Disaggregated Training sections. https://tunix.readthedocs.io/en/latest/performance.html. Fetched 2026-08-11.

[12] Tunix Quick Start guide, Installation section. https://tunix.readthedocs.io/en/latest/quickstart.html. Fetched 2026-08-11.

[13] tunix pyproject.toml at the v0.1.7 release tag (commit ce63a9fd65f02c4c398e74f000f98371c1575eb6). https://raw.githubusercontent.com/google/tunix/v0.1.7/pyproject.toml. Fetched 2026-08-11.

[14] GitHub Git Refs API resolving tag v0.1.7 to its commit. https://api.github.com/repos/google/tunix/git/refs/tags/v0.1.7. Fetched 2026-08-11.

[15] Tunix Metrics page, Metric Loggers / OpenTelemetry section. https://tunix.readthedocs.io/en/latest/metrics.html. Fetched 2026-08-11.

[16] tunix GitHub repository README.md, News section. https://raw.githubusercontent.com/google/tunix/f1922757f84909bd1b931ec879e5248d4e9400c4/README.md. Fetched 2026-08-11.

[17] Tunix Quick Start guide, "Quick start: GRPO" section and AsyncTrajectoryLogger note. https://tunix.readthedocs.io/en/latest/quickstart.html. Fetched 2026-08-11.

[18] Tunix Launching Jobs page, Peft Training on MTNT / GRPO and PPO Training on GSM8K / CLI Scripts Overview sections. https://tunix.readthedocs.io/en/latest/launching.html. Fetched 2026-08-11.

[19] Tunix Launching Jobs page, Tunix CLI overview and Configuration Hierarchy. https://tunix.readthedocs.io/en/latest/launching.html. Fetched 2026-08-11.

[20] Tunix Launching Jobs page, Config Explanation section. https://tunix.readthedocs.io/en/latest/launching.html. Fetched 2026-08-11.

[21] Tunix Performance Considerations page, Batching Config and Collocated vs Disaggregated Training sections. https://tunix.readthedocs.io/en/latest/performance.html. Fetched 2026-08-11.

[22] Tunix Rollout page, vLLM section (config fields, WIP notes, Troubleshooting). https://tunix.readthedocs.io/en/latest/rollout.html. Fetched 2026-08-11.

[23] Tunix Rollout page, SGLang section (config fields, data-parallel and logprobs limitations, Troubleshooting). https://tunix.readthedocs.io/en/latest/rollout.html. Fetched 2026-08-11.

[24] Tunix Reliability page, Fault Tolerance section (InflightThrottler). https://tunix.readthedocs.io/en/latest/reliability.html. Fetched 2026-08-11.

[25] Tunix Design Overview page, SFT and RL pipeline component descriptions (Metrics Logger, Orchestrator). https://tunix.readthedocs.io/en/latest/design.html. Fetched 2026-08-11.

[26] Tunix Metrics page, Collected Metrics / Metric Loggers / Performance Metric Tracing sections. https://tunix.readthedocs.io/en/latest/metrics.html. Fetched 2026-08-11.

[27] Tunix Reliability page, Checkpoint Support section. https://tunix.readthedocs.io/en/latest/reliability.html. Fetched 2026-08-11.

[28] Tunix Models page, Model Sources / AutoModel / Model Download Path sections. https://tunix.readthedocs.io/en/latest/models.html. Fetched 2026-08-11.

[29] Tunix documentation, version-tag URL check: https://tunix.readthedocs.io/en/v0.1.7/index.html (404) versus https://tunix.readthedocs.io/en/stable/index.html and https://tunix.readthedocs.io/en/latest/index.html (200). Fetched 2026-08-11.

[30] Tunix documentation index, left-navigation page list. https://tunix.readthedocs.io/en/latest/index.html. Fetched 2026-08-11.

[31] Tunix Examples and Guides page (examples/ tree contents). https://tunix.readthedocs.io/en/latest/examples.html. Fetched 2026-08-11.

[32] Tunix Talks and Announcements page (Announcements and Blogs / Talks tables). https://tunix.readthedocs.io/en/latest/talks.html. Fetched 2026-08-11.
