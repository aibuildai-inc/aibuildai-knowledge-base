# NVIDIA-NeMo/Skills

A pipeline-orchestration layer, not a trainer: it wraps NeMo-RL and verl behind one `ns` CLI and scales the same command from a laptop to a Slurm cluster.

Nemo-Skills "is a collection of pipelines to improve \"skills\" of large language models (LLMs)", covering synthetic data generation, model training and evaluation, and its README's stated design goal is to "start developing on a local workstation and move to a large-scale Slurm cluster with just a one-line change" [1]. It is built by NVIDIA under the NVIDIA-NeMo GitHub organization and carries a research-only disclaimer: "This project is strictly for research purposes, and not an official product from NVIDIA" [1]. Its API is a `typer`-based CLI, `ns <command>` (e.g. `ns generate`, `ns nemo_rl sft`, `ns nemo_rl grpo`, `ns verl ppo`), each command building a shell command that runs a training/inference/eval script inside a prebuilt container, launched locally or through NeMo-Run onto Slurm [2][3]. It lives at https://github.com/NVIDIA-NeMo/Skills [1].

**When to pick it**: pick Nemo-Skills only if you already want its surrounding pipeline - synthetic data generation, a wide multi-benchmark evaluation suite, and Slurm-scale job orchestration - and are willing to train through NeMo-RL, verl, or a raw Megatron-LM entrypoint underneath; do not pick it as a standalone trainer, because the training pipeline package itself ships four commands (`ns nemo_rl sft`, `ns nemo_rl grpo`, `ns verl ppo`, `ns megatron_lm train`) [3], each just formats a config string and shells out to the wrapped framework's own training script inside its container [4][5][17] - the method's real knob surface, logged metrics and checkpoint format are NeMo-RL's, verl's, or Megatron-LM's, not Nemo-Skills'. If you want a self-contained trainer library, compare against the trl or verl cards directly (cross-reference; not covered here).

**Methods it ships**: four pipeline commands, each a thin CLI wrapper with no method logic of its own: SFT and GRPO run through NeMo-RL (`nemo_skills/pipeline/nemo_rl/sft.py`, `nemo_skills/pipeline/nemo_rl/grpo.py`), PPO runs through verl (`nemo_skills/pipeline/verl/ppo.py`), and a generic pretraining/SFT entrypoint runs through Megatron-LM (`nemo_skills/pipeline/megatron_lm/train.py`) [3]; the verl wrapper's own name is `ppo.py` but its committed default config sets `algorithm.adv_estimator=grpo` unless overridden [5], so the out-of-the-box run under that command is GRPO, not vanilla PPO. The Megatron-LM command takes an arbitrary `--entrypoint` script name (e.g. `pretrain_gpt.py`) rather than shipping a fixed method of its own - it is a generic launcher, not a method wrapper [17]. No DPO/KTO/ORPO/other offline-preference pipeline exists anywhere under `nemo_skills/pipeline/` at this commit [3] - the method math, knobs and training-signal semantics for SFT/GRPO/PPO live on those methods' own cards, not here.

**Scale it handles**: single machine (`--cluster=local`, or no `--cluster` at all with your own installed environment [6]) up to multi-node Slurm, launched through NeMo-Run with a per-cluster YAML config selected by `--cluster` and generated interactively with `ns setup` [7]. `--num_nodes` / `--num_gpus` select node and GPU-per-node counts for both the NeMo-RL and verl commands [3]. NeMo-RL training picks its own parallelism inside the launched job (`--backend=fsdp` or `megatron`, plus `++policy.megatron_cfg.tensor_model_parallel_size`, `expert_model_parallel_size`, `context_parallel_size`, and `++policy.sequence_packing.enabled` for long sequences) [4]; verl/PPO scales via Ray, submitted with `ray job submit` against a local Ray head the launcher starts in the job [5]. No published benchmark numbers for multi-node scale are given by these docs; only the launch mechanism is documented.

**Install**: no PyPI package exists for this project (`pypi.org/pypi/nemo_skills/json` and the dash-named variant both return 404, checked 2026-08-12); the docs' only supported install is git-based - `git clone https://github.com/NVIDIA-NeMo/Skills.git && cd Skills && pip install -e .`, or `pip install git+https://github.com/NVIDIA-NeMo/Skills.git` [6]. `pyproject.toml` requires Python >=3.10 and is Apache-2.0 licensed [8][1]; its version is read dynamically from `nemo_skills/version.py`, which at the screening commit `e06c9b9` reads `0.7.0` [9] - this is ahead of the repository's only GitHub Release-adjacent tag, `v0.1` (commit `74590c6`, an old snapshot unrelated to the current `0.7.x` line) [10], so there is no meaningful pinned release to resolve here: install is a live clone of `main` (or whatever ref you check out) rather than a versioned package. Dependencies come from `core/requirements.txt` and `requirements/pipeline.txt` (declared as dynamic dependency files in `pyproject.toml` [8]); the pipeline extra pins `nemo-evaluator-launcher<0.1.47`, `nemo_run` from its own git repo, and `typer>=0.16,<0.27` [11]; the core requirements pin `litellm[caching]==1.84.10` and `mcp<2.0` [12], but neither file pins a deep-learning core (torch, transformers version) directly - Nemo-Skills itself does not train in-process, so its own package carries no torch pin. Training happens inside separate, prebuilt containers: the NeMo-RL container the sample cluster config points to is the upstream release image `nvcr.io/nvidia/nemo-rl:v0.6.0` [13], and the verl container is built `FROM whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3` with verl checked out at commit `2ed63bbf39c22724e4940d97e4b09e4f3e5f6d68` [14] - CUDA 12.4 and vLLM 0.8.5 are the load-bearing pins for anyone running PPO/GRPO-via-verl. No hardware/CUDA minimum is stated for Nemo-Skills' own package; the containers above are where that constraint actually lives.

**Maintained by**: NVIDIA, under the NVIDIA-NeMo GitHub organization [2]; about 1,021 GitHub stars (not a ranking signal) [2]. Actively pushed - the repository's last push was 2026-08-10 [2] - and the README's News section lists dated entries through 2025-12-15, including a recipe release tied to the NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 model [1]. A repo-top notice states benchmarks and rollouts are migrating to a sibling project, NeMo-Gym, with new benchmark work directed there instead of into Nemo-Skills [1].

## Quick start

The README's quick-start path is inference, not training - Nemo-Skills' own quickstart is `ns generate`, run against an API model with a prompt-format config, quoted from the Getting Started guide [6]:

```bash
export NVIDIA_API_KEY=<your key>
ns generate \
    --server_type=openai \
    --model=meta/llama-3.1-8b-instruct \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --output_dir=./generation \
    --input_file=./input.jsonl \
    ++prompt_config=./prompt.yaml
```

For training, the docs' smallest complete example is `ns nemo_rl sft` for SFT on 8 nodes x 8 GPUs, after preparing data with `nemo_skills.training.prepare_data` [4]:

```bash
python -m nemo_skills.training.prepare_data \
    ++input_files="<path to the generated synthetic data>/output-rs*.jsonl" \
    ++output_path=sft-data.jsonl \
    ++prompt_config=generic/math \
    ++tokenizer=meta-llama/Llama-3.1-8B-Instruct

ns nemo_rl sft \
    --cluster=slurm \
    --expname=my-training-job \
    --output_dir=/workspace/my-training-job/checkpoints \
    --hf_model=meta-llama/Llama-3.1-8B \
    --num_nodes=8 \
    --num_gpus=8 \
    --dependent_jobs=3 \
    --backend=megatron \
    --training_data=/data/sft-data.jsonl \
    ++sft.val_period=0
```

## Start it

- One process, one GPU: omit `--cluster` (or use `--cluster=local`) and Nemo-Skills runs the command directly, printing a warning that only a subset of features is supported outside a defined cluster config [6].
- Multi-GPU / multi-node goes through NeMo-Run's Slurm executor, selected with `--cluster=<name>` picking up a YAML from your `cluster_configs/` folder (or `--config_dir` / `NEMO_SKILLS_CONFIG_DIR`); `ns setup` walks through generating one interactively, and Slurm-only fields cover account, partition, ssh-tunnel, and mounts [7]. `--num_nodes` and `--num_gpus` (GPUs per node) set the job's GPU footprint for both the NeMo-RL and verl commands [3].
- verl/PPO's launcher is Ray underneath: the pipeline submits the training script with `ray job submit --address='http://127.0.0.1:8265' --` inside the scheduled job, not a bare `torchrun`/`accelerate` call [5].
- Effective-batch arithmetic is not Nemo-Skills' own - it forwards to whichever framework's config surface: the default NeMo-RL GRPO config exposes `train_global_batch_size` and `train_micro_batch_size` as separate policy fields you override with `++policy.train_global_batch_size=32 ++policy.train_micro_batch_size=1` [4], and the default verl/PPO command line sets `data.train_batch_size=128`, `actor_rollout_ref.actor.ppo_mini_batch_size=64`, plus `use_dynamic_bsz=True` with a `ppo_max_token_len_per_gpu=32768` cap [5].
- Config surface: each command is a `typer` CLI with its own flags (`--hf_model`, `--backend`, `--num_nodes`, `--wandb_project`, ...) plus a Hydra-style `++key=value` passthrough into the wrapped framework's own YAML config - e.g. `++sft.max_num_epochs`, `++policy.megatron_cfg.tensor_model_parallel_size` for NeMo-RL [4], or `algorithm.adv_estimator=grpo`, `actor_rollout_ref.rollout.gpu_memory_utilization=0.85` for verl [5]. Nemo-Skills does not publish a changed-default table of its own for these method configs; the committed NeMo-RL GRPO default config used by the wrapper sets `checkpointing.save_period: 10`, `logger.wandb_enabled: false`, and `grpo.max_num_steps: 1000000` [15], and the verl/PPO wrapper's own hardcoded defaults (used only when no `--verl_config_name` is given) set `trainer.save_freq=20`, `algorithm.kl_ctrl.kl_coef=0`, and `actor_rollout_ref.rollout.gpu_memory_utilization=0.85` [5].
- Out-of-memory first aid is method-specific and documented only for the NeMo-RL SFT path: the docs recommend tuning micro batch size, max sequence length, and parallelism parameters for optimal performance, and note that when sequence packing is enabled it is best to keep the micro batch size at 1 and instead increase the sequence-packing length if GPU memory is underused [4]; for sequences over roughly 4k tokens the docs recommend enabling `++policy.sequence_packing.enabled=True` together with `++policy.megatron_cfg.context_parallel_size=4` [4]. verl/PPO's generation-side memory knob is `actor_rollout_ref.rollout.gpu_memory_utilization` (default 0.85 in the wrapper's own command) [5]; Nemo-Skills' docs give no further OOM guidance for the verl/PPO path.

## Watch it

This section is mechanics only: what a metric MEANS for SFT, GRPO or PPO lives on that method's own card, and the metric names themselves are NeMo-RL's and verl's, not Nemo-Skills' - Nemo-Skills' pipeline layer only sets the wandb/tensorboard *plumbing* around the wrapped framework's run.

- **Enable it**: both wrapper commands take `--wandb_project`, `--wandb_group` (NeMo-RL only) and `--disable_wandb`; unless disabled, `ns nemo_rl grpo`/`sft` set `++logger.wandb_enabled=True` and construct a wandb run id from `expname`+`wandb_group`+`wandb_project` [4], while `ns verl ppo` sets verl's `trainer.logger=['console','wandb']` (or `['console']` if `--disable_wandb`) [5]. `WANDB_API_KEY` must be exported for this to work; the cluster-config docs list it as "only needed for training (can opt-out with `--disable_wandb`)" [7]. With wandb disabled and no other tracker enabled, a run has console output only - Nemo-Skills does not itself turn on TensorBoard, MLflow, or SwanLab; those are the wrapped NeMo-RL config's own `logger.tensorboard_enabled` / `mlflow_enabled` / `swanlab_enabled` fields, all `false` by default in the committed GRPO config [15].
- **Metric names**: not documented on any Nemo-Skills page read for this card - the wrapper only forwards `logger.*` and `trainer.logger` settings, so the actual scalar names (loss, KL, reward, entropy, etc.) are whatever NeMo-RL or verl log under those backends; consult those projects' own docs/cards for the field lists.
- **Sample-level logging of generations**: verl/PPO's wrapper hardcodes `+trainer.val_generations_to_log_to_wandb=1` in its own default command [5]; the NeMo-RL GRPO config used by the SFT/GRPO wrapper exposes `logger.num_val_samples_to_print` (validation samples pretty-printed to the terminal) [15]. Neither is a Nemo-Skills feature - both are pass-through config keys of the wrapped framework.
- **Evaluation during training**: Nemo-Skills does not run eval concurrently inside the training job; its documented pattern is to chain a separate `ns eval` job after training finishes, using `--run_after=<training-expname>` to set a Slurm dependency, as shown in the docs' Python-API chaining example [4]. NeMo-RL's own `val_period` config key (set to `0` to disable in the SFT quickstart example [4]) controls in-training validation, but that is NeMo-RL's mechanism, not Nemo-Skills'.
- **Stopping rule**: no RL-specific stopping rule, threshold, or patience value is published on any Nemo-Skills page read for this card (README [1], training pipeline docs [4], cluster-configs docs [7], the committed default GRPO config [15]) - training runs for `++sft.max_num_epochs` / `++sft.max_num_steps` (SFT) or the wrapped framework's own step/epoch budget (`grpo.max_num_steps: 1000000` by default in the committed GRPO config [15]; `trainer.total_epochs=30` in the verl/PPO wrapper's own default command [5]), with no adaptive early-stop mechanism documented at the Nemo-Skills layer.

## Save it

- Both wrapper commands point the wrapped framework's checkpoint directory at `<output_dir>/checkpoints` (`++checkpointing.checkpoint_dir={output_dir}/checkpoints` for NeMo-RL [4]; `trainer.default_local_dir={output_dir}/checkpoints` for verl [5]) - the on-disk checkpoint layout inside that directory (which files hold weights, what a retention flag drops) is the wrapped framework's own contract, not documented on any Nemo-Skills page read here.
- Both paths require an explicit conversion step to reach a directly-loadable HF model: the NeMo-RL wrapper runs `nemo_skills.training.nemo_rl.convert_dcp_to_hf` (fsdp backend) or `convert_megatron_to_hf` (megatron backend) against `--training-folder=<output_dir>` and writes to `--hf-ckpt-path`, defaulting to `<output_dir>/final_hf_model` if `--final_hf_path` is not given [4]; a specific step can be targeted with `--conversion_step`, and `--average_steps` triggers `nemo_skills.pipeline.nemo_rl.average_checkpoints` beforehand [4]. The verl/PPO wrapper's checkpoint conversion runs `python3 -m verl.utils.checkpoint.convert_checkpoint` against the actor subdirectory of the latest global step (`<output_dir>/checkpoints/global_step_<N>/actor`), writing by default to `<output_dir>/final_hf_checkpoint` unless `--final_ckpt_path` is given, gated by the boolean flag `--convert_last_ckpt_to_hf` [5].
- No adapter/PEFT save path is mentioned in either wrapper's source read for this card [4][5] - both target full-model training and conversion.
- Loader handoff: the converted `final_hf_model` / `final_hf_checkpoint` directory is a standard HF model directory - the training docs' own follow-up chaining example loads it straight into `ns eval` via `model=f"{output_dir}/final_hf_model"` with `server_type="trtllm"` [4], so an evaluator that accepts a HF model path can load it directly once conversion has run; an un-converted native checkpoint (DCP/Megatron/verl actor shards) is not that format and needs the conversion step above first.

## Find it in the docs

The docs site (`https://nvidia-nemo.github.io/Skills/`, built with `mkdocs` + Material [16]) is a single live build with no version-tag path scheme - `mkdocs.yml` configures no versioning plugin (e.g. `mike`) [16], so there is exactly one URL per page, not a `/vX.Y/` tree; treat every page as `main`-tracking and unpinned.

- Address pattern: `https://nvidia-nemo.github.io/Skills/<section>/<page>` mirroring the `docs/` folder structure in the repo (e.g. `docs/pipelines/training.md` -> `.../pipelines/training/`, `docs/basics/cluster-configs.md` -> `.../basics/cluster-configs/`) [1][4][7].
- Question-to-page map: install/first-run walkthrough -> Getting Started (`docs/basics/index.md`, linked from the README as "steps") [1][6]; how jobs get placed on local vs. Slurm -> Cluster Configs (`docs/basics/cluster-configs.md`) [7]; how to run SFT/GRPO/PPO -> Model Training (`docs/pipelines/training.md`) [4]; the full benchmark list by category (math, code, tool-calling, multilingual, VLM, ...) -> the Evaluation section, linked per-category from the README [1]; how synthetic data / inference jobs are structured -> Generation (`docs/pipelines/generation/`) [1].
- Runnable references beyond the docs: the repo's own `recipes/` tree (e.g. `recipes/proof-gen-verification` reproduces the arXiv:2511.13027 experimental results [1]) and the `releases/` documentation index, which the README calls "Papers & Releases" and links to for every model/dataset combination Nemo-Skills has been used to produce (OpenMathInstruct-2, OpenMathReasoning, OpenReasoning, Nemotron-Math-v2, ...) [1].
- Community layer, curated door first: the docs site runs `mkdocs`'s `blog` plugin with `blog_dir: tutorials` and `post_dir: tutorials/posts` [16], so `https://nvidia-nemo.github.io/Skills/tutorials/` is a maintainer-curated tutorials/blog index, not just the README - the README's own News section links directly into it, e.g. "Added details for reproducing evals" for the NVIDIA-Nemotron-Nano-9B-v2 model at `.../tutorials/2025/08/22/reproducing-nvidia-nemotron-nano-9b-v2-evals/` and a second entry for Llama-3_3-Nemotron-Super-49B-v1_5 at `.../tutorials/2025/08/15/...` [1]. No separate MCP endpoint for the docs was found on the pages read for this card.
- Honest boundary: the pipeline layer covers SFT and GRPO via NeMo-RL, PPO (defaulting to a GRPO advantage estimator) via verl, and a generic entrypoint launcher via Megatron-LM [3][5][17] - no DPO, KTO, ORPO, or other offline-preference method has a pipeline command at this commit, even though the wrapped frameworks may support more methods on their own. A repo-top README notice states that benchmark/rollout development is moving to a sibling project, NeMo-Gym, with an experimental early command, `ns nemo_gym_rollouts`, as the current bridge [1].

## Sources

Every claim above about pipeline behavior is read directly from the `nemo_skills/pipeline/` source at commit `e06c9b900177be3f60d6a3f99135bb5de9af9bed` (`e06c9b9` in short form) - the repository's newest push, fetched 2026-08-12, not a tagged release (no meaningful release tag exists for this project; see the Install field). Docs pages are the live, unversioned `nvidia-nemo.github.io/Skills` build, fetched 2026-08-12 via the mirrored Markdown source in the repository at the same commit. Ecosystem projects Nemo-Skills wraps (NeMo-RL, verl, NeMo-Run, Ray) are named only where the wrapper code itself names them and are not otherwise documented here - their own method math, config surface and logged metrics belong on their own cards.

[1] NVIDIA-NeMo/Skills README. https://github.com/NVIDIA-NeMo/Skills/blob/e06c9b900177be3f60d6a3f99135bb5de9af9bed/README.md. Fetched 2026-08-12.

[2] NVIDIA-NeMo/Skills repository (GitHub API). https://api.github.com/repos/NVIDIA-NeMo/Skills. Fetched 2026-08-12.

[3] `nemo_skills/pipeline/` directory listing at commit e06c9b9 (training-launching commands are `nemo_rl/{sft,grpo}.py`, `verl/ppo.py`, and `megatron_lm/train.py`; no other method directory exists). https://github.com/NVIDIA-NeMo/Skills/tree/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/pipeline. Fetched 2026-08-12.

[4] `nemo_skills/pipeline/nemo_rl/sft.py`, `nemo_skills/pipeline/nemo_rl/grpo.py`, and the Model Training docs page. https://github.com/NVIDIA-NeMo/Skills/blob/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/pipeline/nemo_rl/sft.py ; https://github.com/NVIDIA-NeMo/Skills/blob/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/pipeline/nemo_rl/grpo.py ; https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/docs/pipelines/training.md. Fetched 2026-08-12.

[5] `nemo_skills/pipeline/verl/ppo.py`. https://github.com/NVIDIA-NeMo/Skills/blob/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/pipeline/verl/ppo.py. Fetched 2026-08-12.

[6] Getting Started docs page. https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/docs/basics/index.md. Fetched 2026-08-12.

[7] Cluster Configs docs page. https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/docs/basics/cluster-configs.md. Fetched 2026-08-12.

[8] `pyproject.toml` at commit e06c9b9. https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/pyproject.toml. Fetched 2026-08-12.

[9] `nemo_skills/version.py` at commit e06c9b9. https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/version.py. Fetched 2026-08-12.

[10] Repository tags (GitHub API) and the `v0.1` tag's commit. https://api.github.com/repos/NVIDIA-NeMo/Skills/tags ; https://api.github.com/repos/NVIDIA-NeMo/Skills/commits/74590c6ec828f0efedc091b9731c159bd5cbacaa. Fetched 2026-08-12.

[11] `requirements/pipeline.txt` at commit e06c9b9. https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/requirements/pipeline.txt. Fetched 2026-08-12.

[12] `core/requirements.txt` at commit e06c9b9. https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/core/requirements.txt. Fetched 2026-08-12.

[13] `dockerfiles/README.md` at commit e06c9b9 (states the sample local cluster config defaults `containers.nemo-rl` to `nvcr.io/nvidia/nemo-rl:v0.6.0`). https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/dockerfiles/README.md. Fetched 2026-08-12.

[14] `dockerfiles/Dockerfile.verl` at commit e06c9b9. https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/dockerfiles/Dockerfile.verl. Fetched 2026-08-12.

[15] `nemo_skills/training/nemo_rl/configs/grpo.yaml` at commit e06c9b9 (the committed default config the `ns nemo_rl grpo` wrapper launches with). https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/training/nemo_rl/configs/grpo.yaml. Fetched 2026-08-12.

[16] `mkdocs.yml` at commit e06c9b9. https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/mkdocs.yml. Fetched 2026-08-12.

[17] `nemo_skills/pipeline/megatron_lm/train.py` at commit e06c9b9. https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/pipeline/megatron_lm/train.py. Fetched 2026-08-12.
