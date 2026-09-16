# reasoning-gym

A procedural dataset/environment generator for RLVR, not a trainer: it manufactures verifiable reasoning tasks on demand, and ships a verl-based reference training pipeline that reproduces its paper's GRPO runs rather than a first-party trainer of its own.

**Reasoning Gym** (repository: https://github.com/open-thought/reasoning-gym [2]) is described in its own README as "a community-created Python library of procedural dataset generators and algorithmically verifiable reasoning environments for training reasoning models with reinforcement learning (RL)" [1]. It is built by the Open-Thought community [2] and is the paper behind the NeurIPS 2025 Spotlight "REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards" [3]. Its core API is a factory function, `reasoning_gym.create_dataset(name, size, seed, **task_kwargs)`, returning an iterable dataset whose entries carry a `question`, `answer`, and `metadata`, plus a `score_answer` method for verifying a model's response against that entry [1].

**When to pick it**: pick it as a data source for RLVR post-training - over 100 procedural task generators across algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and games, each with adjustable difficulty and its own verifier, so training data does not need to be pre-collected or human-labeled [1][3]. The paper's own intra-domain GRPO ablation is the number that decides whether this data trains anything: RL-fine-tuning Qwen2.5-3B-Instruct on same-category RG tasks moved Acc@3 from a 5.0 baseline to 16.7 on Algebra (+11.7), 89.7 to 96.0 on Arithmetic (+6.3), and, most tellingly, 0.0 to 3.3 on Games - the base model solved zero Games problems before training and gained measurable capability after [3]. It ships no first-party trainer or model-serving stack; the repository's own `training/` example is a research-reproduction harness built by subclassing verl's `RayPPOTrainer`, pinned to a specific old verl commit, so choosing reasoning-gym does not by itself decide your training framework - it supplies data and reward functions to whatever RL framework you already run [4][5]. The README itself points at the `verifiers` library as the easiest way to start training with RG tasks, while noting the data works with any major RL framework [1].

**Methods it ships**: reasoning-gym ships no trainer classes of its own. The one training method demonstrated in the repository is GRPO, implemented as `RayGRPOTrainer` in `training/trainers/ray_grpo_trainer.py`, a subclass of verl's `RayPPOTrainer` explicitly adapted from verl's `main_ppo.py` at a named upstream commit [5]. Its GRPO config sets `algorithm.adv_estimator: grpo`, `actor_rollout_ref.actor.use_kl_loss: True`, and `actor_rollout_ref.rollout.n: 8` (`> 1 for grpo`) on top of verl's standard PPO-trainer config [6]. Method semantics (GRPO's objective, advantage estimator, KL term) are not restated here - see the GRPO methodology card.

**Scale it handles**: whatever verl at the pinned commit handles, since the training loop is verl's `RayPPOTrainer` unmodified in its distributed mechanics - Ray-orchestrated actor/rollout/ref workers, FSDP sharding (`actor_rollout_ref.actor.strategy: fsdp`), and vLLM rollout generation with its own `tensor_model_parallel_size` [5][6]. The shipped example config targets one 4-GPU node (`trainer.n_gpus_per_node: 4`, `trainer.nnodes: 1`) and the training README shows an explicit 2-GPU rescale by editing `tensor_model_parallel_size` and `n_gpus_per_node` on the command line [4][6]. No multi-node example config or benchmark is published in this repository; the README only states the requirement to install a specific old verl commit paired with vLLM 0.7.3 and warns that newer verl versions may be incompatible because this code overrides verl internals [4].

**Install**: `pip install reasoning-gym`, PyPI version 0.1.25, released 2026-03-28 [7]; Python >= 3.10; Apache-2.0 [1][8]. The base package pins no deep-learning framework at all - dependencies are generator-support libraries (`sympy>=1.13.1`, `magiccube==0.3.0`, `pycosat==0.6.6`, `arckit==0.1.0`, etc.), with no torch/transformers requirement in `pyproject.toml` [8]. Optional extras: `[scoring]` adds `math-verify>=0.7.0` for symbolic-math answer checking; `[cli]`/`[server]` add a FastAPI/Typer experiment-management stack unrelated to training [8]. The `[scoring]` extra is not yet on PyPI - `pyproject.toml` at the tip commit used for this card (2026-04-17, ahead of the v0.1.25 release) already lists it, but the v0.1.25 release tag's `pyproject.toml` does not, so `pip install reasoning-gym[scoring]` will fail against the current PyPI release [8][9]. The separate `training/` pipeline is unpinned by this package's own install line: its README pins verl at commit `c34206925e2a50fd452e474db857b4d488f8602d` with vLLM 0.7.3 and `flash-attn==2.7.3`, installed by hand as a second step, not through `pip install reasoning-gym` [4]. No CUDA/hardware minimum is stated anywhere in the base package's docs; the training README notes its own experiments used Python 3.11 and CUDA 11.8 and that other versions "may" need tweaking [4].

**Maintained by**: the Open-Thought community, contact listed as `andreas.koepf@xamla.com` in the package metadata [8]; the GitHub repository shows 1479 stars (not a ranking signal) and its most recent push is 2026-04-17 [10]. Release history on GitHub runs v0.1.20 (2025-06-04), v0.1.22 (2025-06-06), v0.1.23 (2025-07-05), v0.1.24 (2025-09-29), v0.1.25 (2026-03-28) [11] - active early on, but with an almost six-month gap between the last two tags.

## Quick start

From the README's quickstart, a complete generate-and-verify loop [1]:

```python
import reasoning_gym
data = reasoning_gym.create_dataset('leg_counting', size=10, seed=42)
for i, x in enumerate(data):
    print(f'{i}: q="{x["question"]}", a="{x["answer"]}"')
    print('metadata:', x['metadata'])
    assert data.score_answer(answer=x['answer'], entry=x) == 1.0
```

Task-specific configuration is passed as keyword arguments, e.g. `reasoning_gym.create_dataset('leg_counting', size=10, seed=42, max_animals=20)`, and multiple task types can be mixed into one composite dataset with relative weights via `reasoning_gym.composite.DatasetSpec` [1]. There is no CLI form for dataset generation itself; the `rgc` console script installed by the package is an experiment-management client for a separate FastAPI server extra, not a data-generation CLI [8][12].

## Start it

- reasoning-gym itself starts nothing - it is called from inside whatever training loop you run. The repository's own reference loop is launched as a single Python process that internally drives Ray workers: `python3 -u train_grpo.py --config-path configs/inter_generalisation --config-name algorithmic_qwen_3b`, run from `training/` after installing reasoning-gym, the pinned verl commit, and flash-attention [4].
- Scaling that reference loop down is a command-line override of the same script's Hydra config, not a separate launcher: the README's 2-GPU example runs `python3 -u train_grpo.py --config-path configs/inter_generalisation --config-name algorithmic_qwen_3b actor_rollout_ref.rollout.tensor_model_parallel_size=1 trainer.n_gpus_per_node=2 trainer.project_name=rg-grpo trainer.experiment_name=algorithmic_qwen2.5_3b`, and GPU visibility is controlled by setting `CUDA_VISIBLE_DEVICES` before launch [4].
- Effective batch arithmetic in the shipped `algorithmic_qwen_3b.yaml` config: `data.train_batch_size: 32` prompts per step, each expanded to `actor_rollout_ref.rollout.n: 8` GRPO samples, with PPO updates split into `actor_rollout_ref.actor.ppo_mini_batch_size: 16` and `ppo_micro_batch_size_per_gpu: 8` [6].
- The config surface is entirely verl's `ppo_trainer` config, included via Hydra's `searchpath` and extended with a `reasoning_gym:` block naming which task generators (and per-task kwargs) feed the run, plus a `curriculum:` block for optional automatic difficulty scheduling [6]. reasoning-gym does not change any of verl's own defaults (precision, checkpointing cadence, etc.) - the shipped config sets `actor_rollout_ref.rollout.dtype: bfloat16` and `enable_gradient_checkpointing: True` explicitly, matching verl's own defaults rather than overriding them [6].
- Out-of-memory first aid is not separately documented by reasoning-gym; the generation-side knob available in the shipped config is `actor_rollout_ref.rollout.gpu_memory_utilization: 0.7`, verl's vLLM memory-share setting [6]. No reasoning-gym-specific OOM guidance is published in the README or training README read for this card [1][4].

## Watch it

- Logging is verl's: the shipped config sets `trainer.logger: ['console', 'wandb']`, and the training README tells you to run `wandb login` before training so runs upload [4][6]. reasoning-gym adds no metric names of its own to the logging surface documented here; it supplies reward values through its `score_answer` function, consumed as the accuracy term in the config's `reward:` block (`reward.use_accuracy: True`), alongside optional secondary rewards for format and length that the config lists with per-reward `scaling_factor` weights [6].
- Metric shapes, KL terms, and clip-fraction fields belong to verl's own PPO/GRPO trainer, not to this repository - see the verl card for the live metric-name pages; this card does not restate them.
- No reasoning-gym-specific sample-level generation logging, evaluation-during-training field, or published stopping-rule/threshold was found in the README or training README read for this card [1][4]; evaluation is a separate offline step run after training (see Save it/eval below), not a during-training callback documented here.

## Save it

- Checkpointing during training is verl's FSDP mechanism, configured in the shipped config as `trainer.save_freq: 100`, `trainer.default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}`, and `trainer.resume_mode: auto` (auto-resumes from the last checkpoint found, or starts fresh if none exists) [6]. Checkpoints land as sharded FSDP state, one file per rank under `.../global_step_<N>/actor/`, per the training README's example path `checkpoints/rg-test/.../global_step_400/actor/` [4].
- These sharded checkpoints are not directly loadable by transformers - the repository provides `training/utils/load_fsdp_to_hf.py` to convert them: it loads each `model_world_size_{world_size}_rank_{rank}.pt` shard, concatenates the tensors, and calls `model.save_pretrained(output_path, max_shard_size="10GB")` plus `tokenizer.save_pretrained(output_path)` to produce a standard Hugging Face checkpoint directory [13]. The script hardcodes `world_size = 4` internally rather than reading it from the checkpoint or an argument, so it must be edited by hand for any run that did not use exactly 4 FSDP ranks [13].
- Resume is verl's `trainer.resume_mode`/`resume_from_path` fields on the same config, not a reasoning-gym API [6].
- Loader handoff: only the converted, `save_pretrained`-written directory is a full Hugging Face model directory an evaluator can load with `AutoModelForCausalLM.from_pretrained` directly; the raw FSDP shard directory is not [13].
- Separately, for building static training data ahead of time (rather than sampling reasoning-gym live during RL), `scripts/hf_dataset/save_hf_dataset.py` generates a dataset from named task generators and pushes it to the Hugging Face Hub - `--repo-id` is a required argument and the script's only save path calls `dataset.push_to_hub(repo_id, ...)`, there is no local-save option [14]. The README states each saved row carries `question`, `answer`, and `metadata` columns [1].

## Find it in the docs

reasoning-gym has no hosted docs site (no ReadTheDocs/Hugging Face docs page found from the repository or PyPI listing); the GitHub repository is the source of truth, and this section is a repo map rather than a URL-pattern recipe.

- Repository home: https://github.com/open-thought/reasoning-gym [2]. Task-by-task examples of every registered generator are in `GALLERY.md` at the repo root, linked from the README [1].
- The dataset generators themselves live under `reasoning_gym/` in the package, organized into the domain subpackages named in the README (algebra, arithmetic, computation, cognition, geometry, graph theory, logic, games) [1]; `reasoning_gym.factory.DATASETS` is the name-to-generator registry used by `create_dataset` and by `scripts/hf_dataset/save_hf_dataset.py` to validate requested dataset names [14].
- The GRPO reproduction pipeline is entirely under `training/`: `training/README.md` documents setup (including the pinned verl commit and vLLM/flash-attention versions), `training/train_grpo.py` is the entry point, `training/configs/` holds the Hydra YAML configs actually used for the paper's experiments (e.g. `inter_generalisation/algorithmic_qwen_3b.yaml`), and `training/trainers/ray_grpo_trainer.py` is the verl-subclassing trainer itself [4][5][6].
- Offline evaluation of a trained checkpoint on RG tasks is documented in `training/README.md` via `training/evaluation/evaluate_model.py --config <yaml>`; evaluation against external benchmarks (MATH, GSM8K, MMLU-Pro) is documented there too, run through EleutherAI's `lm-evaluation-harness` with RG-specific task configs supplied under `training/evaluations/lmeh/` [4].
- Community/ecosystem: the README lists downstream projects that build on reasoning-gym data, including `verifiers` (an RL-with-verifiable-environments library the README recommends as the easiest way to start training), NVIDIA's ProRL, Nous Research's Atropos, and PrimeIntellect's SYNTHETIC-2, among others [1]. No official MCP endpoint for these docs was found.
- Trap, stated where it bites: the training README itself warns that the pinned verl commit is required because the reproduction code overrides verl internals, and that "you may alternatively wish to try newer verl versions... However, our code does override some verl code, so there may be incompatibilites with newer versions" [4] - this is a maintainer-authored warning in the docs, not an issue-tracker report; no closed-issue trap report was found in the pages read for this card.
- Honest boundary: reasoning-gym is a data/verifier library, not a trainer - it has no first-party distributed training, checkpointing, or logging system of its own; every scale, save, and watch mechanism documented above belongs to verl, pinned to a specific old commit in the one example the repository ships [4][5][6].

## Sources

All pages were read at the commit or release named per claim; no maintained hosted-docs site was found, so most citations are the repository's own README files and source at the shortlisted commit `49b07130b3fcd12f2d064bba7c43869543a0e7e7` (2026-04-17 push), fetched 2026-08-12.

[1] reasoning-gym README, at commit 49b07130b3fcd12f2d064bba7c43869543a0e7e7. https://raw.githubusercontent.com/open-thought/reasoning-gym/49b07130b3fcd12f2d064bba7c43869543a0e7e7/README.md. Fetched 2026-08-12.

[2] reasoning-gym GitHub repository. https://github.com/open-thought/reasoning-gym. Fetched 2026-08-12 (via GitHub API, https://api.github.com/repos/open-thought/reasoning-gym).

[3] "REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards" (arXiv abstract page, NeurIPS 2025 Spotlight note; Table 1 intra-domain Acc@3 baseline-vs-RG-RLVR numbers read from the full PDF). https://arxiv.org/abs/2505.24760 (abstract), https://arxiv.org/pdf/2505.24760 (PDF, Table 1). Fetched 2026-08-12.

[4] training/README.md, at commit 49b07130b3fcd12f2d064bba7c43869543a0e7e7. https://raw.githubusercontent.com/open-thought/reasoning-gym/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/README.md. Fetched 2026-08-12.

[5] training/trainers/ray_grpo_trainer.py, at commit 49b07130b3fcd12f2d064bba7c43869543a0e7e7 (header credits verl's main_ppo.py at commit a65c9157bc0b85b64cd753de19f94e80a11bd871 as its source). https://raw.githubusercontent.com/open-thought/reasoning-gym/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/trainers/ray_grpo_trainer.py. Fetched 2026-08-12.

[6] training/configs/inter_generalisation/algorithmic_qwen_3b.yaml, at commit 49b07130b3fcd12f2d064bba7c43869543a0e7e7. https://raw.githubusercontent.com/open-thought/reasoning-gym/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/configs/inter_generalisation/algorithmic_qwen_3b.yaml. Fetched 2026-08-12.

[7] reasoning-gym on PyPI (release metadata: version, upload date). https://pypi.org/pypi/reasoning-gym/json. Fetched 2026-08-12.

[8] pyproject.toml, at commit 49b07130b3fcd12f2d064bba7c43869543a0e7e7 (dependencies, extras, license, Python floor, maintainer contact) - this commit is ahead of the v0.1.25 release (its own `version = "0.1.26.dev0"` shows this). https://raw.githubusercontent.com/open-thought/reasoning-gym/49b07130b3fcd12f2d064bba7c43869543a0e7e7/pyproject.toml. Fetched 2026-08-12.

[9] pyproject.toml at the v0.1.25 release tag (diffed against [8] to confirm the `[scoring]` extra is absent from the released version). https://raw.githubusercontent.com/open-thought/reasoning-gym/v0.1.25/pyproject.toml. Fetched 2026-08-12.

[10] reasoning-gym repository metadata (stars, last push date, license, archived status). https://api.github.com/repos/open-thought/reasoning-gym. Fetched 2026-08-12.

[11] reasoning-gym GitHub releases list (tag names and publish dates, v0.1.20 through v0.1.25). https://api.github.com/repos/open-thought/reasoning-gym/releases. Fetched 2026-08-12.

[12] tools/cli/rgc/main.py, at commit 49b07130b3fcd12f2d064bba7c43869543a0e7e7 (the `rgc` console-script entry point talks to a separate experiment-management server, not to dataset generation). https://raw.githubusercontent.com/open-thought/reasoning-gym/49b07130b3fcd12f2d064bba7c43869543a0e7e7/tools/cli/rgc/main.py. Fetched 2026-08-12.

[13] training/utils/load_fsdp_to_hf.py, at commit 49b07130b3fcd12f2d064bba7c43869543a0e7e7. https://raw.githubusercontent.com/open-thought/reasoning-gym/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/utils/load_fsdp_to_hf.py. Fetched 2026-08-12.

[14] scripts/hf_dataset/save_hf_dataset.py, at commit 49b07130b3fcd12f2d064bba7c43869543a0e7e7. https://raw.githubusercontent.com/open-thought/reasoning-gym/49b07130b3fcd12f2d064bba7c43869543a0e7e7/scripts/hf_dataset/save_hf_dataset.py. Fetched 2026-08-12.
