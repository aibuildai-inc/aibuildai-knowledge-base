# TTRL

A research-paper code release: one self-rewarding RL recipe (majority-vote pseudo-labels) bolted onto a vendored, frozen copy of verl - pick it to reproduce or extend TTRL itself, not as a general post-training framework.

**TTRL** (Test-Time Reinforcement Learning) is the code release for a paper that states its own scope as: "We investigate Reinforcement Learning (RL) on data without explicit labels for reasoning tasks in Large Language Models (LLMs)" [1]. It is hosted under the PRIME-RL GitHub organization [2], with the paper's contact authors listed as Kaiyan Zhang and Ning Ding of Tsinghua [1]. Its API shape is not its own: the repository root holds only a `README.md`, `LICENSE`, and a `figs/` folder alongside a full vendored subdirectory named `verl/` [2] - a complete copy of volcengine/verl at internal version `0.4.1.dev` at the shortlist's pinned commit [3] - so a run is started the same way a verl run is started (`python -m verl.trainer.main_ppo`, Hydra config overrides on the command line), with TTRL's own contribution reduced to one Hydra flag, `+ttrl.enable=True`, that swaps the ground-truth label for a majority-vote label inside verl's existing PPO trainer loop [4][5]. The paper's headline result is a 211% pass@1 gain for Qwen-2.5-Math-7B on AIME 2024 using only unlabeled test data, trained with this exact mechanism [1][14]. It lives at https://github.com/PRIME-RL/TTRL [2].

**When to pick it**: pick it only to reproduce the TTRL paper's headline result - a 211% pass@1 gain for Qwen-2.5-Math-7B on AIME 2024 from unlabeled test data alone [1][14] - or to build directly on its majority-vote self-reward mechanism on math reasoning tasks; it is not a competing choice against trl or verl on method breadth, because it adds exactly one technique and otherwise IS a verl checkout - frozen at verl 0.4.1.dev, from a news entry dated 2025-08-17 that says the authors "bump into" that verl release [1][3]. The repository's own last tagged GitHub release, named "verl" (v2.0.0, published 2025-07-11, commit `d187d1e7`), vendors an even older verl `0.2.0.dev` [16] - one version behind the `0.4.1.dev` at the shortlist's pinned commit `5806e11`, which sits untagged on `main` ahead of that release [3][16]. If you want current verl (GRPO, DAPO, and the rest of its recipe library kept up to date, multi-node scale, active maintenance) choose verl directly and port the ~200-line `ttrl_utils.py` module yourself (cross-reference; verl not covered here) [6].

**Methods it ships**: TTRL's own method is the majority-vote reward: at each step it samples `ttrl.n_votes_per_prompt` completions per prompt, extracts and clusters boxed answers, takes the most common answer as a pseudo ground-truth label, computes rewards against that label instead of the real one, and separately computes `label_accuracy`, `reward_accuracy`, `majority_voting_reward`, `ground_truth_reward`, `pass@N`, and `majority_ratio` diagnostics against the withheld real label [4][5]. This mechanism is orthogonal to the advantage estimator - the shipped example scripts run it with GRPO (`advantage=grpo`) [7], but the flag lives inside generic PPO-loop code and is not restricted to GRPO in the source [4]. Everything else the shortlist's `methods_seen` names - DAPO and SPPO (`verl/recipe/dapo`, `verl/recipe/sppo`), SFT (`verl/verl/trainer/fsdp_sft_trainer.py`), a second PPO config, and a reward function under `verl/recipe/char_count/` - is the stock verl `recipe/` and `trainer/` tree copied in wholesale, not something TTRL modified or adds documentation for [3][8]; treat those as "whatever this pinned verl snapshot ships", not TTRL features.

**Scale it handles**: whatever the vendored verl 0.4.1.dev launch path handles (Ray-orchestrated FSDP or Megatron workers, single process up to multi-node) - but the only scale TTRL's own docs report running is one node: "All experiments were conducted on 8 x NVIDIA A100 80GB GPUs" [1], and the shipped `examples/ttrl/Qwen2.5/aime.sh` reproduction script hardcodes `trainer.n_gpus_per_node=8`, `trainer.nnodes=1` [7]. No TTRL-specific multi-node example or benchmark is published; for multi-node mechanism read the vendored verl docs directly (cross-reference; not covered here) [6].

**Install**: `git clone https://github.com/PRIME-RL/TTRL.git && cd TTRL/verl`, Python 3.10 (the quickstart creates a `conda` env pinned to `python==3.10`), then `bash scripts/install_ttrl_deps.sh && pip install -e .` [1]. TTRL is not published to PyPI, so this line installs whatever commit `main` is at when cloned, not a pinned release; the repository's own last tagged release ("verl" v2.0.0, 2025-07-11, commit `d187d1e7`) is two verl minor versions behind the shortlist's pinned commit `5806e11` (vendored verl `0.2.0.dev` versus `0.4.1.dev`) [3][16], so the dependency pins below are read at the pinned commit `5806e11` itself, not at that release. That script pins `vllm==0.8.5.post1`, `torch==2.6.0`, `torchvision==0.21.0`, `torchaudio==2.6.0`, `tensordict==0.6.2`, and `transformers[hf_xet]>=4.51.0`, plus a matching prebuilt `flash-attn==2.7.4.post1` wheel and a `flashinfer==0.2.2.post1` wheel for cu124/torch2.6 [9]. The vendored `verl/setup.py` separately declares `tensordict<=0.6.2` and a `vllm` extra capped at `vllm<=0.8.5`, consistent with the install script's pin [8]. Licensing is split: the repository root is MIT, copyright PRIME-RL 2025 [10], but the vendored `verl/` subdirectory keeps its own Apache-2.0 `LICENSE` file inherited from volcengine/verl [11] - code under `verl/` is Apache-2.0, not MIT. No CUDA/GPU floor is stated in the docs read for this card beyond the reproduction hardware note above; the flash-attn and flashinfer wheels pinned in the install script are built for `cu12`/`cu124`, which is the closest thing to a stated hardware floor [9].

**Maintained by**: the PRIME-RL GitHub organization; last push to the repository was 2026-04-15 [3]. The README's own News section is the dated activity log: a 2025-04-23 announcement of the method, a 2025-04-24 release of the code and experimental logs, a 2025-05-23 rewrite of the code onto verl, an 2025-08-17 note pinning verl v0.4.1, a 2025-09-18 NeurIPS 2025 acceptance announcement, and a 2026-03-10 entry pointing to a separate `urlvr-dev` branch for newer, unsupervised-RLVR follow-on work that is not on `main` [1].

## Quick start

The repository's own quickstart, quoted in form from the README [1]:

```bash
git clone https://github.com/PRIME-RL/TTRL.git

cd TTRL/verl

conda create -n ttrl python==3.10
conda activate ttrl
bash scripts/install_ttrl_deps.sh
pip install -e .
```

```bash
bash examples/ttrl/Qwen2.5/aime.sh
```

The second command reproduces the paper's AIME 2024 result and is a real, complete run, not pseudo-code: it launches `python -m verl.trainer.main_ppo --config-name='ppo_trainer_ttrl.yaml'` with a full set of data, model, actor, critic, and `ttrl.*` overrides baked into the script [7]. Before running it, data must exist as parquet under `verl/data/<TASK>/{train,test}.parquet`; the repo ships `verl/data/preprocess.py` as the JSON-to-parquet converter referenced by the README [1][12].

## Start it

- One process, one node is the only form TTRL documents: `aime.sh` (and its `amc.sh`, `math.sh` siblings under `examples/ttrl/Qwen2.5/`) set `trainer.n_gpus_per_node=8`, `trainer.nnodes=1` directly in the script [7].
- The TTRL-specific config surface is the `ttrl:` block added on top of verl's base `ppo_trainer.yaml` in `verl/verl/trainer/config/ppo_trainer_ttrl.yaml`: `ttrl.enable` (default `false`), `ttrl.n_samples_per_prompt` (rollouts actually trained on, defaults to `actor_rollout_ref.rollout.n`), and `ttrl.n_votes_per_prompt` (rollouts sampled to build the majority-vote label, same default) [4]. `aime.sh` sets these explicitly: `n_votes_per_prompt=64`, `n_samples_per_prompt=32` - i.e. it samples twice as many completions as it trains on, using the extra half purely to build a more reliable vote [7].
- Effective batch arithmetic follows verl's own fields, used as-is in `aime.sh`: `data.train_batch_size=8` prompts per step, each expanded to `n_votes_per_prompt=64` completions for voting, then downsampled to `n_samples_per_prompt=32` completions per prompt for the actual PPO/GRPO update via `ttrl_utils.select_top_k_per_prompt` [7][5].
- Generation runs through vLLM (`actor_rollout_ref.rollout.name=vllm`), colocated on the same 8 GPUs as training in the shipped script (`gpu_memory_utilization=0.8`, `tensor_model_parallel_size=1`) [7]; that is verl's own generation-layout surface, not a TTRL addition (cross-reference verl; not covered here) [6].
- TTRL changes no framework default outside its own `ttrl:` block - `ppo_trainer_ttrl.yaml` is verl's base `ppo_trainer` config with `ttrl.enable: false` layered on via Hydra `defaults:` [4], so all of verl's own precision, checkpointing, and batching defaults apply unchanged.
- Out-of-memory first aid is not documented on TTRL's own pages; the reproduction script's working values (`ppo_micro_batch_size_per_gpu=2`, `max_response_length` capped at `1024*K` tokens) are the only tuned reference point published, and general OOM knobs are verl's, not TTRL's (cross-reference verl; not covered here) [7][6].

## Watch it

This section is TTRL's own diagnostic surface; verl's general metric names and logging mechanics are not restated here (cross-reference verl; not covered here) [6].

- **Enable it**: logging goes through verl's own `trainer.logger` field; the shipped script sets `trainer.logger=['console','wandb']`, so nothing is logged anywhere durable unless `wandb` (or another verl-supported backend) is explicitly listed [7]. A public example run is at the Weights & Biases project linked from the README [1].
- **TTRL-specific metrics**, computed once per step only when `ttrl.enable=True` [4], all logged under a `train/` prefix by the training loop itself, which calls `compute_ttrl_metrics` and writes each key back as `f"train/{key}"` [5][13]:
  - `train/label_accuracy` - whether the majority-vote label matches the real ground truth for the step's prompts (1.0/0.0 per prompt, averaged).
  - `train/reward_accuracy` - fraction of individual rollouts where the majority-vote reward equals the ground-truth reward.
  - `train/majority_voting_reward` and `train/ground_truth_reward` - mean reward under the pseudo-label versus the real label, the direct measure of how much signal is lost to voting.
  - `train/pass@<n_samples_per_prompt>` - whether any rollout in the group actually solved the prompt under the real label.
  - `train/majority_ratio` - mean fraction of votes the winning answer received per prompt (vote-agreement strength).
- What healthy shapes for these look like (e.g. how much `majority_voting_reward` can trail `ground_truth_reward` before training stalls) is method-level interpretation that belongs on a TTRL methodology card, not this framework card - none is asserted here.
- Sample-level logging of generations (`trainer.rollout_data_dir`) and evaluation-during-training (`val_reward_fn`) are verl's own generic fields, read directly from `ray_trainer.py` [13], but the shipped `aime.sh` script sets neither - it sets only `trainer.test_freq=2` for evaluation cadence [7]. No published stopping rule was found on the pages read for this card; general logging and evaluation mechanics are verl's own and not otherwise covered here (cross-reference verl; not covered here) [6].

## Save it

- Checkpointing is verl's own mechanism, read directly from `verl/verl/trainer/ppo/ray_trainer.py` at commit `5806e11`: each save writes to `<trainer.default_local_dir>/global_step_<N>/actor/` (and `/critic/` when a critic is in use), plus a `data.pt` dataloader-state file and a root-level `latest_checkpointed_iteration.txt` [13].
- Retention is controlled by `trainer.max_actor_ckpt_to_keep` / `trainer.max_critic_ckpt_to_keep` (unlimited if unset) [13]. **The shipped `aime.sh` reproduction script sets `trainer.save_freq=2000000` (effectively never) and both retention caps to `0`** [7] - run it as published and no checkpoint is written at all; this is a real trap in the published reproduction, not a hypothetical one.
- Resume is controlled by `trainer.resume_mode`: `"disable"` starts fresh, `"auto"` finds the latest `global_step_*` folder under `default_local_dir` automatically, `"resume_path"` requires `trainer.resume_from_path` to name a specific `global_step_*` folder [13].
- TTRL adds nothing to what gets saved: no adapter path, no TTRL-specific state (the majority-vote label itself is recomputed fresh from the current policy each step, not persisted) [5]. Whether a saved verl actor checkpoint is directly loadable by an evaluator is verl's own loader contract, not documented on this card (cross-reference verl; not covered here) [6].

## Find it in the docs

TTRL publishes no versioned documentation site of its own - the README is effectively the whole of TTRL's own docs, at https://github.com/PRIME-RL/TTRL/blob/main/README.md, unpinned to any tag [1]. For everything below the `ttrl.enable=True` flag - config fields, launch mechanics, checkpoint contract, generation backends - the README explicitly redirects: "For further details regarding the code, please refer to the verl documentation" at https://verl.readthedocs.io/en/latest/index.html [1], which documents the current verl release, not the frozen 0.4.1.dev snapshot vendored here - check any value you read there against the vendored source before trusting it for this repo.
- Runnable references beyond the README: `examples/ttrl/` holds three model families (`Qwen2.5`, `Qwen2.5-Math`, `LLaMA3.1-Instruct`), each with `aime.sh` / `amc.sh` / `math.sh` scripts [1][7]; `data/preprocess.py` is the only documented data-prep entry point [12]; a public Weights & Biases log of the AIME run is linked from the README badge row [1].
- No community tutorial page, MCP endpoint, or curated blog list is published by TTRL itself; the paper's own community footprint is the arXiv listing (https://arxiv.org/abs/2504.16084 [14]) and a linked Hugging Face Papers page and X/Twitter announcement, neither of which is documentation.
- No maintainer reply to a GitHub issue was read for this card, so no trap-from-issue entry is included; the one documented trap - `aime.sh` disabling checkpoint saves - was found in the shipped script itself, not in an issue thread, and is stated above where it bites.
- Honest boundary: TTRL is scoped to math reasoning tasks with boxed, extractable answers - its majority-vote grader (`verl/verl/utils/reward_score/ttrl_math/`) is a boxed-answer extractor and symbolic-equality checker with no support for open-ended or non-math tasks [15]; it documents no method beyond this one self-rewarding mechanism, and it documents no multi-node run.

## Sources

Every reading below is from the pinned commit `5806e119789132fcb43552beaf2532d7d0447213` (the shortlist row's commit) unless a page is explicitly marked otherwise. TTRL is not published to PyPI, so `git clone` plus the quickstart's `pip install -e .` delivers whatever commit `main` is at, not a pinned release; the repository does carry two tagged GitHub releases - "OpenRLHF" (v1.0.0, published 2025-05-23, commit `ad380333`) and "verl" (v2.0.0, published 2025-07-11, commit `d187d1e7`) - but the pinned commit `5806e11` is untagged and sits on `main` after both, vendoring verl `0.4.1.dev` versus the newer tag's `0.2.0.dev` - two verl minor versions ahead [16]. TTRL cites verl only as a cross-reference for material this card does not cover (general launch, logging, and checkpoint mechanics); verl's own documentation is not otherwise mirrored here. Method names (PPO, GRPO, DAPO, SPPO, SFT) are deliberately cited to nothing here - they are inherited verl surface, not TTRL's contribution.

[1] TTRL README, commit `5806e11`. https://github.com/PRIME-RL/TTRL/blob/5806e119789132fcb43552beaf2532d7d0447213/README.md. Fetched 2026-08-12.

[2] TTRL GitHub repository (root). https://github.com/PRIME-RL/TTRL. Fetched 2026-08-12.

[3] TTRL repository metadata (license, push date, default branch) via GitHub API, and vendored verl version file `verl/verl/version/version` at commit `5806e11`. https://api.github.com/repos/PRIME-RL/TTRL ; https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/verl/version/version. Fetched 2026-08-12.

[4] Vendored verl Hydra config `verl/verl/trainer/config/ppo_trainer_ttrl.yaml` at commit `5806e11` (the `ttrl:` block and its defaults). https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/verl/trainer/config/ppo_trainer_ttrl.yaml. Fetched 2026-08-12.

[5] Vendored verl `verl/verl/trainer/ppo/ttrl_utils.py` at commit `5806e11` (majority-vote label construction and TTRL diagnostic metrics). https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/verl/trainer/ppo/ttrl_utils.py. Fetched 2026-08-12.

[6] verl project (cross-reference only; general launch, logging, and checkpoint-loader mechanics for the vendored codebase are not restated on this card). https://github.com/volcengine/verl. Not separately fetched for this card.

[7] TTRL example reproduction script `verl/examples/ttrl/Qwen2.5/aime.sh` at commit `5806e11`. https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/examples/ttrl/Qwen2.5/aime.sh. Fetched 2026-08-12.

[8] Vendored verl `verl/setup.py` and `verl/recipe/` directory listing at commit `5806e11` (dependency extras; recipe tree: char_count, dapo, entropy, genrm_remote, minicpmo, prime, r1, retool, spin, sppo). https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/setup.py ; https://api.github.com/repos/PRIME-RL/TTRL/git/trees/5806e119789132fcb43552beaf2532d7d0447213:verl/recipe. Fetched 2026-08-12.

[9] TTRL install script `verl/scripts/install_ttrl_deps.sh` at commit `5806e11` (pinned torch/vllm/tensordict/transformers versions and flash-attn/flashinfer wheels). https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/scripts/install_ttrl_deps.sh. Fetched 2026-08-12.

[10] TTRL root `LICENSE` file at commit `5806e11` (MIT, copyright PRIME-RL 2025). https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/LICENSE. Fetched 2026-08-12.

[11] Vendored verl `verl/LICENSE` file at commit `5806e11` (Apache License 2.0, inherited from volcengine/verl). https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/LICENSE. Fetched 2026-08-12.

[12] TTRL data conversion script `verl/data/preprocess.py` at commit `5806e11`. https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/data/preprocess.py. Fetched 2026-08-12.

[13] Vendored verl `verl/verl/trainer/ppo/ray_trainer.py` at commit `5806e11` (`_save_checkpoint`, `_load_checkpoint`, checkpoint directory layout, `resume_mode` handling). https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/verl/trainer/ppo/ray_trainer.py. Fetched 2026-08-12.

[14] TTRL: Test-Time Reinforcement Learning, arXiv:2504.16084 (abstract page; the 211% pass@1 gain on AIME 2024 for Qwen-2.5-Math-7B and the maj@n-upper-limit comparison). https://arxiv.org/abs/2504.16084. Fetched 2026-08-12.

[15] Vendored verl `verl/verl/utils/reward_score/ttrl_math/__init__.py` at commit `5806e11` (boxed-answer extraction and symbolic grading used both for the majority vote and for reward scoring). https://raw.githubusercontent.com/PRIME-RL/TTRL/5806e119789132fcb43552beaf2532d7d0447213/verl/verl/utils/reward_score/ttrl_math/__init__.py. Fetched 2026-08-12.

[16] TTRL GitHub tags and releases API, and the vendored verl version file at the "verl" release tag's commit `d187d1e7db9f4226713bcbadec4539a5e303eae6` (confirms the release's vendored verl is `0.2.0.dev`, older than the pinned commit's `0.4.1.dev`). https://api.github.com/repos/PRIME-RL/TTRL/tags ; https://api.github.com/repos/PRIME-RL/TTRL/releases ; https://raw.githubusercontent.com/PRIME-RL/TTRL/d187d1e7db9f4226713bcbadec4539a5e303eae6/verl/verl/version/version. Fetched 2026-08-12.
