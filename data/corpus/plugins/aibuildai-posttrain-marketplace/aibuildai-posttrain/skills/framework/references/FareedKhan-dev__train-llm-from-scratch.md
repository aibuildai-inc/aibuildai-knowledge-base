# train-llm-from-scratch

A single-author teaching repository that hand-writes an entire LLM pipeline in pure PyTorch, from pretraining through SFT, a reward model, PPO, DPO/ORPO/KTO, and GRPO/RLVR - pick it to read and run every algorithm yourself, not to reuse a maintained trainer API.

The repository states it "implemented a transformer model from scratch using PyTorch, based on the paper Attention is All You Need" and that it now "goes all the way from raw text to an aligned, reasoning style model, with every algorithm hand written in plain PyTorch (no `trl`, no `peft`, no `transformers`)" [1]. It is built and maintained by a single named individual, Fareed Khan [1][2], with nine distinct external accounts credited on separate merged pull requests as of this reading (Abellegese, Chamath-Adithya, Jah-yee, OHUVERSE, TianyiQ, XiaoBinGan, eltociear, its-not-rocket-science, konglingfan); one further merged pull request, #28, is attributed to a bot account (`Copilot`), not a human contributor [3]. Its API is a set of standalone scripts, one per pipeline stage (`scripts/train_sft.py`, `scripts/train_reward.py`, `scripts/train_dpo.py`, `scripts/train_ppo.py`, `scripts/train_grpo.py`), each reading a per-stage JSON config and writing a checkpoint, plus a Streamlit control panel that launches the same scripts as background jobs [4][1]. It lives at https://github.com/FareedKhan-dev/train-llm-from-scratch [2].

**When to pick it**: pick it to learn or audit the mechanics of SFT, a Bradley-Terry reward model, PPO, DPO/ORPO/KTO, and GRPO by reading and running from-scratch implementations on a small custom Transformer with no `trl`/`peft`/`transformers` dependency [1]; do not pick it to post-train an existing pretrained checkpoint from the Hub, since every stage assumes the repo's own custom `Transformer` class and its own checkpoint format, not a `transformers`-compatible model - unlike the trl or verl decks, whose trainers take a Hub model id directly (cross-reference; not covered here). The repository's own README frames the tutorial value as authenticity over absolute scores: a roughly 400M model on 2 H100s "shows real before/after gains at each stage, but its absolute GSM8K score stays modest" [4].

**Methods it ships**: `SFT` (prompt-masked next-token cross-entropy with sequence packing), `Reward` (a Bradley-Terry pairwise loss on a scalar head over the SFT backbone; the README's own published run on 7,974 real preference pairs reached a held-out preference accuracy of 0.574, above the 0.5 chance line [1]), `PPO` (actor-critic with a shared-backbone value head, GAE, clipped policy and value losses, and a per-token KL-to-reference penalty), `DPO`/`ORPO`/`KTO` (three objectives in one module, selected by a `--loss_type` flag, sharing a common preference-pair data path; the README's own published DPO run on 7,974 real preference pairs reached an implicit-reward accuracy of 0.574 on held-out pairs, the fraction where the trained policy prefers the chosen response more than the frozen reference does [1]), and `GRPO`/RLVR (group-relative advantages with no critic, a token-level clipped surrogate, and a k3 KL penalty, with an arithmetic curriculum before full GSM8K) [4]. None of the six is marked experimental or stable; the whole suite is version 0.1.0 of a single project with no separate method-maturity taxonomy [5]. The source files are `src/post_training/sft.py`, `reward_model.py`/`reward_train.py`, `ppo.py`, `dpo.py`, and `grpo.py` [4]; import is via `PYTHONPATH=.` or an editable install (`pip install -e .`), not a published package namespace [1]. The live list of stages is the repository's own README table of contents, which recheck at https://github.com/FareedKhan-dev/train-llm-from-scratch#readme [1].

**Scale it handles**: single GPU up to single-node multi-GPU only, via `torchrun --standalone --nproc_per_node=N`, using plain `torch.distributed` DDP (no Accelerate, DeepSpeed, or FSDP) - `src/post_training/distributed.py` states its helpers are "pure torch.distributed -- no accelerate" and that when not launched under torchrun every helper "degrade[s] to no-ops, so the scripts have a single code path" [6]. No multi-node launch form (for example `--rdzv` or `--nnodes>1`) is documented anywhere in the README or `POST_TRAINING.md` [4][1], and the README's own GPU-sizing table only speaks in single-GPU VRAM terms (a single A100-40GB reaches "~6B to 8B" parameters at the largest) [1] - multi-node is out of scope for this codebase, not merely unbenchmarked.

**Install**: `git clone https://github.com/FareedKhan-dev/train-llm-from-scratch.git && cd train-llm-from-scratch && pip install -e .`, with optional extras `.[train]` (datasets, wandb), `.[ui]` (streamlit, pandas, altair), `.[docs]` (mkdocs, mkdocs-material, pymdown-extensions), or `.[all]` [1]. `pyproject.toml` at commit 98f808c pins `requires-python = ">=3.9"`, licence MIT, and core dependencies with no version numbers at all - `torch`, `numpy`, `h5py`, `tqdm`, `tiktoken`, `zstandard`, `requests` - so nothing collides with a version a reader already holds, but nothing is guaranteed compatible either [5]. There is no PyPI package and no GitHub Release or tag: `git tag`/`releases` are empty at the time of reading, so the install line always resolves to whatever commit `main` is at, and commit 98f808c (2026-06-24) is itself the tip of `main`, not a version behind it [7][2]. A second file, `requirements.txt`, pins CUDA 11.8 wheels for the legacy pretraining-only script and is explicitly left untouched by the newer `requirements-post.txt`, which instead targets CUDA 12.1 wheels for the full post-training suite - the two files disagree on CUDA version by design, and neither is used by `pip install -e .` itself [8]. No CUDA/GPU minimum is stated anywhere for the post-training stages beyond "you will need a GPU to train" and a worked sizing table for consumer and datacenter cards [1].

**Maintained by**: Fareed Khan, a single named maintainer [2][1]; about 8,986 GitHub stars, not a ranking signal [2]. The repository was pushed most recently on 2026-06-24 [2], and its issue tracker shows active maintenance through 2026-06-16, when the owner merged fixes for two reported bugs (out-of-memory on the legacy trainer's default config, and a package-import error) in the same week [9][10].

## Quick start

The smallest complete runs, quoted from the repository's own README and `POST_TRAINING.md`, each a real command against the repo's own scripts and data (no pseudo-code) [1][4]:

```bash
# one-time data prep (packs Alpaca + Dolly + GSM8K into fixed-length rows)
PYTHONPATH=. python scripts/prepare_sft_data.py --context_length 1024
# SFT: prompt-masked next-token loss on a pretrained checkpoint
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_sft.py
```

```bash
# GRPO / RLVR on GSM8K, starting from the SFT checkpoint
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_grpo.py --group_size 8
```

The whole alignment chain (SFT to Reward Model to DPO to PPO to GRPO to the evaluation table) runs in one shot with `bash scripts/run_posttraining.sh` [4]. After an editable install (`pip install -e .`), the `PYTHONPATH=.` prefix is unnecessary and every command above still works [1].

## Start it

- **One GPU**: `python scripts/train_sft.py` (or `train_reward.py` / `train_dpo.py` / `train_ppo.py` / `train_grpo.py`), each defaulting to its matching JSON under `configs/` [11].
- **Several GPUs, one node**: `torchrun --standalone --nproc_per_node=N scripts/train_X.py`; only rank 0 logs and checkpoints, and `src/post_training/distributed.py` explicitly does not use Accelerate, DeepSpeed, or FSDP [11][6]. There is no multi-node launcher or config template in the repository.
- **Effective batch**: `batch_size * grad_accum * num_gpus`; the how-to page's own multi-GPU note gives a worked example - at context length 1024 on 2 H100s with no NVLink, use `--batch_size 8 --grad_accum 12` to keep memory down while recovering the effective batch through accumulation [11].
- **Config surface**: each stage is a dataclass in `config/post_training_config.py`, resolved in four layers, lowest precedence first: dataclass defaults, then `configs/base.json`, then the stage's own `configs/<stage>.json`, then any `--field` CLI override [12]. `configs/base.json` sets `amp_dtype: "bf16"` for every stage - a silent assumption that the GPU is bf16-capable (an H100 or A100; not, for example, an older V100) [13]. A parallel `configs/smoke/` tree shrinks the model and step count for a CPU-friendly end-to-end check of the same code path [12].
- **PPO's own defaults**: `gamma: 1.0`, `gae_lambda: 0.95`, `clip: 0.2`, `vf_clip: 0.2`, `kl_coef: 0.05`, read directly from `configs/ppo.json` at this commit [14]. **GRPO's own defaults**: `group_size: 8`, `clip: 0.2`, `kl_coef: 0.04`, `curriculum_iters: 100` [14].
- **Out-of-memory first aid**: the README's GPU-sizing guidance points at the legacy pretraining script's opt-in `--amp`, `--grad-checkpointing`, and `--grad-accum` flags [1]; for the post-training suite specifically, `docs/howto/train.md` recommends lowering `--batch_size`/raising `--grad_accum` and setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, because the educational attention implementation materializes a full `(B, n_head, T, T)` tensor per block, so memory scales with sequence length [11]. A closed issue confirms the *legacy* trainer's committed defaults are a ~3B-parameter model that OOMs on a 40GB A100, fixed by opt-in memory flags added in a later PR - the post-training stages default to a much smaller ~400M model via `configs/base.json` and are not affected by that specific defect [9].

## Watch it

This section is mechanics only - what a reward, KL, or clip-fraction value should look like for a healthy run is method-level judgment, not covered here.

- **Enable it**: `MetricsLogger` always writes one JSON object per logged step to a JSONL file under the run's log directory - "so runs are reproducible and plottable without any external service" - regardless of any tracker [15]. Setting `use_wandb: true` on the config additionally mirrors every logged step to Weights & Biases; if the `wandb` package is not importable, the logger prints a message and silently falls back to JSONL-only, never crashing the run [15].
- **SFT metrics** (from `scripts/train_sft.py`'s own logging calls at this commit): `train_loss`, `lr` every 20 steps; `dev_loss` on the eval cadence [16].
- **Reward-model metrics**: `train_loss`, `train_acc`, `lr`; on eval, `test_acc` and `test_margin` [16].
- **DPO/ORPO/KTO metrics** (shared by all three via `--loss_type`): `train_loss`, `train_acc`, `r_chosen`, `r_rejected`, `lr`; on eval, `test_acc` and `test_margin` [16].
- **PPO metrics**: `reward`, `kl_ref`, `policy_loss`, `value_loss`, `clipfrac`, `resp_len` every 5 iterations; `gsm8k_acc` on the eval cadence [16].
- **GRPO metrics**: `reward`, `informative_groups` (the fraction of sampled groups whose reward has non-zero spread), `loss`, `kl`, `resp_len` every 5 iterations; `gsm8k_acc` on the eval cadence [16].
- **Sample-level logging**: none of the training scripts read for this card write raw generated text to the metrics JSONL; `scripts/chat.py` is the tool for inspecting actual generations from any checkpoint, run manually [1].
- **Evaluation during training**: PPO and GRPO both call `gsm8k_accuracy` against a held-out GSM8K test split on the `eval_every` cadence and log `gsm8k_acc`; SFT, the reward model, and DPO/ORPO/KTO each evaluate on their own held-out split on `eval_steps` [16]. A separate script, `scripts/eval_post_training.py`, scores any saved checkpoint after the fact and appends a row to a cross-stage results table [17].
- **Stopping / health limits**: no early-stopping flag, patience value, or a published KL or reward threshold appears in any config dataclass, JSON file, or docs page read for this card (`config/post_training_config.py`, `configs/*.json`, `docs/06_ppo.md`, `docs/07_grpo.md`, `docs/howto/train.md`) - training always runs for the configured number of epochs or iterations, and the reward-hacking mitigation is architectural (a bounded, correctness-dominant verifier reward plus the KL-to-reference penalty), not a stopping rule [1][4].

## Save it

- Every stage writes checkpoints through one shared function, `save_stage_ckpt`, to a single path per stage (for example `/ephemeral/ckpts/sft.pt`), not a directory of files [18]. The payload is a plain dict with keys `model_state_dict`, `optimizer_state_dict`, `stage`, `cfg` (the fully resolved stage config), `step`, `metrics`, `pytorch_version`, and `cuda_version`; DDP wrappers are unwrapped first "so checkpoints load cleanly on a single GPU" [18].
- Retention is caller-controlled: each script's `save_every` field (for example 500 for SFT, 100 for PPO/GRPO) sets how often `save_stage_ckpt` is called; there is no flag anywhere in the post-training suite that drops the optimizer state to shrink a checkpoint, so every saved checkpoint written by these five scripts is optimizer-state-complete by construction [14][18].
- **No resume support for the post-training stages.** `--resume [latest|<path>]`, restoring model, optimizer, and LR-scheduler state, exists only on the legacy pretraining script `scripts/train_transformer.py`, added in a merged pull request that touched only `README.md`, `config/config.py`, `scripts/train_transformer.py`, `src/models/transformer.py`, and a new test file [19]. A closed feature request explicitly asked for checkpoint resume and the maintainer's fix landed there, in the legacy path, not in `src/post_training/` or any `scripts/train_{sft,reward,dpo,ppo,grpo}.py` [20]. If an SFT/PPO/GRPO/DPO/reward run is interrupted, the newest `save_every` checkpoint on disk holds full model and optimizer state, but no CLI flag reloads it - a reader who needs to continue would have to write that loading code by hand from `save_stage_ckpt`'s own key names [18].
- Reload for inference (any stage, no training state) is `scripts/chat.py --ckpt <path>`, which reads the model's architecture dimensions back out of the checkpoint itself rather than requiring a separate config file, and applies the chat template automatically for anything past the base checkpoint (`--raw` forces plain continuation) [1].
- There is no LoRA/PEFT path in this repository - it does not use `peft` and every save is a full model checkpoint, never an adapter [1].
- Loader handoff: a checkpoint from this repository is only loadable by this repository's own `Transformer` class and `load_backbone_from_ckpt`/`chat.py` - it is not a `transformers`-format directory and cannot be loaded with `AutoModel.from_pretrained` or an evaluator built for Hub checkpoints [18][1].

## Find it in the docs

The docs are the live source; this section teaches the lookup, it does not mirror the content.

- Two doc surfaces exist and cover different depths: the single long-form `README.md`, rendered on the repo's own GitHub page [1], and a built MkDocs Material site at `https://fareedkhan-dev.github.io/train-llm-from-scratch/<slug>/` (verified 2026-08-10: `.../howto/train/` and `.../howto/commands/` both return HTTP 200) [21][11]. The site's nav maps `<slug>` to page names directly - `howto/train`, `howto/configs`, `howto/ui`, `08_evaluation`, `09_inference`, and seven Theory & Pipeline pages `01_data_pipeline` through `07_grpo`, one per pipeline stage (data, pretraining, SFT, reward model, DPO/ORPO/KTO, PPO, GRPO) [22].
- `docs/howto/commands.md` is a MkDocs snippet include (`--8<-- "POST_TRAINING.md"`) rather than page content of its own - fetching the raw file from GitHub returns only that one include directive, while the rendered site page at `howto/commands/` contains the full inlined text of `POST_TRAINING.md`, confirmed by matching section headers present in the rendered HTML but not the raw markdown source [23].
- Question-to-slug map: "how do I configure a stage" -> `howto/configs`; "how do I launch training / multi-GPU" -> `howto/train`; "how do I use the Streamlit app" -> `howto/ui`; "what do I run to compare stages" -> `08_evaluation`; "how do I talk to a checkpoint" -> `09_inference`; the underlying theory and math for each stage lives on its own numbered page (`03_sft`, `04_reward_model`, `05_dpo`, `06_ppo`, `07_grpo`) [22].
- Runnable references beyond the docs: `scripts/run_posttraining.sh` runs the entire SFT-to-GRPO chain end to end [4]; `tests/test_post_training_smoke.py` exercises the core log-prob, masking, and parsing math directly and is described as running "in seconds" [4]; `configs/smoke/*.json` gives a tiny CPU-sized variant of every stage config for a fast smoke test [12].
- Community layer: this project curates no separate tutorials page of its own; the README is written as the tutorial, addressed explicitly to students, developers, and researchers in three different reading paths [1]. No official MCP endpoint for these docs was found.
- Traps found in the issue tracker (closed issues, maintainer replies only): the legacy pretraining script's committed defaults produce a roughly 3B-parameter model that runs out of memory on a 40GB A100, fixed by opt-in memory flags in a merged PR (issue #5, reply by the repo owner, 2026-06-16) [9]; running `python scripts/train_transformer.py` directly from `scripts/` can fail with `No module named config` because that puts `scripts/` rather than the repo root on `sys.path` - fixed by making the project pip-installable (`pip install -e .`) so `config`, `src`, etc. resolve from anywhere (issue #7, reply by the repo owner, 2026-06-16) [10]; checkpoint resume was requested and implemented only for the legacy pretraining script, not the post-training suite (issue #16, reply by the repo owner, 2026-06-16) [20].
- Honest boundary: this is not a `transformers`-compatible trainer and ships no adapter/LoRA path, no multi-node launcher, and no early-stopping or reward/KL threshold; it trains and evaluates its own from-scratch `Transformer` class end to end on a fixed set of public datasets (The Pile, Alpaca, Dolly, Anthropic HH-RLHF, UltraFeedback, GSM8K) [4][1].

## Sources

All GitHub-hosted files are read at commit `98f808c4ea9c4e83e16050358e80288642d0cd80` unless a live/rendered page is named; the MkDocs site pages are unpinned and were fetched 2026-08-10.

[1] train-llm-from-scratch README. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/README.md. Fetched 2026-08-10 (also rendered at https://github.com/FareedKhan-dev/train-llm-from-scratch#readme).

[2] train-llm-from-scratch GitHub repository metadata. https://api.github.com/repos/FareedKhan-dev/train-llm-from-scratch. Fetched 2026-08-10.

[3] Pull request listing for the repository (state=all, up to 100 results), the source for the external-contributor accounts and for PR #28's author being the `Copilot` bot account. https://api.github.com/repos/FareedKhan-dev/train-llm-from-scratch/pulls?state=all&per_page=100. Fetched 2026-08-10.

[4] POST_TRAINING.md. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/POST_TRAINING.md. Fetched 2026-08-10.

[5] pyproject.toml. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/pyproject.toml. Fetched 2026-08-10.

[6] src/post_training/distributed.py. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/src/post_training/distributed.py. Fetched 2026-08-10.

[7] GitHub releases and tags for the repository (both empty). https://api.github.com/repos/FareedKhan-dev/train-llm-from-scratch/releases and https://api.github.com/repos/FareedKhan-dev/train-llm-from-scratch/tags. Fetched 2026-08-10.

[8] requirements.txt and requirements-post.txt. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/requirements.txt and .../requirements-post.txt. Fetched 2026-08-10.

[9] Issue #5, "All default config, it's 3B parameters. Shows OOM with A100 40GB GPU." https://api.github.com/repos/FareedKhan-dev/train-llm-from-scratch/issues/5 and its comments. Fetched 2026-08-10.

[10] Issue #7, "No module named config." https://api.github.com/repos/FareedKhan-dev/train-llm-from-scratch/issues/7 and its comments. Fetched 2026-08-10.

[11] docs/howto/train.md. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/docs/howto/train.md. Fetched 2026-08-10.

[12] docs/howto/configs.md. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/docs/howto/configs.md. Fetched 2026-08-10.

[13] configs/base.json. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/configs/base.json. Fetched 2026-08-10.

[14] configs/ppo.json and configs/grpo.json. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/configs/ppo.json and .../configs/grpo.json. Fetched 2026-08-10.

[15] src/post_training/logging_utils.py. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/src/post_training/logging_utils.py. Fetched 2026-08-10.

[16] scripts/train_sft.py, scripts/train_reward.py, scripts/train_dpo.py, scripts/train_ppo.py, scripts/train_grpo.py (each script's own `logger.log` calls). https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/scripts/train_{sft,reward,dpo,ppo,grpo}.py. Fetched 2026-08-10.

[17] docs/08_evaluation.md. https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/docs/08_evaluation.md. Fetched 2026-08-10.

[18] src/post_training/utils.py (`save_stage_ckpt`, `load_backbone_from_ckpt`). https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/src/post_training/utils.py. Fetched 2026-08-10.

[19] Pull request #28 file list (checkpoint-resume feature, legacy trainer), showing the changed files `README.md`, `config/config.py`, `scripts/train_transformer.py`, `src/models/transformer.py`, and a new test file. https://api.github.com/repos/FareedKhan-dev/train-llm-from-scratch/pulls/28/files. Fetched 2026-08-10.

[20] Issue #16, "Feature Request: Add Checkpoint Resume Support for Interrupted Training." https://api.github.com/repos/FareedKhan-dev/train-llm-from-scratch/issues/16 and its comments. Fetched 2026-08-10.

[21] MkDocs site pages (rendered, unpinned/live, no version parameter). https://fareedkhan-dev.github.io/train-llm-from-scratch/howto/train/ and https://fareedkhan-dev.github.io/train-llm-from-scratch/howto/commands/. Fetched 2026-08-10.

[22] mkdocs.yml (nav slug map). https://raw.githubusercontent.com/FareedKhan-dev/train-llm-from-scratch/98f808c4ea9c4e83e16050358e80288642d0cd80/mkdocs.yml. Fetched 2026-08-10.

[23] Comparison of docs/howto/commands.md raw source (an include directive only) against the rendered page at https://fareedkhan-dev.github.io/train-llm-from-scratch/howto/commands/ (full inlined POST_TRAINING.md content). Fetched 2026-08-10.
