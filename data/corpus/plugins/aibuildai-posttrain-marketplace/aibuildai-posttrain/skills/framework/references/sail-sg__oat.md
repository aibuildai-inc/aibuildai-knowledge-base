# oat

A research-oriented online-alignment framework built around three cooperating process roles (Actor, Learner, Oracle) and one shared CLI argument class - pick it for RLVR/preference-learning research runs where you want the Actor-Learner-Oracle split and exploration algorithms out of the box, not for a broad, stable, ecosystem-integrated trainer library.

**oat** (also released to PyPI as `oat-llm`) is "a simple yet efficient framework for running **online** LLM alignment algorithms" [1]. It is developed by sail-sg (Sea AI Lab) [2][3], and its API is a single `OATArgs` dataclass parsed by `tyro` into a CLI, consumed by a small set of runnable Python entrypoints under `oat/experiment/` (e.g. `python -m oat.experiment.run_math_rl`) that each wire together an Actor class, a Learner class, and an Oracle into a distributed Launchpad program [4][5]. It lives at https://github.com/sail-sg/oat [3].

**When to pick it**: research runs that need oat's distributed *Actor-Learner-Oracle* architecture - vLLM-served Actors for generation, DeepSpeed ZeRO Learners for training, and a Mosec-served or in-process Oracle for preference/reward/verification feedback [1] - especially online exploration algorithms (SEA, APL, XPO) and RLVR-for-math recipes that ship as ready scripts [1][6]. Its own docs describe its taxonomy of `algo`/`critic_type`/`exp_method` values as covering DPO-family offline methods, PPO/GRPO/Dr.GRPO online RL, and exploration variants (below) [7], and there is no separate docs website: the README, a handful of `docs/*.md` guides, and the example scripts are the only prose documentation [1][8]. Weigh trl instead when the work is squarely inside the Hugging Face Hub/`transformers` ecosystem and stability/breadth of trainers matters more than the Actor-Learner-Oracle architecture (cross-reference; not covered here).

**Methods it ships**: method selection in oat is split across three separate dataclass fields on `OATArgs`, not one flat list, read from `oat/args.py` at the screening commit [7]:
- `algo` (Literal): `DPO`, `IPO`, `LR_DPO`, `SLiC`, `SimPO`, `BNF`, `SFT`, `PPO` - default `"DPO"` [7].
- `critic_type` (Literal, only meaningful when `algo="PPO"`): `ppo`, `grpo`, `drgrpo` - default `"drgrpo"`; this is where GRPO and Dr.GRPO live, not in `algo` [7]. The math-RL quickstart script sets `--critic_type drgrpo` with the comment "We use Dr. GRPO by default" [6].
- `exp_method` (Literal, online-exploration add-on): `no`, `EnnBAITS`, `EnnEETS`, `EnnUncertainty`, `EnnPassive` - default `"no"` [7]; the SEA/APL/XPO exploration algorithms named in the README are reached through this field plus dedicated entrypoints (`oat/experiment/run_apl.py`, `oat/experiment/run_xpo.py`), not through `run_math_rl.py` [1][9].
- Some algorithms extend `OATArgs` with their own dataclass rather than adding fields to it: `PPOArgs` (in `oat/algorithms/ppo.py`) adds `num_ppo_epochs`, `kl_penalty_coef`, `non_stop_penalty` [10]; `OfflineArgs` (in `oat/experiment/run_offline.py`) adds `chat_data`, `msg_key` for the SFT path [11]. A reader who only reads `oat/args.py` will not find these fields.
- No taxonomy page comparable to trl's docs index exists to double-check this grouping against; it was cross-checked directly against the `oat/args.py` Literal definitions and the example scripts' actual flags [6][7][9][10][11].

**Scale it handles**: single GPU up to the multi-GPU, single-node case that every shipped example targets (`--gpus 8` in the RL and SFT quickstarts) [6][12], launched either as `python -m oat.experiment.<script>` (RL/preference scripts) or `deepspeed --module oat.experiment.run_offline` (the SFT script) [6][12]. Sharding is DeepSpeed ZeRO via `zero_stage` (default 2) [7]; a `launch_type` field (default `"local_mp"`) and a `DistributedLauncher` helper in `oat/utils/launcher.py` that sets `MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE`/`RANK` env vars from constructor arguments exist for multi-process/multi-node coordination [7][13], but none of the fetched docs or example scripts show a populated multi-node launch command or a published multi-node benchmark - treat multi-node as a documented mechanism, not a demonstrated one. Generation is a separately-configurable vLLM layer: `collocate` (share GPUs with the Learner) plus `vllm_sleep` (offload vLLM weights/KV cache during the optimizer step) and `vllm_gpu_ratio` (default 0.25, pre-allocated GPU memory fraction for vLLM) [7].

**Install**: `pip install vllm==0.8.4 && pip install -U oat-llm`, exactly as the README's Installation section gives it [1]. PyPI's only published build is `oat-llm` 0.2.4, a wheel-only release (no sdist) uploaded 2025-12-23, whose own metadata requires Python `<3.11,>=3.8` [14] - a different range from `pyproject.toml`'s `requires-python = "~=3.10"` at the same v0.2.4 tag [15]: the wheel's range admits 3.8 and 3.9, which `~=3.10` excludes, while `~=3.10` admits 3.11 and above, which the wheel's range excludes; report both rather than resolving the conflict. Apache-2.0 [1][3]. At the v0.2.4 tag, `pyproject.toml` load-bearing pins are `transformers==4.51.3`, `vllm==0.8.4` (the same version the README's install line names), `deepspeed==0.16.8`, `peft==0.15.2`, `flash-attn==2.7.4.post1`, `numpy==1.26.4`; torch is not listed and arrives transitively [15]. This is a case the spec calls out by name: the screening commit (`8697066`, pushed 2026-01-29, five weeks after the v0.2.4 release) has since edited `pyproject.toml` to unpin `transformers` entirely and bump `vllm==0.8.4` to `vllm==0.11.0`, with no corresponding new release yet [16] - so `pip install oat-llm` today still delivers v0.2.4's older pins, not what HEAD's `pyproject.toml` shows. No CUDA or GPU hardware minimum is stated in the README, `pyproject.toml`, or PyPI metadata; the dry-run guide separately notes its examples are "well-tested on A100-40G GPUs," which is a testing note, not a stated minimum [1][14][15][17].

**Maintained by**: sail-sg (Sea AI Lab), an organization account [2]; latest tagged release is v0.2.4, published 2025-12-23, and the repository's most recent push was 2026-01-29 [16][18]. The README's own dated changelog shows continued activity through 2025: FP16-over-BF16 precision-RL notes (31/10/2025), LoRA-RL support (02/10/2025), and Dr.GRPO integration (21/03/2025) [1].

## Quick start

The README points to two runnable scripts as its quickstart, not to inline pseudo-code [1]:

```bash
# R1-Zero-style RL for math reasoning (Dr.GRPO by default)
bash examples/math_rl.sh
```
This invokes `python -m oat.experiment.run_math_rl --critic_type drgrpo ...` with `pretrain`, `prompt_data`, batch-size, and `--use-wb --wb-run-name` flags already filled in for an 8-GPU run [6].

```bash
# Multi-turn SFT
bash examples/multi_turn_sft.sh
```
This invokes `deepspeed --module oat.experiment.run_offline --algo SFT --chat_data robinsmits/ChatAlpaca-20K --msg_key messages ...` on a 4-GPU, `Qwen/Qwen2.5-Math-1.5B` run [12]. A third guide, `docs/alignment_as_cdb.md`, drives preference learning through `python -m oat.experiment.main`, which itself supports `python -m oat.experiment.main -h` to list every `OATArgs` flag [4][19].

## Start it

- One GPU: run any of the scripts above with `--gpus 1` and matching per-device batch sizes; nothing else in the launch form changes.
- Several GPUs, one node: `math_rl.sh` and `math_rl_lora.sh` launch with `python -m oat.experiment.run_math_rl --gpus 8 ...` [6][20]; `multi_turn_sft.sh` instead launches through `deepspeed --module oat.experiment.run_offline --gpus 4 ...` [12] - the launcher differs per example script, so match the one you copy from rather than assuming one form.
- Effective batch: `train_batch_size` (global) and `train_batch_size_per_device` are both explicit `OATArgs` fields; the shipped SFT script sets `rollout_batch_size_per_device` and `pi_buffer_maxlen_per_device` as `$BATCH_SIZE / $GPUS` so the global batch stays constant as GPU count changes [7][12].
- Generation-layout choice for online methods: `collocate` (Actor shares the Learner's GPUs) plus `vllm_sleep` (offload vLLM weights and drop its KV cache during the optimizer step) and `vllm_gpu_ratio` (default 0.25 on `OATArgs`; the base RL, LoRA-RL, and dry-run example scripts all override it to 0.35) are the knobs, all on `OATArgs` [7][6][20][21].
- Config surface: one shared `OATArgs` dataclass (`oat/args.py`), parsed into a CLI by `tyro.cli` via `get_default_args()`, then checked by `default_args_validation()` before a run starts [7] - a different shape from trl's one-Config-subclass-per-trainer pattern (cross-reference; not covered here). Some algorithms extend it with their own dataclass instead of adding fields to the shared one: `PPOArgs` and `OfflineArgs`, as covered above [10][11]. `bf16` defaults `True` on `OATArgs` itself [7] - the same silent bf16-capable-GPU assumption the spec asks to flag, and `default_args_validation()` enforces `not (args.bf16 and args.fp16)` [7]. tyro's CLI accepts both underscore and hyphen spellings of a flag interchangeably (the example scripts mix `--gpus`, `--gradient-checkpointing`, and `--use_wb` freely) and negates a boolean field with a `--no-` prefix, e.g. `--no-use_fused_lm_head` [7][12][6].
- `default_args_validation()` raises `ValueError("fused lm head is not supported for ZeRO-3, please set --no-use_fused_lm_head")` when `use_fused_lm_head=True` (the field's own default) is combined with `zero_stage=3` [7] - the concrete OOM/first-aid fix for that combination is exactly this flag.
- oat ships its own dedicated OOM pre-flight tool rather than only ad-hoc batch-size advice: `dry_run` mode replaces real data with dummy sequences of a chosen length inside the dataset's `__getitem__`, controlled by `--dry_run`, `--dry_run_prompt_len`, `--dry_run_response_len`, and the docs state the shipped dry-run examples are "well-tested on A100-40G GPUs" [17]. Beyond that mechanism, generation-side OOM first aid is `vllm_gpu_ratio` (lower it) and `vllm_sleep` (enable it to offload weights/cache during optimization) [7].

## Watch it

This section is mechanics only; what a logged value means for a given method (healthy shapes, what to tune) lives on that method's own card, not here.

- **Enable it**: set `--use_wb` (`OATArgs.use_wb`, default `False`) plus `wb_org`, `wb_project` (default `"oat-llm"`), `wb_group`, `wb_run_name`; Weights & Biases is the only logging backend wired into the Learner - with `use_wb` off, `self._wandb` stays `None` and `eval_and_log()` still prints to stdout via `self.strategy.pprint(logs_dict)` but writes no persistent record [22][7]. One source-level quirk to know before running unattended: `wandb.login(key=strategy.args.use_wb)` passes the boolean `use_wb` field itself as the `key=` argument when no cached API key is found, rather than a distinct API-key field [22].
- **Metric names actually logged**, read directly from `oat/learners/base.py`'s `eval_and_log()`/`evaluate()`/`get_misc_info()` at the screening commit, since no rendered metrics page exists for oat [22]: under `eval/` when evaluation runs - `eval/rm_win_rate`, `eval/score`, `eval/accuracy`, `eval/eval_count`, `eval/elapse`, `eval/response_tok_len` [22]; under `misc/` every step - `pi_beta_version`, `global_step`, `policy_sgd_step`, `pi_buffer_len`, `prompt_dataset_len`, `elapse`, `update_interval`, `prompt_epoch`, `gradient_update_elapse`, `weight_sync_elapse`, `vllm_go_sleep_time`, `vllm_wake_up_time`, `vram_allocated`, and `lr` (the last read straight from the scheduler as `self.scheduler.get_last_lr()[0]` before the prefix is applied), plus `misc/query_step` and `misc/prompt_consumed` added separately [22]; algorithm-specific `train_info` and per-actor `actor_info` dicts are merged in too but their keys are set inside each algorithm's own learner class (e.g. `oat/algorithms/ppo.py`), not in the shared base file [22][10].
- The math-RL entrypoint (`oat/experiment/run_math_rl.py`) additionally logs per-benchmark evaluation as `eval/<benchmark_name>/...` plus averaged `eval/average/accuracy`, `eval/average/score`, `eval/average/response_tok_len` when multiple eval benchmarks are configured [5].
- **Sample-level logging**: not found as a dedicated flag in `oat/args.py` or `oat/learners/base.py`; `eval_and_log()` does print one random buffer sample (`np.random.choice(self.pi_buffer)`) to stdout on rank 0 each logging step, which is the closest thing to sample inspection this code path offers [22][7].
- **Evaluate during training**: `eval_steps` (default 20) and `eval_query_interval` (default -1, i.e. step-based) gate cadence; `max_eval`, `eval_split`, `eval_batch_size`, `eval_temperature`/`eval_top_p`/`eval_top_k`/`eval_n`, and `eval_generate_max_length` are separate generation-time fields for the evaluation pass, all on `OATArgs` [7]. Per-step evaluation results are additionally written to `{save_path}/eval_results/{steps}.json` (per `evaluate()`) - not just logged as scalars [22].
- **Stopping**: no patience field or reward-magnitude threshold was found on `OATArgs`, in `oat/learners/base.py`, or in the two RL/SFT example scripts read for this card [7][22][6][12], but a query-budget stopping mechanism does exist: `OATArgs.max_queries` (default -1, resolved to `max_train` when unset by `default_args_validation()`) is checked in `run()`'s training loop as `if self.get_current_query() > self.args.max_queries: early_stop = True` [7][22] - a query-count budget, not a reward-conditioned rule. `max_sgd_steps` (default `math.inf`) and `max_epochs` (default 1) are the other run-length controls located; none of the three is reward-conditioned. Read as of 2026-08-12.

## Save it

- Two independent save paths, both gated inside `eval_and_log()` in `oat/learners/base.py` [22]:
  - `strategy.save_model(...)` always runs when the save trigger fires (`save_steps` cadence, or `save_from` reached); it writes to `{save_path}/saved_models/step_<NNNNN>/` [22].
  - `strategy.save_ckpt(...)` only runs if `args.save_ckpt=True` (default `False`); it writes to `{save_path}/checkpoints/step_<NNNNN>/` via DeepSpeed's native `model.save_checkpoint()`, which is the path that carries optimizer and LR-scheduler state [22][23].
- `save_model` (in `oat/utils/deepspeed.py`) gathers the ZeRO-sharded weights, writes them with `save_pretrained` plus a `config.json` and the tokenizer - an HF-format, directly `from_pretrained`-loadable directory - unless the model is a `peft.PeftModel`, in which case it calls the adapter's own `save_pretrained` instead, citing a DeepSpeed compatibility issue in a source comment, and only the adapter weights land on disk (with an extra `adapter_model.bin` write under ZeRO-3) [23]. An adapter directory saved this way is NOT a full model directory - reload it against the base model, not with a bare `from_pretrained`.
- Retention: both `save_model` and `save_ckpt` take `max_num` (from `OATArgs.max_save_num`, default 5) and `max_mem` (from `OATArgs.max_save_mem`, default 1000 GB) and delete the oldest-by-mtime subdirectory in a loop until both bounds are satisfied; `save_model` deletes while `len(subdirs) > max_num`, `save_ckpt` while `len(subdirs) >= max_num` - a one-off difference between the two implementations, read directly from `oat/utils/deepspeed.py` [23].
- Resume: `resume_dir` plus `resume_tag` on `OATArgs` trigger `strategy.load_ckpt(self.model.model, args.resume_dir, args.resume_tag)` at the top of `run()`, which calls DeepSpeed's `model.load_checkpoint(...)` and restores model, optimizer, and scheduler state - but only from a `save_ckpt` (`checkpoints/`) directory, not a `save_model` (`saved_models/`) one, since `load_ckpt` asserts its input `isinstance(model, deepspeed.DeepSpeedEngine)` [22][23]. `run()` carries a source comment marking dataset/dataloader position as a `# TODO`: resume restores training state but not where the data iterator had reached [22].
- Whether the evaluator can load what you saved is the loader's contract, not oat's: a plain `saved_models/step_.../` directory is HF-format and `from_pretrained`-loadable as-is; a PEFT-adapter save under the same path is not, and must be paired with the base model to reconstruct the trained weights [23].

## Find it in the docs

There is no separate docs website for oat; the GitHub repository at a given ref is the whole of the documentation, and this section is a map of it rather than a mirror.

- Root README at `https://github.com/sail-sg/oat/blob/<ref>/README.md`, where `<ref>` is a branch (`main`) or a tag (`v0.2.4`); verified both `main` and `v0.2.4` render, fetched 2026-08-12 [1][24]. The README's own README-content is unchanged between the v0.2.4 tag and the screening commit (byte-identical file), even though `pyproject.toml` has since drifted between the same two points, as covered under Install - check the file you actually need, not just the README, when currency matters [1][24][16].
- Three markdown guides sit under `docs/` at the repo root: `docs/alignment_as_cdb.md` (Contextual-Dueling-Bandit framing of preference learning, linked from the README's Usage section) [1][19], `docs/preference_learning_examples.md` (direct optimizers, preference oracles, and the SEA/EE4LLM/APL/XPO exploration algorithms, each with a runnable command) [25], and `docs/reasoning_examples.md` (PPO-for-math and GRPO-for-CountDown worked examples, not linked from the top-level README's Usage list) [9].
- Stale cross-reference to flag before trusting either guide literally: `docs/alignment_as_cdb.md` gives example commands using `--dap-algo` and points a further reader to `./preference_learning.md`, but at the screening commit the actual CLI flag is `--algo` (confirmed against `oat/args.py` and `oat/experiment/main.py`) and the actual file is `docs/preference_learning_examples.md` - a docs/code and docs/docs drift, not a maintainer-reported trap, so verify flag and file names against the source you are about to run rather than the guide's prose [19][7][4][25].
- Runnable references beyond the docs: the `examples/` tree, including `examples/math_rl.sh`, `examples/math_rl_lora.sh` (LoRA-RL, added per the README's 02/10/2025 changelog entry), `examples/multi_turn_sft.sh`, and `examples/dry_run/` (OOM pre-flight scripts and their own short README) [6][20][12][17]. `python -m oat.experiment.main -h` lists every `OATArgs` flag directly from the code, which is the fastest way to check a flag name against the version you have installed [19].
- Community layer: the README's Adopters section lists four projects built on oat, and its Updates section points to `sail-sg/Precision-RL` and `sail-sg/understand-r1-zero` as related repos (the latter being the source of the Dr.GRPO fix oat incorporated) - no separate curated-tutorials page exists to check for practitioner blog posts the way trl's `community_tutorials` page does [1]. No official MCP endpoint for oat's docs was found.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. All GitHub source-file citations are read at commit `869706620a35ec5304fa80a0c74f0c566cc2bc57` (the screening commit) unless a tag is explicitly named; all page/API reads are dated 2026-08-12 unless stated otherwise in the text. Method names (DPO, PPO, GRPO, SFT, ...) are deliberately cited to nothing here: their defining papers live on the methodology cards.

[1] oat README at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/README.md. Fetched 2026-08-12.

[2] oat GitHub repository API metadata (owner type, description). https://api.github.com/repos/sail-sg/oat. Fetched 2026-08-12.

[3] oat GitHub repository. https://github.com/sail-sg/oat. Fetched 2026-08-12.

[4] oat/experiment/main.py at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/oat/experiment/main.py. Fetched 2026-08-12.

[5] oat/experiment/run_math_rl.py at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/oat/experiment/run_math_rl.py. Fetched 2026-08-12.

[6] examples/math_rl.sh at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/examples/math_rl.sh. Fetched 2026-08-12.

[7] oat/args.py at the screening commit (the `OATArgs` dataclass, `get_default_args`, `default_args_validation`). https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/oat/args.py. Fetched 2026-08-12.

[8] oat repository git tree at the screening commit (confirms no separate docs site; enumerates `docs/`, `examples/` paths). https://api.github.com/repos/sail-sg/oat/git/trees/869706620a35ec5304fa80a0c74f0c566cc2bc57?recursive=1. Fetched 2026-08-12.

[9] docs/reasoning_examples.md at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/docs/reasoning_examples.md. Fetched 2026-08-12.

[10] oat/algorithms/ppo.py at the screening commit (`PPOArgs` dataclass). https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/oat/algorithms/ppo.py. Fetched 2026-08-12.

[11] oat/experiment/run_offline.py at the screening commit (`OfflineArgs` dataclass). https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/oat/experiment/run_offline.py. Fetched 2026-08-12.

[12] examples/multi_turn_sft.sh at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/examples/multi_turn_sft.sh. Fetched 2026-08-12.

[13] oat/utils/launcher.py at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/oat/utils/launcher.py. Fetched 2026-08-12.

[14] PyPI JSON API for oat-llm (version, requires_python, upload_time, packagetype). https://pypi.org/pypi/oat-llm/json. Fetched 2026-08-12.

[15] pyproject.toml at the v0.2.4 tag (dependency pins, requires-python). https://raw.githubusercontent.com/sail-sg/oat/v0.2.4/pyproject.toml. Fetched 2026-08-12.

[16] pyproject.toml at the screening commit, diffed against [15] (transformers unpinned, vllm bumped to 0.11.0 since the v0.2.4 release). https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/pyproject.toml. Fetched 2026-08-12.

[17] examples/dry_run/README.md at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/examples/dry_run/README.md. Fetched 2026-08-12.

[18] oat GitHub releases API (v0.2.4 published_at, and the repository's own pushed_at from [2]). https://api.github.com/repos/sail-sg/oat/releases. Fetched 2026-08-12.

[19] docs/alignment_as_cdb.md at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/docs/alignment_as_cdb.md. Fetched 2026-08-12.

[20] examples/math_rl_lora.sh at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/examples/math_rl_lora.sh. Fetched 2026-08-12.

[21] examples/dry_run/dry_run_rl.sh at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/examples/dry_run/dry_run_rl.sh. Fetched 2026-08-12.

[22] oat/learners/base.py at the screening commit (wandb init, `run()` resume logic, `eval_and_log()`, `evaluate()`, `get_misc_info()`). https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/oat/learners/base.py. Fetched 2026-08-12.

[23] oat/utils/deepspeed.py at the screening commit (`save_model`, `save_ckpt`, `load_ckpt`, `load_model`). https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/oat/utils/deepspeed.py. Fetched 2026-08-12.

[24] README.md at the v0.2.4 tag, byte-compared against [1]. https://raw.githubusercontent.com/sail-sg/oat/v0.2.4/README.md. Fetched 2026-08-12.

[25] docs/preference_learning_examples.md at the screening commit. https://raw.githubusercontent.com/sail-sg/oat/869706620a35ec5304fa80a0c74f0c566cc2bc57/docs/preference_learning_examples.md. Fetched 2026-08-12.
