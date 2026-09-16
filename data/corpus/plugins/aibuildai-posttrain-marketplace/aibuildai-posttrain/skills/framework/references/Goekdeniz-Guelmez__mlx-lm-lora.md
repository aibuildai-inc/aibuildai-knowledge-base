# mlx-lm-lora

A CLI-first post-training library for Apple Silicon: one `mlx_lm_lora.train` command runs LoRA/DoRA/full-precision SFT and eleven preference/RL methods natively on MLX, with no CUDA path.

Its README opens: "With MLX-LM-LoRA you can, train Large Language Models locally on Apple Silicon using MLX", and training works with any model MLX-LM supports [1]. It is a single-maintainer project by Gökdeniz Gülmez [2][3], and it lives at https://github.com/Goekdeniz-Guelmez/mlx-lm-lora [3]. It is built directly on top of the separate `mlx_lm` PyPI package, from which it imports model/adapter loading (`mlx_lm.utils.load`) and the training-callback base (`mlx_lm.tuner.callbacks.WandBCallback`) [4]. The API is one CLI entry point, `mlx_lm_lora.train`, that takes a `--model` id, a `--data` path or Hugging Face dataset name, a `--train-mode` naming the method, and either flags or a YAML `--config` file, then runs to completion [1].

**When to pick it**: training on a Mac (M1-M4, unified memory) when your model and data are on the Hub or local disk and you want one CLI covering SFT plus eleven preference/RL methods without leaving Apple Silicon; there is no CUDA path, so on NVIDIA hardware this is not a candidate at all - the README's own comparison table reports it running only on Apple Silicon, contrasted with Unsloth on NVIDIA GPUs [5]. Two of its preference trainers, ORPO and CPO, shipped a silent zero-gradient bug through the v3.0.0 release; see the trap below before choosing either for anything beyond v3.1.2 [6][7].

**Methods it ships** [1] (method definitions and math live on each method's own card; this card covers only what the library adds):
- Training types: LoRA, DoRA, full-precision, and quantized (QLoRA) training at 4/6/8-bit; Quantization-Aware Training (QAT), which "projects trainable weights onto a quantized grid after each optimizer update", is layered on top of SFT, DPO, or ORPO only [1].
- Eleven trainer modules under `mlx_lm_lora/trainer/`: `sft_trainer.py`, `dpo_trainer.py`, `cpo_trainer.py`, `orpo_trainer.py`, `grpo_trainer.py`, `online_dpo_trainer.py`, `xpo_trainer.py`, `rlhf_reinforce_trainer.py`, `ppo_trainer.py`, `ftpo_trainer.py`, plus a `judge.py` that the online trainers call for preference judgments: its `LLMPairwiseJudge`, `LLMPPOJudge`, and `HumanPairwiseJudge` classes score or rank candidate completions by calling `mlx_lm`'s `generate()` at inference time - this module does not itself train anything (read at commit fb4f39db) [9]. GSPO, Dr. GRPO, and DAPO are not separate trainers: the README lists them under GRPO and reaches them through GRPO flags - `--importance-sampling-level` for GSPO-style sequence-level ratios, `--grpo-loss-type dr_grpo`, and `--epsilon-high` for DAPO's asymmetric clip [1]. FTPO ("Antidoom") is presented as a method for repairing repetition loops rather than a general preference method [1]. The README carries no experimental/stable split or separate import path for any trainer - every method sits at `mlx_lm_lora.train --train-mode <name>` [1].

**Scale it handles**: single machine, single GPU (Apple Silicon's one integrated GPU) is the only form the README documents - `mlx_lm_lora.train ...` run directly, no launcher [1]. All ten online/offline trainers call MLX's own `mx.distributed` primitives (`mx.distributed.init()`, `.all_sum(...)`) to reduce losses, rewards, and token counts across ranks, so multi-process reduction exists at the code level (read at commit `fb4f39db`, the screened v3.0.0 tag) [10], but the README documents no multi-node or multi-process launch command, config template, or benchmark for it - this is a documented-mechanism-only claim, not a published capability.

**Install**: `pip install -U mlx-lm-lora` [1]. The shortlist screened commit `fb4f39db` (tag `v3.0.0`, released 2026-07-14), which was the repository's newest push at screening time; re-checking the live GitHub commits API today (2026-08-12) shows the newest push to `main` is now `4f7c51ce` (2026-08-10T13:40:34Z, matching the repo's `pushed_at` of 2026-08-10T13:48:13Z) - the same commit as the `v3.1.2` tag [13][14]. So `pip install` today delivers v3.1.2, not the screened v3.0.0 - version and dependency claims below are read at v3.1.2 [11][12]. `_version.py` at that tag gives `3.1.2`; PyPI's own metadata confirms `info.version: "3.1.2"` [12][15]. Python `>=3.8` [15]. `requirements.txt` at the v3.1.2 tag pins `mlx>=0.30.6`, `mlx_lm>=0.30.6`, `transformers>=4.39.3`, with `numpy`, `protobuf`, `pyyaml`, `jinja2`, `tqdm`, `datasets` unpinned - identical to the requirements file at the screened v3.0.0 commit, so no floor changed between the two [16][17]. No `pyproject.toml` exists; `setup.py` reads these same pins from `requirements.txt` at install time [18]. Licence: the repository's `LICENSE` file and the GitHub API's own `license` field both give Apache-2.0 [19][3]; `setup.py` and PyPI's `info.license` field instead carry a stale `"MIT"` string that does not match the actual license text [18][15] - Apache-2.0 is the one the license file itself states. No CUDA, torch, or platform minimum is stated anywhere in the install surface; the README's benchmark section states the hardware requirement instead: "Requires Apple Silicon (M1/M2/M3/M4)" [5].

**Maintained by**: Gökdeniz Gülmez, a single maintainer (`author_email` in `setup.py`) [18], with one outside contributor merged in the v3.1.2 release [20]. Development is active: v3.1.2 shipped 2026-08-10, less than a month after v3.0.0 (2026-07-14) [11], and its release notes list a correctness fix, an SSM training-speed optimization "used for the JOSIE-2 model family", and a notebook update [20].

## Quick start

The full basic run from the README, an SFT LoRA fine-tune reading a Hugging Face dataset directly [1]:

```shell
pip install -U mlx-lm-lora

mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--data mlx-community/wikisql \
--iters 600
```

`mlx_lm_lora.train --help` lists all flags; `--config /path/to/config.yaml` loads a YAML config, and CLI flags override the file's values [1]. Method choice is `--train-mode {sft,dpo,cpo,orpo,grpo,...}`; training type is `--train-type {lora,dora,full}` [1][21].

## Start it

- Single GPU is the only launch form documented: run `mlx_lm_lora.train ...` directly, or point `--config` at a YAML file such as the README's basic-LoRA example (`train_type: lora`, `train_mode: sft`, `batch_size: 4`, `learning_rate: 1e-5`, `iters: 1000`, `lora_parameters: {rank: 8, dropout: 0.0, scale: 10.0}`) [1].
- Effective batch is `--batch-size` x `--gradient-accumulation-steps`; the README's own OOM guidance pairs `--gradient-accumulation-steps 4 --batch-size 1` to hold the effective batch while cutting per-step memory [22].
- `TrainingArgs.steps_per_save` (source-level field name, read at `fb4f39db`) is set from the CLI flag `--save-every`, default 100 in `train.py`'s `CONFIG_DEFAULTS` [23]; `--steps-per-report` (default 10) and `--steps-per-eval` (default 200) set report and validation cadence [1][23].
- Config surface: every `train_mode` shares the core flags under the README's Configuration section - `--model`, `--data`, `--train-type`, `--num-layers` (-1 for all), `--max-seq-length` (2048), `--lora-parameters`, `--optimizer {adam,adamw,qhadam,muon}`, `--lr-schedule {cosine,linear,constant}`, `--grad-checkpoint` [21]. GRPO adds `--reward-functions`, `--reward-weights`, and `--list-reward-functions` to enumerate built-in reward functions [1]. No default in `CONFIG_DEFAULTS` changes precision away from the model's own stored dtype; QAT and the `--load-in-{4,6,8}bits` flags are the only precision knobs, and both are opt-in [23][24].
- Out-of-memory first aid, from the README's own guidance: reduce `--batch-size`, quantize the base model with `--load-in-4bits`/`--load-in-6bits`/`--load-in-8bits`, reduce `--num-layers`, enable `--grad-checkpoint`, reduce `--max-seq-length`, or trade batch size for `--gradient-accumulation-steps` [22][25]. The README's own per-size table: 1-3B models suggest `--batch-size 4 --num-layers 16`; 7B suggests `--batch-size 2 --num-layers 8 --load-in-8bits`; 13B+ suggests `--batch-size 1 --num-layers 4 --load-in-4bits --grad-checkpoint` [25].

## Watch it

Mechanics only - what a given metric means for a specific method lives on that method's card.

- No `--wandb` flag means no tracker at all: `train.py`'s `run()` only constructs a `WandBCallback` (from the upstream `mlx_lm` package) when `args.wandb is not None`, and the top-level `main()` calls `run(args)` with no callback argument, so a run without `--wandb <project_name>` writes only console/tqdm progress - no run record survives past the terminal (read at `fb4f39db`, the screened commit; unchanged in the v3.1.2 dependency files) [26].
- SFT's on-`training_callback.on_train_loss_report` payload (`sft_trainer.py`, read at `fb4f39db`) carries `iteration`, `train_loss`, `learning_rate`, `iterations_per_second`, `tokens_per_second`, `trained_tokens`, `peak_memory`; a separate `on_val_loss_report` call carries `iteration`, `val_loss`, and `val_time` [27].
- GRPO's payload (`grpo_trainer.py`, same commit) carries the same core fields plus every key in its `avg_metrics` dict prefixed `train_`: per-reward-function `{name}_mean`, `{name}_std`, `{name}_coverage`; combined `total_rewards_mean`, `total_rewards_std`, `grouped_rewards_mean`, `grouped_rewards_std`; `kl`; `average_generated_tokens`, `min_generated_tokens`, `max_generated_tokens`, `hit_max_tokens_ratio`; and `clip_ratio_low`, `clip_ratio_high`, `clip_ratio_total` [28]. No README page or docs page separately lists these names; they were read directly from the trainer source, so treat this list as commit-pinned rather than a documented contract.
- Sample-level generation logging and a published stopping rule or health-limit threshold were not found: the README's Troubleshooting section gives qualitative advice only ("Convergence Issues: Adjust learning rate, try different optimizers") with no threshold, metric name, or patience value [25], and no other README section (Configuration, Memory Optimization, Advanced Features) publishes one either - search run 2026-08-12 over those four sections plus the full-text CLI flag list in `train.py`.
- Evaluation during training: `--test` plus `--test-batches 500` runs a held-out evaluation pass after training [29]; `--val-batches` (default 25) and `--steps-per-eval` (default 200) control in-training validation cadence and volume [21][23].

## Save it

- Adapter checkpoints are LoRA/DoRA-only weight files, not a full model: `sft_trainer.py`'s save step (read at `fb4f39db`) builds `adapter_weights = dict(tree_flatten(model.trainable_parameters()))` and writes it with `mx.save_safetensors` both to the running `adapter_file` (default `adapters.safetensors`, per the `SFTTrainingArgs` field default) and to a numbered snapshot `{iteration:07d}_adapters.safetensors` in the same directory every `--save-every` steps [30]. `mlx_lm_lora/utils.py`'s `from_pretrained`/save path also writes `adapter_config.json` alongside the weights [31] - together an adapter directory, never a base-model directory.
- Resume: `--resume-adapter-file <path>` reloads a saved adapter checkpoint's weights before continuing training [21].
- Fuse into a full model is a separate, explicit step: `mlx_lm_lora.train --model <model_path> --adapter-path <adapter_path> --fuse` calls `mlx_lm_lora/utils.py`'s `save_pretrained_merged`, whose own docstring says it will "fuse fine-tuned adapters into the base model", with `de_quantize` and `export_gguf` options for a de-quantized or GGUF-format merged output [32][33]. Only after this step does the output directory hold a loadable full model.
- Generation and evaluation of a trained adapter reuse the upstream `mlx-lm` package's own CLI rather than anything in this repo: `mlx_lm.generate --model <model_path> --adapter-path <adapter_path> --prompt "..." --max-tokens 100 --temperature 0.7` [29]. `mlx_lm_lora/utils.py`'s own `from_pretrained` helper likewise calls `mlx_lm.utils.load(model, adapter_path=adapter_path)` internally [31] - the loader contract for both an unfused adapter and a fused model is `mlx_lm`'s loader, not a loader this repo defines itself; confirm against this skill's shared loading-the-result reference before the first save.

## Find it in the docs

There is no separate hosted docs site: the repository has no `docs/` folder (confirmed against the full git tree at commit `fb4f39db`) [8], and the README itself, with its own table-of-contents anchors, is the documentation [1].

- Address pattern: `https://github.com/Goekdeniz-Guelmez/mlx-lm-lora#<anchor>`, where `<anchor>` is the kebab-case heading text from the README's own Contents list - e.g. `#group-relative-policy-optimization-grpo`, `#dataset-formats`, `#performance-comparison` [1]. The Contents block at the top of the README is the fastest way to find a method's section; every training method, plus Configuration, Dataset Formats, Memory Optimization, Evaluation & Generation, and Performance Comparison, has its own anchor there [1].
- Dataset shapes: the README's Dataset Formats section gives the on-disk layout (a `data/` directory of `train.jsonl`/`valid.jsonl`/`test.jsonl`, or a Hugging Face dataset name passed to `--data`) and worked JSONL examples per mode - e.g. SFT chat format is a `messages` list of role/content dicts, SFT completion format is a flat `{"prompt": ..., "completion": ...}` pair - plus flags to remap field names (`--chosen-feature`, `--rejected-feature`, and so on) for datasets that use different keys [34].
- Runnable references beyond the README live in a separate, actively maintained repository, `Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks`, whose own description says it holds "all official MLX-LM-LoRA example notebooks for training on Apple Silicon" [35], last pushed 2026-04-23 [35]; the main README explicitly redirects all example notebooks there, listing simple and detailed SFT, ORPO, DPO, and GRPO notebooks plus a YAML config example [1].
- No curated community-tutorial page or official MCP endpoint was found in the README or the repository tree read at `fb4f39db` [1][8] - this card names none rather than guessing at one.
- Trap, stated where it bites: ORPO's and CPO's trainers computed their forward pass (`get_logps`) outside the closure differentiated by `nn.value_and_grad`, producing a structurally zero gradient, and `orpo_loss` was additionally missing the paper's NLL/SFT anchor term on the chosen sequence - both confirmed present in `orpo_trainer.py` at the screened commit `fb4f39db` (v3.0.0) by reading the source directly. Issue #65 (opened by a reporter with no repository affiliation, `author_association: NONE`; closed 2026-08-10 by owner Goekdeniz-Guelmez, `author_association: OWNER`) reports both bugs with a 35-iteration run showing adapter weights "bitwise unchanged" and a 90-iteration run reaching a reported validation margin while the model had collapsed to a single repeated token; a follow-up comment from the same reporter (2026-07-30) states the identical pattern also exists in `cpo_trainer.py`, "All still present on main (3.0.0, fb4f39db)." [6]. The fix - a new `orpo_loss_from_model` wrapper that puts `get_logps` inside the differentiated closure, plus a restored `chosen_nll` term in `orpo_loss` - is confirmed present by direct source comparison in the v3.1.2 release (commit `4f7c51ce`, 2026-08-10), and the same v3.1.2 release notes credit the CPO half of the fix to a PR "by @swvgjbb78s-coder" [7][20]. Anyone training ORPO or CPO on a version pinned to v3.0.0 or the `fb4f39db` commit is training against a zero policy gradient; v3.1.2 or later is required.
- A second maintainer-confirmed trap, resolved before the screened commit: issue #55 (closed 2026-07-14, the same day v3.0.0 shipped) reported that GRPO's `importance_sampling_level` defaulting to `None` also produced an exactly-zero policy gradient; the owner's reply confirms "fix is merged, the new version will be released later this week!" [36]. `train.py`'s `CONFIG_DEFAULTS["importance_sampling_level"]` reads `"token"` at the screened commit `fb4f39db`, confirming the fix is present in v3.0.0 onward [23].

## Sources

All GitHub pages and the PyPI JSON API are unpinned/live and were fetched 2026-08-12 except where a commit or tag is named; source-code claims are pinned to the two commits stated in the Install field, `fb4f39db66fadec3b71a41441e863d9f1bf87844` (tag v3.0.0, the shortlist's screened commit) and `4f7c51ce7b8d8c5c9cf88b771c19f8edbcb03d86` (tag v3.1.2, the commit `pip install` currently delivers). Method names (SFT, DPO, CPO, ORPO, GRPO, GSPO, Dr. GRPO, DAPO, Online DPO, XPO, RLHF Reinforce, PPO) are deliberately cited to nothing here: their defining papers live on the methodology cards.

[1] mlx-lm-lora README, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/README.md. Fetched 2026-08-12.

[2] mlx-lm-lora setup.py, at commit fb4f39db (author field). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/setup.py. Fetched 2026-08-12.

[3] mlx-lm-lora GitHub repository (item home). https://github.com/Goekdeniz-Guelmez/mlx-lm-lora. Fetched 2026-08-12.

[4] mlx-lm-lora train.py, at commit fb4f39db (imports from the external mlx_lm package). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/train.py. Fetched 2026-08-12.

[5] mlx-lm-lora README, Performance Comparison section, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/README.md. Fetched 2026-08-12.

[6] GitHub issue #65 and its comments (ORPO/CPO zero-gradient and missing-NLL-term bug report). https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora/issues/65 and https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora/issues/65/comments. Fetched 2026-08-12.

[7] mlx-lm-lora orpo_trainer.py, at tag v3.1.2 (commit 4f7c51ce) - direct source comparison confirming the fix. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/4f7c51ce7b8d8c5c9cf88b771c19f8edbcb03d86/mlx_lm_lora/trainer/orpo_trainer.py. Fetched 2026-08-12.

[8] mlx-lm-lora git tree, at commit fb4f39db (repository structure, confirms no docs/ folder). https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora/git/trees/fb4f39db66fadec3b71a41441e863d9f1bf87844?recursive=1. Fetched 2026-08-12.

[9] mlx-lm-lora judge.py, at commit fb4f39db (confirms LLMPairwiseJudge, LLMPPOJudge, and HumanPairwiseJudge call mlx_lm's generate() for inference-time judging, with no training loop in the file). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/trainer/judge.py. Fetched 2026-08-12.

[10] mlx-lm-lora trainer source files (grpo_trainer.py, xpo_trainer.py, rlhf_reinforce_trainer.py, dpo_trainer.py), at commit fb4f39db (mx.distributed usage). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/trainer/grpo_trainer.py (and sibling trainer files at the same commit). Fetched 2026-08-12.

[11] mlx-lm-lora GitHub releases API (publish dates for v3.0.0 and v3.1.2). https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora/releases. Fetched 2026-08-12.

[12] mlx-lm-lora GitHub tags API (tag-to-commit mapping for v3.0.0 and v3.1.2). https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora/tags. Fetched 2026-08-12.

[13] mlx-lm-lora GitHub commits API for the `main` branch (current newest push: commit `4f7c51ce`, committer date 2026-08-10T13:40:34Z). https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora/commits/main. Fetched 2026-08-12.

[14] mlx-lm-lora GitHub repository API (`pushed_at` field: 2026-08-10T13:48:13Z). https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora. Fetched 2026-08-12.

[15] mlx-lm-lora PyPI JSON API. https://pypi.org/pypi/mlx-lm-lora/json. Fetched 2026-08-12.

[16] mlx-lm-lora requirements.txt, at tag v3.1.2 (commit 4f7c51ce). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/4f7c51ce7b8d8c5c9cf88b771c19f8edbcb03d86/requirements.txt. Fetched 2026-08-12.

[17] mlx-lm-lora requirements.txt, at commit fb4f39db (comparison baseline for the screened commit). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/requirements.txt. Fetched 2026-08-12.

[18] mlx-lm-lora setup.py, at commit fb4f39db (license string, Python floor, dependency source). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/setup.py. Fetched 2026-08-12.

[19] mlx-lm-lora LICENSE file, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/LICENSE. Fetched 2026-08-12.

[20] mlx-lm-lora GitHub release notes for v3.1.2. https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora/releases (tag_name: v3.1.2). Fetched 2026-08-12.

[21] mlx-lm-lora README, Configuration section, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/README.md. Fetched 2026-08-12.

[22] mlx-lm-lora README, Memory Optimization section, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/README.md. Fetched 2026-08-12.

[23] mlx-lm-lora train.py, CONFIG_DEFAULTS dict, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/train.py. Fetched 2026-08-12.

[24] mlx-lm-lora README, Quantization Aware Training section, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/README.md. Fetched 2026-08-12.

[25] mlx-lm-lora README, Troubleshooting section, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/README.md. Fetched 2026-08-12.

[26] mlx-lm-lora train.py, run() and main() functions, at commit fb4f39db (WandBCallback construction gated on args.wandb). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/train.py. Fetched 2026-08-12.

[27] mlx-lm-lora sft_trainer.py, at commit fb4f39db (train_info and val_info dict contents). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/trainer/sft_trainer.py. Fetched 2026-08-12.

[28] mlx-lm-lora grpo_trainer.py, at commit fb4f39db (train_info and avg_metrics dict contents). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/trainer/grpo_trainer.py. Fetched 2026-08-12.

[29] mlx-lm-lora README, Evaluation & Generation section, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/README.md. Fetched 2026-08-12.

[30] mlx-lm-lora sft_trainer.py, save step and SFTTrainingArgs field defaults, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/trainer/sft_trainer.py. Fetched 2026-08-12.

[31] mlx-lm-lora utils.py, at commit fb4f39db (from_pretrained, save_pretrained_merged, adapter_config.json handling, calls into mlx_lm.utils.load). https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/utils.py. Fetched 2026-08-12.

[32] mlx-lm-lora README, Fusing Adapters section, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/README.md. Fetched 2026-08-12.

[33] mlx-lm-lora utils.py, save_pretrained_merged docstring, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/mlx_lm_lora/utils.py. Fetched 2026-08-12.

[34] mlx-lm-lora README, Dataset Formats section, at commit fb4f39db. https://raw.githubusercontent.com/Goekdeniz-Guelmez/mlx-lm-lora/fb4f39db66fadec3b71a41441e863d9f1bf87844/README.md. Fetched 2026-08-12.

[35] Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks GitHub repository. https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks. Fetched 2026-08-12.

[36] GitHub issue #55 and its comments (GRPO importance_sampling_level zero-gradient bug report, resolved before v3.0.0). https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora/issues/55 and https://api.github.com/repos/Goekdeniz-Guelmez/mlx-lm-lora/issues/55/comments. Fetched 2026-08-12.
