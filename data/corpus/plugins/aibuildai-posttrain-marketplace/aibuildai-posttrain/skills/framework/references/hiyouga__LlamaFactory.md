# LlamaFactory

A no-code CLI/Web-UI framework for fine-tuning 100+ LLMs and VLMs with a fixed set of training stages (pt/sft/rm/ppo/dpo/kto) - broad model coverage over method breadth, and no GRPO stage in the codebase.

**LlamaFactory** (package name `llamafactory`, formerly LLaMA-Factory and LLaMA-Efficient-Tuning [1]) is "an easy-to-use and efficient platform for training and fine-tuning large language models," letting a user "fine-tune hundreds of pre-trained models locally without writing any code" [2]. It is built by Yaowei Zheng and collaborators and published as an ACL 2024 System Demonstration paper [3]. It lives at https://github.com/hiyouga/LlamaFactory [4]. The API is stage-based: every run sets a `stage` field (`pt`, `sft`, `rm`, `ppo`, `dpo`, or `kto`) on a YAML config, plus a `finetuning_type` (`lora`, `oft`, `freeze`, `full`), and is launched through the `llamafactory-cli train config.yaml` command or the LlamaBoard web UI [5][6].

**When to pick it**: pick LlamaFactory when the priority is fine-tuning a very wide range of already-published LLM/VLM checkpoints (the paper's title claims support for over 100 language models [3]) through a fixed, no-code YAML+CLI/WebUI workflow, with LoRA/QLoRA/OFT/full tuning across pt/sft/rm/ppo/dpo/kto. At commit 62ae362 (committed 2026-07-31), the `stage` field is typed `Literal["pt", "sft", "rm", "ppo", "dpo", "kto"]` [7] - there is no GRPO stage, so a reader wanting GRPO-style online RL should look elsewhere (cross-reference; not covered here).

**Methods it ships**: training stages are `pt` (pretraining), `sft` (supervised fine-tuning), `rm` (reward modeling), `ppo`, `dpo`, and `kto`, each set via the `stage` field [7]; ORPO and SimPO are not separate stages but `pref_loss` variants of the `dpo` stage (`pref_loss: Literal["sigmoid","hinge","ipo","kto_pair","orpo","simpo"]`, default `sigmoid`) [8]. Tuning methods (orthogonal to stage) are full, freeze, LoRA, QLoRA, OFT, and QOFT, all marked supported across every stage in the README's method matrix [9]. The codebase has two parallel architectures gated by the `USE_V1` env var: the default v0 tree (`src/llamafactory/train/{pt,sft,rm,ppo,dpo,kto}/`) is where "most active development happens," while v1 (`src/llamafactory/v1/`, opt-in) is described by the repo's own contributor guide as experimental and currently ships only SFT, DPO, and RM trainers, no PPO or KTO [10]. Advanced training tricks the README lists include GaLore, BAdam, APOLLO, Adam-mini, Muon, DoRA, LongLoRA, LoRA+, LoftQ, and PiSSA, plus FlashAttention-2, Unsloth, and Liger Kernel acceleration [9].

**Scale it handles**: single GPU up to multi-node, all through the same `llamafactory-cli train` entry point. Single-node multi-GPU forces torchrun via `FORCE_TORCHRUN=1 llamafactory-cli train config.yaml` (auto-detects visible GPUs, or set `CUDA_VISIBLE_DEVICES`); multi-node repeats that command on each node with `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` set as env vars, or the run can be launched directly with `torchrun` or `accelerate launch --config_file <yaml>` [11]. Sharding options are DeepSpeed (ZeRO-1/2/3, with optional parameter/optimizer CPU offload) and FSDP/FSDP2, with a docs table comparing which of data/model/optimizer sharding and parameter offload each supports [11]; no multi-node throughput benchmark is published on this page. The paper's own single-GPU efficiency table, run on a single A100 40GB fine-tuning Llama2-7B, is the closest published reference run: full-tuning peaks at 38.72 GB and 1334.72 tokens/s, while QLoRA cuts peak memory to 7.52 GB at 1579.16 tokens/s, and LoRA (with Unsloth attaching adapters) reaches the highest throughput among GaLore, LoRA, and QLoRA at 1954.07 tokens/s versus 1583.77 tokens/s for GaLore - a gradient low-rank-projection method for full-parameter training rather than an adapter method - at a similar 15-16 GB memory footprint for both [3].

**Install**: `git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git && cd LlamaFactory && pip install -e . && pip install -r requirements/metrics.txt` [9]; PyPI package `llamafactory`, latest release 0.9.5 (uploaded 2026-05-30) [12]; `requires-python = ">=3.11.0"` [13]. Licence Apache-2.0 [4]. At the v0.9.5 tag, `pyproject.toml` pins the deep-learning core as `torch>=2.4.0`, `transformers>=4.55.0,<=5.6.0,!=4.52.0,!=4.57.0`, `accelerate>=1.3.0,<=1.11.0`, `peft>=0.18.0,<=0.18.1`, and `trl>=0.18.0,<=0.24.0` [13] - these are tighter/newer than the README's own "Requirement" table, which still shows `transformers>=4.49.0` as a minimum, so the pyproject.toml pins at the release tag are the ones to trust over the README text [13][9]. Optional extras are separate `requirements/*.txt` files rather than PEP 621 extras: `requirements/vllm.txt` pins `vllm>=0.4.3,<=0.11.0`, `requirements/deepspeed.txt` pins `deepspeed>=0.10.0,<=0.18.4` [14][15]. The README's own Requirement table lists CUDA as an "Optional" row with a minimum of 11.6 and a recommended 12.2 (alongside optional-row floors for deepspeed 0.10.0, bitsandbytes 0.39.0, vllm 0.4.3, and flash-attn 2.5.6) [9]. A pre-built Docker image (`hiyouga/llamafactory:latest`) is separately documented as built on Ubuntu 22.04, CUDA 12.4, Python 3.11, PyTorch 2.6.0, and Flash-attn 2.7.4 [9].

**Maintained by**: primary author Yaowei Zheng and the `hiyouga` GitHub account [4][3]; `stargazers_count` and `pushed_at` are live, unpinned GitHub API fields with no revision parameter, so they are reported as of the fetch rather than as of any commit: 73,946 stars and a default-branch push at 2026-08-09T08:00:11Z, both read 2026-08-10 (not a ranking signal) [4]; actively developed, with the v0.9.5 release (2026-05-30) release notes describing added primary support for the Qwen3.5, Qwen3.6, and Gemma4 model families and compatibility with Transformers v5 [16].

## Quick start

The docs' SFT quickstart is a full command line, not a Python snippet: LlamaFactory's config-driven CLI runs a complete training job from a packaged example YAML [17]:

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

Any field in that YAML can be overridden on the command line by appending `key=value` pairs after the config path [17]. The example config sets, among other fields, `model_name_or_path`, `stage: sft`, `finetuning_type: lora`, `lora_rank`, `lora_target`, `dataset`, `template`, `cutoff_len`, `output_dir`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, and `bf16: true` [17]. Other stages are launched identically, only the `stage` field and stage-specific args change: `stage: pt` for pretraining, `stage: rm` (with a preference dataset) for reward modeling, `stage: ppo` (requires a trained `reward_model`), `stage: dpo` (with `pref_loss` choosing sigmoid-DPO/ORPO/SimPO), and `stage: kto` (with a KTO-formatted dataset) [6].

## Start it

Single GPU is the plain form: `llamafactory-cli train config.yaml` with no extra flags [17]. Single-node multi-GPU forces torchrun: `FORCE_TORCHRUN=1 llamafactory-cli train config.yaml`, auto-detecting all visible GPUs (or restrict with `CUDA_VISIBLE_DEVICES`); the same job can instead be launched with a bare `torchrun --nproc_per_node <n> --master_port <port> src/train.py config.yaml` command, or `accelerate launch --config_file <accelerate_config.yaml> src/train.py config.yaml` against an Accelerate config file [11]. Multi-node repeats the `FORCE_TORCHRUN=1` command on every node, with `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` set as environment variables per node [11]. Sharding is selected through DeepSpeed (ZeRO-1/2/3, optional CPU offload of params/optimizer state) or FSDP/FSDP2, chosen via the accelerate/DeepSpeed config passed at launch [11]. Effective batch size is `per_device_train_batch_size * num_gpus * gradient_accumulation_steps`, both fields set directly in the training YAML [17]. Config is the flat YAML/CLI-args surface parsed into dataclasses such as `FinetuningArguments`; the docs' Arguments reference lists RLHF-specific fields including `pref_beta` (default 0.1), `ppo_epochs` (default 4), `ppo_target` (default 6.0), and quantization fields like `quantization_bit`, `quantization_method` (default `bitsandbytes`), `quantization_type` (default `nf4`) [8]. For OOM, the project's own FAQ issue lists, in order: lower `per_device_train_batch_size: 1`; enable compute-kernel replacements `enable_liger_kernel: true` / `use_unsloth_gc: true`; lower `cutoff_len`; use DeepSpeed ZeRO-3 or FSDP to shard weights, or CPU offload; set `quantization_bit: 4` (LoRA only); or switch to the paged optimizer `optim: paged_adamw_8bit` [18]. When inference runs through vLLM as a separate engine (for PPO rollouts or serving), generation-side memory is tuned via `vllm_gpu_util` (default 0.9) and `vllm_maxlen` (default 4096) [8].

## Watch it

This section covers only the logging mechanics; what a metric shape means for a given method lives on that method's card. Enable logging with `report_to` in the training YAML: the monitoring docs demonstrate `report_to: tensorboard`, `report_to: wandb`, and `report_to: mlflow`; SwanLab is enabled separately via `use_swanlab: true` with `swanlab_project`/`swanlab_run_name` fields, and the built-in LlamaBoard web UI plots loss curves without a `report_to` value [19]. Logging cadence is the `logging_steps` field on the training YAML [17].

- **PPO**: the trainer logs `loss`, `reward`, `learning_rate`, and `epoch` to the standard Trainer log history at `logging_steps` intervals, drawing from TRL's internal PPO stats dict (e.g. `ppo/loss/total`, `ppo/learning_rate`); if `log_with` is set it additionally calls TRL's own `log_stats` [20].
- **DPO/KTO**: the trainer logs TRL-style preference metrics with an `eval_` prefix during evaluation: `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `logps/chosen`, `logps/rejected`, `logits/chosen`, `logits/rejected`, plus `sft_loss` when an SFT auxiliary loss is enabled and `odds_ratio_loss` under the ORPO variant [21].
- No sample-level generation logging, evaluation-during-training field list, or a published RL stopping-rule/threshold was found on the trainer or monitoring docs pages read for this card [19][6]; the arguments reference likewise carries no early-stopping or patience field for PPO/DPO/KTO [8].

## Save it

`output_dir` (set per-run in the training YAML) holds the final model, numbered checkpoint subdirectories, the Trainer state, logs, and loss plots [22]. Checkpoint cadence and retention are the standard Trainer fields `save_strategy`, `save_steps`, and `save_total_limit` (which auto-deletes older checkpoints beyond the limit) [22]. Setting `save_only_model: true` saves the model weights only, without optimizer, scheduler, or other training-state files, which reduces disk usage but the docs say this is usually not suitable when a strict full resume of training is needed later - the result is better suited to inference, evaluation, or weight conversion instead [22]. Resume a run by pointing `resume_from_checkpoint` at a specific checkpoint directory, e.g. `resume_from_checkpoint: saves/qwen3_8b_lora_sft/checkpoint-1000` [22]. LoRA/OFT adapters are saved on their own, not merged into the base model: merging is a separate step via `llamafactory-cli export merge_config.yaml`, whose config takes `model_name_or_path`, `adapter_name_or_path`, `template`, `export_dir`, `export_size`, `export_device`, and `export_legacy_format`; the docs warn not to merge "a quantized model or specify quantization bits" during export [22]. An adapter directory alone is therefore not a full model - it is loaded together with the base model, and fine-tuned inference explicitly requires passing `model_name_or_path` + `adapter_name_or_path` + `finetuning_type` + `template` together (`llamafactory-cli chat inference_config.yaml`, or `infer_backend: vllm` for faster serving) [23].

## Find it in the docs

The primary docs live at `https://llamafactory.readthedocs.io/en/latest/<section>/<page>.html`, e.g. `getting_started/installation.html`, `advanced/arguments.html`, `advanced/distributed.html`, `advanced/merge.html`; fetched 2026-08-07 against the `latest` build, which tracks the default branch rather than a pinned release tag - the docs site itself carries no version-tag URL form, so a value read there can be ahead of whatever release you installed. The Arguments page (`advanced/arguments.html`) is the single lookup point for every YAML field, grouped by dataclass (`FinetuningArguments`, RLHF args, `VllmArguments`, `QuantizationArguments`) [8]. Question-to-page map: training-stage configs and RLHF setup -> `getting_started/trainers.html` [6]; SFT walkthrough -> `getting_started/sft.html` [17]; multi-GPU/multi-node/DeepSpeed/FSDP -> `advanced/distributed.html` [11]; checkpoint/merge/quantize -> `advanced/merge.html` [22]; inference/serving -> `getting_started/inference.html` [23]; experiment trackers -> `advanced/monitor.html` [19]. Runnable references beyond the docs: the repo's `examples/` tree ships ready YAML configs by stage and method (the quickstart's `examples/train_lora/qwen3_lora_sft.yaml` among them), and `examples/requirements/` carries feature-specific extra dependency files beyond the top-level `requirements/` directory [9][17]. A pinned FAQ is maintained as a GitHub issue rather than a docs page: issue #4614, "FAQs / 常见问题" [18], which is the source for the OOM checklist above; treat it as maintainer-curated but check its date (opened 2024-06-28) against your installed version before trusting a specific flag name.

The README does not claim GRPO support anywhere and instead points GRPO-seeking readers to a separate sister project: its one GRPO-related changelog entry announces "EasyR1," described as "an efficient, scalable and multi-modality RL training framework" for GRPO training, rather than a GRPO feature inside LlamaFactory itself [9]. The codebase confirms the omission directly: `FinetuningArguments.stage` is typed `Literal["pt", "sft", "rm", "ppo", "dpo", "kto"]` at commit 62ae362 [7], so a reader expecting GRPO (a currently common RLHF method in comparable frameworks) will not find it here and should look to EasyR1 or another framework that lists it explicitly. The v1 rewrite under `src/llamafactory/v1/` (opt-in via `USE_V1=1`) is documented by the repository's own contributor guide as experimental, and its trainers directory contains only SFT, DPO, and RM implementations - no PPO or KTO - so PPO/KTO runs must stay on the default v0 architecture [10].

## Sources

[1] hiyouga/LlamaFactory GitHub repository, former repository names hiyouga/LLaMA-Factory and hiyouga/LLaMA-Efficient-Tuning, as given in the sourcing shortlist row for this card and corroborated by the current repository URL redirecting from the older name. https://github.com/hiyouga/LlamaFactory. Fetched 2026-08-07.

[2] LlamaFactory ReadTheDocs index. https://llamafactory.readthedocs.io/en/latest/index.html. Fetched 2026-08-07.

[3] Zheng et al., "LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models," arXiv:2403.13372, ACL 2024 System Demonstration Track. https://arxiv.org/abs/2403.13372. Fetched 2026-08-07.

[4] hiyouga/LlamaFactory GitHub repository metadata (GitHub REST API, a live endpoint with no revision parameter). https://api.github.com/repos/hiyouga/LLaMA-Factory. Fetched 2026-08-10 (stargazers_count 73946, pushed_at 2026-08-09T08:00:11Z).

[5] LlamaFactory README, Features/CLI section, tag v0.9.5. https://raw.githubusercontent.com/hiyouga/LlamaFactory/v0.9.5/README.md. Fetched 2026-08-07.

[6] LlamaFactory ReadTheDocs, Trainers page (per-stage config examples). https://llamafactory.readthedocs.io/en/latest/getting_started/trainers.html. Fetched 2026-08-07.

[7] `src/llamafactory/hparams/finetuning_args.py`, `FinetuningArguments.stage` field, read at commit 62ae362455801d4900a5132c7a30b23dc5fc3802 (ahead of the v0.9.5 release tag; commit date confirmed via GitHub's commits API, `commit.committer.date` 2026-07-31T10:54:13Z). https://raw.githubusercontent.com/hiyouga/LlamaFactory/62ae362455801d4900a5132c7a30b23dc5fc3802/src/llamafactory/hparams/finetuning_args.py. Fetched 2026-08-07; commit date fetched 2026-08-10.

[8] LlamaFactory ReadTheDocs, Arguments reference. https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html. Fetched 2026-08-07.

[9] LlamaFactory README, Supported Training Approaches table and Installation section, tag v0.9.5. https://raw.githubusercontent.com/hiyouga/LlamaFactory/v0.9.5/README.md. Fetched 2026-08-07.

[10] `.ai/CLAUDE.md` contributor guide (v0/v1 architecture description, v1 experimental status), read via GitHub content API against the default branch. https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/.ai/CLAUDE.md. Fetched 2026-08-07. Trainer directory listings cross-checked via jsdelivr's data API at tag v0.9.5: https://data.jsdelivr.com/v1/packages/gh/hiyouga/LlamaFactory@v0.9.5.

[11] LlamaFactory ReadTheDocs, Distributed Training guide. https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html. Fetched 2026-08-07.

[12] `llamafactory` package on PyPI (JSON API). https://pypi.org/pypi/llamafactory/json. Fetched 2026-08-07.

[13] `pyproject.toml`, tag v0.9.5 (dependency pins, `requires-python`). https://raw.githubusercontent.com/hiyouga/LlamaFactory/v0.9.5/pyproject.toml. Fetched 2026-08-07.

[14] `requirements/vllm.txt`, tag v0.9.5. https://raw.githubusercontent.com/hiyouga/LlamaFactory/v0.9.5/requirements/vllm.txt. Fetched 2026-08-07.

[15] `requirements/deepspeed.txt`, tag v0.9.5. https://raw.githubusercontent.com/hiyouga/LlamaFactory/v0.9.5/requirements/deepspeed.txt. Fetched 2026-08-07.

[16] hiyouga/LlamaFactory GitHub Releases, v0.9.5 release notes (2026-05-30). https://api.github.com/repos/hiyouga/LLaMA-Factory/releases. Fetched 2026-08-07.

[17] LlamaFactory ReadTheDocs, SFT quickstart. https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html. Fetched 2026-08-07.

[18] hiyouga/LlamaFactory GitHub issue #4614, "FAQs / 常见问题," opened 2024-06-28. https://github.com/hiyouga/LlamaFactory/issues/4614. Fetched 2026-08-07.

[19] LlamaFactory ReadTheDocs, Monitor page (experiment tracker integrations). https://llamafactory.readthedocs.io/en/latest/advanced/monitor.html. Fetched 2026-08-07.

[20] `src/llamafactory/train/ppo/trainer.py`, tag v0.9.5 (log fields and TRL stats keys). https://raw.githubusercontent.com/hiyouga/LlamaFactory/v0.9.5/src/llamafactory/train/ppo/trainer.py. Fetched 2026-08-07.

[21] `src/llamafactory/train/dpo/trainer.py`, tag v0.9.5 (logged metric names). https://raw.githubusercontent.com/hiyouga/LlamaFactory/v0.9.5/src/llamafactory/train/dpo/trainer.py. Fetched 2026-08-07.

[22] LlamaFactory ReadTheDocs, Model Saving, LoRA Merging, and Quantization page. https://llamafactory.readthedocs.io/en/latest/advanced/merge.html. Fetched 2026-08-07.

[23] LlamaFactory ReadTheDocs, Inference page. https://llamafactory.readthedocs.io/en/latest/getting_started/inference.html. Fetched 2026-08-07.

