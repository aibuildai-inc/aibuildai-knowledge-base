# torchtune

PyTorch's native post-training library, now in maintenance mode: hackable single-file recipes and YAML configs for SFT, DPO, PPO, GRPO and distillation, with no new features being added.

**torchtune** "is a PyTorch library for easily authoring, post-training, and experimenting with LLMs", built around modular native-PyTorch model implementations, checkpoint-conversion utilities for interoperability with popular model hubs, and YAML-configured training recipes [1]. It was built by the PyTorch Team at Meta [2][3], and on 2025-07-15 the maintainers announced "we are stopping active development on torchtune, effective immediately" and that "no new features will be added to the library", redirecting effort to an unnamed-at-the-time successor project (a maintainer later confirmed the name; see Maintained by), while continuing critical bug and security fixes through the rest of 2025 [4]. The API is a `Recipe` class (training/inference/eval logic) driven by a YAML `Config` plus command-line overrides, launched through a custom `tune` CLI [5]. It lives at https://github.com/meta-pytorch/torchtune [2].

**When to pick it**: pick torchtune only for a PyTorch-native, single-file-recipe way to fine-tune Llama/Qwen/Gemma-family checkpoints on a fixed, working method set (SFT, DPO, PPO, GRPO, distillation, QAT) when you accept that the project has stopped receiving new features and is bound for a successor repo [4]; do not pick it if you need active development, a wide model/method roadmap, or first-class vLLM-backed generation for online RL - trl and verl are both under active development (cross-reference; not covered here) and trl's GRPO trainer optionally offloads generation to vLLM, a capability torchtune's GRPO recipe config does not expose (no vLLM keys appear in the sampled GRPO config; not exhaustively verified across all GRPO configs) [6].

**Methods it ships**, grouped as the recipe overview page groups them [7]: full-finetune and LoRA/QLoRA SFT (single-device and distributed, up to multi-node), knowledge distillation (LoRA/QLoRA only), DPO (full and LoRA/QLoRA), PPO (full-parameter, single-device only), GRPO (full-parameter, multi-device), and quantization-aware training (distributed). The support matrix in the README marks GRPO's single-device row as under construction [1]. GRPO and its Qwen/Llama example configs live only under `recipes/dev/` and `recipes/configs/dev/` at the current repository HEAD (commit bd2a0fc7, 2026-07-31) - they do not exist at the latest tagged release v0.6.1 (2025-04-07): the v0.6.1 `recipes/configs/dev/` directory holds no GRPO config, while the HEAD directory does [8][9]. The recipe overview page itself says its documentation "is currently in construction" and points readers to `tune ls` for the full, current recipe list rather than trusting any static summary [7].

**Scale it handles**: single device up to multi-node. Distributed recipes use FSDP2 (PyTorch's sharded data-parallel API) and launch through `tune run --nproc_per_node N <recipe> --config <config>`, which integrates with `torchrun` [1][10]. Multi-node is documented with a working example: a two-node, 8-GPU-per-node SLURM `sbatch` script that fine-tunes Llama-3.3 70B with full-parameter updates, with the caveat that slow inter-node bandwidth can bottleneck FSDP's all-gather/reduce-scatter collectives [10]. GRPO ships its own multi-node path, `sbatch multinode_grpo.sbatch` in `recipes/dev/` [8]. No multi-node throughput benchmark is published; the README's own single-node reference numbers (batch size 2, sequence length 2048, `compile` on) show Llama-3.1-8B full-finetune reaching 1650 tokens/sec at 18.9 GiB peak memory on one RTX 4090, LoRA reaching 3083 tokens/sec at 16.2 GiB on the same card, and Llama-3.1-405B QLoRA fitting in 44.8 GB peak memory per GPU on 8x A100 at 653 tokens/sec [1].

**Install**: `pip install torchtune`; latest tagged release is v0.6.1, published 2025-04-07 [11]; the package requires Python >= 3.9 per its v0.6.1 `pyproject.toml` [12], and the repository's GitHub-reported license is BSD-3-Clause [3]. torch, torchvision and torchao are NOT pinned as install dependencies - the docs instruct installing them separately first (`pip install torch torchvision torchao`, or a nightly build via the PyTorch nightly index) before `pip install torchtune` [13]. At the v0.6.1 tag, `pyproject.toml` pins `torchdata==0.11.0` as a hard equality pin, plus unpinned-floor dependencies (`datasets`, `huggingface_hub[hf_transfer]`, `safetensors`, `sentencepiece`, `tiktoken`, `blobfile>=2`, `omegaconf`, `Pillow>=9.4.0`, others); the `dev` extra adds `bitsandbytes>=0.43.0`, `wandb`, `comet_ml>=3.44.2`, `mlflow`, and pins `urllib3<2.0.0` [12]. Neither the installation page nor the v0.6.1 `pyproject.toml` states a CUDA or hardware minimum; the README states only that PyTorch nightlies are tested against CUDA 12.6 wheels for that install path [13][1].

**Maintained by**: the PyTorch Team at Meta, published under the `meta-pytorch` GitHub organization (formerly `pytorch/torchtune`) [2][3]. The maintainers' own 2025-07-15 announcement is the clearest public sign of the project's state: active feature development has stopped, only critical bug/security fixes continue through 2025, and Discord/GitHub issues stay open for support [4]. Maintainer `felipemello1` (repository contributor) confirmed on 2025-10-22 that `meta-pytorch/torchforge` is the successor project, linking its announcement blog post; the same issue thread kept receiving replies from other participants through at least 2026-03-13, the thread's most recent comment as read [14].

## Quick start

The first-finetune tutorial's complete flow, quoted at each step [5]:

```bash
tune download meta-llama/Llama-2-7b-hf \
    --output-dir /tmp/Llama-2-7b-hf \
    --hf-token <ACCESS TOKEN>

tune run lora_finetune_single_device --config llama2/7B_lora_single_device epochs=1
```

The same first-finetune tutorial's config-copy pattern for customizing further [5]:

```bash
tune cp llama2/7B_lora_single_device custom_config.yaml
```

A YAML config is edited in place or overridden inline with `key=value` pairs on the same `tune run` command line [5].

## Start it

- One GPU: `tune run <recipe> --config <config> [overrides...]`, e.g. `tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device` [5].
- Multiple GPUs on one node: `tune run --nproc_per_node N full_finetune_distributed --config llama3_1/8B_full`, which integrates with `torchrun` under the hood [1][10]. Config templates live under `recipes/configs/<model_family>/` by naming convention (`<size>_full.yaml`, `<size>_lora_single_device.yaml`, `<size>_lora_dpo_single_device.yaml`, etc.) [8].
- Multi-node: an `sbatch` SLURM script (`#SBATCH --nodes=2 --gpus-per-task=8`) that resolves the head-node IP and calls `tune run` per node from a shared filesystem install [10]; GRPO has its own template, `recipes/dev/multinode_grpo.sbatch`, launched via `sbatch multinode_grpo.sbatch` [8].
- Effective batch size is `batch_size x gradient_accumulation_steps` (no data-parallel multiplier is applied in the config itself); the sampled Llama-3.1-8B LoRA single-device config uses `batch_size: 2` and `gradient_accumulation_steps: 8` for an effective batch of 16 [15]. Gradient accumulation is explicitly incompatible with fusing the optimizer step into the backward pass (`optimizer_in_bwd`) - the full-finetune distributed config comments this directly and ships `optimizer_in_bwd: False` with `gradient_accumulation_steps: 1` by default [16][17].
- Config surface and defaults: every sampled recipe config (LoRA SFT, full SFT, LoRA DPO, GRPO) sets `dtype: bf16` - the docs recommend leaving precision at its default `bfloat16` whenever the hardware supports it, so this is a silent Ampere-or-newer GPU assumption [15][16][18][8][19]. `compile: False` (PyTorch compile) is off by default across the same sampled configs [15][16][8].
- Out-of-memory first aid, from the memory-optimization glossary [19]: activation checkpointing ("Use when you're memory constrained and want to use a larger model, batch size or context length" - trades compute for memory, slows training); activation offloading (similar, used alongside activation checkpointing); gradient accumulation ("Use it when you can already fit at least one sample without OOMing, but not enough of them" - incompatible with `optimizer_in_bwd`); lower-precision optimizers (reduces optimizer-state size, may reduce stability); fusing the optimizer step into the backward pass (needs a large batch, incompatible with gradient accumulation); offloading optimizer/gradient state to CPU ("significantly reduce GPU memory usage at the cost of CPU RAM and training speed" - prioritize last); LoRA (fewer trainable parameters, may reduce accuracy); QLoRA (quantizes the base model, saves "1.5 bytes * (# of model parameters)" at some speed/accuracy cost); DoRA (a LoRA variant, "may improve model performance at the cost of slightly more memory") [19]. No separate generation-side engine or OOM path exists for GRPO - it does not route generation through a separate serving engine such as vLLM in the sampled config [8].

## Watch it

This section covers the mechanics only - what a metric MEANS for a given method lives on that method's card, not here.

- **Enable it**: set the config's `metric_logger` component. Every sampled config (LoRA SFT, full SFT, LoRA DPO, GRPO) defaults to `torchtune.training.metric_logging.DiskLogger`, which writes locally under `${output_dir}/logs` and sends nothing to an external tracker unless you swap in `WandBLogger` or the Comet logger [15][16][8][20][21]. Logging cadence is `log_every_n_steps` (1 in the sampled configs) [15][8].
- **LoRA/full SFT metric names**, read directly from the `log_dict` calls in `lora_finetune_single_device.py` and `full_finetune_distributed.py` (both identical): `loss`, `lr`, `tokens_per_second_per_gpu`, plus memory stats when `log_peak_memory_stats: True`, plus `grad_norm` when gradient-norm clipping is enabled [22][23].
- **PPO metric names**, read from `ppo_full_finetune_single_device.py`'s `log_metrics()`: `scores`, `num_stop_tokens`, `rlhf_reward`, `kl`, `kl_reward`, `loss`, `policy_loss`, `value_loss`, `clipfrac`, `ratios`, `approx_policy_kl`, `response_lengths`, `tokens_per_second_per_gpu_trajectory`, `tokens_per_second_per_gpu_ppo`, plus memory stats when enabled [24].
- **DPO metric names**, read from `lora_dpo_single_device.py`: `loss`, `lr`, `tokens_per_second_per_gpu`, `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `log_probs/rejected`, `log_probs/chosen`, `logits/rejected`, `logits/chosen`, plus memory stats when enabled [25].
- **GRPO metric names**, read from `recipes/dev/grpo_full_finetune_distributed.py` at the current HEAD commit (bd2a0fc7, ahead of the v0.6.1 release - this recipe does not exist at v0.6.1): `rewards` and `successes` (both averaged across ranks), `num_stop_tokens`, `loss`, `policy_loss`, `kl_loss`, `clipfrac`, `ratios`, `approx_policy_kl`, `response_lengths`, plus memory stats when enabled; only the rank-zero process logs [26].
- None of the five inspected recipe scripts (LoRA SFT, full SFT, PPO, DPO, GRPO) calls a `_metric_logger` inside any per-sample generation path with the sample text itself - the docs read for this card do not document sample-level logging of generated completions [22][23][24][25][26].
- No recipe inspected exposes an in-training `eval_dataset` or evaluation-cadence field; evaluation is a separate step, run after training with a dedicated recipe (`recipes/eleuther_eval.py`) that wraps EleutherAI's LM Evaluation Harness [1][22][23][24][25][26].
- No published stopping-rule, threshold, or patience value was found: neither the Memory Optimization Overview (the runtime-tuning page) [19] nor the Weights & Biases logging deep-dive [20] nor the five recipe scripts read [22][23][24][25][26] name an early-stopping, patience, or reward-threshold field. Shapes are published (the metric names above); thresholds are not - this is a plain "none found," not a confirmed absence across the whole codebase.

## Save it

- Checkpoint directory layout, from the checkpointer deep-dive's own `tree` example of a LoRA run's `output_dir`: `epoch_{N}/` subdirectories each holding `adapter_config.json`, `adapter_model.pt`, `adapter_model.safetensors`, the merged full-model safetensors shards, `config.json`, tokenizer files, and a copy of the original checkpoint dir's lightweight metadata files; a `recipe_state/` directory holding `recipe_state.pt` "with the information necessary to restart your training run from the last intermediate epoch"; and a `logs/` directory with the metric logger's output [27].
- torchtune outputs BOTH the LoRA adapter weights and the full merged model weights inside every `epoch_{N}/` folder by default - the merged weights are "a convenience, since it can be used without having special tooling to handle the adapters" [27]. Setting `save_adapter_weights_only: True` saves only the adapter, reducing storage and save time, but the docs state plainly this "does not influence resuming_from_checkpointing" [27].
- Resume: set `resume_from_checkpoint: True`; for LoRA runs also set `adapter_checkpoint: epoch_{YOUR_EPOCH}/adapter_model.pt` so the recipe reloads the original untrained base weights from `checkpoint_dir` and the trained adapter from `output_dir` - the docs warn that loading the merged weights plus the adapter together "would be an error" [27].
- Checkpointer choice determines the on-disk FORMAT, not just the save call: `FullModelHFCheckpointer` reads and writes Hugging Face `transformers`-compatible checkpoints and is "the default format in every torchtune config"; `FullModelMetaCheckpointer` and a torchtune-native checkpointer exist for other formats (the docs page as read does not enumerate the native format's on-disk file names beyond the HF case) [27].
- Loader handoff: an `epoch_{N}/` HF-format directory is a full HF-loadable model directory by the docs' own instruction ("If running inference or pushing to a model hub, you should use this folder directly") [27]; an `epoch_{N}/adapter_model.*` pair alone is NOT a full model - it requires the original base checkpoint to reconstitute a usable model, per the resume-flow warning above [27].

## Find it in the docs

The docs are the live source; this section teaches the lookup pattern, not the content.

- Address pattern: `https://meta-pytorch.org/torchtune/<version>/<page>.html`. `<version>` accepts `main`, `stable`, or a short release-line tag like `0.6` - all three returned HTTP 200 when fetched directly (checked 2026-08-10) [28]. The historical `pytorch.org/torchtune/...` URLs (as printed in the README and PyPI metadata) 301-redirect to the `meta-pytorch.org` domain - the project's GitHub org and doc host both moved from `pytorch`/`pytorch.org` to `meta-pytorch`/`meta-pytorch.org` [2][1].
- Page-slug recipes: top-level pages are flat slugs (`overview`, `install`, `api_ref_config` etc.); tutorials live under `tutorials/<slug>` (`first_finetune_tutorial`, `e2e_flow`, `memory_optimizations`, `multinode`); recipe docs live under `recipes/<slug>` (`recipes_overview`, `dpo`); deep-dives live under `deep_dives/<slug>` (`checkpointer`, `wandb_logging`, `comet_logging`, `configs`, `recipe_deepdive`) [1][7][5][10][19][27][20].
- Question-to-slug map: "what launches distributed?" -> `tutorials/multinode` and `recipes/recipes_overview`; "what format is my checkpoint?" -> `deep_dives/checkpointer`; "how do I log to a tracker?" -> `deep_dives/wandb_logging` or `deep_dives/comet_logging`; "how do configs work?" -> `deep_dives/configs`; "what does `tune` do?" -> `tune_cli`, whose own `--help` output lists the full subcommand set: `download`, `ls`, `cp`, `run`, `validate`, `cat` [29].
- Runnable references beyond the docs: `tune ls` lists every built-in recipe/config pair currently in the package, which the recipe-overview page itself recommends over trusting a static list because "our recipe documentation is currently in construction" [7]. The README's Quick Start commands and its optimization-flags table are runnable as written against Llama-3.1 [1].
- Beyond the official docs: the README's own Community section lists integrations the project maintains ties with - Hugging Face `PEFT` for adapter interoperability, EleutherAI's LM Evaluation Harness for evaluation, Hugging Face Datasets for data loading, and Weights & Biases as a named logging collaborator - but the docs read for this card surface no dedicated curated community-tutorials page (unlike trl's `community_tutorials` slug) [1]. No official MCP endpoint for querying these docs was found in the pages read.
- The maintainers' own successor-project announcement is the honest boundary to read before starting anything long-running here: no new features will land in torchtune [4], and its named successor, `meta-pytorch/torchforge`, was confirmed by a maintainer on 2025-10-22 in the same issue thread, which was still receiving replies as of 2026-03-13, the thread's most recent comment as read [14].

## Sources

All pages are fetched at the dates given below; docs pages are unpinned `main`/`stable`/`0.6` builds unless a commit or release tag is named. Source-code claims are pinned either to the v0.6.1 release tag (commit a6290a5b) or, where explicitly marked, to the current repository HEAD (commit bd2a0fc7, 2026-07-31 push) which is ahead of that release. Method names (SFT, DPO, PPO, GRPO, QAT) are deliberately cited to nothing here; their defining papers live on the methodology cards.

[1] torchtune README at commit bd2a0fc7. https://raw.githubusercontent.com/meta-pytorch/torchtune/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/README.md. Fetched 2026-08-10.

[2] torchtune GitHub repository. https://github.com/meta-pytorch/torchtune. Fetched 2026-08-10.

[3] torchtune GitHub API repo metadata (owner org, license, push date). https://api.github.com/repos/meta-pytorch/torchtune. Fetched 2026-08-10.

[4] GitHub issue #2883, "[IMPORTANT] The future of torchtune," opened by maintainer ebsmothers. https://github.com/meta-pytorch/torchtune/issues/2883. Fetched 2026-08-10.

[5] torchtune "Fine-Tune Your First LLM" tutorial. https://meta-pytorch.org/torchtune/stable/tutorials/first_finetune_tutorial.html. Fetched 2026-08-10.

[6] `recipes/configs/dev/3B_full_grpo.yaml` at commit bd2a0fc7 (no vLLM-related config keys). https://raw.githubusercontent.com/meta-pytorch/torchtune/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/recipes/configs/dev/3B_full_grpo.yaml. Fetched 2026-08-10.

[7] torchtune Recipes Overview page. https://meta-pytorch.org/torchtune/stable/recipes/recipes_overview.html. Fetched 2026-08-10.

[8] GitHub API contents listing, `recipes/configs/dev/` and `recipes/dev/` at commit bd2a0fc7 (GRPO configs and `multinode_grpo.sbatch` exist here). https://api.github.com/repos/meta-pytorch/torchtune/contents/recipes/configs/dev?ref=bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1 and .../recipes/dev?ref=bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1. Fetched 2026-08-10.

[9] GitHub API contents listing, `recipes/configs/dev/` at tag v0.6.1 (no GRPO config present). https://api.github.com/repos/meta-pytorch/torchtune/contents/recipes/configs/dev?ref=v0.6.1. Fetched 2026-08-10.

[10] torchtune Multi-node finetuning tutorial. https://meta-pytorch.org/torchtune/stable/tutorials/multinode.html. Fetched 2026-08-10.

[11] torchtune GitHub API releases list (latest: v0.6.1, published 2025-04-07). https://api.github.com/repos/meta-pytorch/torchtune/releases. Fetched 2026-08-10.

[12] `pyproject.toml` at the v0.6.1 tag (commit a6290a5b). https://raw.githubusercontent.com/meta-pytorch/torchtune/v0.6.1/pyproject.toml. Fetched 2026-08-10.

[13] torchtune Installation page. https://meta-pytorch.org/torchtune/stable/install.html. Fetched 2026-08-10.

[14] Comments on GitHub issue #2883 (maintainer felipemello1, CONTRIBUTOR association, confirms `meta-pytorch/torchforge` as successor on 2025-10-22; thread continues with replies from other participants through 2026-03-13). https://api.github.com/repos/meta-pytorch/torchtune/issues/2883/comments. Fetched 2026-08-10.

[15] `recipes/configs/llama3_1/8B_lora_single_device.yaml` at the v0.6.1 tag. https://raw.githubusercontent.com/meta-pytorch/torchtune/v0.6.1/recipes/configs/llama3_1/8B_lora_single_device.yaml. Fetched 2026-08-10.

[16] `recipes/configs/llama3_1/8B_full.yaml` at the v0.6.1 tag. https://raw.githubusercontent.com/meta-pytorch/torchtune/v0.6.1/recipes/configs/llama3_1/8B_full.yaml. Fetched 2026-08-10.

[17] `recipes/full_finetune_distributed.py` at the v0.6.1 tag (`optimizer_in_bwd`/`gradient_accumulation_steps` incompatibility comment and log_dict call). https://raw.githubusercontent.com/meta-pytorch/torchtune/v0.6.1/recipes/full_finetune_distributed.py. Fetched 2026-08-10.

[18] `recipes/configs/llama3_1/8B_lora_dpo_single_device.yaml` at the v0.6.1 tag. https://raw.githubusercontent.com/meta-pytorch/torchtune/v0.6.1/recipes/configs/llama3_1/8B_lora_dpo_single_device.yaml. Fetched 2026-08-10.

[19] torchtune Memory Optimization Overview tutorial. https://meta-pytorch.org/torchtune/stable/tutorials/memory_optimizations.html. Fetched 2026-08-10.

[20] torchtune "Logging to Weights & Biases" deep-dive. https://meta-pytorch.org/torchtune/stable/deep_dives/wandb_logging.html. Fetched 2026-08-10.

[21] torchtune "Logging to Comet" deep-dive. https://meta-pytorch.org/torchtune/stable/deep_dives/comet_logging.html. Fetched 2026-08-10.

[22] `recipes/lora_finetune_single_device.py` at the v0.6.1 tag (log_dict call). https://raw.githubusercontent.com/meta-pytorch/torchtune/v0.6.1/recipes/lora_finetune_single_device.py. Fetched 2026-08-10.

[23] `recipes/full_finetune_distributed.py` at the v0.6.1 tag (log_dict call, identical fields to [22]). Same URL as [17]. Fetched 2026-08-10.

[24] `recipes/ppo_full_finetune_single_device.py` at the v0.6.1 tag (`log_metrics()` method). https://raw.githubusercontent.com/meta-pytorch/torchtune/v0.6.1/recipes/ppo_full_finetune_single_device.py. Fetched 2026-08-10.

[25] `recipes/lora_dpo_single_device.py` at the v0.6.1 tag (log_dict call). https://raw.githubusercontent.com/meta-pytorch/torchtune/v0.6.1/recipes/lora_dpo_single_device.py. Fetched 2026-08-10.

[26] `recipes/dev/grpo_full_finetune_distributed.py` at commit bd2a0fc7 (this recipe does not exist at v0.6.1). https://raw.githubusercontent.com/meta-pytorch/torchtune/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/recipes/dev/grpo_full_finetune_distributed.py. Fetched 2026-08-10.

[27] torchtune "Checkpointing in torchtune" deep-dive (directory tree example, checkpointer classes, save_adapter_weights_only contract, resume flow). https://meta-pytorch.org/torchtune/stable/deep_dives/checkpointer.html. Fetched 2026-08-10.

[28] Direct fetch confirming `meta-pytorch.org/torchtune/{0.6,main,stable}/index.html` all return HTTP 200. Fetched 2026-08-10.

[29] torchtune CLI reference page (`tune --help` subcommand list). https://meta-pytorch.org/torchtune/stable/tune_cli.html. Fetched 2026-08-10.
