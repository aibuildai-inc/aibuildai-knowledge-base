# trl

Hugging Face's post-training library: one trainer class per method, native to the Hub - start on one machine, scale out through Accelerate.

**trl** (Transformers Reinforcement Learning) is "a full stack library where we provide a set of tools to train transformer language models with methods like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), Reward Modeling, and more", integrated with the transformers library [1]. It is built and maintained by Hugging Face [2][3], and it is native to the Hub - Hugging Face's hosted platform for models and datasets: each method is a trainer class (`SFTTrainer`, `GRPOTrainer`, ...) that takes a model id, a Hub dataset, and method-specific inputs, and runs `.train()` [4]. It lives at https://github.com/huggingface/trl [3].

**When to pick it**: post-training in the Hugging Face ecosystem, when your models and datasets live on the Hub and you want a trainer-class API that starts on one machine - a wide method menu (16 trainers [1]) - and scales out through Accelerate, Hugging Face's launcher layer over torch distributed, when one machine stops being enough [5]. For cluster-scale online RL where Ray (a cluster scheduler) places the training and generation workers, weigh the verl card instead (cross-reference; not covered here).

**Methods it ships** [1] (each method's definition, math, method-level knobs, and training-signal semantics live on its methodology card, which also carries the defining papers - this card does not restate them):
- Online: GRPO, RLOO, Online DPO, Nash-MD, PPO, XPO
- Reward modeling: Reward, PRM
- Offline: SFT, DPO, KTO, BCO, CPO, ORPO
- Distillation: GKD, MiniLLM
- The docs mark ten of the sixteen as experimental - PPO among them, so the parent method is not the stable path here; the stable online trainers are GRPO and RLOO - and five as vLLM-capable for generation (GRPO, RLOO, Online DPO, Nash-MD, XPO) [1]. The taxonomy moves: as of this reading the What's-New note on the same index announces two changes - KTOTrainer graduating to the stable API, and GRPOTrainer gaining per-example environment selection with environment-owned rewards (Harbor and OpenEnv integrations) - so recheck [1] at run time. Experimental trainers import from `trl.experimental.<method>` (the docs give `trl.experimental.ppo.PPOTrainer`), not from the top-level package like the stable ones [6]. The PPOTrainer page itself carries a reference run showing the method works despite its experimental status: on the TL;DR summarization task, a PPO checkpoint reaches a 64.7% preferred rate against the underlying SFT checkpoint's 33.0%, judged by GPT-4o mini [6].

**Scale it handles**: single GPU up to multi-node, all launched through Accelerate (`accelerate config`, then `accelerate launch train.py`), with one model per GPU as the base pattern; DeepSpeed for sharding when one model per GPU does not fit [5]. Sequence-parallel options exist for long-context training - Context Parallelism, implemented as Ring Attention on the FSDP2 backend, and Sequence Parallelism, implemented as ALST/Ulysses on the DeepSpeed backend, each splitting sequences (not model weights) across GPUs and configured through Accelerate's `ParallelismConfig` with `cp_size`/`sp_size` - but the page carries an explicit "Section under construction. Feel free to contribute!" notice, so read [5] at run time instead of trusting a summary; the docs read for this card do not describe FSDP2 as a general model-sharding alternative to DeepSpeed outside this sequence-splitting context. Multi-node is documented as mechanism (the `accelerate config` prompts cover multi-GPU/multi-node setups; ready templates sit under `examples/accelerate_configs/`); [5] shows no multi-node benchmark [5].

**Install**: `pip install trl`; version 1.9.2 released 2026-07-28; Python >= 3.10; Apache-2.0 [2]. Version floors at the v1.9.2 tag (resolved to commit `33f9e462728b98f7f91d38b99328e81adde2faa0`): `transformers>=4.56.2`, `accelerate>=1.4.0`, `datasets>=4.7.0`; torch arrives transitively, unpinned [7]. vLLM generation is an extra - `pip install trl[vllm]` - pinned to `vllm>=0.17.0,<=0.25.1`; also `deepspeed>=0.14.4` and `peft>=0.8.0` as extras [7]. Hardware floors are undocumented: neither the installation page nor the package metadata states CUDA, torch, or platform minimums - requirements flow through transformers and accelerate [8][7]. Note on the commit this card's shortlist row names (`aa592026c7267a52a60ce751a28e90bc303531fe`, pushed 2026-07-31): that is the repository's newest push at screening time, three days ahead of the v1.9.2 release this Install field reports, so it is not the version `pip install trl` delivers - the source-code claims above that carry a commit are read at the release tag, not at that row commit.

**Maintained by**: Hugging Face; about 19k GitHub stars, not a ranking signal [3]. Actively developed as of this reading: the docs index's own What's-New section announces KTO graduating to the stable API and GRPO gaining multi-environment agentic RL support in the same update [1].

## Quick start

Smallest working runs, from the quickstart page [4] - each is a complete program:

```python
from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

```python
from trl import GRPOTrainer
from datasets import load_dataset
from trl.rewards import accuracy_reward

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=load_dataset("trl-lib/DeepMath-103K", split="train"),
    reward_funcs=accuracy_reward,
)
trainer.train()
```

A CLI covers several trainers: `trl sft`, `trl dpo`, `trl reward`, each taking `--model_name_or_path` and `--dataset_name` [4].

## Start it

- One process, one GPU is the base form: the scripts above, as-is.
- More than one GPU goes through Accelerate: run `accelerate config` once and answer for your layout, then launch the same script with `accelerate launch train.py`; ready-made config templates (e.g. multi-GPU) sit in the repo's `examples/accelerate_configs/` and are passed as `accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml train.py <SCRIPT_ARGS>` [5].
- The effective batch is per-device batch x number of devices x gradient accumulation steps; when you scale device count, retune `per_device_train_batch_size` and `gradient_accumulation_steps` together to keep it constant [5].
- GRPO adds a generation-layout choice at start time: with `use_vllm: True`, `vllm_mode` is either `"colocate"` (vLLM shares the training GPUs, the default) or `"server"` (a separate `trl vllm-serve` process on its own GPUs); `vllm_gpu_memory_utilization` sets the generation share of a colocated card, and Hugging Face publishes a Space to recommend a value for a given model and setup [9].
- Configuration is the method's Config class (`SFTConfig`, `GRPOConfig`, ...), a transformers `TrainingArguments` subclass whose trl-changed defaults apply the moment you start: for `SFTConfig`, `logging_steps` defaults to 10 instead of 500, `gradient_checkpointing` defaults to `True` instead of `False`, `bf16` defaults to `True` if `fp16` is unset instead of `False` (a bf16-capable GPU is the silent assumption), and `learning_rate` defaults to `2e-5` instead of `5e-5` [19].
- Out-of-memory first aid: `per_device_train_batch_size=1` with `gradient_accumulation_steps=8` [4]. When generation runs through vLLM, generation-side OOM has its own knob path: lower `vllm_gpu_memory_utilization` (with a small buffer added to the Space's recommendation for stability), and "If you still find you are getting out-of-memory errors set `vllm_enable_sleep_mode` to True and the vllm parameters and cache will be offloaded during the optimization step" [9].

## Watch it

This section is the mechanics: which signals trl emits and how to turn them on. What a signal MEANS for a method - which shapes are healthy, what to tune when they are not - lives on that method's methodology card. trl's docs open their logging page with the reason to care: as reinforcement learning algorithms are historically hard to debug, careful logging matters [11].

- **Enable it**: by default, PPOTrainer and GRPOTrainer "save a lot of relevant information to supported experiment trackers like Trackio, Weights & Biases (wandb) or TensorBoard" [11] - pass `report_to` on the Config (`"trackio"`, `"wandb"`, or `"tensorboard"`) to turn it on; without it, a run keeps no experiment-tracker record. Logging cadence is `logging_steps`, default 10 [19].
- **SFT metric names** (from the SFTTrainer page's Logged metrics section [10]): `global_step`, `epoch`, `num_tokens`, `loss`, `entropy`, `mean_token_accuracy`, `learning_rate`, `grad_norm`.
- **GRPO metric names**: read BOTH the `grpo_trainer` page and the GRPO half of the `logging` page - the two have drifted apart. The trainer page's Logged metrics section [9] lists a wider set than the logging page's GRPO Logging section [11]: `num_tokens`, `step_time`; the `completions/*` family (mean/min/max length, the `*_terminated_length` trio, `clipped_ratio`); the `tools/*` pair (`call_frequency`, `failure_frequency`) when tool use is enabled; per-function `rewards/<name>/mean|std`, combined `reward` and `reward_std`, `frac_reward_zero_std`; the `sampling/*` pair (`sampling_logp_difference`, `importance_sampling_ratio`, each with min/mean/max) logged only when `use_vllm=True` and `vllm_importance_sampling_correction=True`; `policy_loss`, `entropy`, `entropy_coef` (logged when `entropy_coef` is nonzero or `use_adaptive_entropy=True`); `aux_loss` for MoE models; `kl` (only when `beta` is non-zero); and the `clip_ratio/*` family (`region_mean`, `low_mean`, `low_min`, `high_mean`, `high_max`) [9]. The logging page's GRPO section instead names a single fused-kernel `clip_ratio` gated on `use_liger_loss: True` [11], while the trainer page gates the same collapse on `use_liger_kernel=True` [9] - the two pages disagree on the flag's name, one more sign they have drifted; `step_time`, `frac_reward_zero_std`, `policy_loss`, and `entropy_coef` appear only on the trainer page, not on the logging page [9][11]. One trl-specific trap: `use_adaptive_entropy=True` with the default `entropy_target=0.2` protects nothing - typical per-token entropies are 2-10 nats, and the docs state the default "almost never triggers regularization"; set it near the entropy you observe early in training [9].
- **PPO logs a different set** under different names (`objective/kl`, `objective/entropy`, `objective/non_score_reward`, `eps`, `lr`, `episode`, ...) with its own "Crucial values" list (`objective/scores`, `objective/rlhf_reward`, `objective/non_score_reward`) on the same logging page [11].
- **Sample the generations too**: scalars miss degenerate text. `GRPOConfig` carries `log_completions` (default `False`, logs a sample of prompt/completion pairs every `logging_steps` steps to the console via `rich` and/or to wandb/trackio) with `num_completions_to_print`, `log_unique_prompts`, and `log_completions_hub_repo` (pushes the completions to a named Hub repo) [9].
- **Evaluate during training**: both trainers take an `eval_dataset`, which must meet the same shape requirements as `train_dataset` [9][10]; both Config parameter listings name `eval_strategy` (default `"no"`) and `eval_steps` (default `None`), inherited from transformers `TrainingArguments`, plus `per_device_eval_batch_size` (default 8) [9][10]. GRPO adds `num_generations_eval`: "Number of generations to sample during evaluation. This allows using fewer generations during evaluation to save computation. If `None`, uses the value of `num_generations`" [9].
- **Stopping**: trl publishes no RL-specific stopping rule, threshold, or patience value, and here is the search that says so, run 2026-08-10 over the three pages that would carry one: the GRPO trainer page's only threshold-named field is `off_policy_mask_threshold`, which masks off-policy samples (from the DeepSeek-V3.2 paper's delta term) and stops nothing [9]; the SFT trainer page [10] and the logging guide [11] return no match at all for an early-stopping, patience, stopping-rule, or threshold field. The generic transformers callback surface does publish one - `EarlyStoppingCallback`, with `early_stopping_patience` and `early_stopping_threshold` fields that watch a metric via `metric_for_best_model` [18] - but it knows nothing about a reward curve, so it is not an RL stopping rule and is not covered here.

## Save it

- Checkpoints land under the Config's `output_dir` as numbered `checkpoint-<global_step>/` directories, and such a directory is directly loadable - the SFT page's own example loads `"Qwen3-0.6B-Instruct/checkpoint-5000"` into a `transformers` pipeline [10]. (The exact file inventory inside a checkpoint is not enumerated by the docs read for this card; inspect one on disk.)
- Cadence and retention are Config fields (present on every method's Config because it subclasses `TrainingArguments`): `save_strategy` (default `"steps"`), `save_steps` (default 500), `save_total_limit` [12]. **`save_only_model` trades resumability for size**: "Save only model weights, not optimizer/scheduler/RNG state. Significantly reduces checkpoint size but prevents resuming training from the checkpoint. ... You can only load the model using `from_pretrained` with this option set to `True`" [12].
- `trainer.save_model(output_dir)` will save the model "so you can reload it using `from_pretrained()`", saving only from the main process; `trainer.push_to_hub()` uploads it [10].
- Resume: `trainer.train(resume_from_checkpoint=True)` loads the last checkpoint in `output_dir`; passing a path string resumes from that specific checkpoint, with model, optimizer, and scheduler states loaded from it [10]. GRPO's adaptive entropy is checkpoint-safe: "the current entropy coefficient `entropy_coef` is saved alongside each checkpoint and restored on resume, so training is fully resumable" [9].
- PEFT changes what a save IS. PEFT (parameter-efficient fine-tuning, e.g. LoRA) trains a small adapter on top of a frozen base model; with a `peft_config`, saving via `trainer.save_model()` or `model.save_pretrained()` "saves only the adapter weights (~few MB) rather than the full model (~several GB)" [13] - on disk that is `adapter_config.json` plus a several-MB `adapter_model.safetensors` file [14], NOT a full model directory, and a plain `from_pretrained()` on it will not give you the trained model: reload pairs it with the base model (`PeftModel.from_pretrained(base_model, "path/to/adapters")`, optionally `.merge_and_unload()` to fold the adapter into a single full model for inference) or `AutoPeftModelForCausalLM.from_pretrained` [13][14].
- Whether the EVALUATOR can load what you saved is the loader's contract, not trl's - the PEFT adapter case above is exactly where that contract bites: a directory with only `adapter_config.json`/`adapter_model.safetensors` needs the base model at load time.

## Find it in the docs

The docs are the live source; this card teaches the lookup, it does not mirror them.

- Address pattern: `https://huggingface.co/docs/trl/<version>/en/<page>`. `<version>` is `main` (development docs, the version actually fetched for this card) or a release tag in the form `v<X.Y.Z>` - the index page's own version selector lists `v1.9.2` down through `v0.1.1` [1]. `<page>` is a snake_case slug. THE key recipe: every trainer's page is `<method>_trainer` - `grpo_trainer`, `dpo_trainer`, `sft_trainer`, `rloo_trainer`, `ppo_trainer`, and so on [1].
- A trainer page carries the method's full trl story in one place: usage, the Config class with every documented default, and the loss-variant notes. For a config key, open the trainer page and search within it.
- Question-to-slug map [1]: dataset shapes -> `dataset_formats` - that page holds the per-trainer expected-type table (DPOTrainer and RewardTrainer and KTOTrainer/BCOTrainer take preference-shaped data; GRPOTrainer and RLOOTrainer and the experimental NashMDTrainer/OnlineDPOTrainer take prompt-only data; SFTTrainer takes language-modeling or prompt-completion data; the experimental GKDTrainer takes prompt-completion data) [15], the standard-vs-conversational format axis, and ready converters such as `extract_prompt` for turning an implicit-prompt preference dataset into a prompt-completion one [15]. Scaling -> `distributing_training`; memory -> `reducing_memory_usage` (named in [5]'s own cross-links). The index [1] groups everything else into guides / integrations / API; navigate by the slugs above.
- Runnable references beyond the docs: the `examples/` tree of the GitHub repo, including ready Accelerate config templates (`examples/accelerate_configs/`) [5][3]. The quickstart datasets come from `trl-lib`, the TRL organization on the Hub - known-good smoke-test data [4].
- Beyond the official docs - the community layer, curated door first: trl itself curates community tutorials at the `community_tutorials` slug - a table of task / trainer class / author / link, spanning GRPO (Philipp Schmid's Mini-R1 reproduction of the DeepSeek-R1 "aha moment"; a LLaMA-3.1-8B GRPO-with-Unsloth run by Andrea Manzoni; "How to fine-tune open LLMs in 2025 with Hugging Face" also by Schmid), SFT (QLoRA on Gemma by Schmid), DPO (Maxime Labonne's Mistral-7B alignment) and ORPO (Labonne's Llama 3) [16]. Two practitioner-blog domains recur across that curation and keep publishing trl guides: `philschmid.de` and `mlabonne.github.io` [16]. The rule for this layer: prefer what the official page curates, and check a post's date and pinned trl version against yours - community posts age faster than docs, and the curation page itself carries a live deprecation warning on Mayurji's "How to fine-tune a smol-LM with Hugging Face, TRL, and the smoltalk Dataset" tutorial, flagging its now-superseded `SFTTrainer(..., tokenizer=...)` and `setup_chat_format(...)` calls [16].
- MCP (Model Context Protocol - a tool interface an agent connects to): Hugging Face's official MCP server, endpoint `https://huggingface.co/mcp`, ships a built-in `hf_fs` tool that "lets your assistant efficiently navigate the Hub, including semantic searches of Documents and Spaces" - natural-language search over the Hub, including these docs; extra tools (repo creation, sandboxes, job management) are opt-in from the settings page [17]. Without MCP, the URL-pattern recipe above needs no search engine at all.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. All pages are `main`-version docs or live pages unless a tag is named; every statement above is a 2026-08-10 reading, except the stopping-rule search, which is also a 2026-08-10 reading of [9], [10], [11], and [18]. Method names (SFT, DPO, GRPO, ...) are deliberately cited to nothing here: their defining papers live on the methodology cards. Ecosystem tools named in passing (Accelerate, DeepSpeed, FSDP2, Ring Attention, ALST/Ulysses, Liger Kernel, vLLM, Harbor, OpenEnv) are reached through [1] and [5]'s own links and are deliberately not enumerated as references; Ray is named only inside the verl cross-reference; PEFT's own docs are referenced because the save contract depends on them.

[1] trl documentation index. https://huggingface.co/docs/trl/main/en/index. Fetched 2026-08-10.

[2] trl on PyPI. https://pypi.org/project/trl/. Fetched 2026-08-10.

[3] trl GitHub repository. https://github.com/huggingface/trl. Fetched 2026-08-10 (via the GitHub API, https://api.github.com/repos/huggingface/trl).

[4] trl quickstart. https://huggingface.co/docs/trl/main/en/quickstart. Fetched 2026-08-10.

[5] trl distributed-training guide. https://huggingface.co/docs/trl/main/en/distributing_training. Fetched 2026-08-10.

[6] trl PPOTrainer docs (the experimental import path). https://huggingface.co/docs/trl/main/en/ppo_trainer. Fetched 2026-08-10.

[7] trl build metadata at the v1.9.2 tag (dependency floors and extras), resolved via the GitHub API to commit 33f9e462728b98f7f91d38b99328e81adde2faa0. https://raw.githubusercontent.com/huggingface/trl/v1.9.2/pyproject.toml. Fetched 2026-08-10.

[8] trl installation page. https://huggingface.co/docs/trl/main/en/installation. Fetched 2026-08-10.

[9] trl GRPOTrainer docs (vLLM layout and memory knobs, logged-metrics list, adaptive-entropy target warning, completion logging, eval fields, checkpoint fields, adaptive-entropy resume note, off_policy_mask_threshold). https://huggingface.co/docs/trl/main/en/grpo_trainer. Fetched 2026-08-10.

[10] trl SFTTrainer docs (the TrainingArguments-subclass logged metrics, eval fields, checkpoint-directory example, save and reload and resume surface). https://huggingface.co/docs/trl/main/en/sft_trainer. Fetched 2026-08-10.

[11] trl logging guide (PPO and GRPO logged metrics and their crucial-values lists, default tracker behavior). https://huggingface.co/docs/trl/main/en/logging. Fetched 2026-08-10.

[12] transformers Trainer / TrainingArguments docs (save_strategy/save_steps/save_total_limit defaults, the save_only_model contract). https://huggingface.co/docs/transformers/main/en/main_classes/trainer. Fetched 2026-08-10.

[13] trl PEFT integration guide (adapter-only saves, reload and merge calls). https://huggingface.co/docs/trl/main/en/peft_integration. Fetched 2026-08-10.

[14] peft quicktour (adapter file names; AutoPeftModel reload). https://huggingface.co/docs/peft/main/en/quicktour. Fetched 2026-08-10.

[15] trl dataset formats guide. https://huggingface.co/docs/trl/main/en/dataset_formats. Fetched 2026-08-10.

[16] trl community tutorials page (official curation of community content: authors, trainers, links, deprecation warning). https://huggingface.co/docs/trl/main/en/community_tutorials. Fetched 2026-08-10.

[17] Hugging Face MCP server docs. https://huggingface.co/docs/hub/en/hf-mcp-server (the endpoint itself, https://huggingface.co/mcp, renders only its settings page in a browser). Fetched 2026-08-10.

[18] transformers callbacks docs (`EarlyStoppingCallback` and its `early_stopping_patience` and `early_stopping_threshold` fields). https://huggingface.co/docs/transformers/main/en/main_classes/callback. Fetched 2026-08-10.

[19] trl `sft_config.py` source at the v1.9.2 tag (commit 33f9e462728b98f7f91d38b99328e81adde2faa0) documenting the trl-changed TrainingArguments defaults (logging_steps, gradient_checkpointing, bf16, learning_rate). https://raw.githubusercontent.com/huggingface/trl/v1.9.2/trl/trainer/sft_config.py. Fetched 2026-08-10.
