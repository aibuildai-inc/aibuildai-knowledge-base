# axolotl

Item home: https://github.com/axolotl-ai-cloud/axolotl

A YAML-config-driven post-training framework: pick a base model and a method in one file, and `axolotl train config.yml` runs SFT, preference tuning, GRPO, EBFT, or reward modeling without writing training code.

Axolotl "is a free and open-source tool designed to streamline post-training and fine-tuning for the latest large language models (LLMs)" [1]. It is built and maintained by Axolotl AI (GitHub org `axolotl-ai-cloud`, formerly OpenAccess-AI-Collective) [2][3]. Its API surface is a single YAML config passed to a CLI: `base_model`, an optional `rl:` key selecting a preference/RL method (or omitted for SFT), an `adapter` key for LoRA/QLoRA, and a `datasets` list, all consumed by `axolotl train config.yml` [4][5]; its GRPO implementation specifically is documented as building on trl and adding async generation on top of it, rather than replacing trl's trainer [6].

**When to pick it**: you want one YAML file to drive the whole pipeline - model choice, method choice (SFT, DPO/IPO/KTO/SimPO/ORPO, GRPO, EBFT, or reward/PRM training), sharding strategy, and dataset format - without writing a Python training script, and you want a decision guide built into the docs to choose among those methods [5]. The RL paths set the online generation layer through vLLM rather than an in-process choice, similar to trl's vLLM-backed GRPO trainer (cross-reference; not covered here) [6]. Weigh the trl card if you want to write Python training loops directly against Hub datasets, or the verl card if you need Ray-scheduled multi-node RL (cross-reference; not covered here).

**Methods it ships** [5][6][7][8]: SFT (`rl` omitted); preference tuning DPO, IPO, KTO, SimPO, ORPO (`rl: dpo|ipo|kto|simpo|orpo`); GRPO for verifiable-reward RL (`rl: grpo`); EBFT, "Energy-Based Fine-Tuning", which rewards generated text by cosine similarity of internal hidden-state features to ground-truth completions instead of an external reward function or model (`rl: ebft`) [8]; and reward modeling, either outcome-level (`reward_model: true`, `model_type: AutoModelForSequenceClassification`) or process-level/PRM (`process_reward_model: true`, `model_type: AutoModelForTokenClassification`) [7]. The docs' own decision guide names GRPO and reward modeling as the RL/reward paths and states the method taxonomy is chosen via a decision tree on that page, which may be revised - recheck the live page rather than this list for new methods [5].

**Scale it handles**: single GPU up to multi-node. Three mutually exclusive sharding strategies - DeepSpeed (ZeRO stages 1-3, config JSON files fetched via `axolotl fetch deepspeed_configs`), FSDP (PyTorch native, the docs mark it "Recommended", with a documented FSDP1-to-FSDP2 field migration table), and DDP as the default when neither is set - each optionally layered with sequence parallelism [9]. Multi-node is documented with three launch mechanisms and example configs: Accelerate (a preset `accelerate` YAML with `machine_rank`/`num_machines`/`num_processes`), the new `axolotl train config.yaml --launcher torchrun -- --nnodes ... --rdzv_backend c10d ...` form, or legacy direct `torchrun -m axolotl.cli.train config.yaml`, plus a separate Ray Train guide (not read for this card) [10]. The multi-node guide shows configuration mechanism only, no published multi-node throughput or scaling benchmark [10].

**Install**: `uv pip install --no-build-isolation axolotl[deepspeed]` (uv is the documented package manager); PyPI lists version 0.18.0 uploaded 2026-07-17 [11]; per the GitHub repository API's machine-readable `license.spdx_id`, the project is Apache-2.0 licensed (the release's own `pyproject.toml` has a `license = "Apache-2.0"` line, but it is commented out, so the API rather than the file is the source of record here) [12]; the release's `pyproject.toml`, read at the v0.18.0 tag (commit `2f5cb9da62a0fe763a1ddeb7798fc9acb2f4a417`, resolved via the GitHub git-refs API [13]), sets `requires-python = ">=3.10"` [14] and pins `torch>=2.11.0,<=2.12.1`, `transformers==5.14.1`, `accelerate==1.13.0`, `trl==1.8.0`, `peft==0.19.1`, `datasets==4.8.4`, `tokenizers==0.22.2`, `huggingface_hub==1.17.0` as core dependencies [14]. Extras carry their own pins: `deepspeed>=0.18.6,<0.19.0`, `vllm>=0.15.0`, `flash-attn==2.8.3`, `ray[train]>=2.52.1`, and the `[tool.uv]` conflicts table pairs the bare `axolotl` package (no extras) against each of several extras individually (vllm, flash-attn, ring-flash-attn, mamba-ssm, auto-gptq, fbgemm-gpu, llmcompressor) - a uv resolver convention that puts each extra in its own lock group, not a claim that the extras conflict with each other [14]. The installation docs additionally require "NVIDIA GPU (Ampere architecture or newer for bf16 and Flash Attention) or AMD GPU" and "Python >=3.11", "PyTorch >=2.11.0" [15] - a stricter Python floor than the packaging metadata's `>=3.10` [14], so treat 3.11 as the floor the maintainers actually test against. The specific commit read for the code-level claims in this card (`c8440999d34f0b4a2c4f4383f4fd84067c9aba44`, authored 2026-07-28) postdates the v0.18.0 release (2026-07-17) by 11 days but predates the repository's actual last push (2026-08-07T18:26:16Z per a live fetch of the repository API); its `pyproject.toml` is byte-identical to the release's at the point read, so no drift exists to flag for that pair, but this card's commit-pinned claims cover only that 2026-07-28 snapshot, not the repository's true latest push ten days later [12][13][14][16].

**Maintained by**: Axolotl AI (GitHub org `axolotl-ai-cloud`) [2]; a live fetch of the repository API reports its last push as 2026-08-07T18:26:16Z, and the commit read for this card's code-level claims (2026-07-28, authored by maintainer Wing Lian with a co-author credit to NanoCode012) sits ten days before that [12][16]; GitHub's release list shows 11 tagged releases in the twelve months before this card (down to v0.12.1, published 2025-08-11), most recently v0.18.0 on 2026-07-17 [17].

## Quick start

The docs' own first example, LoRA fine-tuning a 1B model [4]:

```bash
axolotl fetch examples
axolotl train examples/llama-3/lora-1b.yml
```

The example config it runs (partial, as shown on the quickstart page) [4]:

```yaml
base_model: NousResearch/Llama-3.2-1B
load_in_8bit: true
adapter: lora
datasets:
  - path: teknium/GPT4-LLM-Cleaned
    type: alpaca
dataset_prepared_path: last_run_prepared
val_set_size: 0.1
output_dir: ./outputs/lora-out
```

Removing `load_in_8bit`/`adapter` runs full fine-tuning; swapping in `load_in_4bit: true` with `adapter: qlora` runs QLoRA [4].

## Start it

- One process, one GPU: the command above, as-is.
- More than one GPU: pick one of three mutually exclusive sharding strategies. DeepSpeed is set with `deepspeed: deepspeed_configs/zero1.json` in the YAML (fetch the templates first with `axolotl fetch deepspeed_configs`) [9]; FSDP is set with `fsdp_version: 2` plus an `fsdp_config:` block (the docs mark FSDP "Recommended" and give a field-by-field FSDP1-to-FSDP2 migration table, e.g. `fsdp_sharding_strategy` becomes `reshard_after_forward`) [9]; DDP is the default when neither key is set [9].
- Multi-node: Accelerate needs a per-machine config (`accelerate config`, or a preset YAML setting `machine_rank`, `main_process_ip`, `num_machines`, `num_processes`) plus the same FSDP block in the training YAML on every machine [10]. Torchrun is offered in two forms, run on each node with `num_nodes`, `gpu_per_node`, `head_node_ip`, `head_node_port`, and a shared `rdzv_id` substituted in - the newer `axolotl train config.yaml --launcher torchrun -- --nnodes $num_nodes --nproc_per_node $gpu_per_node --rdzv_id $rdzv_id --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:$head_node_port"`, called "Recommended", or the legacy direct `torchrun --nnodes $num_nodes --nproc_per_node $gpu_per_node --rdzv_id $rdzv_id --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:$head_node_port" -m axolotl.cli.train config.yaml` [10].
- GRPO's generation layer is a separate vLLM process, started and stopped independently of training: `CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve config.yaml` on one GPU, then `CUDA_VISIBLE_DEVICES=1 axolotl train config.yaml` on another; the docs warn the server takes 30-90 seconds to load and must stay running for the whole run [6]. `vllm_mode` under the `trl:` config key is `"server"` or `"colocate"` [6].
- Configuration surface: standard fields (`learning_rate`, `micro_batch_size`, `gradient_accumulation_steps`, ...) sit at the top level of the YAML; every GRPO-specific field (`use_vllm`, `num_generations`, `reward_funcs`, `reward_weights`, `beta`, sync intervals) lives nested under a `trl:` key [6]. No changed-from-upstream default is documented on the pages read for this card beyond the FSDP1-to-FSDP2 field renames above [9].
- Out-of-memory first aid, in the order the docs give it [18]: drop `micro_batch_size` to 1 and raise `gradient_accumulation_steps` to compensate (for GRPO specifically, the docs give a worked example - `num_generations: 16` with `micro_batch_size: 8` at `seq_len 2048` makes an 8 * 16 * 2048 * 151936 * 2-byte bf16 logits tensor, "~75 GB (way too large)", so GRPO needs `micro_batch_size` down to 2-4); enable `gradient_checkpointing: true` (add `gradient_checkpointing_kwargs: {use_reentrant: false}` except under DeepSpeed ZeRO-3 or EBFT's strided flex-attention mode, which the docs say require `use_reentrant: true`); load in 4-bit/8-bit or `fp8: true`; cut `sequence_len` (and for GRPO, `max_completion_length`); switch to `attn_implementation: flash_attention_2`; or offload optimizer state through DeepSpeed [18].

## Watch it

This section is mechanics only - what a metric means for a given method's health lives on that method's card; axolotl's own docs carry method-specific healthy ranges directly, which are reproduced below because no separate method card in this deck currently covers GRPO's axolotl-specific field names.

- **Enable it**: set `wandb_project` (plus optional `wandb_entity`, `wandb_run_id`, `wandb_name`) or the boolean toggles `use_tensorboard`, `use_mlflow`, `use_comet` in the config; axolotl has no single `report_to` field like trl - each backend is its own boolean/field group [19][20]. `logging_steps` sets cadence (the docs recommend `logging_steps: 1` for RL runs) [19].
- **SFT metrics logged every `logging_steps`** [19]: `train/loss`, `eval/loss`, `train/grad_norm`, `train/learning_rate`, `memory/max_alloc`, `memory/max_reserved`.
- **SFT healthy ranges**, from the training-stability guide's monitoring table [18]: `train/loss` should decrease, typically 0.5-2.0 for chat fine-tuning; `eval/loss` should track train loss with a small gap; `grad_norm` healthy at 0.1-10.0, with spikes above 100 signaling instability; `learning_rate` should follow the configured warmup-then-decay schedule.
- **GRPO/RL metrics logged every step** [19]: per-reward-function `rewards/<name>/mean` and `rewards/<name>/std`; aggregated `reward` and `reward_std`; `frac_reward_zero_std`; `completions/mean_length`, `completions/min_length`, `completions/max_length`, `completions/clipped_ratio`; the terminated-length trio `completions/mean_terminated_length`/`min_terminated_length`/`max_terminated_length`; `kl`; `entropy`; plus `sampling/sampling_logp_difference/mean`, `sampling/importance_sampling_ratio/min`, and `clip_ratio/region_mean` per the training-stability guide's metrics table (that table is not fully reproduced on the logging list itself, so read both pages) [18][19].
- **GRPO health thresholds**, quoted from the docs' own "Quick health checks" and metrics table [6][18]: `rewards/*/mean` should be "> 0.15 within 20 steps - if it stays at 0, test your reward function standalone" [6]; `reward_std` "should be > 0 on most steps - all-zero means no learning signal" [6]; `frac_reward_zero_std < 0.8`, with 1.0 on every step meaning "zero-advantage skip fires constantly, no gradient updates" [18]; `entropy` in "0.05-0.5 - below 0.01 suggests mode collapse" and above 1.0 "suggests the model is not converging" [6][18]; `grad_norm` in "0.001-1.0 - > 10 is unstable, 0.0 is expected when zero-advantage skip fires" [6]; `kl` in 0.0-0.5, with above 2.0 meaning "policy has diverged too far from reference" [18]; `sampling/importance_sampling_ratio/min` near 0 flagged as "stale off-policy data; increase vllm_sync_interval" [18]; `clip_ratio/region_mean` above 0.3 meaning "PPO clipping is too aggressive" [18]; `completions/clipped_ratio` above 0.8 meaning most completions hit `max_completion_length` [18].
- **EBFT-specific metrics**, from the same table [18]: `ebft/alignment` should trend upward, healthy 0.3-0.9; `ebft/diversity` healthy 0.01-0.1, above 1.0 indicating mode collapse; `ebft/cfm_loss` should trend downward, below 10.
- **Sample-level logging of generations**: GRPO's `trl:` config block takes `log_completions: bool` (default `false`) to log sampled completions to W&B, plus `num_completions_to_print: int` to cap how many are printed per step; the same two fields appear in the config reference (`log_completions`, `num_completions_to_print`) [6][21]. The training-stability guide's own debugging advice for a broken reward function is to "Enable `log_completions: true` in the `trl:` config and inspect logged completions in W&B" [18].
- **Evaluate during training**: the reward-modeling examples set `eval_steps: 100` with a held-out `val_set_size` split [7]; general SFT eval cadence fields live in the config reference under the standard `eval_steps`/`evals_per_epoch` group (not separately detailed on the pages read for this card - see the config-reference page directly).
- **Stopping**: axolotl publishes `early_stopping_patience`, described in the config reference as stopping training "after this many evaluation losses have increased in a row", linking to the transformers `EarlyStoppingCallback` docs - this watches eval loss, not a reward signal, so it applies to SFT-style loss curves, not GRPO/EBFT reward curves [22]. No RL- or reward-specific stopping threshold is published on the GRPO page, the training-stability page, or the config reference read for this card - shapes are published (the health-check ranges above) but no automatic reward-based stopping rule is [6][18][22].

## Save it

- `output_dir` (config field, default `./model-out`) holds the saved model; `save_strategy`, `save_steps` (or `saves_per_epoch`, mutually exclusive with `save_steps`), and `save_total_limit` control checkpoint cadence and retention, with `save_first_step: bool` (default `False`) to also checkpoint after step one [21].
- `save_only_model: bool` (default `False`) is the resume-breaking flag: the config reference's own comment reads "Save only the model weights, skipping the optimizer. Using this means you can't resume from checkpoints" [21].
- Resume forms: `resume_from_checkpoint: <path>` resumes from a specific checkpoint directory; `auto_resume_from_checkpoints: true` resumes from wherever a prior run left off in `output_dir` without a `save_only_model` checkpoint in the way, and the docs warn "Be careful with this being turned on between different models" [21].
- LoRA/QLoRA adapters are saved on their own, not merged automatically. `axolotl merge-lora config.yml` (optionally `--lora-model-dir "./outputs/lora-out"` to point at a specific checkpoint, or `--dequant` to write bf16 instead of the format-preserving re-quantized merge for quantized bases) folds the adapter into the base model, and the quickstart states plainly "The merged model will be saved in the `{output_dir}/merged` directory" [4][23] - an adapter checkpoint directory by itself is not a full model and is not what `{output_dir}/merged` produces until this command is run.
- Push to Hub: `hub_model_id` and `hub_strategy` config fields control automatic checkpoint pushing during training [21]; the pages read for this card do not show a separate one-shot `push_to_hub()` call distinct from these config fields.
- Whether an evaluator can load the result directly depends on which artifact you point it at: a merged `{output_dir}/merged` directory loads like any other Hub-format model, while a raw adapter checkpoint directory needs the base model paired with it - this loader contract is not axolotl-specific and is not detailed further on the pages read for this card.

## Find it in the docs

The docs site (docs.axolotl.ai) is a Quarto-built site with no version-tag URL segment and no version banner found on its index page - it is a single live surface, not pinned per release; every claim above from this domain is a 2026-08-10 reading of the current live page [1][4][5][6][7][8][9][10][15][18][19][22][21][23].

- Address pattern: `https://docs.axolotl.ai/docs/<slug>.html`. Guessing `docs/config.html` 404s; the correct slug for the config reference is `config-reference.html`, discoverable from the site's own navigation menu [24][21].
- Question-to-slug map, from the navigation tree read on the index and method pages [1][5]: which method to use -> `docs/choosing_method.html`; full config field list -> `docs/config-reference.html`; CLI commands -> `docs/cli.html`; GRPO -> `docs/grpo.html`; EBFT -> `docs/ebft.html`; reward/PRM -> `docs/reward_modelling.html`; dataset shapes -> `docs/dataset-formats` section (chat_template, alpaca, stepwise_supervised for PRM, and others, named on the reward-modeling and choosing-method pages but not independently fetched for this card) [7][5]; multi-GPU sharding -> `docs/multi-gpu.html`; multi-node launch -> `docs/multi-node.html`; OOM/instability -> `docs/training_stability.html`; install -> `docs/installation.html`; hardware/method support boundaries -> `docs/support-matrix.html`.
- Runnable references beyond the docs: `axolotl fetch examples` downloads the repo's example-config tree, including the quickstart's `examples/llama-3/lora-1b.yml`, into the working directory without cloning the repo [4]; `axolotl fetch deepspeed_configs` likewise fetches the named ZeRO-stage JSON templates [9].
- Community/curated layer: not found on the pages read for this card - no community-tutorials-style curated page (comparable to trl's `community_tutorials`) was located during this session; this is an absence, not a confirmed non-existence, since the full docs nav was not exhaustively crawled.
- No official MCP endpoint for the docs was found on the pages read for this card. Axolotl instead ships a bundled offline equivalent: `axolotl agent-docs` prints docs for AI-agent consumption, with subcommands per method (`axolotl agent-docs sft`, `grpo`, `preference_tuning`, `reward_modelling`, `pretraining`), `axolotl agent-docs --list`, and `axolotl config-schema` for the machine-readable config schema - the README states "these docs are bundled with the pip package, no repo clone needed", and are also available as `docs/agents/` and `AGENTS.md` files when working from a source checkout [1].
- Trap: transformers v5 combined with QLoRA on a mixture-of-experts model regressed into an out-of-memory failure, tracked as GitHub issue #3374. COLLABORATOR winglian's first suggestion (2026-01-27), setting `lora_target_parameters` to the expert projection names, did not fix it on its own - CONTRIBUTOR Nero10578 reported the same OOM still occurred. COLLABORATOR NanoCode012 then pointed to a linked code-level PR (2026-02-10); testing that PR still produced a different error until NanoCode012 clarified that `target_linear` also had to be turned off (2026-03-06), which Nero10578 confirmed fixed it the same day. The full fix needs both the merged PR and `target_linear: false` alongside `lora_target_parameters` - the `lora_target_parameters` config change alone is not sufficient [25][26].
- Honest boundaries, from the support matrix [27]: NVFP4 quantization for MoE models is adapter-training only, not full fine-tune; BitNet (1.58-bit) is full-fine-tune only, "LoRA not supported"; sample packing (multipack) is marked "not with RLHF or multimodal"; fused LoRA-MLP/QKV/O kernels are "SFT + FSDP2 only... not RLHF, not FSDP1"; Flash Attention 2/3 is marked "Ampere or newer only (not Turing)", so pre-Ampere NVIDIA GPUs (e.g. Turing/RTX 20-series) cannot use it.

## Sources

All docs.axolotl.ai pages are unversioned live pages with no URL version segment or version banner found, fetched 2026-08-10; the pyproject.toml claims are pinned to the v0.18.0 release commit as noted; the issue citations carry the specific comment date read.


[1] axolotl README (project description, feature list, `agent-docs` CLI, telemetry). Raw file at commit c8440999d34f0b4a2c4f4383f4fd84067c9aba44: https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/c8440999d34f0b4a2c4f4383f4fd84067c9aba44/README.md. Item home: https://github.com/axolotl-ai-cloud/axolotl. Fetched 2026-08-10.

[2] GitHub organization API for axolotl-ai-cloud (name "Axolotl AI", blog https://axolotl.ai). https://api.github.com/orgs/axolotl-ai-cloud. Fetched 2026-08-10.

[3] axolotl GitHub repository (licence Apache-2.0, org). https://github.com/axolotl-ai-cloud/axolotl. Fetched 2026-08-10.

[4] axolotl Quickstart docs. https://docs.axolotl.ai/docs/getting-started.html. Fetched 2026-08-10.

[5] axolotl "Which Fine-Tuning Method Should I Use?" docs (method taxonomy and decision tree). https://docs.axolotl.ai/docs/choosing_method.html. Fetched 2026-08-10.

[6] axolotl GRPO Training docs (vLLM server/colocate modes, `trl:` config block, health checks, metrics). https://docs.axolotl.ai/docs/grpo.html. Fetched 2026-08-10.

[7] axolotl Reward Modelling docs (outcome and PRM example configs). https://docs.axolotl.ai/docs/reward_modelling.html. Fetched 2026-08-10.

[8] axolotl EBFT Training docs (Energy-Based Fine-Tuning definition and reward formula). https://docs.axolotl.ai/docs/ebft.html. Fetched 2026-08-10.

[9] axolotl Multi-GPU docs (DeepSpeed, FSDP, DDP, FSDP1-to-FSDP2 migration table). https://docs.axolotl.ai/docs/multi-gpu.html. Fetched 2026-08-10.

[10] axolotl Multi Node docs (Accelerate, torchrun launcher forms). https://docs.axolotl.ai/docs/multi-node.html. Fetched 2026-08-10.

[11] axolotl on PyPI (version 0.18.0, upload date, requires_python). https://pypi.org/pypi/axolotl/json. Fetched 2026-08-10.

[12] GitHub repository API for axolotl-ai-cloud/axolotl, live-fetched (license, stargazers, pushed_at 2026-08-07T18:26:16Z). https://api.github.com/repos/axolotl-ai-cloud/axolotl. Fetched 2026-08-10.

[13] GitHub git refs API resolving tag v0.18.0 to commit 2f5cb9da62a0fe763a1ddeb7798fc9acb2f4a417. https://api.github.com/repos/axolotl-ai-cloud/axolotl/git/refs/tags/v0.18.0. Fetched 2026-08-10.

[14] axolotl pyproject.toml at the v0.18.0 release commit (dependency and extras pins). https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/2f5cb9da62a0fe763a1ddeb7798fc9acb2f4a417/pyproject.toml. Fetched 2026-08-10.

[15] axolotl Installation docs (GPU/Python/PyTorch requirements, uv install commands). https://docs.axolotl.ai/docs/installation.html. Fetched 2026-08-10.

[16] GitHub commit API for c8440999d34f0b4a2c4f4383f4fd84067c9aba44 (author Wing Lian, date 2026-07-28, co-author NanoCode012). https://api.github.com/repos/axolotl-ai-cloud/axolotl/commits/c8440999d34f0b4a2c4f4383f4fd84067c9aba44. Fetched 2026-08-10.

[17] GitHub releases API for axolotl-ai-cloud/axolotl, fetched with `per_page=15` to reach the full twelve-month window (11 tagged releases from v0.12.1, 2025-08-11, through v0.18.0, 2026-07-17). https://api.github.com/repos/axolotl-ai-cloud/axolotl/releases?per_page=15. Fetched 2026-08-10.

[18] axolotl Training Stability & Debugging docs (SFT and GRPO/EBFT monitoring tables, OOM debugging steps with the 75GB GRPO logits example). https://docs.axolotl.ai/docs/training_stability.html. Fetched 2026-08-10.

[19] axolotl Training Stability & Debugging docs, "Enabling Logging" and "What Axolotl Logs" sections (logging backends, exact SFT and GRPO metric names). https://docs.axolotl.ai/docs/training_stability.html. Fetched 2026-08-10.

[20] axolotl Config Reference docs (`use_wandb`, `use_tensorboard`, `use_mlflow`, `use_comet` fields). https://docs.axolotl.ai/docs/config-reference.html. Fetched 2026-08-10.

[21] axolotl Config Reference docs (`output_dir`, `save_strategy`, `save_steps`, `saves_per_epoch`, `save_total_limit`, `save_first_step`, `save_only_model`, `resume_from_checkpoint`, `auto_resume_from_checkpoints`, `hub_model_id`, `hub_strategy` fields). https://docs.axolotl.ai/docs/config-reference.html. Fetched 2026-08-10.

[22] axolotl Config Reference docs (`early_stopping_patience` field and its eval-loss-based description). https://docs.axolotl.ai/docs/config-reference.html. Fetched 2026-08-10.

[23] axolotl Command Line Interface docs (`merge-lora` command and flags). https://docs.axolotl.ai/docs/cli.html. Fetched 2026-08-10.

[24] axolotl docs index page (navigation tree used to discover page slugs). https://docs.axolotl.ai/docs/index.html. Fetched 2026-08-10.

[25] GitHub issue #3374, "Transformers-v5 causes OOM on QLoRA config...". https://api.github.com/repos/axolotl-ai-cloud/axolotl/issues/3374. Fetched 2026-08-10.

[26] GitHub issue #3374 comments (COLLABORATOR winglian's initial `lora_target_parameters` suggestion, CONTRIBUTOR Nero10578 reporting it insufficient, COLLABORATOR NanoCode012 pointing to a linked PR and the additional `target_linear: false` requirement, and Nero10578's confirmation dated 2026-03-06). https://api.github.com/repos/axolotl-ai-cloud/axolotl/issues/3374/comments. Fetched 2026-08-10.

[27] axolotl Support Matrix docs (NVFP4-MoE adapter-only, BitNet full-finetune-only, multipack/RLHF exclusion, fused-kernel SFT+FSDP2-only, Flash Attention Ampere-or-newer boundary). https://docs.axolotl.ai/docs/support-matrix.html. Fetched 2026-08-10.
