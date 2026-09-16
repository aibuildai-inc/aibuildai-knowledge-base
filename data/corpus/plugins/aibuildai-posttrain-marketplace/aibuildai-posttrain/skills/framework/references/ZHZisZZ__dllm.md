# dLLM

A research library for training and evaluating diffusion language models: one `transformers.Trainer` subclass per diffusion training objective, plus a GRPO trainer adapted for iterative denoising instead of autoregressive generation.

**dLLM** is described in its own README as "a library that unifies the training and evaluation of diffusion language models, bringing transparency and reproducibility to the entire development pipeline" [1]. It is built by Zhanhui Zhou, Lingjie Chen, Hanghang Tong, and Dawn Song, credited as the authors of the project's own technical report [1][2]; two of them (ZHZisZZ, lingjiechen2) are the maintainers answering issues on the repository [3][4][5]. Its API centers on `transformers.Trainer` subclasses (`MDLMTrainer`, `BD3LMTrainer`) that take a Hugging Face model, tokenizer, and dataset and run `.train()`, plus a `DiffuGRPOTrainer` that subclasses TRL's `GRPOTrainer` for reinforcement learning [1][6][7]. It lives at https://github.com/ZHZisZZ/dllm [3].

**When to pick it**: training or evaluating diffusion language models specifically - masked diffusion (MDLM), block diffusion (BD3LM), and GRPO reinforcement learning for them - including reference recipes for converting an existing autoregressive model (e.g. Qwen, LLaMA, GPT-2, BERT) into a diffusion model [1]. It is not a general post-training library for autoregressive LLMs: its SFT and RL trainers are built specifically around diffusion's masking/denoising loss and generation process, and every quick-start example targets a diffusion checkpoint or an AR-to-diffusion conversion [1][6][7].

**Methods it ships**: two reusable "core" trainer classes under `dllm.core.trainers` - `MDLMTrainer`/`MDLMConfig` implementing the masked diffusion LM loss, and `BD3LMTrainer`/`BD3LMConfig` (subclasses `MDLMConfig`, adds `block_size`) implementing block-diffusion loss with a specialized block-causal attention mask [6][8]. `DreamTrainer(MDLMTrainer)` is a pipeline-level specialization for the Dream model that overrides the loss-weighting scheme (`loss_weight_type="cart[geo_p:0.3]"`) rather than a separate top-level method [9]. For reinforcement learning, `dllm.pipelines.rl.grpo.trainer.DiffuGRPOTrainer` subclasses TRL's `GRPOTrainer` directly and its own docstring states it overrides four methods - `_generate_and_score_completions` (iterative denoising instead of autoregressive generation), `_get_per_token_logps` (diffusion forward-process log-prob), `_compute_loss`, and `_prepare_inputs` - while "PPO clipping, KL regularization, reference model management, and distributed training are inherited from TRL's GRPOTrainer" [7]. This GRPO adaptation follows the external `dllm-reasoning/d1` (diffu-GRPO) reference implementation [10]. There is no hosted "methods" index page to recheck against; the taxonomy above is read directly from the `dllm/core/trainers/__init__.py` exports and the pipeline source at commit ca176752fbceec49c6b4777a2c18ae88e4eb10ed [6][7][9].

**Scale it handles**: single GPU up to multi-node, launched through Hugging Face Accelerate (`accelerate launch --config_file <yaml> <script>`), with ready config templates at `scripts/accelerate_configs/{ddp,zero1,zero2,zero3,fsdp,fsdp2,cpu}.yaml` for DeepSpeed ZeRO and FSDP/FSDP2 sharding [1][11]. Multi-node runs go through a Slurm wrapper, `sbatch scripts/train.slurm.sh --accelerate_config <name> --script_path <script>`, which computes node/world-size from Slurm environment variables and launches `accelerate launch` via `srun` [1][12]. This is a published-benchmark case, not just documented mechanism: in response to an issue, a maintainer (lingjiechen2, COLLABORATOR) posted measured SFT wall-clock times for `tulu-3-sft-mixture` at four GPU counts on their Slurm cluster - 44.24 hours at 8 GPUs, 59.45 hours at 16, 29.90 hours at 32, and 17.22 hours at 64 - and noted that inter-node communication overhead on their specific cluster makes the 16- and 32-GPU points non-monotonic, calling the 8-GPU figure "the most directly referable" reference since it avoids cross-node communication [4]. LoRA (via `peft.LoraConfig`) and 4-bit quantization (`load_in_4bit`) are both supported as memory-reduction options on top of any launcher [1][13].

**Install**: `pip install -e .` from a local clone (not published to PyPI); the only tagged release is `v0.1.0`, tagged/committed 2026-02-27 [14][15]. The screening commit (ca176752fbceec49c6b4777a2c18ae88e4eb10ed, pushed 2026-07-17) is almost five months ahead of that tag; its `pyproject.toml` is dependency-identical to the tag's, so this Install field is reported at the tag pin but the rest of this card's code-level claims are read from the newer, ahead-of-release commit and are flagged accordingly below [15][16][17]. Python floor `>=3.10`; licence Apache-2.0 [16][3]. Version floors at the v0.1.0 tag: `transformers==4.57.0`, `accelerate==1.11.0`, `deepspeed==0.18.0`, `peft==0.17.1`, `datasets==4.2.0`, `sentencepiece==0.2.0` - all pinned with `==`, not floors [16]. torch is not listed in `pyproject.toml` at all; the README instructs installing it manually first, with `torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0` against the CUDA 12.4 wheel index, before `pip install -e .` [1]. Optional extras (`pip install -e ".[optional]"`): `bitsandbytes==0.48.1`, `vllm==0.8.5.post1`, `flash-attn==2.8.3` [16]. No CUDA/hardware minimum is stated anywhere in the README or `pyproject.toml`; the CUDA 12.4 install line is presented as one option ("other pytorch/cuda versions should also work") [1].

**Maintained by**: Zhanhui Zhou, Lingjie Chen, Hanghang Tong, and Dawn Song per the project's own citation entry, with ZHZisZZ and lingjiechen2 answering issues as OWNER/COLLABORATOR [1][3][4][5]. Its most recent README-listed feature announcement is diffu-GRPO support dated 2026/04, and the repository's last push was 2026-07-17 [1][3].

## Quick start

The smallest complete SFT run, quoted from the README's own Training section (the pattern behind every `examples/*/sft.py` entry point) [1]:

```python
import transformers
import dllm

model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model = dllm.utils.get_model(model_args=model_args)
tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
dataset = "..."

trainer = dllm.core.trainers.MDLMTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
    ),
)
trainer.train()
```

CLI form for a concrete example - block-diffusion SFT on `Qwen3-0.6B` over the `tatsu-lab/alpaca` dataset, from `examples/a2d/bd3lm/sft.py` [8]:

```shell
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/bd3lm/sft.py
```

A minimal GRPO run - LLaDA-8B-Instruct on gsm8k, 1 GPU, from `examples/rl/README.md` [10]:

```shell
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/rl/grpo/llada/train.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --load_in_4bit True \
    --dataset gsm8k --max_steps 50 \
    --output_dir .models/LLaDA-8B-Instruct/grpo
```

## Start it

- One process, one GPU: any of the scripts above unchanged, with `--num_processes 1` on the `ddp.yaml` config [1][8].
- More GPUs on one machine or across nodes: run the same script through `accelerate launch --config_file <template>.yaml`, choosing among `scripts/accelerate_configs/{ddp,zero1,zero2,zero3,fsdp,fsdp2,cpu}.yaml` for the sharding strategy [1][11]. The README's own Slurm example runs LLaDA SFT with `sbatch --gres=gpu:8 scripts/train.slurm.sh --accelerate_config "fsdp" --script_path "examples/llada/sft.py"` (add `--nodes=2` for 2-node/16-GPU) [1]; the a2d/bd3lm example's own docstring gives the equivalent form for that script, `sbatch --gres=gpu:8 scripts/train.slurm.sh --accelerate_config "zero2" --script_path "examples/a2d/bd3lm/sft.py"` [8]. The wrapper script derives `NUM_NODES`, `WORLD_SIZE`, and the rendezvous address from Slurm environment variables and forwards all extra flags to the training script [12].
- GRPO's LoRA + 8-GPU ZeRO-2 example sets `--num_generations 6 --per_device_train_batch_size 6 --gradient_accumulation_steps 2 --num_iterations 12`, so effective batch scales as per-device batch x GPU count x gradient accumulation, same as any `transformers.Trainer` [10].
- Configuration is a dataclass subclassing `transformers.TrainingArguments` (`dllm.utils.configs.TrainingArguments`, further subclassed per method as `MDLMConfig`/`BD3LMConfig`/`DiffuGRPOConfig`), and it changes several base defaults the moment training starts: `bf16=True` (base transformers defaults to `False` - a bf16-capable GPU is a silent assumption here), `report_to="wandb"` (base defaults to `"none"` - a run is tracked by default, not silent, unlike a library that logs nowhere), `eval_strategy="steps"` with `eval_steps=0.1` (base defaults to `"no"`), and `save_only_model=True` (see Save it - base defaults to `False`, and this default breaks resume) [13][18][19]. GRPO adds its own defaults on top of TRL's `GRPOConfig`: `block_size=64`, `steps=64` (diffusion denoising steps), `cfg_scale=0.0`, `remasking="low_confidence"`, `p_mask_prompt=0.3`, `scale_rewards=False` [7].
- Out-of-memory first aid: the README's own "Useful tips for training" names `--load_in_4bit True --lora True` as the combination for training under memory pressure, and switching `--accelerate_config` between `"ddp,zero-{1,2,3},fsdp"` for heavier sharding [1]. No dedicated OOM section or generation-side memory knob for GRPO's denoising loop is published; this is read from the root README and the `examples/rl/README.md` and `examples/llada/README.md` pages, none of which carries such a section [1][10].

## Watch it

This section covers only the logging mechanics; what a metric shape means for masked-diffusion or GRPO training is not restated here.

- **Enable it**: `report_to="wandb"` is the library's own changed default (see Start it), so a run is logged to Weights & Biases unless overridden; the underlying field is the standard `transformers.TrainingArguments` surface and accepts its full backend set [13][19].
- **MDLM/BD3LM/Dream metric names**: every SFT-style trainer registers a `torchmetrics`-based `OnEvaluateMetricsCallback` with `{"nll": NLLMetric(), "ppl": PPLMetric()}`, logged as `nll`/`ppl` for the train split and `eval_nll`/`eval_ppl` for the eval split via a `key_for(split, name)` naming scheme, read directly from `dllm/core/trainers/mdlm.py` and `dllm/core/trainers/utils/meters.py` at commit ca176752... [6][20].
- **GRPO metric names**, read directly from `dllm/pipelines/rl/grpo/trainer.py` at commit ca176752...: `num_tokens`; the `completions/length` family (mean/min/max); `completions/clipped_ratio`; the `completions/terminated_length` family (mean/min/max); per-reward-function `rewards/<name>/mean` and `rewards/<name>/std`; combined `reward`, `reward_std`, and `frac_reward_zero_std` [7]. There is no separate hosted metrics page for these names - the source file at the pinned commit is the only place they are enumerated.
- **Sample-level logging**: `DiffuGRPOTrainer` accumulates per-step `prompt`, `completion`, `rewards`, and `advantages` text/values into a `_textual_logs` dict, mirroring TRL's own `log_completions` mechanism it inherits from `GRPOTrainer` [7].
- **Evaluation during training**: `eval_strategy="steps"` with `eval_steps=0.1` (a fraction of total steps) is the library-changed default on every trainer's Config, and `per_device_eval_batch_size` follows the same `TrainingArguments` surface as `per_device_train_batch_size` [13].
- **Stopping**: no RL-specific stopping rule, threshold, or patience value is published anywhere searched for this card - the root README, `examples/rl/README.md`, and the GRPO trainer source at commit ca176752... carry no early-stopping or reward-threshold field; only the general `transformers.TrainingArguments`/`Trainer` callback surface (e.g. `EarlyStoppingCallback`) is available, and it is generic to any HF Trainer run, not specific to dLLM or to a reward curve [1][10][7].
- One maintainer-confirmed trap: for LoRA-trained models, `chat.py` and `eval.py` share the same sampler and differ only in output post-processing (`chat.py` trims after EOS, `eval.py` does not, because lm-eval's own metrics handle that); a maintainer (lingjiechen2, COLLABORATOR) noted this in issue #80 when a user reported apparently inconsistent generation behavior between the two scripts [5].

## Save it

- Trainers save through the standard `transformers.Trainer` call: `trainer.save_model(output_dir)` followed by `trainer.processing_class.save_pretrained(output_dir)`, exactly as shown at the end of `examples/a2d/bd3lm/sft.py`, which writes both to a `checkpoint-final` subdirectory of `output_dir` [8].
- Checkpoint cadence and retention are inherited `TrainingArguments` fields; dLLM's own Config sets `save_steps=0.1` (a fraction of total steps) but leaves `save_strategy` and `save_total_limit` at base transformers' defaults [13].
- **`save_only_model=True` is dLLM's own changed default, and it silently breaks resume.** transformers' own docs state the contract this flag invokes: "Save only model weights, not optimizer/scheduler/RNG state. Significantly reduces checkpoint size but prevents resuming training from the checkpoint. Use when you only need the trained model for inference, not continued training. You can only load the model using `from_pretrained` with this option set to `True`" [18]. Base `transformers.TrainingArguments` defaults `save_only_model` to `False`; dLLM's `dllm.utils.configs.TrainingArguments` sets it to `True` by default, at commit ca176752... [13][18]. A reader who wants to resume training from a dLLM checkpoint must explicitly pass `save_only_model=False`.
- PEFT/LoRA changes what gets saved: dLLM wraps models with `peft.LoraConfig` when `--lora True` (`dllm.utils.utils.load_peft`), and a maintainer directed a user with a LoRA checkpoint to `dllm/tools/merge_peft_adapter.py` before evaluating it "as if it is a non-peft model" - i.e. a LoRA-trained checkpoint from this library is an adapter, not a full model, and needs an explicit merge step before it can be loaded like one [5][21][22].
- Whether a downstream evaluator can load a saved checkpoint directly depends on that evaluator's own loader contract, which this card does not cover; a `save_only_model=True` checkpoint (dLLM's default) loads only via `from_pretrained` for inference, never via `trainer.train(resume_from_checkpoint=...)` [18].

## Find it in the docs

There is no hosted documentation site for dLLM - no `docs/`, mkdocs, or ReadTheDocs configuration exists in the repository tree at commit ca176752... The root README.md is the single entry point, and every training/inference/evaluation recipe for a specific model family lives in its own `examples/<name>/README.md` [1][23].

- Address pattern: `https://github.com/ZHZisZZ/dllm/blob/main/<path>` for any file, or `https://raw.githubusercontent.com/ZHZisZZ/dllm/main/<path>` for raw content. There is no versioned docs URL scheme to swap a tag into - the only tag is `v0.1.0`, and the README does not distinguish per-version docs.
- Page-to-topic map, from the root README's own 9-entry Features list: `examples/llada/README.md` (LLaDA/LLaDA-MoE pretrain/finetune/eval), `examples/llada2/README.md` (LLaDA2.0 inference), `examples/llada21/README.md` (LLaDA2.1 inference), `examples/dream/README.md` (Dream pretrain/finetune/eval), `examples/a2d/README.md` (any-AR-to-diffusion conversion), `examples/bert/README.md` (any-BERT-to-diffusion conversion), `examples/editflow/README.md` (Edit Flows), `examples/fastdllm/README.md` (Fast-dLLM accelerated inference/eval), `examples/rl/README.md` (GRPO) [1].
- Runnable references beyond the README: the `examples/` tree itself is the smoke-test surface - every training script's own module docstring gives the exact `accelerate launch`/`sbatch` invocation for 1 GPU, 8 GPUs, and multi-node, as seen in `examples/a2d/bd3lm/sft.py` and `examples/rl/README.md` [8][10]. `examples/rl/README.md` names five ready-made reasoning datasets for GRPO: `gsm8k`, `countdown`, `sudoku`, `math` (`ankner/math-500`), and `code` (`KodCode/KodCode-Light-RL-10K`) [10].
- Community layer: the repository has GitHub Issues enabled but no Wiki and no Discussions [3]. There is no curated tutorials page; the only community-adjacent content found is maintainer replies inside closed issues, cited in-context above and below.
- No MCP endpoint or docs-search tool is published for this project.

Two more maintainer-confirmed traps, stated where a reader would hit them:
- **Evaluation setup**: cloning the `lm-evaluation-harness` submodule over HTTPS can fail (a 500 error was reported); a maintainer (lingjiechen2, COLLABORATOR) confirmed in issue #97 that switching the submodule remote to SSH resolved it for the reporting user, and attributed it to a network/proxy path issue rather than something repo-specific [24].
- **Multi-node training**: an NCCL watchdog timeout during multi-node pretraining via `train.slurm.sh` was fixed by a maintainer (ZHZisZZ, OWNER) in commit `0dec970ae00a5f80e2d40f510d024fe63922ced0`, per issue #72; the same reply notes that `RandomTruncateWrapper` is "a trick used only in LLaDA pretraining" and can be safely removed from other pretraining scripts [25].

Honest boundary: every recipe in dLLM's README Features list targets a specific published diffusion-LM architecture or an AR/BERT-to-diffusion conversion (LLaDA, LLaDA2.0, LLaDA2.1, Dream, A2D, BERT-Chat, Edit Flows, Fast-dLLM, GRPO), not general post-training for arbitrary autoregressive LLMs, and every quick-start path in this card targets one of those diffusion recipes [1].

## Sources

All pages read at the screening commit ca176752fbceec49c6b4777a2c18ae88e4eb10ed unless a tag or a docs URL is named; fetched/read 2026-08-11.

[1] dLLM root README. https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/README.md.

[2] dLLM technical report citation entry, quoted inside [1] (arXiv 2602.22661, not separately fetched for this card).

[3] dLLM GitHub repository metadata (license, stars, pushed_at, created_at, has_issues/has_wiki/has_discussions). https://api.github.com/repos/ZHZisZZ/dllm.

[4] Issue #102, "SFT runtime and GPU requirement clarification (tulu-3-sft-mixture)", maintainer reply from lingjiechen2 (COLLABORATOR), 2026-03-26. https://api.github.com/repos/ZHZisZZ/dllm/issues/102 and its comments endpoint.

[5] Issue #80, "Extremely slow GSM8K evaluation speed and LoRA support in eval pipeline", replies from ZHZisZZ (OWNER, 2026-01-30) and lingjiechen2 (COLLABORATOR, 2026-02-02 and 2026-02-22). https://api.github.com/repos/ZHZisZZ/dllm/issues/80 and its comments endpoint.

[6] `dllm/core/trainers/__init__.py` and `dllm/core/trainers/mdlm.py`. https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/__init__.py and .../mdlm.py.

[7] `dllm/pipelines/rl/grpo/trainer.py`. https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/rl/grpo/trainer.py.

[8] `examples/a2d/bd3lm/sft.py`. https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/a2d/bd3lm/sft.py.

[9] `dllm/pipelines/dream/trainer.py`. https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/dream/trainer.py.

[10] `examples/rl/README.md`. https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/rl/README.md.

[11] `scripts/accelerate_configs/zero2.yaml` and `scripts/accelerate_configs/fsdp2.yaml`. https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/scripts/accelerate_configs/zero2.yaml and .../fsdp2.yaml.

[12] `scripts/train.slurm.sh`. https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/scripts/train.slurm.sh.

[13] `dllm/utils/configs.py` (`TrainingArguments`, `ModelArguments`, `DataArguments` dataclasses). https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/configs.py.

[14] dLLM git tags list (only `v0.1.0`). https://api.github.com/repos/ZHZisZZ/dllm/tags.

[15] Commit metadata for the `v0.1.0` tag (committer date 2026-02-27). https://api.github.com/repos/ZHZisZZ/dllm/commits/b8d76ff74b2053d359cd88fedfbc6362db17e3d7.

[16] `pyproject.toml` at the `v0.1.0` tag and at commit ca176752... (dependency/extras pins, near-identical between the two). https://raw.githubusercontent.com/ZHZisZZ/dllm/v0.1.0/pyproject.toml and https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/pyproject.toml.

[17] dLLM GitHub Releases list (empty - no Releases published). https://api.github.com/repos/ZHZisZZ/dllm/releases.

[18] transformers `TrainingArguments`/`Trainer` docs (`save_only_model`, `bf16`, `report_to`, `eval_strategy`, `save_steps` default values and the `save_only_model` contract quote). https://huggingface.co/docs/transformers/main/en/main_classes/trainer. Fetched 2026-08-11, `main`-version docs, unpinned and mutable.

[19] transformers `TrainingArguments` `report_to` field, same page as [18].

[20] `dllm/core/trainers/utils/meters.py` (`OnEvaluateMetricsCallback`, `BaseMetricsCallback`, `key_for` naming). https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/utils/meters.py.

[21] `dllm/utils/utils.py` (`load_peft` function building `peft.LoraConfig`). https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/utils.py.

[22] `dllm/tools/merge_peft_adapter.py`. https://raw.githubusercontent.com/ZHZisZZ/dllm/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/tools/merge_peft_adapter.py.

[23] Repository file tree at commit ca176752fbceec49c6b4777a2c18ae88e4eb10ed (used to confirm no `docs/`, mkdocs, or ReadTheDocs configuration exists). https://api.github.com/repos/ZHZisZZ/dllm/git/trees/ca176752fbceec49c6b4777a2c18ae88e4eb10ed?recursive=1.

[24] Issue #97, "Bug: Unable to init submodule", maintainer reply from lingjiechen2 (COLLABORATOR), 2026-03-03. https://api.github.com/repos/ZHZisZZ/dllm/issues/97 and its comments endpoint.

[25] Issue #72, NCCL watchdog timeout during multi-node pretraining, maintainer reply from ZHZisZZ (OWNER), 2026-01-06. https://api.github.com/repos/ZHZisZZ/dllm/issues/72 and its comments endpoint.
