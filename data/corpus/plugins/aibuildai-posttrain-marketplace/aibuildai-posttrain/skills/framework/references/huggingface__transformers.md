# transformers

Hugging Face's model library: not itself a post-training framework, but its `Trainer` class is the generic supervised training loop that trl's method-specific trainers subclass - start on one machine, scale out through Accelerate, DeepSpeed, or FSDP2.

**transformers** describes itself as "the model-definition framework for state-of-the-art machine learning models in text, computer vision, audio, video, and multimodal models, for both inference and training," and positions itself as "the pivot across frameworks" so that a supported model definition is compatible with training frameworks such as Axolotl, Unsloth, DeepSpeed, FSDP, and PyTorch-Lightning, inference engines such as vLLM and SGLang, and adjacent libraries such as llama.cpp [1]. It is built and maintained by Hugging Face [2][3]. Its training-relevant API is `Trainer` plus its `TrainingArguments` config dataclass: construct a `Trainer` with a model, a `TrainingArguments` instance, and a dataset, then call `.train()` [4]. It lives at https://github.com/huggingface/transformers [3].

**When to pick it**: as a post-training tool, pick `transformers.Trainer` directly when the job is a plain supervised loss on a Hub model and dataset and you don't need a preference, RLHF, or distillation trainer - it ships no such trainer classes itself [1][4]. trl's trainers build on this library: its `SFTTrainer` subclasses trl's own `_BaseTrainer`, which itself subclasses this library's `Trainer` [25, commit `2396dfe`] (cross-reference to the trl card; the method-specific losses and generation handling are not restated here). Pick verl instead for cluster-scale online RL orchestrated by Ray (cross-reference; not covered here).

**Methods it ships**: none of the RLHF/preference/distillation family - `transformers` provides one generic trainer, `Trainer`, that runs whatever loss the model's `forward()` returns (typically causal-LM cross-entropy for the fine-tuning tutorial's example) [4]; the shortlist's own repository scan likewise found only this one class, `src/transformers/trainer.py`, when looking for training methods in this repository, read at the v5.14.1 tag's commit `a08ace4` (see Install for why that commit, not the newer commit the screening row names, is the one this card reads). `Trainer` auto-detects a PEFT-wrapped model (`peft.PeftModel`) and special-cases its checkpoint loading and adapter directories accordingly [5, commit `a08ace4`].

**Scale it handles**: single GPU, through multi-GPU DDP, to parameter/gradient/optimizer sharding with FSDP2 or DeepSpeed ZeRO (stages 1-3, with optional CPU/NVMe offload), all launched through Accelerate, which "detects your environment (number of GPUs, distributed backend, mixed precision, etc.) and automatically configures training" [6][7][8][9]. All example configs and launch commands on the pages read for this card - the Accelerate config file, the DeepSpeed `torchrun` command - are single-machine (`num_machines: 1`, `--nproc_per_node` with no node count or rank flags); none of these pages documents a multi-node launch mechanism or benchmark [6][8].

**Install**: `pip install "transformers[torch]"` (the base `pip install transformers` does not pull in torch at all - it is an extra) [10]; version 5.14.1, released 2026-07-16 [2]; Python >= 3.10 [2]; Apache-2.0 [11]. The docs state transformers "has been tested on Python 3.10+ and PyTorch 2.5+" [10], while the pinned floor in the package itself is looser: `torch>=2.4` (unpinned upper bound; torch itself is not in the unconditional `install_requires`, only pulled by the `torch` extra) [12]. Other load-bearing floors at the v5.14.1 tag: `tokenizers>=0.22.0,<=0.23.0`, `huggingface-hub>=1.5.0,<2.0`, `accelerate>=1.1.0` (also an extra), `deepspeed>=0.9.3` (extra), `peft>=0.19.1` (not a hard dependency; imported conditionally) [12]. Full pin list: `setup.py` at the `v5.14.1` tag [12]. No CUDA/hardware minimum is stated beyond the PyTorch 2.5+ recommendation above [10]. The screening row's commit, `3717b9c`, is the repository's newest push (2026-07-31), 15 days ahead of the `v5.14.1` release commit `a08ace4` (2026-07-16) that `pip install` actually delivers; every source-code claim on this card cites `a08ace4`, not the screening commit [2][12].

**Maintained by**: Hugging Face; about 163k GitHub stars (not for ranking) [3]. Actively released: v5.14.1 (2026-07-16), v5.14.0 (2026-07-15), v5.13.1 (2026-07-11), v5.13.0 (2026-07-03), roughly weekly over the month before this reading [13].

## Quick start

From the fine-tuning tutorial [4], a complete supervised fine-tuning run:

```py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

def tokenize(batch):
    return tokenizer(batch["horoscope"], truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

training_args = TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.push_to_hub()
```

There is no dedicated training CLI; the tutorial's own next step for more runnable code is to browse the `examples/pytorch` tree in the repository [4].

## Start it

- One process, one GPU is the script above, as-is; DDP "activates automatically when you launch with a multi-process launcher like Accelerate" [7]: `accelerate launch --num_processes 4 train.py` [7].
- More GPUs, or a different sharding strategy, go through an Accelerate config file (`accelerate config`, saved as `default_config.yaml`), or the equivalent `TrainingArguments` fields (`fsdp`/`fsdp_config`, `deepspeed`, or the `ddp_*` fields) - the config-file route and the `TrainingArguments` route cover the same settings and should not be combined [6].
- Effective batch size is per-device batch x world size x `gradient_accumulation_steps`; `Trainer` skips the DDP all-reduce on intermediate accumulation steps and runs it only on the final micro-batch of each accumulation window [7].
- `TrainingArguments`' own baseline defaults are conservative, not tuned for a specific hardware assumption: `bf16=False` and `fp16=False` (full fp32 unless you opt in), `gradient_checkpointing=False`, `eval_strategy="no"`, `per_device_train_batch_size=8`, `num_train_epochs=3.0`, `dataloader_num_workers=0` [4]. (This is the base library other trainers, such as trl's, override - trl's own Config classes turn bf16 on by default, which this library does not do on its own; see the trl card.)
- FSDP2 is model/optimizer-state sharding across GPUs, recommended "when your model or optimizer states don't fit on a single GPU" [9]; DeepSpeed ZeRO offers three shard stages plus CPU/NVMe offload, with the guidance "Use ZeRO-3 only when your model doesn't fit across GPUs with ZeRO-2" because it costs more inter-GPU communication [8].
- Out-of-memory first aid, from the memory-efficient-evals recipe [14]: evaluation is the more common OOM site than training, because `Trainer` "runs a forward pass on every batch and concatenates the logits into a single tensor on the GPU," which can exhaust memory "even when training on the same hardware works fine." Set `eval_accumulation_steps` to offload accumulated predictions from GPU to CPU every *n* batches, or set `preprocess_logits_for_metrics` to reduce the tensor kept per batch before accumulation [14]. This is a long-standing, maintainer-confirmed gap: in issue #7232, a Hugging Face collaborator (`sgugger`, association COLLABORATOR) confirmed on 2020-09-18 that the eval-side OOM was "a known issue"; the same collaborator followed up on 2020-10-11 naming the workaround available at the time, chunking the dataset into smaller parts [15]; `eval_accumulation_steps` and `preprocess_logits_for_metrics` are the current documented fix [14].

## Watch it

This section is mechanics only - what a training-loss or eval-metric curve should look like for a given method belongs on that method's own card; `transformers.Trainer` has no method-specific signal, only the generic ones below.

- **Enable it**: set `report_to` on `TrainingArguments`, default `"none"`; supported values include `"tensorboard"`, `"trackio"`, `"wandb"`, `"mlflow"`, `"comet_ml"`, `"clearml"`, `"dagshub"`, `"dvclive"`, `"azure_ml"`, `"codecarbon"`, `"swanlab"`, plus `"all"` to report to every installed integration [16]. With the default, a run produces no external tracker record at all. Cadence is `logging_strategy` (`"steps"` or `"epoch"`) and `logging_steps`, default 500 [4][14].
- **Metric names emitted by `Trainer.log`** (read from `src/transformers/trainer.py` at commit `a08ace4`, the v5.14.1 release commit [5]): during training, `loss` (the mean loss since the last log), `grad_norm`, `learning_rate`, and `epoch` are logged unconditionally; `num_input_tokens_seen` is only added when `include_num_input_tokens_seen` is set to something other than its default `"no"` [5]. During `evaluate()`/`predict()`, every reported key is prefixed with `metric_key_prefix` (default `"eval"` for `evaluate()`, `"test"` for `predict()`), so the loss key becomes e.g. `eval_loss` [5]. No separate live docs page enumerates this list in prose; it is read directly from the `_maybe_log_save_evaluate` and `evaluation_loop` code paths [5].
- **Sample the generations**: the base `Trainer` has no completion-sampling or generation-logging field - that capability is specific to trl's online trainers (see the trl card) and is out of scope for this library, which does not run a generation loop of its own outside of `predict()`.
- **Evaluate during training**: `eval_strategy` (default `"no"`) and `eval_dataset` on the `Trainer` constructor; `metric_for_best_model` and `greater_is_better` select which eval metric drives `save_strategy="best"` and `load_best_model_at_end` [4][14].
- **Stopping**: no RL-specific stopping rule exists because this library ships no RL trainer. The generic mechanism is `EarlyStoppingCallback`, which "stops training when an evaluation metric stops improving" - after each evaluation it checks whether the metric improved by more than `early_stopping_threshold`, and stops if it hasn't improved for `early_stopping_patience` consecutive evaluations [17]. The two official pages drift on what it requires: the guide lists two required `TrainingArguments` fields, `metric_for_best_model` and `eval_strategy` [17], while the API reference instead states that the callback "depends on `TrainingArguments` argument `load_best_model_at_end` functionality to set `best_metric` in `TrainerState`", naming `load_best_model_at_end` and not `eval_strategy` [26]. Set `metric_for_best_model`, `eval_strategy`, and `load_best_model_at_end` together to satisfy both pages. This callback watches whatever eval metric you configure - it has no built-in awareness of a reward curve or KL, so it is not a substitute for a method-specific stopping rule.

## Save it

- Checkpoints land under `TrainingArguments.output_dir` as `checkpoint-<step>/` directories; `Trainer` saves one every `save_steps` (or per-epoch with `save_strategy="epoch"`), and keeps all of them unless `save_total_limit` caps the count, or `save_strategy="best"` keeps only the single best-scoring checkpoint by `metric_for_best_model` [14].
- `save_only_model` (default `False`): "Save only model weights, not optimizer/scheduler/RNG state. Significantly reduces checkpoint size but prevents resuming training from the checkpoint. Use when you only need the trained model for inference, not continued training. You can only load the model using `from_pretrained` with this option set to `True`" [18]. Checkpoint resume "requires optimizer and scheduler state files in the checkpoint directory. If those files are missing (for example, when `save_only_model=True`), the optimizer restarts from scratch" [14].
- `trainer.save_model(output_dir)` "will save the model, so you can reload it using `from_pretrained()`" (main process only); `trainer.push_to_hub()` uploads the same artifacts to the Hub [4][19].
- Resume: `trainer.train(resume_from_checkpoint=True)` resumes "from the latest checkpoint in `output_dir`"; passing a path string resumes from that specific checkpoint [14].
- JIT (preemption-safe) checkpointing is a distinct mechanism from periodic saving: setting `enable_jit_checkpoint=True` makes `Trainer` save a checkpoint the moment it receives SIGTERM, finishing the in-flight step first; a sentinel file `checkpoint-is-incomplete.txt` marks a save that was interrupted before completing, and `Trainer` "doesn't check for it automatically," so a reader must check for it before resuming from that directory [14].
- PEFT changes what a save is, the same way it does for trl (see that card): `Trainer._save` calls `model.save_pretrained()`, and for a `peft.PeftModel` that call is PEFT's own override, which writes only the adapter - `adapter_config.json` and `adapter_model.safetensors`, "the extra PEFT weights that were trained," not a full model directory [5][20]. Reload pairs the adapter with its base model, e.g. `AutoPeftModelForCausalLM.from_pretrained(...)`, and `Trainer` itself special-cases resuming a PEFT checkpoint (`model.load_adapter(...)`) rather than loading a full state dict [5][20].
- Whether an evaluator can load what you saved is the loader's contract, not `transformers`'; a plain `checkpoint-<step>/` directory from a full (non-PEFT) run is directly loadable through `from_pretrained()` per the save-model docstring above [19]; an adapter directory is not.

## Find it in the docs

The docs are the live source; this section teaches the lookup pattern, not the content.

- Address pattern: `https://huggingface.co/docs/transformers/<version>/en/<page>`. `<version>` is `main` (development docs) or a release tag written as `v<X.Y.Z>` - checked 2026-08-07: `/docs/transformers/v5.14.0/en/index` loads, `/docs/transformers/v5.14.1/en/index` returns 404 (the docs build for the latest patch release had not gone live at the time of this reading, so `main` or the prior tag `v5.14.0` are the working fallbacks) [21]. `<page>` is a snake_case slug, e.g. `trainer`, `training`, `installation`, `accelerate`, `ddp`, `fsdp`, `deepspeed`.
- Question-to-slug map: how to start a fine-tuning run -> `training`; the `Trainer`/`TrainingArguments` API reference -> `main_classes/trainer`; distributed backends -> `accelerate` (the unified launcher), then `ddp`, `fsdp`, `deepspeed` for the backend-specific config; worked recipes for logging, checkpointing, memory-efficient eval, and custom losses -> `trainer_recipes`; hooking into training events -> `trainer_callbacks`; installing -> `installation`.
- A caveat on templates versus rendered pages, since it changes what a raw fetch shows: `docs/source/en/trainer.md` in the repository is a short template pointing at `training`, `trainer_customize`, `data_collators`, and `trainer_callbacks` rather than the full content [22] - the substantive content for a start-a-run walkthrough sits in `training.md`, and for logging/checkpointing recipes in `trainer_recipes.md`, not in `trainer.md` itself.
- Runnable references beyond the docs: the `examples/pytorch` tree in the repository, covering NLP, vision, audio, and other tasks with `Trainer`-based scripts [4][3]; the `notebooks/README.md` file in the repository lists Hugging Face's own Colab/AWS-Studio/AMD-Dev-Cloud notebooks, including a `Trainer` fine-tuning walkthrough, and points to a separate community-notebooks page for content curated from outside Hugging Face [23].
- Community layer: the official notebooks list above is the curated door; it defers community-authored notebooks to `https://hf.co/docs/transformers/community#community-notebooks` rather than embedding them, so check that page's own curation (and each notebook's pinned `transformers` version against yours) before trusting a community walkthrough [23].
- MCP: Hugging Face's official MCP server at `https://huggingface.co/mcp` ships an `hf_fs` tool that lets a connected assistant search the Hub and its documentation semantically, covering these docs pages as well as models and datasets [24].

Honest boundary: `transformers` ships no preference, RLHF, or online-generation trainer of its own - if the job needs DPO, GRPO, PPO, or any method that samples completions during training, that method-specific loss and generation-layout logic lives in trl (or another post-training library built on top of this one), not here [1][4].

## Sources

All pages are `main`-version docs unless a tag is named, and are unpinned, mutable pages read on 2026-08-07 unless another date is given. Ecosystem tools named in passing (Accelerate, DeepSpeed, FSDP2, PyTorch, PEFT, trl) are reached through the cited pages' own links and are not separately enumerated as references beyond what is cited below.

[1] transformers documentation index. https://huggingface.co/docs/transformers/main/en/index. Fetched 2026-08-07.

[2] transformers on PyPI. https://pypi.org/pypi/transformers/json. Fetched 2026-08-07.

[3] transformers GitHub repository. https://github.com/huggingface/transformers. Fetched 2026-08-07.

[4] transformers fine-tuning tutorial. https://huggingface.co/docs/transformers/main/en/training (source: docs/source/en/training.md on GitHub). Fetched 2026-08-07.

[5] transformers Trainer source at the v5.14.1 release commit. https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/trainer.py. Fetched 2026-08-07.

[6] transformers Accelerate integration guide. https://huggingface.co/docs/transformers/main/en/accelerate. Fetched 2026-08-07.

[7] transformers DDP guide. https://huggingface.co/docs/transformers/main/en/ddp. Fetched 2026-08-07.

[8] transformers DeepSpeed ZeRO guide. https://huggingface.co/docs/transformers/main/en/deepspeed. Fetched 2026-08-07.

[9] transformers FSDP2 guide. https://huggingface.co/docs/transformers/main/en/fsdp. Fetched 2026-08-07.

[10] transformers installation page. https://huggingface.co/docs/transformers/main/en/installation. Fetched 2026-08-07.

[11] transformers PyPI license classifier / LICENSE. https://pypi.org/pypi/transformers/json. Fetched 2026-08-07.

[12] transformers build metadata (`setup.py`) at the v5.14.1 tag. https://raw.githubusercontent.com/huggingface/transformers/v5.14.1/setup.py. Fetched 2026-08-07.

[13] transformers GitHub releases feed. https://api.github.com/repos/huggingface/transformers/releases. Fetched 2026-08-07.

[14] transformers Trainer features recipes (logging, checkpointing, resume, JIT checkpointing, memory-efficient evals). https://huggingface.co/docs/transformers/main/en/trainer_recipes (source: docs/source/en/trainer_recipes.md). Fetched 2026-08-07.

[15] GitHub issue #7232, "trainer.evaluate() aggregates predictions on GPU and causes CUDA out of memory issues for large datasets"; maintainer reply by `sgugger` (COLLABORATOR), 2020-09-18. https://github.com/huggingface/transformers/issues/7232. Fetched 2026-08-07.

[16] transformers TrainingArguments API reference (`report_to` field and its supported integrations). https://huggingface.co/docs/transformers/main/en/main_classes/trainer. Fetched 2026-08-07.

[17] transformers Callbacks guide (`EarlyStoppingCallback`). https://huggingface.co/docs/transformers/main/en/trainer_callbacks (source: docs/source/en/trainer_callbacks.md). Fetched 2026-08-07.

[18] transformers TrainingArguments API reference (`save_only_model` field). https://huggingface.co/docs/transformers/main/en/main_classes/trainer. Fetched 2026-08-07.

[19] transformers Trainer API reference (`save_model` and `push_to_hub` methods). https://huggingface.co/docs/transformers/main/en/main_classes/trainer. Fetched 2026-08-07.

[20] peft quicktour (adapter-only save files, `AutoPeftModel` reload). https://huggingface.co/docs/peft/main/en/quicktour (source: docs/source/quicktour.md on GitHub). Fetched 2026-08-07.

[21] Version-tag check for the transformers docs address pattern (`v5.14.0` loads, `v5.14.1` 404s). https://huggingface.co/docs/transformers/v5.14.0/en/index and https://huggingface.co/docs/transformers/v5.14.1/en/index. Checked 2026-08-07.

[22] transformers Trainer overview page source, a short template pointing elsewhere. https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/trainer.md. Fetched 2026-08-07.

[23] transformers notebooks index (official notebooks list; pointer to the community-notebooks page). https://raw.githubusercontent.com/huggingface/transformers/main/notebooks/README.md. Fetched 2026-08-07.

[24] Hugging Face MCP server docs. https://huggingface.co/docs/hub/en/hf-mcp-server (the endpoint itself, https://huggingface.co/mcp, renders only its settings page in a browser). Fetched 2026-08-07.

[25] trl `_BaseTrainer` source (confirms `SFTTrainer` subclasses trl's `_BaseTrainer`, which subclasses `transformers.Trainer`). https://github.com/huggingface/trl/blob/2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0/trl/trainer/base_trainer.py and https://github.com/huggingface/trl/blob/2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0/trl/trainer/sft_trainer.py. Fetched 2026-08-07.

[26] transformers Callback API reference (`EarlyStoppingCallback`'s stated dependency on `load_best_model_at_end`). https://huggingface.co/docs/transformers/main/en/main_classes/callback. Fetched 2026-08-07.
