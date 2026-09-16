# xTuring

A small, class-per-model Python library for fine-tuning open LLMs on one machine or through DeepSpeed, with an unreleased HEAD that has moved far past what `pip install xturing` actually delivers - read the pinned-vs-live note below before trusting anything else on this page.

**xTuring** makes it "simple, fast, and cost-efficient to fine-tune open-source LLMs ... on your own data - locally or in your private cloud" [1]. It is built and maintained by Stochastic Inc [1][8]. Its API is one call chain per model: `BaseModel.create("<model_key>")` returns a model object, `.finetune(dataset=...)` (or `.dpo_finetune(...)`) trains it, `.generate(texts=[...])` runs inference, `.evaluate(dataset=...)` scores perplexity, and `.save(path)` / `BaseModel.load(path)` round-trip it to disk [1][2]. It lives at https://github.com/stochasticai/xTuring [2].

**When to pick it**: pick xTuring only if you can build from the GitHub HEAD, not from PyPI. **The row's pinned commit (`fb16cc2b3`, 2026-03-04) is roughly two and a half years ahead of the newest actual release, `v0.1.8`, tagged at commit `fbeea1a8` and published 2023-09-07** [3][4] - and PyPI still serves that same `0.1.8` build today [5]. `pip install xturing` therefore installs code that predates Qwen3, GPT-OSS, MiniMax M2, Mamba, and DPO entirely (a keyword grep of the v0.1.8 README for `qwen`, `gpt_oss`, `mamba`, and `minimax` returns zero hits) [6]; every model and method named on this card past that point exists only if you `pip install git+https://github.com/stochasticai/xTuring@fb16cc2b3bd5dde09167a4079580f05f91b5dcda` or clone and `pip install -e .` at that commit. Against that backdrop, xTuring's case is breadth of low-precision variants per model (base/LoRA/INT8/LoRA+INT8/LoRA+INT4 templates) on a single GPU or one DeepSpeed node [7]; it has no cluster-scale launcher and no generation-server-based online RL path, so it is not a fit for multi-node or GRPO/PPO-style rollout-heavy training (cross-reference trl/verl cards for that; not covered here).

**Methods it ships**: supervised fine-tuning via `.finetune()`, run through an in-repo PyTorch Lightning trainer (`lightning_trainer.py`, `config_name = "lightning_trainer"`) that logs a single `"loss"` scalar [9]; and DPO (Direct Preference Optimization) via `.dpo_finetune()`, run through a separate in-house trainer (`dpo_trainer.py`, `config_name = "dpo_trainer"`) that deep-copies a frozen reference model and logs `"loss"` and `"reward_margin"` [10]. Both are HEAD-only, per the pinned-vs-live note above - DPO exists at the row's pinned commit as `src/xturing/trainers/dpo_trainer.py` [row `methods_seen`] but is absent from the `v0.1.8` release. LoRA is not the `peft` package: it is a vendored, Apache-2.0-licensed adaptation of an early Hugging Face PEFT codebase living at `xturing/engines/lora_engine/lora.py`, imported locally rather than from PyPI `peft` [11]. INT8/INT4 quantization goes through `bitsandbytes`, reached via model-key suffixes (`_int8`, `_lora_int8`, `_lora_kbit`) or, for a model outside the built-in registry, the `GenericLoraKbitModel` wrapper class [12][13]. There is no taxonomy page separate from the model-key table; the live list of method x precision combinations per model is the docs' Supported Models page [13].

**Scale it handles**: single CPU, single GPU, or one multi-GPU/DeepSpeed node - every `pytorch_lightning.Trainer(...)` call in both trainers hardcodes `num_nodes=1`, so there is no multi-node path in the code read for this card [9][10]. DeepSpeed is reached as a Lightning `strategy` string (`"deepspeed_stage_2"`, or `"deepspeed_stage_2_offload"` when `optimizer_name="cpu_adam"`), or by passing a `deepspeed_config_path` for a custom `DeepSpeedStrategy` [9]. The README publishes one concrete benchmark for this: fine-tuning LLaMA-7B for one epoch took 21 hours and 33.5 GB GPU / 190 GB CPU RAM with plain DeepSpeed + CPU offloading, versus 20 minutes and 23.7 GB GPU / 10.2 GB CPU RAM with LoRA + DeepSpeed, and 20 minutes and 21.9 GB GPU / 14.9 GB CPU RAM with LoRA + DeepSpeed + CPU offloading [1] - the ~60x wall-clock gap is what LoRA buys on this library's own numbers. The roadmap itself marks INT3/INT2/INT1 precision as not yet supported [1].

**Install**: `pip install xturing` [1] installs PyPI version 0.1.8, uploaded 2023-09-06, `requires_python >= 3.7` [5], Apache-2.0 licensed [14]. Its pinned core dependencies, read from `pyproject.toml` at the `v0.1.8` release tag: `torch >= 1.9.0`, `transformers == 4.31.0`, `datasets == 2.14.5`, `evaluate == 0.4.0`, `bitsandbytes == 0.41.1`, `deepspeed == 0.9.5`, `accelerate == 0.22.0`, `pytorch-lightning` (unpinned), plus an `int4` extra that raises the torch floor to `>= 2.0` [5]. The docs' own install page states only "Python 3.0+" as a floor and names no CUDA or hardware minimum anywhere on that page [15]. These pins are NOT what the pinned commit's own `pyproject.toml` asks for: at `fb16cc2b3`, the unreleased HEAD instead pins `transformers >= 4.36.0`, `deepspeed >= 0.15.1`, and a `pyarrow >= 8.0.0, < 21.0.0` cap, with the same self-reported version string `"0.1.8"` despite the code having diverged substantially from the released `0.1.8` [16] - do not use the HEAD file to size a `pip install`, and do not use the PyPI pins to size a HEAD checkout.

**Maintained by**: Stochastic Inc [1][8]; about 2,672 GitHub stars (not a ranking signal) [row `stars_raw_not_for_ranking`]; the repository is not archived and its most recent push (the row's pinned commit) is dated 2026-03-04 [row `last_push`], but the last tagged GitHub release remains `v0.1.8` from 2023-09-07 - no newer version has been formally released in over two years even though development continues on the default branch [3][4].

## Quick start

The README's own CPU-friendly first example, quoted with its own source [1]:

```python
from xturing.datasets import InstructionDataset
from xturing.models import BaseModel

# Load a toy instruction dataset (Alpaca format)
dataset = InstructionDataset("./examples/models/llama/alpaca_data")

# Start with the lightweight Qwen 0.6B LoRA checkpoint
model = BaseModel.create("qwen3_0_6b_lora")

# Fine-tune and then generate
model.finetune(dataset=dataset)
output = model.generate(texts=["Explain quantum computing for beginners."])
print(f"Model output: {output}")
```

The repository's DPO example, `examples/features/dpo/dpo_finetune.py`, is the smallest complete DPO run: it builds a `PreferenceDataset` from four `prompt`/`chosen`/`rejected` triples, creates `model = BaseModel.create("qwen3_0_6b_lora")`, runs `model.dpo_finetune(dataset=dataset, beta=0.1)`, generates a sample, then calls `model.save(...)` [17]. Both examples require the HEAD checkout, not the PyPI release, per the pinned-vs-live note above.

The docs' finetuning guide additionally shows a CLI form, `xturing finetune --model qwen3_0_6b_lora --data-dir /path/to/your/dataset` [18] - but at the row's pinned commit, `src/xturing/cli/__init__.py` registers only three subcommands, `chat`, `ui`, and `api` [19]; no `finetune` subcommand exists in the code read for this card, so that documented CLI form currently fails at that commit and `.finetune()` must be called from Python instead.

## Start it

- One process is the base form for both CPU and single-GPU runs; `LightningTrainer` branches on the process-wide `DEFAULT_DEVICE` constant, computed once at import time as `torch.device("cuda" if torch.cuda.is_available() else "cpu")` and shared by every model in that process (not a per-model setting) - if no CUDA device is visible it logs a warning and falls back to CPU automatically, with no extra flags needed [9][28].
- Multi-GPU on one node goes through DeepSpeed as a Lightning `strategy` string, selected automatically once `use_lora=True` or `use_deepspeed=True` is set by the model class: `"deepspeed_stage_2"` normally, `"deepspeed_stage_2_offload"` if `optimizer_name == "cpu_adam"`, or a `DeepSpeedStrategy(config=deepspeed_config_path)` built from a user-supplied JSON config path (`finetuning_args.deepspeed_config_path`) [9][10]. There is no separate launcher command to run - Lightning's `Trainer` spawns the worker processes itself from the single `python` invocation, and (per the note above) it caps out at `num_nodes=1`.
- Effective batch size is `batch_size x gradient_accumulation_steps`, both individually configurable Config fields; the global defaults are `batch_size: 1, gradient_accumulation_steps: 1` before any per-model override [20].
- Configuration is per-model-key YAML in `src/xturing/config/finetuning_config.yaml`, read through the model's `.finetuning_config()` accessor; the global defaults block sets `learning_rate: 1e-5, weight_decay: 0.0, warmup_steps: 50, eval_steps: 5000, save_steps: 5000, max_length: 512, num_train_epochs: 1, logging_steps: 10, max_grad_norm: 2.0, save_total_limit: 4, optimizer_name: adamw, output_dir: saved_model`, and named model keys (e.g. the `qwen3_0_6b*` and `gpt_oss_*` variants) override individual fields on top of that [20][21]. `LightningTrainer` itself changes one default silently on the LoRA/DeepSpeed code path: it passes `precision=lora_type` to `pytorch_lightning.Trainer`, and `lora_type` defaults to `16` - i.e. fp16 mixed precision, not bf16 - so a bf16-only accelerator is not this trainer's assumption on the LoRA path [9].
- DPO adds one method-specific knob at start time, `beta` (default `0.1` in `DPOTrainer`, passed explicitly as `0.1` in the example above), controlling how strongly the policy is penalized for diverging from the frozen reference model [10][17].
- Out-of-memory first aid: switch to a `_lora`, `_int8`, `_lora_int8`, or `_lora_kbit` (LoRA + INT4) model-key variant, or load an arbitrary Hub model through `GenericLoraKbitModel` for the INT4 + LoRA path [12][13]; the README's own LLaMA-7B numbers above show LoRA + DeepSpeed alone cutting GPU memory from 33.5 GB to 23.7 GB and CPU RAM from 190 GB to about 10-15 GB versus plain DeepSpeed + CPU offloading [1]. No generation-side memory knobs apply here - generation in xTuring runs through the same `transformers`-backed engine as training, not a separate serving engine.

## Watch it

This section is the mechanics only; what a `"loss"` or `"reward_margin"` curve should look like is a DPO/SFT concern for the corresponding methodology cards, not this one.

- **Enable it**: `LightningTrainer` and `DPOTrainer` both take a `logger` argument (`Union[Logger, Iterable[Logger], bool] = True`) forwarded straight into `pytorch_lightning.Trainer(..., logger=logger)`; with the default `True`, Lightning's own default `CSVLogger` writes local log files, and swapping in a `pytorch_lightning.loggers` instance (e.g. a `WandbLogger` or `TensorBoardLogger`) is how a tracker gets attached, since xTuring adds no tracker integration of its own [9][10].
- **Metric names, standard fine-tuning**: exactly one scalar, `"loss"`, logged every training step via `self.log("loss", loss.item(), prog_bar=True)` in `TuringLightningModule.training_step` [9].
- **Metric names, DPO**: two scalars, `"loss"` and `"reward_margin"` (the gap between the chosen and rejected response's implicit reward), logged from `DPOLightningModule` [10].
- **Sample-level logging of generations**: no such feature was found in `lightning_trainer.py`, `dpo_trainer.py`, or the docs pages read for this card - a run only shows the scalar loss curve unless the caller adds their own callback.
- **Evaluation during training**: not a Config field on either trainer; `.evaluate(dataset=...)` on `CausalModel` is a separate, explicit call after (or between) `.finetune()` calls that computes perplexity as `torch.exp(sum(loglikelihoods) / len(dataset))`, not an automatic periodic eval loop [22].
- **Stopping / health limits**: none found. Search run 2026-08-11 over the pages most likely to carry one - the docs' FAQ page (`/faqs`, which lists five Q&As, none about limits or thresholds) [23], the `/configuration/finetune_configure` parameter table (learning rate, batch size, epochs, save/eval steps - no patience or threshold field) [20], and both trainer source files, which add only a `LearningRateFinder` Lightning callback (triggered when `len(train_dataset) > 100`) and an optional wall-clock `Timer` callback (`max_training_time_in_secs`) - neither is a loss- or reward-based stopping rule [9]. No RL-specific or loss-threshold stopping criterion is published anywhere searched.

## Save it

- `model.save("/path/to/a/directory")` creates the directory if needed and writes, per the docs' own description, two weight files for a LoRA model: `pytorch_model.bin` holding the full model (base weights plus LoRA parameters merged into one `state_dict`), and `adapter_model.bin` holding only the LoRA parameters, alongside the base `config.json` and tokenizer files - confirmed identically in `CausalLoraEngine.save()` and `CausalLoraKbitEngine.save()`, both of which call `torch.save(self.model.state_dict(), .../pytorch_model.bin)` followed by `self.model.save_pretrained(saving_path)` for the adapter [24][25]. A plain, non-LoRA engine's `.save()` is a single `save_pretrained` call for model and tokenizer, with no adapter file [25]. `CausalModel.save()` additionally writes an `xturing.json` config file recording the model's `model_name`, its finetuning config, and its generation config [26].
- Because that `xturing.json` records `model_name`, a saved directory is self-describing: `BaseModel.load("/path/to/a/directory")` reads it and reconstructs the correct model class automatically, with no need to name the model again [26]. Loading a checkpoint that was NOT produced by xTuring's own `.save()` (e.g. a bare Hugging Face checkpoint) instead requires the `GenericModel` family, passing the path directly to `GenericModel`/`GenericLoraModel`/`GenericLoraKbitModel` [12].
- There is no separate `save_total_limit`-driven pruning behavior beyond the `save_total_limit` Config field (default `4`) passed straight into Lightning's `ModelCheckpoint(save_top_k=save_total_limit)` callback on the LoRA/DeepSpeed path [9][20]; no flag was found in the code read for this card that saves weights only while dropping optimizer/scheduler state (unlike, e.g., `trl`'s `save_only_model`).
- Resume: no explicit `resume_from_checkpoint`-style call was found on `LightningTrainer` or `DPOTrainer` in the files read for this card; `on_save_checkpoint` calls `self.model_engine.save(self.saved_path)` on every Lightning checkpoint event, and the ordinary reload path is `BaseModel.load(path)` followed by another `.finetune()`/`.dpo_finetune()` call.
- Loader handoff: an `adapter_model.bin` directory alone is NOT a full model - it must be paired with the base model, exactly as `pytorch_model.bin` already does inside the same save directory; a downstream evaluator that only understands `from_pretrained` on a full Hugging Face checkpoint can load `pytorch_model.bin` directly, but an evaluator that expects a PEFT-style adapter-only directory will not find one in xTuring's default layout, since xTuring always writes the merged full-model file alongside it [24][25].

## Find it in the docs

The docs are a Docusaurus v2.4.1 site at `https://xturing.stochastic.ai/`, server-rendered (full text is in the raw HTML, not a JS-only shell) [1][15]. There is no version-tag path segment in the URLs observed - every page fetched for this card resolves at a bare `https://xturing.stochastic.ai/<section>/<slug>` with no `/v0.1.8/` or `/main/` prefix, so the docs are unversioned relative to any given release and always show whatever is on the default branch; treat every page here as a live, unpinned snapshot fetched 2026-08-11.
- Address pattern and page-slug recipes, confirmed by fetching each: `overview/intro` (landing) [8], `overview/installation` [15], `overview/supported_models` (the model-key x variant table) [13], `overview/quickstart/load_save_models` [24], `overview/quickstart/finetune_guide` [18], `configuration/finetune_configure` (fine-tuning Config field table) [20], `configuration/inference_configure`, `advanced/anymodel` (`GenericModel` family) [12], `advanced/api_server`, `advanced/generate`, and `faqs` [23]. Bare paths without a trailing slash 301-redirect to the trailing-slash form; follow redirects (`curl -L`) when fetching.
- Question-to-slug map for the highest-traffic questions, per the FAQ page's own links [23]: "how do I fine-tune a model" -> `overview/quickstart/finetune_guide`; "how do I load a model not in the supported list" -> `advanced/anymodel`; "how do I use an existing dataset" -> the dataset-loading tutorial linked from that same page; "how do I set up xTuring to contribute" -> the repo's own contributing guide.
- Runnable references beyond the docs: the `examples/` tree in the GitHub repo, which at the pinned commit includes `examples/features/dpo/dpo_finetune.py` (the DPO quickstart quoted above), `examples/features/int4_finetuning/`, `examples/features/evaluation/`, `examples/features/dataset_generation/`, and `examples/features/generic/` [27]; the README's own known-good smoke-test data is the Alpaca-format folder at `examples/models/llama/alpaca_data` [1].
- Community layer: no curated community-tutorials page was found on this docs site (unlike, e.g., trl's `community_tutorials` page) - the docs' own pointers are the Discord server and the GitHub issue tracker, both linked from the FAQ and README, and no other official curation page was located in the pages fetched for this card [1][23].
- No official MCP endpoint for querying these docs was found in the pages read for this card.

Traps: no maintainer-reply trap from a closed GitHub issue is recorded on this card - none was searched, because GitHub's code/issue search API returned "401: Requires authentication" for the unauthenticated queries attempted, and no alternative authenticated path was available in this session; that gap in coverage should not be read as "no traps exist." The honest boundary that IS documented: the roadmap lists INT3/INT2/INT1 precision and Stable Diffusion support as open items, and the trainer code hardcodes `num_nodes=1`, so multi-node training is out of scope for this library regardless of DeepSpeed configuration [1][9][10].

## Sources

[1] xTuring README at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda. https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/README.md. Fetched 2026-08-11.

[2] xTuring GitHub repository. https://github.com/stochasticai/xTuring. Fetched 2026-08-11.

[3] xTuring GitHub releases API (latest release v0.1.8, published 2023-09-07T12:08:04Z). https://api.github.com/repos/stochasticai/xTuring/releases. Fetched 2026-08-11.

[4] xTuring GitHub tags API (v0.1.8 tag commit fbeea1a842935eb3eec2b224dc32bc8c9932677c). https://api.github.com/repos/stochasticai/xTuring/tags. Fetched 2026-08-11.

[5] xturing PyPI JSON API (version 0.1.8, requires_python, requires_dist pins, upload dates 2023-09-06). https://pypi.org/pypi/xturing/json. Fetched 2026-08-11.

[6] xTuring README at tag v0.1.8, grepped for qwen/gpt_oss/mamba/minimax with zero matches. https://raw.githubusercontent.com/stochasticai/xTuring/v0.1.8/README.md. Fetched 2026-08-11.

[7] xTuring docs, Supported Models page (naming templates for base/lora/int8/lora_int8/lora_kbit variants). https://xturing.stochastic.ai/overview/supported_models. Fetched 2026-08-11.

[8] xTuring docs landing page. https://xturing.stochastic.ai/overview/intro. Fetched 2026-08-11.

[9] xTuring lightning_trainer.py at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (TuringLightningModule logging, LightningTrainer device/strategy branching, num_nodes=1, precision=lora_type default 16, ModelCheckpoint/LearningRateFinder/Timer callbacks). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/src/xturing/trainers/lightning_trainer.py. Fetched 2026-08-11.

[10] xTuring dpo_trainer.py at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (DPOLightningModule logging, DPOTrainer defaults including beta=0.1, num_nodes=1). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/src/xturing/trainers/dpo_trainer.py. Fetched 2026-08-11.

[11] xTuring lora_engine/__init__.py and lora.py at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (vendored LoraConfig/LoraModel, Apache-2.0 header attributing origin to Hugging Face). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/src/xturing/engines/lora_engine/__init__.py and .../lora.py. Fetched 2026-08-11.

[12] xTuring docs, "Work with any model" page (GenericModel/GenericInt8Model/GenericLoraModel/GenericLoraInt8Model/GenericLoraKbitModel table and usage). https://xturing.stochastic.ai/advanced/anymodel. Fetched 2026-08-11.

[13] xTuring docs, Supported Models page (full model-key table including Mamba and Stable Diffusion as base-only). https://xturing.stochastic.ai/overview/supported_models. Fetched 2026-08-11.

[14] xTuring LICENSE file at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (Apache License 2.0). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/LICENSE. Fetched 2026-08-11.

[15] xTuring docs, Installation page ("Python 3.0+", pip install instructions, no CUDA/hardware minimum stated). https://xturing.stochastic.ai/overview/installation. Fetched 2026-08-11.

[16] xTuring pyproject.toml at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (HEAD dependency floors: transformers>=4.36.0, deepspeed>=0.15.1, pyarrow>=8.0.0,<21.0.0, version string "0.1.8"). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/pyproject.toml. Fetched 2026-08-11.

[17] xTuring examples/features/dpo/dpo_finetune.py at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda. https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/examples/features/dpo/dpo_finetune.py. Fetched 2026-08-11.

[18] xTuring docs, Finetuning Guide page (Qwen3 0.6B LoRA example and the `xturing finetune --model ... --data-dir ...` CLI form). https://xturing.stochastic.ai/overview/quickstart/finetune_guide. Fetched 2026-08-11.

[19] xTuring src/xturing/cli/__init__.py at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (registers only chat, ui, api subcommands; no finetune subcommand). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/src/xturing/cli/__init__.py. Fetched 2026-08-11.

[20] xTuring docs, Fine-tuning Configuration page (Config field table matching finetuning_config.yaml defaults). https://xturing.stochastic.ai/configuration/finetune_configure. Fetched 2026-08-11.

[21] xTuring src/xturing/config/finetuning_config.yaml at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (global defaults block and per-model-key overrides for qwen3_0_6b* and gpt_oss_* variants). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/src/xturing/config/finetuning_config.yaml. Fetched 2026-08-11.

[22] xTuring src/xturing/models/causal.py at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (CausalModel.evaluate perplexity formula). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/src/xturing/models/causal.py. Fetched 2026-08-11.

[23] xTuring docs, FAQs page. https://xturing.stochastic.ai/faqs. Fetched 2026-08-11.

[24] xTuring docs, Load and Save Models page (pytorch_model.bin vs adapter_model.bin description, BaseModel.load/save). https://xturing.stochastic.ai/overview/quickstart/load_save_models. Fetched 2026-08-11.

[25] xTuring src/xturing/engines/causal.py at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (CausalEngine.save, CausalLoraEngine.save, CausalLoraKbitEngine.save). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/src/xturing/engines/causal.py. Fetched 2026-08-11.

[26] xTuring src/xturing/models/causal.py at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (CausalModel.save/_save_config writing xturing.json; BaseModel.load/load_from_local in src/xturing/models/base.py reconstructing the model class from it). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/src/xturing/models/causal.py and .../src/xturing/models/base.py. Fetched 2026-08-11.

[27] xTuring GitHub API directory listing, examples/features/ at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda. https://api.github.com/repos/stochasticai/xTuring/contents/examples/features?ref=fb16cc2b3bd5dde09167a4079580f05f91b5dcda.

[28] xTuring src/xturing/config/__init__.py at commit fb16cc2b3bd5dde09167a4079580f05f91b5dcda (DEFAULT_DEVICE computed once from torch.cuda.is_available(), CPU-fallback warning). https://raw.githubusercontent.com/stochasticai/xTuring/fb16cc2b3bd5dde09167a4079580f05f91b5dcda/src/xturing/config/__init__.py. Fetched 2026-08-11.
