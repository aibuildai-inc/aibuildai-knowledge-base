# ludwig

A YAML-driven training framework that covers tabular, multimodal, and LLM fine-tuning/alignment in one config schema, scaling from a laptop to a Ray cluster without code changes.

Ludwig is "a **declarative deep learning framework** that lets you train, fine-tune, and deploy AI models — from LLM fine-tuning to tabular classification — using a YAML config file and zero boilerplate Python" [1]. It is hosted by the Linux Foundation AI & Data [1]. The core API is a `LudwigModel` class built from a YAML/dict config, whose `.train()`, `.predict()`, `.save()`, and `LudwigModel.load()` methods drive the whole lifecycle; a `ludwig` CLI wraps the same operations as subcommands (`train`, `predict`, `evaluate`, `serve`, `export_model`, ...) [2]. It lives at https://github.com/ludwig-ai/ludwig [3].

**When to pick it**: config-first training where the same YAML schema spans tabular/multimodal supervised models and LLM fine-tuning or alignment, and you want distributed scale-out (Ray, Accelerate-backed DDP/FSDP/DeepSpeed) without touching training-loop code [1][4]. Its LLM alignment trainers (DPO, KTO, ORPO, GRPO) are one `trainer.type` value each, not separate libraries to learn [5]. Weigh it against a trainer-class library like trl when your workflow is LLM-only and Hub-native (cross-reference; not covered here), or against a cluster-native RL library like verl when the workload is online RL at cluster scale with Ray-placed generation/training workers (cross-reference; not covered here).

**Methods it ships**: supervised training (`trainer.type: finetune` for LLMs, as used in the getting-started fine-tuning example's config and registered at the pinned commit as `@register_llm_trainer_schema("finetune")`, plus the ECD path for tabular/multimodal models, built from separate encoder/combiner/decoder components per Ludwig's own architecture description) [6][7][8], and four LLM preference/alignment trainers reached the same way — DPO, KTO, ORPO, GRPO — selected by `trainer.type: dpo` (or `kto`, `orpo`, `grpo`) [1][5]. The alignment doc's own comparison table: DPO needs `prompt`/`chosen`/`rejected` columns and a `beta` KL-penalty hyperparameter; KTO needs `prompt`/`response`/boolean `label` columns; ORPO needs the same paired columns as DPO but no reference model; GRPO needs `prompt` plus a reward function and a `num_generations` rollout count [5]. At the pinned commit, the registered config field names diverge from those doc names: `DPOTrainerConfig` exposes `dpo_beta` (default 0.1) and the trainer reads `getattr(self.config, "dpo_beta", 0.1)`, not `beta`; `KTOTrainerConfig` exposes only `kto_beta` (default 0.1) and has no `desirable_weight`/`undesirable_weight` field at all, despite the alignment page naming those as KTO's key hyperparameter; `ORPOTrainerConfig` exposes `orpo_beta`; `GRPOTrainerConfig` exposes `grpo_beta` (default 0.04), `grpo_epsilon` (default 0.2), and `grpo_num_generations` (default 4) [7][9]. The `dpo` trainer module's own docstring config example writes `beta: 0.1` under `trainer:`, the same name the alignment page uses [9] — so copying either example's key verbatim leaves the trainer silently on its default rather than the value the reader typed, at commit `5d19f3e632124e2f92266faca773047aeeeb0da4` [9][7]. Method math and defining papers are not restated here.

**Scale it handles**: single machine (local backend, any GPU count via `--gpus`) up to a Ray cluster; distributed training goes through the Ray backend for cluster provisioning and Dask for out-of-memory data processing, with the actual training step delegated to HuggingFace Accelerate for DDP/FSDP/DeepSpeed [4]. `backend.trainer.strategy` defaults to `auto`, which the docs say picks DDP for most models and switches to FSDP or DeepSpeed only for models that do not fit in GPU memory [4] — no published benchmark accompanies that auto-selection claim. Launch is `ray up cluster.yaml` then `ray submit cluster.yaml ludwig train --config config.yaml --dataset s3://...`, or `ludwig train ... --backend ray` against an already-running cluster [4]; prebuilt `ludwigai/ludwig-ray-gpu` Docker images are the documented worker image [4].

**Install**: `pip install ludwig`; PyPI lists version 0.17.8, uploaded 2026-07-27 [10], matching the GitHub release `v0.17.8` published 2026-07-27T00:35:54Z [11]. Requires Python >=3.12 [10]; licensed Apache-2.0 [11]. Resolving the `v0.17.8` tag gives commit `de1d65bbf3530fbd33c99abb21d1a1b9c3a64bad` [12], one commit behind the shortlist's pinned `5d19f3e632124e2f92266faca773047aeeeb0da4` [13] — so code claims below that cite the pinned commit reflect a state one commit ahead of what `pip install ludwig` currently delivers. At that release commit, `pyproject.toml` (identical at both commits) pins `torch>=2.11`, `torchvision>=0.26`, `torchaudio>=2.11`, `transformers>=5.0`, and `safetensors>=0.4` as core dependencies, with no upper bound on any of them [14]. The `llm` extra (`pip install ludwig[llm]`) adds `accelerate`, `peft>=0.10.0`, `torchao>=0.17.0`, `sentence-transformers`, `faiss-cpu`, `loralib`; the `distributed` extra adds `ray[default,data,serve,tune]>=2.9`; a `full` extra bundles `llm`, `distributed`, `serve`, `viz`, `hyperopt`, `explain`, `benchmarking` [14]. No CUDA or hardware-minimum version is stated in `pyproject.toml` or on the installation docs page; hardware requirements flow through the pinned `torch` release [14][15].

**Maintained by**: the Linux Foundation AI & Data [1], with GitHub showing 30 releases; the newest three (v0.17.6, v0.17.7, v0.17.8) landed 2026-06-26, 2026-07-04, and 2026-07-27 (8 and 23 days apart), preceded by a 28-day gap after a burst of six releases within three weeks in May 2026 (v0.16.2 through v0.17.5, 2026-05-08 to 2026-05-29) - an uneven cadence rather than a fixed interval [11].

## Quick start

Tabular example, from the getting-started training page [16]:

```yaml
# rotten_tomatoes.yaml
input_features:
  - name: genres
    type: set
    preprocessing:
      tokenizer: comma
  - name: content_rating
    type: category
  - name: top_critic
    type: binary
  - name: runtime
    type: number
  - name: review_content
    type: text
    encoder:
      type: embed
output_features:
  - name: recommended
    type: binary
```

```bash
ludwig train --config rotten_tomatoes.yaml --dataset rotten_tomatoes.csv
```

```python
from ludwig.api import LudwigModel
import pandas
df = pandas.read_csv("rotten_tomatoes.csv")
model = LudwigModel(config="rotten_tomatoes.yaml")
results = model.train(dataset=df)
```

LLM instruction-tuning (QLoRA) example, from the getting-started LLM fine-tuning page, which the page states was tested on a 12 GiB VRAM GPU [6]:

```yaml
# model.yaml
model_type: llm
base_model: meta-llama/Llama-2-7b-hf
quantization:
  bits: 4
adapter:
  type: lora
prompt:
  template: |
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:
input_features:
  - name: prompt
    type: text
output_features:
  - name: output
    type: text
trainer:
  type: finetune
  learning_rate: 0.0001
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate_scheduler:
    warmup_fraction: 0.01
preprocessing:
  sample_ratio: 0.1
```

```bash
pip install ludwig ludwig[llm]
ludwig train --config model.yaml --dataset "ludwig://alpaca"
```

DPO alignment example, from the alignment doc's Python form [5]:

```python
import logging
import yaml
from ludwig.api import LudwigModel

config = yaml.safe_load("""
model_type: llm
base_model: meta-llama/Llama-3.1-8B
adapter:
  type: lora
  r: 16
  alpha: 32
  dropout: 0.05
trainer:
  type: dpo
  epochs: 1
  learning_rate: 5.0e-7
  batch_size: 2
  gradient_accumulation_steps: 8
  beta: 0.1
input_features:
  - name: prompt
    type: text
output_features:
  - name: chosen
    type: text
backend:
  type: local
""")
model = LudwigModel(config=config, logging_level=logging.INFO)
results = model.train(dataset="data/train.csv")
```

(This is the doc's own example; per the Methods section above, the `beta: 0.1` line under `trainer:` does not reach the registered `dpo_beta` field, so the run keeps the default beta of 0.1 regardless of what value is written there [5][7][9].)

## Start it

- One process, any local GPU count: the CLI/Python forms above, with `--gpus` (CLI) or `gpus=` (`LudwigModel.train`) to restrict which GPUs are used [2].
- Multiple machines: enable the Ray backend either via `backend: {type: ray, processor: {type: dask}}` in the config or `ludwig train ... --backend ray`, against a cluster started with `ray up cluster.yaml`; a partial cluster-config template (head node type, GPU worker node type, Docker image) is shown on the distributed-training doc [4]. Runs are then submitted with `ray submit cluster.yaml ludwig train --config config.yaml --dataset s3://mybucket/dataset.parquet` [4]. In an autoscaling cluster where GPUs may not yet exist at check time, the docs recommend pinning worker count explicitly with `backend.trainer.num_workers` and `backend.trainer.use_gpu: true` [4].
- Sharding is delegated to HuggingFace Accelerate, chosen via `backend.trainer.strategy` (`auto`, `ddp`, `fsdp`, or `deepspeed`); `auto` is the default and the docs describe it as using DDP for most models and switching to FSDP/DeepSpeed only for models that do not fit in memory, without a published benchmark for that switch point [4].
- Config surface: the trainer's own defaults page lists `use_mixed_precision: false` and `compile: false` as the ECD/LLM-finetune trainer defaults [17] — mixed precision is off by default, unlike the bf16-on-by-default posture some peer libraries take, so a reader relying on ambient GPU speed should set it explicitly. `effective_batch_size` and `gradient_accumulation_steps` both default to `auto` in the same schema [17].
- Out-of-memory first aid: for LLM fine-tuning, the getting-started QLoRA example itself uses `batch_size: 1` with `gradient_accumulation_steps: 16` alongside 4-bit `quantization` and a LoRA `adapter` on a stated 12 GiB card [6]; no separate OOM-troubleshooting page or knob list beyond quantization/adapter/batch-size was found in the pages read for this card.

## Watch it

This section covers only the mechanics of what Ludwig emits and how to turn it on; what a metric value means for a given method lives on that method's card.

- **Enabled by default**: the CLI's own help text states "By default Ludwig saves logs for the TensorBoard" and that `-ssl`/`--skip_save_log` is the flag to turn this off — so a Ludwig run logs to TensorBoard automatically unless explicitly disabled, the opposite default from trainers that log nowhere without a tracker configured [2].
- Additional trackers are opt-in CLI flags, each requiring its own account/setup: `--mlflow` (also logs hyperopt parameters and trained models; respects `MLFLOW_TRACKING_URI` for a remote server), `--wandb`, `--comet` (requires a free Comet account), `--aim` [18].
- This card's read of the docs did not reach an enumerated per-trainer metric-name list (e.g., a page listing every scalar tag written to TensorBoard); the pages read name the loggable channels above but not individual metric names, so metric naming is not carried here.
- Sample-level generation logging (e.g., printing sample completions during LLM training) was not found in the pages read for this card.
- Evaluation during training: `--skip_all_evaluation`... is not documented on the pages read; the ECD/LLM trainer schema does list `evaluate_training_set: false` and `skip_all_evaluation: false` as defaults, plus `eval_batch_size`, `eval_steps`, and `checkpoints_per_epoch`/`steps_per_checkpoint` fields controlling how often evaluation runs [17]. The same field's description gives the number that decides whether to flip it: leaving `evaluate_training_set: false` (computing training-set metrics from the values already produced during training) is up to 30% faster than running a separate evaluation pass over the training set, but the separate pass gives less noisy training-set metrics, particularly in early epochs [17].
- Stopping: the trainer schema publishes `early_stop: 5` as a default field name and value on both the ECD and LLM-finetune trainer configs [17] — Ludwig does publish an early-stopping default, unlike libraries that leave it unset; no further description of the metric it watches or its exact trigger condition was found on the pages read for this card, so its full contract is not carried here.

## Save it

- `LudwigModel.save(save_path)` "save[s] the model config, weights, and training metadata to `save_path`", creating the directory if needed and writing `model_hyperparameters.json`, model weight files, and `training_set_metadata.json` [19].
- Training itself defaults to checkpointing: the CLI's `--skip_save_model` flag description states "By default Ludwig saves weights after each epoch the validation metric improves" (disabling it skips saving weights and only reports achievable performance), and `--skip_save_progress` states "By default ludwig saves weights after each epoch for enabling resuming of training" (disabling it saves half as much disk but drops resumability) [2]. Checkpoint internals (at the pinned commit): `ludwig/globals.py` names the subdirectory `TRAINING_CHECKPOINTS_DIR_PATH = "training_checkpoints"`, and the trainer joins it onto the save path as `training_checkpoints_path`, so checkpoints land under a `training_checkpoints/` subdirectory of the model's save directory [20][21]; within it, `checkpoint_utils.py`'s `Checkpoint`/`CheckpointManager` classes write two files per tag - metadata (optimizer/scheduler/global-step state) via pickled `torch.save`, and model weights to a companion `<path>.safetensors` file - with `tag` being `"latest"` or `"best"` [22].
- Resume: CLI flag `-mrp`/`--model_resume_path` takes "path of the model directory to resume training of" [2]; the Python API's `LudwigModel.load_weights(model_dir, from_checkpoint=True)` loads from the latest training checkpoint instead of final weights [23].
- Export formats beyond the training checkpoint: `model.export_model(path, format=...)` supports `"safetensors"` (the default, described as the format Ludwig models are saved in already), `"torch_export"` (the docs describe this as replacing the now-deprecated TorchScript path, producing an ATen-level `.pt2` `ExportedProgram`), and `"onnx"` (via `torch.onnx.export(dynamo=True)`); a separate `ludwig export_mlflow` command exports to MLflow's registry format [24]. The same operations exist as CLI subcommands (`ludwig export_model -m <dir> -o <dir> -f <format>`) [24].
- PEFT adapters: the pages read for this card did not reach a dedicated statement of whether `save()` on an adapter-trained LLM writes adapter-only weights or a merged full model; the `LudwigModel.load` API code excerpt shows a `merge_and_unload` step gated on `is_merge_and_unload_set()` and the presence of an `adapter_config.json` inside the saved weights path, implying adapter weights can be saved unmerged and merged at load time, but the on-disk adapter-only file layout itself is not enumerated in the pages read [25].
- `LudwigModel.upload_to_hf_hub(repo_id, model_path, ...)` uploads "trained model artifacts to the HuggingFace Hub" [26], and `model_path` is documented as either the folder containing `model_weights`/`model_hyperparameters.json` or its parent [26].
- Whether an evaluator can load a Ludwig save directly is the loader's own contract; `LudwigModel.load(model_dir)` is the documented reload path for a `save()` output, and `load_exported_model(...)` auto-detects and reloads the `torch_export`/ONNX export formats [24][23].

## Find it in the docs

The docs are the live source; this section is the lookup, not a mirror.

- Address pattern: `https://ludwig.ai/latest/<slug>/`, confirmed by fetching `https://ludwig.ai/latest/examples/llm/alignment/` and `https://ludwig.ai/latest/user_guide/api/LudwigModel/` on 2026-08-10, both 200 [5][23]. `latest` tracks the current release rather than a pinned version tag; no versioned-docs path (e.g., `/v0.17.8/`) was tested.
- The page-slug source of truth is the `ludwig-docs` repository's `mkdocs.yml` nav block, fetched from its `main` branch on 2026-08-10 [27]. Recipes from that nav: getting-started walkthroughs live under `getting_started/<topic>.md` (e.g. `getting_started/train.md`, `getting_started/llm_finetuning.md`, `getting_started/ray.md`); the trainer config reference is `configuration/trainer.md`; the CLI reference is `user_guide/command_line_interface.md`; the Python API is `user_guide/api/LudwigModel.md`; third-party tracker integrations are `user_guide/integrations.md`; distributed training is `user_guide/distributed_training/index.md` [27].
- Question-to-slug map: "what does `trainer:` accept and what are the defaults" -> `configuration/trainer.md` (renders the full ECD and LLM-finetune trainer default dumps) [17]; "how do I run DPO/KTO/ORPO/GRPO" -> the alignment example was reachable at `examples/llm/alignment/` even though that slug does not appear in the fetched `mkdocs.yml` nav tree, meaning it may not be linked from the site's left-hand navigation despite being live [5][27]; "how do I export a trained model" -> `user_guide/model_export.md` [24]; "how do I scale to a cluster" -> `user_guide/distributed_training/index.md` [4].
- Runnable references beyond the docs: the alignment page points to a full runnable notebook at `examples/alignment/alignment_dpo.ipynb` in the main repo, plus a `examples/alignment/prepare_dataset.py` script that downloads and reformats the Anthropic HH-RLHF dataset for DPO/ORPO or KTO [5]. The Dataset Zoo referenced from the README ships ready-made datasets addressable as `ludwig://<name>` (e.g. `ludwig://alpaca`, used in the quickstart above) [1].
- Community layer: this card's fetch did not reach a dedicated Ludwig-curated community-tutorials page (distinct from the docs nav itself); the README's own Community section instead points to a Discord server, GitHub Issues, an X/Twitter account, and a `medium.com/ludwig-ai` publication as the maintainer-run channels [1]. No MCP endpoint for the Ludwig docs was found in the pages read for this card.
- Honest boundary: a targeted GitHub issue/PR search for the DPO/KTO field-name mismatch described above found no report of it. `dpo_beta` matches only PR #4087, which originally added the DPO trainer, not a bug report; `desirable_weight` returns zero matches; `"not a valid parameter"` matches two unrelated PRs (#3981, freezing pretrained vision-model layers by regex; #3118, adding deprecation warnings for unknown config parameters) [28]. So the mismatch above is stated from the source code and docs directly, not from a maintainer-confirmed issue thread.

## Sources

[1] Ludwig README, `main` branch. https://raw.githubusercontent.com/ludwig-ai/ludwig/main/README.md. Fetched 2026-08-10.

[2] Ludwig CLI reference (`user_guide/command_line_interface.md`, rendered page). https://ludwig.ai/latest/user_guide/command_line_interface/. Fetched 2026-08-10.

[3] Ludwig GitHub repository. https://github.com/ludwig-ai/ludwig. Fetched 2026-08-10.

[4] Ludwig distributed-training guide. https://ludwig.ai/latest/user_guide/distributed_training/. Fetched 2026-08-10.

[5] Ludwig LLM alignment example page. https://ludwig.ai/latest/examples/llm/alignment/. Fetched 2026-08-10.

[6] Ludwig getting-started LLM fine-tuning page. https://ludwig.ai/latest/getting_started/llm_finetuning/. Fetched 2026-08-10.

[7] `ludwig/schema/trainer.py` at commit `5d19f3e632124e2f92266faca773047aeeeb0da4` (DPOTrainerConfig, KTOTrainerConfig, ORPOTrainerConfig, GRPOTrainerConfig, FineTuneTrainerConfig field definitions). https://raw.githubusercontent.com/ludwig-ai/ludwig/5d19f3e632124e2f92266faca773047aeeeb0da4/ludwig/schema/trainer.py. Fetched 2026-08-10.

[8] Ludwig "How Ludwig Works" page (ECD architecture). https://ludwig.ai/latest/user_guide/how_ludwig_works/. Fetched 2026-08-10.

[9] `ludwig/trainers/trainer_dpo.py` at commit `5d19f3e632124e2f92266faca773047aeeeb0da4` (module docstring config example; `getattr(self.config, "dpo_beta"/"kto_beta"/"orpo_beta"/"grpo_beta"/"grpo_epsilon"/"grpo_num_generations", ...)` reads). https://raw.githubusercontent.com/ludwig-ai/ludwig/5d19f3e632124e2f92266faca773047aeeeb0da4/ludwig/trainers/trainer_dpo.py. Fetched 2026-08-10.

[10] ludwig on PyPI (JSON API). https://pypi.org/pypi/ludwig/json. Fetched 2026-08-10.

[11] Ludwig GitHub releases (list API). https://api.github.com/repos/ludwig-ai/ludwig/releases. Fetched 2026-08-10.

[12] GitHub API commit resolution for tag `v0.17.8`. https://api.github.com/repos/ludwig-ai/ludwig/commits/v0.17.8. Fetched 2026-08-10.

[13] GitHub API compare, `v0.17.8` commit `de1d65bbf3530fbd33c99abb21d1a1b9c3a64bad` against `5d19f3e632124e2f92266faca773047aeeeb0da4` (1 commit ahead). https://api.github.com/repos/ludwig-ai/ludwig/compare/de1d65bbf3530fbd33c99abb21d1a1b9c3a64bad...5d19f3e632124e2f92266faca773047aeeeb0da4. Fetched 2026-08-10.

[14] `pyproject.toml` at commit `de1d65bbf3530fbd33c99abb21d1a1b9c3a64bad` (the `v0.17.8` release commit; identical to the `main`-branch copy read for this card, confirmed byte-for-byte). https://raw.githubusercontent.com/ludwig-ai/ludwig/de1d65bbf3530fbd33c99abb21d1a1b9c3a64bad/pyproject.toml. Fetched 2026-08-10.

[15] Ludwig installation page. https://ludwig.ai/latest/getting_started/installation/. Fetched 2026-08-10.

[16] Ludwig getting-started training page. https://ludwig.ai/latest/getting_started/train/. Fetched 2026-08-10.

[17] Ludwig trainer configuration reference (rendered ECD and LLM-finetune trainer default dumps). https://ludwig.ai/latest/configuration/trainer/. Fetched 2026-08-10.

[18] Ludwig third-party integrations page. https://ludwig.ai/latest/user_guide/integrations/. Fetched 2026-08-10.

[19] Ludwig Python API reference, `LudwigModel.save`. https://ludwig.ai/latest/user_guide/api/LudwigModel/. Fetched 2026-08-10.

[20] ludwig/globals.py at pinned commit 5d19f3e632124e2f92266faca773047aeeeb0da4 (defines `TRAINING_CHECKPOINTS_DIR_PATH = "training_checkpoints"`). https://raw.githubusercontent.com/ludwig-ai/ludwig/5d19f3e632124e2f92266faca773047aeeeb0da4/ludwig/globals.py. Fetched 2026-08-10.

[21] ludwig/trainers/trainer.py at pinned commit 5d19f3e632124e2f92266faca773047aeeeb0da4 (joins `TRAINING_CHECKPOINTS_DIR_PATH` onto the save path as `training_checkpoints_path`). https://raw.githubusercontent.com/ludwig-ai/ludwig/5d19f3e632124e2f92266faca773047aeeeb0da4/ludwig/trainers/trainer.py. Fetched 2026-08-10.

[22] `ludwig/utils/checkpoint_utils.py` at commit `5d19f3e632124e2f92266faca773047aeeeb0da4` (`Checkpoint`/`CheckpointManager` classes, `LATEST`/`BEST` tag constants, SafeTensors + pickled-metadata save format). https://raw.githubusercontent.com/ludwig-ai/ludwig/5d19f3e632124e2f92266faca773047aeeeb0da4/ludwig/utils/checkpoint_utils.py. Fetched 2026-08-10.

[23] Ludwig Python API reference, `LudwigModel.load_weights` and `LudwigModel.load`. https://ludwig.ai/latest/user_guide/api/LudwigModel/. Fetched 2026-08-10.

[24] Ludwig model export guide. https://ludwig.ai/latest/user_guide/model_export/. Fetched 2026-08-10.

[25] Ludwig Python API reference, `LudwigModel.load` source excerpt (merge-and-unload / `adapter_config.json` check). https://ludwig.ai/latest/user_guide/api/LudwigModel/. Fetched 2026-08-10.

[26] Ludwig Python API reference, `LudwigModel.upload_to_hf_hub`. https://ludwig.ai/latest/user_guide/api/LudwigModel/. Fetched 2026-08-10.

[27] `ludwig-ai/ludwig-docs` repository, `mkdocs.yml`, `main` branch (nav / slug map). https://raw.githubusercontent.com/ludwig-ai/ludwig-docs/main/mkdocs.yml. Fetched 2026-08-10.

[28] GitHub issue/PR search, repository `ludwig-ai/ludwig`: `dpo_beta` (1 result, PR #4087), `desirable_weight` (0 results), `"not a valid parameter"` (2 results, PR #3981 and PR #3118). https://api.github.com/search/issues?q=repo:ludwig-ai/ludwig+dpo_beta ; https://api.github.com/search/issues?q=repo:ludwig-ai/ludwig+desirable_weight ; https://api.github.com/search/issues?q=repo:ludwig-ai/ludwig+%22not+a+valid+parameter%22. Fetched 2026-08-10.
