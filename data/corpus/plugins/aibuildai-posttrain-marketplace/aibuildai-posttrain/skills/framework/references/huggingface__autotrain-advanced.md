# autotrain-advanced

Hugging Face's no-code wrapper over transformers/trl/peft/bitsandbytes for LLM post-training via a YAML config, CLI, or UI - now declared unmaintained by its own maintainers in favor of the libraries it wraps.

**AutoTrain Advanced**, "developed by Hugging Face, is a robust no-code platform designed to simplify the process of training state-of-the-art models across multiple domains: Natural Language Processing (NLP), Computer Vision (CV), and even Tabular Data analysis" [1]. It is built and maintained by Hugging Face [1][2], and its API shape is one config object (`LLMTrainingParams`, populated from a YAML file, CLI flags, or the UI) that is dispatched to a trainer function per task - `train_clm_sft.py`, `train_clm_dpo.py`, `train_clm_orpo.py`, `train_clm_reward.py` - each of which builds the matching trl trainer class (`SFTTrainer`, `DPOTrainer`, `ORPOTrainer`, `RewardTrainer`) and calls `.train()` [3]. It lives at https://github.com/huggingface/autotrain-advanced [4].

**When to pick it**: pick it only if you already have a YAML config or UI workflow built on it, because both the repository README and the live docs index carry the same maintainer warning: "This project is no longer maintained. No new features will be added and bugs will not be fixed. We recommend using Axolotl, TRL, or transformers.Trainer" [1][4]. For any new post-training work its own maintainers point you at trl directly (cross-reference; not covered here) - this card's own trainer files show autotrain's SFT/DPO/ORPO/Reward paths already just call the matching trl trainer class [3], so choosing trl means the same trl trainer classes without the YAML/config-file indirection layer, from the actively developed sibling this card's own source recommends.

**Methods it ships**: four LLM post-training trainers, selected via the config's `trainer` field [3][5]: `sft` (`SFTTrainer`), `dpo` (`DPOTrainer`), `orpo` (`ORPOTrainer`), and `reward` (`RewardTrainer`), plus a `default` causal-LM trainer that is not a post-training method. Each is a thin call-through: `train_clm_sft.py` builds `SFTConfig(**training_args)` and calls `SFTTrainer(args=args, model=model, ...).train()` with no autotrain-specific loss code, so method behavior is trl's, not autotrain's [3]. DPO and Reward additionally handle 4-bit/8-bit `BitsAndBytesConfig` quantization and PEFT LoRA wrapping inline in the trainer file before constructing the trl trainer [6][7]. There is no separate live "methods" index page for this library; the four names above are read directly from the trainer-selection code at commit 1873aca [3].

**Scale it handles**: single machine only - the launcher always passes `--num_machines 1` to `accelerate launch`, so there is no multi-node path in the code, documented or not [8]. GPU count is auto-detected via `torch.cuda.device_count()`: zero GPUs forces `accelerate launch --cpu` with a logged warning that this "will be super slow"; one GPU uses a fixed single-process `accelerate launch --num_machines 1 --num_processes 1`; more than one GPU defaults to `accelerate launch --multi_gpu --num_processes <n>` (DDP), or, if the config's `distributed_backend` field is set to `"deepspeed"`, DeepSpeed ZeRO stage 3 with `--offload_optimizer_device none --offload_param_device none --zero3_save_16bit_model true` [8]. Unsloth (`config.unsloth`, default `False`) is only wired into the `default` and `sft` trainers, not into DPO, ORPO, or Reward [9]. No published multi-GPU or DeepSpeed benchmark numbers were found in the README or the docs pages read for this card [1][10][11][12].

**Install**: `pip install autotrain-advanced`; latest PyPI release is 0.8.36, uploaded 2025-01-21 [13]; Apache-2.0 [2][13]. The wheel's own METADATA carries no `Requires-Python` field at all, so the Python floor is documented only in prose: the README states plainly that "you will need python >= 3.10 for AutoTrain Advanced to work properly" [4]. Torch is not installed by autotrain either way, but the two pages word it differently: the README says "You also need to install torch, torchaudio and torchvision" [4], while the live quickstart page states, in a note the README does not carry verbatim, that "AutoTrain doesn't install pytorch, torchaudio, torchvision, or any other large dependencies. You will need to install them separately" [10]. Version floors pinned by the 0.8.36 wheel's `Requires-Dist` are exact-equals, not ranges: `transformers==4.48.0`, `accelerate==1.2.1`, `peft==0.14.0`, `trl==0.13.0`, `huggingface-hub==0.27.0`, `torchmetrics==1.6.0`, `datasets[vision]~=3.2.0`, and `bitsandbytes==0.45.0` on Linux only [14]. These pins are identical across the wheel's `base`, `dev`, `docs`, and `quality` extras [14]. No CUDA minimum version is stated in the package metadata; the README's conda recipe installs `pytorch-cuda=12.1` as a worked example, not a stated floor [4]. The repository has no GitHub Releases and no git tags at all (`releases.json` and `tags.json` both return empty arrays), so there is no tag to resolve the 0.8.36 release to a commit; the HEAD commit read for this card, 1873aca349c88684e83c8fd3d79a1c638cfbe636 (pushed 2026-07-21), carries `__version__ = "0.8.37.dev0"` in `src/autotrain/__init__.py` [15][16] - a full year ahead of what `pip install autotrain-advanced` actually delivers, and its `requirements.txt` (47 version-pinned lines) describes that unreleased state, not the installed one - every version number in it is identical to the 0.8.36 wheel's pins, the only differences being cosmetic package-name spellings (`huggingface_hub` vs `huggingface-hub`, `rouge_score` vs `rouge-score`) [17].

**Maintained by**: Hugging Face [2][4]; the repository is not archived and the last push (1873aca, 2026-07-21) is recent, but the maintainers' own "no longer maintained" notice on both the README and the docs index means recent pushes are not a sign of active feature work [1][4].

## Quick start

The README's worked example fine-tunes SmolLM2-1.7B-Instruct with SFT using a YAML config plus the CLI, quoted from the repository README at commit 1873aca [4]:

```yaml
task: llm-sft
base_model: HuggingFaceTB/SmolLM2-1.7B-Instruct
project_name: autotrain-smollm2-finetune
log: tensorboard
backend: local

data:
  path: HuggingFaceH4/no_robots
  train_split: train
  valid_split: null
  chat_template: tokenizer
  column_mapping:
    text_column: messages

params:
  block_size: 2048
  model_max_length: 4096
  epochs: 2
  batch_size: 1
  lr: 1e-5
  peft: true
  quantization: int4
  target_modules: all-linear
  padding: right
  optimizer: paged_adamw_8bit
  scheduler: linear
  gradient_accumulation: 8
  mixed_precision: bf16
  merge_adapter: true

hub:
  username: ${HF_USERNAME}
  token: ${HF_TOKEN}
  push_to_hub: true
```

```bash
$ export HF_USERNAME=<your_hugging_face_username>
$ export HF_TOKEN=<your_hugging_face_write_token>
$ autotrain --config <path_to_config_file>
```

The docs' own config-file page carries a second example that runs ORPO on `meta-llama/Meta-Llama-3-8B-Instruct` against `argilla/distilabel-capybara-dpo-7k-binarized`, setting `params.trainer: orpo` and `data.chat_template: chatml` in the same YAML shape [11]. DPO and Reward configs follow the identical file shape with `trainer: dpo` or `trainer: reward` and the column mapping their data format requires (below) [5].

## Start it

- One GPU: `autotrain --config path/to/config.yaml` is the whole invocation; the library detects `torch.cuda.device_count() == 1` internally and launches `accelerate launch --num_machines 1 --num_processes 1 -m autotrain.trainers.clm --training_config <project>/training_params.json` for you - there is no separate multi-step launch form [8][18]. No GPU falls back to `accelerate launch --cpu` with a logged "will be super slow" warning [8].
- Two or more GPUs on one machine: still `autotrain --config ...` - the same auto-detection appends `--multi_gpu --num_machines 1 --num_processes <n>` (DDP) automatically, or, if the config sets `params.distributed_backend: deepspeed`, `--use_deepspeed --zero_stage 3 --offload_optimizer_device none --offload_param_device none --zero3_save_16bit_model true --zero3_init_flag true --gradient_accumulation_steps <n>` [8]. There is no user-facing accelerate config template in this repository to point at - the flags above are generated in code, not read from a YAML template [8].
- Effective batch size is `per_device_train_batch_size (= params.batch_size) x devices x gradient_accumulation`; the config's `auto_find_batch_size` field (default `False`) can search downward from `batch_size` on OOM instead of you retuning it by hand [19][20].
- Data required per trainer, from the docs' data-preparation page [5]: SFT/`default` need a `text` column; Reward needs `text` (chosen) plus `rejected_text`; DPO/ORPO need `prompt`, `text` (chosen), and `rejected_text`. Both CSV and JSONL are accepted [5].
- Config surface: `LLMTrainingParams` is a pydantic model, not a `TrainingArguments` subclass; it changes several trainer defaults silently the moment training starts - `quantization` defaults to `"int4"` (a CUDA/bitsandbytes assumption the moment `peft: true` is set and `quantization` is left unset), `mixed_precision` defaults to `None` (neither fp16 nor bf16 forced on), and `eval_strategy` defaults to `"epoch"` but is silently overridden to `"no"` in code whenever `valid_split` is unset (see "Save it") [19][21].
- Out-of-memory first aid, read from the parameter docs and the config-building code together, since no single OOM-tuning page exists [22][21]: lower `batch_size` and raise `gradient_accumulation` to hold the effective batch constant; set `peft: true` with `quantization: int4` or `int8` to run under bitsandbytes 4/8-bit quantization; lower `block_size` and, for DPO/ORPO, `max_prompt_length`/`max_completion_length` (the docs warn these two "cannot be greater than" `block_size` or `model_max_length`, or you get an error or NaN losses) [22]; `disable_gradient_checkpointing` (default `False`, i.e. checkpointing is on by default) should stay off to save memory [19].

## Watch it

This section covers the mechanics only; what a logged loss curve or KL value means for a given method lives on that method's own card (SFT/DPO/ORPO/Reward), not here.

- Enable it via the config's `log` field (default `"none"` - no tracker runs unless you opt in), passed straight through as trl/transformers' `report_to`; the README and docs quickstart examples both set `log: tensorboard` [4][19][10].
- What gets logged is whatever the underlying trl trainer (`SFTTrainer`/`DPOTrainer`/`ORPOTrainer`/`RewardTrainer`) reports through `Trainer.log()` - autotrain adds no custom metric names of its own. Its only logging-side addition is `LossLoggingCallback`, which drops the `total_flos` key from each log dict and re-emits the rest through its own logger on the local process [23]. For the exact metric names each trainer reports, consult that trainer's own trl docs page at `https://huggingface.co/docs/trl/main/en/<method>_trainer` (`sft_trainer`, `dpo_trainer`, `orpo_trainer`, `reward_trainer`) - autotrain's own docs do not restate them [24].
- Sample-level generation logging (e.g. printing sampled completions) is not implemented anywhere in the CLM trainer code read for this card; there is no autotrain equivalent of trl's `log_completions` [3][23].
- Evaluation during training only runs if the config sets `valid_split`: `eval_strategy` (default `"epoch"`) and `save_strategy` are forced together to `"no"` whenever `valid_split is None`, so leaving the default unset silently disables both eval and checkpointing, not just one of them [21].
- `logging_steps` (default `-1`) is auto-computed when left at `-1`: 20% of the validation-set length (or training-set length if there is no validation split), divided by `batch_size`, clamped to the range 1-25 [25].
- No stopping-rule or reward/loss threshold is published anywhere in the README, the docs pages read for this card, or the trainer/config code - the search covered the README, the docs quickstart, config, and LLM-finetuning-parameters pages, and `params.py`/`utils.py` in `src/autotrain/trainers/clm/`, and none defines an early-stopping field [4][10][19][21][22].

## Save it

- `trainer.save_model(config.project_name)` writes the trained model into the config's `project_name` directory once training ends; a model card is written alongside it as `README.md` in the same directory [26].
- Checkpointing during training is controlled by `save_strategy`/`save_total_limit` (default `1`), but - as noted in "Watch it" - `save_strategy` is force-set to `"no"` whenever `config.valid_split` is `None` (the default), so with no validation split configured, no intermediate checkpoints are written at all during the run; only the final `post_training_steps` save happens [21][26].
- With PEFT (`peft: true`), the on-save callback (`SavePeftModelCallback`, only attached when PEFT is on and DeepSpeed is not in use) calls `model.save_pretrained(checkpoint_folder)`, which for a PEFT-wrapped model writes adapter-only weights, and additionally writes an empty placeholder `pytorch_model.bin` via `torch.save({}, pytorch_model_path)` into the same checkpoint folder - that placeholder file is not the model and will not load as one [27][19].
- Whether the final PEFT adapter gets merged into a full model is controlled by `merge_adapter` (default `False`). When `peft and merge_adapter` are both true, `post_training_steps` calls a `merge_adapter()` helper that reloads the base model in fp16, resizes embeddings, applies `PeftModel.from_pretrained(model, adapter_path)`, and calls `merge_and_unload()`, then deletes the `adapter_*` files from the output directory - if this step raises, the code logs a warning ("Failed to merge adapter weights... Skipping adapter merge. Only adapter weights will be saved.") and leaves you with an adapter-only directory instead of a full model [28].
- `push_to_hub` (default `False`) is hardcoded to `False` inside the trl/transformers `TrainingArguments` dict built by `configure_training_args`, and handled manually afterward instead: when the config's own `push_to_hub` is `True`, `post_training_steps` calls `HfApi().create_repo(..., private=True, exist_ok=True)` followed by `api.upload_folder(...)` - pushed repos are created **private** by default [21][26].
- Resume-from-checkpoint is not wired into any of the four CLM trainer files read for this card (`train_clm_sft.py`, `train_clm_dpo.py`, `train_clm_orpo.py`, `train_clm_reward.py`); none of them passes `resume_from_checkpoint` to `.train()` [3].
- Loader handoff: a full (non-PEFT, or PEFT-and-merged) `project_name` directory is a standard `transformers` save and loads via `AutoModelForCausalLM.from_pretrained(project_name)`; a PEFT-only directory (merge skipped or `merge_adapter: false`) is NOT a full model and needs the base model plus `PeftModel.from_pretrained(base, project_name)` (or an unloosely-typed adapter loader) to reconstruct the fine-tuned model - the same PEFT reload pattern used across this pipeline's other library cards.

## Find it in the docs

The docs are the live source; this section teaches the lookup, not the content.

- Address pattern: `https://huggingface.co/docs/autotrain/<version>/en/<page>`. `<version>` is `main` (source-install docs) or a release tag in the form `v<X.Y.Z>`; checked 2026-08-10, both `/docs/autotrain/main/en/index` and `/docs/autotrain/v0.8.24/en/index` load. The version dropdown on the docs pages itself lists `main, v0.8.24, v0.7.129, v0.6.48, v0.5.2` - note v0.8.24 is the latest **stable-tagged doc build**, itself behind the 0.8.36 PyPI release, a separate mismatch from the pip/HEAD mismatch described in Install [10][1].
- Page-slug map for the pages read here: `index` (what/why AutoTrain), `quickstart` (local install and CLI), `config` (the YAML config walkthrough), `llm_finetuning` (per-trainer data formats), `llm_finetuning_params` (per-trainer CLI/config parameter list, including the block-size/prompt-length/completion-length constraints quoted in "Start it") [1][10][5][22].
- Runnable references beyond the docs: the repository's `configs/llm_finetuning/` directory holds one example YAML per task (SFT, ORPO, DPO, Reward, generic), linked from the README's Supported Tasks table together with a Colab notebook (`notebooks/llm_finetuning.ipynb`) shared across all four LLM tasks [4].
- No community-tutorials page equivalent to trl's `community_tutorials` slug was found among the pages read for this card; the docs index instead lists a fixed set of maintainer-authored walkthroughs (extractive QA, PaliGemma finetuning, object detection, custom embedding models, SpaceRunner, phi-3 on a MacBook, Mixtral 8x7B, DGX Cloud/H100) with no per-post author or date metadata on the index page itself [1].
- The honest boundary: the maintainers' own warning is the boundary that matters most for a chooser - "no new features will be added and bugs will not be fixed," with Axolotl, trl, and `transformers.Trainer` named as the replacements [1][4]. No GitHub issue search was performed for this card, so no maintainer-reply trap is reported here.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. All docs pages are `main`-version unless a tag is named, unpinned and mutable; every docs and PyPI/GitHub-API reading is a 2026-08-10 fetch, and every source-code claim is read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636, the repository's newest push at screening time - which the Install field notes is a year ahead of the 0.8.36 release `pip install` actually delivers. Method names (SFT, DPO, ORPO, Reward) are deliberately cited to nothing here: their defining papers live on the methodology cards.

[1] AutoTrain docs index. https://huggingface.co/docs/autotrain/main/en/index. Fetched 2026-08-10.

[2] autotrain-advanced on PyPI (author/license fields). https://pypi.org/pypi/autotrain-advanced/json. Fetched 2026-08-10.

[3] `src/autotrain/trainers/clm/train_clm_sft.py`, `train_clm_dpo.py`, `train_clm_orpo.py`, `train_clm_reward.py`. https://github.com/huggingface/autotrain-advanced/blob/1873aca349c88684e83c8fd3d79a1c638cfbe636/src/autotrain/trainers/clm/. Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[4] autotrain-advanced GitHub repository and README. https://github.com/huggingface/autotrain-advanced. Fetched 2026-08-10 (repo API); README read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[5] AutoTrain LLM finetuning data-formats page. https://huggingface.co/docs/autotrain/main/en/llm_finetuning. Fetched 2026-08-10.

[6] `src/autotrain/trainers/clm/train_clm_dpo.py` (BitsAndBytesConfig/LoraConfig construction). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[7] `src/autotrain/trainers/clm/train_clm_reward.py` (BitsAndBytesConfig/LoraConfig construction). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[8] `src/autotrain/commands.py` (`get_accelerate_command`, `CPU_COMMAND`, `SINGLE_GPU_COMMAND`, DDP/DeepSpeed branches). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[9] `src/autotrain/trainers/clm/utils.py` (`get_model`, unsloth gating to `default`/`sft` trainers only). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[10] AutoTrain quickstart page. https://huggingface.co/docs/autotrain/main/en/quickstart. Fetched 2026-08-10.

[11] AutoTrain config-file page (the ORPO/Meta-Llama-3-8B-Instruct config-walkthrough example). https://huggingface.co/docs/autotrain/main/en/config. Fetched 2026-08-10.

[12] AutoTrain LLM finetuning parameters page. https://huggingface.co/docs/autotrain/main/en/llm_finetuning_params. Fetched 2026-08-10.

[13] autotrain-advanced PyPI JSON API (latest version, upload date). https://pypi.org/pypi/autotrain-advanced/json. Fetched 2026-08-10.

[14] autotrain-advanced 0.8.36 wheel METADATA (`Requires-Dist` pins, extras, no `Requires-Python`). https://files.pythonhosted.org/packages/18/fa/81e7f46e903ade3955ef8143aa12fae6a59fca67b13e76aa02df821d02ff/autotrain_advanced-0.8.36-py3-none-any.whl. Downloaded 2025-01-21T07:53:31Z (release upload), inspected 2026-08-10.

[15] GitHub Releases API for huggingface/autotrain-advanced (empty result - no releases). https://api.github.com/repos/huggingface/autotrain-advanced/releases. Fetched 2026-08-10.

[16] GitHub Tags API for huggingface/autotrain-advanced (empty result - no tags); `src/autotrain/__init__.py` `__version__` string. https://api.github.com/repos/huggingface/autotrain-advanced/tags and https://raw.githubusercontent.com/huggingface/autotrain-advanced/1873aca349c88684e83c8fd3d79a1c638cfbe636/src/autotrain/__init__.py. Fetched/read 2026-08-10.

[17] `requirements.txt` at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636 (unreleased 0.8.37.dev0 state). https://raw.githubusercontent.com/huggingface/autotrain-advanced/1873aca349c88684e83c8fd3d79a1c638cfbe636/requirements.txt. Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[18] `src/autotrain/cli/__main__.py` and `src/autotrain/trainers/clm/__main__.py` (training-config entrypoint invoked by the generated accelerate command). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[19] `src/autotrain/trainers/clm/params.py` (`LLMTrainingParams` field defaults: `quantization`, `mixed_precision`, `disable_gradient_checkpointing`, `log`, `auto_find_batch_size`, `unsloth`, `distributed_backend`). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[20] `src/autotrain/trainers/clm/utils.py` (`configure_training_args`: `per_device_train_batch_size`, `gradient_accumulation_steps`, `auto_find_batch_size` wiring). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[21] `src/autotrain/trainers/clm/utils.py` (`configure_training_args`: `eval_strategy`/`save_strategy` forced to `"no"` when `valid_split is None`; `post_training_steps`: `push_to_hub` hardcoded `False` in `training_args`, manual `HfApi` push with `private=True`). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[22] AutoTrain LLM finetuning parameters page (block_size/model_max_length/max_prompt_length/max_completion_length constraints and error/NaN-loss warning). https://huggingface.co/docs/autotrain/main/en/llm_finetuning_params. Fetched 2026-08-10.

[23] `src/autotrain/trainers/common.py` (`LossLoggingCallback`, `UploadLogs`, `TrainStartCallback`). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[24] trl trainer docs pages, checked reachable 2026-08-10: https://huggingface.co/docs/trl/main/en/sft_trainer, /dpo_trainer, /orpo_trainer, /reward_trainer. Fetched 2026-08-10 (existence/URL-pattern check only; metric lists not restated here).

[25] `src/autotrain/trainers/clm/utils.py` (`configure_logging_steps`). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[26] `src/autotrain/trainers/clm/utils.py` (`post_training_steps`: `trainer.save_model`, model-card write, Hub push). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[27] `src/autotrain/trainers/clm/callbacks.py` (`SavePeftModelCallback.on_save`). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.

[28] `src/autotrain/trainers/clm/utils.py` (`merge_adapter` function; `post_training_steps` merge-failure warning branch). Read at commit 1873aca349c88684e83c8fd3d79a1c638cfbe636.
