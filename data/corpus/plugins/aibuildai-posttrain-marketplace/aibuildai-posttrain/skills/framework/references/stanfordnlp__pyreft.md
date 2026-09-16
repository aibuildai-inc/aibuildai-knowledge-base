# pyreft

A Stanford NLP research library for training and sharing Representation Finetuning (ReFT) interventions on Hugging Face models - pick it for parameter-efficient, single-GPU-first activation editing, not as a general multi-method post-training framework.

**pyreft** is described by its GitHub repository's own metadata as a "Stanford NLP Python library for Representation Finetuning (ReFT)" [2]. It is built and maintained by the Stanford NLP group, with the setup metadata naming Zhengxuan Wu as the package author [16], and it sits on top of `pyvene`, the group's own intervention library, which supplies the `IntervenableModel`/`IntervenableConfig` base classes that pyreft's `ReftModel` and `ReftConfig` subclass [3]. Its one-sentence API shape: describe which layer/token positions to intervene on in a `ReftConfig`, wrap a loaded Hugging Face model with `pyreft.get_reft_model(model, reft_config)`, and train the result with a `pyreft.ReftTrainer*` subclass of the `transformers.Trainer` [4]. It lives at https://github.com/stanfordnlp/pyreft [5].

**When to pick it**: you want to reproduce or extend the ReFT/LoReFT line of parameter-efficient fine-tuning - editing hidden representations at chosen layers/token positions instead of adding weight deltas like LoRA - on a Hugging Face causal-LM or sequence-classification model, and you are fine training on a single GPU as the primary supported path. The README's own worked example trains a rank-4 intervention on one Llama-2-7B-chat layer with 32,772 trainable intervention parameters against 6,738,415,616 total model parameters (0.00049% trainable) [4], which is the concrete evidence behind the parameter-efficiency pitch. It is not a wide multi-method framework: the package ships one method family (ReFT, in six intervention variants) plus supervised and sequence-classification trainers, and the DPO example folder wraps `trl`'s own `DPOTrainer` rather than adding a pyreft-native preference-optimization trainer [7] - weigh a library like trl instead if you need a broad menu of RLHF/preference methods (cross-reference; not covered here).

**Methods it ships**: `pyreft/__init__.py` exports six intervention classes - `LoreftIntervention`, `NoreftIntervention`, `ConsreftIntervention`, `LobireftIntervention`, `DireftIntervention`, `NodireftIntervention` - all imported directly from the top-level `pyreft` package, with no experimental/stable split documented [8]. It also exports three trainers: `ReftTrainerForCausalLM` (supervised causal-LM training), `ReftTrainerForCausalLMDistributed` (adds a `DistributedSampler` and restricts checkpoint saving to rank 0), and `ReftTrainerForSequenceClassification` (regression/single-label/multi-label losses plus a custom `evaluate()` loop) [9]. The `examples/reward` folder does not use that shipped classification trainer; it defines its own local `ReftTrainerForRewardModelling(ReftTrainer)` subclass [10]. The `examples/dpo` folder similarly does not add a pyreft trainer: `DPOReftTrainer` there subclasses `trl.DPOTrainer` and overrides three methods to route through the ReFT-wrapped model: `concatenated_forward`, `get_batch_loss_metrics` (which computes the DPO loss and reward metrics by calling the overridden `concatenated_forward` with a `reference=` keyword argument that trl's base class signature does not have), and `save_model` [7]; a maintainer (`frankaging`, repo member) confirmed in a closed issue that DPO support was added this way, as an example rather than a core trainer [11]. This export list is a live file and can move - check `pyreft/__init__.py` at the tag you install [8].

**Scale it handles**: single GPU is the primary, best-documented path - the paper-reproduction guide under `examples/loreft` states plainly that "ReFT only supports a single GPU for now" for those reproduction scripts and instructs setting `CUDA_VISIBLE_DEVICES=0` [12]. A separate instruction-tuning example (`examples/alpaca`) documents a multi-GPU path instead, launched with `torchrun` and Hugging Face Accelerate's FSDP through an `fsdp_config.json`, and the shipped `ReftTrainerForCausalLMDistributed` class exists to support it (distributed sampler, rank-0-only saves) [13][9]. No multi-node benchmark is published; the only documented cost figure is a single-GPU one: the base Alpaca run "will take less than 15 mins on a single A100 (40G MEM) GPU" [13].

**Install**: `pip install pyreft`; version 0.1.0, released 2025-02-04 [14]; Python floor `>=3.8` [14]; licence is listed as "Apache License 2.0" in the package's PyPI license field and as Apache-2.0 on the GitHub repository [2][14], but the same PyPI release's trove classifiers still read "License :: OSI Approved :: MIT License" - a metadata mismatch, so trust the Apache-2.0 declaration on the repository over the stale classifier [14]. Version 0.1.0 is also the newest GitHub release, tagged at commit `f51f4e5c92ddfab234c70d038b8e0c0231ce995e`; reading `requirements.txt` at that tag gives the load-bearing pins: `torch>=2.0.0` (unpinned upper bound), `transformers==4.45.1` (an exact pin - both floor and ceiling, so it can collide with any other library in the same environment wanting a different transformers version), and `pyvene>=0.1.7`, the intervention engine pyreft's model and config classes subclass [15][3]. No CUDA or other hardware minimum is stated in `setup.py`, `requirements.txt`, or the README read for this card [16][15][1].

**Maintained by**: the Stanford NLP group (`stanfordnlp` GitHub org); about 1,577 stars, not a ranking signal [2]. The `main` branch's newest commit is `dafd0995a366d7b47160a337dcc388eda7431821`, dated 2025-02-06 - a checkpoint-resume merge on top of the v0.1.0 release [17][15] - while the repository's overall last-push timestamp is 2026-03-05T11:16:52Z, which lands on the non-`main` branch `aryaman/regret` (tip commit `8abb51a8`, dated 2026-03-05T11:16:51Z) rather than advancing `main` itself [2][18]. So as of 2026-08-12, `main` itself has been stable for roughly a year and a half even though the repository is not archived and at least one other branch shows a more recent push [2][18].

## Quick start

The smallest complete run, from the README's own step-by-step (quoted with light joining of its five code blocks) [4]:

```python
import torch, transformers, pyreft

prompt_no_input_template = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

%s [/INST]
"""

model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048,
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

reft_config = pyreft.ReftConfig(representations={
    "layer": 15, "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=4)})
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device("cuda")
reft_model.print_trainable_parameters()

training_examples = [["Who are you?", "[emoji string, 4 chars]"], ["Who am I?", "[emoji string, 4 chars]"]]  # 10 pairs in the full README example; targets are short emoji sequences, replaced here to avoid rendering emoji on this card
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model, [prompt_no_input_template % e[0] for e in training_examples],
    [e[1] for e in training_examples])

training_args = transformers.TrainingArguments(
    num_train_epochs=100.0, output_dir="./tmp", per_device_train_batch_size=10,
    learning_rate=4e-3, logging_steps=20)
trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)
_ = trainer.train()
```

There is no separate CLI; the README's Alpaca example gives the equivalent script-with-flags form:

```bash
python train.py --model_name_or_path yahma/llama-7b-hf \
    --data_path ./alpaca_data.json --output_dir ./test/ \
    --layers "8;19" --rank 4 --position "f1+l1" \
    --num_train_epochs 1 --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 --learning_rate 2e-5 --logging_steps 1
```
[13]

## Start it

- One GPU: run the quickstart above as-is; this is the path both the README's toy example and the `examples/loreft` paper-reproduction scripts use [4][12].
- Multiple GPUs go through `torchrun` plus an Accelerate-style FSDP config, documented only in `examples/alpaca/README.md`: `torchrun --nproc_per_node=<n> --master_port=<port> train.py ... --fsdp_config ./fsdp_config.json`, with a full example `fsdp_config.json` given on that page (`FULL_SHARD`, `SHARDED_STATE_DICT`, `fsdp_transformer_layer_cls_to_wrap: "LlamaDecoderLayer"`) [13]. The shipped `ReftTrainerForCausalLMDistributed` trainer pairs with this: it wraps the train dataloader in a `DistributedSampler` and gates `save_model` to rank 0 [9].
- Effective batch size follows the standard `transformers.TrainingArguments` arithmetic (per-device batch x world size x gradient-accumulation steps); the alpaca FSDP command uses `per_device_train_batch_size=4` with `gradient_accumulation_steps=8` [13].
- Config surface: pyreft does not define its own training-arguments dataclass and does not override any `transformers.TrainingArguments` default - `ReftTrainerForCausalLM` and its siblings pass `args` straight through to `transformers.Trainer.__init__` [9]. The `torch.bfloat16` dtype in the quickstart is the user's own choice when loading the base model, not a library-imposed default [4]. The library-specific surface is `ReftConfig(representations=...)`, which names the layer, `component` (e.g. `"block_output"`), token positions, `low_rank_dimension`, and the intervention class to train [4].
- Out-of-memory first aid documented in the README: load the base model 4-bit quantized via `transformers.BitsAndBytesConfig` (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`) before wrapping it with `get_reft_model` [4]; for FSDP runs, the example config sets `fsdp_cpu_ram_efficient_loading: true` to reduce peak host/GPU memory while sharding [13]. Since interventions add only a few tens of thousands of parameters, memory pressure comes almost entirely from the base model, not from pyreft's own training state.
- pyreft has no separate generation-time engine: `reft_model.generate(...)` in the README's chat step calls straight into the wrapped Hugging Face model's own `generate`, passing `unit_locations` to say where the trained intervention should fire [4]; there is no vLLM/SGLang-style generation-layout choice to make.

## Watch it

This section covers the mechanics only; pyreft publishes no method-specific guidance on what a healthy loss or reward curve looks like.

- Logging is plain `transformers.Trainer` behavior: `ReftTrainerForCausalLM` and its siblings never set `report_to`, so whatever the base `TrainingArguments` default resolves to (auto-detecting installed integrations, or nothing if none are installed) applies unchanged [9]. The paper-reproduction runs under `examples/loreft` were logged to Weights & Biases and the maintainers published the resulting run pages (commonsense, math, and instruction-following logs) [19].
- No pyreft-specific metric names are published beyond what `transformers.Trainer` already logs (training loss, learning rate, and so on) - `reft_trainer.py` adds no custom `self.log(...)` calls in the causal-LM path [9]. `ReftTrainerForSequenceClassification.evaluate()` is the one custom path: it runs a full pass over `eval_dataset`, calls the user-supplied `compute_metrics`, and logs the result with an `eval_` key prefix [9]. The `examples/dpo/dpo_trainer.py` script logs trl-style preference metrics by name - `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `logps/chosen`, `logps/rejected`, `logits/chosen`, `logits/rejected` - because it is a thin subclass of `trl.DPOTrainer`, not because pyreft defines these [7].
- No sample-level generation logging was found in `pyreft/reft_trainer.py` or the README read for this card - training only reports scalar loss/eval metrics, and inspecting generations means calling `reft_model.generate(...)` yourself as shown in the README's Step 5 [4][9].
- Evaluation during training is the standard `transformers.TrainingArguments` `evaluation_strategy`/`eval_steps` surface (the Alpaca example passes `--evaluation_strategy "no"` by default) [13], plus the custom `evaluate()` override on `ReftTrainerForSequenceClassification` described above [9].
- No stopping-rule or health-limit threshold is published: `pyreft/reft_trainer.py`, `pyreft/config.py`, the top-level `README.md`, and both `examples/alpaca/README.md` and `examples/loreft/README.md` were searched for this card and none states an early-stopping or divergence threshold [9][15][1][13][12].
- A published trap: Hugging Face Trainer's TensorBoard callback can crash pyreft training with `TypeError: Object of type type is not JSON serializable`, tracked as issue #69, which was closed 2024-04-29 [20]. A related issue, #70, carries the title prefix "[P2]" (a maintainer triage convention in the title text, not a GitHub label - the issue's only actual label is `bug`) and was still open as of this card's 2026-08-12 fetch [20]. The workaround stated by maintainer `frankaging` (repo member) in a different closed issue is to pass `report_to="none"` on `TrainingArguments` [21].

## Save it

- `ReftTrainer.save_model(output_dir)` writes to `f"{output_dir}/intervenable_model"` by calling `self.model.save_intervention(save_directory=target_dir, include_model=True)`, which comes from the underlying `pyvene.IntervenableModel` [9]. If that target directory already exists and is non-empty, the save is skipped with a logged warning rather than overwriting it [9]. `ReftTrainerForCausalLMDistributed.save_model` additionally gates this behind `dist.get_rank() == 0` [9].
- `LoreftIntervention` (and the other trainable interventions) override `state_dict()`/`load_state_dict()` explicitly "for data-efficiency" [22], keeping only the small `learned_source` linear layer and the low-rank `rotate_layer` weight rather than any base-model tensors - which is why `include_model=True` is needed on save if you want the base weights captured alongside the intervention in the same directory.
- Resuming: `trainer.train(resume_from_checkpoint=True)` (or a specific checkpoint path) triggers `Trainer._load_from_checkpoint`, which pyreft overrides to call `model.load_intervention(f"{resume_from_checkpoint}/intervenable_model", include_model=True)` [9]. This resume path was added in a "checkpointing" pull request merged 2025-02-06 and is the newest commit on `main` as of this card [17].
- Outside the Trainer, the README's top-level sharing API is `reft_model.save(save_directory=..., save_to_hf_hub=True, hf_repo_name="...")` to push to the Hub, and `pyreft.ReftModel.load("./reft_to_share", model)` to reload - but that reload call takes an already-loaded base `model` as its second argument, meaning the saved artifact is intervention-only, comparable to a PEFT adapter directory, and is NOT a full standalone model directory: the README's own "Generic ReFT model loading" section first loads the raw base model from `meta-llama/Llama-2-7b-chat-hf` via `transformers.AutoModelForCausalLM.from_pretrained` before calling `ReftModel.load` [4].
- Loader handoff: an evaluator can only reload what you saved if they have the exact same base model (same id/revision) available locally or on the Hub, plus `pyreft`/`pyvene` installed, to construct the `model` object that `ReftModel.load` pairs the intervention weights with - a bare `intervenable_model` directory on its own is not loadable into a usable model.

## Find it in the docs

pyreft has no separate hosted documentation site; the GitHub repository's `README.md`, the `examples/` subfolder READMEs, and the source files under `pyreft/` are the docs.

- Address pattern: `https://github.com/stanfordnlp/pyreft/tree/<ref>` for browsing, and `https://raw.githubusercontent.com/stanfordnlp/pyreft/<ref>/<path>` for a specific file at a tag, branch, or commit - checked 2026-08-12 by fetching `README.md` and `requirements.txt` at both the tag `v0.1.0` and the commit `dafd0995a366d7b47160a337dcc388eda7431821`; both forms resolve [1][15].
- There is no page-slug system to learn; the practical recipe is "open the README section" or "open the matching `examples/<name>/README.md`". Question-to-directory map from the top-level README's own example table [6]: instruction-tuning -> `examples/alpaca`; multi-GPU/FSDP -> `examples/alpaca/README.md`'s "Multi-GPU with FSDP" section [13]; reproducing the ReFT paper's commonsense/math/GLUE/instruction numbers -> `examples/loreft/README.md` [12]; reward modeling -> `examples/reward`; safety guardrails -> `examples/safety`; why ReFT is interpretable/composable -> `examples/composition` and `examples/memorisation` [1]. The README's example table does not mention DPO at all; that folder is only discoverable from the repository's own `examples/` directory listing or from the DPO source file and issue thread this card already cites [23][7][11].
- Runnable references beyond the README: a Colab-hosted `main_demo.ipynb`, linked from the README's "Getting Started" badge under the label "ReFT with TinyLlama" [1]; the `examples/` tree itself, with subfolders `agent`, `alpaca`, `chat`, `composition`, `dpo`, `icl`, `jailbreak`, `loreft`, `memorisation`, `overhead`, `peft`, `plots`, `reward`, `safety` [23]. Known-good smoke-test data: the 10-pair emoji list embedded directly in the README's Step 3 code block [4], and `alpaca_data.json` for the Alpaca example [13].
- Community layer: pyreft's README curates no separate tutorials page; the closest official curation is its own "Learn more through other examples" table and three linked Hugging Face Spaces demos - `pyvene/reft_emoji_chat`, `pyvene/reft_ethos`, `pyvene/reft_chat7b_1k` [1]. No third-party blog series recurring in an official curation, and no official MCP endpoint, was found for this card.
- Honest boundary: beyond the single-GPU-only statement on the paper-reproduction path [12], the closed-issue history shows real friction on saved-config compatibility across versions - a maintainer (`frankaging`, repo member) walked a user through hand-editing a broken `representations` field in a saved `config.json` in 2024, and confirmed the underlying serialization bug was only fixed by installing from `main` at the time (issue #51, closed 2024-04-18) [21].

## Sources

All GitHub source files are read at commit `dafd0995a366d7b47160a337dcc388eda7431821` (the repository's `main`-branch HEAD and the shortlist's screening commit) unless a release tag is named; the v0.1.0 release's own `requirements.txt` is read at its tag commit `f51f4e5c92ddfab234c70d038b8e0c0231ce995e`, which is dated earlier than the screening commit above. All fetches for this card were made 2026-08-12.

[1] pyreft README. https://github.com/stanfordnlp/pyreft/blob/dafd0995a366d7b47160a337dcc388eda7431821/README.md. Fetched 2026-08-12.

[2] GitHub API, stanfordnlp/pyreft repository metadata (description, license, stars, pushed_at). https://api.github.com/repos/stanfordnlp/pyreft. Fetched 2026-08-12.

[3] pyreft/config.py and pyreft/reft_model.py (ReftConfig and ReftModel subclass pyvene.IntervenableConfig / pyvene.IntervenableModel). https://raw.githubusercontent.com/stanfordnlp/pyreft/dafd0995a366d7b47160a337dcc388eda7431821/pyreft/config.py and .../pyreft/reft_model.py. Fetched 2026-08-12.

[4] pyreft README, step-by-step Emoji-Chatbot quickstart (code blocks, trainable-parameter printout, generation and save/load calls). Same URL as [1]. Fetched 2026-08-12.

[5] pyreft GitHub repository (item home). https://github.com/stanfordnlp/pyreft. Fetched 2026-08-12.

[6] pyreft README, "Learn more through other examples" table. Same URL as [1]. Fetched 2026-08-12.

[7] examples/dpo/dpo_trainer.py (DPOReftTrainer subclasses trl.DPOTrainer). https://raw.githubusercontent.com/stanfordnlp/pyreft/dafd0995a366d7b47160a337dcc388eda7431821/examples/dpo/dpo_trainer.py. Fetched 2026-08-12.

[8] pyreft/__init__.py (intervention export list). https://raw.githubusercontent.com/stanfordnlp/pyreft/dafd0995a366d7b47160a337dcc388eda7431821/pyreft/__init__.py. Fetched 2026-08-12.

[9] pyreft/reft_trainer.py (ReftTrainerForCausalLM, ReftTrainerForCausalLMDistributed, ReftTrainerForSequenceClassification: save_model, _load_from_checkpoint, evaluate, no report_to override, no custom scalar logging). https://raw.githubusercontent.com/stanfordnlp/pyreft/dafd0995a366d7b47160a337dcc388eda7431821/pyreft/reft_trainer.py. Fetched 2026-08-12.

[10] examples/reward/train.py (local ReftTrainerForRewardModelling subclass, not the shipped ReftTrainerForSequenceClassification). https://raw.githubusercontent.com/stanfordnlp/pyreft/dafd0995a366d7b47160a337dcc388eda7431821/examples/reward/train.py. Fetched 2026-08-12.

[11] GitHub issue #66, "[P0] Adding DPO Support" (closed 2024-05-05), maintainer replies from frankaging and aryamanarora (both repo members) describing the DPO example as an add-on. https://api.github.com/repos/stanfordnlp/pyreft/issues/66 and .../issues/66/comments. Fetched 2026-08-12.

[12] examples/loreft/README.md ("ReFT only supports a single GPU for now" statement; paper-reproduction commands). https://raw.githubusercontent.com/stanfordnlp/pyreft/dafd0995a366d7b47160a337dcc388eda7431821/examples/loreft/README.md. Fetched 2026-08-12.

[13] examples/alpaca/README.md (single-GPU and FSDP multi-GPU commands, fsdp_config.json, "less than 15 mins on a single A100" cost figure). https://raw.githubusercontent.com/stanfordnlp/pyreft/dafd0995a366d7b47160a337dcc388eda7431821/examples/alpaca/README.md. Fetched 2026-08-12.

[14] PyPI JSON API for pyreft (version, release date, requires_python, license field, trove classifiers, requires_dist). https://pypi.org/pypi/pyreft/json. Fetched 2026-08-12.

[15] pyreft requirements.txt at the v0.1.0 release tag/commit f51f4e5c92ddfab234c70d038b8e0c0231ce995e (version-pinned dependencies). https://raw.githubusercontent.com/stanfordnlp/pyreft/f51f4e5c92ddfab234c70d038b8e0c0231ce995e/requirements.txt. Fetched 2026-08-12.

[16] pyreft setup.py (no hardware/CUDA minimum stated). https://raw.githubusercontent.com/stanfordnlp/pyreft/dafd0995a366d7b47160a337dcc388eda7431821/setup.py. Fetched 2026-08-12.

[17] GitHub API, recent commits on stanfordnlp/pyreft (main-branch HEAD dafd0995a3, dated 2025-02-06, "Add resume checkpointing support" #151). https://api.github.com/repos/stanfordnlp/pyreft/commits?per_page=10. Fetched 2026-08-12.

[18] GitHub API, stanfordnlp/pyreft branch list (main HEAD vs. many non-main feature branches). https://api.github.com/repos/stanfordnlp/pyreft/branches and https://api.github.com/repos/stanfordnlp/pyreft/branches/main. Fetched 2026-08-12.

[19] examples/loreft/README.md, links to published Weights & Biases run logs for commonsense, math, and Ultrafeedback reproduction runs. Same URL as [12]. Fetched 2026-08-12.

[20] GitHub issue #69 (closed 2024-04-29, TensorBoard-callback `TypeError`) and issue #70, "[P2] Pyreft tensorboard integration" (open as of fetch, references issue #69, labelled only `bug`). https://api.github.com/repos/stanfordnlp/pyreft/issues/69 and https://api.github.com/repos/stanfordnlp/pyreft/issues/70. Fetched 2026-08-12.

[21] GitHub issue #51, "[P0] Saving and reloading a ReftModel throws an error" (closed 2024-04-18), maintainer frankaging's report_to="none" workaround and config.json hand-fix walkthrough. https://api.github.com/repos/stanfordnlp/pyreft/issues/51 and .../issues/51/comments. Fetched 2026-08-12.

[22] pyreft/interventions.py (LoreftIntervention.state_dict/load_state_dict overridden "for data-efficiency", keeping only learned_source and rotate_layer). https://raw.githubusercontent.com/stanfordnlp/pyreft/dafd0995a366d7b47160a337dcc388eda7431821/pyreft/interventions.py. Fetched 2026-08-12.

[23] GitHub API, contents of the examples/ directory. https://api.github.com/repos/stanfordnlp/pyreft/contents/examples?ref=dafd0995a366d7b47160a337dcc388eda7431821. Fetched 2026-08-12.
