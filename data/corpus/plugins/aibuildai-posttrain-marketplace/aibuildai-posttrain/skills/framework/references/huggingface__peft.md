# peft

Hugging Face's adapter library: wrap a base model in a small `PeftConfig`, train a fraction of the parameters, and hand the result to Transformers, Diffusers, Accelerate, or trl.

**PEFT** (Parameter-Efficient Fine-Tuning) "is a library for efficiently adapting large pretrained models to various downstream applications without fine-tuning all of a model's parameters because it is prohibitively costly," fine-tuning "a small number of (extra) model parameters" while "yielding performance comparable to a fully fine-tuned model," and it is "integrated with the Transformers, Diffusers, and Accelerate libraries" [1]. It is built and maintained by Hugging Face [2][3]. Its API shape is a config-and-wrap pattern: pick a method's config class (`LoraConfig`, and others named below), call `get_peft_model(base_model, config)` to get back a `PeftModel` that a host trainer (Transformers' `Trainer`, or trl) trains like any other model [4]. It lives at https://github.com/huggingface/peft [3].

**When to pick it**: peft is the adapter layer under most Hugging Face post-training stacks - it does not run training loops itself, so pick it alongside a trainer (Transformers `Trainer`, or trl's trainer classes, which accept a `peft_config` argument directly [see the trl card]) when the goal is training a small adapter (LoRA and its many relatives, prompt/prefix tuning, or other parameter-efficient methods) instead of the full model. Its own README's numbers make the case: on an A100 80GB GPU, full fine-tuning of `bigscience/mt0-xxl` (12B params) runs out of memory, while PEFT-LoRA fits it in 56GB GPU / 3GB CPU, and PEFT-LoRA with DeepSpeed CPU offloading fits it in 22GB GPU / 52GB CPU [5]. Choose something else when the task needs full-parameter fine-tuning or a library that also owns the RL/SFT training loop (trl, verl) - peft supplies the parameter-efficient wrapper, not the loop.

**Methods it ships**: the docs' method-overview page groups them by mechanism into three families [6]: Soft Prompting (Cartridges, CPT, Llama-Adapter, P-Tuning, Prefix Tuning, Prompt Tuning), Layer Tuning (BEFT, LayerNorm Tuning, Trainable Tokens), and Adapters (AdaLoRA, AdaMSS, BOFT, C3A, DEFT, DeLoRA, FourierFT, GraLoRA, HiRA, HRA, IA3, Lily, LoHa, LoKr, LoRA - with variants BD-LoRA, DoRA, KaSA, MonteCLoRA, VeLoRA - MiSS, OFT, OSF, PEANuT, Polytropon, PSOFT, PVeRA, RandLora, RoAd, SHiRA, TinyLoRA, UniLoRA, VB-LoRA, VeRA, WaveFT, X-LoRA). A search of this method-overview page and of the LoRA package-reference page for the word "experimental" returns no matches on either [6][7]: unlike trl, peft's docs do not mark any method as experimental. This taxonomy page states it is a living document, so re-check [6] for the current list before relying on a name here. Every method above imports from the top-level `peft` package (e.g. `from peft import LoraConfig`) [4]; there is no experimental sub-namespace to route around.

**Scale it handles**: single GPU up to multi-node, launched through Accelerate - Hugging Face's launcher layer over torch distributed - the same way as trl. peft's own FSDP guide walks a full-parameter-sharded LoRA fine-tune of Llama-2-70B across 8 GPUs on one machine, generated via `accelerate config --config_file fsdp_config.yaml` and run with `accelerate launch --config_file "configs/fsdp_config.yaml" train.py --model_name_or_path "meta-llama/Llama-2-70b-hf" ...` [8]; its DeepSpeed guide gives the same pattern for ZeRO-1/2/3, documents a ZeRO-3 compatibility table that marks LoRA compatible at every stage, and separately calls out QLoRA (quantization + LoRA) with ZeRO-3 as needing extra care, pointed to its own section [9]. Both guides show mechanism (config templates, launch commands) rather than a peft-specific multi-node throughput number - no published multi-node benchmark accompanies either guide [8][9].

**Install**: `pip install peft`; version 0.20.0, released 2026-07-28 [10]. The shortlist's screening commit (`ea8ebf3`, 2026-07-31) is three days newer than the v0.20.0 release commit (`a5526d2`, 2026-07-28) [10][11] - a reader who clones at the screening commit is not on the pinned release; the fields below are read at the release tag, not the screening commit. At that tag, `python_requires=">=3.10.0"` [12], Apache-2.0 [2][12]. Load-bearing pins at v0.20.0: `torch>=1.13.0`, `accelerate>=0.21.0`, `huggingface_hub>=0.25.0`; `transformers` is listed as a dependency with no version bound at all [12][13] - a wide-open floor, so a peft install will not itself force a transformers upgrade or block an old one. No dev/training extras are pinned windows (`dev`, `test`, `docs-specific` extras exist but carry no version ranges of their own) [12]. No CUDA or GPU-hardware minimum is stated in the setup metadata [12]; the live installation page separately claims "PEFT is tested on Python 3.9+" [14], which is a stale claim against the pinned release's own 3.10 floor - resolve the conflict in the release's favor (3.10) since that is what `pip install peft==0.20.0` actually enforces.

**Maintained by**: Hugging Face [2][3]; 21,524 GitHub stars per the live repository-metadata read on 2026-08-10 (not a ranking signal) [3]. Actively developed: the v0.20.0 release landed 2026-07-28, thirteen days before this card was written, and the same live read on 2026-08-10 shows a most-recent push of 2026-08-06 [3][10].

## Quick start

The smallest complete run, quoted from the docs' own quickstart [4]:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

from peft import LoraConfig, TaskType
peft_config = LoraConfig(target_modules=["q_proj"], task_type=TaskType.CAUSAL_LM,
                          inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

from peft import get_peft_model
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
# "trainable params: 524,288 || all params: 1,236,338,688 || trainable%: 0.0424"
```

Training then goes through the Transformers `Trainer` [4]:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="your-name/meta-llama/my-llama3.2-adapter",
    learning_rate=1e-3, per_device_train_batch_size=32, per_device_eval_batch_size=32,
    num_train_epochs=2, weight_decay=0.01, eval_strategy="epoch",
    save_strategy="epoch", load_best_model_at_end=True,
)
trainer = Trainer(model=peft_model, args=training_args,
                   train_dataset=tokenized_datasets["train"],
                   eval_dataset=tokenized_datasets["test"],
                   data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
```

There is no dedicated peft CLI; the README's own quickstart uses the same `get_peft_model` call directly on a chat model (`Qwen/Qwen2.5-3B-Instruct`) before `model.save_pretrained(...)` [15].

## Start it

- One process, one GPU is the base form: the code above, as-is.
- Multi-GPU and multi-node go through Accelerate, the same launcher trl uses: `accelerate config --config_file <yaml>` once, then `accelerate launch --config_file <yaml> train.py <args>`. peft's FSDP guide ships a full example config at `configs/fsdp_config.yaml` and its own launch script `run_peft_fsdp.sh`, tested on Llama-2-70B + LoRA across 8xH100 on one machine [8]. Its DeepSpeed guide gives the equivalent for ZeRO-1/2/3, with `zero_stage`, `gradient_accumulation_steps`, `offload_optimizer_device`, `offload_param_device`, and `zero3_init_flag` as the fields `accelerate config` asks for [9].
- peft adds no config surface on top of `TrainingArguments`/the host trainer's Config; its own knobs live entirely in the method's `*Config` class (`LoraConfig`'s `r`, `lora_alpha`, `lora_dropout`, `target_modules`, etc.) [4]. There is therefore no peft-specific precision default to flag - whatever the host trainer sets (bf16, fp16, fp32) governs, and peft's troubleshooting page separately documents that starting from v0.12.0 it auto-promotes adapter weights from float16/bfloat16 to float32 "where appropriate" unless `autocast_adapter_dtype=False` is passed to `get_peft_model()` [16].
- OOM first aid: peft "makes fine-tuning parameter efficient, but not automatically memory efficient" [17] - the parameter reduction shrinks gradient and optimizer-state memory (Adam holds roughly three times the trainable-parameter count), but activation memory and the frozen base model's own footprint are unaffected by adapter choice [17]. For very large embedding/LM-head layers specifically, the "trainable tokens" method is suggested over full LoRA on those layers [17]. Quantizing the frozen base model (peft keeps the adapter itself at higher precision by default) and DeepSpeed ZeRO-3 CPU/NVMe offload are the two documented memory levers beyond the adapter method itself [9][17].
- One FSDP-specific trap, from a closed GitHub issue: combining FSDP with `model.disable_adapter()` raised a `RuntimeError` (issue #1442). Maintainer BenjaminBossan (association: MEMBER, 2024-02-12) offered a workaround manually toggling `_disable_adapters` on the tuner layers; a reporter (association: NONE, 2024-02-23) confirmed the same setup worked fine under plain DDP but not FSDP, with degraded performance under the workaround; collaborator githubnemo (association: COLLABORATOR, 2026-01-12) later noted the issue was believed at least partially resolved by a torch update and referenced PR #2692, asking anyone still hitting it to report details [18].

## Watch it

peft ships no training loop, logger, or metric names of its own - everything a run reports comes from whatever wraps the `PeftModel` (Transformers `Trainer`'s `report_to`/`logging_steps` surface, or trl's trainer-specific metrics [see the trl card]); a search of the quickstart, method-overview, and LoRA package-reference pages for logging or metric-name content turns up none on any of the three [4][6][7]. peft's own diagnostic is `print_trainable_parameters()`, called once after `get_peft_model()`, which prints the trainable-parameter count, total-parameter count, and trainable percentage as a single line - not a per-step metric, so it belongs in a startup sanity check rather than a dashboard [4].

peft publishes no stopping-rule or early-stopping threshold of its own - this is consistent with it owning no training loop; an evaluate-during-training or stop-training decision is the host trainer's contract, not peft's, and the pages read here (quickstart [4], method overview [6], memory-efficient-training guide [17]) carry none.

## Save it

- `peft_model.save_pretrained("output_dir")` writes exactly three files: `adapter_model.safetensors` (or `.bin`), `adapter_config.json`, and a `README.md` model card that "is not needed to load the model" [19]. safetensors is the default format and is described as "a secure alternative to the bin format, which is known to be susceptible to security vulnerabilities because it uses the pickle utility under the hood" [19].
- Size is the point of this format: the docs' own comparison states "a normal BERT model requires ~420MB of disk space, whereas an IA³ adapter on top of this BERT model only requires ~260KB" [19]; the quickstart's own example puts a LoRA adapter on `facebook/opt-350m` at 6.3MB, versus its separate Llama-3.2-1B example, which reports 524,288 trainable parameters out of 1,236,338,688 total rather than a file size [4].
- **A saved adapter directory is NOT a full model.** It holds only the two adapter files above - `AutoPeftModelForCausalLM.from_pretrained(...)` (or `PeftModel.from_pretrained(base_model, adapter_path)`) is required to reconstruct a runnable model, pairing the adapter with the base model id recorded in `adapter_config.json` [4][19].
- `peft_model.push_to_hub("your-name/my-llama3.2-adapter")` uploads the same three files to the Hub; the docs state plainly that "both methods only save the extra PEFT weights that were trained" [4].
- Merging is a separate, optional step, not a save-time default: `merged_model = model.merge_and_unload()` bakes the adapter into the base model's weights, after which `merged_model.save_pretrained(...)` writes a full model directory. The docs are explicit about the tradeoff: "Once merge_and_unload() is called, you get a basic model without any PEFT-specific functionality... You cannot unmerge the weights, load multiple adapters at once, disable the adapter, etc. Not all PEFT methods support merging weights." and separately, "The whole model will be much larger than the PEFT model, as it will contain all the base weights as well. But inference with a merged model should be a bit faster." [19]
- Resume/reload for training is the host trainer's contract (e.g. Transformers `Trainer`'s `resume_from_checkpoint`) - peft's own pages document reload for inference (`from_pretrained`) rather than mid-training resume [4].
- Whether an evaluator can load what you saved therefore has two answers depending on the artifact: a merged directory loads like any Transformers model with a plain `from_pretrained()`; an unmerged adapter directory does not - a plain `from_pretrained()` on it will not give you the trained model, and it instead needs `AutoPeftModel*.from_pretrained(...)` or `PeftModel.from_pretrained(base_model, adapter_path)` paired with the base model recorded in `adapter_config.json` [4][19].

## Find it in the docs

The docs are the live source; this section teaches the lookup, not the content.

- Address pattern: `https://huggingface.co/docs/peft/<version>/en/<page>`. `<version>` is `main` or a release tag in the form `v<X.Y.Z>` - checked 2026-08-10: `/docs/peft/v0.20.0/en/index` loads, while the bare `/docs/peft/0.20.0/en/index` form serves a redirect notice ("doesn't exist in v0.20.0, but exists on the main version") rather than the pinned page - always include the `v` prefix. The docs default to `main`, so a value read there can differ from an installed release; swap in your tag first.
- `<page>` slugs worth knowing directly: `index` (the library definition and links to every method's own guide) [1], `quicktour` (the get_peft_model / save / AutoPeftModel walkthrough) [4], `install` (Python/PyPI/source install forms) [14], `methods/overview` for the method taxonomy [6] - the README's own link to `conceptual_guides/adapter` is stale: fetching it on 2026-08-10 returned a 302 redirect to `package_reference/lora`, a single method's API reference, not the taxonomy page, so treat any specific conceptual-guide slug as unstable and reach the taxonomy via the index's own links instead [1][6], `accelerate/fsdp` and `accelerate/deepspeed` for multi-GPU launch recipes [8][9], `developer_guides/checkpoint` for the save-file contract [19], `developer_guides/troubleshooting` for dtype and version-mismatch fixes [16], `developer_guides/memory` for OOM guidance [17], `developer_guides/low_level_api` for `inject_adapter_in_model()` on models outside the wrapped-config flow [20].
- A `community_tutorials` slug analogous to trl's does not exist for peft: fetching it on 2026-08-10 returned a 404 page. The closest curated door instead is the method-overview page's own link to the PEFT organization's "notebook collection" Space at `https://huggingface.co/spaces/PEFT/soft-prompting`, and the README's link to a Space meant to list which base models officially support each PEFT method out of the box, `https://stevhliu-peft-methods.hf.space` - but that link is dead: fetching it on 2026-08-10 returned a 404, so skip it and rely on the method-overview page's own per-method notes instead [6][15]. A separate, maintainer-run Space compares PEFT methods interactively but is a live Gradio app rather than a static page, so no single number from it is quoted here [21].
- Runnable references beyond the docs: the repo's `examples/` tree, including `examples/sft/README.md`, `examples/sft/configs/deepspeed_config.yaml`, and `examples/sft/configs/deepspeed_config_z3_qlora.yaml` [3]. The README's own memory-comparison numbers were produced training `bigscience/T0_3B`, `bigscience/mt0-xxl`, and `bigscience/bloomz-7b1` on the `ought/raft/twitter_complaints` dataset - a documented, reproducible smoke-test shape [5].
- Practitioner blogs linked directly from the README (not from a curation page, since none exists): Philipp Schmid's "Fine-tune FLAN-T5 for chat & chain of thought with LoRA" post is cited for the T0_3B accuracy numbers above [5]; a PyTorch blog post on QLoRA + trl on a 16GB GPU, and a Hugging Face blog post on 20B-parameter RLHF on a 24GB GPU with PEFT + trl, are both linked from the README's "quantization" and "trl" sections respectively [15]. Check each post's stated peft version against v0.20.0 before trusting its exact numbers - none of these posts is dated to the current release.

## Sources

Method names (LoRA, DoRA, IA3, prompt tuning, ...) are deliberately cited to nothing here: their defining papers live on the methodology cards, not this one. All docs pages are `main`-version unless a tag is named; all readings are 2026-08-10 unless noted.

[1] peft documentation index. https://huggingface.co/docs/peft/main/en/index. Fetched 2026-08-10.

[2] peft on PyPI. https://pypi.org/pypi/peft/json. Fetched 2026-08-10.

[3] peft GitHub repository, via the GitHub API (`/repos/huggingface/peft`). https://github.com/huggingface/peft. Fetched 2026-08-10.

[4] peft quickstart. https://huggingface.co/docs/peft/main/en/quicktour. Fetched 2026-08-10.

[5] peft README, memory-and-accuracy comparison tables (T0_3B, mt0-xxl, bloomz-7b1 on ought/raft/twitter_complaints). https://raw.githubusercontent.com/huggingface/peft/ea8ebf36da98d93fdfca020ad17ab746e6994c01/README.md. Fetched 2026-08-10.

[6] peft method-overview page (method taxonomy by mechanism; experimental-marker search; notebook-collection and per-model-support Space links). https://huggingface.co/docs/peft/main/en/methods/overview. Fetched 2026-08-10.

[7] peft LoRA package-reference page (experimental-marker and logging/metric-name search). https://huggingface.co/docs/peft/main/en/package_reference/lora. Fetched 2026-08-10.

[8] peft FSDP guide (Llama-2-70B + LoRA on 8 GPUs, config template, launch command). https://huggingface.co/docs/peft/main/en/accelerate/fsdp. Fetched 2026-08-10.

[9] peft DeepSpeed guide (ZeRO-1/2/3 compatibility table, QLoRA+ZeRO-3 note, config fields). https://huggingface.co/docs/peft/main/en/accelerate/deepspeed. Fetched 2026-08-10.

[10] peft v0.20.0 release metadata, via the GitHub API (`/repos/huggingface/peft/releases`, `/repos/huggingface/peft/commits/v0.20.0`, and `/repos/huggingface/peft`). https://github.com/huggingface/peft/releases/tag/v0.20.0. Fetched 2026-08-10.

[11] peft screening commit, via the GitHub API (`/repos/huggingface/peft/commits/ea8ebf36da98d93fdfca020ad17ab746e6994c01`). https://github.com/huggingface/peft/commit/ea8ebf36da98d93fdfca020ad17ab746e6994c01. Fetched 2026-08-10.

[12] peft `setup.py` at the v0.20.0 tag (version, python_requires, install_requires, licence). https://raw.githubusercontent.com/huggingface/peft/v0.20.0/setup.py. Fetched 2026-08-10.

[13] peft PyPI JSON metadata at version 0.20.0 (requires_dist, upload date), cross-checked against [12]. https://pypi.org/pypi/peft/json. Fetched 2026-08-10.

[14] peft installation page ("PEFT is tested on Python 3.9+"; PyPI/source/editable install forms). https://huggingface.co/docs/peft/main/en/install. Fetched 2026-08-10.

[15] peft README (quickstart with Qwen2.5-3B-Instruct, Transformers `add_adapter`/`load_adapter`/`set_adapter` integration, linked practitioner blog posts, per-model-support Space link). https://raw.githubusercontent.com/huggingface/peft/ea8ebf36da98d93fdfca020ad17ab746e6994c01/README.md. Fetched 2026-08-10.

[16] peft troubleshooting guide (autocast_adapter_dtype and the v0.12.0 auto-promotion behavior). https://huggingface.co/docs/peft/main/en/developer_guides/troubleshooting. Fetched 2026-08-10.

[17] peft memory-efficient-training guide (memory breakdown, trainable-tokens suggestion, quantization note). https://huggingface.co/docs/peft/main/en/developer_guides/memory. Fetched 2026-08-10.

[18] peft GitHub issue #1442, "RuntimeError when combining FSDP and disable_adapter" (closed), comments from BenjaminBossan (MEMBER) and githubnemo (COLLABORATOR), via the GitHub API. https://github.com/huggingface/peft/issues/1442. Fetched 2026-08-10.

[19] peft checkpoint-format guide (three saved files, safetensors rationale, size example, merge_and_unload contract and disadvantages). https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint. Fetched 2026-08-10.

[20] peft low-level-API guide (`inject_adapter_in_model()` for models outside the config-and-wrap flow). https://huggingface.co/docs/peft/main/en/developer_guides/low_level_api. Fetched 2026-08-10.

[21] PEFT Method Comparison Space metadata, via the Hugging Face Hub API (`/api/spaces/peft-internal-testing/PEFT-method-comparison`) - an interactive Gradio app, not a static page with a single number to quote. https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison. Fetched 2026-08-10.
