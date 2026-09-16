# unsloth

An open-source fine-tuning and RL library for LLMs that patches the Hugging Face / trl stack for faster, lower-VRAM training on a single GPU, now paired with a desktop UI it also ships.

Its GitHub description, read live, is "the local UI to run and train text and diffusion models, including Kimi K3, Gemma 4, Qwen3.6, DeepSeek-V4, FLUX and more" [1], while the PyPI package summary for the `unsloth` distribution frames the same code as "2-5X faster training, reinforcement learning & finetuning" [2] - the project ships two tracks under one name: "Unsloth Studio," a downloadable local UI, and "Unsloth Core," the Python training library this card covers. It is built and maintained by Daniel Han and Michael Han [5], and lives at https://github.com/unslothai/unsloth [1]. In the Core track, `FastLanguageModel` loads a model and wraps trl's `SFTTrainer` in the SFT quickstart [3] and trl's `GRPOTrainer` in the GRPO quickstart [4]; `FastVisionModel` is the same wrapper for vision-capable models, shown loading DeepSeek-OCR in the troubleshooting docs [20]; and a third entry point, `PatchDPOTrainer()`, patches trl's `DPOTrainer` for preference training [10].

**When to pick it**: single-consumer-GPU fine-tuning or RL where VRAM is the binding constraint - Unsloth's own GRPO memory benchmark (below) is the strongest published case for that framing, and its docs are explicit that multi-GPU is not yet an officially supported, first-class path (see Scale it handles) [7]. It sits ABOVE trl rather than beside it: the quickstarts import `SFTTrainer` [3] or `GRPOTrainer` [4] FROM trl and wrap the model-loading step with Unsloth's own calls, and the preference-training page patches trl's `DPOTrainer` with `PatchDPOTrainer()` [10] - trl's own SFT and DPO trainer docs pages in turn cite Unsloth as an acceleration integration [10], so the two are usually additive, not a fork-off choice, unlike verl or trl on their own (cross-reference; not covered here). Pick trl or verl instead when the run needs official multi-node/cluster placement, since Unsloth's own multi-GPU story is DDP/Accelerate/FSDP-based and explicitly marked as not yet "official support" [7].

**Methods it ships**: the shortlist row's own code-search evidence names KTO and ORPO as detected inside `unsloth/__init__.py` [6]. Beyond that row-level evidence, the docs document a wider menu: SFT (the Alpaca quickstart) [3]; preference/reward methods DPO, ORPO, KTO, and PPO reward-modelling all stated to work, with Colab notebooks provided for GRPO, ORPO, DPO Zephyr, KTO, and SimPO but not PPO [10]; and GRPO as the flagship online RL method, with its own guide, a memory-efficiency guide, and a parameter-reference page [9][11][12]. GRPO is reached through trl's `GRPOTrainer` [4] with Unsloth's `fast_inference=True` for vLLM-backed generation, or `fast_inference=False` to run RL on models vLLM does not support [9]. The row's evidence is the only source-code-level proof point in this card; the docs pages are the source for the rest and are not code-verified here.

**Scale it handles**: single consumer GPU up to a documented VRAM-savings regime, then multi-GPU through Accelerate/DeepSpeed/torchrun/DDP with an explicit non-official caveat. The docs state plainly: "Unsloth currently supports multi-GPU setups through libraries like Accelerate and DeepSpeed," enabling FSDP and DDP, but add "we're working hard to make multi-GPU support much simpler and more user-friendly, and we'll be announcing official multi-GPU support for Unsloth soon," pointing to GitHub issue #2435, which the docs describe as "ongoing" but which is closed as of this fetch [7][8]. The documented DDP path installs from source (`pip install .`) and launches with `accelerate launch train.py` or `torchrun --nproc_per_node N_GPUS train.py` [7][13]; a `device_map="balanced"` flag pipeline-splits a model too large for one GPU (the docs' own example is Llama-3.3-70B-Instruct) [7]. For single-GPU memory, the RL guide gives a measured comparison, not just a mechanism: at 20K context and 8 generations per prompt, Unsloth trains Llama 3.1 8B GRPO in 54.3GB total VRAM against 510.8GB for a standard Flash-Attention-2 implementation, a reduction the docs call "90% less" [11]. The same page breaks that total into training memory (42GB vs. 414GB), the GRPO-specific memory cost (9.8GB vs. 78.3GB), and inference cost (0GB vs. 16GB), with an unchanged 2.5GB KV cache in both cases [11]. Separately, the docs state that 15GB of VRAM is enough to turn any model up to 17B parameters (their examples: Llama 3.1 8B, Phi-4 14B, Mistral 7B, Qwen2.5 7B) into a GRPO-trained reasoning model, and that models of 1.5B parameters or fewer need as little as 5GB [11]. An "Unsloth Standby" feature (`UNSLOTH_VLLM_STANDBY=1`, set before importing Unsloth) reuses the KV-cache/activation memory between vLLM's generation phase and Unsloth's training phase since the two alternate rather than run concurrently; the docs' own measured examples show 1.7x more context for Qwen3-32B LoRA on one H100 80GB (6,144 vs. 3,600 tokens) and 1.13x for Llama-3.1-8B QLoRA (47,500 vs. 42,000 tokens) [12].

**Install**: `pip install unsloth`; latest release 2026.8.10, uploaded to PyPI 2026-08-09 [2]; `requires-python <3.15,>=3.9` [2]; Apache-2.0, confirmed by the GitHub API license field, the LICENSE file text, and the pyproject.toml `license` field at the current release tag [1][14][5]. The PyPI wheel's own dependency metadata for 2026.8.10 pins the full training stack unconditionally: `torch>=2.4.0,<2.12.0`, `transformers>=4.51.3,<=5.5.0` (with named excludes), `trl>=0.18.2,<=0.24.0,!=0.19.0`, `peft>=0.18.0,!=0.11.0`, `accelerate>=0.34.1`, `datasets>=3.4.1,<4.4.0` (excluding 4.0.* and 4.1.0), `bitsandbytes>=0.45.5` (excluding 0.46.0 and 0.48.0), plus `unsloth_zoo>=2026.8.6` and platform-gated `xformers`/`triton` entries [2]. This is a live-endpoint reading of the PyPI JSON API on 2026-08-10, not pinned to a repository commit. It disagrees with the code at the corresponding GitHub release tag `v0.1.527-beta` (commit `154fa227de239009b721b709cd28c74bdda6f19f`, published 2026-08-09): that tag's `pyproject.toml` declares only seven unconditional CLI dependencies (typer, rich, pydantic, pyyaml, nest-asyncio, structlog, click) behind a `unsloth = "unsloth_cli:app"` console script, and gates the entire ML stack (torch, transformers, trl, bitsandbytes, xformers, triton, and per-CUDA/per-torch variants) behind nested extras (`unsloth[huggingface]`, `unsloth[base]`, and hundreds of `cuXXX-torchYYY` combinations) [15]. The docs' own install guidance for "Unsloth Core" recommends the plain `pip install unsloth` (or `uv pip install unsloth --torch-backend=auto`) as the standard path [16], which matches what PyPI's wheel metadata actually delivers, not what the tag's pyproject.toml declares unconditionally - so a reader building from the GitHub source at this tag with no extras gets a CLI-only install, while `pip install unsloth` from PyPI gets the full stack. No CUDA or GPU hardware minimum is stated as a single blanket number anywhere read for this card; the closest documented mechanism is an embedded install-time Python check in the "Advanced Pip Installation" section that raises a `RuntimeError` unless the detected CUDA version is one of `11.8, 12.1, 12.4, 12.6, 12.8, 13.0` (narrowing to `11.8, 12.6, 12.8, 13.0` for torch newer than 2.6.9) [16]. Screening-row commit for this repository is `37c64912d6238067dd6e2802acacb64177c3adea`, pushed 2026-07-31T21:15:30Z per the shortlist row [6] - about nine days behind the `v0.1.527-beta` release this Install field describes, so none of this field's code-level claims are read at the screening-row commit.

**Maintained by**: Daniel Han and Michael Han, listed as maintainers in the pyproject.toml at the current release tag [5]; the repository shows frequent, dated releases: v0.1.527-beta (2026-08-09), v0.1.526-beta ("DSpark + DeepSeek-V4 Flash 0731," 2026-08-04), v0.1.512-beta ("Kimi K3 + DeepSeek-V4 Flash 0731 + Deep Research + Parallel Chat," 2026-07-29), and v0.1.501-beta ("Introducing AMD support," 2026-07-20) [17]. GitHub reports 69,775 stars as of 2026-08-10 (not used here for ranking) [1]; the shortlist row separately records 69,291 stars at an earlier snapshot, likewise not for ranking [6].

## Quick start

The SFT quickstart, condensed from the Llama-3.1 8B Alpaca notebook the docs link (comments and the alternate-4-bit-model list omitted) [3]:

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("unsloth/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)

from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)
trainer.train()
```

The vLLM-backed GRPO quickstart, from the RL guide - loading with fast inference on, then handing the model to trl's `GRPOTrainer` as in the SFT case [9]:

```python
# pip install unsloth vllm
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    fast_inference = True,
)
```

There is also a CLI: `unsloth-cli.py` takes `--model_name`, `--dataset`, `--r`, `--lora_alpha`, `--lora_dropout`, and gradient-checkpointing flags, documented in full via `python unsloth-cli.py --help` [13].

## Start it

- One GPU is the base case: the quick-start scripts above, run as-is.
- Multiple GPUs currently go through Accelerate, DeepSpeed, FSDP, or DDP, but the docs mark this explicitly as not yet an officially supported path [7]. The documented DDP recipe: install from source with `pip install .`, then launch a training script with `accelerate launch train.py` or `torchrun --nproc_per_node N_GPUS train.py` [7][13]. When a single GPU cannot hold the model, `device_map = "balanced"` on `FastLanguageModel.from_pretrained` pipeline-splits it across the visible GPUs (the docs' own example is Llama-3.3-70B-Instruct) [7].
- GRPO's generation layer is a separate choice at start time: `fast_inference=True` on `from_pretrained` turns on vLLM-backed generation; `fast_inference=False` runs RL on models vLLM does not support (the docs' example is Qwen3.5-4B) [9]. `UNSLOTH_VLLM_STANDBY=1`, set as an environment variable before any Unsloth import, together with `gpu_memory_utilization=0.95` on the vLLM side, reuses memory between the generation and training phases for longer context at no extra memory cost, per the docs' own measured examples [12].
- Effective-batch arithmetic for GRPO, from the parameter-reference page: `effective_batch_size = steps_per_generation * num_processes * train_batch_size`, where `train_batch_size` is defined as the number of samples per process per step and the worked examples set it equal to `per_device_train_batch_size`; `unique_prompts = effective_batch_size / num_generations`, which the docs say must be greater than 2 for GRPO to work at all [18]. The page's own worked example: one GPU, `per_device_train_batch_size=3`, `gradient_accumulation_steps=2`, `steps_per_generation=4` gives an effective batch size of 12 [18].
- Configuration for both SFT and GRPO is trl's `SFTConfig`/`GRPOConfig` (a `TrainingArguments` subclass) passed through unmodified as `args=`; the quickstart's own defaults (`optim="adamw_8bit"`, `lr_scheduler_type="linear"`, `learning_rate=2e-4`) are notebook choices, not documented as library-wide changed defaults, and the dtype comment in the notebook itself flags the hardware assumption: `dtype=None` auto-detects "Float16 for Tesla T4, V100, Bfloat16 for Ampere+" [3].
- Out-of-memory first aid: for saving/exporting, lower `maximum_memory_usage` below its 0.75 default (e.g. to 0.5) to use less of GPU peak memory during the save [19]. For evaluation-loop OOM, drop `per_device_eval_batch_size` below 2 and turn on `fp16_full_eval=True` (or `bf16_full_eval=True` on bf16 hardware), which the docs say should already be on by default as of June 2025 [20]. GRPO-specific OOM guidance is not separately itemized in the pages read for this card beyond the vLLM-memory and Standby mechanisms above [11][12].

## Watch it

This section is the mechanics only; what a metric means for a given method lives on that method's card, not here.

- **Enable it**: the SFT quickstart sets `report_to="none"` by default [3], so a default run keeps no external log - `report_to` accepts trl/transformers' usual backends (e.g. `"wandb"`) to turn one on [21]. The RL guide separately states that GRPO reward-vs-step logging, including every individual reward function and the aggregated total, is now built directly into Unsloth, "eliminating the need for external tools like wandb etc." for that specific view [9].
- **Metric names**: the pages read for this card do not carry a single itemized metric-name list comparable to trl's SFT/GRPO "Logged metrics" tables (see the trl card for that list, since Unsloth trains through trl's trainer classes and inherits their emitted metrics) [3][9]. Unsloth's own contribution, per the RL guide, is the built-in reward-vs-step chart per reward function plus the aggregated total reward [9].
- **Sample the generations**: not documented as a separate Unsloth-side toggle in the pages read for this card; trl's own `log_completions`/`num_completions_to_print` fields on `GRPOConfig` apply unchanged since Unsloth trains through trl's `GRPOTrainer` (see the trl card).
- **Evaluate during training**: the FAQ page's documented recipe is `dataset.train_test_split(test_size=0.01, shuffle=True, seed=3407)` (with an explicit warning to always shuffle), passed to `SFTTrainer` as `eval_dataset`; the paired `SFTConfig` fields are `fp16_full_eval=True`, `per_device_eval_batch_size=2`, `eval_accumulation_steps=4`, `eval_strategy="steps"`, `eval_steps=1` [20].
- **Stopping**: Unsloth documents an explicit early-stopping recipe (unlike trl's own docs, which the trl card found publish none): `transformers.EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)` added via `trainer.add_callback(...)`, paired with `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False` on the `SFTConfig` [20][22]. This is stated for SFT via `eval_loss`; it is not shown paired with a reward-based GRPO stopping criterion in the pages read for this card.

## Save it

- Checkpoints: set `save_strategy="steps"` and `save_steps=N` on the trainer's `Config`; `save_total_limit` caps how many are kept on disk (the FAQ's early-stopping example uses `save_total_limit=3`) [20][22]. The docs do not enumerate the file layout inside a `checkpoint-<step>` directory in the pages read for this card; that detail is transformers/trl's own contract (see the trl card) since Unsloth checkpoints through trl's `Trainer.save`.
- Resume: `trainer.train(resume_from_checkpoint=True)` resumes from the latest checkpoint in `output_dir`; passing a Weights & Biases artifact directory downloaded via `wandb.init()` + `run.use_artifact(...)` + `artifact.download()` also works as the `resume_from_checkpoint` argument [22].
- GGUF export is Unsloth-specific and is the documented path to Ollama/llama.cpp/Studio deployment: `model.save_pretrained_gguf("directory", tokenizer, quantization_method="q4_k_m")` (or `"q8_0"`, `"f16"`) saves locally; `model.push_to_hub_gguf("hf_username/directory", tokenizer, quantization_method=...)` uploads to the Hub [19]. The full `ALLOWED_QUANTS` list, sourced from llama.cpp's own quantize.cpp, spans `q2_k` through `q6_k`, `iq2_xxs`/`iq2_xs`/`iq3_xxs`, and aliases like `q4_k`→`q4_k_m` [19]. A manual path exists for building llama.cpp from source and running `convert_hf_to_gguf.py` directly against a 16-bit-merged model [19].
- Chat-template mismatch is a documented trap that bites specifically at export time: the docs state that good results inside Unsloth followed by gibberish or endless generation on another platform (Ollama, vLLM) most commonly comes from an incorrect chat template or eos token surviving the export, and recommend using Unsloth's own conversational notebooks to force the correct template [19].
- Adapter vs. merged model: `save_pretrained` on a LoRA-trained model (the quickstart's `"llama_lora"` output directory) saves an ADAPTER, reloadable with `FastLanguageModel.from_pretrained("llama_lora", ...)`; the docs explicitly discourage the alternative PEFT-only reload path (`AutoPeftModelForCausalLM.from_pretrained("llama_lora", load_in_4bit=...)`), commenting inline "I highly do NOT suggest - use Unsloth if possible" [3]. An adapter directory is NOT a full model; `model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")` produces the full merged model, which is the required input to the manual GGUF-conversion path [19].
- Whether an external evaluator can load what was saved directly is the loader's contract: a `checkpoint-<step>` or merged-model directory saved through trl's `Trainer.save` is loadable the way trl documents (see the trl card's loader-handoff note); a bare adapter directory is not.

## Find it in the docs

The docs are the live source; this section teaches the lookup pattern, not the content.

- Address pattern: `https://unsloth.ai/docs/<slug>.md` - every page on the site is available as Markdown by appending `.md` to its URL, confirmed live on the beginners, saving-to-gguf, multi-GPU, and inference pages fetched for this card [23][7][24]. There is no version-tag form analogous to trl's `/docs/trl/<version>/...`; the docs are unversioned and describe only the current release.
- Page-slug recipes read for this card: `basics/inference-and-deployment.md` is a link-hub page to `unsloth-inference`, `saving-to-gguf`, `api`, `vllm`, `ollama`, `connections`, `sglang`, and a troubleshooting sub-page [24]; `basics/troubleshooting-and-faqs.md` carries the evaluation, early-stopping, and OOM recipes [20]; `basics/multi-gpu-training-with-unsloth.md` and its `/ddp.md` child carry the multi-GPU story [7][13]; `get-started/fine-tuning-for-beginners.md` is a FAQ-style link-hub with no unique technical content of its own [23].
- Question-to-slug map: "how do I save/export my model" -> `basics/inference-and-deployment/saving-to-gguf.md` (not `basics/saving-models.md`, which returns a GitBook 404 for this docs build) [19]; "why is my run OOMing" -> `basics/troubleshooting-and-faqs.md` (not `basics/errors-fixes/oom-out-of-memory-issues.md`, likewise a confirmed 404) [20]; "how do I resume/checkpoint" -> `basics/finetuning-from-last-checkpoint.md` [22]; "how do I use RL/GRPO" -> `basics/reinforcement-learning-rl-guide.md`, with a separate memory-focused page and a parameter-reference page [9][11][12][18].
- Runnable references beyond the docs: the `unslothai/notebooks` GitHub repository, linked throughout the docs, hosts the Colab notebooks the quickstarts point to (Llama-3.1-8B Alpaca SFT, Qwen3 GRPO, and conversational notebooks for Qwen3, Gemma-3, Llama-3.2, Phi-4, Mistral v0.3) [19][3]; a `python_scripts/` subdirectory of that same repo holds `.py` versions of the notebooks for use with `accelerate launch` or `torchrun` [7].
- Community layer: the pages read for this card carry no dedicated curated-community-tutorials page comparable to trl's `community_tutorials` slug; the closest official pointer is the docs' own repeated links to the `unslothai/notebooks` Colab set and a mention of a community Reddit page (`r/unsloth`) on the beginners page [23]. No independent practitioner-blog curation page was found in the pages fetched for this card - this is an absence, not a confirmed non-existence, since the full docs tree was not exhaustively crawled.
- MCP: two unrelated things share the name here, and only one is a docs-query mechanism. Every docs page carries the same generic GitBook query-string mechanism - appending `?ask=<question>&goal=<endgoal>` to any page's `.md` URL runs a natural-language query against that page's documentation platform - confirmed identically worded on the beginners, saving-to-gguf, multi-GPU, and inference pages [23][7][24][19]. A page at the slug `basics/mcp-server` returns a GitBook 404 in this docs build [25]. Separately, the README documents an "opt-in MCP control endpoint" that lets AI clients manage Unsloth Studio's models, training runs, recipes, and exports - a runtime-control surface for the Studio product, not a tool for querying these docs [26]. There is no Unsloth equivalent of trl's dedicated Hugging Face MCP documentation-search endpoint.
- Honest boundaries stated in the docs themselves: multi-GPU is explicitly not yet given "official multi-GPU support" (see Scale it handles and Start it) [7]; GRPO requires `num_generations >= 2` or the group standard deviation used in the advantage calculation is zero and undefined [12]; the docs warn against enabling `mask_truncated_completions` on GRPOConfig, stating "there are some KL issues with this flag, so we recommend to disable it," and link to the KL-NaN mechanism in `unsloth_zoo`'s source at a pinned commit [18].

## Sources

Ecosystem tools named in passing (Accelerate, DeepSpeed, FSDP, DDP, vLLM, llama.cpp, PEFT, Weights & Biases, GitBook) are reached through the docs pages cited below and are not separately enumerated as references. trl is cited only for the cross-reference claims in "When to pick it" and "Watch it"; its own metric names, config surface, and stopping-rule findings are on the trl card, not restated here. All docs pages are unversioned, mutable pages read on 2026-08-10 unless a commit is separately named.

[1] unslothai/unsloth GitHub repository (description, license, stars, live). https://github.com/unslothai/unsloth. Fetched 2026-08-10.

[2] unsloth on PyPI, JSON API (version 2026.8.10, summary, requires_python, requires_dist). https://pypi.org/pypi/unsloth/json. Fetched 2026-08-10.

[3] Llama 3.1 (8B) Alpaca notebook cells (SFT quickstart, dtype hardware comment, adapter-vs-PEFT reload discouragement). https://github.com/unslothai/notebooks (Llama3.1_8B_Alpaca.ipynb). Fetched 2026-08-10.

[4] Qwen3 (4B) GRPO notebook cells (`from trl import GRPOConfig, GRPOTrainer`, `trainer = GRPOTrainer(...)`), linked from the RL guide's GRPO-notebook list. https://github.com/unslothai/notebooks (Qwen3_4B_GRPO.ipynb). Fetched 2026-08-10.

[5] unsloth pyproject.toml at tag v0.1.527-beta, commit 154fa227de239009b721b709cd28c74bdda6f19f (license field, maintainers field). https://raw.githubusercontent.com/unslothai/unsloth/v0.1.527-beta/pyproject.toml. Fetched 2026-08-10.

[6] Shortlist row for unslothai/unsloth (screening commit, last_push, stars snapshot, methods_seen code-search evidence, why_confirmed, evidence lists). Pipeline-provided shortlist data, not independently fetched.

[7] Unsloth docs, Multi-GPU Fine-tuning with Unsloth. https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth.md. Fetched 2026-08-10.

[8] GitHub REST API, unslothai/unsloth issue #2435 ("multi-GPU training", state: closed, closed_at: 2026-06-11). https://api.github.com/repos/unslothai/unsloth/issues/2435 (issue page: https://github.com/unslothai/unsloth/issues/2435). Fetched 2026-08-10.

[9] Unsloth docs, Reinforcement Learning (RL) Guide. https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide.md. Fetched 2026-08-10.

[10] Unsloth docs, Preference Optimization Training - DPO, ORPO & KTO page (trl-docs cross-reference, Colab links). https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto.md. Fetched 2026-08-10.

[11] Unsloth docs, RL Guide GRPO Requirement Guidelines section (VRAM comparison table and figures). https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide.md. Fetched 2026-08-10.

[12] Unsloth docs, Memory Efficient RL / Unsloth Standby page. https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl.md. Fetched 2026-08-10.

[13] Unsloth docs, Multi-GPU Fine-tuning with Distributed Data Parallel (DDP). https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp.md. Fetched 2026-08-10.

[14] unslothai/unsloth LICENSE file (Apache License 2.0 full text). https://raw.githubusercontent.com/unslothai/unsloth/v0.1.527-beta/LICENSE. Fetched 2026-08-10.

[15] unsloth pyproject.toml at tag v0.1.527-beta, commit 154fa227de239009b721b709cd28c74bdda6f19f (unconditional dependencies, console script, extras structure). https://raw.githubusercontent.com/unslothai/unsloth/v0.1.527-beta/pyproject.toml. Fetched 2026-08-10.

[16] Unsloth docs, Pip Install page (Unsloth Core install commands, Advanced Pip Installation CUDA-version checks). https://unsloth.ai/docs/get-started/install/pip-install.md. Fetched 2026-08-10.

[17] unslothai/unsloth GitHub Releases API (tags, publish dates, release names). https://api.github.com/repos/unslothai/unsloth/releases. Fetched 2026-08-10.

[18] Unsloth docs, Advanced Reinforcement Learning Documentation (GRPO Training/Generation/Batch Parameters: effective-batch arithmetic, num_generations >= 2 requirement, mask_truncated_completions warning). https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation.md. Fetched 2026-08-10.

[19] Unsloth docs, Saving to GGUF. https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf.md. Fetched 2026-08-10.

[20] Unsloth docs, Troubleshooting & FAQs (evaluation setup, evaluation-OOM fix, early-stopping recipe). https://unsloth.ai/docs/basics/troubleshooting-and-faqs.md. Fetched 2026-08-10.

[21] trl documentation, logging guide (report_to backend list; cited here only to note that Unsloth trains through trl's trainer classes). https://huggingface.co/docs/trl/main/en/logging. Fetched 2026-08-10.

[22] Unsloth docs, Finetuning from Last Checkpoint (save_strategy/save_steps, resume_from_checkpoint, wandb artifact resume, early-stopping recipe). https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint.md. Fetched 2026-08-10.

[23] Unsloth docs, Fine-tuning for Beginners. https://unsloth.ai/docs/get-started/fine-tuning-for-beginners.md. Fetched 2026-08-10.

[24] Unsloth docs, Inference and Deployment link-hub page. https://unsloth.ai/docs/basics/inference-and-deployment.md. Fetched 2026-08-10.

[25] Unsloth docs, mcp-server slug (confirmed 404 in this docs build). https://unsloth.ai/docs/basics/mcp-server.md. Fetched 2026-08-10.

[26] unslothai/unsloth README.md (Studio vs. Core tracks, opt-in MCP control endpoint for Studio). https://raw.githubusercontent.com/unslothai/unsloth/v0.1.527-beta/README.md. Fetched 2026-08-10.
