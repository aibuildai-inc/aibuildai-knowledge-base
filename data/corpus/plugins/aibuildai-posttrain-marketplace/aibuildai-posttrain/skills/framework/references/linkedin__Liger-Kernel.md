# Liger-Kernel

LinkedIn's Triton-kernel acceleration layer for LLM training: it patches faster fused ops into an existing model or trainer - it is not itself a trainer or orchestration framework.

**Liger Kernel** is described in its own README as "a collection of Triton kernels designed specifically for LLM training" that implements Hugging-Face-compatible RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy, and further kernels beyond those, plus post-training kernels for alignment and distillation losses such as DPO, CPO, ORPO, SimPO, KTO, and JSD [1]. It is built by LinkedIn (GitHub org `linkedin`) [2] and is consumed either by monkey-patching an existing Hugging Face model class (`apply_liger_kernel_to_<model>()`, or the `AutoLigerKernelForCausalLM` wrapper) or by importing individual fused-loss modules (`from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss`) directly into a training loop [1][3]. It lives at https://github.com/linkedin/Liger-Kernel [2].

**When to pick it**: not a competitor to trl or verl - it is a drop-in speed/memory layer those frameworks call into. Pick it whenever you already train with trl (SFT, DPO, GRPO, KTO, or the experimental GKD trainer) and want a one-flag speedup: trl's own docs show every one of those five Config classes accepting `use_liger_kernel=True` [4]. Pick it standalone when you want fused post-training losses (DPO, ORPO, SimPO, CPO, KTO, and an undocumented GRPO loss - see Methods below) without going through a full trainer at all, via the `liger_kernel.chunked_loss` module [1][5]. The README also lists Axolotl, LLaMA-Factory, the plain Hugging Face `Trainer`, SWIFT, and oumi as integrating frameworks [1]. Skip it if you need it to schedule multi-node RL rollouts or manage checkpoints itself - it has no trainer loop of its own beyond one optional `LigerORPOTrainer` subclass (see below), and no save/resume contract independent of whatever host trainer you use.

**Methods it ships**: the README's "Alignment Kernels" table lists five fused losses - CPO, DPO, ORPO, SimPO, KTO - each importable as `liger_kernel.chunked_loss.LigerFusedLinear<Method>Loss` [1]. Reading the module's own export list at commit `08e8ccdc`, `liger_kernel/chunked_loss/__init__.py` exports three more classes the README's tables do not line up with: `LigerFusedLinearGRPOLoss`, which no README table mentions at all; `LigerFusedLinearCosineSimilarityLoss`, likewise absent from both the README's Alignment and Distillation tables; and `LigerFusedLinearJSDLoss`, which duplicates a method the README's "Distillation Kernels" table documents under a different import path, `liger_kernel.transformers.LigerFusedLinearJSD` [6][1]. The same directory also contains a `fused_linear_ppo.py` file, but it exports no public `LigerFusedLinear*Loss` class from `__init__.py` - reading the directory listing, it is internal support code, not a documented PPO loss [6]. Distillation kernels (KLDivergence, JSD, Fused Linear JSD, TVD) are listed separately in the README's "Distillation Kernels" table [1]. Model-side kernels (RMSNorm, LayerNorm, Modulated RMSNorm, RoPE, SwiGLU, GeGLU, CrossEntropy, Fused Linear CrossEntropy, Multi Token Attention, Softmax, Sparsemax, mHC) are not post-training methods and are out of scope for this card's method focus, but are what the monkey-patching functions apply [1]. Liger ships exactly one standalone trainer subclass, `LigerORPOTrainer(ORPOTrainer)`, at `liger_kernel/transformers/trainer/orpo_trainer.py` - the `trainer/` directory contains only that file plus its `__init__.py`, confirmed by listing the directory at commit `08e8ccdc` [7]. The README marks an `experimental` kernel tier (Embedding, int2xint8 matmul) separately from the stable tables [1]. All method math and training-signal semantics live on each method's own methodology card; this card only tracks which fused implementation exists and where.

**Scale it handles**: Liger has no launcher of its own - the README states it "works out of the box with Flash Attention, PyTorch FSDP, and Microsoft DeepSpeed" and lists Multi-GPU support "(PyTorch FSDP, DeepSpeed, DDP, etc.)" as a Key Feature, with no published multi-node benchmark of its own [1]. The scale a run reaches is set entirely by the host trainer or script (trl's Accelerate/DeepSpeed/FSDP launch, or a raw `torchrun`/`accelerate launch` around a HF `Trainer`). The README's own headline number, measured on LLaMA-3-8B with batch size 8, bf16, AdamW, gradient checkpointing, and FSDP1 on 8 A100s, is a 20% increase in multi-GPU training throughput and a 60% reduction in memory usage; under those same conditions the README reports that plain Hugging Face training starts to run out of memory at a 4K context length, while adding Liger Kernel scales the same setup to 16K [1]. The Examples table gives per-example measured deltas at the specific configurations it names: the Hugging Face Trainer example trains Llama-3-8B on Alpaca roughly 20% faster with over 40% memory reduction using 4 A100s with FSDP; the Lightning Trainer example trains Llama3-8B on MMLU with a 15% throughput increase and 40% memory reduction using 8 A100s with DeepSpeed ZeRO3; the Medusa multi-head example reports 80% memory reduction and 40% throughput improvement with 5 LM heads on 8 A100s with FSDP; the Liger ORPO Trainer example reports 50% memory reduction aligning Llama 3.2 with FSDP [1]. Two optional non-Triton compute backends exist and are opt-in via `LIGER_KERNEL_IMPL`: cuTile (needs the `cutile` or `cutile-tileiras` extra) and CuTe DSL (needs the `cutedsl` extra, targets only Hopper SM90 and Blackwell SM100/SM110, and any op without a CuTe DSL kernel "transparently falls back to the default Triton kernel, so selecting the backend is always safe") [1]. Outside those two opt-in backends, the README states the kernels "inherit the full spectrum of hardware compatibility offered by Triton" [1].

**Install**: `pip install liger-kernel`, current stable release v0.8.1, published 2026-07-23 (PyPI upload timestamp) [8], tagged at commit `a5234dfc` (2026-07-22) [9]; a `liger-kernel-nightly` PyPI package also exists [1]. PyPI's `requires_python` field is `None` - no Python floor is declared [8]. License is BSD-2-Clause [1][8]. `setup.py`, read both at the release tag `v0.8.1` and at this card's screening commit `08e8ccdc` (12 days ahead of the release; the two `setup.py` files differ only in a dev-only ruff upper bound, `<0.16.0` vs unbounded, which is not load-bearing) [10], picks dependencies by detecting the local accelerator at install time (`nvidia-smi` -> cuda, else `rocm-smi` -> rocm, else Intel/Ascend probes, else cpu): CUDA/CPU installs pin `torch>=2.1.2, triton>=2.3.1`; ROCm pins only `triton>=3.0.0` (torch is expected to already be installed from PyTorch's ROCm index per the README) [1][10]; XPU pins `torch>=2.6.0`; Ascend NPU pins exact versions `torch==2.7.1, torch_npu==2.7.1, triton-ascend==3.2.1` from a non-default index [1][10]. The README states the CUDA floor as `triton >= 2.3.0`, one patch version below what `setup.py` actually installs (`>=2.3.1`) [1][10]. Extras: `dev` (`transformers>=4.52.0`, `datasets>=2.19.2`, test/lint tooling), `cutile`, `cutile-tileiras`, `cutedsl` [10]. transformers itself is an optional, not a hard, dependency - only required "if you plan to use the transformers models patching APIs" [1]. No CUDA version or GPU-generation floor is stated anywhere for the default Triton path; the Hopper/Blackwell requirement applies only to the opt-in CuTe DSL backend [1].

**Maintained by**: LinkedIn (GitHub org `linkedin`) [2]; 6,560 GitHub stars as of the fetch date, not a ranking signal [2]. Release cadence over the last year: v0.6.4 (2025-11-21), v0.6.5 (2026-02-04), v0.7.0 (2026-02-12), v0.8.0 (2026-04-30), v0.8.1 (2026-07-23) [11]; the repository's most recent push was 2026-08-09, six days after this card's screening commit [2]. The README's Latest News section's most recent dated entry announces a Discord channel and a "Liger Kernel x Triton China Meetup" for mid-January 2026, posted 2025-12-19 [1].

## Quick start

Liger ships no full quickstart script of its own for post-training; its own docs demonstrate patching, and the repo's `examples/alignment/run_orpo.py` is the smallest complete, runnable post-training script (Llama-3.2-1B-Instruct on the `trl-lib/tldr-preference` dataset via the standalone `LigerORPOTrainer`), read in full at commit `08e8ccdc` [12]:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import ORPOConfig

from liger_kernel.transformers.trainer import LigerORPOTrainer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    max_length=512,
    padding="max_length",
)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("trl-lib/tldr-preference", split="train")

training_args = ORPOConfig(
    output_dir="Llama3.2_1B_Instruct",
    beta=0.1,
    max_length=128,
    per_device_train_batch_size=32,
    max_steps=100,
    save_strategy="no",
)

trainer = LigerORPOTrainer(model=model, args=training_args, tokenizer=tokenizer, train_dataset=train_dataset)

trainer.train()
```

The library's own Getting Started page demonstrates the two patching patterns and the standalone-kernel pattern (all three read in full from the raw doc source, `docs/Getting-Started.md`) [3]:

```python
# 1. Simplest: auto-patch a supported HF model type
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")

# 2. Model-specific patching API, with explicit kernel selection
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama(rope=True, swiglu=True, cross_entropy=True,
                             fused_linear_cross_entropy=False, rms_norm=False)
model = transformers.AutoModelForCausalLM("path/to/llama/model")

# 3. Compose your own model from individual kernels
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
loss_fn = LigerFusedLinearCrossEntropyLoss()
loss = loss_fn(model.weight, input, target)
loss.backward()
```

Reading the actual `apply_liger_kernel_to_llama` signature in `src/liger_kernel/transformers/monkey_patch.py` at commit `08e8ccdc`, its real defaults differ from the illustrative call above: `rope=True, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=True` - fused-linear cross-entropy, not plain cross-entropy, is on by default [13].

## Start it

Liger has no launch command of its own - it is started by whichever host trainer or script you run it inside:

- Standalone `LigerORPOTrainer`, one GPU: run the quick-start script above as-is; multi-GPU goes through Accelerate, using the repo's own `examples/alignment/accelerate_config.yaml` as the config template, e.g. `accelerate launch --config_file examples/alignment/accelerate_config.yaml examples/alignment/run_orpo.py` [12][14].
- Inside trl: any of `SFTConfig`, `DPOConfig`, `GRPOConfig`, `KTOConfig`, or the experimental `GKDConfig` (import path `trl.experimental.gkd`) takes `use_liger_kernel=True` to swap in Liger's fused ops for that trainer, confirmed by reading trl's own `liger_kernel_integration.md`, `sft_trainer.md`, `dpo_trainer.md`, and `grpo_trainer.md` docs sources [4][15][16][18]. This is a single boolean, not a chunk-size or kernel-selection knob at the trl layer.
- Effective-batch arithmetic, generation layout, and non-Liger OOM knobs (e.g. `per_device_train_batch_size`, `gradient_accumulation_steps`, vLLM colocate/server mode) are entirely trl's surface, not Liger's - see trl's own card for those.
- Liger-specific memory/speed knob: the fused-linear-loss classes (`LigerFusedLinear<Method>Loss`) take a `chunk_size` constructor argument, documented in `orpo_loss.py`'s own docstring only as "Size of chunks for processing", defaulting to `1`; the chunked-loss module's README never mentions `chunk_size` at all, so this parameter is undocumented at the module-README level despite the README's own claim (see below) that the underlying FlexChunkLoss design is what delivers Liger's memory savings [19][5].
- trl's SFT integration has one interaction to know before enabling Liger: `SFTConfig.loss_type` defaults to `"chunked_nll"`, and trl's own docs state it silently resolves to plain `"nll"` when `use_liger_kernel=True` is also set [17].
- No dedicated Liger OOM-first-aid guide was found; the README's "Fused Linear" note is the closest thing to memory guidance: "combine linear layers with losses, reducing memory usage by up to 80% - ideal for HBM-constrained workloads" [1]. Beyond enabling `use_liger_kernel`/choosing a fused-linear loss class and tuning `chunk_size` down, memory first aid is the host trainer's (trl's OOM section, not covered here).

## Watch it

Liger changes computation, not logging - it adds no metrics of its own beyond changing what a wrapped trainer already logs. What each signal means for a method lives on that method's card; this section only says what appears and where.

- Standalone `LigerORPOTrainer`: reading its `get_batch_loss_metrics` method at commit `08e8ccdc`, it logs 11 keys - `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `logps/rejected`, `logps/chosen`, `logits/rejected`, `logits/chosen`, `nll_loss`, `log_odds_ratio`, `log_odds_chosen` [7]. trl's own `orpo_trainer.md` "Logged metrics" section for the base `ORPOTrainer` documents only 7 of those - it omits `logps/rejected`, `logps/chosen`, `logits/rejected`, and `logits/chosen` [20]; Liger's subclass overrides the loss computation (via `LigerFusedLinearORPOLoss`) and `concatenated_forward` for FSDP compatibility and does not add any metric name beyond what the base `ORPOTrainer` code already emits, but trl's own doc page undercounts that base set by 4 keys [7][20].
- Inside trl with `use_liger_kernel=True` on GRPO specifically, trl's own `grpo_trainer.md` documents one metric-name change: the usual `clip_ratio/*` family collapses to a single `clip_ratio` scalar when the fused Liger GRPO loss path is active [18].
- Logging is otherwise entirely the host trainer's mechanism (trl's `report_to` / logging cadence, or plain HF `Trainer`'s) - not documented separately by Liger.
- No Liger-specific stopping rule, threshold, or health-limit was found; the pages searched for this card were the README, the Getting-Started, Low-Level-APIs pages, and the chunked_loss module README - none publish a runtime-tuning or stopping-rule page of their own [1][3][5]. Read trl's or the host trainer's own card for that.

## Save it

Liger publishes no checkpoint format, save call, or resume call of its own - it patches forward computation and loss functions in place; the patched ops are not persisted as part of a checkpoint. The Getting-Started page (`docs/Getting-Started.md`, read in full) covers only the three patching/composition patterns shown under Quick start above and says nothing about saving or checkpoints [3]. Saving, resuming, and checkpoint layout are entirely the host trainer's contract: `trainer.save_model()` / `resume_from_checkpoint` on `LigerORPOTrainer` inherit unchanged from trl's `ORPOTrainer` base class, since Liger's subclass overrides only `concatenated_forward` and `get_batch_loss_metrics` [7]. Whether a saved checkpoint reloads correctly is therefore the host trainer's and the loader's contract, not Liger's - consult trl's (or the relevant host trainer's) own card for the save/resume/loader details.

## Find it in the docs

The docs are the live source; this section teaches the lookup only.

- Address pattern: `https://linkedin.github.io/Liger-Kernel/<Page-Name>/` - a single, unversioned mkdocs-material site that tracks the `main` branch, not release tags; checked 2026-08-10: `/Getting-Started/` and `/Examples/` both return HTTP 200, while a guessed `/v0.8.1/` path returns 404 [21]. Because the site is unversioned, anything read from it describes `main` at fetch time, not the pinned v0.8.1 release - cross-check against the source at your installed tag when precision matters.
- Page slugs, read from the repo's own `mkdocs.yml` nav at commit `08e8ccdc`: `index` (Home), `Examples`, `Getting-Started`, `High-Level-APIs`, `Low-Level-APIs`, `contributing`, `acknowledgement`, `license` [22].
- Question-to-slug map: "which HF model families are patched, and how" -> `High-Level-APIs` (the AutoModel wrapper and per-model patching-function table); "which individual kernels/losses exist and their import paths" -> `Low-Level-APIs` (Model, Alignment, Distillation, Experimental kernel tables) [1]; "runnable end-to-end examples" -> `Examples` page, which mirrors the README's Examples table (Hugging Face Trainer, Lightning Trainer, Medusa multi-head retraining, Qwen2-VL SFT, the ORPO alignment script) [1].
- Runnable references beyond the docs site: the `examples/` tree in the GitHub repo, e.g. `examples/alignment/run_orpo.py` plus its `accelerate_config.yaml` used above, and `examples/huggingface/run_qwen2_vl.sh` for a vision-language SFT smoke test [1][12].
- Community layer: Liger's README curates two dated external posts under its own "Latest News" section rather than a separate tutorials page - the LinkedIn Engineering Blog post "Liger-Kernel: Empowering an open source ecosystem of Triton Kernels for Efficient LLM Training" (2024-12-05) and a joint PyTorch blog post, "Peak Performance, Minimized Memory: Optimizing torchtune's performance with torch.compile & Liger Kernel" (2025-03-06) [1]. A Liger Kernel Technical Report is also linked directly from the README, hosted on OpenReview [1]. No separate community-tutorials page (of the shape trl's `community_tutorials` page has) was found in the docs nav [22].
- No official MCP endpoint for these docs was found or searched beyond the docs site itself; none is claimed here.
- Honest boundary: Liger has no orchestration, checkpointing, or multi-node scheduling of its own - it is a compute layer under trl, HF `Trainer`, Axolotl, LLaMA-Factory, SWIFT, and oumi, and a chooser who needs a standalone RL trainer or cluster scheduler should read trl's or verl's card instead, not this one (cross-reference; not covered here).
- Trap, from a closed maintainer-answered issue: issue #1246, closed 2026-06-03, reported by CONTRIBUTOR michaelroyzen that after upgrading to v0.8.0, training Qwen3.5 MoE with trl GRPO and DeepSpeed ZeRO-3 on H200/B300 GPUs crashed with illegal-memory-access errors and NaNs; COLLABORATOR Mecoli1219 root-caused it to an index-overflow bug in the fused MoE kernels added by PR #1179 (not the autotune behavior first suspected) and shipped a fix; the reporter's closing comment confirms the fix on H200 only ("Confirming that the new overflow patch fixes the crash on H200!") and explicitly leaves B300 untested ("I think it will work on B300 too but haven't had a chance to test on that platform yet"), so the B300 case named in the original report was never confirmed fixed [23].

## Sources

Ecosystem tools named only in passing (Flash Attention, PyTorch FSDP, DeepSpeed, Accelerate, DDP) are reached through the README's and Get-Started page's own links and are not enumerated separately here. Every claim not otherwise dated is a 2026-08-10 reading. Commit `08e8ccdc6a1838e3da230c528fff808491817b58` (2026-08-03) is this card's screening commit and is 6 days behind the repository's most recent push (2026-08-09) and 12 days ahead of the v0.8.1 release tag `a5234dfc` (2026-07-22); the Install field is read at the release tag itself, not the screening commit, except where noted.

[1] Liger-Kernel README.md at commit `08e8ccdc`. https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/README.md. Fetched 2026-08-10.

[2] Liger-Kernel GitHub repository (API metadata: description, stars, license, push/create dates, topics). https://github.com/linkedin/Liger-Kernel. Fetched 2026-08-10.

[3] Liger-Kernel Getting-Started doc source at commit `08e8ccdc`. https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/docs/Getting-Started.md. Fetched 2026-08-10.

[4] trl Liger Kernel integration guide. https://raw.githubusercontent.com/huggingface/trl/main/docs/source/liger_kernel_integration.md. Fetched 2026-08-10.

[5] Liger-Kernel chunked_loss module README at commit `08e8ccdc`. https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/src/liger_kernel/chunked_loss/README.md. Fetched 2026-08-10.

[6] Liger-Kernel `chunked_loss/__init__.py` export list and the `chunked_loss/` directory listing (15 files, including `fused_linear_ppo.py`), at commit `08e8ccdc`. https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/src/liger_kernel/chunked_loss/__init__.py and GitHub Contents API for the same directory. Fetched 2026-08-10.

[7] Liger-Kernel `transformers/trainer/orpo_trainer.py` and the `transformers/trainer/` directory listing, at commit `08e8ccdc`. https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/src/liger_kernel/transformers/trainer/orpo_trainer.py and GitHub Contents API for the same directory. Fetched 2026-08-10.

[8] liger-kernel on PyPI (JSON API: version, requires_python, license, upload times). https://pypi.org/pypi/liger-kernel/json. Fetched 2026-08-10.

[9] GitHub API tag/commit lookup for `v0.8.1` (commit `a5234dfc`, dated 2026-07-22). https://api.github.com/repos/linkedin/Liger-Kernel/commits/v0.8.1. Fetched 2026-08-10.

[10] Liger-Kernel `setup.py`, read both at the `v0.8.1` release tag and at commit `08e8ccdc` (the two differ only in a dev-only ruff upper bound). https://raw.githubusercontent.com/linkedin/Liger-Kernel/v0.8.1/setup.py and https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/setup.py. Fetched 2026-08-10.

[11] GitHub Releases API for linkedin/Liger-Kernel (last 6+ releases with publish dates). https://api.github.com/repos/linkedin/Liger-Kernel/releases. Fetched 2026-08-10.

[12] Liger-Kernel `examples/alignment/run_orpo.py` and its `accelerate_config.yaml`, at commit `08e8ccdc`. https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/examples/alignment/run_orpo.py. Fetched 2026-08-10.

[13] Liger-Kernel `src/liger_kernel/transformers/monkey_patch.py`, `apply_liger_kernel_to_llama` signature, at commit `08e8ccdc`. https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/src/liger_kernel/transformers/monkey_patch.py. Fetched 2026-08-10.

[14] Liger-Kernel `examples/alignment/` directory listing at commit `08e8ccdc` (contains only `accelerate_config.yaml` and `run_orpo.py`). GitHub Contents API. Fetched 2026-08-10.

[15] trl SFTTrainer docs (`use_liger_kernel` on SFTConfig, `loss_type` interaction). https://raw.githubusercontent.com/huggingface/trl/main/docs/source/sft_trainer.md. Fetched 2026-08-10.

[16] trl DPOTrainer docs (`use_liger_kernel` on DPOConfig). https://raw.githubusercontent.com/huggingface/trl/main/docs/source/dpo_trainer.md. Fetched 2026-08-10.

[17] trl SFTTrainer docs, `loss_type` default and its resolution to `"nll"` when Liger is enabled. https://raw.githubusercontent.com/huggingface/trl/main/docs/source/sft_trainer.md. Fetched 2026-08-10.

[18] trl GRPOTrainer docs (`use_liger_kernel` on GRPOConfig; the `clip_ratio` metric collapse). https://raw.githubusercontent.com/huggingface/trl/main/docs/source/grpo_trainer.md. Fetched 2026-08-10.

[19] Liger-Kernel `src/liger_kernel/chunked_loss/orpo_loss.py` at commit `08e8ccdc` (the `chunk_size` constructor argument and its docstring). https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/src/liger_kernel/chunked_loss/orpo_loss.py. Fetched 2026-08-10.

[20] trl ORPOTrainer docs, "Logged metrics" section. https://raw.githubusercontent.com/huggingface/trl/main/docs/source/orpo_trainer.md. Fetched 2026-08-10.

[21] Liger-Kernel documentation site (live, unversioned, tracks `main`). https://linkedin.github.io/Liger-Kernel/Getting-Started/ and https://linkedin.github.io/Liger-Kernel/Examples/ (HTTP 200); https://linkedin.github.io/Liger-Kernel/v0.8.1/ (HTTP 404). Fetched 2026-08-10.

[22] Liger-Kernel `mkdocs.yml` nav at commit `08e8ccdc`. https://raw.githubusercontent.com/linkedin/Liger-Kernel/08e8ccdc6a1838e3da230c528fff808491817b58/mkdocs.yml. Fetched 2026-08-10.

[23] Liger-Kernel GitHub issue #1246 and its comments (closed 2026-06-03; reporter michaelroyzen, CONTRIBUTOR; fix by Mecoli1219, COLLABORATOR). https://github.com/linkedin/Liger-Kernel/issues/1246. Fetched 2026-08-10.
