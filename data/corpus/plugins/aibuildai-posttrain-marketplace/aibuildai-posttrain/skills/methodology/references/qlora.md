# QLoRA

4-bit-quantize the frozen base model, keep it frozen, and train only 16-bit LoRA adapters on top - backpropagating through the frozen quantized weights - so a 65B model fine-tunes on one 48GB GPU.

**QLoRA** (Quantized Low-Rank Adaptation) is a parameter-efficient fine-tuning method, defined by Dettmers et al. as "an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance" by backpropagating gradients through a frozen, 4-bit quantized pretrained model into LoRA adapters. The paper is at https://arxiv.org/abs/2305.14314 [1]. Its parent is LoRA, which freezes the pretrained weights and injects trainable low-rank decomposition matrices into each Transformer layer [2]. QLoRA keeps that adapter mechanism but additionally quantizes the frozen base weights to 4-bit for storage, dequantizing them to BFloat16 on the fly for each forward/backward matrix multiply, so gradients still flow in 16-bit precision while only the base model sits in 4-bit [1]. Three innovations make this work without the accuracy loss normally seen at 4-bit: 4-bit NormalFloat (NF4), an information-theoretically optimal data type for normally-distributed weights; Double Quantization, which quantizes the quantization constants themselves for further memory savings; and Paged Optimizers, which use NVIDIA unified memory to page optimizer states to CPU RAM and avoid out-of-memory crashes from gradient-checkpointing memory spikes [1].

The paper's own headline result is the Guanaco model family: QLoRA-tuned on OASST1, Guanaco 65B reaches 99.3% of ChatGPT's score on the Vicuna benchmark after 24 hours of fine-tuning on a single professional GPU, and the 7B variant fits in 5GB and outperforms a 26GB Alpaca-13B model by nearly 20 percentage points on the same benchmark (Table 6) [1]. The method is now the default recipe the Hugging Face stack ships for combining quantization with LoRA: `transformers`' bitsandbytes guide has a dedicated "QLoRA" section for setting the 4-bit compute dtype and NF4 type [3], and `peft`'s quantization guide instructs users to call `prepare_model_for_kbit_training()` before wrapping a 4-bit model in `LoraConfig` [4]. LoftQ, published five months later, is a named successor that replaces QLoRA's LoRA initialization with a quantization-aware one to close a further accuracy gap [5]. Lineage: LoRA (2021) [2] -> QLoRA (2023) [1] -> LoftQ (2023) [5]. The paper was confirmed as the correct match against four same-acronym alternates by citation count: 5,192 citations versus a next-highest alternate's 53.

**When to pick it**: pick QLoRA over plain LoRA [2] when the base model itself does not fit in available GPU memory at 16-bit - QLoRA's own comparison shows 4-bit NF4 with double quantization matching 16-bit LoRA's MMLU accuracy at every LLaMA size from 7B to 65B (Table 4, mean 53.1 vs 53.0) [1], so the quantization costs no accuracy at the scales tested. Pick full fine-tuning only if the resource budget allows it and you need every parameter trainable; the paper's own experiments only established parity with full fine-tuning up to 3B parameters (Table 3), not at 33B/65B [1]. Pick LoftQ [5] instead of QLoRA when quantizing to an aggressive bit-width (LoftQ targets 2-bit and mixed 2/4-bit) where QLoRA's LoRA-adapter initialization (small random A, zero B, per [2]) leaves a larger quantization-induced gap [5].

**Variant of**: LoRA [2].

**Data it needs**: whatever the outer training objective needs - QLoRA is a fine-tuning mechanism, not an objective. The paper's own instruction-tuning study used 8 instruction datasets ranging from 9,209 examples (OASST1, filtered from 161,443 raw crowd-sourced entries down to the top reply at each conversation-tree level) to 240,670 examples (Unnatural Instructions), including Alpaca (51,942), Self-Instruct (82,612), FLAN v2, Chip2 (210,289), and LongForm (23,700) [1]. Training is on-policy in neither sense that matters for RL - it is standard supervised fine-tuning on fixed prompt/response text, run once over the dataset; the paper used QLoRA to train more than 1,000 models across scales and datasets for its ablations [1].

**Extra models**: none beyond the base model itself. QLoRA's own definition needs only the frozen 4-bit base model and the trained 16-bit LoRA adapters - no value network, no separate reward model, no reference model [1]. `peft`'s `prepare_model_for_kbit_training()` operates on the same quantized base model passed in; it does not load a second model [4].

**Shipped by**: `transformers` `BitsAndBytesConfig` (`load_in_4bit`, `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant`, `bnb_4bit_compute_dtype`) provides the quantization side [3], `peft` exports `LoraConfig`, `get_peft_model`, and `prepare_model_for_kbit_training` from its top-level package [6] for the adapter side, and `trl`'s `SFTTrainer` accepts both a `quantization_config` and a `peft_config` together to run QLoRA-style supervised fine-tuning in one call [7]. There is no single `QLoRATrainer` class in any of these libraries; QLoRA is assembled from a quantized-loading config plus a LoRA config, all actively maintained as of the 2026-08-06/2026-08-07/2026-08-08 commits read for this card [6][7][8].

## How it works

Each forward/backward pass: dequantize the relevant 4-bit weight block to BFloat16, run the matmul, add the LoRA adapter's contribution, and only ever compute gradients for the LoRA parameters [1].

**LoRA's linear layer**, inherited unchanged [2] ([1], Eq. 3):

$$ Y = XW + sXL_1L_2 $$

where $L_1 \in \mathbb{R}^{h\times r}$, $L_2 \in \mathbb{R}^{r\times o}$, and $s$ is a scalar.

**4-bit NormalFloat (NF4)**: pretrained weights are approximately zero-centered normal with some standard deviation $\sigma$, so QLoRA fixes a single data type for $N(0,1)$ rescaled into $[-1,1]$ rather than estimating quantiles per tensor [1]. The $2^k$ quantization levels of a $k$-bit NormalFloat data type are set from the quantile function $Q_X$ of the standard normal ([1], Eq. 4):

$$ q_i = \frac{1}{2}\left(Q_X\!\left(\frac{i}{2^k+1}\right) + Q_X\!\left(\frac{i+1}{2^k+1}\right)\right) $$

An asymmetric construction (separate quantile sets for the negative and positive halves, with one duplicate zero removed) guarantees an exact representable zero, which matters for quantizing padding values without error [1]. A weight tensor is quantized by absolute-max rescaling into $[-1,1]$ before mapping onto these fixed levels [1].

**Double Quantization**: quantization constants themselves cost memory - a 32-bit constant with a blocksize of 64 for $W$ adds $32/64 = 0.5$ bits per parameter on average [1]. Double Quantization quantizes those first-level constants again, using 8-bit floats with a blocksize of 256 (chosen because 8-bit second-level quantization showed no degradation), reducing the average cost per parameter from $0.5$ bits to $8/64 + 32/(64\cdot256) = 0.127$ bits - a saving of $0.373$ bits per parameter [1]. Worked example at 7B parameters: $0.373 \times 7\times10^9 / 8 \approx 326$ MB saved just from this second quantization pass, on top of the base 4-bit storage.

**Paged Optimizers**: allocate optimizer state in NVIDIA unified memory so pages are automatically evicted to CPU RAM when the GPU runs out of memory during a gradient-checkpointing spike, and paged back in when needed for the optimizer step [1] - a mechanism, not a math term.

**The formal definition** for one linear layer with one adapter ([1], Eq. 5-6):

$$ Y_{BF16} = X_{BF16}\,\mathrm{doubleDequant}(c_1^{FP32}, c_2^{k\text{-bit}}, W^{NF4}) + X_{BF16}\,L_1^{BF16}L_2^{BF16} $$

$$ \mathrm{doubleDequant}(c_1^{FP32}, c_2^{k\text{-bit}}, W^{4bit}) = \mathrm{dequant}(\mathrm{dequant}(c_1^{FP32}, c_2^{k\text{-bit}}), W^{4bit}) = W_{BF16} $$

$W$ is stored in NF4 with blocksize 64 for quantization precision; its constants $c_2$ are stored in FP8 with blocksize 256 to conserve memory [1]. Only the LoRA parameters' gradients $\partial E/\partial L_i$ are computed; the 4-bit weights never receive a gradient, though computing $\partial E/\partial L_i$ still requires dequantizing $W$ to BFloat16 via the same equation to get $\partial X/\partial W$ [1].

**The critical placement finding**: applying LoRA only to the attention query/value projections - standard practice for LoRA - fails to match full fine-tuning at large scale; the paper finds LoRA must be added to all linear layers of every transformer block to recover full-finetuning performance, while the rank $r$ itself has little effect (Figure 2) [1]. `peft`'s quantization guide reflects this directly, recommending `target_modules="all-linear"` for QLoRA-style training [4].

## Cost

**Theory, from the method's own math**: no extra trained model versus LoRA - the memory savings come entirely from storing the frozen base weights at ~4 bits instead of 16, at the cost of a dequantization pass to BFloat16 on every forward and backward matmul that touches the base weights [1]. The paper's own component breakdown for a 7B model trained on FLAN v2 with batch size 1 and LoRA at ~0.2% of base-model weights: LoRA input-activation gradients cost 567MB (dropping to an average 18MB per sequence with gradient checkpointing), the LoRA parameters themselves cost only 26MB, and the 4-bit base model consumes 5,048MB [1] - so with checkpointing the base model dominates memory, not the adapters, and aggressively shrinking the adapters yields only minor savings [1]. A naive reading might expect adapter count/rank to be the main memory lever; the paper's own numbers show it is not, once checkpointing is on [1].

**In practice, per framework**:
- `transformers` `BitsAndBytesConfig`: `bnb_4bit_compute_dtype` defaults to `float32`, not the paper's BFloat16; the docs explicitly instruct changing it to `bf16` "to speedup computation" [3] - leaving the default in place computes the dequantized matmuls in fp32, which is slower and does not match the paper's recipe. Nested (double) quantization is documented as saving "an additional 0.4 bits/parameter" at no performance cost, and the docs give a concrete example of fine-tuning a Llama-13B model on a 16GB T4 GPU at sequence length 1024, batch size 1, with 4 gradient-accumulation steps [3] - a coarser, rounded figure than the paper's own 0.373-bit derivation [1], since it is describing the same mechanism from the framework side rather than re-deriving it.
- `bitsandbytes`: for QLoRA's 4-bit path specifically, the accelerator-support table marks it fully supported on Linux x86-64 NVIDIA GPU (SM60+ minimum, SM75+ recommended), AMD GPU, Intel GPU, and CPU (minimum AVX2, optimized with AVX512F/AVX512BF16), but only "Partially Supported" on Intel Gaudi (Gaudi2/Gaudi3) - and 8-bit optimizers are marked "Not Supported" on Gaudi in the same table, at commit `a2b90e6e` (2026-07-29) [8].
- `peft`: `prepare_model_for_kbit_training()` must be run on the quantized model before wrapping it in `LoraConfig`, per the quantization guide's own recipe [4]; this is a one-time preprocessing call, not a recurring per-step cost.
- `trl` `SFTTrainer`: the `quantization_config` argument is documented as "Ignored if the model is already instantiated" [7] - passing it after loading the model with `from_pretrained` yourself has no effect, a live way to silently train in full precision.

## How to use it

- Load the base model with a `BitsAndBytesConfig` set to `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, and an explicit `bnb_4bit_compute_dtype` (the transformers default is `float32`, not bf16) [3]; the docs recommend NF4 for training 4-bit base models specifically [3].
- Call `prepare_model_for_kbit_training()` on the loaded model, then wrap it with `LoraConfig(target_modules="all-linear", ...)` and `get_peft_model()` [4][6] - `all-linear` is `peft`'s own recommendation for matching the paper's all-layer placement [4].
- No reward or label conventions beyond the outer objective: the paper's own use is standard supervised fine-tuning, masking padding tokens out of the cross-entropy loss the same way any SFT run does [1]; `trl`'s `SFTTrainer` masks padding via an ignore index of `-100` by default [7].
- Key knobs, with each source's own value ("not stated" = the source was checked and gives none; "not checked" = this card did not verify that source's key):

| knob | paper [1] | `peft` `LoraConfig` default [9] | `transformers` `BitsAndBytesConfig` default [3] | `trl` `SFTConfig` default [10] |
| --- | --- | --- | --- | --- |
| LoRA rank $r$ | 64 | 8 | n/a | not checked |
| LoRA $\alpha$ | 16 | 8 | n/a | not checked |
| LoRA dropout | 0.1 (≤13B), 0.05 (33B/65B) | 0.0 | n/a | not checked |
| target modules | all linear layers | `None` (must set `"all-linear"` explicitly) | n/a | n/a |
| compute dtype | BFloat16 | n/a | `float32` (must set to `bfloat16`) | n/a |
| double quantization | on (all experiments) | n/a | `bnb_4bit_use_double_quant=False` [11] | n/a |
| Adam $\beta_2$ | 0.999 | not stated (framework Adam default) | n/a | not checked |
| max grad norm | 0.3 | n/a | n/a | not checked |
| learning rate | 2e-4 (13B) / 1e-4 (33B/65B) | n/a | n/a | 2e-5 |
| LR schedule | constant | n/a | n/a | not checked |

  Two readings of that table: `peft`'s default rank (8) and alpha (8) are far below the paper's tuned values (64 and 16) [1][9] - the paper itself found rank had little effect on quality (Figure 2) [1], so raising it is not obviously necessary, but a reader silently inheriting the library default is not reproducing the paper's setting. And `trl`'s SFT learning-rate default (2e-5) sits an order of magnitude below the paper's own QLoRA learning rates (1e-4 to 2e-4) [1][10], the opposite direction from the usual "SFT rate is too high for adapters" trap - QLoRA needs a comparatively large rate because only the small adapter matrices are updated.
- The paper uses a constant learning-rate schedule after benchmarking linear and cosine alternatives, and groups training examples by length in each batch for efficiency, noting this produces an oscillating loss curve as an expected, not a broken, signal [1].

## While it runs

- **Signals and their healthy shapes**: QLoRA itself adds no new training-time metric beyond what the outer SFT (or RL) objective already logs; `trl`'s `SFTTrainer` reports the standard SFT set - `loss`, `entropy`, `mean_token_accuracy`, `learning_rate`, and `grad_norm` (the L2 gradient norm before clipping) [7]. The paper's own qualitative signal is the loss curve shape: with group-by-length batching enabled, expect an oscillating loss curve, not a smooth one - this is documented as expected, not a sign of a problem [1].
- **Published reference runs**: the paper releases the Guanaco family and its full Table 6 Vicuna-benchmark and Table 1 Elo-rating results, including 95% confidence intervals over 10,000 random match orderings, as its own reference numbers [1]; no additional third-party published QLoRA training curve was checked for this card.
- **Degeneracies and defaults**: `bnb_4bit_compute_dtype` silently defaulting to `float32` in `transformers` [3] trains slower without raising an error. `trl`'s `quantization_config` is silently ignored if the model was already instantiated before being passed to `SFTTrainer` [7], another silent divergence from the intended QLoRA recipe. Reverting to attention-only LoRA placement (the pre-QLoRA "standard practice") is documented by the paper itself to fail to match full-finetuning performance at scale, even though nothing in the config prevents it [1].
- **Named successors**: LoftQ replaces the standard LoRA zero/random adapter initialization with a quantization-aware one, closing a further gap the authors report between full fine-tuning and quantization-plus-LoRA fine-tuning, especially at 2-bit and mixed 2/4-bit precision [5].
- **Known failure modes**: the paper's own Limitations section states it did not establish that QLoRA matches full 16-bit fine-tuning performance at the 33B and 65B scales specifically (only up to 3B in the controlled comparison), due to the resource cost of that experiment [1]; it evaluated only MMLU, Vicuna, and OA benchmarks, not BigBench, RAFT, or HELM, and does not claim those results generalize [1]; and it performed only a limited responsible-AI evaluation, checking Guanaco-65B for one type of social-bias generation and explicitly leaving other bias types to future work [1]. Separately, in `peft`'s closed GitHub issues (repo `huggingface/peft`, searched for "qlora merge" in title/body, closed, sorted by comment count, 2026-08-08), issue #868 reports that `merge_and_unload()` on a QLoRA-trained adapter can silently produce a corrupted, undersized merged model when some layers were offloaded to the "meta" device for lack of memory. Maintainer BenjaminBossan replied that layers on the meta device are not properly handled during the merge, so their weights and the corresponding LoRA weights go missing from the result, and in a follow-up comment explained that those weights end up on meta device because they had to be offloaded for lack of memory in the first place, concluding "I'm really not sure what a solution would look like here" [12].
- **What the gain is - and is not**: the paper's own summary is that 8-bit and 4-bit adapter fine-tuning replicate 16-bit full-fine-tuning and 16-bit LoRA performance on GLUE and Super-NaturalInstructions (Table 3) and on MMLU at 7B-65B scale (Table 4) [1] - the gain is memory reduction at matched quality on the benchmarks tested, not a capability increase. The authors also note that Vicuna-benchmark and MMLU rankings diverge depending on how similar the fine-tuning data is to each benchmark's own distribution (FLAN v2 tracks MMLU, Chip2 tracks the chatbot benchmarks), so a QLoRA run's benchmark score reflects data match as much as method quality [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Framework docs are `main`/unpinned builds; every default quoted above is a 2026-08-08 reading unless a source-code commit is given, and re-checking against the version you install is expected.

[1] Dettmers, Pagnoni, Holtzman, Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs", 2023. https://arxiv.org/abs/2305.14314 - defines QLoRA: NF4, Double Quantization, Paged Optimizers, formal definition (Eq. 3-6), Tables 1-6, Table 9 hyperparameters, Appendix B dataset sizes, Section 8 Limitations. Fetched 2026-08-08 (arXiv PDF, extracted with pdftotext).

[2] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. https://arxiv.org/abs/2106.09685 - the parent method: freezing pretrained weights, injecting trainable rank-decomposition matrices. Fetched 2026-08-08 (abstract page).

[3] Hugging Face `transformers` bitsandbytes quantization guide. https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes - `BitsAndBytesConfig` fields, `bnb_4bit_compute_dtype` default and recommendation, nested-quantization memory saving and example. Fetched 2026-08-08.

[4] Hugging Face `peft` quantization guide. https://huggingface.co/docs/peft/main/developer_guides/quantization - `prepare_model_for_kbit_training()`, `target_modules="all-linear"` recommendation. Fetched 2026-08-08.

[5] Li et al., "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models", 2023. https://arxiv.org/abs/2310.08659 - named successor: quantization-aware LoRA initialization. Fetched 2026-08-08 (abstract page).

[6] `peft` top-level package exports, `src/peft/__init__.py`, `huggingface/peft` GitHub repository, commit `5f55a6331b6a`, 2026-08-06. https://github.com/huggingface/peft - confirms `LoraConfig`, `get_peft_model`, `prepare_model_for_kbit_training` are exported at the top level. Fetched 2026-08-08.

[7] Hugging Face `trl` `SFTTrainer` documentation. https://huggingface.co/docs/trl/main/en/sft_trainer - `quantization_config`/`peft_config` parameters, "Ignored if the model is already instantiated", loss masking with ignore index -100, logged metrics. Fetched 2026-08-08.

[8] `bitsandbytes` GitHub README, `bitsandbytes-foundation/bitsandbytes`, main branch, commit `a2b90e6e`, 2026-07-29. https://github.com/bitsandbytes-foundation/bitsandbytes - supported hardware backends and minimum compute capability/CPU instruction sets. Fetched 2026-08-08.

[9] `peft` `LoraConfig` source, `src/peft/tuners/lora/config.py`, `huggingface/peft` GitHub repository, main branch. https://github.com/huggingface/peft - `r` default 8, `lora_alpha` default 8, `lora_dropout` default 0.0. Fetched 2026-08-08.

[10] `trl` `SFTConfig` source, `trl/trainer/sft_config.py`, `huggingface/trl` GitHub repository, main branch, commit `2396dfe5`, 2026-08-07. https://github.com/huggingface/trl - `learning_rate` default 2e-5. Fetched 2026-08-08.

[11] `transformers` `BitsAndBytesConfig` source, `src/transformers/utils/quantization_config.py`, `huggingface/transformers` GitHub repository, main branch, commit `e8ea728a`, 2026-08-08. https://github.com/huggingface/transformers - `bnb_4bit_use_double_quant` default `False`. Fetched 2026-08-08.

[12] `huggingface/peft` GitHub issue #868, "merge_and_unload issue?", closed. https://github.com/huggingface/peft/issues/868 - maintainer BenjaminBossan on `merge_and_unload()` failing for layers offloaded to the "meta" device. Found via GitHub issue search (`repo:huggingface/peft qlora merge in:title,body is:issue is:closed`, sorted by comments) on 2026-08-08.
