# LOMO

SGD with the gradient-computation and parameter-update steps fused into one, so the optimizer never holds a full gradient tensor or any optimizer state, only the largest single layer's gradient.

**LOMO** (LOw-Memory Optimization) is a full-parameter fine-tuning optimizer for LLMs, introduced by "Full Parameter Fine-Tuning for Large Language Models with Limited Resources" as a way to fine-tune all of a model's weights - not a small adapter - on hardware that cannot hold Adam-style optimizer state [1]. The paper is at https://arxiv.org/abs/2306.09782 [1]. Its parent is plain SGD [2]: the paper argues, and fuses, the update $p \leftarrow p - lr \cdot \partial\mathcal{L}/\partial p$ directly during the backward pass, one parameter at a time, instead of first materializing a gradient tensor for every parameter and then stepping an optimizer over all of them [1]. The paper gives three reasons SGD is workable for LLM fine-tuning specifically, not fine-tuning in general: pretrained LLMs sit on a loss surface that is empirically and theoretically smoother than a general model's, so SGD's classic instability near sharp curvature is less of a problem; fine-tuning only needs a local optimum near the pretrained weights, not SGD's global-optimum guarantee; and the pretrained initialization sits in a "valley" close to the target task, so the saddle points SGD struggles to escape are less likely to be encountered [1]. Given that, the paper's contribution is the fused update itself, which removes the gradient-tensor and optimizer-state memory that would normally sit alongside SGD [1].

LOMO is shipped as a first-party optimizer choice inside Hugging Face's `transformers` `Trainer` (`optim="lomo"` / `"adalomo"`) and inside `accelerate` (`Accelerator.lomo_backward`), both reading from the reference `lomo-optim` PyPI package [3][4][5]. In the paper's own memory profile, switching LLaMA-7B training from AdamW to LOMO cut peak per-GPU memory from 102.20GB to 14.58GB, and from 51.99GB to 14.58GB versus plain (unfused) SGD - a reduction the paper reports as 10.8% of the standard DeepSpeed solution's memory usage, enabling full-parameter fine-tuning of a 65B model on a single machine with 8x24GB RTX 3090s [1]. Lineage in one line: SGD (classic [2]) -> LOMO (2023 [1]) -> adaptive-LR successor AdaLomo (2023 [6]). Between the two candidates surfaced for this row, LOMO was confirmed as the primary paper by citation count: 229 citations against 38 for the alternate candidate, AdaLomo.

**When to pick it**: pick LOMO when the constraint is GPU memory for full-parameter fine-tuning of an LLM and Adam-style optimizer state (two full-precision moment buffers per parameter) does not fit, and when you are willing to hand-tune a plain-SGD-style learning rate instead of getting Adam's per-parameter adaptivity for free [1]. Prefer AdamW [7] when memory is not the binding constraint - the paper's own results show LOMO trailing LoRA on some SuperGLUE tasks even though both were trained on the same 1,000-example sets, which the paper attributes to that training set being too small for full-parameter fine-tuning to show its advantage [1]. Prefer LoRA [8] - the nearest parameter-efficient alternative, which freezes the pretrained weights and trains a small low-rank adapter instead of all parameters [8] - when you do not need every weight updated; the paper reports LOMO combines with LoRA rather than competing with it, improving on LoRA alone in most of its runs [1]. Prefer AdaLomo [6] over plain LOMO whenever you can afford its extra factored second-moment buffers (see Extra models), since the AdaLomo paper reports it fixes LOMO's sensitivity to hyperparameters and matches AdamW's downstream accuracy at similar memory savings [6].

**Variant of**: SGD [2], with the paper's own theoretical case for why plain SGD is viable for LLM fine-tuning laid out in Sec. 3.1 [1].

**Data it needs**: whatever labeled or instruction-formatted fine-tuning data the task calls for; the method reads the same (input, target) batches any supervised fine-tuning loop reads. The paper's own SuperGLUE runs used 1,000 training examples per task per model size [1]. LOMO is on-policy in the trivial supervised sense only - each step consumes a fresh mini-batch from the fixed training set - it is not an RL method and involves no policy sampling.

**Extra models**: none. LOMO trains and holds only the one model being fine-tuned; there is no value network, reference model, reward model, or judge in the method's definition [1]. AdaLomo adds no extra model either, but does add per-parameter non-negative-matrix-factorized second-moment buffers (Adafactor-style row/column factors) plus a grouped-update-normalization step, both held alongside the trained model rather than as separate models [6]. Cost has the memory arithmetic.

**Shipped by**: `transformers` `Trainer`, entry point `TrainingArguments(optim="lomo")` / `optim="adalomo"`, which dispatches to `lomo_optim.Lomo` / `lomo_optim.AdaLomo` - no deprecation or experimental marking found in the reviewed source [3]. `accelerate`, entry point `Accelerator.lomo_backward(loss, learning_rate)` [4]. Both depend on the separate `lomo-optim` PyPI package (the reference implementation from `OpenLMLab/LOMO`) being installed [3][4][5].

## How it works

The loop in one line: for each parameter, as soon as its gradient is computed during backpropagation, apply the SGD step to it immediately and discard the gradient, instead of waiting to finish the full backward pass first [1].

**The fused update** [1]. Vanilla gradient descent is a two-step process:

$$ \mathrm{grad} = \frac{\partial \mathcal{L}}{\partial p}, \qquad p = p - lr \cdot \mathrm{grad} $$

LOMO fuses it into one step:

$$ p = p - lr \cdot \frac{\partial \mathcal{L}}{\partial p} $$

Mechanically, this is meant to be implemented with a PyTorch backward hook registered on every parameter, called as soon as that parameter's gradient is computed (Algorithm 1) [1]; the paper itself notes that current PyTorch APIs do not support an exactly immediate update, so its actual implementation instead bounds gradient memory to at most one parameter's gradient at a time, updating each parameter one by one alongside the backward pass [1]. The reference implementation's own README describes the hook mechanics this produces: a hook fires for a parameter once its gradient is computed (but before that gradient is written to `.grad`), the hook updates the parameter and then clears and frees `.grad`; because the last parameter's hook fires before its own `.grad` is set, the implementation performs one additional scan after the backward pass to update that last parameter [5].

**Gradient normalization without a full gradient tensor.** Standard gradient-norm clipping needs every parameter's gradient at once, which LOMO's fused update never holds. The paper's fix is a second backward pass: one pass computes and accumulates the norm across all parameters without updating anything, and a second pass performs the fused update, scaling by the now-known norm - trading one extra backward pass for norm-based clipping [1]. As a cheaper alternative for small-to-medium learning rates, the paper recommends clipping each gradient tensor by value (elementwise) rather than by norm, and reports this works worse at high learning rates because value-clipping can change a gradient tensor's direction, giving as an example the vector $[1.3, 0.8]$ clipped to $[1.0, 0.8]$ [1]; the paper's own guidance is to use value-clipping for a learning rate below $1\text{e-}3$ [1].

**Mixed precision.** LOMO integrates a dynamic loss scaler: if no overflow occurs for a set number of backward passes the scale factor doubles, and if an overflow occurs that step is dropped and the scale factor halves [1]. Detecting overflow needs a completed backward pass before it is known whether the step is safe to apply, so - like gradient normalization - this also costs a second backward pass, and the paper runs the norm and overflow checks in the same extra pass [1]. Parameters and gradients are cast to full precision only inside these normalization/scaling computations [1].

**Worked example.** With a fused SGD step and $lr = 0.03$ (the lower of the paper's two tested LOMO-alone learning rates, $\{0.05, 0.03\}$ [1]), a parameter $p = 1.000$ whose gradient at that instant is $0.20$ updates to $p = 1.000 - 0.03 \times 0.20 = 0.994$ the moment that gradient is available - no separate `.grad` tensor for that parameter persists afterward, and no momentum or variance term from a prior step is added, because LOMO carries none.

**AdaLomo's own formula** [6]. AdaLomo keeps LOMO's fused, hook-driven update but replaces the fixed learning rate with a per-parameter adaptive one, estimating the second moment with non-negative matrix factorization (an Adafactor-style row/column factorization rather than a full per-parameter buffer) and adding a grouped-update-normalization step the paper reports as necessary to stabilize convergence [6]. This card does not reproduce AdaLomo's update equation; it is a distinct optimizer, and the paper's own memory-vs-AdamW comparison is what its Cost line above covers.

## Cost

**Theory, from the method's own math:**

- Time: one backward pass per step in the base case; two backward passes whenever gradient-norm clipping or dynamic loss-scaling overflow detection is active, since both need the full gradient picture before any parameter can be safely updated [1]. Forward-pass cost is unchanged.
- Memory: no optimizer-state tensors at all - LOMO's update is a fused hook, not a stateful optimizer - and only $O(1)$ gradient memory, bounded by the single largest parameter tensor's gradient rather than the sum over all parameters, because each parameter's gradient is freed right after its update [1]. AdaLomo trades some of that back: it holds factored second-moment buffers per parameter (smaller than Adam's full per-parameter moments, but not zero) [6].
- A naive reading might assume the two-backward-pass norm/overflow trick doubles memory as well as time; the paper is explicit that it costs speed only - "the memory usage leaves unchanged" [1].

**In practice, per framework:**

- `transformers` `Trainer`, via `training_args.py`'s `OptimizerNames.LOMO` / `ADALOMO` and `trainer_optimizer.py`'s `_get_lomo_optimizer`: instantiates `lomo_optim.Lomo` or `lomo_optim.AdaLomo` (chosen by whether `"ada"` appears in the `optim` string), raising `ImportError` if the `lomo-optim` package is not installed, and requires the model object to be passed at construction time [3]. Because the update happens inside the backward hook rather than in an `optimizer.step()` call, `Trainer`'s training step explicitly passes the current learning rate into the loss/backward call for these two optimizers [3].
- `accelerate`'s `Accelerator`: exposes `lomo_backward(loss, learning_rate)`, and sets an internal `has_lomo_optimizer` flag once it detects the prepared optimizer is a `Lomo` or `AdaLomo` instance, routing training-step calls through `lomo_backward` instead of a normal `optimizer.step()` [4].
- The paper's own throughput table (8xRTX 3090, sequence length 1024, batch size 1): LLaMA-7B under AdamW needed all 8 GPUs at 15.76GB/GPU and 67.37 tokens/GPU/sec, while LOMO ran the same model on 1 GPU at 13.61GB and 769.92 tokens/GPU/sec (higher because no cross-GPU sharding was needed); at 65B, LOMO ran on 8 GPUs at 19.18GB/GPU peak and 4.93 tokens/GPU/sec [1].

## How to use it

- Data prep is standard supervised fine-tuning: input/target pairs formatted however the base model's fine-tuning recipe expects (instruction pairs, classification prompts, etc.); LOMO changes the optimizer, not the data pipeline [1].
- No reward or preference labels are involved; loss is whatever supervised loss the fine-tuning task uses (the paper's SuperGLUE runs use standard task losses) [1].
- Key knobs, with each source's own value. "not stated" means the source was checked and does not give the value; "not checked" means this card did not verify that source's key:

| knob | paper (LOMO, SuperGLUE runs) [1] | `lomo-optim` package via `transformers`/`accelerate` [3][4] |
| --- | --- | --- |
| learning rate | `5e-2` or `3e-2` (LOMO alone); `5e-3`/`1e-3`/`5e-4` (LoRA+LOMO) | not stated (package default; not checked) |
| LR schedule | linear | not checked |
| max grad norm | 1.0 | not checked |
| warmup ratio | 0.05 / 0.1 / 0.2 | not checked |
| batch size | 16 | not checked |
| epochs | 10 | not checked |

  The paper's own learning rates for LOMO (`5e-2` down to `5e-4` depending on setup) sit orders of magnitude above a typical AdamW fine-tuning rate (commonly `1e-5`-`1e-4`); this is expected for an unnormalized SGD-style update and is the sensitivity AdaLomo was built to reduce [1][6]. `transformers` and `accelerate` do not set their own optimizer-specific defaults for LOMO/AdaLomo in the reviewed source - they forward whatever the caller passes through `TrainingArguments`/`optimizer_kwargs` to the `lomo-optim` package - so this card cannot state a framework-level default learning rate; the paper's values are the only anchor found [1][3][4].
- Trade-off a run designer faces: choosing the norm-clipping/loss-scaling path (a second backward pass, exact but slower) versus value-based clipping (one backward pass, faster, but the paper's own finding is it degrades at learning rates above roughly `1e-3`) [1].

## While it runs

- Signals and their healthy shapes: the paper's own training-dynamics appendix shows, for LLaMA-7B fine-tuned with LOMO on BoolQ, a loss curve that converges rapidly in the initial phase and then stabilizes and gradually declines, with development-set accuracy trending upward as training steps increase [1]. No loss-value or accuracy thresholds are given beyond that qualitative shape; the appendix is a single dataset/model-size example, not a general reference curve.
- Published reference runs: the paper's Table 3 (SuperGLUE, LLaMA 7B/13B/30B/65B, LOMO vs. LoRA vs. zero-shot) is the closest thing to a reference set - e.g., LOMO reaches an 80.8-point six-task average at 7B and 89.9 at 65B, against LoRA's 78.8 and 89.0 at the same sizes [1]. These are averages over six SuperGLUE tasks with only 1,000 training examples each, per the paper's stated setup - not a general-purpose benchmark for other data scales [1].
- Degeneracies and defaults: the paper reports that clipping gradients by value rather than by norm performs worse at high learning rates specifically, because truncating individual gradient elements can change a gradient tensor's direction, an effect the paper illustrates but does not attach a numeric threshold to beyond "less than `1e-3`" as its own guidance for when value-clipping is safe [1]. No config default is documented by `transformers` or `accelerate` for the optimizer's own hyperparameters (see How to use it) [3][4].
- Named successor: AdaLomo, from the same group, is presented as fixing "the fact that [LOMO's] optimization technique, akin to stochastic gradient descent, is sensitive to hyper-parameters and exhibits suboptimal convergence, failing to match the performance of the prevailing optimizer for large language models, AdamW" [6], by adding a per-parameter adaptive learning rate (factored second-moment estimation) and grouped update normalization [6].
- Known failure modes: from the paper's own results, LOMO underperforms LoRA on some SuperGLUE tasks despite training on the same 1,000-example sets, which the authors attribute to that training set being too small to benefit from full-parameter updates, plus architectural differences between the two methods' training dynamics [1]. No search of a maintainer issue tracker (e.g., `OpenLMLab/LOMO` GitHub issues) was performed for this card; no claim is made about undocumented failure modes beyond the paper's own limitations discussion.
- What the gain is - and is not: the paper's own contribution is stated as a memory reduction that preserves the SGD update itself unchanged ("we ensure that the fine-tuning process remains uncompromised, as the parameter update process is still equivalent to SGD") [1] - LOMO is not claimed to improve optimization quality over SGD, only to make full-parameter SGD fit in less memory; where it beats LoRA (e.g., +2.8 points at 13B in the paper's own comparison [1]), the gain is attributed to updating all parameters rather than a low-rank subset, not to anything specific to the fused-update mechanism.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Framework source reads (`transformers`, `accelerate`) are pinned to the `main`-branch commit each repository reported at fetch time on 2026-08-09; `main` itself is unpinned and mutable, so re-check against the version installed.

[1] Lv et al., "Full Parameter Fine-tuning for Large Language Models with Limited Resources", 2023. https://arxiv.org/abs/2306.09782 - defines LOMO: theoretical case for SGD, fused-update algorithm, gradient-norm/clipping and loss-scaling stabilization, memory profile, throughput, SuperGLUE results, hyperparameters, training dynamics. Fetched 2026-08-09 (HTML full text via arxiv.org/html).

[2] Robbins and Monro, "A Stochastic Approximation Method", Annals of Mathematical Statistics, 1951 - the classic reference behind stochastic gradient descent, LOMO's parent method. Not fetched at check time; cited as the standard attribution for SGD, consistent with [1]'s own framing of LOMO as a fused SGD update.

[3] `transformers` source, `src/transformers/trainer_optimizer.py` and `src/transformers/training_args.py`, commit `e8ea728a3eeeb903e77c7d1bd29267c80a1be71f` (tip of `main` at fetch time). https://github.com/huggingface/transformers - defines `OptimizerNames.LOMO`/`ADALOMO` and `_get_lomo_optimizer`, which imports `Lomo`/`AdaLomo` from `lomo_optim`; `Trainer.training_step` passes the learning rate explicitly for these optimizers. Fetched 2026-08-09.

[4] `accelerate` source, `src/accelerate/accelerator.py`, commit `16cb6eb80dd9aa8b4df1a63ef57863e455d53b83` (tip of `main` at fetch time). https://github.com/huggingface/accelerate - defines `Accelerator.lomo_backward` and the `has_lomo_optimizer` detection path for `Lomo`/`AdaLomo` instances. Fetched 2026-08-09.

[5] `OpenLMLab/LOMO` GitHub repository. https://github.com/OpenLMLab/LOMO - reference implementation (`lomo-optim` PyPI package); README states LOMO and AdaLomo are integrated into `transformers` and `accelerate`, and into the `CoLLiE` library. Fetched 2026-08-09 (README and repository metadata via GitHub API).

[6] Lv et al., "AdaLomo: Low-memory Optimization with Adaptive Learning Rate", 2023. https://arxiv.org/abs/2310.10195 - the named successor: adaptive per-parameter learning rate via non-negative matrix factorization of the second moment, plus grouped update normalization; states LOMO's sensitivity-to-hyperparameters limitation directly. Fetched 2026-08-09 (HTML full text).

[7] Loshchilov and Hutter, "Decoupled Weight Decay Regularization", 2017. https://arxiv.org/abs/1711.05101 - AdamW, the memory-heavier alternative LOMO is compared against throughout [1]. Fetched 2026-08-09 (abstract page).

[8] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. https://arxiv.org/abs/2106.09685 - LoRA, the nearest parameter-efficient alternative referenced throughout [1]'s experiments. Fetched 2026-08-09 (abstract page).
