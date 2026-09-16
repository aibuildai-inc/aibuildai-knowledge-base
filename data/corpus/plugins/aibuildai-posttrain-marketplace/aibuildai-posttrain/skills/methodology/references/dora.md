# DoRA

LoRA with the pre-trained weight split into a frozen unit-norm direction and a trainable magnitude scalar per output column, so the low-rank adapter only has to learn the direction while a cheap extra vector learns the magnitude.

**DoRA** (Weight-Decomposed Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) method for adapting a pre-trained weight matrix to a downstream task, introduced by Liu et al. as a variant of LoRA motivated by a new decomposition analysis of how LoRA and full fine-tuning (FT) update weights differently [1]. The paper is at https://arxiv.org/abs/2402.09353 [1]. Its parent, LoRA, represents the weight update as the product of two low-rank matrices $B$ and $A$ added to the frozen pre-trained weight [2]. DoRA first rewrites the pre-trained weight $W_0$ as a magnitude vector $m$ times a direction matrix $V$ normalized to unit column norm, then keeps $m$ directly trainable while routing only the directional update through a LoRA pair $BA$; the two pieces merge back into one weight matrix before inference, so DoRA adds no inference latency over the frozen model [1]. The paper gives two reasons for this split: a weight-decomposition analysis of the same query weights shows LoRA's magnitude and direction changes across layers and training steps are proportionally correlated, unlike FT's changes, which are not, so LoRA seems to lack the capacity to adjust one independently of the other; and a gradient analysis shows the decomposition scales and reprojects the low-rank update's gradient in a way the authors argue improves optimization stability, by analogy with Weight Normalization [1].

The paper was accepted at ICML 2024 as an Oral presentation, per the method's own repository [3]. DoRA is bundled into Hugging Face PEFT as a flag on the standard LoRA config, and Answer.AI's QDoRA project layered it on top of QLoRA's 4-bit quantization for fine-tuning on consumer GPUs, both described in the paper itself [1]. In the paper's own commonsense-reasoning benchmark (eight tasks, LLaMA-7B), DoRA at matched trainable-parameter count reaches an average accuracy of 78.4% against LoRA's 74.7%, and a half-rank DoRA variant (DoRA†, fewer trainable parameters than LoRA) still reaches 77.5% (Table 1) [1]. Lineage in one line: LoRA (2021 [2]) -> DoRA (2024 [1]) -> QDoRA (Answer.AI, 2024, reported inside the DoRA paper [1]).

**When to pick it**: pick DoRA over plain LoRA [2] when the low-rank adapter is capacity-limited (small rank, or accuracy trailing full fine-tuning) and the extra per-column magnitude parameters and the one-time column-norm computation at adapter init are acceptable; the paper's own low-rank ablation is the deciding case (see How to use it) [1]. Prefer full fine-tuning as the accuracy ceiling DoRA is chasing, not a competing PEFT method, when compute allows it [1]. Prefer VeRA [4] over either when trainable-parameter count must be pushed far below LoRA's, since VeRA freezes a single shared pair of random projection matrices and trains only small scaling vectors; DoRA is compatible with VeRA's projections, and the paper reports the combination (DVoRA) outperforming plain VeRA on instruction tuning (Table 5) [1].

**Variant of**: LoRA [2].

**Data it needs**: whatever the underlying supervised fine-tuning task needs, since DoRA only changes how the weight update is parameterized, not the loss; the paper itself uses cross-entropy fine-tuning on prompt/answer pairs (commonsense reasoning, instruction tuning with a 10,000-example Alpaca subset [1]) and a diffusion loss for text-to-image DreamBooth fine-tuning [1]. It is not tied to on-policy sampling or a reward signal; there is no RL loop in the paper.

**Extra models**: none beyond whatever the base fine-tuning objective already requires (e.g. a reward model only if the surrounding pipeline uses one) - DoRA itself adds only a trainable magnitude vector per adapted matrix plus the usual LoRA $A$, $B$ matrices on top of the frozen pre-trained weight [1]. See Cost for the one-time overhead of computing initial column norms.

**Shipped by**: Hugging Face PEFT, via `LoraConfig(use_dora=True)` (top-level `LoraConfig` export, current default `use_dora=False`; the config's own docstring warns DoRA currently only supports Linear and Conv2D layers and recommends merging weights before inference because of its larger runtime overhead) [5]. trl exposes the same setting as `ModelConfig.use_dora` / the `--use_dora` CLI flag on its training scripts, which it forwards into the `LoraConfig` it builds internally - trl does not implement DoRA itself [6][7].

## How it works

Decompose the frozen weight once, then train direction through a LoRA pair and magnitude directly; recombine before every forward pass.

**Weight decomposition** [1], for a weight matrix $W \in \mathbb{R}^{d\times k}$:

$$ W = m \frac{V}{\lVert V \rVert_c} = \lVert W \rVert_c \, \frac{W}{\lVert W \rVert_c} $$

$m \in \mathbb{R}^{1\times k}$ is the magnitude vector, $V \in \mathbb{R}^{d\times k}$ is the directional matrix, and $\lVert \cdot \rVert_c$ is the norm of each column vector, so every column of $V/\lVert V \rVert_c$ is a unit vector and the matching entry of $m$ is that column's scale [1].

**DoRA's update** [1], initializing $m = \lVert W_0 \rVert_c$ and $V = W_0$ so the adapted weight equals $W_0$ before training starts:

$$ W' = \underline{m}\,\frac{V + \underline{\Delta V}}{\lVert V + \underline{\Delta V} \rVert_c} = \underline{m}\,\frac{W_0 + \underline{B}\,\underline{A}}{\lVert W_0 + \underline{B}\,\underline{A} \rVert_c} $$

Underlined terms are trained: the magnitude vector $m$ directly, and the directional update $\Delta V = BA$ through a LoRA pair, with $B$ and $A$ initialized exactly as in LoRA (so $\Delta V = 0$, hence $W' = W_0$, at step 0) [1]. $V$ itself stays frozen at $W_0$; only $m$ and the low-rank $B, A$ receive gradients [1].

**Worked example** (arithmetic only, not tied to any framework's exact numerics): take a one-column slice of a weight matrix, $w_0 = [3, 4]^\top$. Its column norm is $\lVert w_0 \rVert = 5$, so the decomposition gives magnitude $m = 5$ and unit direction $v/\lVert v\rVert = [0.6, 0.8]^\top$. Suppose training adds a directional update $\Delta v = [0.1, -0.05]^\top$, giving $v' = [0.7, 0.75]^\top$ with $\lVert v' \rVert \approx 1.026$, i.e. a new unit direction $[0.682, 0.731]^\top$; suppose $m$ is separately trained to $5.2$. The adapted weight is then $w' = 5.2 \times [0.682, 0.731]^\top \approx [3.55, 3.80]^\top$ - the magnitude moved from 5 to 5.2 while the direction rotated, independently of each other, which is exactly the independence the paper's decomposition analysis found FT does and LoRA does not (Figure 2) [1].

**Gradient effect**: without detaching, the paper's Eq. 6 gives $\nabla_{V'}L = \frac{m}{\lVert V'\rVert_c}\left(I - \frac{V'V'^\top}{\lVert V'\rVert_c^2}\right)\nabla_{W'}L$ - the ordinary weight gradient scaled by $m/\lVert V'\rVert_c$ and projected away from the current weight direction [1]. The paper argues this scaling and projection make the effective gradient covariance closer to identity, which it links to Weight Normalization's optimization benefits [1]; this is the paper's own argument, not an independently verified stability guarantee. For cost reasons (see Cost), the paper then treats $\lVert V'\rVert_c$ as a constant detached from the gradient graph, which drops the projection term and simplifies the gradient to $\nabla_{V'}L = \frac{m}{\lVert V'\rVert_c}\nabla_{W'}L$ (Eq. 11) [1].

## Cost

**Theory, from the method's own math:**

- Time/memory at inference: none beyond LoRA's, because $m$, $V$, and $\Delta V$ merge into one $d\times k$ matrix before deployment, identical in shape to the base weight [1].
- Time/memory during training: DoRA adds one length-$k$ trainable vector $m$ per adapted matrix (negligible next to $BA$) [1], but naively backpropagating through $\lVert V+\Delta V\rVert_c$ in the denominator would require keeping the full $d\times k$ merged matrix's gradient graph in memory, which the paper flags as extra training memory and removes by detaching that norm from the graph (Eq. 11) - a fix, not a claim that the norm computation itself disappears [1]. That change measured a 24.4% training-memory reduction fine-tuning LLaMA-7B (37.3GB to 28.2GB) and a 12.4% reduction fine-tuning VL-BART (23.4GB to 20.5GB), with accuracy essentially unchanged (78.3 to 78.1 and 77.3 to 77.4 average, respectively) (Table 7) [1].

**In practice, per framework:**

- PEFT: initializing a DoRA adapter is documented by a maintainer as adding a fixed, model-size-dependent overhead beyond plain LoRA - on a 7B model without flash attention, of a 24-second total load-and-init time, about 10.8 seconds were spent in DoRA-specific initialization code, which the maintainer states cannot be avoided given DoRA's extra initialization steps compared to LoRA [8]. Those extra steps are the column-norm computation the method's own definition requires (see How it works) [1]. PEFT's own `use_dora` docstring separately states DoRA carries a bigger runtime overhead than plain LoRA and recommends merging weights for inference [5].
- trl passes `use_dora` straight into the `LoraConfig` it builds for its trainers; trl's own docs give no separate cost figures beyond PEFT's [6][7].

## How to use it

- Reward/label conventions: none specific to DoRA - it inherits whatever loss the surrounding training loop already uses (cross-entropy for the paper's SFT and instruction-tuning experiments) [1].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give it; "not checked" = this card did not verify that source's key):

| knob | paper, LLaMA-7B commonsense [1] | paper, Alpaca instruction tuning [1] | PEFT `LoraConfig` default [5] |
| --- | --- | --- | --- |
| rank $r$ | 16 (DoRA), 8 (DoRA†, half of DoRA's own rank) | not checked | 8 (LoRA default `r`; DoRA reuses the same field) |
| $\alpha$ | 32 | not checked | 8 |
| learning rate | 2e-4 | not checked | not stated (set by the caller) |
| dropout | 0.05 | not checked | 0.0 |
| target modules | Q, K, V, Up, Down | not checked | not stated (set by the caller) |
| `use_dora` | - | - | `False` |

- Rank sensitivity is the paper's deciding ablation: sweeping $r \in \{4,8,16,32,64\}$ on LLaMA-7B commonsense reasoning, LoRA's average accuracy collapses at low rank (40.74% at $r=8$, 39.49% at $r=4$) while DoRA stays far above it (77.96% at $r=8$, 61.89% at $r=4$) - the gap the paper uses to argue DoRA's magnitude/direction split, not just extra parameters, is what closes the FT gap (Sec. 5.5, Table 15) [1].
- Tuning granularity trade-off: Table 6 gives three LLaMA-7B rows - LoRA (0.83% of parameters, m and V both untouched, 74.7% avg), full-module DoRA (updating both magnitude and direction on Q/K/V/Up/Down, 0.84% of parameters, 78.1% avg), and reduced DoRA (magnitude and direction on Q/K/V only, magnitude only on the rest, 0.39% of parameters, 77.5% avg) [1]. The paper states the reduced configuration surpasses LoRA by 2.8 points on LLaMA-7B and 0.8 points on LLaMA-13B while using under half of LoRA's trainable parameters, its deciding number for this ablation [1] - a knob for squeezing trainable-parameter count further once DoRA's default (full-module) configuration is already working.
- Quantized variant: QDoRA (an Answer.AI project, not built by the DoRA authors) combines DoRA with QLoRA's 4-bit NF4 quantized backbone [9] and Fully Sharded Data Parallel for multi-GPU training; on a 100k-example Orca-Math fine-tune, the paper reports it surpasses QLoRA by 0.19 exact-match points on LLaMA2-7B and 0.23 points on LLaMA3-8B, and slightly outperforms full fine-tuning on both models (Fig. 6, reported inside the DoRA paper) [1].

## While it runs

- Signals and their healthy shapes: DoRA introduces no new loss term to monitor beyond whatever the base fine-tuning objective already logs, since it only reparameterizes the weight update [1]. The paper's own diagnostic for whether DoRA is behaving as intended is the correlation between per-layer magnitude change $\Delta M$ and direction change $\Delta D$ across training checkpoints: it reports -0.62 for FT, +0.83 for LoRA, and -0.31 for DoRA, and reads DoRA's negative, FT-like correlation (magnitude and direction moving somewhat independently, sometimes oppositely) as evidence of its closer resemblance to full fine-tuning's learning pattern (Sec. 4.1) [1]; this is not exposed as a training-time metric by PEFT or trl and would need to be computed manually from checkpoints.
- Published reference runs: Table 1 in the paper is the reference curve - final accuracies for DoRA and DoRA† against LoRA on LLaMA-7B/13B, LLaMA2-7B, and LLaMA3-8B commonsense reasoning, with the exact hyperparameters in Table 8 [1]; no raw training logs are published alongside it.
- Degeneracies and defaults: PEFT's `use_dora` defaults to `False`, so a `LoraConfig` copied from a LoRA setup silently trains plain LoRA unless the flag is set explicitly [5]. A PEFT maintainer traced a reported DoRA NaN (on `o_proj` of a Llama-3.1-8B model) to the column-norm denominator: if a whole column of the targeted weight matrix is exactly zero, the division in the decomposition produces a zero norm and propagates NaNs [10]. In that 2024 thread the reporter's own weights were not actually zero in that column, and re-downloading the checkpoint made the NaN disappear, pointing to a corrupted download in that instance [10]; a later comment on the same thread (2026-05-15, not from a maintainer) reports the identical code path producing a NaN on a different, uncorrupted architecture with genuinely zero-valued rows in a targeted linear layer, with a proposed upstream fix (upcasting the norm computation and clamping it away from zero) [10]. So the zero-norm-column failure is not only a download-corruption artifact: it is a live sharp edge of the method's own formula whenever a targeted layer has an exactly- or near-zero row. PEFT's docs also state DoRA currently only supports Linear and Conv2D layers, and that it does not support Megatron core parallelism or `lora_bias=True` [5].
- Named successors: none found as of this card - GitHub search on `huggingface/peft` issues titled with "dora" (16 results, checked 2026-08-08) returned mostly bug reports and cost/compatibility issues, plus one unrelated feature request (issue #2278) asking PEFT to add a different, unrelated "Dynamic Low-Rank Adaptation" method that happens to share the DoRA acronym; none of the 16 results is a follow-on method built on top of this paper's weight decomposition [11].
- Known failure modes: the paper has no dedicated Limitations section; its Impact Statement raises no method-specific limitation [1]. From PEFT's closed issues: the zero-column/zero-row-norm NaN trap above, confirmed on one architecture (Qwen3.5-MoE, 2026-05-15 comment) with genuinely zero-valued rows, while the earlier Llama-3.1-8B report in the same thread was traced to a corrupted checkpoint download rather than a genuine zero-norm reproduction [10]; a maintainer-confirmed, unavoidable extra initialization cost of about 10.8 seconds on a 7B model, from DoRA's initialization code, compared to plain LoRA init [8].
- What the gain is - and is not: the paper's own comparisons are accuracy-per-trainable-parameter on supervised fine-tuning benchmarks (commonsense reasoning, instruction tuning, visual instruction tuning, image/video-text understanding, text-to-image personalization) [1]; it does not claim any new capability over full fine-tuning, only that it narrows LoRA's accuracy gap to full fine-tuning at the same or lower trainable-parameter budget, most clearly at low rank (Sec. 5.5) [1].

## Sources

[1] Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024 (Oral, per the paper's own repository) [3]. https://arxiv.org/abs/2402.09353 - defines DoRA: weight decomposition analysis, method, gradient analysis, training-cost ablation (Table 7), hyperparameters (Table 8), commonsense-reasoning results (Table 1), instruction-tuning/VeRA compatibility (Table 5, Sec. 5.4), rank ablation (Sec. 5.5, Table 15), tuning-granularity ablation (Table 6), QDoRA (Sec. 6.1). Fetched 2026-08-08 (PDF full text via arxiv.org/pdf).

[2] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021 - the parent method's defining paper. https://arxiv.org/abs/2106.09685. Fetched 2026-08-08 (abstract page).

[3] NVlabs/DoRA repository README, ICML 2024 Oral / acceptance-rate claim. https://github.com/NVlabs/DoRA - read at commit 7e2f10abbe8efe212c8fca1d983ae1d04ef13a18 on the `main` branch (2026-03-24), fetched 2026-08-08.

[4] Kopiczko et al., "VeRA: Vector-based Random Matrix Adaptation", 2023 - the nearest-neighbor method DoRA is shown compatible with (DVoRA). https://arxiv.org/abs/2310.11454. Fetched 2026-08-08 (abstract page; DoRA/VeRA compatibility results are from [1], Sec. 5.4).

[5] PEFT `LoraConfig` source, `use_dora` field and docstring. https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py - read at commit 2969e633a9540e4855767639cca653b962fecadc on the `main` branch, 2026-08-08.

[6] trl `ModelConfig.use_dora` field. https://github.com/huggingface/trl/blob/main/trl/trainer/model_config.py - read at commit 01470f1ac6646f8146d16473122d2ac84db0ba83 on the `main` branch, 2026-08-08.

[7] trl PEFT integration docs, `--use_dora` CLI flag. https://github.com/huggingface/trl/blob/main/docs/source/peft_integration.md - unpinned `main` build, read 2026-08-08.

[8] huggingface/peft issue #1593, "Getting Dora Model Is Very Slow", maintainer BenjaminBossan's timing breakdown. https://github.com/huggingface/peft/issues/1593. Fetched 2026-08-08 via the GitHub REST API.

[9] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023 - the 4-bit quantized backbone QDoRA is built on. https://arxiv.org/abs/2305.14314. Fetched 2026-08-08 (abstract page; the DoRA paper cites this work by name for QDoRA, see [1]).

[10] huggingface/peft issue #2049, "Applying Dora to o_proj of Meta-Llama-3.1-8B results in NaN", maintainer BenjaminBossan's explanation of the zero-column-norm cause. https://github.com/huggingface/peft/issues/2049. Fetched 2026-08-08 via the GitHub REST API.

[11] GitHub search, `repo:huggingface/peft dora in:title type:issue`, 16 results, checked 2026-08-08 - basis for the "no named successor found" statement.
