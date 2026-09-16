# LoFiT

Freeze the whole model and learn two small vectors per attention head: which ~3-10% of heads matter for this task, and a per-head offset to add to their output.

**LoFiT** (Localized Fine-tuning on LLM Representations) is a two-step PEFT method introduced by Yin, Ye, and Durrett at https://arxiv.org/abs/2406.01563 [1]. It first picks a small set of attention heads worth modifying for a target task, then trains a fine-tuning step that learns offset vectors added to only those heads' hidden representations [1]. Its parent is the family of localized representation-intervention methods the paper formalizes as a tuple $\langle T, V\rangle$ of a target head set $T$ and offset vectors $V$ - a framework that Inference-time Intervention (ITI) and Representation Engineering (RepE) instantiate in a learning-free way [1][2][3]. LoFiT keeps that same intervention shape but learns both pieces: step 1 trains a per-head scaling vector with an L1 penalty and ranks heads by its norm to pick the top-K; step 2 freezes those scalars and trains a bias vector per selected head with the task's own loss [1]. The paper's stated motivation is twofold: existing PEFT methods that edit representations rather than weights are typically applied to the network uniformly, or treat the choice of which modules to tune as a hyperparameter, without any explicit interpretability-driven localization step, while existing representation-intervention methods such as ITI and RepE compute their offset vectors in a learning-free way rather than training them; the paper sets out to test whether localization can help fine-tuning and whether learned offsets beat intervention-computed ones [1]. The related-work discussion adds that focusing updates on a named subset of heads is presented as a basis for continual learning and model merging, unlike undifferentiated PEFT methods such as LoRA [4] or ReFT [5], which tune weights or a full-network representation subspace without a localization step [1].

LoFiT has no third-party framework adoption at the time of this card - it is a NeurIPS 2024 research method with only the authors' own reference implementation [1][6]. In the paper's own results, LoFiT reaches 74.0/74.3/74.6 average accuracy (TruthfulQA MC1/MC2, MQuAKE EM, CLUTRR EM) on Gemma-7B/Llama-2-7B/Llama-2-13B, against ITI's 49.1/46.4/48.3 and RepE's 53.3/53.8/55.1 at the same 3%-head budget (Table 1) [1]. Against PEFT baselines at a 10%-head budget, LoFiT matches LoRA and outperforms ReFT on average while updating 20-200x fewer parameters than LoRA and RED (Table 3; RED is a scaling+bias method without localization) [1]. Lineage in one line: ITI/RepE (learning-free intervention, 2023 [2][3]) and LoRA (weight-space PEFT, 2021 [4]) -> LoFiT (localized, learned intervention, NeurIPS 2024 [1]), concurrent with ReFT (2024 [5]). The paper was confirmed as the correct source by an exact-title search returning a single candidate, "LoFiT: Localized Fine-tuning on LLM Representations" [1].

**When to pick it**: a truthfulness- or reasoning-style task with a small labeled set (the paper uses 100-500 examples [1]) where you want an auditable, tiny set of tuned attention heads rather than a dense weight update; pick LoRA [4] instead when the task needs resurfacing memorized world knowledge, where the paper found LoFiT falls slightly short of LoRA and RED on commonsense/math benchmarks (SIQA, ARC-c, BoolQ, SVAMP) [1]; pick ITI [2] or RepE [3] instead only when no labeled training data is available at all, since both are learning-free and LoFiT requires a training phase [1].

**Variant of**: the learning-free localized representation-intervention framework instantiated by ITI [2] and RepE [3], made two-step and learned [1].

**Data it needs**: task examples with a gold label per example - cross-entropy fine-tuning on gold responses for CLUTRR and MQuAKE, or preference pairs (gold-truthful vs. untruthful response) for TruthfulQA, trained with DPO [1][7]. No reward model and no scored-completion set is used; this is offline, small-sample supervised/preference fine-tuning, not on-policy sampling. Training scale in the paper: 500 or fewer examples per dataset (300-1000 for the data-efficiency study, 100-350 for the PEFT-benchmark comparison) [1].

**Extra models**: none beyond the base model being tuned. All parameters trained (the head-selection scaling vector $A$ and the bias vector $V$) are new small tensors attached to attention-head outputs; no value network, no separate reward model, and no frozen reference model, except that TruthfulQA's DPO objective itself needs an implicit reference (the pre-DPO policy) as part of DPO's own definition, not LoFiT's [1][7].

**Shipped by**: no PEFT or RL-training library implements LoFiT. Checked by reading the top-level export lists of Hugging Face `peft` (`src/peft/__init__.py`) and `trl` (`trl/__init__.py`) at the `main` branch commit each resolved to on 2026-08-09 and finding no LoFiT-named export in either file [8][9]. The only implementation is the authors' own research code at github.com/fc2869/lo-fit, which patches Llama-2 and Gemma modeling code directly (`models/modeling_llama.py`, `models/modeling_gemma.py`) and exposes training through per-dataset shell scripts rather than a reusable trainer class [6]. Building it inside an existing framework would mean adding two new per-head parameter tensors ($A^i_l$, $v^i_l$) into the attention forward pass and a two-phase training loop (train-then-freeze-then-train-again) - closer to a small architecture patch than a loss swapped onto an existing PPO/SFT trainer.

## How it works

Two sequential training phases on the same small dataset, both with all pre-trained weights frozen [1].

**Step 1 - head selection.** For every attention head $i$ at layer $l$ of an $L$-layer, $H$-head-per-layer decoder, LoFiT adds a learnable scaling vector $A^i_l \in \mathbb{R}^{d_{head}}$ that rescales that head's output activation during the forward pass:

$$ z_t^{(l,i)} \leftarrow (1 + A_l^i) \odot z_t^{(l,i)} $$

where $z_t^{(l,i)}$ is head $i$'s output at layer $l$, time step $t$ [1]. $A^i_l$ is initialized from $\mathcal{N}(0,\sigma_A)$ and trained end-to-end with the task's cross-entropy loss plus an L1 penalty (coefficient $\lambda$) to push most heads' scales toward zero [1]. After training, each head is scored by $S(i,l) = \lVert A^i_l\rVert$, and the top-$K$ heads by this score become the target set $T$; $K$ and $\sigma_A$ are hyperparameters chosen per model and dataset [1].

**Step 2 - bias tuning.** With $T$ fixed, LoFiT freezes $A$ and introduces one bias vector $v_l^i \in \mathbb{R}^{d_{head}}$ per selected head, added to that head's activation:

$$ z_t^{(l,i)} \leftarrow v_l^i \oplus z_t^{(l,i)}, \qquad (l,i) \in T $$

trained with the same task loss on the same data, initialized from $\mathcal{N}(0,\sigma_v)$ [1]. At inference, only these learned biases at the $|T|=K$ selected heads are added to every decoding step; the head-selection scalars $A$ are discarded after step 1 [1]. The paper reports $\sigma_A=\sigma_v=0.001$ uniformly, and states this initialization is robust to random seeds [1].

Worked example at the scale the paper reports for Llama-2-7B: at the 3%-head budget used for the truthfulness/reasoning results, $K=32$ heads [1]. Each selected head gets one $d_{head}$-sized bias vector $v_l^i$ trained in step 2, plus a same-sized scaling vector $A_l^i$ from step 1 that is discarded afterward, so total trainable size scales as $2 \times K \times d_{head}$ before step 1's parameters are dropped; this card did not verify Llama-2-7B's $d_{head}$ against the paper, since the paper does not state it. What the paper does report directly, in Table 3, is the final learned-parameter count after discarding $A$: at a 10%-head budget on Llama-2-7B, LoFiT trains 12K parameters (0.0002% of the model), versus 4.19M for LoRA and 2.10M for ReFT on the same model and task set [1].

Head-selection alternatives the paper itself tests and rejects in favor of the $A$-norm score: random sampling, per-layer probing (RepE-style), and scoring by the norm of a directly-trained bias vector instead of a scale vector (bias-based selection) [1]. Table 2 shows LoFiT's scale-norm selection wins on average across all three models over these three alternatives, and over ITI's own head-selection rule reused as a baseline [1].

## Cost

**Theory, from the method's own math**: both phases run a full forward and backward pass over the base model per training step, since the loss is the ordinary task cross-entropy (or DPO loss) with gradients flowing back through $A$ or $V$ only - no extra model forward pass is added beyond what plain fine-tuning would need. Training happens in two sequential phases on the same data, so total training compute is roughly two full fine-tuning runs' worth of steps, not one; the paper does not report the epoch count split between the two phases separately in the main text. Trainable-parameter memory is negligible ($K \times d_{head}$ floats per phase); the memory floor is dominated by holding the base model's weights, activations, and optimizer state for the frozen-but-backprop-through model, same as any other single-model fine-tune that does not touch pretrained weights.

**In practice, per framework**: the only implementation is the reference code, run by the authors on a single NVIDIA RTX A6000 (48GB) [1][6]; it uses full precision for Llama-2-7B and bfloat16 mixed precision for Llama-2-13B and Gemma-7B to fit that budget [1]. It uses the Hugging Face `transformers` implementation for cross-entropy fine-tuning, and TRL's DPO implementation for the TruthfulQA preference runs [1]. AdamW is used for optimization with $\epsilon=1\text{e-}8$ and weight decay 0.01 [1]. No throughput or step-time numbers are reported in the paper; this card did not find any.

## How to use it

- Data prep: format task examples as prompt/gold-response pairs for cross-entropy tasks (CLUTRR, MQuAKE), or prompt + (truthful, untruthful) response pairs for DPO-trained tasks (TruthfulQA) [1].
- Reward/label convention: no scalar reward function; supervision is either exact gold-text cross-entropy or, for TruthfulQA, DPO's implicit-reward objective with $\beta=0.5$ used uniformly across the TruthfulQA experiments [1].
- Two-phase run order matters: run head selection (phase 1) to convergence first, extract the top-$K$ heads by $\lVert A^i_l\rVert$, discard $A$, then run bias tuning (phase 2) from scratch on the fixed head set - the two phases are not run jointly [1].

| knob | paper's swept range / setting [1] |
| --- | --- |
| head budget K | 3% of heads (main truthfulness/reasoning results); 10% (PEFT comparison in Sec. 6) |
| L1 coefficient $\lambda$ (phase 1) | grid-searched per model/dataset, reported values 5e-4 to 5e-3 (Table 6) |
| phase-1 learning rate | grid-searched, 5e-4 to 5e-3 across settings (Table 6) |
| phase-2 (bias tuning) learning rate | grid-searched, 8e-3 to 2e-2 at 3% heads, 5e-3 to 1e-2 at 10% heads (Table 6) |
| $\sigma_A$, $\sigma_v$ (init std) | 0.001, fixed across all experiments |
| epochs | 5 for TruthfulQA/MQuAKE/CLUTRR/SVAMP; 3 for SIQA/ARC-c/BoolQ |
| batch size | 8 (BoolQ only: 4 for Llama-2-7B, 2 for Llama-2-13B, to fit long passages on one GPU) |
| optimizer | AdamW, $\epsilon=1\text{e-}8$, weight decay 0.01 |
| DPO $\beta$ (TruthfulQA only) | 0.5 |

No framework ships a default configuration for LoFiT, so there is no framework-default column to contrast against the paper's values; every number above is the paper's own grid-search result, not a fixed default. The paper notes that a smaller head budget needs a larger phase-2 learning rate to stabilize training [1].

Trade-off a run designer faces: raising $K$ (head budget) adds more trainable parameters and, per Table 3, closes the gap to LoRA and RED on knowledge-heavy tasks (SIQA, ARC-c, BoolQ, SVAMP), at the cost of the parameter-efficiency and (per the paper's OOD study) some of the base-capability preservation that the 3% setting shows in Table 5 [1].

## While it runs

- Signals: this card found no logging-library integration for LoFiT (no framework instruments it), so there is no framework-standard metric name to point at; the paper's own diagnostic is the head-selection norm score $S(i,l)=\lVert A_l^i\rVert$, used offline after phase 1 to pick $T$, not as a live training-loop signal [1].
- Published reference runs: Table 1 of the paper is the reference curve-equivalent - final test accuracy for LoFiT vs. 0-shot, ITI, and RepE on TruthfulQA/MQuAKE/CLUTRR for Gemma-7B, Llama-2-7B, and Llama-2-13B, all at a 3%-head budget [1]. Figure 4 gives a second reference: LoFiT vs. LoRA vs. RED accuracy as a function of training-set size $n$ on CLUTRR and MQuAKE with Llama-2-7B, showing LoFiT ahead at $n\le100$ and comparable at $n\ge300$ [1].
- Degeneracies and defaults: the paper reports that using fewer heads requires a larger phase-2 learning rate to avoid instability - an under-scaled learning rate at a small $K$ is a documented failure mode, not just a slow-convergence issue [1]. No config default exists to diverge from, since no framework ships one.
- Named successors: a forward-citation query against the Semantic Scholar API for the LoFiT arXiv ID returned 69 citing papers; one, Joint Localization and Activation Editing (JoLA), proposes learning which heads to edit, which intervention type (additive, multiplicative, or both) to apply, and the intervention parameters all together in a single training step, positioned against prior activation-editing methods whose performance depends on separately identifying the right modules to edit [10] - a direct fix to LoFiT's own two-phase locate-then-edit split.
- Known failure modes: the paper's own limitations section states results are only shown on English-language, short-context truthfulness/reasoning tasks, and that different behavior might appear for long-context or long-form generation, and at model scales above the 13B tested here [1]. The GitHub repository's README documents one concrete environment failure: a `RuntimeError: CUDA error: device kernel image is invalid` at inference time, attributed to a PyTorch/CUDA driver version mismatch, with the fix being to match the tested PyTorch 2.2.2+cu121 configuration [6].
- What the gain is - and is not: the paper's own out-of-domain study (Table 5) fine-tunes Llama-2-7B-Chat on TruthfulQA with LoFiT and then evaluates it zero-shot on TriviaQA, MMLU, and Natural Questions; accuracy on the first two benchmarks holds at or above the non-fine-tuned base model's level and rises on the third, whereas ITI, LoRA, and RED baselines each lose accuracy on at least one of these three OOD benchmarks in the same setup [1] - i.e., the paper's evidence is about preserving existing capability under a small, localized update, not about adding new capability the base model lacks.

## Sources

[1] Yin, Ye, and Durrett, "LoFiT: Localized Fine-tuning on LLM Representations", NeurIPS 2024. https://arxiv.org/abs/2406.01563 - defines LoFiT: two-step method, objective, hyperparameters (Table 6), main results (Table 1), head-selection ablation (Table 2), PEFT comparison (Table 3), data-efficiency (Figure 4), OOD generalization (Table 5), limitations. Fetched 2026-08-09 (PDF converted to text via pdftotext).

[2] Li, Patel, Viégas, Pfister, and Wattenberg, "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model", NeurIPS 2023. https://arxiv.org/abs/2306.03341 - ITI, the learning-free intervention baseline LoFiT compares against and reuses for head-selection ablation. Fetched 2026-08-09 (abs page).

[3] Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency", 2023. https://arxiv.org/abs/2310.01405 - RepE, the other learning-free intervention baseline. Fetched 2026-08-09 (abs page).

[4] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022. https://arxiv.org/abs/2106.09685 - LoRA, the PEFT baseline LoFiT is compared against in Table 3 and Figure 4. Fetched 2026-08-09 (abs page).

[5] Wu et al., "ReFT: Representation Finetuning for Language Models", 2024. https://arxiv.org/abs/2404.03592 - ReFT/LoReFT, the concurrent representation-editing PEFT method without a localization step. Fetched 2026-08-09 (abs page).

[6] fc2869/lo-fit GitHub repository (official LoFiT code). https://github.com/fc2869/lo-fit - installation, model support, training scripts, released checkpoints, CUDA troubleshooting note. README fetched 2026-08-09 from raw.githubusercontent.com/fc2869/lo-fit/main/README.md, pinned to commit `ebd344a273d3dab0b63d8bbfd7ecdb5185ae5b7b` (the `main` branch HEAD resolved via the GitHub commits API at fetch time, committer date 2025-01-15); repository metadata (stars, license, archived status) fetched via GitHub API the same day.

[7] Rafailov, Sharma, Mitchell, Manning, Ermon, and Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023. https://arxiv.org/abs/2305.18290 - DPO, used by the paper to fine-tune the TruthfulQA preference-pair experiments. Fetched 2026-08-09 (abs page).

[8] huggingface/peft GitHub repository, `src/peft/__init__.py`. https://github.com/huggingface/peft - top-level package export list. Fetched 2026-08-09 from raw.githubusercontent.com/huggingface/peft/main/src/peft/__init__.py; `main` branch HEAD resolved via the GitHub commits API to commit `5f55a6331b6a1620d8200ddb7c7c517dec722908` (committer date 2026-08-06); grepped for "lofit" (case-insensitive), zero matches.

[9] huggingface/trl GitHub repository, `trl/__init__.py`. https://github.com/huggingface/trl - top-level package export list. Fetched 2026-08-09 from raw.githubusercontent.com/huggingface/trl/main/trl/__init__.py; `main` branch HEAD resolved via the GitHub commits API to commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0` (committer date 2026-08-07); grepped for "lofit" (case-insensitive), zero matches.

[10] Lai et al., "Joint Localization and Activation Editing for Low-Resource Fine-Tuning", 2025. https://arxiv.org/abs/2502.01179 - JoLA, a named successor found via a forward-citation query. Fetched 2026-08-09: citation list via api.semanticscholar.org/graph/v1/paper/arXiv:2406.01563/citations (69 citing papers, live/unpinned endpoint, no revision parameter), and JoLA's own abstract via the same API call plus the arXiv abs page title.
