# DyLoRA

Paper: https://arxiv.org/abs/2210.07558

LoRA where the up/down-projection matrices are trained for a whole range of ranks at once, so one training run yields an adapter that works at any rank in that range without a rank-search or a retrain.

**DyLoRA** (Dynamic Low-Rank Adaptation) is a parameter-efficient fine-tuning method for pretrained language models, introduced to fix two problems the paper identifies in LoRA: rank is fixed after training, and finding a good rank requires exhaustive search [1]. Its parent is LoRA, which freezes the pretrained weights and injects trainable rank-decomposition matrices into each layer, reducing the number of trainable parameters for downstream tasks [2]. DyLoRA keeps LoRA's up-projection/down-projection structure but, at each training step, samples a rank $b$ from a range $[r_{min}, r_{max}]$, truncates both matrices to that rank, and trains on the truncated forward pass, borrowing the idea of ordering information across truncation levels from nested dropout [1]. The paper gives two reasons: this removes the need to retrain from scratch whenever a different rank is wanted, and it removes the exhaustive per-rank search needed to find a good LoRA rank [1].

No landmark production system's adoption of DyLoRA itself was found in this search: a Semantic Scholar citation-graph query returned the 100 most recent papers citing arXiv:2210.07558 (API default order, newest first); manually sorting that batch by citation count found no citation from a widely deployed system, only follow-on parameter-efficient-tuning research such as adaptive-rank and nested-subspace LoRA variants [3]. In the paper's own GLUE (RoBERTa-base) results at the hardest setting, rank 1, DyLoRA averages 84.22 across eight tasks against a plain rank-1 LoRA baseline's 54.90 - a 29.3-point gap - while at rank 8 the two are close, 85.44 for DyLoRA versus 85.27 for LoRA (Table 2) [1]. The paper also reports DyLoRA trains 4 to 7 times faster than sweeping LoRA over ranks, depending on the task [1]; its own time-complexity appendix gives a concrete case, DyLoRA training across all ranks on MRPC in 408.39s versus 399.95s x 8 = 3199.6s to train LoRA separately at ranks 1-8 [1]. Lineage in one line: LoRA (2021 [2]) -> DyLoRA (2022 [1]) -> AdaLoRA, an independent adaptive-rank alternative (2023 [4]) and later nested/rank-flexible LoRA variants surfaced by the citation search (2025-2026 [3]).

**When to pick it**: pick DyLoRA when you want one LoRA-style adapter that can be deployed at several ranks (e.g. to trade accuracy for inference cost per-deployment) without retraining or without running a separate rank sweep; the paper's own evidence for this is the wide-range robustness in Table 2, especially the low-rank regime [1]. Prefer plain LoRA [2] when a single, already-known rank is enough - it is simpler and, per Table 2, matches or slightly beats DyLoRA at the highest ranks tested [1]. Prefer AdaLoRA [4] when the goal is spending a fixed parameter budget efficiently across layers by importance rather than training one adapter to be rank-flexible; AdaLoRA parameterizes updates via SVD and prunes singular values by an importance score, which is a different mechanism from DyLoRA's per-step rank truncation [4]. The alternative DyLoRA replaces outright is a grid search: training separate LoRA adapters at every candidate rank and picking the best, which is what the 3199.6s-vs-408.39s comparison above is measured against [1].

**Variant of**: LoRA [2].

**Data it needs**: whatever the base task needs - the paper's NLU experiments use the GLUE benchmark (labeled single-sentence and sentence-pair classification/regression data) and its NLG experiments use E2E, DART and WebNLG (structured input, target text pairs) [1]; DyLoRA adds no new data requirement over LoRA itself, only a choice of $r_{min}$/$r_{max}$ and a rank-sampling distribution $p_B$ [1]. Training is standard supervised fine-tuning on a fixed labeled dataset, not on-policy rollouts against a reward signal.

**Extra models**: none. Only the frozen pretrained backbone plus the trainable up/down-projection matrices already required by LoRA [1][2]; no value network, reference model, reward model, or judge is part of the method's own definition.

**Shipped by**: no maintained library implements DyLoRA under a top-level export - Hugging Face PEFT's public API lists many LoRA variants (`LORA`, `ADALORA`, `XLORA`, `RANDLORA`, `DELORA`, `GRALORA`, `GLORA`, `VBLORA`, `UNILORA`, `TINYLORA`) but no DyLoRA entry, checked in `peft_types.py` and `__init__.py` at `main` on 2026-08-09 [5]. The only implementation is the authors' own research repo, a fork of Microsoft's LoRA codebase (`loralib`) integrated with a patched `transformers`, at `huawei-noah/Efficient-NLP` under the `DyLoRA/` directory (formerly `huawei-noah/KD-NLP`, which now redirects) [6]; the repository was last pushed 2024-06-04 and is not archived (99 stars) [7], but it ships example scripts for RoBERTa/GLUE, not a general-purpose trainer class. Building DyLoRA into an existing LoRA implementation means adding a per-step rank sample and matrix truncation to the LoRA forward/backward pass (Algorithm 1) [1] - a modification to an existing adapter layer, not a new loss or a new sampling loop.

## How it works

Each training step: sample a rank $b$ from a fixed range, truncate the LoRA up/down matrices to that rank, run the forward pass with the truncated matrices, and update parameters (optionally only the newly-touched row/column) [1].

**LoRA's forward pass**, which DyLoRA truncates, for a pretrained weight $W_0 \in \mathbb{R}^{m \times d}$ with adapter $\Delta W = W_{up}W_{dw}$, $W_{up} \in \mathbb{R}^{m \times r}$, $W_{dw} \in \mathbb{R}^{r \times d}$ [1] (Eq. 4):

$$ h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} W_{up} W_{dw} x $$

$\alpha$ is a constant scale hyperparameter; $W_{up}$ is initialized to zero and $W_{dw}$ to a zero-mean Gaussian [1].

**DyLoRA's truncation.** At each step, sample $b \sim p_B(\cdot)$, $b \in \{r_{min}, \dots, r_{max}\}$, and truncate (Eq. 5) [1]:

$$ W_{dw\downarrow b} = W_{dw}[1{:}b,\,:], \qquad W_{up\downarrow b} = W_{up}[:,\,1{:}b] $$

giving the truncated forward pass (Eq. 7) [1]:

$$ h = W_0 x + \frac{\alpha}{b}\, W_{up\downarrow b}\, W_{dw\downarrow b}\, x $$

**The dynamic loss** replaces the nested-dropout expectation over all ranks with a single sampled rank per step (Eq. 9), which the paper states is what makes the scheme computationally cheap [1]:

$$ \mathcal{L}^{DY}_{\downarrow b} = \sum_{i=1}^{N} l\bigl(f(x_i; W_{dw\downarrow b}, W_{up\downarrow b}),\, y_i\bigr) $$

**Update rule - two modes.** The paper defines a `FROZEN` mode: update only the single row/column newly exposed at rank $b$, denoted $W_{dw}^{\,b}$ (row $b$) and $W_{up}^{\,b}$ (column $b$), leaving rows/columns $1..b{-}1$ untouched (Eq. 10) [1]:

$$ W_{dw}^{\,b} \leftarrow W_{dw}^{\,b} - \eta \nabla_{W_{dw}^{\,b}} \mathcal{L}^{DY}_{\downarrow b}, \qquad W_{up}^{\,b} \leftarrow W_{up}^{\,b} - \eta \nabla_{W_{up}^{\,b}} \mathcal{L}^{DY}_{\downarrow b} $$

The non-frozen mode instead updates all of $W_{dw\downarrow b}$ and $W_{up\downarrow b}$ (rows/columns $1..b$) at every step [1]. The paper's own ablation (Table 4, Uniform-distribution rows) shows the non-frozen full-truncation update scoring higher on average than the frozen single-row/column update at both rank 1 (83.30 vs 81.61) and rank 8 (84.47 vs 83.31) [1].

**Rank-sampling distribution $p_B$.** The paper tests a discrete Uniform distribution over $[r_{min}, r_{max}]$ and a Geometric distribution with $p=0.15$; in Table 4, Uniform sampling with the full-truncation update scores highest at rank 8 (84.47 average), while at rank 1 the full-truncation update scores marginally higher under Geometric sampling (83.42) than under Uniform (83.30) [1].

Worked example: with $r_{min}=1$, $r_{max}=8$ and $\alpha=16$ (the RoBERTa setting in the paper's Table 8 [1]), a step that samples $b=4$ truncates $W_{up}$ to its first 4 columns and $W_{dw}$ to its first 4 rows, computes $h = W_0x + \frac{16}{4}W_{up\downarrow4}W_{dw\downarrow4}x = W_0x + 4\,W_{up\downarrow4}W_{dw\downarrow4}x$, and (non-frozen mode) backpropagates into all 4 rows/columns; the next step may sample a different $b$ and touch a different-sized slice of the same matrices.

## Cost

**Theory, from the method's own math:** DyLoRA adds no extra forward pass and no extra model versus LoRA - the truncation only changes which slice of the already-existing $W_{up}$/$W_{dw}$ is multiplied and updated at a given step [1]. The paper states the training time for DyLoRA is comparable to LoRA trained once at a single rank [1] (Appendix A); the savings claimed are relative to the alternative of training LoRA separately at every candidate rank, not relative to a single LoRA run. Memory footprint per step is bounded by the full-size $W_{up}$/$W_{dw}$ matrices (they must be allocated even when a low $b$ is sampled), so DyLoRA does not save memory versus a LoRA run already sized at $r_{max}$.

**In practice, per framework:** the only released implementation is the authors' own repo, a source-level fork of `loralib` plus a patched `transformers` for RoBERTa [6]; the paper's own comparable-training-time claim (408.39s for DyLoRA across all ranks vs. 399.95s x 8 = 3199.6s to sweep LoRA over 8 ranks on MRPC) is a measurement from this authors' implementation, on a single Tesla V100-PCIE-32GB GPU (Table 8 hardware) [1], not from a maintained framework's benchmark suite. No PEFT-library cost figures exist because PEFT does not implement the method [5].

## How to use it

- Data: same as the base fine-tuning task - GLUE-style labeled classification/regression examples for NLU, or (structured input, reference text) pairs for NLG (E2E, DART, WebNLG in the paper) [1]. No preference pairs, rewards, or judge scores are needed.
- Reward/label convention: none - this is supervised fine-tuning with the task's own loss (e.g. cross-entropy for GLUE classification), truncated per Eq. 9 [1].
- Knobs, with the paper's own values as the only published anchor (no shipping framework exposes these as configurable defaults, since none implements the method):

| knob | paper (RoBERTa-base, GLUE) [1] | paper (GPT-2 Medium, NLG) [1] |
| --- | --- | --- |
| rank range | $r_{max}=8$ (varies 8/32/64 by experiment, Appendix) | $r_q=r_v=4$ |
| LoRA $\alpha$ | 16 | 32 |
| optimizer | AdamW | AdamW |
| learning rate | 4e-4 | 2e-4 |
| batch size | 32 | 8 |
| epochs | 30 | 5 |
| rank-sampling distribution $p_B$ | Uniform or Geometric($p{=}0.15$), tested both | not stated |
| update mode | frozen vs. non-frozen, tested both | not stated |

- Trade-off: a wider $[r_{min}, r_{max}]$ range gives one adapter usable at more deployment ranks but does not change per-step compute (each step still trains one sampled rank); the paper reports testing $r_{max}$ up to 64 in some GLUE ablations [1]. Choosing the frozen update mode trades a small accuracy cost for cheaper per-step updates, since only one row/column is touched instead of up to $b$ of them: across Table 4's four rank x distribution combinations, frozen scores range 81.61-83.31 and full-truncation scores range 83.30-84.47 [1].

## While it runs

- Signals and their healthy shapes: the paper does not describe any live-monitoring metric specific to DyLoRA (e.g. a rank-collapse or degeneracy signal to watch during training); its evaluation is standard task metrics (accuracy, Matthews correlation, F1, Pearson, BLEU, TER) measured at the end of training per rank, not a live-run diagnostic [1].
- Published reference runs: the paper's own Table 2 (GLUE, RoBERTa-base, ranks 1-8, best-rank and full-rank fine-tuning rows) and Table 9 (DART/WebNLG, GPT-2 Medium, ranks 1-4) are the only published reference numbers found for this method [1].
- Degeneracies and defaults: the paper's own ablation (Table 4) shows results are sensitive to two choices with no framework default to fall back on, since no framework ships this method - the sampling distribution ($p_B$: Uniform vs. Geometric) and the update mode (frozen vs. non-frozen) - both documented above under How to use it [1].
- Named successors: AdaLoRA [4] addresses the same "how much rank budget per layer" question with an orthogonal, importance-score-driven SVD-pruning mechanism rather than DyLoRA's per-step rank sampling; a Semantic Scholar citation-graph query (100 citing records) surfaced further rank-flexible LoRA variants describing themselves as successors, including nested-subspace and dynamic-rank-assignment methods, but this card does not verify their claims against DyLoRA's own text since they were not fetched [3].
- Known failure modes: the paper's own Limitations section names three open questions rather than confirmed failure modes - it does not state a value for LoRA's scale hyperparameter $\alpha$ that is known to be best and says further investigation is needed to find one; it shows uniform sampling can match a specific geometric distribution but says further investigation is needed on how the sampling distribution affects other downstream tasks; and it says further research is needed on the impact of the chosen rank range $[r_{min}, r_{max}]$ itself [1]. No GitHub issue search was performed on `huawei-noah/Efficient-NLP` for this card, so no maintainer-reported failure mode beyond the paper's own Limitations text is included here.
- What the gain is - and is not: the paper's own results show DyLoRA's main gain is stability and accuracy across a wide range of ranks without retraining, most visible at low rank (e.g. the 29.3-point rank-1 GLUE-average gap over plain LoRA cited above) [1]; at the ranks where plain LoRA already performs well (rank 8 and above in Table 2), DyLoRA and LoRA are close, so the method's benefit is concentrated in the low-rank / rank-flexibility regime rather than a uniform improvement at every rank [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. A source that could not be opened is cited through the source that quotes it, and says so.

[1] Valipour, Rezagholizadeh, Kobyzev, and Ghodsi, "DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation", EACL 2023 (arXiv 2210.07558, v2). https://arxiv.org/abs/2210.07558 - defines the method: objective, truncation scheme, update modes, all cited tables and appendices. Fetched 2026-08-09 (PDF full text via arxiv.org/pdf/2210.07558, converted with pdftotext).

[2] Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, and Chen, "LoRA: Low-Rank Adaptation of Large Language Models", 2021. https://arxiv.org/abs/2106.09685 - the parent method. Fetched 2026-08-09 (abstract page).

[3] Semantic Scholar Graph API, citations of arXiv:2210.07558, fields title/citationCount/year, limit 100 (API's default order, newest publication year first; this card's own analysis then sorted that batch by citation count for review). https://api.semanticscholar.org/graph/v1/paper/arXiv:2210.07558/citations - used to check for landmark-system adoption and named successors; no citing record in the 100 returned belongs to a large deployed system. Fetched 2026-08-09.

[4] Zhang, Chen, Bukharin, He, Cheng, Chen, and Zhao, "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning", 2023. https://arxiv.org/abs/2303.10512 - the nearest online-neighbor alternative cited for contrast. Fetched 2026-08-09 (abstract page).

[5] Hugging Face PEFT source, `src/peft/utils/peft_types.py` and `src/peft/__init__.py`, `main` branch. https://github.com/huggingface/peft - checked for a DyLoRA entry; none found among the listed LoRA-family methods or the package's exports. Read at `main` on 2026-08-09; unpinned, mutable branch.

[6] huawei-noah/Efficient-NLP repository, `DyLoRA/` directory and its README (formerly `huawei-noah/KD-NLP`, now redirects). https://github.com/huawei-noah/Efficient-NLP/tree/main/DyLoRA - the paper's own footnote names this repo as the official code; README confirms it is a fork of Microsoft's LoRA (`loralib`) with DyLoRA integrated. Fetched 2026-08-09 (raw README) and via GitHub Contents API (directory listing).

[7] GitHub REST API, repository metadata for `huawei-noah/Efficient-NLP`. https://api.github.com/repos/huawei-noah/Efficient-NLP - `pushed_at`, `archived`, and `stargazers_count` fields used for maintenance status. Fetched 2026-08-09.
