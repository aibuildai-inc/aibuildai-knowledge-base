# SEAL

Before fine-tuning on a mixed-quality dataset, train a per-sample weight over it by bilevel optimization against a separate safe dataset, then fine-tune only on the top-ranked, safety-preserving subset.

**SEAL** (Safety-Enhanced Aligned LLM Fine-tuning, full title "SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection") is a data-selection framework for supervised fine-tuning, introduced by Shen, Chen, Das and Chen so that a bilevel optimization stage learns, before fine-tuning, which training samples preserve safety and quality and which do not, so the safe/high-quality ones can be kept and the unsafe/low-quality ones dropped [1]. The paper is at https://arxiv.org/abs/2410.07471 [1]. SEAL's algorithm is explicitly built on the penalty-based reformulation of bilevel optimization, citing it as the inspiration for reformulating the nested problem into a single-level, first-order-only objective [1]; that reformulation is due to Shen and Chen's penalty-based bilevel gradient descent method, its nearest methodological parent [2]. Mechanically, SEAL solves a bilevel problem where the lower level fits model parameters θ to the fine-tuning dataset D reweighted by a learned softmax selector σ(ω), and the upper level adjusts ω so that the resulting θ*(ω) fits a separate safe/alignment dataset D_safe well; after training, the top p% ranked samples of D are kept and the LLM is fine-tuned normally on that subset [1]. The paper gives three reasons for this design: fine-tuning on adversarial or even benign data has been shown in prior work to compromise a model's pre-equipped alignment and safety [1]; large fine-tuning datasets are too big for manual safety curation, so the selection must be automatic and data-driven [1]; and prior bilevel-optimization approaches relied on implicit-gradient or iterative-differentiation methods requiring second-order derivatives, which are memory-inefficient at LLM parameter scale, whereas SEAL's penalty-based algorithm needs only first-order gradients [1].

A Semantic Scholar citation search on the paper's arXiv ID (2410.07471), fetched 2026-08-09, returned 73 citing papers, confirming this is the intended paper via an exact-title match against a single returned candidate (no other paper titled "SEAL" in this space was found by the sourcing pipeline) [1]. Most citing work applies to the same harmful-fine-tuning-defense problem (e.g. Booster, Vaccine, Antidote) rather than adopting SEAL itself, and this card's search of that citation list, of the authors' own repository, and of its GitHub issue tracker found no named production or landmark post-training system that has adopted SEAL; the only implementation located is the authors' own reference repository [3]. In the paper's own experiments, SEAL's average win-rate gain over random data selection is around 8.5% on Llama-3-8B-Instruct (Table 1) and around 9.8% on Merlinite-7B (Table 2) [1]. Lineage in one line: penalty-based bilevel gradient descent (Shen & Chen, 2023 [2]) -> SEAL (2024 [1]) -> no confirmed named successor found in the citation search performed for this card [1].

**When to pick it**: pick SEAL when you must fine-tune on a dataset of unknown or mixed safety quality, you have access to a separate, trusted safe/alignment dataset, and you can afford an extra bilevel training pass before the fine-tuning run itself [1]. It is not a variant of an RL policy-optimization method; the nearest offline alternatives are static, single-pass data-selection heuristics that need no iterative optimization loop: DSIR estimates bag-of-n-gram importance-ratio scores between a target and a raw dataset in one pass [4], and LESS estimates per-sample influence via a one-shot gradient-similarity computation against a target task [5]. SEAL's own paper frames a concurrent bilevel-optimization neighbor, a method that learns weights for whole data sources rather than individual data points, as the closest work in its own class, though this card did not independently fetch that paper [1].

**Variant of**: not a variant of PPO/DPO-style post-training methods; its algorithm is a direct application of the penalty-based bilevel gradient descent method to LLM data selection [1][2].

**Data it needs**: two fixed, pre-collected datasets — a fine-tuning dataset D of N prompt-response pairs of mixed safety/quality, and a safe/alignment dataset D_safe of M prompt-response pairs with known-safe responses [1]. Both are static; SEAL is offline in the sense that no data is freshly sampled from an evolving policy during training, unlike an online RL loop. In the paper's main experiments (Llama-3-8B-Instruct, Merlinite-7B, Pythia-2.8B), D was the RedOrca dataset (about 90k SlimOrca instructions plus 22k potentially unsafe instructions drawn from the Anthropic red-teaming dataset) and D_safe was a withheld 112k-sample subset of SlimOrca, also reused as DSIR's target dataset in the same experiments [1]. In the separate benign-dataset experiment (Section 4.5), D was a 49.9k-sample subset of OpenOrca and D_safe was a 49.9k-sample subset of Alpaca-Cleaned [1]. Anthropic HH appears in the paper only as one of the win-rate evaluation sets (Tables 1, 2, 4, 5) and, for Pythia-2.8B only, as the data for a separate initial DPO safety-alignment step before SEAL is applied — never as SEAL's D_safe [1].

**Extra models**: none of value network, reference model, or reward model by SEAL's own definition; the training signal is each sample's own length-normalized negative log-likelihood loss [1]. The full data-selector algorithm (Algorithm 1) keeps one extra trained LLM copy θ̂ alongside θ during selector training; the paper's own memory-efficient variant (Algorithm 2) drops θ̂ under an approximation, and the paper states this is the variant actually used in its main experiments [1]. Details in Cost.

**Shipped by**: no general post-training library (trl, verl, OpenRLHF core) implements SEAL. The only implementation found is the authors' own reference repository, hanshen95/SEAL, built on top of OpenRLHF, DeepSpeed and Transformers, exposing `examples/train_sft_selector.py` for data-selector training (Algorithms 1/2) and `examples/train_sft.py` for the fine-tuning stage, invoked via `deepspeed ../train_sft_selector.py` from within `examples/scripts/` (which itself holds only shell wrapper scripts, e.g. `train_selector_llama3.sh`); it is installed from source with `pip install -e .`, not published as a PyPI package, and had 24 stars and a last push of 2025-02-20 as of this check [3][6]. Building it into a library that lacks it means adding a second, alternating optimization loop over two datasets on top of an existing SFT trainer, not a new sampling loop — no rollout/generation step is involved.

## How it works

Each step of the data-selector stage samples one safe example and one fine-tuning example, updates the model parameter(s) toward both the safe loss and the weighted fine-tuning loss, and updates the selector weights by the sign and size of a loss gap; after training, the selector ranks D once and the top p% is kept for a normal fine-tuning run [1].

**The bilevel objective** [1, Eq. 1]:

$$ \min_{\omega}\ \frac{1}{M}\sum_{i=1}^{M} \ell(\theta^*(\omega); z_{safe}^i), \qquad \text{s.t. } \theta^*(\omega) = \arg\min_{\theta} \frac{1}{N}\sum_{i=1}^{N} \sigma_i(\omega)\,\ell(\theta; z^i) $$

where a sample is $z=(x,y)$ with target response $y=(y_1,\dots,y_{d_y})$, and

$$ \ell(\theta; z) := -\frac{1}{d_y}\sum_{j=1}^{d_y} \log P_\theta(y_j \mid x, y_{<j}) $$

is the length-normalized negative log-likelihood, and $\sigma(\omega) = (\sigma_1(\omega),\dots,\sigma_N(\omega))$ is the data selector, instantiated as a softmax $\sigma_i(\omega) = \exp(\omega_i) / \sum_{i=1}^N \exp(\omega_i)$ over one scalar $\omega_i$ per fine-tuning sample [1]. The upper level minimizes the safety loss of the model that results from lower-level training on the reweighted fine-tuning set; the lower level fits $\theta$ to $D$ weighted by $\sigma(\omega)$ [1].

**Penalty reformulation.** Because the nested problem is hard to differentiate through directly, the paper defines a penalty term for lower-level sub-optimality [1, Eq. 2]:

$$ p(\omega,\theta) := \frac{1}{N}\sum_{i=1}^{N} \sigma_i(\omega)\,\ell(\theta; z^i) - \min_{\theta'} \frac{1}{N}\sum_{i=1}^{N} \sigma_i(\omega)\,\ell(\theta'; z^i) $$

and folds it into the upper-level loss with penalty strength $\gamma \in (0,1)$ to get a single-level penalized problem [1, Eq. 3]:

$$ \min_{\omega,\theta}\ (1-\gamma)\frac{1}{M}\sum_{i=1}^{M} \ell(\theta; z_{safe}^i) + \gamma\left[\frac{1}{N}\sum_{i=1}^{N} \sigma_i(\omega)\,\ell(\theta; z^i) - \min_{\theta'} \frac{1}{N}\sum_{i=1}^{N} \sigma_i(\omega)\,\ell(\theta'; z^i)\right] $$

Increasing $\gamma$ increases the accuracy of solving for $\theta^*(\omega)$; the paper schedules $\gamma$ to grow from 0 across epochs so early steps warm-start $\theta$ on the safe loss alone [1].

**The updates** (Algorithm 1, the full variant) [1]. Model parameter update, sampling one safe example $z_{safe}^i$ and one fine-tuning example $z^j$ per step:

$$ \theta_{k+1} = \theta_k - \beta_k\left[(1-\gamma_k)\nabla\ell(\theta_k; z_{safe}^i) + \gamma_k\,\sigma_j(\omega_k)\nabla\ell(\theta_k; z^j)\right] $$

An auxiliary model $\hat\theta$ is trained on the weighted fine-tuning loss only (no safe-loss term), approximating $\theta^*(\omega_k)$:

$$ \hat\theta_{k+1} = \hat\theta_k - \beta_k\,\sigma_i(\omega_k)\nabla\ell(\hat\theta_k; z^j) $$

The selector is then updated using the gap between how well $z^j$ is fit under $\theta_k$ (which also serves the safe data) versus under $\hat\theta_k$ (which serves only the weighted fine-tuning data):

$$ \omega_{k+1} = \omega_k - \alpha_k\left[\ell(\theta_k; z^j) - \ell(\hat\theta_k; z^j)\right]\nabla\sigma_j(\omega_k) $$

A large positive gap means $z^j$ fits worse under the safety-aware $\theta_k$ than under the safety-blind $\hat\theta_k$, i.e. it is likely misaligned with $D_{safe}$, so its rank is decreased; a negative gap raises its rank [1]. Concrete illustration of the direction only (numbers are illustrative, not from the paper): if $\ell(\theta_k;z^j)=2.5$ and $\ell(\hat\theta_k;z^j)=1.0$, the gap is $+1.5$ and $\sigma_j$'s weight is pushed down; if $\ell(\theta_k;z^j)=0.8$ and $\ell(\hat\theta_k;z^j)=1.2$, the gap is $-0.4$ and the weight is pushed up.

**Memory-efficient variant (Algorithm 2), used in the paper's main experiments** [1]. Assuming the model has far more parameters than there are fine-tuning samples ($d_\theta \gg N$), the paper argues the lower-level minimum is approximately 0 for any $\omega$, which drops the $\hat\theta$ update entirely and simplifies the selector update to:

$$ \omega_{k+1} = \omega_k - \alpha_k\,\ell(\theta_k; z^j)\,\nabla\sigma_j(\omega_k) $$

**Process-level detail not carried by the light-weight variant**: because $\hat\theta$ is dropped, Algorithm 2's selector update no longer contrasts $\theta_k$ against a safety-blind counterpart — it uses $z^j$'s raw fitted loss under $\theta_k$ directly as the down-rank signal [1].

The four-step framework: (S1) start from a safety-aligned LLM; (S2) train the selector with Algorithm 1 or 2; (S3) rank $D$ by $\sigma(\omega_K)$ and keep the top $p\%$ as $D_{top}$; (S4) fine-tune the LLM on $D_{top}$ with standard SFT [1]. The paper notes S1–S3 can be a one-time cost: once trained on a given fine-tuning dataset, the selector can be reused for all subsequent fine-tuning runs on that dataset [1], and that a selector trained with a smaller model transfers to selecting data for a larger fine-tuning model [1].

## Cost

**Theory, from the method's own math:**

- Time: the selector-training stage (S2) touches both $D_{safe}$ and $D$ at every step, versus a single dataset for plain SFT, and Algorithm 1 additionally runs a third forward/backward pass to update $\hat\theta$ at every step; Algorithm 2 removes that third pass [1]. The fine-tuning stage (S4) is then a normal SFT run over $D_{top}$, so the naive time cost of the full pipeline is (selector training over $D \cup D_{safe}$) + (SFT over $p\%$ of $D$), strictly more than plain SFT over $D$ alone.
- Memory: the selector parameter $\omega \in \mathbb{R}^N$ (one scalar per fine-tuning sample) is negligible next to LLM weights [1]. Algorithm 1 holds two trained LLM copies, $\theta$ and $\hat\theta$, each with its own optimizer state, during selector training; Algorithm 2 holds one [1]. Neither algorithm requires a value network, reward model, or KL-penalized reference model.

**In practice, per framework:**

- Authors' reference implementation (hanshen95/SEAL, built on OpenRLHF/DeepSpeed) [1][3]: Table 3 reports wall-clock and GPU memory on one NVIDIA A6000 in a group of four, for a Merlinite-7B-scale run. Training the data selector with a Pythia-2.8B stand-in model takes 14 hours at 28.3 GB; with Phi-3-mini it takes 20 hours at 34.1 GB. The subsequent fine-tuning stage takes 3 hours at 27.8–27.9 GB when 20% of the data is selected, or 13 hours at the same memory when 80% is selected. A plain no-selection SFT baseline takes 16 hours at 27.9 GB [1]. So on this hardware the full SEAL pipeline (selector training + fine-tuning) costs roughly 1.1x–2x the wall-clock time of plain SFT, dominated by the selector-training stage, and that stage is reusable across future fine-tuning runs on the same dataset [1]. In a GitHub issue, the maintainer states the 8B-model experiments used one A100 GPU, or 4 A6000 GPUs [3].

## How to use it

- Data prep: assemble a fine-tuning dataset $D$ that may contain unsafe or low-quality samples, and a separate, trusted safe/alignment dataset $D_{safe}$ with safe target responses (the paper's main experiments used a withheld 112k-sample subset of SlimOrca as $D_{safe}$ against a RedOrca fine-tuning set; its benign-dataset experiment used a 49.9k-sample subset of Alpaca-Cleaned as $D_{safe}$ against an OpenOrca fine-tuning set — Anthropic HH is used only as an evaluation set, not as $D_{safe}$, in these experiments) [1].
- Loss/label convention: no reward model and no preference labels; both datasets are supervised prompt-response pairs scored by the model's own length-normalized negative log-likelihood [1]. The selector is a lookup table with one scalar weight per row of $D$, not a neural network conditioned on the input — a GitHub issue confirms this design choice and that it does not generalize across datasets of different size, so a selector must be retrained for each new fine-tuning dataset [3].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give it):

| knob | paper, Llama-3-8B-Instruct run [1] | paper, Merlinite-7B run [1] | paper, Pythia-2.8B run (Appendix) [1] |
| --- | --- | --- | --- |
| model learning rate | 1e-5 | 1e-5 | 1e-5 |
| selector learning rate | 5e-3 | 4e-3 | 4e-3 |
| batch size | 64 | 64 | 64 |
| selector training epochs | 3 | 2 | 2 |
| penalty strength $\gamma$ schedule | +3e-2 / epoch, from 0 | +2e-2 / epoch, from 0 | +2e-2 / epoch, from 0 |
| fine-tuning epochs | 2 | 3 | 3 |
| data selection percent $p$ | 80% (shared setting for all Section 4.2 experiments) | 80% (shared setting for all Section 4.2 experiments) | 80% |

  No general-purpose framework default exists to anchor against, since no framework other than the authors' own repository ships this method.
- Group-size/reuse trade-off (the paper's own ablation, Section 4.3): a too-small selection percent filters out more harmful samples but starves the fine-tuning stage of data, hurting target-domain performance; a too-large percent preserves target-domain performance but risks safety breaking; the paper reports safety alignment staying close to the initial aligned model's for $20\% \le p \le 80\%$ on Merlinite-7B [1].
- Cost/quality trade-off from Section 4.4: cost can be eased by lowering $p$ or by training the selector with a smaller model than the one being fine-tuned; the paper reports selecting 20% of the data with Pythia-2.8B as the selector-training model brings total pipeline runtime to 17 hours, close to the 16-hour no-selection baseline, while still improving over standard SFT [1].

## While it runs

- Signals and their healthy shapes: the paper's own monitoring is qualitative rather than a logged-metric list — it inspects the top- and bottom-ranked samples under the trained selector (Tables 6–7 in the paper) and reports that top-ranked data are safe and of good quality while bottom-ranked data have harmfulness-inducing instructions and potentially unsafe target responses, as explainability evidence rather than a live training signal [1]. No framework-level metric names exist for this method since no general framework ships it.
- Published reference runs: Table 1 (Llama-3-8B-Instruct) and Table 2 (Merlinite-7B) are the paper's own reference numbers — win rates of 60.22 (Anthropic HH test), 53.88 (SlimOrca test) and 69.29 (HEx-PHI) for SEAL on Llama-3-8B-Instruct, against 50.78/50.8/56.31 for random selection and 50/50/50 for standard SFT [1, Table 1]. Table 3 gives the paired safety/target-domain deltas at different selection percents and selector models [1, Table 3].
- Degeneracies and defaults: starting the penalty strength $\gamma$ high instead of warm-starting from 0 defeats the schedule's purpose of first fitting $\theta$ to the safe loss before the fine-tuning loss is mixed in [1]. Because the selector is one scalar per training-set row rather than an input-conditioned model, a selector trained on one dataset does not transfer to a differently sized or different fine-tuning dataset — a maintainer states this explicitly in a closed GitHub issue, in response to a question about generalizing a selector trained on 100 samples to a 1000-sample dataset [3]. Algorithm 2's simplification also rests on an explicit assumption, $d_\theta \gg N$, i.e. it is intended for the regime where the model has far more parameters than the fine-tuning set has samples [1]; the paper does not state what happens when that assumption is violated. A separate, unanswered GitHub issue asks why the paper's penalty-strength increments (0.02–0.03 per epoch) are so small, with no maintainer reply at the time of this check [3].
- Named successors: this card's Semantic Scholar citation search on 2410.07471 (73 citing papers, fetched 2026-08-09) found no paper that explicitly frames itself as a SEAL successor fixing a documented SEAL limitation; most citing papers address the same harmful-fine-tuning-defense problem with unrelated methods (e.g. Booster, Vaccine, Antidote, Lisa) rather than extending SEAL directly [1].
- Known failure modes, from the paper's own analysis: the performance gain from SEAL is smaller on the more compact Pythia-2.8B model than on the larger Llama-3-8B-Instruct and Merlinite-7B models, which the paper attributes to the quality of the learned data selector depending on the capacity of the model used to train it [1]. Under a benign (not adversarially mixed) fine-tuning dataset, SEAL improves the safety-domain scores over standard SFT but loses slightly on the target-domain (SlimOrca) score, which the paper attributes to the resulting reduction in fine-tuning dataset size when the discarded data was already good quality [1, Table 4].
- What the gain is — and is not: the paper's own claim is that SEAL enhances safety and output quality by filtering which existing samples a model is fine-tuned on; it does not claim SEAL adds any capability beyond what is already present in the underlying fine-tuning dataset or base model, and its own ablation shows the gain is sensitive to both the selection percentage and the capacity of the model used to train the selector [1].

## Sources

[1] Shen, Chen, Das, Chen, "SEAL: Safety-Enhanced Aligned LLM Fine-tuning via Bilevel Data Selection," arXiv:2410.07471, submitted 2024-10-09. https://arxiv.org/abs/2410.07471 — defines SEAL: problem formulation, penalty reformulation, Algorithms 1–2, framework steps, all reported experiments (Tables 1–5), ablations, and related-work characterizations of DSIR, LESS and the concurrent bilevel data-source-weighting method. Fetched 2026-08-09 (PDF full text via arxiv.org/pdf/2410.07471).

[2] Shen and Chen, "On Penalty-Based Bilevel Gradient Descent Method," arXiv:2302.05185. https://arxiv.org/abs/2302.05185 — the penalty-based bilevel optimization method SEAL's algorithm is built on. Fetched 2026-08-09 (abstract page).

[3] hanshen95/SEAL GitHub repository. https://github.com/hanshen95/SEAL — the only implementation found; README installation/run instructions, repository metadata (24 stars, last push 2025-02-20 as of this check), and issue-tracker comments (Required GPU #1, Model architecture of the selector #3, Why use such a small penalty strength? #4). All source-code and file-path claims (script names, install command) are read at commit `508a56dc782b8c3a49cf132882e6746a8653faf9` on branch `main`, the commit shown by the repository's file browser at fetch time. Fetched 2026-08-09 (repository HTML, README raw file, GitHub REST API for metadata and issues).

[4] Xie et al., "Data Selection for Language Models via Importance Resampling," arXiv:2302.03169. https://arxiv.org/abs/2302.03169 — DSIR, cited as SEAL's nearest offline data-selection alternative. Fetched 2026-08-09 (abstract page).

[5] Xia et al., "LESS: Selecting Influential Data for Targeted Instruction Tuning," arXiv:2402.04333. https://arxiv.org/abs/2402.04333 — LESS, cited as a second nearest offline alternative. Fetched 2026-08-09 (abstract page).

[6] hanshen95/SEAL repository metadata via the GitHub REST API (`/repos/hanshen95/SEAL`). Fetched 2026-08-09.
