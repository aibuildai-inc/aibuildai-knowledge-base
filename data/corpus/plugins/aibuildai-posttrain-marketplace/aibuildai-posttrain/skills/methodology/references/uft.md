# UFT

GRPO-based RFT with an extra log-likelihood loss on a solution "hint" that is concatenated to the prompt and shrunk to zero length over training, so a single run interpolates smoothly from SFT to pure RL.

**UFT** (Unified Fine-Tuning) is an online post-training method that unifies supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) into one training objective, introduced by Liu, Farina, and Ozdaglar (MIT) in "UFT: Unifying Supervised and Reinforcement Fine-Tuning", https://arxiv.org/abs/2505.16984 [1]. Two arXiv papers share the name "Unified Fine-Tuning"; this card follows the shortlist's own resolution of that name collision to this paper over the alternate candidate, "UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function", arXiv:2410.21438 [5]. UFT's parent is GRPO [2]: each training step samples a group of rollouts and updates the policy with GRPO's clipped surrogate objective plus a KL penalty toward a frozen reference policy [1]. UFT modifies this in two ways: it concatenates each prompt with a hint - a prefix of the ground-truth solution whose length is annealed from long to zero over the first $T_{hint}$ steps - and it adds a log-likelihood loss on the hint tokens to the RL objective, so the model is taught to imitate the hint while it explores the rest of the trajectory with RL [1]. The paper gives three reasons for this design: small models often never sample a correct trajectory under a sparse outcome-only reward, so plain RFT teaches them nothing [1]; SFT alone generalizes worse than RFT for larger, more capable models [1]; and the paper proves that RFT's sample complexity to reach 50% pass@1 is exponential in the reasoning-tree height $H$ (lower bound $B^H/4$ explorations, Theorem 4.2), while UFT's is polynomial ($O(BH^5(\log B)^2/\Delta^2)$, Theorem 4.3) [1].

No later paper's adoption of UFT was found in the sources read for this card (the paper's own text, its GitHub repository, and the two arXiv listings); the paper is from May 2025 and reports results only on its own five base models. In the paper's own results, averaged over Countdown, MATH(3,4,5), and Logic, UFT reaches 9.45% on Qwen2.5-0.5B versus 7.28% for the SFT-then-RFT baseline and 3.25% for plain RFT; on Qwen2.5-3B it reaches 30.93%, close to RFT's own 32.15% and well above SFT-then-RFT's 17.34% [1, Table 8]. Lineage in one line: GRPO (DeepSeekMath, 2024 [2]) + R3's hint-based curriculum (2024 [4]) -> UFT (2025 [1]) -> no named successors found.

**When to pick it**: online RL when full reference solutions (not just pass/fail labels) are available and you want one run that behaves like SFT while the model is weak and like RFT once it is strong, rather than a hard-cutover SFT-then-RL pipeline [1]. Prefer plain SFT, the nearest offline alternative, when there is no reward signal to explore against - SFT only maximizes log-likelihood of the reference solution, with no rollout or reward [1]. Prefer R3, the nearest online neighbor, only if you want hints purely as an exploration aid without also training on them as supervision: the UFT paper characterizes R3 as sampling hint length uniformly across all of training, which it says misaligns with the distribution of interest (zero hint length) at evaluation time, unlike UFT's scheduler that anneals hint length to zero and adds a log-likelihood term on the hint [1] (this is UFT's own characterization of R3 [4], not a claim made by [4] itself).

**Variant of**: GRPO [2], extended with the hint mechanism introduced by R3 [4].

**Data it needs**: (question, solution, answer) triples where the solution is a real reasoning trace, not just a correctness label - the paper splits each solution into sentences and groups them into buckets to define hint granularity [1]. Training is on-policy: after the hint prefix, the rest of each trajectory is sampled fresh from the current policy every step using GRPO [1]. The paper's own training scale: batch size 256, 500 steps, models from 0.5B to 3B, at a reported project cost of roughly $10,000 GPU-hours across all experiments and ablations [1, Appendix B.2].

**Extra models**: the same as GRPO - a frozen reference policy for the KL term (coefficient $\beta$, paper default 0.001) - and no value network [1]. The hint's ground-truth tokens come directly from the dataset, not from a separate model. See Cost for what the hint adds on top of GRPO's own footprint.

**Shipped by**: no library ships UFT as a named trainer. The paper's own code is a personal fork of verl, not merged upstream, with two modifications documented in the repository: a change to the RL dataset class that concatenates the hint to the prompt, and a change to the actor's policy-loss function that adds the hint log-likelihood term, both pointed at specific files and line numbers in the fork [3]. Building UFT on top of an existing GRPO trainer would need a hint-length scheduler, a change to prompt construction to concatenate the sampled hint, and an added log-likelihood loss term on the hint tokens inside the existing GRPO loss - a loss-and-data-pipeline modification on top of a GRPO trainer, not a new sampling loop.

## How it works

Each step: sample a hint length $l$ from a schedule; roll the first $l$ steps of the trajectory along the ground-truth solution $\pi^*$; sample the remaining $H-l$ steps from the current policy $\pi$; update $\pi$ on GRPO's clipped objective over the sampled suffix plus a log-likelihood loss on the hint prefix [1].

The paper formalizes problem solving as a search tree of height $H$ and branching factor $B$; a trajectory $(s_h,a_h)_{h=0}^{H-1}$ is a path from the root to a leaf, and only the leaf carries a reward $R(s)\in[0,1]$ [1]. The plain RFT objective it starts from is:

$$ J^{RFT} = \mathbb{E}_{s_0=s_{root},\,(s_h,a_h)_{h=0}^{H-1}\sim\pi}\Big[\, J^{value}\big((s_h,a_h)_{h=0}^{H-1}\big) \;-\; \beta\sum_{h=0}^{H-1} KL\big(\pi(\cdot|s_h)\,\|\,\pi^{ref}(\cdot|s_h)\big) \Big] $$

where $J^{value}$ is GRPO's clipped surrogate over the group's advantages $\hat A$, using the ratio between the current policy $\pi$ and the previous-step policy $\pi^{old}$ [1, footnote 3] (the same clipped-ratio construction as the defining GRPO paper [2]). UFT's objective replaces the first $l$ steps of the rollout with the ground-truth trajectory $\pi^*$ and adds a KL term pulling $\pi$ toward $\pi^*$ over that prefix [1]:

$$ J^{UFT} = \mathbb{E}\Big[\, J^{value}\big((s_h,a_h)_{h=l}^{H-1}\big) - \beta\!\!\sum_{h=l}^{H-1}\!\! KL\big(\pi(\cdot|s_h)\|\pi^{ref}(\cdot|s_h)\big) - \beta\!\!\sum_{h=0}^{l-1}\!\! KL\big(\pi^*(\cdot|s_h)\|\pi(\cdot|s_h)\big) \Big] $$

Since $\pi^*$ is unknown but a sample from it (the dataset's own solution, $a^*_h\sim\pi^*(\cdot|s^*_h)$) is available, the paper rewrites the last term as an unbiased log-likelihood estimator, giving the objective UFT actually optimizes [1]:

$$ J^{UFT} = \mathbb{E}_{l}\Big[\, J^{value}\big((s_h,a_h)_{h=l}^{H-1}\big) - \beta\!\!\sum_{h=l}^{H-1}\!\! KL\big(\pi(\cdot|s_h)\|\pi^{ref}(\cdot|s_h)\big) + \beta\sum_{h=0}^{l-1} \log \pi(a^*_h\,|\,s^*_h) \Big] $$

Remark 3.1 in the paper states the two limits explicitly: with hint proportion $p\equiv 0$ (so $l=0$ always), the added log-likelihood sum is empty and (3.3) reduces exactly to plain RFT; with $p\equiv1$ (so $l=H$), the RL rollout from $h=l$ to $H-1$ is empty and (3.3) reduces to plain SFT, maximizing the log-likelihood of the full solution [1].

**Hint-length schedule.** The paper maintains $p\in[0,1]$, the expected fraction of the solution revealed, cosine-annealed from $p_{high}$ to $p_{low}$ over the first $T_{hint}$ steps [1, Eq. B.1], then samples $l\sim\mathrm{Binomial}(L,p(t))$ where $L$ is the number of solution chunks, giving $\mathbb{E}[l]=p(t)\cdot L$ [1]. Worked example with the paper's own defaults ($p_{low}=0.05$, $p_{high}=0.95$, $T_{hint}=300$ [1, Table 1]): at step $t=150$, halfway through the hint phase, $\cos(\pi\cdot151/300)\approx0$, so $p(150)\approx p_{low}+(p_{high}-p_{low})/2\approx0.5$; with $L=4$ solution chunks, $l\sim\mathrm{Binomial}(4,0.5)$ gives an expected hint of 2 chunks. If $l=2$ is drawn, the added loss term is $\beta\big(\log\pi(a^*_0|s^*_0)+\log\pi(a^*_1|s^*_1)\big)$ - an ordinary teacher-forced log-likelihood on those two ground-truth chunks - while GRPO's clipped objective runs on the remaining $H-2$ steps that the policy itself generates.

The paper's Related Work section notes that RL for reasoning more broadly splits into process supervision (a reward at each step of the trace) and outcome supervision (a single reward for the whole trace); it states that it focuses on outcome supervision for its efficiency and simplicity, and does not define a process-supervision variant of UFT itself [1].

## Cost

**Theory, from the method's own math:**

- Time: relative to plain GRPO, UFT adds one teacher-forced forward pass over the hint tokens (to compute $\log\pi(a^*_h|s^*_h)$, no extra backward beyond the existing policy gradient) and extends the prompt fed to generation by the hint length during the hint phase; this cost decays to zero as $l\to0$ over $T_{hint}$ steps [1].
- Memory: no new trained model versus GRPO - one policy, one frozen reference model for the KL term, no value network [1]. The only addition is the ground-truth hint tokens held alongside each prompt, which is data, not a model.
- A naive reading of the objective might suggest a separate optimal-policy model $\pi^*$ must be loaded to compute the KL term against it; the paper's own rewrite (Eq. 3.3) removes that requirement by replacing the KL term with a log-likelihood on the dataset's own solution tokens, so no extra model is needed for it [1].

**In practice, per framework:**

- The only implementation is the paper's own fork of verl [3]. Its README documents hardware requirements for the experiments in the paper: Qwen2.5-0.5B/1.5B and Llama-3.2-1B were trained on 2 H100 GPUs (or 1 H100 with rollouts reduced from 4 to 2), and Qwen2.5-3B and Llama-3.2-3B needed 4 H100 GPUs [3]. The paper separately reports the total project cost across all reported experiments and ablations as roughly $10,000 GPU-hours [1, Appendix B.2]; this is an aggregate figure, not a per-run cost.
- No other framework has implemented UFT, so no cross-framework comparison of implementation cost is possible here.

## How to use it

- Data: (question, solution, answer) triples; the solution must be decomposable into ordered chunks to serve as hints. The paper splits solutions into sentences, then groups the sentences into $L$ roughly equal buckets (its ablation tests $L\in\{4,5,6\}$) [1, Appendix B.3].
- Reward: outcome-only, three possible values in the paper's setup - accuracy reward 1.0, format-correctness reward 0.1, incorrect reward 0.0 [1, Table 1]; the resulting sub-optimality gap used in the theory is $\Delta = 1.0-0.1=0.9$ [1, Eq. 4.2].
- Key knobs. No framework ships UFT as a named trainer, so the only anchor available is the paper's own setting; the table below is a single column, not a comparison:

| knob | paper value [1, Table 1] |
| --- | --- |
| training batch size | 256 |
| mini-batch size | 64 |
| hint chunks $L$ | 5 |
| learning rate | $10^{-6}$ |
| KL coefficient $\beta$ | 0.001 |
| total steps $T$ | 500 |
| hint-phase steps $T_{hint}$ | 300 |
| rollouts per prompt | 4 |
| $p_{low}$ / $p_{high}$ | 0.05 / 0.95 |
| SFT epochs (SFT baseline only) | 5 |

  All hyperparameters not listed here follow verl's own defaults, per the paper [1]. The paper's own ablations report the method is fairly insensitive to several of these: hint chunk count $L\in\{4,5,6\}$ changed accuracy by at most ~2 points on Qwen2.5-0.5B/MATH(3,4,5) (27.44% / 27.15% / 25.20%) [1, Table 2]; hint-phase length $T_{hint}\in\{200,300,400\}$ changed Countdown accuracy by at most ~2 points (52.15% / 50.39% / 50.00%) [1, Table 5]; and the hint-length distribution (binomial vs. a two-point alternative) changed accuracy by about 2 points (50.39% vs. 48.63%) [1, Table 6]. $\beta$ matters more for small models: on Qwen2.5-0.5B/Countdown, accuracy ranged from 46.09% at $\beta=0.0005$ to 59.95% at $\beta=0.002$, while on Qwen2.5-3B/Countdown the same sweep only moved accuracy between 75.59% and 78.22%, which the paper attributes to larger models' stronger pretrained priors [1, Tables 3-4].
- Trade-off a run designer faces: a longer $T_{hint}$ or higher $p_{high}$ gives the model more imitation signal for longer but delays the point at which it must perform unaided, and the paper's own ablation shows this trade-off is fairly flat within the ranges it tested (above) [1].

## While it runs

- Signals and their healthy shapes: the paper tracks the cumulative average exploration success rate - the fraction of rollouts per step that reach the correct leaf - as a direct diagnostic of whether hints are fixing the sparse-reward problem; on Qwen2.5-0.5B/Logic, RFT alone almost never explores the correct answer while UFT finds it on nearly every step [1, Fig. 6]. It also tracks the average reward on the training batch over steps, showing UFT's cosine-annealed hint schedule converges faster and more smoothly than a step-wise ("staged") hint-length reduction [1, Fig. 4, right].
- Published reference runs: the paper's own training curves for Qwen2.5-0.5/1.5/3B on Countdown, MATH(3,4,5), and Logic compare UFT against Base, SFT, RFT, SFT-RFT, and R3 baselines step-by-step [1, Fig. 2, Fig. 5]; the aggregate end-of-training numbers are in Table 8 [1]. The paper isolates which of UFT's two changes over RFT is load-bearing with an "RFT (cosine)" ablation - RFT equipped only with the annealed hint-length scheduler, but without the added hint log-likelihood loss term. On Countdown, RFT (cosine) beats R3 but still underperforms SFT-RFT for both Qwen2.5-0.5B and Llama-3.2-1B, and for Llama-3.2-1B it scores even below plain SFT [1, Fig. 5]. The paper attributes this to the model's performance being hindered by knowledge it gained through pretraining, which motivates adding the log-likelihood term as UFT's second modification on top of the hint scheduler [1]. Since UFT itself (scheduler plus log-likelihood term together) outperforms both R3 and SFT-RFT in the same figure [1, Fig. 5], this ablation is the paper's own evidence that the log-likelihood term on the hint, not the annealing schedule alone, is what makes UFT work - directly analogous to how PPO's own ablation isolates the clipping term.
- Degeneracies and defaults: with $\beta\equiv$ any value but hint proportion held at $p\equiv1$ throughout training, UFT collapses to pure SFT, and at $p\equiv0$ it collapses to pure RFT, per the paper's own Remark 3.1 [1]. Using a uniform (non-annealed) hint-length distribution across all of training, as R3 does, is reported by the paper to cause a mismatch between the training distribution (hints present) and the evaluation distribution (no hints), leading to performance collapse at test time [1] (this is the paper's characterization of R3 [4], not a claim made by [4] itself).
- Named successors: none found. The search was: the paper's own text, its GitHub repository, and title searches for both arXiv candidates matching "Unified Fine-Tuning" / "UFT" in this dataset's shortlist; no paper describing itself as fixing a documented bias in this UFT was located.
- Known failure modes: from the paper's own Conclusion and Limitations section - all reported experiments use only human-annotated solutions (not model-generated long chain-of-thought data) and GRPO as the sole RL algorithm, so the paper does not report results with other RL algorithms (it names REINFORCE++ and DAPO as directions for future work) or with SFT data distilled from larger models; it also does not evaluate at state-of-the-art (e.g., 70B) scale due to compute constraints, though it reports a smaller-scale result (Qwen2.5-1.5B on the Logic task) as suggestive evidence that the method still helps at that scale [1, Section 6]. Separately, the paper's own Table 8 shows the SFT/RFT/UFT ordering it reports for Qwen2.5 does not hold cleanly for Llama-3.2: on Llama-3.2-3B, UFT's average across the three training tasks is 3.72%, below both SFT-RFT's 4.79% and R3's 5.03% [1, Table 8] - a case the paper's own text does not flag as a limitation, but the numbers themselves show the reported gain is not universal across model families.
- What the gain is - and is not: the paper's own claim is that, averaged over Countdown, MATH(3,4,5), and Logic, small models trained with UFT tend to match SFT-level memorization while larger models trained with UFT tend to match RFT-level generalization, and sometimes exceed it [1, Section 5, Table 8]. The theoretical sample-complexity result (Theorem 4.3) is proved under the paper's own search-tree formalism with a fixed sub-optimality gap and a specific three-valued reward structure (accuracy/format/incorrect); it is not a claim that UFT improves sample complexity under arbitrary reward shapes [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. GitHub documentation is an unpinned, mutable build; the repository state described here was read at the commit hash cited in its own README (`ad3c8ec7d258d4e4cf3f359dd75925d0f127a520`) for the two linked source files, and as of the fetch date below for the rest of the README.

[1] Liu, Farina, and Ozdaglar, "UFT: Unifying Supervised and Reinforcement Fine-Tuning", NeurIPS 2025. https://arxiv.org/abs/2505.16984 - defines UFT: objective, hint scheduler, theoretical sample-complexity results, all experimental tables and ablations, and the Related Work / Limitations sections. Fetched 2026-08-10 (PDF full text via arxiv.org/pdf/2505.16984, converted to text).

[2] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO, UFT's parent method. Fetched 2026-08-10 (abstract page; the GRPO objective quoted on this card is [1]'s own restatement in its footnote 3, not independently re-derived from this source).

[3] liumy2010/UFT GitHub repository (README). https://github.com/liumy2010/UFT - the paper's own implementation, a fork of verl; documents the two modified files/line ranges, install/usage commands, and per-model GPU requirements. Fetched 2026-08-10.

[4] Xi et al., "Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning", ICML 2024. https://arxiv.org/abs/2402.05808 - defines R3, UFT's nearest online neighbor and the source of the hint mechanism UFT extends. Title and arXiv ID confirmed by fetching the abstract page 2026-08-10; all claims about R3's own design on this card are attributed to [1]'s description of it, not to an independent reading of [4]'s full text.

[5] Cheng et al., "UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function", 2024. https://arxiv.org/abs/2410.21438 - the alternate candidate sharing this method's name; cited only to record the disambiguation, not as a source for any method claim on this card. Fetched 2026-08-10 (abstract page).
