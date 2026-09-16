# LCPO

https://arxiv.org/abs/2503.04697

GRPO with a length-aware reward: append a target token count to the prompt, and reward the model for both getting the answer right and landing near that target, so the reasoning model learns to stop when told to.

**LCPO** (Length Controlled Policy Optimization) is an online-RL method for making a reasoning language model's chain-of-thought length controllable at inference time, introduced in "L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning" (Aggarwal and Welleck, CMU, published at COLM 2025) as "a simple reinforcement learning method that optimizes for accuracy and adherence to user-specified length constraints" [1]. Its parent is GRPO [2]: the paper states it adopts GRPO for the policy-gradient update, though it notes the reward design is compatible with other RL algorithms [1]. The mechanism has two parts: every training prompt is augmented with an instruction naming a target length drawn uniformly between 100 and 4000 tokens, and the reward combines an answer-correctness term with a length-adherence term computed from the gap between the target and the generated length [1]. Three reasons the paper gives for existing: reasoning models generate chain-of-thought of uncontrolled length, so a compute budget cannot be allocated to hit a target performance level [1]; the existing length-control baseline, S1's budget forcing, only intervenes at decode time by truncating with an end-of-thinking delimiter or forcing continuation by appending "Wait", which the L1 paper reports severely degrades performance relative to the underlying model [3][1]; and the paper's own supervised-fine-tuning ablation (relabeling generations with their token lengths and training on them) failed to produce length control, which the paper attributes to the narrow length distribution per question and the absence of an online length-sensitive reward signal [1].

No landmark production system was found citing LCPO as an adopted component: a Semantic Scholar citation search on this arXiv ID (2503.04697), sorted by the first 50 returned citing papers, surfaces only 2026 preprints on adjacent efficient-reasoning topics (adaptive budgets, reasoning compression), none of which names itself as a direct successor or production adopter of LCPO's specific reward design [4]. In the originating paper's own results, L1 beats S1's budget forcing by over 100-150% relative and 20-25% absolute pass rate at 512 and 1024 token budgets on math reasoning benchmarks (Figure 2) [1]. Lineage in one line: PPO (2017) [5] -> GRPO (DeepSeekMath, 2024) [2] -> LCPO / L1 (2025) [1] -> no confirmed named successor found in the check above [4]. The paper's name collides with two unrelated prior uses of "LCPO" - "Locally Constrained Policy Optimization" (arXiv:2302.02182, 5 citations) and a 2025 "Length Controlled Preference Optimization" (arXiv:2508.10164, 1 citation); the L1 paper was confirmed as the intended match because it is the far more highly cited paper (356 citations) among the three under an exact-title collision check.

**When to pick it**: pick LCPO when you have a reasoning model already capable of long chain-of-thought and want inference-time control over how many tokens it spends, trading accuracy for latency on a per-request basis [1]. It is a reward design layered on GRPO [2], not a replacement RL algorithm, so choosing it means choosing GRPO (or another on-policy RL algorithm) as the trainer [1]. The paper's own SFT-based length control ablation - the nearest offline alternative - was found ineffective, generating long outputs regardless of the requested length [1]. The nearest inference-time-only alternative is S1's budget forcing, which needs no RL training but forcibly truncates or extends generation with special tokens rather than teaching the model to land near a target, and which L1 outperforms by the margin above [3][1].

**Variant of**: GRPO [2], via a reward-function change, not a change to GRPO's policy-gradient update.

**Data it needs**: a dataset of prompt/final-answer pairs with no reasoning traces required - the paper trained on the DeepScaleR-Preview dataset, about 40K math question-answer pairs drawn from AIME, AMC, Omni-Math, and STILL [1]. Each prompt is augmented at training time with a target-length instruction sampled uniformly from 100 to 4000 tokens; no length-labeled data is collected in advance [1]. On-policy: like GRPO, completions for the length-augmented prompts are sampled fresh from the current policy each step [1].

**Extra models**: a frozen reference model for the KL term, and a correctness verifier (the paper uses a rule-based math-answer checker, not a learned reward model) - no value network, inherited from GRPO [1][2]. The reference implementation (a fork of verl) trains with `use_kl_loss=True` and `kl_loss_coef=0.001` [6], which matches verl's own documented default KL loss coefficient of 0.001 [9] rather than the DeepSeekMath paper's coefficient of 0.04 [2]. Details in Cost.

**Shipped by**: no framework ships LCPO as a built-in trainer or reward recipe as of this check - trl's `trainer/__init__.py` export list contains `DPOTrainer`, `GRPOTrainer`, `KTOTrainer`, `RewardTrainer`, `RLOOTrainer`, and `SFTTrainer`, with no length-control trainer among them [10], and verl's `recipe` collection (searched by directory name for "l1", "lcpo", and "length") also contains none [8]. The only implementation is the paper authors' own repository, a fork that vendors verl and adds a custom reward function (`math_reward_fn` in `math_reward.py`) and launch scripts [6]. Building it on a supported trainer is a small lift, not a new algorithm: trl's `GRPOTrainer` already accepts a custom reward function that receives `completion_ids` and any extra dataset column via `**kwargs` [7], so a target-length column plus a length-penalty term dropped into that function reproduces the method without writing a new sampling loop.

## How it works

Each step: augment every prompt with a sampled target length, sample a group of completions from the current policy, score each completion with a reward that blends correctness and length adherence, and run GRPO's clipped-ratio update with the resulting group-relative advantages [1][2].

**Prompt augmentation.** For a dataset of prompt/answer pairs, each prompt $x_i$ is turned into $x_i^{new} = \text{Concat}(x_i, \text{"Think for } n_{gold,i} \text{ tokens."})$, where $n_{gold,i}$ is sampled uniformly between $n_{min}=100$ and $n_{max}=4000$ [1].

**LCPO-Exact reward** (Equation 1) [1]:

$$ r(y, y_{gold}, n_{gold}) = \mathbb{I}(y = y_{gold}) - \alpha \cdot \left| n_{gold} - n_y \right| $$

$\mathbb{I}(\cdot)$ is 1 if the generated answer $y$ matches the gold answer, $n_y$ is the generated response's token length, and $\alpha$ is a scalar (fixed at 0.0003 in the paper's runs) trading off correctness against exact length adherence [1]. This reward is additive: a wrong answer that hits the target length can still score higher than a correct answer that misses it by a lot. Worked example with $\alpha = 0.0003$: a correct answer at exactly the target length scores $1 - 0.0003 \times 0 = 1$; a correct answer 1000 tokens off target scores $1 - 0.0003 \times 1000 = 0.7$; an incorrect answer exactly on target scores $0 - 0 = 0$.

**LCPO-Max reward** (Equation 2) [1], used to fine-tune from an LCPO-Exact checkpoint so the model respects a maximum rather than an exact length:

$$ r(y, y_{gold}, n_{gold}) = \mathbb{I}(y = y_{gold}) \cdot \operatorname{clip}\!\left(\alpha \cdot (n_{gold} - n_y) + \delta,\; 0,\; 1\right) $$

Here $\delta = 0.5$ so a correct answer with a small overshoot still scores above zero, and the reward is multiplicative rather than additive: an incorrect answer scores 0 regardless of length, which the paper says is needed to keep the GRPO objective's gradient meaningful under a hard length cutoff [1]. Worked example, $\alpha = 0.0003$, $\delta = 0.5$, correct answer: on target ($n_y = n_{gold}$) scores $\operatorname{clip}(0 + 0.5, 0, 1) = 0.5$; 500 tokens under target scores $\operatorname{clip}(0.15 + 0.5, 0, 1) = 0.65$; 2000 tokens over target scores $\operatorname{clip}(-0.6 + 0.5, 0, 1) = 0$. L1-Max is trained with a dual objective: when a prompt's target length is meant to be exact it uses Equation 1, otherwise it defaults to the Equation 2 maximum-constraint mode [1].

**The RL update itself is unmodified GRPO** [2]: the scalar reward from either equation feeds directly into GRPO's group-relative advantage (normalize each completion's reward by its group's mean and std) and clipped-ratio policy-gradient objective; LCPO changes only what enters $r_i$, not how $\hat{A}_{i,t}$ or the clipped loss are computed [1][2].

**Reward-function ablation.** Appendix A.7 tests three variants of Equation 2 against the standard dual-objective L1-Max (which trains with both Equation 1 and Equation 2): a "Single Objective" variant using only Equation 2, an "Addition" variant that adds rather than multiplies the correctness and length-penalty terms ($r = \mathbb{I}(y=y_{gold}) - \alpha \cdot |n_{gold}-n_y|$), and a "Sigmoid" variant that replaces the clip with a sigmoid penalty ($r = \mathbb{I}(y=y_{gold}) \cdot \sigma(\alpha \cdot (n_{gold}-n_y))$), each trained for 120 RL steps from the same L1-Exact checkpoint [1]. The Addition variant reaches a 0% budget-violation rate, but only because the model collapses to trivially short chain-of-thought that satisfies the length constraint while performing substantially worse; excluding Addition, the paper reports the remaining three - standard L1-Max, Single Objective, and Sigmoid - perform similarly with no statistically significant differences, and the paper selects the standard dual-objective form [1].

## Cost

**Theory, from the method's own math:**

- Time and memory beyond GRPO: none from the reward change itself - $r(y, y_{gold}, n_{gold})$ is a scalar computed from the already-generated completion's length and a correctness check, adding no extra forward or backward pass [1]. All of GRPO's own cost structure carries over unchanged: group size G completions generated per prompt, one training pass over the group with no value network, and one frozen-reference forward pass per token if the KL term is enabled [2].
- The only added cost is upstream of the model: prompts are lengthened by the appended target-length instruction, and generation must run long enough to observe $n_y$ before the length term of the reward can be computed - the length penalty cannot be applied to a partial rollout.

**In practice, per framework:**

- The only working implementation is the authors' own fork of verl at commit `93dd330` [6]. Its launch scripts set `algorithm.adv_estimator=grpo`, `actor_rollout_ref.rollout.n=16` (16 completions generated per prompt each step), `actor_rollout_ref.actor.use_kl_loss=True` with `kl_loss_coef=0.001` (one extra frozen-reference forward pass per token, no extra trained model), and a training batch size of 128, run on a single node of 8 A100-80GB GPUs with vLLM as the rollout backend [6]. This is GRPO's own cost profile carried unchanged - a policy-only training pass over 16 completions per prompt plus one reference-model forward pass, no value network - LCPO adds nothing to it beyond the reward function.
- No trl-based implementation exists to report a framework-specific cost for; the estimate above (no added time/memory beyond GRPO's own) is a theoretical reading, not a measured trl number.

## How to use it

- Data prep: a prompt/final-answer dataset with no chain-of-thought needed (the paper used ~40K math QA pairs from DeepScaleR-Preview) [1]; at training time, append a target-length instruction to each prompt with the target sampled uniformly from `nmin` to `nmax` [1].
- Reward convention: LCPO-Exact (Equation 1) for training a model to hit an exact length; switch to LCPO-Max (Equation 2), fine-tuned from the LCPO-Exact checkpoint, when the goal is a maximum-length budget rather than an exact one [1]. The reference implementation's `math_reward_fn` selects between four length-penalty shapes (linear, linear-both-sided, sigmoid, sigmoid-exact) via a `RewardConfig` flag, all variants of the paper's linear (Eq. 1) and multiplicative-sigmoid family (Eq. 2) [6].
- Key knobs, with the paper's own values as the only published anchor (no independent framework default exists for LCPO-specific knobs; GRPO knobs below are the reference implementation's own launch-script values, not a general framework default):

| knob | paper (L1-Exact) [1] | reference repo launch script (L1-Exact) [6] |
| --- | --- | --- |
| target-length range ($n_{min}$, $n_{max}$) | 100, 4000 | not checked beyond training data prep scripts |
| $\alpha$ (Eq. 1) | 0.0003 | `reward_config.alpha=0.0003` |
| $\delta$ (Eq. 2) | 0.5 | not checked |
| GRPO group size G | not stated | `rollout.n=16` |
| learning rate | 1e-6 (same as base DeepScaleR checkpoint) | `actor.optim.lr=1e-6` |
| KL loss coefficient | not stated for LCPO | `kl_loss_coef=0.001` [6], matching verl's own documented default of 0.001 [9], not the DeepSeekMath paper's 0.04 [2] |
| training batch size | 128 | `data.train_batch_size=128` |
| max response length (train) | 4096 | `data.max_response_length=4096` |
| training steps | 700 (L1-Exact), further 120 for L1-Max | `trainer.total_epochs=3` (epoch-based, not directly comparable) |

- Trade-off a run designer faces: $\alpha$ trades correctness against length adherence directly - the paper notes a lower $\alpha$ prioritizes correctness while a higher one enforces stricter length adherence [1]; picking $n_{min}, n_{max}$ too narrow limits the range of budgets the trained model can later be prompted with, since the paper's own limitations section notes the model does not generalize to requested lengths longer than what it was trained on [1].

## While it runs

- Signals and their healthy shapes: the paper's own training-log analysis (Appendix A.3, LCPO-Exact) tracks reward, minimum response length, and a "token adherence score" defined as reward minus mean solve rate; over roughly the first 300 RL steps the model prioritizes raising solve rate while minimum response length keeps falling, then at about step 300 the paper reports a phase transition where reward and token-adherence score both rise sharply while minimum response length drops sharply to converge near the 100-token lower bound of $n_{gold}$ [1]. The paper calls this transition an "Aha Moment" for token adherence [1].
- Published reference runs: Figure 14 in the paper is the training-log plot behind the signal description above (reward, adherence score, and minimum length vs. RL step for LCPO-Exact) [1]; no separate published external reference run (e.g. a public W&B log) was found in this check.
- Degeneracies and defaults: the additive reward-ablation variant (Appendix A.7) collapses to trivially short, low-quality chain-of-thought while still achieving a 0% budget-violation rate - a variant that looks perfectly length-compliant while failing at reasoning quality [1]. The paper's own limitations section states models trained with LCPO do not reliably generalize to requested lengths longer than the training range ($n_{max}=4000$ in the paper's runs) [1].
- Named successors: none found - see the citation-search result and caveat above [4].
- Known failure modes, from the paper's own limitations section: length control does not extrapolate past the training-time maximum target length [1]; the paper trains only on complete-output length rather than reasoning-only length, and flags reward variants that specifically target reasoning-token length as unexplored future work [1]. Also from the paper's own results (not the limitations section, but its analysis of the SFT ablation, Appendix A.6): supervised fine-tuning on length-relabeled data reliably fails to produce length control, generating very long outputs regardless of the requested length, which the paper attributes to a narrow per-question length distribution in the generated data and the absence of an online length-sensitive reward during SFT [1].
- What the gain is - and is not: the paper's own analysis of L1 versus S1 attributes the gap to L1 adapting its chain-of-thought to fit within a length constraint without disrupting the reasoning process, whereas S1 truncates mid-reasoning at a fixed token count [1]. The paper separately reports that models trained with LCPO become unexpectedly strong short-CoT models at very low token budgets - "Short Reasoning Models" - but frames this as a length-control side effect discovered post hoc, not as LCPO adding new reasoning capability beyond what the base model already has [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Framework/library sources are dated readings against a moving target; source-code claims carry the commit read.

[1] Aggarwal and Welleck, "L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning", COLM 2025. https://arxiv.org/abs/2503.04697 - defines LCPO: reward equations, hyperparameters, results, ablations, limitations. Fetched 2026-08-09 (PDF full text via arxiv.org/pdf/2503.04697, version v2, 2025-10-03).

[2] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO, the parent method; its GSM8K/MATH RL run uses KL coefficient 0.04. Fetched 2026-08-09 (PDF full text via arxiv.org/pdf/2402.03300).

[3] Muennighoff et al., "s1: Simple test-time scaling", 2025. https://arxiv.org/abs/2501.19393 - defines budget forcing, the nearest inference-time-only alternative: truncating generation with an end-of-thinking delimiter, or appending "Wait" to force continuation. Fetched 2026-08-09 (PDF full text via arxiv.org/pdf/2501.19393).

[4] Semantic Scholar Graph API, citations endpoint for arXiv:2503.04697, first 50 results by default ordering. https://api.semanticscholar.org/graph/v1/paper/arXiv:2503.04697/citations - used to check for named successors and landmark adopters; found none among the returned titles. Fetched 2026-08-09. This is a partial read (50 of the paper's citing works) and does not rule out an adopter outside that set.

[5] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, GRPO's own parent, cited for the lineage line only. Fetched 2026-08-09 (abstract page).

[6] cmu-l3/L1 GitHub repository, the paper authors' reference implementation (a fork that vendors verl). https://github.com/cmu-l3/L1 - reward function (`math_reward.py`), launch scripts (`scripts/train/run_l1_exact.sh`, `run_l1_max.sh`), and base config (`config/ppo_trainer.yaml`). Read at commit `93dd330102f49450e1bc8227ad8979f8b5482e44` (2025-05-14), fetched 2026-08-09.

[7] trl GRPOTrainer documentation. https://huggingface.co/docs/trl/main/en/grpo_trainer - reward-function interface (`completion_ids`, `**kwargs`) and default `beta=0.0`, used to state what building LCPO on trl would take. This is an unpinned `main`-branch doc; read 2026-08-09.

[8] verl-project/verl-recipe GitHub repository (the `recipe` submodule of volcengine/verl). https://github.com/verl-project/verl-recipe - directory listing searched for an LCPO/L1/length-control recipe; none found among the listed recipes. Unpinned `main` branch, fetched 2026-08-09.

[9] verl GRPO algorithm documentation. https://verl.readthedocs.io/en/latest/algo/grpo.html - states "actor_rollout_ref.actor.kl_loss_coef: The coefficient of kl loss. Default is 0.001." Unpinned `latest` build, fetched 2026-08-09.

[10] trl `trl/trainer/__init__.py` source file. https://github.com/huggingface/trl/blob/main/trl/trainer/__init__.py - the package's own trainer export list (`DPOTrainer`, `GRPOTrainer`, `KTOTrainer`, `RewardTrainer`, `RLOOTrainer`, `SFTTrainer`); no length-control trainer is exported. Unpinned `main` branch, fetched 2026-08-09.
