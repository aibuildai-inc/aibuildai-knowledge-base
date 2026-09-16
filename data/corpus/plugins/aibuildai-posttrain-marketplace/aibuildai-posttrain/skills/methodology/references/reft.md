# REFT

SFT warm-up on (question, chain-of-thought, answer) triples, then online PPO where the reward is a rule that checks the extracted final answer against the ground truth, so the policy can explore many correct reasoning paths instead of memorizing the one annotated path per question.

**ReFT** (Reinforced Fine-Tuning) is introduced by Luong et al. as a way to enhance the generalizability of learning LLMs for reasoning, with math problem-solving as an example, because ordinary SFT on chain-of-thought (CoT) annotations does not show sufficiently strong generalization since training relies on only one annotated reasoning path per question [1]. The paper is at https://arxiv.org/abs/2401.08967 [1]. Its parent is PPO, defined by Schulman et al. as a family of policy-gradient methods that alternate between sampling data and optimizing a clipped surrogate objective [2]; ReFT states explicitly that it employs PPO with a clipped objective for its RL stage [1]. Mechanically, ReFT first warms up the policy with supervised fine-tuning on (question, CoT, answer) tuples, then removes the CoT annotations and runs on-line RL on (question, answer) tuples only: the policy samples a CoT on-policy each step, a rule-based terminal reward checks the extracted answer against the ground truth, and PPO updates the policy and a value head built on top of it [1]. The paper gives two reasons beyond generalization: an abundance of reasoning paths can be sampled automatically and rewards derived naturally from the ground-truth answer, avoiding the cost of collecting more CoT annotations [1]; and the authors tried the offline alternatives DPO and IPO first and dropped them because, being offline, they cannot explore new CoT paths themselves and their own preference data was not well suited to reward modeling [1] (see When to pick it).

The paper reports its own gains on Table 2: on CodeLLAMA-7B, averaged across GSM8K/SVAMP/MathQA_MCQ with N-CoT/P-CoT value accuracy, ReFT reaches 59.31/75.43 versus 52.56/67.96 for the SFT baseline, 55.04/71.41 for an Offline Self-Training baseline, and 53.40/71.37 for an Online Self-Training baseline that shares ReFT's warm-up and on-policy sampling but updates with self-training rather than PPO [1]. This paper title collides with unrelated later work also abbreviated "ReFT" (representation fine-tuning, at 2 citations, and "Nested-ReFT," at 0 citations, among the shortlist's alternates); the pick used here is the 320-citation "ReFT: Reasoning with Reinforced Fine-Tuning" (arXiv:2401.08967), confirmed as the intended paper by being the top-cited match among the name-colliding candidates. No landmark adopter is confirmed here: this card checked the full text of DeepSeekMath [3], DeepSeek-R1 [4], and Tulu 3 [5] for a citation to ReFT or to Luong et al., and found none in any of the three, so no adoption claim is made. Lineage in one line: PPO (2017 [2]) -> ReFT (2024 [1]), an application of PPO to rule-based math reward with no confirmed adopters found in the systems checked.

**When to pick it**: online PPO for a reasoning task where an answer can be extracted and checked by a rule (exact match, or a program's output), so a reward model is unnecessary and the reward function can be used directly [1]. Prefer plain SFT [1] when you only need to imitate the annotated CoTs and generalization to unseen problems is not a concern. Prefer DPO [6], the nearest offline alternative, only when you cannot sample fresh generations at training time; ReFT's own authors tried DPO and IPO on GSM8K and found their performance merely on par with the weaker Offline Self-Training baseline, attributing the gap to DPO/IPO's inability to explore further CoT paths and to preference data sampled from a sub-optimal warm-up policy [1]. The nearest online neighbor is GRPO [7]: like ReFT it can run with a rule-based terminal reward and no reward model, but GRPO removes the value network entirely and computes advantages from a group of same-prompt samples' reward mean and std [7], whereas ReFT keeps a PPO value head trained by GAE [1]; no citation link between the two papers was found by this card's checks, so this is a structural contrast, not a documented lineage.

**Variant of**: PPO [2].

**Data it needs**: warm-up stage needs (question x, CoT e, answer y) triples; the RL stage needs only (question x, answer y) pairs, since the CoT is sampled on-policy at each RL step [1]. The paper's own training scale: GSM8K 7,465 N-CoT / 7,356 P-CoT training triples, SVAMP 3,076/3,043, MathQA_MCQ 14,862/15,250, MathQA_numeric 8,955/7,672, each with CoT annotations obtained via few-shot prompting of GPT-3.5-turbo [1]. The RL stage is on-policy: at each step the current policy samples fresh CoTs for the mini-batch, and the resulting rewards/advantages are only ever computed against that batch's own samples [1].

**Extra models**: one value model, no reward model, and one frozen reference policy. Following Ziegler et al., the value model $V_\phi$ is a linear value head appended to the last hidden states of the policy after warm-up [1]; there is no learned reward model, since the terminal reward is a rule-based comparison of the extracted answer to the ground truth [1]; the total reward also subtracts a KL penalty against the frozen post-warm-up policy $\pi_\theta^{(0)}$, so a frozen reference copy is held for scoring, not for training [1]. See Cost and Shipped by for how this is realized (or not) by frameworks: none of the checked frameworks name a `reft_trainer` object.

**Shipped by**: neither trl [8] nor verl [9] implements a trainer named for ReFT; this card checked trl's top-level exports [8] and verl's algorithms index [10], and neither lists it. The paper's own code release is a set of one-off training scripts (`train_sft_model.py`, `train_rl_reft.py`) rather than a reusable library [11]. Building it from a general PPO trainer takes: (a) a rule-based scoring function plugged in as the reward, (b) a warm-up SFT phase before the RL loop, and (c) KL applied to the reward, not to the loss (see How it works). Two plausible bases: trl's `PPOTrainer` lives only under `trl.experimental.ppo`, a namespace whose own docstring warns anything there "may change (or be removed) in any release without deprecation" [12], and its constructor requires a `reward_model: PreTrainedModel`, i.e. an actual scoring model object, not an arbitrary Python reward function [13] — so ReFT's rule-based reward would have to be wrapped as a fake model rather than passed directly. verl's stable PPO trainer/recipe [9] instead accepts a plain reward function of `(data_source, solution_str, ground_truth, extra_info)` [14], and verl's own pre-implemented GSM8K reward already returns 1 for a fully correct answer, 0.1 for correct format with a wrong answer, and 0 otherwise [14] — the same three-way rule ReFT uses [1] — making verl the closer fit for the RL stage, though the warm-up SFT phase and the KL-in-reward wiring (off by default, see Cost) still need to be added by hand.

## How it works

Warm-up on (x, e, y) with an SFT loss, then loop: sample a mini-batch of questions, roll out a CoT on-policy under the current policy, score it with the rule-based reward net of a KL penalty, compute advantages with GAE, and take PPO-clipped policy/value updates [1].

**Warm-up loss (Eq. 1)** [1]:

$$ L_{SFT}(\theta) = -\mathbb{E}_{e \sim D}\left[\sum_{t=1}^{L} \log \pi_\theta(a_t \mid s_t)\right] $$

Standard next-token cross-entropy over the annotated CoT $e = [a_1, \dots, a_{L-1}, a_L = \texttt{<eos>}]$, where $s_t$ is the question plus tokens generated so far [1].

**Reward** [1]: 0 for every non-terminal action; at the terminal token, a rule compares the extracted answer to the ground truth $y$:

$$ r(s_t, a_t, s_{t+1}) = \begin{cases} 1, & \text{EXTRACT}(s_{t+1}) = y \\ 0.1, & \text{EXTRACT}(s_{t+1}) \neq \text{null}, \neq y \\ 0, & \text{EXTRACT}(s_{t+1}) = \text{null} \end{cases} $$

The 0.1 partial credit applies only on datasets with numeric answers, when an answer can be extracted but is wrong, to soften reward sparsity [1]. The total reward then subtracts a KL term from every token's reward, not from the loss:

$$ r_{total}(s_t, a_t, s_{t+1}) = r(s_t, a_t, s_{t+1}) - \beta\, \mathrm{KL}\!\left[\pi_\theta(\cdot|s_t), \pi_\theta^{(0)}(\cdot|s_t)\right] $$

where $\pi_\theta^{(0)}$ is the frozen policy right after warm-up [1]. This differs structurally from GRPO, which instead adds the KL as a separate term inside the loss rather than folding it into the reward [7].

**GAE and returns** [1]. The paper writes the TD residual as $\delta_{t'} = -V_\phi(s_{t'}) + r_{total}(s_{t'}, a_{t'}, s_{t'+1}) + \gamma V_\phi(s_{t'+1})$ with terminal value $V_\phi(s_{L+1}) := 0$, then the advantage and return:

$$ \hat{A}_t = \sum_{l=0}^{L-t} (\gamma\lambda)^l \delta_{t+l}, \qquad \hat{R}_t = \hat{A}_t + V_\phi(s_t) $$

The paper's own naming is the reverse of the common convention: it defines $\lambda \in (0,1]$ as the discount factor for rewards and $\gamma \in [0,1]$ as the discount factor for TD [1] — the opposite of trl's config fields, where `gamma` is the reward discount and `lam` is the GAE discount [15]. Numerically the settings still line up once the swap is accounted for: the paper's reward-discount is 1 and its TD-discount is 0.95 [1], matching trl's `gamma=1.0`/`lam=0.95` defaults by value, not by name [15].

**PPO-clipped policy and value losses (Eq. 2)** [1]:

$$ L_{policy}(\theta) = -\mathbb{E}_{e \sim \pi_{\theta_{old}}}\left[\min\!\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t,\ \mathrm{clip}\!\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\right] $$

$$ L_{value}(\phi) = \mathbb{E}_{e \sim \pi_{\theta_{old}}}\left[\tfrac{1}{2}\max\!\left((V_\phi(s_t) - \hat{R}_t)^2,\ \left(\mathrm{clip}(\hat{R}_t - V_\phi(s_t), \hat{A}_t - \epsilon, \hat{A}_t + \epsilon)\right)^2\right)\right] $$

$$ L_{RL}(\theta, \phi) = L_{policy} + \alpha\, L_{value} \qquad (\text{Eq. 2}) $$

$\pi_{\theta_{old}}, V_{\phi_{old}}$ are the snapshots used for sampling and for computing $\hat{A}_t, \hat{R}_t$ at the start of each RL step, held fixed across $U$ inner optimization updates on that batch (Algorithm 1) [1]; $\alpha$ weights the value loss against the policy loss, and $\epsilon$ is the shared clip range applied to both the probability ratio and the value-estimate change [1].

**Worked micro-example.** Take a 2-token completion ($L=2$) that ends correctly, with GAE settings from the paper ($\gamma=0.95$ TD discount, $\lambda=1$ reward discount [1]) and value estimates $V_\phi(s_1)=0.3$, $V_\phi(s_2)=0.6$, $V_\phi(s_3)=0$ (terminal, by definition [1]). Only the terminal action gets a nonzero rule reward, $r(s_2,a_2,s_3)=1$; suppose the KL term nets out to 0 at both steps for simplicity, so $r_{total,1}=0$, $r_{total,2}=1$. TD residuals: $\delta_1 = -0.3 + 0 + 0.95(0.6) = 0.27$; $\delta_2 = -0.6 + 1 + 0.95(0) = 0.4$. Advantages: $\hat{A}_2 = \delta_2 = 0.4$; $\hat{A}_1 = \delta_1 + (\gamma\lambda)\delta_2 = 0.27 + 0.95(0.4) = 0.65$. Returns: $\hat{R}_1 = \hat{A}_1 + V_\phi(s_1) = 0.95$; $\hat{R}_2 = \hat{A}_2 + V_\phi(s_2) = 1.0$. Both tokens end up with positive advantage even though only the last token carried a nonzero reward, because GAE propagates the terminal reward backward through the value estimates [1].

## Cost

**Theory, from the method's own math**: two training stages instead of one. The RL stage needs, per step, one on-policy generation pass to sample the CoT, then $U$ (paper: 2 [1]) inner optimization passes over that batch for both $\theta$ and $\phi$ — more forward/backward work per environment step than plain SFT, which the paper's own Limitations section names directly: ReFT optimizes a non-differentiable objective and requires exploration of the generation space, so it needs more epochs to converge than SFT [1]. Memory holds three models at once: the policy $\pi_\theta$ being trained, a value head $V_\phi$ attached to a copy of the policy's backbone (following Ziegler et al., a linear head on the policy's own hidden states, not a separate network) [1], and a frozen reference copy $\pi_\theta^{(0)}$ used only for the KL term, forward passes only [1]. No separate reward model is held, since the reward is a rule [1] — the naive reading that a fourth model is needed does not hold here.

**In practice, per framework**:
- trl [8][12][13]: `PPOTrainer` lives only in the experimental namespace, with a runtime warning unless `TRL_EXPERIMENTAL_SILENCE=1` is set, and the module's own docstring states features there may change or be removed without deprecation [12]. Its `PPOConfig` defaults `num_ppo_epochs=4` (trl's analogue of the paper's $U$) [15], and its `PPOTrainer.__init__` requires an actual `reward_model: PreTrainedModel` and a `value_model: PreTrainedModel`, auto-creating the reference model from the policy if none is passed [13] — reproducing ReFT's rule-based reward means wrapping the rule as a fake scoring model rather than a plain function, an extra engineering step this framework does not remove.
- verl [9][14]: the stable PPO trainer/recipe [9] accepts a plain reward function keyed on `data_source, solution_str, ground_truth, extra_info`, with a pre-implemented GSM8K version already matching ReFT's 1 / 0.1 / 0 rule [14], so no reward-model wrapper is needed. verl's default `use_kl_in_reward` is `False` (`trainer/config/ppo_trainer.yaml`, read 2026-08-02) [16], i.e. its shipped default computes KL as a separate loss term rather than folding it into the reward the way ReFT's reward function does [1] — reproducing ReFT means explicitly setting `use_kl_in_reward: True` rather than relying on the default.

## How to use it

- Data prep: build (question, CoT, answer) triples for warm-up (the paper used GPT-3.5-turbo few-shot prompting to generate both N-CoT and P-CoT annotations from question/answer pairs [1]) and strip to (question, answer) pairs for the RL stage, since RL-stage CoTs are sampled fresh from the current policy [1].
- Reward convention: exact match on the extracted final answer scores 1, an extracted-but-wrong numeric answer scores 0.1 (numeric datasets only), extraction failure scores 0, and every non-terminal token scores 0 before the KL subtraction [1].
- Knobs, paper values as anchors versus the two frameworks' PPO defaults (naming differences noted; "not stated" = checked and absent from that source, "not checked" = not verified here):

| knob | paper [1] | trl `PPOConfig` default [15] | verl PPO defaults [16][17][18] |
| --- | --- | --- | --- |
| reward discount (paper's $\lambda$†; trl `gamma`) | 1 | 1.0 | `gamma` 1.0 |
| TD/GAE discount (paper's $\gamma$†; trl `lam`) | 0.95 | 0.95 | `lam` 1.0 (differs from paper's 0.95) |
| clip range $\epsilon$ | 0.2 | `cliprange` 0.2 | `clip_ratio` 0.2 |
| value-loss clip range | 0.2 (shared with $\epsilon$) | `cliprange_value` 0.2 | `cliprange_value` 0.5 (differs) |
| value loss coefficient $\alpha$ | 5 | `vf_coef` 0.1 (differs) | not applicable — verl's critic is a separate model with its own optimizer, not a weighted loss term [18] |
| updates per rollout batch $U$ | 2 | `num_ppo_epochs` 4 | `ppo_epochs` 1 |
| RL-stage learning rate (actor) | 3e-7 | `learning_rate` 3e-6 (trl docs note this replaces a 5e-5 default) [15] | `actor.optim.lr` 1e-6 |
| critic learning rate | shares the policy's optimizer (shared backbone) | n/a (weighted loss, not separate optimizer) | `critic.optim.lr` 1e-5 (separate optimizer, architecturally unlike the paper's shared backbone) |
| KL coefficient $\beta$ | 0.01 (P-CoT) / 0.05 (N-CoT) | `kl_coef` 0.05 | `kl_coef` 0.001; KL folded into reward only if `use_kl_in_reward` is set True (default False) [16] |

  † The paper names its reward-discount symbol $\lambda$ and its TD-discount symbol $\gamma$, the reverse of trl's `gamma`/`lam` fields; the two "paper" column values above are keyed to trl's fields by value, not by the paper's own Greek letters [1][15].
- Trade-offs the paper names directly: batch size in the RL stage was set to 32, smaller than the warm-up SFT batch of 48, "due to extra memory consumption of the value model" [1]; a larger learning rate speeds convergence but risks policy instability and collapse, while a larger batch size costs more compute [1].

## While it runs

- Signals and their healthy shapes: this card found no first-hand trl or verl logging guide naming metrics specific to a ReFT-style run (neither framework ships ReFT), so no framework-logged-signal is claimed here.
- Published reference runs: the paper's own Table 2 is the reference curve-equivalent for this method — on CodeLLAMA-7B, ReFT's average N-CoT/P-CoT accuracy of 59.31/75.43 versus SFT's 52.56/67.96, Offline-ST's 55.04/71.41, and Online-ST's 53.40/71.37 is the number that shows PPO exploration beats both SFT and same-data self-training, not just SFT alone [1]. Table 4 shows the gap widens further with majority voting and reward-model reranking on GSM8K: reranked CodeLLAMA+ReFT reaches 66.0 (N-CoT) / 81.2 (P-CoT), and the paper states specifically that its best P-CoT reranked result of 81.2 surpasses GPT-3.5-turbo's 78.0 on this benchmark [1]; on N-CoT, ReFT's 66.0 does not surpass GPT-3.5-turbo's reported 75.3 [1].
- Degeneracies and defaults: the paper's Reward Hacking analysis is a documented degeneracy, not a hypothesized one — on MathQA_MCQ N-CoT, where the answer space is limited to {A,B,C,D,E}, a chain of reasoning can reach the correct letter by chance despite a wrong intermediate step, giving a false-positive reward; the paper shows this concretely for one example and names it as a risk specifically when the answer space is small [1]. Framework KL defaults diverge from the paper's KL-in-reward design: verl's `use_kl_in_reward` defaults to `False`, so running verl's PPO recipe unmodified applies KL as a separate loss term rather than the paper's reward-level subtraction [16] (see How to use it).
- Named successors: the paper's Limitations section proposes a more detailed or process-based reward function as future work to mitigate reward hacking, without naming a specific method [1]. Separately, in its own Future Work section (not Limitations), the paper cites Lightman et al.'s finding that a well-trained process-based reward model can significantly enhance performance, and names implementing process-based rewards in RL training as worthwhile future work [1]. This card did not verify whether a paper explicitly named "successor to ReFT" exists beyond this proposal.
- Known failure modes, from the paper's own Limitations section [1]: (1) Training Efficiency — ReFT needs more epochs than SFT to converge because it optimizes a non-differentiable objective and must explore the generation space, and larger learning rates that would speed this up risk instability and collapse [1]. (2) Reward Hacking — the reward depends solely on the final extracted answer, which the MathQA_MCQ experiment shows can be gamed when the answer space is small [1]. This card also checked the paper's Appendix D for a first-hand DPO/IPO failure mode: the authors report DPO and IPO, adapted from the trl framework and trained on preference data sampled from the warm-up checkpoint, performed only on par with the Offline Self-Training baseline on GSM8K, attributed to being offline (no further CoT exploration) and to preference data that is a poor fit for reward modeling since easy questions may have no wrong samples and hard questions no right ones [1]. No trl or verl GitHub issue was searched for ReFT-specific maintainer replies, since neither framework implements the method by name; that search was not performed and no claim is made about it.
- What the gain is - and is not: the paper's own results (Table 2-4) show ReFT and its combinations with voting/reranking outperform SFT and both self-training baselines on value accuracy across GSM8K, SVAMP, and MathQA [1]; the paper does not claim ReFT adds reasoning capability beyond what is reachable by resampling CoTs the base+warm-up policy could already produce — its stated mechanism is exploring more of the CoT space per question, not introducing new problem-solving skill [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. ReadTheDocs pages ([9], [10], [14]) take no revision parameter and are unpinned, moving reads as of their fetch date. GitHub source-file claims ([8], [12], [13], [15], [16], [17], [18]) each carry the exact commit SHA read; that pin covers what loaded at that commit only, not the branch's current state.

[1] Luong, Zhang, Jie, Sun, Jin, Li, "ReFT: Reasoning with Reinforced Fine-Tuning", 2024. https://arxiv.org/abs/2401.08967 - defines ReFT: two-stage procedure, reward function, KL-in-reward, GAE, PPO losses, Algorithm 1, Tables 1-5, Limitations, Appendix D (DPO/IPO attempt). Fetched 2026-08-02 (PDF, extracted with pdftotext -layout after the ar5iv HTML mirror returned an error page).

[2] Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, the parent method. Fetched 2026-08-02.

[3] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - checked full text for a citation to ReFT/Luong et al.; none found. Fetched 2026-08-02 (PDF).

[4] DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025. https://arxiv.org/abs/2501.12948 - checked full text for a citation to ReFT/Luong et al.; none found. Fetched 2026-08-02 (PDF).

[5] Lambert et al., "Tulu 3: Pushing Frontiers in Open Language Model Post-Training", 2024. https://arxiv.org/abs/2411.15124 - checked full text for a citation to ReFT/Luong et al.; none found. Fetched 2026-08-02 (PDF).

[6] Rafailov, Sharma, Mitchell, Manning, Ermon, Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - DPO, the offline nearest alternative; eliminates the need for sampling from the LM during fine-tuning, uses a closed-form optimal policy and a classification loss. Fetched 2026-08-02.

[7] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO, the nearest online neighbor: group-relative advantage without a value network, KL added to the loss rather than the reward. Same paper as [3]; fetched 2026-08-02.

[8] trl repository, `trl/__init__.py`, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. https://github.com/huggingface/trl/blob/2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0/trl/__init__.py - top-level exports (DPOTrainer, GRPOTrainer, KTOTrainer, RewardTrainer, RLOOTrainer, SFTTrainer); no PPOTrainer at top level. Read 2026-08-09.

[9] verl documentation, PPO algorithm page. https://verl.readthedocs.io/en/latest/algo/ppo.html - verl's stable PPO trainer. Fetched 2026-08-02; this is an unpinned `latest`-branch doc build, not a commit-pinned source.

[10] verl documentation, algorithms index/sidebar. https://verl.readthedocs.io/en/latest/ - lists baseline, dapo, dppo, dro, entropy, gpg, grpo, opd, opo, otb, ppo, rollout_corr, spin, sppo; no "reft" entry. Fetched 2026-08-02; unpinned `latest`-branch doc build.

[11] lqtrung1998/mwp_ReFT, the paper's own code release. https://github.com/lqtrung1998/mwp_ReFT - one-off research scripts (train_sft_model.py, train_rl_reft.py, train_rl_sl.py, sampling.py), not a reusable trainer library. README read 2026-08-02 at the repository's default branch head; no commit pin captured.

[12] trl repository, `trl/experimental/__init__.py`, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. https://github.com/huggingface/trl/blob/2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0/trl/experimental/__init__.py - module docstring: contents are unstable/incubating and may change or be removed without deprecation; emits a warning unless `TRL_EXPERIMENTAL_SILENCE=1`. Read 2026-08-09.

[13] trl repository, `trl/experimental/ppo/ppo_trainer.py`, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. https://github.com/huggingface/trl/blob/2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0/trl/experimental/ppo/ppo_trainer.py - `PPOTrainer.__init__` signature: requires `model`, optional `ref_model: PreTrainedModel | None` (auto-created via `create_reference_model` if absent), required `reward_model: PreTrainedModel`, required `value_model: PreTrainedModel`. Read 2026-08-09.

[14] verl documentation, "Implement Reward Function for Dataset". https://verl.readthedocs.io/en/latest/preparation/reward_function.html - reward function signature `(data_source, solution_str, ground_truth, extra_info)`; pre-implemented GSM8K reward: 1 point exact match, 0.1 correct format/wrong answer, 0 incorrect format. Fetched 2026-08-02; unpinned `latest`-branch doc build.

[15] trl repository, `trl/experimental/ppo/ppo_config.py`, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. https://github.com/huggingface/trl/blob/2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0/trl/experimental/ppo/ppo_config.py - `PPOConfig` defaults: `learning_rate` 3e-6 (docstring notes this replaces an earlier 5e-5 default), `num_ppo_epochs` 4, `kl_coef` 0.05, `cliprange` 0.2, `cliprange_value` 0.2, `gamma` 1.0, `lam` 0.95, `vf_coef` 0.1. Read 2026-08-09.

[16] verl repository, `verl/trainer/config/ppo_trainer.yaml`, commit `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71`. https://github.com/volcengine/verl/blob/4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71/verl/trainer/config/ppo_trainer.yaml - `gamma: 1.0`, `lam: 1.0`, `adv_estimator: gae`, `use_kl_in_reward: False`, `kl_penalty: kl`, `kl_coef: 0.001`. Read 2026-08-09.

[17] verl repository, `verl/trainer/config/actor/actor.yaml`, commit `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71`. https://github.com/volcengine/verl/blob/4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71/verl/trainer/config/actor/actor.yaml - `ppo_mini_batch_size: 256`, `clip_ratio: 0.2`, `ppo_epochs: 1`, `lr: 1e-6`. Read 2026-08-09.

[18] verl repository, `verl/trainer/config/critic/critic.yaml`, commit `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71`. https://github.com/volcengine/verl/blob/4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71/verl/trainer/config/critic/critic.yaml - `lr: 1e-5` (separate critic optimizer, no `vf_coef`-equivalent weighting field), `cliprange_value: 0.5`. Read 2026-08-09.
