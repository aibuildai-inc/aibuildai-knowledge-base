# SAPO

GRPO with the hard clip replaced by a sigmoid-shaped soft gate: near on-policy tokens keep their full gradient, and gradient weight decays smoothly (with a steeper decay for negative-advantage tokens) instead of being zeroed past a fixed ratio band.

**SAPO** (Soft Adaptive Policy Optimization) is an online-RL method for fine-tuning an LLM policy against a per-response reward, proposed by the Qwen team as a replacement for the hard clipping used in group-based policy optimization, aimed at token-level importance-ratio variance that is worsened in Mixture-of-Experts models [1]. The paper is at https://arxiv.org/abs/2511.20347 [1]. Its parent is **GRPO** [2]: SAPO keeps GRPO's group-relative, group-normalized advantage and its per-token, per-response objective structure, and swaps only the min/clip surrogate for a bounded sigmoid gate applied to the same token-level ratio [1]. The paper gives three reasons for the change: hard clipping forces a trade-off where tight clipping discards valid samples and loose clipping lets in noisy off-policy gradients; when a sequence has a few high-ratio tokens, a hard clip such as GSPO's sequence-level clip [3] suppresses the gradient for the many near-on-policy tokens in that sequence too, discarding their signal [1]; and negative-advantage updates raise the logits of many irrelevant tokens at once, making them a larger source of instability than positive updates, which the paper addresses with an asymmetric temperature per advantage sign [1].

The paper reports no numeric results table; its evidence is learning curves. In controlled math-reasoning RL from a Qwen3-30B-A3B-Base cold start, the paper states that GSPO and GRPO-R2 (GRPO with routing replay) show early-stage training collapse while SAPO sustains stable training and reaches higher final validation Pass@1 on AIME25, HMMT25, and BeyondAIME (Figure 4) [1]. A temperature ablation comparing $\tau_{neg} > \tau_{pos}$, $\tau_{neg} = \tau_{pos}$, and $\tau_{neg} < \tau_{pos}$ finds the first the most stable and the last the most unstable, which the paper attributes to negative-token gradients being the larger source of instability (Figure 5) [1]. The paper then trains the Qwen3-VL model family with SAPO across mixed text and multimodal tasks and model scales, reporting that SAPO gives steady gains over GSPO and GRPO-R2 under equal compute in a further learning-curve comparison (Figure 6) [1]; this is SAPO's only reported production adoption so far. Lineage in one line: GRPO (DeepSeekMath, 2024 [2]) -> GSPO (2025 [3]) -> SAPO (2025 [1]), which the paper shows reduces to a smooth, GSPO-like sequence-level gate under two stated approximations (small-step ratios and low intra-sequence log-ratio dispersion) [1].

**When to pick it**: online RL with a per-response scalar reward, in the same setting as GRPO, when training shows the instability from hard clipping that the paper targets - particularly high token-ratio variance in MoE policies - and a smooth, tunable trust region is wanted over a fixed clip band [1]. Its parent is GRPO [2], whose hard per-token clip it replaces. Its nearest online neighbor is GSPO [3], which performs sequence-level clipping and rewarding [3]; the SAPO paper argues that clip suppresses the gradient of every near-on-policy token in a sequence containing even a few off-policy ones [1], and SAPO is built to keep those tokens informative while behaving like a smoothed GSPO under the paper's stated approximations [1]. The nearest offline alternative is DPO [4], which trains on fixed preference pairs with no fresh sampling and no reward model, at the cost of not adapting to a changing policy the way on-policy group methods do [4].

**Variant of**: GRPO [2].

**Data it needs**: the same as GRPO - prompts from the target distribution, with G completions sampled fresh from the current policy each step and a scalar reward per completion (on-policy) [1][2]. The paper does not report the exact size of its RL prompt sets, only that the controlled experiments use math-reasoning queries from a Qwen3-30B-A3B-Base cold start and the production run uses a broad mixture of math, code, and logical-reasoning tasks with a fixed per-task sampling ratio per batch, training Qwen3-VL-30B-A3B [1].

**Extra models**: none required by the method's own definition - neither the reprinted GRPO objective nor the SAPO objective in the paper carries a KL term or reference-model dependency, so SAPO's formula needs no frozen reference model and no value network [1]. A reward source is still required for the per-response scalar $R_i$; the paper does not state whether its math/code/logic rewards come from a learned reward model or from rule-based verifiers [1]. In the shipping frameworks, the surrounding trainer's own KL option is orthogonal to the SAPO gate and defaults off: trl's GRPOTrainer `beta` defaults to `0.0` (no reference model loaded) regardless of `loss_type` [5], and verl's `use_kl_loss` defaults to `False` regardless of `policy_loss.loss_mode` [6]. See Cost for the detail.

**Shipped by**: trl (`GRPOTrainer` with `loss_type="sapo"`) [5], verl (`actor_rollout_ref.actor.policy_loss.loss_mode: sapo`, function `compute_policy_loss_sapo`) [7].

## How it works

Each step: sample a group of G completions per prompt from the current policy; compute group-normalized advantages exactly as in GRPO; weight each token's log-probability gradient by a sigmoid gate of its probability ratio, with the gate's decay rate set by the advantage's sign [1].

**The objective**, as printed in the paper [1]:

$$ J(\theta) = \mathbb{E}_{q \sim \mathcal{D},\ \{y_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} f^{i,t}\!\left(r_{i,t}(\theta)\right) \hat{A}_{i,t} \right] $$

where the gate is

$$ f^{i,t}(x) = \sigma\!\left(\tau_{i,t}(x-1)\right) \cdot \frac{4}{\tau_{i,t}}, \qquad \tau_{i,t} = \begin{cases} \tau_{pos}, & \hat{A}_{i,t} > 0 \\ \tau_{neg}, & \text{otherwise} \end{cases} $$

$r_{i,t}(\theta) = \pi_\theta(y_{i,t}|q,y_{i,<t}) / \pi_{\theta_{old}}(y_{i,t}|q,y_{i,<t})$ is the same per-token probability ratio used by GRPO, $\hat A_{i,t}$ is GRPO's group-normalized advantage - the reward $R_i$ standardized by the group's own mean and std, shared across all tokens of completion $i$ [1][2] - and $\sigma$ is the logistic sigmoid [1]. Differentiating gives a weighted policy gradient with gate weight

$$ w_{i,t}(\theta) = 4\,p_{i,t}(\theta)\big(1-p_{i,t}(\theta)\big), \qquad p_{i,t}(\theta) = \sigma\!\left(\tau_{i,t}\big(r_{i,t}(\theta)-1\big)\right) $$

which peaks at $w_{i,t}=1$ exactly at $r_{i,t}=1$ (on-policy) for any $\tau_{i,t}$, and decays smoothly and roughly exponentially as $r_{i,t}$ moves away from 1 - so the gate never zeros a token's gradient outright the way GRPO's hard clip does, and larger $\tau$ makes that decay faster [1]. The paper recommends the asymmetric setting $\tau_{neg} > \tau_{pos}$, using $\tau_{pos}=1.0,\ \tau_{neg}=1.05$ in its own experiments, so negative-advantage tokens are down-weighted faster than positive ones as their ratio drifts [1].

**Worked example**: with $\tau_{pos}=1.0$, a positive-advantage token at $r_{i,t}=1.5$ (50% more likely under the new policy) gets gate weight $p = \sigma(1.0\times0.5) \approx 0.622$, so $w = 4\times0.622\times0.378 \approx 0.940$ - most of its gradient survives. At $r_{i,t}=3.0$, $p=\sigma(2.0)\approx0.881$ and $w\approx4\times0.881\times0.119\approx0.420$ - attenuated but not zero, unlike GRPO's hard clip which would cap the ratio outright past $1+\varepsilon$ [2]. With $\tau_{neg}=1.05$ on a negative-advantage token at the same $r_{i,t}=1.5$, $p=\sigma(1.05\times0.5)=\sigma(0.525)\approx0.628$, $w\approx4\times0.628\times0.372\approx0.934$ - only marginally faster decay than $\tau_{pos}=1.0$ at this ratio, since the paper's recommended temperatures are close together; the asymmetry is a small, deliberate perturbation, not a large gap [1].

**The paper's own reprinted GRPO objective also omits a KL term**: the original DeepSeekMath paper regularizes GRPO by adding $\beta\,D_{KL}[\pi_\theta\Vert\pi_{ref}]$ directly to the loss, with $\beta$ its KL coefficient [2], but neither the GRPO objective (Eq. 1) nor the SAPO objective (Eq. 5) as printed in the SAPO paper carries any KL term [1]. The paper's Section 4 "unified surrogate" frames GRPO's hard token-clip and GSPO's hard sequence-clip as two other choices of the same gating function $f^{i,t}$, and shows that under two approximations - small-step ratios ($r_{i,t}\approx 1$) and low dispersion of token log-ratios within a sequence - SAPO's averaged token gate concentrates to a smooth, GSPO-like sequence-level gate $\mathrm{sech}^2\!\left(\tfrac{\tau}{2}\log s_i(\theta)\right)$, with an explicit second-order error bound in terms of that within-sequence variance [1].

## Cost

**Theory, from the method's own math:**

- Time: identical rollout and forward/backward structure to GRPO - G completions per prompt, one training pass over the policy per completion, no value network - because the gate replaces GRPO's `min`/`clip` with an elementwise sigmoid on the same per-token ratio, which is a negligible extra cost on top of computing that ratio [1][2].
- Memory: same as GRPO with no KL term - one trained policy, plus a reward model only if the reward source is model-based (paper does not state which it uses) [1]. Because the printed objective carries no reference model, a naive reading says SAPO needs no extra models at all; both shipping frameworks preserve that by defaulting their independent KL options off (see below), but a user who turns KL on for either framework adds a frozen reference model exactly as with GRPO [5][6].

**In practice, per framework:**

- trl `GRPOTrainer` [5]: `loss_type="sapo"` only changes which gate function scales the per-token loss; generation, batching, and the `beta` (KL) option behave as in the rest of `GRPOTrainer` - `beta` defaults to `0.0`, and the reference model is not loaded when it is zero [5]. `sapo_temperature_pos` defaults to `1.0` and `sapo_temperature_neg` to `1.05`, matching the paper's own values [1][5].
- verl [7][6]: setting `actor_rollout_ref.actor.policy_loss.loss_mode: sapo` selects `compute_policy_loss_sapo`, which reads `tau_pos` (default `1.0`) and `tau_neg` (default `1.05`) from the actor config - again matching the paper - and, read at commit `d33ddd7` of `verl-project/verl` (2026-08-09), the function hardcodes `loss_agg_mode="seq-mean-token-mean"` inside its own `agg_loss` call regardless of the actor's configured `loss_agg_mode`, so the aggregation mode set elsewhere in the config does not apply to this loss [7]. `use_kl_loss` is a separate actor-level flag, default `False`, unaffected by `policy_loss.loss_mode` [6].

## How to use it

- Prompts and rewards: identical requirements to GRPO - a prompt set and a scalar reward per sampled completion, computed fresh each step [1][2]. The paper's own runs use math-reasoning queries (controlled experiments) and a mixed math/code/logic set with fixed per-task sampling ratios (Qwen3-VL training), without stating the reward source or dataset size [1].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give the value; "not checked" = this card did not verify that source's key):

| knob | trl default [5] | verl default [7][6] | paper [1] |
| --- | --- | --- | --- |
| $\tau_{pos}$ | `sapo_temperature_pos` 1.0 | `tau_pos` 1.0 | 1.0 |
| $\tau_{neg}$ | `sapo_temperature_neg` 1.05 | `tau_neg` 1.05 | 1.05 |
| group size G | `num_generations` 8 | `rollout.n` 1 (must be raised) [8] | not stated |
| KL coefficient | `beta` 0.0 | `kl_loss_coef` 0.001 (only if `use_kl_loss: True`, default False) | not present in the paper's objective |
| mini-batches per rollout batch | not checked | not checked | 4 (controlled experiments), 2 (Qwen3-VL training) |
| learning rate | not checked | not checked | not stated |

  Both shipping frameworks' default temperatures reproduce the paper's own recommended values exactly, which is unusual for a knob table on this corpus and signals the paper's asymmetric-temperature recommendation was carried directly into both implementations [1][5][7][6].
- Trade-off: raising $\tau_{neg}$ further above $\tau_{pos}$ makes negative-token gradients decay faster, trading away some exploration/regularization signal from negative tokens for the stability the paper reports (Figure 5); collapsing $\tau_{neg}=\tau_{pos}$ removes the asymmetry the paper found necessary, and setting $\tau_{neg}<\tau_{pos}$ was the most unstable configuration it tested [1].

## While it runs

- Signals and their healthy shapes: the paper's own monitoring is limited to training reward and validation Pass@1 curves per benchmark (Figures 4-6); it does not name a distinct SAPO-specific logged statistic beyond these [1]. trl exposes standard `GRPOTrainer` metrics (e.g. `num_tokens`, `step_time`) alongside whichever `loss_type` is active, but its docs do not list a metric unique to `loss_type="sapo"` beyond the temperature config values themselves [5].
- Published reference runs: the paper's own Figure 4 (controlled math RL from a Qwen3-30B-A3B-Base cold start) and Figure 6 (Qwen3-VL-30B-A3B training) are the known-good curves to compare against; both are learning curves only, with no accompanying numeric table, and show SAPO avoiding the early-stage collapse the paper reports for GSPO and GRPO-R2 baselines under the same compute budget [1].
- Degeneracies and defaults: as with GRPO, a group of one gives the advantage no baseline - verl's `rollout.n` defaults to 1 and must be raised to use group-based methods including SAPO [8]. Setting $\tau_{neg} < \tau_{pos}$ is a documented degeneracy: the paper's own ablation shows it is the most unstable of the three temperature configurations tested [1].
- Named successors: none found in the paper's own text (arXiv:2511.20347, submitted 2025-11-25), which as a method proposed only nine months before this card was written has had no time to accumulate documented successors [1]. A GitHub code search for `sapo` in `verl-project/verl` was attempted two ways and neither could be completed: the web UI search page loaded logged out and returned zero results (`"logged_in":false`, `"result_count":0`), and the `api.github.com/search/code` endpoint returned an outright authentication-required error (`"message":"Requires authentication"`, status 401) rather than any result set. Neither attempt is evidence of absence; this search remains unresolved, not a completed "none found."
- Known failure modes: the paper's own text states that "all methods may ultimately exhibit signs of instability" including SAPO, differing only in how long stable learning is sustained before divergence sets in - it does not claim SAPO eliminates instability, only delays it relative to GSPO and GRPO-R2 in its reported runs [1]. No maintainer-reported failure mode was found: this card searched trl's and verl's SAPO-related documentation text fetched above for caveats and found none beyond the temperature-default note already covered in How to use it [5][7].
- What the gain is - and is not: the paper's own claims are about training stability and Pass@1 under a fixed compute budget, not about new capability - it reports SAPO sustaining coherent learning longer and reaching higher final Pass@1 than GSPO/GRPO-R2 baselines before those baselines collapse, not that SAPO raises a model's reachable ceiling beyond what stable GRPO/GSPO training could also reach given enough stability [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Framework docs and source are unpinned `main`/`latest` builds; every trl default above is a 2026-08-09 reading of the hosted `main` docs, and every verl default and code claim above is read at commit `d33ddd7140f44d392e0e10b48a8902651a1340f4` of `verl-project/verl` (fetched 2026-08-09) - re-check against the version you install.

[1] Gao, Zheng, Chen, Dang, Liu, Yu, Yang, Bai, Zhou, Lin (Qwen Team, Alibaba), "Soft Adaptive Policy Optimization", 2025. https://arxiv.org/abs/2511.20347 - defines SAPO: objective, gate function, asymmetric-temperature justification, GSPO/GRPO unified-gating-function connection, controlled and Qwen3-VL experiments. Fetched 2026-08-09 (PDF full text via `arxiv.org/pdf/2511.20347`).

[2] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO, the parent method: group-relative advantage, hard-clipped token-level objective, and its $\beta\,D_{KL}$ regularization term. Fetched 2026-08-09 (PDF full text via `arxiv.org/pdf/2402.03300`).

[3] Zheng, Liu, Li, Chen, Yu, Gao, Dang, Liu, Men, Yang et al., "Group Sequence Policy Optimization", 2025. https://arxiv.org/abs/2507.18071 - defines GSPO, cited here as SAPO's nearest online neighbor and as the source of the sequence-level hard-clip objective reprinted and analyzed in [1]. Fetched 2026-08-09 (abstract page metadata).

[4] Rafailov, Sharma, Mitchell, Ermon, Manning, Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - DPO, cited here as the nearest offline alternative. Fetched 2026-08-09 (abstract page metadata).

[5] trl GRPOTrainer documentation. https://huggingface.co/docs/trl/main/en/grpo_trainer - `loss_type="sapo"`, `sapo_temperature_pos`/`sapo_temperature_neg` defaults, `beta`/reference-model default, `num_generations` default. Fetched 2026-08-09.

[6] verl source, `verl/workers/config/actor.py`, class `ActorConfig`. https://github.com/verl-project/verl/blob/main/verl/workers/config/actor.py - `tau_pos`/`tau_neg` defaults, `use_kl_loss`/`kl_loss_coef` defaults, `policy_loss.loss_mode` default `"vanilla"`. Commit d33ddd7140f44d392e0e10b48a8902651a1340f4, fetched 2026-08-09.

[7] verl source, `verl/trainer/ppo/core_algos.py`, function `compute_policy_loss_sapo` (registered as `"sapo"` in `POLICY_LOSS_REGISTRY`). https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py - SAPO loss implementation, hardcoded `seq-mean-token-mean` aggregation. Commit d33ddd7140f44d392e0e10b48a8902651a1340f4, fetched 2026-08-09.

[8] verl GRPO documentation. https://verl.readthedocs.io/en/latest/algo/grpo.html - `actor_rollout.ref.rollout.n` default of 1 and the instruction to raise it for group sampling; the algorithm-specific `ActorConfig.rollout_n` field in [6] carries no default of its own (`MISSING`, asserted non-missing at runtime) and is overridden from this rollout config. Fetched 2026-08-09.
