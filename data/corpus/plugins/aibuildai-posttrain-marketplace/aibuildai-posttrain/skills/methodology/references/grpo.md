# GRPO

PPO with the value network deleted: sample a group of G completions per prompt and use the group's own reward mean and std as the baseline.

**GRPO** (Group Relative Policy Optimization) is an online-RL method for fine-tuning an LLM policy against a per-response reward, introduced by the DeepSeekMath paper as a variant of PPO that cuts its memory cost by removing the value network [1]. PPO, its parent [2], trains a value network alongside the policy - a second model of comparable size - whose per-token return estimate serves as the baseline [1]. GRPO replaces that with a group baseline: for each prompt it samples G completions and normalizes their rewards by the group mean and std [1]. The paper is at https://arxiv.org/abs/2402.03300 [1]. Two reasons beyond memory: with an outcome reward only the last token gets a reward score, so training a per-token value function is hard; and a group-relative baseline fits the comparative nature of reward models, which are trained on comparisons between outputs to the same question [1].

The method carried DeepSeek's reasoning line: DeepSeek-R1 adopted it to reduce RL training costs [3], and R1-Zero - pure GRPO RL on the base model, no SFT - took AIME 2024 pass@1 from 15.6% to 71.0% [3]; Qwen3 likewise used GRPO for its reasoning-RL stage [4]. In the originating paper, RL with GRPO lifted DeepSeekMath-Instruct 7B from 82.9% to 88.2% on GSM8K and from 46.8% to 51.7% on MATH [1]. Lineage in one line: PPO (2017 [2]) -> GRPO (DeepSeekMath, 2024 [1]) -> adopted by DeepSeek-R1 and Qwen3 (2025 [3][4]) -> bias-fixing successors DAPO and Dr. GRPO (2025 [5][6]). The paper was confirmed as the defining source by an exact-title arXiv search returning DeepSeekMath as the top-cited match for "Group Relative Policy Optimization" among the candidates considered.

**When to pick it**: online RL when every sampled completion can be scored with a scalar reward (a programmable function or a reward model) and memory is tight - the group baseline replaces PPO's value network [1]. Prefer PPO [2] when a learned per-token critic earns its extra model. Prefer DPO [7] when you have preference pairs and no way to score fresh samples: DPO is an offline method: it fits on a static dataset of paired preferred/dispreferred responses to a prompt, optimizing the log-likelihood margin between them directly, with no reward model and no sampling step in the loop [8]. Nearest online neighbor is RLOO [9]: its baseline is the leave-one-out mean of all other samples in the same batch [10], its default single-step setting collapses the probability ratio to 1 so training reduces to standard REINFORCE [11] with the clip term inactive [10], and its KL is used purely for reward shaping (folded into the scalar reward under no-gradient) while GRPO adds an explicit, differentiable KL term to its objective [10].

**Variant of**: PPO [2].

**Data it needs**: prompts from the distribution you care about (the paper's RL stage used around 144K chain-of-thought questions related to GSM8K and MATH from the SFT data [1]), plus a reward source. On-policy: completions are sampled fresh from the current policy each step; a pre-collected response set cannot be used [1].

**Extra models**: never a value network. The objective carries a KL term with coefficient $\beta$ (the paper ran $\beta = 0.04$ [1]), which needs a frozen reference model - a copy of the policy you started RL from, forward passes only. Setting $\beta = 0$ removes both the term and the model; trl's default does exactly that [12], verl's GRPO recipe keeps the term on [13]. Details in Cost.

**Shipped by**: trl (`GRPOTrainer`) [12], verl (`algorithm.adv_estimator: grpo`) [13].

## How it works

Each step: sample a group of G completions per prompt from the current policy snapshot $\pi_{\theta_{old}}$; score each completion; turn scores into group-relative advantages; update the policy on the clipped objective [1].

**The objective** [1]:

$$ J_{GRPO}(\theta) = \mathbb{E}_{\,q \sim P(Q),\ \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)}\; \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\!\left[ \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}\,\hat{A}_{i,t},\; \operatorname{clip}\!\left( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}, \,1-\varepsilon,\,1+\varepsilon\right)\hat{A}_{i,t} \right]
- \beta\, D_{KL}\!\left[\pi_\theta \,\|\, \pi_{ref}\right] \right\} $$

The fraction $\pi_\theta/\pi_{\theta_{old}}$ is the per-token probability ratio between the policy being trained and the snapshot that generated the group; $\operatorname{clip}$ caps how far one update can move that ratio, and $\varepsilon$ is the clip range inherited from PPO's clipped surrogate objective [2]. Two structural facts sit inside the formula: every completion is length-normalized by $1/|o_i|$, and the $\beta\, D_{KL}$ penalty sits inside the per-token braces, applied at every token [1].

**Advantage - outcome supervision** [1]; one scalar reward per completion:

$$ \hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})} \quad \text{for every token } t \text{ of } o_i, \qquad \mathbf{r} = \{r_1, \dots, r_G\} $$

The scalar reward becomes a token-level signal by giving every token of completion i the same normalized advantage. Worked example, G = 4 with rewards $\{1, 0, 0, 0\}$: mean $0.25$, population std $\sqrt{0.75/4} \approx 0.433$; the correct completion gets advantage $(1-0.25)/0.433 \approx +1.73$ on every token, each wrong one gets $\approx -0.58$, and the four advantages sum to zero - the group is its own zero-mean baseline. (With sample std the numbers become $+1.5$ / $-0.5$; which convention an implementation uses is not checked here.) Rewards $\{1,1,1,1\}$ or $\{0,0,0,0\}$ give std $= 0$ - the degenerate case in While it runs. The std division is itself a method-level knob in practice: trl `scale_rewards` (default `"group"`) [12], verl `algorithm.norm_adv_by_std_in_grpo` (`False` gives Dr. GRPO's mean-only version [6]) [13] - see While it runs for the documented bias.

**Advantage - process supervision** [1]: a process reward model scores each reasoning step; all step rewards in the group are normalized together by the same mean/std recipe, and a token's advantage is the sum of the normalized rewards of every step that ends at or after it: $\hat{A}_{i,t} = \sum_{\,index(j)\, \ge\, t} \tilde{r}_i^{\,index(j)}$, where $index(j)$ is the end-token position of step $j$. Both trl and verl ship outcome supervision [12][13]; the rest of this card assumes it.

**KL regularization** is part of the method's definition, not a framework choice: GRPO regularizes by adding the KL divergence between the trained policy and the reference policy directly to the loss, instead of subtracting a per-token KL penalty from the reward as PPO does [1]. The KL uses the unbiased estimator of Schulman [14] ([1]), computed per token and always non-negative:

$$ D_{KL}\!\left[\pi_\theta \,\|\, \pi_{ref}\right] = \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}
- \log \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}
- 1 $$

$\pi_{ref}$ is a frozen copy of the model you started RL from (the SFT checkpoint) [1]. In the paper's iterative GRPO variant, the reference is not frozen forever: each iteration generates a new training set for the reward model from the current policy's samples, retrains the old reward model with a replay mechanism that mixes in 10% of historical data, and then resets $\pi_{ref}$ to the current policy before continuing training [1]. Neither framework's docs describe that iterative loop; both present a static reference model [12][13].

**Loss aggregation in practice is not the paper's**: the paper's $1/|o_i|$ per-completion normalization is exactly what later work found length-biased [5][6]. trl's `loss_type` defaults to `"dapo"` (token-level aggregation [5]; its docs describe the original `"grpo"` aggregation as averaging losses by token within a sample and then across samples, giving each sample equal weight [12]); verl's `loss_agg_mode` defaults to `"token-mean"`, and its docs note that the original GRPO paper's sample-level loss (`"seq-mean-token-mean"`) may be unstable in long-CoT scenarios [13]. Know which loss you actually ran (see While it runs).

## Cost

**Theory, from the method's own math:**

- Time: each prompt needs G completions, so generated tokens per prompt scale with G times average completion length. (Whether generation dominates wall-clock depends on lengths, batching, and hardware; the theory-level quantity is the token count.) Training passes run over all G completions, for the policy only: PPO's value-network forward and backward are gone [1]. With $\beta > 0$, add one reference-model forward pass over the same tokens - no gradients, no optimizer state.
- Memory: one trained model less than PPO [1]. Each trained model costs weights + gradients + optimizer state - Adam keeps a first-moment and a second-moment estimate per parameter [15], about two extra weight-sized tensors - so PPO's critic roughly doubles the trained footprint, and the paper describes the value function in PPO as typically another model of comparable size to the policy model [1]. GRPO with $\beta = 0$ trains and holds the actor alone, plus the reward model only if the reward is a model.
- A naive reading of the objective says a whole group's activations must sit in memory at once (x G). The two framework bullets below [12][16] show implementations do not do that: the group has to coexist only as G reward scalars at advantage-normalization time - bytes, not activations.

**In practice, per framework:**

- trl `GRPOTrainer` [12]: `beta` defaults to `0.0`, and its docs state that if 0.0 (the default) the reference model is not loaded [12], reducing memory; with `beta > 0` the frozen reference model is loaded. Generation is bounded by the per-device generation batch size, or handed to vLLM [17] (`use_vllm`); the training pass is bounded by the usual per-device batch size and gradient accumulation. `num_generations` defaults to 8 [12].
- verl [13][16]: the GRPO recipe tells you to set `actor.use_kl_loss: True` - its default is `False`, so flipping only `adv_estimator: grpo` runs without the KL term - with `kl_loss_coef` default 0.001 and `kl_loss_type` choosing the estimator (k1 / abs / mse-k2 / low_var_kl-k3 / full) [13]. Forward/backward memory is bounded by the batch keys, not by the group size: `ppo_mini_batch_size` splits a rollout batch into gradient-update sub-batches, and `ppo_micro_batch_size_per_gpu` bounds one forward pass, trading speed for GPU memory (log-prob recomputation has its own `log_prob_micro_batch_size_per_gpu`) [16]. Peak activation memory follows the micro-batch setting, not G.
- Do not carry one framework's behavior to the other: the same method name runs with a reference model under verl's recipe [13] and without one under trl's default [12].

## How to use it

- Prompts: from the target task distribution; difficulty must roughly match the current policy, because uniform-reward groups teach nothing (see While it runs). The paper's RL stage used ~144K questions [1].
- Reward: one scalar per completion. In trl [12], each reward function is called with keyword arguments `prompts`, `completions`, `completion_ids` plus every extra dataset column via `**kwargs`, and must return a list of floats [12]; functions may be `async`; a function may return `None` for samples it does not apply to, and that reward function is excluded from the reward calculation for that sample [12]; several functions combine as a sum, or as a weighted sum if `reward_weights` is provided [12]. A reward model is passed as a model-ID string and loaded with `num_labels=1`; trl's docs state that only sequence classification models are supported as `PreTrainedModel` reward models [12].
- Key knobs, with each source's own value (paper values are anchors, not defaults - note how far the framework defaults sit from them). "not stated" = the source was checked and does not give the value; "not checked" = this card did not verify that source's key:

| knob | trl default [12] | verl default [13] | paper [1] |
| --- | --- | --- | --- |
| group size G | `num_generations` 8 | `rollout.n` 1 (see While it runs) | 64 |
| KL coefficient $\beta$ | `beta` 0.0 (KL off) | `kl_loss_coef` 0.001 (recipe: `use_kl_loss: True`) | 0.04 |
| clip range $\varepsilon$ | `epsilon` 0.2 (`epsilon_high` for an asymmetric upper bound) | `clip_ratio` 0.2 | not stated |
| learning rate | `1e-6` | not stated | `1e-6` |
| updates per rollout batch ($\mu$) | `num_iterations` 1 | not checked | one update per exploration stage |
| max completion length | `max_completion_length` 512 | not checked | 1024 |
| sampling temperature | `temperature` 1.0, `top_p` 1.0 | not checked | not stated |

  Three readings of that table: the learning rate is `1e-6` in both the paper and trl's default [1][12], well below what SFT stages typically run, so carrying an SFT learning rate into GRPO is a live mistake; at `num_iterations` 1 - both the trl default and the paper's setting - the ratio starts at 1 on the single update, so the clip seldom binds; and PPO's own design point was multiple epochs of minibatch updates from one rollout batch [2], so $\varepsilon$ starts mattering when you raise $\mu$ and reuse a rollout batch.
- Group size trade-off: larger G gives a steadier baseline and more rollout cost per prompt; at a fixed token budget it means fewer prompts per batch. The paper used 64 [1] against trl's default 8 [12].
- Install, launch, parallelism, and the rest of the framework surface belong to the framework cards (trl, verl); this card stops at the knobs that define the method.

## While it runs

- Signals while it trains, and their healthy shapes (trl publishes the rare first-hand guidance; the framework file carries the metric names and how to enable them): trl's logging guide states that `reward` is the primary objective, reflects the group-wise normalized rewards the policy is achieving, and should generally increase during successful training [18]; a collapse in entropy means the policy is becoming overconfident and deterministic, often too early, and can stall learning by reducing exploration [18], with typical per-token entropies of 2-10 nats for language models [12]; a high `clip_ratio` (or the finer-grained `clip_ratio/low_mean`, `clip_ratio/high_mean`, `clip_ratio/region_mean` for the standard, non-Liger loss) indicates how often updates are being constrained by GRPO's clipping mechanism, and a very high value can suggest the policy is trying to change too drastically - potentially due to large advantages, a learning rate that is too high, or an epsilon clipping range that is too restrictive [18]; a high `completions/clipped_ratio` suggests the model frequently generates completions cut off by `max_completion_length` [18]; with $\beta > 0$, the `kl` metric tracks divergence from the reference model and should be watched to ensure the policy does not stray too far, which can lead to instability [18]; per-reward-function means (`rewards/{name}/mean`) separate which behavior the model is learning versus struggling with when several rewards combine [18]; and the fraction of uniform-reward groups (trl logs it as `frac_reward_zero_std`, described as samples with a reward std of zero because all answers are equally correct or incorrect) measures the degeneracy two bullets below, live [12].
- Neither framework runs the paper's loss aggregation by default (trl `loss_type="dapo"` [12], verl `loss_agg_mode="token-mean"` [13]; both document why - length bias / long-CoT instability). Before comparing a run to the paper or to another framework's run, state which loss actually ran.
- KL defaults diverge three ways: paper 0.04 [1], verl recipe 0.001 [13], trl 0.0 [12]. trl's docs justify the zero default by noting that this choice is motivated by the observation that the KL divergence term is not always necessary [12] - but with $\beta = 0$ nothing pulls the policy toward the reference, so nothing bounds drift and reward hacking except the reward itself. Choose $\beta$ deliberately; do not inherit a default.
- A group of one has no baseline: verl's `rollout.n` defaults to 1 and its docs say to set it to a value larger than 1 for group sampling with GRPO [13].
- Uniform-reward groups: if all G rewards are equal the advantage is 0/0 - an implementation must guard that case (check yours) - and the prompt teaches nothing that step; near-uniform groups make the std tiny and inflate advantages. Dividing by std also carries a documented bias: trl's docs cite Dr. GRPO's finding that turning off std-based scaling removes a source of bias, at the cost that update magnitudes then depend directly on the raw reward scale [12][6] - controlled by trl `scale_rewards` / verl `norm_adv_by_std_in_grpo` (`False` = Dr. GRPO's fix [6]).
- Length bias is real enough that both frameworks changed their defaults for it. DAPO [5] found that the original GRPO sample-level loss gives each sample equal weight regardless of length and made token-level aggregation its fix (now trl's default loss [12]); its system, built on the verl framework, reports 50 points on AIME 2024 with a Qwen2.5-32B base model, versus 47 points for DeepSeek-R1-Zero-Qwen-32B using half the training steps [5]. Dr. GRPO [6] removes GRPO's length and std normalization terms, calling the result an unbiased optimization method that improves token efficiency while maintaining reasoning performance, and reports 43.3% accuracy on AIME 2024 with a 7B base model [6]; verl's docs describe setting `use_kl_loss: False` and `norm_adv_by_std_in_grpo: False` for the Dr. GRPO algorithm [13]. Other trl `loss_type` values (dr_grpo, sapo, cispo, vespo) are named here only as config values [12]; their defining papers are deliberately not covered on this card.
- Know what the gain is: the originating paper's own analysis, comparing Pass@K and Maj@K accuracy of the Instruct and RL models, finds that RL enhances Maj@K's performance but not Pass@K, and concludes this is because RL renders the output distribution more robust rather than enhancing the fundamental capabilities of the model [1] - GRPO reranks what the base model can already sample; do not expect it to add capability the base model cannot reach.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Framework docs are `main` / `latest` builds, unpinned and mutable; every default quoted above is a 2026-08-08 reading - re-check against the version you install.

[1] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO: objective and motivation, outcome/process supervision, iterative GRPO, hyperparameters, results, Maj@K/Pass@K analysis. Fetched 2026-08-08 (HTML full text at arxiv.org/html/2402.03300).

[2] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, the parent method: clipped surrogate objective, clip range ablation (Table 1), multiple epochs per rollout batch. Fetched 2026-08-08 (abstract page plus PDF full text; no arXiv HTML conversion exists for this paper).

[3] DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025. https://arxiv.org/abs/2501.12948 - GRPO adoption, R1-Zero AIME 2024 results. Fetched 2026-08-08 (HTML full text, v1).

[4] Qwen Team, "Qwen3 Technical Report", 2025. https://arxiv.org/abs/2505.09388 - GRPO in the reasoning-RL stage. Fetched 2026-08-08 (HTML full text).

[5] Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", 2025. https://arxiv.org/abs/2503.14476 - token-level policy gradient loss, AIME 2024 results, verl-based implementation. Fetched 2026-08-08 (HTML full text).

[6] Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective", 2025. https://arxiv.org/abs/2503.20783 - introduces Dr. GRPO, length and difficulty bias analysis. Fetched 2026-08-08 (HTML full text).

[7] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290. Fetched 2026-08-08 (abstract page).

[8] trl DPOTrainer documentation. https://huggingface.co/docs/trl/main/en/dpo_trainer. Fetched 2026-08-08.

[9] Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs", 2024. https://arxiv.org/abs/2402.14740. Fetched 2026-08-08 (abstract page).

[10] trl RLOOTrainer documentation. https://huggingface.co/docs/trl/main/en/rloo_trainer. Fetched 2026-08-08.

[11] Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", Machine Learning, 1992 - REINFORCE. Not fetched at check time; cited as the classic reference behind [10]'s description of RLOO reducing to standard REINFORCE.

[12] trl GRPOTrainer documentation. https://huggingface.co/docs/trl/main/en/grpo_trainer. Fetched 2026-08-08.

[13] verl GRPO documentation. https://verl.readthedocs.io/en/latest/algo/grpo.html. Fetched 2026-08-08.

[14] Schulman, "Approximating KL Divergence", 2020. http://joschu.net/blog/kl-approx.html - defines the k3 unbiased KL estimator, $k_3 = (r-1) - \log r$. Fetched directly 2026-08-08; the DeepSeekMath paper [1] independently states that its KL term is this unbiased estimator.

[15] Kingma and Ba, "Adam: A Method for Stochastic Optimization", 2015. https://arxiv.org/abs/1412.6980 - maintains running estimates of the first and second moment of the gradient per parameter, the two per-parameter buffers behind the optimizer-state sizing. Fetched 2026-08-08 (abstract page plus PDF full text).

[16] verl configuration reference. https://verl.readthedocs.io/en/latest/examples/config.html - batch-key meanings (`ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu`, `log_prob_micro_batch_size_per_gpu`). Fetched 2026-08-08.

[17] Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", 2023. https://arxiv.org/abs/2309.06180 - vLLM. Fetched 2026-08-08 (abstract page).

[18] trl logging guide (the GRPO crucial-values list in While it runs). https://huggingface.co/docs/trl/main/en/logging. Fetched 2026-08-08.
