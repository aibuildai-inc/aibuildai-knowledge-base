# RLHF

Fine-tune a supervised model in three stages: train a reward model on human rankings of its outputs, then run PPO against that reward model with a KL penalty back to the supervised model, so the policy is optimized for what labelers actually preferred rather than for the next-token loss alone.

**RLHF** (Reinforcement Learning from Human Feedback) is the three-step pipeline - supervised fine-tuning (SFT), reward model (RM) training on labeler comparisons, and reinforcement learning via PPO against the RM - that Ouyang et al. use to turn GPT-3 into InstructGPT [1]. The technique of using human preferences as a reward signal for RL was introduced by Christiano et al. for control and Atari tasks [2]. InstructGPT's own methodology section states it follows Ziegler et al. (2019) and Stiennon et al. (2020), who applied that recipe in the stylistic-continuation and summarization domains [1]; InstructGPT's Step 3 follows the same PPO-against-a-learned-reward-model structure as Stiennon et al.'s summarization work, including its KL-penalty computation [1][3]. The RL optimizer in Step 3 is PPO itself, unmodified in its clipped-surrogate mechanics [4]; RLHF is not a variant of PPO but a fine-tuning pipeline that uses PPO as its RL step and adds an RM and a KL-to-SFT penalty around it [1]. The paper is at https://arxiv.org/abs/2203.02155 [1]. It gives three reasons for building the pipeline instead of scaling supervised loss alone: language-modeling loss on internet text is not the same objective as following instructions helpfully and safely; human comparisons are cheaper to collect at scale than gold demonstrations for every prompt; and a learned reward model lets the policy be optimized for preferences that are easier for people to judge than to write out as a supervised target [1].

RLHF's confirmation for this card is a top-cited pick among several arXiv entries surfaced under the name: the search settled on the paper with the most citations that also matches the technique InstructGPT itself claims to use, "reinforcement learning from human feedback" [1], not a title-based match. Landmark adoption: Llama 2's paper states that RLHF "is a model training procedure that is applied to a fine-tuned language model to further align model behavior with human preferences and instruction following," collecting pairwise comparisons to train a reward model and then optimizing the policy against it [5]. In the originating paper, human labelers preferred outputs from the 175B PPO-ptx (InstructGPT) model over 175B GPT-3 outputs 85 ± 3% of the time, and over few-shot-prompted 175B GPT-3 71 ± 4% of the time; the 1.3B InstructGPT model's outputs were preferred to the 175B GPT-3's despite having over 100x fewer parameters [1]. Lineage in one line: RL from human preferences (Christiano et al., 2017 [2]) -> applied to language-model fine-tuning for summarization (Stiennon et al., 2020 [3]) -> RLHF for instruction-following (InstructGPT, 2022 [1]) -> adopted by Llama 2 and other chat models [5] -> offline alternative DPO (2023 [6]) and AI-feedback variant RLAIF (2023 [7]).

**When to pick it**: pick RLHF when you can collect human comparisons over model outputs (or already have a trained reward model) and can afford an online RL loop that samples fresh completions from the current policy every step [1]. Prefer DPO [6] when you have a fixed set of preference pairs and no reward model or RL infrastructure: DPO reparameterizes the reward implicitly and optimizes the policy directly on the pairs, offline, with no PPO loop and no separate reward or value model [6]. Prefer RLAIF when human comparison labels are the bottleneck: it keeps the same reward-model-plus-PPO structure but trains the reward model on AI-generated preference labels instead of human ones, and the paper's abstract reports comparable performance to RLHF across summarization, helpful dialogue generation, and harmless dialogue generation [7].

**Variant of**: not a variant of a single named post-training method - RLHF is the parent framework here, built by applying RL from human preferences [2] (carried into language-model fine-tuning for summarization by Stiennon et al. [3]) as a three-stage SFT-then-RM-then-PPO pipeline [1], with PPO itself supplying the RL step unchanged [4].

**Data it needs**: three separate datasets in the paper - an SFT set of about 13k prompts with labeler-written demonstrations; an RM set of about 33k prompts, each with K=4 to K=9 labeler-ranked model outputs, turned into $\binom{K}{2}$ pairwise comparisons per prompt; and a PPO set of about 31k prompts drawn only from the API, with no human labels attached, used purely to sample rollouts [1]. The RL step (Step 3) is on-policy: PPO samples completions from the current policy each iteration and cannot reuse a static response set [1]; the SFT and RM stages that precede it are ordinary offline supervised training on fixed datasets [1].

**Extra models**: three beyond the policy - the frozen SFT model used both as the PPO starting point and as the reference for the KL penalty; a reward model (6B parameters in the paper, shared across all policy sizes) that scores each rollout; and a PPO value network (also 6B, initialized from the reward model's weights) that provides the per-token baseline [1]. This is the heaviest model count on this corpus: unlike GRPO, which drops the value network, InstructGPT's RLHF keeps PPO's full critic in addition to the reward model. See Cost for what each model costs to hold.

**Shipped by**: trl ships a `PPOTrainer`, but only under the experimental namespace `trl.experimental.ppo.PPOTrainer`, marked experimental in trl's own trainer taxonomy [8]; the RM and SFT stages use trl's non-experimental `RewardTrainer` and `SFTTrainer` [9]. verl's PPO is selected with `algorithm.adv_estimator: gae` (generalized advantage estimation, the same estimator InstructGPT's Appendix C.4 describes using with no discount) and requires a separate critic model, which verl's own docs contrast with GRPO and RLOO as not needing one [10]. OpenRLHF, describing itself as the first Ray-plus-vLLM RLHF framework aimed at production use, ships the full pipeline as separate CLI entry points: `openrlhf.cli.train_sft`, `openrlhf.cli.train_rm`, and `openrlhf.cli.train_ppo_ray` [11].

## How it works

Three stages, run in sequence: fine-tune the pretrained model on demonstrations (SFT); train a reward model on labeler comparisons of that model's sampled outputs (RM); then run PPO to maximize the reward model's score minus a KL penalty back to the SFT model, optionally mixed with the pretraining loss [1].

**Step 2 - reward model loss**, a pairwise ranking loss over all $\binom{K}{2}$ comparisons from each K-way ranked prompt [1]:

$$ \operatorname{loss}(\theta) = -\frac{1}{\binom{K}{2}}\, E_{(x,y_w,y_l)\sim D}\left[ \log\!\big(\sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big)\big) \right] $$

$r_\theta(x,y)$ is the scalar reward-model score for prompt $x$ and completion $y$; $y_w$ is the completion the labeler preferred over $y_l$ in that pair; $D$ is the set of human comparisons. All $\binom{K}{2}$ pairs from one labeling task are treated as a single gradient update, not $\binom{K}{2}$ separate ones, because the paper found that training on them as independent, shuffled datapoints overfits after a single pass [1].

**Step 3 - PPO objective**, maximized over the RL policy $\pi_\phi^{RL}$ [1]:

$$ \operatorname{objective}(\phi) = E_{(x,y)\sim D_{\pi_\phi^{RL}}}\left[ r_\theta(x,y) - \beta\log\left(\frac{\pi_\phi^{RL}(y\mid x)}{\pi^{SFT}(y\mid x)}\right) \right] + \gamma\, E_{x\sim D_{pretrain}}\left[ \log\big(\pi_\phi^{RL}(x)\big) \right] $$

$r_\theta(x,y)$ is the trained reward model's score for the sampled completion; the $\beta \log(\pi_\phi^{RL}/\pi^{SFT})$ term is a per-episode KL penalty that keeps the policy near the SFT model it started from, computed the same way as in Stiennon et al. [1][3]; the last term mixes in the ordinary pretraining log-likelihood loss over $D_{pretrain}$, weighted by $\gamma$, to offset regressions on public NLP benchmarks the paper observed without it [1]. Setting $\gamma=0$ recovers the paper's plain "PPO" variant; the paper's headline InstructGPT models use $\gamma>0$ and are called "PPO-ptx" [1]. Inside this objective, the token-level advantage that PPO's clipped surrogate actually optimizes is computed with generalized advantage estimation over the reward-minus-KL signal, using the value network as baseline, exactly as in the parent PPO algorithm [1][4].

**Worked example** for the RM loss with K=3 ranked outputs $y_1 \succ y_2 \succ y_3$ for one prompt: $\binom{3}{2}=3$ comparisons are formed - $(y_1,y_2)$, $(y_1,y_3)$, $(y_2,y_3)$ - and all three losses from those pairs are averaged into one gradient step for that labeling task, rather than being split across three separate updates [1].

## Cost

**Theory, from the method's own math**: RLHF holds four models during Step 3 - the trained policy, a frozen reference copy for the KL term, a reward model, and a value network - versus one model (plus optimizer state) for supervised fine-tuning alone [1]. Each PPO step needs a forward pass through the reward model and the reference model on every sampled completion (no gradients, no optimizer state for either), plus a full forward-backward pass through the policy and the value network (both trained, so both carry optimizer state) [1][4]. Whether the reward-model and reference-model passes are cheap relative to policy training depends on their relative sizes; the paper made all of the reward model, value network, and policy models trainable at up to 175B parameters, so the naive assumption that scoring is cheap does not hold without checking model sizes.

**In practice, per framework**: the paper itself fixes a 6B reward model and 6B value network across all three policy sizes (1.3B, 6B, 175B), specifically so that policy-size effects are not confounded with reward-model or critic size [1]. trl's `PPOTrainer` takes `reward_model` and `value_model` as separate `PreTrainedModel` arguments, and `ref_model` defaults to a copy of the policy model when not supplied, i.e. four model instances resident by default even with no separate reference checkpoint of your own [8]. verl's PPO keeps a distinct critic model alongside the actor and states explicitly that GRPO and RLOO, unlike PPO, do not require a critic model - flagging the critic as PPO's specific extra cost within verl [10].

## How to use it

Data preparation follows the paper's three splits: demonstration data for SFT (prompt plus a single gold completion); comparison data for the RM (a prompt with K ranked completions, expanded to $\binom{K}{2}$ pairs during training); and prompt-only data for PPO rollouts, with no additional labels needed at that stage [1]. Reward convention: the RM outputs one scalar per (prompt, completion) pair, and Step 3 only ever consumes that scalar minus the KL term - it never sees the human rankings directly [1].

| knob | paper (InstructGPT) [1] | trl `PPOTrainer` default [8] | verl PPO default [10] |
| --- | --- | --- | --- |
| KL coefficient $\beta$ | 0.02 | not checked | governed by `algorithm.kl_ctrl`, not checked here |
| pretraining-mix coefficient $\gamma$ | 27.8 (PPO-ptx); 0 for plain "PPO" | not applicable (trl has no pretraining-mix term) | not applicable |
| PPO clip ratio | 0.2 | not checked | not checked |
| rollout batch size | 512, minibatch 64, 1 inner epoch | not checked | governed by `data.train_batch_size` / `critic.ppo_mini_batch_size`, not checked here |
| reward/value model size | fixed at 6B for all policy sizes | separate `reward_model` / `value_model` arguments, any size you pass | separate critic model, any size you configure |
| GAE discount | none (undiscounted) | not checked | `algorithm.gamma` / `algorithm.lam`, not checked here |

The central trade-off a run designer faces beyond the standard PPO knobs is the pretraining-mix coefficient $\gamma$: the paper found that RLHF without it regressed public NLP benchmark scores (e.g. SQuADv2, DROP), and fixed this by mixing in 8x more pretraining gradient steps than RL episodes, at the cost of extra compute per PPO step [1]. That mechanism is specific to InstructGPT's own pipeline and is not part of PPO itself [1][4].

## While it runs

**Signals and their healthy shapes**: the paper's own diagnostic is the RM validation accuracy on held-out comparisons, used to pick RM training checkpoints and to check that the RM has not overfit after more than one epoch over the $\binom{K}{2}$-expanded comparisons [1]; beyond that, this card found no first-hand, method-specific "what to watch while it trains" guidance for the original 3-stage RLHF pipeline in the InstructGPT paper itself (a check of the paper's Section 3 and Appendix C found training-configuration values but no logged-metric health guidance of the kind PPO or GRPO framework docs provide).

**Published reference runs**: InstructGPT's Figure 1 human-preference win rates against the 175B SFT baseline are the paper's own reference curve for judging whether an RLHF run reproduces the paper's result: 85 ± 3% preference for 175B PPO-ptx over 175B GPT-3, 71 ± 4% over few-shot-prompted 175B GPT-3 [1]. verl separately publishes a reference PPO run on Qwen2.5-0.5B-Instruct on GSM8K, going from a 36.4 pretrained-model score to 56.7 after PPO training, as its own known-good curve for the PPO algorithm it ships [10] - useful as a framework-level sanity check, not a reproduction of InstructGPT's human-preference numbers.

**Degeneracies and defaults**: the paper reports that without the pretraining-mix term ($\gamma=0$, the "PPO" variant rather than "PPO-ptx"), RLHF caused measurable regressions on public NLP datasets such as SQuADv2 and DROP - a default that silently trades general capability for preference alignment if left unmitigated [1]. Beyond that, this card did not find InstructGPT-specific guidance on collapse or reward-hacking symptoms in the paper's own limitations discussion (see Known failure modes).

**Named successors**: DPO removes the reward model and PPO loop entirely, optimizing the same preference data with a single cross-entropy-style objective, and its paper frames itself explicitly against RLHF's three-stage complexity [6]. RLAIF keeps RLHF's RM-plus-PPO structure but replaces human preference labels with AI-generated ones in the RM training stage; its abstract reports comparable performance to RLHF across summarization, helpful dialogue generation, and harmless dialogue generation, and a direct-RLAIF variant that skips RM training altogether by scoring rollouts with an off-the-shelf LLM during RL, which it reports outperforms canonical RLAIF [7].

**Known failure modes**: the paper's own limitations section states that InstructGPT still generates simple mistakes - it can fabricate facts, hedge excessively, fail to notice false premises, and produce toxic or biased outputs when explicitly instructed to - and attributes these partly to the RLHF fine-tuning objective being misaligned with genuine helpfulness and honesty rather than to the RL mechanics themselves [1]. This card checked the InstructGPT paper's own limitations discussion for these findings; it did not search a library's issue tracker for RLHF-specific (as opposed to PPO- or GRPO-specific) maintainer reports, so no maintainer-reply failure mode is recorded here.

**What the gain is - and is not**: the paper's central claim is a preference gain, not a new capability - labelers prefer InstructGPT's outputs for helpfulness, truthfulness, and reduced toxicity over the base and SFT models on the paper's prompt distribution, with only minor "alignment tax" regressions on some public NLP datasets that pretraining-mix training mitigates [1]. RLHF here does not claim to add reasoning or knowledge the base model lacks; it reshapes which of the base model's already-reachable outputs get sampled.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Framework docs are `main` / `latest` builds, unpinned and mutable; every default quoted above is a 2026-08-08 reading - re-check against the version you install.

[1] Ouyang et al., "Training language models to follow instructions with human feedback", 2022. https://arxiv.org/abs/2203.02155 - defines the SFT/RM/PPO pipeline, RM loss, PPO objective, dataset sizes, hyperparameters, human-preference results, limitations. Fetched 2026-08-08 (ar5iv HTML full text).

[2] Christiano et al., "Deep reinforcement learning from human preferences", 2017. https://arxiv.org/abs/1706.03741 - originates RL from human preference comparisons as a reward signal. Fetched 2026-08-08 (abstract page).

[3] Stiennon et al., "Learning to summarize from human feedback", 2020. https://arxiv.org/abs/2009.01325 - applies RL-from-preferences to language-model summarization; source of the KL-penalty computation InstructGPT reuses in Step 3. Fetched 2026-08-08 (abstract page).

[4] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - the RL algorithm used unmodified as RLHF's Step 3 optimizer. Fetched 2026-08-08 (abstract page).

[5] Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023. https://arxiv.org/abs/2307.09288 - landmark adopter; defines and applies RLHF in its own pipeline. Fetched 2026-08-08 (ar5iv HTML full text).

[6] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - offline alternative to RLHF's RM-plus-PPO pipeline. Fetched 2026-08-08 (abstract page).

[7] Lee et al., "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback", 2023. https://arxiv.org/abs/2309.00267 - AI-feedback variant of RLHF; comparative human-preference results. Fetched 2026-08-08 (abstract page).

[8] trl PPOTrainer documentation. https://huggingface.co/docs/trl/main/en/ppo_trainer - `trl.experimental.ppo.PPOTrainer` signature, `ref_model`/`reward_model`/`value_model` arguments. Fetched 2026-08-08.

[9] trl trainer taxonomy / index page. https://huggingface.co/docs/trl/main/en/index - marks `PPOTrainer` experimental versus `RewardTrainer` and `SFTTrainer` unmarked. Fetched 2026-08-08.

[10] verl PPO documentation. https://verl.readthedocs.io/en/latest/algo/ppo.html - `algorithm.adv_estimator: gae` entry point, critic-model requirement, reference GSM8K run. Fetched 2026-08-08.

[11] OpenRLHF README. https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/main/README.md - Ray+vLLM RLHF framework description, `openrlhf.cli.train_sft` / `train_rm` / `train_ppo_ray` entry points. Fetched 2026-08-08.
