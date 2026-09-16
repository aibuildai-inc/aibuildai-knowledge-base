# ProRL

GRPO with DAPO's decoupled clipping and dynamic sampling, plus an explicit KL penalty against a reference policy that is periodically hard-reset (weights and optimizer state) to the current policy, run for thousands of steps across a diverse verifiable-reward task mix instead of the few hundred steps typical of short RL runs.

**ProRL** (Prolonged Reinforcement Learning) is a training methodology introduced by Liu et al. to test whether reinforcement learning with verifiable rewards can expand a language model's reasoning boundary beyond what the base model can already reach under extensive sampling, rather than only re-weighting outputs already latent in its distribution [1]. Its core algorithm is GRPO [2], which the paper summarizes as removing PPO's [3] value model and estimating the baseline from group scores instead [1]; GRPO's own defining paper describes it as a variant of PPO [2]. On top of GRPO, ProRL adopts DAPO's decoupled clipping ($\epsilon_{low} \neq \epsilon_{high}$) and dynamic sampling that filters prompts with accuracy exactly 0 or 1 [4], adds an explicit KL-divergence penalty between the online policy and a reference policy, and periodically hard-resets that reference policy to a recent policy snapshot together with the optimizer state [1]. The paper is at https://arxiv.org/abs/2505.24864 [1]. The paper gives three reasons for this recipe: high sampling temperature alone only delays entropy collapse rather than preventing it [1]; recent work argues for dropping the KL penalty entirely, but the paper finds that argument applies to base models before supervised fine-tuning, whereas ProRL starts from an SFT checkpoint already producing coherent chain-of-thought, where a KL penalty still helps stability and sustained entropy [1]; and as training progresses the KL term can grow to dominate the loss and stall updates, which the periodic reference reset relieves [1].

The paper trains Nemotron-Research-Reasoning-Qwen-1.5B, a 1.5B model, with ProRL for more than 2k RL training steps on a 136K-example verifiable dataset spanning math, code, STEM, logic puzzles (Reasoning Gym), and instruction following, using verl [5] as the training framework [1]. The resulting model outperforms its DeepSeek-R1-Distill-Qwen-1.5B [6] starting checkpoint by +15.7% on math, +14.4% on code, +25.9% on STEM, +22.0% on instruction following, and +54.8% on the text-based logic puzzles of Reasoning Gym, and also beats domain-specialized 1.5B baselines by +4.6% on math and +6.5% on code (Sec. 3, no separate table number given for these headline deltas) [1]. Lineage: PPO (2017 [3]) -> GRPO (DeepSeekMath, 2024 [2]) -> DAPO's decoupled-clip/dynamic-sampling additions (2025 [4]) -> ProRL (2025 [1]), combining GRPO+DAPO with KL control and reference reset for long-horizon runs; the paper's own release is a model checkpoint and technical report, not a named "ProRL" library. This card treats the shortlist row's exact-title match against arXiv:2505.24864 as confirmed, since the row itself was produced from a single-candidate title match with no alternates to disambiguate.

**When to pick it**: pick ProRL when you plan to run RL for thousands of steps (the paper ran past 2k) to push a model's reasoning boundary rather than a few hundred steps to sharpen it, particularly when starting from a weaker base model - the paper finds the biggest pass@128 gains where the base model's initial reasoning boundary is lowest [1]. Its parent GRPO [2] alone entropy-collapses over long runs according to the paper's own diagnosis [1]; DAPO alone [4] supplies the decoupled clip and dynamic sampling but not the KL penalty or reference reset that ProRL adds for long-horizon stability [1]. As an offline contrast, DPO [7] needs no online rollouts at all, trading exploration for a fixed preference dataset; ProRL's entire value proposition is prolonged online exploration, so DPO is not a substitute for this goal.

**Variant of**: GRPO [2], with DAPO's decoupled clipping and dynamic sampling folded in [4].

**Data it needs**: prompts with a programmatically verifiable reward, no chosen/rejected pairs and no learned reward model - the paper's 136K-example mix uses answer matching for math, test execution for code, and rule-based constraint checking (IFEval-style) for instruction following [1]. On-policy: each step samples n=16 fresh completions per prompt from the current policy at temperature 1.2 [1]. Training scale in the paper: 136K prompts, batch size 256, mini-batch 64 (4 gradient updates per rollout step), on 4x8 NVIDIA H100-80GB nodes for about 16k GPU-hours [1].

**Extra models**: no value network (inherited from GRPO [2]); one frozen reference policy for the KL term, which the method's own definition requires to be periodically hard-reset - weights and optimizer state - to a recent snapshot of the online policy, monitored via a held-out validation blend [1]. No reward model: every domain in the paper's mix uses a rule-based verifier, not a learned judge [1]. Cost detail below.

**Shipped by**: no library ships a "ProRL" trainer. The paper's own run was built on verl [5], whose GRPO documentation exposes a single unified `actor_rollout_ref.actor.clip_ratio` (default 0.2), not the decoupled low/high clip range the paper itself reports using [1][5] - the decoupled clip is DAPO's own contribution [4], and this card did not find it as a verl config key. ProRL's KL-penalty-plus-reference-reset loop is likewise not a standard verl config flag: the paper describes it as a custom outer training loop that monitors validation performance and hard-resets the reference model and optimizer when it stagnates or degrades [1]; the verl-recipe collection (commit e7f8895, checked 2026-08-09) ships a `dapo` recipe but no `prorl` recipe [8]. Building this recipe means combining an existing GRPO/DAPO trainer's KL-loss option with scripted checkpoint-swap-and-optimizer-reinit logic, not writing a new sampling loop.

## How it works

Each step: sample a group of completions per prompt from the current policy, filter out prompts with accuracy 0 or 1 (dynamic sampling), compute group-relative advantages under GRPO, update with DAPO's decoupled clip, and add a KL penalty against a reference policy that is occasionally hard-reset [1].

**Base objective - GRPO** [1], quoting the paper's own compressed form:

$$ \mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[ \min\big(r_\theta(\tau) A(\tau),\ \operatorname{clip}(r_\theta(\tau), 1-\epsilon, 1+\epsilon)\, A(\tau)\big) \Big] $$

$r_\theta(\tau)$ is the probability ratio between the current and old policy before each actor update, and $A(\tau)$ is the group-relative advantage computed from the group of scores $\{R_i\}_{i \in G(\tau)}$ for the sampled trajectory group, with no critic model involved [1].

**DAPO's decoupled clip** [4], as ProRL applies it: the single clip range $\epsilon$ is split into separate lower and upper bounds,

$$ \operatorname{clip}\big(r_\theta(\tau),\ 1-\epsilon_{low},\ 1+\epsilon_{high}\big) $$

with the paper setting $\epsilon_{low}=0.2$, $\epsilon_{high}=0.4$ [1]. A wider upper bound lets the ratio for a positive-advantage token rise further before clipping engages, which the paper adopts (together with dynamic sampling and the KL term) specifically to counter entropy collapse [1].

**KL-regularized loss** [1]:

$$ L_{KL\text{-}RL}(\theta) = L_{GRPO}(\theta) - \beta\, D_{KL}\!\left(\pi_\theta \,\|\, \pi_{ref}\right) $$

The paper states its motivation and mechanism for $\pi_{ref}$ but does not give a numeric value for $\beta$ in the main text [1]. Periodically, $\pi_{ref}$ is hard-reset to a more recent snapshot of the online policy $\pi_\theta$, and the optimizer state is reinitialized at the same time, which the paper reports both restores training stability and lets the policy diverge further from the base model than a fixed reference would allow [1].

**Dynamic sampling** [4], as used here: prompts where the group's accuracy is exactly 1 or exactly 0 are dropped before the update, because a group with no variance in reward carries no GRPO learning signal - the same degeneracy noted on the GRPO card - and this filtering keeps the training signal concentrated on intermediate-difficulty prompts [1].

**Worked example of why zero-variance groups are filtered**: for a group of n=16 completions on one prompt, if all 16 receive reward 1 (or all 16 receive reward 0), the group mean equals every individual reward, so the group-normalized advantage is 0 for every completion in the paper's GRPO-style normalization (see the GRPO card's worked example for the general case); dynamic sampling removes that prompt from the batch rather than training on sixteen zero-advantage tokens [1][4].

## Cost

**Theory, from the method's own math**: prolonged training multiplies GRPO/DAPO's per-step cost (n=16 rollouts per prompt, no critic forward/backward) by the number of steps run - the paper runs past 2k steps versus the few hundred typical of shorter RL recipes, which is the method's central resource trade [1]. The reference-policy reset adds no new model class beyond the KL term's existing frozen copy, but it does require periodically re-copying full policy weights into the reference slot and reinitializing optimizer state (discarding accumulated Adam moments), which is a point cost at reset time rather than a per-step cost.

**In practice, per framework**:

- verl [5]: the paper's own training uses verl as the RL framework, with GRPO's advantage estimator (`algorithm.adv_estimator: grpo`) and a decoupled clip range plus dynamic-sampling filter that the paper itself configured [1]; verl's own GRPO documentation, as read at [5], exposes only a single `clip_ratio` (default 0.2) rather than a separate low/high pair, so the decoupled range is DAPO's contribution layered on top, not a documented verl GRPO default [4][5]. The hard reference-and-optimizer reset is not a verl config flag; the paper implements it as an outer-loop procedure gated on a validation blend (AIME 2024, Codeforces, GPQA-diamond, IFEval, and the Reasoning Gym task graph_color), triggered when validation performance stagnates or degrades [1].
- The paper reports extended training on 4x8 H100-80GB nodes for about 16k GPU-hours to reach its final checkpoint through eight sequential training runs, several of which include a hard reset [1] - this is the paper's own measured cost, not a per-framework guarantee, since no other framework ships the recipe (see Shipped by).

## How to use it

- Prompts and rewards: assemble a verifiable-reward mix; the paper's 136K examples span math (answer matching), code (unit-test execution), STEM (multiple-choice/short-answer matching), logic puzzles from Reasoning Gym (programmatic checking), and instruction following (rule-based constraint checking, e.g. verifying a generated essay actually has three paragraphs when the instruction asks for it) [1].
- Context window and length: the paper caps responses at 8k tokens for most of training against a 128k-token base-model limit, to avoid long-sequence rollouts, then raises the cap to 16k tokens for a final ~200-step stage and reports the model adapting quickly [1].
- Reward shaping for termination: two of the paper's eight training runs (Runs 4-5) add a penalty for responses that fail to terminate with an end-of-sequence token, after observing runaway response-length growth from repeated non-terminating generations; this was found to modestly shorten responses without ProRL-specific tuning guidance beyond that empirical fix [1].
- Knobs table, paper values only (no framework ships defaults for this recipe, so there is no framework column to compare against):

| knob | ProRL paper [1] |
| --- | --- |
| rollouts per prompt (n) | 16 (Runs 1-5), later reduced to increase to 32 in Runs 6-7, then back to 16 with the extended context in Run 8 |
| clip range | $\epsilon_{low}=0.2$, $\epsilon_{high}=0.4$ (DAPO's decoupled clip [4]) |
| rollout temperature | 1.2 |
| batch size / mini-batch size | 256 / 64 (4 gradient updates per rollout step) |
| optimizer / learning rate | AdamW, constant $2\times10^{-6}$ |
| response length cap | 8k tokens for most of training, 16k tokens in the final ~200-step stage |
| KL coefficient $\beta$ | not stated in the main text |
| training length | more than 2k steps, across 8 sequential runs with periodic hard resets |

- Group-size and step-budget trade-off: the paper's own change from n=16 to n=32 rollouts across Runs 6-7 (with two hard resets in that stretch) was followed by rising response length alongside validation improvement, illustrating the trade between rollouts-per-prompt and total steps affordable within a fixed compute budget [1].

## While it runs

- **Signals and their healthy shapes**: the paper monitors a validation blend (AIME 2024, Codeforces, GPQA-diamond, IFEval, Reasoning Gym's graph_color) and reports that pass@1 and pass@16 on this blend consistently improved and scaled with added training compute across its runs [1]. It also reports a positive but non-decisive correlation between average response length and validation score - some training stages improve without longer responses - so response length alone is not a reliable proxy for progress [1]. Entropy is the trigger for intervention: the paper states that its DAPO components plus the KL-divergence loss are what let the model avoid entropy collapse over the extended run [1].
- **Published reference runs**: the paper's own Figure 2 (training dynamics) and Figure 8 (KL divergence across the eight sequential runs, with reset points marked) are the published reference curves for this recipe; this card did not extract the raw log values, only the paper's textual description of their shape [1].
- **Degeneracies and defaults**: a prompt group with accuracy exactly 0 or exactly 1 across all rollouts carries no GRPO learning signal, which is why dynamic sampling filters those prompts out before the update, the same zero-variance degeneracy documented on the GRPO card [1][4]. The paper's own $\beta$ (KL coefficient) value is not stated, so a practitioner rebuilding this recipe has no paper-given default to anchor to and must tune it, unlike the clip range and learning rate, which are given exactly.
- **Named successors**: none identified as of this card; the paper itself is the most recent recipe in this lineage and was submitted 2025-05-30 [1].
- **Known failure modes**: from the paper's own limitations section - extended RL training with multiple stages, periodic resets, and long reasoning-chain sampling is computationally expensive, which may be prohibitive for smaller organizations [1]; it is untested whether the recipe scales to larger models than the paper's 1.5B, and compute requirements grow with parameter count [1]; the periodic hard-resets of reference policy and optimizer add training-process complexity and may produce results less consistent than more stable single-phase training methods [1]; and the training/evaluation task mix, though broad, is still a subset of possible reasoning tasks, so generalization to domains entirely outside it is not guaranteed [1]. No maintainer-issue-level failure modes are reported here because no library ships this recipe as a trainer to file issues against (see Shipped by); this card did not search GitHub issues for a "prorl" recipe since none exists in the verl-recipe collection checked [8].
- **What the gain is - and is not**: the paper's own analysis finds RL's effect on the reasoning boundary (measured by pass@128) is strongly, negatively correlated with how well the base model already does on a task - tasks where the base model already scores well see minimal or negative gains in reasoning breadth after RL, narrowing toward solutions the model already favors, while tasks where the base model starts weak see the largest expansions in both pass@1 and the breadth of reasoning paths explored [1]. The paper links this to a creativity-index measure: tasks with the smallest post-RL gains tend to have low creativity-index scores against the DOLMA pretraining corpus, i.e. the base model likely saw similar data during pretraining [1]. So the gain from prolonged RL is largest exactly where a base model is weakest, not a uniform boost across all reasoning tasks.

## Sources

[1] Liu et al., "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models", 2025. https://arxiv.org/abs/2505.24864 - defines ProRL: GRPO+DAPO base, KL penalty, reference/optimizer hard reset, training recipe (Appendix E), results, pass@k analysis, limitations. Fetched 2026-08-09 (HTML full text and abstract page).

[2] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO, ProRL's core algorithm. Fetched 2026-08-09 (abstract page; formula details taken from [1]'s own restatement).

[3] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, GRPO's parent method. Fetched 2026-08-09 (abstract page).

[4] Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", 2025. https://arxiv.org/abs/2503.14476 - defines decoupled clipping and dynamic sampling, adopted by ProRL; reports 50 points on AIME 2024 with a Qwen2.5-32B base in its own open-sourced system. Fetched 2026-08-09 (abstract page).

[5] verl GRPO documentation. https://verl.readthedocs.io/en/latest/algo/grpo.html - confirms verl implements the GRPO advantage estimator ProRL's training used, and documents its `clip_ratio` config key. This page's own embedded metadata reports its readthedocs version slug as "latest", a floating alias rather than a pinned release or commit, so it is an unpinned, moving build; every default or config key quoted from it is a 2026-08-09 reading and may not hold for a later edit of the page. Fetched 2026-08-09.

[6] DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025. https://arxiv.org/abs/2501.12948 - source of the DeepSeek-R1-Distill-Qwen-1.5B checkpoint ProRL starts from, named in [1]. Fetched 2026-08-09 (abstract page).

[7] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - offline preference-pair alternative contrasted in When to pick it: trains on pairs of completions to the same prompt without online rollouts. Fetched 2026-08-09 (abstract page).

[8] verl-project/verl-recipe GitHub repository contents listing, commit e7f889574b8301cc0f0fc1d57c6d67f31ffeb689 (the `recipe` submodule of volcengine/verl as of 2026-08-09). https://github.com/verl-project/verl-recipe - lists a `dapo` recipe directory and no `prorl` directory. Fetched 2026-08-09 (GitHub API contents listing).
