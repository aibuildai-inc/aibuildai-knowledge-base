# StepGRPO

GRPO with the outcome-only reward replaced by two rule-based, step-wise rewards - one that checks whether a reasoning path hits the key intermediate steps of a reference solution, one that checks the path is structurally complete and logically ordered - so a multimodal model gets a training signal even when its final answer is wrong.

**StepGRPO** (Step-wise Group Relative Policy Optimization) is an online-RL method for fine-tuning multimodal LLMs (MLLMs) on step-by-step reasoning, introduced in the paper "R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization" as a two-phase framework: SFT warm-up followed by step-wise online policy optimization [1]. Its parent is GRPO, defined in the DeepSeekMath paper as a variant of PPO that drops the value network and estimates the advantage from a group of sampled completions [2]; PPO itself is GRPO's grandparent [3]. StepGRPO keeps GRPO's group-relative advantage but replaces the single outcome-level reward with two rule-based, dense step-wise rewards - Step-wise Reasoning Accuracy Reward (StepRAR) and Step-wise Reasoning Validity Reward (StepRVR) - computed over each sampled reasoning path before the group is normalized [1]. The paper gives two reasons: MLLMs' limited reasoning ability means outcome-only reward is sparse, so few sampled paths in a group receive any signal, which is unstable to learn from; and dense step-level supervision can be obtained with hand-written rules instead of training a separate process reward model (PRM), avoiding that model's extra cost [1]. The paper is at https://arxiv.org/abs/2503.12937 [1].

StepGRPO trains R1-VL, a pair of models (2B and 7B) built by applying it to Qwen2-VL-2B and Qwen2-VL-7B [1]; a third variant, R1-VL-7B*, applies it to Qwen2.5-VL-7B with training data from the follow-up work R1-ShareVL [4][1]. Against its own Qwen2-VL baselines, Table 1's eight-benchmark-average column rises from 37.5 to 41.6 for the 2B model (+4.1) and from 48.7 to 52.1 for the 7B model (+3.4); the paper's own Section 4.3 prose rounds these to "4.6%" and "3.8%" respectively, a discrepancy this card notes rather than resolves. On the MathVista column of the same table, R1-VL-7B (63.5) scores 0.4 points above Mulberry-7B (63.1) and 9.1 points above LlamaV-o1-11B (54.4), though Section 4.3's prose states the gaps as "0.6%" and "9.3%" [1]. An ablation on Qwen2-VL-7B over MathVista shows warm-up alone at 61.2%, StepRAR or StepRVR alone at 62.4% and 61.9%, and both together at 63.5% (Table 2), and a separate comparison shows step-wise reward beating outcome-level reward at the same warm-up starting point, 63.5% versus 62.3% (Table 4) [1]. The right paper was confirmed by an exact-title arXiv search that returned this paper as the top-cited match among candidates for "Step-wise Group Relative Policy Optimization" / "StepGRPO", ahead of a five-citation paper using a similarly-worded name for an unrelated retrieval method. Lineage in one line: PPO (2017 [3]) -> GRPO (DeepSeekMath, 2024 [2]) -> StepGRPO (R1-VL, 2025 [1]) -> cited by the same group's follow-up R1-ShareVL, which targets a different failure mode (sparse reward and advantage vanishing over an expanded question space) with a different mechanism (Share-GRPO) [4].

**When to pick it**: online RL for MLLM step-by-step reasoning when outcome-only reward is too sparse for a weak or newly-warmed-up policy to learn from, and you have (or can extract with an LLM) a reference reasoning path to score intermediate steps against - StepGRPO's own motivation is exactly this sparse-reward regime [1]. Prefer plain GRPO [2] when the policy is already strong enough that outcome-only reward yields non-degenerate groups. Prefer training a PRM (as in ReST-MCTS*, cited by the paper as the process-reward-model alternative [1]) when you need step-level judgments that a fixed rule set cannot express, at the cost of a separate reward model. Nearest offline alternative is SFT: the paper's own comparison at matched training steps on Qwen2-VL-7B/MathVista shows StepGRPO beating continued SFT on the same warm-up checkpoint (Fig. 3), because SFT only imitates the reference path while StepGRPO scores self-sampled paths [1].

**Variant of**: GRPO [2].

**Data it needs**: two phases, both multimodal (image + text question). Phase 1 (warm-up SFT) needs question-plus-reference-CoT-path pairs; the paper uses Mulberry-260K [1]. Phase 2 (RL) needs the same kind of question, plus - only for StepRAR - a set of key reasoning steps pre-extracted from the reference path with GPT-4 and augmented into equivalent surface forms for soft matching (e.g. "6/3 = 2" also stored as "6 divided by 3 equals 2") [1]; the paper samples 10K questions from Mulberry-260K for this stage, released as the R1-VL-10K dataset [1][5]. On-policy: each step samples M=4 fresh completions per question from the current policy (temperature 1.2, max length 1024 tokens) [1].

**Extra models**: no value network - inherited from GRPO [2]. A frozen reference model is required by the method's own KL term (the paper's default coefficient is $\beta=0.04$, attributed to the trl library [6][1]); both the policy and reference model are initialized from the post-warm-up checkpoint [1]. StepRAR's key-step extraction uses GPT-4 as an offline pre-processing step before RL begins, not a model held during training [1]. See Cost for what this costs at training time.

**Shipped by**: no library implements StepGRPO - it does not appear in trl's or verl's trainer lists as of this reading. The paper's own code is a standalone repository (github.com/jingyi0000/R1-VL) [7], which supplies its own RL launch scripts, `src/r1-vl/run_grpo_2b_vllm.sh` and `src/r1-vl/run_grpo_7b.sh`, and states its RL stage is "based on" R1-V (github.com/Deep-Agent/R1-V, now hosted as StarsfieldAI/R1-V) [7]. R1-V is itself a script-based GRPO training setup: its `src/r1-v` package holds the trainer code and a base launch script `run_grpo.sh`, while its vLLM-accelerated launch scripts `run_grpo_vllm.sh` and `run_grpo_vllm_qwen25vl.sh` live in a sibling directory, `src/scripts/`; none of this is a call into trl's or verl's packaged trainer [8]. Building StepGRPO on a maintained framework is a reward-function addition, not a new sampling loop: trl's `GRPOTrainer` already accepts a list of custom `reward_funcs` that each return a per-completion float [9], so StepRAR and StepRVR could be written as two such functions (StepRAR needs the pre-extracted key steps as an extra dataset column) and combined the way trl already combines multiple reward functions.

## How it works

The loop in one line: after a warm-up SFT phase, sample M reasoning paths per question, score each with two rule-based step-wise rewards, normalize by the group's mean/std, and update the policy on a KL-regularized policy-gradient loss [1].

**Phase 1 - warm-up.** Standard token-level negative log-likelihood on the reference CoT data $D_s=\{Q^n,\tau^n\}_{n=1}^N$ [1]:

$$ \mathcal{L}_{warm-up} = -\mathbb{E}_{\tau\sim D_s}\Big[\sum_{t=1}^{T}\log\big(\pi_\theta(a_t|s_t)\big)\Big] $$

**Phase 2 - StepRAR (Eq. 2)** [1]. Let $y$ be the ground-truth answer, $\text{ans}(s_{t+1})$ the answer extracted from the path so far, $\mathbf{v}=\{v_1,v_2,\dots\}$ the pre-extracted key steps, and $k^i=|\mathbf{v}_{match}|/|\mathbf{v}|$ the fraction of key steps matched by path $i$ via soft matching:

$$ r_{auc}^i(s_t,a_t,s_{t+1}) = \begin{cases} 1+\alpha k^i, & \text{ans}(s_{t+1})=y \\ \alpha k^i, & \text{ans}(s_{t+1})\neq \text{null},\ \neq y \\ 0, & \text{ans}(s_{t+1})=\text{null} \end{cases} $$

$\alpha$ (the paper uses 0.1 [1]) weights the step-matching bonus; a path with the correct final answer and every key step matched scores $1+\alpha$, a wrong-but-complete path can still earn up to $\alpha$, and an incomplete path scores 0.

**Phase 2 - StepRVR (Eq. 3)** [1]. With completeness indicator $\delta^c$ (background analysis, step-by-step reasoning, and a final answer are all present) and logic indicator $\delta^l$ (that content appears in the right order - background before steps, steps before the answer):

$$ r_{val}^i(s_t,a_t,s_{t+1}) = \begin{cases} 1, & \mathbb{I}(\delta^c(s_{t+1}))\cdot\mathbb{I}(\delta^l(s_{t+1}))=1 \\ 0, & \text{otherwise} \end{cases} $$

**Combined reward and advantage.** The two rewards sum per path, $r^i=r_{auc}^i+r_{val}^i$, and are normalized against the group exactly as in GRPO [1][2]:

$$ \hat{A}^i = \frac{r^i-\text{mean}(\{r^1,\dots,r^M\})}{\text{std}(\{r^1,\dots,r^M\})} $$

**Objective (Eq. 5)** [1]:

$$ \mathcal{L}_{StepRL} = -\mathbb{E}_{Q\in D_s}\Big[\frac{1}{M}\sum_{i=1}^{M}\Big(\frac{\pi_\theta(\mathbf{c}^i|Q)}{[\pi_\theta(\mathbf{c}^i|Q)]_{\text{no grad}}}\hat{A}^i - \beta D_{KL}(\pi_\theta\|\pi_{ref})\Big)\Big] $$

The denominator is written as a stop-gradient copy of the numerator, $[\pi_\theta(\mathbf{c}^i|Q)]_{\text{no grad}}$, not GRPO's separate old-policy snapshot $\pi_{\theta_{old}}$ [1][2]; at the point the ratio is computed its value is 1, so the fraction functions as a gradient-routing device for the policy-gradient-with-baseline term rather than an importance-sampling ratio, and the equation carries no clip/min term - unlike the DeepSeekMath paper's own GRPO objective, $\mathcal{J}_{GRPO}(\theta)$ (its Eq. 3), which keeps PPO's $\min[\cdot,\text{clip}(\cdot)]$ structure over the old-policy ratio $\pi_\theta/\pi_{\theta_{old}}$ [2]. This is not a departure from every implementation: trl's own documented GRPO loss uses the identical stop-gradient ratio with no clip term as its default loss formula, and trl's docs describe the clipped, old-policy-snapshot form as a generalization needed only when doing multiple gradient updates per rollout batch [6] - StepGRPO's own Algorithm 1 runs exactly one "Optimize policy model" call per sampled group, matching that single-update case [1]. The KL term uses the same unbiased estimator GRPO cites [1][2]:

$$ D_{KL}(\pi_\theta\|\pi_{ref}) = \frac{\pi_{ref}(\mathbf{c}^i|Q)}{\pi_\theta(\mathbf{c}^i|Q)} - \log\frac{\pi_{ref}(\mathbf{c}^i|Q)}{\pi_\theta(\mathbf{c}^i|Q)} - 1 $$

**Worked example**, $\alpha=0.1$, group of $M=4$ with rewards $r=\{1.1,\ 0.1,\ 0,\ 1.0\}$ (a fully-matched correct path, a matched-but-wrong path with $k=1$, an incomplete path, and a correct path with $k^i=0$ and $\delta$ satisfied): mean $=0.55$, population std $\approx 0.487$; the best path's advantage is $(1.1-0.55)/0.487\approx+1.13$, the incomplete path's is $(0-0.55)/0.487\approx-1.13$. This card verifies only the arithmetic convention shown here, not which std convention (population vs. sample) any implementation runs, since no framework ships this loss (see Shipped by).

## Cost

**Theory, from the method's own math:**

- Time: the warm-up phase is one SFT pass, no rollouts. The RL phase needs M completions generated per question (the paper's default M=4 [1]) plus one training forward/backward pass over the sampled tokens for the policy. Because $\beta=0.04>0$ in the paper's setting, an additional reference-model forward pass (no gradients) runs over the same tokens for the KL term [1] - the same structural cost GRPO incurs when its KL term is on [2]. StepRAR's key-step extraction with GPT-4 runs once, offline, before RL training starts, not inside the training loop [1].
- Memory: like GRPO, no value network is trained [1][2]. With $\beta>0$ a frozen reference model is held in memory in addition to the trained policy; with $\beta=0$ (not the paper's default here) that model would not be needed, by the same reasoning GRPO's documentation gives for its own $\beta$ [1][6].
- A naive reading of the objective's $1/M\sum_i$ suggests all M paths' activations must coexist in memory; as in GRPO, the group only has to coexist as M reward scalars at the advantage-normalization step, not as M sets of activations (see While it runs for the paper's own $M$ ablation, which is about statistical stability, not memory).

**In practice, per framework:**

- No published framework ships StepGRPO, so there is no framework-level practice claim to make here (see Shipped by). The only concrete resource report available is the paper's own reference run: all experiments, including RL with M=4, were conducted on 4 H100-80GB GPUs [1] - a data point about the paper's setup, not a framework's documented behavior, and it does not by itself say whether generation or the training pass dominated wall-clock time.

## How to use it

- Data preparation: warm-up needs question + reference step-by-step CoT pairs (the paper strips "reflective" reasoning data out of Mulberry-260K for this stage, per the official repo's instructions) [7]. RL needs the same kind of question plus, for StepRAR, a pre-extracted, format-augmented set of key steps per question; the official repo ships a prompt for extracting these from your own data, or points to the pre-built R1-VL-10K dataset [7][5].
- Reward/label conventions (paper's own choices, no shipping framework to defer to): $r^i=r_{auc}^i+r_{val}^i$ is a simple sum of the two rule-based rewards, not a weighted combination [1]; StepRAR needs a parsed final answer and pre-extracted key steps per question, StepRVR needs no external reference beyond the format check (background/steps/answer, in order) [1].
- Key knobs, with the paper's own values as the only source (no framework anchors exist for this method):

| knob | paper value [1] |
| --- | --- |
| group size M | 4 |
| key-step bonus $\alpha$ | 0.1 |
| KL coefficient $\beta$ | 0.04 |
| sampling temperature | 1.2 |
| max sequence length | 1024 |
| policy learning rate (RL phase) | 1e-6 |
| warm-up learning rate | 1e-5 (2B model), 5e-6 (7B model) |
| RL batch size | 4 |
| warm-up batch size | 128 |

- Group size trade-off: the paper's own sweep over M on Qwen2-VL-7B/MathVista shows a roughly monotonic gain from M=2 (62.5%) to M=6 (63.7%), attributed to a more stable group-mean baseline at larger M, traded against higher per-question generation cost; the paper fixes M=4 as its default balance (Table 3) [1].
- The paper's $\beta=0.04$ sits far from trl's own current documented GRPO default of $\beta=0.0$ (KL term off, reference model not loaded) [6], even though the paper attributes the 0.04 choice to the trl citation [1]; trl's docs justify their zero default by noting recent work has questioned whether the KL term is necessary at all [6]. Whoever reruns this method should pick $\beta$ deliberately rather than assume either value.

## While it runs

- Signals and their healthy shapes: the paper's own ablation (Table 2) is the first-hand signal to watch during development, not a live-training metric. The base model scores 58.2% on MathVista; adding warm-up alone raises it to 61.2%; from that warm-up checkpoint, adding StepRAR alone reaches 62.4% and adding StepRVR alone reaches 61.9% (StepRVR alone scores lower than StepRAR alone); combining warm-up with both StepRAR and StepRVR reaches the table's best score, 63.5% [1]. The two single-reward rows are not a monotonic ladder; only the three-way combination clears both of them. No framework logs StepGRPO-specific metric names (e.g. a `frac_reward_zero_std`-style signal), because none ships it (see Shipped by); a practitioner building it on trl's GRPOTrainer would reuse that trainer's existing group-reward and clip-fraction logging [6][9], but that is this card's own inference about a hypothetical build, not a documented StepGRPO signal.
- Published reference runs: Table 1's baseline-vs-R1-VL comparison is the paper's own reference curve in table form - Qwen2-VL-2B 37.5 to R1-VL-2B 41.6 (8-benchmark average), Qwen2-VL-7B 48.7 to R1-VL-7B 52.1 [1]. No raw training-log files (loss/reward curves over steps) are published in the paper or the official repository as read here.
- Degeneracies and defaults: a uniform-reward group ($r^1=\dots=r^M$) gives std $=0$ and an undefined $\hat A^i$, the same degeneracy GRPO has [1][2]; the paper does not state a guard for this case. A group of one collapses the whole group-relative mechanism, as in GRPO [2].
- Named successors: the same authors' R1-ShareVL replaces StepGRPO's step-wise-reward fix for sparse reward with a different mechanism, Share-GRPO, which expands and shares reasoning trajectories across an enlarged question space to address both sparse reward and advantage vanishing [4].
- Known failure modes: the paper's own conclusion and discussion sections describe what StepGRPO fixes (sparse reward, need for a PRM) but state no limitations or failure modes of the method itself [1]. This card checked the paper's abstract, introduction, conclusion, and discussion sections (Section 4.5) for a stated limitation and found none; the official GitHub repository's issue tracker was not queried, so "none found" here covers only the paper text, not maintainer reports.
- What the gain is - and is not: the paper's own ablation attributes the improvement specifically to denser supervision along the whole reasoning trajectory rather than only at the final answer - StepRAR and StepRVR each independently outperform warm-up-only, and outperform outcome-level reward when compared directly at the same warm-up starting point (63.5% vs. 62.3% on MathVista, Table 4) [1]. The paper frames this as StepGRPO teaching the model to self-improve via reward-guided exploration rather than by imitating a fixed positive reasoning path, and reports it beats continued SFT at matched training steps (Fig. 3) [1]; it does not claim StepGRPO adds reasoning capability the base MLLM cannot already produce with some sampled path.

## Sources

[1] Zhang, Huang, Yao, Liu, Zhang, Lu, Tao, "R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization", 2025. https://arxiv.org/abs/2503.12937 - defines StepGRPO: warm-up loss, StepRAR/StepRVR reward definitions, advantage and objective equations, KL estimator, implementation details, main results, ablations, discussion. The abstract page lists two revisions, v1 (2025-03-17) and v2 (2025-08-04); fetched 2026-08-09 as v2, the version live at the unversioned URL at fetch time (HTML full text via arxiv.org/html/2503.12937).

[2] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO, the parent method: removal of PPO's value network, group-sampled baseline (Sec. 4.1.1), the clipped objective $\mathcal{J}_{GRPO}$ (Eq. 3), KL estimator. The abstract page lists three revisions, v1 (2024-02-05), v2 (2024-02-06), v3 (2024-04-27); fetched 2026-08-09 as v3, the version live at the unversioned URL at fetch time (HTML full text via arxiv.org/html/2402.03300).

[3] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, the grandparent method and source of the clipped surrogate objective that StepGRPO's Eq. 5 omits. The abstract page lists two revisions, v1 (2017-07-20), v2 (2017-08-28); fetched 2026-08-09 as v2, the version live at the unversioned URL at fetch time (abstract page).

[4] Yao et al., "R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO", 2025. Cited via [1]'s reference list and the official R1-VL repository's news entry, which states it targets sparse reward and advantage vanishing via reasoning-trajectory sharing over an expanded question space; the paper itself (arXiv:2505.16673) was not separately fetched, so this card reports only what [1] and [7] state about it.

[5] R1-VL-10K dataset. https://huggingface.co/datasets/jingyiZ00/R1-VL-10K - the 10K-question RL-stage dataset referenced by the official repository. Not fetched directly at check time; cited via the official repository's README [7].

[6] trl GRPOTrainer documentation. https://huggingface.co/docs/trl/main/en/grpo_trainer - cited by [1] as its reference [39] (the trl library) as the source the paper follows for its KL coefficient default of 0.04; this card notes only what [1] attributes to this source, not trl's own current default value. Fetched 2026-08-09.

[7] R1-VL official repository README. https://github.com/jingyi0000/R1-VL - code availability, training/data-preparation instructions for both phases, dataset and prompt links, RL launch script paths (`src/r1-vl/run_grpo_2b_vllm.sh`, `src/r1-vl/run_grpo_7b.sh`), dependence on R1-V for the RL stage. Fetched 2026-08-09 as the raw README at commit `679be5cf0d04d852f0eb3856e6a9b36dbf1c0558` (HEAD of `main` at fetch time); the repository takes no revision parameter for casual browsing, so this citation covers only that commit, not the live default branch.

[8] R1-V repository (StarsfieldAI/R1-V, formerly Deep-Agent/R1-V), the codebase the official StepGRPO RL implementation is built on. https://github.com/Deep-Agent/R1-V - README script listing (`run_grpo.sh`, `run_grpo_vllm.sh`, `run_grpo_vllm_qwen25vl.sh`) and `src/r1-v` source-directory listing, checked to confirm it is a standalone script-based GRPO setup, not a call into a packaged trainer. Fetched 2026-08-09 at commit `e35f97e566357631a2602accdd79610b752d8f0b` (HEAD of `main` at fetch time); the repository's own trl dependency, if any, was not confirmed before the `setup.py` fetch timed out, so this card does not claim it does or does not depend on trl.

[9] trl GRPOTrainer documentation, `reward_funcs` interface. https://huggingface.co/docs/trl/main/en/grpo_trainer - confirms the trainer accepts custom reward functions and combines multiple functions' outputs, the interface a from-scratch StepGRPO build would extend. Fetched 2026-08-09 (same page as [6]).
