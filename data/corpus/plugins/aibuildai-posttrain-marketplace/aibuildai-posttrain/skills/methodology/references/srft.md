# SRFT

Single-stage fine-tuning that adds an SFT term and an off-policy demonstration-augmented GRPO term to an on-policy GRPO rollout loss, with two entropy-derived scalars that automatically damp the SFT term when the policy is uncertain and damp the on-policy positive-sample term when the policy is over-confident.

**SRFT** (Supervised Reinforcement Fine-Tuning) is a single-stage method for LLM reasoning that unifies SFT and RL through entropy-aware weighting mechanisms, defined in Fu et al.'s paper of the same title [1]. Its parent is GRPO, the group-relative, value-network-free variant of PPO introduced by Shao et al. [2]; SRFT keeps GRPO's group-normalized advantage but augments every rollout group with demonstration responses, following an off-policy RL design similar to LUFFY [3], and adds an SFT loss and a self-exploration RL loss on top, each scaled by a policy-entropy-derived weight [1]. The paper motivates the design with two claims from its own analysis: SFT makes coarse-grained, global changes to the token distribution while RL makes fine-grained, selective changes, and policy entropy is a useful indicator for balancing the two [1]; running SFT and RL as two sequential stages instead shows an asymmetric failure mode in the paper's own preliminary experiment - SFT after RL (RL-then-SFT) consistently yields suboptimal performance and its entropy plateaus after about 90 training steps, showing limited capacity for further learning, while RL after SFT (SFT-then-RL, the ordering existing methods use) achieves substantial gains [1]. The right paper was confirmed by an exact-title arXiv search for "SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning" returning a unique arXiv:2506.19767 match, cross-checked against the OpenReview submission linked from the authors' own GitHub README [4].

No landmark production system was found citing or adopting SRFT: a Semantic Scholar query for citations of arXiv:2506.19767 was read for its first 50 results (of a shortlisted total of 86 citing papers), and those 50 were all 2025-2026 research papers that compare against or build on it, none of them a named production training system - this card's "no landmark adopter" finding covers only that first batch, not the remaining ~36 citing papers [5]. In the paper's own main results table (Table 2), SRFT scores a 59.5 in-distribution average (AIME24 35.3, AMC 74.3, MATH500 89.8, Minerva 39.7, Olympiad 58.3) and a 62.5 out-of-distribution average (ARC-C 85.3, GPQA-Diamond 46.4, MMLU-Pro 55.9), each the best or second-best of every baseline in the table [1, Table 2]. The same table's own prose, directly below it, instead states a repeated 59.1% in-distribution average with a "+9.0 points over the best baseline" margin, a "+4.8" margin over SFT baselines, and a "+3.4" margin over other SFT+RL baselines, and a 62.5% OOD average with a "+4.7" margin [1, Table 2 discussion] - so Table 2's own row (59.5) and Table 2's own prose (59.1) disagree with each other on SRFT's in-distribution average. Table 3, a separate ablation table further in the paper, reports its own full-SRFT row as 35.3/72.2/89.8/39.7/58.3, average 59.1 - matching the prose's 59.1 but not Table 2's own AMC score of 74.3 [1, Table 3]. This card cites Table 2's per-benchmark row for the headline in-distribution/OOD scores above, and flags that the paper's own prose and its own Table 3 both instead carry 59.1. Lineage: PPO [6] -> GRPO [2] -> off-policy demonstration augmentation (LUFFY [3], ReLIFT [7]) -> SRFT [1], with no successor method identified within the first 50 of 86 citing papers checked [5].

**When to pick it**: pick SRFT when you have both a demonstration dataset (prompt plus a reference reasoning trace, e.g. DeepSeek-R1 traces) and the ability to score fresh rollouts with a verifiable reward, and you want a single training run instead of an SFT-then-RL pipeline - the paper's own preliminary experiment found sequential SFT-then-RL entropy-plateaus and stops improving once RL has already run first [1]. Prefer plain GRPO [2] if you have no demonstrations at all. Prefer LUFFY [3] if you want the off-policy demonstration augmentation without the entropy-weighted SFT/self-exploration terms SRFT adds on top. Prefer SFT alone (no RL) if you only need to imitate the demonstrations and have no reward signal.

**Variant of**: GRPO [2], via an off-policy demonstration-augmentation design similar to LUFFY [3].

**Data it needs**: a demonstration set of (prompt, reference reasoning trace) pairs plus a scalar-checkable ground truth for each prompt. The paper trains on OpenR1-Math-46k-8192, 46,000 NuminaMath-1.5 problems with DeepSeek-R1-generated responses, filtered by Math-Verify to exclude unverifiable answers or responses over 8,192 tokens [1]; the Hugging Face copy's first-row schema exposes columns `data_source`, `prompt` (list of role/content turns), `target` (the reference reasoning trace, list of role/content turns), `ability`, `reward_model` (`ground_truth`, `style`), and `extra_info` (`index`, `split`) [8]. Training ran 500 steps with 8 rollouts sampled per prompt at each step [1]. The demonstration/SFT and off-policy-demonstration terms are offline (reused per step from the fixed dataset); the self-exploration RL term is on-policy, sampled fresh from the current policy each step [1].

**Extra models**: none required by the method's own definition beyond the policy itself - GRPO's group-normalized advantage needs no value network, and the paper's off-policy demonstration term substitutes a fixed behavior-policy probability of 1 instead of loading a separate model to score demonstrations [1]. A reward model is only needed if the reward is not a verifiable checker; the paper uses a rule-based 1/correct, 0/incorrect reward, not a learned reward model [1]. The authors' own reference training script explicitly disables the reference-model path (`actor_rollout_ref.ref.use_ref=False`, `actor_rollout_ref.actor.use_kl_loss=False`, `kl_loss_coef=0.00`) [9], so the actual run that produced the paper's numbers loads no reference model either. See Cost.

**Shipped by**: no mainline framework (trl, upstream verl) implements SRFT as of this check. The only implementation found is the authors' own repository `fyqqyf/SRFT` [4], a full fork of verl with a custom `mix_src` submodule (`main_mix_ppo.py`, `mix_actor.py`, `mix_core_alg.py`, `mix_trainer.py`, `mix_vllm_rollout.py`, `math_verify_reward.py`, `reward_with_format.py`, `rl_dataset_with_target.py`) [10], entry point `python3 -m verl.mix_src.main_mix_ppo` [9]; the repository carries no stated support status (no maintenance/experimental label in its README) and had 59 stars at the time of this check [4]. Building it on a framework that does not vendor this code would mean writing a new combined loss (SFT term + off-policy PPO/GRPO-style term + entropy-weighted self-exploration term) on top of an existing GRPO trainer, plus a data path that feeds demonstration rows into the same rollout batch as on-policy samples - a loss-and-batching change on top of an existing trainer, not a new sampling loop, since it reuses ordinary GRPO-style rollouts.

## How it works

Each step: sample G on-policy rollouts per prompt from the current policy, augment that group with the prompt's demonstration response(s), compute one group-relative advantage over the augmented group, then update the policy on three loss terms - an SFT term and an off-policy RL term on demonstrations, and an entropy-weighted self-exploration RL term on the on-policy rollouts - each carrying its own entropy-derived weight [1].

**Augmented group and advantage.** The rollout group is augmented with demonstrations before advantage computation:

$$ G_{\text{aug.}} = \{(\bm{x}_i,\bm{y}_i)\}_{i=1}^{|G_{\text{roll.}}|} \cup \{(\bm{x}_j,\bm{y}_j)\}_{j=1}^{|G_{\text{demo.}}|} $$

$$ \hat{A}_k = \frac{r(\bm{x},\bm{y}_k) - \operatorname{mean}(\{r(\bm{x},\bm{y}_k)\}_{k=1}^{|G_{\text{aug.}}|})}{\operatorname{std}(\{r(\bm{x},\bm{y}_k)\}_{k=1}^{|G_{\text{aug.}}|})} $$

where $G_{\text{roll.}}$ is the on-policy rollout group and $G_{\text{demo.}}$ is the demonstration group added to it [1]. Because demonstration responses tend to score higher reward, adding them raises the advantage of the whole augmented group, which the paper describes as promoting optimistic exploration [1].

**SFT term on demonstrations**, entropy-weighted:

$$ w_\text{SFT} = 0.5 \cdot \texttt{stop\_grad}\!\left(\exp(-\mathcal{H}(\pi_\theta))\right), \qquad \mathcal{L}_{\text{SFT}}^{\text{demo.}}(\theta) = w_\text{SFT} \cdot \mathbb{E}_{(\bm{x},\bm{y})\sim\mathcal{D}_{\text{demo.}}}\left[-\log\pi_\theta(\bm{y}|\bm{x})\right] $$

$\mathcal{H}(\pi_\theta)$ is the current policy's entropy; when it is high (the policy is uncertain) $w_\text{SFT}$ shrinks toward 0, so the SFT loss's influence is diminished exactly when the demonstration's behavior policy is most likely to be mismatched with the current one [1]. `stop_grad` means the weight itself carries no gradient - only the log-likelihood term is optimized [1].

**Off-policy RL term on demonstrations**, a GRPO/PPO-style clipped surrogate with the behavior policy $\pi_\beta$ substituted for the old policy:

$$ \mathcal{L}_{\text{RL}}^{\text{demo.}}(\theta) = -\mathbb{E}_{(\bm{x},\bm{y})\sim\mathcal{D}_{\text{demo.}}}\left[\min\left\{r_{k,t}(\theta)\cdot\hat{A}_k,\ \operatorname{clip}(r_{k,t}(\theta),1-\epsilon,1+\epsilon)\cdot\hat{A}_k\right\}\right], \qquad r_{k,t}(\theta)=\frac{\pi_\theta(\bm{y}_{k,t}|\bm{x}_t)}{\pi_\beta(\bm{y}_{k,t}|\bm{x}_t)} $$

Following LUFFY and ReLIFT, the paper sets $\pi_\beta = 1$ to avoid recomputing behavior-policy probabilities for off-the-shelf demonstration data, and drops the clip entirely because clipping becomes unstable once $\pi_\beta=1$ [1].

**Self-exploration RL term.** The paper derives this from a decomposition of the standard REINFORCE-style objective under a stated binary reward $\{1,-1\}$ into a positive-sample and a negative-sample piece [1]:

$$ \mathcal{L}_{\text{RL}}^{\text{self-rollout}} = -\mathbb{E}_{\bm{x}\sim\mathcal{D},\bm{y}\sim\pi_\theta}\left[R(\bm{x},\bm{y})\log\pi_\theta(\bm{y}|\bm{x})\right] = \underbrace{\mathbb{E}_{\bm{y}^+\sim\pi_\theta}\left[-\log\pi_\theta(\bm{y}^+|\bm{x})\right]}_{\text{positive sample}} + \underbrace{\mathbb{E}_{\bm{y}^-\sim\pi_\theta}\left[\log\pi_\theta(\bm{y}^-|\bm{x})\right]}_{\text{negative sample}} $$

An entropy weight is then applied only to the positive-sample term:

$$ w_\text{RL} = 0.1 \cdot \texttt{stop\_grad}\!\left(\exp(\mathcal{H}(\pi_\theta))\right), \qquad \mathcal{L}_{\text{RL}}^{\text{self-rollout}}(\theta) = w_\text{RL}\cdot\mathbb{E}_{\bm{y}^+\sim\pi_\theta}\left[-\log\pi_\theta(\bm{y}^+|\bm{x})\right] + \mathbb{E}_{\bm{y}^-\sim\pi_\theta}\left[\log\pi_\theta(\bm{y}^-|\bm{x})\right] $$

Unlike $w_\text{SFT}$, this weight grows with entropy: self-exploration drives entropy down rapidly toward deterministic outputs, so the paper damps the positive-sample (imitation-like) term when entropy is already low, to preserve exploration diversity [1]. Note the reward the paper actually trains with in its appendix is $\{1,0\}$ (correct/incorrect), not the $\{1,-1\}$ assumed for the decomposition above [1]; this card states both facts as written rather than reconciling them.

**Total loss**:

$$ \mathcal{L}_{\text{SRFT}}(\theta) = \mathcal{L}_{\text{SFT}}^{\text{demo.}}(\theta) + \mathcal{L}_{\text{RL}}^{\text{demo.}}(\theta) + \mathcal{L}_{\text{RL}}^{\text{self-rollout}}(\theta) $$

Worked example for the entropy weights only, at a representative per-token entropy of $\mathcal{H}=1.0$ nat: $w_\text{SFT} = 0.5\cdot e^{-1.0}\approx 0.184$, so the demonstration SFT loss is scaled down to about 18% of its unweighted value; $w_\text{RL} = 0.1\cdot e^{1.0}\approx 0.272$, scaling the self-exploration positive-sample loss to about 27%. At a lower entropy of $\mathcal{H}=0.2$, $w_\text{SFT}\approx 0.5\cdot0.819\approx0.409$ (demonstrations trusted more) while $w_\text{RL}\approx0.1\cdot1.221\approx0.122$ (self-exploration imitation damped more) - the two weights move in opposite directions as entropy falls, which is the paper's stated intent [1].

## Cost

**Theory, from the method's own math:**

- Time: no value network forward/backward (inherited from GRPO [2]), but three loss terms are computed per step instead of GRPO's one - an SFT pass over demonstrations, an off-policy RL pass over the same demonstrations, and an on-policy RL pass over freshly sampled rollouts - plus generation for the G on-policy rollouts per prompt each step [1]. The off-policy terms reuse a fixed dataset (no extra generation), so the added generation cost over plain GRPO is limited to whatever demonstrations are concatenated into the token batch, not extra sampling.
- Memory: the method's own definition needs no value network and, with $\pi_\beta=1$, no separate behavior-policy model to score demonstrations [1] - so in theory SRFT need hold no more trained/frozen models than the policy itself. A naive reading might assume a frozen reference model is required for KL control the way PPO uses one; the paper's formulas carry no reference-model term at all, and the authors' own training run disables the reference-model path entirely (below) [9].

**In practice, per framework:**

- Authors' `fyqqyf/SRFT` fork of verl [4], reference training script `exp_scripts/train.sh` [9]: run on a single setup of 4 nodes x 8 A100-80GB GPUs (`trainer.n_gpus_per_node=8`, `trainer.nnodes=4`) [9]. The script sets `actor_rollout_ref.ref.use_ref=False`, `actor_rollout_ref.actor.use_kl_loss=False`, and `algorithm.kl_ctrl.kl_coef=0.00`, so no reference model is loaded and no KL penalty is computed in the run that produced the paper's numbers [9]. The SFT loss is toggled with `+actor_rollout_ref.actor.use_sft_loss=True` and `+actor_rollout_ref.actor.sft_loss_coef=-0.5`, and the off-policy demonstration loss with `actor_rollout_ref.actor.use_off_policy_loss=True` and `off_policy_reshape="p_div_p_0.1"` [9] - both add compute on top of a standard verl GRPO actor update rather than a new model. Rollout generation runs through vLLM (`actor_rollout_ref.rollout.name=vllm`) with `gpu_memory_utilization=0.8` and tensor-parallel size 2 [9]; the per-GPU training batch is bounded by `ppo_micro_batch_size=64` with dynamic batching capped at `ppo_max_token_len_per_gpu=32768` [9], the same batching knobs verl's GRPO recipe uses (see the verl framework card) - the extra cost from SRFT's own terms is additional forward/backward passes over the demonstration batch, not a wider per-token activation footprint.
- The repository's `compute_token_on_off_policy_loss` in `mix_core_alg.py` (commit `8d4e10a`) applies `pos_entropy_exp_coeff = 0.1 * entropy.exp().detach()` to the on-policy positive-sample loss, matching the paper's $w_\text{RL} = 0.1\cdot\texttt{stop\_grad}(\exp(\mathcal{H}))$ term in form [11]. Its SFT-side term differs from the paper: the function computes `entropy_exp_coeff = entropy.exp().detach()` and applies it as `-entropy_exp_coeff * log_prob`, with no 0.5 factor and a positive-entropy exponent rather than the paper's $w_\text{SFT}=0.5\cdot\texttt{stop\_grad}(\exp(-\mathcal{H}))$ [1][11]. Separately, the coefficient this function's `sft_loss` output is scaled by is applied one level up in `mix_actor.py` (commit `8d4e10a`), which computes `policy_loss - sft_loss * self.sft_loss_coef`; with the launch script's `sft_loss_coef=-0.5` [9] this yields a net `+0.5 * sft_loss` contribution, matching the paper's 0.5 magnitude and sign despite the flag's own negative value [12].

## How to use it

- Build a demonstration dataset of (prompt, high-quality reasoning trace, ground-truth answer) triples; the paper's own set is OpenR1-Math-46k-8192, DeepSeek-R1 traces on NuminaMath-1.5 problems filtered by Math-Verify for verifiability and an 8,192-token cap [1], with columns `prompt`, `target`, `reward_model.ground_truth` in the published copy [8].
- Reward convention: the paper's actual training reward is rule-based and binary, 1 if the final answer verifies as correct and 0 otherwise, checked with Math-Verify and OAT-Grader at evaluation time [1] - not the $\{1,-1\}$ reward assumed in the self-exploration loss's own derivation (see How it works).
- Chat template: the paper uses a single system prompt that encourages step-by-step reasoning across SFT, RL, and SRFT runs, following the same template choice as LUFFY and ReLIFT [1].
- Key knobs, paper values versus the authors' own reference run (no other framework ships this method, so there is only one implementation column):

| knob | paper (main text / appendix) [1] | authors' `train.sh` [9] |
| --- | --- | --- |
| rollouts per prompt (G) | 8 | `rollout.n=8` |
| training steps | 500 | not stated as a step count (script sets `trainer.total_epochs=30` instead) |
| RL learning rate | fixed at $1\times10^{-6}$ | `actor.optim.lr=1e-6` |
| SFT learning rate (pre-training stage baselines) | $5\times10^{-6}$ | not applicable - this script trains SRFT jointly, not the separate SFT baseline |
| $w_\text{SFT}$ coefficient | 0.5 (times $\exp(-\mathcal{H})$) | `sft_loss_coef=-0.5`, net effect `+0.5` once combined with `mix_actor.py`'s subtraction (see Cost) |
| $w_\text{RL}$ coefficient | 0.1 (times $\exp(\mathcal{H})$) | not exposed as a separate flag; folded into `mix_core_alg.py`'s fixed `0.1` factor [11] |
| clip range $\epsilon$ (on-policy term) | not stated as a numeric value | not checked |
| max response length | 8,192 tokens | `data.max_response_length=8192` |
| RoPE theta / context window | extended 10,000 -> 40,000, window to 16,384 | not checked in this script (handled at model-conversion time, before this script runs) [1] |
| batch size | not stated for the joint SRFT run (SFT-only baseline uses 128) | `data.train_batch_size=128`, `ppo_mini_batch_size=64` |

- Group-size and demonstration-mix trade-off: every rollout group already carries the demonstration response(s) added on top of the G on-policy samples, so raising G raises on-policy generation cost the same way it does for plain GRPO, while the demonstration side of the batch is reused from a fixed dataset each step and does not scale with G [1].

## While it runs

- Signals and their healthy shapes: the paper's own entropy-dynamics analysis is this method's central live signal - it reports that plain RL exhibits a rapid entropy decline toward deterministic outputs, while SRFT "maintains more stable entropy," which the paper reads as evidence the policy keeps exploring during training [1]. The paper also separately tracks training reward and response length as standard RL training-dynamics signals during SRFT runs [1], though it gives no numeric healthy range for either.
- Published reference runs: the paper's own Figure 6 plots training rewards, response lengths, and training entropy for SRFT versus plain RL over the run [1]; no raw log files were located outside the paper's own figures during this check [1].
- Degeneracies and defaults: a group with a single response (G=1, no group at all) or a group where every reward is identical gives a zero-std normalization for $\hat{A}_k$, the same degeneracy that affects plain GRPO's advantage formula [1][2] - the paper does not separately discuss a guard for this case for SRFT. The authors' reference training run sets `use_kl_loss=False`, `use_ref=False`, and `kl_ctrl.kl_coef=0.00` [9], meaning nothing in that run bounds drift from a reference policy - unlike GRPO's own optional KL term [2], the run that produced the paper's numbers has no reference-model regularization at all.
- Named successors: none identified. The Semantic Scholar citations list for arXiv:2506.19767 (~50 entries, checked at the time of this card) contains only papers that cite or compare against SRFT, none naming themselves as a direct successor method [5].
- Known failure modes: the paper's own Limitations paragraph states that its entropy-based control uses "relatively simple" exponential weighting functions, and that the method assumes access to high-quality demonstrations, leaving open how it degrades with imperfect demonstrations [1]. A live query of the `fyqqyf/SRFT` GitHub issues API (`state=all`) returned zero open or closed issues, so no maintainer-reported failure mode exists there to cite [13].
- What the gain is - and is not: the paper attributes SRFT's in-distribution gain over pure RL baselines to combining demonstrations (coarse-grained behavior-policy approximation) with self-exploration (fine-grained refinement), and its OOD gain to the same combination improving generalization [1]; the paper does not claim the method adds reasoning capability beyond what is reachable by SFT plus RL individually - it argues the single-stage integration realizes more of that combined capability than running the two stages sequentially [1].

## Sources

[1] Fu et al., "SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning", 2025. https://arxiv.org/abs/2506.19767 - defines SRFT: loss terms, entropy weights, reward design, training/evaluation setup, main results (Table 2), sequential-integration ablation (Table 1), weighting-mechanism ablation (Table 3), limitations. Fetched 2026-08-09 (LaTeX source via arxiv.org/src/2506.19767, plus the abstract page).

[2] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO, the parent method. Fetched 2026-08-09 (abstract page).

[3] Yan et al., "Learning to Reason under Off-Policy Guidance" (LUFFY), 2025. https://arxiv.org/abs/2504.14945 - off-policy demonstration-augmented GRPO design SRFT follows; also the source of the OpenR1-Math-46k-8192 training dataset. Fetched 2026-08-09 (abstract page).

[4] `fyqqyf/SRFT` GitHub repository. https://github.com/fyqqyf/SRFT - README (ICLR 2026 acceptance note, OpenReview link, verl/vLLM dependencies, citation block), repository file tree, `mix_src` module listing. Star count (59) and metadata (`open_issues_count: 0`) read live via the GitHub repo API, not tied to a commit; code/file-listing claims are pinned at commit `8d4e10a586266a27359086188f78303e320a9266` (see [9]-[12]). Fetched 2026-08-09.

[5] Semantic Scholar citations list for arXiv:2506.19767. https://api.semanticscholar.org/graph/v1/paper/arXiv:2506.19767/citations - checked for landmark adopters and named successors; returned approximately 50 citing papers, all 2025-2026 research papers, none a named production system or explicit successor method. Fetched 2026-08-09 (live API query; not a fixed snapshot).

[6] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, GRPO's parent, cited for the lineage line. Fetched 2026-08-09 (abstract page).

[7] Ma et al., "Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions" (ReLIFT), 2025. https://arxiv.org/abs/2506.07527 - interleaved online fine-tuning baseline compared against SRFT in Table 2. Fetched 2026-08-09 (abstract page).

[8] Hugging Face datasets-server first-rows API for `Elliott/Openr1-Math-46k-8192`. https://huggingface.co/datasets/Elliott/Openr1-Math-46k-8192 - column schema (`data_source`, `prompt`, `target`, `ability`, `reward_model`, `extra_info`) and dataset README (confirms it is LUFFY's training dataset). Fetched 2026-08-09 (live dataset viewer API and README; not pinned to a dataset revision).

[9] `fyqqyf/SRFT` repository, `exp_scripts/train.sh`, read at commit `8d4e10a586266a27359086188f78303e320a9266` (`main` branch HEAD at fetch time). https://github.com/fyqqyf/SRFT - the reference training launch script: entry point, cluster size, all `verl`/`mix_src` config flags used for the run reported in the paper. Fetched 2026-08-09 (raw file at the commit above; the repository is a live, unpinned target beyond that commit).

[10] `fyqqyf/SRFT` repository, `srft/verl/verl/mix_src` directory listing, read at commit `8d4e10a586266a27359086188f78303e320a9266`. https://github.com/fyqqyf/SRFT - confirms the module files implementing SRFT's custom loss and rollout logic. Fetched 2026-08-09 (GitHub contents API at the commit above).

[11] `fyqqyf/SRFT` repository, `srft/verl/verl/mix_src/mix_core_alg.py`, read at commit `8d4e10a586266a27359086188f78303e320a9266`. https://github.com/fyqqyf/SRFT - source code for `compute_token_on_off_policy_loss`: the RL-side entropy-weight term matches the paper's formula in form; the SFT-side term differs (no 0.5 factor, opposite entropy sign). Fetched 2026-08-09 (raw file at the commit above).

[12] `fyqqyf/SRFT` repository, `srft/verl/verl/mix_src/mix_actor.py`, read at commit `8d4e10a586266a27359086188f78303e320a9266`. https://github.com/fyqqyf/SRFT - source line `policy_loss - ret_dict["sft_loss"] * self.sft_loss_coef`, showing `sft_loss_coef=-0.5` nets to a `+0.5` contribution. Fetched 2026-08-09 (raw file at the commit above).

[13] `fyqqyf/SRFT` repository issues, GitHub REST API `state=all`, read at fetch time (issue state is not tied to a specific commit). https://github.com/fyqqyf/SRFT/issues - returned zero open or closed issues. Fetched 2026-08-09 (live API query).
