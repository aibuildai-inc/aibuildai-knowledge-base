# WARM

Paper: https://arxiv.org/abs/2401.12187 [1]

Train M reward models with the same pretraining but different fine-tuning hyperparameters, then average their weights (not their logits) into one reward model that costs no more than a single RM at inference time.

**WARM** (Weight Averaged Reward Models) is a reward-modeling method for RLHF, introduced by Ramé et al. of Google DeepMind as a strategy that first fine-tunes multiple RMs, then averages them in the weight space [1]. Its parent is prediction ensembling (ENS) of reward models, the strategy of averaging the output logits of M independently trained RMs, first used for RLHF reliability by Christiano et al.'s original RLHF paper and analyzed further in later reward-ensembling work [2][3][4]; WARM instead initializes each RM from the same supervised-fine-tuned (SFT) checkpoint with a linear-probed classifier head, fine-tunes M copies with diverse hyperparameters, and linearly interpolates the M resulting weight vectors into one reward model [1]. The paper gives three reasons for existing: unlike ENS, weight averaging keeps only one model at inference so it adds no memory or compute overhead; it relies on linear mode connectivity (LMC) so the averaged model is at least as accurate as the interpolation of the individual accuracies; and a theoretical analysis under a simplified feature model shows weight averaging suppresses low-probability, run-specific (often spurious or memorized) features more than prediction averaging does, which the paper argues improves reliability under distribution shift and robustness to noisy preference labels [1].

The paper does not mention any code release, and does not name an external system that adopted WARM; the search performed for this card (fetching the Gemma 2 technical report and checking for the string "WARM") found no mention [5]. The paper's own DeepMind follow-up, WARP (Weight Averaged Rewarded Policies), extends the weight-averaging idea from reward models to policies themselves [6]. In the paper's own summarization experiments (TL;DR dataset, PaLM-XXS reward models, PaLM-XS policy), a policy RL-fine-tuned with WARM (M=6) reaches a 79.4% win rate, judged by an AI preference oracle, against a policy RL-fine-tuned with the single best individual RM, and a 99.8% win rate against the SFT policy after 3500 steps [1]. Lineage in one line: prediction ensembling of RMs (Christiano et al., 2017 [2]) and model soups / linear mode connectivity (2020-2022 [7][8]) -> WARM (2024 [1]) -> WARP (2024 [6]).

**When to pick it**: pick WARM when you are training a reward model for RLHF from a shared pretrained/SFT checkpoint, want the reliability benefits of ensembling multiple RMs, and cannot afford to keep M models loaded at inference time for reward scoring during RL [1]. Its nearest offline alternative is a single reward model trained once by minimizing the Bradley-Terry negative log-likelihood, the standard RLHF recipe WARM builds on [1][9]; its nearest online neighbor is prediction ensembling (ENS), which averages the logits of M separately trained RMs at inference time instead of their weights, keeping all M models loaded and paying inference cost that grows linearly with M [1][3][4].

**Variant of**: prediction ensembling of reward models [2][3][4], replacing logit averaging with weight averaging in the style of model soups [7], applied to RMs trained with the standard Bradley-Terry loss [1][9].

**Data it needs**: a preference dataset of (prompt, chosen response, rejected response) triples, the same format used to train a single Bradley-Terry reward model [1]. WARM's own experiments used the Reddit TL;DR summarization dataset with AI-generated preference labels from a prompted PaLM-L model, training M reward models for 10k steps each at batch size 128 [1]. It is offline with respect to the reward-modeling step itself: all M RMs fine-tune on the same fixed preference dataset, only varying training hyperparameters, data order, or (for the paper's "Baklava" variant) which SFT checkpoint each RM's featurizer is initialized from [1]. The resulting averaged RM is then used to score fresh, on-policy samples during a downstream RL loop (or to rerank best-of-N samples), the same way any RM would be [1].

**Extra models**: none beyond what the M underlying RM fine-tuning runs each require — each of the M runs starts from the same SFT checkpoint plus a linear-probed classifier head, and the M resulting weight sets are averaged offline into a single deployed RM, so no extra model is held or loaded at reward-inference or RL time [1]. This differs from a value network or a frozen KL-reference policy, which are required by the downstream RL algorithm (e.g., PPO's value network [10], or REINFORCE-with-baseline as used in WARM's own RL experiments [1]), not by WARM itself. Details in Cost.

**Shipped by**: no WARM implementation was found in trl or verl, the two libraries checked for this card; other RLHF/post-training libraries were not searched. Checking trl's exported top-level package (`trl/__init__.py`, `main` branch) and its `RewardTrainer`/callback documentation found no weight-averaging-of-reward-models trainer or callback; the closest object, `BEMACallback`, performs an exponential moving average of the policy model's own weights during training, not offline averaging of multiple independently fine-tuned reward models [11]. verl's exported top-level package (`verl/__init__.py`, `main` branch) likewise has no WARM-named trainer or utility [12]. Building it requires no new loss and no new sampling loop: it is M repeated runs of an existing reward-model trainer (e.g., trl's `RewardTrainer`, which already implements the Bradley-Terry loss [13]) with varied learning rate/dropout/data order, followed by a short offline script that loads each checkpoint's state dict and averages the tensors elementwise before saving one merged checkpoint.

## How it works

The loop is: initialize M reward models from the same (SFT weights, linear-probed classifier) pair; fine-tune each on the same preference dataset with a different random seed / hyperparameter draw; average the M resulting weight vectors elementwise; use the single averaged model as the reward model [1].

**Reward-model loss** (per RM, before averaging), the Bradley-Terry negative log-likelihood [1][9]:

$$ \mathcal{L}_R(r_\phi, \mathcal{D}_{train}) = -\mathbb{E}_{(x,y^+,y^-)\in\mathcal{D}_{train}}\left[\log\sigma\big(r_\phi(x,y^+) - r_\phi(x,y^-)\big)\right] $$

where $r_\phi(x,y)$ is the scalar reward the model with weights $\phi$ assigns to prompt $x$ and generation $y$, $y^+$ is the preferred generation, $y^-$ the dispreferred one, and $\sigma$ is the logistic function [1]. Each of the M runs minimizes this same loss on the same $\mathcal{D}_{train}$, differing only in training hyperparameters, data order, or initialization checkpoint [1].

**Weight averaging** [1]:

$$ \phi^{\text{WARM}} = \frac{1}{M}\sum_{i=1}^{M} \phi_i $$

**Linear mode connectivity**, the empirical property that licenses this averaging [1]: for two fine-tuned weight vectors $\phi_1,\phi_2$ sharing a pretraining, and any $\lambda \in [0,1]$,

$$ \mathrm{Acc}\big(r_{(1-\lambda)\phi_1 + \lambda\phi_2}, \mathcal{D}_{test}\big) \geq (1-\lambda)\times\mathrm{Acc}(r_{\phi_1}, \mathcal{D}_{test}) + \lambda\times\mathrm{Acc}(r_{\phi_2}, \mathcal{D}_{test}) $$

i.e. the interpolated model's pairwise accuracy is at least as good as the linear interpolation of the two individual accuracies [1]. The paper attributes this to the shared pretraining constraining the fine-tuning runs to a convex region of the loss landscape, and notes it does not hold for weights trained from scratch even with a shared random initialization [1].

**Diversity sources.** LMC requires shared pretraining, but WARM also needs the M runs to differ, since diversity across fine-tuned weights is what drives the accuracy gains of weight averaging [1]. The paper uses three sources: different orderings of the same training data across runs; different sampled hyperparameters (learning rate drawn from {1e-5, 4e-5, 1e-4}, dropout from {0.05, 0.1}) [1]; and a new initialization scheme the paper calls "Baklava," which initializes each RM's featurizer from a different checkpoint collected along one SFT trajectory, relaxing model soups' requirement of an identical initialization down to only a shared pretraining [1][7]. The paper reports that initializing from early SFT checkpoints instead of the final one gave a worse accuracy-diversity trade-off, so WARM's main experiments use only the last SFT checkpoint for every RM [1].

**Worked micro-example (why weight averaging beats prediction averaging under label noise).** The paper's theoretical analysis assumes a binary classifier $r=\omega^\top f$ over $F$ orthogonal features, where feature $j$ is learned by a given fine-tuning run with probability $p_j$ [1]. As the number of averaged models $M \to \infty$, prediction ensembling's coefficient on feature $j$ tends to $p_j$, while weight averaging's coefficient tends to $p_j^2$ [1]. Concretely, a feature learned by only 30% of runs ($p_j=0.3$) keeps a weight of 0.3 under ENS but shrinks to 0.09 under WA, while a feature learned by 90% of runs keeps 0.9 under ENS versus 0.81 under WA — WA suppresses low-probability, run-specific features (the paper's proxy for memorized or spurious signal from noisy labels) much more than the frequently-learned, run-invariant ones [1].

## Cost

**Theory, from the method's own math:**

- Time: M full reward-model fine-tuning runs on the same preference dataset, each the same cost as training one ordinary Bradley-Terry RM, plus one cheap elementwise weight-average over the M checkpoints at the end [1]. This is M times the compute of training a single RM, but the M runs are independent and embarrassingly parallel — the paper notes this "updatable machine learning" property removes the need for inter-server communication between runs [1].
- Memory: at reward-inference and RL time, exactly one RM's weights are held, the same footprint as a single ordinary RM [1]. This is the point of the method: prediction ensembling (ENS) instead keeps all M RMs loaded, so ENS's inference memory and compute scale linearly with M [1]. Naive readings that assume WARM's training-time cost is also "free" are wrong: the M-fold training cost is real and is paid once, before the RL loop starts.

**In practice, per framework:** no shipping framework was found (see Shipped by), so no framework-specific cost claim can be made here; a practitioner using trl's `RewardTrainer` [13] to build WARM manually would pay trl's own per-run reward-training cost M times, a detail that belongs to the trl reward-trainer skill, not to this card.

## How to use it

- Data preparation: the same preference-triple format as any Bradley-Terry reward model — (prompt, chosen response, rejected response) — with no extra columns required by WARM itself [1][9]. WARM's own preference labels were produced by prompting a PaLM-L model with chain-of-thought (an RLAIF-style setup), rather than by human annotation, though the method is agnostic to the labeling source [1][14].
- Reward/label convention: same as the underlying RM loss — the model is trained so $\sigma(r_\phi(x,y^+)-r_\phi(x,y^-))$ is high, i.e., higher scalar reward for the preferred completion [1][9].
- Key knobs, with the paper's own values as the only published anchor (no framework ships this method, so no framework-default column exists):

| knob | paper value [1] |
| --- | --- |
| number of averaged RMs (M) | 6 in main RL/BoN results; swept 1-10 in ablations |
| RM fine-tuning steps | 10,000 |
| RM training batch size | 128 |
| RM learning rate (sampled per run) | one of {1e-5, 4e-5, 1e-4} |
| RM dropout (sampled per run) | one of {0.05, 0.1} |
| RM optimizer | Adafactor |
| classifier head init | linear-probed (not random) |
| downstream RL policy learning rate | 1e-5 |
| downstream RL KL coefficient $\alpha$ (clean labels) | 0.003 |

- Trade-offs a run designer faces: raising M gives a more reliable, more label-noise-robust RM and, per the paper's ablations, pushes the reward-hacking collapse later and to a higher control reward, but multiplies reward-model training cost linearly with M and only pays off once the M runs are diverse enough (identical hyperparameters and identical SFT-checkpoint initialization erode the accuracy gain) [1]. The Baklava initialization diversifies runs "for free" (no extra compute versus using the final SFT checkpoint for all M runs) but requires having saved multiple SFT checkpoints along one SFT trajectory in the first place [1].

## While it runs

- Signals and their healthy shapes: the paper's own monitoring signal is the "control reward" — a separate, larger, disjointly-pretrained RM used only for evaluation, not for RL — plotted against RL training steps or against the KL between the RL policy and its SFT initialization; a healthy run keeps the control reward rising, and reward hacking shows up as the control reward eventually declining while the proxy (WARM) reward keeps climbing [1]. The paper reports this control RM reaches 80.1% accuracy on its own out-of-distribution test set, and states that increasing M delays the point where the control reward collapses and raises its peak value [1].
- Published reference runs: the paper's own Figure 8/9 curves are the reference: WARM with M=6 reaches a 99.8% win rate against the SFT policy after 3500 RL steps, and a 79.4% win rate against a policy RL-trained with the single best individual RM after 3000 steps, both judged by the same AI preference oracle used for training labels [1]. No raw logs are published alongside the paper.
- Degeneracies and defaults: LMC (and thus the accuracy benefit of averaging) requires the M RMs to share the same pretraining; the paper states LMC does not hold for weights trained from scratch even with a shared random initialization, so averaging RMs from different pretrained bases is a documented way to break the method [1]. Using a random classifier-head initialization instead of linear probing is also flagged as harmful to LMC, since it risks feature distortion during fine-tuning [1]. Making all M runs identical (same hyperparameters, same data order, same checkpoint) removes the diversity the paper says is required for WA's accuracy gains [1].
- Named successors: WARP (Weight Averaged Rewarded Policies) extends the weight-averaging idea from reward models to the RL-trained policies themselves, aiming to better trade off KL regularization against reward optimization [6].
- Known failure modes, from the paper's own limitations section: compared to prediction ensembling, WARM cannot benefit from RMs of different architectures or pretrainings (all M RMs must share pretraining for LMC), and cannot expose prediction disagreement across RMs as an uncertainty signal [1]. The paper also states that WARM does not eliminate all forms of spurious correlation — if every individual RM relies on a shared spurious feature such as summary length, WARM is likely to replicate that reliance rather than correct it [1]. No search of a library's issue tracker was performed, since no library implements WARM.
- What the gain is - and is not: the paper's own claim is that WARM improves the reliability of the RM under distribution shift and its robustness to noisy preference labels, at no extra inference cost versus a single RM [1]; it is explicitly not a fix for the other challenges in RLHF (e.g., reward misspecification from the human-preference collection process itself), and the paper states WARM must be considered within a larger responsible-AI context rather than as a complete solution to reward hacking or misalignment [1].

## Sources

[1] Ramé et al., "WARM: On the Benefits of Weight Averaged Reward Models", 2024. https://arxiv.org/abs/2401.12187 - defines WARM: procedure, LMC property, diversity sources, theoretical p_j vs p_j^2 analysis, TL;DR experimental setup and results, limitations. Confirmed as the correct paper via a single-candidate exact title match against the shortlisted title. Fetched 2026-08-09 (ar5iv HTML full text and PDF text extraction).

[2] Christiano et al., "Deep Reinforcement Learning from Human Preferences", 2017. https://arxiv.org/abs/1706.03741 - the original RLHF paper, cited by WARM as the first to use prediction ensembling of reward models. Fetched 2026-08-09.

[3] Lakshminarayanan, Pritzel, and Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", 2017. https://arxiv.org/abs/1612.01474 - the prediction-ensembling reference WARM cites for ENS. Fetched 2026-08-09.

[4] WARM's own related-work discussion of reward-model ensembling for hacking mitigation (Eisenstein et al. and Coste et al., cited as references 41-42 in [1]) - cited via [1]; not independently fetched.

[5] Gemma Team, "Gemma 2: Improving Open Language Models at a Practical Size", 2024. https://arxiv.org/abs/2408.00118 - checked for any mention of WARM; none found in the full HTML text. Fetched 2026-08-09.

[6] Ramé et al., "WARP: On the Benefits of Weight Averaged Rewarded Policies", 2024. https://arxiv.org/abs/2406.16768 - named successor extending weight averaging to policies. Fetched 2026-08-09 (abstract page).

[7] Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time", 2022. https://arxiv.org/abs/2203.05482 - the weight-averaging technique WARM applies to reward models. Fetched 2026-08-09.

[8] Frankle, Dziugaite, Roy, and Carbin, "Linear Mode Connectivity and the Lottery Ticket Hypothesis", 2020. https://arxiv.org/abs/1912.05671 - one of the two LMC references WARM's Observation 1 relies on. Fetched 2026-08-09.

[9] Bradley and Terry, "Rank analysis of incomplete block designs: I. the method of paired comparisons", Biometrika, 1952 - the pairwise-preference model behind the reward-model loss. Cited via [1]; not independently fetched (pre-arXiv era).

[10] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - the RL algorithm WARM's own downstream RL experiments compare against as a standard choice; WARM's own RL experiments instead use REINFORCE with a baseline [1]. Fetched 2026-08-09.

[11] trl source, `trl/__init__.py` and `trl/trainer/callbacks.py`, GitHub `huggingface/trl`, commit `2396dfe5d2be` (main branch HEAD at fetch time), and the `RewardTrainer` documentation page. https://github.com/huggingface/trl - checked for a WARM or weight-averaging-of-reward-models trainer/callback; none found. Fetched 2026-08-09 (raw source files pinned to the above commit; docs page is an unpinned, dated reading).

[12] verl source, `verl/__init__.py`, GitHub `volcengine/verl`, commit `4a2cba76f7f6` (main branch HEAD at fetch time). https://github.com/volcengine/verl - checked for a WARM-named trainer or utility; none found. Fetched 2026-08-09 (raw source file pinned to the above commit).

[13] trl `RewardTrainer` documentation. https://huggingface.co/docs/trl/main/en/reward_trainer - confirms trl's reward trainer implements a Bradley-Terry-style pairwise loss but no weight-averaging step. Fetched 2026-08-09.

[14] Lee et al., "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback", 2023. https://arxiv.org/abs/2309.00267 - the AI-feedback labeling procedure WARM's experiments follow for generating preference labels. Fetched 2026-08-09.
