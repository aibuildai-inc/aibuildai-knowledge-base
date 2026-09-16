# URM

A reward model whose value head outputs a Gaussian distribution over each preference attribute instead of one fixed score, so per-prediction uncertainty can be measured and used to filter unreliable reward signals in downstream alignment.

**URM** (Uncertainty-aware Reward Model) is a reward-model training method introduced in "Uncertainty-aware Reward Model: Teaching Reward Models to Know What is Unknown," which proposes URM and an ensemble variant URME to address aleatoric and epistemic uncertainty in reward modeling [1]. Its parent is the multi-attribute reward model, exemplified by Nemotron-4 340B's reward model, which maps a response to several human-preference attribute scores (e.g. helpfulness, coherence) instead of one scalar [2]. URM replaces the deterministic value head of a multi-attribute RM with a probabilistic one: it takes the base model's last hidden state and outputs a mean and log-standard-deviation per attribute, parameterizing a normal distribution from which the attribute score is treated as sampled, and the aleatoric uncertainty is read off as the variance of that distribution [1]. The paper gives three reasons for this: point-wise reward models cannot represent that human preferences are inherently probabilistic rather than deterministic; standard reward models cannot assess the reliability of their own predictions; and a per-attribute distribution lets an ensemble of such models (URME) further quantify epistemic uncertainty from disagreement between ensemble members, so unreliable predictions can be identified and filtered [1]. The paper is at https://arxiv.org/abs/2410.00847 [1]. Because the name "Uncertainty-aware Reward Model" is shared with an unrelated paper on proxy-based uncertainty estimation for instruction following, the paper cited here was confirmed as the intended one by being the top-cited match under an exact-title search for the name, well ahead of the same-acronym paper.

URM is not adopted by other named training systems in the source read for this card; its own reported adoption is as a plugged-in reward model for existing alignment pipelines. In the paper's own RewardBench evaluation, URM built on the Skywork reward model as its base, URM(S), scores 92.9 overall against 92.5 for that base model, and URM built on the FsFairX reward model, URM(F), scores 89.9 against 84.4 for its base model [1, Table 1]. Applying URM inside iterative DPO and PPO-based RLHF loops, with uncertainty-based filtering of unreliable reward predictions, was shown to raise win rate over the SFT model relative to using the same reward predictions without filtering [1]. Lineage in one line: multi-attribute reward models such as Nemotron-4's (2024 [2]) -> URM (2024 [1]), read out downstream by iterative DPO [7] and PPO-based RLHF [3].

**When to pick it**: pick URM when a reward model is trained offline on a static, attribute-annotated preference dataset and the downstream loop (best-of-n selection, iterative DPO, or PPO-style RLHF) can use a per-example uncertainty score to discard or penalize unreliable reward predictions [1]. The nearest offline alternative is a standard Bradley-Terry (BT) reward model that outputs one deterministic scalar per response and has no notion of prediction reliability, as used in InstructGPT-style RLHF [3]; a closer offline neighbor is the concurrent QRM, which also models a reward distribution but via quantile regression rather than URM's Gaussian MLE or regression losses [6]. URM does not itself sample fresh completions during training; the "online" step happens one layer up, in the DPO or PPO loop that consumes URM's scores [1].

**Variant of**: multi-attribute reward models, e.g. the reward model in Nemotron-4 340B [2], with the deterministic value head replaced by a probabilistic one [1].

**Data it needs**: for the attribute-regression stage, prompt-response pairs with multi-attribute human ratings; the paper trains on HelpSteer2 [1, App. A.1][5], and the released checkpoint's model card lists the five HelpSteer2 attributes URM regresses against as Helpfulness, Correctness, Coherence, Complexity, and Verbosity [4]. For the second, gating-layer stage, chosen/rejected response pairs are needed to learn how to combine the attribute scores into one reward via a Bradley-Terry loss; the paper uses Skywork-Reward-Preference-80K [1, App. A.1]. Stage 1 trains for one epoch over HelpSteer2; stage 2 trains the gating layer alone for 4000 steps at batch size 256 over Skywork-Reward-Preference-80K [1, App. A.1]. Both stages train on a fixed, pre-collected dataset; URM is offline, not on-policy [1, App. A.1].

**Extra models**: none required to run URM itself beyond its own base LLM, which the paper initializes from an existing reward model - Liu and Zeng (2024) for the Llama3.1 URM and Dong et al. (2023a) for the Llama3 URM [1, Sec. 5.1] - identified in the released checkpoints' model cards as Skywork-Reward-Llama-3.1-8B for URM(S) and FsfairX-LLaMA3-RM-v0.1 for URM(F) [4] - before adding the probabilistic value head. URME, the ensemble variant, needs multiple independently trained URM instances - the paper uses three, differing in random seed, value-head initialization, and training mini-batches [1, Sec. 5.1]; each adds a full extra copy of the base model. No value network, reference model, or judge model is part of URM's own definition; those apply only to whatever downstream RLHF/DPO loop consumes URM's reward.

**Shipped by**: no RL/alignment framework trainer implements URM. The authors released trained checkpoints on the Hugging Face Hub, `LxzGordon/URM-LLaMa-3.1-8B` and `LxzGordon/URM-LLaMa-3-8B`, loadable for inference with `AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)` [4]. Their training code is adapted from the general Bradley-Terry reward-model trainer in the RLHFlow/RLHF-Reward-Modeling repository, whose README makes no mention of URM's probabilistic value head or its losses as of the version fetched here [1, App. A.1][8]; a GitHub repository-search API query for `URM uncertainty-aware reward model` returned `total_count: 0` [9]. **Building URM from scratch on top of an existing sequence-classification reward-model trainer** means swapping its scalar output head for a mean/log-std head per attribute and its loss for the MLE or regression loss below - an architecture-plus-loss change on an existing trainer, not a new sampling loop.

## How it works

Train a probabilistic value head to output a Gaussian over each preference attribute, combine the attributes into one scalar reward, and (for URME) measure the ensemble's disagreement as an uncertainty signal.

**Value head.** For hidden state $h$ of the base model, the head outputs $\mu_i$ and $\sigma_i$ for each attribute $i$, parameterizing $\mathcal{N}(\mu_i, \exp(2\sigma_i))$, from which the preference score for that attribute is treated as sampled; attributes are assumed independent (diagonal covariance) [1, Sec. 4.1].

**Loss 1 - maximum likelihood estimation (MLE)** [1, Eq. 2]:

$$ L_2 = -\mathbb{E}_{x,y\sim D}\left[\sum_{i=0}^{n} \log P_\theta(R_i \mid x, y)\right] $$

$R_i$ is the $i$-th attribute's label, and $\log P_\theta(R_i\mid x,y)$ is the log-probability of that label under $\mathcal{N}(\mu_i, \exp(2\sigma_i))$ [1, Eq. 2]. Worked example for one attribute: if the head outputs $\mu = 3.0$, $\sigma = \log(0.5) \approx -0.693$ (std $=0.5$) and the label is $R=3.4$, the log-density is $-\log(0.5\sqrt{2\pi}) - \frac{(3.4-3.0)^2}{2\cdot 0.25} = -0.226 - 0.32 \approx -0.55$; this term (negated and summed over attributes) is what $L_2$ minimizes.

**Loss 2 - attribute regression with reparameterization** [1, Eq. 3]:

$$ L_3 = \mathbb{E}_{x,y\sim D}\left[\sum_{i=0}^{n}\big(r_i(x,y) - R_i\big)^2\right], \qquad r_i = \mu_i + \alpha\exp(\sigma_i),\ \ \alpha\sim\mathcal{N}(0,1) $$

$r_i$ is a sample from the attribute distribution, drawn via the reparameterization trick so gradients flow through $\mu_i$ and $\sigma_i$ [1, Eq. 3]. Worked example: $\mu=3.0$, $\sigma=\log(0.5)\approx-0.693$, and a drawn $\alpha = 0.8$ gives $r = 3.0 + 0.8\times 0.5 = 3.4$; if the label $R=3.4$, this sample contributes $(3.4-3.4)^2=0$ to $L_3$ for this attribute.

**Combining attributes into one reward** uses either fixed, pre-determined weights as in Nemotron-4 [2], or a learned gating layer inspired by ArmoRM [10], trained with a Bradley-Terry loss to prioritize chosen over rejected responses while the base model and value head are frozen [1, Sec. 4.1][4]. The paper's own multi-attribute URM sums the per-attribute variances as its aleatoric-uncertainty estimate [1, Sec. 4.1].

**URME's epistemic-uncertainty estimators**, over an ensemble of $N$ URMs indexed $i,j$:

$$ u_1(x,y) = \max_{i,j}\big(r^{(i)}(x,y) - r^{(j)}(x,y)\big) \qquad [1,\text{ Eq. 4}] $$

$$ u_2(x,y) = \max_i \big(\lVert \Sigma^{(i)}(x,y) \rVert_F\big) \qquad [1,\text{ Eq. 5}] $$

$u_1$ is the largest pairwise disagreement in the combined reward across ensemble members; $u_2$, taken from the offline model-based RL estimator of Yu et al., is the largest Frobenius norm of any one member's (diagonal) attribute covariance, and the paper notes it captures both uncertainty sources at once [1, Eq. 5][11]. Worked example for $u_1$ with three URMs scoring one response $0.90, 0.85, 0.40$: $u_1 = 0.90 - 0.40 = 0.50$, a large disagreement flagging low reliability.

**Which variant frameworks ship**: neither loss nor either uncertainty estimator has a framework-native implementation; only the trained checkpoints are shipped (see Shipped by). The paper's own ablation (Table 2) trains both losses from two base models and finds attribute regression ($L_3$) scores higher on RewardBench, while MLE ($L_2$) gives better-calibrated uncertainty, so the paper uses regression for accuracy comparisons and MLE for all uncertainty-quantification experiments [1, Sec. 5.2.2].

## Cost

**Theory, from the method's own math:**

- Time: URM's own training is not more expensive than training a standard sequence-classification reward model - the value head only replaces one scalar output with a mean and log-std per attribute, a negligible parameter and compute addition to a forward/backward pass over the base LLM [1, Eq. 2-3]. The two-stage recipe differs in cost: stage 1 backpropagates through the whole base model, while stage 2 freezes the base model and value head and trains only a small gating layer [1, App. A.1]. URME multiplies stage-1 training cost by the ensemble size $N$ (the paper uses $N=3$), since each member is trained independently [1, Sec. 5.1].
- Memory: one URM instance costs about the same as a standard reward model of the same base size - weights, gradients, and optimizer state for the base LLM, plus a small extra head. URME must hold or serially run $N$ full base-model copies at inference to compute $u_1$ or $u_2$; a naive reading of the ensemble formulas implies all $N$ members' outputs must exist simultaneously for one input, which at inference is $N$ forward passes over the same sequence, not $N\times$ the training-time activation memory.

**In practice:**

- No framework doc reports a production training or serving cost profile for URM specifically. The paper's own runs used 8 H800 GPUs, fp16 weights (fp32 tested with no measured accuracy gain over the extra memory it costs), global batch size 64 (4 per device, 2 gradient-accumulation steps), max sequence length 4096, for 1 epoch on HelpSteer2; the gating layer was then trained for 4000 steps at batch size 256 on Skywork-Reward-Preference-80K [1, App. A.1].
- The only shipped artifact is inference-time: `LxzGordon/URM-LLaMa-3.1-8B` is an 8B-parameter causal LM with a classification head, loaded like any other Hugging Face sequence-classification model [4]; its memory footprint at inference matches its 8B base model, with no extra models loaded unless the caller runs the 3-member URME ensemble itself.

## How to use it

- Data prep, stage 1 (attribute regression): responses with multi-attribute human ratings. The paper's HelpSteer2 run uses five attributes: Helpfulness, Correctness, Coherence, Complexity, Verbosity [1, App. A.1][4][5].
- Data prep, stage 2 (gating layer): chosen/rejected response pairs, e.g. Skywork-Reward-Preference-80K, used only to train the small gating network with base model and value head frozen [1, App. A.1][4].
- Reward convention: the combined scalar reward is a weighted sum of the sampled per-attribute scores, with weights either fixed as in Nemotron-4 or produced by the learned gating layer; the paper reports removing the gating layer changes RewardBench scores only slightly (Chat 0.955, Chat-Hard 0.864, Safety 0.909, Reasoning 0.976, versus URM(S)'s gated 95.5/88.2/91.1/97.0) [1, App. A.1].
- Knobs, with each source's own value ("not stated" = the source was checked and does not give the value):

| knob | paper (stage 1) [1, App. A.1] | paper (stage 2, gating) [1, App. A.1] |
| --- | --- | --- |
| learning rate | $2\times10^{-6}$ | not stated |
| weight decay | $10^{-3}$ | not stated |
| batch size | 64 (4/device x 2 accum.) | 256 |
| training length | 1 epoch | 4000 steps |
| max sequence length | 4096 | not stated |
| precision | fp16 | not stated |
| ensemble size (URME) | 3 URMs | n/a |

- Loss trade-off: the paper's ablation (Table 2, RewardBench overall) shows `URM-Reg` scoring $92.9\pm0.1$ (Skywork base) / $89.9\pm0.2$ (FsFairX base) against `URM-MLE` at $91.7\pm0.3$ / $87.6\pm0.4$, both above a deterministic-head ablation `URM-Det` at $92.0\pm0.1$ / $88.3\pm0.3$, and both above the un-modified base models at $92.5$ / $84.4$ [1, Table 2]. Pick regression for raw accuracy, MLE when the downstream loop needs calibrated uncertainty, per the paper's own recommendation [1, Sec. 5.2.2].
- Downstream filtering trade-off: in the paper's iterative DPO experiments, keeping the 50% of chosen/rejected pairs with lowest uncertainty each iteration was used, rather than all pairs [1, Sec. 5.2.4]; in PPO-based RLHF, an uncertainty threshold on generated responses was swept at 10, 30, 50, and "inf" (no filtering); a tighter threshold discards more data and can starve training, while no filtering exposes PPO to unreliable, out-of-distribution reward predictions [1, Sec. 5.2.4].

## While it runs

- Signals and their healthy shapes: the paper's own downstream metric is win rate against the SFT model under GPT-4-0125-preview-as-judge, evaluated with position-swapping to reduce judge positional bias [1, Sec. 5.2.4]; for PPO with no uncertainty filtering ("inf" threshold), win rate stayed at 60.2%, which the paper attributes to unreliable reward predictions, while filtering by uncertainty raised it, with the paper reporting the best PPO result at threshold 50 among the thresholds tested (10, 30, 50, inf) - the paper gives exact win-rate numbers for those runs only in a figure, not in text, so only the 60.2% "inf" figure is confirmed here from the read text [1, Sec. 5.2.4]. The fraction of an ensemble's disagreement ($u_1$) or largest per-member variance ($u_2$) rising for a given input is the live signal that a reward prediction should be discounted [1, Eq. 4-5].
- Published reference runs: RewardBench overall/category scores in Table 1 are the paper's own reference numbers - URM(S) 92.9 (Chat 95.5, Chat-Hard 88.2, Safety 91.1, Reasoning 97.0) against base Skywork-RM's 92.5 (95.8/87.3/90.8/96.2); URM(F) 89.9 (96.9/78.7/88.2/95.7) against base FsFairX-RM's 84.4 (99.4/65.1/86.8/86.4) [1, Table 1].
- Degeneracies and defaults: the paper reports that RewardBench identified overlap between its test set and the Skywork-Reward-Preference-80K-v0.1 training set used for the gating layer, i.e. unintentional test contamination for that stage; the paper notes the gating layer is not a necessary component and reports near-identical scores with it removed [1, App. A.1]. Both training stages assume attribute independence (diagonal covariance) as a simplifying default, not a property verified from data [1, Sec. 4.1].
- Named successors: none identified in the source read for this card; URM itself names one concurrent, not successor, method - QRM, which models the reward distribution via quantile regression rather than URM's Gaussian MLE or reparameterized regression, and which the paper distinguishes as not also studying an uncertainty-aware ensemble [1, Sec. 3.2][6].
- Known failure modes: the paper's own analysis states that because URM is not specifically trained to match the initial SFT policy, it can produce many out-of-distribution reward evaluations when used naively inside iterative DPO, performing poorly and even causing model degeneration if uncertainty-based filtering is not applied [1, Sec. 5.2.4]. The same GitHub repository-search query, `URM uncertainty-aware reward model`, found no dedicated URM repository beyond the generic RLHFlow/RLHF-Reward-Modeling base codebase the authors adapted [9], so no maintainer-reported issue thread on a URM-specific repository was checked for numerical traps; this does not cover code-level mentions inside unrelated repositories, since GitHub's code-search endpoint requires authentication this card's fetches did not use.
- What the gain is - and is not: the paper's own contribution is reliability, not raw ranking accuracy - RewardBench gains over the un-modified base reward models are modest (92.9 vs 92.5; 89.9 vs 84.4) [1, Table 1], and the larger claimed benefit is that low-uncertainty reward predictions are more reliable and, when used to filter data or penalize generations, improve downstream BoN, iterative DPO, and PPO alignment results relative to using the same reward model without uncertainty filtering [1, Sec. 5.2.3-5.2.4]. URM does not add a new reward signal or capability beyond what its base reward model already scores; it adds a confidence estimate on top of that signal.

## Sources

[1] Lou, Yan, Shen, Yan, Xie, Zhang, "Uncertainty-aware Reward Model: Teaching Reward Models to Know What is Unknown," 2024. https://arxiv.org/abs/2410.00847 - defines URM/URME: architecture, MLE and regression losses (Eq. 2-3), ensemble uncertainty estimators (Eq. 4-5), RewardBench results (Table 1), ablation (Table 2), BoN/iterative DPO/PPO experiments, training details (App. A.1). Fetched 2026-08-09 (HTML full text at arxiv.org/html/2410.00847).

[2] Adler et al., "Nemotron-4 340B Technical Report," 2024. https://arxiv.org/abs/2406.11704 - multi-attribute reward model that URM's value head extends; source of the fixed-weight attribute-combination scheme. Cited via [1]'s description and reference list; not independently fetched.

[3] Ouyang et al., "Training language models to follow instructions with human feedback," 2022. https://arxiv.org/abs/2203.02155 - InstructGPT, the RLHF/PPO and Bradley-Terry reward-model baseline referenced as URM's nearest offline, non-distributional alternative. Cited via [1]'s preliminaries (Sec. 2); not independently fetched.

[4] Hugging Face model cards, `LxzGordon/URM-LLaMa-3.1-8B` and `LxzGordon/URM-LLaMa-3-8B`. https://huggingface.co/LxzGordon/URM-LLaMa-3.1-8B - architecture summary, two-stage training description, attribute list, usage code, datasets used, base model (Skywork-Reward-Llama-3.1-8B). Fetched 2026-08-09 (raw README via Hub API at revision `5344f0f4eb1ad2f8e5775c8a17bdc7f4819b49ec`, Hub `lastModified` 2025-02-21T07:57:08Z); this revision is what the card's claims about this model card are pinned to, and the Hub's live listing (download/like counts, and any later revision) is not covered. https://huggingface.co/LxzGordon/URM-LLaMa-3-8B - base model (FsfairX-LLaMA3-RM-v0.1). Fetched 2026-08-09 (raw README via Hub API at revision `888cd0f929ca815988efba98bbfb95c580e69a5f`, Hub `lastModified` 2024-10-11T08:11:25Z); same pinning caveat applies.

[5] Wang, Dong, Delalleau, Zeng, Shen, Egert, Zhang, Sreedhar, Kuchaiev, "HelpSteer2: Open-source dataset for training top-performing reward models," 2024. https://arxiv.org/abs/2406.08673 - the multi-attribute dataset URM's stage-1 training uses. Cited via [1] and [4]; not independently fetched.

[6] Dorka, "Quantile Regression for Distributional Reward Models in RLHF," 2024. https://arxiv.org/abs/2409.10164 - QRM, the concurrent distributional-reward-model alternative discussed in [1]'s related work. Cited via [1]'s related-work section (Sec. 3.2); not independently fetched.

[7] Xiong, Dong, Ye, Wang, Zhong, Ji, Jiang, Zhang, "Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint," 2024. https://arxiv.org/abs/2312.11456 - iterative DPO, the online-loop consumer of URM's reward scores in [1]'s downstream experiments. Cited via [1]'s preliminaries (Sec. 2) and Sec. 5.2.4; not independently fetched.

[8] RLHFlow/RLHF-Reward-Modeling GitHub repository (README, `main` branch). https://github.com/RLHFlow/RLHF-Reward-Modeling - the generic Bradley-Terry reward-model codebase URM's training was adapted from; its README does not mention URM's probabilistic value head or losses. Fetched 2026-08-09 (raw README via GitHub API); an unpinned, mutable default-branch file - re-check against the commit you build from.

[9] GitHub REST API, repository search endpoint. https://api.github.com/search/repositories?q=URM+uncertainty-aware+reward+model - queried for a dedicated URM repository; returned `total_count: 0`, no matching repositories. Fetched 2026-08-09; a live, unauthenticated query against this endpoint (code-level search was not run, since GitHub's code-search endpoint requires authentication this card's fetches did not use).

[10] Wang, Xiong, Xie, Zhao, Zhang, "Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts," 2024. https://arxiv.org/abs/2406.12845 - ArmoRM, whose gating layer inspired URM's attribute-combination alternative to fixed weights. Cited via [1]'s reference list and [4]'s model card; not independently fetched.

[11] Yu, Thomas, Yu, Ermon, Zou, Levine, Finn, Ma, "MOPO: Model-based Offline Policy Optimization," 2020. https://arxiv.org/abs/2005.13239 - source of the $u_2$ ensemble-variance uncertainty estimator (Eq. 5) URM reuses from offline model-based RL. Cited via [1]'s Eq. 5 attribution; not independently fetched.
