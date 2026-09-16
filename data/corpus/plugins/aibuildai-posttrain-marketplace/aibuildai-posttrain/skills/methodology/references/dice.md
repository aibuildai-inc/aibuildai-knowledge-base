# DICE

Take a model already fine-tuned once with DPO, use the reward the DPO loss implicitly defines to rank its own fresh samples into new preference pairs, then run DPO again on a mix of those pairs and the original offline data - repeat for a couple of rounds.

**DICE** (self-alignment with DPO ImpliCit rEwards) is an iterative-DPO method that bootstraps a DPO-tuned language model using no reward model, no external judge, and no extra human labels, introduced in "Bootstrapping Language Models with DPO Implicit Rewards" [1]. Its parent is DPO (Direct Preference Optimization) [2], which DICE treats as a black box run twice: once to produce the starting checkpoint, and again inside the loop. DPO's training objective implicitly defines a reward, $r(\mathbf{x},\mathbf{y})=\beta\cdot[\log\pi^{\star}(\mathbf{y}|\mathbf{x})-\log\pi_{\mathrm{ref}}(\mathbf{y}|\mathbf{x})]$ [1]; DICE samples $K$ completions per prompt from the current policy, scores them with this implicit reward, keeps the best and worst as a new preference pair, and feeds the resulting dataset back into DPO [1]. The paper is at https://arxiv.org/abs/2406.09760 [1]. The paper gives two reasons for two added refinements rather than using the implicit reward directly: the raw implicit-reward ranking is heavily length-biased (mean length gap between chosen and rejected of 1031 characters in the first-round generated data) [1], and relying solely on the model's own (imperfect) reward risks forgetting the knowledge in the initial DPO-tuned policy [1].

No adoption by a landmark trained system was found in the sources read for this card; DICE's own paper is its only reported deployment, on Llama-3-8B-DPO and zephyr-7B-beta [1]. On AlpacaEval 2, two rounds of DICE raised the length-controlled win rate of Llama-3-8B-DPO from 18.20 to 27.55 and of zephyr-7B-beta from 12.69 to 20.71, an increase of more than 8 points for both base models (Table 1) [1]; on the leaderboard snapshot the paper reports (Table 2), the Llama-3-8B-DPO result (27.55 LC) sits above Gemini Pro's 24.38 LC and Mixtral 8x7B's 23.69 LC, though below Claude 2 (28.15 LC) and GPT-4 0613 (30.18 LC) [1]. The paper's own Table 1 also carries the number that shows why the refinements are necessary and not just marginally better: the "Offline DPO" baseline (naive continued DPO on replayed data, without DICE's length-regularized reward or replay tuning) collapses from a round-1 LC win rate of 13.40 to 4.96 for zephyr-7B-beta and from 14.13 to 1.89 for Llama-3-8B-DPO by round 2, while DICE keeps improving over the same two rounds [1]. Lineage in one line: DPO (2023 [2]) -> DICE (2024 [1]), a sibling of Self-Rewarding Language Models' iterative-DPO loop (2024 [3]) that swaps the LLM-as-judge signal for DPO's own implicit reward [1]; no named successor to DICE itself was found in the sources read for this card.

**When to pick it**: pick DICE when you already have a DPO-tuned checkpoint plus the offline preference data used to train it, want a second round of improvement, and want to avoid standing up any extra reward model or judge model [1]. Training a separate scalar reward model (the paper's own "IntlRM" baseline, trained on the same offline data) is not needed to match this signal: in the paper's own alignment-rate comparison against GPT-4o labels, DPO's implicit reward scores 0.698 versus IntlRM's 0.624, and also beats a larger externally-trained reward model (ERM-555k, trained on substantially more data) at 0.656 (Table 5) [1]. Prefer Self-Rewarding Language Models [3] as the nearest online neighbor when you want to start self-improving from an SFT model rather than a DPO-tuned one, and are willing to prompt the model itself as an LLM-as-a-judge instead of using DPO's implicit reward [1].

**Variant of**: DPO [2], used twice - once to produce the starting policy, once per bootstrapping round.

**Data it needs**: a DPO-tuned starting policy plus the offline preference dataset used to train it (the paper draws ~10k pairs from UltraFeedback) [1]; no chosen/rejected labels are needed for the new data, since DICE manufactures its own pairs by ranking K=16 fresh samples per prompt with the implicit reward [1]. On-policy in the sense that each round's preference data comes from sampling the current policy, not a static pre-collected set [1]; each round's DPO update is itself the standard offline pairwise loss run on that round's freshly built dataset [1]. Paper's training scale: 2 rounds, 300 steps per round, 9.6k preference pairs per round, global batch size 32, 8 A100 GPUs [1].

**Extra models**: a frozen reference model is required by definition, exactly as in ordinary DPO - the implicit reward is computed from the ratio of the trained policy to this reference [1]; the paper sets the reference for round $t$ to the policy from round $t-2$ ($\pi_\theta^{(t-1)}$ is the target, $\pi_\theta^{(t-2)}$ the reference) [1]. No value network, no separate reward model, and no judge model - the paper explicitly excludes approaches requiring external models such as a scalar reward model or an LLM-as-a-judge from its own method [1]. Cost is one extra full model held frozen for the reference, same as DPO.

**Shipped by**: no library implementation was found in the sources read for this card - trl's DPOTrainer docs [4] and verl's documentation index and algorithms overview page [6] contain no mention of DICE, and the paper's own code release is a standalone research repo, not a library integration (github.com/sail-sg/dice, described by its own repository page as the official implementation of the paper) [5]. Building it on top of an existing DPO trainer (trl or verl) would take an outer sampling-and-relabeling loop around the trainer - generate K completions per prompt, score them with the implicit reward formula using the current and a lagged policy checkpoint, apply the length-regularization search of Eq. 6, mix with replayed offline data, then call the existing DPO loss - not a new loss function.

## How it works

Each round: sample K completions per prompt from the current policy; score each with a length-regularized implicit reward; keep the best and worst as a new pair; mix that dataset with replayed offline data; run one round of ordinary DPO on the mixture [1].

**DPO's implicit reward**, the base signal, from the DPO loss's reparameterization of the reward in terms of the optimal policy [1]:

$$ r(\mathbf{x},\mathbf{y}) = \beta \cdot \left[\log \pi^{\star}(\mathbf{y}|\mathbf{x}) - \log \pi_{\mathrm{ref}}(\mathbf{y}|\mathbf{x})\right] $$

$\beta$ is DPO's own temperature-like coefficient, $\pi^{\star}$ the trained policy, $\pi_{\mathrm{ref}}$ the frozen reference [1].

**Length-regularized (LR) reward**, DICE's first refinement, added because the raw implicit reward favors longer responses (Eq. 5 in the paper) [1]:

$$ r_{\mathrm{LR}}(\mathbf{x},\mathbf{y};\alpha) = \beta \log \frac{\pi_\theta(\mathbf{y}|\mathbf{x})}{\pi_{\mathrm{ref}}(\mathbf{y}|\mathbf{x})} - \alpha |\mathbf{y}| $$

$|\mathbf{y}|$ is the response's string length and $\alpha$ a penalty strength; among the K sampled completions for a prompt, the one with highest $r_{\mathrm{LR}}$ is labeled $\mathbf{y}_w$ and the lowest $\mathbf{y}_l$, giving a candidate preference dataset $\mathcal{D}(\alpha)$ [1]. The paper searches for the $\alpha$ that makes this dataset length-unbiased (Eq. 6) [1]:

$$ \alpha^{\star} = \arg\min_\alpha \left| \mathbb{E}_{(\mathbf{y}_w,\mathbf{y}_l)\sim\mathcal{D}(\alpha)}\left(|\mathbf{y}_w| - |\mathbf{y}_l|\right) \right| $$

The paper's worked example: with $\alpha=0$ the average length gap between chosen and rejected in the first-round data is 1031 characters; setting $\alpha^{\star}=0.023$ (found via this search) brings the gap to $-21$ characters [1].

**Experience replay**, DICE's second refinement: the round's training set $\mathcal{D}_t$ mixes $\gamma$ fraction of data resampled from the original offline dataset $\mathcal{D}_{\mathrm{offline}}$ with $(1-\gamma)$ fraction from the freshly built $\mathcal{D}(\alpha^{\star})$ [1]; the paper's ablation finds $\gamma=0.5$ gives the best AlpacaEval 2 LC win rate in its Zephyr setting, with pure offline data ($\gamma=1$) or pure generated data ($\gamma=0$) both underperforming [1].

**The loop** (Algorithm 1): for round $t=1,2,\dots$, sample $\mathbf{y}_{1:K}\sim\pi_{\theta^{(t-1)}}$ for prompts drawn from the offline dataset's prompt set; build $\mathcal{D}(\alpha^{\star})$ by optimizing Eq. 6; mix it with $\mathcal{D}_{\mathrm{offline}}$ per $\gamma$ into $\mathcal{D}_t$; run DPO on $\mathcal{D}_t$ with $\pi_{\theta^{(t-1)}}$ as both the initial policy and the reference, producing $\pi_{\theta^{(t)}}$ [1]. The paper's own reference-model bookkeeping is not the naive "reference = previous checkpoint" reading: for round $t$'s reward computation, the target policy is $\pi_{\theta^{(t-1)}}$ and the reference is $\pi_{\theta^{(t-2)}}$ (one round further back), with the very first round's reference being the reference policy from the original DPO training [1].

## Cost

**Theory, from the method's own math**: each round needs K forward-generation passes per prompt (the paper uses K=16) before any training step, on top of the DPO training pass itself over the resulting pairs - strictly more generation than a single DPO run, scaling linearly in K [1]. Memory-wise, DICE needs the same two models DPO needs (trained policy plus one frozen reference), no more; the K completions per prompt are generated and scored one at a time before pair selection, so peak memory does not scale with K the way training-time activations would.

**In practice, per framework**: no shipping framework was found (see Shipped by), so no framework-level cost claim can be sourced; the paper's own runs used 8 Nvidia A100 GPUs per experiment, training each round for 300 steps at global batch size 32 [1].

## How to use it

- Prepare an offline preference dataset paired with a DPO-tuned checkpoint from that same dataset; DICE assumes both already exist before bootstrapping starts [1].
- At each round, sample K completions per prompt (paper: K=16) with a fixed decoding temperature/top-p (paper: $T=0.9,p=1.0$ for the Llama-3 setting, $T=0.7,p=0.9$ for the Zephyr setting) [1].
- Score with the LR implicit reward and select the $\alpha^{\star}$ that minimizes the average $|\,|\mathbf{y}_w|-|\mathbf{y}_l|\,|$ gap over the candidate dataset (Eq. 6); the paper notes this can be solved with any black-box optimizer over a scalar $\alpha$ [1].
- Mix the resulting pairs with replayed offline pairs at ratio $\gamma$ (paper's ablation best: $\gamma=0.5$) [1] and run DPO with the standard pairwise loss.

Knobs, with the paper's own values as the only reported anchor (no framework ships this method, so there is no second column):

| knob | paper [1] |
| --- | --- |
| samples per prompt K | 16 |
| DPO $\beta$ | tuned over $\{0.01, 0.1\}$ per model/method on AlpacaEval 2 |
| replay ratio $\gamma$ | 0.5 (best in Zephyr ablation) |
| length penalty $\alpha$ | $\alpha^{\star}=0.023$ found by Eq. 6 in the paper's first-round Zephyr run |
| learning rate | 5e-7, constant schedule, 50-step warm-up |
| training steps per round | 300 |
| global batch size | 32 |
| rounds | 2 (paper's main results); no continued improvement observed beyond 3 iterations |

The paper flags $\alpha$ and $\gamma$ as the two design choices it found critical: without LR reward shaping ($\alpha=0$) the policy suffers length exploitation despite raising raw win rate, and doubling $\alpha^{\star}$ ($\alpha=2\alpha^{\star}$) over-penalizes length at some cost to response quality [1]. Group-size-style trade-off: larger K gives more candidates to pick the best/worst pair from at the cost of more generation per round; the paper does not report an ablation over K itself [1].

## While it runs

- **Signals and their healthy shapes**: the paper's own diagnostic is the length-difference histogram between $\mathbf{y}_w$ and $\mathbf{y}_l$ in the constructed dataset - a skewed, large positive mean (1031 characters at $\alpha=0$) signals length exploitation, while a distribution centered near zero (like the offline UltraFeedback data's own near-zero-centered gap) signals the LR shaping is working [1]. No other first-hand monitoring guidance (entropy, clip fraction, reward-margin curves) was found in the sources read for this card; DICE reuses DPO's own training loss and offers no additional method-specific metric beyond the length-gap diagnostic.
- **Published reference runs**: Table 1 of the paper is the reference curve across two rounds for two base models, reporting both AlpacaEval 2 (LC and raw win rate) and Arena-Hard scores for Offline DPO, Offline DPO w/ new reference, LLM-as-a-judge, and DICE at Iterations 1 and 2 side by side [1]; use it as the known-good comparison since no other published run of this method was found.
- **Degeneracies and defaults**: the paper's own limitation section states that if the implicit reward model is not well-trained, it can lead to a collapse of the training pipeline [1]; the paper also reports it did not observe continuous improvement beyond three iterations, describing this as an open question shared with other self-improving methods such as Self-Rewarding Language Models [1][3]. Setting $\gamma=1$ (all replay, no generated data) reduces DICE to continued offline DPO with a refreshed reference model each round, one of the paper's own weaker baselines [1]; that baseline is not merely weaker but collapses outright in Table 1 - its round-2 LC win rate drops to 4.96 (zephyr-7B-beta) and 1.89 (Llama-3-8B-DPO), versus round-1 values of 13.40 and 14.13, showing that skipping DICE's length-regularization and replay tuning breaks the bootstrapping loop rather than just underperforming it [1].
- **Named successors**: none found in the sources read for this card (search: the DICE paper's own text and a title/abstract read of its arXiv listing; no later paper naming itself a DICE successor was located).
- **Known failure modes**: from the paper's own Section 5 - reliance on a well-trained DPO implicit reward model as a precondition, shared with the classic RLHF pipeline's dependence on a well-trained reward model [1]; and no continued improvement beyond three iterations, an issue the paper says is open across this family of methods [1]. No maintainer-issue failure reports were found, since no library ships this method.
- **What the gain is - and is not**: the paper reports DICE lifts win rate on AlpacaEval 2 and Arena-Hard using only the model's own samples and its own implicit reward, without any external feedback [1]; it does not claim the method adds new capability - the gain is bounded by what the base DPO-tuned policy can already sample and by the quality of the offline data being replayed; the paper's own alignment-rate comparison (Table 5) also shows the implicit reward is not merely a cost-saving substitute for a trained reward model but outscores one (0.698 versus IntlRM's 0.624) [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. The trl and verl documentation entries below ([4], [6]) are `main` / `latest` builds, unpinned and mutable; the "no mention of DICE" finding is a dated reading of those moving targets, not a release-pinned claim, and could change on either project's next docs update.

[1] Chen, Liu, Du, Pang, Liu, Sinha, Varakantham, Lin, "Bootstrapping Language Models with DPO Implicit Rewards", 2024. https://arxiv.org/abs/2406.09760 - defines DICE: implicit reward, LR reward shaping (Eq. 5), $\alpha^{\star}$ search (Eq. 6), experience replay, Algorithm 1, experimental setup, Table 1/Table 2 results, ablations, limitations. Fetched 2026-08-09 (HTML full text via arxiv.org/html).

[2] Rafailov, Sharma, Mitchell, Ermon, Manning, Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - DPO, the parent method. Fetched 2026-08-09 (abstract page).

[3] Yuan, Pang, Cho, Li, Sukhbaatar, Xu, Weston, "Self-Rewarding Language Models", 2024. https://arxiv.org/abs/2401.10020 - the nearest online neighbor: iterative DPO using LLM-as-a-judge self-scoring rather than DPO's implicit reward, cited by [1] as its closest related-work comparison. Fetched 2026-08-09 (abstract page).

[4] trl DPOTrainer documentation. https://huggingface.co/docs/trl/main/en/dpo_trainer - checked for any mention of DICE or a DICE-specific trainer; none found. Fetched 2026-08-09.

[5] sail-sg/dice GitHub repository. https://github.com/sail-sg/dice - the paper's own code release, its page describes it as the official implementation of the paper. Fetched 2026-08-09.

[6] verl documentation, index and algorithm-baseline pages. https://verl.readthedocs.io/en/latest/index.html and https://verl.readthedocs.io/en/latest/algo/baseline.html - checked for any mention of DICE; none found. Fetched 2026-08-09.
