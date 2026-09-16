# TDPO

DPO with the KL constraint moved from the sentence level to the token level: keep DPO's implicit-reward Bradley-Terry loss, but subtract a per-token forward-KL imbalance term so the dispreferred response isn't allowed to drift from the reference model faster than the preferred one.

**TDPO** (Token-level Direct Preference Optimization) is an offline preference-optimization method for aligning an LLM policy to pairwise human preferences, introduced by Zeng et al. (https://arxiv.org/abs/2404.11999) as a token-level extension of DPO that adds forward KL divergence constraints for each token [1]. Its parent is **DPO** (Direct Preference Optimization), Rafailov et al., which reparameterizes the reward in a Bradley-Terry preference model directly in terms of the policy, turning RLHF into a single classification-style loss with no separate reward model or RL loop [2]. TDPO keeps that reparameterization but redefines it at the token level: it reformulates the Bradley-Terry model as an equivalent Regret Preference Model over an advantage function (Theorem 4.5), which lets it add a divergence term measured per token instead of per sequence [1]. The paper gives two reasons for doing this: DPO's implicit KL constraint is a reverse KL applied once over the whole sequence, and reverse KL is mode-seeking, so it can reduce output diversity; and empirically, on the same data DPO lets the sequential KL divergence (SeqKL, Definition 4.3) of the dispreferred response grow faster than that of the preferred response, an imbalance the paper's Figure 1 traces through training and that TDPO is built to correct [1]. A prior attempt at this problem, f-DPO, only swaps DPO's KL direction from reverse to forward without changing its sentence-level, un-regularized structure, which the TDPO paper argues leaves the underlying imbalance unaddressed [3][1].

A citation search on the Semantic Scholar citations API for papers citing arXiv:2404.11999 (50 results returned, the API's default page, ordered by recency rather than citation count) turned up no landmark or production RLHF system when the results are re-sorted by citation count; the highest-cited entries are smaller research papers, led by a 32-citation paper on cooperative policy-reward optimization for RLHF ("CoRLHF") and a 14-citation "Demystifying Group Relative Policy Optimization" analysis [4]. In the paper's own results, on Anthropic HH with a Pythia-2.8B base, TDPO2 reaches 67.33% preference accuracy on a held-out evaluator and a response entropy of 4.915, against 59.43%/3.196 for DPO and 54.71%/4.708 for f-DPO(FKL) (Table 1) [1]. On MT-Bench, judged by GPT-4, TDPO2 gets the highest win rate among SFT, PPO (via the trlx framework), DPO, f-DPO(FKL) and TDPO1 [1]. Lineage in one line: Bradley-Terry preference model (1952) -> DPO (Rafailov et al., 2023 [2]) -> TDPO1/TDPO2 (Zeng et al., 2024 [1]) -> no landmark adopters or named successors found in this search. The shortlist flagged a name collision with two lower-cited papers that also use "DPO" variant names — arXiv:2506.03541, whose own abstract page title is "Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement," and arXiv:2505.12299, titled "MobileIPL: Enhancing Mobile Agents Thinking Process via Iterative Preference Learning" — neither of which is titled or abstracted as a token-level extension of DPO, so this card's claims are checked against arXiv:2404.11999 only [1][5][6].

**When to pick it**: offline preference fine-tuning, same setting as DPO (a fixed dataset of preferred/dispreferred pairs, no reward model, no sampling loop), when DPO's training runs show the entropy/diversity collapse the paper documents and a tighter per-token KL trade-off is wanted; TDPO adds no new data requirement over its parent, DPO [2], and stays offline, unlike PPO-style online RLHF, which needs a reward model and fresh rollouts each step and is the paper's own PPO/trlx baseline in its MT-Bench comparison [1][7]. The nearest offline alternative is f-DPO, which the paper's own Table 1 shows scoring lower on both accuracy and entropy than either TDPO variant on Anthropic HH [1][3].

**Variant of**: DPO [2].

**Data it needs**: the same triples as DPO — a prompt $x$ and a preferred/dispreferred completion pair $(y_w, y_l)$, no per-token or per-step labels beyond that pairing [1]. Offline: TDPO trains on a fixed preference dataset, no on-policy sampling from the trained model during training. The paper's own runs used IMDb sentiment continuations with a GPT-2 Large base, and Anthropic HH with a Pythia-2.8B base; for MT-Bench evaluation the HH-trained policy was scored by GPT-4 [1].

**Extra models**: one frozen reference model $\pi_{ref}$, the same requirement as DPO — no value network and no separate reward model at training time, because the reward is implicit in the policy/reference log-ratio, as in DPO [1][2]. Every token-level forward pass needs $\pi_{ref}$'s per-token vocabulary distribution (not just its log-prob of the sampled token), because the forward-KL term sums over the whole vocabulary at each position — see How it works and Cost.

**Shipped by**: no established framework. Grepping trl's `dpo_trainer` and docs-index pages, and verl's algorithm docs index, for "TDPO" and "token-level" returned no matches in either [8][9]. The only implementation is the paper's own reference repository, `Vance0124/Token-level-Direct-Preference-Optimization` (156 stars at last check), a fork of eric-mitchell's DPO codebase that adds a `loss=tdpo` config option, run via a Hydra-style CLI (`loss=tdpo loss.alpha=0.5 loss.beta=0.1 trainer=FSDPTrainer`) rather than a pip-importable trainer class [10][11]. Building TDPO into an existing trainer means writing a new loss function on top of an existing DPO trainer — it needs the reference model's full per-token vocabulary distribution at each position (not just the sampled-token log-prob DPO needs), so it is a loss-function change, not a new sampling loop.

## How it works

Per training batch: run both the policy and the frozen reference model over the preferred and dispreferred completions, take the per-token log-probabilities of the actual tokens (as DPO does) and additionally the full per-token vocabulary distributions of both models, combine them into the token-level advantage/regret expression, and take the paper's clipped logistic loss on that expression.

**Sequential KL divergence** (Definition 4.3, Eq. 10) is the sum of per-token forward KL terms along a response $y$ of length $T$ conditioned on prompt $x$:

$$ D_{SeqKL}(x, y; \pi_{ref} \,\|\, \pi_\theta) = \sum_{t=1}^{T} D_{KL}\!\left(\pi_{ref}(\cdot \mid [x, y^{<t}]) \,\|\, \pi_\theta(\cdot \mid [x, y^{<t}])\right) $$

At each position $t$, this is the KL divergence between the reference and current policy's next-token distributions, summed over the response — no sampling expectation, matching the paper's own Appendix B code, which computes this term as a plain `.sum(-1)` reduction over the vocabulary dimension at each position [1].

**Bradley-Terry as a Regret Preference Model** (Lemma 4.4, Theorem 4.5, Eq. 11-12): the paper shows the Bradley-Terry preference probability can be rewritten using an advantage function under discount factor $\gamma = 1$ (the value the paper sets throughout) [1]:

$$ P_{BT}^{*}(y_1 \succ y_2 \mid x) = \sigma\!\left(u^{*}(x, y_1, y_2) - \delta^{*}(x, y_1, y_2)\right) $$

**The reward-difference term**, identical in form to DPO's implicit reward [2], is (Eq. 13):

$$ u(x, y_1, y_2) = \beta \log \frac{\pi_\theta(y_1|x)}{\pi_{ref}(y_1|x)} - \beta \log \frac{\pi_\theta(y_2|x)}{\pi_{ref}(y_2|x)} $$

**The KL-imbalance term**, new to TDPO, is (Eq. 14):

$$ \delta(x, y_1, y_2) = \beta\, D_{SeqKL}(x, y_2; \pi_{ref}\,\|\,\pi_\theta) - \beta\, D_{SeqKL}(x, y_1; \pi_{ref}\,\|\,\pi_\theta) $$

**TDPO1 loss** (Eq. 15), with $y_1=y_w$ (preferred) and $y_2=y_l$ (dispreferred):

$$ \mathcal{L}_{TDPO1}(\pi_\theta;\pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\!\left(u(x,y_w,y_l) - \delta(x,y_w,y_l)\right)\right] $$

The paper's gradient analysis (Eq. 16) finds TDPO1 pushes the SeqKL of the preferred response up together with the dispreferred one's, which is not the intended, asymmetric correction [1]. **TDPO2** (Eq. 17) fixes this with a stop-gradient (denoted $sg(\cdot)$) on the preferred-response KL term and a separate scaling coefficient $\alpha$:

$$ \mathcal{L}_{TDPO2}(\pi_\theta;\pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\!\left(u(x,y_w,y_l) - \alpha\,\delta_2(x,y_w,y_l)\right)\right] $$

$$ \delta_2(x,y_w,y_l) = \beta\, D_{SeqKL}(x,y_l;\pi_{ref}\|\pi_\theta) - sg\!\left(\beta\, D_{SeqKL}(x,y_w;\pi_{ref}\|\pi_\theta)\right) $$

$sg(\cdot)$ blocks gradient flow through the preferred-response KL term, so training can still push the dispreferred response's KL down without being pulled by an entangled gradient from the preferred term; $\alpha$ separately scales how strongly that gap enters the loss [1]. The paper's own PyTorch implementation (Appendix B) matches this: it computes `per_position_kl` as the reference-vs-policy forward KL at every position, and for TDPO2 detaches (`.detach()`) exactly the preferred-response ($y_w$) KL term before subtracting, matching the $sg(\cdot)$ in Eq. 18 [1].

**Worked example.** Take $\beta=0.1$. Suppose the summed policy/reference log-ratio is $2.0$ over the preferred response's tokens and $-1.0$ over the dispreferred response's tokens, giving $u = 0.1\times(2.0-(-1.0)) = 0.3$. Suppose the preferred response's SeqKL to the reference is $0.5$ nats and the dispreferred response's is $1.5$ nats (the imbalance direction the paper documents in Figure 1) [1]. Then $\delta = 0.1\times(1.5-0.5) = 0.1$, so TDPO1's loss argument is $u-\delta = 0.2$ — smaller than DPO's plain $u=0.3$, because the excess divergence on the dispreferred side is being subtracted back out of the reward gap. With $\alpha=0.5$, $\delta_2$ has the same numeric value $0.1$ here (only its gradient path differs from $\delta$), giving TDPO2's argument $u - 0.5\times0.1 = 0.25$, between DPO's and TDPO1's.

## Cost

**Theory, from the method's own math:** relative to DPO, TDPO adds no new trained model — same one frozen $\pi_{ref}$, same lack of a value network or reward model [1]. The added cost is per-token: DPO needs each model's log-probability of only the sampled token, while TDPO's SeqKL term needs each model's full next-token distribution at every position of both responses, to compute the KL sum in Eq. 10. That is an extra reduction over the vocabulary dimension at every token, not an extra model or an extra forward pass — the same reference-model logits already computed for DPO's log-prob term are reused, just reduced differently.

**In practice, per implementation:** the only implementation is the paper's own reference repository, run with `trainer=FSDPTrainer` [11]. Its published example command for TDPO2 on Anthropic HH with Pythia-2.8B sets `batch_size=64`, `eval_batch_size=32`, `gradient_accumulation_steps=2`, and `model.fsdp_policy_mp=bfloat16`, and expects an SFT checkpoint passed in via `model.archive` [11]. The paper's Appendix B lists RMSprop as the optimizer with a learning rate linearly warmed up from 0 to 5e-6 over 150 steps [1]. No other framework's cost profile can be cited here because none implements TDPO (see Shipped by).

## How to use it

- Data prep: the reference repo's `preference_datasets.py` consumes the same $(x, y_w, y_l)$ triples DPO uses, loaded via a two-stage recipe — first an SFT run, then TDPO preference training initialized from the SFT checkpoint (`model.archive=.../policy.pt`) [11].
- Reward/label convention: no explicit reward or label beyond which of $y_w$/$y_l$ is preferred; the model's own log-ratio to the reference stands in for the reward, exactly as in DPO [1][2].
- Knobs, from the paper's own settings (no framework ships this method, so there is no second source column to compare against):

| knob | paper / reference repo default [1][11] |
| --- | --- |
| $\beta$ | 0.1 |
| $\alpha$ (TDPO2 only) | 0.5 |
| discount factor $\gamma$ | 1 (fixed throughout the paper) |
| optimizer | RMSprop |
| learning rate | 5e-6, linear warmup over 150 steps |
| batch size | 64 |

- Trade-off on $\alpha$: in the paper's IMDb sweep, $\alpha \in \{1, 1.5, 2, 5\}$ made the reward harder to optimize as $\alpha$ grew, which is why the paper fixed $\alpha=0.5$ for every other experiment [1] — raising $\alpha$ tightens the KL-imbalance penalty at the cost of optimization difficulty.
- TDPO1 vs TDPO2: the paper's Table 1 numbers (accuracy 60.08 vs 67.33, entropy 4.727 vs 4.915) are the basis for treating TDPO2, not TDPO1, as the default choice [1].

## While it runs

- **Signals and their healthy shapes**: the paper's own diagnostic is the gap between the preferred and dispreferred response's SeqKL divergence to the reference model (Eq. 10, tracked across training in the paper's Figure 1) — DPO's failure mode is the dispreferred SeqKL growing faster than the preferred one, and TDPO's $\delta$/$\delta_2$ term is built to keep that gap in check [1]. No metric-logging framework ships this signal under a fixed name, since no framework implements TDPO (see Shipped by); tracking it means computing Eq. 10 directly against a held-out or generated response set, as the reference repo's maintainer described doing in response to a GitHub issue asking how KL divergence was measured: generate responses to the test-set prompts, score those responses with the Siebert/sentiment-roberta-large-english classifier for reward, and separately compute the SeqKL divergence between the policy and reference model on the same generated responses [12].
- **Published reference runs**: the paper's Table 1 (Anthropic HH, Pythia-2.8B) is the known-good comparison point: TDPO2 67.33% accuracy / 4.915 entropy vs DPO 59.43% / 3.196, TDPO1 60.08% / 4.727, f-DPO(FKL) 54.71% / 4.708 [1]. A maintainer of the reference repo pointed a user with a training-stability question to a public Weights & Biases run log for the paper's demos as the loss-curve reference [13].
- **Degeneracies and defaults**: the paper's IMDb ablation shows large $\alpha$ (tested up to 5) makes the TDPO2 objective harder to optimize, which is why the shipped default is $\alpha=0.5$, not a larger value [1].
- **Named successors**: a Semantic Scholar citation search on this paper (50 citing papers returned, the API's default page, re-sorted here by citation count) found none that name themselves as a fix to a documented TDPO bias — the citing works are unrelated applications and analyses (e.g. a GRPO-analysis paper, an RLHF-safety paper), not TDPO successors [4].
- **Known failure modes**: the reference repo's maintainer reports that replacing the frozen reference policy with on-policy updates based on the policy itself — an alternative the team tested — often led to instability in the training environment of LLMs, resulting in training failures, which is why the shipped method keeps the reference policy fixed throughout training [14]. The paper's own text was checked for the string "limitation," which appears once and describes DPO's shortcoming (the motivation for TDPO), not a limitation of TDPO itself [1]. Checked all 8 issues (open and closed) on the reference repository `Vance0124/Token-level-Direct-Preference-Optimization`: besides the instability reply above, one is an unrelated Python `random.seed()` type bug fixed by the maintainer, one asks for and receives a loss-curve link, one asks how KL divergence was measured, and one asks for the evaluation scripts and is told they are not yet released, with a pointer to f-DPO's evaluation code as a substitute; none of the remaining threads reports a numerical or training failure mode specific to the method [15].
- **What the gain is - and is not**: the paper's own comparison is about the accuracy/entropy trade-off — TDPO2 raises both preference accuracy and response entropy over DPO and f-DPO on Anthropic HH (Table 1), and wins more MT-Bench judgments than SFT, PPO, DPO, f-DPO(FKL) and TDPO1 [1]. This is a diversity-preserving alignment improvement over DPO on the same preference data, not a claim of new capability beyond what the base/SFT model already provides.

## Sources

Framework docs are `main`/`latest` builds, unpinned and mutable; every default quoted above was read on 2026-08-09. The reference-repository README and metadata ([11]) were read at the `master` branch HEAD as of 2026-08-09. GitHub Issue threads ([12], [13], [14], [15]) are not part of the repository's git history and the Issues API takes no branch or commit parameter, so they are not commit-pinned; they are cited as a live endpoint read on 2026-08-09, and could be commented on again after that date.

[1] Zeng et al., "Token-level Direct Preference Optimization," 2024. https://arxiv.org/abs/2404.11999 — defines TDPO1/TDPO2, SeqKL divergence, the Bradley-Terry/Regret-Preference-Model equivalence, Table 1 and MT-Bench results, Appendix B reference code and default hyperparameters. Fetched 2026-08-09 (arXiv HTML full text).

[2] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023. https://arxiv.org/abs/2305.18290 — defines DPO, the parent method. Fetched 2026-08-09 (arXiv abstract page).

[3] Wang et al., "Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints," 2023. https://arxiv.org/abs/2309.16240 — f-DPO, cited by [1] as the nearest prior divergence-swapping variant of DPO and used as [1]'s own experimental baseline (f-DPO(FKL) in Table 1). Cited via [1]'s own description and reference-list entry; not independently fetched.

[4] Semantic Scholar citations API, papers citing arXiv:2404.11999, 50 results returned on the API's default page (ordered by recency, not citation count; re-sorted by citation count for this card's analysis). https://api.semanticscholar.org/graph/v1/paper/arXiv:2404.11999/citations — used to check for landmark adopters and named successors. Fetched 2026-08-09.

[5] arXiv:2506.03541 abstract page. https://arxiv.org/abs/2506.03541 — title "Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement," checked to rule out a name collision with TDPO's "DPO" variant naming. Fetched 2026-08-09.

[6] arXiv:2505.12299 abstract page. https://arxiv.org/abs/2505.12299 — title "MobileIPL: Enhancing Mobile Agents Thinking Process via Iterative Preference Learning," checked to rule out a name collision with TDPO's "DPO" variant naming. Fetched 2026-08-09.

[7] Schulman et al., "Proximal Policy Optimization Algorithms," 2017. https://arxiv.org/abs/1707.06347 — PPO, the online-RL baseline [1] itself runs (via the trlx framework) in its MT-Bench comparison. Fetched 2026-08-09 (arXiv abstract page).

[8] trl `DPOTrainer` documentation. https://huggingface.co/docs/trl/main/en/dpo_trainer — grepped for "TDPO" and "token-level," no matches. Fetched 2026-08-09.

[9] verl algorithms documentation index. https://verl.readthedocs.io/en/latest/algo/index.html — grepped for "tdpo" and "token-level dpo," no matches. Fetched 2026-08-09.

[10] GitHub repository search for "TDPO preference optimization," 1 result. https://api.github.com/search/repositories?q=TDPO+preference+optimization — confirms `Vance0124/Token-level-Direct-Preference-Optimization` is the only matching repository. Fetched 2026-08-09.

[11] Vance0124/Token-level-Direct-Preference-Optimization repository, README and metadata, `master` branch. https://github.com/Vance0124/Token-level-Direct-Preference-Optimization — describes the eric-mitchell-DPO-codebase fork, the `loss=tdpo` Hydra config, the FSDPTrainer example command with its batch/optimizer settings. Fetched 2026-08-09 (raw README.md and GitHub API repo metadata).

[12] Vance0124/Token-level-Direct-Preference-Optimization, issue #8 comments (live GitHub Issues endpoint, not commit-pinned). https://github.com/Vance0124/Token-level-Direct-Preference-Optimization/issues/8 — maintainer reply on how KL divergence was measured. Fetched 2026-08-09.

[13] Vance0124/Token-level-Direct-Preference-Optimization, issue #3 comments (live GitHub Issues endpoint, not commit-pinned). https://github.com/Vance0124/Token-level-Direct-Preference-Optimization/issues/3 — maintainer reply linking a Weights & Biases run log. Fetched 2026-08-09.

[14] Vance0124/Token-level-Direct-Preference-Optimization, issue #7 comments (live GitHub Issues endpoint, not commit-pinned). https://github.com/Vance0124/Token-level-Direct-Preference-Optimization/issues/7 — maintainer reply describing tested on-policy updates causing training instability, and the proof clarification for the advantage-function derivation. Fetched 2026-08-09.

[15] Vance0124/Token-level-Direct-Preference-Optimization, issues #1-#8 list and threads for #2, #3, #4, #7, #8 (live GitHub Issues endpoint, not commit-pinned). https://github.com/Vance0124/Token-level-Direct-Preference-Optimization/issues — checked for method-specific failure modes. Fetched 2026-08-09.
