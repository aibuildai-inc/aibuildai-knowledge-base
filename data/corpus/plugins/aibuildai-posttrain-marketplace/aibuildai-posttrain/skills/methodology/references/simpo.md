# SimPO

Paper: https://arxiv.org/abs/2405.14734

DPO with the reference model deleted: score each response by its own length-normalized average log-probability, and require the winning response's reward to beat the losing response's by a fixed margin.

**SimPO** (Simple Preference Optimization) is an offline preference-optimization method for fine-tuning an LLM policy on chosen/rejected response pairs, introduced by Meng, Xia, and Chen as a simpler yet more effective approach than Direct Preference Optimization (DPO) [1]. Its parent, DPO, reparameterizes the RLHF reward as a closed-form function of the policy and a frozen reference model, then fits that reward to the Bradley-Terry preference objective [2]. SimPO replaces DPO's reference-based reward with the policy's own average per-token log-probability of the response (no reference model at all), and adds a target reward margin term $\gamma$ to the Bradley-Terry objective so the winning response's reward must exceed the losing response's by at least $\gamma$ [1]. Two reasons the paper gives for the change: DPO's reward requires a reference model during training, adding memory and compute cost, and it creates a mismatch between the reward optimized in training (a log-ratio against a reference) and the average log-likelihood metric that actually guides generation, so that in the paper's own training runs only about half of the training triples satisfied a consistent reward-vs-likelihood ranking under DPO [1]. A margin term is added because the paper found generation quality improves as the target margin grows, then degrades past a point [1].

trl ships SimPO as a loss mode of its `CPOTrainer`: selecting it swaps in the reward margin and length normalization described above and turns off the trainer's behavior-cloning regularization term [3]. In the originating paper's own headline results, a Gemma-2-9B-it model trained with SimPO reached a 72.4% length-controlled win rate on AlpacaEval 2 and ranked 1st among sub-10B models on Chatbot Arena with real user votes [1]; on the Llama-3-8B-Instruct setting, SimPO outperformed DPO by 4.4 points length-controlled win rate and 2.6 points raw win rate on AlpacaEval 2, and 1.2 points win rate on Arena-Hard (Table 4) [1]. Lineage in one line: DPO (2023 [2]) -> SimPO (2024 [1]) -> shipped as a `CPOTrainer` loss mode in trl [3], with the CPO-SimPO and AlphaPO variants documented alongside it in the same trainer [3].

**When to pick it**: offline preference optimization on chosen/rejected pairs when you want to drop the reference model entirely for memory and compute savings, and can tolerate an extra margin hyperparameter to tune [1]. Prefer DPO, its parent, when you want the reference-model KL anchor as an explicit regularizer against reward hacking [2] (SimPO's paper reports the caveat that, in principle, SimPO could reward-hack without that regularization, though the paper did not observe collapse with proper tuning [1]). Prefer ORPO, the nearest reference-free offline alternative, when you want the preference signal folded directly into the supervised fine-tuning step instead of a separate post-SFT stage: ORPO's own paper studies SFT's role in preference alignment and builds its method around penalizing the disfavored style during that stage [4]. SimPO's own paper describes ORPO as a recent reference-free objective and reports SimPO beating it across their benchmarks [1]. Prefer an online method such as PPO-driven RLHF, the nearest online neighbor, when you can sample fresh completions from the current policy and score them with a reward model each step [5]; SimPO, like DPO, is fit once on a static offline dataset of pairs [1].

**Variant of**: DPO [2].

**Data it needs**: a preference dataset of (prompt, chosen response, rejected response) triples — the same shape DPO uses, no explicit reward model required. The paper's main runs used the UltraFeedback dataset for the preference-optimization stage (Base setups) or a self-generated version of it (Instruct setups, ranked with PairRM), following an UltraChat-200k SFT stage for the Base setups [1]. SimPO is offline: it trains on a fixed, pre-collected set of pairs and does not sample from the policy during training [1].

**Extra models**: none by design — no reference model, no value network. This is the method's central claim: SimPO's reward is computed entirely from the policy being trained, unlike DPO's, which needs a frozen copy of the starting (SFT) model at every training step [1]. trl's `CPOTrainer` correspondingly loads no reference model when `loss_type="simpo"` [3]. (A judge or reward model may be used only offline, to build the preference dataset itself, as in the paper's PairRM- or ArmoRM-ranked Instruct-setup data [1] — that is data preparation, not a training-time model.)

**Shipped by**: trl, via `trl.experimental.cpo.CPOTrainer` with `loss_type="simpo"` — the docs list `CPOTrainer` under the "Experimental" section of the trainer index, alongside other loss-mode variants (CPO-SimPO, AlphaPO), rather than as a standalone top-level trainer [3]. No other framework's SimPO support was checked for this card.

## How it works

Each step: for every (prompt, chosen, rejected) triple, compute each response's length-normalized average log-probability under the current policy, subtract to get a margin, and push that margin above $\gamma$ through the Bradley-Terry sigmoid loss [1].

**DPO's reward, for contrast** [2], reparameterizes the reward with the optimal policy in closed form:

$$ r(x,y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x) $$

**SimPO's reward** drops $\pi_{ref}$ and $Z(x)$ entirely, using the length-normalized average log-probability of the response under the policy itself as the reward [1]:

$$ r_{\text{SimPO}}(x,y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x) = \frac{\beta}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i \mid x, y_{<i}) $$

Length normalization (dividing by $|y|$, the token count) is explicitly chosen over the raw summed log-probability because the paper found the unnormalized sum favors artificially inflating the probability of the longer response whenever the winning response happens to be longer, which increases the risk of degeneration [1].

**The full objective** plugs this reward into a margin-augmented Bradley-Terry loss [1]:

$$ \mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma \left( \frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma \right) \right] $$

$\beta$ scales the reward difference (the paper recommends 2.0-2.5 [1], well above the generic $\beta = 0.1$ default trl's `CPOConfig` uses for the whole CPO loss family [3]); $\gamma > 0$ is the target reward margin, so the winning response's reward must exceed the losing response's by at least $\gamma$ before the sigmoid saturates [1]. Worked example, $\beta = 2.0$, $\gamma = 1.0$: if the chosen response (12 tokens) has summed log-prob $-6.0$ and the rejected response (8 tokens) has summed log-prob $-5.6$, the length-normalized rewards are $2.0 \times (-6.0/12) = -1.0$ and $2.0 \times (-5.6/8) = -1.4$; the margin is $-1.0 - (-1.4) - 1.0 = -0.6$, so $\sigma(-0.6) \approx 0.35$ and the loss is still large — training pushes to widen that gap past $\gamma$. Both key designs are load-bearing in the paper's own ablation: on Mistral-Base, removing length normalization drops AlpacaEval 2 LC from 21.5 to 11.9, and setting $\gamma = 0$ drops it from 21.5 to 16.8 (Table 5) [1].

## Cost

**Theory, from the method's own math:**

- Time: one forward+backward pass over the policy for each of the chosen and rejected sequences per step — no reference-model forward pass, unlike DPO [1].
- Memory: only the policy model needs to be held with gradients and optimizer state; no second (reference) model's weights need to sit in memory at all, unlike DPO [1].
- A naive reading says this should cut memory and time roughly in half relative to DPO (one model instead of two); the practice numbers below show a smaller, framework-measured gain because a well-implemented DPO can decouple the reference forward pass, and the reference model in DPO is never trained, so it never carries gradients or optimizer state.

**In practice, per framework:**

- trl `CPOTrainer` [3]: with `loss_type="simpo"` and `cpo_alpha=0.0`, no reference model is instantiated (SimPO is inherently reference-free, unlike CPO's own BC-regularized default) [3]. The paper's own measurement (Llama-3-Base setting, 8×H100 GPUs, comparing to a standard/vanilla DPO implementation): SimPO cuts run time by roughly 20% and reduces GPU memory usage by about 10% relative to DPO, and the paper notes DPO could match SimPO's memory efficiency if implemented to separate the reference forward pass from the policy pass, though that is not standard practice [1]. This measurement is the paper's own benchmark, not a trl-specific number.

## How to use it

- Data: a static set of (prompt, chosen, rejected) triples. trl's `CPOTrainer` (which implements SimPO) accepts both conversational and standard preference-dataset formats and applies the chat template automatically for conversational data [3].
- Reward/label convention: no explicit reward model at training time; "reward" is simply the length-normalized log-probability under the policy itself, computed identically for chosen and rejected sequences [1]. trl logs the same fields as CPO generally: `rewards/chosen` and `rewards/rejected` (mean scaled log-probabilities), `rewards/accuracies` (fraction of pairs where chosen > rejected), `rewards/margins` (mean chosen-minus-rejected gap), and `nll_loss` [3].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give the value; "not checked" = this card did not verify that source's key):

| knob | trl `CPOConfig` default (`loss_type="simpo"`) [3] | paper's own setting (varies by run) [1] |
| --- | --- | --- |
| $\beta$ | 0.1 (CPO-wide default; not SimPO-tuned) | 2.0 (Mistral-Base, Llama-3-Base) to 2.5 (Mistral-Instruct, Llama-3-Instruct) |
| $\gamma$ (`simpo_gamma`) | 0.5 | 0.3 to 1.6 across the four settings (Table 8) |
| learning rate | 1e-6 | 3e-7 to 1e-6 across the four settings (Table 8) |
| batch size (per device) | 8 | 128, fixed across all preference-optimization runs |
| training epochs | 3.0 | 1 |

  trl's default $\beta = 0.1$ is far below the paper's own recommended 2.0-2.5 [1][3] — carrying trl's generic CPO default into a SimPO run understates the reward scale the paper tuned for; the SimPO authors' own GitHub README states SimPO requires a much larger $\beta$ than DPO [6].
- Trade-off a run designer faces: $\gamma$ interacts with $\beta$ rather than acting independently — the paper's own reference implementation recommends tuning the ratio $\gamma/\beta$ (suggesting 0.5 as a starting point, grid-searched between 0 and 1) rather than $\gamma$ in isolation, since the same absolute $\gamma$ has a different effect at different $\beta$ [6]. Batch reuse is not a live knob here as it is for on-policy methods — SimPO trains once through a fixed dataset for one epoch in the paper's own runs, so the batch-size/epoch trade-off is standard-SFT-like, not rollout-driven [1].

## While it runs

- Signals and their healthy shapes: trl's `CPOTrainer` logs `rewards/margins` (mean chosen-minus-rejected reward) and `rewards/accuracies` (fraction of pairs with chosen reward exceeding rejected) for any CPO-family loss including SimPO [3]; a margin trending toward or past the configured $\gamma$ and an accuracy trending toward 1.0 indicate the model is separating chosen from rejected as intended, though this card found no SimPO-specific published threshold for either metric beyond that framework-defined logging.
- Published reference runs: the paper's own GitHub repository publishes full Weights & Biases training curves for three released checkpoints — Llama3-Instruct-SimPO, a v0.2 rerun of it, and Gemma2-IT-SimPO — as the concrete reference to compare a new run against [7].
- Degeneracies and defaults: a target margin of $\gamma = 0$ is a real degenerate config, not a hypothetical — the paper's own ablation shows it measurably underperforms a tuned $\gamma$ (21.5 to 16.8 LC win rate on Mistral-Base, Table 5) [1]; the trl default of $\beta = 0.1$ (CPO's general default, not SimPO-tuned) sits far below the paper's recommended 2.0-2.5 [1], and the SimPO authors' own GitHub README separately states a much larger $\beta$ than DPO is required for SimPO to work well [6].
- Named successors: trl's own `CPOTrainer` docs list AlphaPO and CPO-SimPO as later variants reachable through the same trainer — AlphaPO reshapes the reward via an extra $\alpha$ parameter and is linked from the docs to its Hugging Face paper page, while CPO-SimPO combines SimPO's loss with CPO's behavior-cloning regularizer for reportedly more stable training and is linked from the docs to a separate GitHub repository [3]. Neither variant is mentioned in the SimPO paper's own repository README [6].
- Known failure modes: the paper's own Limitations section states that its choice of target margin requires manual tuning with no automatic method yet, that safety and honesty are not explicitly optimized for (though the paper reports high TruthfulQA scores as an indirect signal), and that preference optimization in general — including SimPO — tends to lower downstream math (GSM8K) performance, hypothesized to relate to training data, hyperparameters, or a chat-template mismatch at evaluation [1]. No maintainer-reported failure modes were found in this card's check: no issue search was performed against trl's or the SimPO repository's issue trackers, so absence of a finding here reflects that this card did not look, not that none exist.
- What the gain is - and is not: the paper reports SimPO consistently beating DPO and other offline baselines on instruction-following benchmarks (AlpacaEval 2, Arena-Hard, MT-Bench) without substantially increasing response length, which the paper frames as evidence against length exploitation rather than as a claim about new capability [1]; it does not claim SimPO improves reasoning-heavy task performance, and in fact reports the opposite trend for math (see above) [1].

## Sources

[1] Meng, Xia, and Chen, "SimPO: Simple Preference Optimization with a Reference-Free Reward", 2024. https://arxiv.org/abs/2405.14734 - defines SimPO: reward formulation, target margin, full objective, hyperparameters (Table 8), ablations (Table 5), main results (Table 4), efficiency measurement, limitations. arXiv carries three revisions (v1 23 May 2024, v2 8 Jul 2024, v3 1 Nov 2024); the arxiv.org/html/2405.14734 rendering fetched 2026-08-08 resolves to the current version at fetch time, v3 - every claim from this source reflects v3's text, not v1's original submission.

[2] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - DPO, the parent method: reward reparameterization (Eq. 5), Bradley-Terry objective. arXiv carries three revisions (v1 29 May 2023, v2 13 Dec 2023, v3 29 Jul 2024); the arxiv.org/html/2305.18290 rendering fetched 2026-08-08 resolves to v3.

[3] trl CPOTrainer documentation. https://huggingface.co/docs/trl/main/en/cpo_trainer - SimPO loss mode (`loss_type="simpo"`), import path `trl.experimental.cpo`, Experimental trainer-index placement, default `CPOConfig` values (including `per_device_train_batch_size=8`, `num_train_epochs=3.0`), logged metrics, CPO-SimPO and AlphaPO variant links. This is the unpinned `main`-branch build, not the stable release: the page's own version banner states "You are viewing main version, which requires installation from source" and points to "v1.9.2" as the latest stable version instead - the trl defaults quoted on this card are the `main` branch's defaults at fetch time and may differ from `v1.9.2`'s. Fetched 2026-08-08.

[4] Hong, Lee, and Thorne, "ORPO: Monolithic Preference Optimization without Reference Model", 2024. https://arxiv.org/abs/2403.07691 - nearest reference-free offline alternative. arXiv carries two revisions (v1 12 Mar 2024, v2 14 Mar 2024); the abstract page fetched 2026-08-08 resolves to v2.

[5] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - the clipped-surrogate policy-gradient method behind online RLHF, cited here as the nearest online neighbor's mechanism. arXiv carries two revisions (v1 20 Jul 2017, v2 28 Aug 2017); the abstract page fetched 2026-08-08 resolves to v2.

[6] princeton-nlp/SimPO GitHub repository, README hyperparameter-tuning guidance, and its statement that SimPO requires a much larger beta than DPO. https://github.com/princeton-nlp/SimPO - beta and gamma_beta_ratio tuning advice from the paper's own authors; this guidance appears only in this README, not in the paper itself. Raw README fetched 2026-08-08 from the `main` branch, which resolved via the GitHub API to commit `1b3e8f3` (2025-02-16); the file may have changed since.

[7] princeton-nlp/SimPO GitHub repository, Weights & Biases training-curve links. https://github.com/princeton-nlp/SimPO - published reference runs for Llama3-Instruct-SimPO, its v0.2 rerun, and Gemma2-IT-SimPO. Raw README fetched 2026-08-08 from the `main` branch, which resolved via the GitHub API to commit `1b3e8f3` (2025-02-16); the file may have changed since, and the linked wandb.ai run pages themselves were not fetched for this card.
