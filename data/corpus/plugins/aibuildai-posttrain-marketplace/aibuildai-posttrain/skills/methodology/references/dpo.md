# DPO

Skip the reward model and the RL loop: collect a fixed dataset of (prompt, preferred completion, dispreferred completion) triples and train the policy directly with a binary cross-entropy loss that implicitly makes the policy's own log-probability ratio act as the reward.

**DPO** (Direct Preference Optimization) is an offline preference-tuning method for LLMs, introduced by Rafailov et al. as a way to solve "the standard RLHF problem with only a simple classification loss" [1]. The paper is at https://arxiv.org/abs/2305.18290 [1]. Its parent is the PPO-based RLHF pipeline described in Ziegler et al., which runs supervised fine-tuning, then trains a reward model on human comparisons, then optimizes the policy against that reward model with RL [2]. DPO's mechanism: it shows that the KL-constrained reward-maximization objective RLHF actually optimizes (Eq. 3) has a closed-form optimal policy in terms of the reward (Eq. 4), and that this relationship can be inverted to re-express the Bradley-Terry preference probability directly in terms of policy log-ratios, turning reward-model fitting plus RL into a single maximum-likelihood classification loss over a static preference dataset (Eq. 7) [1]. The paper gives two reasons for existing beyond simplicity: RLHF is "a complex and often unstable procedure" that first fits a reward model and then fine-tunes with RL to maximize that estimated reward [1], and DPO removes the need to sample from the policy during training or to tune RL-specific hyperparameters [1].

Landmark adopters: Zephyr-7B is trained with what its paper calls "distilled direct preference optimization (dDPO)" on AI-feedback preference data, and reports that the result "sets the state-of-the-art on chat benchmarks for 7B parameter models" [3]. Tülu 3's post-training recipe lists "Direct Preference Optimization (DPO)" as one of its training algorithms, alongside SFT and RL with verifiable rewards [4]. In the originating paper's own results, DPO reached a 61% win rate against reference summaries on TL;DR summarization at temperature 0, exceeding PPO's 57% win rate at its own optimal sampling temperature, and in a head-to-head human evaluation DPO samples at temperature 0.25 were preferred over PPO samples at temperature 0 58% of the time [1]. Lineage in one line: PPO-based RLHF (Ziegler et al., 2019 [2]) -> DPO (2023 [1]) -> adopted by Zephyr and Tülu 3 (2023-2024 [3][4]) -> named successors IPO and SimPO (2023-2024 [5][6]).

Confirmation note: "DPO" collides with several unrelated uses of the acronym and full name across the shortlisting search terms; the paper above was confirmed as the intended match by being the top-cited result under an exact-title search against the candidate pool (9,852 citations among 259 ranked candidates, versus the next-highest name-colliding candidate at 1,073 citations for a paper that spells out "Direct Preference Optimisation" as a different, more general framework, not the DPO algorithm itself) [1].

**When to pick it**: pick DPO when you already have (or can cheaply construct) a static dataset of paired preferences and do not want to run generation inside the training loop. It is the offline alternative to its own parent, PPO-based RLHF, which requires a trained reward model plus on-policy sampling and RL optimization [1][2]. Against the nearest offline neighbor, KTO, which needs only a per-example binary "good/bad" label rather than a matched pair for the same prompt [7], prefer DPO when paired comparisons are actually available, since DPO is defined directly on pairs [1]. Against the nearest online neighbor, PPO-based RLHF, prefer PPO-style methods when a reward function or reward model can score fresh, on-policy samples and the extra reward-model plus RL infrastructure is affordable [2]; DPO's own paper reports it matching or exceeding PPO's TL;DR win rate without that infrastructure [1].

**Variant of**: the PPO-based RLHF pipeline of Ziegler et al. [2], of which DPO is a reparameterization that removes the explicit reward model and the RL optimizer [1].

**Data it needs**: a static dataset of `(prompt, chosen, rejected)` triples — trl's `DPOTrainer` accepts this as `{"prompt": ..., "chosen": ..., "rejected": ...}` in standard format or the equivalent conversational format with role/content lists [8]. Offline: the same fixed dataset is reused across the whole run, with no fresh sampling from the policy during training [1]. The paper's own preference datasets ranged from a synthetic IMDB sentiment task to the TL;DR summarization dataset and a single-turn subset of Anthropic HH, trained with batch size 64 [1].

**Extra models**: one frozen reference model, a copy of the policy at the start of training, used only for forward passes to compute the log-ratio in the loss (Eq. 7) [1]. No value network and no explicit reward model — the reward is implicit in the policy's own log-probabilities [1]. In practice, trl's `DPOTrainer` can avoid holding the reference model in memory during training by precomputing its reference log-probabilities once up front with `precompute_ref_log_probs` [8]; details in Cost.

**Shipped by**: trl (`DPOTrainer`) [8]. verl ships no standalone DPO trainer or config recipe: its documented algorithms list (PPO, GRPO, DAPO, SPIN, SPPO, GPG, and related online recipes) does not include one [9]. verl's own docs instead carry a design-pattern tutorial, "Extend to other RL(HF) algorithms," that walks through building an on-policy "Online DPO" on top of verl's actor/rollout infrastructure with placeholder pseudocode (a `generate_sequences` sampler, a `ReferencePolicy` worker, and a `DPOActor.update` with `self.loss_fn = xxx` left unfilled) — it is a walkthrough, not a runnable recipe [10].

## How it works

The loop, once, on the full dataset: for each training step, take a batch of `(prompt, chosen, rejected)` triples, compute log-probabilities of the chosen and rejected completions under both the trained policy and the frozen reference model, and take a gradient step on the loss below — no generation happens during training [1].

**The objective** [1, Eq. 7]:

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right] $$

Here $x$ is the prompt, $y_w$ the preferred (winning) completion, $y_l$ the dispreferred (losing) completion, $\pi_\theta$ the policy being trained, $\pi_{\text{ref}}$ the frozen reference policy, $\sigma$ the logistic function, and $\beta$ a temperature that controls how strongly the loss penalizes deviation from the reference policy [1]. This is derived by inverting the closed-form optimal policy of the KL-constrained RL objective $\max_{\pi_\theta}\mathbb{E}_{x,y\sim\pi_\theta}[r_\phi(x,y)] - \beta\,\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]$ [1, Eq. 3] into the Bradley-Terry preference model, replacing the reward-model MLE loss $\mathcal{L}_R(r_\phi,\mathcal{D}) = -\mathbb{E}[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]$ [1, Eq. 2] with an equivalent loss over the policy itself [1].

**The gradient** [1] makes the update rule explicit:

$$ \nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))\big[\nabla_\theta\log\pi(y_w\mid x) - \nabla_\theta\log\pi(y_l\mid x)\big]\Big] $$

where $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ is the model's own implicit reward. Each example is weighted by $\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))$ — how wrong the implicit reward currently has the pair ordered — and the update increases $\log\pi(y_w\mid x)$ while decreasing $\log\pi(y_l\mid x)$ [1]. The paper reports that dropping this weighting term causes the model to degenerate (its Appendix Table 3) [1].

Worked example, $\beta=0.1$ (the paper's default [1, Appendix B]): suppose for one pair $\log\pi_\theta(y_w|x)-\log\pi_{\text{ref}}(y_w|x) = 0.5$ and $\log\pi_\theta(y_l|x)-\log\pi_{\text{ref}}(y_l|x) = -0.3$ (log-ratios in nats). The loss argument is $\beta(0.5-(-0.3)) = 0.1\times0.8 = 0.08$, so $\mathcal{L} = -\log\sigma(0.08) \approx 0.653$, close to the $-\log(0.5)\approx0.693$ a completely unconfident model would produce, showing the model has barely started separating the pair; a well-separated pair (log-ratio gap of, say, 8.0) drives $\mathcal{L}$ toward 0.

The paper also derives a Plackett-Luce variant for ranked lists of more than two completions [1, Appendix A.3], not used in this card's worked example; trl's `DPOTrainer` implements the pairwise case [8].

**Loss-function variants actually shipped**: trl's `DPOTrainer` exposes `loss_type`, defaulting to `"sigmoid"` (the Eq. 7 loss above) [8]. Other selectable values include `"ipo"` (the IPO paper's identity-transform loss, argued to resist the overfitting the sigmoid loss is prone to) [8][5], `"hinge"` (an SLiC/RSO-style hinge loss where $\beta$ becomes the margin's reciprocal) [8], and `"sigmoid_norm"` (SimPO's fix for the sigmoid loss's length bias, normalizing by the number of non-masked tokens) [8][6]. These are configuration choices inside one trainer, not separate methods; this card treats `"sigmoid"` as the method's definition per [1].

## Cost

**Theory, from the method's own math:**

- Time: no generation inside the training loop — completions are already fixed in the dataset [1]. Each step needs one forward pass through the policy and one forward pass through the frozen reference model, over the same chosen/rejected token sequences; only the policy pass carries a backward pass and optimizer step.
- Memory: two models' weights must be available at once in the naive reading of the loss — the trained policy (weights + gradients + optimizer state) and the frozen reference model (weights only, no gradients, no optimizer state) [1]. No value network and no reward model are part of the method's own definition [1].
- The naive reading says the reference model must sit in GPU memory for the whole run; the framework bullet below shows an implementation can avoid that by precomputing reference log-probabilities once [8].

**In practice, per framework:**

- trl `DPOTrainer` [8]: `precompute_ref_log_probs`, when enabled, computes the reference model's log-probabilities for the entire training dataset once before training begins, explicitly to save memory during training "as the reference model does not need to be kept in memory" [8], with `precompute_ref_batch_size` controlling that one-time pass. Sequence length is bounded by `max_length` (default 1024) with truncation [8]. `use_liger_kernel=True` is documented as incompatible with `precompute_ref_log_probs=True` and with using more than one `loss_type` at once [8], so the memory-saving path and the Liger-kernel path are mutually exclusive choices in this implementation.
- verl has no DPO trainer in its documented algorithms list, so no framework-level cost entry exists for a configured run [9]; verl's own "Extend to other RL(HF) algorithms" tutorial sketches an on-policy Online DPO instead of the paper's offline formulation — that variant still needs a generation/rollout step each round, unlike the offline, rollout-free training over a static pair dataset this card otherwise describes [10].

## How to use it

- Data: a static preference dataset of `(prompt, chosen, rejected)` triples, or an "implicit prompt" pair of full `chosen`/`rejected` sequences if no separate prompt field exists; trl's `DPOTrainer` accepts both the plain-text standard format and a conversational (role/content) format, automatically applying the chat template in the latter case [8].
- Reward/label convention: there is no explicit reward model call; the label is simply which of the two completions in a row is `chosen` versus `rejected` [8]. trl logs the model's own implicit reward per side as `rewards/chosen` and `rewards/rejected`, each computed as $\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ [8].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give the value):

| knob | trl `DPOConfig` default [8] | paper [1] |
| --- | --- | --- |
| $\beta$ | 0.1 | 0.1 (0.5 for TL;DR summarization) |
| learning rate | 1e-6 (DPOConfig overrides TrainingArguments' 5e-5 default to this value) | 1e-6, linearly warmed up over 150 steps |
| batch size | not stated as a DPO-specific default in the docs text checked | 64 |
| optimizer | `adamw_torch_fused` | RMSprop |
| `max_length` | 1024 tokens | not stated |
| `loss_type` | `"sigmoid"` | Eq. 7 (equivalent to `"sigmoid"`) |

  Two figures match exactly between the sources: $\beta$ at 0.1 for the paper's non-summarization tasks [1][8] - the paper doubles it to 0.5 specifically for TL;DR summarization [1], a per-task adjustment neither framework default encodes - and the learning rate, where trl's DPO-specific override of 1e-6 [8] lands on the same value the paper reports using [1].
- Trade-off: raising $\beta$ tightens the effective KL constraint to the reference policy, trading a larger training signal (bigger loss gradients from confidently-wrong pairs) for less deviation from the reference; the paper's TL;DR-specific $\beta=0.5$ suggests this needs per-task tuning rather than a single global default [1]. Because training reuses the same fixed dataset, there is no group-size or batch-reuse knob analogous to an on-policy method's rollout batch — the only dataset-scale lever is how many preference pairs are collected up front.

## While it runs

- Signals and their healthy shapes: trl's `DPOTrainer` logs `rewards/chosen` and `rewards/rejected` (the model's own implicit reward per side, $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ for each completion) [8], `rewards/margins` (the average gap between the two) [8], and `rewards/accuracies` ("the proportion of examples where the implicit reward for the chosen completion is higher than that for the rejected completion") [8] — a margin trending up and an accuracy trending toward 1 indicate the policy is separating the pairs; `logps/chosen` and `logps/rejected` (average log-probabilities) are logged alongside these to see whether the gap is driven by the chosen side rising or the rejected side falling [8].
- Published reference run: on Anthropic-HH one-step dialogue, the paper reports that DPO converges to its best performance relatively quickly over the course of training, based on the win-rate-vs-temperature curves in Figure 3 [1]; the paper does not publish raw training logs, only these summarized win-rate curves.
- Degeneracies and defaults: the paper's own ablation (Appendix Table 3) shows that removing the $\sigma(\cdot)$ weighting term from the gradient — training on preferred/dispreferred pairs without that per-example confidence weighting — causes the language model to degenerate [1]. $\beta$ defaults diverge only for one task: paper 0.1 generally, 0.5 for TL;DR [1]; trl 0.1 uniformly [8] — a TL;DR-style run left at trl's default is running a smaller KL constraint than the paper used for that task.
- Named successors that fix documented biases: IPO identifies that DPO "still heavily relies on" the approximation that substitutes pairwise preferences with pointwise rewards, and proposes an identity-transform objective ($\Psi$PO with $\Psi=\text{Identity}$) that it reports as empirically superior to DPO on illustrative examples where DPO overfits [5]; trl's docs describe the IPO authors' argument that "the logit transform can overfit" the sigmoid-based DPO loss [8][5]. SimPO removes the reference model entirely by using the average log-probability of a sequence as the implicit reward and adds a target reward margin to the Bradley-Terry objective, which trl exposes as `loss_type="sigmoid_norm"` to correct DPO's original sigmoid loss's length bias [6][8].
- Known failure modes: the paper's own Discussion section raises open questions rather than confirmed failures — whether DPO policies generalize out-of-distribution as well as PPO-trained ones is only preliminarily tested, and the paper explicitly asks whether "the slight decrease in performance" it observes in one of its own generalization figures (Figure 3, right) is an instance of reward over-optimization, without resolving the question [1]. No maintainer-reported numerical trap was found in a targeted search of huggingface/trl's issue tracker for "DPO" combined with "degenerate" (1 result, unrelated to a numerical trap) and "DPO" combined with "likelihood displacement" (0 results) [11].
- What the gain is — and is not: the paper's own experiments show DPO matching or exceeding PPO-based RLHF's win rate on summarization and single-turn dialogue while being "substantially simpler to implement and train" [1] — the gain reported is training-time simplicity and stability at matched or better preference win-rate, not a claim that DPO adds any capability beyond what the reference/SFT model can already produce.

## Sources

[1] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - defines DPO: objective derivation (Eq. 2-7), gradient, degenerate-weighting ablation, TL;DR and Anthropic HH results, Discussion/limitations. Fetched 2026-08-08, arXiv v3 (arxiv.org/html/2305.18290v3 full text).

[2] Ziegler et al., "Fine-Tuning Language Models from Human Preferences", 2019. https://arxiv.org/abs/1909.08593 - the RLHF pipeline (SFT, reward-model fitting, RL optimization) that DPO's Preliminaries section cites as the pipeline it reviews and replaces. Fetched 2026-08-08, arXiv v2 abstract page (body not fetched).

[3] Tunstall et al., "Zephyr: Direct Distillation of LM Alignment", 2023. https://arxiv.org/abs/2310.16944 - "distilled direct preference optimization (dDPO)" adopter, chat-benchmark state-of-the-art claim. Fetched 2026-08-08, arXiv v1 abstract page (body not fetched).

[4] Lambert et al., "TÜLU 3: Pushing Frontiers in Open Language Model Post-Training", 2024. https://arxiv.org/abs/2411.15124 - lists DPO among Tülu 3's post-training algorithms. Fetched 2026-08-08, arXiv v5 abstract page (body not fetched).

[5] Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences", 2023. https://arxiv.org/abs/2310.12036 - introduces IPO ($\Psi$PO with identity $\Psi$), identifies DPO's reliance on the pairwise-to-pointwise approximation and its overfitting pitfall. Fetched 2026-08-08, arXiv v2 abstract page (body not fetched).

[6] Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward", 2024. https://arxiv.org/abs/2405.14734 - reference-free, length-normalized reward, target reward margin. Fetched 2026-08-08, arXiv v3 abstract page (body not fetched).

[7] Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization", 2024. https://arxiv.org/abs/2402.01306 - unpaired binary-feedback alternative to DPO's paired preferences. Fetched 2026-08-08, arXiv v4 abstract page (body not fetched).

[8] trl DPOTrainer documentation. https://huggingface.co/docs/trl/main/en/dpo_trainer - loss definition and loss_type variants, DPOConfig defaults (beta, max_length), precompute_ref_log_probs, dataset formats, logged metrics. Fetched 2026-08-08 (main/latest build, unpinned).

[9] verl documentation site search results for "DPO", which render the full site sidebar. https://verl.readthedocs.io/en/latest/search.html?q=DPO - the sidebar's Algorithms list (PPO, GRPO, DAPO, SPIN, SPPO, GPG, and related recipes) does not include a DPO trainer. Fetched 2026-08-08 (main/latest build, unpinned).

[10] verl documentation, "Extend to other RL(HF) algorithms." https://verl.readthedocs.io/en/latest/advance/dpo_extension.html - tutorial walkthrough for building an on-policy "Online DPO" on verl's actor/rollout infrastructure, with placeholder pseudocode rather than a runnable trainer. Fetched 2026-08-08 (main/latest build, unpinned).

[11] GitHub API search of huggingface/trl issues for "DPO degenerate" (1 result, unrelated) and "DPO likelihood displacement" (0 results). https://api.github.com/search/issues - queried 2026-08-08.
