# MPO

DPO's preference loss plus BCO's quality loss plus an SFT-style generation loss, weighted and summed into one objective, so a vision-language model learns relative preference, absolute response quality, and how to generate the preferred answer at once, from the same chosen/rejected pair.

**MPO** (Mixed Preference Optimization) is an offline preference-optimization method for multimodal LLMs, introduced by Wang et al., "Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization" (https://arxiv.org/abs/2411.10442), to fix a failure the authors observed when training on large-scale multimodal preference data with plain DPO: the resulting models could fail to generate reasonable rationales and produced repetitive responses, a pattern the paper says aligns with prior analysis in the Smaug work [1]. Its parent is DPO, which fits a policy directly to preference pairs under a Bradley-Terry assumption without an explicit reward model [2]. MPO keeps DPO's pairwise loss as one term but adds two more: a quality loss borrowed from BCO that scores each response's absolute quality against a shifting threshold [3], and a generation loss that is length-normalized SFT negative log-likelihood on the chosen response [1]. The three terms are combined as a weighted sum, $\mathcal{L}=w_p\mathcal{L}_p+w_q\mathcal{L}_q+w_g\mathcal{L}_g$ [1]. The paper gives two reasons for the design: an effective preference-optimization process should teach relative preference between pairs, absolute quality of individual responses, and the process of generating a preferred response, all three at once; and DPO alone, trained on large-scale preference data, degrades a model's own likelihood of both the chosen and the rejected response together, which the added generation loss counteracts [1]. The paper's own name collides with an unrelated older RL method, V-MPO ("Maximum a Posteriori Policy Optimization") [4]; the exact-title search that resolved the collision on this card matched the paper's full title against the highest-cited candidate under related preference-optimization search terms, confirming the multimodal MPO as the correct hit.

MPO trained the paper's own InternVL2-8B-MPO and InternVL2-76B-MPO release [1], and the method was carried forward into every subsequent InternVL generation: InternVL2.5's MPO-tuned checkpoints are reported to outperform their non-MPO counterparts by an average of 2 points across all model scales on the OpenCompass leaderboard [5], and InternVL3.0 and the later InternVL3.5 CascadeRL pipeline both ship an MPO training stage, with InternVL3.5 adding a further on-policy RL stage on top of the MPO-tuned checkpoint [5]. In the paper's own headline result (Table 2), InternVL2-8B-MPO reaches 67.0% on MathVista against 58.3% for InternVL2-8B before MPO, an 8.7-point gain [1]. Lineage in one line: DPO (2023 [2]) + BCO (2024 [3]) -> MPO (2024 [1]) -> InternVL2.5/3.0/3.5 MPO stage [5], later paired with an online RL stage in InternVL3.5's CascadeRL [5].

**When to pick it**: offline preference tuning for a multimodal (image-conditioned) reasoning model when you have chosen/rejected response pairs and plain DPO on that data is known to degrade chain-of-thought generation quality or produce repetitive output [1]. Prefer plain DPO with only an added SFT term ("DPO+" in the paper's own ablation) if you want a simpler two-term objective and don't need the quality loss's absolute-quality signal [1]. Prefer an on-policy method such as GRPO [6] when you can score freshly sampled completions each step rather than training on a fixed pair set; InternVL3.5's CascadeRL runs exactly this online RL stage after its MPO stage [5].

**Variant of**: DPO [2], with BCO's quality loss [3] and an SFT-style generation loss [1] added as extra terms.

**Data it needs**: preference pairs with a fixed schema — image, question, chosen response, rejected response per training example, no scalar reward column [7]. The defining paper's own MMPR dataset totals about 750K pairs without a clear ground-truth answer plus 2.5M pairs with one (roughly 3.25M pairs) [1]; a separate reproduction guide for the same InternVL2.0 MPO release states the released MMPR data contains about 3 million pairs of which only around 1.0 million are actually used in training, via a per-dataset repeat/mixture setting [7] — the two figures are from different documents and are not reconciled here. Offline: all pairs are collected before training and reused across epochs; no fresh sampling occurs inside the training loop [1].

**Extra models**: one frozen reference model $\pi_0$, required by both the preference loss and the quality loss's implicit-reward term — a copy of the initial policy, forward passes only [1]. No value network and no separate reward model: BCO's quality loss trains the policy's own implicit reward (the scaled log-ratio against $\pi_0$) rather than a standalone classifier [1][3]. Details in Cost.

**Shipped by**: trl (`DPOTrainer` with `DPOConfig(loss_type=["sigmoid","bco_pair","sft"], loss_weights=[0.8,0.2,1.0])`) — trl's own docs name this exact combination as implementing MPO, and multi-loss support was merged into `main` on 2025-07-21 [8][9]. OpenGVLab/InternVL ships the original training entry point, `internvl/train/internvl_chat_dpo.py`, launched via shell scripts under `internvl_chat/shell/internvl2.0_mpo/`, built on trl 0.9.6's `DPOTrainer` with `--loss_type sigmoid,bco_pair --rpo_alpha 1` [10][7]. verl: no MPO support found — its docs index and dedicated DPO algorithm page (`algo/dpo.html`, checked 2026-08-09) return no mention of MPO and a 404 respectively [11].

## How it works

One step: for a batch of (image, question, chosen, rejected) tuples, compute four log-probabilities per example — $\pi_\theta$ and $\pi_0$ on the chosen response, $\pi_\theta$ and $\pi_0$ on the rejected response — then combine them into three losses and sum them with fixed weights [1].

**Preference loss**, identical to DPO's loss [1][2]:

$$ \mathcal{L}_p = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_c\mid x)}{\pi_0(y_c\mid x)} - \beta\log\frac{\pi_\theta(y_r\mid x)}{\pi_0(y_r\mid x)}\right) $$

$x$, $y_c$, $y_r$ are the query, chosen response, and rejected response; $\beta$ is the KL penalty coefficient; $\pi_\theta$ is initialized from $\pi_0$ [1].

**Quality loss**, from BCO [3], trains the chosen and rejected terms independently against a moving reward-shift threshold $\delta$ [1]:

$$ \mathcal{L}_q = \mathcal{L}_q^{+} + \mathcal{L}_q^{-}, \qquad
\mathcal{L}_q^{+} = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_c\mid x)}{\pi_0(y_c\mid x)} - \delta\right), \qquad
\mathcal{L}_q^{-} = -\log\sigma\!\left(-\Big(\beta\log\frac{\pi_\theta(y_r\mid x)}{\pi_0(y_r\mid x)} - \delta\Big)\right) $$

$\delta$ is the reward shift, computed as a moving average of previous rewards to stabilize training [1].

**Generation loss**, length-normalized SFT negative log-likelihood on the chosen response only [1]:

$$ \mathcal{L}_g = -\frac{\log\pi_\theta(y_c\mid x)}{|y_c|} $$

**Total objective** [1]:

$$ \mathcal{L} = w_p\mathcal{L}_p + w_q\mathcal{L}_q + w_g\mathcal{L}_g $$

The paper sets $w_p=0.8$, $w_q=0.2$, $w_g=1$, and $\beta=0.1$ for its main InternVL2-8B-MPO run [1]. Worked illustrative example (arithmetic only, not a real training log): suppose $\beta=0.1$ and, for one pair, $\log\pi_\theta(y_c|x)-\log\pi_0(y_c|x)=2$, $\log\pi_\theta(y_r|x)-\log\pi_0(y_r|x)=-1$, and $\delta=0.15$. Then the preference-loss argument is $0.1(2)-0.1(-1)=0.3$, so $\mathcal{L}_p=-\log\sigma(0.3)\approx0.56$; $\mathcal{L}_q^{+}=-\log\sigma(0.1(2)-0.15)=-\log\sigma(0.05)\approx0.67$ and $\mathcal{L}_q^{-}=-\log\sigma(-(0.1(-1)-0.15))=-\log\sigma(0.25)\approx0.58$, so $\mathcal{L}_q\approx1.25$; with an illustrative per-token generation NLL of $0.5$, $\mathcal{L}_g=0.5$. Total: $\mathcal{L}=0.8(0.56)+0.2(1.25)+1(0.5)\approx1.20$. The paper defines no other supervision variant (unlike, e.g., GRPO's outcome/process split); the ablation in Table 7 instead varies which preference loss and which extra terms are combined, and the sigmoid+bco_pair+sft combination the paper adopts is the one trl ships as "MPO" [1][8].

## Cost

**Theory, from the method's own math**: all three losses are functions of the same four scalars per example — $\log\pi_\theta(y_c|x)$, $\log\pi_0(y_c|x)$, $\log\pi_\theta(y_r|x)$, $\log\pi_0(y_r|x)$ — which a plain DPO forward/backward pass already computes [1][3]. Reading the formulas literally, MPO therefore adds no extra forward pass over vanilla DPO: it reuses DPO's four log-probabilities and adds elementwise arithmetic (the BCO threshold-shift and the length-normalized NLL) plus a running scalar $\delta$. Memory is set by DPO's own requirement: the trained policy plus one frozen reference model, no value network, no separate reward model [1]. (Whether the extra loss terms measurably change wall-clock time in a given framework is an empirical, not a theoretical, question — see the practice layer below.)

**In practice, per framework**:

- trl `DPOTrainer` [8][9]: multi-loss combination is a documented feature — pass `loss_type` as a list with matching `loss_weights` [8]. A merged trl fix (PR #5079) describes the trainer's internal `dpo_loss()` function computing all requested loss types from one shared set of chosen/rejected log-probabilities: before the fix, the `sft` loss branch's correctly-zeroed `chosen_rewards`/`rejected_rewards` were being overwritten by an unconditional DPO-style reward computation that ran regardless of loss type, inflating the logged reward metrics for any multi-loss run that includes `"sft"` — exactly MPO's configuration [12].
- InternVL/`internvl_chat_dpo.py` [10]: the training script's own executable defaults are 256 GPUs, per-device batch size 1, and a global batch size of 256 (so a gradient-accumulation step of 1), training the full model (vision encoder, connector, and LLM all unfrozen) for 1 epoch under DeepSpeed ZeRO stage 1 with gradient checkpointing — the memory cost in this recipe comes from full-parameter fine-tuning of an 8B (or 76B) vision-language model plus the frozen reference copy, not from the extra MPO loss terms [10]. (A stale header comment in the same script says "64" GPUs and "~4" per-GPU batch size and "8" epochs; the card reports the script's executable defaults, which the paper's own 1-epoch, 5e-6 learning-rate statement matches [1].)
- verl: not applicable — no MPO implementation was found [11].

## How to use it

- Data: pairwise triples of (image, question, chosen response, rejected response), no scores or separate reward labels; the InternVL reproduction docs give the JSON/JSONL schema directly (`image`, `question`, `chosen`, `rejected` keys) [7]. Chosen/rejected pairs in the paper's own MMPR pipeline come from two sources: a correctness-based pipeline (sample up to 32 solutions per question, correct ones become chosen, incorrect ones rejected) for tasks with a ground-truth answer, and a Dropout Next Token Prediction pipeline (truncate a positive response by half and let the model complete it without the image, producing the rejected sample) for tasks without one [1].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give the value):

| knob | trl `DPOConfig` default [8] | InternVL2-8B-MPO run (paper / training script) [1][10] |
| --- | --- | --- |
| loss combination | single `"sigmoid"` (DPO only); `loss_weights` default to equal (1.0) if a list is given without weights | `sigmoid` (0.8) + `bco_pair` (0.2) + generation/NLL term (weight 1, via `rpo_alpha 1` in trl 0.9.6) |
| KL coefficient $\beta$ | 0.1 | 0.1 |
| reward shift $\delta$ (BCO term) | not present: trl's `bco_pair` branch computes `chosen_rewards`/`rejected_rewards` as $\beta$ times the log-ratios with no shift term subtracted, so trl's own `bco_pair` loss omits $\delta$ entirely [13] | moving average of previous rewards, not a fixed number [1] |
| learning rate | not stated in `DPOConfig`'s own docs (inherited from the base `TrainingArguments`) | 5e-6 |
| global batch size | not stated (framework-generic) | 256 (256 GPUs, per-device batch size 1) |
| epochs | not stated (framework-generic) | 1 |

  The $\beta=0.1$ match between trl's default and the paper's own value means a user who only sets `loss_type`/`loss_weights` and leaves `beta` alone reproduces the paper's KL strength exactly; every other paper-specific value above must be set explicitly. The paper's own hyperparameter sweep on M3CoT found that raising the learning rate to 5e-6 reached the sweep's optimal result, surpassing the baseline (the model before MPO) by 19.6 points, while a further increase to 5e-5 caused a sharp performance drop, so learning rate is the most sensitive of the knobs above at fixed loss weights [1].
- Loss-weight trade-off is the run-designer's main choice specific to this method: the paper's own loss-combination ablation (Table 7, M3CoT) shows plain DPO scores Direct 75.8 / CoT 72.7 ($\Delta=-3.1$, i.e. DPO alone makes CoT answers worse than direct answers), while adding the quality and generation terms as MPO gives Direct 77.7 / CoT 79.1 ($\Delta=+1.4$) — the quality and generation terms, not the preference term alone, are what let CoT performance exceed direct-answer performance [1].

## While it runs

- Signals and their healthy shapes: MPO reuses trl's standard DPO logging, since the trainer only adds extra loss terms — `rewards/chosen` and `rewards/rejected` (the implicit reward $\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ for each side), `rewards/margins` (the average gap between them), `rewards/accuracies` (fraction of pairs where the chosen implicit reward exceeds the rejected one), and `logps/chosen`, `logps/rejected` [8]. Because of the trl PR #5079 bug described in Cost, on any pre-fix trl version these reward metrics are inflated for MPO's `sft`-inclusive configuration and should not be trusted for absolute comparison until the fix is present [12].
- Published reference runs: the paper's own InternVL2-8B-MPO run is the reference point — MathVista 67.0% versus 58.3% before MPO, and the Table 7/Table 8 numbers above, on the MMPR dataset [1]; no separate published training-curve log (loss or reward-margin over steps) was found in the paper or in the cited framework docs.
- Degeneracies and defaults: the paper's own Discussion section reports that training on DPO's preference loss alone, at scale, decreases the absolute generation probability of both the chosen and the rejected response simultaneously even while their relative gap grows — the generation loss $\mathcal{L}_g$ is added specifically to keep the chosen response's probability rising rather than falling, which the paper frames as preventing model collapse [1]. Leaving `loss_weights` unset in trl gives all specified loss types equal weight (1.0 each), not the paper's 0.8/0.2/1.0 split [8].
- Named successors: none found specific to MPO itself — a search of the trl repository's issues and PRs for "MPO" (2026-08-09, 8 results) returned only the PRs that added, documented, or fixed MPO support in `DPOTrainer` (#2544, #5079, #5089, #3799, #3766, #3906) and two unrelated issues, with no successor method proposed against MPO's own biases [14].
- Known failure modes: from the paper's own Discussion, plain DPO on large multimodal preference data can fail to generate reasonable rationales and produce repetitive responses, which motivated adding the generation loss [1]; from trl's merged PR #5079, combining `"sft"` with other loss types (MPO's own configuration) previously caused the logged reward metrics for the SFT component to be silently overwritten with DPO-style implicit rewards instead of the intended zeros, inflating `rewards/*` metrics for any MPO run before the fix [12].
- What the gain is - and is not: the paper's own analysis targets a specific failure — CoT reasoning underperforming direct-answer responses on the same model — and MPO's own ablation (Table 6, Table 7) shows this is where it helps most; the paper does not claim MPO improves general VQA or hallucination benchmarks by comparably large margins, and Table 8 shows smaller, sometimes mixed, gains there (MMHalBench 3.3 to 3.5, POPE 86.9 to 88.1) [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Framework docs (trl, verl) are `main`/`latest` builds served with no revision parameter; every default quoted above is a 2026-08-09 reading of those live pages and may drift — re-check against the version you install. OpenGVLab/InternVL sources ([5], [10]) were read from the `main` branch at commit `2410d1dbf208f0e799459aff9376e5747dbf41a2` (2025-09-22); that commit is the pin for those two claims, and any later commit to `main` is not covered by it. GitHub API results (issue/PR search and metadata, [9], [12], [14]) were fetched 2026-08-09 and are pinned to the specific issue/PR numbers and their state at that time, not to a repository commit; [13] is an unpinned `main`-branch source-code read, not a GitHub API search result.

[1] Wang et al., "Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization", 2024. https://arxiv.org/abs/2411.10442 - defines MPO: the three loss terms, weighted combination, MMPR dataset construction, implementation details/hyperparameters, main results (Table 2, Table 8), loss-combination ablation (Table 6, Table 7), hyperparameter ablation (Section 8.4), Discussion section. Fetched 2026-08-09 (arXiv HTML full text).

[2] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - DPO, the parent method's loss and Bradley-Terry framing. Fetched 2026-08-09 (abstract page).

[3] Jung et al., "Binary Classifier Optimization for Large Language Model Alignment", 2024. https://arxiv.org/abs/2404.04656 - BCO, source of MPO's quality loss. Fetched 2026-08-09 (abstract page; loss form confirmed via [1]'s own citation and formula).

[4] Song et al., "V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control", 2019. https://arxiv.org/abs/1909.12238 - the unrelated RL method behind the "MPO" name collision noted in the shortlist row. Not separately fetched; cited only to record the collision.

[5] OpenGVLab/InternVL GitHub repository README (raw, `main` branch at commit `2410d1dbf208f0e799459aff9376e5747dbf41a2`, 2025-09-22). https://github.com/OpenGVLab/InternVL - InternVL2.5-MPO's reported +2 point average OpenCompass gain, InternVL3.0/3.5 shipping an MPO training stage, InternVL3.5 CascadeRL's added online RL stage. Fetched 2026-08-09.

[6] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO, cited here only as the nearest on-policy neighbor for the "When to pick it" contrast. Fetched 2026-08-09 (abstract page).

[7] InternVL documentation, "Mixed Preference Optimization" (InternVL 2.0). https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html - MMPR pair schema (`image`/`question`/`chosen`/`rejected`), reported dataset size (~3M pairs, ~1M used in training), the `pip install trl==0.9.6` dependency note. Fetched 2026-08-09.

[8] trl `DPOTrainer` documentation. https://huggingface.co/docs/trl/main/en/dpo_trainer - `loss_type`/`loss_weights` multi-loss combination and its explicit MPO example, `beta` default (0.1), logged metrics (`rewards/chosen`, `rewards/margins`, etc.). Fetched 2026-08-09.

[9] huggingface/trl pull request #2544, titled "MPO" (title also carries a leading emoji, dropped here). https://github.com/huggingface/trl/pull/2544 - merged 2025-07-21, adds MPO/multi-loss support to `DPOTrainer`. Fetched 2026-08-09 (GitHub API).

[10] OpenGVLab/InternVL repository, MPO training script (raw, `main` branch at commit `2410d1dbf208f0e799459aff9376e5747dbf41a2`, 2025-09-22). https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0_mpo/preference_optimization/internvl2_8b_internlm2_7b_dynamic_res_mpo_full.sh - entry point (`internvl/train/internvl_chat_dpo.py`), DeepSpeed/batch/epoch/learning-rate/loss-weight flags for the released InternVL2-8B-MPO run. Fetched 2026-08-09.

[11] verl documentation index and DPO algorithm page. https://verl.readthedocs.io/en/latest/index.html and https://verl.readthedocs.io/en/latest/algo/dpo.html - no mention of MPO in the index; the DPO algorithm page returned HTTP 404. Fetched 2026-08-09.

[12] huggingface/trl pull request #5079, "Fix SFT loss type rewards being overwritten in dpo_loss()". https://github.com/huggingface/trl/pull/5079 - merged 2026-02-16; documents the reward-metric bug for multi-loss configurations that include `"sft"`, i.e. MPO's own configuration. Fetched 2026-08-09 (GitHub API).

[13] huggingface/trl source, `trl/trainer/dpo_trainer.py`, function `dpo_loss()`, `main` branch. https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py - the `bco_pair` branch computes `chosen_rewards`/`rejected_rewards` as `self.beta * chosen_logratios` / `self.beta * rejected_logratios` with no reward-shift term subtracted, confirming trl's `bco_pair` loss omits BCO's own $\delta$ term. Fetched 2026-08-09 (unpinned `main`-branch read; re-check against the version you install).

[14] GitHub search of huggingface/trl issues and pull requests for "MPO" (query `repo:huggingface/trl MPO`, 8 results: PRs #5089, #5079, #2544, #3799, #3766, #3906, and issues #4071, #4407). https://github.com/huggingface/trl - searched 2026-08-09; no successor-method proposal found among the results, only the PRs that added, documented, or fixed MPO support, plus two unrelated issues (a tutorial bug report and a paper-index request). Fetched 2026-08-09 (GitHub API).
