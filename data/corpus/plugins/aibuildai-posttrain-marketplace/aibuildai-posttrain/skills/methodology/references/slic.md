# SLiC-HF

https://arxiv.org/abs/2305.10425

Skip the reward model and the rollout: fine-tune the SFT model so its log-likelihood ranks a human-preferred completion above a dispreferred one, using a hinge margin loss plus a regularization term to a fixed target sequence, entirely offline.

**SLiC-HF** (Sequence Likelihood Calibration with Human Feedback) is an offline preference-tuning method for LLMs, introduced in the paper of the same name to show that Sequence Likelihood Calibration (SLiC) - originally built to rank decodes by similarity to a reference sequence - can instead be driven by human preference labels [1]. The paper was confirmed by an exact title match on arXiv 2305.10425 against a single returned candidate [1]. Its parent, SLiC, is defined in Zhao et al., which trains a model so decoded candidates' sequence likelihood tracks their similarity to a reference (ROUGE, embedding distance), and offers four calibration-loss variants (rank, margin, list-rank, expected-reward) plus two regularization variants (cross-entropy, KL) [2]. SLiC-HF keeps SLiC's rank calibration loss and cross-entropy regularization exactly, but replaces the similarity-to-reference ranking signal with human preference pairs (x, y+, y-) [1]. Two reasons the paper gives for existing: PPO-style RLHF [3] holds a reward network and a value network - both comparable in size to the policy - in memory during training and decodes inside the training loop, which slows optimization steps and complicates tuning; SLiC-HF needs neither a value network nor decoding inside the loop, and can reuse off-policy human-feedback data collected for a different model [1].

RSO, a later paper, places SLiC's reference-free hinge loss and DPO in one unified family of loss functions and reports that its own rejection-sampling method outperforms both on held-out evaluations [4]. trl's `DPOTrainer` ships SLiC's hinge loss as `loss_type="hinge"`, added directly from RSO's normalized-hinge formula [5][7]. In the originating paper's own ablation, T5-Large SLiC-HF-sample-rank (ranking-model variant) raised the T5-XXL ranking model's win rate over human references from 44.96% (SFT baseline) to 86.21% (Table 1) [1]; in a 2-way human evaluation against the 6B RLHF-PPO model of Stiennon et al., the same SLiC-HF configuration won 66% to 34%, a statistically significant margin (Table 3) [1]. Lineage: SLiC (Zhao et al., 2022/2023) [2] -> SLiC-HF (2023) [1] -> unified with DPO and beaten by RSO (2023) [4] -> hinge loss shipped inside trl's `DPOTrainer` (2023) [5].

**When to pick it**: pick SLiC-HF when you already have, or can build, preference triples (x, y+, y-) and want an offline method that in its own definition needs no reward model, no value network, and no reference model at all - only the trained policy and a fixed regularization target [1]. Its parent, SLiC, calibrates against reference-similarity metrics (ROUGE, embeddings), not preference labels, so it needs reference summaries rather than preference pairs [2]. The nearest offline neighbor is DPO, which also trains on preference pairs but every one of its loss variants is defined through a log-ratio to a frozen reference model [6]. The nearest online alternative is RLHF-PPO [3], which needs a reward model, a value network, and generation inside the training loop [1].

**Variant of**: SLiC [2].

**Data it needs**: preference triples (x, y+, y-), plus SFT pairs (x, yref) for the cross-entropy regularization term [1]. Triples can come two ways: SLiC-HF-direct takes (y+, y-) straight from an off-policy human-feedback dataset; SLiC-HF-sample-rank decodes m candidates from the frozen SFT policy and ranks them with a separately trained reward or ranking model to pick pairs [1]. Paper's own scale: DSFT has 117k/6k/6k train/val/test examples, DHF has 64k human preferences, and the sample-rank variant used m=8 sampled decodes by default (m=64 tested but gave little extra gain) [1]. Offline: SLiC-HF-direct trains on a fixed preference set collected in advance; SLiC-HF-sample-rank also decodes once from a frozen SFT snapshot before calibration training starts, not fresh at every step, so neither variant samples from the model currently being trained the way online RL does [1].

**Extra models**: none required by the method's own objective - the calibration term uses the trained model's raw log-likelihoods of y+ and y-, and the regularization term is a plain cross-entropy to a fixed target sequence, not a KL against a stored reference-model copy [1][2]. SLiC-HF-sample-rank additionally needs a trained reward or ranking model, but only to score/rank candidates before calibration training begins, not inside the loss [1]. trl's shipped `loss_type="hinge"` differs here: it is RSO's reference-normalized hinge variant, not the paper's own reference-free one, so it does load and hold a frozen reference model during training [5][4] - see How it works and Cost.

**Shipped by**: trl, as `loss_type="hinge"` inside the stable, top-level `DPOTrainer` (`from trl import DPOTrainer`) [5]. No framework was found shipping the paper's own reference-free rank-calibration-plus-cross-entropy loss under a SLiC or SLiC-HF name: trl's experimental submodule list (`trl.experimental.*`) has no SLiC entry [8], and verl's algorithm documentation index lists no DPO or SLiC page [9]. Building the paper's exact loss would mean writing a custom hinge-on-raw-log-likelihood loss plus a cross-entropy regularizer on top of an existing SFT trainer - a loss swap on an existing training loop, not a new sampling loop, since the method needs no rollout [1].

## How it works

The loop in one line: build (x, y+, y-) triples offline, then fine-tune the SFT model so the ranking calibration loss pushes apart the log-likelihoods of y+ and y- while the regularization term holds it near a fixed target.

**Calibration loss** (SLiC's rank-loss variant, the one SLiC-HF uses) [2][1]:

$$ L_{cal}(\theta) = \max\bigl(0,\; \beta - \log P_\theta(y_+|x) + \log P_\theta(y_-|x)\bigr) $$

$P_\theta$ is the policy being trained, $y_+$ and $y_-$ are the preferred and dispreferred sequences for input $x$, and $\beta$ is the margin hyperparameter, matching the symbol used for this same rank-loss form in both SLiC's own equation (1) [2] and SLiC-HF's equation (2) [1]. The loss is zero once the log-likelihood gap $\log P_\theta(y_+|x) - \log P_\theta(y_-|x))$ exceeds $\beta$, and otherwise grows linearly as the gap shrinks or reverses.

**Full SLiC-HF objective**, adding cross-entropy regularization to a fixed target - this is a separate, later equation in the paper (its equation (4)), which switches the margin symbol to $\delta$ [1]:

$$ L(\theta) = \max\bigl(0,\; \delta - \log P_\theta(y_+|x) + \log P_\theta(y_-|x)\bigr) - \lambda \log P_\theta(y_{ref}|x) $$

$\lambda$ weights the regularization term, and $y_{ref}$ is either the original SFT target or the best-ranked sampled candidate; the paper found little difference between the two choices [1]. Neither term references a frozen copy of the model - $\log P_\theta(y_{ref}|x)$ is computed by the same model being trained, so this objective needs only one model in memory [1]. The paper's hyperparameter section reports the value it trained with as "ranking margin $\beta$ of 1.0" [1], using the equation (2)/(1) symbol even though the trained objective is equation (4); this card follows the paper's own usage and reports that value as $\beta = 1.0$.

Worked example, using the calibration-loss form above with margin $\beta = 1.0$ (the paper's value [1]): if $\log P_\theta(y_+|x) = -12.0$ and $\log P_\theta(y_-|x) = -13.4$, the gap is $1.4 > \beta$, so the calibration term is $\max(0, 1.0 - 1.4) = 0$ and only the regularization term contributes; if instead $\log P_\theta(y_-|x) = -12.3$, the gap is $0.3 < \beta$ and the calibration term is $\max(0, 1.0-0.3) = 0.7$, pushing $y_+$'s likelihood up and $y_-$'s down.

**RSO's reference-normalized variant, shipped by trl**: RSO restates SLiC's loss with the same reference-free form as equation (9), $L_{hinge} = \mathbb{E}[\max(0, 1-[\gamma\log\pi_\theta(y_w|x) - \gamma\log\pi_\theta(y_l|x)])]$ with $\gamma = 1/\delta$, then proposes a normalized version (its equation (10), "hinge-norm") that divides by a fixed policy $\pi_{sft}$ inside the log before applying the same hinge [4]. trl's `DPOTrainer` with `loss_type="hinge"` implements exactly this normalized form: it computes `chosen_logratios = chosen_logps - ref_chosen_logps`, `rejected_logratios = rejected_logps - ref_rejected_logps`, `delta_score = chosen_logratios - rejected_logratios`, and the per-sequence loss is `relu(1 - beta * delta_score)`, where `beta` plays the role of $\gamma$ and `ref_*` are computed by a frozen reference model [7] (source read at commit `85709ab`). This is RSO's hinge-norm loss, not SLiC-HF's own reference-free rank loss - it needs the extra reference model that the paper's own objective does not [4][7].

## Cost

**Theory, from the method's own math**:

- Time: no rollout inside the calibration loop - unlike PPO [3], decoding happens once, offline, to build the (x, y+, y-) set before training starts (for SLiC-HF-sample-rank) or is skipped entirely (SLiC-HF-direct) [1]. Training itself is one forward+backward pass per example for the policy model, same shape as ordinary SFT.
- Memory: the paper's own comparison table reports parameter memory usage for training as $4p$ for RLHF-PPO [3] (policy, value, reward, and SFT reference models, all roughly size $p$) versus $p$ for both SLiC-HF variants, and parameter updates per step as $2p$ for RLHF-PPO versus $p$ for SLiC-HF [1]. The paper also reports RLHF-PPO decoding 1M episodes inside the training loop versus SLiC-HF-sample-rank decoding 800k sequences entirely offline, and SLiC-HF-direct decoding none [1].
- A naive reading might assume the sample-rank variant needs the reward/ranking model resident throughout training since it "scores" candidates; the paper's own accounting places that model's cost entirely in the offline candidate-generation phase, not in the $p$-vs-$4p$ training-memory comparison above [1].

**In practice, per framework**:

- trl `DPOTrainer` with `loss_type="hinge"` [5][7]: because this is RSO's reference-normalized variant, not the paper's reference-free one, a frozen reference model is loaded unless `precompute_ref_log_probs=True` precomputes and discards it, per the trainer's own `precompute_ref_log_probs` option [7]. This adds one extra frozen-model forward pass per example versus the paper's own zero-extra-model accounting above - a framework-specific cost that does not apply to the method as the paper defines it [1][7].
- No verl page implements this method's loss to report framework-specific cost against [9].

## How to use it

- Dataset: trl's `DPOTrainer` expects a preference dataset with `prompt`, `chosen`, and `rejected` columns (standard or conversational format) [5] - this maps onto the paper's (x, y+, y-) triples, but note the Extra Models caveat: this trainer's hinge loss is RSO's normalized variant, so it also computes reference-model log-probs from the same columns [7].
- Regularization target: the original paper's own cross-entropy term needs a fourth column, $y_{ref}$, either the SFT reference or the best-ranked sampled candidate [1]; trl's `DPOTrainer` hinge loss has no such term - normalization against the reference model's log-ratio plays that role instead in the shipped version [7].
- Knobs, with each source's own value ("not stated" = the source was checked and does not give it; "not checked" = this card did not verify that source's value):

| knob | paper's own value [1] | trl `DPOConfig` default for `loss_type="hinge"` [5][10] |
| --- | --- | --- |
| margin ($\beta$) † | 1.0 | 0.1 |
| learning rate | $10^{-5}$ | $10^{-6}$ (trl's own doc calls out this default as different from its base `TrainingArguments` default of $5\times10^{-5}$ [10]) |
| generation batch size | 32 (calibration training), 8 sampled decodes at temperature 0.7, top-k 40 | not stated (trl's `hinge` path performs no generation; sampling happens outside `DPOTrainer` if you build sample-rank data yourself) |
| sampled candidates per prompt (m) | 8 (64 tested, little extra gain) | not applicable (trl's dataset is already paired; no per-prompt sampling in this trainer) |

† Not the same quantity: the paper's column is the margin $\beta$ itself, but trl's `beta` config field is its reciprocal ($\text{beta}_{trl} \approx 1/\beta_{paper}$) per trl's own documentation [5], so trl's default of 0.1 corresponds to a margin of 10, not 0.1.

- Trade-off a run designer faces that the paper documents directly: SLiC-HF-direct is simpler to run (no reward/ranking model, no decoding) but the paper reports its calibration loss decreasing while output length keeps growing without converging to a stable value, which it attributes to the off-policy human-feedback decodes being out-of-distribution for the SFT policy; SLiC-HF-sample-rank converges robustly instead, at the cost of decoding m candidates per prompt and training a reward or ranking model first [1].

## While it runs

- Signals and their healthy shapes: trl's `DPOTrainer` documentation and source, the only files this card fetched on the shipped implementation, describe no `loss_type="hinge"`-specific logged metric beyond the standard reward/accuracy metrics common to all `DPOTrainer` loss types [5][7]; a dedicated trl logging guide was not fetched for this card and is not cited here. The paper's own signal is the calibration loss itself, which it reports decreasing in both variants, and output length, which it reports as a warning sign for SLiC-HF-direct specifically (see Degeneracies below) [1].
- Published reference runs: the paper's own Table 1 ranker win rate progression (SFT 44.96% -> continue-SFT-on-best-decodes 65.43% -> SLiC-HF-sample-rank-by-ranking 86.21%) and Table 3's human-eval win rate against RLHF-PPO (66%* vs 34%*, starred as statistically significant) are the paper's published reference points; no raw training logs were found published alongside the paper [1].
- Degeneracies and defaults: SLiC-HF-direct's length-divergence failure above is a documented degeneracy of that specific variant, not of SLiC-HF-sample-rank [1]. The reward-vs-ranking-model choice also matters: the paper's own ranking model scored 73.23% accuracy against DHF's validation split versus 71.34% for the pointwise reward model, and SLiC-HF-sample-rank with the ranking model beat the reward-model variant by about 3 points of ranker win rate in Table 1 [1]. Framework default divergence: trl's `beta` default of 0.1 versus the paper's own margin of 1.0 is a tenfold gap in the same numerical slot (trl calls `beta` the reciprocal of the margin, so trl's default margin is 10 versus the paper's 1.0) [1][5][10] - and because trl's `loss_type="hinge"` is RSO's reference-normalized form, this default is not a direct restatement of the paper's own hyperparameter (see Extra Models and How it works).
- Named successors: RSO (Statistical Rejection Sampling Optimization) reframes SLiC's loss and DPO's loss into one family and reports it outperforms both SLiC and DPO on its evaluations, motivated by SLiC's own restriction to sampling preference pairs only from the SFT policy [4].
- Known failure modes: the paper's own limitations discussion is the length-divergence problem of SLiC-HF-direct described above, attributed to out-of-distribution off-policy decodes relative to the SFT policy's own distribution [1]. Search for maintainer-reported failure modes: queried trl's GitHub issue search for "hinge repo:huggingface/trl" (16 results) and "SLiC repo:huggingface/trl" (6 results, 2026-08-09); none of the returned issues report a numerical or implementation failure mode specific to `loss_type="hinge"` - the only substantive hit is PR #866 adding the loss from RSO's equation (12) region, with no follow-up bug reports found in this search [11]. No verl issue search performed, since no verl page implements this method [9].
- What the gain is - and is not: the paper's own experiments are confined to the Reddit TL;DR summarization task, comparing ROUGE, a T5-XXL ranking model's win rate, and human side-by-side judgments; it reports SLiC-HF improving over SFT and matching or beating a 6B RLHF-PPO model on this task, but reports an expected drop in ROUGE scores because the method has less incentive to match reference text, and does not claim gains beyond this single-task summarization setting [1].

## Sources

[1] Zhao, Joshi, Liu, Khalman, Saleh, Liu, "SLiC-HF: Sequence Likelihood Calibration with Human Feedback", 2023. https://arxiv.org/abs/2305.10425 - defines SLiC-HF: objective (eq. 4), the two data-construction variants, hyperparameters, Table 1 ablation, Table 3 human evaluation vs. RLHF-PPO, Table 5 compute/memory comparison, and the SLiC-HF-direct length-divergence limitation. Fetched 2026-08-09 (PDF full text and abstract page).

[2] Zhao, Khalman, Joshi, Narayan, Saleh, Liu, "Calibrating Sequence Likelihood Improves Conditional Language Generation", ICLR 2023. https://arxiv.org/abs/2210.00045 - defines SLiC and its four calibration-loss variants (rank, margin, list-rank, expected-reward, eq. 1) and two regularization variants (cross-entropy, KL, eq. 2); the parent method. Fetched 2026-08-09 (PDF full text and abstract page).

[3] Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, the RL algorithm behind RLHF-PPO named as SLiC-HF's comparison point throughout the paper. Fetched 2026-08-09 (abstract page).

[4] Liu, Zhao, Joshi, Khalman, Saleh, Liu, Liu, "Statistical Rejection Sampling Improves Preference Optimization", ICLR 2024. https://arxiv.org/abs/2309.06657 - unifies SLiC's reference-free hinge loss (eq. 9) and its reference-normalized "hinge-norm" variant (eq. 10) with DPO, and reports RSO outperforming both. Fetched 2026-08-09 (PDF full text and abstract page).

[5] trl DPOTrainer documentation. https://huggingface.co/docs/trl/main/en/dpo_trainer - `loss_type="hinge"` cites RSO and SLiC and describes `beta` as the reciprocal of the margin; dataset format (prompt/chosen/rejected). Fetched 2026-08-09.

[6] Rafailov, Sharma, Mitchell, Ermon, Manning, Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - DPO, the nearest offline neighbor; every loss variant is defined via a log-ratio to a frozen reference model. Fetched 2026-08-09.

[7] trl `dpo_trainer.py` source. https://github.com/huggingface/trl/blob/85709aba2feca3d0cd121217fdda4f82227ac097/trl/trainer/dpo_trainer.py - implements `loss_type="hinge"` as `relu(1 - beta * delta_score)` on reference-normalized log-ratios; `precompute_ref_log_probs` option. Commit read: `85709ab` (latest commit touching this file on the `main` branch, as of 2026-08-09).

[8] trl `experimental` submodule directory listing. https://github.com/huggingface/trl/tree/c9fad2d1a17e25e49c0bf7ad0fff81d67d920ea8/trl/experimental - lists experimental trainers (a2po, cpo, kto, ppo, etc.); no SLiC or SLiC-HF entry. Commit read: `c9fad2d` (latest commit touching this path on the `main` branch, as of 2026-08-09; GitHub contents API).

[9] verl documentation index. https://verl.readthedocs.io/en/latest/index.html - algorithm-page listing (ppo, grpo, dapo, dro, gpg, opd, opo, otb, sppo, spin, rollout_corr, entropy, baseline); no DPO or SLiC page. Fetched 2026-08-09.

[10] trl `DPOConfig` source. https://github.com/huggingface/trl/blob/dd7cbafb9f409ad11b449e66455317102f10ac6b/trl/trainer/dpo_config.py - `beta` default 0.1, `learning_rate` default `1e-6` (documented as different from base `TrainingArguments`' `5e-5`), `loss_type` default `["sigmoid"]`. Commit read: `dd7cbaf` (latest commit touching this file on the `main` branch, as of 2026-08-09).

[11] trl GitHub issue/PR search for "hinge repo:huggingface/trl" and "SLiC repo:huggingface/trl", and PR #866 ("[DPO] add SLiC hinge loss to DPOTrainer", citing RSO's equation (12)). https://github.com/huggingface/trl/pull/866. Queried via GitHub search API 2026-08-09; no bug reports specific to the hinge loss found among the returned results.
