# RAFT

Sample K completions per prompt from the current model, keep only the reward-model-ranked best one per prompt, and fine-tune on that filtered batch with plain SFT - no critic, no policy-gradient loss.

**RAFT** (Reward rAnked FineTuning) is an alignment method for generative foundation models that replaces reinforcement-learning-style policy optimization with an iterative best-of-K sampling and filtering loop, introduced by Dong et al. in a paper that frames it as an alternative to the "predominant PPO algorithm" for RLHF [1]. Its parent is best-of-K (rejection) sampling, the inference-time policy of drawing K responses and returning the one the reward model scores highest, used in WebGPT [2] and in the verifier-based re-ranking of Cobbe et al. [3]; RAFT's own paper describes the intuition of RAFT as that "the model iteratively learns from the induced best-of-K policy" of that scheme [1]. The paper is at https://arxiv.org/abs/2304.06767 [1]. Mechanically, each stage samples a batch of $b$ prompts, generates $K$ completions per prompt from the current model, keeps the single highest-reward completion per prompt (or, in the optional KL-regularized extension, the completion with the highest reward minus a KL penalty against a frozen reference model), and fine-tunes the model on that batch with a standard supervised loss before repeating [1]. The paper gives three reasons for this design: PPO's on-policy training and critic model make it unstable and memory-heavy, so decoupling data generation from fine-tuning into separate SFT-like steps needs only one model in memory at a time; ranking by relative reward order rather than absolute reward value makes the method insensitive to reward scaling; and using the best-of-K selection directly as training data is a data-quality control that a practitioner can inspect and monitor to catch reward-model exploitation [1].

This card found no confirmed landmark-system adopter: a check of 100 of the paper's citing works (one unsorted page returned by the Semantic Scholar citations API [17], out of 758 total citations Semantic Scholar's separate paper-lookup endpoint reports for this paper as of 2026-08-09 [18], titles scanned for known model/lab names on 2026-08-09) turned up no deployed LLM that names RAFT as the method it used; this covers only that one page, not the full citation list, so the absence is reported as what this specific check found, not as an exhaustive result. The closest large-scale parallel is Meta's Llama 2, whose "Rejection Sampling fine-tuning" - sample $K$ outputs per prompt, keep the highest-reward one, fine-tune on it - is structurally the same loop, but the Llama 2 paper attributes the idea to Bai et al. (2022b) and Scialom et al. (2020a), not to RAFT, and does not cite RAFT's arXiv id anywhere in its text [4]. A closer confirmed link is citational rather than architectural: the RLOO paper explicitly benchmarks against "newly proposed 'RL-free' methods such as DPO and RAFT" and reports that its own REINFORCE-style variants outperform both [5]. In RAFT's own LLaMA-7B/HH-RLHF experiments, the RAFT-aligned model reached a mean held-out reward of 2.294 versus 2.077 for a tuned PPO baseline, 0.772 for the LLaMA-7B-SFT starting checkpoint, and 1.873 for the dataset's own human-preferred ("chosen") responses, while holding a lower (better) perplexity than PPO (4.031 vs 4.156) (Table 3) [1]; in GPT-4 and human pairwise evaluation on 100 held-out prompts, RAFT-K32 beat a tuned PPO ($\beta{=}0.1$) baseline 65-32-3 by GPT-4 and 66-14-20 by human raters (Table 4) [1]. Lineage in one line: best-of-K sampling (WebGPT, Cobbe et al., 2021 [2][3]) -> RAFT (Dong et al., 2023 [1]) -> cited as an "RL-free" baseline by RLOO (2024 [5]) -> no landmark adopter found in the 100-citation page checked.

**When to pick it**: when a reward model (or any scalar scorer) can score fresh completions and you want an alignment loop that only ever runs plain SFT - no critic, no clipped policy-gradient loss, no held reference model unless you opt into the KL-penalized ranking variant [1]. Prefer PPO [6], RAFT's explicit point of comparison, when you already have PPO infrastructure and want per-token credit assignment from a trained critic rather than one reward per completion [1]. Prefer DPO [7] when you have static preference pairs and no ability to sample fresh completions from the current model: DPO's own paper states it eliminates the need for sampling from the LM during fine-tuning [7], so it is offline where RAFT is on-policy (RAFT resamples $K$ completions from the current policy every stage) [1]. Nearest online neighbor is RLOO [5], which also does full on-policy sampling and scalar-reward scoring but keeps a REINFORCE-style policy-gradient update over all $K$ samples rather than discarding all but the single best and fine-tuning on it with a supervised loss: RLOO's own paper states that far simpler REINFORCE-style optimization variants outperform both PPO and "RL-free" methods such as DPO and RAFT [5], and trl's implementation of it uses each sample's leave-one-out mean over the other samples in its group as the baseline [8].

**Variant of**: best-of-K (rejection) sampling, cited by the RAFT paper to WebGPT [2] and Cobbe et al. [3]; RAFT turns the inference-time best-of-K policy into an iterative training loop [1].

**Data it needs**: a prompt set only - no chosen/rejected pairs, no per-token labels - plus a reward function that scores a completion given a prompt [1]. On-policy: every stage's $K$ completions per prompt are freshly sampled from the current model, not drawn from a fixed offline set [1]. The paper's LLM experiments used 82,147 prompts (filtered from the 112K-example HH-RLHF training set to a 256-token context window) on a LLaMA-7B policy with an Open-LLaMA-3B reward model, run on 8x A40 (48GB) GPUs with bf16 [1].

**Extra models**: a reward model to score completions (required); no value network, ever - the paper positions removing PPO's critic as a core reason for the method [1]. A frozen reference model is needed only if the optional KL-regularized ranking (Eq. 5 in the paper) is used, and then only transiently during the ranking step to compute the log-probability ratio, not during the fine-tuning forward/backward pass [1]; the base method as defined in Section 3.2 uses no reference model at all [1]. See Cost for what this means for peak memory.

**Shipped by**: LMFlow (`lmflow.pipeline.raft_aligner.RaftAligner`, the toolkit built by the RAFT paper's own co-authors [1][9]), but only conditionally - `AutoPipeline` in LMFlow's source registers `RaftAligner` under the key `raft_aligner` only when the installed `transformers` version is below 4.35.0, and otherwise leaves it unregistered [10] (read at commit `ea19450`, 2026-08-09). A directory listing of LMFlow's `scripts/` and `scripts/archive/` folders shows its example launch script, `run_raft_align.sh`, sitting only under `scripts/archive/`, not the top-level `scripts/` directory that current examples use [11] (read 2026-08-09). trl and verl do not export a RAFT trainer: trl's top-level package exports `DPOTrainer`, `GRPOTrainer`, `KTOTrainer`, `RLOOTrainer`, `RewardTrainer`, and `SFTTrainer`, with no rejection-sampling or RAFT-named trainer among them [12] (read at commit `2396dfe`, 2026-08-09), and verl's README has no mention of the method beyond an unrelated substring match on "draft" [13] (read at commit `4a2cba7`, 2026-08-09). Building it on any of these frameworks would not need a new loss - it needs a sampling-and-filtering loop (generate $K$ completions per prompt, score with a reward model, keep the top-ranked one) that feeds a standard SFT trainer already shipped by trl or LMFlow [1][12].

## How it works

The loop in one line: sample $K$ completions per prompt, keep the best one per prompt by reward, run one step of SFT on the kept batch, repeat.

**Problem setup** [1]. For an initial generative model $G_0 = g(w_0, x)$ producing $y \sim p_{G_0}^{1/\lambda}(y|w_0,x)$ from a prompt $x$, and a reward function $r(x,y)$, RAFT's stated objective is:

$$ \max_{w} \; \mathbb{E}_{x\sim \mathcal{D},\, y\sim p_g(\cdot|w,x)}\; r(x,y) $$

$\lambda$ is a temperature controlling sampling diversity. Because searching the full output space $\mathcal{Y}$ for the exact maximizer is infeasible, RAFT approximates the (unreachable) deterministic optimal policy by iteratively fine-tuning on the best sample the current model can already produce [1].

**The three-step loop**, per stage $t+1$ [1]:

- Step 1, data collection: sample a batch of $b$ prompts $\mathcal{D}_t = \{x_1^t,\dots,x_b^t\}$, and for each prompt generate $K$ completions $y_1,\dots,y_K \sim p_{G_t}^{1/\lambda}(\cdot\,|w_t,x_i^t)$.
- Step 2, data ranking: score every completion with the reward function, and for each prompt keep $y := \arg\max_{y_j \in \{y_1,\dots,y_K\}} r(x,y_j)$, collecting one kept completion per prompt into a batch $\mathcal{B}$ of size $b$.
- Step 3, model fine-tuning: fine-tune the current model on $\mathcal{B}$ with a standard supervised loss, then start the next stage.

The paper notes the selection criterion is a reward *ranking* rather than a reward *threshold*, which is what makes RAFT insensitive to how the reward is scaled [1].

**Worked example.** With $K=4$ completions for one prompt and rewards $\{0.9, 0.3, 0.7, 0.5\}$, only the $0.9$-reward completion is kept; the other three are discarded entirely, contributing no gradient signal (unlike DPO's contrastive loss over a pair) [1]. This repeats independently for each of the $b$ prompts in the stage, and the resulting batch of $b$ single completions is what gets one step of SFT.

**Optional KL-regularized ranking** [1]. To trade off reward against staying close to the initial model, the paper defines a regularized objective:

$$ \max_{w}\Big[\, \mathbb{E}_{x\sim \mathcal{D},\, y\sim p_g(\cdot|w,x)}\, r(x,y) \;-\; \beta\, Q(w) \,\Big], \qquad Q(w) = \mathbb{E}_{x\sim\mathcal{D}}\, \mathrm{KL}\big(p_g(\cdot|w,x)\,\|\,p_{G_0}(\cdot|w_0,x)\big) $$

which is folded into ranking (not into the fine-tuning loss) by replacing $r$ in Step 2 with a modified reward:

$$ \tilde r(x,a) = r(x,a) - \beta \log \frac{p_g(y|w,x)}{p_{G_0}(y|w_0,x)} $$

using the paper's own symbol $a$ for the ranked sample on the reward side of the equation, while the probability ratio inside the log keeps the paper's $y$ [1]; computed by querying both the current model and the frozen initial model's logits over the $K$ sampled completions [1]. $\beta$ is optional; the base algorithm in Section 3.2 has no KL term and no reference model at all [1].

**Hyperparameters** the paper's own Table 1 lists as RAFT's complete configuration: batch size $b$, acceptance ratio $1/K$ (larger $K$ means a stronger preference for high reward), temperature $\lambda$ (larger $\lambda$ means more diverse generation), and the optional KL coefficient $\beta$ (larger $\beta$ means more regularization) [1].

## Cost

**Theory, from the method's own math:**

- Time: each stage generates $b \times K$ completions (Step 1) before any gradient step, so wall-clock is dominated by inference volume that scales linearly in $K$; the fine-tuning pass (Step 3) trains on only $b$ examples per stage - one per prompt, since $K-1$ of every $K$ generations are discarded - so the gradient-step cost itself does not scale with $K$ [1].
- Memory: no value network and, in the base method, no reference model, so exactly one trained model needs weights, gradients, and optimizer state at fine-tuning time [1]. The paper contrasts this with PPO's requirement to load four models concurrently - the trained model, a reference model, a critic, and the reward model [1]. Even with the optional KL variant, the reference model is used only inside Step 2's ranking pass (forward-only, no gradient) and does not need to coexist with the optimizer state of Step 3.
- A naive reading of "sample $K$ per prompt" might suggest holding $K\times$ activations in memory for backpropagation; the objective shows this is not the case, since only the single kept completion per prompt is ever backpropagated through - the other $K-1$ never enter the fine-tuning graph [1].

**In practice, per framework:**

- LMFlow `RaftAligner` [10][14]: decouples the same way the paper's math implies - generation runs with `model.config.use_cache = True` and gradient checkpointing disabled, then the SFT step runs from a fixed prompt-completion batch [10]. Its `collection_strategy` config has two modes: `"local"` reproduces the paper's per-prompt argmax over $K = 1/\text{top\_reward\_percentage}$ samples per prompt, while the **default**, `"top"`, instead generates one completion per prompt and keeps the globally top `top_reward_percentage` fraction across the whole batch regardless of which prompt they came from [10][14] - a materially different selection rule from the one described in the paper's Section 3.2, and the choice a run designer must make explicitly to match the paper.
- As covered under Shipped by, `RaftAligner` is gated out of `AutoPipeline`'s registry entirely once `transformers >= 4.35.0` is installed [10], so on a current `transformers` install there is no cost profile to report - the pipeline is unreachable through the documented entry point without pinning an older `transformers`.

## How to use it

- Data: prompts only, from the distribution you want aligned; LMFlow's own RAFT example builds this from the `Dahoas/full-hh-rlhf` dataset's prompt column, discarding the chosen/rejected response columns entirely for the RAFT stage itself (it uses them only in an earlier, separate reward-modeling step) [15].
- Reward: LMFlow's default RAFT example trains a GPT-Neo-2.7B reward model following the InstructGPT recipe and also ships a pre-trained copy so the reward-modeling step can be skipped [15].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give the value):

| knob | RAFT paper [1] | LMFlow default [10][14] |
| --- | --- | --- |
| samples per prompt $K$ | 32 (main HH-RLHF result, "RAFT-K32") | 5 (`collection_strategy="local"`, derived as $1/\text{top\_reward\_percentage}$) or 1 generation with global top-20% keep under the default `"top"` strategy |
| batch size $b$ | 2048 | `raft_batch_size` 1024 |
| SFT learning rate | $2\times10^{-5}$ | not exposed as a default in `args.py` (`learning_rate` is a required run-time argument) [14]; LMFlow's own example launch script sets it to `2e-5` [16] |
| SFT epochs per stage | 2 (linear decay scheduler) | not exposed as a default in `args.py` [14]; LMFlow's own example launch script sets `num_train_epochs` to `4` [16] |
| number of stages | not stated as a single number (run "until the reward converges") | `num_raft_iteration` 20 |
| KL coefficient $\beta$ | optional, ablated in Appendix A.3; not fixed in the main HH-RLHF run | not exposed as a config field in `raft_aligner.py`'s argument set found in this check |
| generation temperature | varied per experiment (paper reports both $\lambda=1.0$ used for the main table and other values ablated) | `temperature` 0.85 (fixed in `generation_kwargs`, not exposed via CLI) |

- Trade-off: a larger $K$ gives each kept completion a stronger best-of-$K$ selection pressure (closer to the true reward maximizer) at the cost of $K\times$ generation for that stage's prompts; the paper's own ablation compares RAFT-K32 against RAFT-K8, with RAFT-K32 winning 48-37-15 by GPT-4 and 40-24-36 by human evaluators on the same 100 held-out prompts (Table 4) [1].
- Install, launch flags, and distributed-training setup belong to the LMFlow framework skill; this card stops at the knobs that define the method.

## While it runs

- Signals: the paper's own headline metric is mean held-out reward, tracked stage-over-stage until it converges, alongside perplexity and diversity metrics (MSSTR, Distinct-1/2, Unique-1/2) to catch the "alignment tax" trade-off between reward and fluency/diversity that the paper explicitly names as a known effect in the literature [1]. LMFlow's `RaftAligner` records a `reward_seq` and a `train_reward` list across iterations, giving a per-stage reward trace to plot [10].
- Published reference run: the paper's Table 3 numbers - RAFT-K32-$\lambda$1.0 reaching reward 2.294 / perplexity 4.031 from an SFT starting point of reward 0.772 / perplexity 3.781, versus a tuned PPO run at reward 2.077 / perplexity 4.156 - is the known-good comparison point for a LLaMA-7B, HH-RLHF-style run [1].
- Degeneracy: because Step 2 always returns a strict per-prompt argmax over $K\geq 1$ samples, $K=1$ collapses ranking to a no-op (the "kept" completion is just whatever was sampled, with no selection pressure) - the paper's own framing of $1/K$ as an "acceptance ratio" implies $K>1$ is required for the method's selection effect to exist [1]. Separately, LMFlow's default `collection_strategy="top"` does not match the paper's per-prompt selection at all (see Cost); a run using LMFlow's defaults without switching to `"local"` is not running the paper's algorithm as defined in Section 3.2.
- Named successor: none found. This check searched one unsorted page of 100 of the paper's citing works [17], out of 758 total citations reported by Semantic Scholar's paper-lookup endpoint [18], for a paper that explicitly frames itself as fixing a documented RAFT bias or limitation, analogous to how DAPO and Dr. GRPO name themselves as fixes to GRPO; none of the titles in that one page make that claim about RAFT, and the remaining citations were not checked.
- Known failure modes: the paper's own limitations are narrow - its Discussion and Conclusion section states only that RAFT's performance heavily depends on the quality of the data set derived from the best-of-$K$ policy, which in turn depends on hyperparameter choices [1]. This check did not search LMFlow's GitHub issue tracker for maintainer replies about RAFT-specific numerical traps; none are reported on this card because that search was not performed.
- What the gain is - and is not: the paper positions RAFT as matching or exceeding PPO's mean reward and perplexity trade-off on its own HH-RLHF benchmark, and separately observes that RAFT-aligned outputs are longer and (at temperature 1.0) more diverse than the SFT baseline on its diversity metrics [1]; the paper does not claim RAFT adds any capability beyond what the underlying model and reward model can already express through sampling - it is explicitly framed as an alternative optimizer for the same alignment objective PPO targets, not a source of new capability [1].

## Sources

[1] Dong et al., "RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment", Transactions on Machine Learning Research, 2023. https://arxiv.org/abs/2304.06767 - defines RAFT: problem setup, three-step algorithm, KL-regularized extension, hyperparameter table, HH-RLHF experiments and results (Tables 3-4), limitations discussion. Fetched 2026-08-09 (ar5iv full HTML rendering of arXiv:2304.06767).

[2] Nakano et al., "WebGPT: Browser-assisted question-answering with human feedback", 2021. https://arxiv.org/abs/2112.09332 - cited by [1] for the best-of-K policy RAFT is framed as iteratively learning from. Fetched 2026-08-09 (abstract page).

[3] Cobbe et al., "Training Verifiers to Solve Math Word Problems", 2021. https://arxiv.org/abs/2110.14168 - cited by [1] alongside [2] for the best-of-K interpretation. Fetched 2026-08-09 (abstract page).

[4] Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023. https://arxiv.org/abs/2307.09288 - describes "Rejection Sampling fine-tuning" and attributes it to Bai et al. (2022b) and Scialom et al. (2020a), not to [1]; does not cite arXiv:2304.06767. Fetched 2026-08-09 (ar5iv full HTML rendering of arXiv:2307.09288).

[5] Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs", 2024. https://arxiv.org/abs/2402.14740 - names RAFT alongside DPO as an "RL-free" method it benchmarks against. Fetched 2026-08-09 (abstract page).

[6] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, the method [1] positions RAFT against and uses as its experimental baseline. Fetched 2026-08-09 (abstract page).

[7] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - states DPO eliminates the need for sampling from the LM during fine-tuning, the basis for calling it offline. Fetched 2026-08-09 (abstract page).

[8] trl RLOOTrainer documentation. https://huggingface.co/docs/trl/main/en/rloo_trainer - documents that trl's RLOO implementation uses each sample's leave-one-out mean over the other samples in its group as the baseline. Fetched 2026-08-09.

[9] Diao et al., "LMFlow: An Extensible Toolkit for Finetuning and Inference of Large Foundation Models", cited by [1] as https://optimalscale.github.io/LMFlow/. GitHub repository: https://github.com/OptimalScale/LMFlow. The README content itself was fetched from the GitHub Contents API with `ref=main`, a live pointer; the commit it resolved to at fetch time was independently confirmed by calling `GET /repos/OptimalScale/LMFlow/commits/main`, saved as `lmflow_commit.json`, which returned SHA `ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4`. Both calls made 2026-08-09.

[10] LMFlow source, `src/lmflow/pipeline/raft_aligner.py` and `src/lmflow/pipeline/auto_pipeline.py`. https://github.com/OptimalScale/LMFlow - the RaftAligner implementation, its registration guarded by a `transformers` version check, and its `collection_strategy`/generation-decoupling behavior. Fetched via the Contents API with `ref=main`; commit confirmed as SHA `ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4` by the same `lmflow_commit.json` lookup as [9]. Fetched 2026-08-09.

[11] GitHub API directory listing of `scripts/` and `scripts/archive/` in https://github.com/OptimalScale/LMFlow - shows `run_raft_align.sh` exists only under `scripts/archive/`, not the top-level `scripts/` directory. Fetched via the Contents API with `ref=main`; commit confirmed as SHA `ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4` by the same `lmflow_commit.json` lookup as [9]. Fetched 2026-08-09.

[12] trl source, `trl/__init__.py`, and https://github.com/huggingface/trl README. https://github.com/huggingface/trl - trl's top-level trainer exports (`DPOTrainer`, `GRPOTrainer`, `KTOTrainer`, `RLOOTrainer`, `RewardTrainer`, `SFTTrainer`), showing no RAFT-named trainer. Fetched via the Contents API with `ref=main`; the commit it resolved to was independently confirmed by calling `GET /repos/huggingface/trl/commits/main`, saved as `trl_commit.json`, which returned SHA `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. Both calls made 2026-08-09.

[13] verl README. https://github.com/volcengine/verl/blob/main/README.md - checked for a RAFT/rejection-sampling trainer; the only case-insensitive "raft" substring match is inside the unrelated word "draft". Fetched via the Contents API with `ref=main`; commit confirmed by `GET /repos/volcengine/verl/commits/main`, saved as `verl_commit.json`, which returned SHA `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71`. Both calls made 2026-08-09.

[14] LMFlow source, `src/lmflow/args.py`. https://github.com/OptimalScale/LMFlow - default values for `num_raft_iteration`, `raft_batch_size`, `top_reward_percentage`, `collection_strategy`; `learning_rate` and `num_train_epochs` are required run-time arguments with no default set here. Fetched via the Contents API with `ref=main`; commit confirmed as SHA `ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4` by the same `lmflow_commit.json` lookup as [9]. Fetched 2026-08-09.

[15] LMFlow documentation, "RAFT" example page. https://optimalscale.github.io/LMFlow/examples/raft.html - describes the HH-RLHF prompt/reward-model setup used in LMFlow's own RAFT walkthrough; this is a rendered docs page, not a GitHub Contents API call, so no commit SHA applies. Fetched 2026-08-09.

[16] LMFlow example launch script, `scripts/archive/run_raft_align.sh`. https://github.com/OptimalScale/LMFlow - sets `--learning_rate 2e-5` and `--num_train_epochs 4` for the RAFT SFT step. Fetched via the Contents API with `ref=main`; commit confirmed as SHA `ea19450ed4d69ba7a3d8ddfd023321f78c9ce6d4` by the same `lmflow_commit.json` lookup as [9]. Fetched 2026-08-09.

[17] Semantic Scholar Graph API, citations endpoint for arXiv:2304.06767 (`/paper/arXiv:2304.06767/citations`), saved as `raft_citations.json` - one page of 100 citing works (`offset: 0`), not sorted by citation count (its first entries have `citationCount: 0`); the response carries no total-citation-count field. Fetched 2026-08-09.

[18] Semantic Scholar Graph API, paper-lookup endpoint for arXiv:2304.06767 (`/paper/arXiv:2304.06767?fields=title,citationCount`), saved as `raft_s2_paper.json` - returns `citationCount: 758` as of the fetch time; this is a live, moving count and covers only the value at that timestamp. Fetched 2026-08-09.
