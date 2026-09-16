# RRHF

Rank k pre-collected responses per prompt by a frozen reward score, then train the policy with a hinge loss that pushes its own length-normalized log-probabilities into the same order, plus a supervised loss on the best response.

Paper: https://arxiv.org/abs/2304.05302. Official repository: https://github.com/GanjinZero/RRHF.

**RRHF** (Rank Responses to align Human Feedback, full title "RRHF: Rank Responses to Align Language Models with Human Feedback without tears") is an offline alignment method introduced by Yuan et al. as a simpler alternative to PPO-based RLHF that scores sampled responses via a logarithm of conditional probabilities and learns to align these probabilities with human preferences through ranking loss [1]. Its parent, in the paper's own framing, is PPO as used in InstructGPT's RLHF pipeline [2][3]: for each prompt x, RRHF collects k responses from any mix of sources (the policy itself, other LLMs, human-written text), scores each response i by its length-normalized log-probability under the model being trained, $p_i = \sum_t \log P_\pi(y_{i,t}\mid x, y_{i,<t}) / \|y_i\|$ [1], and optimizes a ranking loss that penalizes any pair whose model-score order disagrees with the reward order, added to a cross-entropy loss on the highest-reward response [1]. The paper gives three reasons for existing over PPO: PPO needs a policy, value, reward, and reference model simultaneously, which the paper calls memory-unfriendly [1]; PPO's advantage estimation needs a learned value-model baseline, while RRHF gets its baseline for free by comparing sampled responses' log-probabilities directly [1]; and because RRHF samples before training rather than during it, "the KL term degenerates" and no reference model is needed unlike PPO's KL-penalized objective [1].

A Semantic Scholar citation-graph query returned an unsorted first batch of 100 papers citing RRHF (offset 0; the API's own `next` field shows more citing papers exist beyond this batch), with only each citing paper's title, year, and citation count returned - no abstract or full text [4]. Sorting that batch locally by citing-paper citation count, the most-cited entry by title is "A survey on LLM-as-a-judge" (70 citations, 2026); no title in this 100-paper batch names a production system whose training method is RRHF [4]. Titles alone cannot show how any of these papers actually uses RRHF (e.g., as a baseline versus an adopted method), and this checks only one batch, not the full citing-paper list, so it cannot rule out landmark adoption stated in a title-invisible way or outside this batch. The paper's own adopter is its own model: the authors trained Wombat, an Alpaca-7B model fine-tuned by RRHF on ChatGPT-scored responses, and released its weights [1][5]. In the paper's main HH-dataset experiment, Alpaca trained with RRHF using top-p sampling (RRHF$_{SP}$) reached an average reward-model score of -0.96, versus -1.03 for the paper's own PPO baseline on the same Alpaca initialization and -1.24 for the unmodified good responses from the dataset (Table 2) [1]. The paper's own ablation shows the ranking loss is what makes RRHF work: removing $L_{rank}$ from the total loss (Table 7) drops the Alpaca-initialized run's average reward from -1.03 to -1.14, and the paper states that without the ranking loss "models cannot learn from how one response is better than another and obtain a worse average reward score" [1]. Lineage in one line: PPO (2017 [2]) -> RRHF (2023 [1]) -> Wombat (2023, the paper's own downstream model [1]).

**When to pick it**: offline alignment when you already have, or can cheaply generate, several ranked or reward-scored responses per prompt and want to avoid running PPO's four-model training loop [1]. Prefer PPO [2] (its nearest online neighbor per the paper's own comparison) when you can afford online sampling and a value model and want on-policy updates - PPO "leverages π for sampling" while RRHF "is sampling before training to get rid of the KL divergence term" [1]. Prefer DPO [6] as the nearest offline alternative when you have exactly two responses per prompt (chosen/rejected pairs) rather than RRHF's k-way ranking over an arbitrary number of responses [1]; [6] is a naming citation only, not independently fetched for this card, and [1] does not discuss DPO.

**Variant of**: PPO [2], per the paper's own framing of RRHF as a simplification of the PPO stage of RLHF [1].

**Data it needs**: per prompt, k sampled responses (any mix of self-generated, other-LLM-generated, or human-authored) each with a scalar reward score; the official training script consumes one JSON object per line with fields `query`, `responses` (list), and `scores` (list) [7]. Offline: all responses are sampled from fixed source policies before training begins, not refreshed against the model being trained during training - the paper states "RRHF is sampling before training" [1], though it also reports an online variant (OP-k, resampling from the live policy every k steps), which the paper states takes about 30 hours versus 4-6 hours for the offline sampling policies used in its main results, and an iterative variant (IP-n, retraining a new round of samples from the previous round's output), for which the paper does not give a comparable training-time figure [1]. The main HH-dataset experiments used k=4 sampled responses (plus 2 dataset-provided responses in most settings) per query, and training ran on 8×80GB A100 GPUs for 3 epochs [1].

**Extra models**: none required by the method's own definition beyond the reward source used to score responses before training - no value network and no reference model, since the KL term is absent from the loss (Eq. 5) [1]. If the reward source is itself a model (as in the paper's Dahoas/gptj-rm-static proxy reward model [1]), that reward model runs only during data preparation, not during the RRHF training loop itself. The official training script's `RRHFTrainer` holds a single model in memory [7].

**Shipped by**: no major RL-post-training library was found to implement RRHF. Neither trl nor verl lists an RRHF trainer or advantage estimator in their source trees as read at commit-level snapshots below [8][9], and a GitHub repository-search for "RRHF" turned up no adopting framework besides the authors' own repository [10]. Building it on top of an existing SFT trainer requires two additions the paper itself describes as small: a ranking-hinge loss over length-normalized sequence log-probabilities, and a per-batch max-reward cross-entropy term - the authors state the change "only adds 30 lines to SFT training code" [1]. The authors' own reference implementation, GanjinZero/RRHF, subclasses Hugging Face's `transformers.Trainer` as `RRHFTrainer` and overrides `compute_loss` [7]; it is a standalone research script, not an installable package with a stated support status.

## How it works

Score every sampled response with the model being trained, then push the model's own score order into agreement with the reward order via a hinge loss, and separately do next-token cross-entropy on the best-reward response.

**Scoring** [1]: for query x and response $y_i$ sampled by some source policy $\rho_i$,

$$ p_i = \frac{\sum_t \log P_\pi(y_{i,t}\mid x, y_{i,<t})}{\|y_i\|} $$

$p_i$ is the length-normalized conditional log-probability of $y_i$ under the model $\pi$ currently being trained (Eq. 1) [1].

**Ranking loss** [1]:

$$ L_{rank} = \sum_{r_i < r_j} \max(0,\, p_i - p_j) $$

summed over every pair of responses whose reward order ($r_i < r_j$, from the external reward function $R(x,y_i)=r_i$) is violated by the model's own score order (Eq. 2) [1]. The paper explicitly drops the margin term $\lambda_{ij}=(j-i)\lambda$ used by the summarization ranking-loss paper it credits as its inspiration, reporting good empirical results without it and citing the cost of tuning $\lambda$ [1].

**SFT loss** [1]: with $i' = \arg\max_i r_i$ the index of the highest-reward response,

$$ L_{ft} = -\sum_t \log P_\pi(y_{i',t}\mid x, y_{i',<t}) $$

an ordinary next-token cross-entropy loss on that one best response (Eqs. 3-4) [1].

**Total loss** [1]:

$$ L = L_{rank} + L_{ft} $$

the paper states this is an unweighted sum and reports that giving $L_{rank}$ a larger weight (10 or 100, as the summarization-loss paper it draws on suggests) gave worse results in its preliminary experiments (Eq. 5) [1].

Worked example, k=3 responses to one query with rewards $r=\{2, 5, 1\}$ (so response 2 is best): the model computes length-normalized log-prob scores, say $p=\{-0.40,\,-0.55,\,-0.30\}$. Reward order says $r_3<r_1<r_2$. Checking every pair with $r_i<r_j$: pair (3,1): $r_3<r_1$ but $p_3-p_1=-0.30-(-0.40)=+0.10>0$, so it violates and contributes $\max(0,0.10)=0.10$; pair (3,2): $r_3<r_2$, $p_3-p_2=-0.30-(-0.55)=+0.25>0$, contributes $0.25$; pair (1,2): $r_1<r_2$, $p_1-p_2=-0.40-(-0.55)=+0.15>0$, contributes $0.15$. $L_{rank}=0.50$ over this query (arithmetic follows Eq. 2 directly; it is not attributed to any framework default since none ships this loss). $L_{ft}$ is the ordinary cross-entropy on response 2, the highest-reward response.

The official implementation's `get_score` divides the summed log-probability by `length ** length_penalty` rather than by length directly, with `length_penalty` defaulting to 1.0 in its `TrainingArguments` (so length_penalty=1.0 reduces to the paper's $p_i$) [7]; its `rrhf_loss` implements the same pairwise-violation hinge as Eq. 2 [7].

## Cost

**Theory, from the method's own math**: the training forward/backward pass runs once over k responses per prompt (for the k-way ranking loss and the SFT term), all under the single model being trained - no value network, no reference model, no reward model inside the training loop, since scoring by an external reward function happens before training starts to build the (query, responses, scores) dataset [1]. This removes two of PPO's four models from the training-time memory footprint entirely (the paper states PPO "utilizes 4 models during training, whereas RRHF requires only 1 or 2 models" - the policy being trained, plus optionally a reward model if scoring is done online rather than pre-computed) [1]. The paper's own limitations section flags the naive reading directly: "RRHF requires multiple responses as inputs which increases the GPU usage for a single query compared to PPO," and separately notes that its online-sampling variant (OP-k) is slower than both offline RRHF and PPO [1].

**In practice, per framework**: the authors' official `train.sh` script runs RRHF as an FSDP-sharded `transformers.Trainer` subclass on 8×80GB A100 GPUs with `per_device_train_batch_size 1` and `gradient_accumulation_steps 8` [7]; the paper reports 4-6 hours of training time on that hardware for the offline (BP/SP/DP) sampling policies used in its main results, versus about 30 hours for the online-sampling (OP) variant [1]. No packaged framework (trl, verl) was found to ship this trainer, so no framework-level batching or memory-management defaults exist to report beyond the official script [8][9].

## How to use it

Data preparation, per the official script's `ScoreDataset`/`DataCollatorForSupervisedDataset`: one JSON object per line with a `query`, a list of `responses`, and a matching list of scalar `scores`; every response in the list is tokenized and concatenated to the query, with the query tokens masked out of the loss labels [7]. The paper's main experiments used the Anthropic Helpful and Harmless (HH) dataset, scoring sampled responses with the proxy reward model Dahoas/gptj-rm-static so PPO and RRHF could be compared under an identical reward signal [1]; its Wombat variant instead used ChatGPT itself as the scorer over Alpaca-format prompts [1].

Key knobs, with the paper's own values as the anchor and the official script's defaults noted separately, since no packaged framework ships this method:

| knob | paper (main HH experiment) [1] | official script default [7] |
| --- | --- | --- |
| responses per query (k) | 4 sampled + up to 2 dataset-provided | not fixed by the trainer; set by the data file |
| rrhf_weight (weight on $L_{rank}$) | 1 (unweighted sum, per Eq. 5) | `TrainingArguments.rrhf_weight` defaults to 100.0 in the dataclass, but the shipped `train.sh` overrides it to `--rrhf_weight 1` to match the paper [1][7] |
| length_penalty † | implicit (plain length normalization, exponent 1) | `TrainingArguments.length_penalty` defaults to 1.0 [7] |
| learning rate | 2e-5, warmed up then decayed linearly to 0 | `train.sh` passes `--learning_rate 2e-5` [1][7] |
| epochs | 3 | `train.sh` passes `--num_train_epochs 3` [1][7] |
| max sequence length | 192 tokens (query+response) | `train.sh` passes `--model_max_length 192`; the dataclass default is 512 [1][7] |
| margin term $\lambda$ | disabled (no margin in Eq. 2) | not implemented in `rrhf_loss` [1][7] |

† Raising `length_penalty` above 1 does not penalize length the way its name implies: since `get_score`'s summed log-probability is negative, dividing by `length ** length_penalty` with an exponent above 1 makes the score less negative (higher) for longer responses, not lower - see While it runs, Known failure modes [7][12].

The paper's own ablation is the trade-off a run designer faces on sampling policy rather than batch size: reward score is highly correlated with the quality of the k sampled responses fed in, to the point that the paper characterizes RRHF as learning "from best-of-n sampling" - the trained model's reward tends to approach the maximum reward among its own sampled training responses rather than exceed it by a wide margin [1].

## While it runs

- **Signals and their healthy shapes**: the paper reports that its loss and average reward-model score are negatively correlated during training, so the loss curve can be used to track the reward trend, and states that losses "converge at the third epoch" in its main HH run, with average reward also reaching its maximum around the same point [1]. No framework-level logging convention exists to point at, since no packaged framework ships this trainer [8][9].
- **Published reference runs**: the paper's own Table 2 is the reference point - on the HH dataset with an Alpaca initialization, RRHF with diverse-beam sampling (RRHF$_{DP}$) reached reward -1.03 and perplexity 14.75, RRHF with top-p sampling (RRHF$_{SP}$) reached reward -0.96 and perplexity 14.41, against a PPO baseline on the same Alpaca initialization at reward -1.03 and perplexity 13.84, and unmodified "good" dataset responses at reward -1.24 [1]. Its human-evaluation table (Table 3) reports RRHF$_{DP}$ beating the dataset's own "good" responses 59 wins to 11 losses (30 ties) and beating its PPO baseline 27 wins to 25 losses (48 ties), on Alpaca-initialized models judged by human annotators [1].
- **Degeneracies and defaults**: the ranking loss (Eq. 2) sums over pairs with $r_i<r_j$ only where the model's own order currently disagrees ($p_i>p_j$); if all k sampled responses carry the same reward, no pair satisfies $r_i<r_j$ and $L_{rank}$ contributes nothing for that query, so a batch built from uniform-reward samples trains only via the SFT term on whichever response the max-reward tie-break happens to select. The official script's `rrhf_weight` dataclass default (100.0) diverges sharply from the value the same script's shipped launch command actually passes (1, matching the paper) - running the trainer with only the dataclass default and no override would not reproduce the paper's setting [7].
- **Named successors**: RRHF-V applies a ranking-response approach to multimodal large language models specifically to mitigate hallucinations, per its own repository description [11].
- **Known failure modes**: from the paper's own Limitations section - RRHF requires multiple responses per query, which "increases the GPU usage for a single query compared to PPO"; the online- and iterated-sampling variants are "prone to over-optimization to cheat the reward models" in the authors' preliminary experiments, a problem the paper states is shared with PPO and best-of-n sampling rather than unique to RRHF; and the method's alignment quality is only as good as the reward signal used to score responses, so "malicious or harmful reward scores or human preference ratings may mislead the LLM to generate unsafe results" [1]. A search of the GanjinZero/RRHF issue tracker (`state=all`, 57 issues returned) turned up a maintainer-thread numerical trap in open issue #51: a user points out that `get_score`'s summed log-probability is already negative, so dividing it by `length ** length_penalty` with `length_penalty > 1` makes the score less negative (higher), the opposite of what the name "length_penalty" suggests; the maintainer's reply ("length is always positive") does not address that the numerator, not the denominator, is negative, and the issue remains open with no code fix [12].
- **What the gain is - and is not**: the paper's own analysis states that performance is "highly related to sampling quality," concluding RRHF is fundamentally a best-of-n learner - it drives the trained model's own reward toward the maximum reward already present among its sampled training responses, rather than discovering higher-reward behavior the sampled responses never demonstrated [1]. In the paper's Wombat comparison against ChatGPT on the 80-question Vicuna test set, Wombat still underperformed ChatGPT, which the paper attributes mainly to weaker logical reasoning ability, naming it a direction for future work rather than a gap RRHF's ranking loss closes [1].

## Sources

[1] Yuan et al., "RRHF: Rank Responses to Align Language Models with Human Feedback without tears," 2023 (NeurIPS 2023; v3 revised 2023-10-07). https://arxiv.org/abs/2304.05302 - defines RRHF: objective, loss, relation to SFT/reward-model/PPO, HH-dataset experiments, ablations, Wombat, Limitations. Fetched 2026-08-09 (PDF via arxiv.org/pdf/2304.05302, extracted with pdftotext).

[2] Schulman et al., "Proximal Policy Optimization Algorithms," 2017. https://arxiv.org/abs/1707.06347 - PPO, cited by [1] as reference [28] for the clipped surrogate objective. Not independently fetched for this card; PPO's formula as used in [1]'s own comparison (Eq. 6 of [1]) is cited via [1].

[3] Ouyang et al., "Training language models to follow instructions with human feedback," 2022. https://arxiv.org/abs/2203.02155 - InstructGPT, cited by [1] as reference [22] for the SFT/reward-model/PPO RLHF pipeline that RRHF simplifies. Not independently fetched for this card.

[4] Semantic Scholar Graph API, citations of arXiv:2304.05302, offset 0, first 100 entries returned by the endpoint's default (unsorted) ordering, not sorted by citation count. https://api.semanticscholar.org/graph/v1/paper/arXiv:2304.05302/citations - used to check for landmark-system adoption among this batch of citing papers, sorted locally by citation count after fetching; this covers only this one 100-paper batch, not the full citing-paper list. Fetched 2026-08-09.

[5] GanjinZero/RRHF README. https://github.com/GanjinZero/RRHF - the README's title describes the repository as covering the move "from RLHF to RRHF" and the release of the Wombat model, and states the Wombat model weights were released 2023-04-13. Fetched 2026-08-09, read at commit `e1a2b61f7d91fbee4cfaa3923327fcc5c5c733de` on branch `main`.

[6] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023. https://arxiv.org/abs/2305.18290 - DPO, the nearest offline pairwise-preference alternative. Not independently fetched for this card.

[7] GanjinZero/RRHF, `train.py` and `train.sh`. https://github.com/GanjinZero/RRHF/blob/main/train.py and https://github.com/GanjinZero/RRHF/blob/main/train.sh - `RRHFTrainer`, `TrainingArguments` defaults, `ScoreDataset`/`DataCollatorForSupervisedDataset` data format, and the shipped launch command's hyperparameters. Fetched 2026-08-09, read at commit `e1a2b61f7d91fbee4cfaa3923327fcc5c5c733de` on branch `main`.

[8] huggingface/trl, repository file tree. https://github.com/huggingface/trl - `trl/__init__.py` and full recursive tree contain no file or export path matching "rrhf". Fetched 2026-08-09, read at commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0` on branch `main`.

[9] verl-project/verl, repository file tree (repository moved from `volcengine/verl`). https://github.com/verl-project/verl - full recursive tree contains no file or export path matching "rrhf". Fetched 2026-08-09, read at commit `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71` on branch `main`.

[10] GitHub repository search for "RRHF", sorted by stars. https://api.github.com/search/repositories?q=RRHF - the only alignment-relevant result is the authors' own GanjinZero/RRHF (805 stars); no RL post-training framework repository appears among the top 20 results. Fetched 2026-08-09.

[11] chengq1001/RRHF-V, repository description, "[COLING'25] RRHF-V: Ranking Responses to Mitigate Hallucinations in Multimodal Large Language Models with Human Feedback." https://github.com/chengq1001/RRHF-V - description only, surfaced via the same GitHub repository search as [10]; the RRHF-V paper itself was not fetched for this card. Fetched 2026-08-09.

[12] GanjinZero/RRHF, issue #51 and its comments (open as of fetch date). https://github.com/GanjinZero/RRHF/issues/51 - user "IT-five" flags that `get_score`'s length-penalty exponent makes the score less negative, not more, for longer responses when the exponent exceeds 1, because the summed log-probability being divided is negative; maintainer "GanjinZero" replies that length is always positive, which does not address the numerator's sign. Fetched 2026-08-09 via the GitHub REST API (`issues/51` and `issues/51/comments` endpoints).
