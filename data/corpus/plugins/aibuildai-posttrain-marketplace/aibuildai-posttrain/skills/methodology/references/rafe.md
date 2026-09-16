# RaFe

**Paper**: https://arxiv.org/abs/2405.14431

Train a query-rewriting model for RAG with a frozen, off-the-shelf reranker's score as the reward, so no human-labeled documents or answers are needed.

**RaFe** (Ranking Feedback improves Query Rewriting) is a two-stage framework for training a small query-rewriting model for retrieval-augmented generation (RAG), introduced to remove the need for annotated labels or task-specific reward design in query-rewriting feedback training [1]. The paper's own description: current feedback-driven query-rewriting methods rely on annotated labels such as relevant documents or answers, or pre-designed rewards tailored to specific domains, and RaFe instead leverages a publicly available reranker to score a rewrite without requiring additional labels [1]. Its parent methods are the three feedback-training algorithms it plugs the reranker score into: DPO [2], KTO [3], and PPO [4], with the PPO variant additionally using GAE for its advantage estimate [5]. Mechanism: RaFe first trains an initial rewrite model with standard supervised fine-tuning on LLM-produced rewrites, then runs a second feedback-training stage in which the reranker scores each rewrite's retrieved documents and that score drives either offline preference optimization (DPO/KTO) or online reinforcement learning (PPO) [1]. The paper gives two reasons for building it this way: query rewriting is hard to evaluate directly, so prior work fell back on annotated passages or LLM-based scoring, and the traditional IR pipeline already reranks documents retrieved by a rewritten query, so a reranker's score is a natural, label-free feedback signal for the same objective [1].

RaFe is a research method, not (as of this card) reused by a named downstream system; a Semantic Scholar citation-graph query for papers citing arXiv:2405.14431 returned 84 records as of 2026-08-09, and reading their titles found survey and query-rewriting papers referencing RaFe rather than production systems documenting adoption of it [6]. In the paper's own results, RaFe(PPO/DPO/KTO) improves over the Original Query Retrieval (OQR) baseline on most of six QA benchmarks; on the English NQ dataset, OQR (the same baseline row Table 1 lists once, ahead of both the Substitute-Raw and Expand-Raw blocks) scores 51.36 QA / 32.35 Prec@5, while RaFe(KTO)'s Substitute-Raw row scores 51.61 QA / 32.71 Prec@5, and the largest reported gain is in the Expand-Ranked setting of Table 2, where RaFe's QA score surpasses all baselines including OQR by 2-3 percentage points, a pattern the paper says also holds in Table 8 [1]. Lineage in one line: PPO (2017 [4]) / DPO (2023 [2]) / KTO (2023 [3]) -> RaFe applies reranker-derived, annotation-free rewards to all three for query rewriting (2024 [1]) -> no named successor method found as of this card's search in [6]. The paper match was confirmed by fetching the arXiv HTML full text and checking the page's own `<title>` tag against the shortlist row's title, which matched exactly with a single candidate considered [1].

**When to pick it**: pick RaFe when you are training a small query-rewriting model for RAG and want a reward/preference signal without labeled relevant documents or answers; a public reranker's ranking score over the rewrite's retrieved documents stands in for that label [1]. Contrast with its own Precision-feedback baseline (retrieval-metric feedback built from Prec@5 against annotated relevant documents) and its own LLM-feedback baseline (Qwen-32b-chat judging QA correctness): the paper's Table 4 reports RaFe needs no annotation and takes 0.67h to construct feedback for 30k instances, versus Precision feedback which needs annotation and takes 0.01h, and LLM feedback which needs annotation and takes 78h [1]. Table 3 reports QA-accuracy scores on the English FreshQA and NQ datasets, each under both Raw and Ranked retrieval, for the Substitute and Expand settings: under Substitute, RaFe(KTO) scores 62.12/62.71 (FreshQA Raw/Ranked) and 51.61/51.86 (NQ Raw/Ranked), against LLM(KTO) at 62.32/62.39 and 51.34/51.54 and Precision(KTO) at 60.54/61.34 and 49.76/50.12; under Expand, RaFe(KTO) scores 62.65/64.85 and 52.48/52.86, against LLM(KTO) at 61.87/64.08 and 51.89/52.23 and Precision(KTO) at 61.79/63.15 and 50.69/51.32 [1]. Among RaFe's own three feedback-training variants, DPO and KTO are offline (trained on a fixed batch of pre-scored good/bad rewrites) [1][2][3], while PPO is online, generating, retrieving, and scoring fresh rewrites inside the training loop each step [1][4].

**Variant of**: DPO [2], KTO [3], and PPO [4] (RaFe is the same three algorithms with the reward/preference label replaced by a reranker score; it does not change any of their objectives) [1].

**Data it needs**: for the SFT stage, query/query-rewrite pairs produced by prompting an LLM (the paper used Qwen-max to produce rewrites for six QA datasets spanning English: HotpotQA, TriviaQA, NQ, and Chinese: baike, webqa, sougouqa, squadzen, balle, coig, totaling 123,500 instances by summing Table 6's own per-dataset counts: EN hotpotqa 12,471 + triviaqa 28,083 + nq 19,445 = 59,999; ZH baike 6,552 + webqa 16,486 + sougouqa 9,488 + squadzen 6,294 + balle 9,601 + coig 15,080 = 63,501) [1]. For feedback training, no additional labels: the offline (DPO/KTO) variants need the same queries scored by the reranker on their own retrieved documents to be split into good/bad rewrites by a threshold; the online (PPO) variant needs only the queries and live access to a retriever plus the reranker at training time [1]. RaFe(DPO)/RaFe(KTO) are offline (one pass over a pre-built good/bad rewrite set); RaFe(PPO) is on-policy, sampling and scoring new rewrites at every step [1].

**Extra models**: a frozen reranker used only for scoring, never fine-tuned - the paper uses the public BAAI/bge-reranker-base [7]. A frozen reference policy $\mathcal{M}_{ref}$ (the SFT checkpoint) is required by all three variants' KL terms [1]. RaFe(PPO) additionally needs a value network $V_\phi$, initialized from the policy being trained [1] - this is the one place RaFe still carries PPO's extra trained model; DPO and KTO train no value network. See Cost for what each variant actually holds in memory.

**Shipped by**: no library ships "RaFe" as a named trainer; it is a training recipe (reranker score -> DPO/KTO/PPO) built on existing trainers. The paper's own PPO training was carried out with the TRL repository [8][1], and its DPO/KTO defaults followed the ContextualAI HALOs repository [9][1]. In the current trl main branch (commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`, 2026-08-07), `DPOTrainer` and `KTOTrainer` are both exported from the top-level `trl` package [10], but `PPOTrainer`/`PPOConfig` are not: they live under `trl.experimental.ppo`, imported as `from trl.experimental.ppo import PPOConfig, PPOTrainer` in trl's own example script, and are absent from `trl/__init__.py`'s export list and from `trl/trainer/`'s file list at that commit [11][10][12]. Building RaFe from scratch on any other trainer would require only a reward/labeling function - compute the reranker score per rewrite, threshold it into good/bad pairs for DPO/KTO or feed it directly as the PPO reward - no new training loop.

## How it works

The loop: SFT once to get a working rewrite model, then repeatedly rewrite -> retrieve -> reranker-score -> update, using that score as either a preference label (offline) or a reward (online) [1].

**SFT stage.** The rewrite model $\mathcal{M}_\theta$ is trained on LLM-produced rewrites with a standard next-token loss over the rewrite tokens conditioned on the original query [1]:

$$ \mathcal{L}_{\text{sft}} = -\sum_{q' \in Q'} \sum_t \log \mathcal{M}_\theta(q'_t \mid q'_{<t}, q) $$

**Reranker score.** For a rewrite $q'$ that retrieves documents $D'$, the reranker $\mathcal{M}_r$ scores each document against the *original* query $q$ and the rewrite's score is the mean over its retrieved set [1]:

$$ S(q, q') = \frac{1}{|D'|} \sum_{d' \in D'} \mathcal{M}_r(q, d') $$

**Offline good/bad threshold.** A single scalar threshold $\mu$ is the mean reranker score over all feedback-training instances; any rewrite scoring above it is labeled good, otherwise bad, giving pairs $(q, q'_g, q'_b)$ with no further labeling [1]:

$$ \mu = \frac{1}{|T_f|}\sum_{(q,q') \in T_f} S(q,q') $$

**DPO loss**, applied to the good/bad pairs, is the paper's own Eq. 5, unmodified from DPO's objective except that $q'_g$/$q'_b$ come from the reranker threshold rather than human preference [1][2]:

$$ \mathcal{L}_{dpo} = -\mathbb{E}_{(q,q'_g,q'_b)\sim T_f}\left[\log\sigma\!\left(\beta\log\frac{\mathcal{M}_\theta(q'_g\mid q)}{\mathcal{M}_{ref}(q'_g\mid q)} - \beta\log\frac{\mathcal{M}_\theta(q'_b\mid q)}{\mathcal{M}_{ref}(q'_b\mid q)}\right)\right] $$

**KTO loss**, applied to unpaired good/bad-labeled rewrites $(q,q';\rho)$, $\rho \in \{\text{good}, \text{bad}\}$ [1][3]:

$$ \mathcal{L}_{kto} = \mathbb{E}_{(q,q';\rho)\sim T_f}\left[w(q')\left(1-h(q,q';\rho)\right)\right], \qquad g(q,q';\rho) = \beta\log\frac{\mathcal{M}_\theta(q'\mid q)}{\mathcal{M}_{ref}(q'\mid q)} - \mathbb{E}_{q'\sim T_f}\!\left[\beta\,\mathrm{KL}(\mathcal{M}_\theta\Vert\mathcal{M}_{ref})\right] $$
$$ h(q,q';\rho) = \begin{cases} \sigma(g(q,q';\rho)) & \rho=\text{good} \\ \sigma(-g(q,q';\rho)) & \rho=\text{bad}\end{cases}, \qquad w(q') = \begin{cases}\lambda_{good} & \rho=\text{good}\\ \lambda_{bad} & \rho=\text{bad}\end{cases} $$

$\lambda_{good}$ and $\lambda_{bad}$ default to 1; when good/bad counts are imbalanced the paper requires $\dfrac{\lambda_{good}n_{good}}{\lambda_{bad}n_{bad}} \in [1, \tfrac{4}{3}]$ [1].

**PPO reward and objective.** The action at step $t$ is generating the next rewrite token; the reranker score is used directly as reward, penalized by a KL term against the reference policy [1][4]:

$$ R(s_t,a_t) = S_{\text{reranker}}(q'\mid q) - \beta_{KL}\,\mathrm{KL}(\mathcal{M}_\theta \Vert \mathcal{M}_{ref}) $$

A value network $V_\phi$ initialized from $\mathcal{M}_\theta$ produces a GAE advantage [5][1]:

$$ \delta_t = R(s_t,a_t) + V_\phi(s_{t+1}) - V_\phi(s_t), \qquad A(s_t,a_t) = \sum_{t'=0}^{\infty}\lambda^{t'}\delta_{t+t'} $$

and the final loss is PPO's clipped policy loss plus a value-regression loss [4][1]:

$$ \mathcal{L}_\theta = \mathbb{E}\left[\min\!\left(\frac{\mathcal{M}_\theta(s_t,a_t)}{\mathcal{M}_{ref}(s_t,a_t)}A(s_t,a_t),\ \mathrm{clip}\!\left(\frac{\mathcal{M}_\theta(s_t,a_t)}{\mathcal{M}_{ref}(s_t,a_t)},1-\epsilon,1+\epsilon\right)A(s_t,a_t)\right)\right],\quad \mathcal{L}_\phi = \mathbb{E}\left[(V_\phi(s_t)-R_t)^2\right],\quad \mathcal{L}_{ppo}=\mathcal{L}_\theta+\mathcal{L}_\phi $$

Worked example for the offline threshold: if a batch of four rewrites gets reranker scores $\{0.8, 0.5, 0.3, 0.2\}$, then $\mu = 0.45$, so the two rewrites scoring 0.8 and 0.5 are labeled good and the two scoring 0.3 and 0.2 are labeled bad - purely from the reranker's own numbers, with no external label consulted [1].

RaFe also defines a process-style variant for documents rather than single rewrites: the paper does not define one - it treats the whole rewrite as a single unit scored once per completion in every variant [1].

## Cost

**Theory, from the method's own math:**
- SFT: one forward/backward pass per token, one model held (the rewrite model) [1].
- Offline (DPO/KTO): each update needs one forward pass of $\mathcal{M}_\theta$ and one of the frozen $\mathcal{M}_{ref}$ per good/bad pair (DPO) or per labeled rewrite (KTO); no retrieval or reranker call happens inside this stage because good/bad labels were computed once beforehand [1]. Two models held: the trained policy (weights+gradients+optimizer state) and the frozen reference (weights only, no gradients).
- Online (PPO): every optimization step needs a fresh generation from $\mathcal{M}_\theta$, a live retrieval call, a reranker forward pass to score it, and a reference-model forward pass for the KL term - four models/components touched per step (policy, value network, reference model, reranker), versus DPO/KTO's two [1]. The value network is trained (weights+gradients+optimizer state), so PPO is the only RaFe variant carrying PPO's usual extra trained-model cost [4].

**In practice, per framework:**
- trl `DPOTrainer` / `KTOTrainer` [10]: both load a separate reference model whenever `beta > 0` in their respective configs (both default to `beta=0.1`) [13][14]; `KTOConfig` additionally exposes `desirable_weight` and `undesirable_weight`, both defaulting to `1.0` [14], matching the paper's $\lambda_{good}=\lambda_{bad}=1$ default [1]. Neither trainer performs retrieval or reranking itself - RaFe's good/bad labeling and reranker scoring must be done by the caller before the batch reaches the trainer.
- trl `trl.experimental.ppo.PPOTrainer` [11][12]: `PPOConfig` defaults to `cliprange=0.2`, `kl_coef=0.05`, and `num_ppo_epochs=4` at commit `2396dfe` [15] - the paper's own PPO run used $\epsilon=0.2$ (matching the current default) but $\beta_{KL}=0.2$, four times the current `kl_coef` default; the paper describes both values as the framework's default at the time it ran on an earlier trl version [1]; do not assume the paper's KL coefficient matches today's trl default. This trainer sits under an experimental import path rather than the top-level `trl` package, distinct from the still-supported `DPOTrainer`/`KTOTrainer`/`GRPOTrainer`/`RLOOTrainer` exports [10][11].
- Neither the paper nor either trl trainer's docs specify retriever/reranker latency; that cost depends entirely on the retrieval backend and reranker model chosen by the caller.

## How to use it

- Data prep: produce an initial SFT set of (query, rewrite) pairs from an LLM (the paper prompted Qwen-max) [1]; split it into an SFT portion and a feedback-training portion $T_f$ [1]. For $T_f$, run the current rewrite model, retrieve with each rewrite, and score the retrieved set with the reranker per Eq. 3 [1].
- Reward convention: the reranker score is computed against the *original* query $q$, not the rewrite $q'$, for every document the rewrite retrieved [1] - scoring against the rewrite itself would let a degenerate rewrite game its own retrieval.
- Offline vs. online trade-off: DPO/KTO reuse a single pre-scored batch (cheaper per step, no live retrieval during training) but freeze the good/bad labels at construction time; PPO rescoring is on-policy every step, so labels track the current policy but retrieval and reranker calls become part of the training loop's critical path [1].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give the value; "not checked" = this card did not verify that source's key):

| knob | RaFe paper value [1] | trl default [13][14][15] |
| --- | --- | --- |
| DPO/KTO $\beta$ | 0.1 (following HALOs [9]) | `beta` 0.1 (`DPOConfig`/`KTOConfig`) |
| KTO $\lambda_{good}$/$\lambda_{bad}$ | 1 / 1 (bounded to $[1,4/3]$ under imbalance) | `desirable_weight`/`undesirable_weight` 1.0 / 1.0 |
| DPO/KTO learning rate | 5e-6, 1 epoch | not checked |
| SFT learning rate / epochs | 5e-5, 2 epochs | `learning_rate` 2e-5 (overrides the base `TrainingArguments` default of 5e-5) / epochs not checked [16] |
| PPO clip range $\epsilon$ | 0.2 (paper's stated framework default at the time) | `cliprange` 0.2 |
| PPO KL coefficient $\beta_{KL}$ | 0.2 (paper's stated framework default at the time) | `kl_coef` 0.05 |
| PPO batch size / steps | 32 / 1000 steps (~1.067 epochs) | not checked |

  The PPO row is the one to check before reproducing: the paper reports its own $\epsilon$ and $\beta_{KL}$ both as the trl default at the time it ran, but today's trl `PPOConfig` default for `kl_coef` (0.05) is a quarter of the paper's stated 0.2, so the trl default that produced the paper's number is not the one shipping now [1][15].
- Design trade-off specific to this reward: the offline threshold $\mu$ is a single mean over the whole feedback set (Eq. 4), so a feedback batch dominated by easy queries pushes $\mu$ up and can relabel genuinely good rewrites on hard queries as "bad" purely from batch composition; the paper does not report re-running $\mu$ per batch versus once for the whole set, so which cadence it used is not verified here.

## While it runs

- Signals and their healthy shapes: this is a research paper, not a maintained framework, so it publishes no live training-curve monitoring guidance; the closest first-hand signal is the paper's own validation that the reranker actually separates good from bad rewrites - Table 5 reports retrieved-document precision of 46.14 (Prec@5) for rewrites the threshold labeled good versus 30.74 for those labeled bad versus 41.41 for the unrewritten original query, i.e. the good/bad split the reranker produces is not noise: good rewrites clearly beat the original query and bad rewrites clearly underperform it [1]. Falling clip fractions or KL divergence for the PPO variant would use trl's own `PPOTrainer` logging, which this card did not separately verify against RaFe's paper.
- Published reference runs: the paper's Table 1/Table 2 numbers across FreshQA/NQ/TriviaQA/HotpotQA/WebQA are the reference curve for this method, but only as final-epoch/final-step numbers - no per-step training curve or raw log is published in the fetched HTML [1].
- Degeneracies and defaults: if all rewrites in a feedback batch score above or below $\mu$ by construction, every good/bad pair collapses to the same label; the paper does not discuss guarding this case, so treat it as unverified. Separately, the PPO KL-coefficient default drift noted in How to use it (paper 0.2 vs. current trl `kl_coef` default 0.05) is a default divergence a reproduction must catch explicitly [1][15].
- Named successors: none found - the citation-graph check in [6] returned survey and application papers, not a method paper explicitly built as a fix to RaFe's own biases.
- Known failure modes, from the paper's own Limitations and Appendix A.4.1: (1) no cross-domain validation - all reranker feedback experiments use one general-purpose reranker, with no domain-specific reranker tested [1]; (2) the method's ceiling is bounded by the reranker's own capability, since RaFe cannot improve rewriting beyond what its reranker can distinguish [1]; (3) under the Substitute-Raw setting, RaFe shows only marginal gains, and on the hard NQ dataset a case study shows a semantically faithful rewrite ("What do you call the cross-like symbol on a letter 't'?") can still hurt retrieval precision versus the plainer original query by introducing vagueness a search engine cannot resolve [1]; (4) RaFe's improvement over baselines shrinks as the base model's parameter count grows (comparing Qwen-max to Qwen-32b), because larger models already produce good rewrites on the easy cases where RaFe helps most, while the harder cases where it still deviates remain [1]. No maintainer-issue-level failure modes were searched, since RaFe has no maintained library implementation to file issues against.
- What the gain is - and is not: RaFe's own case studies attribute its gains to preserving the original query's semantics better than plain SFT rewrites, normalizing terminology into forms retrievers handle better, and restructuring phrasing for retrieval even in cases with no obvious intuitive fix [1]. It is not a capability increase in the base rewrite model - the paper's own analysis (Appendix A.4.1) finds the benefit is concentrated in easy cases and shrinks with model scale, meaning RaFe mainly compensates for a smaller model's rewriting weaknesses rather than adding reasoning ability a larger model lacks [1].

## Sources

[1] Mao et al., "RaFe: Ranking Feedback Improves Query Rewriting for RAG", 2024. https://arxiv.org/abs/2405.14431 - defines RaFe: task formulation, SFT loss, reranker score, offline threshold, DPO/KTO/PPO losses, training details, all result tables, ablations, limitations. Fetched 2026-08-09 (HTML full text at https://arxiv.org/html/2405.14431).

[2] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - DPO, one of RaFe's parent methods, cited by [1] for the offline feedback-training loss. Not independently re-fetched for this card; the loss formula on this card is taken from [1]'s own restatement (Eq. 5).

[3] Ethayarajh, Xu, Jurafsky, and Kiela, "KTO: Model Alignment as Prospect Theoretic Optimization" (cited in [1] as Kawin et al., 2023, from the Contextual AI HALOs technical report), 2023. https://arxiv.org/abs/2402.01306 - KTO, one of RaFe's parent methods, cited by [1] for the offline unpaired feedback-training loss. Not independently re-fetched for this card; the loss formula on this card is taken from [1]'s own restatement (Eq. 6-7).

[4] Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, RaFe's online parent method, cited by [1] for the clipped policy objective. Not independently re-fetched for this card; the loss formula on this card is taken from [1]'s own restatement (Eq. 8-10).

[5] Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2016. https://arxiv.org/abs/1506.02438 - GAE, used inside RaFe's PPO advantage estimate, cited by [1]. Not independently re-fetched for this card; the formula on this card is taken from [1]'s own restatement (Eq. 9).

[6] Semantic Scholar Graph API, citations of arXiv:2405.14431. https://api.semanticscholar.org/graph/v1/paper/arXiv:2405.14431/citations?fields=title,year,citationCount&limit=100 - returned 84 citing-paper records as of the fetch; titles were read to check for named successor methods or landmark-system adoption of RaFe, none found. Fetched 2026-08-09 (live API, not commit-pinned; a re-query on a later date can return a different citation count).

[7] Xiao, Liu, Zhang, and Muennighof, "C-Pack: Packed Resources For General Chinese Embeddings", 2023 (defines the BGE reranker family used by RaFe as BAAI/bge-reranker-base). https://huggingface.co/BAAI/bge-reranker-base - cited in [1] as the specific reranker checkpoint used for all open-domain QA experiments. Not independently re-fetched for this card; cited via [1]'s own reference to the model page.

[8] von Werra et al., "TRL: Transformer Reinforcement Learning", 2020. https://github.com/huggingface/trl - the library RaFe's own PPO implementation was built on, per [1]'s Appendix A.2.1. Not independently re-fetched for this card as a paper; the trl repository itself was fetched directly, see [10]-[12], [13]-[15].

[9] ContextualAI, HALOs repository. https://github.com/ContextualAI/HALOs - source of the default DPO/KTO temperature $\beta=0.1$ that RaFe's own training followed, per [1]'s Appendix A.2.1. Not independently re-fetched for this card; cited via [1]'s own footnote reference.

[10] trl `__init__.py`, huggingface/trl, main branch, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0` (2026-08-07). https://github.com/huggingface/trl/blob/main/trl/__init__.py - top-level package exports: DPOTrainer, KTOTrainer, GRPOTrainer, RLOOTrainer, RewardTrainer, SFTTrainer and their configs; no PPOTrainer/PPOConfig. Fetched 2026-08-09 (raw file at that commit).

[11] trl example script `examples/scripts/ppo/ppo.py`, huggingface/trl, main branch, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py - imports `PPOConfig`, `PPOTrainer` from `trl.experimental.ppo`, confirming PPO support now lives under an experimental import path rather than the top-level package. Fetched 2026-08-09 (raw file at that commit).

[12] trl repository contents API, `trl/trainer` and `trl/experimental/ppo` directories, huggingface/trl, main branch, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. https://github.com/huggingface/trl/tree/main/trl/trainer and https://github.com/huggingface/trl/tree/main/trl/experimental/ppo - confirms `trl/trainer/` has no `ppo_trainer.py`, while `trl/experimental/ppo/` contains `ppo_trainer.py`, `ppo_config.py`, and `modeling_value_head.py`. Fetched 2026-08-09 (GitHub contents API at that commit).

[13] trl `dpo_config.py`, huggingface/trl, main branch, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py - `beta` defaults to `0.1`. Fetched 2026-08-09 (raw file at that commit).

[14] trl `kto_config.py`, huggingface/trl, main branch, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. https://github.com/huggingface/trl/blob/main/trl/trainer/kto_config.py - `beta` defaults to `0.1`, `desirable_weight` and `undesirable_weight` default to `1.0`. Fetched 2026-08-09 (raw file at that commit).

[15] trl `ppo_config.py`, huggingface/trl, main branch, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`, under `trl/experimental/ppo/`. https://github.com/huggingface/trl/blob/main/trl/experimental/ppo/ppo_config.py - `cliprange` defaults to `0.2`, `kl_coef` defaults to `0.05`, `num_ppo_epochs` defaults to `4`. Fetched 2026-08-09 (raw file at that commit).

[16] trl `sft_config.py`, huggingface/trl, main branch, commit `2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0`. https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py - `learning_rate` field defaults to `2e-5`, documented in the file's own docstring as overriding the base `TrainingArguments` default of `5e-5`; no override of `num_train_epochs` was found in this file. Fetched 2026-08-09 (raw file at that commit).
