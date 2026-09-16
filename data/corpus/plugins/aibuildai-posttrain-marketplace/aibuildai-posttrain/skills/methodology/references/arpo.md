# ARPO

Item home: https://arxiv.org/abs/2507.19849 [1]

GRPO for tool-using agents: sample fewer full trajectories per prompt and spend the saved budget branching mid-trajectory right after a tool call, exactly where token entropy spikes, then average the shared prefix's advantage across the branches it feeds.

**ARPO** (Agentic Reinforced Policy Optimization) is an online-RL algorithm for training multi-turn, tool-using LLM agents, introduced because the authors judge that existing RL training methods do not properly weigh a model's step-by-step reasoning skill against its skill at repeatedly calling and reacting to external tools [1]. Its parent is GRPO, introduced by Shao et al. as a memory-saving variant of PPO for mathematical reasoning [2]; ARPO restates GRPO's own clipped, group-normalized surrogate objective as the base it builds on, including the group-relative advantage and reference-policy KL term [1]. The mechanism has two parts: an entropy-based adaptive rollout that spends part of a fixed per-prompt sampling budget branching new trajectory forks right after tool calls, where token entropy is measured to spike, instead of only sampling full independent trajectories as GRPO does [1]; and an advantage-attribution step that keeps the branches' shared prefix tokens on a common advantage value instead of letting importance-sampling ratios silently blend shared and diverged tokens [1]. Three reasons the paper gives: a pilot study found token entropy rises sharply in the first 10-50 tokens after each tool call and remains elevated afterward, which trajectory-level sampling does not specifically target [1]; naive per-trajectory branching from scratch costs O(n^2) in trajectory length, and the paper's rollout mechanism reduces this to between O(n log n) and O(n^2) [1]; and GRPO's own token-level importance ratio already treats identical token prefixes identically, which the paper's Advantage Attribution Estimation formalizes rather than leaves implicit [1].

ARPO is the authors' own algorithm, evaluated only within its own paper at the time of this card. A Semantic Scholar API query for the arXiv 2507.19849 citation graph, read on 2026-08-09, returned the 50 citing-paper titles requested via the query's own `limit=50` parameter (the full set of results retrieved, not the endpoint's own default page size); none names a landmark model-training system such as DeepSeek-R1 or Qwen3, all instead being other RL-agent-training method papers, so this card did not find a landmark third-party system's technical report describing adoption of ARPO [3]. In the originating paper, on 7B/8B/14B Qwen and Llama backbones ARPO outperformed GRPO, DAPO, and REINFORCE++ across 10 combined math and knowledge-reasoning benchmarks (Table 1), and on deep-search tasks trained with only 1K samples it beat GRPO by 6 points on GAIA and WebWalkerQA (Table 2 discussion) [1]. Lineage in one line: PPO -> GRPO (DeepSeekMath, 2024 [2]) -> ARPO (2025 [1]), itself compared against the trajectory-level GRPO variant DAPO [4] and REINFORCE++ [1]. The right paper was confirmed by a top-cited match among 3 candidates under an exact title search, resolving a name collision with two unrelated works also abbreviated ARPO (Adversarial Robust Policy Optimization, and a reverse-prompt-optimization method) that had far fewer citations.

**When to pick it**: online RL for training an LLM agent that calls external tools (search, browser, code) across multiple turns, when the reward is scored at the end of a trajectory (exact-match or F1 against a ground-truth answer in the paper's setup) and you want exploration concentrated at the specific tool-call steps where the policy is most uncertain, rather than spread evenly across whole trajectories [1]. Prefer the parent GRPO [2] when your task is single-turn or when tool calls do not introduce a measurable entropy spike worth targeting. Prefer DPO [5] when you have static preference pairs and cannot sample fresh multi-turn trajectories against live tools: DPO is offline, trained on pairs of completions to the same prompt without an explicit reward model [5]. Nearest online neighbor is DAPO, another GRPO-derived trajectory-level algorithm that decouples the clip ranges and adds dynamic sampling, reported to reach 50 points on AIME 2024 with a Qwen2.5-32B base [4]; the ARPO paper reports DAPO performs well on single-turn reasoning but underperforms in multi-turn tool-call interaction compared to ARPO [1].

**Variant of**: GRPO [2], restated inside the ARPO paper with the same clipped surrogate, group-normalized advantage, and reference-policy KL term [1].

**Data it needs**: prompts paired with a checkable final answer, scored during RL by a piecewise reward built from format compliance, answer accuracy, and a multi-tool-use bonus (defined exactly in How to use it), not by a plain exact-match/F1 score — the paper's RL phase used 10K Tool-Star reasoning/knowledge prompts for the deep-reasoning setting and only 1K mixed hard-search prompts (from SimpleDeepSearcher and WebSailor) for the deep-search setting [1]. On-policy: each rollout regenerates completions, including the branched partial rollouts, from the current policy snapshot each step; no fixed offline trajectory set is used [1]. A separate cold-start SFT stage on ~54K Tool-Star samples precedes RL in the paper's pipeline, and is not part of the RL algorithm itself [1].

**Extra models**: no value network, inherited from GRPO's group-baseline design [2]. The objective as written keeps a KL term against a frozen reference policy [1], but the paper's own implementation detail states the KL divergence coefficient in GRPO is set to 0 to stabilize training in both the deep-reasoning and deep-search settings, so no reference-model forward pass runs in the reported experiments [1]. No reward model: the RL-phase reward is computed by a rule-based piecewise scoring function over format compliance, answer accuracy, and tool-use behavior, not a learned reward model (formula in How to use it) [1]. Details in Cost.

**Shipped by**: no mainline framework ships ARPO — a case-insensitive text search for "arpo" in trl's GRPOTrainer documentation page and verl's GRPO algorithm documentation page, both read 2026-08-09, returned zero matches in either file [6][7]. The only running implementation is the authors' own fork of verl, bundled in their GitHub repository under `ARPO/verl_arpo_entropy/`, launched via `python3 -m verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` and rollout flags `rollout.n`, `rollout.initial_rollouts`, `rollout.beam_size`, `rollout.branch_probability`, `rollout.entropy_weight` added to a modified vLLM rollout worker (not the installable `verl` package) [8]. Building it on top of an existing GRPO trainer requires adding entropy-triggered mid-generation branching to the rollout loop plus a shared-prefix advantage-attribution pass — a new sampling loop, not just a loss change.

## How it works

Each step: sample a group of trajectories per prompt, but generate only N of a global budget M as independent full rollouts; for the remaining M-N, monitor token entropy after each tool call and branch new partial trajectories from high-entropy tool-call steps instead of always continuing; score every completed trajectory; assign advantages so tokens shared by several branches get one averaged advantage; update on the clipped GRPO objective [1].

**Entropy signal.** Token-level generation entropy at step t is [1]:

$$ H_t = -\sum_{j=1}^{V} p_{t,j}\log p_{t,j}, \qquad \bm{p}_t = \pi_\theta(\cdot \mid \mathcal{R}_{<t}, x; T) = \operatorname{Softmax}\!\left(\frac{\bm{z}_t}{\tau}\right) $$

where $V$ is vocabulary size, $\bm{z}_t$ the pre-softmax logits, and $\tau$ the decoding temperature; this reflects uncertainty in the generation distribution, not any one token's identity [1]. The paper's pilot study on search- and Python-tool agents found entropy rises sharply in the first 10-50 tokens after each tool call, stays elevated afterward, and rises more after search feedback than after Python feedback [1].

**Entropy-based adaptive rollout.** Given a global rollout budget $M$, the policy first generates $N$ full trajectories by ordinary trajectory-level sampling, reserving $M-N$ for partial sampling [1]. For each tool-call step $t$, the entropy change relative to the trajectory's initial entropy is normalized by vocabulary size [1]:

$$ \Delta H_t = \text{Normalize}(H_t - H_{\text{initial}}) $$

The branching probability at step $t$ is [1]:

$$ P_t = \alpha + \beta\cdot\Delta H_t, \qquad \text{Action}(P_t) = \begin{cases} \text{Branch}(Z), & \text{if } P_t > \tau \\ \text{Continue}, & \text{otherwise} \end{cases} $$

where $\alpha$ is a base sampling probability and $\beta$ a stability weight on the entropy signal; when $P_t$ exceeds threshold $\tau$, the trajectory forks into $Z$ new partial paths from that node, otherwise it continues [1]. Branching stops once the forked-path count reaches the reserved budget $M-N$; if all paths terminate first, the shortfall is filled with ordinary full trajectory samples [1]. The paper reports this reduces per-rollout computational complexity from trajectory-level RL's $O(n^2)$ to between $O(n\log n)$ and $O(n^2)$, for global expansion size and per-trajectory token count both denoted $n$ [1].

**Advantage attribution.** Given $d$ trajectories that share a token prefix and diverge afterward, each trajectory's own per-token advantage under the usual group-normalized reward is [1]:

$$ \hat{A}_{i,t} = \frac{r_i - \operatorname{mean}(\{R_i\}_{i=1}^G)}{\operatorname{std}(\{R_i\}_{i=1}^G)} $$

The paper defines two ways to combine this at the shared prefix. Hard Advantage Estimation explicitly averages the $d$ trajectories' own advantages over the shared segment [1]:

$$ \hat{A}_{i,t}^{\text{shared}} = \frac{1}{d}\sum_{i=1}^{d} \hat{A}_{i,t} $$

Soft Advantage Estimation instead folds the shared/individual distinction into the GRPO objective itself via the importance-sampling ratio, which is identical across trajectories on shared tokens ($y_{i,<t}=y_{j,<t} \Rightarrow r_{i,t}(\theta)=r_{j,t}(\theta)$) and differs once trajectories diverge [1]:

$$ J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\!\Big(r_{i,t}(\theta)\hat{A}_{i,t},\ \operatorname{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\Big) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right] $$

where $r_{i,t}(\theta) = \pi_\theta(y_{i,t}\mid x, y_{i,<t}) / \pi_{\text{ref}}(y_{i,t}\mid x, y_{i,<t})$ [1]. Worked micro-example: two branches fork after a shared 20-token prefix and a tool call; branch A ends correct ($r_1=1$) and branch B ends incorrect ($r_2=0$) with two other independent trajectories in the same group of $G=4$ scoring 0 and 0 — mean $0.25$, population std $\approx 0.433$, giving branch A advantage $\approx+1.73$ and branch B $\approx-0.58$ on their own diverged tokens (arithmetic per the group-normalization formula above [1]); under Hard Advantage Estimation the shared 20-token prefix instead gets the average of the two branches' own advantages, $(1.73 + (-0.58))/2 \approx 0.58$, rather than either branch's individual value.

**Theoretical foundation.** The paper derives a Generalized Policy Gradient (GPG) Theorem treating each branch/continuation decision as a macro-action over a segmented token sequence, intended to justify the rollout mechanism for any Transformer-based policy; the full derivation is in the paper's Appendix D [1].

Both Advantage Attribution variants are defined by the paper, which reports that in its own Figure 5 comparison the soft setting reaches consistently higher reward with greater training stability than the hard setting, and states that ARPO defaults to the soft setting for advantage estimation as a result [1]. The code repository's launch scripts expose `beam_size`, `branch_probability`, and `entropy_weight` as rollout config flags, but this card did not verify from the source code itself whether the shipped reward pipeline's own default matches the paper's stated soft-setting default [8].

## Cost

**Theory, from the method's own math:**

- Time: no per-trajectory savings are free — the paper's own complexity result assumes the entropy-based branching stays between $O(n\log n)$ and $O(n^2)$ against trajectory-level RL's $O(n^2)$, neglecting the added cost of computing per-token entropy at each tool-call boundary [1]. Whether this bound is realized in wall-clock time depends on how many tool calls actually trigger branching, which is data- and threshold-dependent and not given as a single number in the paper.
- Memory: same as GRPO — no value network [1][2]. With the KL coefficient set to 0, as the paper's own experiments do, no reference-model forward pass is needed either [1]; a naive reading of the objective (which keeps the $\beta D_{KL}$ term) would otherwise require one frozen reference-model copy, as in GRPO.
- A naive reading of "branch into $Z$ partial paths" suggests holding all $Z$ branches' activations in memory simultaneously from the fork point onward; whether the shipped rollout worker does this or streams branches is not verified from source here (see below).

**In practice, per framework:**

- The paper's own fork of verl (`verl_arpo_entropy`) [1][8]: Appendix C.2 states all training runs used a total training batch size of 128, a PPO mini-batch size of 16, a global rollout size of 16, and an initial sampling size of 8, on 8 NVIDIA H800 GPUs for 7B/8B deep-reasoning and deep-search runs, and 16 H800 GPUs for the 14B deep-search run [1]. Per-response length is capped at 4096 tokens for deep-reasoning and math tasks and extended to 8192 tokens for deep-search tasks [1]. This card did not open the rollout worker's own memory-management code, so whether concurrent branches share KV-cache pages or are computed sequentially is not checked here.
- No mainline framework (trl, verl) implements ARPO, so no second framework's cost profile exists to compare against; the framework skill's own cards for trl/verl carry their GRPO baseline costs.

## How to use it

- Data preparation: prompts need a checkable final answer. The paper's cold-start SFT stage draws on ~54K Tool-Star samples plus an 0.8K STILL math subset before RL begins [1]; the RL stage itself uses 10K Tool-Star prompts for deep-reasoning tasks or 1K SimpleDeepSearcher/WebSailor prompts for deep-search tasks [1].
- Reward convention: the RL training reward, following Tool-Star, combines a correctness term with a format check and a multi-tool bonus $r_M$ [1]:

$$ R = \begin{cases} \max(Acc.+r_M,\ Acc.) & \text{if Format is Good \& } Acc.>0 \\ 0 & \text{if Format is Good \& } Acc.=0 \\ -1 & \text{otherwise} \end{cases}, \qquad r_M = \begin{cases} 0.1 & \text{if both } \texttt{<search>} \text{ and } \texttt{<python>} \text{ are used} \\ 0 & \text{otherwise} \end{cases} $$

  This is separate from the paper's reported benchmark evaluation metric, which is not the training-time reward: for post-training evaluation the paper scores four knowledge-intensive QA benchmarks with F1, scores other benchmarks with Qwen2.5-72B-Instruct as an LLM judge, and extracts the final answer from a `\boxed{}` span under pass@1 sampling at temperature 0.6 / top-p 0.95 [1]. This card did not verify from source what specific accuracy check the training-time $Acc.$ term uses.
- Knobs, paper values as the only published anchor since no framework ships defaults of its own:

| knob | paper (Appendix C.2) [1] | repo launch script (7B reasoning) [8] |
| --- | --- | --- |
| global rollout size $M$ | 16 | `ROLLOUT_N` 16 |
| initial full-trajectory samples $N$ | 8 | `INITIAL_ROLLOUTS` 8 |
| branch count $Z$ per fork | not stated as a fixed number | `BEAM_SIZE` 2 |
| branch probability param | $\alpha=0.5$, threshold $\tau=0.5$ | `BRANCH_PROBABILITY` 0.5 |
| entropy weight $\beta$ | 0.2 | `Entropy_weight` 0.2 |
| KL coefficient | 0 (set to stabilize training) | `algorithm.kl_ctrl.kl_coef` 0.0 |
| PPO mini-batch size | 16 | `PPO_MINI_BATCH_SIZE` 16 |
| training batch size | 128 | `TRAIN_BATCH_SIZE` 128 |
| max response length | 4096 (deep-reasoning), 8192 (deep-search) | `MAX_RESPONSE_LENGTH` 4096 (7B reasoning script) |
| RL epochs | 2 (deep-reasoning, 7B/8B); 5 (deep-search, 1K-sample setting) | `TOTAL_EPOCHS` 2 (7B reasoning script) |
| max tool calls per trajectory | not stated in the paper text checked here | 5, per a maintainer reply to a GitHub issue [9] |

  The repo's single `BRANCH_PROBABILITY` flag is not confirmed from source code to separately implement both the paper's base rate $\alpha$ and threshold $\tau$; this card only verifies that both take the value 0.5 in the paper and 0.5 in the script.
- Trade-off: raising $N$ (more full trajectories, less branching budget) moves ARPO toward plain GRPO; raising $Z$ (branch count) concentrates more of the fixed budget $M$ on fewer high-entropy tool-call sites per trajectory rather than spreading across more sites.

## While it runs

- Signals: the entropy quantity $H_t$ and its branching trigger $P_t$ are the method's defining signals per the paper's own formulas [1]. Unlike trl, which publishes a dedicated logging guide for GRPO's crucial values [6], no equivalent ARPO-specific logging guide was found: this card searched the repository's README [8] and its 30 most recent issues (the full set returned by the GitHub issues API at the time of the 2026-08-09 read [10]) for maintainer guidance on healthy ranges for $H_t$ or $P_t$ and found none.
- Published reference runs: the paper's own Table 1 and Table 2 numbers are the only published reference curves — e.g. on Qwen2.5-3B-Instruct, GRPO scored 50.4 average across 10 math/knowledge tasks against ARPO's 52.8 [1]; on Qwen3-14B deep-search with 1K training samples, GRPO scored 36.9 average GAIA against ARPO's 43.7 [1]. No public training-curve logs (e.g. a wandb report) are linked from the repository's README [8].
- Degeneracies and defaults: setting the KL coefficient away from the paper's 0 default reintroduces a reference-model cost with no ARPO-specific guidance on the right value, since the paper's own runs all used 0 [1]. A GitHub issue reports that `tool_agent.py`'s branching code uses `random.random() > self.branch_probability`, which reads as branching decided by chance rather than by the entropy formula; the author replied that this file is deprecated logic and the real entropy-based branching lives in `vllm_rollout_with_tools.py`, so a reader inspecting the wrong file would misjudge what triggers a branch [11].
- Named successors: the same 50-title Semantic Scholar citation sample used above for adoption was checked for a paper naming itself as a fix to a documented ARPO bias or naming ARPO in its own title; none of the 50 titles do either, though several (e.g. "Tool-Aware Optimization with Entropy Guidance for Efficient Agentic Reinforcement Learning", "AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning") extend entropy-based rollout ideas without naming ARPO as the method being fixed, so this card does not treat them as confirmed successors [3].
- Known failure modes: the paper's own text does not contain a dedicated Limitations section (checked Sections 1-6 and the Conclusion of the arXiv HTML full text, 2026-08-09) [1]. From the repository's closed issues: one user's initial single-run evaluation of the released Qwen3-8B-AEPO-DeepSearch checkpoint scored HLE 8.8% and GAIA 24.3%, against the paper's reported 11% and 45.6% - roughly half on GAIA, about 80% of the paper's value on HLE [12]. The author replied that the discrepancy is most likely caused by a problem with the search process and asked for three independent evaluation rounds to help debug [12]. The user's three-round follow-up covered HLE only, returning accuracy values of 0.076, 0.088, and 0.086 - the middle value an exact match to the original single-run 8.8% figure, the other two below it - with no corresponding three-round GAIA numbers posted in the thread; no confirmed resolution is recorded [12]. A separate reply from the author states the RL-phase tool-call limit is 5, a value not given in the paper text checked here [9].
- What the gain is - and is not: the paper's own comparisons are against other trajectory-level RL algorithms (GRPO, DAPO, REINFORCE++) trained on the same cold-start checkpoint and datasets, showing gains attributed specifically to step-level tool-use exploration rather than to a stronger base model or more training data — the deep-search results use only 1K training samples, which the paper frames as evidence of sample efficiency rather than of a fundamentally higher performance ceiling [1].

## Sources

[1] Dong et al., "Agentic Reinforced Policy Optimization", 2025. https://arxiv.org/abs/2507.19849 — defines ARPO: entropy signal, adaptive rollout, advantage attribution, GPG theorem, experimental setup, results tables, hyperparameters. Fetched 2026-08-09 (arXiv HTML full text at https://arxiv.org/html/2507.19849, and the abstract page).

[2] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 — defines GRPO, ARPO's parent method. Fetched 2026-08-09 (abstract page; GRPO's own objective as restated and cited inside [1]).

[3] Semantic Scholar Graph API, citations for arXiv:2507.19849. https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.19849/citations?fields=title,year&limit=50 — first 50 citing-paper titles and years, used to check for landmark adopters and named successors. A live, unpinned API endpoint; read 2026-08-09.

[4] Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", 2025. https://arxiv.org/abs/2503.14476 — DAPO, the nearest online neighbor compared as a baseline in [1]; reports 50 points on AIME 2024 with a Qwen2.5-32B base. Fetched 2026-08-09 (abstract page).

[5] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 — DPO, the nearest offline alternative. Fetched 2026-08-09 (abstract page).

[6] trl GRPOTrainer documentation. https://huggingface.co/docs/trl/main/en/grpo_trainer.html — searched for the string "arpo" (zero matches); also the source for trl's published GRPO logging guidance. An unpinned `main`-branch build; read 2026-08-09.

[7] verl GRPO algorithm documentation. https://verl.readthedocs.io/en/latest/algo/grpo.html — searched for the string "arpo" (zero matches). An unpinned `latest` build; read 2026-08-09.

[8] RUC-NLPIR/ARPO GitHub repository (renamed from dongguanting/ARPO). https://github.com/RUC-NLPIR/ARPO — README installation/training instructions, `ARPO/scripts/ARPO_7B_Reasoning_1node.sh` launch script, `ARPO/verl_arpo_entropy/` forked-verl implementation directory, MIT license. `main` branch pinned to commit `67b343dfffff75d24596056872a8a66312ed80b9` (dated 2026-07-13T07:06:27Z by GitHub's commits API), read 2026-08-09; every claim citing [8] on this card reflects the repository's contents at that commit, a moving `main` branch beyond it is not covered.

[9] dongguanting, reply on RUC-NLPIR/ARPO issue #32, "default value about the call_limit or max_turns", GitHub. https://github.com/RUC-NLPIR/ARPO/issues/32 — states the RL-phase tool-call limit is 5. Fetched 2026-08-09.

[10] GitHub REST API, issues list for RUC-NLPIR/ARPO. https://api.github.com/repos/RUC-NLPIR/ARPO/issues — the 30 most recent issues (all states) returned by the endpoint at read time, searched for maintainer guidance on entropy-signal monitoring ranges (none found). A live, unpinned API endpoint; read 2026-08-09.

[11] dongguanting, reply on RUC-NLPIR/ARPO issue #30, "Discrepancy: Random branching used instead of entropy-based Adaptive Beaming as described in paper", GitHub. https://github.com/RUC-NLPIR/ARPO/issues/30 — clarifies `tool_agent.py`'s random branching is deprecated logic, not the entropy-based mechanism. Fetched 2026-08-09.

[12] zjd2001, dongguanting, and Phoenix-Leo, RUC-NLPIR/ARPO issue #47, "The evaluation did not achieve the desired results as described in the original text", GitHub. https://github.com/RUC-NLPIR/ARPO/issues/47 — reports a reproducibility gap on HLE/GAIA with the released checkpoint, and a three-round HLE-only follow-up evaluation. Fetched 2026-08-09.
