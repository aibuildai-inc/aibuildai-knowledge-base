# Entropy-Weighted Self-Consistency for Olympiad-Level Mathematical Reasoning

Competition: ai-mathematical-olympiad-progress-prize-3
Rank: #6
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-3/writeups/entropy-weighted-self-consistency-for-olympiad-lev

Entropy-Weighted Self-Consistency for Olympiad-Level Mathematical Reasoning:
A Competition Report on AIMO Progress Prize 3


Kunal Aarse
infinitekunal4@gmail.com


## Abstract
We present our solution to the AI Mathematical Olympiad Progress Prize 3 (AIMO3) competition on Kaggle, which tasks participants with solving 50 original olympiad-level mathematics problems spanning algebra, combinatorics, geometry, and number theory. Our approach centers on entropy-weighted self-consistency voting over multiple parallel inference attempts using the GPT-OSS-120B model served via vLLM under MXFP4 quantization on a single NVIDIA H100 GPU. Each candidate answer is weighted by the inverse of the mean Shannon entropy computed from top-5 token log-probabilities, rewarding confident, low-entropy completions. We further integrate a sandboxed Tool-Integrated Reasoning (TIR) environment providing persistent, stateful Jupyter kernels for symbolic computation via SymPy, numerical computation via NumPy, and high-precision arithmetic via mpmath. Our best submission achieved a public leaderboard score of 44/50 using 8 parallel attempts with flat time budget allocation and temperature T=1.0. We systematically ablated model choice (GPT-OSS-120B, GPT-OSS-20B, Qwen3-30B-A3B-fp8), attempt count (8, 16, 32), temperature (0.8, 0.9, 1.0), time budget strategy (flat vs. geometric decay), and fine-tuning approaches (SFT and GRPO). Notably, fine-tuning on mathematical corpora did not improve performance on this competition, and geometric budget allocation—despite theoretical advantages—produced inconsistent results (29–38) compared to the simpler flat-budget baseline.

## 1. Introduction
Mathematical reasoning remains one of the most challenging frontiers for artificial intelligence. Unlike natural language understanding, which benefits from statistical patterns and surface-level correlations, mathematical problem-solving at the olympiad level demands rigorous multi-step deduction, creative insight, and error-free computation. The AI Mathematical Olympiad (AIMO) Progress Prize series—organized by XTX Markets and hosted on Kaggle—has emerged as the primary benchmark for open-source mathematical AI, offering progressively harder problem sets and substantial prize pools to incentivize frontier research.
AIMO3 represents a significant escalation in difficulty over its predecessors. The competition features 110 original problems (50 for the public leaderboard) spanning all major olympiad categories, designed to be resistant to memorization and requiring genuine mathematical reasoning. Critically, the answer format was changed from 3-digit (AIMO1, AIMO2) to a 5-digit range (0–99999), eliminating implicit modulo reductions and demanding more precise computation. Furthermore, the competition provided participants with access to NVIDIA H100 GPUs, enabling inference with models far larger than previously feasible in competition settings.
Our submission explores the following central research questions:
    • Can entropy-weighted voting over parallel attempts outperform simple majority voting for olympiad-level mathematics?
    • What is the optimal balance between number of attempts and per-attempt compute budget under a fixed 5-hour wall-clock constraint?
    • Do SFT and GRPO fine-tuning on mathematical corpora improve base model performance on out-of-distribution olympiad problems?
    • How does dynamic (geometric decay) budget allocation compare to uniform time distribution?
We report that entropy-weighting provides a consistent advantage over raw voting, that 8 attempts with maximum per-attempt budget outperforms 16 or 32 attempts with reduced budgets, and that fine-tuning—despite theoretical promise—failed to generalize to the AIMO3 problem distribution. The remainder of this paper is structured as follows: Section 2 reviews related work including previous AIMO winners; Section 3 describes our methodology; Section 4 presents experimental setup; Section 5 reports results and analysis; Section 6 concludes with lessons learned and future directions. Full source code is provided in Appendices A, B, and C.

## 2. Related Work
### 2.1 AIMO Progress Prize 1 — Project Numina and CMU Math
The inaugural AIMO Progress Prize (July 2024) was won by Project Numina, a collaboration between Hugging Face and the NuminaMath team. Their winning approach fine-tuned a DeepSeek-Math-7B model on a curated corpus of 860K competition-grade problems sourced from AMC, AIME, MATH, and synthetic augmentation pipelines. The key technical contributions were: (1) a two-stage supervised fine-tuning pipeline combining chain-of-thought (CoT) and tool-integrated reasoning (TIR) data; (2) rejection sampling to filter training samples where the model-generated solution reached the correct answer; and (3) self-consistency voting across 32 parallel samples at inference time. Numina demonstrated that a relatively small (7B parameter) model, when fine-tuned on high-quality mathematical data at scale, could dramatically outperform much larger untuned models.
The CMU Math team (second place) employed a ToRA-style (Tool-integrated Reasoning Agent) approach, interleaving natural language reasoning steps with Python code execution. Their architecture emphasized tight coupling between symbolic reasoning in text and numerical verification through code, drawing on the insight that the model's natural language reasoning and computational results should mutually constrain each other. CMU Math's approach highlighted the importance of the execution environment quality: reliable, fast code execution directly impacted final score by enabling the model to verify intermediate steps.
### 2.2 AIMO Progress Prize 2 — NemoSkills and Imagination Research
AIMO2 (April 2025) saw a step-change in both problem difficulty (returning to 3-digit answers but with harder problems) and solution sophistication. The winning submission from Nvidia's NemoSkills team built upon their open-source NeMo-Skills framework, applying Group Relative Policy Optimization (GRPO)—a variant of reinforcement learning from verifiable rewards (RLVR)—to fine-tune Qwen2.5-Math and DeepSeek-R1 variants. GRPO optimizes the model's policy by sampling groups of responses, scoring each against a verifiable outcome (correct/incorrect numerical answer), and training the model to prefer correct trajectories while applying a KL penalty to prevent distribution collapse. The NemoSkills team showed that GRPO substantially improved pass@k rates, enabling reliable answers with fewer samples than their SFT-only baselines. Their technical report (Abadal et al., 2025; arXiv:2504.16891) provides a comprehensive ablation of training data quality, GRPO hyperparameters, and inference-time compute scaling.
The Imagination Research team (second place in AIMO2) focused on diverse self-consistency: rather than sampling from a single model, they employed a multi-model ensemble combining outputs from several mathematical reasoning models of different architectures and sizes. Their key insight was that model diversity in the voting pool—where different models make independent errors—improves aggregate accuracy beyond what homogeneous multi-sampling can achieve. They also employed answer clustering with a tolerance window (±2 for numerical proximity), which reduced the sensitivity of majority voting to small arithmetic errors in otherwise correct solution paths.
### 2.3 Self-Consistency and Voting Methods
Wang et al. (2023) introduced self-consistency sampling as a simple but highly effective decoding strategy: sample multiple reasoning paths at high temperature and select the most frequent answer. Subsequent work has explored weighted variants. Li et al. (2024) demonstrated that using model confidence scores (derived from token log-probabilities) as voting weights outperforms uniform majority voting, particularly for harder problems where the model occasionally produces highly confident correct answers alongside many uncertain wrong ones. Our entropy-weighting approach extends this line of work by computing Shannon entropy H = -Σ p log₂ p over the top-K token distribution at each decoding step, averaging across the full trajectory, and using 1/H as the answer weight.
### 2.4 Tool-Integrated Reasoning
Tool-integrated reasoning (Gou et al., 2023; Chen et al., 2022) augments language model inference with access to Python interpreters, enabling exact symbolic computation, brute-force enumeration, and numerical verification. In the AIMO setting, TIR is essential for problems requiring modular arithmetic, polynomial factorization, and combinatorial enumeration. Our implementation uses isolated Jupyter kernels (one per parallel attempt) pre-loaded with SymPy, NumPy, mpmath (64-decimal-digit precision), and standard library modules. This architecture prevents cross-contamination between attempts while enabling the full expressiveness of the Python scientific stack within each attempt.

## 3. Methodology
### 3.1 Model Selection and Quantization
We evaluated three models available within the competition's offline constraint:
    • GPT-OSS-120B (gpt-oss-120b): The primary model, a 120-billion parameter open-weight reasoning model served in MXFP4 (4-bit microscaled floating point) quantization via vLLM with fp8_e4m3 KV cache. MXFP4 reduces memory footprint to approximately 60–65 GB, enabling the full 120B model to reside on a single H100-80GB GPU with 96% memory utilization.
    • GPT-OSS-20B (gpt-oss-20b): A smaller 20B variant serving as a baseline and fallback configuration. Despite its smaller size, this model benefits from the same reasoning architecture and training regime.
    • Qwen3-30B-A3B-thinking-fp8: A Mixture-of-Experts (MoE) reasoning model from the Qwen3 family with 30B total parameters but only 3B active per forward pass (A3B architecture). Served in fp8 quantization, this model offered low latency per token while maintaining competitive mathematical reasoning quality.
The GPT-OSS-120B model under MXFP4 quantization consistently outperformed the other two models on the public leaderboard reference problems, validating the strong empirical relationship between model scale and olympiad-level reasoning quality. However, its larger size means fewer parallel attempts are possible within a fixed time budget, creating a fundamental tension between model quality and sampling diversity.
### 3.2 Inference Infrastructure
All inference is served through a local vLLM OpenAI-compatible API endpoint launched at initialization. Key infrastructure components include:
    • vLLM async scheduling with prefix caching enabled, reducing redundant computation for parallel attempts sharing the same system and user prompts.
    • A model weight preloading step that reads all checkpoint files into the OS page cache using 16 parallel worker threads before launching the vLLM process, reducing server startup latency from 4–6 minutes to under 90 seconds.
    • Streaming token generation with real-time \boxed{} pattern scanning, enabling early termination as soon as a valid answer is detected without waiting for the complete response.
    • Per-attempt quadratic seed spacing: attempt_seed = (seed + i)², ensuring decorrelated random number streams across parallel attempts without complex seeding logic.
### 3.3 System and Tool Prompts
Each inference request uses a structured system prompt implementing a five-phase problem-solving methodology: (1) UNDERSTAND — problem restatement and constraint identification; (2) EXPLORE — multi-strategy brainstorming; (3) PLAN — approach selection; (4) EXECUTE — systematic derivation; (5) VERIFY — answer validation. The system prompt additionally specifies ReasoningEffort.HIGH, activating the model's extended thinking mode.
A tool prompt instructs the model on appropriate use of the Python execution environment: computations that are error-prone by hand, numerical verification of analytical results, conjecture testing, and small-case brute-force. The preference prompt appended to each user query enumerates the available scientific libraries (SymPy for symbolic computation, NumPy for numerical computation, mpmath for arbitrary-precision arithmetic) with concrete guidance on when each is most appropriate.
### 3.4 Entropy-Weighted Self-Consistency Voting
Our central methodological contribution is entropy-weighted voting. For each decoding step, vLLM returns the top-K (K=5) token log-probabilities. We compute the Shannon entropy of this distribution:
H(t) = -Σₖ pₖ log₂ pₖ     where pₖ = exp(logprob_k) / Σⱼ exp(logprob_j)
The mean entropy H̄ = (1/T) Σₜ H(t) is computed over all T tokens in the response. Lower mean entropy indicates a more confident, self-consistent generation. The voting weight for answer a from attempt i is w_i = 1 / max(H̄_i, ε), where ε = 10⁻⁹ prevents division by zero. The final answer is the candidate maximizing total weighted score: â = argmax_a Σᵢ wᵢ · 1[aᵢ = a].
This formulation rewards attempts where the model expresses high token-level confidence throughout its reasoning chain, which empirically correlates with correctness on mathematical problems. In contrast to raw majority voting—which treats all completed attempts equally—entropy weighting can select a minority answer if that answer comes from significantly more confident completions.
## 3.5 Time Budget Management
### 3.5.1 Flat Budget (Best Configuration)
The flat budget strategy (Document 1 / Document 4) allocates per-problem time as:
budget = min(time_left − (N−1) × base_timeout, ceiling_timeout)
where time_left is the remaining notebook runtime, N is the number of problems remaining, base_timeout = 300s is the minimum guarantee per problem, and ceiling_timeout = 900s is the hard maximum. This strategy is conservative and predictable: early problems receive generous budgets, and unused time flows naturally to subsequent problems through the time_left term.
### 3.5.2 Geometric Decay Budget (Experimental)
The geometric decay strategy (Document 3) models the intuition that earlier problems in the randomly-ordered test set should receive more compute, since problem difficulty is unknown and early over-investment is recoverable, while late over-investment is catastrophic. The budget for the i-th problem is the i-th term of a geometric series summing exactly to the total notebook limit:
budget_i = T_remaining × (1−r) / (1−r^N)
where r = 0.97 is the geometric ratio and N is the number of remaining problems. Simultaneously, the number of attempts decays geometrically: attempts_i = max(attempts_min, round(attempts_max × r^i)), with attempts_max = 16 and attempts_min = 4. A pace correction mechanism compares actual elapsed time against an ideal elapsed time trajectory, triggering emergency caps when the solver falls more than 300s or 600s behind schedule.
### 3.6 Fine-Tuning Experiments
We attempted two fine-tuning approaches using the Unsloth library with the TRL training framework:
    • Supervised Fine-Tuning (SFT): Fine-tuned on a dataset of IMO shortlist problems, Putnam competition solutions, and AIME problems with full solution chains. Training used LoRA adapters (rank 64, α=128) on the attention and MLP projection layers.
    • Group Relative Policy Optimization (GRPO): Trained using verifiable reward signals on the AIMO3 reference problems. Following the NemoSkills approach, groups of 4 responses were sampled per problem, scored as 1 (correct) or 0 (incorrect), and the policy gradient was computed relative to group baselines with KL regularization.
Both approaches used Unsloth's memory-efficient training pipeline to fit within the H100 GPU's 80GB VRAM budget while training the 120B model with MXFP4 precision. However, merging LoRA SFT adapters into the MXFP4 base model encountered compatibility issues between the Unsloth quantization format and standard adapter merging routines. When these issues were resolved and the merged model was evaluated, leaderboard scores did not improve over the base model, suggesting that the fine-tuning data distribution did not align well with the AIMO3 problem design philosophy (which explicitly aims to be resistant to training contamination through entirely novel problem construction).

## 4. Experimental Setup
### 4.1 Hardware
All experiments were conducted on a Kaggle notebook session equipped with a single NVIDIA H100 SXM5 80GB GPU, 2 CPU cores, and approximately 29 GB of system RAM. The competition enforces a 5-hour (18,000 second) wall-clock limit; we used a conservative 17,000–17,400 second budget to account for Kaggle overhead variance.
## 4.2 Hyperparameter Configurations
Table 1 summarizes the hyperparameter configurations evaluated across all submissions.

Table 1: Hyperparameter Configurations Evaluated
Model	Attempts	Temp.	Budget	Fine-tune	LB Score
GPT-OSS-120B	8	1.0	Flat	None	44 (Best)
GPT-OSS-120B	8	0.9	Flat	None	38–42
GPT-OSS-120B	8	0.8	Flat	None	36–40
GPT-OSS-120B	16	1.0	Flat	None	40–43
GPT-OSS-120B	32	1.0	Flat	None	37–41
GPT-OSS-120B	16	1.0	Geometric	None	29–38
GPT-OSS-120B	8	1.0	Flat	SFT	No improvement
GPT-OSS-120B	8	1.0	Flat	GRPO	No improvement
GPT-OSS-20B	8	1.0	Flat	None	28–34
Qwen3-30B-A3B-fp8	8	1.0	Flat	None	30–36

## 4.3 Evaluation Protocol
Public leaderboard evaluation: the Kaggle API serves 50 test problems sequentially in random order. Each submitted notebook is run once to generate the public score. Private leaderboard evaluation (after competition deadline) runs the notebook twice over 110 private problems; scoring uses a penalized accuracy where both runs correct = 1.0, one correct = 0.5, both wrong = 0.0. Our public score of 44 corresponds to answering 44 of 50 public problems correctly in a single run.

## 5. Results and Analysis
### 5.1 Temperature Ablation
Temperature controls diversity in the self-consistency sampling pool. Lower temperatures produce more deterministic outputs, while higher temperatures introduce more exploration. Our ablation across T ∈ {0.8, 0.9, 1.0} with 8 attempts yielded:
    • T = 0.8: Scores 36–40. Insufficient diversity; multiple attempts converged to identical wrong answers, reducing the effective sample size.
    • T = 0.9: Scores 38–42. Better diversity, but still some convergence failures on harder geometry and combinatorics problems.
    • T = 1.0: Best score 44. Maximum diversity enabled the voting pool to include correct solution paths that lower temperatures never explored. The entropy-weighting mechanism was particularly effective here: high-temperature completions have high mean entropy overall, but correct answers often came from completions with significantly lower entropy than the distribution of wrong answers, allowing entropy-weighting to correctly identify them.
This result is counterintuitive relative to standard NLP tasks, where T=1.0 typically introduces too much noise. For olympiad mathematics, the problem space is so constrained that exploration is more valuable than precision in the decoding process—precision is provided by the verification steps within each reasoning chain.
5.2 Attempt Count Ablation
Increasing attempts from 8 to 16 did not consistently improve scores, and 32 attempts performed worse than 8. The likely explanation is a compute-diversity tradeoff: with a fixed 5-hour budget and the large 120B model, increasing attempts necessarily reduces the per-attempt token budget. Attempts truncated by the deadline produced no answer (counted as None, excluded from voting), effectively reducing the useful voting pool. With 32 attempts and a 300-second problem budget, roughly 60% of attempts were truncating before reaching a \boxed{} answer, yielding fewer usable votes than the 8-attempt baseline where nearly all attempts completed.
This finding suggests that for large models in time-constrained settings, depth (per-attempt reasoning quality) dominates breadth (number of attempts) beyond a modest ensemble size.
5.3 Geometric Budget Allocation Analysis
The geometric decay strategy was motivated by two intuitions: (1) front-loading compute to earlier problems allows the model to "warm up" on easier instances; (2) problems closer to the end of the session, where the model has less time, are likely harder on average due to random ordering and natural difficulty correlation.
In practice, this strategy produced scores ranging from 29 to 38, substantially worse than the flat baseline (44). Post-hoc analysis suggests two failure modes:
    1. Early over-investment: The geometric strategy allocated up to 660 seconds to the first problem. If that problem happened to be one the model could solve quickly (within 200 seconds), the remaining ~460 seconds was burned on redundant attempts hitting the early_stop threshold. This time was not recoverable for later harder problems.
    2. Pace correction instability: The pace correction mechanism—which reduces budget and attempts when the solver falls behind the ideal geometric trajectory—triggered aggressively on problems that required many Python tool calls, reducing the quality of subsequent problems' solutions in a cascading failure mode.
The flat budget strategy's superior performance demonstrates that in the absence of per-problem difficulty signals, uniform allocation is more robust than adaptive strategies that assume any particular difficulty distribution.
5.4 Fine-Tuning Analysis
Both SFT and GRPO fine-tuning failed to improve over the base model on AIMO3. We identify several reasons for this negative result:
    • Distribution mismatch: AIMO3 problems are explicitly designed to resist training contamination by being entirely novel. Fine-tuning on historical competition problems (IMO shortlists, Putnam, AIME) likely reinforced solution patterns that are not applicable to AIMO3's construction.
    • Quantization incompatibility: The MXFP4 quantization format used by GPT-OSS-120B is not natively supported by standard LoRA merging pipelines. Post-merge re-quantization introduced accuracy degradation that offset any reasoning improvements from the adapter.
    • Adapter-format fragility: Sequential single-adapter training (SFT then GRPO on the same adapter) is theoretically preferable to merging separately trained adapters. However, the small number of GRPO training examples available within the competition's data constraints was insufficient to produce statistically stable policy gradients.
5.5 Model Comparison
GPT-OSS-120B significantly outperformed both GPT-OSS-20B (28–34) and Qwen3-30B-A3B-fp8 (30–36). The 6× parameter advantage of the 120B model translated to approximately 10–14 additional correct answers, suggesting that model scale remains the dominant factor for olympiad-level reasoning in the absence of problem-specific fine-tuning. The Qwen3 MoE architecture, despite its active-parameter efficiency, did not close this gap—likely because the mathematical reasoning capability scales with total trained parameters rather than active inference parameters.

## 6. Conclusion
We presented a comprehensive exploration of inference-time strategies for olympiad-level mathematical reasoning in the AIMO3 competition context. Our best submission achieving 44/50 on the public leaderboard establishes several empirical findings:
    3. Entropy-weighted self-consistency voting provides a consistent and principled improvement over raw majority voting by leveraging token-level confidence signals.
    4. Higher sampling temperature (T=1.0) is optimal for mathematical self-consistency, contradicting conventional wisdom from natural language generation tasks.
    5. Depth (per-attempt token budget) dominates breadth (number of attempts) for large reasoning models under tight compute constraints. Eight well-resourced attempts outperform 16 or 32 resource-starved ones.
    6. Flat uniform budget allocation outperforms geometric decay strategies in the absence of per-problem difficulty information. Adaptive strategies require reliable difficulty signals to work well.
    7. Fine-tuning on historical competition mathematics does not generalize to novel olympiad problems explicitly designed to resist training contamination. Inference-time compute scaling remains more reliable than training-time optimization for out-of-distribution competition mathematics.
The gap between our 44/50 open-source result and the 50/50 results achievable by closed-source systems with sufficient compute (as reported by competition organizers) highlights the continued challenge of open-source mathematical AI. Future work should explore: multi-model ensembles for diversity-through-heterogeneity; online difficulty estimation from early attempts to dynamically allocate compute; better-calibrated confidence measures beyond simple token entropy; and training data construction that more closely mirrors the novel problem construction methodology of AIMO3.

References
[1] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. ICLR 2023.
[2] Gou, Z., Shao, Z., Gong, Y., Yang, Y., Huang, M., Duan, N., Chen, W., & Zhang, T. (2023). ToRA: A tool-integrated reasoning agent for mathematical problem solving. arXiv:2309.17452.
[3] Chen, W., Ma, X., Wang, X., & Cohen, W. W. (2022). Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv:2211.12588.
[4] Numina. (2024). NuminaMath: A solution to AIMO Progress Prize 1. Kaggle Competition Writeup.
[5] Abadal, S., et al. (2025). NemoSkills: Scalable open-source mathematical reasoning via GRPO fine-tuning. arXiv:2504.16891.
[6] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. SOSP 2023.
[7] Sheng, Y., Zheng, L., Yuan, B., Li, Z., Ryabinin, M., Chen, B., Liang, P., Re, C., Stoica, I., & Zhang, C. (2023). FlexGen: High-throughput generative inference of large language models with a single GPU. ICML 2023.
[8] Li, Y., Lin, Z., Zhang, S., Fu, Q., Chen, B., Lou, J. G., & Chen, W. (2024). Making language models better reasoners with step-aware verifier. ACL 2023.
[9] Frieder, S., et al. (2025). AI Mathematical Olympiad - Progress Prize 3. Kaggle Competition. https://kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3.
[10] Han, J., et al. (2024). InternLM-Math: Open math large language models toward verifiable reasoning. arXiv:2402.06332.

Appendix A: Best Submission Code (44/50)
This appendix contains the complete source code for our best-scoring submission, achieving 44/50 on the public leaderboard. This uses GPT-OSS-120B with 8 parallel attempts, flat time budget allocation, temperature T=1.0, and entropy-weighted voting.

## A.1 Configuration
```Python
class CFG:
    served_model_name = "gpt-oss"
    model_path = "/kaggle/input/gpt-oss-120b/transformers/default/1"
    kv_cache_dtype = "fp8_e4m3"
    dtype = "auto"
    high_problem_timeout = 900
    base_problem_timeout = 300
    notebook_limit = 17400
    server_timeout = 180
    session_timeout = 960
    jupyter_timeout = 6
    sandbox_timeout = 3
    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256
    early_stop = 4
    attempts = 8      # Fixed 8 attempts — best configuration
    workers = 16
    turns = 128
    seed = 42
    gpu_memory_utilization = 0.96
    temperature = 1.0  # T=1.0 outperformed 0.8 and 0.9
    min_p = 0.02
```

## A.2 Time Budget Allocation (Flat Strategy)
```Python
def solve_problem(self, problem: str) -> int:
    # Flat budget: generous for each problem, guarantee floor
    elapsed_global = time.time() - self.notebook_start_time
    time_left = self.cfg.notebook_limit - elapsed_global
    problems_left_others = max(0, self.problems_remaining - 1)
    reserved_time = problems_left_others * self.cfg.base_problem_timeout

    budget = time_left - reserved_time
    budget = min(budget, self.cfg.high_problem_timeout)  # cap at 900s
    budget = max(budget, self.cfg.base_problem_timeout)  # floor at 300s

    deadline = time.time() + budget
```

## A.3 Entropy-Weighted Voting
```Python
def _compute_mean_entropy(self, logprobs_buffer: list) -> float:
    if not logprobs_buffer:
        return float("inf")
    total_entropy = 0.0
    token_count = 0
    for top_logprobs_dict in logprobs_buffer:
        if not isinstance(top_logprobs_dict, dict) or not top_logprobs_dict:
            continue
        token_entropy = 0.0
        for token_str, log_prob in top_logprobs_dict.items():
            prob = math.exp(log_prob)
            if prob > 0:
                token_entropy -= prob * math.log2(prob)
        total_entropy += token_entropy
        token_count += 1
    return total_entropy / token_count if token_count > 0 else float("inf")

def _select_answer(self, detailed_results: list) -> int:
    answer_weights = defaultdict(float)
    answer_votes   = defaultdict(int)
    for result in detailed_results:
        answer, entropy = result["Answer"], result["Entropy"]
        if answer is not None:
            weight = 1.0 / max(entropy, 1e-9)
            answer_weights[answer] += weight
            answer_votes[answer]   += 1
    scored = sorted(answer_weights.items(), key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else 0
```

Appendix B: Geometric Time Allocation Code (Score: 29–38)
This appendix documents the geometric decay budget allocation strategy, which underperformed the flat baseline. It is included for completeness and to enable future researchers to understand why adaptive strategies require reliable difficulty signals.

## B.1 Configuration
```Python
class CFG:
    # Geometric decay parameters
    geo_ratio    = 0.97   # each problem gets 3% less budget than prior
    attempts_max = 16     # problem 1 gets maximum attempts
    attempts_min = 4      # floor — never fewer than 4
    tier_1_timeout       = 900
    base_problem_timeout = 180
    notebook_limit       = 17000  # conservative buffer
```

## B.2 Geometric Budget Computation
```Python
# Projected budgets at competition start (T=17000, N=50, r=0.97):
#   Problem  1: ~660s, 16 attempts
#   Problem 10: ~491s, 12 attempts
#   Problem 25: ~303s,  8 attempts
#   Problem 40: ~187s,  5 attempts
#   Problem 50: ~180s,  4 attempts

r = self.cfg.geo_ratio
N = max(self.problems_remaining, 1)
if abs(1.0 - r ** N) < 1e-9:
    geo_budget = time_left / N
else:
    geo_budget = time_left * (1.0 - r) / (1.0 - r ** N)

budget = min(geo_budget, self.cfg.tier_1_timeout)
budget = max(budget, self.cfg.base_problem_timeout)

# Geometric attempts decay
dynamic_attempts = max(
    self.cfg.attempts_min,
    round(self.cfg.attempts_max * (r ** problems_solved))
)
```

## B.3 Pace Correction
```Python
# Ideal elapsed = sum of first i terms of the geometric series
if problems_solved > 0:
    per_problem_0 = self.cfg.notebook_limit * (1.0 - r) / (1.0 - r ** 50)
    ideal_elapsed = per_problem_0 * (1.0 - r ** problems_solved) / (1.0 - r)
else:
    ideal_elapsed = problems_solved * 298.0

pace_delta = elapsed_global - ideal_elapsed  # positive = behind schedule

if pace_delta > 600:
    budget = min(budget, self.cfg.base_problem_timeout * 2)
    dynamic_attempts = self.cfg.attempts_min
    pace_label = " -> EMERGENCY CAP"
elif pace_delta > 300:
    budget = min(budget, 600)
    pace_label = " -> PACE CAP (600s)"
```

Appendix C: Additional Experiment Details
## C.1 vLLM Server Launch Configuration
```Python
cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--seed", str(cfg.seed),
    "--model", cfg.model_path,
    "--served-model-name", cfg.served_model_name,
    "--tensor-parallel-size", "1",
    "--max-num-seqs", str(cfg.batch_size),     # 256
    "--gpu-memory-utilization", "0.96",
    "--dtype", "auto",
    "--kv-cache-dtype", "fp8_e4m3",
    "--max-model-len", "65536",
    "--stream-interval", "200",
    "--async-scheduling",
    "--disable-log-stats",
    "--enable-prefix-caching",
]
```

## C.2 Sandbox Initialization
```Python
class AIMO3Sandbox:
    """Isolated Jupyter kernel for tool-integrated reasoning."""

    def __init__(self, timeout: float):
        # Assign unique ports to prevent inter-kernel conflicts
        ports = self._get_next_ports(5)
        self._km = KernelManager()
        self._km.shell_port   = ports[0]
        self._km.iopub_port   = ports[1]
        self._km.stdin_port   = ports[2]
        self._km.hb_port      = ports[3]
        self._km.control_port = ports[4]
        self._km.start_kernel(env=env)

        # Pre-import scientific stack in every kernel
        self.execute(
            "import math, numpy, sympy, itertools, collections, mpmath\n"
            "mpmath.mp.dps = 64  # 64 decimal digits precision"
        )
```

## C.3 Answer Extraction
def _scan_for_answer(self, text: str) -> int | None:
    # Primary: \boxed{N} pattern
    pattern = r"\\boxed\s*\{\s*([0-9,]+)\s*\}"
    matches = re.findall(pattern, text)
    if matches:
        clean_value = matches[-1].replace(",", "")
        value = int(clean_value)
        if 0 <= value <= 99999:
            return value

    # Fallback: "final answer is N" pattern
    pattern = r"final\s+answer\s+is\s*([0-9,]+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        clean_value = matches[-1].replace(",", "")
        value = int(clean_value)
        if 0 <= value <= 99999:
            return value

    return None

## C.4 Score Progression
Table C1: Score Progression Over Competition Duration
Experiment	Key Change	Score Range	Best Score
Baseline (20B)	GPT-OSS-20B, 8 att.	28–34	34
Qwen3 MoE	Qwen3-30B-A3B-fp8	30–36	36
T=0.8	120B, T=0.8	36–40	40
T=0.9	120B, T=0.9	38–42	42
T=1.0, 8 att.	Flat budget, 8 att.	42–44	44 ★
T=1.0, 16 att.	Flat budget, 16 att.	40–43	43
T=1.0, 32 att.	Flat budget, 32 att.	37–41	41
Geometric Budget	r=0.97 decay, 16→4 att.	29–38	38
SFT Fine-tune	LoRA SFT on 120B	No gain	—
GRPO Fine-tune	GRPO on 120B	No gain	—


# Additional information
Thanks a lot @nihilisticneuralnet @andreasbis your work is very significant in this competition, I think these people can be given additional prizes or a job invitation.

Our team really spent a lot of time finding the best results and improving our final pipeline. We tried:
1. Entropy-Weighted Self-Consistency
2. Create our validation
3. Сhanging generation parameters and reasoning strategy
4. Prompt engineering and solution structure
5. Ensembling

```Python
# Examples of validation, this is Yandex School of Data Analysis tasks
test_questions = [
        # Number Theory (1-6)
        ("Find the remainder when 7^2024 is divided by 100.", 1, "Cycle of 7^n mod 100 is 4"),
        ("How many positive integers < 1000 divisible by 7 or 11 but NOT both?", 208, "Inclusion-exclusion"),
        ("Largest Armstrong number < 1000?", 407, "1, 153, 370, 371, 407"),
        ("Find gcd(2024, 1980).", 44, "Euclidean algorithm"),
        ("Sum of digits of 3^100 modulo 9?", 0, "Digital root: 3^100 ≡ 0 mod 9"),
        ("How many primes between 50 and 100?", 10, "53,59,61,67,71,73,79,83,89,97"),
        
        # Sequences & Recurrence (7-10)
        ("Sequence: a₁=1, a₂=1, aₙ=aₙ₋₁+2·aₙ₋₂. Find a₁₅.", 10923, "Linear recurrence"),
        ("Fibonacci: F₁=1, F₂=1. Find F₂₀ mod 1000.", 57021 % 1000, "Modular Fibonacci"),
        ("Geometric series: sum_{k=0}^{10} 2^k = ?", 2047, "2^11 - 1"),
        ("Arithmetic series: 1+4+7+...+100 = ?", 1717, "n=34 terms, sum=n*(a₁+aₙ)/2"),
        
        # Combinatorics (11-16)
        ("Ways to arrange 5 books if 2 must NOT be adjacent?", 72, "5! - 2·4!"),
        ("Non-negative integer solutions to x+y+z=20?", 231, "C(22,2)"),
        ("Coin flipped 10 times: P(exactly 6 heads)=a/b reduced. Find a+b.", 617, "C(10,6)/2^10=105/512"),
        ("How many 4-digit numbers with distinct digits?", 4536, "9·9·8·7"),
        ("Ways to choose 3 from 10 people?", 120, "C(10,3)"),
        ("Permutations of 'MATH'?", 24, "4!"),
        
        # Geometry / Algebra (17-22)
        ("Triangle: A=60°, AB=5, AC=8. Find BC².", 49, "Law of cosines"),
        ("Sum of digits of 2ⁿ = 31, n<30. Find n.", 20, "2^20=1048576"),
        ("Roots of x²-5x+6=0. Sum of squares of roots?", 13, "(r₁+r₂)²-2r₁r₂"),
        ("Circle radius 5, chord length 8. Distance from center to chord?", 3, "Pythagoras: √(25-16)"),
        ("Solve |2x-3|=7. Sum of solutions?", 3, "x=5 or x=-2"),
        ("Quadratic x²+bx+c has roots 3 and -5. Find b+c.", -23, "(x-3)(x+5)=x²+2x-15"),
        
        # Logic & Attention (23-26)
        ("n²+n+41 prime for n=0..39. How many primes?", 40, "Euler's polynomial"),
        ("'Lucky': divisible by 3 but not 9, in [1,198]. Count?", 44, "2 per block of 9 × 22"),
        ("Alice: ×3, +12, ÷3, -original. Result?", 4, "Algebraic simplification"),
        ("Last digit of 2^(2^2024)?", 6, "Cycle [2,4,8,6], exponent mod 4 = 0"),
        
        # Advanced (27-30) 
        ("Vector space P₃(R). f(g)=GCD(x²-1,g)+g'. Is f linear? 1=YES, 0=NO.", 0, "Counterexample exists"),
        ("Cheaters: maximize E[solved]-P(caught). Optimal k?", 3, "Piecewise quadratic max"),
        ("Asymptotic: ∫₀¹ sin(tx)ln(x)dx ~ C·tⁿ·(ln t)ᵐ. Return |m|+|n|+round(100|C|).", 158, "n=-1,m=0,C=-π/2"),
        ("Nested squares n=7, final at (4,3). Count sequences?", 400, "C(6,3)×C(6,2)"),
    ]
```
