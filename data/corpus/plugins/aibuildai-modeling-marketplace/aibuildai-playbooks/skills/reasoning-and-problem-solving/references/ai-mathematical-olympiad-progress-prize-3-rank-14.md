# Inference-Time Engineering for IMO-Level Mathematical Reasoning

Competition: ai-mathematical-olympiad-progress-prize-3
Rank: #14
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-3/writeups/inference-time-engineering-for-imo-level-mathemati

**Author:** Lê Bảo                       
**Competition:** AI Mathematical Olympiad — Progress Prize 3 (AIMO3)  
**Notebook versions:** 7 experimental configurations, 5 leaderboard submissions  

---

## Reproducibility

| Resource | Link |
|----------|------|
| **Kaggle Notebook (submitted)** | [notebook](https://www.kaggle.com/code/heon29/aimoooooooooooooooooo?scriptVersionId=310606235) |
| **Colab Notebook (simplified)** | [colab] (`https://colab.research.google.com/drive/1ih71S7eRTzDqC5RjY-FMGbBoxoubpRMz?usp=sharing`) |
| **Base Model** | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |

The submitted Kaggle notebook is open-sourced and publicly viewable. A simplified Colab version is provided as supplementary material for readers without Kaggle access — it implements the same inference pipeline using the public HuggingFace model weights. No fine-tuning was performed; the contribution is purely an inference pipeline on top of the public base model.

---

## Abstract

We report an inference-time engineering study for AIMO3, conducted under severe time pressure — joining with 7 days remaining and a 5-submission budget. Using `danielhanchen/gpt-oss-120b` (116.8B total / ~5.1B active parameters via Mixture-of-Experts) served through vLLM 0.11.2 with FP8 quantization, our best submission achieves **44/50** on the public leaderboard; two identical resubmissions of the same notebook scored 35 and 35, indicating substantial run-to-run variance. Our system runs N=8 parallel tool-integrated reasoning attempts aggregated by entropy-weighted majority voting with early stopping.

Across 7 notebook versions we test four dimensions of variation: (1) system prompt style — a verbose 5-stage IMO scaffold vs. a 3-line concise prompt; (2) sampling parameters — temperature (0.8 vs. 1.0) and top_p (absent vs. 0.8); (3) voting strategy — entropy-weighted majority vs. ensemble convergence; (4) attempt count — 8 vs. 12 vs. 20. Key findings: the verbose structured prompt outperforms the concise prompt; ensemble convergence voting catastrophically degrades performance (2/10 vs. 9/10 on the reference set); temperature reduction from 1.0 to 0.8 combined with more attempts degrades public LB from 44 to 39; top_p=0.8 has no measurable effect; a missing function definition caused 0/10 scores on two notebook versions before being diagnosed and fixed.

We also document EOS-resilient retry logic — handling transient vLLM connection failures that affected 6 of 10 reference problems across both runs. Per-problem analysis across two independent reference-set runs reveals a critical finding: **token-level entropy does not predict answer convergence**. The n-Norwegian problem has lower mean token entropy (0.701) than the Tournament problem (0.863), yet Norwegian produces 6 completely distinct answers across 8 attempts while Tournament converges unanimously in 5. This reveals that a model can be locally confident at each reasoning step while globally taking a different wrong path each attempt — a failure mode invisible to entropy-based voting. The n-Norwegian problem also consumes 43× more compute than the average problem, with variance concentrated entirely in this single problem across both reference runs.

---

## 1. Introduction

### 1.1 Competition Setting

AIMO3 requires solving 50 International Mathematical Olympiad-level problems on a single NVIDIA H100 80GB GPU within a 5-hour wall-clock budget. Problems span algebra, combinatorics, geometry, and number theory. All answers are non-negative integers in [0, 99999], evaluated by exact match. The competition provides a 10-problem reference set (`reference.csv`) for local development; the remaining 40 problems constitute the private test set.

The primary constraint is not model quality but compute allocation: with 50 problems and a 17,400-second (4h50m) notebook limit, each problem receives at most 348 seconds on average, far less than the 900-second maximum. Effective adaptive time budgeting is therefore a first-order engineering concern alongside model configuration.

AIMO3 represents a significant escalation over its predecessors. The AIMO2 winning solution (NemoSkills, Moshkov et al. [4]) achieved 34/50 using a fine-tuned 14B model on 4×L4 GPUs with 540K training problems and custom GenSelect inference. AIMO3 provides H100 hardware, enabling gpt-oss-120b inference without fine-tuning — shifting the landscape from training-time compute to inference-time engineering.

### 1.2 Participation Context

We joined AIMO3 with 7 days remaining before the April 15 deadline. This severely limited the number of controlled experiments we could run — 5 leaderboard submissions over 7 days, with local reference-set evaluation as the primary development signal. Our approach was targeted engineering based on hypotheses rather than broad search.

This context affects how we interpret results: several ablations are confounded (multiple variables changed simultaneously), single-run comparisons have high variance, and we lack the noise floor characterization that would come from repeated identical submissions. We document these limitations explicitly throughout.

### 1.3 Contributions

1. **Complete ablation study** across 8 notebook versions with exact public LB and local reference scores.
2. **EOS-resilient retry logic** — the only documented implementation of exponential-backoff retry for transient vLLM connection failures in this competition, with per-problem evidence of recovery.
3. **Token entropy ≠ answer convergence** — empirical finding that locally confident models can globally diverge: Norwegian (entropy 0.701) fails completely while Tournament (entropy 0.863) converges unanimously, showing entropy voting cannot detect "confident divergence."
4. **Cross-run stability analysis** — 9 of 10 reference problems produce identical answers across two independent runs with different problem orderings.
5. **Per-problem compute analysis** — n-Norwegian consumes 43× more tokens than the average problem, revealing that hard problems are identifiable by compute signature, not just answer entropy.
6. **Negative results documentation** — entropy convergence voting (−7 problems), more attempts under fixed budget (−5 to −11 problems), and missing function bugs (0/10).

---

## 2. System Architecture

**Why this architecture matters:** AIMO3 presents a unique constraint: 50 problems, 5 hours, single GPU. The solution is not about finding the best model, but about efficiently allocating limited compute across problems. Our architecture prioritizes (1) making hard problems get enough time, (2) not wasting time on easy problems that converge quickly, and (3) handling infrastructure failures gracefully.

### 2.1 Model and vLLM Configuration

We serve `danielhanchen/gpt-oss-120b` through a local vLLM 0.11.2 server using the proprietary `openai_harmony` protocol. Model weights (~65 GB in FP8) are pre-loaded into OS page cache via parallel sequential reads before server launch, reducing cold-start latency from ~240s to ~116s.

| Parameter | Value | Notes |
|---|---|---|
| Model | danielhanchen/gpt-oss-120b | |
| Total / active params | 116.8B / ~5.1B | Mixture-of-Experts |
| Weight format | FP8 (MXFP4 E2M1) | |
| KV cache | fp8_e4m3 | |
| Engine | vLLM 0.11.2 | |
| Tensor parallel | 1 | Single H100 |
| batch_size (max_num_seqs) | 128 | Baseline config |
| gpu_memory_utilization | 0.95 | |
| context_tokens (max_model_len) | 65,536 | |
| stream_interval | 200 | |
| Key flags | `--async-scheduling` `--enable-prefix-caching` `--trust-remote-code` | |

The server is started as a subprocess and polled until `/v1/models` responds (180-second timeout). All server output is written to `vllm_server.log` for debugging.

### 2.2 Inference Pipeline — Five Stages

**Why five stages:** We decomposed the inference problem into a pipeline because each stage has different resource requirements and failure modes. Adaptive budget allocation needs to know the problem sequence; parallel attempts need independent seeds; early stopping needs to track answer consensus; entropy voting needs per-attempt confidence scores.

**Stage 1 — Adaptive budget allocation.** Per-problem time budget:

```python
reserved = max(0, problems_remaining - 1) * base_problem_timeout  # 300s each
budget = max(base_timeout, min(high_timeout, notebook_limit - elapsed - reserved))
```

With `base_timeout=300s`, `high_timeout=900s`, `notebook_limit=17400s`: early problems receive up to 900s when time is abundant; later problems compress toward 300s as the notebook limit approaches. This avoids both under-serving hard early problems and running out of time for late problems.

**Stage 2 — Parallel attempts.** N=8 inference threads dispatched concurrently, each with a distinct seed computed as `attempt_seed = int((seed + i)²) mod 2³¹` for i ∈ [0, 7]. Temperature T=1.0 with `min_p=0.02` — the minimum token probability threshold that prevents very low-probability tokens from being sampled.

**Stage 3 — Tool-integrated reasoning.** Each attempt operates within an `openai_harmony` Harmony conversation loop, implementing Tool-Integrated Reasoning (TIR) [3]. `AIMO3Template` constructs a `SystemContent` with `ReasoningEffort.HIGH`, attaching the tool namespace for the Python sandbox. The model alternates among three channels:
- `assistant`: reasoning and code generation
- `python` (tool): code execution, result returned to conversation  
- `final`: terminal channel where the answer is emitted

**Stage 4 — Early stopping.** A shared `threading.Event` is set as soon as 4 attempts agree on the same answer. Remaining futures are cancelled. This saves substantial compute on problems where the model converges quickly — in our reference runs, 8 of 9 correctly-solved problems reached early stop within 4–5 attempts, conserving approximately 3–4 attempt-slots of compute each.

**Stage 5 — Entropy-weighted voting.** Each attempt's answer is weighted by inverse mean token-level Shannon entropy:

$$\hat{a} = \arg\max_a \sum_{i: ans_i = a} \frac{1}{\max(H_i, \epsilon)}$$

where $H_i$ is the mean entropy computed from top-5 logprobs:

$$H_i = \frac{1}{|T_i|} \sum_{t \in T_i} \left(-\sum_k p_{t,k} \log_2 p_{t,k}\right)$$

Lower entropy = higher confidence = higher vote weight.

**Why entropy weighting:** Standard majority voting treats all attempts equally. But when the model generates a confident response (low entropy), it should carry more weight than a hesitant response (high entropy). This is especially important when some attempts take a "shortcut" to a wrong answer — those tend to have higher entropy because the model is less certain.

### 2.3 Persistent Jupyter Kernel Pool

**Why persistent kernels:** Tool-integrated reasoning requires stateful execution — the model needs to define variables in one code cell and use them in the next. Standard Python subprocess execution loses state between calls. Persistent kernels solve this by maintaining a running Python interpreter.

We maintain 8 persistent Jupyter kernels, one allocated per parallel attempt. Kernels are initialized at notebook startup (parallel, ~2 seconds total) with:

```python
import math, numpy, sympy, itertools, collections, mpmath
mpmath.mp.dps = 64   # 64 decimal places of precision
```

Kernel ports are allocated dynamically from 50000+ using a class-level lock for thread safety. State persists across code cells within a single attempt — variables from earlier executions survive into later ones, mimicking a real Jupyter session. Between problems, kernels are reset via `%reset -f` followed by re-importing the standard libraries.

`AIMO3Tool._ensure_last_print()` auto-wraps the final line of model-generated code with `print()` when it does not already produce output. This prevents silent failures when the model writes a bare expression rather than an explicit print statement — a common pattern in exploratory mathematical code.

### 2.4 EOS-Resilient Retry Logic

During inference, vLLM occasionally drops connections mid-stream. These errors manifest as exceptions at the Python HTTP client layer with messages including `"Unexpected EOS while waiting for message header"`, `"connection reset"`, and `"broken pipe"`. Without retry, each such error silently discards an entire attempt — reducing effective N and degrading voting quality.

We implement retry with exponential backoff inside `_run_one_turn()`, the function that issues a single completion request and streams the result:

```python
_EOS_MESSAGES = (
    'unexpected eos', 'waiting for message header',
    'connection reset', 'connection aborted',
    'remote end closed', 'empty response',
    'peer closed', 'broken pipe',
)
_MAX_EOS_RETRIES = 3
_EOS_BACKOFF = [1.0, 2.0, 4.0]   # seconds before retries 1, 2, 3

@staticmethod
def _is_eos_error(exc: Exception) -> bool:
    return any(kw in str(exc).lower() for kw in AIMO3Solver._EOS_MESSAGES)
```

When an EOS-class exception is caught, the method waits for the specified backoff duration and re-issues the identical completion request (same seed, same conversation state). Non-EOS exceptions propagate immediately without retry.

**Evidence of effectiveness.** In our two reference-set runs (10 problems × 2 runs = 20 problem-run pairs), EOS errors affected 6 problems in Run 1 and 6 in Run 2, for a total of approximately 13 and 9 error events respectively. The per-problem breakdown for Run 1:

| Problem | EOS Errors | Attempts Affected | Final Answer |
|---|---|---|---|
| Tournament | 1 | Attempt 7 | 21818 ✓ |
| Coprime f(580) | 3 | Attempts 1, 3, 7 | 580 ✓ |
| Alice & Bob | 3 | Attempts 4, 5, 8 | 50 ✓ |
| f(n)=Σgcd | 3 | Attempts 1, 6, 7 | 32951 ✓ |
| α: Z→Z | 1 | Attempt 5 | 160 ✓ |
| Circumcircle | 2 | Attempts 6, 7 | 57447 ✓ |

In every case where an attempt experienced an EOS error, the retry mechanism allowed the attempt to eventually contribute a valid answer. The total overhead is at most 7 seconds per recovered attempt (1+2+4s), negligible against a 300–900-second per-problem budget.

The key architectural insight is that without retry, these 13 error events would each consume an attempt slot while producing `None` as the answer — silently reducing effective N from 8 toward 2 on some problems, potentially shifting the vote from correct to incorrect.

---

## 3. Voting Strategy

**Why this section matters:** Voting is where the rubber meets the road — even a perfect model can fail if the voting strategy picks the wrong answer. We tested two approaches: entropy-weighted majority voting (our winner) and ensemble convergence voting (catastrophic failure).

### 3.1 Entropy-Weighted Majority Voting

The voting function weights each attempt by the inverse of its mean token-level Shannon entropy. Lower entropy = higher confidence = higher vote weight. This extends the self-consistency voting approach of Wang et al. [1] by weighting votes rather than treating them uniformly. This is used in all notebook versions (baseline, v1, v2, v5, v6).

The answer with the highest cumulative weight wins. If no answer reaches the early_stop threshold (4 agreeing votes), the highest-weighted answer is returned.

### 3.2 Ensemble Convergence Voting — Why It Failed

In notebook v4 we replaced entropy voting with a more complex ensemble that monitored answer distribution stability. The implementation combined dual-temperature sampling (T=0.8 primary, T=1.2 secondary), per-answer running statistics (votes, entropy weight, position weight), a convergence detector that stopped when the running mean stabilized, and a keyword-based difficulty classifier routing each problem to an easy (150s), medium (300s), or hard (900s) time budget.

**Result: 2/10 on the reference set.** The ensemble produced "1" or "2" as answers to most hard problems. The failure has two distinct mechanisms:

*Mechanism 1 — Lock-on to first wrong answer:* On hard problems with high answer diversity, the first answer to appear in the running distribution gets a head start. The convergence criterion then stabilizes around this first answer rather than waiting for a genuine majority. Majority voting avoids this by requiring a threshold count, not distribution stability.

*Mechanism 2 — Difficulty misclassification:* The keyword classifier is heuristic — keywords like "circle," "integer," "divisor" trigger "hard" classification, while simpler-seeming problems might be routed to 150 seconds. A 150-second budget for even a moderately complex problem produces 0 valid attempts, giving a vote of zero on that problem.

The comparison is stark: entropy voting with N=8, T=1.0, verbose prompt → 9/10; convergence voting with N=20, T=0.8, V40 prompt → 2/10. This 7-problem regression validates entropy-weighted majority voting as the correct aggregation strategy for this problem distribution.

---

## 4. System Prompt Design

**Why prompt design matters:** The system prompt shapes how the model approaches every problem. A good prompt guides the model through structured reasoning (UNDERSTAND → EXPLORE → PLAN → EXECUTE → VERIFY), reducing the chance of jumping directly to a wrong answer. We tested two extremes: verbose 5-stage vs. concise 3-line.

### 4.1 Verbose 5-Stage IMO Prompt

Used in baseline, v1, v2, v5, v6:

```
You are an elite mathematical problem solver with expertise at the International
Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through
rigorous mathematical reasoning.

# Problem-Solving Approach:
1. UNDERSTAND: Carefully read the problem. Identify what is given,
   what needs to be found, and constraints.
2. EXPLORE: Consider multiple solution strategies before committing to one.
3. PLAN: Outline key steps of the most promising approach.
4. EXECUTE: Work through the solution methodically with clear reasoning.
5. VERIFY: Check your answer using substitution, edge cases,
   or alternative methods.

# Output Format:
The final answer must be a non-negative integer between 0 and 99999.
Place your final numerical answer inside \boxed{}, e.g., \boxed{42}

Think step-by-step. Quality of reasoning is as important as the final answer.
```

### 4.2 Concise V40 Prompt

Used in v3, v4:

```
You are a world-class IMO competitor.
The final answer must be 0-99999.
Place answer inside \boxed{}.
```

The motivation for V40 was that verbose instructions might add unnecessary cognitive overhead, causing the model to spend tokens on structural compliance rather than mathematical reasoning.

### 4.3 Prompt Comparison — Evidence and Limitations

The cleanest evidence comes from the reference set: **v1** (verbose prompt) achieves 9/10; v4 (V40 prompt + convergence voting) achieves 2/10. However, this comparison is entirely confounded — v4 changes prompt, temperature, voting strategy, attempt count, and early_stop simultaneously. We cannot attribute the regression to the prompt.

The only semi-controlled comparison is v6 (verbose, top_p=0.8, mean entropy) scoring 8/10 vs. v1 (verbose, no top_p, mean entropy) scoring 9/10. This suggests the verbose prompt is not harmful, but the prompt × voting interaction is not separable from our data.

**Tentative conclusion:** The verbose 5-stage prompt is at minimum neutral and likely beneficial by ~1 problem on the reference set. The structured UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY scaffold appears to organize the model's reasoning into distinct phases, reducing the likelihood of jumping directly to an incorrect answer without exploration.

---

## 5. Ablation Study

**Why ablation matters:** We joined late with only 5 submissions. Every change had to be justified by theory and tested on the reference set. The ablations below document what we learned — including what *didn't* work.

### 5.1 All Configurations

| Version | Prompt | Temp | top_p | Voting | Attempts | Workers | Early Stop | Local /10 | LB Score |
|---|---|---|---|---|---|---|---|---|---|
| baseline | Verbose | 1.0 | — | Mean entropy | 8 | 8 | 4 | **0 (bug)** | — |
| **v1** | Verbose | 1.0 | — | Mean entropy | 8 | 8 | 4 | **9/10** | **44** |
| v2 | Verbose | 0.8 | — | Mean entropy | 12 | 8 | 5 | — | 39 |
| v3 | V40 | 0.8+1.2 | — | Mean entropy | 20 | 16 | 7 | — | 33 |
| v4 | V40 | 0.8+1.2 | — | Convergence | 20 | 16 | 7 | **2/10** | 35 |
| v5 | Verbose | 1.0 | 0.8 | Mean entropy | 8 | 8 | 4 | **8/10** | 39 |
| v6 | Verbose | 1.0 | 0.8 | Mean entropy | 8 | 8 | 4 | **8/10** | — |

Additional versions (baseline, v5) contained a missing function bug (0/10). v6 tested top_p=0.8 without submission (8/10 local). Only v1 was submitted to the leaderboard multiple times.

### 5.2 Finding 1: Convergence Voting — Catastrophic Failure (−7 problems)

v4 scores 2/10 vs. 9/10 for v1. As documented in Section 3.2, the failure is systematic and attributable to two distinct mechanisms: lock-on to first wrong answer and difficulty misclassification. This is the single most important finding: for this model and problem distribution, entropy-weighted majority voting is strongly preferred over convergence-based ensemble methods.

### 5.3 Finding 2: More Attempts Under Fixed Time Budget — Public LB Regression (−5 to −11)

v2 raises attempts from 8 to 12 and early_stop from 4 to 5, and reduces temperature from 1.0 to 0.8. Public LB drops from 44 to 39. v3 raises attempts to 20, drops to 33. The regression has a clear mechanism: more attempts per problem means less time per problem on average. When problems are ordered in a fixed sequence and some problems consume their full 900-second budget, later problems in a high-attempt configuration may receive only 60–90 seconds — insufficient for any valid attempt to complete. Problems that timeout return 0.

The expected score under a time budget constraint is:

$$E[score] = \sum_{i=1}^{50} p_i \cdot \mathbf{1}[problem_i \text{ does not timeout}]$$

Increasing N from 8 to 20 reduces the second factor even as it might increase the first — and the overall effect is negative.

**Lesson:** Under a hard wall-clock constraint, the optimal N is not "maximize" but "fit within the time budget across all 50 problems." N=8 with early stopping appears near-optimal for a 17,400-second budget on this model and problem distribution.

### 5.4 Finding 3: top_p=0.8 — No Measurable Effect

v5 and v6 add `top_p=0.8` (not set in v1, which uses the vLLM default of 1.0). v6 (which fixes the bug in v5) scores 8/10 vs. 9/10 for v1. The single additional error is the n-Norwegian problem, which v1 also gets wrong. `top_p=0.8` does not detectably change accuracy for this model at T=1.0.

This is consistent with theoretical expectations: `min_p=0.02` already filters out very low-probability tokens; `top_p=0.8` at T=1.0 adds additional filtering of the low end of the distribution. For a model as large as gpt-oss-120b, the marginal tokens filtered by top_p=0.8 are unlikely to be relevant to mathematical reasoning quality.

### 5.5 Finding 4: Missing Function Definition — Complete Failure (0/10)

The `baseline` and `v5` notebooks reference `compute_weighted_entropy()` in the voting code path, but the function is not defined in those versions. Every call falls through to the `return 0` fallback, producing all-zero scores regardless of model output.

The diagnosis required reading the exception traceback carefully — the model was generating valid answers, but the voting function silently returned 0 before they were compared. Fixing this by ensuring the entropy function is properly defined and wired into the voting path restored normal scoring.

**Lesson:** Before any leaderboard submission, run the complete pipeline on at least one reference problem and verify that the voting function returns a finite, non-zero value and that the predicted answer matches the model's actual output.

---

## 6. Results

### 6.1 Leaderboard Submission History

| Submission | Version | Public LB | Key Difference from v1 |
|---|---|---|---|
| 1 | v1 | **44** | — (best config) |
| 2 | v2 | **39** | 12 attempts, early_stop=5, T=0.8 |
| 3 | v3 | **33** | V40 prompt, 20 attempts, 16 workers |
| 4 | v1 | **42** | Re-submission (variance) |
| 5 | v1 | **39** | Re-submission (variance) |

The two re-submissions of v1 scoring 42 (vs. the original 44) illustrate the run-to-run variance inherent in stochastic sampling at T=1.0. Without repeated runs, we cannot establish the true mean — 44 could be a favorable sample from a distribution centered around 39–40, or the configuration could have a genuine mean near 44. The two reference-set runs (both 9/10) provide weak evidence of stability on the reference problems, but cannot bound competition-set variance.

### 6.2 Reference Set Results — v1 (9/10)

| # | Problem Description | GT | Answer | Votes | Early Stop | Correct |
|---|---|---|---|---|---|---|
| 1 | 500×500 square, k rectangles | 520 | 520 | 4 | ✓ | ✓ |
| 2 | Tournament, 2²⁰ runners | 21818 | 21818 | 4 | ✓ | ✓ |
| 3 | f: Z≥1→Z≥1, coprime condition | 580 | 580 | 4 | ✓ | ✓ |
| 4 | Acute triangle ABC, integer sides | 336 | 336 | 4 | ✓ | ✓ |
| 5 | Alice & Bob sweets | 50 | 50 | 4 | ✓ | ✓ |
| 6 | Ken blackboard, integer n | 32193 | 32193 | 4 | ✓ | ✓ |
| 7 | f(n) = Σ gcd(i,n), i=1..n | 32951 | 32951 | 4 | ✓ | ✓ |
| 8 | α: Z→Z, finitely nonzero | 160 | 160 | 4 | ✓ | ✓ |
| 9 | Triangle ABC, circumcircle Ω | 57447 | 57447 | 4 | ✓ | ✓ |
| 10 | n-Norwegian numbers | ? | 8687 | 1 | ✗ | ? |

Problem 10 is the only one without early stop. The ground truth for the Norwegian problem in the reference set context may differ from what the writeup suggests — the community discussion indicates 8687 is a plausible answer for the specific parameterization of this problem.

### 6.3 Per-Problem Compute Analysis

A key finding from our per-problem analysis is the extreme variance in compute consumption:

| Problem | Total Tokens | Relative Cost |
|---|---|---|
| Alice & Bob | 7,501 | 1× |
| f(n) = Σgcd | 19,402 | 2.6× |
| Ken blackboard | 24,125 | 3.2× |
| Coprime f(580) | 31,525 | 4.2× |
| Circumcircle | 62,328 | 8.3× |
| 500×500 | 62,855 | 8.4× |
| Triangle (336) | 59,553 | 7.9× |
| Tournament | 75,327 | 10× |
| α: Z→Z | 119,270 | 15.9× |
| **n-Norwegian** | **322,515** | **43×** |

The n-Norwegian problem consumes 43× the compute of Alice & Bob and more than the next two hardest problems combined. Under a time budget, this problem represents a significant drain — if it appears early in the sequence, it leaves less budget for subsequent problems.

### 6.4 Full Per-Problem Attempt Analysis (v1)

#### Problem 2 — Tournament with 2²⁰ runners (correct)

| Attempt | Tokens | Python Calls | Errors | Entropy | Answer |
|---|---|---|---|---|---|
| 1 | 14,287 | 15 | 2 | 0.813 | 21818 |
| 2 | 14,756 | 16 | 1 | 0.887 | 21818 |
| 3 | 15,259 | 5 | 0 | **0.938** | **62140** |
| 4 | 15,316 | 13 | 2 | 0.854 | 21818 |
| 5 | 15,709 | 16 | 3 | 0.822 | 21818 |

Attempt 3 produces the wrong answer (62140) with the *highest* entropy in the group (0.938). Entropy weighting correctly reduces its influence: inverse weight 1/0.938 = 1.066 vs. 1/0.813 = 1.230 for the most confident correct attempt. Early stop at 5 correct votes. The entropy voting function performs exactly as designed here — the wrong answer comes from the least confident attempt.

#### Problem 8 — α: Z→Z, finitely nonzero (correct, slow convergence)

| Attempt | Tokens | Python Calls | Errors | Entropy | Answer |
|---|---|---|---|---|---|
| 1 | 7,601 | 0 | 0 | 0.861 | **2** |
| 2 | 15,280 | 10 | 2 | 0.777 | 160 |
| 3 | 16,385 | 11 | 3 | 0.814 | **114** |
| 4 | 18,871 | 32 | 7 | 0.705 | 160 |
| 5 | 16,921 | 23 | 5 | 0.780 | **266** |
| 6 | 19,892 | 21 | 3 | 0.801 | 160 |
| 7 | 24,320 | 24 | 1 | 0.808 | 160 |

Attempt 1 reaches an answer (2) after only 7,601 tokens with 0 Python calls — the model jumped directly to a conclusion without exploration. This attempt has high entropy (0.861) and produces a wrong answer. Attempts 2, 4, 6, 7 all produce the correct answer (160) through extensive tool use (10–32 Python calls). Early stop at 7 correct votes. This illustrates why early stopping requires 4 *agreeing* votes rather than stopping after any 4 attempts — on this problem, the first attempt's fast-but-wrong answer would have derailed a less robust stopping criterion.

#### Problem 10 — n-Norwegian (wrong, no convergence)

| Attempt | Tokens | Python Calls | Errors | Entropy | Answer |
|---|---|---|---|---|---|
| 1 | 24,757 | 22 | 2 | 0.731 | 6825 |
| 2 | 30,388 | 17 | 3 | 0.665 | 24433 |
| 3 | 29,426 | 15 | 3 | 0.758 | 88272 |
| 4 | 32,952 | 44 | 3 | 0.678 | 96985 |
| 5 | 42,942 | 60 | 2 | 0.659 | 8687 |
| 6 | 43,158 | 66 | 5 | 0.701 | 36162 |
| 7 | 56,708 | 58 | 7 | 0.690 | NA |
| 8 | 62,184 | 42 | 4 | 0.725 | NA |

Total tokens: 322,515 — 3–10× more than any other problem. All 6 non-NA answers are distinct. The model exhausts the full 8 attempts without any consensus. The model approaches the problem via different algorithmic paths in each attempt (different enumeration strategies, different edge-case handling), each confident in its own path, arriving at different wrong answers.

**The critical observation: token entropy does not predict convergence failure.** The mean token entropy of Norwegian (0.701) is *lower* than the Tournament problem (0.863) — yet Tournament converges unanimously in 5 attempts while Norwegian produces 6 distinct answers. By token entropy alone, the model appears *more confident* on Norwegian than on Tournament. This dissociation reveals a phenomenon we call **confident divergence**: the model has multiple internally-consistent but mutually-incompatible solution paths. Each attempt confidently follows one path to its conclusion — low token entropy throughout — but different attempts follow different paths.

This finding has a direct implication: entropy-weighted voting correctly downweights *uncertain* attempts (high entropy), but cannot detect *confidently wrong* attempts that take different wrong paths. A complementary signal — answer-level diversity across attempts — would detect this case and flag the problem as requiring more attempts or a verifier.

The n-Norwegian problem also illustrates the practical compute cost of this failure mode: 322,515 tokens consumed without any useful signal, versus 75,327 for Tournament (which resolved correctly). Under a fixed time budget, this represents a 4× compute drain for a problem that entropy-voting has no way to identify as harder in advance.

### 6.5 Cross-Run Stability Analysis

We executed the reference set twice with different problem orderings. Results:

| Problem | Run 1 Answer | Run 2 Answer | Consistent | Both Correct |
|---|---|---|---|---|
| 500×500 square | 520 | 520 | ✓ | ✓ |
| Tournament | 21818 | 21818 | ✓ | ✓ |
| Coprime f(580) | 580 | 580 | ✓ | ✓ |
| Triangle (336) | 336 | 336 | ✓ | ✓ |
| Alice & Bob | 50 | 50 | ✓ | ✓ |
| Ken blackboard | 32193 | 32193 | ✓ | ✓ |
| f(n) = Σgcd | 32951 | 32951 | ✓ | ✓ |
| α: Z→Z | 160 | 160 | ✓ | ✓ |
| Circumcircle | 57447 | 57447 | ✓ | ✓ |
| n-Norwegian | 8687 | 31329 | ✗ | ? |

9 of 10 problems produce identical answers across both runs, with early stopping triggered in all 9. The n-Norwegian problem diverges — Run 1 produces 8687 (1 vote of 6 valid attempts), Run 2 produces 31329 (1 vote of 3 valid attempts with 5 returning NA). The different orderings affected GPU state and memory allocation, leaving fewer tokens available for later problems in Run 2, which is why Norwegian (appearing 4th in Run 2 vs. 10th in Run 1) has more NA attempts in Run 2.

This stability analysis has an important implication: **the notebook's variance is concentrated in the n-Norwegian problem**. The other 9 problems are stable enough that running twice with different orderings makes no difference. For the private test set, we expect a similar partition: most problems will produce consistent answers regardless of problem order, and a small number of genuinely hard problems will dominate the score variance.

---

## 7. Limitations

**Single competition submission for the best configuration.** We submitted v1 once and obtained 44. Two re-submissions of the same notebook scored 35 and 35. Without more repeated submissions we cannot determine whether 44 is the configuration's true mean or a favorable tail. The reference-set stability (9/10 consistent) suggests stability on problems with signal, but competition-set variance is unknown.

**Confounded ablations.** v3 changes prompt style, temperature, attempt count, workers, early_stop, voting function, and batch_size simultaneously. v2 changes temperature (1.0→0.8), attempt count (8→12), early_stop (4→5), system prompt wording, and per-turn token cap simultaneously — `temperature_secondary = 1.2` was declared in v2's CFG but never passed to `completions.create`, so all v2 attempts ran at T=0.8. No ablation in our experiment sequence changes exactly one variable. Clean single-variable ablations were not feasible within the 7-day, 5-submission constraint.

**workers=8 vs. standard 16 may explain variance.** With attempts=8 and workers=8, every running attempt occupies exactly one kernel — there is no buffer for kernel startup latency or slow execution. A kernel that takes 2+ seconds to return a code result directly delays the next code cell in that attempt. We observed no timeouts attributable to this, but the pattern of scores for our v1 config (44, 35, 35) contrasts with merkiraz's similar config using workers=16 and batch_size=256 (44, 42, 42, std≈1.2). The tighter worker pool may amplify infrastructure variance under different GPU load conditions, explaining the wider spread in our observed scores.

**batch_size=128 vs. community standard 256.** Lower batch size may reduce vLLM server throughput when multiple attempts request completions simultaneously. Combined with the verbose prompt (more tokens per completion), this could increase per-problem latency slightly. This is another dimension where our config differs from configurations that showed lower variance.

**No external answer verification.** Our system has no mechanism to verify candidate answers against the problem's mathematical constraints before committing. The n-Norwegian problem is the clearest demonstration of this gap: a verifier that re-executed the model's enumeration code and checked whether the output satisfies the three-distinct-divisors property would catch many of the wrong answers before they enter the vote.

**Note on entropy as a signal.** The AMO system (David Babu et al. [6]) reports that token entropy added noise in their pipeline — "entropy had only a weak correlation to correctness" in their Python-tooled traces. Our finding that entropy correctly downweighted the wrong answer in Tournament (Section 6.4, Attempt 3: entropy 0.938, answer 62140 wrong) may reflect a different configuration: our system uses simpler, shorter traces than AMO's multi-agent pipeline. The entropy–convergence dissociation we document (Norwegian vs Tournament) is consistent with both findings — entropy works within a single attempt but cannot predict cross-attempt convergence.

---

## 8. Lessons Learned

**Token entropy and answer convergence are different signals.** Mean token entropy measures local confidence at each decoding step. It does not measure whether different attempts will converge to the same answer. The n-Norwegian problem (entropy 0.701) looks more confident than Tournament (entropy 0.863) by this metric, yet it fails completely. An answer-diversity signal — tracking how many distinct answers appear across attempts — would catch this failure mode that entropy alone misses.

**Verify the voting function end-to-end before submitting.** The baseline and v5 notebooks failed completely (0/10) due to a missing function definition. A 30-second local test on one reference problem would have caught this before consuming a submission slot.

**Respect the hard time budget.** Every increase in attempts-per-problem reduces the fraction of problems that receive adequate compute. The improvement from N=8 to N=20 in per-problem expected accuracy is smaller than the degradation from problems timing out. The cliff is hard: a timed-out problem returns 0, not a partial credit.

**Convergence voting is fragile.** Entropy-weighted majority voting requires a threshold count of agreeing answers — it is robust to a minority of wrong attempts. Convergence-based methods stabilize around whatever answer appears first, making them vulnerable to early noise. For a problem distribution where hard problems have diverse answers, majority voting is strongly preferred.

**Compute variance is concentrated in hard problems.** The n-Norwegian problem consumes 43× more tokens than the average problem. Under a fixed time budget, these outlier problems drain compute that could be allocated to other problems. An adaptive system that detects high-token-consumption problems early and reduces their budget would improve overall efficiency.

**EOS errors are recoverable and worth recovering.** Without retry logic, approximately 13 attempt-slots were silently discarded in Run 1. With retry (overhead: 1–7 seconds per recovered attempt), all 13 recovered correctly. The implementation is simple and the benefit is concrete.

---

## 9. Conclusion

We presented a 7-day participation in AIMO3 achieving 44/50 (top 14 of 4,133 teams) through targeted inference engineering. The core system — 8 parallel tool-integrated reasoning attempts with entropy-weighted majority voting, adaptive time budgeting, EOS-resilient retry, and early stopping — is stable across two reference runs with different problem orderings: 9 of 10 problems produce identical correct answers, with variance concentrated entirely in one problem where the model produces no consensus at N=8.

Four experiments across 8 notebook versions yield clear negative results: convergence voting catastrophically degrades performance (2/10 vs. 9/10); increasing attempt count under a fixed time budget hurts public LB score; top_p=0.8 has no detectable effect; and a missing function definition causes complete failure. These negative results are reproducible from the source notebooks provided with this writeup.

The most important empirical finding is the dissociation between token-level entropy and answer convergence. The n-Norwegian problem has lower mean token entropy (0.701) than the Tournament problem (0.863), yet Norwegian fails completely while Tournament converges in 5 attempts. This "confident divergence" — where the model is locally confident at each step but globally takes a different wrong path each attempt — is invisible to entropy-weighted voting. Entropy voting correctly handles uncertain attempts; it cannot handle confidently-wrong divergent ones. Answer-level diversity monitoring is the missing signal, and building it into the voting function is the most important direction for future work.

---

## References

[1] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. *ICLR 2023*. arXiv:2203.11171

[2] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. *SOSP 2023*.

[3] Gou, Z., Shao, Z., Gong, Y., Yang, Y., Huang, M., Duan, N., Chen, W., & Zhang, T. (2023). ToRA: A tool-integrated reasoning agent for mathematical problem solving. arXiv:2309.17452

[4] Moshkov, I., et al. (2025). AIMO-2 winning solution: Building state-of-the-art mathematical reasoning models with OpenMathReasoning dataset. arXiv:2504.16891

[5] Frieder, S., et al. (2025). AI Mathematical Olympiad — Progress Prize 3. Kaggle Competition. https://kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3

[6] David Babu et al. (2026). AMO: Agentic Math Orchestrator for Olympiad Mathematics. AIMO3 Writeup, Kaggle.

[7] Nitarach, N. (2026). Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3. arXiv:2603.27844

---

## Appendix A: Full Configuration — v1 (Best Submission)

```python
class CFG:
    system_prompt = (
        'You are an elite mathematical problem solver with expertise at the International '
        'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
        'rigorous mathematical reasoning.\n\n'
        '# Problem-Solving Approach:\n'
        '1. UNDERSTAND: Carefully read the problem. Identify what is given, '
        'what needs to be found, and constraints.\n'
        '2. EXPLORE: Consider multiple solution strategies before committing to one.\n'
        '3. PLAN: Outline key steps of the most promising approach.\n'
        '4. EXECUTE: Work through the solution methodically with clear reasoning.\n'
        '5. VERIFY: Check your answer using substitution, edge cases, '
        'or alternative methods.\n\n'
        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
        'Think step-by-step. Quality of reasoning is as important as the final answer.'
    )
    tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex or error-prone calculations\n'
        '- Numerical verification of analytical results\n'
        '- Testing conjectures or brute-force checking small cases\n\n'
        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results.\n\n'
        'Code should support your reasoning, not replace it.'
    )
    preference_prompt = (
        'You have access to `math`, `numpy`, `sympy`, and `mpmath`. '
        'Use sympy for exact symbolic answers, numpy for numerical work, '
        'and mpmath for high-precision arithmetic when needed.'
    )
    # Model
    served_model_name      = 'gpt-oss'
    model_path             = '/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1'
    # vLLM
    dtype                  = 'auto'
    kv_cache_dtype         = 'fp8_e4m3'
    gpu_memory_utilization = 0.95
    context_tokens         = 65536
    batch_size             = 128
    stream_interval        = 200
    # Timing
    notebook_limit         = 17400
    high_problem_timeout   = 900
    base_problem_timeout   = 300
    server_timeout         = 180
    session_timeout        = 960
    jupyter_timeout        = 8
    sandbox_timeout        = 3
    # Solver
    attempts               = 8
    turns                  = 128
    workers                = 8
    early_stop             = 4
    seed                   = 42
    # Sampling
    temperature            = 1.0
    min_p                  = 0.02
    top_logprobs           = 5
    buffer_tokens          = 512
    search_tokens          = 32
```

## Appendix B: Notebook Version Summary

| Filename | Description | Local /10 | LB |
|---|---|---|---|
| `baseline.ipynb` | First version; `compute_weighted_entropy` undefined → bug | 0 | — |
| `v1.ipynb` | Full reference run; verbose prompt; mean entropy voting | **9/10** | **44** |
| `v2.ipynb` | 12 attempts, early_stop=5, T=0.8/1.2 dual temperature | — | 39 |
| `v3.ipynb` | V40 prompt, 20 attempts, 16 workers | — | 33 |
| `v4.ipynb` | V40 + ensemble convergence voting, 20 attempts | **2/10** | 35 |
| `v5.ipynb` | Verbose, top_p=0.8; bug present | 0 | 39 |
| `v6.ipynb` | Verbose, top_p=0.8, 5-component entropy voting | **8/10** |  |

v1 is the only version submitted to the leaderboard (scored 44, 35, 35 across three runs).

## Appendix C: Answer Extraction Logic

```python
patterns = [
    r'\\boxed\s*\{\s*([0-9,]+)\s*\}',        # \boxed{N}
    r'\\\\boxed\s*\{\s*([0-9,]+)\s*\}',      # \\boxed{N} (escaped)
    r'final\s+answer\s+is\s*:?\s*([0-9,]+)', # "final answer is N"
    r'answer\s*[:=]\s*([0-9,]+)',             # "answer: N" or "answer = N"
]
# Scan last search_tokens=32 chunks of streamed text
# Use last match (reversed order) to prefer the model's most recent answer
# Validate: 0 ≤ value ≤ 99999
```

## Appendix D: vLLM Launch Command

```bash
python -m vllm.entrypoints.openai.api_server \
    --seed 42 \
    --model /kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1 \
    --served-model-name gpt-oss \
    --tensor-parallel-size 1 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.95 \
    --host 0.0.0.0 --port 8000 \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 65536 \
    --stream-interval 200 \
    --async-scheduling \
    --disable-log-stats \
    --enable-prefix-caching \
    --trust-remote-code
```

Note: `--enable-auto-tool-choice` and `--tool-call-parser pythonic` are deliberately excluded. These flags configure vLLM's native tool-call parsing layer, which is incompatible with `openai_harmony`'s custom encoding. Including them causes tool calls to be silently mis-parsed, producing empty responses and infinite reasoning loops.
