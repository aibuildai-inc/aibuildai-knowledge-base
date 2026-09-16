# Less Prompting, More Trust

Competition: ai-mathematical-olympiad-progress-prize-3
Rank: #2
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-3/writeups/less-prompting-more-trust

# AIMO Progress Prize 3

## Acknowledgments

This solution builds on [Sachchidananda Daki's public notebook (v45)](https://www.kaggle.com/code/sachchidanandadaki/aimo-3-submission), which scored 40/50 on the public leaderboard using vLLM serving, Harmony encoding, a sandboxed Jupyter tool, parallel attempts, entropy-weighted voting, early-stopping, and adaptive time budgeting. My changes were to the prompts and several hyperparameters (detailed in the Prompts and Configuration sections below), which raised the score to 45/50. Thanks also to andreasbis/aimo-3-utils for the offline wheel bundle.

## Overview

The solver runs `gpt-oss-120b` on a single H100 via vLLM, with fp8 KV cache, prefix caching, and a 64K context. For each problem, the system issues 8 independently seeded attempts in parallel at temperature=1.0 and reasoning_effort=HIGH. Each attempt has access to a stateful Jupyter sandbox (math, numpy, sympy, itertools, collections, mpmath with 64 digit precision) that the model can call mid-generation via Harmony tool-use messages. Output is streamed token by token, and whenever a } appears, the last 64 tokens are scanned. If \boxed{...} is found, the attempt terminates immediately with that answer.

Once an attempt produces a final answer, it is pooled with the others. If four attempts agree on the same answer, the remaining attempts are cancelled (early stop). Otherwise the system waits for all 8 attempts to complete and then selects the answer using entropy-weighted voting.

Time is budgeted globally within the notebook's total runtime limit of 17,400s. Each problem reserves a base budget of 300s but may extend up to 900s if remaining time exceeds what the remaining problems need.




The rest of this section zooms in on the pieces referenced above: the per-attempt turn loop, the entropy-weighted voting used for tie-breaking, the three prompts, the configuration, and how answers are extracted from streamed output.

### Turn loop

The model can reason freely, but it cannot execute Python on its own. When it writes code inside a tool-call message and emits the hand-off token, generation has to stop so the orchestrator can actually run that code in a sandbox and feed the real output back. Without this pause, the model would just keep generating and hallucinate what it thinks the output should be. A turn is exactly one round of generation bounded by these hand-offs.

After a turn ends, the orchestrator looks at the last message. If it's on the final channel, the boxed answer is extracted and the attempt is done. If it's a Python tool call, the code runs in the sandbox and the real output is appended as a tool-response message before the next turn begins. 

### Entropy-weighted voting

During generation, at every token position the model produces we record the top-5 log-probabilities for that position. At the end of the attempt, each log-probability is converted to a probability (via `exp`), and the Shannon entropy at that position is computed as `-sum(p_i * log2(p_i))` across those 5 values. Summing those per-position entropies and dividing by the token count gives the attempt's mean entropy: a single number where low means the model was consistently sure of itself, high means it was uncertain throughout.

Each attempt's vote is then weighted by 1 / mean_entropy. Weights are summed per candidate answer, and the answer with the highest total weight wins.

A worked example makes the behavior clearer. Suppose a problem launched 8 attempts and 3 failed to produce a valid answer (timed out, crashed, or never emitted a `\boxed{}`). The remaining 5 valid attempts look like this:

| Attempt | Answer | Mean entropy |
|---|---|---|
| 1 | 8687 | 0.75 |
| 2 | 8687 | 0.69 |
| 3 | 48673 | 1.77 |
| 4 | 48673 | 1.12 |
| 5 | 14 | 0.88 |

No answer reached 4 agreements, so early-stop didn't fire, so we fall into the voting path.

**Step 1: Invert each entropy into a weight (`1 / entropy`):**

| Attempt | Answer | Entropy | Weight |
|---|---|---|---|
| 1 | 8687 | 0.75 | 1.334 |
| 2 | 8687 | 0.69 | 1.449 |
| 3 | 48673 | 1.77 | 0.565 |
| 4 | 48673 | 1.12 | 0.893 |
| 5 | 14 | 0.88 | 1.136 |

**Step 2: Sum weights per candidate answer:**

- 8687: 1.334 + 1.449 = 2.783
- 48673: 0.565 + 0.893 = 1.458
- 14: 1.136

**Step 3: Pick the highest. 8687 wins with 2.783.**

8687 and 48673 both had 2 raw votes, so a plain majority vote would have tied them. But 8687's attempts had low entropy (0.75 and 0.69), while 48673's were much noisier (1.77 and 1.12), so 8687 wins the weighted vote. Answer 14 was individually confident (entropy 0.88) but only had one attempt behind it, so it couldn't catch up.

## Prompts

Three prompts are used. All three are deliberately minimal. They state the contract and the available tooling, and do not prescribe reasoning steps.

The v45 baseline used a much more elaborate set of prompts (~65 lines total), explicitly scaffolding the model's reasoning through an UNDERSTAND → EXPLORE → PLAN → EXECUTE → VERIFY protocol, along with detailed mathematical reasoning principles, verification requirements, and library best-practices:

**System prompt:**

```python
system_prompt = (
    'You are an elite mathematical problem solver with expertise at the International '
    'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
    'rigorous mathematical reasoning.\n\n'
    '# Problem-Solving Approach:\n'
    '1. UNDERSTAND: Carefully read and rephrase the problem in your own words.\n'
    '2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems.\n'
    '3. PLAN: Select the most promising approach and outline key steps before executing.\n'
    '4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n'
    '5. VERIFY: Check your answer by substituting back, testing edge cases.\n\n'
    '# Mathematical Reasoning Principles:\n'
    '- Break complex problems into smaller, manageable sub-problems\n'
    ...
    '# Verification Requirements:\n'
    ...
    '# Output Format:\n'
    ...
)
```

**Tool prompt:**

```python
tool_prompt = (
    'Use this tool to execute Python code for:\n'
    '- Complex calculations that would be error-prone by hand\n'
    ...
    'Remember: Code should support your mathematical reasoning, not replace it. '
    'Explain what you\'re computing and why before running code.'
)
```

**Preference prompt:**

```python
preference_prompt = (
    'You have access to `math`, `numpy`, and `sympy` for:\n\n'
    '# Symbolic Computation (sympy):\n'
    ...
    '# Numerical Computation (numpy):\n'
    ...
)
```

This baseline scored 40/50 on the public leaderboard.

The new prompts collapse this to ~10 lines total, removing the reasoning scaffold entirely and keeping only the task contract and tool description.

**System prompt** (sets the task contract):
```python
system_prompt = (
    'Solve the following math problem step by step.\n'
    'Use Python code for calculations, verification, or brute-force search.\n'
    'The answer is a non-negative integer in [0, 99999].\n'
    'Put your final answer in \\boxed{}.'
)
```
**Tool prompt** (describes the Python sandbox, attached to the Harmony `ToolNamespaceConfig`):
```python
tool_prompt = (
    'Execute Python code in a stateful Jupyter notebook. '
    'Available: math, numpy, sympy, itertools, collections, mpmath (64-digit precision). '
    'Use print() to see results.'
)
```
**Preference prompt** (appended to each problem; nudges library choice, not reasoning):
```python
preference_prompt = (
    'Use sympy for exact symbolic computation. '
    'For number theory: use sympy.ntheory (factorint, divisors, totient, isprime). '
    'Verify analytical solutions with numerical checks or brute-force when feasible.'
)
```

With these changes (along with the configuration tweaks described in the next section), the score moved from 40/50 to 45/50. The prompt simplification appears to be a major factor, though it was not isolated from the other hyperparameter changes described in the Configuration section.

As DeepSeek-R1 noted, heavy chain-of-thought scaffolding (UNDERSTAND → EXPLORE → PLAN → EXECUTE → VERIFY style) can interfere with models that already have strong internal reasoning. I believe gpt-oss-120b at `reasoning_effort=HIGH` is such a model, so I tried simplifying the prompts. They specify *what* the model must do (answer format, answer range) and *what* it has access to (tooling), but not *how* to think. Reasoning is left entirely to the model.

### Configuration

Compared to v45, fewer attempts run in parallel (16 → 8), but each gets more compute and time: workers per problem doubled (8 → 16), the early-stop threshold rose from 3 to 4 agreements, and the per-problem time budget grew (145s → 300s base, 250s → 900s ceiling). The full final configuration is below.

| Parameter | Value | Notes |
|---|---|---|
| Model | `gpt-oss-120b` | served via vLLM, fp8 KV cache, prefix caching |
| `reasoning_effort` | HIGH | Harmony system content |
| Context | 65,536 tokens | `max_model_len` |
| Per-turn max tokens | 4,096 | truncation guard |
| Temperature | 1.0 | |
| `min_p` | 0.02 | |
| Attempts per problem | 8 | parallel, independently seeded |
| Workers | 16 | worker threads per problem |
| Turns per attempt | 128 | max tool-use turns |
| Early-stop threshold | 4 agreements | |
| Voting | entropy-weighted | `weight = 1 / mean_token_entropy` |
| Base per-problem timeout | 300s | |
| Max per-problem timeout | 900s | extended if notebook time allows |
| Notebook runtime limit | 17,400s | about 4h 50m |
| Seed | 42 | per-attempt seed = `(seed + i)^2` |

### Answer extraction

During streaming, each chunk is watched for a `}` character. When one appears, the last 64 tokens of text are searched with the regex `\\boxed\s*\{\s*([0-9,]+)\s*\}`. The trailing match wins, commas are stripped, and the value is kept if it falls in [0, 99999]. If the full message completes without a `\boxed{}` match, a fallback regex `final\s+answer\s+is\s*([0-9,]+)` is applied. Attempts without a valid answer contribute no vote.

## What didn't work

The following variants were tested during development and did not improve on the configuration above:

1. **Two-round retry**: If the initial 8 attempts produced ≤2 agreement, launch another 8 attempts with an additional 900s budget, then entropy-vote across all 16. Result: 40/50. 

2. **Temperature sweep**: A sweep over {0.85, 1.00, 1.15} was run on 10 problems from reference.csv. The nine easy/medium problems were solved at all three settings. On the single hard problem, out of 8 parallel attempts, temp 0.85 yielded 1 correct, temp 1.00 yielded 0, and temp 1.15 yielded 3. Another run at temp 1.15 on the same problem yielded different hit counts, suggesting that factors beyond temperature alone (such as vLLM internal state, KV cache reuse, or scheduling nondeterminism) also affect outcomes. The signal was judged too noisy to justify deviating from temp 1.0.

3. **Concept-first agentic loop**: Four-step pipeline: (a) LLM enumerates relevant concepts for the problem, (b) LLM drafts a strategy from {problem + concepts}, (c) LLM identifies potential pitfalls and wrong assumptions, (d) LLM solves the problem with the full accumulated context. On the hard problems where the baseline already struggled, the extra stages did not produce more correct answers, and on some runs zero out of 8 attempts reached the correct answer.

4. **Solver/verifier loop**: The solver proposes a solving strategy, and the verifier returns either approved or rejected. The loop continues until a strategy gets approved, and the solver then uses that approved strategy to solve the problem. This also did not work out well. The verifier often approved flawed strategies or rejected reasonable ones, and the resulting solutions were no more accurate than single-shot solving. Correct and incorrect answers frequently tied in the final vote.

5. **AIMO2-style verifier wave (from NVIDIA's AIMO2 solution)**: NVIDIA's AIMO2 protocol generates 64 candidates and runs a tournament-style verification: 32 times, the candidates are grouped into sets of 8, the best one from each group is picked, and then the best one is picked among those winners. The final answer is decided by majority vote across the 32 tournament winners. I tried a simpler version. If the initial 8 attempts had consensus ≤2, a second stage summarizes the 8 solutions and runs 8 verifier passes, where each pass sees the summarized candidates in a randomly shuffled order and picks the best one. The candidate with the most votes across the 8 passes wins. This did not work out well. The pipeline often ended up with the correct answer getting two votes and an incorrect one also getting two votes, with no clear winner. The approach might have worked better with more verifier passes, but I did not try that due to the time budget. One possible factor is that NVIDIA's GenSelect used a dedicated selector model trained on matched data, while I was using an off-the-shelf gpt-oss-120b for both solving and verification.

6. **Per-answer verifier scoring wave**: If the initial 8 attempts had ≤2 consensus, a second stage runs where every unique candidate answer is shown to the verifier 8 times. Each pass assigns a score in {1, 2, 3} reflecting how likely the candidate is correct. Scores are summed per candidate across all passes, and the candidate with the highest total wins. Same underlying problem as (5). The verifier was not a meaningfully stronger judge than the solver.

Across the variants I tested, adding structure to guide the model's reasoning (extra stages, verifier loops, scaffolded prompts) did not improve accuracy. The configuration above (minimal prompts, single-shot parallel sampling, entropy voting) outperformed all tested elaborations. NVIDIA's AIMO2 solution showed that verifier-based selection can work well, but it relied on a dedicated selector trained on matched data. It seems that imposing structure on how a strong reasoner thinks tends to hurt rather than help, unless the model has been trained for that specific structure.


## Conclusion

Across every variant I tested, adding structure (verifier loops, scaffolded prompts, multi-stage pipelines) hurt rather than helped. The minimal configuration outperformed all elaborations. It seems that for a model with strong internal reasoning, prompt and pipeline design should clarify the contract, not direct the thinking. Future work likely lies in better selection across attempts (a stronger judge than the solver itself), not in further structuring the solver.
