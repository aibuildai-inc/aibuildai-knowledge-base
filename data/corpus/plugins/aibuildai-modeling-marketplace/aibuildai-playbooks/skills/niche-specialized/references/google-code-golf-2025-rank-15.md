# 15th Place Solution

Competition: google-code-golf-2025
Rank: #15
Source: https://www.kaggle.com/c/google-code-golf-2025/writeups/15th-place-solution

We would like to thank the host and the Kaggle team for organizing such a creative and inspiring competition.
It was our first experience with code golf, and we learned a lot through this exciting challenge.

Team Member: @chimaki821 @isakatsuyoshi @denden12 @hikari30 @kosirowada

table of contents:
1. Code Golf by Hand
2. Code Golf by LLM
3. Code Golf by Rulebase
4. Code Golf by Compression

---

## 1. Code Golf by Hand

Since it was difficult for LLMs to solve tasks from scratch or fundamentally improve the algorithms, we manually created many “initial solutions”.
During this process, the discussions on the forum were extremely helpful. In particular, the two discussions we posted:

[1] [How do you golf neighbor checks? (task279)](https://www.kaggle.com/competitions/google-code-golf-2025/discussion/611930)

[2] [Diagonal Extension (task034)](https://www.kaggle.com/competitions/google-code-golf-2025/discussion/612504)

led to productive exchanges with many participants, and we were able to incorporate the insights gained from them. We sincerely thank everyone who contributed.

### 1.1. Algorithmic Improvements

We focused on simplifying algorithms by observing the test cases carefully and exploiting structural patterns in the grids.
Our main strategies were as follows:

#### (a) Simplify by Observation

* Identify fixed input/output sizes or recurring numbers.
* Detect common layouts, such as one object in the upper half and another in the lower half.
* Apply operations to entire rows or grids whenever possible.

#### (b) Write Concise and Efficient Logic

* Use high-level functions like `any`, `all`, `zip`, `map`, `filter`, and `sum`.
* Replace nested loops and index access with more direct operations.
* Prefer `list.index` or `enumerate` over manual `range` loops.

#### (c) Smart Techniques
* Employ **recursion** and grid **rotation** to reduce repeated neighbor checks.
* Leverage **regex** where possible — we solved 31 tasks with regex-based solutions.
  * see [chimaki’s notebook](https://www.kaggle.com/code/chimaki821/step-by-step-regex-practice) 

### 1.2. Identifying Similar Problems

While solving tasks manually, we noticed that many problems shared similar structures. Recognizing these patterns allowed us to apply successful strategies from one task to another.
Thanks to @denden12, who compiled a PDF summarizing all the tasks, we were able to discover such similarities more easily.

For example:

* task154 and task390
* task351 and task400
* task246 and task335

Since these problems were almost identical, we tackled them together and aimed for score improvements simultaneously.

---

## 2. Code Golf by LLM 

We racked up a lot of points by taking strong human‑written seed solutions and then “follow‑up code‑golfing” them with AIs like GPT5 Pro.

We adopted this workflow for several reasons:

- We wanted to conserve human effort and let AI handle what AI can.
- Current high-end AI can relatively easily golf code when the approach direction is clear.
  - However, AI struggles with 0->1 creation or escaping local optima, where manual code golf remains effective.
- Diversity is key to escaping local optima, such as moving from a manual solution to AI, or from one AI to another, is the key to escaping local optima.

### 2.1. LLMs We Used

- We tried GPT, Claude, Gemini, and Grok.
- In the early phase, we used Claude Code with DSL implementation to achieve AC results.
- After GPT‑5 was released, we shifted toward it. GPT‑5’s performance was astonishing, achieving many 0→1 AC results.
- In the end, we mainly used GPT‑5 Codex and GPT‑5 Pro.

### 2.2. Prompt Engineering

We put significant effort into prompt design and system-level context construction.
The key idea was to provide good examples and rich context so that the LLM could learn how to golf effectively.

- Added numerous examples of codes that had already matched the first-place scores.
  - This enabled the models to perform advanced transformations such as recursion, which are typically difficult for LLMs.
- Included before/after examples of effective golfing tips.
- Used [GPT-5 Prompt Optimizer](https://platform.openai.com/chat/edit?models=gpt-5&optimize=true)  to systematically improve output quality and consistency.
- Supplemented prompts with custom text files, which included:
  - Programming-contest–style test cases.
  - Translated [hints originally written in Chinese (by 暗黑AGI-san)](https://www.kaggle.com/code/boristown/agi-chinese-hints-for-all-400-tasks) , converted into Japanese and English via the [PLaMo API](https://plamo.preferredai.jp/) .
- Combined these into a structured “prompt file” for each problem, containing code examples, test data, and textual hints.
  The image below shows an example of such a prompt file used for AI-driven code golf experiments.



- Using the tricks learned from writing shorter code manually helped the prompt get better scores. With this smart idea, even gpt-5-mini got a better score about one time out of every 50 tries.

These structured prompt files helped the models understand both the problem format and the essence of golfing strategies, resulting in consistently better code compression.

### 2.3. Custom Evaluation Script for AI Agents

We modified the official evaluation script to better support LLM-based experimentation:

- Displayed both plain byte count and zlib/zopfli-compressed size.
- Computed and displayed the gap from the first-place score.
  - For example, GPT-5 Codex often reduced a ~50-point gap to match the 1st-place score.
- Implemented a hack filter to reject invalid or exploitative outputs, such as:

  ```python
  p=lambda g:type("",(),{"__eq__":lambda*_:1})()
  ```
  (a 46-byte snippet that passes any test case locally but fails on the leaderboard).

### 2.4. API Automation

After collecting a sufficient number of strong initial solutions, we automated the process:

For each task, we prompted the models with:

- A set of programming-contest–style test cases,
- The currently shortest known code, and
- 30–50 randomly selected solutions from other problems.

Using gpt-5, gpt-5-mini, and grok-4, we performed iterative improvements.
On average, gpt-5 produced a few-byte improvement once every 50 problems, and gpt-5-mini once every 100 problems.

---

## 3. Code Golf by Rulebase

Our team maintained two synchronized datasets throughout the competition:

1. Plain Shortest Codes : the shortest uncompressed version.
2. Best Score Codes : the version achieving the best score considering compression.

After each update or improvement, both datasets were automatically refreshed through a post-processing pipeline.

### 3.1. Rule-based Post-processing Notebook

We developed 23 rule-based notebooks that automatically executed sequentially every minute.
Each notebook applied a specific transformation to further shorten the code while maintaining correctness.
The following summarizes the main rules used:

1. Remove comments
2. Remove unnecessary whitespaces
3. Remove soft (non-critical) whitespaces
4. Expand `if True:` blocks and remove `if False:` blocks entirely
5. Replace `True/False` with `1/0`
6. Replace `len(x) == 0` with `not x`
7. Replace `x == False` with `not x`
8. Replace `x == True` with `x`
9. Safely remove `pass` statements
10. Shorten `return None` → `return`
11. Remove trailing semicolons `;`
12. Remove redundant parentheses
13. Remove unused imports
14. Remove unnecessary `list()` calls for literals
15. Automatically shorten local variable, function, and argument names to one character
16. Apply global one-character renaming via AST transformation
17. Remove indentation and compress function bodies into single lines
18. Replace `l.append(x)` with `l += [x]`
19. Shorten `from X import Y` → `from X import *`
20. Minify identifiers using external libraries (e.g., pyminifier-based tools)
21. Reduce indentation width from 4 spaces to 1 space
22. Inject short aliases for built-in functions (e.g., `range → R`)
23. Remove all `print` statements

This automated pipeline continuously refined submissions to approach minimal byte counts.
By keeping these transformations modular and incremental, we ensured that even minor edits propagated safely to the latest code versions.

### 3.2. AI-based Post-processing

Beyond the rule-based framework, we also automated AI-based post-processing notebooks using OpenAI API, Gemini API, and Grok API.
These notebooks are periodically executed to perform more flexible, non-rule-based optimizations and capture improvements that purely static rules could not.


### 3.3. pysearch

We applied targeted transformations using [pysearch](https://github.com/lynn/pysearch).
By specifying desired input/output pairs, pysearch synthesizes the shortest logic that satisfies them, which improved scores for some problems.

Example:
For `task144.py`, we needed `3*(x==y)` with `x∈[0,7]` and `y∈[0,2]`.
pysearch rewrote it to `3>>x+y`, saving 2 bytes.

---

## 4. Code Golf by Compression

- **Grammar optimization**: Considering zlib/zopfli compression, we sometimes intentionally increased raw byte count to gain a better compressed score.
  For example, aliasing `range` as `R=range` and using `R()` can reduce raw bytes, but not aliasing can compress better.
  We first performed some of these conversions manually, then turned them into prompt examples and used GPT‑5 to revise code accordingly.

- **Variable-name optimization**: Based on Garry Moss’s notebook
  [`compressed-variable-name-optimization`](https://www.kaggle.com/code/garrymoss/compressed-variable-name-optimization),
  we accelerated and tuned the search.
  We deferred the expensive correctness checks, first computing compressed byte counts to speed up iterations, and reset the trial counter whenever the score improved.
  In practice, we ran the compression-aware rewrites first and then applied variable-name optimization, which yielded better overall compression efficiency.
