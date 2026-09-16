# 11th place solution

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #11
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/573086

I’m grateful to Kaggle and the competition organizers for hosting this challenge. I’d also like to extend my thanks to the participants who shared insightful discussions and valuable datasets.


# Summary

Using [**deepseek-r1-distill-qwen-14B-awq-casperhansen**](https://www.kaggle.com/models/huikang/deepseek-r1/Transformers/deepseek-r1-distill-qwen-14b-awq-casperhansen/1) with **SC-TIR**.
By **modifying vllm**, the inference can end early when Self-Consistency (SC) converges or diverges.  
Evaluation is done using the **most recent 100 AIME problems**. No machine resources were available to do GRPO. 



# Strategy

### 1. Model

- Adopted **deepseek-r1-distill-qwen-14B-awq-casperhansen**  
- Compared 7B, 14B, and 32B:
  - 32B is too slow and not perfect.
  - 14B > 7B (in terms of accuracy), 14B < 7B (in terms of speed)  
  - Increasing the number of generations for 14B made it more competitive

### 2. Self-Consistency

- Its effectiveness was already demonstrated in AIMO1.  
- It was also a standard approach in AIMO2.

### 3. Custom vllm

- Employed 14B and modified **vllm** to allow as many generates as possible.
- For easy problems → outputs don’t vary much, so there’s no need to generate up to `max_num_seqs`.  
- For overly difficult problems → the output often ends up being 0 and -1 or diverging (no unique solution). Even generating the full `max_num_seqs` won't lead to a correct answer in many such cases.  
- As a result, the number of problems that actually require the full `max_num_seqs` length to determine the outcome is surprisingly small.



- **There are too many -1 and 0**  
  - If there are a certain number of -1 or 0 (invalid responses), it ends inference early.  
  - 5 ~ 7 responses → terminate if 5 or more are -1/0  
  - 8 ~ 11 responses → terminate if 4 or more are -1/0  
  - 12 ~ 16 responses → terminate if 6 or more are -1/0

- **There responses have diverged**  
  - Once there are at least 8 responses, if there are 6 or more different valid responses, I consider it too scattered and terminate.

- **The winner has been decided**  
  - If the difference in counts between the most frequent answer and the second-most frequent answer exceeds a certain threshold, the winner is considered determined, and inference stops.  
  - 4 ~ 7 responses → difference ≥ 3  
  - 8 ~ 11 responses → difference ≥ 2  
  - 12 ~ 16 responses → difference ≥ 1  

### 4. Validation dataset

- Used the 100 most recent AIME problems.
- Because the score varies, I tested several times and took the average for pipeline evaluation.
- The [public notebook](https://www.kaggle.com/code/huikang/thought-engineering-r1-distill-qwen-7b-awq) score was about 63–68, while my pipeline scored around 75–79.

### 5. one-shot

- Even problems that are mathematically difficult to solve might still be solved by a **TIR** exhaustive search.
- I included the **exhaustive search approach** in the few-shot prompt.
- Prompts such as `Let's have Python do the tedious calculations for us!` and  
  `There are multiple ways to solve this problem, so find the most efficient one.`  
  contributed slightly to improved scores.

```
You are a Python code assistant. 

You will be given a mathematical problem that has integer solutions.
Your task is to convert this complex math problem into Python code.
Let's have Python do the tedious calculations for us!

There are multiple ways to solve this problem, so find the most efficient one.

- The final answer should be an integer.
- The final answer should be modulo 1000.
- Please return Python code only following the format below.

\`\`\`python
import math 

...
# Intermediate calculations
...

print(<answer> % 1000)
\`\`\`

Here is an example.

User:
Find the number of ordered pairs $(m, n)$ such that $m$ and $n$ are positive integers in the set $\{1, 2, ..., 30\}$ and the greatest common divisor of $2^m + 1$ and $2^n - 1$ is not $1$ .

Assistant:
\`\`\`python
import math

answer = 0

# Check all m, n in [1, 30]
for m in range(1, 31):
    for n in range(1, 31):
        # Compute GCD of (2^m + 1) and (2^n - 1)
        if math.gcd(2**m + 1, 2**n - 1) != 1:
            answer += 1

# Print the result modulo 1000
print(answer % 1000)
\`\`\`
```




Public LB: 29
Private LB: 28
Inference Notebook: https://www.kaggle.com/code/farsail/aimo2-inference
Custom vllm: https://www.kaggle.com/code/farsail/pip-install-aimo2-custom-vllm
