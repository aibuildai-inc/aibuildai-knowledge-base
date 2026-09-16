# 41st Place Solution

Competition: ai-mathematical-olympiad-prize
Rank: #41
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/516868

Thank you to the hosts and participants for hosting this competition. I am very happy to get the silver medal and become a master.

My solution is as follows.

#### ・Prompt

I emphasized self-doubt. This may make the AI ​​think repeatedly and arrive at the correct answer. Typing "Be skeptical of your answers." three times was the most effective against LB.

```python
code = """Below is a math problem you are to solve (Non-negative integer answer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even an idiot can follow your instructions, and remember, your final answer should be Non-negative integer, not an algebraic expression!
Be skeptical of your answers.
Be skeptical of your answers.
Be skeptical of your answers.
Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final Non-negative integer answer within \\boxed{}.

Approach:"""


cot = """Below is a math problem you are to solve (Non-negative integer answer!):
\"{}\"
Be skeptical of your answers.
Be skeptical of your answers.
Be skeptical of your answers.
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final Non-negative integer answer within \\boxed{}.\n\n"""

promplt_options = [code,cot]
```
#### ・Temperature
Temperature is set to 0.7 to improve stability.

#### ・Time management
Some public notes could exceed the time limit. So made sure they could be solved within the time limit. Also added a variable called "time_for_difficult" to allocate more time to difficult problems.

```python
time_for_difficult = 9000

solve_time_standard = max(300,int((32200 - consumed_time)/(51 - problem_count)))
solve_time = solve_time_standard
if problem_count <= 45 and time_for_difficult >= 1 and consumed_time < 29200:
    solve_time += min(600,time_for_difficult)    
time_for_difficult -= 200

for jj in tqdm(range(n_repetitions)):
    best, best_count = best_stats.get(i,(-1,-1))
    problem_time = time.time()-PROBLEM_START_TIME

    if problem_time >= 720 and best_count >= 4:
        time_for_difficult -= problem_time - solve_time_standard
        return best_stats[0][0]
    if problem_time >= 900 and best_count >= 2:
        time_for_difficult -= problem_time - solve_time_standard
        return best_stats[0][0]
```
