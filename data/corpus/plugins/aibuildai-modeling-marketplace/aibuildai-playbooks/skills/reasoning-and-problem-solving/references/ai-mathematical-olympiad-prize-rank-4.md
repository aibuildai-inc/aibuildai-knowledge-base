# 4th place solution

Competition: ai-mathematical-olympiad-prize
Rank: #4
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/518960

First, we would like to thank Kaggle and XTX Markets for bringing us this exciting competition. Through this competition, we have learned a lot. We also want to thank every team member for their hard work. They are truly amazing, with many excellent ideas and engineering skills.

Our solution is based on the great work of [AbdurRafae](https://www.kaggle.com/abdurrafae), [Improved Code Interpretation (kaggle.com)](https://www.kaggle.com/code/abdurrafae/improved-code-interpretation). His Early Sharing Prize is well-deserved. Similarly, [Anren](https://www.kaggle.com/anrenk)'s reorganization of this code provided us with great help.

Below is a brief introduction to our solution's overall process and some unique tricks. Our solution code in here

[4th Dual gpu reasoning and comparing answers (kaggle.com)](https://www.kaggle.com/code/winter2003/4th-dual-gpu-reasoning-and-comparing-answers)



### Local CV Strategy

We used this 71k dataset: [AIMO-24: Processor (Art Of Problem Solving) (kaggle.com)](https://www.kaggle.com/code/dinhttrandrise/aimo-24-processor-art-of-problem-solving/output?select=problems.csv). From all AMC_12A problems, we selected the latest 50 for our local validation (excluding problems with the `[asy]` string in the text, which will be explained later). This showed good correlation with the public leaderboard.



### Model Selection

We used `deepseek-math-7b-rl`, with parameters: temperature of 0.9, top_p of 1.0, and max tokens of 2048. This model with code tools, can achieve 58.8% on MATH benchmarks.

We deployed and ran this model in parallel on two T4 GPUs to maximize self-consistency for final prediction accuracy. Dual-GPU inference increased the total repetitions per problem from 21 to 30-40.



### Time Management

**TimeManager Class**

We created a `TimeManager` class to manage time allocation. It tracks the time spent on each problem and dynamically adjusts the number of attempts for subsequent problems to ensure as many problems are solved as possible within the given time.

```python
class TimeManager:
    def __init__(self, total_problems, total_time, default_attempts):
        self.total_problems = total_problems  # Total number of problems
        self.total_time = total_time  # Total time available
        self.default_attempts = default_attempts  # Default number of attempts
        self.time_spent = 0  # Time spent so far
        self.current_problem = 0  # Current problem index
        self.last_problem_time = 0  # Time spent on the last problem
    
    def update_time_spent(self, time_for_problem):
        self.time_spent += time_for_problem  # Add time spent on the problem to the total time spent
        self.current_problem += 1  # Increment the problem index
        self.last_problem_time = time_for_problem  # Record the time spent on the current problem
    
    def get_next_attempts(self):
        if self.current_problem >= self.total_problems:
            return 0
        
        remaining_time = self.total_time - self.time_spent  # Remaining time
        remaining_problems = self.total_problems - self.current_problem  # Remaining problems
        average_time_per_problem = remaining_time / remaining_problems  # Average time per problem
        
        if self.current_problem == 0 or self.time_spent / self.current_problem <= average_time_per_problem:
            next_attempts = self.default_attempts
        else:
            next_attempts = max(1, int(self.default_attempts * (average_time_per_problem / (self.time_spent / self.current_problem)) * 0.8))
        
        return next_attempts
    
    def print_status(self):
        remaining_time = self.total_time - self.time_spent  # Remaining time
        print(f"Time spent: {self.time_spent} seconds")  # Print total time spent
        print(f"Remaining time: {remaining_time} seconds")  # Print remaining time
        print(f"Current problem: {self.current_problem}/{self.total_problems}")  # Print current problem progress
        print(f"Time spent on current problem: {self.last_problem_time} seconds")  # Print time spent on the current problem
```

**Reducing Attempts for Difficult Problems**

We manually reduced the number of attempts for problems we believed the model could not solve, providing more attempts for the remaining problems. Specifically, we only allowed 3 repetitions for problems containing the `[asy]` symbol. This symbol indicates a geometric problem with a figure, surrounded by `[asy]` in LaTeX. In our local tests, the model rarely answered these problems correctly, and even when it did, we suspected the problem might have been in the model's training set.

**Reducing Waiting Time Between Dual GPU Runs**

When using `ThreadPoolExecutor` for parallel inference, one GPU often finishes faster. If we see the first completed GPU has enough candidate answers, we will terminate the slower GPU's attempts early. And then, merge the candidate answers generated by the two gpu's. This ensures more efficient use of dual GPUs and increases the average number of repetitions for all problems.



### Candidate Answer Generation

DeepSeekMath is initialized with [DeepSeek-Coder-v1.5 7B](https://huggingface.co/deepseek-ai/deepseek-coder-7b-base-v1.5), which can effectively solve and prove mathematical problems by writing programs. We followed [Anren](https://www.kaggle.com/anrenk)'s method with two prompts, letting the model solve problems using COT and write code methods.

We made some improvements:

1. **Limiting the Model's Code Modify Opportunities**

   We limited the model to only 3 code modify opportunities per repetition, as input text that is too long significantly reduces the output quality of LLMs. This feature is controlled by the parameter `while_limit`, which we set to 6(as each code input and output processing requires two while loop).

2. **Weighted Count Sorting for Candidate Answers**

   We discovered that the model tends to output small integers like 0, 1, 2, 3, 4, 5 when answering incorrectly. We reduced the weight of these numbers to 0.25, while other numbers retained the normal weight of 1.

3. **Fixing Some Incorrect Behaviors**

   For example, when the code output is not a valid answer (e.g., error, decimal, complex number), the original method still tried to parse the text answer, which was almost always incorrect. In such cases, we skipped recording the text answer.



### Other

Our logs recorded detailed data for each experiment, including candidate answers, program code, code accuracy, text accuracy, etc. These detailed logs greatly facilitated our quick experiments and result observations. Our local experiment score was around 21, with pass1@ reaching 32 and pass2@ reaching 23.

Of course, some luck might be needed in the competition. Our solution on the public leaderboard, using different seeds, could result in a 3-point score difference.



**Detailed Local Log (parameters slightly different)**
[pic]
