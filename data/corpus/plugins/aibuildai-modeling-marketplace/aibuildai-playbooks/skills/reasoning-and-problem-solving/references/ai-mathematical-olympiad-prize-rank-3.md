# 3rd place solution

Competition: ai-mathematical-olympiad-prize
Rank: #3
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/517206

# vLLM with some tricks

**[Code for our final solution can be found here.](https://www.kaggle.com/code/davidd2/3rd-place-solution-afterexams)**

Our final selected solution was an adaptation of the notebook of [AbdurRafae](https://www.kaggle.com/code/abdurrafae/improved-code-interpretation). We used DeepSeek-Math-7B-RL without any finetuning and did majority voting with a scoring rule we made.

## vLLM
We decided to go with vLLM due to the better speed when compared to the Huggingface Transformers.
- We generate a large number of candidate solutions (usually between than 120-160). 
- We found that making the KV cache be in FP16 improved our score.

## Generation
We generate the solutions in one batch using iterations. An iteration is made for each code that has to be run.
- We found that a large number of iterations >6 helped the score.
- To ensure the model consistently outputs the answer in the \boxed{} format, we used the following approach. When the generation finishes without an answer, we append a string such as "The final answer is \boxed{" to the output and generate a few more tokens. This prompt forces the model to output the answer correctly.
- For each iteration we took all code that needed to be executed and did it in parallel in batches. This has reduced the amount of time we spent on code execution significantly.

## Scoring rule
Regarding the scoring rule we found that at least on our validation, the model often had two types of errors as the final result:
1. Numbers which are small (<10), this often happened because of errors in the code.
2. Numbers which were part of the problem statement. We found this to be especially pronounced and the number of times this happened outweighed the chances for the problem statement to actually contain the answer.

We thus penalized these results as a percentage of the total number of generations we attempted. We have observed a significant boost in score by penalizing numbers from the problem statement.

# What else we tried
For the final submission we chose this old version due to the possibility of having ties in the final leaderboard.
We have also created a number of different versions.

### BF16
We observed that running the model in BF16 improved performance and thus we tried to make our solution work with BF16. The problem is that vLLM does not support this by default on the T4s. One can however perform the computations in FP32 and cast back to BF16.

We have thus slightly modified the vLLM library code to remove the check for BF16 support and wrapped the attention module so we convert back and forth between fp32 and bf16.

We thus have two versions of the libraries:
- [Full bf16](https://www.kaggle.com/datasets/davidd2/vllm-0-4-0-post1-bf16/data)
- [Only attention and kv cache in bf16](https://www.kaggle.com/datasets/davidd2/vllm-0-4-0-post1-only-attn-in-bf16/data)
The changes we made to vLLM were pretty minimal: [Github](https://github.com/RD211/vllm-bf16)

We found that the making the MLP be in bf16 made the model very slow and thus we also created the version that keeps everything but the MLP in bf16.
Models with these libraries performed well but were not stable enough for our liking. They gave us the highest score we have on our validation but did not go over 25 on the public leaderboard.

### Executing code during generation
We observed that making multiple calls to vllm took a significant amount of time, further one cannot early exit reliably with intermediate results of solutions. We have thus made a new version of the notebook we initially scored 26-27 with that feeds the result of the code generation directly back to the model without needing two vllm generate calls.

We do this by using the logit processors and forcing the model to output the tokens of the code output. We also enforce all early breaking and pruning rules directly in the logit processor function.
[Code for this can be found here](https://www.kaggle.com/code/davidd2/vllm-continous-generation-with-code-outputs/notebook)
We personally liked this submission more than our final one but due to time constraints we could not make it score as high as the old one
