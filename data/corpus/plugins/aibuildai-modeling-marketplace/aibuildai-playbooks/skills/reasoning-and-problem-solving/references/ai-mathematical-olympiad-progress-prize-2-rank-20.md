# 20th place solution - generate lots of code with a 32b model

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #20
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/575172

I see that all 10 published so far higher ranked solutions are using 14b models, and (maybe except for the top solution) they seem to be focused on generating reasoning paths long enough so that they will end with generating a text (boxed) answer.

My solution is a little different. I was using a 32b model and I gave up on trying to generate sufficient amount of tokens to obtain a boxed answer.
Instead, at 5k-6k tokens I force the model to generate code multiple times. In total, up to 108 programs in Python per task are generated.
The final answer is chosen not by majority voting, but by a weighting algorithm with carefully chosen weights.

[First solution](https://www.kaggle.com/code/arkadiuszpaterek/aimo-2-new-solution-10?scriptVersionId=230095668) (version 118 of the notebook)
private score 27, public score 26

[Second solution](https://www.kaggle.com/code/arkadiuszpaterek/aimo-2-new-solution-10?scriptVersionId=230875528) (version 137 of the notebook)
private score 25, public score 25

How to run it locally:
1. Copy the set of questions from the cell 2, without the first line, to the appropriate file (dq_data5e.py or dq_data4h.py)
2. Copy the code from the cell 3 (begins with T = True) to the file solve.py
3. Change to:
submit = F
conf_pc = T
4. Change llm_path to the location of the model
5. Either set prior_time_per_task to the average time per task from previous runs, or set use_control = F
6. Run with the command:
python3 solve.py save_dir=1 save_prefix=A runs=1 gpus=0,1
Change save_prefix for every run.

How it works:

Quick summary:
* 32b model served with vLLM
* begin with initial 7 prompts
* generate up to 108 programs per task, and up to 9 boxed answers
* end generating after around 7k tokens
* balance the importance of boxed and code answers
 - shrink the weights (decrease importance) of groups of code answers that have long common prefix
 - large bonus for identical boxed and code answer 
 - penalty for short code
 - penalty for small value of the answer

Additionally:
* temperature decreases from 0.7 to 0.2, and code is generated with temperature 0.15
* some prompts rewrite the question before forcing generating the code
* some initial prompts ask the LLM to first rewrite the question in mathematical notation
* async-await is utilized to run code with 12s timeout, simultaneously with generating text by the LLM
* adaptive control chooses the max number of tokens to generate, depending on time remaining and on a regularized estimate of time used per task

In parallel to generating the code I continue generating the reasoning path that sometimes will give a good boxed answer.
I structure launching prompts in a way that all unfinished reasoning paths and the last wave of unfinished code chunks end at the same token number, so that the tree of generated tokens is cut at a fixed depth.
All that is fast, because vLLM can reuse KV cache entries for prompts with the same prefix.

The program automatically uses less time for easier tasks, because if the LLM returns a boxed answer early, then the whole subtree will be generated sooner, and that will speed up generating the remaining subtrees.

I used a model from a source popular in this contest, without fine-tuning:
https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-32b-awq
This is Deepseek R1 distilled to the Qwen 2.5 architecture with 32 billion weights, with 4-bit AWQ.

Obvious possible improvements:
* Use a better model. The code works with models of any size, just the constant prior_time_per_task needs to be adjusted.
* As of today, LLM inference engines don't fully utilize multiple GPUs, and particuarly when using speculative decoding with a smaller draft model, both models quantized to 4-bit. More than 2x faster token generation should be possible here.
