# 3rd Place Solution Report

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #3
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/573314

**Notebook**: https://www.kaggle.com/code/mavicbf/3rd-place-solution-aliev
**Model**: https://www.kaggle.com/models/shelterw/deepseek-r1/Transformers/deepseek-r1-distill-qwen-14b-awq/1
 
First of all, I’d like to thank Aimo and Kaggle for organizing this competition, and all the participants who shared their insights and ideas throughout. Thanks to you, I learned a lot and had an incredible amount of fun during the entire journey.

Although my solution was relatively simple, I ended up securing **3rd** place.
**Public set score: 25/50**
**Private set score: 30/50**

 
## 1.     Model
No additional training, no DPO, GRPO, or anything like that — I just took the most popular model (possibly after the 7B version) used in open solutions — *Deepseek-R1-Distill-Qwen-14B* with AWQ quantization by *ShelterW*.
 
## 2.     Motivation
-       While running the model on the 10 validation problems, I was really annoyed by the fact that during the solving of simple tasks,  where most solution branches usually lead to the same answer, we spend way too much time. I’d really like to dynamically control the moment when further generation becomes pointless. This is especially relevant in cases where 9 out of 10 solutions are already correct, and the last one gets stuck in a loop and keeps generating until the final token.
 
-       I also followed the following intuition - most likely, the closer we are to the beginning of the solution, the more common ground there is between all solution branches. It seems that the probability of hallucinating increases as the generation progresses. Based on my experience manually reviewing the validation problems, the first ~4096 tokens don’t differ much from each other in any fundamental way.
 
## 3.     The CORE part of the solution:
 
So let’s get to the idea – based on the motivation I implemented the following algorithm: 
 
 
1) We start with 5 solution branches and generate the first 4096 tokens for each.  
2) Then we duplicate each of the branches (so we’ll have 10 solutions in the next iteration) and generate another 4096 tokens. 
3) At this point, it's likely that we already have enough solutions to make a decision, so I implemented the following rule:  
     - If after 8192 tokens we already have more than 6 completed solutions (detected using a regular expression for `\boxed{}`) and one of the answers makes up more than 70% of all, we stop the generation. Indeed, the probability that the model will produce a different answer from this point on is quite low, and we can save valuable tokens (especially considering the increasing computational complexity of attention with longer context).  
     - If not, we move to step 4.  
4) Since duplicating every solution would result in 20 branches, which is too much, I randomly select 7 unfinished branches, duplicate them - resulting in 14 branches - and generate the next 4096 tokens. 
5) We then collect all of the solutions and apply a majority vote.



 
Since in the worst case we’d have to recompute the same KV cache (even twice) — it’s important to set the flag `enable_prefix_caching=True` in `vllm.LLM`.
 
This way, we can generate 14 solutions of up to 12k tokens + those 0-3 branches that might have reached an answer at 8k tokens (in fact, even at 4k tokens, the code often remains the same, but I’m omitting that detail due to how rarely such a case occurs and for the sake of simplicity). I didn’t collect full statistics, but based on my observations, this usually takes about 6–7 minutes. Considering that we’ll often get early answers, on average we should be within the time limits.
 
This optimization isn’t without flaws — for example, the solutions will still be somewhat correlated due to the shared prefix. However, it seems this isn’t too critical and partially condensed by different prompts across the branches.
 
Also, if we're running out of time (based on the `cutoff_times` array), we start with 6 branches, duplicate them after 4096 tokens, and stop at 8192 tokens.

All extra details (temperature, prompts and so on) you can check on the notebook.

My score was partly due to luck, however, the core idea seems interesting, and I’d be happy to see it further developed.

I’ll be glad to answer any questions you may have, so feel free to ask!
