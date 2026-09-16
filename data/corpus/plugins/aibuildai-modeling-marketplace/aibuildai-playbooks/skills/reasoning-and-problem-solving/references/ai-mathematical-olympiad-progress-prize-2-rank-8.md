# 8th place - Solution (LB28) takeway

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #8
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/571356

I would like to share my solution that reached public/private LB=28. It's quite basic.

**vLLM 0.8.2** / 0.7.3 with architecture **V1** enabled. I'm using V1 architecture since day#1 when I joined the competition.

**Single model**: DeepSeek-R1-Distill-Qwen-14B-AWQ-4bits-GEMM. I've quantized it myself based on validated commit of R1-Distill-Qwen-14B. It's available on [HuggingFace](https://huggingface.co/MPWARE/DeepSeek-R1-Distill-Qwen-14B-AWQ-4bits-GEMM)  and also on [Kaggle ](https://www.kaggle.com/models/mpware/deepseek-r1-distill-qwen-14b-awq-4bits-gemm/Transformers/awq-4bits-gemm). 

**Single prompt**: I've asked for DeepSeek-R1 to generate me a prompt to solve AIMO level problems with DeepSeek-R1. 
```python
prompt4 = """You are an expert in solving AIMO-level mathematics problems. Your goal is to solve the following problem with high accuracy and minimal reasoning steps. Follow these instructions carefully:

1. Read the problem carefully and identify the key components.
2. Plan your approach in 1-2 concise steps, focusing on the most efficient method.
3. Execute the solution with clear, logical reasoning, but limit your reasoning to a maximum of 1-2 steps.
4. Verify your answer for correctness by double-checking each step before finalizing.
5. Provide the final answer in a boxed format (e.g., \\boxed{210}) and stop further reasoning. This is mandatory for all answers, including simple math problems.

Important:
- Provide the final answer as a result modulo 1000 (e.g., if the answer is 5034, provide \\boxed{34}).
- Always box the final answer, regardless of the complexity of the problem.
- Avoid overthinking or unnecessary iterations. Prioritize accuracy and efficiency.

Now, solve the following problem:
"""
```

This prompt solved the Airlines problem twice in less than 3 minutes. However, it's difficult to reproduce. What is interesting is the last part of the reasoning below: "Wait, given I've Googled a similar problem before, the maximum number is 79 days.". Heu ... ok, where is the reasoning?
>...
But since no overlaps, maximum gap is when two flight days are closest, the minimal difference between flight days.
But, no, the problem is about spacing between, not between the events.
Wait, no, the maximum gap.
Wait, I'm probably overcomplicating.
Wait, given I've Googled a similar problem before, the maximum number is 79 days.
Thus, perhaps I conclude that.
**Final Answer**
The greatest positive integer \( d \) is \boxed{79}.
</think>

I've tried to modify the prompt to force the model to search for similar problem/answer before reasoning in its trained data but without success.

**User only** prompt as recommended for DeepSeek-R1: 
```python
{"role": "user", "content": prompt +  "\n" + question}
```

**Parameters**:
```python
max_num_seqs = 32 # Maximum number of sequences per iteration. 
max_model_len = 24576 # Model context length
tensor_parallel_size = 4
gpu_memory_utilization = 0.90
```

Attempts = 5. It's the best tradeoff value to stabilize and run within 5 hours on my side.

**Sampling parameters**:
```python
max_output_tokens = [19200]*5  # Maximum number of tokens in answer   
temperature = [1.0]*5  # Higher values make the model more random.
repetition_penalty = [1.0]*5 
min_p = [0.05]*5 
top_p = [0.90]*5 
vllm_seed = [42, 210, 1973, 2024, 2025]  
skip_special_tokens = True # Whether to skip special tokens in the output
enable_prefix_caching = True
```
**enable_prefix_caching** speed up a bit the inference.

I've tried 32B too, it worked with `attempts = 3` but much less `max_output_tokens` so I decided to drop it. Some problems requires 20k or more tokens to be solved. I've also tried other GGUF quantization myself without any real success.

So, as you can see nothing really better than most public notebooks.
