# 7th place solution (pure luck)

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #7
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/572760

I was surprised that my "just give it a try" submission made within a week ended up in the gold zone for this competition. Though this is pure luck (public 23 -> private 29), I would like to share what I learned from this competition.


## The task and GPU resources

In this competition, we were given L4x4 GPUs for inference. The good is we didn’t have to suffer with the ancient T4s in 2025. The bad is L4s aren't ideal for LLM decoding either—they only provide 300GB/s on paper memory bandwidth each, totaling 1200GB/s with 4 GPUs.

Long-context decoding (for reasoning) and batched decoding (for majority voting) were both important for this task. Under this kind of inference settings, as context length grows, KV cache can become the main bottleneck (larger than the model weights), further increasing the demand on GPU bandwidth.

## Inference engines

I tested the inference speeds of LMDeploy, SGLang, and vLLM. With 4-bit quantized weights and fp16 KV cache, the speed ranking was: LMDeploy > SGLang > vLLM v1 >> vLLM.
LMDeploy supports int8/int4 KV cache quantization, while SGLang and vLLM offer FP8 KV cache quantization. Using KV cache quantization enables larger decoding batch size within the same time budget.
Unfortunately, int4 KV cache quantization caused too much accuracy loss, so I ended up choosing LMDeploy with int8 KV cache quantization for inference.

## Submission strategy

Before submission, I profiled inference time of decoding to 16k tokens (worst case scenario) under different batch sizes. During the actual submission, I tracked the actual inference time for each question, then add `planned time - actual time` into the time budget for the next question. Batch size for the next question was selected on that new time budget and my earlier profiling results.

This is a fairly conservative strategy, using smaller batch sizes for first received questions. And I didn’t profile for longer max decoding lengths, so there’s room for improvement.

The inference notebook is public [here](https://www.kaggle.com/code/tascj0/aimo2-submit?scriptVersionId=231101822). Simply a rushed implementation using `casperhansen/deepseek-r1-distill-qwen-14b-awq` and my submission strategy. I haven’t done enough testing so there might be bugs.
