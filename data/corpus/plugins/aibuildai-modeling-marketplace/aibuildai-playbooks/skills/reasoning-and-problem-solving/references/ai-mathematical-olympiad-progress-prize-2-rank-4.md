# 4th place solution

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #4
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/573671

4th Place Solution – AIMO Reasoning Challenge

Summary

This solution is based on a single quantized 14B DeepSeek model, using a simple, fixed inference strategy without time management or system prompts. I focused on robustness over leaderboard tuning, avoiding overfitting by validating locally on a custom set of math problems. Despite its simplicity, the setup performed surprisingly well on the private leaderboard.


1. Model

I used the DeepSeek-R1-Distill-Qwen-14B-AWQ-4bits model, specifically the AWQ-quantized version by casperhansen, available on Hugging Face and Kaggle:

https://www.kaggle.com/models/konstantinboyko/qwen-14b-awq-casperhansen

Like many others, I found the 14B model to be a good compromise between the quality of the larger DeepSeek 32B and the speed of the 7B. The distilled version made it feasible to run efficiently on Kaggle while still delivering strong reasoning performance.


2. Motivation

My main motivation was to keep things simple and avoid overfitting to the public leaderboard. I took two specific measures to support this:

1.	Local validation was done using a set of 50 handpicked problems — a mix of AIME, AIME 2025, and the provided reference questions. For each model or parameter change, I ran multiple validation passes, shuffling the questions each time. This gave me a more stable and realistic picture of model performance than relying solely on the public leaderboard.

2.	No time management was used during generation. While many strong solutions relied on dynamic token allocation, I chose a fixed strategy — splitting tokens and sequences equally across the 50 problems. Since questions were served in random order, the effectiveness of time management became extremely sensitive to question sequence. That likely explains some of the large fluctuations seen in public notebook results.

Instead of tuning for leaderboard score, I used the public set only to help select values for max_tokens and number of sequences — prioritizing throughput over score. This approach was intentionally conservative and focused on robustness. Given that, the final result was unexpected — but not unusual in competitions like this, where significant leaderboard shakeups are common.

3. Inference

Inference was performed using vLLM 0.7.3 with FlashInfer 0.2.2 enabled.

I used a single prompt. Following DeepSeek’s Usage Recommendations, I did not use a system prompt. Instead, I appended the instruction directly to the user message. This seemed to improve output quality on my validation set, so I kept this approach throughout.

I also tested different sampling settings:
	•	temperature between 0.6 and 0.8
	•	max_p = 0.95

However, I didn’t see clear improvements compared to the defaults used in most public notebooks. In the end, I stuck with the common settings:
temperature = 1.0, min_p = 0.01, and top_p = 1.0.



4. Final Notes & Acknowledgements

I didn’t apply any postprocessing beyond selecting the best model variant based on token and sequence throughput — not leaderboard score. I chose the configuration that generated the most content across the 50 validation problems, which gave more consistent results.

A few acknowledgements:
	•	DeepSeek – for releasing the R1 reasoning model mid-competition. It was a game-changer and quickly became central to nearly all top solutions.
	•	Md Boktiar Mahbub Murad and others for publicly sharing DeepSeek-based notebooks that helped shake up the leaderboard and inspired many.
	•	Casper Hansen – for providing the AWQ-quantized 14B model that worked smoothly with vLLM and made this solution possible.

Thanks

Thanks again to Aimo and Kaggle for organizing this competition. I learned a lot and really appreciated the chance to explore the problem. Also, thanks to the Kaggle community — the shared discussions and ideas made this a genuinely enjoyable experience.

Although my solution was extremely simple, I ended up placing 4th.
Public score: 25/50
Private score: 29/50

Notebook: https://www.kaggle.com/code/sorenravn/aimo-2-4th-place
Model: https://www.kaggle.com/models/konstantinboyko/qwen-14b-awq-casperhansen
