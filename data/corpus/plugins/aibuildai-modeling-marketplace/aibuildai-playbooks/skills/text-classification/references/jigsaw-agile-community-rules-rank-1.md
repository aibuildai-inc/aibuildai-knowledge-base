# 1st place solution

Competition: jigsaw-agile-community-rules
Rank: #1
Source: https://www.kaggle.com/c/jigsaw-agile-community-rules/writeups/1st-place-solution

Link to code is [here](https://www.kaggle.com/code/wowfattie/notebookc4e4a9206d).


I would like to thank Kaggle and Jigsaw for organizing and hosting this interesting competition. It's an excellent opportunity to explore practical aspects of LLM finetuning.


**Validation strategy**

What we know:
- train.csv contains two rules with comments (bodies).
- test.csv includes six rules, four of them are new.
- The four new rules have already been identified. 

However, to build a strong local validation, we still need to generate realistic comments for the four new rules — a nontrivial task.

I believe the Public LB is a better choice for validation because of the following:
- The host mentioned that the public/private LB split is random, which means the public LB serves as an unbiased estimator of the private LB.
- The total test data size is not small, with the public LB representing about 30% of it. This ensures low variance in score estimation.

Therefore, all models were finetuned online and validated using the public LB.




**Data processing**

This [notebook](https://www.kaggle.com/code/neibyr/lb-0-878-train-on-test-data-qwen2-5-0-5b) provides an excellent baseline for data processing and finetuning. The idea is to use positive and negative examples in the test.csv as training data. The main modifications I made to the data processing pipeline are as follows:
- Used the LLM built-in chat templates.
- I excluded the subreddit column before deduplication, since including it can create unnecessary duplicates across different subreddits. The performance improvement is significant after this change!



**Finetuning**

Initially, I used vanilla transformers + pytorch code, which works great, achieved 0.931 on public LB early with an ensemble of Qwen3-4b-instruct-2507, llama3.2-3b, and phi-4-mini-instruct. However, I couldn’t scale beyond 7B models due to memory limitations. Later, I switched to Unsloth, which is much more memory-efficient — it can easily fit a 14B model within 16GB of GPU memory.
Before finetuning, I upsampled examples from test.csv once to increase their weight, and then finetuned for one epoch. This is equivalent to one epoch for train.csv and two epochs for test.csv. 
Chat templates can introduce unnecessary tokens before and after the the "Yes" or "No" targets, such as "<think>\n\n</think>\n\n" in Qwen3 models, as well as one or more tokens following the "Yes" or "No" targets. By default, finetuning will also apply loss to these extra tokens which is unnecessary and can slow down training. 
To address this, I manually excluded these tokens from the loss computation, ensuring that only the "Yes" or "No" token contributes to the loss. An additional benefit of this step is that it standardizes the training behavior across different models, leading to more consistent convergence speeds and allowing me to use a single set of hyperparameters across models.


**Inference**

Since the last linear layer maps to the whole vocabulary, even after finetuning on the Yes/No tokens, there could still be a non-trival probabilitiy that the probs will be scattered on other tokens. So I constructed a candidate set of first tokens for many variants:
- ["Yes","YES","Y","yes","True"] -> Yes_ids 
- ["No","NO","N","no","False"] -> No_ids
- plus their space-prefixed forms. 

The variants are obtained by local experiments outputting the most frequent topK predicted tokens. 
For scoring, I output probs equivalent to the sigmoid of log-odds of Yes_ids vs No_ids.

Other tricks to speed up inference is:
- Sort test data by length
- Use forward() only. Avoids all the other operations during generation because all we need is the logits of the last token.

**Postprocessing**

For each model’s output, I computed per-rule rankings, normalized the scores to the range [0, 1], and then ensembled based on these normalized values. This postprocessing step provided a consistent improvement, increasing the public LB score from 0.929 to 0.931 in early tests.

**Model performance and final ensemble**

- Larger models generally performed better. 
- Qwen3 is better than the other common LLMs.
- The final ensemble is a weighted average of these six models.

| model  | Public LB | Private LB |
| --- | --- | --- |
| Qwen3-14b | 0.9297 | 0.9239 |
| Qwen2.5-14b | 0.9287 | 0.9232 |
| Qwen3-8b | 0.9272 | 0.9236 |
| Qwen3-4b-instruct-2507 | 0.9258 | 0.9198 |
| llama3.1-8b | 0.9257 | 0.9202 |
| Ettin-400M | 0.8991 | 0.8944 |
| ensemble | 0.9344 | 0.9293 |
