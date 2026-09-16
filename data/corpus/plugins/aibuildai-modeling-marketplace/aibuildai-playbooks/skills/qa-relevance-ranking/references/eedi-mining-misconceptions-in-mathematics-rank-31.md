# 31st place solution

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #31
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/553317

Hopefully it's not too late to post my solution🙃

I am so excited that I got my first silver medal. 
Since last year, I have started participating in kaggle competitions in my spare time of work. As a beginner, sometimes I feel it really hard to construct a runnable pipeline for submission, but thanks to the kaggle community users for their selfless sharing, I learned a lot of tricks and techniques, and many useful insights to solve the given task.  Also, I would like to thank Kaggle and EEDI for hosting such an interesting competition. 
# Solution
Like many other players, my solution is a 2-stage approach:
## Retrieve
The retriever is a Qwen-14b-awq from [https://www.kaggle.com/code/anhvth226/eedi-11-21-14b](https://www.kaggle.com/code/anhvth226/eedi-11-21-14b). For each query, select the top 25 likely misconceptions.
## Rerank
Given a question, a correct answer, a wrong answer and a list of possible misconceptions (9 candidates each query) from the previous stage, let Qwen2.5-32b-awq predict the most likely misconception index. 
Please refer to the hot public notebook [https://www.kaggle.com/code/jagatkiran/qwen14b-retrieval-qwen32b-logits-processor-zoo](https://www.kaggle.com/code/jagatkiran/qwen14b-retrieval-qwen32b-logits-processor-zoo). 
And my infer notebook: [https://www.kaggle.com/code/doublezeta/notebook0d961c3b94](https://www.kaggle.com/code/doublezeta/notebook0d961c3b94)
# What's new in my solution
## Lora
I finetuned (lora) a Qwen2.5-32b-awq reranker via trl.SFTTrainer. Here are my hyper parameters
```
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none"
)
training_args = SFTConfig(
    packing=False,
    max_seq_length=tokenizer.model_max_length,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.001,
    gradient_accumulation_steps=3,
    gradient_checkpointing=True,
    fp16=True,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
)
```

My demo training notebook: [https://www.kaggle.com/code/doublezeta/train-reranker](https://www.kaggle.com/code/doublezeta/train-reranker) Actually I did the training in Colab.

I construct my SFT training dataset by
1. run the retrieval (stage-1) on the competition train dataset, obtaining 25 candidates misconceptions for each question-option pair.
2. From the original 25 candidates, 8 wrong misconceptions are selected and, together with the correct misconception, form a sample with 9 candidates. The label is the index of the correct misconception. Since there are 25 candidates, I can generate 3 samples for each question-option pair.

## Train Data Augmentation
The competition overview mentions "The goal is to create a model that not only aligns with known misconceptions but also generalizes to new, emerging misconceptions". 
There are ~900 misconceptions not appeared in the train set. I suppose the private LB would focus more on the unseen misconceptions.
Therefore I slightly adjust my strategy for constructing the SFT training set. Among the 8 wrong candidates of each sample, 4 are from the retrieval result, and 4 are from the 900 unseen misconceptions. 
I believe increasing the ratio of the unseen misconceptions can help the LLM enhance the ability of reasoning upon these misconceptions.

It indeed brought me sort of improvement on the private board.
|  | Public Score | Private Score |
| --- | --- | --- |
| with Augmentation | 0.539 | 0.531 |
| without Augmentation | 0.554 | 0.514 |

Notebook to build SFT training set : [https://www.kaggle.com/code/doublezeta/train-data-sft](https://www.kaggle.com/code/doublezeta/train-data-sft)

I found similar insight in the 3rd place solution: [https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551498](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551498), maybe I should push it further🥂

# Tried but not worked
I noticed that the reranker just select the most likely misconception without changing the order of the rest candidates. 
Inspired by 
[https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519) 
[https://openreview.net/pdf?id=vhLAb1dpIw](https://openreview.net/pdf?id=vhLAb1dpIw), 
I used LLM as a pointwise scorer by getting the **Yes** token's logprob.  
I finetuned (lora) a Qwen2.5-32b-awq. During inference, I prompt it to predict "Yes" or "No" for given a question, a correct answer, an incorrect answer, and a possible misconception.
But the Public LB is poor (0.3+), I guess it's because
- I did not correctly finetune the model. I forgot to train on completions only.
- Unlike Pointwise inference, Listwise inference allows the LLM to compare between different candidates.

# Some experience
- Speed up vLLM inference by setting `enable_prefix_caching=True`
- Correctly manage your python environment
As a beginner, I spent quite a lot of time on figuring out the dependencies among different packages (transformers, torch, vllm ...). 
For example, I was struggling on running vLLM with LoraRank >= 64 on T4 GPU until I found such links
[https://github.com/vllm-project/vllm/issues/5199](https://github.com/vllm-project/vllm/issues/5199)
[https://github.com/vllm-project/vllm/issues/3934](https://github.com/vllm-project/vllm/issues/3934)
For convenience I fixed the dependencies I need in a notebook: [https://www.kaggle.com/code/doublezeta/eedi-dependencies](https://www.kaggle.com/code/doublezeta/eedi-dependencies)
