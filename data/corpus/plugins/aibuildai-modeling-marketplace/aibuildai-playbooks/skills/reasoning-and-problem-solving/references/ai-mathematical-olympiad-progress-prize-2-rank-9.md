# [LB Pub 29 Pvt 28] Enhancing the efficiency of a reasoner using SFT and GRPO [Fast-Math-R1-14B]

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #9
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/571252

First of all, congratulations to all the participants for making it through this long and challenging competition. We’d also like to express our sincere appreciation to the organizers for designing such an interesting competition with a high-quality dataset, and to the Kaggle team for providing the extensive computational resources that made this possible.

Our team identified the main challenge of this competition as the redundancy in the reasoning process of the DeepSeek R1 series models — specifically, their long output token lengths, which made it difficult to solve 50 problems within the 5-hour time limit. To address this, we aimed to fine-tune a distilled version of the R1 model to reason more efficiently.

Our final model scored **29 on public LB and 28 on private LB**.

## Paper
ICML2025 AI for Math Workshop
[A Practical Two-Stage Recipe for Mathematical LLMs: Maximizing Accuracy with SFT and Efficiency with Reinforcement Learning](https://arxiv.org/abs/2507.08267)

## Github
https://github.com/analokmaus/kaggle-aimo2-fast-math-r1

## Model
https://huggingface.co/RabotniKuma/Fast-Math-R1-14B
https://www.kaggle.com/models/analokamus/fast_math_r1_14b/

## Dataset
https://huggingface.co/datasets/RabotniKuma/Fast-Math-R1-SFT
https://huggingface.co/datasets/RabotniKuma/Fast-Math-R1-GRPO

# First stage: intensive SFT using a high-difficulty dataset
## Dataset
[OpenR1 Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k): We randomly sampled 3000 examples where the R1’s trace had more than 12800 tokens and an accuracy of over 50%, along with another 3000 examples where the accuracy ranged between 50% and 75%.
[openr1_hard](https://huggingface.co/datasets/hoanganhpham/openr1_hard):  "~2.5k hard samples from open-r1-math-220k. Samples deemed as hard were unsolvable by r1-distill-32b after 4 tries." (Thank you for the dataset, @andy2709 )
[Light-R1-SFTData](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData): We used the 2nd stage data from Light-R1-SFTData. (Big thanks to LightR1 team, @zouhaosheng )

We merged all the datasets mentioned above, removed duplicates, and selected the correct generation with the shortest token length. For samples in the Light-R1 dataset where ground truth answers were not provided, we extracted and substituted the answers from the R1 traces. As a result, we constructed a **high-difficulty dataset consisting of 7900 problem - R1 trace - answer sets**.

## Training
Based on our prior experiments, we observed that the 14B models demonstrated more stable performance. Therefore, we chose DeepSeek-R1-Distill-Qwen-14B as our starting point. A full-parameter supervised fine-tuning training was conducted on a machine with 8 H200 GPUs, using the SFTTrainer from the trl library.

### Key parameters
- per_device_train_batch_size = 1
- gradient_accumulation_steps = 8
- num_train_epochs = 20 
 - This number is unusually big, but we found that meaningful performance gains only emerged after long long training
- max_seq_length = 24000
- packing = True
- learning_rate = 1e-5
- lr_scheduler_type = cosine
- system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."

Training time: approx. 10 hours (8× H200 GPUs)

## Evaluation
We evaluated the model’s performance using a dataset of 40 problems, consisting of **10 reference problems and 30 from AIME 2025**.
Initially, we generated answers using 16k tokens × 32 prompts. We then applied post-hoc filtering based on token length and the number of prompts to assess performance under various inference conditions.

## Results

| Experiment                    | Token budget | Accuracy (majority@32) | Accuracy (pass@32) | Num of answers collected | Average generation length | Public LB (quantized model) |
|-------------------------------|---------------------|--------------------------|---------------------|--------------------------|---------------------------|-----------|
| DeepSeek-R1-Distill-Qwen-14B  | 16384               | 0.700                    | 0.775               | 21.775                   | 9684                      | 25        |
| DeepSeek-R1-Distill-Qwen-14B  | 12800               | 0.675                    | 0.775               | 16.775                   | 8331                      |           |
| DeepSeek-R1-Distill-Qwen-14B  | 9000                | 0.525                    | 0.600               | 12.500                   | 4725                      |           |
| SFT          | 16384               | 0.750                    | 0.825               | 20.725                   | 10396                     | 23        |
| SFT         | 12800               | 0.725                    | 0.725               | 15.700                   | 7024                      |           |
| SFT        | 9000                | 0.550                    | 0.550               | 11.600                   | 4387                      |           |


Our local validation scores improved remarkably after the first-stage SFT. However, we observed that the Public LB scores tended to be slightly lower.
We believe this was **due to increased reasoning redundancy introduced by SFT, causing many examples to fail to reach a conclusion within the time limit**.
To address this, our next objective was to apply reinforcement learning to encourage the model to reach accurate conclusions using fewer tokens, while maintaining performance.

# Second stage: GRPO for more efficient reasoning
## Dataset
[Light-R1-SFTData](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData): We used the 2nd stage data from Light-R1-SFTData.

## Training
We used the [faster version of trl GRPOTrainer](https://github.com/nhannguyen2709/open-r1) created by @andy2709 (again, thank you so much!).

We used the following reward function:
1. Format reward
In our submission, generation is stopped at the `</think>` tag to save time, therefore we designed the reward to match the pattern `r"^.*?oxed{(.*?)}.*?</think>.*?$"`.
2. Cosine reward (correct [1.0, 0.1], incorrect [-0.1, -1.0], max_len=30000)
Compared to a normal accuracy-based reward, cosine reward applies a continuous penalty to longer correct reasoning traces and shorter incorrect ones.
3. Length reward
Length-based rewards to discourage overthinking and promote token efficiency.
Paper: https://arxiv.org/abs/2501.12599

### Key parameters
- num_generations = 8
- beta = 0.04
- per_device_train_batch_size = 2
- gradient_accumulation_steps = 8
- num_train_epochs = 1 
- max_completion_length = 16384
- learning_rate = 4e-6
- lr_scheduler_type = cosine
- system_prompt = (
        'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
        'You should think step-by-step. Return final answer within \\boxed{{}}.'
    )

Training time: approx. 10 hours (8× H200 GPUs)

## Evaluation
Same as first stage.

## Results

The reward was optimized steadily throughout training, but after step 60, catastrophic shifts occurred, causing a substantial decline in performance. We thus decided to use checkpoints from earlier steps for evaluation.

| Experiment                    | Token budget | Accuracy (majority@32) | Accuracy (pass@32) | Num of answers collected | Average generation length | Public LB (quantized model) |
|-------------------------------|---------------------|--------------------------|---------------------|--------------------------|---------------------------|-----------|
| DeepSeek-R1-Distill-Qwen-14B  | 12800               | 0.675                    | 0.775               | 16.775                   | 8331                      | 25        |
| DeepSeek-R1-Distill-Qwen-14B  | 9000                | 0.525                    | 0.600               | 12.5                     | 4725                      |           |
| SFT         | 12800               | 0.725                    | 0.725               | 15.7                     | 7024                      | 23        |
| SFT         | 9000                | 0.550                    | 0.550               | 11.6                     | 4387                      |           |
| SFT + GRPO (best checkpoint)       | 12800               | **0.725**                | **0.775**           | **18.5**               | 6817                      | **29**    |
| SFT + GRPO (best checkpoint)       | 9000                | **0.625**                | **0.700**           | **15.25**                | 4759                      |           |


GRPO enabled us to train a model that preserved accuracy while significantly improving inference efficiency through shorter token lengths. 

# Submission inference setup
## Quantization
We applied 4-bit quantization using [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).
We observed some degradation in validation performance after quantization, and attempted to recover it through calibration on a math-specific dataset. Unfortunately, this approach did not succeed in preserving the original performance.

## Inference time scheduling
We trained a ModernBERT model to predict the shortest token length of correct R1 traces for each problem in the OpenR1 Math dataset, thereby quantifying problem difficulty.
On the validation set, we observed a moderate correlation between the predicted difficulty and the actual number of tokens generated, as shown in the figure below.



Using this model, we scaled the output token length dynamically during inference. This approach stabilized the Public LB scores and led to an improvement of approx. +1 point (though it could be placebo).

## Misc.
- Prompts: prompt_config0 * 8 + prompt_config1 * 2 (10 prompts in total)
```python
prompt_config0 = dict(
    system=(
        'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
        'You should think step-by-step. Return final answer within \\boxed{{}}, after taking modulo 1000.'
    ),
    prompt='{question}'
)

prompt_config1 = dict(
    system=(
        'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
        'You should think step-by-step. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{{}}.'
    ),
    prompt='{question}'
)
```
- Output token length: 10500 - 13300 (dynamic scaling described above)
- vLLM version 0.7.3 with V1

# What didn’t work
## Constrained decoding
Many studies have explored controlling the reasoning processes of LLMs through constrained decoding methods, such as:

- https://arxiv.org/pdf/2501.18585
- https://arxiv.org/pdf/2412.21187

We experimented extensively with these approaches and tuned various parameters; however, none of them proved effective on our validation dataset. Our hypothesis is that these methods might only be suitable for relatively simple problems.

## Rewritiing the reasoning process and retraining
We also experimented with rewriting the original R1-Qwen model’s reasoning process into a more compact one, following the [SkyThought](https://github.com/NovaSky-AI/SkyThought) approach, and retraining the model via SFT and DPO. Although this method significantly reduced the number of tokens generated before reaching an answer, it also caused a substantial drop in accuracy.

For highly capable LLMs, forcibly altering their natural reasoning process externally might not be an effective strategy, particularly when dealing with complex problem-solving scenarios like ours.
