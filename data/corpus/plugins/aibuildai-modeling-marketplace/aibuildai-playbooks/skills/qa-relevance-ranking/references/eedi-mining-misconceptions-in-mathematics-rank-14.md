# 14th Place Solution

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #14
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551492

First of all, I would like to thank the organisers for hosting the competition, kaggle team, and my team mates from AIRI, Skoltech and VeinCV
[rolf110](https://www.kaggle.com/rolf110)
[danyaivanov](https://www.kaggle.com/danyaivanov)
Shout-out to [pipmos](https://www.kaggle.com/pipmos)!

## Summary

### Overall Inference Pipeline
[Inference Pipeline]

Our solution:
- `Qwen2.5-14B` retriever to obtain top-25 misconceptions
- `Qwen2.5-32B` reranker used to score each of the misconceptions received by retriever. We do it iteratively, by passing one misconception at a time to obtain a probability of it being the one that leads to the incorrect answer. 

The whole notebook runs approximately 7 hours. We didn't use any upper-level frameworks such as vllm to run the inference pipeline.

All models were fine-tuned using the competition data, consisting of 4370 (question, construct, correct answer, incorrect answer) quadruplets, +   13921 synthetic (question, incorrect answer, correct answer, incorrect answer) samples generated using gpt-4o.

## Description

### Synthetic Data

- For each `MisconceptionId` in the given database, we generated new (`QuestionText`, `ConstructName`, `Correct Answer`, `Incorrect Answer`) samples, ensuring at least 7 examples is presented for each `MisconceptionId` in the final real+synthetic dataset. We then filtered those examples that have a full-match in the `QuestionText` (approximately 500 samples removed).
- To provide some guidance to the LLM, we passed a reference sample from the train dataset corresponding to the closest misconceptions for the one considered for generation. We used `SFR-Embedding-2_R` embeddings of `MisconceptionName` to obtain "closest" misconceptions. We ensured that each reference is different between different generations for a particular `MisconceptionId`.
- To construct the prompt, we followed the ideas from [here](https://arxiv.org/pdf/2403.04706). The template of the `prompt` we passed to gpt-4o:

```python
'''Please act as a professional math tutor.
Your goal is to create high quality math problems to help students learn math.
You will be given a misconception. Please create a multiple choice math question with two options: one correct answer and one distractor (incorrect answer). The distractor should be based on the Misconception. 
Follow the instructions below.
To achieve the goal, you have four tasks:
1. Please generate a brief and formal description of the concept or skill being tested in the question. DO NOT mention or refer to the Misconception in the description.
2. Create a realistic and contextually appropriate math question that aligns with the description. Provide two options: the correct answer and a distractor (incorrect answer). The distractor should be a plausible answer that a student might choose if they hold the Misconception.
3. Check the question by solving it step-by-step to find out if it adheres to all principles.
4. Modify the created question and options according to your checking comment to ensure it is of high quality.
You have the following principles to guide you:
1. Ensure the question is realistic, natural, and contextually appropriate, adhering to common sense and fundamental mathematical principles.
2. Ensure the question asks for only one specific thing and is clearly stated.
3. Ensure your student can answer the question correctly by using only the information provided in the question. If visual data is needed (e.g. plot, histogram, bar, etc.), describe it's content explicitly in text.
4. Ensure the distractor is directly related to the question, stems from the Misconception, and is a plausible answer that a student might select if they have that Misconception.
5. If the created question already follows these principles upon verification, keep it without modification.

Your output should be in the following format:
DESCRIPTION: <Brief and formal description of the concept or skill being tested>
QUESTION: <Your created question>
CORRECT ANSWER: <Correct answer to the question>
DISTRACTOR: <Distractor (incorrect answer) to the question based on the provided Misconception>
VERIFICATION AND MODIFICATION: <Solve the question step-by-step and modify it to follow all principles>
FINAL QUESTION: <Your final created question>
FINAL CORRECT ANSWER: <Your final correct answer>
FINAL DISTRACTOR: <Your final distractor (incorrect answer)>

Here is an example of the final result corresponding to Misconception "{misconception}". Use it for your reference, but ensure the student can answer your created question without the given example:
DESCRIPTION: {description}
FINAL QUESTION: {question}
FINAL CORRECT ANSWER: {correct_answer}
FINAL INCORRECT ANSWER: {incorrect_answer}
'''
```

We manually checked some of the generated examples, and they were pretty good and close to the real data.
Other staff we found important for generation:
- **verification and modification** stage inside the prompt helped to fix some rare errors and obtain good questions and valid answers.
- When specifying a reference, it is important to add "ensure the student can answer your created question without the given example" to the prompt. Otherwise in some rare cases the model didn't specify some important information in the generated question because it was stored in the reference and the model probably thought that it will be also given to a student. 
- The only thing we couldn't entirely fix is that in rare cases (approx. 3-4% of the data) LLM generated quite a strong hint for the correct misconception inside the description (`ConstructName`) field; but we found that such cases were also observed in the competition data, so we decided to keep it as is.

### Validation

We used 20% of the competition data for validation, which we additionally split on 2 subsets:
  - **out-of-distribution (OOD):** no intersection for train/test between `MisconceptionId` and stratification based on the number of misconceptions per question.
  - **in-distribution (ID):** no intersection for train/test between `QuestionId` and stratification based on the number of misconceptions per question.

We used the **ID** test to account for how the model works on the misconceptions it knows, while the **OOD** test was used to account for how the model generalizes to the unseen misconceptions. However, we mainly tracked the **OOD** data because it was well-aligned with the lb and accounted for the most complex scenario.

### Retriever

To obtain fine-grained hard negatives, we first fine-tuned `Qwen2.5-14B` on real+synthetic data. Then we repeat the training with the exact same setup, but using the hard-negatives extracted using our trained `Qwen2.5-14B`. After each epoch, we collect new hard negatives using the latest checkpoint.

#### Training
- negatives_range & num_negatives: 100 & 8
- qlora config: rank 32, alpha 64
- loss: InfoNCE Loss
- mask_token_probability: 0.1 (applied only to the query)
- lr: 5e-5
- epochs: 3

We also utilized in-batch negative sampling during training. To mitigate the problem of treating negatives which appear to be the same misconceptions as the positive ones, we applied dynamic masking by setting the similarities of such misconceptions to `-inf` before calculating the loss for each sample in the batch.

#### Score

| Data | Recall@25 | MAP@25 |
| --- | --- | --- |
| CV OOD | 0.88 | 0.42 |
| Public LB | None | 0.438 |
| Private LB | None | 0.434 |

### Reranker

Judging from the experiments provided in the paper [Novice Learner and Expert Tutor](https://arxiv.org/abs/2310.02439) and some research done by other participants during the competition, we had come up with the idea to pass misconceptions list as an additional input to better guide the reranker model. 

We didn't want the model to output the misconception prediction as a text because it was not very reliable as we thought, and the multi-classification problem didn't fit here either. Finally, we have decided to solve a binary classification problem for each misconception separately:
1. First, we pass the question, construct, correct answer, incorrect answer, and top-25 misconceptions as a list received by retriever through the backbone and store `past_key_values`. This helps to save a lot of runtime because we only need to compute it once for the given sample.
2. Then we form a batch of misconceptions taken from the list, and pass it to the backbone where each element gets the same `past_key_values` from the previous step as cache. We use a batch size of 5, so 5 iterations is required for the whole list of misconceptions.
3. We take the final logits and feed them to the classification head trained from scratch. This head maps from the logits space to a single score indicating whether the misconception leads to the incorrect answer or no.
4. We sort the misconceptions by predicted scores.

#### Training Dataset

Given the (question, incorrect answer) pair, we create 24 `label = 0` and 1 `label = 1` samples for training. The main difference between them is the way we construct the "list of possible misconceptions":
1. First, we identify the top 50 negative misconceptions using our trained retriever.
2. From these top 50 misconceptions, we randomly select 24 negatives as a subset for further sampling.
3. For each of the negatives sampled at the 2nd step, we add 28 random negatives, the specific negative being evaluated, and the correct positive misconception tied to the (question, incorrect answer) pair. This results in 24 `label = 0` samples per pair.
4. When the model is queried with a positive misconception (`label = 1`), we include 29 randomly selected negatives from the top 50 to create a set of 30 misconceptions we pass to the model.

To mitigate the positional bias caused by the retriever, we randomly shuffle the list for every input sample.

Example of the input prompt:

```python
Question: {Question}
Brief description of the concept or skill being tested in the Question: {ConstructName}
Correct Answer: {Correct Answer}
Incorrect Answer: {Incorrect Answer}
List of possible misconceptions:
1. {Misconception_1}
...
30. {Misconception_30}
Misconception: {n}. {Misconception_n}
Does this Misconception lead to the Incorrect Answer? 
```

To account for high imbalance between 0/1 classes, we oversampled positive class 4 times.

#### Training

We used [h2o-studio](https://docs.h2o.ai/h2o-llmstudio/) framework to train our model.

- loss: cross-entropy loss with 0.05 label smoothing to account for possible errors introduced by synthetic data
- qlora config: rank 16, alpha 32
- lr & head lr: 2e-5 & 1e-5
- epoch: 1
- mask token probability: 0.1

#### Score

| Data | MAP@25 |
| --- | --- |
| CV OOD | 0.571 |
| Public LB | 0.589 |
| Private LB | 0.558 |

### What did not work
- Re-write positive misconceptions with gpt4o-mini to improve oversampling procedure
- Add other incorrect answer choices to the prompt

### Public Data

reranker adapter and classification head weights: https://www.kaggle.com/models/andreygalichin/qwen2.5-32b-r32-a16-synth-merge-smooth-full/
