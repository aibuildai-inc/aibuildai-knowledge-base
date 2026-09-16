# 15th Place Solution Writeup

Competition: map-charting-student-math-misunderstandings
Rank: #15
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/15th-place-solution-writeup

# 15th Place Solution Writeup

First of all, thanks to my team members: @jjinho @ky8ori @masasato1999 @lizhecheng we work together to achieve the 15th rank on the private leaderboard. Even though the shake down made us miss the gold medal, we have a wonderful journey in this great competition.

## lizhecheng and ky8ori part

We use 8 x A100 80GB GPUs for parallel experiments.

During the first period, we mainly focus on following models:

1. Qwen/Qwen3-14B
2. Qwen/Qwen3-8B
3. Qwen/Qwen3-4B and Qwen/Qwen3-4B-Instruct-2507
4. google/gemma-2-9b-it
5. deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
6. deepseek-ai/deepseek-math-7b-instruct

We use five fold cross validation and LoRA to train each model with 8 different parameters settings:

``rank=(32 32 32 32 64 64 64 64)``

``learning rate=(1.75e-4 2.0e-4 2.25e-4 2.5e-4 1.75e-4 2.0e-4 2.25e-4 2.5e-4)``

We constantly set lora_alpha to 32, training epochs to 2, lr_scheduler to 'linear' and we mainly use the simplest input prompt shown below:

```
def format_input(row):
    x = "Yes" if row['is_correct'] else "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Student Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}\n"
    )
```

Our results shown that:

- Qwen3 series models gave the best results, and larger models are having both higher scores on CV and LB.
- gemma2-9b-it is very hard to train, usually the CV are pretty lower with issue such as gradient exploding etc, which we decided not to move on with this model more
- deepseek-math-7b-instruct and Qwen2.5 series models are not as good as Qwen3 series, which we also dropped.

Additional training methods we used:

- Training with full classes, this is the most original code, which gives fair results.
- Training with fewer classes, which means we delete the prefix True or False for each label, since we all know whether the answer is correct, this gives some improvements on both CV and LB, maybe around 0.001.
- Training with masked loss (got this idea from M sato), since we found that only certain labels appear in the certain questions, so we do not calculate the loss for the prediction probabilities and corresponding to the labels actually would not appear for the certain questions. This method improves the CV score, but the LB score does not appear to improve.
- Training with advanced prompts such as adding the options into the input or extending the input questions to be more clear, this method improves the CV score, but the LB score decreases. Also because the input is much longer, we need to set the inference length to 300 or even longer, so we drop this method.
- Training with other different losses: focal loss, bootstrapping loss, labeling smoothing. However, none of them give consistent CV boost on all five folds validation. We tried a 5-fold CV ensemble train with regular CE loss, but replacing some folds where bootstrapping loss gave better CV scores. This approach boosted lb score by 0.001, which could be just noise.
- Training with soft labels. With embedding models and manual inspections, we grouped student misconceptions that have the same semantic meaning. For each group, we calculate soft labels based on distribution of its ground truth misconception labels. Again, we saw CV improvement on some folds, but not across all folds. This could be due to inadequate semantic grouping.
- Training with shared labels. See 'JJ part'.

Post-processing method that we tried:

After doing eda with the oof predictions, we found that since the category Correct:NA and Neither:NA have a large portion in the training and test data, and they have a lot of errors with the prediction of whether put one category in the first position or the second position.

We extracted the data belonging to these two categories from the original training dataset, and trained a binary classification post-processing model to try to verify and switch the place when these two categories appear at the first and second position.

The MAP score for {1, 0.5} could reach 0.965+, but after adding to one previous submission, the LB actually decreased, which does not make sense, but we have to drop this idea.

Final submissions:

For our final submission, since we decided to focus on using a single model that fully trained on the given training dataset, we mainly retrained the Qwen/Qwen3-14B model using worked methods mentioned above with the parameter setting that achieves higher CV score across 8 different settings. Also, we trained microsoft/phi-4 and microsoft/Phi-4-reasoning since our teammates found phi-4 series models work well in this competition.

During the inference, we only care about the output probabilities that their corresponding labels would appear for the certain question ids. We mainly use the same weight for all full training models to ensure not overfitting and the diversity, our final submission mainly contains 5-6 models. We do have more than 15+ submissions with a private score 0.948+, unfortunately, we did not choose them.

We will release all the models trained in this competition, with their performance scores in the model description.

## t fuku part

I use RTX5090 GPU for my experiments.

#### 1. Models

Early in the competition, I tested 62 models (size * model) and many type of prompts.
The CV rankings when training with unified prompts were as follows:

- Public scores are measured after completing full training.
- lr: 2e-4
- scheduler:  cosine-scheduler
- epoch: 3

| Model                         | Local CV | Public LB | LoRA Rank | LoRA Alpha |
| :---------------------------- | :------: | :-------: | :-------: | :--------: |
| Phi-4                         |  0.9483  |   0.948   |    64     |    128     |
| Phi-4-reasoning-plus          |  0.9480  |   0.948   |    64     |    128     |
| Qwen3-32B-4bit-quantization   |  0.9483  |   0.946   |    64     |    128     |
| Qwen2.5-32B-4but-quantization |  0.9485  |   0.946   |    64     |    128     |

We also tested several smaller models, and the 32B model achieved the same score as the 14B model on LB. However, when used in an ensemble, the 32B model yielded higher LB scores.

#### 2. Prompts

Initially, we presented prompts listing all options for the LLM to choose from. However, as the prompts grew longer, both training and inference times increased.
In @cdeotte's [public code](https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945), a much simpler prompt was used, and employing that prompt yielded better results than providing all options within the prompt. While reviewing various public codes, I noticed most prompts did not utilize tags. I proceeded to build the model training and prompts while incorporating tags.
I tried adding phrases like “You are a mathematics expert. Please perform the task of identifying student misconceptions” at the beginning of the prompt, but this didn't significantly improve results. However, prompts containing “Let me analyze this mathematical misconception...” within the <think> tag substantially boosted my CV. My best prompt is as follows. (phi-4)

```
prompt = (
    "<|user|>\n"
    f"[Mathematical Misconception Analysis Task]\n\n"
    f"Question: {row['QuestionText']}\n"
    f"Answer: {row['MC_Answer']}\n"
    f"Correct?: {status}\n"
    f"Explanation: {row['StudentExplanation']}\n"
    "<|end|>\n"
    "<|assistant|>\n"
    "<think>\n"
    "Let me analyze this mathematical misconception...\n"
    "</think>\n\n"
)
```
#### 3. Ensemble

A point for reflection is that I overestimated LB and excessively adjusted the ensemble weights. By ensembleing all models: phi-4, phi-4-reasoning-plus, qwen3-32b, and qwen3-14b, with weights set to 1 using the program below, there was potential to achieve a Private score of 0.948 and aim for a gold medal.

#### 4. Code

The model training and inference programs are saved below.
<https://github.com/T3pp31/map-charting-student-math-misunderstandings>

#### 5. Data
I also found some interesting things in the data. For specific problem numbers, Wrong_Fraction was actually Wrong_fraction. I wonder if this affected the scores. I thought that by standardizing Wrong_fraction to Wrong_Fraction for training, the model would have one less thing to classify, thereby improving accuracy. However, it didn't work out.

## M Sato part

We believed that using the same problems for both training and test data would be key to this competition, so we tried several approaches.

#### 1.Reproduction of options given to students (preprocessing)

In this competition, students were actually given four options: a, b, c, and d. Based on StudentExplanation, we manually mapped each option to a corresponding pair of a, b, c, or d. → This had little effect on either cv or lb.

#### 2.Masked Loss model（train）

During model training, we dynamically selected labels for loss calculation based on training IDs to focus on label learning for each problem. → This improved cv but had no effect on lb.

#### 3.Misunderstanding Filtering Based on QuestionID in Training Data (Post-Processing)

The model's generalized output was filtered based on QuestionId labels. → This proved effective for both cross-validation and label balancing

## JJ part

During this competition I had access to 4090 x 2 and towards the end a RTX 6000 Pro Blackwell Workstation Edition for my experiments.

I also performed multiple experiments with various models training on LoRA but unlike my team mates, I had problems with training Qwen3 series model in that training and inference on my local machine would work well, but inference on Kaggle’s infrastructure would result in vastly inferior results. This was despite training using quantization to mimic the Kaggle environment. However the Phi4 series of models did not give me those issues so I stuck with them. I also tried training Gemma2/3 series of models but the CV was also worse than Phi4.

I used five fold cross validation. Set LoRA to 64/128 (experimented with combinations of alpha/rank ranging from 8 to 128) and learning rate 2e-4 (experimented with ranges from 5e-5 to 5e-4) and 3 epochs (experimented with ranges from 2 to 4).

When it came to prompt also started with the simplest prompt, but later switched to model appropriate prompts.

While the models were trained on the classification task, over the course of the competition I also changed the way in which I set up the labels. Hypothesis was that removing True_/False_ prefixes would help the model focus on less classes (since that information was already given to us). Additionally later during error analysis, we noticed that the model was having the highest error rates between the Neither/Correct type labels. One observation was that Neither type language was fairly constant throughout all questions, while Correct type language was very question specific. Therefore I created labels in which Neither was shared across all questions (e.g. “Q:Neither:NA”) while the correct labels were questions specific (e.g. “31772:Correct:NA”). This change gave the best CV (0.9496). Other variations on that labeling scheme (e.g. sharing Correct:NA across all questions and making Neither:NA question specific) were also tried, but did not achieve as high CV (0.949 in the inverted label case), but were added to the ensemble in an attempt to obtain increased diversity.

## Conclusion
During ensemble training, we combined various models and adjusted their weights, achieving a score of 0.952 on LB. What we lacked this time was proper experiment management, cross validation skills with how CV was constructed. We thank everyone who competed with us until the very end in the competition.
