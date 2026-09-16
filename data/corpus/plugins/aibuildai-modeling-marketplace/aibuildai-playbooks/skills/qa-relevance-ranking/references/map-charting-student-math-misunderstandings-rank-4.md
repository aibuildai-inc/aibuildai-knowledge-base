# 4th Place Solution (single model pb 0.948 lb 0.951)

Competition: map-charting-student-math-misunderstandings
Rank: #4
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/4th-place-solution-single-model-pb-0-948-lb-0-951

First, I would like to express my sincere gratitude to the organizers and Kaggle for hosting such a great competition. I have personally learned a tremendous amount throughout this competition. In the following section, I will share my solution and the entire score-improvement process.
#  1. Data Preprocessing
- First, I used StratifiedKFold split the data into 5 folds. Then, I selected a fold with a cross-validation score that was slightly below average as the validation set, in order to avoid using folds with scores that were either too high or too low.
- As discussed [here](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/589400)，In train data there are 12 wrong category labels. Namely 12 rows with QuestionId 31778 have MC_Answer 9 and Category True when the correct answer is 6. Therefore, I have applied the same processing method described in the discussion thread during both the training and inference stages.
# 2.Modeling
- Since the training set and the online test set share the same 15 questions, determining whether the selected answer is correct does not require prediction. Therefore, the task can be simplified to only predicting the type of explanation (Correct, Misconception, or Neither in Category), which reduces the number of prediction labels from 65 to 37.
- Based on the provided train.csv, it can be statistically determined that each question has between 4 to 6 candidate targets, which further reduces the target space.
```python
train['candidate_targets'] = train['QuestionId'].map(train.groupby("QuestionId")['target'].apply(set).apply(list))
train['neg_candidate_targets'] = train.apply(lambda x : [y for y in x["candidate_targets"] if x["target"] != y], axis=1)
``` 
While the online test set might present different scenarios—such as potential new misconception Labels for each question—I have observed that samples labeled as "Correct" and "Neither" constitute the majority. Therefore, I intend for the model to focus primarily on the patterns observed in the currently available samples.
- My ranker models was trained with a pointwise approach, where it sees one label at a time in the context window, As discussed [the 1st place solution in "Eedi - Mining Misconceptions in Mathematics"](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution). I configured the training process so that each step processes only one row_id sample. Since each row_id has 4 to 6 different candidate labels, all of these must be included in a single batch, so the actual batch size is 4 to 6.
# 3.Backbone
The experimental models I tested are primarily from the Qwen series. The experimental results for each model are summarized from multi-fold data:
>Qwen3-32B >= Qwen3-Reranker-8B >= QWQ-32B >= Qwen3-14B >= Qwen2.5-Math-7B-Instruct > Qwen2.5-14B-Instruct > Qwen3-8B > Qwen2.5-7B-Instruct

Since Qwen3-Reranker-8B and Qwen2.5-Math-7B-Instruct demonstrated outstanding performance, I decided to use Qwen3-Reranker-8B for iterative optimization, while employing the larger Qwen2.5-Math-72B-Instruct as the teacher model for knowledge distillation. Regarding the choice of the teacher model, due to my limited GPU resources, I did not experiment with many other large-scale models.
# 4.Effective Optimization Process
**1. Example of Prompt_v1 (base prompt)**
```python
<|im_start|>system
You are an expert in detecting grade-school level math misconceptions. The task performs 2 steps: 
1.Assesses whether the explanation contains a misconception. (Correct, Misconception, or Neither)
2.Identifies the specific misconception present, if any.<|im_end|>
<|im_start|>user
Question: What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]
Student's Answer: \( \frac{1}{3} \)
This answer is correct.
Student's Explanation: I think that 1/3 is the answer, as it's the simplest form of 3/9.

Now we judge the student's explanation is Correct, Does this diagnosis is correct? (Yes/No)<|im_end|>
<|im_start|>assistant

```
- Prompt_v1 + Qwen3-Reranker-8B with 1 epoch gave me cv score is 0.9473, public score is 0.944, private score is 0.942
- Prompt_v1 + Qwen3-Reranker-8B + distill with 1 epoch gave me cv score is 0.9480, public score is 0.946, private score is 0.945

**2.Example of Prompt_v2 (base prompt + few shots for target)**
```python
<|im_start|>system
You are an expert in detecting grade-school level math misconceptions. The task performs 2 steps: 
1.Assesses whether the explanation contains a misconception. (Correct, Misconception, or Neither)
2.Identifies the specific misconception present, if any.<|im_end|>
<|im_start|>user
Question: What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]
Student's Answer: \( \frac{1}{3} \)
This answer is correct.
Correct's Explanation Samples:
- Because there are 9 triangles and 3 arent shaded it is 3/9 and they are both multiples of 3 so it's 1/3
- The answer is 3/9 which can be simplified to 1/3
- I think this because there are 9 triangles and 3 are not shaded and 3/9 is 1/3.
- there are 9 triangles shaded and 3 are not shaded=3/9=1/3
- I think this because the answer is three over nine then they can be simplified to one over three.
- There are 9 triangles and 3 is not shaded. 3/9 is a possibility but the question says to put it in its simplest form. 9 divided by 3 is 3 and 3 divided by 3 is one. You cannot simplify it anymore so the answer is 1/3.
--
Neither's Explanation Samples:
- i counted the in-shaded parts then the shaded part then i made them a fraction.
- there are 9 triangles in total and 3 are not shaded. this means that 3 times 9 is 3 and 1 is 9.
- I think this because it is hard of the number
- 1/3. 3/9 shaded, and to find simplest from we divide the top and the bottom by 3.
- Because I counted them and seen the awnser
- I think this is because the fraction of the shape shaded is 3/9 but simplified it is 1/3.
--
Misconception:Incomplete's Explanation Samples:
- there are 9 boxes altogether and 3 of them are not shaded.
- there are 9 triangles andd 3 are not shaded
- Because there is nine triangular spaces and three are unshaded
- There are 9 squares but as you can see theree are 3 not shaded.
- I think this because there are nine triangles and thee are not shaded so it’s three over nine.
- i think this because 3 out of nine triangles are not shaded
--
Misconception:WNB's Explanation Samples:
- 6 is the total and 3 is blank
- because 3 arr white and 6 are blue.
- 6 triangles are shaded so 3 aren’t shaded
- 3 of the triangles are not shaded and 6 are shaded so therefore it is 3/6
- i counted the white ones and the blue ones.
- because 3 are shaded and 6 are not
--
Student's Explanation: I think that 1/3 is the answer, as it's the simplest form of 3/9.

Now we judge the student's explanation exhibited the misconception of WNB, Does this diagnosis is correct? (Yes/No)<|im_end|>
<|im_start|>assistant

```
1).Prompt_v2 + Qwen3-Reranker-8B + distill with 1 epoch gave me cv score is 0.9493, public score is 0.948, private score is 0.946, Regarding the number of few-shot examples in prompt_v2, I randomly sampled 1 to 3 examples per target during training, and 6 examples per target during online inference.
2).Prompt_v2 + Qwen3-Reranker-8B + distill + full data with 1 epoch gave me public score is 0.950, private score is 0.947.
3).Prompt_v2 + Qwen3-Reranker-8B + distill + full data with 2 epoch gave me private score is 0.951, public score is 0.948.(**best single model**), Here's an interesting episode: the model trained on the full dataset for 2 epochs encountered issues during online inference. Since the T4 GPU does not support bfloat16 type in vllm, converting the model to float16 resulted in overflow and produced a large number of NaN values. By this time, the competition was nearing its deadline, leaving me with limited time for adjustments. As an alternative, I used the 2-epoch model from the local validation phase as the backtone, augmented it with validation data, and trained for another 2 epochs to approximate the effect of training on the full dataset for 2 epochs.
4).Prompt_v1 + Qwen3-32B + distill with 1 epoch gave me cv score is 0.9502, After using full data, public score is 0.949, private score is 0.946.
5).I performed an ensemble using Model 3 (my best single model) and Model 4. The ensemble achieved a public score of 0.952 and a private score of 0.948. This was one of my final selected submissions. However, since Model 4, despite having a high cv score and public score, demonstrated relatively lower performance on the private leaderboard, the final result of the ensemble did not surpass that of my best single model.


**3.conclusion**
| model | task | cv | lb | pb | use full | inferrence time |
| --- | --- |
| prompt_v1 + Qwen3-Reranker-8B + 1 epoch  | CausalLM | 0.9473 | 0.94419 | 0.94270 |  | 40 min |
| prompt_v1 + Qwen3-Reranker-8B + distill + 1 epoch  | CausalLM | 0.9480 | 0.94674 | 0.94502 |  | 40 min |
| prompt_v2 + Qwen3-Reranker-8B + distill + 1 epoch  | CausalLM | 0.9493 | 0.94875 | 0.94617 |  | 4 hours |
| prompt_v2 + Qwen3-Reranker-8B + distill + 1 epoch  | CausalLM | 0.9493 | 0.95014 | 0.94737 | ✅ | 4 hours |
| prompt_v2 + Qwen3-Reranker-8B + distill + 2 epoch  | CausalLM | 0.9499 | 0.95120 | 0.94873 | ✅ | 4 hours |
| prompt_v1 + Qwen3-32B-AWQ + distill + 1 epoch  | CausalLM | 0.9502 | 0.94955 | 0.94655 | ✅ | 2.5 hours |
| final seleced model (Best Qwen3-Reranker-8B + Qwen3-32B-AWQ)  | CausalLM | 0.9518 | 0.95211 | 0.94835 | ✅ | 6.5 hours |

#  5.Not Worked for me
- listwist ranker
- chain of thought: I personally spent the most time on Chain-of-Thought (COT), but it ultimately didn't yield any improvements.
- add the most relevant explanation to prompt: I fine-tuned a retrieval model (Qwen3-Emebedding-8B) to find the N most relevant explanations to the current student explanation and incorporated them into the prompt. This method achieved a decent CV score of 0.9493, but get worse score on both the public and private leaderboards.
```python
<|im_start|>system
You are an expert in detecting grade-school level math misconceptions. The task performs 2 steps: 
1.Assesses whether the explanation contains a misconception. (Correct, Misconception, or Neither)
2.Identifies the specific misconception present, if any.<|im_end|>
<|im_start|>user
Question: What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]
Student's Answer: \( \frac{1}{3} \)
This answer is correct.
Top relevant explanations with their tags:
- Because there are 9 triangles and 3 of them were not shaded (Correct)
- because there is 9 triangles and 3 of them were not shaded (Correct)
- there aree 9 triangles in total and 3 of them are not shaded. this gives you 3 / 9. (Correct)
--
Student's Explanation: I think that 1/3 is the answer, as it's the simplest form of 3/9.

Now we judge the student's explanation is Correct, Does this diagnosis is correct? (Yes/No)<|im_end|>
<|im_start|>assistant
```
# 6.Not attempted
Generate more explanations for data augmentation

# 7.Reference
[kaggle_eedi](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview)
[offline-install-vllm-0.10.0](https://www.kaggle.com/code/hiranorm/offline-install-vllm-0-10-0-i-qwenemdding-llama)
[Qwen](https://huggingface.co/Qwen)
[vllm](https://docs.vllm.ai/en/latest/)
