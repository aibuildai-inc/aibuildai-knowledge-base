# 5th place solution

Competition: map-charting-student-math-misunderstandings
Rank: #5
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/5th-place-solution

### Introduction

Firstly I would like to express my sincere gratitude to the authors of the brilliant public notebooks that were incredibly helpful in this competition. Without your shared insights, achieving a gold medal would not have been possible. Particularly, I found the following five notebooks especially helpful. Thank you very much.

[1] [Gemma2-9B-it - CV 0.945](https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945)
[2] [DeepSeekMath-7B LB:0.944](https://www.kaggle.com/code/jaytonde/deepseekmath-7b-lb-0-944)
[3] [[LB 0.942][Train] FullFT Qwen3-8B by SFTTrainer](https://www.kaggle.com/code/sinchir0/lb-0-942-train-fullft-qwen3-8b-by-sfttrainer)
[4] [[LB 0.942][Infer] FullFT Qwen3-8B by SFTTrainer](https://www.kaggle.com/code/sinchir0/lb-0-942-infer-fullft-qwen3-8b-by-sfttrainer)
[5] [Some considerations. Fine tuning Gemma2-2b-it.](https://www.kaggle.com/code/ueueue/some-considerations-fine-tuning-gemma2-2b-it)

-----

### Strategy Overview: Emphasis on Diversity

My primary strategy for this competition was to ensure **diversity** in training methodologies.

The baseline model was already able to correctly predict about 94% of the entire dataset. Moreover, since many people's scores were clustered around the baseline level, to further improve the score, I believed it was necessary to focus on the small subset of data that was difficult to predict. Specifically, I hypothesized that the following three types of data were prone to misclassification by the model due to their low volume and lack of variation:

1.  Data where the student's answer is correct, but the explanation is incorrect (`True_Misconception:XXX`)
      * The model tends to predict these as `Correct` or `Neither`.
2.  Data where the student's answer is wrong, but the explanation is correct (False_Correct:NA)
      * The model tends to incorrectly assign some `Misconception` label.
3.  Data with ambiguous explanations
      * These are difficult to distinguish between `Correct` / `Neither` or `Neither` / `Misconception`.

Since the `Question` is provided in this competition, the correctness of the answer can be determined beforehand. This means the True/False component of the prediction can be answered with nearly 100% accuracy. Consequently, the problem is essentially a 37-class classification task (`Correct`, `Neither`, and 35 types of `Misconceptions`).

Among these classes, `Correct` and `Neither` can be labels for any question, so I reasoned that models are more likely to include them in their prediction candidates. Therefore, I concluded that I didn't need to intensively focus on improving the second and third hypotheses (the distinction between `Correct` / `Neither`) above.

Instead, the key challenge was to ensure the model included the correct `Misconception` as one of its top three candidates for data with ambiguous explanations.

For such difficult-to-classify data, a single model or training approach would likely lead to predictions biased by that model's specific "quirks" or "perspective." To address this challenge, I decided that an ensemble of models incorporating diverse approaches was crucial. I'm not certain if this hypothesis was correct, but I'm relieved that it resulted in a gold medal.

-----

### Ensemble Strategy and CV/LB Correlation

For my cross-validation (CV) strategy, I initially used `StratifiedKFold` based on the `Misconception` labels. However, I noticed that some `StudentExplanation` were identical, which could lead to an overly optimistic evaluation. I then switched to `StratifiedGroupKFold`, using groups based on the `StudentExplanation` after removing spaces and periods.

Once a single model's Public LB score reached around 0.947, I observed that the correlation between the CV score and the LB score almost completely disappeared. A similar phenomenon was observed with the model ensembles. I speculate that this was because the hard-to-predict data I identified earlier is rare, and the test set likely contains expressions not present in the training data.

Based on this hypothesis, I chose my models using the following policy:

  * Single Models: Selected based on a combination of both CV and LB scores.
  * Ensemble: Ignored CV scores and made adoption decisions based solely on Public LB score improvement.

I made the decision not to use a single model, even if it achieved a high LB score like 0.949, if it did not contribute to improving the ensemble's LB score. To prevent overfitting to the LB score, we minimized adjustments to the ensemble weights. 
This was a bit of a gamble, but it ultimately paid off.

To ensure diversity and incorporate approaches I might not have conceived of myself, I included models from two excellent public notebooks directly into my ensemble. This also freed up my time to focus on training other models. For the ensemble itself, I used a simple weighted average of the predicted probabilities from each model. Once again, I deeply thank the authors who shared their notebooks and model weights.
- [2] [DeepSeekMath-7B LB:0.944](https://www.kaggle.com/code/jaytonde/deepseekmath-7b-lb-0-944)
- [4] [[LB 0.942][Infer] FullFT Qwen3-8B by SFTTrainer](https://www.kaggle.com/code/sinchir0/lb-0-942-infer-fullft-qwen3-8b-by-sfttrainer)

-----

### Number of Classification Classes

The training data contains 65 unique labels. However, the problem can be reduced to a 37-class classification task since the True/False component of the answer can be determined with almost perfect accuracy.

To maximize model diversity, I built and ensembled both 65-class and 37-class models.

-----

### Data Augmentation

I performed data augmentation to address the first type of difficult data I hypothesized: cases where "the student's answer is correct, but the explanation is incorrect (`True_Misconception:XXX`)". This was applied to only one of the ensembled models.

While the improvement in accuracy was minimal, I believe it contributed to the overall diversity of the ensemble.

Procedure:

1.  For each `Question`, extract data where the "answer is incorrect and contains a specific Misconception (`False_Misconception:XXX`)".
2.  Randomly sample 10% of this extracted data.
3.  Rewrite only the student's answer part of the sampled data to be correct and change the label to `True_Misconception:XXX`.

-----

### Post-Processing

By analyzing the training data, I found that the types of possible `Misconceptions` are limited for each `QuestionId`. To leverage this domain knowledge, I applied the following post-processing step:

1.  From the training data, create a set of possible `Misconceptions` corresponding to each `QuestionId`.
2.  During inference, for a given `QuestionId`, forcibly set the predicted probabilities to zero for any `Misconceptions` that did not appear for that question in the training data.

Code to Create the Misconception Map:

```python
def create_question_misconception_map(self, train: pd.DataFrame) -> Dict[int, set]:
    question_misconception_map = {}
    
    train["Category_del_prefix"] = train.Category.apply(
        lambda x: x.split("_", 1)[-1] if "_" in x else x
    )
    train["target"] = train.Category_del_prefix + ":" + train.Misconception
    
    unique_question_ids = train["QuestionId"].unique()
    for question_id in unique_question_ids:
        question_data = train[train["QuestionId"] == question_id]
        possible_targets = set(question_data["target"].unique())
        question_misconception_map[question_id] = possible_targets
        
    return question_misconception_map
```

-----

### List of Models Used

| \# | Model | \# Classes | Training Method | Data Aug. | CV (\*1) | Public LB | Private LB | Weight | Note |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Gemma2-9b-it | 37 | LoRA | - | 0.946 | 0.946 | 0.943 | 0.25 | - |
| 2 | Qwen2.5-14B-Instruct | 37 | QLoRA | ☑️ | 0.945 | 0.946 | 0.943 | 0.25 | - |
| 3 | Gemma2-9b-it | 65 | QLoRA | - | - | 0.947 | 0.942 | 0.25 | - |
| 4 | Qwen2.5-14B-Instruct | 37 | QLoRA | - | 0.938 | - | - | 0.1 | - |
| 5 | Qwen3-8B | 65 | Full FT | - | - | 0.942 | 0.938 | 0.1 | Public Notebook [4] |
| 6 | Deepseek-math-7b-instruct | 65 | QLoRA | - | - | 0.944 | 0.940 | 0.25 | Public Notebook [2] |

(\*1) Model \#1 was evaluated with `StratifiedKFold`; all other models were evaluated with `StratifiedGroupKFold`.

-----

### Prompts Used

I used different prompts for each model. (Variable omitted.)

Model # 1

```python
<start_of_turn>user
You are an expert math educator analyzing student responses for mathematical misconceptions.

### TASK
Your goal is to analyze the student's explanation for a given math problem and classify it into a specific category of understanding or misconception.

### INPUT
- Question:{row['QuestionText']}
- Answer Choices:{choices}
- Student's Answer:{row['MC_Answer']}
- Correct?:{correctness}
- Student's Explanation:{row['StudentExplanation']}
- Classification:<end_of_turn>
<start_of_turn>model
```

Model #2
```python
<|im_start|>user
Question: {row["QuestionText"]}
Answer: {row["MC_Answer"]}
{correctness}
Student Explanation: {row["StudentExplanation"]}<|im_end|>
<|im_start|>assistant
```

Model #3
```python
<start_of_turn>user
You are an expert math educator analyzing student responses for mathematical misconceptions.

Question: {question}
Correct Answer: {answer}
Student's Explanation: {explanation}

CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception

TASK: Classify this student's response using EXACTLY ONE of these {len(all_labels)} labels:

{labels_text}

Classification:<end_of_turn>
<start_of_turn>model
```

Model #4
```python
<|im_start|>user
Question: {row["QuestionText"]}
Ground Truth: {ground_truth}
Student Explanation: {row["StudentExplanation"]}<|im_end|>
<|im_start|>assistant
```
