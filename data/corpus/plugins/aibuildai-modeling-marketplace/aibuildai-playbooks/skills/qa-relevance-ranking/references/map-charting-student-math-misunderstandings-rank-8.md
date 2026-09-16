# 8th Place Solution

Competition: map-charting-student-math-misunderstandings
Rank: #8
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/8th-place-solution

Many thanks to the Kaggle team and organizers for running such an interesting challenge. I’m really happy to have achieved a solo gold medal.

# Overview
- Inference acceleration with vLLM + an ensemble of many models
- Add the question’s answer choices to the prompt
- Search for the optimal learning rate and batch size
- Create synthetic data for labels with few samples or that are hard to predict

# Model
- Task type: Chosen as Causal LM because it is easy to accelerate with vLLM.
- Architecture: Based on my experiments, Qwen2.5 and Qwen3 performed best among the architectures I explored.
- Final models used:
 - **Qwen/Qwen2.5-7B**
 - **Qwen/Qwen2.5-14B-Instruct-AWQ**
 - **Qwen/Qwen3-8B**
 - **Qwen/Qwen3-14B-AWQ**
- Training: All models were trained with LoRA.

# Prompting
From [this discussion](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/589350), I found that students answer by choosing one option from a–d.



Therefore, I reconstructed the correspondence between `MC_Answer` and choices a–d from `StudentExplanation`, and modified `QuestionText` as follows:

`Which number is the greatest? (a) 6 (b) 6.2 (c) 6.079 (d) 6.0001`

I believe this helped with prediction by allowing the model to compare the student’s `MC_Answer` with the other options and identify the corresponding `Misconception`.

The final prompt was as follows:
```text
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.] (a) 1/3 (b) 3/6 (c) 3/8 (d) 3/9
Answer: 1/3
Correct: Yes
Student Explanation: 0ne third is equal to tree nineth
Type of Misunderstanding:
```

For readability during later synthetic data creation and EDA, I removed LaTeX notation from `QuestionText` and `MC_Answer` (**Note**: this had no impact on the score).

Inspired by [this notebook](https://www.kaggle.com/code/aleaiest/lb-0-945-qwen2-5-32b-gptq), I mapped the target labels to the following symbols and generated a single token.

```python
special_character_list = [
    '■', '□', '▲', '△', '▼', '▽', '◆', '◇', '○', '●', '★', '☆',
    '♦', '♥', '♠', '♣', '§', '†', '‡', '※', '∞', '±', '≠', '≈', '√',
    '∑', '∏', '∆', 'Ω', 'μ', '∂','→', '←', '↑', '↓', '↔', '↕', '〈', '〉',
    '『', '』', '│', '─', '┌', '┐', '└', '┘', '┼', '█', '▓', '▒','£',
    '¥', '€', '₩', '©', '®', '™', '♪','♫', '☀', '☁', '☂', '☃', '☎'
]

special_token = [tokenizer(i, add_special_tokens=False).input_ids[0] for i in special_character_list]
```
Each of these symbols corresponds to a single token in the Qwen tokenizer.

# Training

- **Epochs:** 3  
- **Optimizer:** AdamW (weight_decay = 0.01)  
- **Scheduler:** cosine (warmup_ratio = 0.03)  
- **Batch size:** 12  
- **Learning rate:** 0.0002  

- **LoRA Configuration**  
  - **r:** 16  
  - **alpha:** 16  
  - **dropout:** 0.05  
  - **target_modules:**  
     - `q_proj`  
     - `k_proj`  
     - `v_proj`  
     - `o_proj`  
     - `gate_proj`  
     - `up_proj`  
     - `down_proj` 

I noticed during hyperparameter trials that small changes to batch size and learning rate could cause large swings in CV scores. Therefore, I tuned those two carefully and searched for the best combination.
Changing LoRA `r` or `alpha` did not yield a meaningful improvement in scores.
Compared with full-parameter fine-tuning, LoRA achieved better CV in my runs (possibly due to insufficient tuning on my side for full-parameter training).

# Data
Because the competition labels showed imbalance and differences in prediction difficulty, I attempted to create synthetic data using generative AI (`gpt-5`  and `o3`).
I first trained a model on the competition data and computed out-of-fold (OOF) predictions with MAP@3. I then aggregated the mean MAP@3 and the number of samples for each (`QuestionId`, `Category`, `Misconception`) pair. For pairs with low mean MAP@3 or few samples, I generated synthetic data using the competition data as few-shot examples. The generation prompts included the `Category` and `Misconception` descriptions and instructed the model to create up to 50 new samples. Few-shot examples were sampled either uniformly at random or at random from items with low OOF scores. I repeated generation multiple times to obtain approximately 30,000 synthetic samples in total.

I trained models with different synthetic data configurations and used them in the ensemble. The contribution from synthetic data appeared modest—likely ≤ 0.001 on the leaderboard. (If leaderboard precision increases, I plan to investigate this in more detail.)

# Final submission

- **Ensemble:** 10 experimental settings (each setting trained with 5 folds → 10 × 5 = 50 models in total)
- **Final submission:** selected both the LB-best ensemble and the CV-best ensemble; the LB-best performed better on the Private LB.
- **Weighting:** applied weights based on LB and CV results

## LB-best results
**Public LB:** 0.951, **Private LB:** 0.948
| Model                          | Synthetic Data | Public LB (Rank) | Private LB | Weight |
|--------------------------------|----------------|-------|-------|--------|
| Qwen/Qwen2.5-7B               | ✅             | 0.949 (#7)     | 0.944     | 0.5    |
| Qwen/Qwen2.5-7B               | ✅             | 0.949 (#8)     | 0.945     | 0.5    |
| Qwen/Qwen2.5-7B               | ✅             | 0.949 (#1)    | 0.946     | 1.5    |
| Qwen/Qwen2.5-14B-Instruct-AWQ | ✅             | 0.949 (#6)     | 0.946     | 1.5    |
| Qwen/Qwen3-8B                 |                | 0.949 (#5)     | 0.945     | 1.0    |
| Qwen/Qwen3-8B                 | ✅             | 0.949 (#4)     | 0.945     | 1.0    |
| Qwen/Qwen3-8B                 | ✅             | 0.949 (#9)     | 0.944     | 0.5    |
| Qwen/Qwen3-8B                 | ✅             | 0.949 (#10)     | 0.946     | 0.5    |
| Qwen/Qwen3-14B-AWQ            |                | 0.949 (#3)     | 0.945     | 1.5    |
| Qwen/Qwen3-14B-AWQ            | ✅             | 0.949 (#2)     | 0.945     | 1.5    |

*Rank indicates the submission’s position when the Submissions page is sorted by Public LB.
