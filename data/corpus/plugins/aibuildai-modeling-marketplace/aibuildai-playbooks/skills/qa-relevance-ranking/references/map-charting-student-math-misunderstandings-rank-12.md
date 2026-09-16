# 12th Place Solution

Competition: map-charting-student-math-misunderstandings
Rank: #12
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/12th-place-solution

First and foremost, big thanks to the host and the Kaggle community that made this competition possible. I learned so much from this competition and my first solo gold is definitely the cherry on top🥇

# **Overview**
My final submission was an ensemble of five Qwen2.5-14B-Instruct models. The distillation from larger 72B models was the trick that boosted the score the most (+0.003~0.005). Models were fine-tuned on the full training dataset, utilizing EMA for stability. 

# **Preprocessing**
The True_Neither class appeared the third most and was very difficult to classify. In an attempt to generate these samples, I made the following prompt for evaluating them. Even though the generated data didn’t not improve the CV, the mean MAP@3 varied across different score groups. Therefore, I applied this prompt to the training data and used the score to re-create the CV.

```python
Below is a student explanation for a multiple-choice math question. Evaluate it as an experienced math educator for evidence of understanding and communication quality. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
    
    - Add 1 point for topical relevance: The explanation references the question, choices, or relevant quantities/relationships. Purely off-topic or generic commentary earns 0.
    - Add 1 point if the explanation is off-topic or generic but the chosen answer is correct (<correct>yes</correct>), indicating possible tacit knowledge despite weak articulation.
    - Add 1 point for absence of misconceptions: The explanation contains no incorrect mathematical claims, contradictions, or misapplied rules. Vague or hedged language is acceptable. Purely off-topic or generic commentary is not considered a misconception.
    - Add 1 point for partial conceptual grasp (competent but imperfect): The explanation indicates the correct idea, operation, or relationship—even if incomplete, imprecise, or with a minor slip—showing the student “knows what it is” but struggles to fully express it.
    - Add 1 point for reasoning trace: There is at least a minimal, coherent rationale linking the chosen answer to relevant concepts, steps, comparisons, or definitions. The chain can be brief or underspecified but should be logically connected and non-contradictory.
    - Subtract ALL points if the explanation is logically flawless/perfect (fully correct and complete, with a coherent, gap-free reasoning chain, no errors or contradictions, and no unjustified leaps), regardless of tone or register. Simple/childlike wording is acceptable; the penalty targets logical perfection, not expert-like phrasing.
    - Subtract ALL points if the explanation contains any mathematical misconceptions (incorrect mathematical claims, contradictions, misapplied rules, or fundamental misunderstandings of mathematical concepts). Purely off-topic or generic commentary alone does not trigger this penalty.
```

# **Model Training**
- Qwen2.5-72B-Instruct & Qwen2.5-Math-72B (AutoModelForSequenceClassification)
Used QLoRA for fine-tuning with r=64 and lora_alpha=128. They were trained on 4-folds for 2 epochs with a learning rate of 1e-4.
- Qwen2.5-14B-Instruct (AutoModelForSequenceClassification)
Used QLoRA for fine-tuning with r=64 and lora_alpha=128. (There was no outstanding performance gain with LoRA.) Trained with a custom head for distillation for 3 epochs with a learning rate of 1e-4. Five variants were trained by setting different distillation loss rates, while all variants utilized EMA with the decay value set to 0.999.

In earlier experiments, I tried to distill into smaller models (e.g., ettin-encoder-1b). However, these models couldn’t surpass the performance of the 14B model (-0.004~0.002 worse on average).

# **Final Submission**
Took a simple average over the probabilities of all the variants. Ensemble boosted the public LB to (+0.001~0.003) as well as the private. Using torch.compile(), I could barely fit the five merged models.

# **Other Things That Didn’t Work**
- Include multiple choices in the prompt
- Inference with vllm
- Hard negative sampling
- ...
