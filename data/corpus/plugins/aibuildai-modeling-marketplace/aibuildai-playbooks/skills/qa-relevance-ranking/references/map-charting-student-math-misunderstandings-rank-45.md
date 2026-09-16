# 45th Place (Silver Medal) : A Multi-LLM Ensemble Approach [LB-0.947]

Competition: map-charting-student-math-misunderstandings
Rank: #45
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/45th-place-silver-medal-a-multi-llm-ensemble-a

## Quick Summary

🎯 **Competition Goal**: Predict student math misconceptions from explanations  
📊 **Final Score**: 0.947 MAP@3 (Private LB)  
🏅 **Rank**: 45th place / Silver Medal  
🤖 **Approach**: 6-model LLM ensemble with intelligent voting  
⚡ **Key Innovation**: Family-based filtering + multi-factor scoring


## Acknowledgments 🙏

This Silver Medal achievement would not have been possible without the generous contributions of the Kaggle community. I'm deeply grateful to:

**Competition & Platform:**
-   **Kaggle** for providing this incredible platform to learn and compete
-   **Competition Hosts** for organizing the MAP competition and creating such a meaningful real-world challenge

**Community Contributors:**
-   **[Chris Deotte](https://www.kaggle.com/cdeotte)** - His Ettin-Encoder-1B discussion and data preprocessing approach provided the foundation for my data pipeline.
-   **[Kishan Vavdara](https://www.kaggle.com/kishanvavdara)** - His ensemble notebook was instrumental in developing my multi-model voting strategy.
-   **[Raja Biswas](https://www.kaggle.com/conjuring92)** - His pre-trained Eedi CoT 14B model from the previous competition gave me a strong starting point.

**The Kaggle Community:**
-   All the participants who shared their insights, notebooks, and discussions throughout the competition
-   The spirit of collaboration and knowledge-sharing that makes Kaggle such a special place to learn

This is my first Silver Medal, and it's a testament to how much you can achieve by standing on the shoulders of giants. Thank you all! 🚀

## Competition Overview
The MAP (Misconception Annotation Project) competition challenged participants to develop NLP models that predict students' mathematical misconceptions from their open-ended explanations. The goal was to help teachers efficiently identify and address incorrect thinking patterns that hinder student learning.
Evaluation Metric: Mean Average Precision @ 3 (MAP@3)

### Final Results:

- Public LB: 0.949
- Private LB: 0.947 (45th place, Silver Medal 🥈)

### Solution Architecture
My solution leveraged an ensemble of 6 different LLMs, each contributing unique strengths to the final prediction. The key insight was that different model architectures and training configurations capture different aspects of mathematical misconception patterns.


| # | Model | Public LB | Training Method | GPU Setup | LoRA Config | Learning Rate | Epochs | Training Time |
|---|-------|-----------|-----------------|-----------|-------------|---------------|--------|---------------|
| 1 | **Hunyuan 7B Instruct** | 0.945 | LoRA | 2×L4 | R:64, α:128 | 2e-4 | 3 | 4h 34m |
| 2 | **Qwen 3 4B** | 0.945 | LoRA | 2×L4 | R:512, WD:0.3 | 2e-5 | 3 | 2h |
| 3 | **Qwen 2 8B** | 0.945 | QLoRA 4-bit | 2×L4 | R:64, α:32 | 2e-4 | 2 | 4h 23m |
| 4 | **Qwen 3 14B** | 0.944 | QLoRA 4-bit | 4×L4 | R:16, α:32 | 2e-4 | 3 | 11h 34m |
| 5 | **Qwen 3 8B** | 0.943 | Full Fine-tune | 4×L4 | N/A | 2e-5 | 3 | 3h |
| 6 | **DeepseekMath 7B** | 0.944 | Full Fine-tune | 4×L4 | N/A | 2e-5 | 3 | 3h 24m |


### Data Preprocessing Pipeline
The preprocessing was consistent across all models to ensure fair comparison:
```
# Create target labels
train['target'] = train.Category + ":" + train.Misconception
train['label'] = le.fit_transform(train['target'])

# Identify correct answers
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1 # Mark these as correct answers

# Merge 'is_correct' flag into the main training DataFrame
train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
```

### Input Format

Each training example was formatted as:
```
Question: {QuestionText}
Answer: {MC_Answer}
Is Correct Answer: {Yes/No}
Student Explanation: {StudentExplanation}
```

This structure provides the model with full context: the question, the student's answer choice, whether it's correct, and their reasoning.

## Training Strategies

### 1. **LoRA (Low-Rank Adaptation)**
- Used for: Hunyuan 7B, Qwen 3 4B
- **Advantages:** Memory efficient, faster training, prevents catastrophic forgetting
- **Configuration variations:** Experimented with ranks from 64 to 512

### 2. **QLoRA (Quantized LoRA)**
- Used for: Qwen 2 8B, Qwen 3 14B
- **4-bit NF4 quantization** with bfloat16 compute dtype
- Enabled training of larger models on limited GPU resources
```
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### 3. **Full Fine-tuning**
- Used for: Qwen 3 8B, DeepseekMath 7B
- Complete parameter updates for maximum model adaptation
- Required more GPU memory but achieved strong performance

### Key Training Hyperparameters
```
# Common settings across models
num_train_epochs = 3
learning_rate = 2e-4 to 2e-5  # Varied by model
lr_scheduler_type = "cosine"
warmup_ratio = 0.1
gradient_accumulation_steps = 16
fp16/bf16 = True
```

### Inference Strategy
Probability Extraction
For each model, I generated full probability distributions over all classes (not just top-3), which was crucial for effective ensembling:
```
# Generate probabilities
logits = trainer.predict(test_dataset).predictions
probs = softmax(logits, axis=1)

# Sort and save top-K classes with probabilities
top_indices = np.argsort(-probs, axis=1)
for i in range(num_classes):
    prob_dict[f"prob_{i}"] = probs[i, top_indices[i]]
    
# Save for ensembling
prob_df.to_csv(f"submission_{model_name}_prob.csv", index=False)
```

### Ensemble Methodology
The ensemble was the critical component that boosted performance from individual 0.943-0.945 scores to 0.949 on public LB.

### Family-Based Filtering
A key insight was recognizing that predictions must respect the category family (True vs False):
```
# Build family map from correct answers
fam_map = test_df.merge(correct_answers, on=['QuestionId','MC_Answer'])
fam_map['family'] = fam_map['is_correct'].map({1: 'True_', 0: 'False_'})
```

### Weighted Voting with Multi-Factor Scoring
The ensemble combined three scoring components:

1. Weighted Probability Sum (34%): Aggregate model confidence
2. Agreement Bonus (33%): How many models agree on this class
3. Confidence Bonus (33%): Maximum confidence from any model

```
for class_name in all_classes:
    base_score = class_total_prob[class_name]
    agreement_bonus = class_votes[class_name] / n_models
    confidence_bonus = class_max_prob[class_name]
    
    final_scores[class_name] = (
        base_score * 0.34 +
        agreement_bonus * 0.33 +
        confidence_bonus * 0.33
    )
```

### Family Filtering & Backfilling
```
# Filter by family
final_scores = {k: v for k, v in final_scores.items() 
                if k.startswith(family_prefix)}

# Backfill if < 3 predictions
fillers = [f"{family}_Neither:NA"]
if family == "True_":
    fillers.append(f"{family}_Correct:NA")
```

### Model Weights
All models were weighted equally (weight=1) as they performed similarly:
```
weights = [1, 1, 1, 1, 1, 1]  # Equal weighting
```

## Key Insights & Learnings

### 1. **Diversity Matters**

-   Combining different architectures (Qwen, Hunyuan, DeepseekMath) captured complementary patterns
-   Different training strategies (LoRA, QLoRA, full fine-tuning) provided diverse perspectives

### 2. **Family Constraint is Critical**

-   Enforcing category family (True/False) consistency prevented nonsensical predictions
-   Simple rule-based filtering provided huge gains

### 3. **Probability Calibration**

-   Saving full probability distributions (not just top-3) enabled sophisticated ensembling
-   Multi-factor scoring (probability + agreement + confidence) outperformed simple averaging

### 4. **Efficient Training**

-   LoRA and QLoRA enabled training large models (up to 14B parameters) on accessible hardware
-   4-bit quantization had minimal impact on final performance

### 5. **Transfer Learning**

-   Using Raja Biswas's pre-trained Qwen2 model from the previous Eedi competition provided a strong starting point
-   Domain-specific pre-training on math misconception data was highly valuable

----------

## Computational Resources

-   **GPUs:** Primarily 2×T4, 2×L4 and 4×L4 configurations
-   **Total Training Time:** ~30+ hours across all models
-   **Inference Time:** ~6-7 hours for full test set with all 6 models

## What Didn't Work

### Models Explored but Not Included:
- **OpenAI GPT 20B**: Training complexity outweighed benefits
- **Generative Training Approach**: Text generation inferior to classification
- **Small Models (smoLM2/3)**: 360M-3B parameter models lacked capacity
- **50+ Other Models**: Including Llama 3.1 8B, Phi 4 12B, and others
- **Qwen 32B**: Insufficient time for full inference pipeline in 4-bit quantization

### Techniques That Underperformed:
- **Heavy Prompt Engineering**: Structured input format worked better than complex prompts
- **Unequal Ensemble Weights**: Equal weighting performed as well or better
- **Single Model Approaches**: None matched ensemble performance

## What Could Be Improved

1.  **Hyperparameter Optimization:** More systematic tuning of LoRA ranks, learning rates, and ensemble weights
2.  **Data Augmentation:** Synthetic misconception generation to increase training diversity
3.  **Ensemble Weights:** Learning optimal weights rather than equal weighting
4.  **Cross-Validation:** Better validation strategy to detect overfitting

## Conclusion

This competition demonstrated that **ensemble diversity**, **efficient fine-tuning techniques**, and **domain-specific constraints** are key to achieving top performance in specialized NLP tasks. The combination of 6 carefully trained LLMs with intelligent ensembling achieved a **0.947 MAP@3 score** and a **Silver Medal** finish.

----------

## Resources and Credits

-   **Data Preprocessing:** [Chris Deotte' Ettin-Encoder-1B Discussion](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/590326)
-   **Ensemble Technique**[Kishan Vavdara's ensemble Notebook](https://www.kaggle.com/code/kishanvavdara/ensemble-gemma-qwen-deepseek?scriptVersionId=260098381&cellId=9)
-   **Competition:** [MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)
-   **Pre-trained Model:** [Raja Biswas's Eedi CoT 14B](https://www.kaggle.com/models/conjuring92/eedi-cot-14b-dec6-awq/)
-   **Libraries:** HuggingFace Transformers, PEFT, BitsAndBytes, scikit-learn

**Thank you for reading! Feel free to ask questions or discuss the approach.** 🚀
