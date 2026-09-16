# MAP Competition 13th Place Gold Medal Solution

Competition: map-charting-student-math-misunderstandings
Rank: #13
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/map-competition-13th-place-gold-medal-solution

Thanks to the competition organizers, data providers, and all participants who contributed valuable discussions.

## Strategy

**Two Approaches × Multiple Models × Ensemble**

- Classification approach: 14B model × 2
- Listwise approach: 72B model × 2 + 32B model × 2
- Total of 6 models in hybrid ensemble
- All 6 models trained with different seeds

Used all 36,696 samples (no validation split)

## Model-wise Scores

| Model | Task | Weight | Private Score | Public Score |
|--------|--------|--------|---------------|--------------|
| Qwen3-14B-Base | Classification | 0.9 | 0.943 | 0.949 |
| Qwen3-14B-Base | Classification | 0.9 | 0.944 | 0.948 |
| Qwen3-32B | Listwise | 1.0 | 0.943 | 0.949 |
| Qwen3-32B | Listwise | 1.0 | 0.943 | 0.948 |
| Qwen2.5-72B-Instruct | Listwise | 1.0 | 0.945 | 0.947 |
| Qwen2.5-72B-Instruct | Listwise | 1.0 | 0.945 | 0.949 |
| **Ensemble** | **-** | **-** | **0.947** | **0.952** |

## Approach

### Key Design Decisions

The competition uses a fixed set of 15 questions, and each sample is labeled with Category (Correct/Neither/Misconception) and Misconception tags. We made three key observations:

1. **Target Reduction**: Each question has only 4-6 possible targets (out of 37 total classes excluding True/False). Both approaches leverage this constraint.

2. **True/False Separation**: The True/False label indicates whether the student's answer is correct or incorrect. Since the competition uses fixed questions, we can determine the correct answer from training data without prediction. We remove True/False from the prediction target and append it as a prefix to final predictions.

3. **Input Enhancement**: We extracted the 4 answer choices for each question from training data and added them to all input prompts.

### Classification Model
- Model: Qwen3-14B-Base
- Task: Train on all 37 classes using SequenceClassification
- Training: QLoRA 4bit, 2 epochs
- Inference: 4bit quantization + LoRA adapters, extract scores for 4-6 targets per question

### Listwise Model
- Models: Qwen2.5-72B-Instruct (×2), Qwen3-32B (×2)
- Task: Present 4-6 targets per question as options using CausalLM, generate option numbers
- Training: QLoRA 4bit, 1 epoch, option order randomly shuffled
- Post-processing: LoRA merge → (72B: intermediate_size padded 29568→29696 for vLLM tensor parallelization) → GPTQ 4bit quantization using  `gptqmodel`
- Inference: Use vLLM to obtain logprobs for each option

## Ensemble Method

Convert each model's scores to unified scale:
- Classification models: L1 normalize probabilities
- Listwise models: Softmax transform logprobs

Select Top 3 using weighted average. Add True/False prefix to final predictions.

## Resources
- Training: 14B (RTX 5090, 2h), 32B (RTX 5090, 10h), 72B (H200, 4h)
- Inference: Kaggle T4×2, 8 hours

## Discussion

**Data characteristics:**
- Labels are subjective with inconsistent annotation: misconception tags are abstract, containing multiple distinct concepts, with many labeling errors and borderline cases

**Consequences of these characteristics:**
- Multi-model ensemble worked effectively
- Synthetic data generation and CoT methods did not work (synthetic data was used for model characteristic evaluation, but had weak correlation with LB scores)

**About model selection:**
- Model selection was challenging: Public Score showed overly optimistic results, and nearly any combination yielded similar scores
- Our experiments revealed several insights that guided our decisions:
  1. Accuracy improved from 8B→14B→32B but plateaued beyond 32B, with 72B showing no additional gains
  2. Scores varied by approximately ±0.002 across different random seeds
  3. In synthetic data evaluation, listwise approaches consistently outperformed classification models
- We therefore used 2 models at each configuration to account for seed variation, and weighted listwise models higher based on their synthetic data performance
- This approach proved critical: relying solely on Public Score rankings would likely have cost us the medal
- Post-competition analysis revealed additional insights: Qwen3-32B underperformed compared to Qwen2.5-72B, but separate experiments showed Qwen2.5-32B matched Qwen2.5-72B, indicating Qwen3 was less suitable for this task. Furthermore, Private Score showed smaller seed variation than Public Score, suggesting a more diverse ensemble with different architectures (rather than multiple seeds) might have been more beneficial
