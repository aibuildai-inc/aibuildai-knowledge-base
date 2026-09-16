# Public 6th | Private 31st solution - Multi-head structure

Competition: map-charting-student-math-misunderstandings
Rank: #31
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/private-31st-public-5th-solution-multi-head-ar

Thanks Kaggle and community for this fun competition. 

I spent significant time training and testing various approaches, but eventually shook out of the gold zone. However, I did anticipate this to happen. As I submitted more versions, the unstable LB scores and CV-LB inconsistency unveiled the randomness inherent in this competition. 

My major takeaway: control experimental randomness and systematically figure out what works and what doesn't.

---

# 1. Model Structure: Dual-Head Hybrid Architecture

### Major Decisions
- Only classification approach is used (no causalML)
- Qwen-32B as backbone. Tests show that: Qwen performs better than others; Qwen-32B outperforms smaller versions (14B, 7B)
- Multi-head structure (1 global + 15 local) combining global knowledge with local experts
- Model only classifies misconceptions (37 labels); correct or incorrect is labeled afterwards by hard code.

.png?generation=1760995592574729&alt=media)


### Global Head (37 dims)
- Learns cross-question universal misconception patterns
- Normal gradient flow to LoRA backbone

### Local Heads (15 heads, variable dims)
- One head per QuestionId
- Each outputs only the candidate misconceptions for that question
- Example: Question 31772 has 3 candidates [0,1,5], Question 32835 has 4 candidates [17,18,19,20]

### Gradient Blocking Mechanism
Prevents local head overfitting from harming backbone generalization:

```python
if eta_local2lora == 0:
    feat_local = feat.detach()  # Hard blocking
else:
    feat_local = feat.detach() + eta * (feat - feat.detach())  # Soft blocking
```

Optimal: eta_local2lora = 0.2 (20% gradient flow)

### Inference Fusion

```python
final_score = lambda * p_local37 + (1 - lambda) * p_global
```

Optimal lambda range: 0.0~1.0 (test multiple during validation)

---

# 2. Training Strategy


.png?generation=1760993973100608&alt=media)

### Cross-Validation
5-fold Stratified KFold based on QuestionId + MisconceptionId combination

### Input Format
Trained multiple models with different query formats to increase diversity for ensemble:
- Query 1: Question + StudentAnswer + is_correct + StudentExplanation
- Query 2: Question + AllChoices + StudentAnswer + is_correct + StudentExplanation
- Query 3: Question + CorrectAnswer + StudentAnswer + is_correct + StudentExplanation
- ...

### Multi-Phase LR Scheduling

Learning rate is very critical in this competition. The settings below yielded the best outcome:
- LoRA: lr=2e-4, wd=0.01
- Global head: lr=5e-5, wd=0.05
- Local heads: lr=5e-5, wd=0.05
- Loss weights: alpha=0.6 (local), beta=1.0 (global)

Best CV score is typically achieved near the end of the second epoch (epoch 1). Epoch 2 is kept but rarely leads to better CV scores.
- Phase 1 (Epoch 0-1): cosine 100%→5%
- Phase 2 (Epoch 2):   cosine 5%→0%

---

# 3. Submission Strategy

Selected models trained from different folds with different input formats for better generalization. Assigned different models to different QuestionIds based on their CV performance, which further improved the LB score slightly.

```python

cands = { # candidates' list
    "model1":[lora_path_1, lambda_1, query_format_1],
    "model2":[lora_path_2, lambda_2, query_format_2],
    "model3":[lora_path_3, lambda_3, query_format_3],
    "model4":[lora_path_4, lambda_4, query_format_4],
    "model5":[lora_path_5, lambda_5, query_format_5],
    ...
}


cands_sub = { # weights
    "model1": {31772: 0.4, 31774: 0.0, 31777: 0.4, 31778: 0.4, 32829: 0.4, 32833: 0.4, 32835: 0.4, 33471: 0.4, 33472: 0.4, 33474: 0.0, 76870: 0.4, 89443: 0.4, 91695: 0.4, 104665: 0.0, 109465: 0.4},
    "model2": {31772: 0.0, 31774: 0.4, 31777: 0.0, 31778: 0.0, 32829: 0.0, 32833: 0.0, 32835: 0.0, 33471: 0.0, 33472: 0.0, 33474: 0.4, 76870: 0.0, 89443: 0.0, 91695: 0.0, 104665: 0.4, 109465: 0.0},
    "model3": {31772: 0.5, 31774: 0.5, 31777: 0.5, 31778: 0.5, 32829: 0.5, 32833: 0.5, 32835: 0.5, 33471: 0.5, 33472: 0.5, 33474: 0.5, 76870: 0.5, 89443: 0.5, 91695: 0.5, 104665: 0.5, 109465: 0.5},
    "model4": {31772: 0.4, 31774: 0.4, 31777: 0.4, 31778: 0.4, 32829: 0.0, 32833: 0.4, 32835: 0.0, 33471: 0.4, 33472: 0.4, 33474: 0.4, 76870: 0.0, 89443: 0.0, 91695: 0.0, 104665: 0.4, 109465: 0.0},
    "model5": {31772: 0.0, 31774: 0.0, 31777: 0.0, 31778: 0.0, 32829: 0.0, 32833: 0.0, 32835: 0.4, 33471: 0.0, 33472: 0.0, 33474: 0.0, 76870: 0.4, 89443: 0.4, 91695: 0.0, 104665: 0.0, 109465: 0.0},
    "model6": {31772: 0.0, 31774: 0.0, 31777: 0.0, 31778: 0.0, 32829: 0.4, 32833: 0.0, 32835: 0.0, 33471: 0.0, 33472: 0.0, 33474: 0.0, 76870: 0.0, 89443: 0.0, 91695: 0.4, 104665: 0.0, 109465: 0.4},
}

```

Eventually, the disagreement handling ensemble method is borrowed from a public notebook (https://www.kaggle.com/code/kishanvavdara/ensemble-gemma-qwen-deepseek), which yielded a slightly better public LB result than simple weighted ensemble:

```python
base[c]  = sum(w_m * p_m[c])      # Weighted probability sum
conf[c]  = max(w_m * p_m[c])      # Highest confidence
agree[c] = (top3 votes) / M       # Consensus score

final[c] = 0.6*base[c] + 0.3*agree[c] + 0.1*conf[c]
```

---

# 4. What Didn't Work

- Label smoothing (both traditional uniform and smart frequency-based): no improvement, kept eps=0.0
- Data augmentation (oversampling, synthetic data): no improvement
- Add AnswerId format (A/B/C/D) into the prompt
- Pure local heads (lambda=1.0): severe overfitting
- Hard negative mining: no improvement
- CausalML approach (pair-wise rerank): classification-only approach worked better
