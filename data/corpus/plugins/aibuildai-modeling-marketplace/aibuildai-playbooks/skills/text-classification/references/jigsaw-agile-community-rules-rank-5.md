# 5th Place solution - Diverse Ensemble

Competition: jigsaw-agile-community-rules
Rank: #5
Source: https://www.kaggle.com/c/jigsaw-agile-community-rules/writeups/5th-place-solution-diverse-ensemble

First of all thanks to the organizers, wonderful public notebooks, and the community and everyone who shared ideas along the way.

# Overview 

Our final solution is a deliberately diverse ensemble. We combined per-rule specialists (DeBERTa NLI training; embedding models with logistic regression + k-NN), Inference-only large model (Qwen-32B trained on the official data and inferred with similarity-based sampling), and several test-time-trained (TTT) components across all rules (public DeBERTa training; public FAISS + triplet pipeline; Qwen-3 1.7B full finetune; Llama-3.1 8B LoRA). All models used a consistent instruction-only prompt and Yes/No-constrained decoding, then we blended them with per-rule rank-normalized weighting.

# Data Preprocessing 
- Drop `subreddit` - Treated as noisy/unstable signal; removing it improved consistency.
- Deduplicate rows: Removed exact duplicates, reducing train size to <10k examples, faster training with no measurable drop on lb.

# Performance Evaluation
We used the “trust public LB” method.

# Models

| Model                                        | Public LB | Private LB | Runtime*  |
|----------------------------------------------|-----------|------------|-----------|
| LLaMA-3.1 8B Instruct (TTT)                  | 0.92084   | 0.92618    | ~130 min  |
| Qwen-3 8B (TTT)                              | 0.91971   | 0.92314    | ~130 min  |
| Triplet + FAISS (public, improved)           | 0.90082   | 0.91131    | ~70 min   |
| 6× Embedding models → LR + kNN (per-rule)    | 0.90067   | 0.90976    | ~90 min   |
| DeBERTa (public training recipe)             | 0.90149   | 0.90767    | ~45 min   |
| DeBERTa NLI (custom)                         | 0.89853   | 0.90636    | ~90 min  |
| Qwen-32B (Inference-only, similarity sampling)   | 0.89065   | 0.89671    | ~300 min  |

\* End-to-end runtime on our 2×T4 setup.


# Training and Inference: 

## LLaMA-3.1 8B Instruct & Qwen-3 8B (LoRA, DDP)

- Setup: 2×T4 with DDP, fp16 + grad checkpointing, seq len 512 for train + inference.
- LoRA: ```r=256, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"].```
- Opt/Schedule: paged_adamw_8bit, LR≈2e-4, cosine, warmup≈10%.


#### Prompt :

```
 Instructions: You are a expert Reddit Moderator, you are given a comment and a rule. Your task is to classify whether the comment violates the rule. Only respond Yes/No.
<Rule>{rule}</Rule>
<Comment>{body}</Comment>
Answer: ```

#### Inference:
 vLLM TP=2, max_model_len=512; constrained to {Yes, No}, max_tokens=1, take Yes prob; if over budget, truncate only <Comment>.



## Embedding Models 

Families used: E5 (base, large), Qwen3-Embedding (0.6B), BGE (small, large), GTE (large).

#### Per-rule pipeline
- De-dup & de-bias:  cluster near-duplicates within each rule (cosine > 0.95–0.97) and down-weight them so the classifier doesn’t overfit repeated patterns.
- Train two light heads (6-fold CV):
      - Logistic Regression: grid over 31 C values .
      - KNN: grid over K=10–15, distance-weighted.
      - For each embed model & rule, keep the best CV setting.
- Head blending (per rule): 0.7 · LR + 0.3 · KNN to couple calibration (LR) with local neighborhood cues (KNN).
- Model ensemble search (per rule): GPU-accelerated simplex/grid over weights for the six embed models to maximize CV AUC (constrained, non-negative, sum to 1).
- Stabilization: final per-rule score = 0.9 · ensemble probability + 0.1 · rank-mean (guards against tail bias / scale drift across rules).
- Output: calibrated probability per row, per rule—later blended with other model families in the overall ensemble.
(This embed stack was our earliest strong baseline; we iterated heavily on de-dup thresholds, CV folds and embed models )


## DeBERTa NLI (rule-aware zero-shot → TTT)

Base: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
NLI framing: 
         - Premise: comment body
         - Hypothesis: “This comment violates the rule: {rule}.” (plus 8 paraphrases)
         - Label map: Entailment → violation, Contradiction/Neutral → compliant

Per-rule Test-Time Training (TTT):
      - Build a small rule-specific set from test canonical positives/negatives.
      - Upsample to ~25k pairs; train 2 epochs.
      - Update only biases + classifier head + LayerNorms (backbone frozen), faster training.
Balancing: dynamic pos/neg multipliers to ~50/50 (cap 32×).


## Qwen 2.5 32B LLM with Asymmetric Example Selection

Core idea: Before each inference, pick contextually relevant examples using embeddings—then prompt the LLM.
   - Positive examples: choose the most similar to the comment (clear violation prototypes).
   - Negative examples: choose the least similar (hard, diverse non-violations).
 This asymmetric pairing tightens the decision boundary.

Pipeline:
   1. Embed canonicals (2 pos + 2 neg per rule) with Qwen3-0.6B embeddings.
   2. For each comment: encode (rule+comment) → compute cosine sims → argmax pos, argmin neg, with simple quality thresholds.
   3. Build the prompt with the selected pos/neg examples.

Infer with Qwen-32B-Instruct (GPTQ) + LoRA; constrain outputs to Yes/No and softmax logits.
Result: +0.006 vs vanilla few-shot—showing that relevant positives + hard negatives noticeably improve classification.

## Triplet + FAISS
Include the train set in public solution. 
Train for extra epoch

# Ensemble (stack-as-you-go, per-rule ranks)

We ensembled incrementally, whenever a new strong model landed, we stacked it on top of the current blend rather than rebuilding from scratch. Scores from each submission were rank-normalized per rule to remove calibration drift across models (LLMs vs. NLI vs. retrieval). We then applied a hierarchical blending schedule: start with a 50/50 of the strongest retrieval (Embeddings LR+kNN) and the strongest NLI (DeBERTa-NLI), and progressively mix in FAISS/Triplet, Qwen-32B (train-only), DeBERTa-public, Qwen-3-8B (TTT), and LLaMA-3.1-8B (TTT). Each addition used a fixed mixing ratio; the previous stack was down-weighted accordingly (geometric decay), which naturally regularizes and prevents any single model from dominating. Finally, we clipped to [1e-6, 1−1e-6] . This strategy leveraged diversity across paradigms (retrieval, NLI, LLM-TTT) and kept calibration stable.


# Citation

https://www.kaggle.com/code/neibyr/lb-0-878-train-on-test-data-qwen2-5-0-5b
https://www.kaggle.com/code/nahidhossainredom/deberta-v3-base-3-epochs-lb-0-906
https://www.kaggle.com/code/datafan07/jigsaw-speed-run-10-min-triplet-and-faiss
