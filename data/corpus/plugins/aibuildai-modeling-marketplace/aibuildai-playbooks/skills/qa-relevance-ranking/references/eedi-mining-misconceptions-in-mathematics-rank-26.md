# 26th place solution(Public 14th)

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #26
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551406

Thank you for hosting a wonderful contest.  
Congratulations to those who placed at the top, and I would like to praise everyone who participated.
I would like to briefly share my solution.

## Overview
- Used Retrieve Model to obtain the top 160 misconception candidates for each `QuestionID_Answer`.
- Rerank Model evaluated the candidates and selected the top 25.
- Efficient training and inference achieved through Prefix Caching and Negative Downsampling with vLLM.

## Validation Strategy
- Group KFold (group = question_id)

## Training Phase

### Retriever
- Trained using Sentence Transformer based on a public notebook.
- Backbone: Salesforce/SFR-Embedding-2_R.
- Fine-tuned with LoRA.
- `num_retrieve`: 100.
- Negative samples downsampled:
  - For each `questionId_Answer`, randomly sampled 3 out of 100 examples for training.

### Reranker
- Leveraged out-of-fold (OOF) predictions from the Retriever stage.
- Extracted top K (K = 100) misconception candidates for each `QuestionID_Answer` using the retriever.
- Input: `question_text + correct answer + wrong answer + misconception`.
- Task: Output `Yes` or `No`.
- Negative samples were downsampled in the same way as the retriever.
- Prompt: 

```
<|im_start|>user
Is the incorrect answer caused by the misconception? Answer Yes/No.
<Problem>: {Question}
<Construct>: {ConstructName}
<Subject>: {SubjectName}
<Correct Answer>: {Correct_Answer}
<Incorrect Answer>: {Answer}
<Misconception>: {PredMisconceptionName}

<|im_end|>
<|im_start|>assistant
Answer:{label}
```

- Backbone: Qwen 32b AWQ or GPTQ.
- Fine-tuned with LoRA.

## Inference Phase

### Retriever
- Merged LoRA model for inference with vLLM.
  -  since vLLM embedding models do not natively support LoRA.
- Calculated similarities and selected the top K(K=160).

### Reranker
- Used vLLM.
- Forced output of either `Yes` or `No` token and output the corresponding token probabilities for reranking.
- Implemented Prefix Caching in vLLM for significant speedup:
  - Required inference for top K candidates per question-answer pair.
  - Prompt content up to the final misconception remained the same.
  - Prefix Caching allowed efficient inference for many candidates using the Rerank Model.

## What Doesn't Work
- Train models before quantization (without AWQ or GPTQ) and attempted custom quantization.
- Hard Example Mining:
  - Improved CV but not LB.
- QWQ model.
- Concatenating other high-ranking misconceptions in the Retriever.
- Concatenating other options.
