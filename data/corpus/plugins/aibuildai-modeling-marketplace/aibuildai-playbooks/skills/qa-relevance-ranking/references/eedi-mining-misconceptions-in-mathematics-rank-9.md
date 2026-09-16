# Private 9th (Public 7th) Place Solution

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #9
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551420

I must show my appreciation with respect to the eedi organization and kaggle team, for hosting such a stunning competition. This is my first participation of a kaggle competition, and I am very happy to take the gold medal.

### Overview
1. Synthetic data generation (GPT-4o)
2. Rationale generation (Qwen 2.5 32B Instruct AWQ with vllm)
3. Retriever (Qwen 2.5 14B Instruct)
4. Reranker (Qwen 2.5 14B Instruct and Qwen 2.5 32B Instruct with vllm)

### Synthetic data generation
We generated SubjectName, ConstructName, QuestionText, Correct Answer, Incorrect Answer by providing misconceptions to GPT-4o.
Next, start filtering:
- For misconceptions in `train.csv`, we filtered generated data with base model (training Qwen 2.5 14B Instruct without synthetic data scored 0.447 on LB).
- For misconceptions not in `train.csv`, do nothing.

### Rationale generation
Input the SubjectName, ConstructName, QuestionText, Correct Answer and Incorrect Answer into Qwen 2.5 32B Instruct AWQ to generate a briefly rationale (up to 50 words).
Prompt:
```python
Prompt = """Here is a question about {ConstructName} ({SubjectName}).
Question: {Question}
Correct Answer: {CorrectAnswer}
Incorrect Answer: {IncorrectAnswer}

Generate a brief rationale for the Incorrect Answer in a sentences, within 50 tokens. Describe the reasoning process that might lead to this choice, including any logical errors or misconceptions.

Keep the explanation concise and focused on the key points, and ensure it does not exceed 50 tokens
***Important***
1.There is no need to analyze the relationship between the wrong answer and the correct answer. Simply describe the rationale of the incorrect answer in a few words.
2.There is no need to have fields such as "Incorrect Answer:" and "Incorrect Answer is". Just give the rationale directly.
3.Output results in English.
"""
```

### Retriever
Following @sayoulala and his discussion https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519.
- Upsample original trainset question once for training, where misconceptions occur less than 3.
- Training retriever by contrastive learning and using sfr for hard negative sample mining.

### Reranker
- Binary classification reranker (14B), predict whether incorrect answer and misconception are matched. (we didn't use it finally because of unefficiency)
- Causal LM reranker (32B), using lm head take yes token logits as ranking score.Once trained, AWQ quantization is performed.

32B inference referer to https://www.kaggle.com/code/cdeotte/infer-34b-with-vllm.

| Model | Public LB | Private LB |
| --- | --- | --- |
| Retriever | 0.547 | 0.491 |
| Binary classification reranker | 0.606 | 0.542 |
| Causal LM reranker | 0.641 | 0.571 |


btw, we started training Reranker only a week before the competition cut-off and not much time for attempt😭
