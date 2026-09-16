# 15th Place Solution

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #15
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551501

First, I would like to extend my heartfelt gratitude to the organizers for hosting such a fantastic competition. This has been the most engaging contest I’ve ever participated in.

---

## Model Overview and Features

### Overview

My solution is quite straightforward. I used a retrieval model to select 25 candidates and re-ranked them using a single model. I applied this approach with different combinations of training data. For the 25 selected candidates, duplicate IDs had their scores averaged, while unique IDs retained their original scores.

---

### Score Results

- **Final Score:** 0.553 (private) / 0.617 (public)
  - **Retriever Model 1:** 0.513 (public) / 0.455 (private) - Qwen2.5-32B
  - **Retriever Model 2:** 0.515 (public) / 0.450 (private) - Qwen2.5-32B
  - **Reranker Model 1:** 0.610 (public) / 0.538 (private) - Qwen2.5-32B-Instruct-bnb-4bit
  - **Reranker Model 2:** 0.596 (public) / 0.536 (private) - Qwen2.5-32B-Instruct-bnb-4bit

---

### Synthetic Data Generation

Using GPT-4o, I provided 10 sample questions and the misconceptions I aimed to generate. Questions were then created by reverse-engineering from the misconceptions.

- **Misconceptions never appearing in competition data:** 2 types (985 samples × 2)
- **Misconceptions appearing only once:** 1 type (847 samples)

---

### Training Methodology

#### Retriever Training:
1. Extracted the top 200 candidates using the SFR-Embedding-2\_R pre-trained model.
2. Used Qwen25-14B to randomly select 30 hard negatives from the top 200 candidates and conducted contrastive learning. Extracted the top 200 candidates again.
3. Repeated the same process using Qwen25-32B.

This process might seem counterintuitive. However, the role of the retrieval model was not to identify a single correct answer, but rather to ensure that the correct answer was included within the top 25 candidates. Therefore, the model needed to adopt a broader focus to achieve this objective.

#### Reranker Training:
1. Extracted the top 25 candidates for each question using the retriever model.
2. Randomly selected 2-25 options from these candidates. If there was no correct answer, one option was replaced with the correct answer.
3. Conducted instruction tuning to predict the correct answer from the set of options. Varying the number of options helped capture the core nature of the task.

---

### Ensembling Method

The scores for overlapping IDs from the two prediction results were averaged, while unique ID scores were retained. Surprisingly, this approach worked well. Since the retriever and reranker models had different strengths due to their random negative sampling, this ensembling method boosted the score by 0.02.

---

## Final Thoughts

This competition allowed me to achieve my best-ever ranking. I stayed in the gold medal range for a long time during the contest, which was incredibly exciting. Though I narrowly missed the gold medal, I am eager to apply these learnings in future competitions.

Thank you all very much!

### Inference Code (Version 2)
https://www.kaggle.com/code/tkyiws/sub-embedder-reranker-ensemble-v3?scriptVersionId=212088687
