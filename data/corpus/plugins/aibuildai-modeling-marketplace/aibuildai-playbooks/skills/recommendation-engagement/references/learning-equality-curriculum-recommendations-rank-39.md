# 39th place solution

Competition: learning-equality-curriculum-recommendations
Rank: #39
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/394896

Thanks to Kaggle and The Learning Agency Lab for this exciting competition, and thanks to kagglers and my great teammate @youjods .

# Summary
Our model is a 2-stage configuration(Retriever and Reranker).
We used models downloaded from [huggingface](https://huggingface.co/) and trained with the [sentence-transformers](https://www.sbert.net/) library.
The following points contributed significantly to the score:
- Using sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1 for backbone model
- Using the Retriever model as a backbone of Reranker training
- Using OnlineContrastiveLoss for Reranker

# Codes
- [Training](https://github.com/calpis10000/kaggle-lecr)
- [Inference](https://www.kaggle.com/code/calpis10000/lecr-calpis-exp037)

# CV Strategy
We used GroupKGold, which is handled differently depending on category.
- category=='source' topics were all used for training.
- The other categories are divided by GroupKFold keyed by channel, and the following data are used for validation.
  - 1 fold topics (as unknown channel topics for train data)
  - Other fold topics sampled same number of above fold (as known channel topics for train data)

Valid-scores were calculated for known and unknown channels, respectively.

# preprocess
We referred to @conjuring92 's discussion: [Topic Context Matters in Supervised Pipeline](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/376873)

topics: channel + language + level + title + description + context(title) + context(description) + children_title
content: kind + language + title + description+ text 

We have cut the title and discussion to some length. The cut length differs between Retriever and Reranker. (Reranker is shorter)


# Stage1: Retriever
We trained models from huggingface using sentence-transformer library, and we used MultipleNegativesRankingLoss.
We have tried various backbone models and the following conditions produced the best recall score.
- backbone: sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1
- epoch: 20
- batch_size: 128
- lr: 2e-5

Recall@100 score resulted in:
- 0.8745 for whole valid-data
- 0.93929 for known channel
- 0.80972 for unknown channel


# Stage2: Reranker
In stage2, top100 nearest contents were extracted for every topic using Reranker model.
Then we finetuned Retriever model using sentence-transformer library with OnlineContrastiveLoss.
We first trained with simple binary-classification, but OnlineContrastiveLoss boosted the f2-score as following:
- binary-classification: CV 0.4565, LB: 0.553
- OnlineContrastiveLoss: CV 0.5414, LB: 0.619

# Not worked
- Ensemble
  - ensemble improved our validation-score, but worsed LB-score.
- Other pretrained model (e.g. sentence-transformers/all-MiniLM-L12-v2)
- LightGBM Reranker
