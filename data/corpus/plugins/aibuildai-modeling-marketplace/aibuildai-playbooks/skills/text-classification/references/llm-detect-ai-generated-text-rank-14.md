# 14th Place Solution

Competition: llm-detect-ai-generated-text
Rank: #14
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470193

I have been working hard towards achieving Solo Gold for several years. 
This time, I am very happy to have reached Solo Gold, and I would like to express my gratitude for the many discussions and notebooks that have taught me along the way. 
Thank you.

I apologize for only having a simple solution prepared, butI would like to share it briefly.

# 14th Place Solution

My solution is based on the following notebook. 
[LLM DAIGT Analyse edge cases](https://www.kaggle.com/code/batprem/llm-daigt-analyse-edge-cases)
(special thanks)

While there are some minor adjustments, I made significant contributions with two main additions:
1. roberta-large-openai-detector [1]
2. Pseudo-Label

These additions have played a crucial role in enhancing the overall solution.

## roberta-large-openai-detector
I considered the addition of BERT model-based classification crucial to differentiate my approach from methods based solely on word frequency.

Among various experiments, this model achieved a score of 0.813 on the Leaderboard without requiring tuning. (private score → 0.839)

I experimented with various tuning techniques, but unfortunately, I couldn't manage to control overfitting.

So, I added the output without tuning as a feature to my model.

Since there is a limitation on the number of input tokens in the model, I used the first 512 tokens from the head and the last 512 tokens from the tail to obtain two sets of predictions.

[LLM DAIGT Analyse edge cases](https://www.kaggle.com/code/batprem/llm-daigt-analyse-edge-cases)

I trained LightGBM and XGBoost models independently using these BERT predictions as well as the word frequency features from the publicly available notebook.

While the LB score of each individual model falls behind the one in the public notebook, ensembling them with the predictions from the public notebook led to an improvement in the LB score. (LB Score 0.965, Private Score 0.930)


## Pseudo-Label
Considering the high LB score and along with the tendency for overfitting during training, I believe Pseudo-Labeling is effective.

I picked up outputs from the first-stage predictions in which I had confidence.

The outputs of the BERT model excelled at predicting automatically generated ones (label 1), and considering the limited number of automatically generated examples (label 1) in the training data, I utilized the top 10% for label 1 and the top 7.5% for label 0.

Furthermore, to avoid bias towards specific topics, I assigned Pseudo-Labels based on prompt_id.

While there are various small refinements, these two contributions, especially, made significant impacts.
((example) adjustment of training weight, noise data filtering, reducing features for faster processing, etc...)


[1] Solaiman, I., Brundage, M., Clark, J., Askell, A., Herbert-Voss, A., Wu, J., ... & Wang, J. (2019). Release strategies and the social impacts of language models. arXiv preprint arXiv:1908.09203.
