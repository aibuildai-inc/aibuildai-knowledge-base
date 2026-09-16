# 17th place solution

Competition: otto-recommender-system
Rank: #17
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/383121

First, thanks to OTTO and Kaggle for organizing this great competition.

Here I would like to share my part.

# Overview
- My part
    - Two-stage recommendation 
    - 100-200 candidates per session
    - 100 features for ranking model
    - CV strategy
        - For local validation, train by week3 and validate by week4
        - For submission, train by week4
    - Score -> public: 0.598, private: 0.597
- Team Solution
    - Ensemble the outputs of my model and those of my teammates' models with rank vote
    - Score -> public: 0.601, private: 0.600

# Candidates generation
Generate candidates by following methods. 

Median numbers of candidates per session are 100 for clicks, 180 for carts and 200 for orders.
- Previously actioned
- Co-visitation
    - matrix
        - action2action
        - action2click, action2cart, action2order
        - action2buy
        - buy2buy
    - weight
        - type
        - recency
        - time interval
- word2vec (use Gensim Word2Vec)
- node2vec (use PyTorch Geometric Node2Vec)

# Feature engineering
Around 100 features were created to train models. 

I think my features are not special compared to other competitors, but some examples are given below.
- CF score
    - candidates score
    - candidates rank
- Item
    - number of {type}*
    - ratio of number of {type} to number of actions
    - average number of {type} in one session
    - probability of {type} after {type} in one session
- User
    - session size
    - unique item number
    - ratio of unique item number to session size
- Item x User
    - number of {type}
    - number of {type} weighted by recency
    - ratio of number of {type} to session size
    - elapsed time from last {type}
- Item similarity
    - cosine similarity of embeddings by word2vec and node2vec
        - last aid to candidate
        - average of last 3 aids to candidate
        - average of last 5 aids to candidate

*{type} denotes clicks, carts or orders.

# Ranker model
- I used LightGBM ranker
    - objective: lambdarank
    - boosting_type: gbdt

# Ensemble
- For each type, ensemble the output of two LightGBM rankers with slightly different candidate generation
    - Score -> public: 0.598, private: 0.597
- After that, ensemble the outputs of my model and those of my teammates' models with rank vote
    - @dehokanta & @zakopur0 model
        - Score -> public: 0.598, private: 0.597
        - Details -> https://www.kaggle.com/competitions/otto-recommender-system/discussion/383493
    - @t88take model
        - Score -> public: 0.594, private: 0.594
        - Details -> https://www.kaggle.com/competitions/otto-recommender-system/discussion/382886
    - Final score -> **public: 0.601, private: 0.600**

# What worked
- Item similarity between last aid and candidate by word2vec contributes significantly. This feature boosts my score 
+0.003.
- Inspired by [this paper](https://arxiv.org/abs/1804.04212), I tuned hyperparameters of word2vec (such as epochs, window, ns_exponent,,,) and this increased my score +0.001.
- I used different N of co-visit candidates based on recency and gained +0.0015. Take action2click matrix as an example, 
    - For last aid in a session, join top 50 
    - For aids within 30 minutes of last action, join top 20
    - For aids older than 30 minutes, no candidates are joined

# What did not work
- In a recent tabular competition I participated in (H&M and Amex), LightGBM dart was better than gbdt. So, I tried dart but accuracy did not improve. It just made training time longer.
- I tried different models (LightGBM classifier, CatBoost classifier, CatBoost ranker) but LightGBM ranker was best in my case.

# Environment
Google Colab Pro+

# Acknowledgement
Thank @radek1 [introducing Polars](https://www.kaggle.com/competitions/otto-recommender-system/discussion/366194). I did not know Polars until taking part in this competition. Polars helped me to accelerate experiments and try many ideas.

Thank @cdeotte for sharing very helpful knowledge and codes. My co-visit matrix candidates generation is totally based on [his notebook](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575).
