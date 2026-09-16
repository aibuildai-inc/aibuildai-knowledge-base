# 28th place solution

Competition: riiid-test-answer-prediction
Rank: #28
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210041

I want to thank the hosts for organizing a meaningful competition that reflected the critical constraints of real-world difficulties (time series API, limited RAM, and inference time). Also, I thanks my teammate NARI who worked hard with me until the end of the competition.

# Overview



Our final solution is an ensemble of Catboost and Transformer. Although Transformer could not create a high CV model in time like the top teams, it contributes a lot to boosting the stacking model score. Our solution's validation strategy and feature engineering pipeline were heavily influenced by tito's notebooks ([validation strategy](https://www.kaggle.com/its7171/cv-strategy), [feature engineering](https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering)). We definitely would not have reached this score if he had not been shared early in the competition.

# Features

We extracted 160 features for training Catboost. 

The main features are as follows. The detailed contribution to the score of each feature has not been confirmed, but the feature importance by CatBoost is available [here](https://github.com/haradai1262/kaggle_riiid-test-answer-prediction/blob/main/notebook/check_feature_importance.ipynb)

- Content-related features
    - Aggregation features
        - mean, std for answered_correctly
        - mean for elapsed time
        - We also used aggregation in records only for the second and subsequent answers to the same question by the user.
    - Word2vec features

        - We extract Word2vec features using each user's answer history as sentences, each question_id and its associated part, tag_id, and lecture_id as words.
        - We also extracted cases in which users' answer histories for only correct answers, and only incorrect answers were treated as separate sentences.
        - The similarity and clustering by the word2vec feature was also used for other feature extraction.

    - Graph features

        - Based on the users' answer histories, we created a directed graph, whose nodes are question_ids and edges' weights are defined by the number of users' transitions, and used the following node feature as question features.
            - node metrics (eigenvector_centrality, betweenness_centrality, trophic_levels [NetworkX document](https://networkx.org/documentation/stable/reference/algorithms/centrality.html))
            - node embedding (DeepWalk [paper](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf), struc2vec [paper](https://arxiv.org/pdf/1704.03165.pdf), [implementation](https://github.com/shenweichen/GraphEmbedding))
            - SVD for adjacency matrix
- User history features
    - Answer count,  Correct answer rate
        - We use the user's correct answer rate and the correct answer rate in the last N times.
        - We also used the correct answer rate in the first ten times, one day and one week.
    - Time-related
        - difftime (timestamp - previous_timestamp) worked especially well
        - We used some features related to difftime, including difftime with the most recent 5 step timestamp and their statistics.
    - Question-related
        - Whether the user has answered the target question in the past or not, and the number of times the user has answered the question in the past were worked.
        - We also use correct answer rate for questions in the same cluster (clustered by k-means using similarities based on Word2vec features described above, with the number of clusters set to 100)
        - We also used some features related to the similarity between the target question and the recent questions or these parts (the similarities were based on Word2vec features described above).
    - Part-related
        - The correct answer rate in the part of the question users are answering was worked.
        - We also used the count and correct answer rate of each part.
    - Tag-related
        - The correct answer rate for each tag of the user is kept. The statistics (max, min, mean) of the user's correct answer rate for each tag in the question being answered are used.
    - Lecture-related
        - The number of answers from the user's most recent lecture worked best for lecture-related features.


# Model


### Catboost

- Data split: tito CV
- Input
    - 160 features
- CV: 0.803~0.804, PublicLB: 0.801~0.802

### Transformer (SAINT-like model)

- Data split: tito CV
- Input (sequence length 120)
    - Encoder
        - question_id, part, tag, difftime
    - Decoder
        - answered_correctly, elasped_time
- CV: 0.797~0.798

# Stacking

- Model: Catboost
- Data split: k-fold CV (k=9) of 2.5M records
- Input
    - Predictions of Catboost (6 models, random seeds) and Transformer (2 models, a slight variation of hyperparameters and  structure)
    - Top 75 higher importance features based on feature importance of CatBoost
- 3 Seed average
- CV: 0.810, PublicLB: 0.807, PrivateLB: 0.809

---

Thank you for your attention! Our code is available in

- [https://github.com/haradai1262/kaggle_riiid-test-answer-prediction](https://github.com/haradai1262/kaggle_riiid-test-answer-prediction)
- [Inference (kaggle notebook)](https://www.kaggle.com/haradataman/riiid-28th-solution-inference-only)
