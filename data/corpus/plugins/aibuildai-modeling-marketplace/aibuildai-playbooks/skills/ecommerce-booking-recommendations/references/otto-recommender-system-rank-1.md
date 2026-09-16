# 1st place solution

Competition: otto-recommender-system
Rank: #1
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/384022

Thank you very much for organizing this fun competition.
The problem set up was relatively close to my actual work, and I was glad to learn a lot.

## Candidates
The average number of candidates is around 1200.
- visited aids in session
- covisitation matrix
  - use multiple versions with different weighing by type and aggregation period
  - apply covisitation matrix at multiple times like beam search
- NN that predicts subsequent aids
   - use multiple versions to create candidates and rerank features
   - NN structure is MLP or transformer (there was no big difference)
   - I tried to focus on samples that are not predicted well
   - I used the same embedding for x_aid and y_aid.
   - I used multiple aids in future as positive targets.
   - I used prediction target aid type information when calculating session embedding so that session embedding is adjusted according to the prediction target aid type.
   - some models are trained by using only non visited aids as targets to avoid overlapping information with revisitation based candidates and features.





## Reranker
### model

single LGBMRanker : LB 0.604  
ensemble of 9 LGBMRankers with different hyperparameters : LB 0.605  
I performed ensemble by averaging the predicted scores of the rankers.  
- I haven't tested if this is a better method than voting etc.

### features
- session * aid
    - rank by covisitation matrix at candidate generation
    - cosine similarity by NN at candidate generation
    - aid info in the session (when it appeared, what type it is, etc)
- aid
    - popularity of aids
        - It worked well when ranked
        - calculated by multiple time windows
    - ratio of types
- session
    - length
    - aid dupplication rate
    - ts between the last aid and the second last aid

about 200 features were created  
select about 100 features for each target by lgbm gain importance to reduce memory usage  

#### negative sampling rate
clicks : 5% 
carts  : 25%  
orders : 40%  
I set these values so that the training data can be handled by my machine (the data size is around 35GB for each).

## Cv strategy
I followed radek's set up. https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991  
I can get almost perfect correlation between local validation and LB.  
For quick iteration of improvements, I conducted experiments by training with 5% of the data and evaluating with other 10% of the data.

## ablation study
ablation study by local validation.  
Information that is involved in both candidate generation and reranker features is removed from both.

|  condition  |  clicks_recall@20  | carts_recall@20 | orders_recall@20 | weighted_recall@20 |
| ----------- | -----------------  | --------------- | ---------------- | ------------------ |
|  my solution (LB604)             | 0.556607 | 0.436375 | 0.669644 | 0.588359 |
|  without visited aid             | 0.555677 | 0.435616 | 0.666456 | 0.586126 |
|  without covisitation            | 0.547493 | 0.430180 | 0.665553 | 0.583136 |
|  without nn                      | 0.544811 | 0.429904 | 0.666004 | 0.583055 |
|  without aid feats               | 0.550472 | 0.433442 | 0.666275 | 0.584845 |
|  without session feats           | 0.555922 | 0.435805 | 0.669734 | 0.588174 |
|  only single nn                     | 0.532279 | 0.410148 | 0.564768 | 0.515133 |
