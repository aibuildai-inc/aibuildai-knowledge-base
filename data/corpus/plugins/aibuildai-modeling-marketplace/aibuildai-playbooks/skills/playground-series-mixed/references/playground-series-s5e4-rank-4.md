# Rank 4 approach - lots of features, lots of simple models and a ridge blend!

Competition: playground-series-s5e4
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s5e4/discussion/575782

Hello all,

Thanks to Kaggle for a good CV-LB correlated regression tabular assignment! Also thanks to my fellow participants for their contribution to the forums as well! I am happy to present my solution for rank 4 here as below-

# CV scheme
I used a simple 10-fold cv scheme as below-
`Kfold(10, random_state = 42, shuffle = True)`

# Overall model design

The figure below describes my overall model effort for this competition - 


<br>As indicated in the figure above, I opted for a simple pipeline with brute-force feature engineering and simple models blended with a ridge regression. The below sections illustrate the process in detail- 

## Feature engineering
- I opted for simple interaction features featuring bigrams, trigrams, 4-grams, 5-grams, 6-grams and 7-grams across the dataset. I converted numerical columns to string datatypes and included them in the interactions. 
- I also used a few random numeric column integrations involving numeric columns in the data. I also created a feature for the number of nulls across a row, featuring the starting columns only
- I dropped a few features that had extremely low cardinality/ quasi constant features. This step did not have any meaningful CV impact though

## Model training

I trained **382 single models over the month** mixing and extracting features from my data store explained above. I opted for a simple blend involving common boosted tree models involving - 

#### Xgboost 
- performed well in the initial stage of the competition, with lower features in the feature store
- as the number of features increased, this model took longer to train and CV scores were sub-optimal to LGBM
- increasing tree depth proved to be a useful tactic for cv-lb boost
- learning rate at 0.005 - 0.0075 was the best range for my models throughout
- early stopping rounds in the range of 550-600 balanced my hardware constraints with suitable models 
- training models on an A6000 GPU was a good experience. I used 128 GB RAM throughout 

#### LightGBM
- the best single model choice - I used both gbdt and goss options and both performed extremely well individually. My 10-best single models are all LGBM gbdt and goss options 
- tree depth of -1 was an odd choice, but a good choice 
- learning rate of 0.01 and below was a good choice 
- tuning more parameters turned out to be a sub-optimal cost-benefit exercise, so I preferred to hand tune parameters and build models 
- I trained these models on an A6000 Ada 128 GB RAM and the results were good

#### Catboost
- did not perform extremely well on the CV scheme, but provided needed diversity to the ensemble
- memory issues surfaced with a wider feature set, so I used this model option with smaller number of features (< 350- 400)
- learning rate of 0.01 and tree depth of 12 proved to be good options for me
- I preferred an A6000 GPU and 256 GM RAM for these models

#### Autogluon
- I elicited feature importance using my single tree models and shortlisted 25-100 repeatedly important features across most model options
- I prepared 8 feature sets using these *prime importance features** and trained Autogluon models 
- I used L4 GPU on Colab and 12-18 hour runtimes for the AutoML run

## Ensemble blend
- I choose a simple ridge model to blend my single models for a submission
- A simple StandardScaler ensured that all predictions were scaled before feeding into the ridge model

## Post-processing
- I chose to round off all my predictions to the nearest target value. 
- This entailed a very small CV score improvement and the same score on the public leaderboard though!

## CV-LB details
As indicated earlier, I enjoyed a near-perfect CV-LB relation all throughout with the below CV scores across single models and the ensemble-

### Single model details

| Model algorithm | CV score  | Public Leaderboard | Number of features |
| --- | --- | -------- | ----------- | 
| XgBoost |  11.89035 - 12.76437 | 11.91435 - 12.79546 | 717 - 10 |
| LightGBM gbdt|  11.80357 - 12.87457 | 11.83467 - 12.91435 | 925 - 10 |
| LightGBM goss|  11.83367 - 12.84623 | 11.85891 - 12.87594 | 817 - 10 |
| Catboost |  12.00325 - 12.86532 | 12.00134 - 12.89734 | 385 - 10 |

### Final Submission details

| Model algorithm | CV score  | Public Leaderboard | Private Leaderboard | Number of features / model components|
| --- | --- | -------- | ----------- | --------- |
| Ridge |   11.61414226 | 11.64459 | 11.54182 |382 |
| Ridge with post-processing| 11.61414171 | 11.64460| 11.54182 | 382 |

# Key takeaways
- Simple models have lots of power and should be used well to elicit a good score
- Participating in competitions where CV-LB relation is strong feels good and is highly enjoyable!
- Building reusable code is of great value! I reused my pipeline from [Playground S5-E2](https://www.kaggle.com/competitions/playground-series-s5e2/discussion/565542) almost completely to best effect!
- A good GitHub repo with relevant code is akin to gold!
- Data stores are valuable when one wishes to iterate through features and build models quickly and effectively
- I did not use any public code in my pipeline this time as I thought it would be great to test my own indigenous models and experiment thereby. I think my idea prevailed!

# References

- Selected single models - https://www.kaggle.com/datasets/ravi20076/playgrounds5e4modelsv1
- Selected model CV scores - https://www.kaggle.com/code/ravi20076/playgrounds5e4-modelscores-v1

# Final comments

Sincere thanks to Kaggle and my fellow participants for a great experience and best wishes for a successful journey ahead!
All the best, happy learning and regards!

Ravi Ramakrishnan
