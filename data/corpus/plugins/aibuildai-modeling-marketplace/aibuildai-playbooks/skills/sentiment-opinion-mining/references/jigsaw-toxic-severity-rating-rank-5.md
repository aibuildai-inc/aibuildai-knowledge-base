# 5th place solution

Competition: jigsaw-toxic-severity-rating
Rank: #5
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306390

I decided to focus on maximizing only validation score in the last month of the competition. Public LB did not seem quite useful because it had only 5% of data and there was not correlation between validation score and public LB score. 

I used 2 datasets for training models: Ruddit dataset, Toxic comments dataset (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). I tried other datasets, but they did not improve my validation score at all.

I used  “validation_data.csv”  for validation only.

## Ruddit dataset / models

	Ruddit dataset includes tuples (A, B, C, D) where “A” is the least toxic and “D” is the most toxic. Therefore, I used those tuples to create pairs (X, Y) where X is less toxic than Y. 

Then I trained 2 models with MarginRankingLoss:

1)	“roberta-large” with linear layer 
2)	“unitary/unbiased-toxic-roberta” with linear layer

	I used layerwise learning rate decay (from 1e-5 to 1e-4). Training was made by using 5-folds cross-validation. Each model was trained with 3 epochs, after each epoch I saved the model with the best accuracy.


##  Toxic comments dataset / models

I used that dataset to train a regression models. Data preparation: 1) remove items that have the same text in validation_data.csv 2) compute toxic target (y = toxic*0.32 + severe toxic*1.82 + obscene*0.16 + threat*1.5 + insult*0.64 + identity_hate*1.5) 3) create a train dataset with positive toxic target and negative toxic target.

After data preparation I trained 3 models:


1)	TfIdf + Ridge. I used cleaned text and 4 extra features: length, portion of punctuation in text, portion of chars with uppercase in text, count of swear words (these features slightly improve score on validation_data). I trained models on 8 folds with grid search of hyperparameters (~~“min_df”, “max_df”, ~~ “ngram_range”, “alpha”, “fit_intercept”) and chose the model with the highest score on validation_data.
2)	Regression model based on “unitary/unbiased-toxic-roberta” with linear custom layer and MSELoss. I trained that model by using 5-folds cross-validation with 5 epochs. The model with the minimal loss on validation fold was saved. Models were trained with layerwise learning rate. 
3)	Regression model based on “unitary/unbiased-toxic-roberta” with attention custom layer and MSELoss. I trained that model by using 5-folds cross-validation with 5 epochs. The model with the minimal loss on validation fold was saved. Models were trained with layerwise learning rate.

## Models ensemble

After training the models I used optuna to weigh models in ensemble. Steps:
1) normalizing logits of models using min, max scaling 2) weighing models using weights 0 or 1. 
After some experimentation I got maximum validation score - 0.709 with public LB score 0.819. 



Final ensemble – sum of the models:
|**№**| **Model** | **Dataset**  | **Count of models** |
| --- | --- | --- | --- |
1 | roberta-large with MarginRankingLoss | Ruddit |         2
2 |unitary/unbiased-toxic-roberta with MarginRankingLoss | Ruddit  |         3
3 |Tfidf + RIDGE | Toxic comments  |         3

4 |unitary/unbiased-toxic-roberta with custom linear layer with MSELoss  | Toxic comments  |         3
5 |unitary/unbiased-toxic-roberta with custom attention layer with MSELoss  | Toxic comments  |         2




## Notebook

Link to my submission - https://www.kaggle.com/alexander1980/best-cv-ens?scriptVersionId=87213915

P.S.: Sorry for duplications in my code.
