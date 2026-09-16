# 8th place solution

Competition: playground-series-s4e10
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s4e10/discussion/543772

Firstly, congratulations to @hardyxu52, @omidbaghchehsaraei, and @nadavcherry for placing in the top 3! It’s great to have you, @hardyxu52, back in the playground competitions. I hope to see more of you in future competitions.

### Data Preprocessing
I created five pipelines to train each of my base models. These pipelines included different types of preprocessing for the categorical features. In three of the pipelines, I used the original dataset, but only during training. Validation was done using only the competition dataset. For CatBoost, I treated all features as categorical, as this showed an improvement in both CV and public LB scores.

### Modeling
I used CatBoost, XGBoost, LightGBM (with three different boosting types: GBDT, DART, GOSS), histogram gradient boosting, gradient boosting, AutoGluon, and neural networks in this competition. Except for the neural networks, I trained all of these models on the five pipelines previously mentioned and saved their OOF predictions.

I also modified the CV strategy used by @omidbaghchehsaraei in [his notebook](https://www.kaggle.com/code/omidbaghchehsaraei/ensemble-modeling-for-loan-approval) to be the same as my other models and used some of his models in my final solutions. The neural networks in my solution were inspired by [this notebook](https://www.kaggle.com/code/paddykb/ps-s4e10-no-keras-no-loan-cv-0-963) by @paddykb.

### Ensembling
In my 8th place solution, I gave all the OOF files I had collected to AutoGluon with mostly default settings and let it handle the ensembling. This model achieved a CV score of 0.970887, a public LB score of 0.97329, and a private LB score of 0.96900. Unfortunately, I do not have a notebook for this on Kaggle, as I had to do this on my own computer. This result was obtained after 24 hours of training, which Kaggle notebooks do not allow.

However, I do have a [notebook](https://www.kaggle.com/code/ravaghi/loan-approval-prediction-8th-place-solution) on Kaggle for my other submission that achieved the same private LB score as my AutoGluon solution and would also place in the top 10. This solution consisted of ensembling OOF files using ridge and logistic regression, followed by another ensembling step using a weighted average approach. It’s worth noting that the previously mentioned AutoGluon solution used OOF files from 52 models, but in this multilevel ensemble approach, I found that reducing the number of OOF files improved CV and public LB scores. Initially, I tried using RFECV to select models, but I achieved better results by identifying the best models for the ensemble using a simple brute-force approach. In the end, I used 19 models in both ridge and logistic regression (though not the same 19 models in each).

### Results
The figure below shows the 10-fold CV score of all my base models as well as my ensembles. Note that the AutoGluon scores are not shown in the figure, but as I mentioned previously, the 10-fold CV score it achieved was 0.970887, which is the highest CV score I have achieved in this competition.


