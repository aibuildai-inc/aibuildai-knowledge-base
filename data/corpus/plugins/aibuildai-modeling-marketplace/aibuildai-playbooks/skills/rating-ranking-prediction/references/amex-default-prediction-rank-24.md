# 25th place solution

Competition: amex-default-prediction
Rank: #24
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/347880

## Intro
As a senior AI researcher in a start-up company called **pecan.ai** I mostly deal with similar types of transational data. I used my work experience to get a smooth start. Also a huge thanks to my bosses for providing me with computational power :) It will absolutely paid off by amount of usefull directions I learned from such a warm community :)
 

## Feature Engineering:

We used a couple of versions of feature engineering (it was kind of chaotic) and not all techniques described here were applied for all models. It’s done for couple of reasons:

- Make datasets slightly more diverse
- Reengineering features after some time to be sure that there are no bugs.

  as a basis, we used @raddar 's dataset


### 1. Pre-flattening - engineering done on sequences:

We started with “after-pay” features and then advanced into feature interaction: We iterated over all possible combinations of the numerical features, calculated the difference between each pair of features and filled NAN with 0, if the linear correlation was significantly higher than either one of the standalone features then this difference, became a new standalone feature and was added to the model, e.g. `P_2–B_3`.
We also used amex metric instead of linear correlation.

In one of datasets categorical features was one-hot encoded into multiple binary sequences.

### 2. Flattening - reducing to a single value:

Basics - mean, min, max, std, first, last (indeed).

Also - middles: `mean(1:12)`, `mean(2:13)`, `mean(2:12)`. The reason - post-flattening of `last-mean(1:13)` makes less sense because `mean` already has `last` inside, so `last-mean(1:12)` sounds as more correct solution. In practice, I do not know to measure how it is usefull.


### 3. Post-flattening - operating on created features:

* Last - first, last - middle, middle - first, relative_pos: (last - min) / (max - min)

* Row features: count of NANs, sum of normalised values, difference between normalised values of P,D,S,B,R e.g. sum(normalisedP) – sum(normalisedD)

### Specialized features
* feature value momentum
* linear predictions of each feature 180 days into the future using the last 3 months only
*  days since last NAN value
* (last value – prev value ) / (last_date – prev_date) 
* and some other variation of the above features.

Feature Engineering was done in Python and C# .Net:

### SequentialEncoder
We also created sequential features described in [this](https://www.kaggle.com/code/pavelvod/27-place-sequentialencoder?scriptVersionId=104154431) notebook (they was very significant).  
 
 
 ## Feature Selection
 
 We tested some simple methods, but they was reducing our score. We decided that loosing 4th point  of CV does not worth it, we are still able to run a model with 3.5k features dataset (our biggest one), so risk will not paid off.
  
## Modeling:


### GBDT:
LightGBM Dart, CatBoost, XGBoost 

### Tabnet
We did not manage to get useful results (all ensembles almost nullify its contribution).

**But!** then we tried the trick I tested a couple of years ago. We took our best model (lightgbm dart) and calculated **shap values** in an out-of-fold manner. Back then I called it self-supervised pre training, but technically it's a feature transformation. As a result we got a dataset which is much easier to digest for NN models, because it was extracted with GBT. Then I trained tabnet using this dataset and achieved **0.797** on LB. That solution not so diverse, as straight-forward tabnet model. But Tabnet learned predictions in different manner than GBT, so it still was very useful for the ensemble. 

You can find more detailed explaination [here](https://www.kaggle.com/code/pavelvod/gbm-supervised-pretraining) (from some previous competition)


 ### Node
But my greatest excitement was trying the **NODE** model for this competition. It achieved good results - I do not think my results were optimal, I believe If I would finetune it more - it would achieve better results. But It was probably the heaviest tabular model I ever tried - my 12 gb GPU (thanks **pecan.ai**) was screaming with only 64 batch size.

### Public models:
We used TF transformer by @cdeotte with 3 seeds.

### Sequential models
We used `tsai` library which has plenty of sequential models, such as LSTM, 1D-CNN and many many more their advanced versions and implementations. Nothing was found usefull for us.

### Training parameters
We used logloss for all our models both as loss and stopping metric. We tried FocalLoss and RankingLoss, but they not worked for us.
We used 5-fold CV.
Almost all models was trained and averaged with 4 seeds (3 seeds was stratified by target and 4th was stratified by target and P_2_last - idea by @bogorodvo 

### Ensemble

As an EnsemblerClassifier, we developed an iterative process, known as a forward selection process, which was able to find the optimum weights to maximize AMEX score.
We also tried other methods which optimised LogLoss, but they was not good enough.

## Worth trying

Couple of key ideas that worked well for other people was also in our plans, but we somehow we gave up on this ideas. I will not mention them.


The only thing I regret that I did not tried (and still not found someone tested it) is Tabnet self-supervised pretraining on the whole train and test data and then finetuning on the train data. I started to regret after seeing the knowledge distillation solution of @cdeotte
