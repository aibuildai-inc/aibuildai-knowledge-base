# 17th solution - seed selection, NN, ensembling..

Competition: porto-seguro-safe-driver-prediction
Rank: #17
Source: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44655

Congratulation to all winners - it was strange, but very trainable competition ) Thanks to all who share their approaches - I have learnt a lot from @CPMP, @Tilii, @Olivier, @Scirpus, @Andy Harless and many others kagglers as well.

My solution is not unique - just ensemble of high performance diverse models. But I want to share some approaches I use, may be it will be helpful to somebody.

**Cross-Validation**

I use standard 5-Folds Stratified split, but at first I try to find "magic" seed(s) that split data to folds with more-or-less similar internal statistical characteristic of data to reduce intra-fold variance. I discussed this approach here (https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/42785#240324). In addition to it I use averaging of 3-run per each fold with the same model but different (random!) seeds.

This scheme can be extended to @CPMP version of CV with random seeds (https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44614#251014) by changing [0,1,2,3,4] :) to seeds with smallest variance. I still not sure that my approach is more robust to bias/noise that other, possible it is my luck that I don't catch bias, but... in this competition it helps me to get gold )

// I checked (and submitted) several other models with another seed - all of them had worst CV/LB.

**Model to use**

Nothing special - XGB, LGB, CatBoost, RGF (with 2 or 3 base set of features):  
- no "calc" / OHE / remove unimportant features  
- no "calc" / LabelEncoded (for CatBoost / LGB-Cat)  
- no "calc" / OHE / +some base FE (iteractions, counters, ....)  

FFM (added some value to ensemble)

Thank you @Scirpus for https://www.kaggle.com/scirpus/libffm-generator-lb-280,   
@Chia-Ta Tsai for https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43741,    
@Oscar Takeshita for https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43741#245562 )

NN (based on Keras/TF)

I use my own version of Embedding NN and try several architecture of model:

- Concat All Ind/Reg/Car category features to three groups 
- Embed groups separately
- Merge with Base columns
- Merge with Calc (or drop Calc)
- Dense*N

===

- Concat All Ind/Reg/Car/Calc features to 4 groups
- Dense over each group
- Merge output
- Dense*N

// very nice version of embedding NN you can find here (https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44601)  

Hyperparameters to tune:  
- embed size (it's strange, but best result for me was size=2 of all category features)  
- dense: one or two layers with 8-64 cells and dropout 0.25-0.1

NN had high weight in my ensemble.

**Feature selection**

To reduce dataset (after adding some pack of features) I use the feature selection method:  
- at first run lgb (it is fast!) with small depth (5) in loop and remove features with 0 importance  
- then run shuffling selection of features using lgb (again it is fast!)  
- final tuning - remove features with 0 importance on the new dataset  

// more detailed description you can find in coments below

I have tried RFECV/Boruta (you can remember several nice discussion/kernel about this topic), but hold out using it (remove importance/shuffling is much faster with very nice result)

**Ensemble**

I use 2-step ensembling method. At first stage I stack all high-performance model of the same type (LGB-XGB/FFM/NN/...).  Then I stack all L2 models together.

The ensembling procedure:  
- use LRCV to obtain correct C for current set of models  
- create cor_matrix for prediction (I try pearson/spierman)  
- get 2 models with smallest corr_coef and combine it  
- iteratively add model with smallest corr_coef to current ensemble (drop model if adding don't get improvement in score)  
- at the end build LR with C on selected models  

// I try to use NN / LGB as stacker but LR was better (at least in my case)


**What don't fly**

* Clustering and all method of dimensionality reduction
* KNN and all attempt to play with similarities
* Feature Engineering...


**Main lesson from this competition**

1. Robust CV is 50% of success ) // remember Mercedes )
2. Trust your CV!!! (I have submit to 12 place with better CV but not to choose it as I think it's overfit)
3. Correct Ensemble scheme is nice ) : Use same folds / Use High performance diverse models / Be careful about overfitting/leakage.
4. Don't looking for best n_rounds. Averaged Folds test predictions works well!

Thank to all! Good luck and happy kaggling!
:)

@kruegger  
P.S. you can upvote if you want )
