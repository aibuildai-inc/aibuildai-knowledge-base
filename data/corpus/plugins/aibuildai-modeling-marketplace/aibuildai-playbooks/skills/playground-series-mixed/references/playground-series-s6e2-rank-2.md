# 2nd place solution --- Avoid leaks and overfitting

Competition: playground-series-s6e2
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/2nd-place-solution-avoid-leaks-and-overfitting

I'm very happy to have come in second place.

I took into consideration the high-quality discussions and code from the community. I'd like to thank all the participants.

I particularly took reference from the following three code and one discussion.

Thank you very much.

[https://www.kaggle.com/code/yashanathaniel/catboost-oof-encoding-heartdisease-0-95379](https://www.kaggle.com/code/yashanathaniel/catboost-oof-encoding-heartdisease-0-95379)

[https://www.kaggle.com/code/yashanathaniel/authentic-ensemble-10-catboost-0-95391](https://www.kaggle.com/code/yashanathaniel/authentic-ensemble-10-catboost-0-95391)

[https://www.kaggle.com/code/omidbaghchehsaraei/the-best-solo-model-so-far-realmlp-lb-0-95397?scriptVersionId=299097960](https://www.kaggle.com/code/omidbaghchehsaraei/the-best-solo-model-so-far-realmlp-lb-0-95397?scriptVersionId=299097960)

[https://www.kaggle.com/competitions/playground-series-s6e2/discussion/672393](https://www.kaggle.com/competitions/playground-series-s6e2/discussion/672393)



I would also like to express my gratitude to the Kaggle organizers for giving me this opportunity. Thank you very much.


# Content

From the EDA stage to the end, I couldn't decide which was the correct choice:

Should I treat the target statistics as global features and calculate them outside the CV loop, or should I target encode them inside the CV loop as usual?

I noticed that the adversarial AUC during EDA showed almost no difference between the distributions of the train and test data (Train vs Test AUC (XGB): 0.5017 ± 0.0013). 

Therefore, I thought it was highly likely that calculation outside the CV loop, where the CV would be higher, would be effective. However, if this judgment was incorrect, the CV score would be falsely high due to leakage. 

In other words, I was unable to determine how to handle the difference of 0.5017 - 0.5 = 0.0017.

Therefore, I decided to continue creating calculation models for inside the CV loop and outside the CV loop.

The final selection would be made by submitting one model from each category.


The results of Private LB showed that the model that encodes the target within the CV loop and calculates the target statistics is slightly better.

However, the difference is so small that it is still too small to say for sure. The answer may be that either is fine. 

It may have been a good decision not to mix the two when stacking.


■Final submitted model results

-Model calculated within the CV loop (2nd place model):

CV score:0.955759

Public LB:0.95394

Private LB:0.95535


-Model calculated outside the CV loop:

CV score:0.955774

Public LB:0.95394

Private LB:0.95534


# The 2nd place solution is as follows:

The number of models examined was 105. The results were selected for stacking using NN as follows:


   105models(xgb,lgbm,catboost,realmlp,tabm,pairwise ranking auc NN) 

-> select 50models(Selecting the model to calculate inside the CV loop) 

-> select 15models (Only average results of multi-seed are selected) 

-> select 4models (Model select based on correlation coefficient of 0.9999.) 

-> 6models (Added two models by Rank Transformation. 2 catboost models, 4 realmlp models) 

-> NN stacking


## notebook links:

★Final notebook using Rank transformation and stacking with NN

[https://www.kaggle.com/code/satokin13m/s6e2-nn?scriptVersionId=300608701](https://www.kaggle.com/code/satokin13m/s6e2-nn?scriptVersionId=300608701)

In this notebook, a total of six results are stacked using NN. 

These six results are the results of narrowing down to four using correlation coefficients in a subsequent notebook, and the results of applying rank transformation to two models without rank transformation.


Model used: NN stacking

Cross-validation: StratifiedKFold 5 splits


★This notebook selects 4 models that are effective for stacking from 50 models that are the result of target encoding within the CV loop.

[https://www.kaggle.com/code/satokin13m/s6e2-nn?scriptVersionId=300590084](https://www.kaggle.com/code/satokin13m/s6e2-nn?scriptVersionId=300590084)

To increase versatility, I decided to consider only the averaged results of the multi-seed calculations. 

This narrowed down the 50 results to 15.

Furthermore, I removed results with similar correlation coefficients to accommodate the NN, narrowing it down to four results.

This notebook also uses NN stacking, but the key is the method of feature selection. I perform the selection in the following three steps:

Step 1: Removing highly correlated features (threshold=0.9999)

Step 2: Forward Selection

Step 3: Backward Elimination


★The three notebooks and self-made datasets that created the six final results (the two that were rank transformed are the code above, so the remaining four results)

●notebook1

[https://www.kaggle.com/code/satokin13m/s6e2-ynong](https://www.kaggle.com/code/satokin13m/s6e2-ynong)

This notebook is the most important one. This notebook is a copy of one I ran on my local PC.

Since I ran out of Kaggle's GPU available time for a week, I ran it on my local PC.

Since I haven't run it in the Kaggle environment, I can't guarantee that this code will work perfectly.

This is a modified version of the following notebook, with the target encoding moved into the CV loop.

[https://www.kaggle.com/code/yashanathaniel/authentic-ensemble-10-catboost-0-95391](https://www.kaggle.com/code/yashanathaniel/authentic-ensemble-10-catboost-0-95391)

Special thanks to the author ([https://www.kaggle.com/yashanathaniel](https://www.kaggle.com/yashanathaniel)).

Without this notebook, I would have overlooked the rank transformation without even thinking about it. Thank you so much.


Model used: CatBoostClassifier

Cross Validation: StratifiedKFold 5 splits


●notebook2

[https://www.kaggle.com/code/satokin13m/s6e2-multi-seeds-realmlp?scriptVersionId=296465842](https://www.kaggle.com/code/satokin13m/s6e2-multi-seeds-realmlp?scriptVersionId=296465842)

I ran the calculation in a Kaggle environment using a model that used the results of optimizing the hyperparameters of realmlp using optuna on my local PC. 

I was delighted when I found '1-auc_ovr' in val_metric_name.


Model used: RealMLP_TD_Classifier

Cross Validation: StratifiedKFold 5 splits


●notebook3

[https://www.kaggle.com/code/satokin13m/s6e2-multi-seeds-realmlp?scriptVersionId=299135216](https://www.kaggle.com/code/satokin13m/s6e2-multi-seeds-realmlp?scriptVersionId=299135216)

The hyperparameters in the following notebook were superior to those in notebook2. I changed n_ens from 8 to 20, but left the rest unchanged.

[https://www.kaggle.com/code/omidbaghchehsaraei/the-best-solo-model-so-far-realmlp-lb-0-95397?scriptVersionId=296615900](https://www.kaggle.com/code/omidbaghchehsaraei/the-best-solo-model-so-far-realmlp-lb-0-95397?scriptVersionId=296615900)

We would like to thank the author ([https://www.kaggle.com/omidbaghchehsaraei](https://www.kaggle.com/omidbaghchehsaraei)).


Model used: RealMLP_TD_Classifier

Cross Validation: StratifiedKFold 5 splits


●Dataset

[https://www.kaggle.com/datasets/satokin13m/s6e2-myhillclimbdata](https://www.kaggle.com/datasets/satokin13m/s6e2-myhillclimbdata)

The CSV file containing the results of the study was created as a dataset. 

It was originally created for hill climbing, so the dataset name includes "hillclimb."

The necessary results were read from this dataset and then selected and stacked.



This concludes the explanation of the 2nd place solution.


# another stacking solution

From here on, I'll write about a different stacking solution (so I'll minimize the code I share, sorry).

In addition to NN stacking, I also tried stacking during hill climbs.

This model performs a hill climb on the results of a model that calculates target statistics within a CV loop, or on the results of all models.

I'm not sure if it's actually correct to call this hill climbing, but this is an algorithm I created based on the name hill climbing.

I thought that simple hill climbing would result in overfitting. 

I considered combining it with cross-validation. 

This algorithm tests the weights obtained from the training data on the validation data, and only if the AUC of both the training data and validation data improves will the results and weights of that model be adopted; if the AUC of either is degraded, the results of that model are rejected.

The results of public LB were similar to the NN stacking method I used, but the CV score was too high, so I rejected it as I thought there was a possibility of overfitting.


-Hill climb model with CV loop for "the results of the model in which the target feature is calculated within the CV loop":

[https://www.kaggle.com/code/satokin13m/s6e2-hill-climb?scriptVersionId=299993972](https://www.kaggle.com/code/satokin13m/s6e2-hill-climb?scriptVersionId=299993972)

CV score:0.9557888

Public LB:0.95394

Private LB:0.95535


-Models that performed hill climbing with CV loops on the "Results for all models" 
(only models that calculate target features outside the CV loop are adopted):

[https://www.kaggle.com/code/satokin13m/s6e2-hill-climb?scriptVersionId=300034509](https://www.kaggle.com/code/satokin13m/s6e2-hill-climb?scriptVersionId=300034509)

CV score:0.9558057

Public LB:0.95394

Private LB:0.95534


It seems that this is not a case of overfitting, but rather, as Masaya-san([https://www.kaggle.com/masayakawamata](https://www.kaggle.com/masayakawamata)), who produced the first solution, said, the CV-LB relationship is determined for each model, and it is important to adopt the model with the higher CV within the same model.

So, next I don't know what reasons are used to determine whether they are the same model or different models, and this is an issue that will need to be addressed in the future.


Congratulations to Masaya-san for your first place! 

I'm always amazed at how fast and accurate your start is. 

I also admire your deep insight and your willingness to invigorate the community.



# Impressions:

The more I do data science, the more I don't understand.

That's why I think it's so interesting.

I'm still learning.

I rely heavily on everyone's code and AI for my Python code.

However, I'm very grateful to be able to quickly code my ideas using AI.

I'm not very knowledgeable about how to write in Jupyter Notebook, so there are many things I don't understand.

This solution description may also not follow proper etiquette.

I'm not sure if what I'm writing now is correct.

I apologize if there are any unpleasant or confusing parts.


Kaggle is fun.
