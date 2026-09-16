# 18th place solution

Competition: commonlitreadabilityprize
Rank: #18
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258497

Thank you to Kaggle and Commonlit for providing this opportunity. 

Also thanks to @rhtsingh and @andretugan for sharing excellent kernels. 

We had 24 large models and 3 base models, utilized full 3 hours of runtime. We stacked using SGD Regressor and TF-IDF features.  Models in our stack described below : 

**Cross Validation** - All models trained with 5 folds using kincaid readability feature. All our models were using 3 seeds (42, 77, 2021). Even with 3 seeds we noticed a LB difference of ~0.005 but 3 seeds were the max we could do. Nothing specific for each fold - all folds were trained with the same head/parameters. 

**Roberta Large**  (3 Roberta large with 3 seeds each - total 9 models)-   All Roberta models were using pretrained model  using Competition data, Kids frontiers data and [Simple wiki](https://github.com/LGDoor/Dump-of-Simple-English-Wiki). 
- Attention head - along with TF-IDF features with TabNet - differential learning rates 
- weighted layer pooling - multisample dropouts, No differential learning rates 

**Deberta Large**   (3 Deberta large with 3 seeds each - total 9 models) 
All Deberta models were trained for 25 epochs with validation at the end. No pretraining. 
- Deberta Large with pseudo labeling on [Text Simplification dataset](https://cs.pomona.edu/~dkauchak/simplification/) 
- Deberta large with pooled output, attention head and multisample dropouts along with TabNet for TF-IDF features. 
	For the multi-sample dropout architecture, the model outputs 5 values, the mean of which correspond to our targets and their standard deviation to the standard error. Thus, both the quantities are used for training.

**Electra Large** - Attention head - along with TF-IDF features with TabNet

**Ernie** - Attention head - along with TF-IDF features with TabNet. 

**Roberta Base** - (Pretraining on competition data) This model was almost same as [public kernel](https://www.kaggle.com/andretugan/pre-trained-roberta-solution-in-pytorch), we added predictions from LGBM and Ridge to this model

Our best model was Roberta Large with public LB score 0.461 (CV: 0.473) and everything else was between 0.461 and 0.473.  TF-IDF features gave us a boost of ~0.002 in every model. 

Things that didn’t work - 
-Pretraining with other datasets for Roberta Large - we tried pretraining with many many other datasets 
-Pseudo label with Roberta Large - It worked on private LB but it was not in our stack. 
-Pretraining for Ernie 
-Many many different architectures and combinations of different architectures. 

Disappointed that we missed gold zone by one rank, it was a pleasure collaborating with the team @kpriyanshu256 @aman1391 @gyanendradas and Adarash S -  Thank you.
