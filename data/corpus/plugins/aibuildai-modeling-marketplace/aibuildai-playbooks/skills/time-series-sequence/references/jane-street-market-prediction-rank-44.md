# Current 29th place approach

Competition: jane-street-market-prediction
Rank: #44
Source: https://www.kaggle.com/c/jane-street-market-prediction/discussion/227167

I'm relatively new to kaggle, so I'm surprised to be high in the leaderboard (maybe because of much luck). I learned a lot from this competition. I would like to share my approach. I published [inference notebook](https://www.kaggle.com/hedwig100/js-inference-notebook) and [training notebook](https://www.kaggle.com/hedwig100/nn-training-notebook). 

Later in this competition, I noticed ensembling models boosts my local CV. I think this is caused by stable prediction. So, I trained various models, and ensembled these models.  Details are shown below. 

<br> 

#Training Strategy#
- PurgedGroupTimeSeriesSplit(5fold and 20gap)
- but maybe leakage was caused when training encoder... 
- watched AUC and UtilityScore in each fold.
- used Early Stopping by valid-auc 

<br>

#Preprocess#
- fill NaN by 0 and FeatureNeutralization(p=0.25)
- fill NaN by mean 
- denoising autoencoder

<br>
#Model#
- simple NN, CNN and DenseNet. 
- I also used [this famous notebook](https://www.kaggle.com/a763337092/pytorch-resnet-starter-training)'s model architecture. I trained this model with my CV strategy. 
- In first submission, I used my 3model. 
- In second submission, I used my 3model + 1pytorch model
- For each model, I used weight trained in last fold, and weight trained with all data(except for weight = 0 and first 85days)
- second submission achieves higher score. 

<br>
#Ensemble#
- weight average
- weight was decided by CV. 

<br> 

Thank you for reading!
