# 30th place solution

Competition: lish-moa
Rank: #30
Source: https://www.kaggle.com/c/lish-moa/discussion/200679

First of all I want to thank the organizers for interesting data and competition. 
I'm not sure that choosing logloss as primary metric is good for pursuing practice goals, but it's convenient for modeling multilabel tasks for competitors ^^
Also, huge thanks to users @gogo827jz, @tolgadincer and @namanj27. Pretty appreciate your discussions and NJ7's pytorch starter is a gem.
Personally, it's the best result in Kaggle competitions, so I'm pretty happy about it.

Solution is relative easy and based on average blending of 4 models: 
1) Shallow MLP 
https://www.kaggle.com/alturutin/moa-mlp
2) "Resnet"-like multihead NN with cluster embeddings
https://www.kaggle.com/alturutin/moa-resnet
3) TabNet with hyperparameters almost similar to public notebooks
https://www.kaggle.com/alturutin/moa-tabnet
4) Shallow MLP with pretraining on nonscored data with and finetuned to maximize CV to GELU activation
https://www.kaggle.com/alturutin/moa-mlp-gelu
*) Ensemble notebook:
https://www.kaggle.com/alturutin/moa-inference
*) Viz, eda notebook:
https://www.kaggle.com/alturutin/moa-t-test-outliers-segmentation-interactions
Most of the code incapsulated in dataset scripts.

OneCycleLR scheduler for training MLP and Resnet and ReduceLROnplateu for TabNet.
Trained 2 type of models: for CV based on train data, because organizers wrote that private is randomized, and, just in case, grug_id based cv.

Preprocessing similar to public notebooks: quantile transform -> feature selection, pca, feature stats, clusters.
I suppose main difference from other works may be using RFE to MLP permutation importance scores. It's given big boost on CV.
Also, I've used prediction bias as metric for model finetuning: for example, if train prior mean = 0.0037 I've tried to get prediction mean > 0.0037 to avoid overfitting, because models tended to overfit to 0. 
It's possible to solve this problem using regularization, for example, label smoothing, but estimates is not very concise and little lucky-ish I think :)
Pseudo labeling worked too and provided small boost, with lr=1e-5 and SGD optimizer.
Postprocessing is just clipping to inteval [1e-5; 1 - 1e-5]
Pytorch was preferred over tensorflow because it just has given better results for similar hyperparameters.

Finally, sorry for my english (it's not my native), I've learned a lot in this competition. 

List of trained features below.

===========================================================
worked:
+ 10 folds oof mkf (multilabel stratified)
+ bagging over seeds & splits
+ label smoothing
+ predictions clipping
+ pca
+ pseudo labels
+ quantile transform preprocessing
+ resnet with categorical embeddings
+ tabnet
+ average blending
+ stats feature engineering
+ RFE feature selection
+ stacking top labels
+ nonscored pretraining
+ clusters from kmeans, GMM and outlier detection methods

===========================================================
not worked:
- fit unsupervised on full data (train + test) -> overfitting
- gbm (too long to train)
- kmeans dists
- dbscan->svm clustering
- self-norm nn
- ttest feature selection
- using drug group cv
- MLSMOTE augmentation
- noise augmentation
- feature interactions like sum & multiplications

===========================================================
had no time to check:
* online pseudo labeling on private dataset
* TVAE; GAN augmentation
* transformer/attention model
* convert features to image and fit CNN
* DAE embeddings
* contrastive learning (with triplet loss for example) embeddings
