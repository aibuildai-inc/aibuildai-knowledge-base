# 16th Place Solution Summary

Competition: open-problems-multimodal
Rank: #16
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/368052

Thanks to the competition host, Kaggle team, Saturn cloud team and congrats to all the winners!

Although many great solutions have already been posted and my solution may not contain new approaches, I would like to leave my efforts over the past two months. In this term I learned a lot. Thanks to the all competitors for a great game.

Please forgive me if it is difficult to read this post or see the schematic diagram below, as my command of English is not very good and my educational background was different from computer science or machine learning area.
***
# Overview:
The machine learning algorithms used were as follows:
- Two MLPs (4 and 9 hidden layers, the variants of [Laurent Pourchot's model](https://www.kaggle.com/code/pourchot/all-in-one-citeseq-multiome-with-keras?scriptVersionId=108466116))
- Conv1d (almost all the same as the [tmp’s 1D-CNN model for tabular data on MoA competition](https://www.kaggle.com/c/lish-moa/discussion/202256))
- LGBM

In my case, stacking scheme boosted the score. The outputs of level 1 models were concatenated and then used as input for level 2. It was effective to apply dimensionality reduction to concatenated level 1 outputs after standardization. When the outputs were just concatenated without dimensionality reduction, the score of level 2 was rather lower than that of level 1.

Also, ensemble worked well. I created several models with slight difference (different dimensionality reduction algorithms, feature extraction methods and loss functions) and blended them. In addition, each learning process was performed on 15 random-seeds and results were averaged.
<br>
# CV scheme:
I used simple KFold (k = 5). Fortunately, I resulted in shakeup in private LB.
<br>
# Citeseq:
The diagram of my Citeseq stacking scheme is as follow.


#### Preprocess
Before dimensionality reduction or feature extraction in level 1, the set of all features which are constant in the train or test were eliminated according to the [AmbrosM’s Code](https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart).
- Dimensionality reduction: tSVD and PCA (n_components = 64) were used separately and inference results were finally blended.
- Feature extraction: According to the [Fabien Crom's Code](https://www.kaggle.com/code/fabiencrom/msci-correlations-eda-citeseq/notebook), I picked features with high Pearson correlation coefficients for the targets. In order to gain diversity as much as possible, I changed the picking query for each models. For example:
・Extract the RNAs in order of highest **average** of correlation to 140 proteins
・Extract the RNAs that have a high correlation value for a **single** protein, **not the average**
・Change how many of the top RNAs are extracted
・With or without dimensionality reduction after extracted
<br>
# Multiome:
The scheme is almost the same as that of Citeseq. The differences are as follows:
- Extracted features were not used. In the Multiome case, the score was deteriorated when they were concatenated with 64 dims compressed features.
- For NN algorisms, only MSE was used as loss function.
- The target vectors were compressed to 512 (for NN) or 128 (for LGBM) dims by tSVD and used in training. The inferred vectors were decompressed to 23418 dims by inverse SVD.
