# 40th place solution | MLPs, ResNet, TabNet & Clustering

Competition: lish-moa
Rank: #40
Source: https://www.kaggle.com/c/lish-moa/discussion/200552

Congratulations to the winners!

Here is my approach:

# CV: 
The most important part of any Kaggle competition, aka finding a good cross-validation framework, was a nightmare in this competition. I started with 10-fold CV, but it was not generalizing at all to the public LB and I switched to MultilabelStratified 10-fold. Generalization to the LB was weak, so I decided to **only keep changes that improved both my CV and the LB**. I think that this was key in this competition. I experienced with [Chris Deotte's CV strategy ](https://www.kaggle.com/c/lish-moa/discussion/195195)but found that it was not generalizing better than my old CV, so I sticked to MultilabelStratified 10-fold. 

# Models: 
Here are the models used in my final ensemble:
- MLPs (4 models):
I took inspiration from [this notebook](https://www.kaggle.com/riadalmadani/pytorch-cv-0-0145-lb-0-01839/output?scriptVersionId=45510876&select=submission.csv).
In my final ensemble, I used 4 MLPs: 2 different architecture (varying number of layerss), one with the new CV framework (from Chris Deotte), and one predicting non-scored labels as well.
CV (for the best MLP): 0.01571 // LB: 0.01834
- ResNet (1 model):
I started by re-implementing [this notebook](https://www.kaggle.com/rahulsd91/moa-multi-input-resnet-model) locally, which I converted to Pytorch.
CV: 0.01601 // LB: 0.01854
- TabNet (1 model)
TabNet was doing amazingly well in this competition. I was not familiar with the model and it was a pleasant discovery. I started with the implementation described [here](https://www.kaggle.com/hiramcho/moa-tabnet-with-pca-rank-gauss?scriptVersionId=45407015), and brought it down to 0.01827 LB by just changing parameters. Setting n_shared = 1 and n_independent = 1 was key. Also switching to OneCycleLR scheduler, and reducing the weight decay. 
CV: 0.01569 // LB: 0.01827

# Training:
A few elements were essential. I used these hyper-parameters across all models: 
- Averaging 5 models with different random seeds per fold
- weight decay 8e-6
- label smoothing 1e-4
- OneCycleLR scheduler with 25 epochs, and max_lr 5e-3 and div_factor 1e3

# Ensembling:
Nothing fancy, just a weighted average. 
Best weights: 
0.5 * MLPs + 0.1 * ResNet + 0.4 * TabNet
CV (excluding the MLP trained with the new CV framework): 0.01556 // LB: 0.01821

# Clustering:
We know that some drugs appear in both the training set and the test set. Besides, once drug_id was given, we discovered that 8 drugs were dominating in terms of frequency. So I thought that we could do some clustering, and find test drugs close to clusters formed by these 8 drugs. I first trained a t-sne model to reduce the genes+cells feature space to dimension 2, for these 8 drugs + the public test set. Then, when plotting, I noticed that 5 out of these 8 drugs had clear, separate clusters. So for test data points close (in dimension 2, and with L2-distance) to the cluster centers of these 5 drugs, I blended my ensemble's current predictions with the labels from these 5 drugs. Amazingly, that helped my public LB from 0.01821 to 0.01819, which converted to 0.01611 and 0.01609 on the private LB respectively. I did not even check how much this helped CV. To keep safe I only used this clustering with one of my two final submissions. 



In red is the public test set, the 8 other colours are the 8 drugs with highest frequency. We can see the 5 forming clear clusters: orange (bottom left), purple (left), green (top left), light blue (top left), and blue (top). When looking at this plot, it is very natural to want to label the red points falling into these clusters with the same label as these drugs :) 

The drug_ids of the 5 clusters are: 292ab2c28, 87d714366, 8b87a7a83, d08af5d4b and d1b47f29d.

# What did not work:
- xgboost
- Clustering on the full training set
- Clustering in the original feature space
- 2nd-stage models trained on model predictions
- Deep stacking 
- DAE features
- Removing targets with too few positives
- Other optimizers than Adam 
- T-SNE features for neural nets
