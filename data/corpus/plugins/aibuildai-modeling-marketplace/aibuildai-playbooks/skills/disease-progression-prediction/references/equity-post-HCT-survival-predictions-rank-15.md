# 15th Place Solution

Competition: equity-post-HCT-survival-predictions
Rank: #15
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566756

15th place solution

First of all, I would like to thank the Kaggle community for sharing great ideas and engaging discussions. I would also like to thank the hosts for organizing this interesting task competition.

# Overview
- Ensemble which focused on the mechanism of the metric and the performance of the model
- Stacking GBDT models and various transformed targets

# Target transformation
I used public note transform ( Kaplan , Nelson etc. ) and Rank Gauss transform.
Because the number of unique values of the target decreases by using Kaplan and so on , I concerned that amount of information decreases.
Therefore I used Rank Gauss transform as not decreasing uniques , then LB was improved slightly.

[simple xgb result]
|  | CV | Public LB | Private LB |
| --- | --- | --- | --- |
| Kaplan | 0.675 |  |  |
| Nelson | 0.680 | 0.685 | 0.688 |
| Rank Gauss | 0.680 | 0.686 | 0.689 |
 
# Model
I implemented stacking 30 models twice.
When n time stacking , I use n-1 time OOF as feature.
 
30 models = GBDTs(XGB,CAT,LGB) : 3 * Various targets and settings : 10
 
[ensemble result(detail below)]
|  | CV | Public LB | Private LB |
| --- | --- | --- | --- |
| No stack and below ensemble | 0.687 | 0.689 | 0.693 |
| 1st stack and below ensemble | 0.690 |  |  |
| 2nd stack and below ensemble(final submission) | 0.694 | 0.691 | 0.695 |
 
# Ensemble
I implemented ensembling as follows using Rank Gauss transform target.
1. Ensembling efs=1 data trained model and all data trained model 
2. Substituting all data trained model prediction for above prediction  in case fallling below the threshold ageinst all data trained model prediction.
    `y_pred.loc[oof['pred']<threshold,'prediction'] = oof['pred']`
 
I consider that the competition score is calculated from two part. One part (a) is to compare each efs=1 prediction ,other (b) is to compare efs=0 prediction against efs=1 prediction.
By training only efs=1 data , ‘a’ part score will be very high. 
‘a’ part score can be improved by ensemble 1.
‘b’ part score can be improved by ensemble 2.

[2nd stack result]
|  | 'a' part (efs=1) CV | CV(a+b part) |
| --- | --- | --- |
| efs=1 data model | 0.745 | 0.561 |
| All data model | 0.597 | 0.686 |
| Ensemble 1 | 0.684 | 0.669 |
| Ensemble 2(final submission) | 0.680 | 0.694 |
 
# Didn’t work
・Target encoding
・Umap/t-SNE as a feature
