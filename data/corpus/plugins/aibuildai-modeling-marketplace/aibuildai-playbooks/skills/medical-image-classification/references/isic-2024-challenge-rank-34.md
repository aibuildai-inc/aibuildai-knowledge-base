# 34th place solution

Competition: isic-2024-challenge
Rank: #34
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/533046

I used catboost, xgboost and lgbm for tabular data and tabular + cnn features. (total 6 models). As stacking method I applied StackingCVClassifier from mlxtend with LogisiticRegression as a metaclassifier. 
CNNs densenet201 (224 x 224) and efficient_net_b0 (384, 384). Data from previous competitions was added. 
I used cutmix with alpha=0.5 for regularization and data augmentation from 2020 first solution. It works well on public LB. My max was 0.163 from densenet201, but failed on private LB. 
Bagging with different seed was working (0.171 PB), but I did not select it.
