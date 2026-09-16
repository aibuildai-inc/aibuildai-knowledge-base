# 1st Place Solution-2 Targets and Ensemble

Competition: equity-post-HCT-survival-predictions
Rank: #1
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566550

First, congratulations to the teams that finally "survived" in this competition, and thank you to the participants who shared their experiences and provided help in the forum.

I also trained classifier and regressor models independently and then combined them together with a magic function, just like 2nd and 4th place solution. The main idea is that when a patient has a high probability of `efs == 0`, we should give them a high score. Otherwise, we should give them a relatively lower score and pay more attention to the rank of `efs_time`. So the regressor just needs to focus on the ranking task where `efs == 1`.

Here is the simplified [code](https://www.kaggle.com/code/minerppdy/1st-place-simplified-private-lb-0-700).

## 1. Feature Engineering
1. Original features  
2. One-hot encoding for all categorical features(but still keep the original category features)  
3. Copy continuous features as categorical features  

## 2. CV Strategy
1. The CV is very unstable with different splits Strategy. To make the results comparable, I used 10-fold splits with same seeds for all classifier and regressor model training.
```python
y_combine = data['efs'].astype('str')+'|'+data['X']['race_group'].astype('str')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=888)
skf.split(X,y_combine)
```
2. without use early stoping to reduce the risk of overfitting, especially when I use a large number of folds.



## 3. Classifier
### Target: `P(efs=0)`
### Models Used  
1. XGBoost, LightGBM, CatBoost  
2. NN, TabM
3. GNN  
4. NN/TabNet/GNN with pairwise-rank-loss
### tricks
1. Gnn method. I use KNN to find nearest 25 nodes and  create edges(use Euclidean Distance),then use graphsage to fit the targets.
2. when use rank loss, the predition would shift over different fold because rank loss is not sensitive to shift, which may cause lower CV when calculate metric in the entire set. So you need to fix these shifts to keep same mean/mid prediction value over models trained in differece fold.
3. GNN also have this problem even if I used logloss，I still don't know why.
### auc metrics

|  model| auc |
| --- | --- |
|lightgbm | 0.7603  |
|catboost| 0.7617  |
|xgboost|  0.7606  |
|nn| 0.75902  |
|tabm| 0.7596  |
|gnn|  0.7591  |
|ranknn| 0.7598  |
|ranktabm| 0.7597  |
|rankgnn| 0.7586  |
### interesting finding
1. The best max depth for LGB and XGB is 2 (for CatBoost, it's 6, and I still don't know why). This is nearly unheard of in my experience. It means there is little valuable feature interaction information, and the task is easy to fit. Maybe that's why NN like models also work well in this tabular data.
## 4. Regressor
### Target  
Grouped and normalized rank by `efs`  
```python
efs_time_norm[efs == 1] = efs_time[efs == 1].rank() / sum(efs == 1)  
efs_time_norm[efs == 0] = efs_time[efs == 0].rank() / sum(efs == 0)  
```
### Models used
XGBoost lightgbm catboost( NN like model didn't work well for me, if any one succeed in NN like model, please let me know.)
### Tricks
1. Add efs as a feature when training and focus on the performance at samples where `efs==1`. Set `efs=1` when making inference on LB data. 
2. apply sample weight on efs==1 and efs==0 with 0.6:0.4
3. Here's why this trick is used. Initially, I trained the regressor only on samples where `efs==1`. When using this model to make inference on samples where `efs==0`, it still showed an obvious correlation between prediction and ground truth. This is strange because efs_time for `efs==0` is meaningless due to various reasons could cause data censoring. I guess it's because of the SurvalGAN algorithm. By the way, it turns out that there are similar patterns between samples where `efs==0` and `efs==1`. So, by adding samples where `efs==0`, the regressor's performance on `efs==1` improved significantly!

### Raw Concordance Index on efs==1
|  model| c-index |
| --- | --- |
|xgboost|0.770229|
|lightgbm|0.767615|
|catboost|0.769340|


## 5.Merge Function
```python
def model_merge(Y_HAT_REG,Y_HAT_CLS,a=2.96,b=1.77,c=0.52):
    '''
    Y_HAT_REG and Y_HAT_CLS need be scaled to 0~1
    a,b,c need to be tuned with optuna 
    '''
    y_fun = (Y_HAT_REG>0)*c*(np.abs(Y_HAT_REG))**(b)
    x_fun =(Y_HAT_CLS>0)*(np.abs(Y_HAT_CLS))**(a)
    res = (1-y_fun)*x_fun+y_fun
    res = pd.Series(res).rank()/len(res)
    return res
```
This is what the merge function looks like(without rank transform), where x is the probablity of efs==0 and y is predicted efs_time(scaled to 0~1) 


## 6 Ensembling Method
1. Create combinations between classifiers and regressors, then find the best (a, b, c) for the merge function by with 5-folds and then get the merged predictions.

    Here is Stratified Concordance Index of combinations. Ignoring the rank transform in merge function, pretty sure there is no leakage

    |cls|reg|CV Stratified c-index|
    |---|---|---|
    |xgboost|xgboost | 0.6938821198444668    |
    |xgboost|catboost|0.693390761259049|
    |xgboost|lightgbm|0.6920553689042972|
    |catboost|xgboost|0.6946771227057665|
    |catboost|catboost|0.6942415329389815|
    |catboost|lightgbm|0.6928675187239672|
    |lightgbm|xgboost|0.6935306031275579|
    |lightgbm|catboost|0.693123814234674|
    |lightgbm|lightgbm|0.6917244174625249|
    |tabm|xgboost|0.6943607674628444|
    |tabm|catboost|0.6936616324722884|
    |tabm|lightgbm|0.6922428015362854|
    |nn|xgboost|0.6939542378441973|
    |nn|catboost|0.6932931191267933|
    |nn|lightgbm|0.6920700218390199|
    |gnn|xgboost|0.6940062476514388|
    |gnn|catboost|0.6932012736212297|
    |gnn|lightgbm|0.6921340326054011|
    |ranktabm|xgboost|0.693141398733159|
    |ranktabm|catboost|0.6927626179572159|
    |ranktabm|lightgbm|0.6912730733179469|
    |ranknn|xgboost|0.6944776101013886|
    |ranknn|catboost|0.6936542865352446|
    |ranknn|lightgbm|0.6924528052605144|
    |rankgnn|xgboost|0.6934854451860402|
    |rankgnn|catboost|0.692783734834767|
    |rankgnn|lightgbm|0.6916329456191074|

2. Ensemble the merged predictions using a weighted average with Optuna and 5-folds.
3. Since there are 9 classifiers × 3 regressors = 27 combinations, there is a risk of overfitting. Therefore, I set the Optuna search range for weights to be between 0.1 and 1, rather than 0 and 1. I think this is a form of regularization.
4.  The  final CV Stratified c-index is round 0.6965
5.  Adding noise to some race groups significantly improved cross-validation performance, but it didn't work on either the public or private LB.
