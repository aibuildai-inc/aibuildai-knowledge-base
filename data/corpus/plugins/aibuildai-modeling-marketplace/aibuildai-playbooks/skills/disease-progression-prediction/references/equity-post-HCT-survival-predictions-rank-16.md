# 16 th Solution

Competition: equity-post-HCT-survival-predictions
Rank: #16
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/567643

**1.Acknowledgement**
First and foremost，I'd like to start with a big respect to the following open source authors and my team:
1.one of our model from:[https://www.kaggle.com/code/albansteff/refactoring-nn-pairwise-ranking-loss](url)
2.some feats from:[https://www.kaggle.com/code/yunsuxiaozi/cibmtr-yunbase](url)
3.use the classifier to make post-processing：[https://www.kaggle.com/code/kendontcare11/public-classifier-cat-xgb-lb-0-688](url)
4.The target constructor provided by **all the great open source workers**.
5.And the three brothers(**leaf,yueming,DeepMayNotLearn**),who have been preparing the postgraduate entrance exams ,thank them for their hard work
 (the above ranking is in no particular order)

**2.Plan introduction**
**2.1 NN**
**2.1.1 Multi-target NN **
A：target one： CV of CoxPHFitter
B：target two：custom_target to split the 0 1 as much as possible

`def custom_target(self): 
        cus_target =  self.data[['efs','efs_time']]
        cus_target['y'] = self.data.efs_time.values
        mx = cus_target.loc[cus_target.efs==1,"efs_time"].max()
        mn = cus_target.loc[cus_target.efs==0,"efs_time"].min()
        cus_target.loc[cus_target.efs==0,"y"] = cus_target.loc[cus_target.efs==0,"y"] + mx - mn
        cus_target.y = cus_target.y.rank()
        cus_target.loc[cus_target.efs==0,"y"] += 2*len(cus_target)
        cus_target.y = cus_target.y / cus_target.y.max()
        cus_target.y = np.log( cus_target.y )
        cus_target.y -= cus_target.y.mean()
        cus_target.y *= -1.0    
        return cus_target
`

We trained two simple models （CNN，DNN） to fit the two targets

**2.1.2 Opensource ODST **
we make some changes：   
1.fill the unseen category with train[col].mode()[0] ,rather than np.nan
`if ind.any():
    val.loc[ind, col] = train[col].mode()[0]`
2.use weights（race_index of each fold) to ensemble each fold

**2.1.3 Our ODST（pure torch version without torch lightning）**  
1.some additional feats from yunsuxiaozi‘s notebook   
2.slights change the Hyperparameters   
3.use weights to ensemble each fold best of our metric   
4.use **seed_torch(42+fold)** to train the model of each fold

**2.2 GBDT Tree**
  For the tree model part, we use two classification trees with different features to make post-processing.And use  xgboost，lightgbm and catboost to train the common target mentioned in the public code.The features our tree used are easy to see in the public.

**3.Overview of the Plan**
For more details：https://www.kaggle.com/code/kudosn/nn-tree-ensemble 
![https://www.helloimg.com/i/2025/03/12/67d05e23c134a.png] (url to Plan Structure)
**Tips：If want to run this notebook please change "from cib_metric import score" with "from metric import score",we just find the UTILITY SCRIPTS has been deleted.**
**4.Eventually, Allow me to show great respect to the open source workers once again**
![https://www.helloimg.com/i/2025/03/12/67d05e22c6492.png] (url to a surprise)
