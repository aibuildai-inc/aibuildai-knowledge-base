# Solution Sharing

Competition: springleaf-marketing-response
Rank: #7
Source: https://www.kaggle.com/c/springleaf-marketing-response/discussion/17081#96804

I use pandas and numpy in python.

     def get_ctr_features(data, test, y, ctr_cols, dctr, num):
            data["target"] = y
            dcols = set(test.columns)
            kf = cross_validation.StratifiedKFold(y, n_folds=4, shuffle=True, random_state=11)
            tr = np.zeros((data.shape[0], len(ctr_cols)))
            for kfold, (itr, icv) in enumerate(kf):
                data_tr = data.iloc[itr]
                data_te = data.iloc[icv]
                for t, col in enumerate(ctr_cols):
                    if col not in dcols:
                        continue
                    ctr_df = data_tr[[col, "target"]].groupby(col).agg(["count", "sum"])
                    ctr_dict = ctr_df.apply(lambda x: calc_ctr(x, num), axis=1).to_dict()
                    tr[icv, t] = data_te[col].apply(lambda x: ctr_dict.get(x, dctr))
        
            te = np.zeros((test.shape[0], len(ctr_cols)))
            for t, col in enumerate(ctr_cols):
                if col not in dcols:
                        continue
                ctr_df = data[[col, "target"]].groupby(col).agg(["count", "sum"])
                ctr_dict = ctr_df.apply(lambda x: calc_ctr(x, num), axis=1).to_dict()
                te[:, t] = test[col].apply(lambda x: ctr_dict.get(x, dctr))
            del data["target"]
            return tr, te

[quote=Bishwarup B;96803]

[quote=Gzs_iceberg;96802]

congratulations to all winners!

Here's my solution. I list some key points below.

(1) category features: use likelihood to encode it, the way how you do is important, it's easily leaky. I use a cross validation to do it.

(2) logistic regression: feed it's prediction into xgboost. it's similar to likelihood features.

(3) feature engineering: only location and date work.

(4) tuning parameters: best paramter I got: 
max_depth=18 colsample_bytree=0.3 min_child_weight=10 subsample=0.8 num_round=9666 eta=0.006 

(5) merging different xgboost's model by using different features and paramters.


[/quote]

Icebarg,

Do you mind sharing your likelihood approach to encode the categorical features. I tried to encode with randomized target rate but every time it resulted in a massive overfit. I would be happy if you share how you cross-validate the optimal encoding here. 

Congrats for finishing in the top 10 ladder.

[/quote]
