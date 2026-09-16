# [13th] My solution: 2nd stage of modeling

Competition: tabular-playground-series-may-2021
Rank: #13
Source: https://www.kaggle.com/c/tabular-playground-series-may-2021/discussion/243028

Hello everybody 😊
This competition is over and i'd like to share some ideas..
In this opportunity i share my concrete solution. You can see the workflow below
[wf]
## 1st Stage
For the 1st stage I used this type of code to get the models:
```
def lmodelv(X_train,y_train,X_val,y_val,lgb_params):
    d_train = lgb.Dataset(X_train, label=y_train)
    d_valid = lgb.Dataset(X_val, label=y_val)
    watchlist = [d_train, d_valid]
    model = lgb.train(lgb_params,
                      train_set=d_train,
                      valid_sets=watchlist,
                      verbose_eval=0,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    return model
```
```
def training_lgbv(X_train,y_train,X_test,txt='lgb'):
    skf=StratifiedKFold(n_splits = NUM_FOLDS,shuffle = True,random_state = RANDOM_STATE)
    yv=np.zeros((len(X_train),4))
    yt=np.zeros((len(X_test),4))
    for fold,(idx_tr,idx_vl) in enumerate(skf.split(X_train,y_train)):
        X_tr,y_tr=pd.DataFrame(X_train).iloc[idx_tr],y_train.iloc[idx_tr]
        X_vl,y_vl=pd.DataFrame(X_train).iloc[idx_vl],y_train.iloc[idx_vl]
        model=lmodelv(X_tr,y_tr,X_vl,y_vl,lgb_tune)
        # See the pattern inference, evaluation and forecast
        yv[idx_vl]=model.predict(X_vl)
        yt+=model.predict(X_test)/NUM_FOLDS #predict_proba
        print(f"Found Metric in {fold}:{log_loss(y_vl,yv[idx_vl])}")
    print(f'Results of the training: {log_loss(y_train,yv)}')
    np.save(path+f'preds/train_{txt}_{log_loss(y_train,yv)}.npy',yv)
    np.save(path+f'preds/test_{txt}_{log_loss(y_train,yv)}.npy',yt)
    return yv,yt
```
The first is used to define the model, and the other for the training. I did this because you can optimize the algorithm in two ways. (Ex Fine tuning https://www.kaggle.com/awwalmalhi/extreme-fine-tuning-lgbm-using-7-step-training)
In similar way I did a code for the preprocessing and hyperparameter tuning.
```
def num_encode(train_df, test_df, column, encoding):
    new_feature = [f"num_{col}" for col in column]
    encoding.fit(train_df[column])
    train_df[new_feature] = encoding.transform(train_df[column])
    test_df[new_feature] = encoding.transform(test_df[column])
    return train_df[new_feature],test_df[new_feature]
```
```
def get_params(study, obj,dir_save):
    study.enqueue_trial(lgb_tune)
    study.optimize(obj, timeout=TIME, show_progress_bar=True)
    study.trials_dataframe().to_csv(dir_save,index = False) # Save results
    return study
```
About preprocessing, i think is better try diversity of datas but i'm not sure. I didn't use autoencoder but i should did.
Also, see the way how you save the information of the training.
## 2nd Stage
Then, in the 2nd stage: You can see the notebook (see the code section, i have a ban for using gpu and I cant share the link) and this dataset.
The general idea is: Some submissions can merge with linear weigths, but this can be optimized with a metric (Finally the cv is the most important thing)
Well, I think the code and all this workflow can be better, but in general, this works properly (in this case).
And Congrats to the winners! I'd like to see their solution.
