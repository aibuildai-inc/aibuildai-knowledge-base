# 17th place solution | Hand-Tuning and features

Competition: playground-series-s3e11
Rank: #17
Source: https://www.kaggle.com/c/playground-series-s3e11/discussion/399393

### First, our key method was to take 3 different approaches without telling each other what was our plans and then ensemble. Like this: 
.

**On this topic, I'll only tackle my solution, which you can find a good part [here](https://www.kaggle.com/code/janmpia/feature-eng-xgb-cat-ensemble-0-29265).** The ensemble has been dealt with by @shashwatraman 


@shashwatraman solution : https://www.kaggle.com/competitions/playground-series-s3e11/discussion/399485
@ifreenibrahim solution : https://www.kaggle.com/competitions/playground-series-s3e11/discussion/399438



**Global Idea:**
My main focus for this model was the hyperparameters of my models, and to make sure my CV was rebust enough to allow for such search, without overfitting either the train or even the CV valid set. For the tuning part, I did it all by hand so I can't give you a go-to guide on optuna, I beleive it is too expensive in terms of GPU time for poor improvement unpon a certain point. Hand tuning works best for me and you also learn a great deal about the model's architecture.

**Features:**
As many have pointed out, not all features were relevent, my selection was:
`FEATS = ["total_children", "num_children_at_home", "avg_cars_at home(approx).1", "store_sqft", "coffee_bar", "video_store", 'florist',"prepared_food"]`

I also created some features for my catboost model which are the following:
concat_train_hold_test = pd.concat([train,hold,test],ignore_index=True)
```python
for feature in INIT_FEATS:
    if feature in ['units_per_case','store_sales(in millions)','total_children']:
        avg_df[f'avg_{feature}'] = concat_train_hold_test.groupby('store_sqft')[feature].mean()
        avg_df_test[f'avg_{feature}'] = concat_train_hold_test.groupby('store_sqft')[feature].mean()
        avg_df_hold[f'avg_{feature}'] = concat_train_hold_test.groupby('store_sqft')[feature].mean()
        avg_df_original[f'avg_{feature}'] = concat_train_hold_test.groupby('store_sqft')[feature].mean()
        
        CAT_FEATS.append(f'avg_{feature}')
```

You might be wondering what that code does and whats the purpose of those features, check out [this topic](https://www.kaggle.com/competitions/playground-series-s3e11/discussion/397431).

**Models:**
- XGB
- CAT
- LGBM

Thats it for my solution ! Kindly check out my mates solution once they are posted !
