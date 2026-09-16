# 7th Place Solution (Updated Late Submission Finding)

Competition: elo-merchant-category-recommendation
Rank: #7
Source: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82055

Congrats to all winner teams and new masters,experts.It's a very hard competition,I think every team work hard to try everything and fight with overfitting every day,no one fails who does his best.
I started at very early stage so maybe I just did more work than others,let me share my solution briefly.

#**Preprocessing**
sort the historical and new data by timestamp then create 8 kinds of dataset for different features.

 1.  Only historical data
 2. Only historical data with authorized_flag=1
 3. Only new data 
 4. Merge of historical data with authorized_flag=1 and new data 
 5. Merge of historical data  and new data 
 6. Merge of historical data and merchants data
 7. Merge of new data and merchants data
 8. Merge of historical data  and new data and merchants data

#**Feature Engineering**
## Aggregation
I think there are not big difference with everyone's aggregation features, I want to thank raddar's magic that purchase_amount_new can bring additional features beside of original purchase_amount.

## Interval
days interval(the difference from current purchase_date to next purchase_date)
purchase_amount interval(the difference from current purchase_amount to next purchase_amount)

## Interaction
new.purchase_date.max() / historical.purchase_date.max() is very strong

## SVD
dimession reducing from original card_id's merchant|merchant_category transaction sequence to 5 
```
def svd_feature(prefix, df, traintest, groupby, target,n_comp):
    tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=None)
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    df_bag[target + '_list']=df_bag['list'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',',' '))
    
    df_bag = df_bag.merge(traintest,on=groupby,how='left')
    df_bag_train = df_bag[df_bag['target'].notnull()].reset_index(drop=True)
    df_bag_test = df_bag[df_bag['target'].isnull()].reset_index(drop=True)
    
    tfidf_full_vector = tfidf_vec.fit_transform(df_bag[target + '_list'])
    tfidf_train_vector = tfidf_vec.transform(df_bag_train[target + '_list'])
    tfidf_test_vector = tfidf_vec.transform(df_bag_test[target + '_list'])
    
    svd_vec = TruncatedSVD(n_components=5, algorithm='arpack')
    svd_vec.fit(tfidf_full_vector)
    svd_train = pd.DataFrame(svd_vec.transform(tfidf_train_vector))
    svd_test = pd.DataFrame(svd_vec.transform(tfidf_test_vector))

    svd_train.columns = ['svd_%s_%s_%d'%(prefix,target,x) for x in range(n_comp)]
    svd_train[groupby] = df_bag_train[groupby]
    svd_test.columns = ['svd_%s_%s_%d'%(prefix,target,x) for x in range(n_comp)]
    svd_test[groupby] = df_bag_test[groupby]
    #df_svd = pd.concat([svd_train,svd_test],axis=0)
    print ('svd_train:' + str(svd_train.shape))
    print ('svd_test:' + str(svd_test.shape))
    return svd_train,svd_test
```

## Word2vec
create each merchant_id's , merchant_category_id's , purchase_date's ... word embedding,then group by card_id's min/max/mean/std
```
def word2vec_feature(prefix, df, groupby, target,size):
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag[target] = df_bag[target].astype(str)
    df_bag[target].fillna('NAN', inplace=True)
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    doc_list = list(df_bag['list'].values)
    w2v = Word2Vec(doc_list, size=size, window=3, min_count=1, workers=32)
    vocab_keys = list(w2v.wv.vocab.keys())
    w2v_array = []
    for v in vocab_keys :
        w2v_array.append(list(w2v.wv[v]))
    df_w2v = pd.DataFrame()
    df_w2v['vocab_keys'] = vocab_keys    
    df_w2v = pd.concat([df_w2v, pd.DataFrame(w2v_array)], axis=1)
    df_w2v.columns = [target] + ['w2v_%s_%s_%d'%(prefix,target,x) for x in range(size)]
    print ('df_w2v:' + str(df_w2v.shape))
    return df_w2v
```

## Transaction-based Model
use above 6,7,8 dataset, merge train's target to each transaction row,then create a lightgbm to output each transaction's meta feature , then groupby card_id's min/sum to create card_id's meta feature , it can boost 0.005 ~ 0.006  both on cv and lb.

## Feature Selection
at early stage I add new features one by one,sometimes group by group,at late stage I use target permutation learned from https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
I created thousands of features,finally get 12 different feature sets with number from 200 ~ 700

## Top 20 Feature Importance 
```
('reborn_new_min_submodel', 						3212145.126355231)
('reborn_hist_new_min_submodel', 					2897647.0672301054)
('reborn_hist_min_submodel', 						2571620.7657690495)
('reborn_hist_sum_submodel', 						1556242.6892476082)
('reborn_hist_new_sum_submodel', 					1495420.6608867645)
('card_purchase_date_max_ratio', 					1176560.0546371937)
('reborn_new_sum_submodel', 						988300.9342432022)
('hist_new_purchase_date_max_ratio', 				751098.8485645056)
('hist_new_sum_category_1_1_purchase_amount', 		634349.3938679695)
('hist_sum_category_1_1_purchase_amount', 			626576.9190101624)
('new_card_sum_purchase_amount_new', 				573281.6908314228)
('hist_card0_max_month_diff', 						539578.2926783562)
('new_card_var_day_diff', 							537170.215294838)
('new_card_last_day', 								532604.059334755)
('hist_card_sum_month_lag_0_purchase_amount_new', 	530102.9405331612)
('hist_card0_mean_month_diff', 						506467.2120089531)
('svd_hist_new_merchant_id4', 						454442.183167696)
('hist_card0_sum_month_diff', 						445077.63865828514)
('hist_card_std_merchant_category_id_w2v_4', 		436928.8732688427)
('hist_new_card_std_merchant_category_id_w2v_1', 	434849.51894950867)
```

# **Single Model**
## Lightgbm
My best single model is a lightgbm has 385 features with CV:3.6144,LB:3.668,PB:3.593,maybe it can reach to 3th place.It also happened at homecredit competition,best single model is much better than stacking,I will doubt if stacking is a good choice.
## Nueral Netowork
My best nn has CV:3.6407,LB:3.680,PB:3.605,it's just a simple 3layers-dense network.

```
def ann(input_shape):
    model = Sequential()
    model.add(Dense(2 ** 10, input_dim = input_shape, init='random_uniform', activation='relu'))
    model.add(Dropout(0.25))    
    model.add(BatchNormalization())
    model.add(Dense(2 ** 9, init='random_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25)) 
    model.add(Dense(2 ** 5, init='random_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))      
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam') 
    return model
```

## Others
Xgboost and Catboost got not bad results but almost no improvement to stacking.
Also Tried Ridge,KNN,FFM...no success to me.


# **Ensemble**
I built 12 lightgbm models and 40 nn models for Stacking stage1,then use lightgbm and nn as stage2,finally blend them with average.CV can reach to 3.603,but lb got worse,and final pb also got worse.

# **Postprocessing**
Isotonic regression give me 0.005 ~ 0.006 boost both on cv and lb but pb score become worse,that's the reason I shakedown.
Combination with a non-outliers model using binary model's best threshold give me 0.001 ~ 0.002 boost both on cv and lb and pb.

# **Leaderboard Probing**
I used 100 subs to probe the lb,that let me grasp 24 public-test outliers and 6 non-outliers.
I tried Pseudo-Labelling but no boost,finally it just can be used to decide a "perfect threshold" to override the outliers but failed.

#**Other tricks**
at very early stage I tried target transformation,using minmaxscaler to transform from 0 ~ 1, then use xentropy boosting type,finally reversed to original value range,it didn't improve cv and lb, but I found pb is very good, a very eary stage lightgbm model is enough to get a gold medal.

#**Late Submission Founding**
Isotonic regression overfit both on cv and lb, should be careful.
Stacking can work if stage-2 model is nn or ridge ,lgb and extratree overfit too much.
