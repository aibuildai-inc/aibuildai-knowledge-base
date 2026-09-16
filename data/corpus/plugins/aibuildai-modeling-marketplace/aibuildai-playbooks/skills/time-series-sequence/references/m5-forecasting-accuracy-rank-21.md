# 21st Place Solution (Nodalpoints)

Competition: m5-forecasting-accuracy
Rank: #21
Source: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/164685

# M5 Forecasting - Accuracy Writeup

##Thanks  
Me  and Ouranos would like to thank the organizers for putting out such a exceptional competition on this scale.  We also need to mention that our involvement in the M5 competition was both tasked and sponsored by our company (Nodalpoint LTD, Athens, Greece).  This gave us almost one month and half to be active in the competition.

##Warm up (Costas)
First I would like to share a personal opinion regarding the hierarchical nature of this competition. I began working on hierarchical stuff one a half month ago focusing on R packages 'forecast',  'hts' (hierarchical time series) and Professor Hyndman's m4 winning solution repository. I was modelling time series in every level-1  beeing the most aggregated and level-12 the most disagregated (item_ids) in the compensate. 

I soon after compensating 30490 low level time series, I had some descent results on levels 10, 11, 12 (better from the best performing LightGBM model), but when aggregated on top levels they produced worst results. So even if the lightGBM model failed to capture the individual sales count for each item, by daily aggregation the errors were cancelling out each other producing better high-level time series. That realization made me hold back from the classical time series approach and focus to other directions for the last one month. 

I am currently experimenting to see how this approach (hts and m4-winning forecasting per time series) would eventually score by itself and if it could further help our blend.

## Solution
We followed the most common strategy:  “treating the problem as a regression one”. So, no classic time series models or even a proper recurrent one. One of the key ingredients of our approach was the validation set selection:

*	Validation set 1: ['d_1914', 'd_1915', …, 'd_1941'] (=public LB data)
*	Validation set 2: ['d_1886', 'd_1887', ', …, 'd_1913']
*	Validation set 3: ['d_1578', 'd_1579', …, 'd_1605’] (exactly one year before the private LB)

All modelling and blending were based on improving the weighted mean and standard deviation on these three folds. We used 4 models in our final blend:

### Model 1 (lgb_nas) Tabular regression using 10 LightGBM models on each store
*    Features
      We used publicly available features (thanks to the exceptional work of Konstantin Yakovlev.  
      The features were divided in the following categories:
      1. Categorical
           item\_id, dept\_id, cat\_id

      2. Price related
          price\_max, price\_min, price\_std, price\_mean, price\_norm, price\_unique, price\_momentum, 
         momentum\_m, momentum\_y

      3. Calendar related 
          event\_name\_1, event\_name\_2, event\_type\_1, event\_type\_2, snap\_CA, snap\_TX, snap\_WI, 
          tm\_d, tm\_w, tm\_m, tm\_y, tm\_wm, tm\_w\_end
     4.  Lag related
           1.	Lag only (15 features): sales\_lag\_28, sales\_lag\_29, …, sales\_lag\_42

           2.	Rolling only (8 features): rolling\_mean\_7, rolling\_mean\_14, rolling\_mean_30,  
           rolling\_std\_30, rolling\_mean\_60,  rolling\_std\_60, rolling\_mean\_180,  rolling\_std\_180  

           3.	Lag and roll (12 features): rolling\_mean\_tmp\_i, j for i in [1, 7, 14] and j in [7, 14, 30, 60]. These features were king of tricky because they need recursive evaluation during.
      5.  Mean encodings
           enc\_cat\_id\_mean,  enc\_cat\_id\_std, enc\_dept\_id\_mean, enc\_dept\_id\_std, 
            enc\_item\_id\_mean, enc\_item\_id\_std

*	Model 
LightGBM model using the following configuration:
 ```
{      
  'boosting_type': 'gbdt',
   'objective': 'tweedie',
   'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'subsample': 0.6,
    'subsample_freq': 1,
    'learning_rate': 0.02,
    'num_leaves': 2**11-1,
    'min_data_in_leaf': 2**12-1,
    'feature_fraction': 0.6,
    'max_bin': 100,
    'n_estimators': …,   
    'boost_from_average': False,
    'verbose': -1,
    'num_threads': 12
   }	
  ```
 
We trained 10 models, one for each store using variable number of estimators:

```
    rounds_per_store1={
     'CA_1': 700,
     'CA_2': 1100,
     'CA_3': 1600,
     'CA_4': 1500,
     'TX_1': 1000,
     'TX_2': 1000,
     'TX_3': 1000,
     'WI_1': 1600,
     'WI_2': 1500,
     'WI_3': 1100
    } 
```

### Model 2 (lgb_cos): Tabular regression using 1 LightGBM for all data
* Features
We used publicly available features, divided in the following categories:
    1.	Categorical
       item\_id, dept\_id, cat\_id, store\_id

    2.	Price related 
        price\_max, price\_min, price\_std, price\_mean, price\_norm, price\_unique, price\_momentum, momentum\_m, momentum\_y

    3. Calendar related
event\_name\_1, event\_name\_2, event\_type\_1, event\_type\_2, snap\_CA, snap\_TX, snap\_WI, tm\_d, tm\_w, tm\_m, tm\_y, tm\_wm, tm\_w\_end

    4.	Lag related
        1. Lag only (15 features): sales\_lag\_28, sales\_lag\_29, …, sales\_lag\_42.  

        2. Rolling only (8 features): rolling\_mean\_7, rolling\_mean\_14, rolling\_mean\_30,  rolling\_std\_30, rolling\_mean\_60,  rolling\_std\_60, rolling\_mean\_180,  rolling\_std\_180  

        3. Lag and roll (12 features): rolling\_mean\_tmp\_i, j for i in [1, 7, 14] and j in [7, 14, 30, 60]. These features were king of tricky because they need recursive evaluation during.

    5.	Mean encodings 
enc\_cat\_id\_mean,  enc\_cat\_id\_std, enc\_dept\_id\_mean, enc\_dept\_id\_std, enc\_item\_id\_mean, enc\_item\_id\_std

• Model 
A single LightGBM model using the following configuration:
```
{
 ‘boosting_type’: ‘gbdt’,
 ‘objective’: ‘tweedie’,
 ‘tweedie_variance_power’: 1.1,
 ‘metric’: ‘rmse’,
 ‘subsample’: 0.5,
 ‘subsample_freq’: 1,
 ‘learning_rate’: 0.03,
 ‘num_leaves’: 2047,
 ‘min_data_in_leaf’: 4095,
 ‘feature_fraction’: 0.5,
 ‘max_bin’: 100,
 ‘n_estimators’: 1300,
 ‘boost_from_average’: False,
 ‘verbose’: -1,
 ‘num_threads’: 8
}
```

### Model 3 (keras_nas): NN regression using Keras model with embeddings
We created a neural network model using Keras/Tensorflow that uses low-dimensinoal embeddings for the categorical inputs:
* Features: We used the following subset of the publicly available features. All continuous features were scaled in [0, 1] using min/max values from the training set.
    * Continuous
        *	sell\_price, 
        *	sell\_price\_rel\_diff,
        *	rolling\_mean\_28\_7,
        *	rolling\_mean\_28\_28,             
        *	rolling\_median\_28\_7,
        *	rolling\_median\_28\_28, 
        *	logd:  np.log1p(sales.d-sales.d.min())
        *	snap\_CA, 
        *	snap\_TX,
        *	snap\_WI,
    *	Categorical
        *	wday, 
        *	event\_name\_1,
        *	event\_type\_1,
        *	event\_name\_2,
        *	event\_type\_2, 
        *	dept\_id, 
        *	store\_id, 
        *	cat\_id, 
        *	state\_id


### Model 4 (fastai_cos): NN regression using a fastai tabular package

*	Features
Same as Model 2 (lgb\_cos). Embeddings for item_id were reduced at 10. Used fillna(-1) to fill in non-available entries. No scaling used.
 
*	Model

    * Architecture 
    Following the fastai tabular interface the model was created using
    ```
    tabular_learner(data, 
                layers=[128, 64, 32, 16], 
                ps=[0.05, 0.05, 0.05, 0.05], 
                emb_drop=0.04,
                emb_szs={'item_id': 10})
    ```
The model above was trained for 8 epochs using a custom Tweedie loss function 
    * Loss function
Custom Pytorch implementation of Tweedie loss, that clipped zeros at 1.0e-6 to avoid NaNs
### 
Final blend
For the final blending, we chose weighted geometric mean based on the average metric on three validation sets. We noticed that **keras_nas**  NN predictions had an unexpected large peak on the last day (day 1969). See for example the plot below:


After confirming this behavior to be dominant in almost all cases, we decided to exclude that model’s prediction for the last day. So our final ensembling looks like: 

 


In the table below we present the score of each  model  and the final ensemble for each validation set.				

|    |lgb\_nas|	lgb\_cos|	keras\_nas|	fastai\_cos|	ensemble|
| --- | --- |--- |--- |--- |--- |
|weights days 1-27  	|3.5	|1.0	|1.0	| 0.5	|0.531
|weight day 28 	|3.0	| 0.5	| 0.0	|  1.5	|0.531
|Val. set 1	|0.474	|0.470	|0.715	|0.687	|0.531
|Val. set 2	|0.641	|0.671	|0.577	|0.631	|0.519
|Val. set 3	|0.652	|0.661	|0.746	|0.681	|0.598
|Average|0.589	|0.602	|0.679	|0.667	|0.549
|Std	|0.08	|0.09	|0.05	|0.02	|0.03

		
			
Correlation matrix flatten()-ing the 30490 x 28 prediction:

| Final pred. |  lgb_nas| lgb_cos|keras_nas	|fastai_cos|
| --- | --- |--- |--- |--- |
| lgb_nas |  1.0000| 0.9941|0.9420 |0.9251 |
| lgb_cos | 0.9941 | 1.0000 |0.944  | 0.9281 |
| keras_nas |0.942 |0.9441 | 1.0000| 0.8952|
| fastai_cos | 0.9251 | 0.9281|0.8952 |1.0000 |

## Summary
*	Validation sets are very important. We were lucky enough to select those three because of it was relatively fast to calculate predictions for them (tripled the computation). 
* In time series modelling  you need to be extra careful not to leak information from train to validation set.
*	Model diversity was really important. Ouranos spent many hours trying to stabilize keras_nas outcomes. A tedious process with a lot of trial-and-error that resulted in a hand full of features that gave great diversity
*	Our local validation was **0.549** and our final private score was **0.552** without any magic multiplier. We did tried out various multipliers but 1.0 was the best choice. This was by far the most important result of our methodology. We manage to emulate the performance on the private test. We were confident about the final score but we just didn't know were all the competitors would end up.
*	Talking about multipliers: if we 've used the magic number 0.97 we would end up on the top of the leaderboard  
*	We regret not adding classical methods (R packages hts, forecast) to the ensemble 
*	We ‘ve also played around with recurrent implementations, but they never reached the level maturity so as to include them in the final blend
* Our best submission was our last one!!
