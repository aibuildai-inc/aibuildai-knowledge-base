# 12th place solution

Competition: otto-recommender-system
Rank: #12
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/382834

### 1. Retrieval Stage

    • Re_visitition:
	All past events

    • Co_visitition:
	Top_150 of [ Top_140(type_weighted) + Top_140(buy2buy) + Top_50(time_weighted) ]
	
    • SRGNN:
	Top_172

	DNN model can boost the retriever recall significantly in my solution by using even slightly smaller average candidates number.
	The recall here indicates recall@full_candidates

| retrieval_strategy        | recall_clicks           | recall_carts  | recall_orders | recall_weighted_sum | average candidates number per session
| ------------- |:-------------:| -----:|  -----:| -----:| -----:|
| all_past_events \| top_258_co_visitition  | 0.697| 0.555 | 0.732 | 0.675 | 236.47 |
| all_past_events \| top_150_co_visitition \| top_169_CORE_trm     | 0.723   |   0.575| 0.742| 0.690 | 234.26 |
| all_past_events \| top_150_co_visitition \| top_172_SRGNN     | 0.734  |   0.581 | 0.746 | 0.695| 234.07 |

Combine multiple DNN models does not work for me, single SRGNN performs best under dozens of experiments.


#### Data preparation & augmentation for SRGNN:

Delete items appearing less than 3 times.

train_set: 3rd_week + truncated_4th_week_former_part

valid_set: truncated_4th_week_latter_part (with former part as basis)

##### Augmentattion: 
Train_set: For each sequence, iteratively seperate last item as target, truncate or pad the rest to max_length of 50 as training sequence, until only 1 item left.

Valid_set: For each sequence, use truncated_4th_week_former_part as basis, then iteratively concatenate truncated_4th_week_latter_part as target, until last item in the latter part, as illustrated below:
.png?generation=1675234390597141&alt=media)

As in test case, we have former part as basis for prediction, so I used this method to best mimic this scenario. It increased the validation samples as well compared with using latter part only, in which case the first item of latter part can not be used as target.

The validation metric is recall@20, which correlates very well with my retrieval strategy's final recall across all experiments & DNN models.

I did not try other augmentations, such as Swapping, Shuffling, etc..., simply because the data scale is already considered huge for my machine.

### 2. Ranking Stage

#### CV strategy for ranking:

Using only the 4th week as training data for forming the session-item pairs.
For each type: clicks/carts/orders, drop all sessions without positve targets, then do GroupKfold for each, respectively.

Negative sample downsampling: positive:negative = 1:20

#### Feature Engineering:
1. Session features: statistics about session: 

    session_clicks/carts/orders_count, 
    
    session_length
    
    session_items_nunique
    
    session_duration 
    
    etc...

2. Item features: statistics about item:

    simple statistics
    
    item_total_count
    
    item_count as clicks/carts/orders
    
    clicks/carts/orders_counts in each week
    
    clicks/carts/orders_counts changing trends in temporal order, weekly basis
    
    clicks/carts/orders_unique_session_counts in each week
    
    clicks/carts/orders_unique_session_counts changing trends in temporal order, weekly basis
    
    etc...

3. Session-item interation features:

    item appearing count in session
    
    item appearing as clicks/carts/orders count in session
    
    item recent 1 hour/day/week clicks/carts/orders count from prediction timestamp
    
    relative time gap between the item clicked/carted/ordered last time and prediction timestamp
    
    co_visition weights(time weight, type weight, just count) for items in each session
    
    dnn model cosine similaries between session and item embeddings
    
    etc...


#### Models used:
Lgbm, Catboost, Xgboost, MLP

Classification and ranking algorithm perform similarly in my solution.

And ensembling classification and ranking does not provide any boost.

So only binary classification is used at the end.


#### Ensemble:
Blending of 4 models, each with 5 folds.

Stacking with meta model leads to a silimar performance, thus only blending is used.

#### Post processing:
There are 3327 sessions with less than 20 candidates.

So I re-calculated the co_visitition items for these sessions without constraining the time range, namely with full data.

After this supplement, the average candidates number is above 19.

Using hottest items within half day around the prediction timestamp to supplement the last drop.


### Acknowledgement

Thank @radek1  for introducing polars to the community, which saves me in this competition with limited ram.

Its lazy evalution and native multithreading support make it such a nice Lib on CPU platform.

Thank @cdeotte for sharing the knowledge and valuable notebooks as always.

Thank all guys who shared their knowledge and insights in notebooks and discussions. Learned a lot from you guys.
