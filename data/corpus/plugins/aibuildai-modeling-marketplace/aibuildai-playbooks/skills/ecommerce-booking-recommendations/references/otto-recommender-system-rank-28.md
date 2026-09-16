# 28th Place Solution

Competition: otto-recommender-system
Rank: #28
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/382812

First of all, we would like to thank OTTO for organizing this competition. It was the first RecSys experience for most of us and we learned a lot. I'll try to talk about the things we've all tried at each stage of our solution, and the ones that didn't work at the end.

In addition, @tetsuro731 has opened a separate thread for [the stacking method he applied](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382955)

**In addition, the links to our public repos are below:**
Anil: [GitHub Repo](https://github.com/nlztrk/OTTO-Multi-Objective-Recommender-System)
Gunes: [GitHub Repo](https://github.com/gunesevitan/otto-multi-objective-recommender-system)  
Tetsuro: [GitHub Repo](https://github.com/tetsuro731/OTTO-kaggle-tetsuro731)  
***
## Generating Covisitation Matrices
#### Anil & Ayberk
- Generated **Top-100** AIDs for three well-known covisitation schemes given in: [link](https://www.kaggle.com/code/tuongkhang/otto-pipeline2-lb-0-576)
#### Gunes
- 7 covisitation matrices like the ones in public but slightly different (click/cart/order weighted, only click/cart, click/order, cart/order and time weighted)
- Created a counter from **7** covisitation matrices and **50** nearest neighbor of last aid in session using fasttext model + annoy
#### Tetsuro
- Generated co-visitation matrix with several parameters/weights based on the public notebook.
***
## Splitting the Data as Train/Val
#### Anil & Ayberk
- Used the [local validation scheme](https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation) given by [Radek](https://www.kaggle.com/radek1). The local train data was also splitted into two by sessions in order to avoid possible leakage during model training and score calculation. The implementation can be seen in the corresponding notebook.
#### Gunes
- Radek’s split is used as the validation set. Every approach is applied on two different datasets; training + validation and entire dataset. First one is used for the validation and second one is used for the submission.
#### Tetsuro
- Didn’t use 5 weeks and used 4 weeks for training because the data for the prediction have only 4 weeks.
***
## Generating Co-Occurrence Matrices
#### Anil
- Generated all pair occurrences for all AIDs among all sessions for all action pairs (click-cart, cart-order, etc.). This is used for feature extraction later.
***
## Candidate Generation
#### Anil & Ayberk
- Used the [public candidate generation script](https://www.kaggle.com/code/tuongkhang/otto-pipeline2-lb-0-576) and generated **100** candidates for all action types. The rank of the candidates generation with several parameters/weights are also used for the features for the ranking model.
#### Gunes
- Generated co-visitation matrices slightly different than publicly shared ones
    - 12 hour difference 15x click weighted
    - 12 hour difference 15x cart weighted
    - 12 hour difference 15x order weighted
    - 12 hour difference only click and cart
    - 12 hour difference only click and order
    - 14 day difference only cart and order
    - 24 hour difference time weighted
- Trained and tuned a FastText (0.547 lb score) skipgram which worked better than gensim word2vec and it was faster because of c++ bindings
- Created a counter from 7 covisitation matrices and 50 nearest neighbors of last aid in the session using FastText model and annoy (100 trees)
- 80-100 candidates are selected using the approach above and session unique aids are concatenated to candidates for each session
#### Tetsuro
- Created **Top-100** candidates.
- Added click/cart/order Top-50 Popular items for each candidate. Finally, 150 candidates are generated which are used for the next ranking phase. The rank of the candidates generation with several parameters/weights are also used for the features for the ranking model.
***
## Feature Extraction

Generated features for following data subsets:
- Items
- Sessions
- Item-Session Combinations
- Covisitation and Co-Occurrence Statistics

### Item Features
####Anil
- Statistics generated from hour, weekday and weekend status
- Count features (bool for >0 and >1, rank among all)
- Unique count features (unique count and rank among all)
- Distribution of action types in percentiles
- Inclusion rate by all sessions
- Occurrence rate in the last week of data
- Average number of times seen in the same sessions at different times
- All of the above with filtered separately for all action types
#### Gunes
- type mean/std
- day of week mean
- hour mean
- is session start/end mean
- candidate score mean/std/min/max
- timestamp ratio/difference

All of those aggregations generated for each type, last week, last 1, 2, 3, 4, 5, 6, 7 days (all of the aid aggregations with count and nunique are replaced with rank percentile because two datasets had different sizes)
#### Ayberk
- Unique count features
- Count features
- Time features
- Rank of a item in sessions
- How many times item clicked after bought
- Session based features like how many time item clicked per session
- Weekly change rates
- click-to-cart , cart-to-buy ratios

These aggregations generated per last 2 week and type.

#### Tetsuro
- Popular item feature
- The number/rank/unique number of click/cart/order during 1/2/4 weeks
- Counts/Unique counts of click/cart/order for each session
- The ratio of these features.

### Session Features
#### Anil
- Statistics generated from hour, weekday and weekend status
- Count features (bool for >0 and >1, rank among all)
- Unique count features (unique count and rank among all)
- Distribution of action types in percentiles
- Length of the session
- Features generated by extracting mini-sessions according to the time differences between actions
- Statistics generated from multiple purchases made in a single basket
- Rates of taking products to the next action within the same session (click->cart, cart->order)
- All of the above with filtered separately for all action types
#### Gunes
- Aid count
- Unique count
- Count of last aid
- Type of last aid
- Last aid itself
- Timestamp ratio/difference
#### Ayberk
- Session unique aid count, length, time features
- Session click-to-cart , cart-to-buy ratios

These aggregations generated per type.
#### Tetsuro
- Mean number of click/cart/order counts/unique counts
- Session duration (min and max diff of session timestamp)

### Item-Session Combination Features
####Anil
- Statistics generated from hour, weekday and weekend status
- Count features (bool for >0 and >1, rank among all)
- Unique count features (unique count and rank among all)
- Distribution of action types in percentiles
- Reversed order of the item in the session
- Time difference between the latest occurrence of the item and the start - end of the session
#### Gunes
- Candidate count in session
- Candidate click, cart and order count in session
#### Ayberk
- Count features
- How much time passed since last action.
- Inverse rank place in session

These aggregations generated using all of the types and per type.
#### Tetsuro
- Diff/ratio between item and session which are related to similarity.

### Covisitation and Co-Occurrence Statistics
#### Anil
- Statistics generated from covisitation and co-occurrence scores between candidate items and items in the session's history
#### Ayberk
- Co-visitation scores used.
***
## Training
#### Anil
- **Model:** XGBoost
- **Fold Scheme:** 5-Fold (Grouped by "session")
- **Negative Sampling Fraction:** 15%
- Dropped sessions with no positive labels
- Used the first half of splitted local training set
#### Gunes
- **Model:** XGBoost and LightGBM
- **Fold Scheme:** 5-Fold (Grouped by "session")
- **Negative Sampling Fraction:** 30%
- Dropped sessions with no positive labels
#### Ayberk
- **Model:** LightGBM
- **Fold Scheme:** 5-Fold (Grouped by "session")
- **Negative Sampling Fraction:** 20%
- Early stopping used.
- Dropped sessions with no positive labels
- Used the first half of splitted local training set
#### Tetsuro
- **Model:** LightGBM
    - Metrics: nDCG@20
- **Fold Scheme:** 5-Fold (Grouped by "session")
    - Checked recall for each fold
- **Negative Sampling:** sampling negative to 2.5% positive.
- Dropped sessions with no positive labels
***
## Inference
#### Anil
- Used mean blending
- Executed on the second half of splitted local training set when running local validation
#### Gunes
- Used mean blending
#### Ayberk
- Used mean blending
- Executed on the second half of splitted local training set when running local validation
- Filling orders predictions in the end with carts first non duplicate predictions (last 2)
#### Tetsuro
- Used mean blending
***
## Submission Blending
Our blending approach was pretty straightforward.
- Scale each prediction with robust scaler
- Outer join different predictions on session and aid
- Fill missing values with 0
- Do a weighted sum based on OOF scores
***
## Didn't Work & Improve
#### Anil
- Weekday-Specific aggregations
- Word2Vec features
- Different models (CatBoost, LGBM)
- Comprehensive pair scores (because of OOM errors)
- Max-median blending
- Early-stopping
- Higher negative fractions
- Different objective metrics
- Different fold counts
#### Gunes
- **Collaborative filtering:** Either the scores aren't good or the embeddings become too large and inference is too slow
- **Matrix factorization:** Same as collaborative filtering
- **Doc2vec:** Model doesn't learn anything
- **Sequential models from recbole library (GRU4Rec, BERT4Rec and etc.):** Training is too slow and they are not competitive
- **General models from recbole library (BPR, CF and MF models):** Training is fast but inference is too slow because models don't scale
- **Models from surprise library:** Same as general models from recbole library
- **TF-IDF + pairwise similarity:** Very slow inference time since I was using argsort to get top 20 most similar aids
#### Ayberk
- Daily aid features
- User2User Similarities
- Tried MLP architecture with various settings (pairwise type prediction, session-based multi class)
but couldnt beat heuristic approach by Chris and as i used only kaggle resources it was hard for me to keep improving so i dropped this approach at early stages.
- Candidate generation using item embeddings from MLP architecture
- Higher negative fractions
- Candidate ensemble with best @20 recall and best@N recall as it would rank candidates better but didnt work
#### Tetsuro
- Weekday-Specific aggregations
- improve nDCG but local recall hadn’t improved
- Wanted to try W2C based method but I didn’t have time to do.
