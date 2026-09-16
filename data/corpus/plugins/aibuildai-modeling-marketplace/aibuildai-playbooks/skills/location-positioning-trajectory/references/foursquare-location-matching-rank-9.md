# 9th place solution

Competition: foursquare-location-matching
Rank: #9
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/336415

## setting
evaluate local by 2fold CV
groupKfold by POI

## 1st : candidate selection
- selected 400 candidate for each id
  - knn with latlon: 350 candidate
  - knn with latlon + name tfidf: 50 candidate
-  Max IoU: 0.994, 420M candidates

## 2nd : modelling light model
- catboost with tfidf similarity 
  - features (40 features)
     - text similarity with tfidf
     - raw & diff int of text
     - raw & diff latlon
  - threshold is 0.005
  - Max IoU: 0.987, 3M candidates

## 3rd : modelling
- catboost and lightgbm
    - catboost with text encoding each text
        - 0.010 improved with text encoding
  - features (2nd40 + 100 features)
    - text similarities with gesh, LCS, leven, jaccard... 12 methods
    - count ids, match ids
    - target encoding each pair of ['name', 'name_match'], ['categories', 'categories_match'] (2fold)

## 4th : post_process
- calc the node predictions with avg of neighbors predictions, order by predictions
    - thresholds: 0.4

## score
- Final CV: 0.905, 2fold CV
- LB: 
    - catboost(cb): 0.925
    - cb + post_process(pp): 0.930
    - cb + lgb + pp: 0.933
    - cb + lgb + pp + training all data(1fold): 0.943
    - cb + lgb + pp + all data + ensemble 5models  0.945
    - cb + lgb + pp + all data + ensemble + increase iterations(over fit): 0.948
