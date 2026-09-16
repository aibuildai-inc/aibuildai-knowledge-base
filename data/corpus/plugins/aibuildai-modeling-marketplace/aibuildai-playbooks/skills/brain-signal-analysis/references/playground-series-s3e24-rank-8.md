# #8 Private LB #7 Public - Solution Approach

Competition: playground-series-s3e24
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s3e24/discussion/455268

Hello All!

Thanks to Kaggle for this episode, the size of the dataset had resulted in a consistent Public to Private LB. I would like to thank the other participants for their contriubtions & the publicly available notebooks.

Special mentions to the following individuals for their notebooks & being highly active in this episode:
1) [@paddykb](https://www.kaggle.com/paddykb), his [notebook](https://www.kaggle.com/code/paddykb/pg-s3e24-brute-force-and-ignorance) has contributed in predictions of many other notebooks including mine. 
2) [@cv13j0](https://www.kaggle.com/cv13j0), amazing notebook for reference
3) [@ravi20076](https://www.kaggle.com/ravi20076), thanks for the contributions as always and congrats again!


The appraoch is based on my publicly available [notebook](https://www.kaggle.com/code/arunklenin/ps3e24-eda-feature-engineering-ensemble/notebook)

My approach: 

###Data Processing

Thanks to [@paddykb](https://www.kaggle.com/paddykb) for the data pre-processing

###Feature Engineering

1) All features with frequency above 2 were treated as discrete and many encoding techniques were applied. The dataset is large enough to enable this approach. 
2) New features are created using brute force search for arithmetic combinations of existing features based on performance ciriteria. 
3) Selected the union of top N features from CatBoost, XGBoost, LightGBM where N is 50, 100. Higher values of N resulted in "Run Time Exceed Error"

### Modeling Framework

1) XGBoost, Catboost, lightGBM, Artificial Nueral Networks, Logistic Regression, & DecisionTrees are ensembled to maximize the AUC score using Optuna
2) Adding ANNs into ensembling framework was recently added and gives decent performance. 
3) My predictions are again ensembled with publicly available predictions with ranks based on public LB as weights. 

### Takeaways

1) Running my notebook took a lot of time and a many submissions are stopped because of runtime limits. 
2) A lot of notebooks using predictions from other notebooks and provided weights based on trial & error approach and there are discussions if this is the right approach or not. My thought is simply to use an outcome that is produced from a differently feature engineered data to make a generelized prediction. howvever, I would suggest to use a logical way to assign weights because we only have a part of the information available in the Public LB and anything can happen in the private LB. 
3) For users finding difficulty in identifying true notebooks that do not use other's predictions, one way is to look at the number of inputs which is in general the main dataset along with the original dataset in PS series. 


####Thank you all! All the best for the next episode!

####I wish everyone a Happy Diwali!
