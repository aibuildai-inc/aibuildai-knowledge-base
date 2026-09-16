# #3 place solution

Competition: bnp-paribas-cardif-claims-management
Rank: #3
Source: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion/20258

I think our solution may sound too plain compared to what the top 2 teams did :) .

We spent quite some time reading forum threads and applied many of things discussed with some success like :

1) rounding features to remove noise
2) finding the correct denominator of features to get them back to the initial integer state
3) converting categorical variables to likelihoods

They all helped  a little bit but the most important thing for us (and thanks to clobber for finding it! ) was interacting the categorical variables together , especially  v22 with all the rest categorical variables in 2way and even 3way manner. 

Another thing that helped was rounding strongly ( 2 decimals) all the numerical features and creating interactions (as categorical "numericalrounded_catgeorical" with some of the categorical ones.) 

Stans was kinda of our features'factory machine! - apart from likelihood and other target-based feature transformations on categorical data, he did a lot of extra work in creating group-by type of aggregate features (like averages) of v50 and other variables (up to 3 way). Although we never understood exactly why this works, stans was able to capture some very useful patterns here .

Our best single xgboost would have finished 4th (0.4242 in private) and was consisted of more than 7.5 K features (mostly interactions).

We also produced many different base level models without much Feature engineering, just different input format types (like load all categorical variables as counts, or as onehot encoding etc).

Our ensemble was consisted of 223 models. Faron did a lot of work in removing noise and discarding many of these in order to get to our bets score with a lvl2 ensemble of geomean weights between an ET ,  2NN and 2 Xgmodels. 

We burnt many cores in case you might have questions about it!
