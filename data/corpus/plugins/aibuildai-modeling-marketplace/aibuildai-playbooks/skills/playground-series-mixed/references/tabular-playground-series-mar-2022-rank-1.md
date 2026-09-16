# [1st] Disbelief

Competition: tabular-playground-series-mar-2022
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-mar-2022/discussion/316271

I was quite surprised to find that the LB shakeup was as high as it was.  I almost didn't check tonight because I was so disappointed in my scores.  While I would like to congratulate myself as being some sort of data-science genius, I think I got more than a little bit lucky in this case.  I know many smarter, more experienced competitors worked much harder than me this month.  Sometimes you win the lottery, I guess. 

I am grateful to all the kagglers who have shared their insight via the discussions and code.  Specifically, I was helped by [@Pack-Man's](https://www.kaggle.com/packinman) [AutoML notebook](https://www.kaggle.com/code/packinman/tps-mar-2022-automl-pycaret-regression) and [@Martynov Andrey's](https://www.kaggle.com/martynovandrey) [Hybrid Regressors notebook](https://www.kaggle.com/code/martynovandrey/tps-mar-22-hybrid-regressors).  From the first I took a way to likelihood encode based on the cross between hours-minute time and location.  From the second, I liked the post-processing.  

# Broad Overview 
The winning score was a single lgbm with no post-processing.  I kinda ran out of time.  First, I created a substantial amount of likelihood encodings and lag features.  Unlike most public kernels, I used many other features than the medians.  Second, I found the best subset of the features using Optuna.  

For validation, I used the data that were exactly 1 week before the test.  I filtered the train data to only be during the same day and hour-minute combinations as the test period.  The feature creation of the winning model is [here](https://www.kaggle.com/code/ottpocket/encodings-and-lags?scriptVersionId=91804908) and the feature selection of the winning model is found [here](https://www.kaggle.com/code/ottpocket/feature-selection-notebook).  
## The Lags and Likelihood Encodings
[This notebook](https://www.kaggle.com/code/ottpocket/encodings-and-lags) is the code I used to create both the lag features and the encodings.  For the lagged features, I found means, variances, medians, minimums, maximums, and 1 interval shifts.  I took this for every x-y-direction combination on both the day and weekday.  I used both 3,5, and 10 day rolling windows and expanding windows.  

The likelihood encodings took the minimus, maximums, medians, variances, and means for every `xy` and `x-y-direction` combination at all `hour-minute` combinations.  
The whole process takes around 5-10 minutes to run.

## Optuna for Feature Selection and Train Filtering
After step 1 I had too many features.  Given the surplus of features, I used Optuna to pick the best features.  Specifically, I used `trial.suggest_categorical(feat_in_question, [True, False])`, for each of the features I considered.  After 300 trials, Optuna got very good at finding the best features.  Moreover, I saw that many notebooks only training on weekday 0.  I had optuna find whether I should train on only day 0 and, additionally, if I should only train on the same hour-minute times as the test.  After the Optuna trials, I just picked the best features and submitted the notebook.  The code for this is [here](https://www.kaggle.com/code/ottpocket/feature-selection-notebook).

## Failures
[Wide Neural Networks](https://www.kaggle.com/code/ottpocket/wide-neural-predictions-with-optuna).  Given the spatial relationship of the data, it would make sense to train a regressor on all the spatial points at a given time simultaneously.  This did not work out for me, however.  


# Special Thanks

I want to give a special thanks to the "How to Win a Data Science Competition" course on coursera, put on by the Higher School of Economics.  This course taught me the basics of likelihood encodings, which account for much of the score here.  The encodings I did here were not very pretty as they did not use k-fold validation to prevent leakage.  Another case of not having enough time.  

I have to thank the general kaggle community for all their help along my journey as a data scientist.  From dropping out of school two years ago to working as a data scientist now, kagglers have helped me learn much more than I ever could on my own.  Thank you, reader, for being part of this great community.  Keep kaggling.

It is 11:30 here, and I wake up at 4:45 to go to work.  Thanks for reading!
