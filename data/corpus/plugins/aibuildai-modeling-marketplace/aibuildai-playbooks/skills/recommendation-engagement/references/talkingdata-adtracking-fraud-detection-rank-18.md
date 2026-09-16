# 18th Place Brief Solution Overview

Competition: talkingdata-adtracking-fraud-detection
Rank: #18
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56422

kazanova has, IMHO correctly, pointed out that the 1st and 4th place finishes are perhaps the best ones in any competition. I'd argue that finishing one place out of gold might be the hardest one. A slightly different choice of final blend, one or two more tricks that we didn't use, could have made all the difference. It calls for a lot of soul-searching, and makes you engage in a lot of counterfactual thinking. Nonetheless, my team and I are overall pretty happy with what we had accomplished. I want to thank my teammates - @psilogram, @andrenaef, and @dicksonchin93. They are all incredibly talented, smart, fun, and kind people. I'm sure you'll be seeing a lot more of them in the upcoming competitions. 

Here is a brief overview of our solution(s):

Silogram's model was a single LGB bagged 10 times with different training data samples. First, data was split into four 24-hour datasets (day7 - say10) matching the hours in test_supplement. All features were then generated within each day without considering data from other days. Features consisted of the typical frequency stats by group and seconds to previous/next clicks that have been mentioned by others, as well as target stats from the previous day. The final model had 24 features. For validation, the model was trained on day 8 and validated on  day 9 (looking only at hours that matched the test set hours). To save time, training was done on a sample of 2% negative observations plus all of the positive observations. For the submission, the model was trained on a sample of data from day8 and day9, with the number of iterations increased 1.9 times over the best validation iteration count. The LB score was 0.9814 and the PB score was 0.9820.

André's model was a single LGBM based on ~90 features trained on 5% of the data, plus all positives. Beside common features, he was also using log odds from a statistics timeframe, and applying SVD to those to fill in the blanks.

Chin (Ee Kin) model:

Build primarily models on LGBM and XGB. Fixed Validation split of day 9 4am to 3pm mimicking time period of test set, trained both types of models with validation and after that used best iteration from cv to train on full data set, got the max increase of 0.0006 for LGBM with this method and XGB 0.0022. For my LGBM full training dataset is used without sampling, for XGB the focus is to create variation to our models for ensembling so resorted to sampling methods, tried multiple configurations but for my best XGB model is with a ratio of 1:5 positive to negative examples, with oversampling positive samples without replacements at 10%. Utilised @Nano Mathias Bayesian optimization script to obtain lgbm and xgb parameters for training with slight manual tweaking based on intuition. Besides common features, was using two step next clicks and previous clicks, some statistical info based on time and normalized 0-1 for some features which only contributed a little the final model but its used, around 40+ features. Didn’t use ip as a feature for training directly based on several popular discussions made on it. Things that didn’t work for me is target encoding and pseudo-labelled target encoding, burnt two weeks trying different versions...

Bojan's model(s):

I also used more or less the same features as everyone else. I used three different validation schemes: 4am to 3pm on the 9th day, all of the 9th day, and random splits. I built several LGBM and XGB models, and even used one fm_ftrl model based on @anttip's now famous kernel. Training the model on different subsets of training data gave a nice boost for blending, as did using different boosting algos. I wish I had the time to build a catboost and RGF models. Early in the competition I was also considering building a solid NN model, but since most people were not able to blend it well with the rest of their submissions, it seemed that it would not help much, so I never went down that rout. It also didn't help that I have access to a machine with lots of RAM and a machine with lots of GPU power, but not one that has both. :-) 

Our final submission was a blend of various models. We experimented with various weights for our submodels, but it seems that something close to almost equal weights would have been as good as almost all the other options that we tried.
