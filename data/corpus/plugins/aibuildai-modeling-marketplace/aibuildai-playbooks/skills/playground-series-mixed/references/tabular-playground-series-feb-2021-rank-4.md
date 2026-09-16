# 4th Place Solution

Competition: tabular-playground-series-feb-2021
Rank: #4
Source: https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222791

# TPS Feb 2021 - 4th Place Solution

First of all, congratulations to the winners of the competition @ryanzhang, @davidedwards1, and @kntyshd. The DAE solutions by @ryanzhang and @davidedwards1 are very elegant. As well, thanks to everyone for sharing thoughts, ideas, and code with the community. There were some fantastic EDAs, excellent notebooks, and really great discussions about approaches, model types, encodings, noise, and pre-processing that were very helpful. It's been a lot of fun!

I'll break down my solution into two parts, the technical bits, and then some of the useful lessons learned. I think that the useful lessons are probably more insightful, because the technical solution isn't incredibly novel and is actually quite boring. My 4th place standing is really a combination of luck and intuition when it came to trusting my own metrics vs. the public leaderboard.

## Technical Information

My submission was generated using a two level stacked approach:

* Level 1:
  * 14 CatBoost models (categoricals treated with CatBoost encoding)
  * 17 XGBoost models (categoricals treated with Leave One Out encoding)
  * 27 LightGBM models (categoricals treated with Label encoding)
  * 10 Random Forest models (categoricals treated with Leave One Out encoding)
  * 10 Ridge regression models (categoricals treated with Leave One Out encoding)

* Level 2:
  * 1 Ridge regression model

I'll get into why I generated so many level 1 models in a moment. Each level 1 model was trained using 10-fold cross validation (importantly, each model used the same set of fold data during training). The training predictions for each fold were saved for use as a training set for the second level. Predictions on the test data were generated out-of-fold and became the test input for the second level. 

The level 2 model was trained on the previous level's training predictions, again using 10-fold cross validation. The out-of-fold test predictions from the previous level were used as the test set, and again, final test predictions were made out-of-fold. The end result was a model that had a cross validated RMSE score of 0.84159, and a public leaderboard score of 0.84190.

All of these models were trained using a custom Python framework. The idea was to create an abstract regression model class that hid the inner workings of how to build CatBoost, XGBoost, LightGBM, Random Forst, and Ridge regression models (plus a few others that I didn't have time to incorporate). Taking inspiration from the discussion thread and notebook posted by @hamzaghanmi about [LGBM Hyperparameter Tuning Using Optuna](https://www.kaggle.com/hamzaghanmi/lgbm-hyperparameter-tuning-using-optuna), the framework performs a random search of each model's parameter space and saves models and out-of-fold training and test predictions that fall below a certain RMSE score threshold. The framework can then aggregate the result of multiple models, turning them into the training and testing sets for the second level. Theoretically it can stack as many levels as you want, although my own testing showed that stacking more than 2 levels yielded poor results. Additionally, the second level model could have been a LightGBM or CatBoost model, but again, my own testing and cross validation showed that those models performed worse than a Ridge regression. For those interested, it took around 36 hours to generate the models using a combo of two machines, one with a 3rd gen core i5 and 24 GB of ram, and the other with a Ryzen 5 3600X with 32 GB of ram and a GTX 1060 GPU.

## Lessons Learned

Here's where I think I can contribute some more useful information.

### Noise

Early on in the competition, several EDAs showed that features were not highly correlated with one another, and experiments demonstrated that dropping features with low importance produced models that performed poorly in comparison to others that kept all the features intact. Additional experiments with feature engineering such as polynomial feature generation, binning, normalization, scaling, and others, yielded poor results. I found similar results with various encoding methodologies. All of that meant that the competition was going to come down to how well a model could deal with the noise in the data. Looking back at the previous month (which I didn't get a chance to take part in), DAEs very much appeared to be the superior solution. While I spent a fair amount of time experimenting with DAEs, I wasn't able to make much progress with finding optimal noise mixtures (but there's always next month). 

### Multiple Models

At that point, I started experimenting with a single LightGBM model to see how well I could tune it. Digging deeper, I learned two things:

* Widely different values for parameters such as `max_bin`, `cat_smooth` and `num_leaves` were generating decent models. My thought was that each model was adjusting for noise in different ways. If this were true, then stacking was probably going to give a better result than a single model, since the stack could exploit the different fit of each model overall.
* While a single model could be hyper-tuned to give a good leaderboard score, I was worried that I was going to overfit both the training data, and the leaderboard.

This is what prompted me to look at stacking. I generated large numbers of the same type of level 1 model by searching through the parameter space of each model type. While the resultant predictions from each model type were highly correlated, they weren't identical (in general, you want to stack models that produce results that are _not_ highly correlated with one another). I figured the second level in the stack could deal with the high correlation, which is exactly what the Ridge regression did.

### Trust Your Cross Validation

My hyper-tuned single model result was doing better than my stacked models on the public leaderboard, but my cross validation results were saying something different. The question was which model should I trust? The stacked approach was doing much better with cross validation, and to me that signalled that the single model solution probably wasn't going to generalize well. I saw the difference between my position and 4th place on the public leaderboard was a score of 0.00011, and I figured that gap could easily be closed with differences between the public and private leaderboard data distributions and a model that generalizes better. So I stopped following the leaderboard and concentrated more on ensuring the stacked model CV was good. 

### Always Save Fold Predictions and Models

I learned late in the month that I should have been saving my out-of-fold predictions and models every time I generated a model (I've read this as well on several Kaggler's websites that I can't find the links to at the moment). This was particularly obvious in hindsight after running something like `GridSearch` or `Optuna` for several hours. Why throw away predictions of somewhat decent models when you can use them later in a stacked approach? It was painful to find a set of decent parameters, and then have to retrain the model later to make predictions again. Also, make sure you use cross validation in your parameter searching to make sure you're getting a good look at how well the model parameters perform!

# Conclusions

The competition was fun! Thanks again to everyone who took part in discussions and code sharing! It's great to be a part of such a friendly community!
