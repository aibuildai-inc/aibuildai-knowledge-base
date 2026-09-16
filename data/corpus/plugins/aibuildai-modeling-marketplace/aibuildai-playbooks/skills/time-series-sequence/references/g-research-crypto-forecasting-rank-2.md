# 2nd place solution

Competition: g-research-crypto-forecasting
Rank: #2
Source: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/323098

Hi everyone,

This is great. I'm relieved to have maintained second place in the final update, as I was nervous that anything could happen. By my estimate, there was a 60% chance I would drop below 2nd on the 6th update, due to the noisy nature of this kind of data. I also estimate that on a 7th update, Meme Lord Capital would have a 75% chance of maintaining first place. They finished with a big lead on everyone. Good job, Meme Lord Capital!

I'd like to thank G-Research for sponsoring the competition, and everyone involved in organizing and hosting it.

# Introduction

In this writeup I'll strive to provide insight into my methods, but without giving away model details that could be profitable for the host. When a good financial indicator becomes common knowledge, everyone uses it, therefore it loses profitability. As such, I won't be sharing my code or any specific insight on features. 

I decided to enter this competition because it concerns asset price prediction using actual market data (not pre-engineered features). That's the kind of data I like: having an explicit time dimension, many interacting entities, a lot of noise, and a need for creative method development. In 2014-2017, I researched stock price analysis and prediction in my own time. It didn't end up being profitable long-term as a single DIY investor, but I did learn a lot about predicting time series and asset prices.

# Toolkit

My toolkit consisted of CPython, Numba, Jupyter, Pandas, LightGBM, Matplotlib, and scikit-learn. Using the Numba compiler instead of CPython for feature generation resulted in high performance. Doing so, I was able to experiment with a wide variety of features, utilizing the entire dataset and iterating quickly.

# Cross validation

Setting up good N-fold cross validation (CV) is essential. We should try to see if our ideas fail in the most real environment we can make. Every time we have a research question, it should (eventually, after a period of creative exploration) be answered in that manner. If we don't have a good CV setup, our decisions will mostly be mere guesses. Good N-fold CV is easy to set up, taking less than 100 lines of code, without using scikit-learn or any timeseries framework. 

In this competition I used 6-fold, walk-forward, grouped cross validation. The group key was the timestamp. In a typical setup, train folds were 40 weeks long, test folds were 40 weeks long, there was a gap of 1 week between test and train folds, and the ends of training folds were incremented by 20 weeks each fold. 

I chose to overlap my folds so that they could be long, but I could still have 6 folds. With non-overlapping folds, I would have to decide between many short folds or a few long ones. 

The advantage of having many folds is that average CV scores will have lower variance. On the other hand, the advantage of having long folds is that the model sees more data, so you get a better picture of how it will perform with the full dataset. Instead of choosing between many short folds or a few long ones, I let them overlap so I could have many long folds. I could have used more than 6 folds, or run CV multiple times, changing seeds. I didn't do these things because CV score variance was decently low, and CV was already almost too slow.

(If you don't pay attention to CV variance, you could waste days optimizing your model, only to find that your early decisions were based on noise and incorrect. An easy way to see if your CV scores have too much variance is to look at a plot of CV score vs. a parameter you're tuning. A good plot will usually be smooth, with a knee and a plateau, or maybe a peak or a valley. If the plot looks too noisy to clearly see those things, then try it again, with different seeds, and see if you get different results. If you do, then your CV results are too noisy. You can try using more folds, or running CV several times with different random seeds and averaging.)

The one week gap between train and test data was to prevent CV results from being too rosy. With no gap, a model can cheat at the the beginning of the test period, because the end of the train period is very similar. There was no gap between train and test data in the final submission.

# Public leaderboard scores

As I was developing my model, naturally I wanted to have an idea of how it would place against competitors. Because public leaderboard scores were unrealistically high due to probing and overfitting, I had to find a way to interpret them. So I looked at masters and grandmasters with "low" scores. I figured these competitors were good enough to submit good kernels, but had been careful to keep the public leaderboard period out of their training data. I did not expect true scores to go above 0.1, based on experience and discussions here. Plotting the sorted scores of the "low" scoring masters and grandmasters, there was a clear plateau around 0.08, as I recall. That was the score I aimed to beat in my local CV. 

Of course, I did not use the public leaderboard as a guide in any optimizations. That's always a bad idea (see the paragraph above about CV variance). I only made a total of two submissions in the competition.

# Feature engineering

Because my labor and computational resources are finite, it’s important to know where to direct them. As I developed my model, feature engineering was guided by feature importance. As I made new features, I focused further effort on developing features sets that already performed well (had high importance), or for which transformations led to easy increases in importance. 

As I said in the introduction, I won't be discussing specific feature insight or give my code. I'll only be giving my two cents on methods.

# Learner

The learner was trivial: a LightGBM GBDT regressor with squared loss. There was no ensembling other than the gradient boosting in GBDT. The only parameters I changed from defaults were the number of estimators, number of leaves, and the learning rate. There was no regularization, augmentation, or feature neutralization. I checked with CV whether these things would help, found that they wouldn't, and decided to go with a simpler submission.

LightGBM parameters were not tuned exhaustively. Tuning was taking a long time, and CV results were indicating that regularization would have little effect on model performance. So I decided to not push on that wall and instead placed my focus back on feature engineering. CV is essential in making these resource allocation decisions.

# Submission kernel

Because this was a code competition with a large dataset, performance was crucial. Our kernels were limited to running in 9 hours or less, using up to 16GB of RAM. As we all can attest, it's easy to bump into those limits.

The hardest limitation I dealt with was the 16GB RAM limit. I wanted to train with the entire dataset, because CV had shown me that scores just kept improving with longer training data. To use the entire dataset, I had to find and fix code that resulted in unnecessary RAM copies of arrays. The Pandas operation df.update() was a frustrating offender. The solution was to select small subsets of data, use df.update() on those subsets, then append to a list of dataframes, and then at the end, concatenate the list into one dataframe.

On the other hand, my kernel is really fast. It takes 25 minutes to import data, generate features, and train the model. When submitting, predictions are generated at a rate of about 50 timestamps per second, not including the overhead of the submission mechanism. Generating predictions for 2 weeks of data takes around 10 minutes, on top of the the 25 minutes to train the model.

Speed wasn't a limiting factor in my submission because I used Numba to write feature generation code and my learner was simple. My main goal was to have a codebase that allowed me to test feature ideas quickly and easily. The fast submission kernel was a side effect.
