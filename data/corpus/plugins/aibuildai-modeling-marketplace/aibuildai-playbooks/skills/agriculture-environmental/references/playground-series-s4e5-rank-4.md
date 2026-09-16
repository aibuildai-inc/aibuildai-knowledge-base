# 4th Place Solution: Hill climbing through the noise

Competition: playground-series-s4e5
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s4e5/discussion/509044

Hi everyone,

Thank you for another exciting Playground Series competition! Before presenting my solution, I'd like to suggest that a two-week duration might have been more appropriate for this competition. The final stretch felt like trying to squeeze out the last few drops of optimization. Now, onto my solution:

The crucial strategy for this competition was to blend as many models as possible. This approach was necessary because we were essentially predicting noise.





## hillclimbers
I experimented with various model blending methods, including Lasso and Ridge regression, but ultimately achieved the best results with hill climbing. I used my Python package, [hillclimbers](https://github.com/Matt-OP/hillclimbers), to determine the optimal weights.

## Feature Engineering

I utilized 3 sets of features to train my models, denoted by FE2 and FE3 seen in the bar chart of selected weights. These sets included the following features in different variations:
Statistical features: mean, median, mode, max, min, standard deviation, skewness, kurtosis, quantiles, etc. 
I also included features for the counts of each unique value in a row which was originally proposed in the [[AutoML Grand Prix] 1st place solution write-up](https://www.kaggle.com/competitions/playground-series-s4e5/discussion/500700).

## Hyperparameter turning 

I employed various sets of hyperparameters, some manually tuned and others optimized using Optuna. For example, my DecisionTreeRegressor ensemble utilized multiple hyperparameter sets, and the resulting predictions were then blended together.

[Link](https://www.kaggle.com/code/mattop/ps-s4-e5-4th-place-solution-with-hillclimbers/notebook) to the notebook.
