# 7th place solution 🥇 : The very simple method !

Competition: march-machine-learning-mania-2025
Rank: #7
Source: https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572540

Before starting to join this competition, I could not imagine that I will get such the greatest score where I participated in the past competitions. I am awesome to this result.

In my strategy, I only focused on the notebook which kaggle  tier person created and the solutions which were  successful in past NCAA kaggle competitions due to the period of competition limited.

 My winning solution is below URL in 3rd version!  (The title is 7th place🥇notebook_NCAA2025)
https://www.kaggle.com/code/takumuwakamatsu/7th-place-notebook-ncaa2025

To create the notebook of the result ,  I only did 2 steps.  

① Baseline notebook was  below URL which @raddar developed (The latest version before the deadline of  submitting) 
https://www.kaggle.com/code/raddar/vilnius-ncaa

This notebook  mentioned in terms of feature engineering is below 5 points.

- Easy difficulty features  (merge the Seed Information dataset)
- Medium difficulty features (We combine the averages of T1 and T2 to predict the outcome of the match between T1 and T2)
- Hard difficulty features (Introducing EOL ratio)
- Hardest difficulty features (Caliculating  the strength using GLM method)
- After creating in 4step, I choose the 25 features in these steps.

Finally, XGBoost parameter is below  in notebook and done 5 cross validation of k-fold method. 
Then I predicted each year of same parameter. 
 
```python
param = {}
param["objective"] = "reg:squarederror"
param["booster"] = "gbtree"
param["eta"] = 0.01
param["subsample"] = 0.6
param["colsample_bynode"] = 0.8
param["num_parallel_tree"] = 2
param["min_child_weight"] = 4
param["max_depth"] = 4
param["tree_method"] = "hist"
param['grow_policy'] = 'lossguide'
param["max_bin"] = 32
num_rounds = 700
```
Finally,  I correct the result of XGBoost using spline interpolation.  

② I added to the below URL notebook which @kaito510 created after final part of ① Baseline notebook.
https://www.kaggle.com/code/kaito510/updated-goto-conversion-winning-solution

 This solution is below and in order to calculate this I added the dataset of the '538data'(https://www.kaggle.com/datasets/kaito510/538data) . 

- Submission with Optimal Strategy Below is a mathematical proof that the optimal strategy to win a medal under Brier Score is when we assume a team with 33.3% chance of winning a match to win that match.

- The expected return when we risk on a given game can be expressed as:

```python
f(p) = p(1 - p)^2 where p is the probability of success and (1-p)^2 is essentially the reward for the risk taken if the risk succeeds

This implies f'(p) and f''(p) can be expressed as:

f'(p) = -2p + 2p^2 + (1-p)^2

f''(p) = -4 + 6p

argmax_p f(p) = 1/3 with tedious mathematical working omitted.
```
- Thus, expected reward is maximised when we assume a team with 1/3 chance of winning a match to win that match.


Finally, I appreciate that @raddar and @kaito510 developed such a great notebook! 
Without you, I could not have gotten such an awesome result in this competition!!!! Thank you very much.
