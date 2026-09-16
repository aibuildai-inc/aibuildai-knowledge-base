# 4th place: Factorized modelling approach

Competition: equity-post-HCT-survival-predictions
Rank: #4
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566528

Full model* code: https://www.kaggle.com/code/herrahuu/4th-place-solution
Private score: 0.69936, Public score: 0.69724, total runtime: 4h
*this version also includes LightGBM model not mentioned below, but impact is minimal in ensemble

For comparison solo model scores (without bagging):
CatBoost: 0.69784, 0.69500, 30min
XGBoost: 0.69765, 0.69538, 5min
TabM: 0.69636, 0.69383, 6min
LightGBM: 0.69793, 0.69516, 5min

~~~~

It was fun to make a small comeback to Kaggle after a long break. Especially given this nice "old school" competition, not too big & tabular dataset. Below is a short summary of my solution.

# Introduction
The goal of this competition was to rank patients based on their risk scores for allogeneic HCT transplantation events. As a first step it's useful to think what a perfect solution would look like. In this case that would be:
1.  all patients with no event (efs=0) should be at the bottom of the list (no order defined between them)
2. for efs = 1, patients are ordered by the event times

Clearly this ranking problem is non-differentiable, especially as the 1) introduces step function like behavior for the desired ranks. One common way to handle this type of problems is to introduce some sort of soft/smoothed version of the original task. However, in my experience, such approaches tend to make things unstable during training, or just don't really match to the original problem that well if smoothed too much. 

So instead, my main idea for the competition was to follow the steps 1-2) directly and divide the problem into two parts: 
1.  predict the probability of the event
2.  predict the expected ranking position among patients with events

Finally, given these two predictions, we can calculate the expected ranking position as a risk score.

# Model formulation:
```python
risk_score = P(event = 0)*(s0_group/2) + P(event = 1)*(s0_group + (1-s0_group)*E[rank% | event = 1])
, where s0_group = sum_{race_group = group} P(event = 0) / N_group 

```
## P(event):
- binary classification for efs events
- censored data points are treated as a partial observations
- weights for those data points are calculated as cumulative densities given by Kaplan-Meier estimator and scaled to (0,1) range.

## E[rank% | event = 1]:
- conditional regression model to predict time based rankings for data points where event did happen
- the model is trained using only data points which have efs = 1
- target is rank% = rank(-time)/N. 
- so, target values are between 0 and 1, 0 = longest time and 1 = shortest
- with all algorithms, I'm using target transformation of the form: inverse_normal_cdf(rank%) "z-score", and then predictions are transformed back to percentages by normal_cdf(pred)

# Algorithms:
For both of these tasks, three different models are implemented using the following packages/algorithms:
- TabM
- CatBoost
- XGBoost

So, in total six models are trained. First their predictions are merged by just calculating weighted sums (separately for both time and event using different weights), and then finally risk scores are calculated by the the formula given above.
