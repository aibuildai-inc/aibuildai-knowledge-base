# 3rd place solution

Competition: tabular-playground-series-nov-2022
Rank: #3
Source: https://www.kaggle.com/c/tabular-playground-series-nov-2022/discussion/370126

Thanks for kaggle competitions.I have learned a lot from these great Kagglers.
There is no complicated tricks for my solution, and the shakeup shocked me.
### Team name
@guoyaobit , solo

### Team place
172nd public LB and 3rd private LB

### The brief description of the solution
1. Drop features contain values out of range of (0,1).
2. Drop features using spearman corrlation with ground truth. 0.42 is the threadhold.
3. Isotonic calibration.
4. Train the model using AutoGluon.

### Several ideas that doesn't work
1. Drop features using ece.
2. FLAML

### Lessons learned from the competition
1. Trust your local CV.
2. Calibration and model stacking is useful tricks.
