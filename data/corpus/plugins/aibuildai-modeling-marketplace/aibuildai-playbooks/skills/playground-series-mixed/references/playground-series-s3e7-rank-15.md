# 15th place solution

Competition: playground-series-s3e7
Rank: #15
Source: https://www.kaggle.com/c/playground-series-s3e7/discussion/390977

Congratulations to all the winners ! 👍

My solution:

- The code:
[**PS-3-7 EDA CV weights optimizer**](https://www.kaggle.com/code/martynovandrey/ps-3-7-eda-cv-weights-optimizer)

- CV:
30 folds stratified. The CV function was written with full classification report and 5 plots to compare solutions.

- Ensembling
I think that was *the main feature of the solution*, ensembling was made in **each fold** with weight optimized, using what i called it "**Averager**" - class with fit and predict methods to find optimal weights for N solution given. I'm to use it in the future. See the code in the notebook.

- Models
3 regressors + 3 classifiers (XGB, CAT, LGB) + Random Forest

- FE
None

In the notebook I collected CV and LB progress data with comparision table and plot, it helped a lot to choose final submission.



*Many thanks to everyone who read my humble work!*
