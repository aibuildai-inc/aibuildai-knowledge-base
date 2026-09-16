# 4th private (7th public) place (part II)

Competition: data-science-bowl-2019
Rank: #4
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127312

There is the second part of our solution.

RNNs are trained by @sergeifironov. We train all models in this submission on the train data only (17690 observations), nevertheless TFiDF has been fitted with the test set information.

The 'simple' scheme is following: 


Additional info:
- All models are trained as a regression.
- Tree based models perform worse with direct usage of the test data.
- We round final predictions in this submission with original @artgor coefficients. They magically show better results.

What doesn`t work for us (tree based models):
- Counters, matrix factorisations (ALS).
- TFiDF on only assessments / only titles / only assessments scores sequences. 
- Adding additional components to the token (world, number of successful attempts, etc).
