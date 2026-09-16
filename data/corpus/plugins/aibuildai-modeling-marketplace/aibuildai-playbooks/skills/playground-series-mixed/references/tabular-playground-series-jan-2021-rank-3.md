# 3rd place solution

Competition: tabular-playground-series-jan-2021
Rank: #3
Source: https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216087

Congrats everyone and especially @springmanndaniel for his great winning. I was really happy at the time that he let the rest of us know that NN models can also work on this dataset. When you know that something is doable, then it's really more doable. (even though most of us failed :)) After that time I worked almost only on NN models and managed to get 0.700X cv/lb with a single model, which ended up improving my final stacking quite well.

My solution is based on stacking diverse models. And I had 4 main models:
**1 -** **regular lgbm modeling** with plain features.
**2 -** **2-stage lgbm modeling**. For this part, I turned the real target column into a binary column with a threshold and then run a classification model as 1st-stage modeling. Let's say my threshold was 8, then targets having a lower value than 8 were classified as 0 and the rest as 1. Then I got predictions for the probability of being in class 1. At my 2nd-stage modeling, again I had 2 separate models. First, I run my models with training data having the target higher than the threshold and got predictions for both validation and test. Then did the same with the training data having the target values lower than the threshold. In the end, I calculated final predictions as:

`final_prediction = probability_of_being_0 * (predictions_from_training_0) + probability_of_being_1 * (predictions_from_training_1)`

Just by changing the threshold value, I managed to get many diverse models easily, and accumulated their oof and test set predictions.

**3 -** **regular lgbm with the augmented training dataset**. I tried to apply DAE on all data set. Even though my dae was far from being successful as 1st place's dae solution, using the output of my DAE for augmenting training data during the CV ended up improving final stacking well enough.

**4 -** **MLP with Embedding layers after crafting categorical features from the original continous columns**. By adding embedding layers along with the original inputs I managed to get 0.700X CV and LB with an NN model.

As the final step, I applied stacking with a linear regression on top of OOF predictions of my each saved model. It was much better than manual blending. Also as a very final squeezing step, I've created non-linear interaction features between oof features during stacking and managed to improve both the CV and LB in 4th-5th decimals more.

My kernels are coming on the way. Hope you can find some useful parts from them.

NN kernel: https://www.kaggle.com/fatihozturk/nn-with-embedding-part-of-3rd-place-solution?scriptVersionId=53252723

LGBM models and Stacking part: https://www.kaggle.com/fatihozturk/models-stacking-3rd-place-solution?scriptVersionId=53266247
