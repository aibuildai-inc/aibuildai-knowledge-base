# 3rd Place Solution

Competition: equity-post-HCT-survival-predictions
Rank: #3
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566574

[Full code](https://www.kaggle.com/code/churkinnikita/3rd-place-solution)

[Best "solo" model (gold medal top-11 Private LB — 0.69694, full training from scratch)](https://www.kaggle.com/code/ukrainskiydv/3rd-place-solution-nn-solo-0-69694-private-lb)

###### First of all, we are grateful to the competition organizers for the possibility to contribute to HCT therapy!

Our solution has the following steps.

1. Validation is the most important one. We made a 4-fold CV (as the public test part is 25%) and evaluated the score by 20–100 random split seeds (depends on the computational complexity). During all the competition, we had excellent CV–LB correlation when measuring average fold scores (4 x NSeeds). The public "fold" was approximately at the 75-quantile of the score histogram in our experiments. We emulated and tracked both "public" and "private" scores in this way and created a second submission as more private-oriented. It had a lower LB score but significantly better emulated (and real) private score.

2. We made a uniform target in the range [0, 1] for uncensored (`efs = 1`) observations and in the range [1.345, 1.355] for censored (`efs = 0`) cases. The target was calculated within train folds and valid folds separately for every race group.

3. Then we divided the task into three parts: separate regressions within uncensored and censored data weighted on the probability of belonging to the classes `efs = 1` and `efs = 0`.

4. Regression within zeros is the most insignificant part of the composition. For NN models, we just collapse all the `efs = 0` observations to the constant 1.35 (that was tuned to show maximal performance in terms of the concordance index).

5. The target range [0, 1] for the regression task allowed us to train the models via binary cross-entropy loss, which is much more profitable in comparison with mean squared error.

6. Thus, the prediction has the form

and we obtain an overall scatter plot of type

7. Our model zoo consists of CatBoost, LightGBM, XGBoost, MLP with ODST, and TabM models (separate exemplars of each for regression and classification). NNs were the best at classification, and GBMs are the champions for regression. Ranking models have weaker performance in the competition.

8. The data is noisy, and the classic methods of noise reduction worked well. We averaged the models with fixed hyperparameters / architectures over random initialization seeds and eliminated the observations with giant regression errors from the training. But in the case of classification, outlier denoising leads to overfitting (ROC AUC becomes better, but LogLoss worsens at the same time).

9. On the very final stage, we blended the regressors by convex combination and stacked the classifiers by logistic model. Stacking by logreg shows spectacular performance on the private part of the test data that we saw via out-of-fold imitation of the public / private split. All blending weights were optimized simultaneously to maximize the concordance index on 20 OOF predictions (20 random split seeds) using Bayesian optimization

The resulting ensemble is

Overall performance of the components:
| Model | CV ("Public" estimate — folds x seeds) | Public LB | Private LB |
| --- | --- | --- | --- |
| CatBoost | 0.68586 | 0.69424 | 0.69577 |
| LightGBM | 0.68530 | 0.69318 | 0.69572 |
| XGBoost | 0.68616 | 0.69252 | 0.69493 |
| NN | 0.68554 | 0.69540 | 0.69694 |
| Blend GBMs | 0.68932 | 0.69608 | 0.69794 |
| Blend All + LogReg | 0.69139 | 0.69692 | 0.69937 |

To sum up, the most important trick in this competition was to divide the task into separate regression and classification parts. You could see this trick, for example, by looking at the scatter plot within the first straightforward modeling attempts. The picture below shows training using all data (left), training using `efs = 1` data only (center), and training using `efs = 0` data only (right)


My teammates disclose the details of the solution in the comment section.
