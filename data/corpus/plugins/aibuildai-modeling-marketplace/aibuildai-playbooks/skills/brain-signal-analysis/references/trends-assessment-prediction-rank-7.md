# 7th Place Solution

Competition: trends-assessment-prediction
Rank: #7
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162787

I have a 3D Conv Net originated from @shentao 's great baseline. I want to thank him for that. I made some small architectural changes on resnet10 and started to boost from median predictions. It converged much faster that way, so that I could make the augmentation less severe. It achieved 0.1695 CV without using any other features.

I have used its OOF in my models. Then I have almost equally performing 4 models on top of these OOF predictions and loading + fnc features:
- LGB: Before feeding the features, I oversample loading features 50x times and then I use very low feature sampling fraction like 0.015.
- MLP: Different dropout rates for 3D CNN OOF, fnc and loading features.
- SVM: Same as https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging
- Ridge: An overfitting ridge was contributing the best compared to a robust BayesianRidge or HuberRegressor.

I made the CV-LB gap closer by using Adversarial Validation predictions as sample weights. At the end, I had 0.1562 CV and 0.1566 LB. I was expecting shaking up thinking that there are even more site2 in private LB (Kaggle usually does that way) but I guess it wasn't the case and this trick lost its importance.
