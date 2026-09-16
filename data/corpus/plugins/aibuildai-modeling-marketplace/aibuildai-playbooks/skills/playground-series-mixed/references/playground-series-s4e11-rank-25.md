# 25th Place - GBDT plus NN - Trust CV

Competition: playground-series-s4e11
Rank: #25
Source: https://www.kaggle.com/c/playground-series-s4e11/discussion/549194

Hi everyone. Thank you for the lively discussions and generous sharing. I enjoyed participating in this competition with everyone and I plan to participate in December's competition [here][1]! 🎉

This is my second playground competition. In my previous playground competition, I built 100+ models and used hill climbing [here][7] (discussion writeup [here][2]). However from my first experience, I learned that more simple feature engineering and simple solutions work well in playground competitions, so in this competition my final solution is an ensemble of only 3 models (with simple feature engineering). Namely CATboost plus XGBoost plus NN. This is a powerful diverse ensemble combination! 🔥

# Use AUC as proxy for ACC (accuracy) metric
The metric in this competition was ACC. This metric is not smooth and has lots of random variance when trying to use it to optimize models and decisions. Therefore I used the more reliable metric AUC to locally find the best CV score. Then I chose my best CV AUC ensemble/model as my final submission.

# 33% CatBoost - CV ACC=0.9401 (AUC=0.9751), LB Public=0.9433, Private=0.9405
My CatBoost model was based on top scoring single model CatBoost public notebook [here][5] by @abdmental01

# 33% XGBoost - CV ACC=0.9400 (AUC=0.9755), LB Public=0.9439, Private=0.9400
My XGBoost model was based on top scoring single model XGBoost public notebook [here][6] by @adyiemaz

# 33% NN (MLP) - CV ACC=0.9399 (AUC=0.9756), LB Public=0.9427, Private=0.9413
My NN by itself would achieve 68th rank on private LB! It's a strong model! I encoded all columns the same way as my public notebook [here][3]. Namely I converted every column into categorical strings (and transformed rare values to value = "RARE", and nan to value = "NAN"). Then I used my NN code from September's playground competition [here][4]. All hyperparameters, learning schedule, and architecture were the same.

# Ensemble - CV ACC=0.9406 (AUC=0.9762), Public LB=0.9438, Private LB=0.9415
I tried adding some other models, but the three above achieved the best ensemble CV. So my final ensemble is only the three models described above and achieved 25th place in Kaggle's "Exploring Mental Health Data" competition! 💪

[1]: https://www.kaggle.com/competitions/playground-series-s4e12
[2]: https://www.kaggle.com/competitions/playground-series-s4e9/discussion/537202
[3]: https://www.kaggle.com/code/cdeotte/fast-gpu-hill-climbing-starter-cv-0-94-lb-0-94
[4]: https://www.kaggle.com/code/cdeotte/nn-starter-lb-72300-cv-72800
[5]: https://www.kaggle.com/code/abdmental01/emh-multi-model/notebook?scriptVersionId=205479699
[6]: https://www.kaggle.com/code/adyiemaz/this-code-fixed-my-depression/notebook?scriptVersionId=205824442
[7]: https://www.kaggle.com/code/cdeotte/fast-gpu-hill-climbing-starter-cv-0-94-lb-0-94
