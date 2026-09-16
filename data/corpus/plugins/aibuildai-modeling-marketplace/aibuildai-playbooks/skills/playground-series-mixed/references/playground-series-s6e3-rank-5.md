# 5th Place Solution! 149 Models -> 6 Meta Models -> 3 Blends

Competition: playground-series-s6e3
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s6e3/writeups/5th-place-solution-149-models-6-meta-models

# 1st Anniversary in Kaggle Competitions
I’m excited to share my solution write-up for this binary classification competition—my approach helped me reach 5th place, and it’s been a tough yet rewarding learning journey from start to finish. March 2026 marks exactly one year in Kaggle Competitions, and as someone from a non-scientific/mathematical discipline, grasping the concepts behind different architectures and algorithms has been a steep curve. I genuinely owe a lot to the great teachers and participants who openly share their knowledge and solutions with the public. Appreciation goes to @cdeotte @tilii7 @siukeitin @optimistix @ravi20076 @mahoganybuttstrings @masayakawamata @mikhailnaumov @yekenot @include4eto, and last but not least @mpwolke for the thoughtful preliminary insights at the start of every competition. 

I appreciate you and everyone who shared valuable knowledge with the community.

## Diversity: 199 Base Models
| Model | Count | Variant | Key Parameter | Best CV |
| --- | --- |
| LightGBM | 40 | gbdt, goss | data_sample_strategy, alpha, lambda | 0.91932 | 
| XGBoost | 30 | gbdt, lossguide, gradient_based | colsample_bytree, alpha, lambda | 0.91914 | 
| CatBoost | 12 | plain, ordered | bagging_temperature, random_strength | 0.91924 | 
| xLearn | 9 | linear, fm, ffm | lr, alpha | 0.91851 |
| TabNet | 26 | - | lr | 0.91703 |
| PyTabkit: NN | 12 | RealMLP, TabM, MLP_PLR_D, MLP_RTDL_D, Resnet_RTDL_D | - | 0.91887 |
| PyTabkit: Trees | 6 | Catboost_TD, XGB_TD, LGBM_TD | - | 0.91925 |
| PyTabkit: RealMLP | 3 | - | embedding_size, ls_eps, n_ens | 0.91941 🏆 |
| KerasNN: FeatureSpace | 3 | MLP, TabTransformer | - | 0.91532 |
| LogisticRegressionCV | 6 | - | C | 0.91667 |
| YDF | 12 | global, local | growing_strategy, categorical_algorithm | 0.91817 |
| H2OAutoML | 2 | - | num_models | 0.91791 | 
| HistGradientBoosting | 4 | - | max_iter, lr, max_features | 0.91770 | 
| BARTZ | 2 | - | ntree, k | 0.918358 | 

## Hillclimb, Ridge, BayesianRidge, Optimized Blend
| Model | Score | LB | PB |
| --- | --- |
| BayesianRidge | 0.91983 | 0.91733 | 0.91846 🏆 | 
| Ridge | 0.91986 | 0.91730 | 0.91845 | 
| Hillclimb  | 0.91989 | 0.91730 | 0.91840 | 
| Optimized Blend (3 above) | 0.91989 | 0.91732 | 0.91842 | 

## Sumission
Unfortunately, I ended up submitting `Ridge` and `Optimized Blend` rather than selecting the single most promising personal-score configuration; the submission ultimately landed at `0.91845` with `Ridge`, while `BayesianRidge` had the best result at `0.91846`.
