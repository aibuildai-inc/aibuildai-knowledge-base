# Public 71st Solution Writeup (Private 39th)

Competition: predict-energy-behavior-of-prosumers
Rank: #39
Source: https://www.kaggle.com/c/predict-energy-behavior-of-prosumers/discussion/472598

# Public 71st Solution Writeup

Firstly, thanks Kaggle and competition hosts for hosting this exciting competition. It's the exact first time I try to work as a team player, an amazing experience! Hence, thanks my teammates, @patrick0302 and @chunweishen for all the supports, discussion and hardworking. Also, thanks all kagglers for sharing your valuable insights, special thanks to @vitalykudelya for providing high-quality notebook, which our final solutions are highly based on.

## 1. Overview
* Split data into 3 target types for modeling, including `prod`, `cons_c` and `cons_b`.
	* `prod`: `is_consumption == 0`
	* `cons_c`: `is_consumption == 1 and is_business == 0`
	* `cons_b`: `is_consumption == 1 and is_business == 1`
* Transform target as shared in the public notebooks and add `target / eic_count`.
	* Combined with target type splitting, we train different models with `model_type` summarized in the following table, 
[[Screenshot-2024-02-01-at-3-14-00-PM.png]](https://postimg.cc/Hjkvw9Hy)
> "raw" stands for `target`, "diff" for `target / target_lag2d`, "dcap" for `target / installed_capacity` and "deic" for `target / eic_count`

* Train both xgb and lgbm versions of each `model_type`.
* Select custom feature set for each model, with the number of features in the range of 27 ~ 55.
* Use 3-fold time series CV with validation interval `202209 ~ 202211`, `202212 ~ 202302`, and  `202303 ~ 202305`.
* Ensemble models from the model pool and retrain them in final evaluation phase (elaborated later).

## 2. Base Model Training
As mentioned in overview, we train each model with custom selected features. Also, we train each model with different samples.
### *Data Sampling*
We sample the data for model training with the following logics,
* `p_x` models use all production data, where `x` can be any target transformation in the table above.
* `c_raw` and `c_diff` use all consumption data and models are shared for `cons_c` and `cons_b`.
* `cc_dcap` and `cc_deic` use only `cons_c` data.
* `cb_dcap` and `cb_deic` use all consumption data.

Because we observe `cons_b` data is a drag on `cons_c` performance when using the `dcap` and `deic` targets, we train `cc_dcap` and `cc_deic` with data only from the same target type.
### *Feature Engineering and Selection*
Most of the features we use are from public notebooks. We fail to create other features which can boost CV scores except for the following one,
* `target_lag?d_xpc`: The plain lagged target features **crossing production and  consumption**. That is, we consider production/consumption lagged targets as the features for consumption/production models.

We believe there exist more powerful features, waiting for more to be shared! 
Then, we do some experiments to verify that most of the features on public notebooks can be dropped directly. After all,  **less is more** sometime. Hence, we use a 2-stage method to select feature set for each model.
1. Set-wise forward selection
		We add features set-by-set (*e.g.,* local mean of the forecast weather) and keep them if the CV score boosts for all the 3 folds. For most models, we end up with a features set with 70 ~ 80 features.
2. Element-wise backward elimination
		Then, we eliminate features element-by-element (*e.g.,* `target_lag4d`, `target_lag5d`) simply based on feature importances. Again, features are dropped if removing these features doesn't worsen 3-fold CV scores (fold-by-fold, not average). Sometimes, we even observe CV boost in this stage. Finally, we get selected feature sets with 27 (`p_diff_lgb`) ~ 55 features.
### *Model Training and Hyperparameters*
We just manually tune hyperparameters (mainly `n_estimators` without early stopping) in the very beginning, and fix them. About two weeks before the submission deadline, we revisit it with the final selected feature sets but can't observe CV/LB sync on hyperparameter tuning. Hence, we rollback to the fixed one at start.

## 3. Retraining and Ensemble Strategies
### *Model Retraining*
To compensate for the 8-gap data and align CV setup with the final evaluation scenario, we retrain each model using one of the following methods,
1.  **Retrain at start** : Retrain models **only once** after the first `currently_scored` has been triggered. That is, the gap 8-month data is completely collected.
2. **Monthly retrain**: Retrain models on the first day of each month.

[[Screenshot-2024-02-01-at-3-14-16-PM.png]](https://postimg.cc/Dmm3cXK8)
To determine which retraining strategy to choose, we compare CV scores of two CV setups, 3-fold (3 months for each) mimicking **retrain at start** and 9-fold (1 month for each) mimicking **monthly retrain**. Then we pick the one performs better for each single model.
### *Model Ensemble*
We ensemble models based on the following logics,
* For `?_raw` and `?_diff` (where `?` can be `p`, `cc` or `cb`), we average them with the equal weight (*e.g.,* `p_raw_xgb * 0.5 + p_diff_xgb * 0.5`, which is named as `p_ens_xgb`).
* For each `model_type`, we average xgb and lgbm with the equal weight (*e.g.,* `p_dcap_xgb * 0.5 + p_dcap_lgb * 0.5`, which is named as `p_dcap_xl`).
* Each model is trained with 3 random seeds.

Then, the prediction strategy is described as follows,
1. For `prod`, `dcap` is selected as a base prediction. If `installed_capacity` is missing, we use `raw * 0.5 + diff * 0.5` as prediction. If `target_lag2d` is again missing, we use only `raw`.
2. For `cc`,  `d_cap * 0.5 + d_eic * 0.5` is used as base for segments seen in the training set. Otherwise, the same prediction hierarchy used in `prod` is applied.
3. For `cb`, `dcap` or `deic` have decent CV boosts but fail to sync with LB. Hence, one submission uses `raw * 0.5 + diff * 0.5`, and the other one takes `deic` into consideration `(raw + diff + deic) / 3`. 

## 4. Some More Studies
### *Cascading or Rolling Training Set*
We observe that some models have slightly better performance when trained with the latest 1-year data, instead of all available data till `202109`. My question is which training set would you choose if both have similar performance?
### *Retrain at Start or Monthly Retrain*
As described above, we choose retraining strategies by comparing performance of 2 CV setups, 3-fold 3-month and 9-fold 1-month. An example of the consumption raw model `c_raw` is illustrated as follows, 

[[Screenshot-2024-02-01-at-2-55-55-PM.png]](https://postimg.cc/WhWnwdTb)

In this case, **monthly retrain** performs better than **retrain at start**. As the month gap increases (from 0 to 2), the performance boost of monthly retrain becomes greater.
### *Unseen Segment Test*
Considering there'll be new units in private LB, we test which target form is more suitable for unseen segments for each target type. As for implementation, we mask one unit per county in training set after data splitting is done. That is, we can validate on segments not appearing in the training set.
For `prod` , we choose base prediction `dcap`. For `cc`, `raw * 0.5 + diff * 0.5` performs better than base `dcap * 0.5 + deic * 0.5`. And, we use base `raw * 0.5 + diff * 0.5` for unseen  `cb` segments.

## 5. What Remains
1. We spend about 3 days to build a single NN model achieving LB 74 but have no time to work on this. It's based on WaveNet and co-train 24-hour production and consumption together, which can be seen as a multi-horizon forecasting problem.
2. We try to select best models for prediction on the fly based on the historical error patterns, but observe only little improvement compared with simple blending.

## 6. What Didn't Work for Us
1. Generate more powerful features.
2. Tune hyperparameters.
3. Increase the number of random seeds more than 3 for each model.
4. Use `dcap` or `deic` for `cb` (CV boosts a lot, but LB is unstable).
