# 6th place solution

Competition: playground-series-s3e15
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s3e15/discussion/413839

Hello everyone! In this solution I used model blending with 5 LGBM models based on different imputation techniques and parameters and concatenating their train set with original set (when the data is splitted on valid and train set, in other words inside CV loop) , and with a little feature engineering/imputation techniques also. Also note, there will be a lot of techniques which are already discussed in [my discussion](https://www.kaggle.com/competitions/playground-series-s3e15/discussion/411353). I'll point all moments which are already discussed in my discussion to not waste time for explanation

# Feature Engineering
In feature engineering I've only applied logarithm to `D_e`, `D_g` and `chf_exp` in order to skew their distribution to right a bit, since they had left-skewed distribution, as was discussed in my discussion. And that's whole feature engineering

# Feature Imputation
I was blending 5 LGBM models using different feature imputation techniques, so, about them I will tell about it in **Blending** section. But there are also some general feature imputation techniques which are applied to all models in the blending.

- As was discussed in my disscusion, I've imputed `geometry` values using `author` values
- As @aadinesarkar pointed in my discussion, I've imputed `geometry` values `D_e` and `D_h` values.
    - If they are equal, then `geometry` is `tube`
    - If `D_e` is equal to 15.0 or `D_h` is equal to 120.0, then `geometry` is plate
    - As for `annulus` there are a few values of `D_e`, the imputation can be inaccurate, so I've omitted it
- Similarly as we imputed `geometry` values, we impute `D_e` and `D_h` in reverse

# Blending

### Notes
- We will use general imputation for all models.
- We are tuning only these parameters - `learning rate`, `num_leaves`, `max_depth`, `colsample_bytree`, `subsample`, `min_child_samples`.
- We are concatenating train set with original set (valid set must be from synthetic dataset)
- When we say "without imputing <features>" it doesn't mean that imputer don't fit on these features, it means that we just leave these features as they are

- `lgbm_without_imputation` this model is just a LGBM without any feature imputation (except the general imputation) and optimized by optuna

- `lgbm_lgbmimputer1` this model is trained on imputed data by `LGBMImputer(n_iters=200)`, but without imputing `D_h`, `length`, `D_e`(we are still using general imputation, as for all models). Also we are using `lgbm_without_imputation` tuned parameters changing `max_depth` and `min_child_samples` parameters

- `lgbm_lgbmimputer2` this model is trained on imputed data by `LGBMImputer(n_iters=200)`, but without imputing `mass_flux`, `D_e`, `length`, `D_h`. Also we are using `lgbm_without_imputation` tuned parameters changing `max_depth` and `min_child_samples` parameters

- `lgbm_lgbmimputer3` this model is trained on imputed data by `LGBMImputer(n_iters=200)`, but without imputing `mass_flux`, `D_e`, `D_h`. Also we are using `lgbm_without_imputation` tuned parameters without changing

- `lgbm_lgbmimputer4` this model is trained on imputed data by `IterativeImputer(LinearRegression(), max_iter=20, initial_strategy='constant')`, but without imputing `mass_flux`, `pressure`. Also we are using `lgbm_without_imputation` tuned parameters changing only `max_depth` parameter

- And concatenating it using LinearRegression without interecept

That's it! Also, I wanted to tune parameters by minimizing loss of the other model, which is trying to predict `chf_exp` feature, since `x_e_out` is the most important feature to predict `chf_exp`, but the score didn't change, so I didn't do things complicated.

*The implementation of all these things is in [my solution notebook](https://www.kaggle.com/code/demko1/6th-place-solution-s3e15)*

Good luck in the next competitions!
