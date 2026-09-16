# 7th place solution - lots of trial-and-error

Competition: tabular-playground-series-sep-2021
Rank: #7
Source: https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/277256

Hi all, I wanted to say thank you to all of you who have shared their expertise in this competition. I wouldn't have done this without all of your amazing work!

At first, my models only produced mediocre results (`0.5xx` - `0.79x`) because I'm using `SimpleImputer` without capturing the usefulness of NaN values. After learning that the NaN values are actually useful and capturing it before imputation, the results got better even with just the base models.

0. Feature engineering
	- Thanks to @realtimshady for FE ideas on this [notebook](https://www.kaggle.com/realtimshady/single-simple-lightgbm)
	- Thanks to @dwin183287, @lucamassaron for the EDA and notebook about missing values
	- Thanks to @craigmthomas, @mohammadkashifunique, @davidcoxon for answering my questions on the discussions. 


1. LightGBM - Refer to this [notebook](https://www.kaggle.com/stevenrferrer/tps-september-2021-lightgbm-baseline)
	- I forgot whose hyperparam I'm using, please let me know in the comments if you know who so I can properly mention here.
	- Most likely using @mlanhenke hyperparams here

2. CatBoost - Refer to this [notebook](https://www.kaggle.com/stevenrferrer/tps-september-2021-catboost-baseline)
	- I'm using @mlanhenke hyperparam in this [notebook](https://www.kaggle.com/mlanhenke/tps-09-single-catboostclassifier).
	- Thanks to @shenurisumanasekara for the [catboost step-by-step notebook](https://www.kaggle.com/shenurisumanasekara/catboost-step-by-step-improved).

3. HistGradientBoost - Refer to this [notebook](https://www.kaggle.com/stevenrferrer/tps-september-2021-histgradientboosting-baseline)
	- I'm using @tunguz hyperparam in this [notebook](https://www.kaggle.com/tunguz/tps-09-21-histgradientboosting-with-optuna).

4. XGBoost - Refer to this [notebook](https://www.kaggle.com/stevenrferrer/tps-september-2021-xgboost-baseline)
	- Also forgot whose hyperparam I'm using, please do let me know in the comments if you know who so I can properly mention here.
	- Most likely using @mlanhenke hyperparams here

5. VotingClassifier LightGBM/CatBoost/HistGradientBoost - Refer to this [notebook](https://www.kaggle.com/stevenrferrer/tps-september-2021-votingclassifier-baseline)
	- I'm using the same hyperparams as from the above, just combined them in the voting classifier.
	- The reason I didn't include XGBoost in the voting is because takes long to train (9 hours and did not finish) plus, I ran out of GPU. Turns out it was good because I only achieved good score in the voting using CPU.
	- Thanks to @dmitryuarov for the [voting notebok](https://www.kaggle.com/dmitryuarov/tps-voting-xgb-cb-lgbm).

6. Blend LightGBM/HistGradientBoost/CatBooost/XGBoost/Voting - Refer to this [notebook](https://www.kaggle.com/stevenrferrer/tps-september-2021-blend-lgb-xgb-cb-hgb-voting)
	- For blending, I combined all of the above models to produce the final submission.
	- Thanks to @ankitkalauni for the visualisations on model correlations.

I've compiled all the prediction data in this [dataset](https://www.kaggle.com/stevenrferrer/tps-september-2021-preds).

Again, thank you very much to you all for sharing your knowledge to beginners like me. I hope we will continue to learn and grow in this platform!

P.S.: Please, please, please 🙏 do let me know if I forgot to mention you or someone else in this post, I'd be happy to update this to give proper credit. Thanks!
