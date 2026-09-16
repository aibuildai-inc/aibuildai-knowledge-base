# 12th Place Solution: Stacked Ensemble with Ordered Target Encoding

Competition: playground-series-s6e4
Rank: #12
Source: https://www.kaggle.com/c/playground-series-s6e4/writeups/12th-place-solution-stacked-ensemble-with-ordered

# 12th Place Solution: Stacked Ensemble with Ordered Target Encoding

Thanks to Kaggle for the Playground Series. Final placement: 12 of 457, private LB 0.98082.

Special thanks to [@cdeotte](https://www.kaggle.com/cdeotte) for the [PS6E3 1st place write up](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm). I used it as the starting point for my whole setup on this competition: pipeline structure, validation discipline, stacking strategy, and the bar for what counts as enough diversity. The full thanks list is at the bottom.

## Summary

My solution is a 41 model stacked ensemble. The base models span six algorithm families. A LightGBM stacker combines their out of fold probabilities. A log bias correction tunes the final argmax thresholds for the imbalanced classes.

The single biggest lever was Ordered Target Encoding with 4x shuffle augmentation. The second biggest lever was adding deliberately weak diversity models that the stacker could route around the strong models on hard samples.

Raw data flows into two parallel feature pipelines. Each pipeline feeds a strong booster group and a diversity group. All 41 base models hand their OOF probabilities to a LightGBM stacker. The stacker output passes through a log bias correction before becoming the final submission.

## Final Picks

| Pick | File | Public LB | Private LB |
|------|------|-----------|------------|
| 1 | sub_v15_lgb_stack_log_bias.csv | 0.98081 | **0.98082** |
| 2 | sub_v11_lgb_stack_30m.csv | 0.98072 | 0.98046 |

Pick 1 was the v15 stacker over 41 base models. Pick 2 was a more conservative v11 stacker over 30 base models with the tightest CV vs LB gap. Pick 1 won on private. v11 dropped from public 0.98072 to private 0.98046, which lined up with the warning that public LB at 20 percent test is noisy. A later v17 run added two more models for 43 total. v17 had higher CV but lower public LB, so I did not pick it.

## Validation

5 fold StratifiedKFold with seed 42, applied identically across every base model and the stacker. The class distribution is heavily skewed: Low 58.7 percent, Medium 37.9 percent, High 3.3 percent. Stratification was essential for stable balanced accuracy CV.

Identical folds everywhere meant clean stacker training. No fold mismatch, no leakage between OOF folds and the meta learner.

## Feature Engineering

The breakthrough was Ordered Target Encoding adapted from [emanuellcs's public notebook](https://www.kaggle.com/code/emanuellcs/predicting-irrigation-need-xgboost-ote). Three pieces:

1. **Digit extraction** from numeric columns to expose sub feature structure that the raw values masked.
2. **Ordered Target Encoding** with 4x shuffle concatenation. Each fold receives four shuffled copies of the training data with leave one out target encoding applied in order. This reduces target leakage versus standard target encoding.
3. **Class weights** tuned by Optuna in the public notebook, reused as a starting point and lightly retuned per model.

I kept a second pipeline for diversity. The v3 pipeline used magic ratios, domain features, standard target encoding, and 2 way interactions. Models trained on this pipeline lagged OTE by 0.001 to 0.002 CV individually, but contributed unique signal to the stacker.

## Base Models

Six algorithm families. All CV scores are 5 fold balanced accuracy.

**Top tier OTE models:**
- LightGBM OTE: 0.97942
- XGBoost OTE: 0.97938
- CatBoost OTE: 0.97919
- XGBoost OTE shallow (Kaggle T4): 0.97927
- LightGBM OTE shallow / deep depth variants

**v3 pipeline gradient boosters:**
- 7 CatBoost variants from 0.97786 to 0.97834
- 4 LightGBM variants from 0.97131 to 0.97701
- 6 XGBoost variants from 0.97185 to 0.97561

**Neural networks:**
- RealMLP mahog: 0.97802
- RealMLP v3fix: 0.97108
- TabM v3fix: 0.97053

**Diversity models, individually weak but key for the stacker:**
- ExtraTrees OTE: 0.96156
- cuML RandomForest OTE: 0.96005
- cuML SVM RBF OTE: 0.96154
- Logistic Regression L1, L2, ElasticNet: 0.937 to 0.956
- KNN k=5 and k=15: 0.717 to 0.907
- cuML GaussianNB: 0.90860

## Stacking

LightGBM at level 2, with all base model OOF class probabilities as features. After the stacker produces fold predictions, I apply log bias correction to the argmax thresholds.

The log bias step is a simple search over additive shifts to the log probability of each class. It is fit on OOF predictions with balanced accuracy as the objective. For this dataset it consistently added 0.0005 to 0.001 CV.

v15 stacker CV: 0.98045
v15 stacker public LB: 0.98081
v15 stacker private LB: 0.98082

## What Worked

**Ordered Target Encoding.** All three top single models use it. Public CV jumped from 0.978 to 0.979 with no other change. The 4x shuffle was load bearing. Without it the OTE features overfit.

**Many weak models in the stacker.** Adding ExtraTrees at CV 0.96, KNN at CV 0.78, and Logistic Regression at CV 0.94 hurt my greedy ensembles but helped the stacker. The stacker learns when to trust each model. Per sample weighting beats fixed weights once you have a heterogeneous pool.

**Log bias threshold tuning.** Post hoc class shift on the stacker output added more CV than tuning any individual base model. Imbalanced 3 class problems with a balanced accuracy metric are very sensitive to argmax calibration.

**StratifiedKFold seed 42 everywhere.** Identical folds across all 41 base models and the stacker. No reindexing, no shuffled OOFs.

## What Did Not Work

**GraphSAGE GNN.** Trained on SageMaker T4 after the Kaggle P100 turned out to be sm_60 incompatible with current PyTorch. CV 0.97073, but adding it dropped LB by 0.0001. Removed from the final pool.

**RealMLP on OTE features.** OTE produces NaNs. fillna(0) on an MLP destroys the signal. CV 0.96509 versus 0.978 for RealMLP on the v3 pipeline. Median imputation might fix this but I ran out of time.

**TabPFN.** Hit the 12 hour Kaggle GPU limit on fold 1 of 5 even with subsampling. Promising architecture but infeasible at 630K rows on free tier T4.

**Pseudo labeling.** Hard labels from v11 and v12 agreement on 99.9 percent of test, weight 0.5, combined with the original train. CV crashed to 0.95390. The script lost the 4x OTE shuffle augmentation, and 99.9 percent label agreement adds almost no new signal anyway.

**Greedy forward selection.** v8 greedy with 8 models hit CV 0.98081 and LB 0.97973. Greedy overfits CV at scale. I stopped using it for final submissions early.

**Correlation pruning.** All my models correlate above 0.995 on OOF. Pruning removed signal without removing redundancy. v16 corrprune stacker dropped CV to 0.98015.

## Key Insights

**Public LB at 20 percent test is noisy.** A 0.0005 swing on public is essentially noise on private. CV on 5 fold 630K rows is a stronger private LB predictor than public LB. I picked v15 over v11 partly because v11 had a CV vs LB gap that suggested public split luck. The private result confirmed it. v11 lost 0.00026 from public to private.

**Stackers scale, greedy does not.** Greedy ensembles peaked around 7 to 9 models then degraded as more were added. The LightGBM stacker improved monotonically from 22 models through 41 models then saturated. The marginal gain per model decayed but never went negative.

**Diversity beats individual strength after a point.** Once I had three strong OTE models, adding a fourth strong gradient booster gave less signal than adding a weak ExtraTrees or KNN. Different error structures matter more than incremental accuracy.

**Stop trusting CV gains below 0.0001 once the stacker saturates.** v17 hit CV 0.98057, the highest of any stacker, but LB 0.98041, worse than v15. Past saturation, CV improvements are within fold variance and do not generalize.

**Blind blending is competition specific.** Equal weight averages and quick public notebook blends do work in some settings. Smooth regression metrics, balanced classes, large public test sets, and low correlation between public and private all lower the overfit risk. None of those held here. Three way imbalance with High at 3.3 percent, balanced accuracy as the metric, and a 20 percent public split made naive blends very brittle. Equal weights and weight optimisation by greedy or differential evolution all overfit CV and lost on private LB in my experiments. A learned stacker with proper OOF discipline is the safer default whenever the metric is sharp at the argmax boundary or the public test is small.

## Workflow

**AWS training.** Each base model had a Python launcher that built a SageMaker estimator and called `.fit()`. One shell command from my laptop kicked off a job on c5.9xlarge or g4dn.xlarge. The container wrote OOF and test predictions to S3. A second command synced them back to my local `predictions/` directory. Same pattern for the LightGBM stacker. This let me run several jobs in parallel without ever opening the AWS console.

**Kaggle GPU.** Same shape, different backend. `kaggle kernels push` to submit, `kaggle kernels output` to pull results. Each notebook source lived as a `.py` file in the repo. A small build step compiled it to `.ipynb` on demand. T4 only. P100 was unusable for current PyTorch wheels.

**Claude Code for notebook ports.** This was the biggest single accelerator. When a strong public notebook appeared on the leaderboard, I pointed Claude Code at it and asked for a port to my CV strategy. The output was a training script that:

- Used my StratifiedKFold seed 42 splits identically to every other base model
- Wrote OOF and test probabilities as `.npy` files matching my naming convention
- Plugged into the stacker model list with a one line edit

A 2 to 3 hour manual port turned into a 10 to 20 minute review and run. That is the main reason my model pool grew from 22 to 41 in the last two weeks. The stacker rewards adding diverse models, and conversion cost was the only bottleneck. Removing that bottleneck is what let me cover six algorithm families at depth. Same trick applied to refactoring scripts, debugging SageMaker job failures, and writing the launchers themselves.

## Compute

SageMaker handled the heavy training, mostly c5.9xlarge for boosters and g4dn.xlarge for cuML and neural nets. Kaggle GPU notebooks for one off T4 jobs and the public notebook.

## Code

- Public notebook: [PS6E4 12th Place: 41-Model Ensemble with Stacking](https://www.kaggle.com/code/wguesdon/ps6e4-30-model-ensemble-with-stacking)
- OOF dataset: https://www.kaggle.com/datasets/wguesdon/ps6e4-30model-oof-predictions
- Training code dataset: https://www.kaggle.com/datasets/wguesdon/ps6e4-training-code-30

## Thanks

Big thanks to the Kagglers whose public notebooks I built on:

- [@cdeotte](https://www.kaggle.com/cdeotte) for everything. The structural method I used here is lifted from his [PS6E3 1st place write up](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm), which I treated as a playbook for this competition. Beyond that, the [original data exact formula](https://www.kaggle.com/code/cdeotte/original-data-exact-formula) notebook unlocked feature engineering for me, and more broadly the volume of educational content Chris shares notebook after notebook, competition after competition is the single biggest reason I have been able to climb on Kaggle. Thank you.
- [@emanuellcs](https://www.kaggle.com/emanuellcs) for the [XGBoost OTE notebook](https://www.kaggle.com/code/emanuellcs/predicting-irrigation-need-xgboost-ote) that anchored my whole pipeline. The 4x shuffle ordered target encoding came straight from this work.
- [@yunsuxiaozi](https://www.kaggle.com/yunsuxiaozi) for the [PSS6E4 XGB CV 0.97981 notebook](https://www.kaggle.com/code/yunsuxiaozi/pss6e4-xgb-cv-0-979805) and a LightGBM advanced variant. Source for digit extraction and shallow XGB hyperparameters.
- [@mahoganybuttstrings](https://www.kaggle.com/mahoganybuttstrings) for the 2 way interaction TE pipeline used in the RealMLP base model. The mahogany feature pipeline was the core of the strongest neural network in my pool.
- [@ravi20076](https://www.kaggle.com/ravi20076) for threshold tuning ideas, neural net diversity discussions, and data drift analysis on the competition forum.
- [@utaazu](https://www.kaggle.com/utaazu) for the CatBoost pairwise TE notebook with logit bias tuning. Inspired my log bias post processing.
