# [7th Place] Reversing the Polarity of the Information Flow

Competition: playground-series-s3e8
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s3e8/discussion/392937

Thanks to Kaggle for another interesting and enjoyable competition in this great series, and to everyone who participated and shared their insights (and models!). This competition gave me a chance to try something wacky which, while it’s the main focus of this write-up, actually yielded less than a quarter of the final ensemble.

**Executive Summary:**
I used the following three-step prediction workflow to engineer some extra features and generate more models:

- (Original) Training -> Test 
- (Reverse) Training <- Test 
- (Double-back) Training -> Test 

**0.	The Set-up**
Using the predictions of models on both training and test sets as features is something I’ve thought about doing before. What I did was largely for interest, and it would perhaps with hindsight have been easier to obtain model predictions for the training set through cross-validation. However, the relatively even size ratio of the training and test sets suggested that this might be a favourable competition for the reversal and double-back approach. More pertinently, I hadn’t even considered using cv to get training predictions until the reversal was largely a sunk cost of several hours’ work. I chose the following seven publicly available models to serve as the chassis for my reverse models, These were chosen because they were all both high scoring and apparently capable of being run without dependencies on private datasets:
Model 1: [PS3E8 Xgb/Lgbm/Cat Ensemble Baseline](https://www.kaggle.com/code/tetsutani/ps3e8-xgb-lgbm-cat-ensemble-baseline) by @tetsutani, v11. 
Model 2: [CB_TPGS_S3E8_v1](https://www.kaggle.com/code/omarvivas/cb-tpgs-s3e8-v1) by @omarvivas, v17. 
Model 3: [Samurai_s3e8ens+featureEngineering_ CB,XB,LB](https://www.kaggle.com/code/samuraikaggle/samurai-s3e8ens-featureengineering-cb-xb-lb) by @samuraikaggle, v 9. 
Model 4: [Gemstone Price Prediction - S3E8](https://www.kaggle.com/code/satoshiss/gemstone-price-prediction-s3e8) by @satoshiss, v11. 
Model 5: [PS-S3-E8 EDA and modeling](https://www.kaggle.com/code/francescoliveras/ps-s3-e8-eda-model-en-es) by @francescoliveras, v11. 
Model 6: [XGboost regression baseline](https://www.kaggle.com/code/rkoirala129/xgboost-regression-baseline) by @rkoirala129, v3. 
Model 7: [playground_season_3_episode_8](https://www.kaggle.com/code/pawebiegun/playground-season-3-episode-8) by @pawebiegun, v10.

**1.	Running the Original Models**
Since I was going to be working on adaptations of these models, it was essential to get them all working for me in their original states; anything that I couldn’t run “as is” would be abandoned as a lost cause for reversing. In practice, all seven models ran without too much difficulty. This yielded predictions which should all have been the same as those in the original notebooks, except that I later discovered that I had mistakenly copy-and-edited the ‘wrong’ version of Model 3, v9 rather than the better scoring v4.

**2.	Engaging Reverse Gear**
I now had each model’s predictions for the test set, but still needed corresponding numbers for the training set before I could use each model’s output as a newly engineered feature. In order to obtain them, I required price values for the test set, which were obviously the target of the competition and hence could only be estimated. I used the predictions of the then best-scoring public model as pseudo-ground truth here, since this model was independent of my seven (an apparent dependency on a private dataset meant that I hadn’t tried to adapt it):
Model 0: [PS S3E8, 2023 EDA and Submission](https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e8-2023-eda-and-submission) by @sergiosaharovskiy, v18.
I thus added the predictions of @sergiosaharovskiy’s model as the price column to the test set, see [here](https://www.kaggle.com/datasets/jbomitchell/gemstone-price-modelling?select=test_as_train.csv), using this to train the reversed models, which in turn would predict price for the original training set (with its ground truth deliberately obscured).
Next, I adapted each of the seven models to predict backwards, which was largely a matter of respecifying their training and test sets by editing a few paths. All seven of them produced sets of predictions without too much trouble.
Thus, I now had predictions from all seven models for both test and training sets, and these seven columns were henceforth to be treated as engineered features. See the expanded sets of features: [training](https://www.kaggle.com/datasets/jbomitchell/gemstone-price-modelling?select=train_m.csv) and [test](https://www.kaggle.com/datasets/jbomitchell/gemstone-price-modelling?select=test_m.csv)

**3.	Double-back**
With the expanded feature sets in hand, the next objective was to use the models to generate new predictions for the test set, with the hope that the predictions would be at least meaningfully different from the existing ones, adding diversity, and more optimistically better scoring. Changing the feature sets and removing dependencies on the original data were required here. Given more time, patience and coding skills, this could doubtless have been accomplished for all seven models. However, avoiding throwing good debugging time after bad, I ultimately settled for generating four double-back (DB) models. Their scores were:
Model 1: (Original Priv 571.75670, Pub 575.16974; DB Priv 573.17408, Pub 577.21897);
Model 4: (Original Priv 573.08148, Pub 577.55176; DB Priv 572.40829, Pub 576.53189);
Model 6: (Original Priv 573.89711, Pub 578.15487; DB Priv 571.99507, Pub 575.98026);
Model 7: (Original Priv 577.24019, Pub 578.28833; DB Priv 572.94775, Pub 576.50537)
Thus, I improved the scores for three out of four models by adding the engineered features through this reverse and double-back workflow.

**4.	Ensemble**
Given the number of diverse, but similarly scoring, models, median ensembling seemed a good strategy for the final submission. I looked at adding in any public models that were better than the weakest of mine, and also sought to take the median of an odd number of predictions. Hence, I added;
Model 8: [Simple Model](https://www.kaggle.com/code/satyaprakashshukl/simple-model) by @satyaprakashshukl, v6.
Model 9: [Playground-S3_E8-Gemstones-XGB,LGB,CB-ensemble](https://www.kaggle.com/code/eamonntweedy/playground-s3-e8-gemstones-xgb-lgb-cb-ensemble) by @eamonntweedy, v1 (two output models).
Model 10: [Plotly EDA | Feature Engineering | Optuna](https://www.kaggle.com/code/ch124uec/plotly-eda-feature-engineering-optuna) by @ch124uec, v6.
In my [eventual submission](https://www.kaggle.com/code/jbomitchell/gem-median-ensembler), I took the median of seventeen models, of which four were the ones I engineered through the reversal. The 17 models were models: 0, 1, 2, 3 (v4 & v9), 4, 5, 6, 7, 8, 9 (two submission files), 10, 1DB, 4DB, 6DB and 7DB.
This generated the scores of (Priv 570.52851, Pub 574.77083), and allowed me the somewhat atypical experience of benefitting for once from the shake-up, moving from 15th to 7th.
