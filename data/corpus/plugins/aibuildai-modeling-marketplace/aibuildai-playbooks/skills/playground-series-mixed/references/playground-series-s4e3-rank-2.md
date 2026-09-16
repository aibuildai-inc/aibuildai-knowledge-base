# 2nd place solution: OOF Ensemble

Competition: playground-series-s4e3
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s4e3/discussion/488106

Hello everyone, 

It never comes to me that I would be sharing my solution here. As a student, this is my first time fully participating in a Kaggle competition.

**Context:**
Steel Plate Defect Prediction Competition (Playground Series - Season 4, Episode 3)

Business Context: https://www.kaggle.com/competitions/playground-series-s4e3
Data Description: https://www.kaggle.com/competitions/playground-series-s4e3/data

**My Solution Code Link:** https://www.kaggle.com/code/yunqicao/2nd-place-solution-steel-plate-defect-prediction

Here is my solution:


**1. Data**

I combine the provided data with the original dataset and exclude rows with multiple labels, as they constitute only a small portion.


**2. Feature Engineering**

I create three features and remove some features. My decisions on feature selection are primarily guided by cross-validation scores, feature importance, and pairwise correlation.


    def feature_engineering(data):

        data['Ratio_Length_Thickness'] = data['Length_of_Conveyer'] / data['Steel_Plate_Thickness']
        data['Normalized_Steel_Thickness'] = (data['Steel_Plate_Thickness'] -data['Steel_Plate_Thickness'].min()) / (data['Steel_Plate_Thickness'].max() - data['Steel_Plate_Thickness'].min())
        data['X_Range*Pixels_Areas'] = (data['X_Maximum'] - data['X_Minimum']) * data['Pixels_Areas']

        return data

    features_to_drop = ['Y_Minimum', 'Steel_Plate_Thickness', 'Sum_of_Luminosity', 'Edges_X_Index', 'SigmoidOfAreas', 'Luminosity_Index', 'TypeOfSteel_A300']

Thanks to:

https://www.kaggle.com/code/thomasmeiner/ps4e3-eda-feature-engineering-model

https://www.kaggle.com/competitions/playground-series-s4e3/discussion/482401


**3. Cross Validation and Tuning Parameters**

I choose four multi-class models(xgb, lgbm, cat, hgbc), implement a 10-fold cross-validation and tune hyperparameters using Optuna. For the final predictions, I calculate the average of the predictions across the 10 folds.

Thanks to:

https://www.kaggle.com/code/ankurgarg04/steel-plate-defect-detection-technique-exploration


**4. Out-of-Fold Ensemble**

I create an ensemble of three models by utilizing out-of-fold (OOF) files. To ensure consistency, I replicate or adapt their publicly shared notebooks to generate OOF files with an identical train-validation split. The weights for the ensemble are optimized for cross-validation score using Nelder-Mead optimization.

Model 1: My model

Model 2: Replicated from https://www.kaggle.com/code/noepinefrin/0-89534-clustered-feature-lgbm-xgb-cat

Model 3: Adapted from https://www.kaggle.com/code/arunklenin/ps4e3-steel-plate-fault-prediction-multilabel

Thanks to:

https://www.kaggle.com/code/lucamassaron/steel-plate-eda-xgboost-is-all-you-need

https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175614


**5. Simple Ensemble**

At the final step, I combine my submission with outstanding public submissions. In addition to the notebooks previously referenced, I also incorporate output from https://www.kaggle.com/code/cyrilbourgeois/playground-s4e03-xgboost-ensembler. I had limited opportunities to submit in the final step, so I assigned weights based on intuition. As it turned out, using average weights is effective.


**6. Other Thoughts**

**Other approches I tried but not work:**

1. Pseudo labeling;

2. Stacking approach with XGB as the meta model;

3. Change eval_metric to 'roc_auc'

Thanks to:

https://www.kaggle.com/code/arnogils/basic-xgb-for-steel-plate-defects-0-8948

https://www.kaggle.com/competitions/playground-series-s3e26/discussion/464863


**7. Code**

**My Solution Code Link:** https://www.kaggle.com/code/yunqicao/2nd-place-solution-steel-plate-defect-prediction

By the way, simply ensembling the outputs from the public notebooks I mentioned with equal weights can get an excellent result. This approach achieves a public leaderboard score of 0.89684 and a private leaderboard score of 0.88923, marking my highest score on the public leaderboard. Although I choose not to use this version as my final submission, it shows that these public notebooks are distinguished.


**Many Thanks to Everyone!**

Thank you for reviewing my solution. I'm grateful for all the public notebooks and discussions. They are really valuable.  I look forward to sharing my insights during the next competition.
