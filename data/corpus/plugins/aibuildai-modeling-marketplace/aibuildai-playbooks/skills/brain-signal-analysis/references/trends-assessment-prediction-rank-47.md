# 47th Place - Site 2 Correction

Competition: trends-assessment-prediction
Rank: #47
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162758

The best public notebook [here][1] scores LB 0.1590. If you analyze the predictions, you will see that it underestimates target `age` and underestimates target `domain2_var1` for site 2 brain scans. If you adjust these 2 targets, you increase it's LB score to LB 0.1585. That's a 0.0005 LB Boost !

# Classify Brain Scans as Site 2
Using an ensemble of logistic regression with L2 regularization and RAPIDS cuML support vector classification, you can build a site 2 classifier with AUC = 0.80. Below is the classifier's probability distribution. There are 3507 test samples with `prob = 0` of being site 2. There are 800 test samples with `0 &lt; prob &lt;= 0.5` and 508 test samples with `0.5 &lt; prob &lt; 1` and 1062 with `prob = 1`. (The 1062 includes the known 510).



# Determine Target Bias of Site 2
Next if you plot the targets versus site 2 probability, we see that the targets `age` and `domain2_var1` decrease as the probability of site 2 brain scans increase. If we **assume** that site 1 and site 2 have the same `age` and `domain2_var1` mean then this is a statistically significant model bias based on site feature drift. 

(For samples of size 1000, the sample mean's standard deviation of target is `0.3 = OOF std / sqrt(1000)` therefore our observation is not random chance. )



# Post Process Correction - Boost LB 0.00050!
From the plot we observe that this biased model needs to add `1.9` to the test target `age` for site 2 brain scans with `prob = 1` and needs to add `0.67` to the test target `domain2_var1` for site 2 brain scans. Furthermore we can adjust `0.5 &lt; prob &lt; 1`, `0 &lt; prob &lt;= 0.5`, and `prob = 0` based on the plot. 

### 1590 Public Notebook With PP



### 1590 Public Notebook Without PP


# Private Leaderboard
I was hoping that private leaderboard would have more site 2 data. Then this trick would increase private LB more than public LB and I would have climbed rank on private leaderboard :-) Instead this PP only increased private LB by +0.00025 and I dropped rank on private LB :-(

**UPDATE**: Two simple experiments suggest that private test has 510 site 2 brain scans and public test has 1020 site 2 brain scans. The experiments are described [here][5]

# Final Models
My final model was an ensemble of these 4 great notebooks:
[trends-multi-layer-model][1] by NQ @david1013
[BaggingRegressor + RAPIDS Ensemble][2] by Andy @andypenrose
[TReNDS：Simple NN Baseline][3] by Tawara @ttahara
[TRends: EDA,DNN for Predicting Age][4] by Hema @hemavivekanandan 

Ensemble achieved LB 0.1588 with weights `0.60 * model_1 + 0.20 * model_2 + 0.15 * model_3 + 0.05 * model_4` and then PP increased this to LB 0.1583.

[1]: https://www.kaggle.com/david1013/trends-multi-layer-model
[2]: https://www.kaggle.com/andypenrose/baggingregressor-rapids-ensemble
[3]: https://www.kaggle.com/ttahara/trends-simple-nn-baseline
[4]: https://www.kaggle.com/hemavivekanandan/trends-eda-dnn-for-predicting-age
[5]: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162907
