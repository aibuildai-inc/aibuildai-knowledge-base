# First Place Solution

Competition: march-machine-learning-mania-2025
Rank: #1
Source: https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572717

Thanks to Kaggle for hosting this competition again. 

**Background**
My journey this year on this competition was not smooth, on mid February ( weeks before the tournament ) I decided to start slowly collecting different data to build my model based on, mainly: Kenpom, MassyOrdinal, and 538. I had kenpom data prepared and tested on data up to 2024 ( last year ), however I missed the fact that 538 is not available  for 2025, and MassyOrdinal ( was available late ). This is when I had to shift my approach, drawing inspiration from  @raddar notebook.

**Details**
The notebook provide really well engineered feature, the feature include the provided data, features derived using average statistics and team quality.

I built my approach based on that. 

Although I tried tunning  different algorithms ( Castboost, LightGBM ), XGBoost performed the best, so I made feature selection ( 29 feature ) instead of ( 25  feature selected by raddar )
`
features = [
    "men_women",    
    "T1_seed",
    "T2_seed",
    "Seed_diff",
    "T1_avg_Score",
    "T1_avg_FGA",
    "T1_avg_OR",
    "T1_avg_DR",
    "T1_avg_Blk",
    "T1_avg_PF",
    "T1_avg_opponent_FGA",
    "T1_avg_opponent_Blk",
    "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score",
    "T2_avg_FGA",
    "T2_avg_OR",
    "T2_avg_DR",
    "T2_avg_Blk",
    "T2_avg_PF",
    "T2_avg_opponent_FGA",
    "T2_avg_opponent_Blk",
    "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    "T1_elo",
    "T2_elo",    
    "elo_diff",
    "T1_quality",
    "T2_quality",
]
`


I tunned XGBoost parameters further.
`
param = {}
param["objective"] = "reg:squarederror"
param["booster"] = "gbtree"
param["eta"] = 0.0093 
param["subsample"] = 0.6
param["colsample_bynode"] = 0.8
param["num_parallel_tree"] = 2 
param["min_child_weight"] = 4
param["max_depth"] = 4
param["tree_method"] = "hist"
param['grow_policy'] = 'lossguide'
param["max_bin"] = 38
num_rounds = 704 `

The performance was better, and achieved a lower brier score.

Then comes the final part, since I had a bumpy  experience this year, I decided to boost the model prediction, so I did : 
- Increase prediction that is below 85% by 10% ( Men & Women Predictions )
- Manually overriding (30-70% increase ) few games in the men tournament, that the model did not seem confident even though, for early rounds higher seed has higher chances of winning + Experts confirmed it : 

| Match | ID | Pred | Pred-New |
| --- | --- | --- | --- |
| Baylor v Mississippi State | 2025_1124_1280 | 0.58185013 | 0.98185013 |
| BYU vs VCU | 2025_1140_1433 | 0.683304479 | 0.980304479 |
| ST Mary CA vs Vanderbilt | 2025_1388_1435 | 0.714379186 | 0.954379186 |
| Mississippi vs North Carolina | 2025_1279_1314 | 0.664069204 | 0.964069204 |
| Texas A&M vs Yale | 2025_1401_1463 | 0.853761116 | 0.953761116 |
| UCLA vs Utah St | 2025_1417_1429 | 0.673425592 | 0.973425592 |

Initially, the features helped to achieve a lower Brier score. Tuning XGBoost further improved the score, and finally, manual adjustments corrected a few bad predictions in the early rounds, resulting in an overall better Brier score.

**Source**
Final code: https://www.kaggle.com/code/modeh7/final-solution-ncaa-2025
