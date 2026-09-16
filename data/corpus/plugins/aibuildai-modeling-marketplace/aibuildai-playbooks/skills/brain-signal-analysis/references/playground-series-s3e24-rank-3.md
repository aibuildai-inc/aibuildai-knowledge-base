# #3 Private 8 Public approach - simple ensemble with probing

Competition: playground-series-s3e24
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e24/discussion/455248

Hello all,

Firstly wishing one and all a happy Diwali! I hope the festival of lights brings joy, good health and lots of success to one and all! I wish to extend sincere thanks to Kaggle for an interesting episode. It was indeed a great experience with lots of takeaways and learning that we can use outside of Kaggle too. I also wish to thank others participants for their generous contributions with special mention of the below users-
1. @cv13j0 -- his pseudo label notebook was great and offered a slightly different perspective to the assignment in the early stages 
2. @arunklenin -- his notebook was one of the most comprehensive work in the episode, especially his EDA and feature engineering sections
3. @oscarm524 -- his notebook provided a good reference at the start of the competition
4. @paddykb -- I appreciate his consistent and valuable contribution to the assignment with this light GBM pipeline

I outline my approach as below-
## Feature engineering
1. I initially started off with my [post](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/450314) regarding secondary features but found that not using them was a better alternative with regard to the CV score on the synthetic dataset. 
2. I used brute-force features very similar to @arunklenin [notebook](https://www.kaggle.com/code/arunklenin/ps3e24-eda-feature-engineering-ensemble). My approach was similar to his notebook, but I preferred to restrict myself to 80-120 features only. I eliminated features using [permutation importance](http://scikit-learn.org/stable/modules/permutation_importance.html). This takes a long time to execute, hence kernel time management is key. I used my local PC to engender this task. 
3. I used a 10x1 stratified k-fold CV strategy based on the target classes. I tried the 10x3 repeated stratified K-fold strategy too, but it did not give me any added advantage in the challenge, so I reverted to the 10 x 1 stratified K-fold to good effect
4. I used the original dataset for my models but not the adjutant original data mentioned in this [post](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/450510). Using this data did not bode well for me at the start of the competition, so I decided to use the competition synthetic data and the [original data](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals) only. 

## Model development 
I used a wide variety of models in my pipeline including the below-
1. Catboost - I used 3 models with varied parameters 
2. LightGBM - I used 5 LGBM models with varied parameters 
3. XGBoost - I used 3 diverse parameter models 
4. Random Forest
5. Logistic Regression
6. TabNet Classifier -- my work without this option was almost equally good, so this gave me a puny advantage
7. Multi-layer perceptron -- the neural network marginally contributed to improving my ensemble
8. Generalized Additive Model

## Ensemble
I used the below ensemble strategies using my single models as above
1. Hill-climbing 
2. Optuna 
3. Stacking

I assessed the effectiveness of the ensemble using my post-ensemble CV score and its relation to the leaderboard. I found that my Optuna framework correlated well to the LB and I was able to fine-tune the weights based on probing, so I decided to go ahead with this approach. 
My final submission consisted of an Optuna ensemble and probing. I manually modified some model weights in-fold based on the leaderboard score using probing. I found a marginal impact on the CV score using this strategy, this perhaps worked for me. My final submissions consisted of a probed Optuna tuned ensemble and another without the probing. Both submissions worked well, but the probed submission was just a bit better, awarding me the third place. 

## What did not work
1. Using more than 130-140 features -- I tried this in a version to no good CV score improvement and it did not add to my LB score as well, reinforcing the need to prune lowly useful features 
2. Neural networks and TabNet -- their weights in my final ensemble were puny. Perhaps not using them in totality would not have made a great difference to the overall model framework, so they could be included in this list
3. Blind probing-- I tried this based on some public approaches but did not go ahead with this considering my past experience with the playground series. A blind ensemble of top public work seldom generalizes, so one may probe with care and caution for the best results. 

## My takeaways
1. Rely on the CV score 
2. Evaluate models on a good and appropriate CV strategy
3. Do not over-complicate when not needed- simple models have a lot of power
4. Model parameter tuning is much secondary to feature engineering. 
5. Probe with care- one may resist the temptation to probe into the public LB using weights that do not impact the CV positively. 

Best regards and happy learning!
