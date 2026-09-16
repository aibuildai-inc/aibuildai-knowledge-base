# 15 Public, 30 Private - Solution Writeup (Public 0.417 Private 0.427)

Competition: um-game-playing-strength-of-mcts-variants
Rank: #30
Source: https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549587

First of all, I would like to thank @stefanoclss @xyzdivergence for teaming up!
*More to be added*

**Dataset**
- Dropped all Ludus_Coriovalli games from the training data
- No external data was generated
- Data augmentation was done: i.e double the number of rows by swapping agent1, agent2, AdvantageP1 and utility_agent1 (CV: +0.005, LB: -0.006)

**Cross Validation**
StratifiedGroupKFold by the first alphabet of the GameRulesetName (i.e. Zuz_Mel_7x7 -> Z), repeated 3 times. I noticed that similar games tend to have the same first letter in the GameRulesetName, so this method was used to reduce potential data leakage in the CV, since the test dataset is likely to have a shift.

**Preprocessing**
Transformed the target (to "correct" the unusual target distribution) before training (LB: -0.001)
```
df["utility_agent1"] = df["utility_agent1"] - (- 1 + 2 * df["AdvantageP1"])
```
Before:

After:

One-hot encoded the agent specific features (LB: -0.000 but speed improvements ~70%)
```
onehot_cols = [
            ['selection1', ['ProgressiveHistory', 'UCB1', 'UCB1GRAVE', 'UCB1Tuned']], 
            ['selection2', ['ProgressiveHistory', 'UCB1GRAVE', 'UCB1', 'UCB1Tuned']], 
            ['exploration_const1', ['0.1', '0.6', '1.41421356237']], 
            ['exploration_const2', ['0.6', '0.1', '1.41421356237']], 
            ['playout1', ['MAST', 'NST', 'Random200']], 
            ['playout2', ['Random200', 'NST', 'MAST']], 
            ['score_bounds1', [True, False]], 
            ['score_bounds2', [True, False]],
        ]
        
created_feats = 0
for col, unique in onehot_cols:
    for u in unique:
        df[f'{col}_{u}'] = (df[col] == u).astype(np.int8)
        created_feats += 1
```

**Feature Engineering and Selection**
We used it very sparingly as initial experiments showed that it is very easy to overfit the CV with FE and FS
- Feature Engineering: 17 features from [this public notebook](https://www.kaggle.com/code/litsea/mcts-baseline-fe-lgbm) was added (LB: -0.002)
- Feature Selection: CatBoost RFE was done on all our folds in the cross validation and we selected the feature set that performed the best on LB, irrespective of CV (LB: -0.002)

**Models used**
- CatBoost mainly
- LightGBM was used, but we did not get a satisfactory performance if GameRulesetName was dropped from the feature set
- DeepTables NN

**Postprocessing**
We tried two methods of postprocessing, improvement was similar for both
1. Multiply all predictions by 1.15
2. Train two quantile loss models, one targeted towards the -1 direction (alpha=0.2) and the other targeted towards the +1 direction (alpha=0.8). The initial negative predictions were ensembled with the alpha=0.2 model, while the initial positive predictions were ensembled with the alpha=0.8 model, with a weight of 0.8 for the initial predictions and 0.2 for the quantile loss prediction.

CV and LB improvement was about 0.004-0.005

**Ensembling**
Hill Climbing based on CV score (with both augmented and non-augmented models). We probably would have a chance for gold if we did it based on LB score 😔 but we believed the CV was robust and data leakage is not present.

**Reflection**
**What we could have done better**
- If CV and LB is close, it may be a green flag (compared to just optimising for the best CV)
- Two stage stacking
- Analyze the data more thoroughly to find a better CV split from the very beginning instead of sticking with GroupKFold for the entire of the first month
- I relied on CV a little too much in the first half of the comp, and did not submit promising ideas if they don’t work on the CV. I had started trying this augmentation in early October and did not submit due to the significantly worse CV score (and we almost gave up on this idea). It was only until we ran out of ideas towards the end then @stefanoclss decided we should try subbing the model with that to check LB, and to our surprise, it got **0.419**

**Failed Ideas**
- TFIDF (both on training set and test set)
- Pseudo-labelling
- Other target transformations
- Most feature engineering ideas
- Using the public notebooks. In fact, we referenced very little from the public notebooks when designing this solution.
- RandomOverSampler on GameRulesetName
- Spending too much time on feature engineering and feature selection
