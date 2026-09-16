# 5th Place Solution

Competition: um-game-playing-strength-of-mcts-variants
Rank: #5
Source: https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549585

Huh! Due to the saturation in the top scores, we were expecting a sad shake-up, again. But we had crossed our fingers on our ensemble submission. And it didn’t let us down, me and @sercanyesiloz got our first gold medal and became competition masters!

I will write a brief summary about the solution for now, will add details tomorrow.

---

### Anil's Pipeline [[Link]](https://www.kaggle.com/code/nlztrk/um-5th-place-solution-catboost-part)
- CatBoost
- Stratified Group 10-Fold (similar to @yunsuxiaozi's scheme)
- Created augmented training rows by switching agent1-agent2, inverting advantage and utility label
- Generated augmented rows for inference and mean blended with original row predictions
- **CV OOF:** 0.4059 - 0.4030 (TTA)
- **LB:** 0.421 / 0.427

Also added adjusted advantages as features:
```               
((pl.col("AdvantageP1") * pl.col("Completion")) + (pl.col("Drawishness")/2)).alias("adv_p1_adj"), ((pl.col("AdvantageP2") * pl.col("Completion")) + (pl.col("Drawishness")/2)).alias("adv_p2_adj")
```
and didn't use any other feature!

**Catboost params:**
```
cb_params = {
    "random_state": 42,
    "iterations": 3000,
    "learning_rate": 0.085,
    "depth": 10,
    "verbose": 100,
    "use_best_model": False,
    "task_type": "GPU",
    "l2_leaf_reg": 0.,
    "border_count": 254,
    "objective": "RMSE",
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
}
```

---

### Sercan's Pipeline
- Blend of CatBoost + LightGBM + DeepTables

#### DeepTables

- Min-Max Scaling
- Group 6-Fold
- Learning Rate Scheduler
- Adam Optimizer

```
nets = ['dnn_nets'] + ['fm_nets'] + ['cin_nets'] + ['ipnn_nets']
hidden_units = ((1024, 0.0, True), (512, 0.0, True), (256, 0.0, True), (128, 0.0, True))
embeddings_output_dim = 4
embedding_dropout = 0.1
apply_gbm_features = True
epochs = 7
learning_rate = 0.001
```


#### GBDT

- Stacked LightGBM & CatBoost
- Group 5-Fold
- Some TF-IDF features from EnglishRules, LudRules, agent1 and agent2
- Post process on final predictions
- Weighted ensemble

```
lgb_params = {
    'objective': 'regression',
    'min_child_samples': 24,
    'num_iterations': 13000,
    'learning_rate': 0.07,
    'extra_trees': True,
    'reg_lambda': 0.8,
    'reg_alpha': 0.1,
    'num_leaves': 64,
    'metric': 'rmse',
    'device': 'cpu',
    'max_depth': 24,
    'max_bin': 128,
    'verbose': -1,
    'seed': 42
    }

ctb_params = {
    'loss_function': 'RMSE',
    'learning_rate': 0.03,
    'num_trees': 13000,
    'random_state': 42,
    'task_type': 'GPU',
    'border_count': 254,
    'reg_lambda': 0.8,
    'depth': 8
    }

```

---

### Final Ensemble [[Link]](https://www.kaggle.com/code/sercanyesiloz/um-mcts-5th-place-solution)

- Weighted blend of Anil's and Sercan's model sets
