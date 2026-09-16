# [2nd Place] Winning based on ID Shift Analysis

Competition: playground-series-s5e12
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s5e12/writeups/2nd-place-solution-winning-based-on-id-shift-an

# Thanks

Happy New Year! This month-long marathon is finally over! Congratulations to all the winners! I want to thank the organizers for hosting this interesting competition.

This was my first time participating in a Playground Series, and I truly poured my heart and soul into it. I started learning ML and joined Kaggle in March 2025, so it has been 9 months now. This is a wonderful New Year's gift. I hope to share this joy with all Kagglers in the community, and I wish you all the best in 2026!!!



# Learning from discussion

Since the start of this competition, I almost never looked at the public notebooks in the code section because it was filled with "upvote bait"—notebooks where people simply mix others' kernels to farm upvotes. Additionally, there were one or more arrogant cheaters using multiple accounts to perform "blind blending."

In contrast, the discussion section contained a lot of high-quality content. I read every single post in the discussions in detail and drew significant inspiration from them:

### Orig FE

@masayakawamata shared in [this post](https://www.kaggle.com/competitions/playground-series-s5e12/discussion/648186) that using statistical features from the original dataset could lead to score improvements. This indeed increased my score slightly; however, I didn't perform any aggressive Feature Engineering, so I only attempted to use the most fundamental statistics.

### The weird dataset

@tilii7 mentioned in [this post](https://www.kaggle.com/competitions/playground-series-s5e12/discussion/652262) that this dataset is very strange. The organizers distorted the feature distributions and removed decisive strong features, resulting in an inconsistency between the Train and Test distributions. To be honest, at that point, I even considered giving up on the competition.

### ID Shift

The turning point came with [this post](https://www.kaggle.com/competitions/playground-series-s5e12/discussion/659313) by @laureanoarcanio . I strongly recommend everyone read this post as it was the pivotal moment of this competition. He mentioned a phenomenon: using `id` as a feature surprisingly boosts the LB score.

Furthermore, @alan1305 pointed out:

> One possible reason could be the interaction of id and physical_activity_minutes_per_week. If we plot the rolling mean of physical_activity_minutes_per_week, it shifts to test distribution at the end of train dataset. Therefore, including id may help indicate which row is "closer" to test set.

I conducted an Adversarial Validation analysis (Micro-Scan of the Tail Transition) and found that this phenomenon indeed exists: **the closer the data is to the tail, the closer its distribution is to the test set.**

This is actually a serious case of ID leakage, but given the significant shift between train and test, this finding was crucial. (I honestly don't know why @alan1305 only received 5 upvotes... This was clearly a game-changing finding, yet people were busy upvoting public blind-blending notebooks...)

### A validation strategy

In [this post](https://www.kaggle.com/competitions/playground-series-s5e12/discussion/662501), @masayakawamata proposed a new validation strategy. Since the tail data better "represents" the test set, it makes sense to use the tail data directly for CV. He also introduced sample weighting.

### Concept Shift

@siukeitin mentioned in [this post](https://www.kaggle.com/competitions/playground-series-s5e12/discussion/663033) that simple sample weighting doesn't work well because there is **Concept Shift**.

I also conducted experiments regarding this. More specifically, within the training set, the more the data resembles the test set (low covariate shift), the greater the concept shift becomes. I divided the data into 10 bins and found that samples with the smallest covariate shift had the largest discrepancy in label distribution, whereas the original dataset did not suffer from this issue.
```
CUTOFF_ID = best_id

cat_cols = train.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))

train_ids = train['id'].values
cutoff_mask = train_ids >= CUTOFF_ID

X_tail = train.loc[cutoff_mask].drop(columns=['id', 'diagnosed_diabetes'])
y_adversarial_tail = pd.Series(1, index=X_tail.index)

X_head = train.loc[~cutoff_mask].drop(columns=['id', 'diagnosed_diabetes'])
y_adversarial_head = pd.Series(0, index=X_head.index)

X_adv_train = pd.concat([X_head, X_tail], axis=0)
y_adv_train = pd.concat([y_adversarial_head, y_adversarial_tail], axis=0)

print(f"1. Training Discriminator: Head({len(X_head)}) vs Tail({len(X_tail)})...")
params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 500, 
    'learning_rate': 0.05,
    'num_leaves': 31,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

model = lgb.LGBMClassifier(**params)
model.fit(X_adv_train, y_adv_train)

print("2. Evaluating Head samples...")
head_similarity_score = model.predict_proba(X_head)[:, 1]

head_true_labels = train.loc[~cutoff_mask, 'diagnosed_diabetes'].values

df_eval = pd.DataFrame({
    'tail_similarity': head_similarity_score,
    'true_label': head_true_labels
})

# Binning
df_eval['bin'] = pd.qcut(df_eval['tail_similarity'], q=10, labels=False, duplicates='drop')

agg = df_eval.groupby('bin', observed=False).agg({
    'true_label': ['count', 'mean'],
    'tail_similarity': 'mean'
})
agg.columns = ['count', 'target_mean', 'similarity_mean']

true_tail_mean = train.loc[cutoff_mask, 'diagnosed_diabetes'].mean()

print(agg)
print(f"\nTail Set True Mean: {true_tail_mean:.4f}")

plt.figure(figsize=(12, 6))
bars = sns.barplot(x=agg.index, y=agg['target_mean'], color='skyblue', alpha=0.8)

for i, v in enumerate(agg['target_mean']):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.axhline(true_tail_mean, color='red', linestyle='--', linewidth=2, label=f'True Tail Mean ({true_tail_mean:.3f})')

plt.title(" Do 'Tail-like' Head samples have correct labels?", fontsize=14)
plt.xlabel("Similarity to Tail (0=Unlikely, 9=Indistinguishable from Tail)", fontsize=12)
plt.ylabel("Diabetes Rate (Target Mean)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()
```
Result:
```
1. Training Discriminator: Head(677000) vs Tail(23000)...
2. Evaluating Head samples...
     count  target_mean  similarity_mean
bin                                     
0    67700     0.643648         0.019489
1    67700     0.645583         0.021456
2    67700     0.642836         0.022520
3    67700     0.641832         0.023592
4    67700     0.649926         0.025069
5    67700     0.655465         0.027066
6    67700     0.639764         0.029891
7    67700     0.612186         0.034075
8    67700     0.581004         0.041102
9    67700     0.521359         0.070540

Tail Set True Mean: 0.6214
```

Therefore, I decided to apply weighting to the Original dataset as well.


# Modeling

Based on the validation strategy derived from the analysis above, I trained a diverse set of models. The core philosophy was to force the models to learn the distribution of the "Tail" (which resembles the Test set) and the "Original" dataset (which holds the ground truth logic), while suppressing the noise from the "Head" of the training data.

Here is the summary of the single models used in the final ensemble:

| Model Type       | Model Name     | Key Strategy / Features                                                                                                                        | Cutoff AUC |
| :--------------- | :------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- | :--------- |
| **LightGBM**     | `lgb`          | Baseline with interaction features & Categorical encoding.                                                                                     | 0.70435    |
|                  | `lgb_te`       | **Tail-based Target Encoding**: TE calculated using only Tail + Orig data.                                                                     | 0.70382    |
|                  | `lgb_safe`     | **Feature Selection**: Dropped "toxic" features (high concept shift) like `physical_activity`.                                                 | 0.67464    |
|                  | `lgb_pseudo`   | **Pseudo-Labeling**: Trained on Tail + Pseudo-labeled Test data (Top/Bottom confidence). <br>Applied **Quantile Mapping** to fix distribution. | 0.70081    |
| **XGBoost**      | `xgb`          | Baseline XGBoost with standard parameters.                                                                                                     | 0.70335    |
|                  | `xgb_weighted` | **Sample Weighting**: **Tail (16x)**, **Orig (8x)**, Head (1x).                                                                                | 0.703??*   |
|                  | `xgb_binned`   | **Binning**: Extensive binning of numerical features + Target Encoding from Original data.                                                     | 0.69898    |
|                  | `xgb_pl_soft`  | **Soft Pseudo-Labeling**: Regression objective (`reg:squarederror`) on soft labels from best sub.                                              | 0.70231    |
|                  | `xgb_residual` | **Residual Modeling**: Boosting over a fixed base margin derived from Tail Mean.                                                               | 0.70474    |
| **CatBoost**     | `cat_weighted` | **Sample Weighting**: Same 16x/8x/1x strategy as XGBoost.                                                                                      | 0.70489    |
|                  | `cat_inter`    | **Explicit Interactions**: Manually created interaction features (e.g., `gender_age`).                                                         | 0.70489    |
| **NN**           | `nn_dae`       | **DAE** + **Fine-tuning on Tail only**.         | 0.70034    |
|                  | `nn_stacking`  | Stacked Neural Network (Input: OOF predictions).                                                                                               | 0.70616    |
| **Linear/Other** | `tail_lr`      | **Tail-Only LR**: Logistic Regression trained *only* on the last 10% of data. <br>Post-processed with Quantile Mapping.                           | 0.70345    |
|                  | `tail_gam`     | **Tail-Only GAM**: Generalized Additive Model trained on Tail data.                                                                            | 0.69325    |

*> Note: The Cutoff AUC is calculated on the validation set defined by `id >= 678260` (approx. last 3% of train data).*

## Ensemble Strategy

Given the diverse nature of the models (Linear, GAM, NN, and various GBDTs), I used **Bagged Hill Climbing** to find the optimal weights.

* **Method:** Bagged Hill Climbing (50 bags, 2000 iterations per bag) optimizing for **Cutoff AUC**.
* **Final Result:** The ensemble significantly improved the robustness.

  * **Ridge Regression (Positive)**: 0.70751 Cutoff AUC
  * **Bagged Hill Climbing (Best)**: **0.70771 Cutoff AUC**

The final submission was generated using the weights from the Bagged Hill Climbing method.

check [here](https://www.kaggle.com/code/daylighth/2nd-place-solution-ensumble) if you need my notebook.  
what's more, I also have a notebook to record my hypothesis. You can check [here](https://www.kaggle.com/code/daylighth/ps-s5e12-hypothesis)

# Some advice for beginners

I noticed that many participants in this competition appear to be new to machine learning. As a beginner myself, I completely understand the learning curve, so I’d like to share a few simple tips based on my experience:

1. **Conduct purposeful EDA.**
   The ultimate goal of Exploratory Data Analysis (EDA) is to guide your modeling strategy. If your data analysis doesn't provide insights for feature engineering or model selection—and is just about plotting pretty charts—then I believe it lacks real value. Always ask yourself: "How does this plot help me build a better model?"
2. **Don't obsess over micro-tuning hyperparameters.**
   Spending too much time fine-tuning parameters is often meaningless. Changing the learning rate from `1e-4` to `1e-5` is rarely worth wasting a submission. The most efficient approach is to spend that time trying out diverse models or exploring new feature engineering ideas.

Finally, Happy New Year again to everyone! I wish you all the best and hope you win gold medals in your upcoming competitions!
