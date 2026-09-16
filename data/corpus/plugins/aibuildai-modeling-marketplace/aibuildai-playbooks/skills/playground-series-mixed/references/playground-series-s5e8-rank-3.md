# 3rd Place Solution — OOF Stacking + AutoGluon

Competition: playground-series-s5e8
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/3rd-place-solution-oof-stacking-autogluon

Many thanks to **Kaggle** for organizing such a wonderful Playground competition! This month’s dataset was excellent, and I truly enjoyed the process and learned a lot. 

Special thanks to @mulicmu ’s “[Dive into Deep Learning](https://d2l.ai/)”, the [AutoGluon ](https://auto.gluon.ai/) library, @optimistix  for the ideas shared in the [1st place solution](https://www.kaggle.com/competitions/playground-series-s4e8/writeups/optimistix-1st-place-solution-72-oofs-a-whole-lott), @cdeotte for his enthusiastic sharing, and to everyone in the community who openly shared their work and ideas.

## 📌 Solution Approach

My main idea was to use iterative OOF stacking combined with AutoGluon. The workflow was as follows:

1. **Collect OOF and submission files**: Gather OOF predictions and submissions from high-quality notebooks and my own models, while filtering out those with leakage or inflated CV scores.
2. **Train with AutoGluon**: Use the selected OOF files as features for AutoGluon, train new models, and generate fresh OOF and submission files while recording feature importance.
3. **Filter features**:  Analyze feature importance and select high-quality features for the next round.
4. **Iterate the process**: Repeat steps 1–3, continually adding new models and OOFs to steadily improve ensemble performance.

## 🏆 Model Development Overview

| Version  | Model / Notes             | Private | Public  | Type                                        | Gap vs Best Private |
| -------- | ------------------------- | ------- | ------- | ------------------------------------------- | ------------------- |
| Baseline | AutoGluon baseline        | 0.96677 | 0.96691 | Multi-model (time\_limit=1000)              | -0.00947            |
| V1       | GPU with KNN              | 0.97085 | 0.97111 | Multi-model (best\_quality)                 | -0.00539            |
| V2       | Added pseudo-label data   | 0.97085 | 0.97106 | Multi-model (pseudo-labels)                 | -0.00539            |
| V3       | GBM                       | 0.97082 | 0.97100 | Single model                                | -0.00542            |
| V4       | NN\_TORCH                 | 0.96528 | 0.96559 | Single model                                | -0.01096            |
| V5       | CATBoost                  | 0.96910 | 0.96951 | Single model                                | -0.00714            |
| V6       | XGB                       | 0.97008 | 0.97046 | Single model                                | -0.00616            |
| V7       | FASTAI                    | 0.96505 | 0.96512 | Single model                                | -0.01119            |
| V9       | Dual GPU, removed KNN     | 0.97096 | 0.97113 | Multi-model (best\_quality)                 | -0.00528            |
| V10      | Random Forest (RF)        | 0.96650 | 0.96694 | Single model                                | -0.00974            |
| V11      | Extra Trees (fast, 2 hrs) | 0.96478 | 0.96516 | Single model                                | -0.01146            |
| V12      | Multiple OOF re-training  | 0.97624 | 0.97663 | **OOF / Stacking** (CV ROC AUC: 0.97589771) | 0.00000             |



**Key insight:**

* Single models and the standard AutoGluon multi-model perform reasonably well (≈0.971),
* OOF/Stacking ensembles further improve the score to 0.976, demonstrating that advanced ensembling strategies clearly outperform single models or standard multi-models.

| Version | Model / Notes                          | eval\_metric | CV ROC AUC | Private LB | Public LB |
| ------- | -------------------------------------- | ------------ | ---------- | ---------- | --------- |
| V34     | AutoGluon,50 OOF Files, | 'roc_auc'| 0.977498   | 0.97772 | 0.97806      |
| V41     | AutoGluon, adjusted eval\_metric       | 'pac'        | 0.977486   | 0.97772       | 0.97807|
| V39     | AutoGluon, adjusted eval\_metric       | 'nll'        | 0.977486   | 0.97771       | 0.97805|
| V38     | AutoGluon, adjusted eval\_metric       | 'mcc'        | 0.976961   | 0.97770 | 0.97805      |

* Changing `eval_metric` from `'roc_auc'`  to `'pac'` or `'nll'` has almost no effect on CV or LB scores.
* Using `'mcc'` slightly decreases CV ROC AUC but keeps LB stable.
* Overall, the metric choice did not significantly improve the leaderboard scores.
* Interestingly, `'pac'` gave the best Private LB, which aligned with my intuition at the time, but I didn’t dare to bet on it and stuck with the official `'roc_auc'` metric.


## 🏆 Best Single Model Evolution

| Model / Version  | Notes / Evolution                                                                                                                                                         | CV ROC AUC | Private LB | Public LB |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- | --------- |
| Baseline         | Based on [original notebook](https://www.kaggle.com/code/mahoganybuttstrings/pg-s5e8-single-xgb-cv-0-975782-lb-0-97681?scriptVersionId=254278105) by @mahoganybuttstrings | 0.97578    | 0.97636    | 0.97681   |
| V1 - Single XGB  | Added categorical **3-way combinations** (Cat 3) | 0.97576    | 0.97633    | 0.97681   |
| V2 - Single XGB  | Added numerical **3-way combinations** (Num 3) | 0.97577    | 0.97638    | 0.97676   |
| V9 - Single XGB  | Added high-importance features and applied **TE & CE** on them | 0.97592    | 0.97645    | 0.97684   |
| V10 - Single XGB | Added high-importance features (without extra TE/CE) | 0.97602    | 0.97660    | 0.97702   |
| V13 - Single XGB | Adjusted parameters                                                                                                                                                       | 0.97618    | 0.97683    | 0.97716   |
| V19 - Single XGB | Removed 2 lowest-importance features                                                                                                                                      | 0.97621    | 0.97687    | 0.97717   |
| V28 - Single XGB |Selected highest-correlation features and created product interactions; CV slightly changed, but both private and public LB decreased.                                     | 0.97632    | 0.97646    | 0.97685   |
| V30 - Single XGB | Highest Private LB single model (modified XGB), 10-fold CV                                                                                                                | 0.97639    | 0.97693    | 0.97727   |
| V32 - Single XGB | 40-fold CV                                                                                                                                                                | 0.97637    | 0.97683    | 0.97722   |
| V34 - Single XGB | 20-fold CV                                                                                                                                                                | 0.97644    | 0.97688    | 0.97725   |
```python

# Product interactions
df['duration_div_duration_rank'] = df['duration'] / (df['duration_rank'] + 1e-5)
df['duration_minus_log_duration'] = df['duration'] - df['log_duration']
df['log_duration_mul_age'] = df['log_duration'] * df['age']
df['duration_relmean_minus_std'] = df['duration_rel_mean'] - df['duration_rel_std']
df['duration_per_campaign_mul_age'] = df['duration_per_campaign'] * df['age']

df['duration_mul_age_squared'] = df['duration'] * df['age_squared']
df['balance_log_mul_campaign'] = df['balance_log'] * df['campaign']

```

## 📌 [Feature Engineering Summary](https://www.kaggle.com/code/bestwater/v10-of-pg-s5e8-single-xgb?scriptVersionId=255135348)

Based on the raw data, I designed and constructed around **30 new features** from **duration**, **balance**, **age**, **month**, and **day**, mainly including:

1. **Nonlinear transformations**: log, sqrt, square, cube, and exponential transformations.
2. **Binning features**: quantile-based binning (5/10/20/30 bins) to generate categorical-like features.
3. **Statistical features**: indicators for long/very long duration, absolute balance and squared terms.
4. **Ratio features**: duration-to-campaign ratio, balance-to-duration ratio.
5. **Cyclical features**: sine and cosine encodings for the day variable.
6. **Seasonal features**: indicator for peak business months (March, April, May, September, October, November).
 

## **Overall Conclusion**

* **Feature Interactions**

   * **Pairwise combinations (length = 2):** Provided consistent and noticeable improvements, the most effective interaction strategy so far.
   * **Triple combinations (length = 3):** Tried, but gains were minimal and sometimes introduced noise, leading to limited benefit.

* **Target / Categorical Encoding (TE & CE)**

   * Very effective, among the most impactful techniques for performance improvement.

* **Manually Engineered Features**

    * Added some value, but overall impact was weaker compared to **feature interactions** and **TE/CE**.

* For single models, increasing the number of cross-validation folds often helps improve performance, though too many folds can be counterproductive.
* Best path forward: Focus on strong pairwise interactions, then integrate TE/CE.
* Other engineered features can serve as complementary additions but are not the main drivers of performance.
* This may be partly explained by the nature of Kaggle Playground datasets, which are synthetically generated. In such data, handcrafted feature engineering often has limited effectiveness compared to systematic interactions and encoding methods.



## 🔹 Presets Comparison

| Version | #OOF Files | Presets               | CV ROC AUC | Private LB | Public LB |
| ------- | ---------- | --------------------- | ---------- | ---------- | --------- |
| V46     | 52         | best                  | 0.97752    | 0.97779    | 0.97809   |
| V48     | 52         | experimental\_quality | 0.97751    | 0.97778    | 0.97809   |
| V53     | 48         | experimental\_quality | 0.97754    | 0.97780    | 0.97809   |
| V58     | 48         | best                  | 0.97753    | 0.97779    | 0.97810   |

**Analysis:**

1. Using **experimental\_quality** presets has almost no effect on Public LB, sometimes slightly decreasing it, while slightly improving Private LB.
2. This observation is from the mid-to-late stage of the competition; earlier or most of the time, **experimental\_quality** performed well on both Public and Private LB.
3. The Public LB results made me somewhat skeptical of **experimental\_quality**, which influenced some of my decisions.
4. Due to concerns about overfitting, I mostly chose **best** presets in the later stage.


## AutoGluon Manual CV Experiments

* **Goal:** Test whether manually setting CV folds improves performance when `auto_stack=True`.


| Version | Base | auto\_stack | CV Setup | CV ROC AUC | Private LB | Public LB | 
| ------- | ---- | ----------- | -------- | ---------- | --------- | ---------- |
| **V34** | –    | ✅           | default  | 0.97750    | 0.97772   | 0.97806    |
| **V35** | V34  | ✅           | 10-fold  | 0.97745    | 0.97765   | 0.97801    |
| **V36** | V34  | ✅           | 5-fold   | 0.97745    | 0.97768   | 0.97805    |
| **V42** | –    | ✅           | default  | 0.97745   | 0.97770   | 0.97805    |
| **V44** | V42  | ✅           | 5-fold   | 0.97745    | 0.97769   | 0.97806    |


```python
from sklearn.model_selection import KFold

train.loc[:, "fold"] = -1
split = KFold(n_splits=5, random_state=42, shuffle=True).split(train, train["y"])

for i, (_, val_index) in enumerate(split):
    train.loc[val_index, "fold"] = i

TabularPredictor( 
    groups="fold",      
)
```
* **Summary:**

  * Manual 5-fold or 10-fold CV (V35, V36, V44) **did not improve** scores; CV ROC AUC slightly decreased in some cases.
  * AutoGluon’s built-in stacking (`auto_stack=True`) already handles CV effectively.

## 🏅 Final Submitted Models


| Model / Version  | Notes / Evolution                                                             | CV ROC AUC | Private LB | Public LB | Final Submission |
| ---------------- | ----------------------------------------------------------------------------- | ---------- | ---------- | --------- | ---------------- |
| V103 - AutoGluon | AutoGluon Ensemble on CPU; same workflow as final submitted model; seed = 105 | 0.97761    | 0.97788    | 0.97820   |                  |
| V107 - AutoGluon | Same workflow as V103 but run on GPU T4 x2; seed = 105                        | 0.97761    | 0.97788    | 0.97819   |                  |
| V109 - AutoGluon | Same workflow as V103 but run on GPU T4 x2; seed = 5781                       | 0.97763    | 0.97790    | 0.97821   | ✅ Selected       |



## 
✅ What Worked Well

1. My best Public LB score was also my best Private LB score — It was my third-to-last submission on the final day, where I had only changed a random seed. This shows that a bit of luck also plays a role.
2. In most competitions, single models are hard to outperform carefully designed ensembles. For me, OOF stacking + AutoGluon brought a significant boost.
3. More diverse ensembles tend to boost CV scores.


## 🔹 AutoGluon Blending Experiments

| Version / Blend           | Description / Notes                                              | CV ROC AUC | Private LB | Public LB | Final Submission            |
| ------------------------- | ---------------------------------------------------------------- | ---------- | ---------- | --------- | --------------------------- |
| V70                       | GPU run of Best presets                                          | 0.977634   | 0.97789    | 0.97820   |                             |
| V76                       | GPU run of Extreme presets                                       | 0.977618   | 0.97787    | 0.97817   |                             |
| V79                       | CPU run of Best presets                                          | 0.977610   | 0.97787    | 0.97818   |                             |
| V87                       | GPU run of Best presets                                          | 0.977615   | 0.97789    | 0.97820   |                             |
| V103                      | CPU run of Best presets                                          | 0.977612   | 0.97788    | 0.97820   |                             |
| V108                      | GPU run of Best presets                                          | 0.977634   | 0.97789    | 0.97820   |                             |
| V109                      | GPU run of Best presets                                          | 0.977631   | 0.97790    | 0.97821   |                             |
| Blend Search2             | Average of V70 + V76                                             | 0.977644   | 0.97789    | 0.97819   |                             |
| Blend Search3             | Average of V70 + V76 + V79                                       | 0.977644   | 0.97789    | 0.97819   |                             |
| Blend Search4 V1          | Average of V70 + V76 + V79 + V87                                 | 0.977643   | 0.97789    | 0.97819   |                             |
| Blend Search4 V2          | Average of V70 + V76 + V108 + V109                               | 0.977654   | 0.97789    | 0.97820   | ✅ Selected Final Submission |
| Blend Weighted Search4 V4 | Weighted blend of V70 + V76 + V108 + V109 \[0.18,0.18,0.27,0.36] | 0.977655   | 0.97790    | 0.97820   |                             |


##
⚠️ What Didn’t Work Well

1. Simple and weighted blends improved CV scores but did not increase my LB score.
2. Adding more original training data did not improve the leaderboard score.
 
---

## 💡 Reflections & Learnings

1. This iterative OOF stacking approach lets me record and make use of every attempt, ensuring that all effort contributes to progress and keeps me motivated to learn.
2. Reading and learning from other excellent solutions and open-source notebooks is one of the fastest ways to improve.

---

I regret not being able to run more interesting experiments, and my notebooks ended up a bit messy — but here’s the [notebook](https://www.kaggle.com/code/bestwater/load-of-v109-of-autogluon-playground-s5e8?scriptVersionId=259211334).

> *“Dao can be spoken, but it is not the eternal Dao. Name can be named, but it is not the eternal name.”*
> Learning is a continuous process of accumulation — quantitative change eventually leads to qualitative breakthroughs.
> Enjoy the journey of learning ahead, and keep pushing your boundaries!


Good luck!
