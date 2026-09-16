# 8th Place - Ensemble and TrustCV

Competition: playground-series-s6e2
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/8th-place-ensemble-and-trustcv

First of all, thanks to Kaggle for organizing this amazing competition, and congratulations to all the winners! 👏

I would also like to extend my gratitude to:

* @omidbaghchehsaraei
* @include4eto
* @mikhailnaumov
* @abisheksrivastav

Their public kernels were extremely insightful and helped shape my final solution.

---

#  Phase 1 — XGBoost Exploration (Core Focus)

I started the competition focusing primarily on **XGBoost**. My goal was to build a diverse set of models using:

* Different feature combinations
* Multiple target encoding strategies
* Variations in fold setup
* DMatrix baseline model

###  Experiment Results

| Exp Name          | CV      |
| ----------------- | ------- |
| xgb-5fold-3TRG_FE | 0.95533 |
| xgb-5fold-2TRG    | 0.95532 |
| xgb-5fold-TRG     | 0.95532 |
| xgb-5fold-ORG     | 0.95535 |
| xgb-dmatrix-5fold | 0.95522 |



### 🔎 Explanation

* **3TRG / 2TRG / TRG** → Number of target encodings per fold
* **ORG** → Aggregate features engineered from original dataset
* **DMatrix** → Baseline XGB trained only on competition dataset

The performance difference was marginal (~0.0001), which indicated:

> Feature engineering mattered more than hyperparameter tuning at this stage.

I reused and improved my previous month's target encoding framework.

---

#  Phase 2 — Tabular Deep Learning (TabM)

Since **pyTab models** were performing extremely well in this competition, I decided to experiment with them.

However, due to my ongoing mid-semester exams, I couldn’t afford extremely long training runs. So instead of heavier models, I opted for:

> **TabM** (fast and competitive)

###  Results

| Exp Name       | CV      |
| -------------- | ------- |
| TabM-Org-5fold | 0.95532 |
| TabM-5fold     | 0.9547  |



TabM was competitive with XGB but did not clearly outperform it in my experiments.

---

#  Phase 3 — Neural Networks (Failed Attempt 😅)

I experimented with multiple neural network architectures, including:

* Embedding layers
* Learning rate scheduling
* Exponential decay
* Early stopping

Unfortunately, none of them surpassed tree-based models.

###  Results

| Name               | CV      |
| ------------------ | ------- |
| NN_lr1e-3_pembd_es | 0.95473 |
| NN_lr1e-3_pembd    | 0.95472 |
| NN_lr1e-3_NoExpDec | 0.95362 |
| NN_Improved        | 0.95393 |
| NN_Pytorch_5       | 0.95313 |



Key Takeaway:

> For this dataset, gradient boosting was simply stronger and more stable than deep neural networks.

After this phase, I temporarily moved away from pure neural nets.

---

#  Phase 4 — Stacking & Ensembling

Initially, I tried **Logistic Regression stacking**, but it did not give significant improvements.

In the final two days, I shifted towards:

>  Neural Network Stacking

###  Results

| Name    | Public  | Private |
| ------- | ------- | ------- |
| Best LR | 0.95388 | 0.95530 |
| Best NN | 0.95394 | 0.95533 |

The neural stacking slightly outperformed logistic stacking.
Here's the detailed breakdown of the neural stacking experiments


| Name    | CV  | 
| ------- | ------- | 
| NN_stacking_1 | 0.95538 | 
| NN_Stacking_2 | 0.95569 | 

The drastic change in CV occurred due to the following change in the training setup:
* Introduced Exponential Decay
* Introduced Early Stopping with a patience of 7



---

#  Models Used in Stacking

I used predictions from the following public kernels:

* [RealMLP+Temperature](https://www.kaggle.com/code/abisheksrivastav/realmlp-temperatures)
* [RealMLP](https://www.kaggle.com/code/omidbaghchehsaraei/the-best-solo-model-so-far-realmlp-lb-0-95397) 
* [Single XGB](https://www.kaggle.com/code/include4eto/single-xgb-cudf-pseudo-labels-cv-0-95573) 
* [Resnet](https://www.kaggle.com/code/omidbaghchehsaraei/resnet-predicting-heart-disease-lb-0-95385) 
* [TabM1](https://www.kaggle.com/code/include4eto/tabm-cv-0-95565) 
* [TabM2](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-predicting-heart-disease-cv-0-95553)

These models brought diversity in architecture and inductive bias.

---

#  Final Blend

At the very end, I couldn’t resist incorporating **@mikhailnaumov’s solution**.

Since OOF predictions weren’t available in the public kernel and I was exhausted by that point, I opted for blending instead of stacking.

### Final Strategy:

```
Final Prediction =
    0.5 × My NN Stacking
  + 0.5 × Mikhail’s Ensemble
```

This gave me a slight boost and stabilized private LB performance.

---

#  Final Thoughts

* XGBoost was extremely strong and stable throughout the competition.
* Target encoding diversity was critical.
* TabM was surprisingly competitive.
* Neural networks did not shine for this dataset.
* Ensembling provided the final edge.

---
