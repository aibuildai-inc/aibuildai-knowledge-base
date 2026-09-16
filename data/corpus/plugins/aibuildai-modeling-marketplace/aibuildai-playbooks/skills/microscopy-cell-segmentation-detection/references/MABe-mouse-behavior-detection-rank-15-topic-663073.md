# 15th place solution

Competition: MABe-mouse-behavior-detection
Rank: #15
Source: https://www.kaggle.com/c/MABe-mouse-behavior-detection/writeups/15th-place-solution

I built on this excellent baseline and made the following changes:

https://www.kaggle.com/code/ravaghi/social-action-recognition-in-mice-xgboost

Thank you @ravaghi.


* **Excluded non-test datasets from training**: removed **CalMS21**, **MABe22**, and **CRIM13** since they do not appear in the test set. (This made CV correlate better with LB.)
* **Trained models per lab**: trained **XGBoost** and **CatBoost** separately for each lab. (LightGBM performed worse for some reason.)
* **Removed AdaptableSnail 25 fps from training**.
* **Bidirectional EMA smoothing**: applied EMA in both forward and backward directions, and searched for the best smoothing timescale.
* **Ego-centric features**: used **body_center** as the origin (for labs without body_center, I used the mean of the annotation coordinates), rotated coordinates so the motion direction aligns with the **positive Y-axis**, and generated ego-centric features.
* **Feature pruning**: removed features whose importance was **zero in both CatBoost and XGBoost**.
* **Fair tie-breaking when multiple classes exceed thresholds**: `argmax()` is not fair under class imbalance, so I used a Z-score–based margin:

  * `z = (prob - threshold) / sigma`, where **sigma is the standard deviation of OOF predictions**.
  * Selected the class with the highest z-score.
* **Ensembling**: averaged **EMA-smoothed probabilities** across models and used the **average threshold** for decisions; when multiple classes competed, selected the one with the highest **mean Z-score across models**.

### What didn’t help (for me)

* Post-processing after inference: filling with `max_gap` and removing spikes via `min_duration`.
* Left-right flip augmentation for GBDT (**CV↑ but LB↓**).
* Handling AdaptableSnail 25 fps based on discussion ideas (scaling frames by **1.2×**, removing `rear`, `submit`, `approach`).
