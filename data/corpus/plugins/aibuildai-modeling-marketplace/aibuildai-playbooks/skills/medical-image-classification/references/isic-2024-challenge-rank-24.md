# Private 24th / Public 1st Approach

Competition: isic-2024-challenge
Rank: #24
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532564

Well that was an unpleasant shakedown for me for sure. It was my second time getting kicked out of the gold zone so I'm kind of used to it. 💀 I didn't use image models for a long time, so this was a good opportunity to remember and getting sync'd with the state-of-the-art. I want to congratulate all the winners and thank the organizers for the competition. This was a very educational process.

Links:
- **My best private tabular-only submission (181/161):** https://www.kaggle.com/code/nlztrk/isic-only-tabular-solution-181-161
- **My best private solo stack submission (188/170):** https://www.kaggle.com/nlztrk/best-solo-stack-submission-4th-version

---

### Solution Wrap-Up
I tried many different things and also tried involve most of them in my final selected submission blends. The most notable ones among them were the following:

---

#### Image Backbones
- Resnet18 with using one-hot-encoded sex, tile type and site information
- EfficientNetB0 with using one-hot-encoded sex, tile type and site information
- Dreamsim VitB-16 Embeddings + LinearSVC

---

#### Image Features
I also tried manual image features but they lost their effect when I started to add NN OOF predictions:

**Centered Lesion Ratio:** Grayscaled + thresholded all the images, then I calculated the proportion of pixels with a mask value of 1 that are located within the central 50% section of the image.
**HSV Features:** Calculated hue-saturation 2D matrices from all the images and calculated distributional statistics on them.

---

#### Tabular Features
I started to use the starter features from the earliest notebooks. I didn't use any categorical features directly. The biggest improvements came from the group aggregations. They were simply:

**Group by:** (patient_id) and (patient_id, tbp_lv_location)
**Aggregations:** max, mean, min, median, skew, std, nunique, sum, z-score, ecdf

---

#### The Model
The model was only a CatBoost (I mean 5 CatBoost models blended for a 5-fold CV) with the following parameters:

```
cb_params = {
    "random_state": 42,
    "iterations": 4500,
    "learning_rate": 0.015,
    "depth": 3,
    "verbose": 100,
    "use_best_model": True,
    "auto_class_weights": "Balanced",
    "eval_metric": "AUC:use_weights=false",
    "task_type": "GPU",
    "bootstrap_type": "Poisson",
    "l2_leaf_reg": 200.,
    "border_count": 254,
}
```

---

#### CV and Feature Selection
Used a naive stratified group k-fold like:

```
sgkf = StratifiedGroupKFold(n_splits=5)
df_train["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(sgkf.split(
    df_train,
    df_train["target"],
    groups=df_train["patient_id"]
)):
    df_train.loc[val_idx, "fold"] = idx
```

Tried many feature selection schemes like SHAP, LOFO, Top-K importance. The improvements for both CV and LB from the top-k one. CatBoost seems really good at it.
