# 16th Place Solution - Ridge Ensembling

Competition: playground-series-s6e3
Rank: #16
Source: https://www.kaggle.com/c/playground-series-s6e3/writeups/16th-place-solution-ridge-ensembling

I'm happy to share my 16th place solution write-up for this Playground competition. Huge thanks to Kaggle for hosting this competition. This was my first time experimenting with ensembling, so I wanted to document my approach.

---

# Overview

My final solution was a **34-model stacked ensemble** with a **Ridge meta-learner** augmented with **polynomial interaction features**.

No exotic architectures — just systematic ensembling done carefully.

---

# Base Models

I didn't have enormous compute or time to train hundreds of models from scratch, so my approach was to improve existing public notebooks and add my own variants.

### Model Set Included

- LightGBM — multiple models with 20-seed averaging and different parameters  
- XGBoost — multiple models with 20-seed averaging and different parameters  
- CatBoost — multiple parameter and seed variants  
- LightGBM DART — separate diversity variant  
- RealMLP (multiple variants) — different feature representations  
- TabM (5 variants) — trained with different configurations  
- TabTransformer — improved public model  
- Trompt — improved public model  
- PyTabkit Ensemble — ensemble of RealMLP, XGB, CatBoost and LGBM  
- cdeotte GPU NNs — nn_v312, v102, tepair_logit3  
- BARTZ and GNN — from starter notebooks  

All models were trained using:

`5-Fold StratifiedKFold CV`

---

# Feature Engineering

For my own trained models I built a reusable feature engineering module covering:

### Numeric & Statistical Features
- Frequency encoding of numeric columns  
- Arithmetic features  
  - charges deviation  
  - monthly-to-total ratio  
  - average monthly charges  
- Service count features  

### ORIG Dataset Mapping
- ORIG_proba mappings from the original Telco dataset  
- Applied to all categorical and numeric columns  

### Distribution Features
- Percentile ranks vs churner/non-churner distributions  
- Quantile distance features at q25, q50, q75  

### Digit-Level Features
For:
- tenure  
- MonthlyCharges  
- TotalCharges  

Generated:
- first digit  
- last digit  
- mod10  
- mod12  
- round number flags  

### Categorical Interactions
- N-gram categorical interactions  
- Bigrams and trigrams of top contract/service columns  

### Linear Embeddings
- Min-max projections  
- Rank-based projections  
- Z-score projections  

### Periodic Embeddings
- Sin/Cos embeddings at multiple frequencies  
- Captures:
  - 12-month tenure cycles  
  - round charge patterns  

### Meta Features
- I Trained 6 NN Models and added their OOF as Feature Input for my Boosting Models.

---

# Ensembling

I experimented with both Hill Climbing and Ridge meta-learning.

With 34 models, Hill Climbing converged too quickly and produced a very simple 3-model equal-weight blend. Ridge generalized better across the full model set because it could assign continuous weights to all 34 models simultaneously rather than greedily selecting a subset.

### Ridge Meta-Learner Performance

- CV: 0.91976  
- Private LB: ~0.91828

---

# Adding Polynomial Features to the Meta-Learner

This was the single biggest improvement in my pipeline.

Instead of giving Ridge just the 34 raw OOF prediction columns, I expanded them with all pairwise interaction terms:

- Degree = 2  
- interaction_only = True  
- include_bias = False  

**Feature Expansion**
- 34 → 561 features  
- (34 original + 527 interactions)

### Intuition

Plain Ridge learns weighted averages.  
Polynomial Ridge learns complementary model relationships.

Examples:
- lgbm × catboost  
- realmlp × tabm  

```
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly  = poly.transform(x_test)

ridge_poly = Ridge(alpha=best_poly_alpha)
ridge_poly.fit(StandardScaler().fit_transform(x_train_poly), true)
```

# Alpha Tuning

- Plain Ridge: ~75
- Polynomial Ridge: ~235

Improvement

- CV: 0.91976 → 0.91984
- Private LB: 0.91829 → 0.91837

---

Final Ridge + Polynomial Performance

Final CV: 0.91984
Private LB: 0.91837

---

# Isotonic Calibration

I also experimented with isotonic calibration applied on top of the Ridge + Polynomial predictions.

This fits a monotonic mapping from raw OOF predictions to true labels, adjusting probability values without changing their ranking order (AUC preserved).
```
from sklearn.isotonic import IsotonicRegression

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof_meta, true)
calibrated_test_preds = iso.predict(raw_test_preds)
```

On public LB this submission scored slightly lower than Ridge+Poly alone, but I selected it as my second final submission.

---

Acknowledgements

Thanks to the public notebook authors whose work formed the foundation of this ensemble, Including @cdeotte, @yekenot, @include4eto, @blamerx, @badalkrsharma, Also not limited to them!
