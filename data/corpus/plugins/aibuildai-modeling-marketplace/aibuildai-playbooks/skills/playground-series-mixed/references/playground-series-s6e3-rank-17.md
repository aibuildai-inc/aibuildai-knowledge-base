# # 17th Place Solution

Competition: playground-series-s6e3
Rank: #17
Source: https://www.kaggle.com/c/playground-series-s6e3/writeups/17th-place-solution

***

## Overview

Our final solution is a large ensemble of 138 models with greedy hill climbing, achieving a final OOF AUC of 0.91974.

---

## What Gave the Big Punches

The hill climbing results tell the story clearly:

| Model | HC Weight | Key Feature Group |
|---|---|---|
| xgb_31 | **+0.34** | CATNUM (groupby cat→agg num, zero TE) + DIGITQUAD |
| realmlp_90 | **+0.18** | RealMLP with DIST/QDIST vs orig distribution |
| xgb_33 | **+0.17** | Multi-scale clustering (KMeans+GMM+DBSCAN) |
| tabm_61 | **+0.15** | FREQ2/3/4 combinations (zero TE) |
| xgb_123 | **+0.10** | Periodic embeddings sin/cos (multi-period) |
| xgb_112 | **+0.10** | Boosting residuals (Ridge→RealMLP residual→XGB) |

The pattern is unmistakable: **the highest-weight models use zero target encoding**. Models #31, #33, #61, #123 all avoid in-fold TE entirely.

### Feature Engineering Insights

**Scaling to an 800+ Feature Pool (The Diversity Buffet):**
To feed 138 different models and ensure maximum orthogonality, we programmatically generated a massive pool of over 800 zero-leakage features. We achieved this extreme scale through combinatorial expansion. For example, crossing 11 high-IV categorical columns with 3 numerical columns across 8 different statistical aggregations (mean, std, median, q25, q75, IQR, skew, CoV) instantly creates hundreds of features. Expanding this to 2-way categorical pairs exponentially grows the matrix. While throwing all 800+ features into a single GBDT caused overfitting, this massive pool served as a "feature buffet". We trained different models on different subsets of this pool, forcing them to learn completely different perspectives of the data.

**CATNUM (groupby cat → agg numeric from orig IBM dataset):**
The single most powerful feature family. `groupby(Contract)['tenure'].mean()` and its siblings (std, min, max, median, Q10–Q90, IQR) computed from the IBM dataset provide rich group-level statistics with **zero leakage**: no in-fold TE needed. The key insight: you're not encoding the target, you're encoding the numeric distribution within each categorical segment. Extended to pairwise CATNUM (`groupby(Contract, InternetService)['TotalCharges'].mean()`) for another boost.

**Periodic Embeddings:**
Inspired by the competition writeup noting periodic embeddings bypass spectral bias. For tenure: periods of 1, 6, 12, 24, 36 months (annual cycles). For MonthlyCharges: periods of 10, 25, 50, 100 (billing round numbers). This transforms smooth numerics into multi-frequency representations that tree models can split on cleanly.

**Distribution Distance Features (DIST/QDIST):**
For each row, compute its percentile rank against the IBM original's churn vs non-churn distribution: `pctrank(TotalCharges | churn=1) - pctrank(TotalCharges | churn=0)`. When this gap is large and positive, the customer's billing pattern aligns strongly with churners.

**FREQ2/3/4:**
Joint frequency of categorical combinations: C(16,2)=120 pairs, C(16,3)=560 triples, C(10,4)=210 quads. No target involved, just how rare or common each combination is. Rare combinations often signal unusual customer profiles with distinctive churn rates.

---

## Model Architecture

We trained 138 models across diverse architectures and feature sets:

**Tree-based (XGB, LGB, CatBoost, RF, ExtraTrees, HistGBT):**
- Standard depth (5-6): Ridge→XGB stacking, CATNUM+DIGITQUAD, clustering features
- Low depth stumps (max_depth=2): "Raw is Law" philosophy, OHE + minimal FE
- Balanced depth (max_depth=4): Periodic embeddings + cuML TE
- DART and GOSS variants of LightGBM
- CatBoost with native ordered TE (zero external TE)

**Neural Networks:**
- RealMLP (PyTabKit) with n_ensembles=8: multiple feature configurations
- PyTorch MLP with learnable periodic embeddings + entity embeddings
- PyTorch FREQ-only MLP: purely frequency signal, zero TE
- ResNet, TabNet, FTTransformer, TABM, DeepFM, MLP-PLR

**Stacking architectures:**
- Ridge→XGB (proven best single: 0.91903)
- Ridge→RealMLP→XGB (3-stage)
- LGB→XGB, ElasticNet→XGB
- Knowledge distillation: RealMLP teacher → XGB student (soft labels, T=0.5, α=0.7)
- Boosting residuals: Ridge predicts → RealMLP predicts Ridge's error → XGB combines both

**Novel feature spaces explored:**
- Epidemiology features: Odds Ratio, Risk Ratio, Attributable Risk, NNT, PAR, Mantel-Haenszel adjusted RR
- Within-group percentile rank (position of row in its group, not group stats)
- Bayesian TE: Beta posterior mean + variance + credible interval width
- NMF embeddings on cat co-occurrence matrix (rank-8)
- Tensor CP decomposition: 3-way tensor of top 3 cats (rank-5)
- Chi-squared residuals per cat value
- Word2Vec on categorical sequences

---

## What Didn't Work

- Polynomial feature interactions (GP confirmed features are largely independent)
- Complex mathematical FE (Wasserstein distance, Hellinger divergence): negative or near-zero ensemble weights
- Extremely large feature sets (1200+ features): tended to overfit on this synthetic data when used entirely in a single model, hence the need for feature subsampling.
- Pseudo-labeling at high thresholds: marginal gain, not worth the complexity

---

## Conclusion

The more orthogonal your ensemble members, the more each iteration of hill climbing can find something genuinely new to add.

With 138 models across 15+ architecture families and 20+ distinct feature engineering strategies, the ensemble had enough diversity to squeeze meaningful signal from every orthogonal direction.
