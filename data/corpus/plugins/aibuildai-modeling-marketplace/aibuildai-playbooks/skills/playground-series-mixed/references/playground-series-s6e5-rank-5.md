# 5th place solution  — a 99-model logit stack

Competition: playground-series-s6e5
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s6e5/writeups/5th-place-solution-a-99-model-logit-stack

Thanks to the Kaggle team for a challenging and enjoyable Playground round, and to everyone who shared notebooks and ideas. For me this was a competition where model quality mattered less than how carefully you blended and which submission you trusted. Below is a summary of what I built and what I learned.

---

## The final model

My chosen submission was an sklearn LogisticRegression meta-model stacked on 99 diverse base models. The construction is simple:
- One feature per base model: each model's out-of-fold probability is converted to a logit, `log(p / (1 − p))`, clipped at ±30.
- The meta-model is fit fold-wise on those 99 columns to produce an honest OOF, then refit on the full data and averaged over 5 seeds for the test prediction.
I first saw this logit-stacking idea from @cdeotte in S6E3 and reused it in S5E4; it has been reliable across competitions.
Two hyperparameter choices, both following directly from the metric being AUC:
- class_weight=None— AUC is rank-based, so class balancing only distorts the probabilities without improving the ranking.
- C=1.0, L2 penalty — light regularization was enough; with one clean logit per base model there was no need for the aggressive shrinkage a wider feature set would require.

No post-processing was applied — no argmax, no offset, no calibration. Because AUC depends only on the ordering of predictions, absolute calibration does not affect the score, so the submission is the raw averaged probability.

### OOF diagnostics

| Metric | Value |
| --- | --- |
| OOF AUC | 0.95536 |
| Log-loss | 0.2119 |
| Brier | 0.0660 |
| Per-fold AUC (mean) | 0.95537 |
| Per-fold AUC (std) | 0.00081 |
| Per-fold AUC (range) | 0.0021 |

One diagnostic stood out: the OOF calibration slope was 0.082, indicating the meta-model was strongly overconfident in absolute probability terms. This does not affect AUC, since ranking is unchanged, but I read it as a signal that the meta-model was working close to its limit — and it informed the conservative submission choice described at the end.

The largest-magnitude meta coefficients came from a mix of the strongest single models and a few ensembles, each carrying rank information no other base supplied: a single XGBoost, a 4-model Ridge stacker, a 7-LGBM rank-mean ensemble (with a negative coefficient, acting as a correction term), the pseudo-label LightGBMs, and the adaptive-learning-rate LightGBM.

---

## The base models

The values below are OOF AUC ranges for each family.

| Family | OOF AUC range | Members |
| --- | --- | --- |
| **Gradient-boosted trees** (backbone) | 0.946 – 0.954 | LightGBM in many variants (Optuna-tuned, DART, monotonic, native-categorical, frequency-encoded, target-encoded, quantile-binned, pseudo-labeled / self-distilled, adaptive cosine LR, hazard/survival reframing, per-year and per-race specialists); XGBoost (deep, low-colsample, heavy-bagging, stint-FE, Tweedie); CatBoost (Optuna, numeric-as-categorical); HistGradientBoosting; YDF GBT |
| **MLP** | 0.950 – 0.954 | RealMLP (pytabkit) — 5-seed PyTorch and Keras ports, an "original-data-as-stats" variant, several FE ablations; multi-task NN with auxiliary losses; categorical-embedding MLP |
| **Attention / token tabular** | 0.944 – 0.952 | FT-Transformer, SAINT, TabM, TabTransformer, Trompt, AutoInt, GateNet |
| **Other neural** | 0.918 – 0.948 | ResNet, GANDALF, VSN, DANet, SELU self-normalizing net, RFF-Kernel net, DAE + transfer learning |
| **Sequence** | 0.941 – 0.945 | GRU, BiLSTM over the lap sequence, causal TCN fed survival-aware hazard channels |
| **Graph** | 0.940 – 0.946 | GraphSAGE, GNN |
| **Factorization machines** | 0.918 – 0.936 | Deep FFM, xLearn FFM |
| **Kernel / other** | 0.925 – 0.948 | Nyström kernel, exact-kernel subsampled RBF SVM, Random Forest, ExtraTrees, classical Bagging trees, NGBoost, EBM, BART, logistic GAM, LDA/QDA, TE-logistic-regression |
| **AutoML / meta** | 0.951 – 0.955 | AutoGluon (LGBM-only and NN-only), H2O AutoML |

The pattern: trees set the ceiling, MLPs matched them, and a long tail of weaker but orthogonal models (factorization machines, kernels, sequence and graph nets) did the real decorrelation work.

---

## Diversity

This was a high-correlation problem. Across all 4,851 model pairs, the mean pairwise Spearman correlation was 0.950 (Pearson 0.957, top-K Jaccard 0.82). When predictions correlate that tightly, AUC stacking depends on the tails of the diversity distribution rather than the average.

The per-model redundancy ranking (each model's mean Spearman against the other 98) sorted almost perfectly by family:

| Group | Models | Note |
| --- | --- | --- |
| **Most redundant** (mean Spearman ≈ 0.965–0.968) | the LightGBM cluster — pseudo-label / AutoGluon / frequency-FE / native-categorical LGBMs | Several were near-twins (pairwise Spearman > 0.998). Keeping all of them was insurance, not new signal. |
| **Most diverse** (mean Spearman ≈ 0.85–0.91) | Deep FFM (0.852), BART (0.851), Nyström kernel (0.884), the GNN, GRU, EBM, xLearn FFM | Lowest standalone AUC, highest contribution per slot. |

The point worth stressing for anyone stacking on a saturated metric: a 0.918-AUC Deep FFM was worth more to the blend than a fourth 0.953 LightGBM. The marginal LightGBM adds ranking information the stacker already has; the FFM adds a direction nothing else covers. The weakest models earned their place precisely because they were weak in an uncorrelated way.

I also ran an OOF-versus-test consistency check on every base (distribution drift, q90 gap). Only three models flagged — Deep FFM, GRU, and the LNN — all on tail (q90) drift, none on the mean. The most diverse models are also the ones whose score distributions wander most between train and test. That is the honest cost of diversity, and another reason I kept the meta-model regularized.

---

## Features

The dataset was clean, with no missing values, so I spent more time on disciplined feature-engineering ablations than on any single model. The one robustly positive signal was appending the original F1-strategy dataset as extra training rows. Everything else was marginal or neutral:

| Feature idea | Effect |
| --- | --- |
| Original data as extra training rows | The reliable win |
| Arithmetic interactions | Weak positive (~+0.0005), kept in a few bases |
| Target encoding (year-aware, Race×Year, Race×Compound, fold-fitted) | Small, consistent gains |
| Frequency encoding (log1p, fit on train ∪ original) | Useful baseline |
| Generator-fingerprint FE (digit features, snap-to-grid, anchor TE, TF-IDF) | No meaningful impact here |
| Lag / sequence features | Limited, used mainly in the sequence models |

For diversity I deliberately built FE variants of the same model: numeric-as-categorical duplication, quantile-rank encoding, triple-cross target encoding, a "wrong-target" LightGBM, KNN-target features (FAISS), and DAE/SCARF self-supervised embeddings. These rarely beat the baseline on AUC, but decorrelation — not standalone accuracy — was the goal, and they delivered it.

Two post-processing ideas were built and validated on OOF before any use on the test set: a near-duplicate neighbor blend (kNN label transfer was ~99% accurate on the closest 1% of test rows, but the OOF gate did not clear convincingly) and a non-linear residual LightGBM on top of the stack logits (no honest OOF re-ranking gain). 

---

## The submission that scored 0.95505

After the deadline I revisited how the two strongest blends were combined. My two best candidates were the logit stack (LS) and a hill-climbing ensemble (HC) built on the identical set of 132 models. They rank almost identically (Spearman ≈ 0.996) but sit on different probability scales: LS is calibrated near the base rate, while HC's scores are compressed toward the middle.

Because the two scales differ, a plain average of the probabilities is dominated by whichever model has the wider spread. Averaging the two models' ranks instead puts them on a common ordering scale, so each contributes equally — which is exactly what AUC rewards. The rank average scored 0.95505 private, ahead of the raw probability average (0.95503) and enough to top the private leaderboard.

It is a one-line change and a good default whenever you blend models that may be on different scales:

```python
from scipy.stats import rankdata
rank_avg = 0.5 * (rankdata(hc) / len(hc) + rankdata(ls) / len(ls))
```

---

## The conservative submission (1st public → 5th private)

Hill-climbing on the OOF gave a higher CV, and the corresponding HC submissions later scored private 0.95501–0.95502, which would have placed 2nd or 3rd. My best post-deadline logit-stack variant (public-equivalent 0.95571) also reached private 0.95502. So HC would have moved me up.

I did not pick it, on purpose. The CV-to-public gap on the HC selections was wider than on the flat 99-model stack, and when the entire leaderboard sits inside 0.0001 AUC, a wider gap is usually exactly what gets punished on the 80% private set. A flat L2 blend over all 99 bases is much harder to overfit than a greedily selected subset, so I chose the lower-variance option.

It cost me a few places. The real lesson is that the better hedge would have been to split my two final submissions across both philosophies — "trust the conservative blend" and "trust hill-climbing" — rather than spending both on conservative variants. That is the one thing I would change.

---

## Acknowledgements
This solution stood on a lot of shared work. With thanks to:
- @cdeotte — for the logit-stacking meta-model approach;
- @yekenot — for the RealMLP (pytabkit) approach and feature ideas that became my strongest non-GBM line;
- @mikhailnaumov — for feature-engineering ideas and several base models (YDF, HGB, CatBoost, XGB);
- @masayakawamata — for the LightGBM GOSS variant;
- @include4eto — for the pseudo-labelling pipelines;
- @rohit8527kmr7518 — for the CatBoost configuration I re-implemented in LightGBM;
- @analyticaobscura — for the within-stint lag/rolling feature ideas.

Apologies to anyone I have missed — this stack drew on many public notebooks, and the diversity it gained from them is exactly what made the final blend competitive.

Happy to share my codes and answer questions in the comments.
Happy coding!
