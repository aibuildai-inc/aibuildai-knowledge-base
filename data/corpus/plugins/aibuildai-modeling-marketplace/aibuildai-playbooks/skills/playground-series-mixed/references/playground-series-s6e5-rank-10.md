# Stacking stacked predictions

Competition: playground-series-s6e5
Rank: #10
Source: https://www.kaggle.com/c/playground-series-s6e5/writeups/stacking-stacked-predictions

# 10th Place Solution – Predicting F1 Pit Stops Playground Competition

## Overview

I finished **10th place** in the Predicting F1 Pit Stops Playground competition.

My solution was built around a simple philosophy:

> Generate as many diverse prediction sources as possible, treat OOF predictions as reusable assets, and continuously search for better ensemble architectures.

While I experimented extensively with feature engineering and model development, the biggest gains ultimately came from stacking and ensemble design.

One particularly interesting outcome was that, in hindsight, one of my unselected submissions would have ranked approximately **5th place on the private leaderboard**. Like many Kaggle competitors, I selected submissions based on a combination of OOF performance and public leaderboard score. The competition ended up being a useful reminder that neither metric tells the full story.

---

# Competition Summary

| Metric                   | Result                    |
| ------------------------ | ------------------------- |
| Final Rank               | 10th Place                |
| Primary Metric           | ROC-AUC                   |
| Main Stacker             | cuML Logistic Regression  |
| Best CV Ensemble         | 221 Models                |
| Best Submission Ensemble | 129 Models                |


---

# Dataset Impressions

As an F1 fan, I was particularly excited for this competition.

However, the dataset turned out to be less enjoyable than expected due to inconsistencies in the underlying source data. Several variables contained irregularities that made modeling more challenging and occasionally reduced confidence in leaderboard movements.

As a result, I relied heavily on cross-validation and OOF analysis throughout the competition.

Despite the data quality issues, the competition provided an excellent environment for experimenting with advanced ensembling techniques.

---

# Feature Engineering

I explored:

Frequency-based encodings
Target encodings
Pairwise interaction features
Alternative categorical representations of numerical variables
Digit-level transformations for selected columns

For interaction features, I primarily focused on combinations involving variables such as Driver, Compound, Race, LapNumber, Stint, TyreLife, Position, and RaceProgress. Examples include Driver × Compound, Race × LapNumber, and Stint × TyreLife.

Target encoding was performed using RAPIDS cuML's TargetEncoder, allowing efficient experimentation with both single-column and pairwise encodings.

While no individual feature engineering technique dramatically transformed leaderboard performance, many of these features contributed incremental gains across different model families. Their collective value became most apparent once incorporated into the larger ensemble framework.

---

# Model Diversity

A major focus of my solution was maximizing model diversity.

Rather than searching for a single dominant model, I attempted to generate many strong yet different prediction sources.

## Model Families Tested

| Family                 |
| ---------------------- |
| Logistic Regression    |
| CatBoost               |
| XGBoost                |
| LightGBM             |
| TABM                   |
| RealMLP                |
| Neural Networks        |
| Graph Neural Networks  |
| Bartz                  |
| TABPFN                |
| AutoGluon Models       |

Many models were only moderately competitive individually but contributed positively once incorporated into the ensemble.

---

# OOF Predictions as First-Class Assets

One principle that guided my entire workflow was:

> OOF predictions are valuable reusable assets, not temporary by-products.

Every trained model produced:

* OOF predictions
* Test predictions

These prediction files became building blocks for future experiments.

Rather than repeatedly training models, I spent considerable time analyzing:

* Correlations between predictions
* Ensemble performance
* Alternative stackers
* Meta-model architectures

This approach allowed me to iterate rapidly on ensemble designs. At some point I stopped training models and started training predictions. OOF files became my actual dataset.

---

# Multi-Level Stacking Architecture

The strongest aspect of my solution was the ensemble architecture.

The workflow eventually evolved into a hierarchical stacking framework.

```text
Base Models
      ↓
Level-1 Stacks
      ↓
Community/Public Ensembles
      ↓
Level-2 Stacks
      ↓
cuML Logistic Regression
      ↓
Final Submission
```

Importantly, many inputs to the final stacker were themselves stacked ensembles.

As a result, the final solution effectively became a multi-level stacking system rather than a traditional single-layer stack.

---

# Public Notebooks Were Extremely Valuable

I incorporated a number of public notebook predictions into the stacking framework.

Rather than blending them blindly, they were evaluated through OOF validation and treated as additional prediction sources.

Interestingly, several community solutions ended up receiving some of the largest weights in the final stacker.


This reinforced the value of leveraging diverse community solutions when building large ensembles.

---

# Stacking Experiments

I experimented with multiple meta-models.

## Stackers Tested

| Stacker                  |
| ------------------------ |
| cuML Logistic Regression |
| CatBoost                 |
| Ridge Regression         |
| Hill Climbing Ensemble   |

The most consistent performer across my validation framework was:

**cuML Logistic Regression**

Its combination of speed, stability, and ability to handle large numbers of correlated prediction inputs made it an ideal choice for the final ensemble.

---

# Best CV vs Best Submission

One of the most interesting lessons from the competition came from comparing my best validation solution to my best competition submission.

## Highest CV Ensemble

| Metric | Value    |
| ------ | -------- |
| Models | 221      |
| CV AUC | 0.955630 |

## Final Selected Submission

| Metric | Value    |
| ------ | -------- |
| Models | 129      |
| CV AUC | 0.955294 |

Despite having a lower CV score, the smaller ensemble generalized significantly better on the private leaderboard.

---

# What Probably Went Wrong

My strongest CV solution relied heavily on a large collection of AutoGluon-generated predictions and multiple generations of stacked models.

In hindsight, this ensemble likely crossed the point of diminishing returns.

Some evidence for this came from the final logistic regression weights.

### Largest Negative Weights

| Model                        | Weight |
| ---------------------------- | -----: |
| AUTOGLUTON_LightGBMXT_BAG_L3 | -0.535 |
| RealMLP4_7                   | -0.202 |
| AUTOGLUTON_LightGBM_BAG_L2   | -0.158 |

The stacker appeared to spend significant effort correcting redundant or potentially overfit signals.

While the larger ensemble achieved the strongest validation score, it likely sacrificed some generalization ability.

---

# What I Did Not Do

One approach I intentionally avoided was blind blending.

Every ensemble decision was supported by OOF validation.

While leaderboard-driven blending occasionally succeeds, I preferred maintaining a validation-first workflow throughout the competition.

---

# Lessons Learned

This competition reinforced several lessons that I consider broadly applicable to tabular Kaggle competitions:

1. Diverse prediction sources are often more valuable than marginal improvements to a single model.
2. OOF predictions should be treated as reusable assets.
3. Multi-level stacking can continue extracting signal after individual models plateau.
4. Larger ensembles are not automatically better ensembles.
5. Public leaderboard performance can sometimes discourage solutions that generalize better.
6. Community predictions can be extremely valuable when incorporated thoughtfully.
7. Validation remains essential, especially when working with noisy datasets.

---

# Closing Thoughts

Although the dataset itself was not as enjoyable as I initially hoped due to various inconsistencies, the competition turned into one of my most educational experiences in large-scale ensembling.

The final solution was less about finding a single winning model and more about orchestrating hundreds of diverse prediction sources into a coherent stacking framework.

Finishing 10th place was rewarding, but perhaps the most valuable takeaway was gaining a deeper appreciation for ensemble design, model diversity, and the importance of treating OOF predictions as long-term assets rather than temporary artifacts.
