# 11th Place: In the midst of entrance exams

Competition: playground-series-s6e5
Rank: #11
Source: https://www.kaggle.com/c/playground-series-s6e5/writeups/11th-place-in-the-midst-of-entrance-exams

Thanks to @cdeotte, @masayakawatama, @yekenot, and @amantar for their notebooks and ideas.

This was a fairly standard competition overall. I used around **250 diverse OOFs**, **5 to 6 different feature sets**, and a few experimental ensembling ideas. The finish was unexpected because I spent most of the competition fluctuating around ranks #200 to #300. I still had several ideas left to explore, especially around OOF fusion and ensemble construction, but my entrance exams took most of my time.

I did not use AI for modelling. I only used Codex occasionally for small coding tasks.

# Models

Where do I even begin?

I trained a wide range of models:

* **GBDTs** such as XGBoost, LightGBM, and their DART variants
* **CatBoost**, which worked particularly well for me, possibly because of Ordered Boosting
* **AutoGluon** and **FLAML** trained across all six feature sets
* Decision Trees, YDF, Random Forests, ExtraTrees, and linear models such as Logistic Regression
* **TabNet** wasn't helpful for me. It was stuck around ~.93



Some of the feature sets included 5 fold target encoding.

I also experimented with **LogisticGAM**. Trained over **50 stochastic samples of 10k training examples**. It consistently outperformed Logistic Regression for me.

Another small improvement came from dropping the `Driver` column and retraining the previous models.

I also used:

* OOFs after Hill Climbing
* Level 2 and Level 3 stacking
* Weak but surprisingly interesting ideas such as Isotonic Regression on approximately monotonic functions

For hyperparameter tuning, I used Optuna with `n_trials=50` and `n_jobs=-1`. I mainly tuned parameters such as `colsample_bytree`, `subsample`, and `learning_rate`, while keeping settings like `n_estimators=2000` and `early_stopping_rounds=100` fixed.

I like to think of this setup as forcing Optuna to search for good solutions under constrained conditions. In my experience, this tends to improve model diversity.

# Feature Engineering

I used `PolynomialFeatures` along with meta features from @mikhailnaumov and @masayakawatama's notebooks.

I also used Ordered Target Encoding for XGBoost and LightGBM.

For interaction features:

* I used a selected set of 5 columns for trigram features
* I used the entire categorical feature set for bigrams

# Ensembling

My default ensembling approach is **Hill Climbing** because it naturally filters useful OOFs. I implemented a GPU based Hill Climber and used it throughout the competition. I also used **Ridge blending** extensively. I experimented with blending Hill Climbing and Ridge predictions using Optuna, although the gains were minimal.

A **Logistic Regression stacker** did not outperform Ridge, but its predictions were still useful as inputs to higher level ensembles. The final ensemble maneuver involved blending the outputs of **RealMLP and TabM** after projecting them through my final OOF sample space.

What surprised me most was how effective Ridge turned out to be. With 250 OOFs, there are theoretically `2^250` possible subsets. There is a reasonable chance that some subset performs better than the full set itself. To explore this idea, **I randomly sampled 200 subsets, each containing at least 50 model OOFs**. I trained a Ridge ensemble on each subset, producing 200 new OOF predictions. Averaging these predictions gave results that were only slightly weaker than my final ensemble setup.

I wanted to push these ensembling ideas further, but I was too busy studying chemistry for my entrance exams.

# Closing

These competitions become extremely competitive near the end, and there is never a shortage of new ideas to try.

I do not necessarily aim for rank #1 or even rank #11. I simply enjoy thinking about problems, and sometimes those ideas happen to work.

My entrance exam, JEE Advanced, is now over, and I obviously did not crack it. Hopefully, the Kaggle competitions I participate in will work out a little better for me.
