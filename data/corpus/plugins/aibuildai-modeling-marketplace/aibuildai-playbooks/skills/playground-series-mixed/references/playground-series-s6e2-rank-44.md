# 44th Place - Simple as Slime

Competition: playground-series-s6e2
Rank: #44
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/44th-place-simple-as-slime

First of all, congratulations to @masayakawamata for securing 1st place; I learned a lot from your notebooks and insights. I also want to thank @mirko45, @tilii7, and @cdeotte for the knowledge shared in discussions, @omidbaghchehsaraei for the excellent solo RealMLP notebook, and @mpwolke for the discussion on cardiovascular risk factors—it was very helpful!

This competition was unique for me because the CV-LB relationship was the most critical factor, as highlighted by @masayakawamata in his winner's write-up.

### My Approach

**1. Feature Engineering**
I experimented extensively with feature engineering using a baseline XGBoost model. I followed a Manual selection approach: adding a batch of features, checking if the CV improved, and keeping only the ones that provided a boost. This resulted in a core set of features used across all models:
    
-  Frequency Encoding: Applied to numerical columns across Train, Test, and Original datasets to capture the distribution of values.
- Target Encoding: Performed using 5-fold Stratified CV on the combined Train and Original datasets.
- Numeric Binning: Used pd.qcut to create 10-quantile bins for numerical features.

**2. Modeling Strategy**
I trained several diverse models using the feature engineering described above. I also experimented with different feature subsets (e.g., only original features, only categorical, etc.) to increase diversity for the final ensemble.

| Model | CV | Private LB | Public LB |
| --- |
| XGBOOST | 0.955445 | 0.95518 | 0.95373 |
| XGBOOST TUNED | 0.955622 | 0.95526 | 0.95383 |
| CATBOOST | 0.955647 | 0.95527 | 0.95387 |
| CATBOOST TUNED | 0.955702 | 0.95528 | 0.95386 |
| LIGHTGBM | 0.955447 | 0.95505 | 0.95363 |
| REALMLP | 0.955688 | 0.95530 | 0.95390 |


To increase diversity for the ensemble, I developed several model variations with different feature subsets:
- XGB + All Categorical: All original features were cast to the category dtype.
- XGB + Original Features: Used only the baseline features without engineering.
- XGB + Numerical Only: Focused strictly on numerical columns.
- XGB + Categorical Only: Focused strictly on categorical columns.
- CatBoost + All Categorical:All original features were cast to the category dtype.

**3. Blending & Results**
I collected the OOF predictions and test predictions from all models (approximately 10-15 variations) and applied Rank-based Hill Climbing to find the optimal weights.
- Final Ensemble Score: (CV: 0.95574 | Private LB: 0.95532 | Public LB: 0.95396)

### Lessons Learned:

- Feature Diversity: Instead of using the same feature engineering for every model, I should create "specialist" models with different feature sets to decrease correlation between errors.
- CV-LB Relationship: I focused primarily on improving my CV score but should have paid more attention to how those changes reflected on the Leaderboard to ensure better generalization.
- Model Quantity: I only built 10-15 models. Expanding the ensemble with more variations (different seeds, different architectures) would likely have yielded a more robust final blend.

I titled this "Simple as Slime" because I stuck to a specific set of features and a relatively small number of models. Sometimes simplicity works, but more diversity is the key to climbing higher!
