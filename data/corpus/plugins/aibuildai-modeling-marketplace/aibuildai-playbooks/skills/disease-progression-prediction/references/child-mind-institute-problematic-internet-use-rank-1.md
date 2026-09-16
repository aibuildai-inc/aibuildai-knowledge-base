# First Place Write-Up: Or How I Won the Lottery

Competition: child-mind-institute-problematic-internet-use
Rank: #1
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552638

**First and foremost, I would like to thank the organizers and Kaggle for making this competition possible. Tackling real-world noisy data was both a challenging and rewarding experience.**

**To be perfectly honest, luck played a major role in my success. I was especially lucky actually selecting the best possible notebook. Nevertheless, I’d like to present what I did. ([Link to the notebook](https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368))**

Interestingly, I dropped out of the competition two months ago only to re-enter recently. My focus during this return was on improving the robustness of the solution.

# Final Model Overview
The final solution was a voted ensemble consisting of:
•	LGBMRegressor
•	Two XGBoost Regressors
•	CatBoostRegressor
•	ExtraTreesRegressor

# Target, Cross-Validation, and Sample Weights
•	**Target Variable:** Instead of using the provided sii labels, I used the 'PCIAT-PCIAT_Total' score and converted the predictions to sii labels. 
•	**Distribution and Weighting:** The target’s distribution, especially the excess zeros, led to two approaches: exploring sample weights and alternative objectives for regression like **Tweedie**. Equidistant bins were used to define the sample weights. Weighting did not improve the optimized scores directly but it brought the unoptimized scores closer to the optimized scores.
•	**Cross-Validation:** A 10-fold stratified KFold was used, stratified by the bins. Seeds were frequently changed to ensure stability. Submissions with different seeds were used to get further feedback from the public Leaderboard on the variance. The LB-score itself was mostly ignored to minimize overfitting to the leaderboard.
# Data Cleaning, Feature Engineering, and Imputation
**Data Cleaning:** 
- Implausible values, such as body fat percentages over 60% or negative bone mineral content, were removed and replaced with NaN.

**Feature Engineering:** 
-	Various descriptive actigraph features were created, with separate masks for day and night.
-	Dimensionality reduction of the actigraph data using PCA retained 15 components.
-	Additional features included normalized values based on age group means and other features that seemed sensible like the difference between the daily energy expenditure and the basal metabolic rate.
-      Quantile binning was applied to a good chunk of the features to deal with the noise. Which worked surprisingly well.

**Imputation:** 
-	Lasso was used for feature imputation, due to the moderately high dimensionality and noise.
-	Features were imputed using Lasso. For each target column, a model was trained using features with fewer than 40% missing values, and missing values in these features were imputed based on the trained model. If no valid features (i.e., features with less than 40% missing values) were found for imputation, or if the number of valid samples was too small, the solution defaulted to mean imputation.
# Parameter Tuning and Feature Selection
Early in the competition, it became apparent that typical parameter tuning with regular cross-validation setups resulted in unstable outcomes. To address this:
•	Repeated Stratified KFold was employed during parameter tuning. With 10 to 20 repeats. Which was computationally more expensive but yielded more robust results.
•	Feature selection was done manually based on feature importance, reducing the dataset to 39 features
