# 12th place solution

Competition: allstate-claims-severity
Rank: #12
Source: https://www.kaggle.com/c/allstate-claims-severity/discussion/26414

I’m guessing there is an easier way to get to where I ended up, but here is how I got to my solutions.  First a couple of notes:

1.	I used a stratified 10-fold CV for all models.  The stratification was based on the normalized loss (z-score).
2.	All the stacking models use out-of-fold predictions from the predecessor models.
3.	I did post-processing in two ways, both using a 10-fold CV to test the results and calculate the OOF predictions:

•	Method 1: For each fold x, use KMeans clustering (k=25) to partition the predictions (where fold != x).  Predict clusters for OOF (fold==x) and the test file.  Then if the maximum prediction in a cluster was greater than 10,000 add c (c in range 0 to 20000) and test MAE for improvement for each c.  For c where MAE improves the most, add c to OOF and test data in the same cluster.

•	Method 2:  For 10-fold CV, for each fold x, use rows fold!=x to determine the best threshold and additive value c by seeing if the MAE improves when adding c to all prediction>threshold. For this method I tested all thresholds between 0 and max(prediction), and all c between 0 and 1400.

4.	Because my CV scores tracked well with my public LB scores I didn’t worry so much about PLB score vs. CV score when choosing two submissions as my final one.  I was more worried about the post-processing overfitting the training data because my CV went down on a couple, while the PLB score went up.  Ultimately, I chose one model without and post-processing and one model with.  The model without any post-processing scored better on the private LB.

**Feature Engineering**

For additional features I built most off of three sets of the original features; the binary categorical features (each has only two levels, cat1 – cat72), multi-level categorical (more than 2 levels, cat73 – cat116), and the numeric features (cont1 – cont14).

The additional features I created included (summary stats = minimum, mean, median, maximum, stdev, etc.):

•	binary encoded categorical columns (cat1 - cat72)

•	multi-level categorical columns (cat73 - cat116) label encoded

•	multi-level categorical columns (cat73 - cat116) ascii value encoded

•	multi-level categorical columns (cat73 - cat116) One-Hot encoded

•	multi-level categorical columns (cat73 - cat116) minimum log(loss) per level

•	multi-level categorical columns (cat73 - cat116) maximum log(loss) per level

•	multi-level categorical columns (cat73 - cat116) mean log(loss) per level

•	multi-level categorical columns (cat73 - cat116) stdev log(loss) per level

•	paired categoricals, ascii value encoded

•	multi-level categorical columns (cat73 - cat116) mean log(loss) per level summary stats

•	binary categorical probabilities of occurrence

•	multi-level categorical probabilities of occurrence

•	Kinetic values (sum (probability of occurence**2) ) for categorical columns

•	multi-level categorical expected values (probability * mean value)

•	base numeric features summary stats

•	skew-corrected, numeric columns

•	skew-corrected, standard-scaled numeric columns

•	Square root transformed numeric columns

•	Log transformed numeric columns

•	PCA features (59 features => 90% of the variance)

•	distance to binary column centroid

•	distance to multi-level, ascii encoded centroid

•	distance to multi-level, mean(log(loss)) centroid

•	distance to base numeric centroid

•	distance to skew-corrected, standard-scaled centroid


From these features I built XGBoost, Keras, kNN, SVM, Regression, LightGBM, ExtraTrees and RandomForest models on various subsets of the features using either log(loss) or loss**(1/4) as the target feature.

**First Level Models**

I eventually narrowed these down to a set of models that gave the best 10-fold CVs and worked best in stacking. I used the out-of-fold predictions from these 28 models to train Keras, XGBoost and RandomForest models (stacking):

•	8 XGBoost models (best CV: 1126.27, PLB: ~1107)

•	4 Keras models (best CV: 1130.32, PLB: 1107.4)

•	5 ExtraTrees models (best CV: 1164.61, PLB: not submitted)

•	2 RandomForest models (best CV: 1167.07 PLB: not submitted)

•	4 k-Nearest Neighbor models (best CV: 1279.55, PLB: not submitted)

•	2 SVM models (best CV: 1145.16, PLB: not submitted)

      o	Both were LinearSVR

•	3 Regression models (best CV: 1145.57, PLB: not submitted)

      o	1 OLS (best CV), 1 BayesianRidge, and 1 ElasticNet

**Stacking into 2nd & 3rd level models**

For choosing models to include in next-level stacking I looked at OOF correlations and their performances on various percentiles.  That is, I calculated the MAE for all of the OOF predictions in 0-10th%, 10-20th, etc. and picked the ones that performed the best on a percentile range.

The stacking combinations that led to my final two submissions (the numbers are just how I track each model output) is attached.
