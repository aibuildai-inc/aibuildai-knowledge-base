# #1 solution

Competition: tabular-playground-series-jul-2022
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-jul-2022/discussion/341023

Here is a write-up of my approach ; see the actual code at https://www.kaggle.com/code/ymatioun/tps22jul-custom-em


1. Data structure.
Only 14 variables are relevant to clustering, the others are not used at all (this has been discovered early on in the competition). Variables f_07 - f_13 are integers (7 of them), and f_22 - f_28 are floating point (also 7 of them).

Floating point variables already appear to have a normal distribution, so there is no need to transform them.

But integer variables appear to come from some sort of mixed Poisson distribution. I tried to construct a model for them, but could not come up with anything that worked (that is, produced negative correlations but no negative values, and had a tractable probability mass function). So instead i did the same thing as everybody else - power-transformed them into a more 'normal' shape and treated them as multivariate normal vars. This is probably where the model could be improved the most [would be nice if competition organizers told us exactly how they constructed those vars, for future reference].


2. Cluster structure.
Data consists of 42 clusters, structured as 7 groups of 6 clusters each. Each cluster group is treated as one cluster as far as cluster labels are concerned, resulting in 7 distinct cluster labels.

Integer variables only vary by 7 cluster groups (so they truly form 7 clusters, not 42). Their means and covariances seem to be completely independent across 7 cluster groups (some covariances are zero).

Floating variables have two distinct distributions: for each cluster group, 2 floating variables have mean=0, standard deviation=1, and are uncorrelated with any other integer of floating point variables. Remaining 5 floating point variables have means that are -1 or +1, and full 5x5 covariance matrix. They are not correlated with integer variables, so full 14x14 covariance matrix consists of 7x7 block for integer vars, 5x5 block for correlated floating point vars, and 2x2 unit matrix for uncorrelated floating point vars.


3. Background

This data structure can be easily uncovered by running GaussianMixture(n_components=42) on floating point variables only, and looking at the means of mixture components.

I realized this when i looked at the source code of BayesianGMMClassifier (doesn't everybody do that? -:) ), trying to understand why it improved results so much, and saw that it used the same n_components=7 for each of the 7 clusters, resulting in total of 49 clusters. I thought that was a mistake, and made it use 1 component for each cluster, but that greatly reduced the LB score. That demonstrated that true data structure consists of > 7 clusters; further experimentation showed that each cluster group actually has 6 clusters, not 7.


4. Model.

I wrote a custom EM (Expectation Maximization) algorithm for a Gaussian Mixture Model, taking into account details of the data structure described above. I hard-coded means of all the floating point variables (0, -1, 1) to ensure convergence to true cluster centers. And this is what resulted in my best LB score.
