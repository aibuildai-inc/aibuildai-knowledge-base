# 6th place - Kernel Density Estimation

Competition: facebook-v-predicting-check-ins
Rank: #6
Source: https://www.kaggle.com/c/facebook-v-predicting-check-ins/discussion/22123

My final solution was an ensemble of XGBoost, random forests, k-nearest neighbors, and kernel density estimation. My XGBoost, RF, and KNN methods ended up being highly similar to what has already been discussed in the forums so in this post I will share my KDE method, which scored ~0.604 on the public LB.

The idea of this problem is very simple, given a new check-in, can we predict the business of the check-in out of 100000+ businesses? Using, KDE, the goal is to generate a kernel density for every business in the training set using relevant properties. Then, for each observation in the test set, compute the probability of the check-in being at each of the 100000+ businesses and sort the results from highest to lowest. 

**Problems faced:**

1. 100000+ business is too many. 

Solution: break the data set into smaller grids as in the KNN and XGBoost methods

2. How to incorporate the various properties of the data?

Solution: estimate a kernel for each property (e.g., x, y, log(accuracy), day of week, hour of day), and then multiply the probabilities. This assumes that the properties are independent (which I believe they are) so you're basically computing Pr(check-in = business | x, y, log(accuracy), day of week, hour of day) = Pr(check-in = business | x)*Pr(check-in = business | y)*Pr(check-in = business | log(accuracy))*Pr(check-in = business | day of week)*Pr(check-in = business | hour of day).

3. How to select the bandwidth/kernel type?

Solution: I used the kernel density code from sklearn. They offer a number of different kernels. The best way is to start with the Gaussian kernel, and to start with Scott's rule for bandwidth selection (described in the Scipy kde documentation). Then, tweak the values to optimize the kernel and bandwidth with your validation set. I found that the Exponential kernel was slightly better than the Gaussian kernel.

**Solving the three problems above produces a score of ~0.59.**

Next, we know that check-ins do not happen consistently across time. Some businesses have check-ins early in the time period and others later and some happen at seemingly randomly periods (while still following day of week and hour of day trends). One theory to test is that businesses that have check-ins later in the training set are more likely to appear in the test set. This can be modeled in the KDE framework by simply multiplying the final probability by the average date of check-in for the business, so that businesses with later check-ins will have higher probabilities than businesses with earlier check-ins. Doing this improves the score to ~0.595. 

Another way to do the above is to estimate a KDE for time (e.g., week number), and set all of the test set's week number to be that of the final week in the training set. Doing this instead of the above method scores ~0.594, but an ensemble of both methods (while keeping the other parts the same) improves the score to 0.602 on the public LB.

Finally, the grid method cuts off boarders at specific locations which limits kernels to be estimated using only the data within the grid. By shifting the grid slightly in both directions (x and y), you end up using a different data set to estimate the kernels. Ensembling the results from the two different grids improves score to 0.604.

Attached is a not-so-clean Python script that I used. Please let me know if something is unclear or wrong in the script. Hope this was useful/interesting for people!
