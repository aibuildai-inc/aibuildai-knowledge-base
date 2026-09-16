# First place solution

Competition: dont-overfit-ii
Rank: #1
Source: https://www.kaggle.com/c/dont-overfit-ii/discussion/91766#latest-539866

First of all, I want to say that I feel incredibly honored to win. [The original Don’t Overfit](https://www.kaggle.com/c/overfitting/leaderboard) was my first Kaggle competition and coincides with when I discovered machine learning.  I remember poring over forum posts to learn how to fit models with the caret package and writing R code on the commuter train on my way to job interviews in Boston, a city where I’d just arrived. On those train rides, I daydreamed about one day having a job where I did machine learning for work.

I can’t believe it’s been 8 years.

The original Don’t Overfit competition was the start of a trajectory that took me to the career I have today.  Thank you @salimali for piquing my interest, and thank you to the Kaggle community for all you’ve given to me.

***

This competition, of course, was completely different from the original Don’t Overfit 😁 

In the original competition, we only had 5 submissions total for the private leaderboard and no leaderboard feedback. 

 I believe my critical insight was being among the first to realize leaderboard probing would be key to winning. I started a dedicated effort to submit every individual variable as a single submission.  I was able to get the [AUC for all 300 variables on the public leaderboard](https://www.kaggle.com/zachmayer/300-probed-aucs?scriptVersionId=13927278), and as @cdeotte already [discussed](https://www.kaggle.com/c/dont-overfit-ii/discussion/91683#latest-528739), used (AUC public) - 0.50 to come up with “coefficients” based on AUC. If Chris had had time for 250 more submissions, he would have had a good chance of beating me. Just using the 300 public LB “coefficients”, with no further processing, gives a private LB score of **0.853**, which would have been good enough for 110th place or so.  

I additionally used the AUCs on the training data to add a little bit of extra information to this model, using the following formula: `0.11*(AUC train) + 0.89*(AUC pubic) - 0.5`
The weight for AUC train is `(rows train)/[rows train + rows public]`
The weight for AUC public is `(rows public)/[rows train + rows public]`

This gives a private leaderboard score of **0.856**, good enough for 60th place or so.  (About where I finished in the first iteration of Don’t Overfit!).

It also gives the following plot of public AUC vs training AUC, which led me to believe all 300 variables were used to create the target variable.  This is also a useful plot to calibrate simulated datasets (more on this later):
[Public vs Training]

This methodology overfits the public leaderboard, as there are many small AUCs that should probably be zero.  Using Chris’ capping heuristic, we can set every coefficient with abs(cf) &lt; 0.04 = 0, which yields the following equation:
```
-0.0661*X16 + -0.0448*X29 + 0.1778*X33 + -0.0526*X45 + -0.0736*X63 + 0.171*X65 + -0.059*X70 + -0.1047*X73 + -0.118*X91 + -0.0422*X106 + -0.0446*X108 + -0.0985*X117 + -0.0448*X132 + 0.0462*X164 + -0.0514*X189 + 0.1117*X199 + -0.0704*X209 + -0.1198*X217 + -0.0456*X239
```

This is very similar to Chris’ solution, just with more variables. This gives a private leaderboard score of **0.877**, which is good enough for first place.

***

I don’t like capping, and I wanted a solution that smoothly shrinks smaller AUCs to zero.  For example, variable 239 in the above equation has a coefficient of -0.0456, and there’s a good chance that is really a zero (but I’m not certain)!

So what I spent most of my time doing was looking for an equation that would smoothly shrink small coefficients to zero, while leaving large coefficients more or less alone.  I dove down a bit of a rabbit hole here, but it was a fun rabbit hole, and I’m interested to see what approaches other people take to solving the same problem.

The approach I took was:
1. Simulate a dataset with 20,000 rows, 300 columns, sampled from the random normal distribution.  AKA `matrix(rnorm(20000*300), ncol=300)` in R.  I am 99.99% sure this is how Kaggle simulated the data for this competition.
2. Apply my probed coefficients to the data.  This isn’t perfect, but the coefficients I probed should be pretty representative of the real distribution of coefficients.  AKA `matrix(rnorm(20000*300), ncol=300) %*% probed_cf` in R.
3. Add a little random noise.
4. Choose a classification cutoff that gives a target distribution in the training data of 160 0s and 90 1s.

I then fiddled with steps 2-4 until I got (AUC train - 250 rows) vs (AUC public - 1975 rows) plots that looked like the above plot.

I also simulated “leaderboard probing” to try to get simulated results that had training AUCs of .95 and public AUCs around .92, as those were my results on the real leaderboard.

***

Once I had my simulated data, the problem becomes: predict the “real” coefficients, given the probed coefficients.  I played around with many different methodologies for this, and finally ended up using [Eureqa Desktop](https://www.nutonian.com/products/eureqa-desktop/) to discover the following equation:
```
penalized_cf = 0.0320794465661013 + 1.38501844798023*cf - 9.51464300036321*cf^3 - 
    0.0654557703259389*plogis(85.9772069321308*cf)
```
(Where plogis is the sigmoid function in R)
Here is a plot of the penalized coefficients vs. the raw coefficients.  Note that coefficients near zero get shrunk almost completely to zero, but not entirely:
[Penalized]

***

So to recap, my final submission was:
```
cf = 0.11*(auc_train) + 0.89*(auc_pubic) - 0.5
penalized_cf = 0.032 + 1.385*cf - 9.515*cf^3 - 0.065*plogis(85.977*cf)
sub = X_sub %*% penalized_cf
```

This achieves a private LB score of **0.886** and a public LB score of 0.923

I’ll post more detailed code (along with the probed AUCs) in some public kernels.
