# My approach

Competition: allstate-claims-severity
Rank: #4
Source: https://www.kaggle.com/c/allstate-claims-severity/discussion/26411

I'm really curious to hear from the rest of the top 10, but I'll go first:

1. First, I started with the best public script, in particular the XGBoost scripts and keras scripts that were floating around.  Picking the best tuned single model didn't seem to matter a whole lot, e.g. I used one of the the PB LB 1108 XG scripts and never got around to running the 1106 one (maybe that was my mistake!).  I edited most of the keras scripts to run 10 CV folds and 10 bags, and most of the XGBoost scripts to use 10 folds and no bags.
2. I modified each public script to generate stacked predictions, if it didn't already.   All the keras script generated stacked predictions, but I had to modify some of the XGBoost scripts to generate stacked predictions.  This was key.
3. Next, I generated a bunch of different datasets to run through those scripts.  This was the part where I got a bit creative.  I can go into more detail here later, but I did things like PCA on all the numeric variables, PCA on all the categorical variables, PCA on all the numeric + categorical variables, all the numeric variables differences and ratios, all the categorical variable interactions, etc.
4. Steps #2 and #3 were my core loop.  Generate new datasets, and then run them through my XGBoost and Keras scripts to generate stacked predictions and test-set predictions. New datasets usually gave me more lift than tuning my models.
5. After a few iterations, I'd end up with a bunch of models with stacked predictions and test-set predictions.  Before any ensembling, I ran these models through wilcox.test in R, to make sure the stacked predictions were similar to the test set predictions.  Anything with a p.value of less than .10 I dropped from my ensemble.  Sometimes this step indicated models where I'd done something silly, like forgetting to un-transform the test-set predictions.
6. Then I took the set of remaining stacked predictions, and ran PCA on them.  I added the first PCA component to the stacked predictions and the test set predictions.
7.  Now for the key step.  I ran the stacked predictions through the optim() function in R, to find linear weights that minimized MAE.  I used the BFGS solver for optim.
8. Later in the competition, I split the stacked predictions into thirds, based on the PC1 component, and used optim on each third.  I then applied these 3 sets of weights to the test set, based on the PC1 component for the test set.  I'm not sure if this helped or hurt me at the end of the competition; it may have been unnecessary complexity.
9. I also added some L1 regularization to my call to the optim function.  I simple added  mean(abs(w))*(a penalty factor) to the objective function for optimizing the weights.  I'm pretty sure this regularization helped at the end of the competition.
10. I made 2 subs at the end: my best leaderboard score, and my best local CV score.  I think my best local CV score ended up being the best on the private LB.  Based on the small shakeup at the end of the competition, I'm happy I made one sub with my best local CV.  Trust your cross-validation!
 
Once I see the public vs private LB for all my subs, I'll have more details on what helped or what hurt.
