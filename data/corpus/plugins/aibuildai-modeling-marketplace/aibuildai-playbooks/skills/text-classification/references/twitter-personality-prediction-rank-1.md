# What I learned and did

Competition: twitter-personality-prediction
Rank: #1
Source: https://www.kaggle.com/c/twitter-personality-prediction/discussion/2195

<p>Forum postings in other competitions have helped me a lot so I thought I'd share what I learned and did in this competition.</p>
<p>I learned that luck plays a big role in determining the competition winner. In cleaning my code for submission, I ran through everything again. While it beat my best model on the public leaderboard, it would not have been good enough to win on the private
 leaderboard.</p>
<p>Another important thing I learned is to always check the submission file. I wasted two submissions from badly formatted data frames and another two from transferring the model weights incorrectly from my cross validation code to the final submission code.
 Luckily, I caught the latter mistake a day before the competition ended and the resubmitted model won the competition.</p>
<p>Since the data set is so small, I combined the training and test data sets and used Excel and conditional formatting for data exploration. It's similar to what Jeremy Howard did (http://media.kaggle.com/MelbURN.html) Two patterns pop out.</p>
<p>Blocks of zeros tend to occur together for the Fvar variables with highly correlated non-zero values. I replaced those columns with averages (Fvar29-33, Fvar38-46, Fvar53-55, Fvar64-65, Fvar75-78). I also deleted columns with no variation (Avar9, Fvar23-28).</p>
<p>The second pattern relates to missing values. Some observations have no features and some observations had no Fvar or Lvar values. So I split the data set into three and imputed/modeled them separately. The motivation was to impute as little as possible
 to reduce overfitting to the data. kNN in the R imputation package was used.</p>
<p>No other feature selection was performed since, in a way, feature selection was already done in creating these derived variables. All I did was “feature elimination” and throw away really bad ones.</p>
<p>Faced with many unknown and likely un-important features, I concentrated on ensembles of weak learners and those that do feature space transformations. All tuning was done in R with repeated 2-fold cross validation. The first four models I fitted (random
 forests, gradient boosted machines, support vector machines, and multivariate adaptive regression splines) ended up in the final weighted average blend. None of the subsequent models improved the blend so were not used. I didn't try more sophisticated blending
 techniques.</p>
<p>I'm curious as to what others did in feature selection. And any other good models besides the ones I mentioned? I'm especially curious about people that did well against both the public and private leaderboards. How did you avoid overfitting?</p>
<p>&nbsp;</p>
