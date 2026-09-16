# 2nd place solution overview

Competition: santander-value-prediction-challenge
Rank: #2
Source: https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63848

First of all, thanks to Giba for sharing the [first set of columns][1], Moshin hasan for the [kernel][2] with the way to use them, and all the discussion participants and kernel creators for insights. Apart from a healthy dose of luck, my solution consisted of the following:

**Row and column order.** The procedure I used was similar to the one that have emerged in the forums. One of the differences was that at each iterations I organised the resulting rows/columns into sequences and only used the sequences above a certain length (e.g. 5) for the next iteration to try to avoid using false-positives for matching. Creating sequences also helped to assign IDs to the groups of rows that were likely to belong to the same client.

**Validation.** I used a 5-fold CV but adjusted such that all the rows belonging to a single client were never split between the train and validation sets. Since the sizes of these groups varied a lot, this introduced some additional variance, so all the final predictions were averages over several CV seeds.

**Data augmentation.** I added the test leaks to the training set - this improved the local CV however given only two decimals it's unclear whether it had much effect on the private leaderboard.

**Model.** The best performing model was relatively simple: feature set consisted of various non-zero aggregations (mean, max, exponential smoothing, count, etc.) applied to all the 96 forty-observation-long sequences and their subsets of the last 7,14,21,28 "days"; plus leading non-zero values in each of the sequences and aggregations over all non-zero values in a row. This resulted in ~3500 features that were reduced to ~1000 via selection. 

The final model was a blend of lightGBM, XGBoost and a random forest, all using those ~1000 features.


  [1]: https://www.kaggle.com/titericz/the-property-by-giba
  [2]: https://www.kaggle.com/tezdhar/breaking-lb-fresh-start
