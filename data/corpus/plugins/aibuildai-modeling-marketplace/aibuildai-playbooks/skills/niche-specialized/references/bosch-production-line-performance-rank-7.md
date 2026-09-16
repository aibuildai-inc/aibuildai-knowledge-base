# Solution Sharing

Competition: bosch-production-line-performance
Rank: #7
Source: https://www.kaggle.com/c/bosch-production-line-performance/discussion/25348

The competition only has a couple minutes left so I think it is safe to start sharing. Don’t know whether our solution will drop like a rock or not, but here it is:

•	We only trained on rows that were part of duplicates – specifically row N was very similar to row N+1. It turns out that this set was almost exactly the same as Faron’s feature that measured whether the IDs were consecutive in a time-group.

•	We also only predicted for rows where N was very similar to N+1. We set all others to zero.

•	Some features were:

•	How many rows were in a particular time-group.  If there were many rows, there tended to be higher failure, and if there were few rows, there tended to be higher failure. Lowest failure was in the middle ranges.

•	Some time groups simply had higher failure rates than others.  So proximity by time was an important feature.

•	Total duration of a row from first timestamp to last timestamp.

•	Some values of L3_S32_F3854 were particularly valuable so we one-hot-encoded that.

•	The rows immediately before and after row N were important.  Specifically the Response on those rows; were they exact duplicates of row N or simply partial duplicates; the values of L3_S38_D3957 and L1_S24_F1844 on row N+1 were helpful.

•	Not all of the duplicates came in pairs.  Some were in triples, quadruples, or more. So we measured how many rows were in the duplicate “set” and what position this row was (e.g. 3rd of 4).

•	After feature creation, we used XGBoost and Random Forests and created a weighted average of their predictions.

•	I usually prefer simpler solutions to more complex ones, but for this competition, we found that to squeeze the last 0.01 out, we had to create some complex ensembles of runs that had different subsets of the features, different algorithms, etc.  In fact, one of our selected submissions simply took all of our submissions from the last month and predicted 1 if that row was 1 in 11 or more of the submissions.

We will see how it goes.
