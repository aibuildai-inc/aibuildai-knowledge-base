# Solution sharing

Competition: machinery-tube-pricing
Rank: #4
Source: https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/16264#91207

If you were only using the benchmark, the problem isn't that it's a regression problem. The problem is that the benchmark did not come close to incorporating all the information available. There was a lot of data preparation involved in this competition, and the benchmark was only the very start.

I'll work on a full writeup soon. But in terms of representing the data, there were three main insights:

1. The numeric data in the component files was extremely valuable, albeit too sparse and high dimensional in its raw form. So we took various statistical aggregates of the weight, length, and other physical properties of the tube components. For example, for each row, we grouped together all numeric columns with "weight" in the column name and took the mean, min, and max. These derived aggregates were very predictive.

2. Adding soft leakage and encoding categorical variables. Grouping by tube assembly and adding aggregate information for each assembly was very valuable. For example, simply adding the count of unique quantities per tube assembly id. Likewise,  grouping by supplier and taking the mean, max, min of quantity, annual usage, etc. helped a lot. Adding these encodings was critical because the one-hot encoding was too sparse, in that the test set included suppliers, components, etc. that were not in the train set.

3. Perhaps the biggest "aha" moment was realizing that the tube assembly id's were autocorrelated. This was discovered largely by accident; internal CV dropped substantially when I left the row order in my training set. Thus, the solution was to map the order of the tube assembly ids to an integer index, so that TA-0001 was 1, TA-0002 was 2, etc. 

That's just the feature engineering part. My partners are really responsible for the machine learning pipeline, and I'm sured they'd be happy to share as well. Hopefully we'll come up with a cohesive writeup.
