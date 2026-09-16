# 4th place solution

Competition: bosch-production-line-performance
Rank: #4
Source: https://www.kaggle.com/c/bosch-production-line-performance/discussion/25370

First of all: congratulations to all 3 winning teams for your great results, as well to all other participants who put considerable work and finished in good (but maybe not top) positions! Also thanks, to kernel/forum contributors!

In fact, I had practically quit after seeing the word "leakage" in the forum. However, the beautiful post by JohnM with the workflow visualization that I had read early on, made me reconsider in the last week as I had since the intention to try out stacking over station groups and I thought it could be worth a try anyway (and as a bonus it turns out that this also somewhat demystifies the "leak"). This is why I submitted the sample submission on the cut-off day and then rushed to finish with 12 submissions the last 3 days.

Due to the latter and the robust cv approach, my scores seemed to follow the pattern: private > public > cv (and model performance order was consistent). However, a downside is that I did not effectively use any feedback from the LB, which maybe could be helpful... but also very tricky!

Here is an overview of my solution:

* "Batch features": These seem equivalent (or a tiny bit more powerful/expressive) compared to the "magic features".
1. Identify samples that are exactly identical (it turns out these always have consecutive Ids), and count the number of duplicates (powerful) and the Id rank (somewhat useful as this is additional info). 
2. Count number of *different* samples in almost identical "batches", using workflow for guidance (very powerful). It is convenient that these have consecutive Ids too, although Id here does not give extra info as these seem to be ordered by consecutive columns. So, lines can be subdivided into (almost) mutually exclusive station groups (L0: [S0-S11] and [S12-S23], L1 and L2 by station, L3: [S29-S37, S38] and [S39-S51], where S38 must be considered separately in the end), and starting from earlier to later groups, "batches" are samples that are duplicates up to this point (the most common pattern consists of samples followed by exact duplicates everywhere except in S38).
3. The value of the Id rank consists in almost excluding the very last Id in a given "batch" (with simultaneously both above definitions) from potential failures. Everything considered, this identifies 5265 failures in ~70k rows (from <7k failures in ~1.1M rows) in the train set, and correspondingly ~70k rows in the test set. 

* Stacking based on different feature subset combinations:
1. Level1, using all data, do almost all combinations of the following: a. train models with and without numeric features, b. with and without "batch" features, c. use the described workflow decomposition to train on separate station groups
2. Level2: blend the metafeatures with (a selection and minor adaptation of) the original features, training only on previously identified "batches" (ie ~70k rows).
3. Algorithms: for level1 just xgboost..., for level2 (in order of performance) xgboost, random forest, extratrees, keras. The final solution was a simple average.

* Trading off efficiency in favor of robustness:
1. Level1 1 cross validation based on a customized "StratifiedLabelKFold" with shuffling, split on "batches" instead of samples. Each fold contained all corresponding (exact or approximate) duplicates, and models were tuned based on auc. The trade-off is that the test metafeatures were (probably) more accurate compared to cv metafeatures, and I thought this would make for a more robust approach, at the cost of some loss of information as regards absolute performance.
2. In level2 tuning was based on plain stratified folds, using the lower bound of the 99% confidence interval of a moving average of mcc versus  prediction rank as evaluation metric. The latter (is simpler than it might sound and) was helpful in finding a robust cut-off threshold (namely, ~2860 ones in my final solution).

(edit: copyedit)
