# Hints on 13th place solution.. and finally Grandmaster!

Competition: talkingdata-adtracking-fraud-detection
Rank: #13
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56333

Hi everyone! First of all, congratulations to the winners, and thanks to all of those who shared ideas or kernels in the forum.
This competition has been very challenging, considering the amount of data involved and the tricky time-variant problem.

I'll just give some hints on my solution, besides what is already described in many posts and kernels:

 1. Use test supplement: it is crucial, as it gives extra data to calculate next click features;
 2. Focus on a clean and correct approach first, then on CV / local validation (time-variant problem + extremely imbalanced dataset = impossible to have a 100% reliable validation, at least for me), finally on the public LB (clearly too small and not representative of the private LB) 
 3. Careful with next click feature: a click at 14.30 in the test set, for instance, has a maximum next click at 90 minutes (using test supplement), in train it could be over a day. I ended up using three 24 hours time frames and calculating the next click within this time frame, otherwise there would be a clear difference between training and test features.

Other notes:

 - I used R + xgboost and lightgbm (and heavy use of data.table for grouping, counting, shifting etc), training on days 7-8-9 in some models and on 8-9 in others;
 - Validation setup: random 10mln rows in validation, 25mln in train plus all the "ones" not present in validation (remember AUC cares only about order) - repeat with different samples
 - Memory: 32GB of ram and 60GB of swap (never used swap in previous kaggle competitions). 
 - Other features: some "previous day" counts and ratios proved to be not particularly effective, but useful to create diverse models.

At the end of the day, 6 competitions, 5 gold medals, 4 solo gold medals, 1 prize in private masters-only competition, it's time for a break! But I always come back to Kaggle sooner than expected.
It is an addicition with no effective therapy and a great way to learn, deal with real world problems and keep up to date!
