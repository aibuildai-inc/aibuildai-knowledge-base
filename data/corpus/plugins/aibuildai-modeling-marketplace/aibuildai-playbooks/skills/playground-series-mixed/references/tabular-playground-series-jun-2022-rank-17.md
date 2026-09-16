# 17th place solution

Competition: tabular-playground-series-jun-2022
Rank: #17
Source: https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334343

This is my best achievement so far in the TPS competitions. Thanks a lot to Kaggle for hosting this month over month.

My solution involved 2 main aspects:

1. The finding by @ehekatlact to model based on the number of missing values in each record. This finding proved to be the key to this month's TPS.
2. Discovering the distribution of the public test dataset.

For majority of the first 10 days, I was deeply invested in working with the F_2 features to find some meaning. These features were categorical and had no missing values. I found it very hard to just discard these without investigating. The work is laid out [here](https://www.kaggle.com/code/nikhilkhetan/f-2-features-search-for-meaning). After slicing and dicing in every possible way using the F_2 features and not finding anything, I decided to move on.

By then, the general idea in the discussion forum was to Mean Impute F_1 and F_3 and model for F_4. I decided to start with modelling F_4 and publicized my work to [set up a Validation set](https://www.kaggle.com/code/nikhilkhetan/setting-up-a-validation-set) for this competition. This helped in comparing models and paving a path.

My first submission was without including the NA counts and had a score of 1.04555. The same model after including NA counts and some modifications, achieved a score of 0.94694.

After this, I decided to spend some time modelling F_1 and F_3 as well rather than just Mean Imputing. I tried lots of models and tactics but the public score hardly moved. This is when an idea struck me that maybe F1 and F3 are simply not a part of the public dataset. After a very simple probe, I was able to figure out that most of the public dataset is comprised of F4 columns and maybe a few records with missing in F1 and F3. This also led me to believe that Mean imputation may not be the best for F1 and F3 as we simply do not have any line of sight for the change in performance.

I decided to use the 2 allowed submissions to capture both the paths. I submitted one solution with Mean Imputing F1, F3 and modelling F4 and another one with modelling for everything. The 2nd solution was better both on public and private but only by a very thin margin(~0.00005) which translates to about 3 places down in the leaderboard. So, I guess utlimately it did not matter much.

I used the same set of 2 models for separately modelling each of F1, F3 and F4 and took a simple average of their results as described below:

1. A NN model based on [this](https://www.kaggle.com/code/ehekatlact/tps2206-the-na-count-of-each-record-is-critical) notebook by @ehekatlact but implemented in Tensorflow.
2. A combination of LightGBM and XGboost regression with NA count.

I had developed a few other models but chose the final submission based on the results from the public leaderboard. Initially, I only worked on the modelling for F4 and used mean imputation for F1 and F3 to save time. After trying various combinations of models, I selected the above as it gave the best score on F4 based on the public leaderboard. I then used that combination of models for F1 and F3 and generated my submission.

Thanks a lot to all the folks who contirbuted publicly to this month's competition. Got to learn a lot from the posts.

And special thanks to @ehekatlact without whose contribution, this would not have been possible.
