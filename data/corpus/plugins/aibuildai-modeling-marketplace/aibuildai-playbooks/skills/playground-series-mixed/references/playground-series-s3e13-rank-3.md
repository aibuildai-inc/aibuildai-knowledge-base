# #3 solution

Competition: playground-series-s3e13
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e13/discussion/406409

3rd place solution Playground Series - Season 3, Episode 13

I want to express my gratitude to the organizers at Kaggle for putting together the Playground Series. As a beginner, participating in these competitions can be very challenging but they can teach valuable skills.

I would also like to give a special thanks to @Matt OP, @Tanoi, @Elric and @broccoli beef for sharing their expertise in their respective notebooks:  

- https://www.kaggle.com/code/mattop/ps-s3-e13-important-features-for-each-disease/notebook 
- https://www.kaggle.com/competitions/playground-series-s3e13/discussion/402933
- https://www.kaggle.com/competitions/playground-series-s3e13/discussion/403493,
- https://www.kaggle.com/competitions/playground-series-s3e13/discussion/403156

Their insights were invaluable and helped me greatly.

Here's what worked for me:

- To start, I wanted to come up with a reliable validation method. Based on the previous Playground competition, I decided to use RepeatedStratifiedKFold. I selected k and n based on the average validation score, standard deviation, and standard error of the mean. I found that 10 repeats of 10-folds seemed to be a good choice to assess the validation scores.
- For my baseline method, I used SVC, which had a validation score of 0.367 +- 0.025 std.
- I used the features provided by Tanoi and found that including 'kidney-failure' resulted in a cross-validation score of 0.370 +- 0.025.
- I also grouped features based on their names and added up their values. For example, I added up the values of features that contained the words pain, loss, or inflammation. Including the summed-up features containing the word pain improved the cross-validation score to 0.375 +- 0.026.
- I also included polynomial features and selected a feature pair that resulted in the highest cross-validation score: 'back_pain' and 'yellow_skin', this gave a score of 0.3937. Next, I further tested including additional pairs of polynomial features ('itchiness', 'bullseye_rash', and 'diarrhea', 'hypoglycemia'): the corresponding validation score further improved to 0.3989.
- Feature selection: I preselected some algorithms that performed well on the initial feature set and checked what would be a good VarianceThreshold cutoff rate for them. I tested XGBClassifier, SVC, BernoulliNB, NuSVC, and LGBMClassifier. All of them resulted in the highest cross-validation score when using a threshold of 0.1. These were all the algorithms that I used for the final ensemble without weighing.
- The public score of my final submission was 0.37089, and the private score was 0.52302.

Here's what did not work or might have worked:

- Due to the vast amount of original features in the dataset, I thought that clustering techniques would be beneficial to this task. I tried several of them, including MeanShift, Birch, BisectingKMeans, and KMeans. MeanShift had the best cross-validation score of 0.357 +- 0.039, so it did not provide any improvement.
- I also tried dimension reduction techniques, such as PCA, but did not invest too much time into them. It might be worth exploring more in the future.
