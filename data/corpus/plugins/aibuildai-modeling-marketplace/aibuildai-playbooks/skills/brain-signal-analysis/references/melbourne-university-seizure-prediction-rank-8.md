# Solution (8th)

Competition: melbourne-university-seizure-prediction
Rank: #8
Source: https://www.kaggle.com/c/melbourne-university-seizure-prediction/discussion/26268

First I would like to thank the organizers and all the participants, this competition was very interesting, I really enjoyed it and it would be great to see some useful ideas coming out of this competition.

So here is a summary of my final solution based on **Classification decision trees** which scored **0.80396 AUC** on public leaderboard and **0.79074 AUC** on private leaderboard:

Software
--------
Matlab 2014a

Data
----
I used all data files marked as safe in *train_and_test_data_labels_safe.csv*. No preprocessing was done.

Features
--------
Features were calculated on the whole 10 minute files for each channel without splitting into any shorter epochs. 

I basically took all the features from sample submission script and added few more based on my hunch and some articles about this topic. The features included:

-	mean value, standard deviation, skewness, kurtosis, spectral edge, Shannon’s entropy (for signal and Dyads), Hjorth parameters, several types of fractal dimensions
-	<del>eigenvalues</del> singular values of 10 scale wavelet transformation using Morlet wave
-	maximum correlation between channels in interval -0.5,+0.5 seconds (the maximum was pretty much always at 0 second delay anyway), correlation between channels in frequency domains, correlation between channels power spectrums at each dyadic level

I had 73 features in total for each channel, only the real part of features was used.

Cross validation
----------------
I used *cvpartition* from Statistical toolbox which can create random partitions where each subsample has equal size and roughly the same class proportions. I did not care about sequences which caused my local AUC results to be around 0.1 higher than the leaderboard ones.

Model
-----
A classification decision tree model was created for each channel and patient, the mean output across channels for the patient was used as the outcome. Models were trained with 10 fold cross validation to prevent overfitting.
Because we had only 2 classes the *Exact* training algorithm was used (see Matlab documentation for *fitctree* for more details).

Training models and generating output for each patient took in all cases under one minute.

The most important predictors across all decision trees were: correlation between channel power spectrums at dyadic levels, spectral edge, Shannon’s entropy for Dyads, mean value and 3rd estimate of fractional Brownian motion.

Discussion
----------
I tried to make both single subject models and general model as the general model would be more useful. Both approaches actually reached the same scores on leaderboard, which is probably caused by the fact that the general model was trained on all subjects on which it was then tested. That might allow decision trees to grow different branches for each subject at some part and thus reaching the same score as single subject models. When I tried to cross validate general model by subject (2 subjects used for training, 1 subject for testing) the results got much worse, around 0.6 AUC. In guess I would have to normalize the features to make the general model work for unseen subjects, but I did not have time to pursue this idea.
