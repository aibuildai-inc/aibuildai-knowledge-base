# The 3d place solution

Competition: kkbox-music-recommendation-challenge
Rank: #3
Source: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/45971

Thanks KKBox, Kaggle and WSDM for this competition! 

The main hack of this competition is that we can generate features from future. Data is ordered chronologically. We can see, for example, listen or not the same artist in the future.

I use xgboost and catboost. Fit their on features among which was matrix_factorization, where I user LightFM. Catboost is better than xgboost for about 0.001. 

I use last 35% of train for generation table of features, where the rest of the data was used as a history. For test – train is history. For validation analogous scheme, but the size of X_train, X_val was about 20% of data.

From hisotry it can be generated features using target. Other features can be generated using all data. I generate features in the context of categorical features (their pairs and triples):

1)	mean of target from history by categorical features (pairs and triples)

2)	count – the number of observations in all data by categorical features (pairs and triples)

3)	 regression – linear regression target by categorical features (msno, msno + genre_ids, msno+composer and so on)

4)	time_from_prev_heard – time from last heard by categorical features (msno, msno + genre_ids, msno+composer and so on)

5)	time_to_next_heard – time to next heard by categorical features (pairs and triples which include msno)

6)	last_time_diff – time to the last heard by categorical features

7)	part_of_unique_song – the share of unique song of artist that the user heard on all unique song of artist

8)	matrix_factorization – LightFM. I use as a user msno in different context (source_type) and the msno was the feature for user.

I use last 35% for fit. Also I drop last 5% and again use 35%. Such a procedure could be repeated many times and this greatly improved the score (I did 6 times on validation and this gave an increase of 0.005, but in the end I did not have time to do everything and did only 2 times on test).

As a result, I fit xgboost and catboost on two parts of the data using and without the matrix_factorization feature. And finally I blend all 8 predictions. 
Sorry for my English. 

The code is available here:

https://github.com/VasiliyRubtsov/wsdm_music_recommendations/blob/master/pipeline.ipynb
 
But I don't check it yet :)
