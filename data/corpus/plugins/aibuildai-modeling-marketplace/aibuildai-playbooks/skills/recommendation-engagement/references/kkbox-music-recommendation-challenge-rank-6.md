# My solution without time series assumption

Competition: kkbox-music-recommendation-challenge
Rank: #6
Source: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/45999

Hello everyone!

Thanks to KKboxs, WSDM and kaggle for this competition. Congratulate all competitors. especially, @lystdo, @Magic Recommenders and @rubtov Vasiliy. 

This is the first Kaggle competition that I really attend.  As a Master Student, I am really lack of experience for my first Kaggle competition. At the end of the competition, I realize that there is a strong time order in the dataset. But it is too late. 
So my solution did not use any time series assumption in the dataset.

I will release all the codes for feature engineering and model later, if allowed.

**Model**

I use an ensemble (average) of lightgbm.  The best single model public LB 0.726

**Feature Engineering**

1) raw features for both song, user and context

2) SVD matrix factorization latent space of user-song interaction matrix

3) SVD matrix factorization score of user-song interaction matrix

4) collaborative filtering (CF) score (top 100 neighborhoods ) of user-song  interaction

5) similarity of artist, lyricist, composer, genre,language, source_system_tab, source_type, soure_screen_name,within each user.

6) source_system_tab, source_type, soure_screen_name categorical variable rate  of each user.

7) word embedding of of  artist, lyricist, composer, genre.


I use 10 cross validation, where train / valid set are splitted randomly. 

I wish you can understand.

2 stupid questions about When and how to submit the code and How to submit paper to workshop?
Thanks
