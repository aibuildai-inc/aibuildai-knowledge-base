# 2nd place solution - brief summary

Competition: playground-series-s3e1
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s3e1/discussion/377179

First of all, thanks to the organizers for a fun, short competition. Also thanks to the many notebook contributors in this competition!!!🎉

My final solution is an ensemble of some public contributions with some of my own personal ideas. 

-I felt like the boosting path was well-explored by many public notebooks and designed to base my boosting approach on these methods; as mentiond, distance using coordinates plays a big part.

-I noticed many people added the external data to their dataset and computed their CV score using this data. Since the competition data is an **adaptation** of the original dataset, I think this is why some CV scores weren't very well aligned. Instead, I ran a CV split on the original data, and added the external data to the training set afterwards. This forces validation on the supplied dataset, better representing the LB score. Ensembling this with methods that split on the full merged data seemed to diversify a lot and improve LB.

-Lastly, I also training a NN in keras using [keras_tuner](https://keras.io/guides/keras_tuner/getting_started/) on the standard features + coordinate features. Local CV was only 0.59, but this also added significant diversification in the full blend. Model summary can be seen below:

.png?generation=1673332971801417&alt=media)
