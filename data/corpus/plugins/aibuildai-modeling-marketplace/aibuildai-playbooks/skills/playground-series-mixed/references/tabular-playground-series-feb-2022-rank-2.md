# #2 solution

Competition: tabular-playground-series-feb-2022
Rank: #2
Source: https://www.kaggle.com/c/tabular-playground-series-feb-2022/discussion/310367

First off congratulations and well done to everyone who took part in this competition. I for one definitely got the opportunity to learn a lot of new techniques, so thank you to Kaggle for hosting this competition.

My final solution consisted of a voting ensemble of 6 solutions (two tree based, two basic clustering based, two optimised clustering methods).

**Tree based methods**
The two tree based methods I used were an Extra trees classifier, and a lightGBM classifier. Both were individually optimised by RandomSearchCV.

**Initial clustering**
Initially, I started with two tuned KNN clustering solutions, one using Euclidean distance and another using Manhattan, with the Manhattan KNN outperforming the Euclidean KNN when run individually.

**Optimised clustering**
I then tried to use a method that I thought would be appropriate for the data. As the data consisted histogram data, I started looking at distance metrics appropriate for histograms. Also, exploiting information about the similarity between the features, e.g. the difference between A10T0G0C0 and A9T1G0C0 is much smaller than the difference compared to A0T0G0C10. This lead me to a metric that I’d never heard of before called the Earth Movers Distance (EMD) and essentially measures the amount of work needed to shift one histogram into another. The logic behind this method being that identical species of bacteria will require less work needed to shift their histogram from one sample to the next compared to the amount of work needed to shift one species histogram to another. The challenge with this method was the high computational cost of calculating then EMD distance. It wasn’t feasible to work out the EMD between all samples, so instead I found the 30 closest neighbours for each sample by Euclidean and Manhattan distances and of these 30 found the closest neighbour by EMD and assigned it the same class. I suspect this method could have been improved further had I have had more time to compute all of the distances.

Once again, I’d like to thank Kaggle for hosting this competition and to everyone who took part. 

Best wishes,

Jamie
