# 11th place solution

Competition: trackml-particle-identification
Rank: #11
Source: https://www.kaggle.com/c/trackml-particle-identification/discussion/63303

Hi,
it's been a very interesting competition and as always these challenges are a very good place to learn. 
I tried many different approaches but eventually, as many of you, I used clustering (dbscan) on unrolled helix with z shifting and track extension.

I don't want to explain here the tricks that I used on the clustering part (some of them has already been shared by @yuval and @CPMP) but I do want to explain the feature that allowed me to increase the score from approx. 0.68 (obtained using the clustering algo) to about 0.76, namely a supervised track extension. 

**Supervised track extension**
The base code is similar to the one shared by @HengCherKeng where I integrated a gradient boosting tree (LightGBM) to establish if a given hit belongs to a given track. As input it takes approx. 60 features of the track (constructed using the clustering algo) and the proposed hit, and it outputs the probability of the hit to belong to the track. In this way, I could take into consideration a larger number of potential hits and let the algorithm decide which one is the best candidate.
I trained the decision tree over only 10 events and I used pretty naive features, meaning that it can be improved much more.

From a performance point of view, the clustering algorithm takes about 1h per event on a single core instead, the extension algorithm takes about 20min (at inference time). 

If you are interested, in the next days I will share the code.

Thanks to all of you who shared your ideas during the competition, cheers!

NEWS!

The code and documentation are available on https://github.com/andri27-ts/GoldTrackML
