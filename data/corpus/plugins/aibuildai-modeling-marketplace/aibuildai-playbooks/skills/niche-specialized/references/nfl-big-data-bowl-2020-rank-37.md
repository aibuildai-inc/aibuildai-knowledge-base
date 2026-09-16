# Private LB 37th Place Solution

Competition: nfl-big-data-bowl-2020
Rank: #37
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/125162

First off, congratulations to the winners and thanks to NFL &amp; Kaggle for hosting the competition!

This was one of my first major Kaggle competitions and was quite surprised to end up in the top 50 so thought I’d share a few details of my solution (I waited until the very end for fear of a disastrous code break in the final weeks). I started 52nd on the public LB and ended up in 37th.

I focussed mainly on feature engineering with the spatial variables and ended up passing these into a fully connected 128 (dense) x 64 (dense) x 199 (softmax) neural network.

One of my regrets was spending so much time on an initial model which had fundamental issues with the features and expecting small tweaks to yield a breakthrough.

An entire rethink of my feature engineering yielded a breakthrough when I finally got a grip on the spatial and temporal features.

The first portion of my feature vector as input was fairly standard rusher specific info which you would expect, but the majority of the vector was filled with spatial counts (both boxed and radial) of defenders relative to the rusher. This was done using box and radial counts at T = 0, T = 0.5, T = 1.0, using the speed and direction field to ‘move’ the play forward in time.  

Pictures always make more sense:







Adding a few aggregate stats (min, mean) on the distance of defenders relative to rusher at each time gave me around 80 features in the final model.

I had wanted to incorporate the offensive players, but simple counts didn’t yield me any significant improvement. More nuanced treatment was needed, given their role as blockers.

Beyond this, I spent most of the time making my code as robust as possible (rather than trying any particular tweaks – data augmentation, post processing etc. ) which seemed to turn out ok!
