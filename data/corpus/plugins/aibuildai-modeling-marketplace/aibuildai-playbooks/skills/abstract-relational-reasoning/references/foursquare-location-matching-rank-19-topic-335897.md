# Why was the private train set replaced?

Competition: foursquare-location-matching
Rank: #19
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/335897

As already pointed out in the leak thread, the train set (not test set!) of the private run has been replaced with different data than the public.

@tkm2261 has [pointed out](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335799#1847503) that there is a precedent in a past competition, but in that competition, the data description clearly [states](https://www.kaggle.com/competitions/landmark-recognition-2021/data) that train.csv will be replaced, whereas in this competition, there is no explanation at all. Frankly, **replacing private train set without any annoucement** was beyond my imagination.

While irrelevant for many participants who do not use train.csv in their inference notebook, this is a very unfair situation for solutions including our team that use train.csv during inference time, such as target encoding.

The results of the post-competition LB probing revealed the following facts:
- The private train set has a different number of rows than public train set
- The private train set has more than 1 million rows.

What exactly is this private train set? I believe this is a bug introduced by the host, not an intentional one (is it a draft version of competition data?).

Whether the host should fix the bug and rescore is a sensitive issue, but the host should at least explain the cause for the change. @zisispetrou @addisonhoward
