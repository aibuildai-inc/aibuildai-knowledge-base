# 8th place solution

Competition: ubiquant-market-prediction
Rank: #8
Source: https://www.kaggle.com/c/ubiquant-market-prediction/discussion/338236

First of all thanks to all the people organizing and running this competition, it's been a fun experience for me throughout regardless of the result.

My model is a weighted ensemble of 10 of the same LGBM, 30 of the same NN and a handbuilt model. The reason for taking so many of the same model was to try to average out some of the randomness due to initial conditions.

Both the NNs and the LGBM used the same features, but because of the 16GB memory limitations of kaggle notebooks I actually had to split the dataset into two. This, gladly, during testing on smaller subsets of the data, didn't seem to have any noticeable impact on the metric score which could be distinguised from random chance.

The features were:

- the features as given by ubiquant
- the average of each feature per time id
- the rolling average of a few hand-selected features over time  for the particular investment id minus the current value of the feature

The reason for taking the average of the features by time id was to have some pool of performance to compare yourself to, probably better results could be achieved if you sorted this by sector.

the hand selected features were the following:
33, 120, 225, 242, 266, 293
and
15, 31, 83, 157, 164, 189, 197, 215, 226, 231, 237, 239, 243, 250

The process of finding these features was to go through each of the features and look at whether, when the feature value is higher or lower by some amount from the rolling average of the feature until that point, whether that correlates with a higher probability of the target then being either positive or negative.

More specifically I looked for properties I considered to be "good" properties, which basically means that when the value goes up by some amount I want the probability of the target being either negative or positive to go up. Any features that were jumpy, in the sense that they didn't consistently follow the pattern of higher or lower value->higher probability, were discarded. I did this with every feature over many different baskets of investment ids, and when, for any of these batches, the "good" property was lost, I discarded that feature.

Using only these handmade rolling average subtracted features and summing them all up the "good" property was of course preserved and I could predict the sign of the target up to a point of about 70% accuracy... although at that point the amount of times this opportunity would come would be less than 1% of the cases. Which, considering that we're trying to predict future market results, seems like an entirely reasonable number for a well working model.

For me, personally, this would have been my final model because I thought that with purely anonymous features you really couldn't do much better. But this only translated to a metric score of about 0.08 which was far below what could be achieved by just naively putting the data into your favorite machine learning model.

For these ML models however, some of the "good" property got lost! One way to think about this is that when the output of the ML model goes up it should translate to a higher probability of the target being positive and vice versa when the output goes down. The problem is that it just hovers between 50-57% not even breaching 60% let alone getting anywhere near 70% my hand-built model could achieve. The good thing however is that the pearson correlation metric was a lot higher for these ML models. To try to fix that problem a bit I just added my handbuilt model on top of the ML models when my handbuilt model reaches some threshold like 60% probability.

So my final model  turned out to be:

**((10xLGBM)/2 + (30xNN)/2)x0.92 + (handbuilt model)x0.08**

Low weight for the handbuilt model because the predictions tended to be higher in absolute terms. This at least made it such that it manages to breach the 60% probability mark.

For validation I used the last 25% of the data which gave me a CV of around 0.17 

For the final submission I used the entire dataset which translated to a LB of around 0.15.

Finally to make use of the two submission that were given I used one submission for a bear market by multiplying all negative predictions by 1.4 and dividing all positive predictions by 1.4 and vice versa for a bull market.
The reason for choosing the value of 1.4 was to just go through a bunch of different values from 1 to 2 and seeing how that affected the CV. It's been a while but I think on average it was something like:

1.1: +0.001
1.2:+0.002
1.3:+0.003
1.4:+0.0035
1.5:+0.003
1.6:+0.001

In the case of my final submission it seems to have made a difference of something between +0.001 and +0.002.
The ranking over the 3 months was something like 
~900 LB ->~100->~20->~30->~20->8

Without knowing what any of the features mean it's hard to tell if anything I did made any sense and I believe that much better things could be done when knowing the features' meanings.
