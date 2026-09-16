# 16 place mini write-up (before LB swaps)

Competition: santander-value-prediction-challenge
Rank: #16
Source: https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63760

Congrats to the winners and especially to Gilberto for winning as well as for reclaiming the top 1 spot (if my calculations are correct). 

Also kudos for sharing the leak and levelling the field. At the end of the day I think ml (probably) played quite some part  in getting a good score past the leak and that is still useful to the organizer.

I also hope they fix the issue with people selecting different submissions after the end of the competitions and moving onto the leaderboard including me (https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63752)

My solution consisted (like most people I presume) of 2 parts:

Finding the leaky rows
======================

I have attached all the patterns I managed to find (4390 columns) that helped me reach the 3887 leaks in the training data .  I found them based on rows that I knew were sequential  (which I knew from the columns some people shared in kernels).

It took me around one day to find them via exhaustive brute-force search and some heuristics to make the search faster (like group together columns that tend to have similar unique values). 

I used the Jiahzen kernel to find the leaks and mark them .


Modeling
======================

This was used for when a leak was not found.

I Built a few lightgbms and xgbost models using aggregated values over the whole rows (like means, stds, kurtosis , count of zeros, skewness , mode etc ). 

I created a few times series features on the first 40 rows namely:

     ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'],

 it included exponential smoothing and weighted moving averages on different lag selections.

I had some string features on the whole row, like count of unique numbers in values, count of values with more decimals, count of integers .

I made a few deep learning models (with lstm and gru) using these 40 features as inputs. 

Then some PCA features along with kmeans and other clustering features.  

**I should mention that I validated on the test data too**, since I knew the leak had most probably  assigned  the correct labels for these cases. I also assessed which public kernels were better and final weights based on the leaky test values too.  

Stacking
======================

i stacked 15 different models . I did a 2nd level stacking with an nn, a lightgbm model and Extratrees. I averaged a few of the public kernels too.
