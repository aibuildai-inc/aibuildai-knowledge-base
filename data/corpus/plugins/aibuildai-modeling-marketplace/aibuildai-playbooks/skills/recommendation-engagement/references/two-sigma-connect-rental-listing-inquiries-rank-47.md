# Solution sharing (#47, w/source code link)

Competition: two-sigma-connect-rental-listing-inquiries
Rank: #47
Source: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32104

This was a great competition, and congrats to the winners!

Here's my solution - it's a bit rough, since I was writing code in a hurry near the end of the competition (I remembered that I could do out-of-bag validation for my stacking basically the last night) - and it's definitely "B" quality in code organization, but it's still got some interesting stuff IMO.

Source code's at https://github.com/happycube/kaggle2017/tree/master/renthop

### Best 'single model'... with a 'huh?'

My best single-model lightGBM (an average of 5 folds) got .0.51072 privateLB, 0.51050 public, 0.50514 CV... 

... and after feeding that model to my XGB stacker it's score dropped to .50678priv/.50715pub/0.501258042911 CV.  The lightGBM version was almost as effective (.50762/.50796/.50208) 

 So the stacking model provides about half the score gain from stacking, without even actually stacking models - it's like prior correction on steroids!

### Things I did:

- I wrote a stacking class (~100-150 LOC) and formatted my models to output data compatible with it.  It was nice and orthagonal until last night* - in that I could concat the train+test output and use it for L2, and the test output (with listing_id as index) could be submitted directly from the dataframe's CSV output.

Originally I used a Keras NN, but wound up going with lightgbm for testing and xgb for submitting (I got the NN idea from my CNN Description processing, which I didn't actually use in my final submissions, and lopped off the bottom bit of it to make a stacker... which gave me another key idea:

(* - I decided to keep each fold's test predictions seperated, and I kludged the models output to be the original dataframe along with an np.array.  Crude but it actually worked.)

- I made both classification and regression models, relying on the stacker to convert 0 to 1 into the three categories.  This allowed me to make two somewhat different models with xgboost and lightgbm.  (edit:  It didn't help as much as I thought - stacking the two lightgbm models only added about .0006)

- In addition, I ran with the idea behind https://www.kaggle.com/luisblanche/two-sigma-connect-rental-listing-inquiries/price-compared-to-neighborhood-median/comments, and made an xgboost model that predicted the fair value by basing it on only the medium category.  My thought was that it would give models something to anchor to, and it worked.

- I also cleaned up the TF-IDF features feature engineering a bit.

### Things I used:

- Bits and pieces of various starter scripts, including the lightgbm starter.  Since I do my non-GPU Kaggling using the official Docker container, I didn't have any setup issues.

- Kaz's L1 StackNet models, with my own stacking code.

- The leak.  I have the picture .zip file and could have pretty quickly reproduced Kaz's data, but I just used his .zip.

- Adams' features posted yesterday.

### What I could have done differently

- Gotten used to stacking a lot sooner (true for Kaggle as a whole really)

- Realizing I could (and should!) do an out of fold CV to verify quality of my stacking before the 11th hour ;)   I implemented it Monday night so I wasn't able to utilize it to tune my models as much as I should have, but ah well.

- Made more usable models to feed the stacker.  That and a bit more feature engineering might've gotten me into gold metal contention.

### Some thoughts:

- Had trouble with increasing overfitting as my score went up.  By the end I was at about .01/.009 public/private LB overfit.  I probably should have varied features more between models.

(post subject to further editing)
