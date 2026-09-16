# 7th place The Zoo

Competition: santander-customer-transaction-prediction
Rank: #7
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89023#latest-518634

I was planning to do a much longer writeup like I did in Quora, but disappointment about not making top 5 is still too high, specifically as it is only 0.00005 gap. I only browsed the solutions of top teams and I think we do most of the things very similarly, but apparently failed on combining them to boost us on top.

## The "Magic"

- Count the feature frequency
- Only count for train + nonfake test data
- Profit (and be creative in how to use these features)

## Models

Given our time being in top range, I assume we have the most different set of models. For a long time, our models focused on modeling each feature separately and then combining the individual predictions with product or mean logit. Here are a few models we use:

**LGB** 
Not much to say here, LGB on top of features + counts. 

**NN**
This is a structure we came up with quite early and that was for a long part very important to the submission. Basically it is same idea as before, we use 200 Dense layers for each feature separately including large dropouts (0.8) and then just have a final layer lazily combining them.

**NN-Boost**
We use abovementioned NN structure, and predict the 200 Dense layers separately for each feature. So we end up with 200 * 512 features. We then fit an LGB separately on each feature and combine predictions. This also made it to final blend, and I am happy about that because I also wanted to make such a model work :) 

We played A LOT around with spline regression, kernel regression and similar things. Basically the idea here is that we model for each variable unique and non-unique values separately as they show different functions to be modeles. So overall we have 400 models.  

**Logreg**
This is modeling the features based on some knot transformed additional variables.

**Splines**
We transform the features with a kernel and knots to spline feature and fit linear reg on top.

**KDE**
We calculate KDE of unique/non-unique separately and multiply the probabilities.

**Naive Bayes**
This is a model that was also for a long time very useful. It is a hand-crafter NB that calcualtes conditional probabilities on a smoothed version of different bins. So you bin the data (again separately for unique/nonunique) into for example 5-15 bins, then calculate NB probabilities for each bin separately, and then average.

**NN**
Of course we also had NNs fit on single features, but they were slightly too weak in the end.

Probably many other version that I now can't remember.

Problem of overfitting:
Unfortunately, with doing 200 models, it is very very easy to overfit. As an example, if you do LGB and early stop each variable separately, you can go to 93 in CV, but sucks in splitout test or LB. So we mostly tuned hyperparas by taking the same one for each model.

Only in the last few days (probably too late), we went back a bit and tried to fit all features together. Here we two important models for final submission.

**Pivot LGB**
@Giba had the idea (and I think this is similar to very top solutions) to fit one LGB on all features together, but just model it as one feature and have a separate indicator of which feature it is. The best one here had CV of 926.

**Full LGB**
Instead of using additional count features, @dott had the idea to simply mask unique/non-unique features and produce additional features, so ending up with either 400 or 600 features. This model is also an important part of final blend.

**Creative stuff:**
We had so many more creative and cool ideas. One is by @dott who changed underlying LGB source code to only combine feature+count in each boosting round as there are no interactions. This was a bit too weak though for the blend.

## Features and tricks

We tried so many different features and nothing else helped. I don't even want to talk much about that because it is a bit frustrating.

Regarding tricks it is a bit unfortunate for us. Of course we also tried things like pseudo tagging which also gave boosts in CV but was not that helpful on LB so we dropped it. Apparently it could have given us a slight boost and we should have experimented further with it. We also had this idea of using pseudo tags for early stopping which looked promising, but again was not that good on holdout or LB. We never tried to add those things to the blend though and probably we should have. 

We also only use train augmentation rarely and basically only for the Full LGB model. For the single fit models it does not help, because the model already learns 0/1 classes separately. We were also thinking quite a lot about doing test augmenation, but couldn't find a proper way. I am sure there is some room for improvement here.

## Final blend

The final blend is a weighted combination of several different models from different categories from above. We use a hill climber algo for combining, and our top private solution is also our top CV solution with a CV score of 0.927.

## The magical night and the following despair

As most of you know, we were the first to breahc the 924 gap, and it all happened within a few hours. Basically @dott found the count features and could see nice boosts on CV, but as many of you we could not directly utilize it for LB. We were tinkering around and finally a few hours later I found the 200k fake examples in test. I quickly fixed the counting and submitted a full feature LGB and boom 912. Dmitry was at Ikea and he said he will further boost it when he is home. I decided to play a quick game of Apex (which I won - I btw haven't played since) and waited. And voila when he got home we got to 920 or something by fitting each feature separately and taking product. Then a little hyperpara tuning 922 and then adding the NN model from above 924.

What came next was frustrating. As fast as we found this solution, as long we were stuck on 924. We tried so many things but just couldn't get our score up. I think we were stuck for around a month at 924. I guess all top teams know that things get hard after 924, and at least we could crack 925 in the end. But I feel like we should have had more of an advantage of finding and combining the puzzle pieces so early. Specifically the kernel about fake test data was a hit in the face for us :)

I want to sincerely thank my two partners @dott and @giba who are both incredible data scientists and combining our ideas and also slightly different ways of warking was very fruitful and I could learn a lot. I am sure I have forgotten many things already and might add a few details here and there if I remember, maybe my teammates will also chime in or make separate posts.

Clean version of first 922 model fitting within ~3 minutes by @dott:
https://www.kaggle.com/dott1718/922-in-3-minutes

LGB by @giba:
https://www.kaggle.com/titericz/giba-single-model-public-0-9245-private-0-9234
