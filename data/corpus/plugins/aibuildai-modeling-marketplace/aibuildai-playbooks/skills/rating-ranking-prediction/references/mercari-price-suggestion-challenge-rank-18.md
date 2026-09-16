# 18th Place with no Neural Nets - LGB and FM - 0.40604

Competition: mercari-price-suggestion-challenge
Rank: #18
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50252

So we didn't figure out how to get to 0.39X, but given that at the beginning of the competition I didn't expect anyone (least of all me) to crack 0.42X, I'd say we did pretty well.

We also made a solution without any neural nets -- just a Wordbatch FM scoring ~0.419 (thanks antiip!) and a custom LGB scoring ~0.410. The weighted average gets our score of ~0.406.

**Our kernel is here: https://www.kaggle.com/peterhurford/lgb-and-fm-18th-place-0-40604**

When DataGeek and I merged, we were surprised to see we both had roughly the same solution (LGB and FM average with similar LGB architectures), so we tried to combine the best features from each of our approaches. The general idea was building a LGB on all the categorical data and using Ridge models as variables in the LGB to represent the text data. We did 2-fold CV to get out-of-fold predictions to use in the LGB. We also did a lot of feature engineering for both the FM and LGB to learn what the TFIDF and Wordbatch would not learn, which is things like spacing, punctuation, ratios, etc.

Two great boosts to our model included using a Multinominal Naive Bayes submodel in the LGB to correct for the tendency of the Ridge models to underestimate and a `SelectKBest` and a regression to find the top performing individual wordbatch vectors and included them in our LGB. Both of these tricks improved our position by ~0.003 each.

Overall, our LGB had 48000 individual wordbatch vectors found via `SelectKBest` and a regression; 37 custom feature engineering variables made and tested by hand; target encoded versions of brand, category 1, category 2, category 3, and the first word of the item description; and 4 2-fold submodels (one Ridge trained on everything, one MNB trained on everything, one Ridge trained on just the name vectors, and one Ridge trained on just the item description vectors). We tried running the FM as a submodel in the LGB also, but it ended up surprisingly more valuable as a standalone model that we would average with.

We had a big scramble at the end to get Stage 2 ready (definitely should've started preparing for that earlier!) and took a 0.005 drop from our best non-ready submission. I'm still not exactly sure what happened there and would've loved more time to push us up further, but oh well. We ended up submitted two different approaches -- one with batched predictions and one with not. I was surprised the one without batched predictions made it and didn't OOM or run out of time, but both our submissions crossed the finish line. The one running in batch was only 0.001 worse.

There's also a lot more I would of loved to tune more -- I wish I knew more about wordbatch params and FM params as we weren't able to get much lift from hand tuning in those areas. We did notice however that we were able to get pretty good *speed* improvements from some FM tuning relative to what was in the public kernels.

Thanks for the awesome competition and kudos to everyone who managed to be Stage 2 ready. Big congrats to everyone who cracked to 0.39X - I can't wait to see how they did that! I wish there was less stress and uncertainty around Stage 2, but I'm glad we made it, and I overall did really like the Kernel-only format.
