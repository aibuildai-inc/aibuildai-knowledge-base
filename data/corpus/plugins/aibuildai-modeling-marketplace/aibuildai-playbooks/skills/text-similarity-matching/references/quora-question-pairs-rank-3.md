# Overview Of 3rd Place Solution

Competition: quora-question-pairs
Rank: #3
Source: https://www.kaggle.com/c/quora-question-pairs/discussion/34288

I'll update this post with more detail and the code when I have documented it better, but I figure I'll quickly describe my part of our solution. I'll focus on my parts, Sean provided very good NLP, nn and graphical features so I'll let him describe these.

Overall we used around 1300 features for the 1st level models, which were nn, lightgbm and xgb models. Lightgbm worked really well for this competition, being up to 5x faster than xgb and only a little less accurate. These features were primarily NLP features and what I'll call meta-features. The meta-features were frequency features and graph features, such as intersection count, the frequency of intersecting questions, question frequency, question frequency just for q1, question frequency just for q2 and so on. NLP features were things like matching words, if the word before a "?" matched, frequencies of matching words, frequencies of not matching words, how similar matching and non-matching words were, etc.

These models, roughly 15 of them (some nn were bagged, I'm counting that as 1 model) were then stacked together. Before stacking we introduced features such as q1_hash_by_mean_pred, q1_hash_by_min_pred, sum of the question hashes etc. Our best single model was an xgb model and got around 0.185 on cv, stacking got down to 0.157 on cv. It was important to include meta features when stacking, as some models were built just on NLP and this allowed them to still be useful. Sean also found a feature in the last couple days which was related to the OOF predictions of NLP models to the intersections somehow, and this provided significant gains.

Edit: Another interesting find by Sean was that averaging the stacked models with the individual models (so the final ensemble was 0.75 stacked + 0.25 * (base models)). This lead to gains of ~ 0.001 on public lb, but harmed cv. So doing this somehow ensured better generalization, still not sure exactly why but it did also translate to private lb.

For one idea which probably a lot of teams did not do, was to selectively adjust our predictions based on question frequency. There was a systematic difference between lb and cv, and it was somehow related to question frequency. For instance we used the link function (adjust the odds down) by a factor of 4.7 when the question frequency sum was 2 (both questions unique) and scaled this gradually down with increasing question frequency.

That is all for now, there is a bunch of other things we did and I'll try and post the code when it is cleaned and documented. Thanks for a fun competition, even if it was about more than just NLP.
