# 5th Place Solution Summary

Competition: quora-question-pairs
Rank: #5
Source: https://www.kaggle.com/c/quora-question-pairs/discussion/34359

Hey there,

congratz to the winners and especially to the **"DL guys"** for an impressive victory. Well done! Thanks goes to my teammates Kaza and Qingchen for another nice "kaggle experience" and a special one goes to the guys from [H₂O][1] (they know why ^^).

That one was a tough and fast paced competition with quite some twists and lots of things to discover, which made it really interesting but also a bit exhausting. ^^

Our final model is an XGB with ~600 features, of these ~25 are oof-model predictions (LightGBMs, NNs/LSTMs and some SGDs) or likelihoods and the others are "normal" features. We used an oversampled training set with prior ~0.13 to train our submission models and we validated on the last *n* rows. Out-of-fold predictions have been created via 5 stratified folds on non-oversampled data.

For the NLP part, it turnt out the be very useful to create features based on differently pre-processed questions texts (raw, interrogative forms, stemmed, cleaned, stopwords only, stopwords removed,..) and token bags (shared &amp; non-shared tokens). For instance, count vectorization applied to those bags yielded very predictive features. Another source of NLP-like feature extraction were the longest common subsequences (lcs) of both questions in a row computed via dynamic programming. This was also useful to enhance the token matching for questions pairs with grammar or punctation errors in only one of them or just to detect those and creating stats upon. Especially, the mutal information of lcs and orginal question texts was useful.

Beside some silly-looking features like min(qid1, qid2), qid-deltas and indices-deltas from a sorted list of all questions texts, it was predictve to categorize each row based on the train-test-appearances of both questions like "both questions appear in train &amp; test". In general, train-test-split information carried alot of target information but was very prone to overfitting. 

As generally known, the book of spells has been the underlying graph structure of the question comparisons. Beside the stats like |common neighbors|, |unique neighbors|, |paths of length *n* between q1 &amp; q2|, max. clique size, component size, etc. pp. we put each of our features as weights *w* to the edges and computed stats based on that (for instance *mean(w)* of common neighbors). <br>
We also used out-of-fold predictions as edge-weights to get something ouf of the transitive relation y(q1, q3) = a = y(q2, q3) =&gt; y(q1, q2) = a (which is inconsistent given the ground truth). Those features provided a significant gain. Last but not least, we treated strongly connected components in the graph as markov chains (state transitions could be an oof-prediction or an feature) and re-weighted given features with the estimated steady state distributions of those MCs. I don't know yet, if that added anything, but I found it interesting enough to try. 

[**???**][2]

cheers, <br>
Faron


  [1]: https://www.h2o.ai/
  [2]: https://www.quora.com/What-does-it-feel-like-to-be-addicted-to-Quora
