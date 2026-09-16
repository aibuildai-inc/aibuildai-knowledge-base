# 18th Place Solution - Careful Ensembling + Resampling Diversity

Competition: porto-seguro-safe-driver-prediction
Rank: #18
Source: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44579

Edit: [here][1] is a blog post I wrote describing some aspects of my solution in more detail. It's focused on stacking, entity embedding neural networks, and resampling strategies.

This was great fun and an incredible learning experience. What an awesome surprise to land in the top 100, let alone top 20. I hope that this post is useful!

My approach was influenced by many public kernels and discussions, so I owe a huge thank you to the community and many individuals. To mention specific people whose kernels I directly used or adapted in my solution -- [Andy Harless / Olivier][2], [Bojan][3], [xbf][4], and [Snow Dog][5]. 

This last kernel deserved a lot more attention. Building models that used its sandwich downsampling technique made the difference between top 100 and top 20 for me. Looking back on it, I'm quite surprised that there wasn't more in-depth discussion of resampling strategies given that class imbalance was a central part of this problem.    

My final solution included 16 base models, stacked with a regularized logistic regression. I decided early on that trying to get too fancy with stacking could lead to a bad overfit. I also prioritized trying to find model diversity over hyper-tuning individual models. Most models were trained with the 5-fold cv / average method seen in the kernels. To select which models to add to the stack I looked at spearman correlations of the test predictions and 5-fold cv scores (with a final mean cv of .2924, not far off my private score). I carefully chose which public kernels to consider including based on their having good cvs.  

I did a tiny bit of "feature engineering", but it was mostly data processing and probably didn't add much. For some models I combined the ps06-09 and ps16-18 bin variables into single category columns (these were clearly one-hot-encoded or functionally equivalent to it), and I also imputed ps_reg_03 from the other reg features with a simple linear regression. I also sometimes added interaction terms based on boosting feature importances. I added a sum of null values column, and dropped the _calc columns. 

Diversity in my stack comes from 3 main sources: 1) different models, 2) different features 3) different resampling of the training data. Here's a breakdown, most models used 2x resampling or class weights unless stated otherwise:

**Gradient Boosting**

4 LGB models:  1 with OHE categoricals, 1 with target encoding, 1 with entity embedding features, 1 with 5-fold sandwich downsampling (so total of 26x5 = 130 individual models).

3 XGB models: 1 with OHE (xbf's), 1 with target encoding (Andy / Olivier), 1 with sandwich downsampling.

**Neural Network**

1 entity embedding network, basically my [public kernel][6].
 
**Regularized Greedy Forest**

3 models: 1 with OHE, 1 with target encoding (Bojan's), 1 with the sandwich.

**Field-Aware Factorization**

2 models: 1 without resampling and 1 with the sandwich (yes, writing a shell script to train these 130 models was very fun).

**The Stuff That May Have Been A Bad Idea**

In the end I think I took resampling diversity a bit too far, and actually included some LGB models where I upsampled the 1s enough to be the majority class in a fold. Those are the remaining models in my stack, and though they improved my CV from .2920 to .2924 and my public LB score, they worsened my private score. I think in the CV they only really improved my score on one fold, so I should have been more skeptical. If I had stuck with my highest LB score solution 3 days ago instead of my highest LB solution today, I would have finished at .29183, in 11th place.        
 

   
I hope that this writeup is helpful and please let me know if I can answer any questions about what I did. I may write about some of the methods I used in more detail in a blog post that I'd share here later on. 

Thanks again, and happy kaggling!


  [1]: https://jeddy92.github.io/JEddy92.github.io/seguro/
  [2]: https://www.kaggle.com/aharless/xgboost-cv-lb-284
  [3]: https://www.kaggle.com/tunguz/rgf-target-encoding-0-282-on-lb
  [4]: https://www.kaggle.com/xbf6xbf/single-xgb-lb284
  [5]: https://www.kaggle.com/snowdog/xgb-sandwich
  [6]: https://www.kaggle.com/aquatic/entity-embedding-neural-net
