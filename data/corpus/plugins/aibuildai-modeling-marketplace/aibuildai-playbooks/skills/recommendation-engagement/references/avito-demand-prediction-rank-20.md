# 20th place solution

Competition: avito-demand-prediction
Rank: #20
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/59936

Firstly thank you to Avito for hosting such an awesome competition and thank you to both my teammates for all their hard work. I'll try to keep this brief.

**Price model from active data**

I actually wrote about this [here][1] and am surprised that many others didn't do this - it was perhaps our single biggest success. Whilst a price model on active data was quite poor in error metric terms it made a large contribution to our score. We used NNs on all train and test active data and Rob trained a per category xgb price model after extracting lots of text features (numerical fields for things like cars/houses etc...) which helped a lot.

**NNs and LGBM/XGB**

Like others we had good success with NNs and trained a variety on different text embeddings (fasttext cc and wiki, self-trained w2v with and without stemming). We didn't add many other features to the NNs but did use a variety of architectures with and without attention. Modelling raw log1p(price) as a categorical and then embedding it worked well too, as did lower batch sizes and averaging predictions over several epochs. It was my first time using DL in a competition so I was very happy to be reaching ~0.2195 with a NN without many new features.

**CV/stacking**

Everything was done on a 5 fold basis and using a lgbm stacker. We got a lot of success from this but perhaps could have been more careful in trying to construct diverse base models and using a more rigorous feed forward selection. Antoine in particular did a lot of work ensuring we were robust in what we finally included.

**Other thoughts**

 - We did try to train models to predict period from the active data to no avail.
 - We didn't use images beyond basic features
 - We tried translating all the text, probably too late, but google's api gave up on us. The hope was the translation, whilst average, would help with the cases in the Russian language - as well as letting us finally understand what was being sold!
 - We trained models on a few different targets (e.g a NN to predict if an item was a 0 or not, treating target as categorical) which then went into the stack. The pdf entropy from these classifications was a good feature also.
 - We did train a user_id model (only using users that were in test and train) but ended up excluding any models which used user_id as were still finding too unstable CV-LB relationship. Be interested to hear if anyone included it successfully.
 - I second some of the other comments about workflow: we saved all our transformations down and then simple read in .pkl files and concatenated. Some of the transformations were expensive and this allowed us to get going quickly.
 - [Gensim][2] is a cool package, though slightly steeper learning curve than sklearn.
 - A note on lgbm: we actually found better results by sometimes excluding strong categorical features, particularly from the stack. My intuition for this is that given lgb fits in a leaf-wise (best first) manner it perhaps over-uses categorical features which always offer easy splits but ultimately might not be the best choice/don't allow slower exploration of weaker features or more subtle relationships. Keen to hear what others think of this.

I'll let Rob and Antoine add anything if they see fit. Thanks again for a fun competition!

Mark

  [1]: https://www.kaggle.com/c/avito-demand-prediction/discussion/57410
  [2]: https://radimrehurek.com/gensim/
