# 14th solution，CNN+finetune+FM

Competition: mercari-price-suggestion-challenge
Rank: #14
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50275

My solution is an ensemble of CNN and FM.  The most important part is finetuning on CNN.

1. CNN
------

1) embeddings of field `name` and `item_description` are inititalized from glove.twitter.27B.d.txt, I choosed the `twitter` version sinced twitter is UGC, which may be like customer reviews.
<br>2)  each feature including shipping and item_condition_id are embeded into a 50 dim vector.
<br>3)  single layer cnn, used 40 filters of size 2 and 40 filters of size 3, I haved tried filters with size 1 and size 4, not too much improved, GlobalAveragePooling1D was used.
<br>4)  concate every feature and give them to BN-dense(512, relu, he_normal)-dropout(0.1)-BN-dense(64, relu, he_normal)-dropout(0.1)-BN-dense(32, relu, he_normal)-dropout(0.1)-dense(1, linear, he_normal)
<br>5)  I found RMSProp worked better than Adam with my features, could imporve ~0.001
<br>6)  epochs=3, batch_size=2048
<br>7)  rmsle is between 0.415~0.418 on my CV, the result is not stable.

2. Finetune
-----------

1) since different category has different price distribution, After train the CNN model for 3 epochs, I trained each category for 2 epochs with the same architecture, only on items of the exact category. 
<br>2) This improved rmsle to ~0.408 on my CV. The result is more stable.

3. FM
-----

1) folked from [anttip's solution][1], It is amazing. Thanks a lot.
<br>2) rewrite the `wordbatch.WordBatch` function to let it run faster
<br>3) rmsle is  ~0.425

4 Ensemble
----------

1) CNN(finetune) * 0.6 + FM * 0.4, to get 0.399x~0.400x on my CV, and 0.401 on LB

  [1]: https://www.kaggle.com/anttip/wordbatch-ftrl-fm-lgb-lbl-0-42555?scriptVersionId=2122276
