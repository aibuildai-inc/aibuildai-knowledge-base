# "Dance with Ensemble" Sharing Thread

Competition: avito-demand-prediction
Rank: #1
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/59880

**Preface**

First of all, I would like to thank Avito and Kaggle for hosting such an interesting competition! Lots of ways to do feature engineering, data quality is not bad, and data size is arguably accessible to everyone. 

I would also like to thank my three awesome teammates, Arsenal, Georgiy Danshchin and thousandvoices (ordered alphabetically) ! You all are truly amazing feature engineering masters! 

Arsenal has been a long-time friend of mine and we decided to go with “Dance with Ensemble” again in honor of our last loss under the same team name (which was 3 years ago already!) I believe that time we dropped out of top 3 mainly because we didn’t use NN at all. And the funny thing is, this time (R)NN is one of the main reasons that we were always staying ahead of other teams.

**Summary**

So our approach is probably not different than what other top teams did in a high level. We have some lgb models, some NN models, some xgb models as first layer, and some lgb models, some xgb models and some NN models as second layer, and one NN as the final layer. But honestly, the complicated structure (3 layer) probably gave us about 0.0002 - 0.0004 improvement. Just several models with a simple linear stacking should be able to achieve not exactly the same, but quite similar score.

A few days ago our best single models were both 215X (on public LB) for NN and lgb. And then amazingly Georgiy Danshchin discovered a few features based on active train+test and immediately it boosted the best single lgb to 213X! Purely including them to my NN didn't help much but I couldn't find much time tweaking it (and honestly I was lazy at that point). I think in the end, it had 0.0007 ish improvement on our final score. I will leave this black magic to Georgiy Danshchin to disclose (Hint: RNN is involved there).

Stacking is extremely important here, which means that building diversified models are extremely important. I remember when four of us merged, just a linear blending of our models could get to 0.2133. 

**NN**

I exclusively worked on NNs for this one and didn’t do much feature engineering otherwise. So I would like to share how you can achieve 0.215X with a single NN, and leave the rest (truly amazing stuff) to my awesome teammates.

All features matter here. Text, categorical, numerical, images (and probably in this order).  And to my best memory, here is how I did it:

 - I got 0.227X with numerical features and categorical embedding
 - And then I included titile and description with 2 RNNs, with fastText pretrained embedding, with some tuning, the score dropped to 0.221X.
 - Played with self training fastText embedding on train+test, and also train active, test active. It turned out that self training on train+test was the best. Score got to 0.220X.
 - Added VGG16 top layer with average pooling. It made my score worse. Did some tuning, specifically, had a separate layer before merging text, image, categorical, numerical features together, and started to see the improvement . Got to about 0.219X.
 - Tried to tweak text models, with CNN or Attention etc. None worked. In the end, went with 2 layer LSTM followed by a dense layer. Probably 0.0003 improvement here.
 - Tried different CNN models for images. None of the "fine tuning" models worked (and GOD it was slow). But fixed ResNet50 middle layer helped by probably another 0.0005. Now the score became 0.218X.
 - Started doing all sorts of tuning (based on intuition mostly). And found that adding spatial dropout between text and LSTM helped quite a bit, probably 0.0007 - 0.001. And fine tuned dropout ratio overall helped too. In the end, about 0.001 - 0.0015 improvement here. So now the score was around 0.2165 - 0.217.
 - Started including all engineered features from teammates. Lots of engineered features from them (the ones based on text) didn't help but others did. So in the end, a NN with 0.215X!
 - If you kept saving models along the way (models with fewer features, models with more features that got worse result, etc.), you could train a fully connected NN on top of them and for me it was around 0.008 improvement in addition. In other words, you can easily get into top 10 with only NN!


I also attached a simple sketch of what the model architecture looks like. 


And this thread is **To Be Continued** by my amazing teammates!

Jump to: 

**Arsenal's approach:**
https://www.kaggle.com/c/avito-demand-prediction/discussion/59880#349563

**thousandvoices's approach:** 
https://www.kaggle.com/c/avito-demand-prediction/discussion/59880#349386

**Georgiy Danshchin:** 
https://www.kaggle.com/c/avito-demand-prediction/discussion/59880#349710

![NN Model][1]


  [1]: https://pbs.twimg.com/media/DgvX3pWUYAAlCAK.jpg:large


Once again, thank you all for this amazing experience!
