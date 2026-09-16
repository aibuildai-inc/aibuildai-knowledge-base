# 11th Solution Overview

Competition: avito-demand-prediction
Rank: #11
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/60154

What a competition this has been - it was such a rare opportunity to be able to tackle image, text,  tabular and time series competition in one single competition.   I personally found that the ability to iterate and execute ideas as quickly as possible is really critical in a competition like this. and it is for the same reason that I really admire all the solo participants who did well here, none more so than @dmitrylarko, @kelexu and @wenbozhao   Despite managing to finish in the gold medal zone, it is obvious to me that the distance between us and the top teams is very big, so it is somehow humbling to see there is still so much room to improve.  We will definitely keep going to get better 

Our team names during this competition more or less reflect our journey in the last 7 weeks when we were active in the competition. First of all, 每天进步一点点means making small progress every day, while 挣扎到最后一刻 means fighting til the bitter end.  I don't think we have come across any single magical features that improve our score by more than 0.001,  while we were working frantically till the last seconds of the competition to make sure we can stay in the gold medal zone. The fight for the last gold medal positions was probably the most intensive that I have witnessed will multiple teams making sudden big jumps, so we are really happy that we manage to stay with them. 

I started the competition with @cxl923cc, @zhiqiangzhong and @yl1202, we are close friends and colleagues with each other and wanted to use this exercise to form a group that can share the burden of kaggle competitions (seems it worked out well). Towards the merger deadline, we formed with @mzr2017 and @oyxuan - because their modelling approach was quite different from ours, and the two groups complemented each other well. 

<h2>Feature Engineering</h2>
I will describe the features we have been working on within our original team and let our teammate add theirs in following threads.  Most of our features are already covered by other top solutions, but anyway I will provide a list here. The following are shared by all models:

 - text feature with TFIDF vectorizer for title, description and params,
   we  played around with many different combination of TFIDF
   parameters, and it all added to our model diversity.
 - SVD of TFIDF vectorizer features 
 - text statistical features such as length of text, number of number
 - text features on ngrams, text distance features 
 -  various groupby statistics between different categoricals 
 - aggregated features like the one shared in the kernels, 
 - LDA features 
 - Image features like the one that is extacted from pretrained models, as well as dullness, brightness
 - rolling statistics such as number of ad in the same category in different time windows

For non-NN models we specifically also created:

 - rnn extracted features： rnn features extracted from rnn architecture (Bidirectional-GRU, attention, global max, global avg), and feed into non-nn model as tabular features. 
 - sentence2vec features
both group of features played important roles in our models. 

For NN model, we made heavy use of pre-trained word embedding models - three variant of FastText models shared in external data thread, and the self-train embedding model trained on all data. 

Aside from the above, we have also added the following features trained on five-fold oof manner:

 -  "zero deal probability rate" - five-fold oof probability prediction on if the deal probability is zero. 
 - target encoding features - again five-fold oof target encoding features for each original categorical features apart from user-id 

All these oof features are used directly on level 1 of the stacking, along side with other lv1 meta features (oof train and test)

<h2> Modelling Approach </h2>

We have used a wide range of models/algorithms in this competition. LightGBM with different parameters, XGB, Wordbatch FTRL_FM, Ridge, LinearSVR, RNN with Keras, Fully-connected NN with keras.

In the last week of the competition, @mzr2017 found out that using xentropy in LightGBM, and building model by different parent categories helps to improve scores and give more diversity - and we started to retrained our better models with the corresponding settings.  We achieved our best LB model at public LB 0.2186 with xentropy training on all data - it was a model with 400+ features and some dense rnn &amp; sentence2vec features, and trained for more than 30 hours.  meanwhile Our best RNN model was around 0.2195 in public LB, courtesy of our teammates.

<h2> Stacking Approach </h2>
Stacking turned out to be very effective in this competition, and we were able to generate stacking score that is almost 0.004 better than our best l1 models. 

With the range of model/algorithms mentioned above, we generate 154 level 1 models. we found that almost all models add some diversity to stacking, and all efforts to trim the selection resulted in worse CV/LB, so in the end we went with including all level1 models. 

The stacking approach we used were very similar to the one that I have described in my [ensemble kernel][1] in the Porto competition. Except from the fact that we went for weight averaging for level 3 instead of stacking. I found that weight averaging always performed better on this level with a combination of lgb and keras level 2 oof. Didn't attempt to perform stacking beyond level 3, which we could have done - but this way we would have to start stacking a bit more earlier into the competition. 

We found that training with alternative objective functions on level 1, and retraining level 1 model by different combinations of category helped to increase the diversity.  Having discovered this we focused solely on retraining our existing models in the final day of the competition and managed to generate more than 40 models to add to our mix. This contributed a lot to our stacking effort with more 0.0005 gained in the last day. 


  [1]: https://www.kaggle.com/yifanxie/porto-seguro-tutorial-end-to-end-ensemble
