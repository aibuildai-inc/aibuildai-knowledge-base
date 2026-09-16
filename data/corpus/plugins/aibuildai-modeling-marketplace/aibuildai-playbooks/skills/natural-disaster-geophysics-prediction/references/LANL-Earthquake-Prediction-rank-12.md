# 12th place solution

Competition: LANL-Earthquake-Prediction
Rank: #12
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94500#latest-553545

As kaggle tradition dictates, one shall share if he/she/they win the gold. So here I am, writing my very first solution post.

I would like the thank kaggle and the LANL for hosting such an interesting competition, drawing fellow kagglers/data science hobbyists' attention to these earth acoustic data which otherwise would not have been made well known.

I would also like to thank [Yifan](https://www.kaggle.com/yifanxie), who had introduced me to kaggle 3 years ago and given me many valuable guidance and advice, as well as other kagglers I was fortunate to team-up with in the past competitions. I have learn very valuable experiences from you. It's exactly this, the openness and the sharing spirit that made kaggle the most wonderful place to learn, exchange ideas/knowledge and improve. 

Now about how I landed the 12th position. (I am sorry about the long post. I just wrote as my memory rewind so the text lacks structure)

**The thought process:**
From my past competition experiences, I find the most important thing in any competition is that one must make the best effort to understand the data.

This was exactly what I was aiming to do at the beginning. I was using a lgbm model with public features, but made many attempts on different CV strategies in order to understand why there is such a big different in the CV score. 

I began with just stratifying by target, then by the 17 earthquakes, then manually selected groups of earthquakes. These experiments led to the following understanding:

- earthquakes with similar length can CV very well. for example earthquake #1 lasted 11.54 seconds, #10 - 11.42s, #11 - 11.02s. If you do a 3 fold cv on these three, you get very good MAE. Even for longer earthquakes like #2. #7 and #14, CV result is very good and reliable. 

- the problem is between the groups. If you train a model on shorter earthquakes your model will consistently underestimate long quakes, and model trained on long quakes will consistently overestimate the short ones.

I have then made some brief attempts to classify different groups of earthquakes. However the more I thought about it the more I realized there are many levels of challenges involved. To make a maybe inappropriate metaphor, predicting an earthquake is like predicting a man's death, you can probably predict accurately his aging speed, the eventual end however has many random factor involved that is simply unpredictable. That's why similar lengths earthquakes CV very well.

I have also thought about estimating the test data distribution between different type/lengths of earthquakes, maybe using some kind of Gaussian mixture model, and then quickly realized this is not possible. And it is this thought process that play an important part in my final submission. I was trying to image what the test data would look like given the above understanding. The competition designer would probably want to know if someone can come up with a generic model that does well on earthquakes of different length. So my hypothesis is: it's less likely that the test blocks of 150000 data points are taken randomly from different earthquakes. Instead it's probably from a number of complete quakes of different lengths. This means the test data distribution should like very much like that of the training data, i.e. uniform on the left and start to taper off on the right due to different earthquake lengths.

So my final two submissions are based on two CV strategy.

**Technical Detail**

The wining CV strategy is based on the following earthquake splits:
below 9s              (class 0): 0, 3, 5, 6, 8, 12, 13
between 10-12s   (class 1): 1, 9, 10, 11, 15, 16
above 12s             (class 2): 2, 4, 7, 14
note I put 0 in class 0 after confirming on CV results.
The CV is a repeated stratifiedKfold of n=2, repeat=5.
The first fold is used for making models CV'ed within each earthquake class and make prediction for the second fold.
Then within the second fold is another repeated stratifiedKfold of n=2, repeat=5 to stacking.

for the first fold training I have used a lgbm model, a xgb model, 2 svr models, a kernel ridge and a NN model. So basically the same as what Andrew did in his kernel https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples
On top of that I have made another 2-layer GRU model with custom time-series features. This RNN model has respectable validation score very close to the lgbm.
I had to remove catboost model at the end because it was taking too long. A few crash/bug fix left me not enough time to do more experiments.

for the second stacking fold I have only used a lgbm model and a NN model. 

I convinced myself that such approach is ok because first of all there is no leakage with split between earthquakes, and secondly I know that similar earthquakes CV reasonably well and I am counting on it to pass some kind of earthquake class information to the stacking model, and lastly I used the repeated stratified kfold to exhaust different earthquake combinations as much as possible to cover different cases and exploit the power and bagging (if you make enough reasonable attempts, the average is bound to be not far off right?)

In terms of features, 90% of the features used were based on public features. I have made some additional 60 features just based on welch power spectral density method. But to be honest I am even sure how much impact they've made.
As a result I have about 400 tabular features, and out of laziness I made a further 400 by extracting the same features from first quarter and the last quarter of the same 150000 blocks and subtract them. :'D
I had made 45 timeseries features for the RNN model which were also inspired by the public features. I just selected a few and made some adaption based on feature importance analyses from the public kernels.

These features are used for fold 1 base model training. Then the tabular features were concatenated with predictions from the base models for stacking training.

As a result I have observed reasonable CV improvement at the stacking level (I monitor the score for each earthquake in validation)

However the public LB result is way too bad. In fact the wining submission scored 1.61. What gave me the confidence (or courage) to submit it was my hypothesis about the test data distribution (plus i was never in the high ranks throughout so nothing to lose). The ttf distribution is close enough to that of the train (I will update a plot), i.e. gentle/similar to uniform on the left, and tapers off from 10s onwards on the right.

The second submission was a blend of different models (lgbm, xgb, cat, NN, and 3 RNNs) using repeat straitfiedkfold result, based on this split
7s                    (class 0): 6, 8
8s                    (class 1): 0, 3, 5, 12, 13
9s                    (class 2): 9, 15
11-12s                (class 3): 1, 10, 11, 16, 4
14-16s                (class 4): 2, 7, 14

The CV score for this strategy for different models were quite consistent between MAE 1.8-2.2. and it has a public LB of 1.449. So I meant for it to be a conservative/submission. But the distribution was not as pretty and it did turn out that it was pretty bad on the private board.

I have also made a 1D CNN model hoping that it may capture any potential useful information in the time gaps (This approach was motivate by some interesting discussion a while ago about data sampling, etc.).
The model has consistent predictive power, i.e. the MAE is always around 2.3-2.5 for different type of quakes, which I think might be interesting. However I couldn't add it in the stacking as well because it took too long to train and I simply ran out of time.

If you made it all the way to here than thank you very much for you patience and I hope this post provides you some useful information.

I will definite improve/slim down my posting next time.

Thanks you to all the fellow kagglers.
