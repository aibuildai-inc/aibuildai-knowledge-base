# Sharing of my experience so far

Competition: elo-merchant-category-recommendation
Rank: #24
Source: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/75935

Hello all, I hope everyone is enjoying a very nice period toward the end of the year, and for me, it is definitely great fun to have some dedicated time to immerse myself in a kaggle competition without thinking about other daily stuff.

It is mid-way through this nice competition(or maybe not so nice depends on how your look at it). I would like to share some of my experience so far. Some are successful, some are not, but they are all good experience. 

so here we go:
-  Feature engineering: my personal feeling is that regular feature engineering (typical groupby-agg, counting, categorical encoding) would bring LB score down (i.e. RMSE)  to 3.680, but no more. I personally starting to find it difficult to improve beyond the 3.690 mark. Beyond this line, I started to look for other ways that would give me features. Some of these have already been mentioned in the forum, such as count vector, FM/FFM, NN, and out of fold model.

- Feature selection: I have found this dataset to be quite sensitive to 1) selection of features, and 2) order of features in the dataset - especially if you are using algorithms that choose sub-set of features (i.e. GBM). By switching columns order, I have had scores with 0.005 difference on the LB (obviously with very similar CV). Usually I don't do too much feature selection in a kaggle competition, but now I found this (often overlooked) technique to be useful BOTH in improving CV/LB score, and also helps to reduce computation time. I strongly encourage everyone to have a look at the post from  @peterhurford titled "[Less is More][1]". Everything I used so far w.r.t. feature selection comes from that post.

- Get your basics right: Each and every kaggle competition shall be treated as a work/study project in which you should setup a proper workflow to structure your work. and I can not emphasise more about this - every model you chunk out, MAKE SURE IT IS REPRODUCIBLE - ok some algorithms/tool are non-deterministic - but then again you should make sure you can reproduce your model with similar baseline performance. Personally, for each model I create, I use a good-old spreadsheet to capture the following: algorithm used, parameters, features used, seed, cross-validation approach(kfold, stratified kfold, etc), CV score, LB score(if submitted), score at each fold, std of CV score. everyone's approach differs, but still I encourage everyone to write these down - in chinese we have the saying "a good memory is not as good as a used pen" :)

- Think "out-of-box": the elephant(s) in the room is that there are 2000+ outliers which contribute to most of the RMSE, and this requires us to take time to think how to address this. there is quite a lot of discussion on this, and the [nice kernel about post-processing][2] from @waitingli, and also the [discussion thread][3] from @lblhandsome.  I don't have much to add to this, but what I would say is make sure you spend time away from your computer(s). Take a walk, spend time with your family/friends, and eat/sleep properly. For me, the several ideas that really work came when I was not facing the screen busy coding. 

- Manage your code:  always keep your code in proper modules or classes, so that they can be reused for your future projects and competitions. A lot of code I am using to push myself up the LB has from my own little repo - so I have probably saved hundreds of hours by reusing my previous stuff. In a sense, you need to find a way to team up with your previous selves :)

What I have not yet managed to figured out is that magical something both  @senkin13 and @lblhandsome have found. So actually everything I have done so far have been fully shared by many others in many previous competitions - I can confirm that without anything "magic" or "leaky" you could get up to at least 3.67x in public LB - well, private LB is another story, and probably worthy of another post to discuss how much shake-up there will be - insert :trollface: 

Anyway, I hope everyone enjoys this competition, and have a great time! 
 


  [1]: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73937
  [2]: https://www.kaggle.com/waitingli/combining-your-model-with-a-model-without-outlier
  [3]: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/75034#440855
