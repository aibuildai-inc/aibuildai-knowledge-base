# 7th solution, CPMP view

Competition: LANL-Earthquake-Prediction
Rank: #7
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94407#latest-555528

First of all, I want to thank organizers for this interesting and tricky challenge, as well as my team mates @antoine and @stecasasso for their interesting views.  You can find @antoine's view [here](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94359#latest-542974), and @stecasasso [here](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94408).  I also want to thank @mykper for his great analysis of public test data.  

Our solution is based on few pieces.  I'll start by what I did before teaming.

**Cross Validation**

We used 16 fold CV where each fold includes a time to failure (TTF) reset.  Lately we realized that a simple unshuffled  16 folds would be equivalent  to what we did.  Even with 16 folds, it was quickly clear that CV LB correlation was poor, and that public LB was not a reliable.  I decided to use a nested CV approach where one fold Ti is used as test data, and CV is run on the other 15 folds.  15 models are trained using the 15 fold cv and their predictions on Ti are averaged.  This mimics the submission process 16 times, and provides a reliable indicator of a model strength by averaging the mae of Ti folds.  Correlation with public LB was very good.

**Data Sampling**

I used 150k segments starting every 50k rows.  Every segment overlaps with 2 segments before it and 2 segments after it.  I tested more overlap, and less overlap, and this one was the best trade off between running time and result accuracy.  A potential issue could be that segments at fold boundary overlap.  I tried removing those overlap but result accuracy decreased a bit, hence I kept all segments.

**Outliers**

TTF is not reset at acoustic data peaks which makes the problem difficult.  I decided to treat the points between acoustic peak and the following TTF reset as outlier, and trained a binary model to predict outliers.  I also trained model where the target is TTF outside outliers, and TTF is reset at acoustic peaks.  

Final prediction is 0.2 if binary prediction is above a threshold (0.45 to 0.7 depending on the runs), and an average of models trained on original TTF and modified TTF as above elsewhere.

**Features**

I used 7 MFCC from librosa.  To make it work I pretended than data was sampled at 40kHz.  Not doing so means the useful info is way higher, around the 20th MFCC.  I also used 4 features based on std of various signal quantiles.  Last, I used a binary feature indicating high variance segments that correspond to acoustic peaks very accurately.

**Models**

Mostly lgb with conservative settings, like 7 leaves only.  But also general additive models using pygam.  I also trained a knn model which was surprisingly good.  In hindsight, gam was better than lgb which was better than knn, the gam model would get a gold medal alone at 2.348.  For lgb I tried mse, gamma and huber objective.  With the above I got to 1.288 on public LB at which point I teamed.

After teaming, progress was in many areas, but here are the three most important ones.  

**Ensembling**

First, model diversity as my team mates were using different data sets.  By averaging our top public LB @areveillon had a 1.286 public LB) we got to 1.275 and took the lead.  TMy team mates adapted their models to the binary vs regression models.    We also reused some features from each other dataset, which added to model diversity.  We then worked on stacking, ending with a stack based on a lgb, a gam, and a knn from me, a lgb from @Antoine and a NN from @stecasasso .  lgb was used for the second level model.  We validated stacking using our nested CV which gives us some confidence that stacking was indeed working.  Private LB confirms that stacking was improving over base models.

**Train / Test Difference**

I must admit that I hate LB probing, and avoid it as plague.  But here, using pictures from academic papers could lead to a good estimate of test data, and it was the way to go given the significant difference between train and test data.  @Antoine had done an estimate before teaming and was using it already.  Few days before end he convinced me that we should really base our final sub on it and I measured precisely length of EQ cycles in that picture and could estimate their duration by running a linear regression on train data.  From it I estimated its mean to be 6.35 which is quite close to the actual 6.32.

From this we can estimate density function of TTF for train and test.  We then used sample weights to map the train distribution to the test distribution.  I see that top 2 teams (at least) rather resampled train, and this may explain why they are ahead.  There are two ways to compute weights.  I tried to base them on density function, i.e. weights only depend on the TTF, while @Antoine was pushing for weights per eq cycle.  In hindsight he was right.  Given we could not agree before competition end we decided to produce two submissions, one that optimizes my weights while the second one was optimizing @areveillon 's weights.   @stecasasso  then had the idea of using the weights not only for evaluating our models, but also for training models.  In the end, submissions optimizing @areveillon 's weights were the best ones.

**Post-processing**

@stecasasso found that we could set all high variance segments TTF to 0.31.  @Antoine looked at how to best combine binary models to fix segments TTF at 0.2.  Applying these as postprocessing improved submissions.  It also moved our best public LB from 2.75 to 2.45.

**What did not work**

I could not get NN models to work well enough to be useful in our stack.  @stecasasso managed to get one, but it was the weakest model on the stack.  lgb and gam are better.  I think NNs could be useful on processed data, either sftt or Hilbert envelope, but I did not have the time to try seriously.

**Answers**

I was asked about what was the 'no magic' feature. It was the 4th MFCC I was using.  I also was asked how to overfit public LB.  Well, first answer is that it was easy given how many people had way worse results on private Lb than public LB.  But in our case it was just by capping test prediction to 9.6 then to 9, given @mykper had shown public test TTF was below 10. This led us to 1.236 from 1.245.  I'm curious about how kopeyka team overfitted public test so effectively.  Last, I was asked about how I survive shakeups like this one or in previous competitions: it is because I only trust my CV and do not rely on LB probing.
