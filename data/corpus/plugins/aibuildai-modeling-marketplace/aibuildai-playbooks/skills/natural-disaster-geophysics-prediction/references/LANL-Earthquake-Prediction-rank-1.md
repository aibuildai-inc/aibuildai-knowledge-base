# 1st place solution

Competition: LANL-Earthquake-Prediction
Rank: #1
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390#latest-568449

Thanks a lot to the hosts of this competition and congratz to all participants and of course to my amazing teammates.

What made this competition tricky was to find a proper CV setup that you believe in as the public LB gave bad feedback for private LB. This was my first competition where this was the case and it took me a while to completely ignore public LB, but it was necessary.

I will now try to summarize some of the main points that helped us to win this competition. I am posting these elaborations in the we-form as we are a team and everyone contributed ideas and knowledge. Special thanks to @ilu000 @dott1718 @returnofsputnik @dkaraflos @pukkinming who worked hard the last few weeks on the comp.

**Acoustic signal manipulation and features**

As has been discussed in the forums and shown by adversarial validation, the signal had a certain time-trend that caused some issues specifically on mean and quantile based features. To partly overcome this, we added a constant noise to each 150k segment (both in train and test) by calculating ```np.random.normal(0, 0.5, 150_000)```. Additionally, after noise addition, we subtracted the median of the segment. 

Our features are then calculated on this manipulated signal. We mostly focused on similar features as most participants in this competition, namely finding peaks and volatility of the signal. One of our best final LGB model only used four features: (i) number of peaks of at least support 2 on the denoised signal, (ii) 20% percentile on std of rolling window of size 50, (iii) 4th and (iv) 18th Mel-frequency cepstral coefficients mean. We sometimes used a few more features (like for the NN, see below) but they are usually very similar. Those 4 are decently uncorrelated between themselves, and add good diversity. For each feature we always only considered it if it has a p-value &gt;0.05 on a KS statistic of train vs test.

**Differences between train and test features**

After doing abovementioned signal manipulation, we had more trust in our calculated features and could focus on better studying differences between train and test data feature distributions. We found that the test data should look different to training data in a few ways when comparing features by e.g., applying KS statistics between train and test. That’s when we decided to sample the train data to make it look more like we expect test data to look like (only from looking at feature distributions). We started by manually upsampling certain areas of train data, but gave up on that after a few tries and then we found a very nice way of aligning train and test data.

So what we did is that we calculated a handful of features for train and test and tried to find a good subset of full earth-quakes in train, so that the overall feature distributions are similar to those of the full test data. We did this by sampling 10 full earthquakes multiple times (up to 10k times) on train, and comparing the average KS statistic of all selected features on the sampled earthquakes to the feature dists in full test. A visualization for this looks like this (this is a limited visualization and not necessarily the one we chose to make our final selection of EQs):

 [KS statistic train subsample vs. test]

The x-axis is the average target of the selected EQs in train and the y-axis is the KS statistic on a bunch of features comparing the distribution of that feature for the selected EQs vs the full test data. We can see that the best average KS-statistic is somewhere in the range of 6.2-6.5. You can also see nicely here that a problematic feature like the green one deviates clearly from the rest, this would be a feature we would not select in the end.

After careful examination of these results, we decided in the end to subsample the train data to only consider earthquakes [2, 7, 0, 4, 11, 13, 9, 1, 14, 10] numerating all 17 earthquake cycles we have in train. The mean of this sample is 6.258 and the median is 6.031.

**CV**

Now that we had sampled train data that we though to be similar to test just purely based on statistical analysis, and now that we had features that should not have any time leaks, we decided on doing a simple shuffled 3-fold on that data. Higher fold results are similar. We now tried to improve this CV as well as possible. 

**Models**

Our final submit is a hillclimber blend of three types of models: (i) LGB, (ii) SVR, (iii) NN. The overall CV score on this was ~1.83. The LGB is using a fair loss with relatively moderate other hyperparameters. The SVR is also quite simply set-up. The NN is a bit more complicated with a few layers on top of a bunch of features. The real interesting thing here is that we do multi-task learning by specifying additional losses next to the ttf loss that we weight higher than the others. We have one additional binary logloss with the target specifying if the ttf is &lt;0.5 and one further MAE loss on the target of time-since-failure. This helped to balance some of the predictions out a bit and specifically helped to better predict some of the areas at the end of earthquakes that make some weird spikes. The NN had the best single MAE, but blending improved. Actually, just blending LGB and NN would have produced the best private LB score (2.25909). Adding SVR did improve CV though.

With all the steps described above, we also managed to make the distribution of test predictions very similar tho oof predictions. The following image shows for a single LGB the oof (blue) vs. test predictions (orange). The KS-test between those two does not reject the null hypothesis of them being equally distributed.

[LGB oof (blue) vs. test (orange) prediction dist]

**Ideas that have potential**

We had quite some ideas that have potential but did not make it into our final submission. One area is to better use the time-since-failure prediction, which we used only as an additional loss in our NN. Modeling tsf works better than ttf. It can help to manually adjust a few predictions which have large discrepancies between tsf and ttf predictions, like the end of EQs. Also, they can be a reasonable proxy for predicting the approximate length of the EQ. So for example, we had one model that normalized the ttf targets to be in range 0-1 and then predicts this normalized target and scales it by ttf+tsf prediction. This was usually very close to our simpler models so we did not tune it extensively, I just feel that this has further potential. 

The following kernel runs a LGB model on most of what I explained above and also would score 1st place with 2.279 private MAE:

https://www.kaggle.com/ilu000/1-private-lb-kernel-lanl-lgbm/

The following kernel runs a blend between LGB and NN scoring 2.25993 on private LB:

https://www.kaggle.com/dkaraflos/1-geomean-nn-and-6featlgbm-2-259-private-lb
