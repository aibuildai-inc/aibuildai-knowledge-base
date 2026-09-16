# 22nd place solution described (molaee)

Competition: LANL-Earthquake-Prediction
Rank: #22
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94903#latest-547809

Its my pleasure to share the ad-hoc method I used in this competition with you. I am not that expert in ML, however I hope that my method opens some fruitful discussions for all of us.

My kernel was forked initially from here [here](https://www.kaggle.com/artgor/even-more-features) (Thank you @artgor )

**Feature extraction:**
After running the kernel, I realized that the TTF is not estimated very well in the vicinity of failure times (it becomes saturated). I worked a lot on feature extraction to overcome this issue. My focus was mostly on time-frequency and MFCC related features and could increase the Pearson correlation from 0.65 (in the original kernel) to 0.68 (when measured on the entire data). The pearson correlation increased from ~0.25 to 0.38 when measured on the data in the vicinity of the failure times (&lt;3.2 s).
My best feature: The area under 0.2 quantile of the cumulative STFT  (presented in log scale) of the audio signal (NFFT=128, signal de-trended).
After a couple of days, I had about 1300 similar features!! -&gt; LB public score improved 0.1 from ~1.5 to 1.4

**Benefit from the leak:**
I became aware of the leak only 2 days before the submission deadline. I did not carefully analyzed the leak data, nor did match the mean of predicted TTF with the one presented in the paper. I just, removed the 5th and 6th EQ data from the training set. Visually inspection of the test and train data, I had the feeling that after removing these two EQs, test and train data become very similar to each other.

**Model:**
I got the lgb model presented in the forked Kernel. Then created 3 categories of data set: (A) entire data excluding the 2 above mentioned EQs, (B) data very close to failures (3.2 s and less) , (C) data far from failures (6 s and more)
For each data set, a separate lgb model was trained. The model corresponding to (A) had a mastering role. If its prediction was below a threshold (3 s) or above a threshold (7 s) then its output was being merged with that of (B) or (C) in a linear manner.

`
for ind, row in X_test.iterrows():
    if prediction_lgb_m[ind] &lt; thr_l:
        alpha = (thr_l - prediction_lgb_m[ind])/thr_l
        prediction_lgb[ind] = (alpha)*prediction_lgb_short[ind] + (1-alpha)*prediction_lgb_m[ind]
    elif prediction_lgb_m[ind] &gt; thr_h:
        prediction_lgb[ind] = prediction_lgb_long[ind]
    else:
        prediction_lgb[ind] = prediction_lgb_m[ind]
plt.hist([prediction_lgb,prediction_lgb_m],50)
`
TTF histogram over test data using model (A) [orange] and Combined modes (A,B,C) [Blue]:


**Topics for discussions:**

1) Does it make sense to have a cascade of lgb models as I described? In my opinion, this method should add another high-level branch to the tree of the original model (A). But I could not get the same results when I increased the number of layers or leafs in (A) !!!
Indeed, when I used models (B) and (C) I could get pretty lower/higher TTF values for the training data close/far to the failure time. The model (A) could not provide such results.

2) After one month working on feature extraction, I had about 2300 similar and correlated features. Compared to my original 1300 features, I could obtain a bit better CV values (2.01 --&gt; 1.99) and I could see less fluctuations over (y - ypred) training data set. However, LB public score became a bit worse (1.399 --&gt; 1.43) !! As if I had over fitted. Right? What is the general procedure to reduce the number of features from 2300 to a lower value?

3) I tried to reduce the number of features by removing highly correlated features (0.95 and more) from the feature list. This reduced the number of features to about 800. However, CV increased (+0.01), as well as LB public (+0.03).

Finally, I decided to use my entire 2300 features, 3 LGBs, and the removed EQs 5 and 6 from the training set.

**What did not work:**
1) PCA over the features increased the LB public score. So I did not follow it.
2) Combination of LGB with other methods was not really a big success. I preferred not to step in that way.
