# 15th Place Memo and CV Scheme

Competition: LANL-Earthquake-Prediction
Rank: #15
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94556#latest-553091

## Thank You!

It was my first Kaggle challenge. (... I pursued to the end. I got into [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification) but got distracted with other stuff in live.) I'm pretty happy to have survived the shake up and have finished so high up in the ranking. I'd like to thank the community to share insights and ideas so well (especially to @CPMPml and his [no magic](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/92679#latest-543125) feature selection)! That's the spirit of Kaggle! It was a great learning experience, and I will try to give something back here.

## What Matters

I made that huge jump in the LB and finished well. I'm still trying to figure out, to what degree it was luck and how much it was skill and a robust model. The shakeup might suggest a large influence of luck. Notching the model slightly, like adjusting the mean, can have a big impact on your standing in the final LB as @sushize showed [here](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94324#latest-543708). So the final scoring was indeed quite fragile and one had to be lucky to win a gold medal. I don't think, it's pure luck despite the big shakeup, though. We still see the grandmasters in high spots and the winning solutions used models with few features and robust CV strategies.

I think this challenge taught us that applying fancy ML techniques is not enough and it sometimes matters more to understand the basics. We have to:
* get a good grasp of the problem and understand the data well
* build a good and robust CV strategy
* avoid overfitting by all means

As the data was so little, these key ML ingredients played a major role in this competition and models ware relatively less important. Trust your thinking, think well, and do not apply stuff, you don't fully understand (though try to!).

## The Heart: Cross Validation and Significance

After some initial struggle with the large data and spending quite some time trying to get NN's to work, time was getting short for me. That helped me to concentrate on the important. I committed myself to the models that worked best, which were GBM's (I used a blend of LightGBM, XGBoost, and CatBoost in the end). Most importantly, I realised that a robust CV strategy *you stick to* is crucial.

We've seen that the quality of practically all models depended on the actual TTF. They do well for intermediate values and bad for very low or high values of TTF. So, depending on the distribution of TTF in the test / validation portion affects the score a lot. Hence, I tried to split the training data, such that this effect was minimised. At the same time, I got the feeling that I should take entire cycles in or out. Hence, I searched for splits of the 17 cycles into groups of three to four cycles that had similar TTF *and* size. One of those splits was
| cycles | fraction | mean TTF |
| --- | --- | --- |
| 0, 3, 7, 8 | 21.0% | 5.86 |
| 1, 10, 11 | 20.8% | 5.67 |
| 2, 12, 15 | 19.9% | 5.68 |
| 4, 5, 9, 16 | 19.7% | 5.56 |
| 6, 13, 14 | 18.6% | 5.61 |

Groups 0 and 16 are the very short "cycles" before and after the first and last earth quake. Note that with this split, I also don't have leakage between overlapping segments.

These folds still didn't give me the stable CV scores I wanted. I created some more of these even splits and decided to actually do four of these 5-fold group splits and take the mean MAE of each of the four 5-fold splits, such that I had four scores. This resulted in a rather slow CV scheme, but it was stable and I could not only work with a CV score, but also an uncertainty: `np.std(cv_mae,ddof=1)`, where `cv_mae` are the four averaged scores of the 5-folg group splits. These uncertainties were typically of order `0.005`. Note that this scheme assumes that the (private) test set has the same TTF distribution as the training set.

After establishing that, I was in a situation where I could not only test models and features with a robust CV scheme, but also had a measure for what a significant improve in the score is.

## What We Predict On

The assumption that the distributions of features and TTF in the test set, I've made in setting up my CV scheme, is quite reasonable. Some said otherwise, but I contradict. It was claimed by @gpreda that [the feature distribution was different for train and test set](https://www.kaggle.com/gpreda/lanl-earthquake-new-approach-eda). I could not find that for my features as I've subtracted the mean of the acoustic data on each segment. That makes physically sense and I already did it before I knew about the comparison.

Also the assumption that the TTF distributions are similar should be reasonable. @mykper did some [nice work](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/91583) on the public test set. We've seem a somewhat different distribution of the TTF and they also maxed out at some value below 10, whereas the training data had values up to 16. That's true. However, he also found out, that there are only two cycles in the public test set. These were quite typical, if you compare them with the training set. No reason to assume that we will predict on structurally different data than what we have seen in the training data.

Also one **note on the public LB**:

The public LB was calculated on just about 350 segments. This small number together with the high variance in the predictions alone should tell you that you cannot trust the public LB. Now compare that number with the size of the CV: we use several fold and will eventually use the entire training set for the score. That is 4200 segments. That's more than 10 times as much! When using overlapping segments - as I also did - the ratio gets even bigger, although the information gain will not grow linearly with the number of segments anymore. In anyway: the public LB score is calculated on a tiny set, especially when compared with the CV on the training data. I've never trusted the public LB in this competition.

## The Features and Minimalism

I took @CPMPml approach of feature selection he explained in the [no magic](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/92679#latest-543125) discussion and augmented it with my uncertainty of my CV scores. As we have little data, and I would have ended up with almost no features, if I'd only taken features with high significance. I decided to take features that improved my model by at least half a standard deviation.

Some side note from a physicist:
If you have two measurements `m_1` and `m_2` with standard deviations `s_1` and `s_2`, the difference of the measurements `m_1 - m_2` does *not* have an uncertainty / standard deviation of `s_1 + s_2`, but of `sqrt(s_1^2 + s_2^2)`.

With that approach, I ended up with just 7 features!

## Final Words

That's pretty much it. Some ensembeling of different GBM's (LightGBM, XGBoost, CatBoost) with (a little) optimised hyper-parameters and, as I've said in the beginning, probably a bit of luck.

There are some things I've seen in other solutions, that I'd like to have used, too. There is the special treatment of outliers as them ABC did (see their approach [here](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94407#latest-543731)). Also nested CV would probably have done some good.

Much of the detective work on the test set, was not so fruitful, I think. Despite we even [know where it came from](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/90664). The only think that was useful - and probably quite a bit, although I didn't use it - was the information about its mean TTF. It seems to have helped a few teams. So after all, my assumption that the test set is not as strictly fulfilled as I've assumed in my CV scheme.
