# 12th Place Solution

Competition: PLAsTiCC-2018
Rank: #12
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75237

First of all, thanks to Kaggle and LSST teams for holding this fantastic competition. It is also very interesting and made me know a lot more about what mysterious things our astronomers are doing. I am very grateful for the community and learned very much from kagglers' generous kernels and discussions. Congratulations to the incredible [@Kyle][1], all the medal winners, my teammates [@lucaskg][2], [@zuoweijian][3], [@xietian6578][4], and especially my teammate [@strideradu][5] who became a Kaggle Competition Master.

Our final model is an ensemble of LightGBM, XGBoost, and binary classifiers. Final CV=0.3252(with specz)/0.3842(without specz), Public LB=0.814, Private LB=0.826.

Here is a summary of what we did during the last tough several weeks.

**1. Feature Engineering**
==========================

**1.1 Frequency non-related features**
----------------------------------------

We performed feature extraction manually one by one, inspired by the functions listed on [FATS][6]. I guess they are basically similar to those extracted from packages such as tsfresh and cesium. Around a total of 50-60 kinds of features are extracted to characterize each light curve. We did both passband-level and object-level feature extractions.

**1.2 Frequency related features**
------------------------------------

We referred to [this kernel][7] published by [@Scirpus][8]. We evaluated the periodogram and grouped the result into ~20 frequency bins. The amplitude was summed in each frequency bin.

**1.3 Bazin**
------------------------------------

My teammate [@lucaskg][9] did the curve fitting, using [this paper][10] for reference.

**1.4 Flux adjustment &amp; Features while detected == 1**
--------------------------------------------------------

We performed feature extractions on different datasets, including:
 
 - original flux features,

 - original flux features while detected == 1,

 - magnitude1 (flux * specz * specz) features,

 - magnitude1 features while detected == 1,

 - magnitude2 (flux * photoz * photoz) features,

 - magnitude2 features while detected == 1

**1.5 Feature difference among different passbands or passband groups**
---------------------------------------------------

Besides aggregating passband-level features using 'std' and 'mean', we constructed some features that reflect the difference among passbands or passband groups, such as:

 - 'flux max passband i' - 'flux max passband j',
 - ('flux mean passband0' + 'flux mean passband1' + 'flux mean passband2') - ('flux mean passband3' + 'flux mean passband4' + 'flux mean passband5')

**1.6 Meta Features from passband-level prediction**
------------------------------------------------------

Please refer to Part 2.1 below.

**2. Modeling**
===============

We built three models: galactic, extragalactic with specz, and extragalactic without specz.

**2.1 Passband-level modeling**
-------------------------------

We ran a LightGBM on passband level. In this model, we only employed those features related to the shape of the light curve, such as flux skewness and frequency features. All features related to the magnitude of the flux, such as flux max and flux mean, are ignored. After we got the passband-level probability prediction, we flattened them as a group of meta-features.

It may be noteworthy that **in the galactic model, we removed all the data with passband == 0**. We found the prediction can be more accurate without them. At least, CV improved with doing that.

**2.2 Object-level modeling**
-----------------------------

We applied LightGBM at most of the time during the competition and tried XGBoost on the last day. To avoid overfitting, feature numbers are restricted to 100 for galactic model and 140 for extragalactic model, based on the feature importance obtained by LightGBM. The performance of single models is as follows.
 

 - Galactic: LGB CV=0.03149, XGB (no time to run)
 - Extragalactic with specz: LGB CV=0.48724, XGB CV=0.50902
 - Extragalactic without specz: LGB CV=0.58838, XGB CV=0.61177
 - Total: LGB CV=0.3448(with specz)/0.4143(without specz), Public LB=0.831, Private LB=0.847

**2.3 Binary Classification**
-----------------------------

Besides the multi-class classification models, my teammate [@lucaskg][11] conducted a binary classification on each of extragalactic classes.

**2.4 Ensemble**
----------------

I tried to stack LightGBM and XGBoost by allocating them different weights, but CV did not improve at all. I guess the reason could be the high correlation between them. So I turned to use LightGBM to attempt an ensemble on LightGBM, XGBoost and binary classification predictions. It turned to be useful with CV improved from 0.3448 to 0.3252(with specz), 0.4143 to 0.3842(without specz), public LB score improved from 0.831 to 0.814, and private LB score improved from 0.847 to 0.826.

 **3. Class_99 adjustment**
===============
Actually, what we did on class_99 is not worth mentioning because we only got a 0.001 boost on the leaderboard. However, considering we are finally just 0.0004 ahead of 13th place (who is the first of silver medal winners) on private LB,  it was very crucial to help us achieve the gold medal. 

We applied [@olivier][12]'s method and [@Scirpus][8]' [method][13], and found the latter is always a little better than the former on public LB. So I tried to interpolate class_99 by setting it to:

class99Prob = class99scirpus + (class99scirpus - class99olivier)

which boosted the LB score by 0.001. Here, we would like to express our sincere thanks and appreciation to [@olivier][12] and [@Scirpus][8], helping us make our gold dream come true.


**Thanks to the Kaggle community and You'll never walk alone!**
---------------------------------------------------------------

  [1]: https://www.kaggle.com/kyleboone
  [2]: https://www.kaggle.com/lucaskg
  [3]: https://www.kaggle.com/zuoweijian
  [4]: https://www.kaggle.com/xietian6578
  [5]: https://www.kaggle.com/strideradu
  [6]: https://pypi.org/project/FATS/
  [7]: https://www.kaggle.com/scirpus/lomb-scargle
  [8]: https://www.kaggle.com/scirpus
  [9]: https://www.kaggle.com/lucaskg
  [10]: https://arxiv.org/pdf/0904.1066.pdf
  [11]: https://www.kaggle.com/lucaskg
  [12]: https://www.kaggle.com/ogrellier
  [13]: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/72104
