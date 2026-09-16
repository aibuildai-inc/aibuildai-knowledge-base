# 10th place solution (Amjad's view)

Competition: LANL-Earthquake-Prediction
Rank: #10
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94467#latest-547452

First of all, I would like to thank the organizers for providing an interesting research problem (even though badly designed) and kaggle for providing the computational resources that I used throughout this competition. My biggest thanks of course go to Giba who put his magical touch on my models. By the way, take a look at his views [here](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94466#latest-543723). You can check out my kernel [here](https://www.kaggle.com/amjad85/10th-place-feature-engineering?scriptVersionId=15179664), and Giba’s kernel that uses it [here](https://www.kaggle.com/titericz/top-1-lb-2-254-private).

**Preprocessing:**

I used denoising with discrete wavelet transform (DWT). However, the wavelet length was important here. Short wavelets like Haar filter too much useful information. I used  the best localized wavelet 14 (l14 from wmtsa R package).

**Feature generation**

Since the early days, one feature stood out, that is zero-crossing. So I generalized this feature to compute the rate of crossing points at various quantile levels of acoustic data. I computed crossing points features on several levels of DWT decomposed signals (started with 5 levels and went up to 10). Then I expanded the one DWT decomposition to three wavelets with different length. In addition to that, I added features representing the mean and variance at rolling windows of the spectrum of the signal. I also had some features counting peaks, which I took from public kernels and translated to R, and some autocorrelation features from the tsfeatures R package. I had also statistical features computed on different levels of DWT, but we didn’t end up using those given their different distribution.

**Cross validation**

I used shuffled KFold stratified by both time and quakes to enhance the features. This had the lowest variance among different strategies, and correlated well with public LB. I kept an eye open on other cross-validation schemes to know what causes overfitting. Towards the end, I also implemented nested cross validation, which is more robust than a simple leave-one-quake out cross-validation.

**Model training**

I started with Catboost at the beginning and regretted it given how slow it is. Once I switched to LGB, I immediately jumped the public LB.

**The twist**

Transferring the prediction to test distribution was pulled out by Giba at the last moment. See his post for explanation.

**Thing that didn’t work for me**

-	Classifying long vs short quakes
-	Augmentation with overlapping segments
-	MFCC features, rolling features, skew, kurtosis
-	Deep learning. My best model was 2-channel 1d-CNN (mean and sd) – bi-GRU scored 1.499 on public LB.

**My tips to beginners**

I’m not new to machine learning, but this competition was my first project with gradient boosting, so it was a great learning experience. Also I’m pretty much new to kaggle competitions. I participated in one some years ago and it was a struggle on my crappy laptop. Now it’s a much better experience. 
Here are some tips to other beginners:
-	Spend some time on making your code runs faster and easy to develop
-	Start with a ML algorithm that runs fast for development
-       Spend some time implementing your cross-validation
