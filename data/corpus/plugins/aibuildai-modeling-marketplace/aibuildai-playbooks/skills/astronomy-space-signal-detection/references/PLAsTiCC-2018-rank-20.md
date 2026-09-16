# 20th Place Solution

Competition: PLAsTiCC-2018
Rank: #20
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75262

First of all, thanks Kaggle and the sponsors for this great, clear and leak free competition :-)

This competition I started working hard in the beggining and practically stopped competing during last 10 days. My final solution is a stack ensemble and a blend of 4x LGB and 2x SVC models trained using different subsets of features. 

#Features
I used some subsets of around 250 features for the final models, but I'm sure I tried more than 1000 during the competition and dropped useless features. My features are built using all kind of aggregations and statistics, feets lib, coeficients of linear regressions for each passband, coeficient  of linear regression after a peak and custom made features base on observation of the light curves, for example: raise rate of peaks and valeys, peak ratios of 'time to raive' vs 'time to decrease to X%', etc...

#Models
For each model fit I augmented the train set concatenating noise versions of the train set. That simple augmentation improved my score by about 0.04 and decreased the CV x LB gap.

Model 1:  multi-class LGB trained using all 250 features and all whole set.

Model 2:  multi-class LGB trained using a subset of the 250 features, selected via feature selection, using whole train set.

Model 3:  multi-class LGB trained using another subset of the 250 features, selected via other feature selection algorithm, using whole train set.

Model 4:  multi-label LGB trained using a subset of the 250 features, selected via feature selection for each target individually, using whole train set.

Model 5:  multi-label SVC using rbf kernel trained using all 250 features and whole train set.

Model 6:  multi-class LGB trained using all 250 features. Fit one model for intra galaxy and other for extra galaxy.

Model 7:  multi-label LGB trained using a subset of the 250 features, selected via feature selection for each target individually.  Fit one model for ddf==0 and other for ddf==1.

Stack ensemble 1, multi-class, models 1,2,3,4,5.

Stack ensemble 2, multi-class, model 6.

Stack ensemble 3, multi-class, model 7.

Blend Stack Ensemble 1,2,3 using simple geometric weighted average.

For class 99 I ended using Scirpus(thanks @scirpus) equation plus some minor LB probings. 




#What didn't worked for me:

 - List item

 - FFT, wavelet and periodogram features.

 - DAE

 - Curve reconstruction

 - Fit a model for hoztgal_specz using train+test

 - CNN models ( metric &gt; 1.3 )

 - Target encoding

 - Flux correction by redshift

 - Flux passband correction using redshift

 - Semi-supervised Learning (improved just 0.0001)

 - SMOTE


Giba
