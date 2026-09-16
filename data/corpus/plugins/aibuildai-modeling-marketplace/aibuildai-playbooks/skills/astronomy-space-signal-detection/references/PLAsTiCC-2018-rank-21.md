# 21st Solution ~super tough road~

Competition: PLAsTiCC-2018
Rank: #21
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75156

Hi, dear kagglers.
First of all, big thanks to Kaggle and LSST teams for holding such a flawless and leakage-free competition. And congratulations to the winners and all Kagglers.

It’s really painful and tough during last few weeks since our score didn’t improve well. Although our team were unable to get gold medal. but we still write down our solution here in order to give some feedback to this wonderful community :))

Our final model is a blending of 4 LGBMs. CV=0.41653, Public=0.85191, Private=0.86777, We’ll write what we’ve tried detailedly below :

### Feature engineering

+ Aggregation   
agg: {mean, median, max, skew, kurt, percentile(10,25,75,90,99), iqr, max-min}  
apply agg on :
{flux, flux_err, flux_by_flux_ratio_sq, flux_ratio_sq, shift flux, phase shift flux, rolling flux, phase rolling flux, detected, normalized flux, flux*photoz, flux*photoz^2, flux / photoz, flux / photoz^2}  
different group : {object_id}, {object_id,passband}, {object_id,detected}, {object_id,passband,detected}, {kmeans of hostgal_photoz}


+ Color features   
  difference between the {max,mean,median} flux of different passbands (like [this](https://arxiv.org/pdf/1701.05689.pdf), see 3.1, but not only computed with adjacent bands), these features give us really big boost.   
  
+ Features from paper and different library  
  -- [cesium library](https://github.com/cesium-ml/cesium), we computed all the features in it.  
  -- [tsfresh](https://github.com/blue-yonder/tsfresh), just tried some of it.  
  -- [FATS library](http://isadoranun.github.io/tsfeat/FeaturesDocumentation.html), we've tried some features based on the importance plot in these paper : [paper_1](https://arxiv.org/pdf/1506.00010.pdf), [paper_2](https://arxiv.org/ftp/arxiv/papers/1710/1710.06804.pdf), [paper_3](https://arxiv.org/pdf/1809.00763.pdf)  
  -- Some other features from [this paper](https://arxiv.org/pdf/1801.07323.pdf) and [how to compute](https://arxiv.org/pdf/1511.03456.pdf) : 4 passband independent
time scale features, Autocorrelation Integral, Shannon entropy, HL Ratio, Median Absolute Deviation, Von-Neumann Ratio and some Statistic like Shapiro-Wilk or Jarque-Bera. but most of them ending up with no improvement.
  
+ some other useful features   
  -- max(mjd) - min(mjd) when detected == 1  
  -- max(mjd) - min(mjd) when detected == 1 and only select flux that greater than mean(flux)  
  -- Some diff features like "max(flux) - mean(flux)" or ratio features  
  -- Percentage of passband at max flux per time, use mjd and phase, see below :  
![class][1]
  -- PCA on whole dataset(train + test)  
  -- Making a lgbm to predict hostgal_specz, using oof as a feature  
  -- Number of "going up" and "going down"
  
+ something not worked for us this time  
  -- Measuring rising time and declining time of the light curve in different passband.  
  -- Decay speed of flux after peak in different time window   
  -- Conv1D, RNN for time series features  
  -- kmeans of some important features then doing WOE, target encoding
ex. kmeans of hostgal_photoz cluster=15, then target encoding or WOE per cluster.
  -- Stacking, still can't figure out why  
  -- Gaussian Process with RBF Kernel and Spline kernel in "kernlab" package in R, and doing same aggregation like above didn't improve our CV.  
  -- Training a model with objective = "ova" in lgbm, this one is only worse about 0.03-0.05 than gbdt one, but it didn't give any positive feedback when blending.


### Special Sauce
There are some secret sauce that improved our score a lot, one is we use GPyOpt to optimized the sample weight in lightgbm, it suprisingly gave us another 0.04 boost in LB. We use following sample weight in our best single model :

&gt; labels2weight = {6: 4.343454,
 15: 3.283419,
 16: 2.074140,
 42: 0.2,
 52: 3.671259,
 53: 17.475224,
 62: 0.837049,
 64: 9.155481,
 65: 0.728893,
 67: 3.800462,
 88: 4.003533,
 90: 0.1,
 92: 2.984069,
 95: 4.337909
}

And another thicks is to doing some post-processing, which @fatihozturk will share details with validation in another write-up, we have done some optimization with following formula : pred_class[X] * param[X] and X = 1~14, we optimized the parameter with Nelder-Mead solver, and it also gain me another boost of 0.03. Also, applying 

&gt; pred[i] = pred[i].apply(lambda x: 1.1 if x&gt;0.76 else x )   

can improve lb by 0.005 ! 


And that’s all, thanks for your reading!

[fatih’s post processing thread](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75140)

[1]:https://imgur.com/9kK71x2.png
