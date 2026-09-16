# 8th place solution

Competition: ariel-data-challenge-2024
Rank: #8
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543776

Thanks Kaggle and competition host for this interesting competition and also specials thanks to sergeifironov for the great starter notebook, I basically follow the same way and add some details.

**Data Processing**
- almost same as public notebooks, remove two more dimensions for last 60 wavelengths

**Phase Split**
- detrend by polynomial fitting with boundary points
- find gap center with accumulated gradient
- find exact gap edge based on monotonic condition

**model1**
- transit_depth=(oot-it)/oot, transit_depth is coefficient before oot term in linear equation, so this can be solved by OLS directly
- fit the average of all wavelengths
- fit by quantile regression first and remove points with large residual then fit by OLS. Same points will also be removed in model2 without running quantile regression again. This makes model robust to unclear phase edges.

**model2**
- apply svd to all wavelengths and reconstruct them with principal components, different wavelengths are weighted based on SNR before svd
- exponential smoothing over nearby wavelengths, different wavelengths are further weighted based on SNR
- past two operations only apply to ch0
- fit all wavelengths independently with same way in model1
- coefficient covariance based on coefficient standard error of OLS and residual correlation

**ensemble**
- basic models are run on data with bins 12*10, 10, then I average 10 results with shifted start index. When bin size becomes smaller the error in variables start to have significant impact

**Post-Process**
- apply svd to model2 coefficient to get first k ( 3/4 is best for train/test set, since test molecules are super set of training collection) principal components, then reconstruct coefficients by GLS with covariance shrinkage, the covariance of model2 are transformed based on OLS since simple GLS covariance seems underestimate the actual variance.
- base_coef=0.6*coef1+0.4*coef2
- base_coef*mean(coef2)/mean(coef1) to deals with bias caused by imbalanced energy distribution across wavelengths
- replace base_coef with coef2 when significant difference exists between coef1 and coef2, this is done in planet-wavelength level and planet level.
- sigma for base_coef: max(|coef1-coef2|, coef_std2) Imagine we ’observe’ highly biased model1 from less biased model2. Sigma for coef1 is bounded by average gap between coef1 and coef2 globally, this works when accurate coef2 estimation is infeasible
- sigma for coef2: (2*coef_std+max coef gap between nearing wavelength), since model2 is still biased and there exists unknown bias in data. I extend coef_std2 based on some heuristic rules
- I add some rules to capture cases when the models might fail, and replace by default results, but seems they never work on training set and public test set, not sure if it worked in private part.

**Reproduce**
https://www.kaggle.com/code/w5833946/adc-final-reproduce
I also run it with training set in ver1.
