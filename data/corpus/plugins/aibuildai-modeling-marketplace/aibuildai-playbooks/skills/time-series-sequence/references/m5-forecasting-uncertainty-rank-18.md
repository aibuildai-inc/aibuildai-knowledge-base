# 18th place (silver medal) solution

Competition: m5-forecasting-uncertainty
Rank: #18
Source: https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/164191

Thanks for all selfless public kernel authors! My work just basicly based on your notebooks. It's more important to concern knowledge but not result, right? This sharing is to thanks all community members. It's you made Kaggle such a friendly, wise community. 

### Accuracy-Competition

+ FE: adding a `weekends` feature.

+ I used 4 single models:
    
    + time series split 3 fold cv - rmse loss function
    
    + [time series split 3 fold cv - poisson loss function](https://www.kaggle.com/rikdifos/timeseriessplit-cv-poisson)
    
    + time series split 3 fold cv - self-defined loss function
    
    + rolling prediction

+ Then I calculate all prediction's mean value, the I recode very tiny prediction to 0. Finally, I use a magic multiplier close to 1.

### Uncertainty-Competition

+ Use [public kernel](https://www.kaggle.com/szmnkrisz97/point-to-uncertainty-different-ranges-per-level) released by [KrisztianSz](https://www.kaggle.com/szmnkrisz97) to convert point prediction to uncertainty. I use accuracy final submission as input and get a single submission 1.

+ Use a [public kernel](https://www.kaggle.com/ulrich07/quantile-regression-with-keras) released by [Ulrich GOUE](https://www.kaggle.com/ulrich07) as a single submission 2.

+ Finally, I did an averaging from above 2 submission files and recode all negative prediction to 0.

All code could be downloaded from my github repo:
https://github.com/songxxiao/m5_compete
