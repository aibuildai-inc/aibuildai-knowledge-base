# Minimum linear regression (5th-place)

Competition: tabular-playground-series-jan-2022
Rank: #5
Source: https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304369

I thank Kaggle for hosting this introductory tabular competition. I am looking forward for joining table/sales competitions with medals.

I did not understand the long-term evolution other than the GDP and gave up a week ago, but since I accidentally get 5th place, I write here what I understood.

### Model summary

- Linear regression with 29 coefficients + 1 bias.
- All years treated equally and no extrapolation to year 2019, because I did not understand the trend.
- Just minimizes the mean squared error of log(num_sold/GDP), i.e., standard linear regression.
- Train SMAPE 4.29899, Public LB, 4.13522, Private 4.66955.

Code available in notebook: https://www.kaggle.com/junkoda/holiday-kernel

Note that I did not think this is the best, I simply gave up before the deadline. The model has a small number of parameters and therefore avoids overfitting, but giving up future extrapolation cannot be the best approach. If there is something good about my model, it is probably the way I handle the holidays.

### Features

[1] Country, store, product, and weekday (Friday and weekend) have constant factors; constant offsets in log(num_sold), as in AmbrosM's great notebook (congratulations for the 1st place!):

https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model

7 parameters: Friday, weekend, Norway, Sweden, Hat, Sticker, Rama	

[2] Products mug, hat have **pure** cosine and sine annual modulations respectively, none for the sticker. I don't see phase shift or higher Fourier modes. 2 parameters for mug cosine and hat sine amplitudes.

[3] Holiday boosts have a **common** Gaussian shape peaking at 4.5 days after the holiday, except for Christmas which has much larger height.
- 10 parameters for 10 days of standard holiday. The Features are binary flags for "today is n days after the holiday" (0 ≦ n < 10); this representation can handle overlapping holidays, which change every year due to fixed vs non-fixed dates.
- 10 similar parameters for Chrismas

Nonlinear fitting with Gaussian works equally well and got shift ~ 0.45 days, amplitude ~ 0.15 in log(num_sold), and width σ ~ 3. Gaussian fitting have smaller statistical error due to smaller number of parameters, but can underfit; I did not do detailed comparison. Christmas is slightly wider but might have the same shift and width by choosing more than 1 day for Christmas, but I did not try. 

The holiday depends on the country:




(Inline figure temporarily not possible now?)

### What I could not understand

[residual]

[linear in residual]

- The residual look piecewise linear in time, which are different and discontinuous each year. The slope can be common among 3 countries, but offset seems different.
- There are large error at the begging of 2015
- There seem to be correlated error that continues for 1-2 months, but I didn't see any pattern

I looked at some external data, but did not find anything useful other than the GDP and the dates of official holidays. I thought probing for the linear function for the year 2019 using the public leaderboard would be useful but I was lazy doing that.

GDP by Carl McBride Ellis:
https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298911

There are several things that must be done, but I did not:

- Rounding the prediction to integer; beautiful figure by AmbrosM:
https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/301249

- Since the error seems Gaussian in log(num_sold), using mean squared error is reasonable, but prediction value should be tuned for SMAPE
