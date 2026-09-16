# #1 solution - lucky error

Competition: womens-machine-learning-competition-2018
Rank: #1
Source: https://www.kaggle.com/c/womens-machine-learning-competition-2018/discussion/53597

This result was totally unexpected, as I only spent 4 hours in total in this one, as my main focus was men's competition for which I spent 3 days. However, the ideas in men's competition transferred well to women's, and I am happy they did!

First, I have not used any external data and only used regular season data. After my final submission I noticed a small error made from my side - I forgot to include stage2 regular season file, and therefore used Prelim2018 file. It turns out these files are very different, as 2017 season was also affected. If I have used full data, my same model would have placed in 10th place only! 

Surprisingly, my model did a good job at the only tourney upsets this year - #6 Oregon St. winning over #2 Baylor (16%) and #1 Notre Dame winning over #1 UConn (33%). 

My model is a single xgboost model. I only used tournament data as my training set. I modeled the win margin using MAE metric together with [cauchy][1] objective function, which is suitable for MAE optimization. Then the score difference predictions were used as inputs for smoothing splines GAM model to smoothly transform them to probabilities.

This year I focused on very careful feature selection and it turned out to be very important factor - I have tried but failed to create a better performing private LB model so far. So it seems my solution is close to optimal with the data and ideas I had.

My overall CV logloss was 0.4039551 (5-fold random split over all tournament data). CV year-by-year:

    Seas logloss
    2010 0.4131130
    2011 0.3587928
    2012 0.4074102
    2013 0.3928945
    2014 0.3836127
    2015 0.3529199
    2016 0.5117136
    2017 0.4111838

CV estimates are pretty much stable and this year's results were on average no different from previous tournaments.

Big thank you for organizers for adding women's competition this year and changing the rules not to share everyone's predictions during tournament - it really made the tournament more enjoyable to watch. 

Also big thanks to people who believed in me who have wanted me to win this competition!


  [1]: https://www.kaggle.com/c/allstate-claims-severity/discussion/24520#140163
