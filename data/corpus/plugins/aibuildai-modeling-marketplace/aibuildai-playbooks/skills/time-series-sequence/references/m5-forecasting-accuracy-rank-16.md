# 16th Place Solution - Kalman Filters and Boosted Trees

Competition: m5-forecasting-accuracy
Rank: #16
Source: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/169085

I just posted my team's solution [on my company Nousot's blog](https://nousot.com/blog/how-we-won-gold/) and our code is [on Github](https://github.com/nousot/Public/tree/main/Kaggle/M5). In the article, the Section entitled "The Public Notebook Component," is where the methodological description starts. Cheers to @kyakovlev for the excellent data pipelines.

Our custom component used Kalman Filters via the R package [KFAS](https://cran.r-project.org/web/packages/KFAS/index.html), an awesome time-series tool if you gravitate towards state space modeling. Our data pipeline for this solution was constructed from scratch and relatively simple. However, at the end of the contest we realized that our solution was loosing big for the sparse series, so we used a loess fit to help us choose between a boosted tree solution from the public notebooks and the Kalman Filter solution for each time series. I suppose that makes it a "mixture of experts."

Here's a case where the Kalman Filter looked more reasonable to us than the boosted tree solution:



And here's a case where the boosted tree's solution was more reasonable:


It was a great contest. Thanks everyone who competed!

Ben
