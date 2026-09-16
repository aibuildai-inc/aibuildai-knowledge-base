# 9th place solution

Competition: playground-series-s3e19
Rank: #9
Source: https://www.kaggle.com/c/playground-series-s3e19/discussion/428300

This was the first time I finished with single digit rank in a Playground/ Featured competition.

Many thanks to Kaggle for organizing the playground series and listening to @ravi20076 's feedback regarding the variety of challenges, and @onurkoc83 @nivedithavudayagiri for being great teammates!

In this post, I will be outlining my approach to the S3E19 competition.

**Handling 2020 sales data**

I smoothed the data using this dataset https://www.kaggle.com/datasets/shivanimalhotra91/playground-s3e19-covid-data-smoothed, not just from Mar to May 2020, but from **Mar to Dec 2020** [I also tried Jan to Dec 2020 but that got me a lower validation score] to remove the anomalous trend for the entire year.

The data now looks like this (with no signs of deviation)


**Preprocessing**

Added additional features such as
(a) Standard date features like year, month, dayofweek ...
(b) Holidays (ref: @nivedithavudayagiri)
(c) Seasonality features e.g. month_sin, month_cos


**Approach**

I did not do any cross validation. I used 2017-2020 smoothed data for training and 2021 for validation. The validation metric was not the entire 2021 data, but only the **SMAPE from Apr to Dec 2021** (https://www.kaggle.com/competitions/playground-series-s3e19/discussion/423657, which hinted that we will be evaluated on Apr to Dec data).

Postprocessing by multiplying final predictions was used, ref: @paddykb [Link](https://www.kaggle.com/competitions/playground-series-s3e19/discussion/425538).

During the competition, I published two public notebooks with different forecasting models, namely
(1) https://www.kaggle.com/code/yeoyunsianggeremie/s3e19-prophet-baseline-with-postprocessing
Validation score: 9.39
(5) https://www.kaggle.com/code/yeoyunsianggeremie/s3e19-catboost-smoothing-post-processing [Version 35]
Validation score: 8.50

**Both of the notebook submissions were used as candidates** for blending into the final submission, together with these

(2) https://www.kaggle.com/code/tetsutani/ps3e19-eda-ensemble-ml-pipeline-rnn-by-skorch by @tetsutani 
(3) https://www.kaggle.com/code/paddykb/ps-s3e19-gammy-sales [be careful!] by @paddykb 
(4) https://www.kaggle.com/code/oscarm524/ps-s3-ep19-eda-modeling-submission by @oscarm524 

After some experiments, the best subset of submissions for blending are (2), (3) and (5). This gave a public LB score of 5.42826 and a **private LB score of 6.01720, good for 9th place**.


**The final submission notebook is attached here:** https://www.kaggle.com/code/yeoyunsianggeremie/s3e19-9th-place-submission

EDIT: Could have been 6th if I chose the submission with the best Public LB instead of the best CV. Lol

Thanks for reading!
