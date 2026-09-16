# 2nd Place solution: The Godaddy Microbusiness Data Cleaning Challenge

Competition: godaddy-microbusiness-density-forecasting
Rank: #2
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/395264

Interesting problem, and it was also interesting to see the solutions came up with.   I'm pretty new to these competitions.  The biggest uncertainty to me is how much probing of the hidden data is allowed, especially in a case like this where it matters a lot for time series prediction.

I also though SMAPE was a weird choice of metric that didn't make sense in a business sense.  A change in a small county would have a large influence in SMAPE, where if you are trying to capture revenue streams, wouldn't you care a lot more about a change in the number of active forecasts in Los Angeles?  I can only imagine this would be useful for allocation of advertising dollars.

The code: [Godaddy Microbusiness Final](https://www.kaggle.com/code/danielphalen/godaddy-microbusiness-final/notebook?scriptVersionId=133942554)

## The biggest issue - Data quality ##

As has been pointed out by many people, SMAPE is a relative metric.  With the kind of forecast values we are looking at (~1.2-1.5 for 1 month, ~3.2 for months 3-5 together), we can look at the contribution of an individual CFIPS.  

$$\Delta_{SMAPE,i} = \frac{200}{n_{counties}} \frac{|F_i - A_i|}{F_i+A_i}$$

Which, in the case of something going from zero to non-zero can be ~0.0638 per month difference.  At the same time, a prediction for most of these models seems to be on the order of 0.5% change per month, which is far below the smallest change in active entries for many 25% of the counties.  

Second, there are many CFIPS where the data is terrible.  The hosts acknowledged that they had a methodology change in Jan 2021, leading to a number of jumps, and I suspect there was another change after the first month.  

The one which always bothered me was CFIPS 56033, Sheridan Co, WY, where there are 2.36 microbusinesses/working age person.  I can only guess there is some bug where if there is a misclassification they dump it in that CFIPS.  I also wondered if there was some fraud happening during COVID times as people chased PPP loans given the large jumps.  However, many of these issues would eventually revert, like the below:



So the question I think that make or breaks this, if we identify a large jump, will it revert?  In this I believe we are helped by the gap between December and the first forecast month, March.  I also hope GoDaddy gets better in their data collection procedures, which would help.  In then end, if they want to use this in business, it would probably not be that useful to have a large machine learning error correction model for data collection issues.

My solution is a mix of public leaderboard probing for individual CFIPS changes, reversion for outlier CFIPS, and a forecast of the smooth changes.  

### Public leaderboard - still very useful information ###

Given the equation above and as @petersorenson360 suggested, and that the leaderboard was active to 4 decimal places, you are sensitive to changes of less than 1% in the value an individual value.  Given you had about 20 days from revealed test to final submission deadline, you could probe about 100 CFIPS for final values.  It was actually easier if you made the value worse since the equation above would allow you to work out the exact number of active entries.  

Many large jumps would the revert, which we tried to work out how long that might take.  This will probably be the difference in the end of who wins and loses, and I think the couple month gap between the test and private data will help ensure they revert.   This was by far the top influence.

### Continuous Model ###

After seeing GiBa's @titericz notebook, his data cleaning method reminded me of something used in futures algorithmic trading, called a `continuous contract`.  Basically, you need to take a discontinuous series of prices and make it smooth.  So for what I will call the continuous forecast, I used a data cleaning method where I looked for large jumps in the number of active entries, then did a shift to smooth them out.  So the month where there was a large active jump became zero active jump.  This smoothing method gave the best CV score of the number that I tried.

I then setup the CV environment.  For the model used XGBoost, added extra indicators, and prevented peeking in the future:

- lagged density changes and lagged active values
- pct_bb, pct_college, pct_it, pct_foreign_born, median_hh_inc of the last year (Many public books used an implicit forward bias in their features by looking at 2021 census data when training on 2019 data)
- Labor Force and Labor force participation for the county
- 10 year average pct population change for the county
- latitude and longitude
- engagement, participation, and MAI_composite from the Godaddy website.
- The difference between the microbusiness density and the average of the neighboring counties, weighted by population.



I found a population cut of about 5,000 was helpful.  Rounding the model to an integer number of active helped a bit.   A lot of these external indicators proved more helpful for the longer term forecasts, where there is some reversion to an average.

Other notes:
- I actually found training on the 1 month forward change then feeding that prediction back into the model and recalculating all indicators gave the best CV, as opposed to directly forecasting the 3 month ahead forecast.  It turned out to be less biased.
- The `scale` variable used seemed to work in forecasting the public leaderboard, but did terribly in my CV environment.
- Again, noting that the last value is pretty good and I was using a roll forward model, I used the tuned public models to patch in a January forecast as an estimate for the ground truth, then rolled forward the XGBoost model from that.
