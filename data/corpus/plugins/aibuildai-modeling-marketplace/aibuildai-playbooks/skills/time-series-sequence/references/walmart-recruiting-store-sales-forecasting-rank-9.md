# Thank You and #2 rank model

Competition: walmart-recruiting-store-sales-forecasting
Rank: #9
Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023#43874

<p>I used all Python (pandas, sklearn, statsmodels) for all of my work.</p>
<p>The way I dealt with the holidays was to create distribution of sales for each holiday based on the date, with 3 parameters, width, skew and location (relative to the actual date). I made these distributions on a daily grid and then summed them up to weekly totals. Using skewed distributions was key here for me. I largely fit the parameters by eye to begin with, but I was working on more automated methods towards the end but ran out of time. I think I could have squeezed a lot more out of this method.</p>
<p>For each store/dept combination, after calculating the trend in the data, I used a linear model with L1 regularization to fit the holidays to the detrended data. Then after subtracting the holiday fit and trend, I took an average of value for the week over the years to find the residual weekly cycle that was not due to the holidays.</p>
<p>Then I fit the trend + deseasonalized data using the Unemployment, Fuel Price and CPI, using another linear model with L1 regularization. I calculated the missing data using an simple AR model. This fit gave a small improvement over using the pure trend.</p>
<p>I used cv over the first two years / last 39 weeks split to pick whether the trend was constant or linear, and also to look for bad fits, for which I looked for fixes. For example something happend at store 14, which caused a dramatic drop in sales across all departments, so I applied a step function to account for this.</p>
<p>I ignored the Markdown data. My conclusion was that since we only had 1 year of markdown data it was impossible to extract anything useful from it, as we could not see the effect from one year to the next.</p>
