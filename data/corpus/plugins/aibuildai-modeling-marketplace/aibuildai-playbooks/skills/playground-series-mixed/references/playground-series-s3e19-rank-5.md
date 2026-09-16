# 5th place solution

Competition: playground-series-s3e19
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s3e19/discussion/428347

First, I would thank organisers for a very interesting competition. My special thanks also to @siukeitin, @paddykb, and @ravi20076. Discussions with you helped me a lot!

I have previously published almost all my solution, which is based on the multiplication of several different factors (https://www.kaggle.com/competitions/playground-series-s3e19/discussion/424520). 
$$y_{t, c, s, p} = \text{GDP}(c, \eta) \cdot \text{F}(p, t) \cdot \text{S}(s) \cdot \text{w}(w) \cdot \text{Covid}(m, \eta)  \cdot \text{D}(d) \cdot \text{H}(c, d)$$

1) \\( \text{GDP}(c, \eta) \\) is relative GDP per capita in each country \\(c\\) ant year \\(\eta\\), which was sugested in [this post](https://www.kaggle.com/competitions/playground-series-s3e19/discussion/423725).

2) \\( \text{F}(p, t) \\) is the averaged sales of each product \\(p\\)  with applied fourier filtering (only few lower frequency component are left); \\(t\\) t is the consequentive number of the day, starting from 2017-01-01.

3) \\( \text{S}(s) \\) is the constant factor depending on store \\(s\\).

4) \\( \text{W}(w) \\) is the constant factor depending on weekday \\(w\\);

5) \\( \text{Covid}(m, \eta) \\) is the COVID factor depending on year \\( \eta \\) and on month \\( m \\). It is always equal to 1, except for year 2020. And in 2020 it is equal to the total sales made per month in 2020 divided by the averaged total sales made per month in all other years;

6) \\( \text{D}(d) = D_1\cos(\frac{\pi d}{365}) + D_2\sin(\frac{\pi d}{365})+D_3\cos(\frac{2\pi d}{365})+D_4\sin(\frac{2\pi d}{365}) \\) is the factor depending on day-of-year \\(d\\); It is responsible to lower sales in summer and higher sales in winter season. \\(D_j\\) values are selected with linear regression. To do this we also have to get rid of holidays.

7) \\( \text{H}(c, d) \\) is the holiday factor.  To deal with it, I noticed, that the form of each holiday response is the same. They all have gauss form with standard deviation equal to 2, and the maximum shifted forward at 4.5. I used ridge regression to estimate the amplitude of each such function. All holidays are grouped by their name, as python's holiday library recognise them.

https://www.kaggle.com/code/kdmitrie/5th-place-solution-timeseries-decomposition
