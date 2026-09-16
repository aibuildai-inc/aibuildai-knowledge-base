# 4th place solution

Competition: playground-series-s5e1
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s5e1/discussion/560692

Congratulations @georgekoussa for his winning notebook! And congratulations to @kdmitrie for a great 3rd place. Whenever I was in first place for a short time, he quickly put me in second place a couple of hours later 😉 

This forecasting competition with an artificial dataset was kind of like a puzzle with it's various ratios to be computed. The total sales per year, however, remained a mistery to me. So I'm glad I ended on fourth place, just like on public LB. I expected a greater shakeup.

In a nutshell, this is my solution:
- See the yearly totals as given (I used a mean from earlier predictions and some public notebooks). All the following steps refer to ratios.
- Use World Bank GDP/capita figures per year for country ratios. Since there were some major discrepancies to the country ratios in the training data, I used a simple scipy linear regresssion to make the ratios fit better, especially for Kenya.
- Use constant store ratios. I could not identify any seasonal or whatsoever patterns.
- For each country, separately:
   - Compute the mean product ratios per day, separately for even and odd years (looking at the sincos curves there's an obvious two-year-pattern), then apply some FFT smoothing (thanks @kdmitrie who discussed it somewhere)
   - For the day-of-year-ratios I did linear regression with Sklearn's Ridge and HuberRegressor (not much of a difference). I did some extensive feature engineering. Besides sinus-cosinus features and day-of-week, I tried lots of country-specific holidays, both movable and immovable. The peaks in sale figures usually occurred some days after the holidays. Since Ridge runs pretty fast, I tried to identify the very days that worked per country. Interestingly, some country/holiday combinations had an effect even though the holidays library didn't have them.
- Finally, compute the absolute figures from the ratios and yearly totals.

Since the country ratios were somewhat flawed, I probed some factors (like Kenya * 1.012) against the public LB once I had no good ideas left to submit. But it did not make much of a difference and posed a risk with relation to private LB.

My best notebook reached 0.04385 on public lb and 0.4560 on private lb, both earning me fourth place.
