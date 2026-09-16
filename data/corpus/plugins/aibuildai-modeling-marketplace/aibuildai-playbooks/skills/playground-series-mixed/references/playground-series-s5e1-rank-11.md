# 9 Public, 11 Private (Single Model) - Solution Writeup (Public LB 0.04605, Private LB 0.04766)

Competition: playground-series-s5e1
Rank: #11
Source: https://www.kaggle.com/c/playground-series-s5e1/discussion/560569

First of all, thanks to Kaggle for organizing a playground forecasting competition after 16 months! (finally). This was the first time I got a top 1% in a competition **solo**, and done so **without** copying or ensembling public notebooks, but using my own intuition of what should work for this competition.

I will briefly outline my solution here while elaborating on parts not seen on the public notebooks/discussions. The solution code is [here](https://www.kaggle.com/code/yeoyunsianggeremie/ps5e1-11th-place-solution?scriptVersionId=220253622) - will be adding some comments to it later today.

(A) Baseline
=
As @siukeitin shared in [this discussion](https://www.kaggle.com/competitions/playground-series-s5e1/discussion/554349), the "distribution" of sales follow a certain pattern and can be decomposed by country (GDP Ratio), product and store

**Store**
No visible trend across years was observed -> a constant value was assigned across all years for each store
**Product**
Seasonality was observed with frequency either 1 or 2 years depending on product. Thus a Fourier series with frequency = 2 years was fitted for each product
 
**Country**
Yearly values of country GDP / GDP sum for each year is used
**Day of week**
Sunday > Saturday > Friday > any other days in sales volume

Note: Kenya and Canada sales should be excluded when computing these ratios as their NULLs may introduce bias into them.

(B) Holidays
=
Through EDA, we can find that the effect of a holiday lasts beyond the actual day of the holiday itself, and mostly up till 7-9 days after the holiday. 
1. The holidays are first sorted in order of date so that the most recent holiday is prioritised (in the case of overlapping holidays effect)
2. Group by the country and the holiday names
3. For each day from 1-9 compute an individual multiplier
3a. Assume ```day T``` is one of these days impacted by the holiday, compute ```sales for day T / sales for day T-7```, if ```day T-7``` is a holiday, propagate the ratio back until a non-holiday is found
3b. clip the minimum to 1

(C) Estimating the total sales and multiplier
=
Now that we have a rough idea of the "breakdown" of the sales, it leaves us to estimate the total sales for each calendar year from 2010 to 2019.

The simplest way would be to fit a linear least squares to estimate yearly sales from the total GDP (due to the high R^2 of 0.997) - but this comes with some challenges. 
1. As per the discussion in (A) - it was shown that Kenya's GDP from worldbank is too high. I subtracted a constant value of 200 from Kenya's GDP as a result
2. The linear model isn't a good fit for sales vs GDP as it violates the assumption where the errors are independent




While the line of least squares estimates the slope (or Total Sales / GDP) as 84.54, we notice that the residuals follow a periodic trend!
Denoting R(T) as the ratio of Total Sales/GDP for the year T:



Therefore, a multiplier is necessary and it should be expected that the multiplier for
- 2017 > 2016 (1.04)
- 2018 < 2017
- 2019 > 2018

Through LB Probing it was found that the optimal multiplier for 2017 is 1.08. Unfortunately, I did not find any heuristics to accurately estimate the multiplier and had to guess it. If anyone has found one feel free to leave it in the comments

For one submission I chose
- 2017: 1.08, 2018: 1.07, 2019: 1.11

For the other submission I chose
- 1.08 for all three years

The second submission scored better on the private LB, which showed that the estimation of 1.11 as the multiplier for 2019 is not accurate.

Notes
= 
I want to express my sincere gratitude to @cdeotte for sharing beginner-friendly tutorials over the years! Back then when I was competing for the first time in LLM Science Exam, your [starter notebooks](https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-2) were incredibly helpful in guiding us toward improving our solution, especially during the times when I, and likely many others, were struggling. Your generosity in sharing knowledge has been invaluable, and I truly appreciate the impact it has had on my growth. Thank you!
