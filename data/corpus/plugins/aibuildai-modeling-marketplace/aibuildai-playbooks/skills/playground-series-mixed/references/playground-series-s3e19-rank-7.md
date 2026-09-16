# 7th Place Solution

Competition: playground-series-s3e19
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s3e19/discussion/428514

Overall, this was a pretty enjoyable competition! The competition can be roughly divided into 3 parts/periods: implementing a solution, probing phase (credit to @paddykb), and ensemble. 

**Implementing a solution**

The whole first half of the competition was implementing this notebook [https://www.kaggle.com/code/kitadakiyoto/tpssep22-predict-by-linear-regression-1st-place ](url) onto this competition, using the holidays library to extract holidays and using the log of the relative GDP as an additional feature. Linear regression shows ~70 public MAPE, Random Forest shows ~30 public MAPE after some attempts at postprocessing.

**Probing**

Inspired by paddykb's findings in his notebook, I set the relative GDP to be 0.4 for all countries in 2022, effectively uplifting all countries to be around the same level. ~7.5 public MAPE after this one simple change. After some testing of standard regression models, Extra Trees was the best pick, with ~6.13 public MAPE after converting num_sold to int and deducting by 1. 

**Ensemble**

I got lazy here and just used my best public scoring submission and paddykb's public notebook (almost forgot about the surprise here). Inspired by the discussion [https://www.kaggle.com/competitions/playground-series-s3e19/discussion/428123](url), I added 1 to paddykb's public csv. The best private score of ~5.93 MAPE (~5.00 public MAPE) was obtained with 0.825*(paddykb's csv) + 0.175*(my csv), followed by rounding the numbers. 

**Some thoughts**

Blindly following public LB instead of using CV scores seemed to work out for me in this competition; I think the time series is too trend following & periodic for public LB to not follow in 2022 as well. Conclusively, I think some work and inspiration from public notebooks and discussions allowed me to get a decent placement, although luck is definitely involved, as in all things statistical.
