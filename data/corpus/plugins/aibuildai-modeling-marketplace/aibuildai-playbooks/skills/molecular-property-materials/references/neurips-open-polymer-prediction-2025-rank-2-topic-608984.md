# 2nd Place Solution

Competition: neurips-open-polymer-prediction-2025
Rank: #2
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/2nd-place-solution

# Initial Observations

I started by analyzing the best-scoring public notebooks at the time and the public leaderboard, leading to the following observations:
- There were many different models among the top-scoring public notebooks.
- The top public notebooks weren't far behind the top performers on the public leaderboard.
- Complex models didn't score significantly better than simpler ones.
- There was a benefit to ensembling.

# Choosing a Strategy

I entered this competition without domain expertise in chemistry. Furthermore, since many different models, both simple and complex, were getting very similar results, I did not see it as productive to create my own models. Rather, I would rely on what seemed promising from public notebooks as a starting point. I used several different public notebooks as a base over the course of the competition.

I'm not going to go into details about the models I ended up using, because I was treating them as black boxes. I focused my early efforts on optimizing the ensembling step at the end.

# The Data Anomaly

After the test data was changed, it was noticed that adding 273.15 to Tg would greatly increase public leaderboard score. This was attributed to a units issue (Celsius and Kelvin). But upon my own investigation, I noticed that adding 300 would yield an even higher score than 273.15. After the units issue was fixed, I found that adding 30 still yielded an improved score. This became the focus of my efforts. I first checked the other targets to see if they had a similar issue, but found that they did not. So, I focused on finding the best adjustment for Tg. While I tried many things, in the end I found a simple +40 shift worked best.

# Multiply by 9/5, Then Add 32?

Towards the end of the competition, I was trying to determine why a shift was effective and realized it could be another units issue. I tried the Celsius -> Fahrenheit conversion formula ((9/5)x + 32), but it reduced the public leaderboard score, so I abandoned the idea. However, after the competition ended I noticed that this submission had a private LB score of 0.068, significantly better than the 1st place submission.

While this looks a units issue, this is not necessarily the case. A transformation being effective does not mean it is optimal. After the competition ended, I found that using (9/5x) + 45 performs even better (0.066 private LB score).

# Conclusion

After the competition ended, I went back to my very first submission, which used an ExtraTreesRegressor and would have placed around 1300th on the private LB. I added the (9/5) x + 32 transformation and reran it. The resulting private LB score — 0.077 — was the same as my final submission (which used the less effective +40 transformation with an ensemble of models).

So In the end, despite there being five properties to predict, performance in this competition was determined by a distribution shift in just one property. This distribution shift was present in the public leaderboard data, but it was stronger in the private leaderboard data.
