# 3rd place solution

Competition: tabular-playground-series-sep-2022
Rank: #3
Source: https://www.kaggle.com/c/tabular-playground-series-sep-2022/discussion/356643

3rd and I suspect also 4th, 5th, 6th & 7th (based on the public score -- apologies if it is not the case) are a variation on [this GAM](https://www.kaggle.com/code/paddykb/tps-2022-09-compare-to-best-public-notebook)

No tricks, no ensemble. I used *“leave 3 month chunks out”* to get a stableish CV. 

The GAM learns the patterns and behaviours from the training years, which are then projected forward based on strong assumptions about which of those behaviours continues. 

The only difference in the final submission is a change to capture *Christmas into New year* also using a spline. And, of course, blind luck being on holiday for the last week so I couldn’t tinker and do something silly :)
