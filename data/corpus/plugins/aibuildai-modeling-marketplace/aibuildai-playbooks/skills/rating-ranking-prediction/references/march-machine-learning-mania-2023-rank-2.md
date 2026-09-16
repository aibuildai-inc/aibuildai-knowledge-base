# 2nd Place Solution

Competition: march-machine-learning-mania-2023
Rank: #2
Source: https://www.kaggle.com/c/march-machine-learning-mania-2023/discussion/401578

**Private leaderboard place:** 2nd
**Private leaderboard score:** 0.17522

UPDATE: Notebooks now available. Each notebook is seeded with public data, built from the prior notebook, to allow anyone to run any notebook. This keeps RAM requirements under the 30GB limit for non-paying accounts. 
1. **[Prep men's and women's data](https://www.kaggle.com/code/matthiask/01-prep-march-mania-2023-data-round-1)**
2. **[Feature engineering](https://www.kaggle.com/code/matthiask/02-march-mania-2023-feature-engineering)**
3. **[Modeling](https://www.kaggle.com/code/matthiask/03-march-mania-2023-modeling)**
4. **[Smoothing predictions](https://www.kaggle.com/code/matthiask/04-march-mania-2023-smooth-predictions)**
5.  **[Model evaluation technique](https://www.kaggle.com/code/matthiask/05-march-mania-2023-submission-evaluation)**

### Intro
My original solution used a blend of pre- and post-processing steps in R, and API calls to DataRobot's AutoML platform to leverage feature engineering and AutoML functionality. I happen to work at DataRobot and have access to their most recent software, but I understand that many do not. For that reason I have attempted to replicate the entire process in R, where I have achieved outputs with > 0.993 correlation to my original final submission and a very similar leaderboard score. I have uploaded a series of notebooks linked above.

I have a BA in mathematics and an MS in statistics, with 10+ years of professional experience as a university professor, actuary, and data scientist. I would be remiss not to point out that with just 126 scoring observations, I very well could have been lucky. But I am optimistic about some of these ideas working year-over-year. Here is the basic pipeline:

R: pre-processing -> R: conference effects offset -> DataRobot time-aware feature engineering -> DataRobot AutoML -> R: smoothing probability predictions and outputting in Kaggle format.

### Primary objective and methodology
I wanted to find a way to leverage information from entire seasons, while also allowing the model(s) to focus on how teams are playing leading up to the tournament, and focus on games that are more like tournament games. I leveraged AutoML to save time, because my 5- and 7-year-olds are my real bosses, and they are demanding ;-) To that end, here are the most unique things about my approach.

1) Derived conference effects for every season from the regular season non-conference schedule, i.e., information that would be available when the tournament starts. This came in iteration 2 or 3, when I saw the South Dakota State women as my #2 overall team. Sure they went 18-0 in conference, but they also play in the Summit conference, which isn't very good. Clearly my model was not picking up conference effects very well before this addition.

For this, I used R's glmnet on only non-conference, regular season games, and I leveraged four primary features: 
- each team's in-conference point differential to control for how good the team is within their conference
- home court indicator
- the season (as a factor), as many conferences have gone up and down over the years
- conference labels (and interaction between season and conference). This is the main effect I'm looking for, really. The rest are just controls.

Maybe the real magic here is in the creation of the model matrix, where instead of duplicating conference features for home and away, I made a single column for each conference. The home team's conference was labeled +1 and the away team's conference was labeled -1. This allowed me to isolate an entire conference's information into one column, rather than splitting it up between home[conf] and away[conf]. I used two targets: home team's point differential in the game and home team win indicator. Then I extracted the season-conference coefficients for each of home margin and home win models (log-odds), and appended them onto each game in my training dataset, to the home and away teams, respectively. I took the difference between the home team's conference effects and away team's conference effects, and these became offsets in my main models to help control for conference effects that are hard to pick up in the relatively few tournament games that are played.

2) Related to the above, I had to do something about neutral site games, as the entire NCAA tournament is played on neutral site courts (for the men, and some for the women, as well). So from the Kaggle-provided data, I duplicated all neutral site rows flipping "home team" and "away team", and used a "home" indicator = 0 for when games were on neutral courts. This made all neutral site games have similar impacts on the influence of "home" and "away" features, as well as doubling the weight of most NCAA tournament games in model training. When I went to predict tournament games, I predicted two rows per every matchup and then averaged the results. 

3) I built four models in the main training portion of the work. Point differential and home win models for each of men and women. There was more information given for men's tournaments, and I thought it best to just separate out the genders. I also believe you can extract more information from differentials--as a 20-point win isn't the same as a 1-point win--so I wanted to make sure I adjusted my win probabilities by margins. To do so, using the outputs of the four models described above, I built a `gam()` with the `mgcv` package in R that regressed log-odds of predicted home win probability on the predicted home point differentials to "smooth" the probabilities out and leverage those point differentials. Anecdotally, this helped the predictions for the Iowa women, as their predicted point differentials were relatively greater than their original predicted win probabilities. 

4) Time-aware feature engineering: I noted that I wanted to identify teams that may be getting relatively better or worse, and this is how I achieved it using DataRobot's Feature Discovery tool:
- Derived rolling 2-week averages, medians, maxes, mins, and standard deviations of each Massey ranking (for men)
- Derived rolling averages, medians, maxes, mins, and standard deviations for box score statistics, going back 6 months (effectively all season), 3 months, 1 month, and 2 weeks. 
- In addition to the box score stats provided, I derived team differentials for each game that were aggregated up. So, for example, for the designated home team in an upcoming game I would have their "average offensive rebounding differential over the past 2 weeks" as one such feature.
- For all these box score statistics, I normalized them before aggregation on a per-67 possessions basis. 

5) AutoML revealed that elastic net regressors did very well on this data. I think that makes some sense in sports, as typically effects are pretty monotonic and linear. You can't really be "too good" at something, such that you'd need xgboost or a comparable algorithm to detect non-monotonicity or non-linearity. 

### Top features
- The home effect was quite large, equivalent to between 3-4 points per game. For example, the men's (un-standardized) coefficient effect was +1.72, which represents the gap between playing at home and playing on a neutral court. Thus the home-vs.-away effect is more like 2x1.72 = 3.44, which jives with what you can typically infer from betting lines.
- Conference effects were very large, especially in early round games where you often see power conferences playing mid-majors.
- Lagged, aggregated offensive rebounding differential: showed up highest for men, less so for women.
- Lagged, aggregated versions of score differential (points, FGM, FTM): showed up high for men *and* women. In general, longer windows were more valuable (e.g., whole season over 2-week lag), but FTM for men was important in the last month. I hypothesize that getting to the line is something a team has a fair amount of control over, and there is signal in shorter rolling windows. 
- Individual Massey ranking systems showed up early and often for men. I would have thought some of my aggregations (like median rank across systems) would have been top contributors, but instead it was DOK and 7OT that both made the top 10.
