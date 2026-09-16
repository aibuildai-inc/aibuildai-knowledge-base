# 5th Place Solution

Competition: march-machine-learning-mania-2023
Rank: #5
Source: https://www.kaggle.com/c/march-machine-learning-mania-2023/discussion/401382

[Notebook link](https://www.kaggle.com/code/mbund1237/marchmadness/notebook)

**Private leaderboard place:** 5th
**Private leaderboard score:** 0.17619

### Intro
Before getting into it, I just want to thank Kaggle for getting all the different datasets together and for hosting this competition. I've learned a lot and this was a great way to challenge myself. I also want to thank anyone who has shared a notebook or participated in any discussion with the goal of helping others! However, it's sad to see that ~60% of the top 100 was taken up by a fork of one notebook (without any changes). I was surprised at the number (and profile rank) of people who put more value on winning an arbitrary medal vs. improving their skills and challenging themselves. I hope that future competitions will put more emphasis on original submissions.

### Background
I graduated in August 2022 with a BS in Business Administration from San Diego State University. I don't have any professional experience (yet) but hope to land a data-driven role. I am self-taught and took up machine learning at the beginning of 2023. However, I do have experience playing sports, sports analytics, and building betting models for basketball primarily (personal projects). I am definitely not the best modeler in the competition, but I do feel as if my knowledge in this subject specifically is what gave me a bit of an edge.

### Approach Introduction
One of my favorite movies of all time is Moneyball. In Moneyball, the Oakland Athletics took a statistical approach to building a baseball team. One of the main metrics they judged hitters by was OBP (on-base percentage). They strategized that winning in baseball can primarily be broken down to getting batters on base. I have always wanted to apply a similar strategy to basketball, but it's tough as it is a very different game. 

### My Approach
My approach to recreate the Moneyball strategy was to only use features that normalized any difference in pace of play, playstyle, bias, noise, etc. In order to do this, **I focused on how well teams played on a possession-by-possession basis only with an emphasis on their efficiency. Teams who play more efficiently per possession are more likely to beat their opponent, irrespective of strength/seed. ** I am somewhat confident that this approach would do fairly well every year and could possibly be expanded to the NBA as well.

### Features
During the testing phase, most of my time spent was on feature selection. As mentioned before, having previous knowledge and experience using some advanced basketball statistics made the feature selection much easier. I also decided to drop "NumOT" and "WLoc" from all of the datasets that I used. I view "NumOT" as noise due to the fact that it doesn't give any value in determining the strength of either team. "WLoc" is irrelevant in my opinion as I see the NCAA Tournament as a neutral location so there is no home-court advantage.

Here is a breakdown of each of my features, why I used them, and their importance (will average between the two teams... i.e. AWinRatio and BWinRatio will just be referred to as WinRatio and I will use their average importance):

| Feature | Formula | Explanation | Importance |
| --- | --- |
| Win Ratio (Win %) | WR = W / G | This one is fairly straightforward. A team with a higher win percentage is more likely the stronger team. Teams with a high win percentage and a low seed may be stronger than anticipated.  | 0.327 |
| Seed | - | This one is also fairly straightforward. A team with a high seed is more likely to be the stronger team than a low-seeded team. | 0.052 |
| PPP (points per possession) | PPP = PTS / Poss | PPP metric represents the efficiency at which a team is able to generate points on a possession-by-possession basis. The higher the PPP, the better a team is at generating efficient points each time they have possession. A low PPP may be indicative of bad shot selection (i.e. long 2's), poor efficiency, or bad offense. | 0.014 |
| Team | - | Some teams have performed better than others (and vice-versa). This may be due to better recruiting, good coaching, conference differences, or even luck.  | 0.028 |
| EFG% (Effective field goal %) | EFG =  (FG + 0.5 * 3P) / FGA | EFG is a shooting efficiency metric that accounts for the fact that a 3 pointer is worth 1 more point than a 2 pointer (50% more) when looking at FG% (field goal percentage). | 0.008 |
| TOV% | TOV% = (TOV / (FGA + 0.475 * FTA + TOV))  | TOV% (Turnover percentage) is an estimate for the number of times a team turns over the ball per 100 possessions. This can be used to measure a team’s ability to take care of the ball on offense as teams who turn the ball over often have lower efficiency and allow the defense to get back possession of the ball. Note: Some calculations use 0.44 as the weight instead of 0.475.  | 0.008 |
| OREB% | OREB% = OREB / (OREB / OppDREB) | OREB% (Offensive rebound percentage) is an estimate of how often a team gets an offensive rebound. This measurement is useful as a team with a high OREB% is more likely to create second chance opportunities after a missed shot.  | 0.008 |
| Set  | 1 = Tournament, 0 = Regular Season  | This feature was created by me in order to allow the model to possibly assign a different weight to regular season games vs tournament games. Winning a tournament game can be seen as more important than winning a regular season game.  | 0.008 |
| TS% | TS% = PTS / (2 * FGA + 0.475 * FTA)  | Another advanced shooting efficiency metric that takes into account free throws when calculating field goal percentage (FG%). Traditional FG% only utilizes total field goals (made and attempted) while TS% takes into account the difference between a free throw, 2 pointer, and 3 pointer when calculating. Note: Some calculations use 0.44 as the weight instead of 0.475.  | 0.008 |
| FTR | FTR = FTA / FGA | FTR (Free throw rate) represents how often a team is able to draw shooting fouls relative to how many shots they take. Teams with a high FTR are able to get to the free throw line often when attempting shots. This metric can be useful as a team with a high FTR is not only able to generate additional points but can also get the other team into foul trouble. | 0.007 |
| DREB% | DREB% = DREB / (DREB + OppOREB) | DREB% (Defensive rebound percentage) measures a team’s ability to secure a defensive rebound after a missed shot by the offense. Teams with a high DREB% minimize the opponent’s chances to get an offensive rebound after a missed shot and give their team possession of the ball. | 0.007 |
| Season | - | I wasn’t sure if I should leave this feature in but it seemed to perform well. My rational was that teams that have performed well the last few seasons may be more likely to perform well present day. It can also be used to quantify playstyle differences throughout the years (i.e. 3 pointers are taken at a much higher rate than in the early 2000’s). | 0.007 |
| 4F | -  | Please see references for an explanation. The 4 factors concept was created by Dean Oliver in his book, “Basketball On Paper”. | 0.007 |
| Gender | 1 = Men, 0 = Women | This feature is used to try and quantify the playstyle differences between men and women. Due to only training one model (for both men and women), I figured this could be a possible solution. Next year I hope to train two separate models. It didn’t carry any weight in the final feature importance list.  | 0.000 |
| ORTG | ORTG = 100 * (PTS / Poss) | ORTG (Offensive rating) is used to measure the efficiency of an offense. It’s value represents the number of points a team scores per 100 possessions. Looking back, PPP (points per possession) does almost the same thing- just doesn’t adjust it per 100 possessions. While testing the model in the March Madness Warmup, this held more importance than it did here. | 0.000 |
| DRTG | ORTG = 100 * (OppPTS / OppPoss) | DTRG (Defensive  rating) is used to measure the number of points a team’s defense allows per 100 possessions. Similar to ORTG, PPP is highly correlated with this. | 0.000 
| Opponent Stats | - | In addition to calculating all of these metrics for a team, I did it for their opponent’s averages as well. For example, if a team has a very low “OppEFG” for the season, it means their opponents do not shoot efficiently against them. This is highly indicative of a strong defense. Another example is OppFTR. If a team has a very low “OppFTR”, it means that the team does not foul shooters often and doesn’t allow the opposing team to shoot free throws at a high rate. In general, you want your Opponents’ metrics lower than yours (other than TOV%).  | See Full Importance List Below

[Full importance dataframe]

### Datasets & Modeling

**Datasets:** I only used a total of 6 datasets and did not incorporate any outside data (i.e. 538, Massey, KenPom). For both men's and women's I used TourneyDetailedResults, NCAATourneySeeds, and RegularSeasonDetailedResults. I also used the sample submission when creating my entry. 

**Scaling:** I tested RobustScaler, MinMaxScaler, and StandardScaler. I got the best CV scores using MinMaxScaler so that is what I used.

**Cross-validation:** I used KFold with shuffling and tested 5, 8, and 10 splits with 10 performing the best. With this said, this is one area where I need to improve, and hope to cross-validate more effectively.

**Modeling:** I decided to take a classification approach- 1 being a win and 0 being a loss. It seems as if many people took a regression approach to predict the scoring margin. However, I feel that this isn't the right way to approach the problem as we are not being scored on the accuracy of the margin. There is no difference if a team wins by 2 or 15 in this competition- only the fact that they won (or lost). Many people tried to extrapolate a win probability based off of this margin but I don't think that approach would be as accurate. 

I tested 5 different classification models, LogisticRegression, as well as a voting ensemble. The classification models included XGBoost, LightGBM, RandomForest, KNearestNeighbors, and StochasticGradientDescent. While testing, the voting ensemble was taking too long to train so I decided to scrap that approach. In the end, XGBoost performed the best so that is what I used for my submission.

### Post Processing
Unlike many others, I didn't manually override any of the predictions. I considered this, but felt it would be incorporating my own bias into the modeling (which I wanted to avoid). With this said, any manual overrides seem fairly safe for the women's tournament but much riskier for the men's. 

### Conclusion
I'm very happy with my results and am curious to see how this approach would score on previous tournaments. I fully intend to work on a submission for next year's tournament!

### Wishlist and suggestions
- Individual player data for each season
- Score the competition using the scoring margin rather than the probability of winning. There is a huge difference between a team winning by 20 vs 2, especially if you consider the seeds. A 16-seed blowing out a 1-seed by 20 points would be incredibly unlikely (looking at you Virginia) and could be more indicative of a team's strength.
- No direct forks of public notebooks (or cluster them as a team like I saw someone suggest)
- Sub-competitions? I feel as if this could be interesting. I saw a few people mention that this competition is getting a bit repetitive or boring. Having different sub-competitions for March Madness could be really fun. I.e. separate competitions or scoring for only the first round, whoever can predict the closest to the sum of all the scores, etc. Some people may enjoy the variety and additional challenge(s) it brings.

### References

About. THE POWER RANK. (n.d.). Retrieved from https://thepowerrank.com/cbb-analytics/
All 68 Ncaa Tournament Teams in 12 Minutes. (2023). Youtube. Retrieved from https://www.youtube.com/watch?v=1LUgGiw-cHw&.
Basketball Reference. (n.d.). Four factors. Four Factors. Retrieved from https://www.basketball-reference.com/about/factors.html
Kaggle. (2019). Ncaa March Madness Workshop | led by Darius Darius Barušauskas | Kaggle Days Paris. YouTube. Retrieved from https://www.youtube.com/watch?v=KmhGNc7gcCM.
Kotzias, K. (2018, March 15). The four factors of basketball as a measure of success. Statathlon. Retrieved from https://statathlon.com/four-factors-basketball-success/
squared2020. (2017, September 6). Introduction to oliver's four factors. Squared Statistics: Understanding Basketball Analytics. Retrieved from https://squared2020.com/2017/09/05/introduction-to-olivers-four-factors/  

Glossary links:
[Basketball Reference](https://www.basketball-reference.com/about/glossary.html#tsa)
[NBAStuffer](https://www.nbastuffer.com/analytics101/)

Thanks again and congratulations to all the winners! Feel free to ask questions below, I will answer them all to the best of my ability.
