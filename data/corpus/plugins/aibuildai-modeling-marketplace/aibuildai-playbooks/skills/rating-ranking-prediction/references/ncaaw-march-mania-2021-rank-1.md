# 1st Place Solution - Brief Overview

Competition: ncaaw-march-mania-2021
Rank: #1
Source: https://www.kaggle.com/c/ncaaw-march-mania-2021/discussion/231528

Thanks again to everyone who has expressed their support for me after the end of the competition. Your words mean so much to me. If you want to know more about my experience in this competition, and my overall Kaggle journey, you can read about them [here](https://www.kaggle.com/c/ncaaw-march-mania-2021/discussion/230725).

Thank you all also for patiently awaiting this post. I am still trying to crawl out from underneath a big pile of backlogged work. I will provide here the basic outline of my solution, and hopefully, as the time permits, fill in some more details later. If you know anything about how I approach ML modeling, you would not be surprised to learn that my work is all over the place, both literally and Physically. I use one Kaggle notebook, two languages (R and Python), three of my own machines, and at least a couple dozen Python notebooks with names such as Untitled7.ipynb. You know the drill. So may take me w while to try to streamline my modeling process.

1. My basic modeling approach was very similar to the one that has been used in many top solutions over the past few years: use only season data for the team that had made it to the tournament, focus primarily on predicting the spread, and then try to convert those predictions into probabilities. 

2. For my features I relied on @imoore [R notebook](https://www.kaggle.com/imoore/2019m-1st-solution-with-parameter-optimization). It's been a while since I've done anything in R, I did not want to spend to much of my time trying to convert his work into Python, so I decided to just export the feature datasets inside of a [Kaggle notebook](https://www.kaggle.com/tunguz/2019-1st-solution-features-only/) and then download them to my local machine for further work in Python.

3. I wanted to see if I could make some other features that would be useful in predicting team rankings. A long, long time ago in grad school, I thought of an ingenious idea to calculate [Page Rank](https://en.wikipedia.org/wiki/PageRank) for NCAA teams based on their season performance. If I remember correctly, I had coded it up in Perl, and it gave a pretty decent overall consistency with other team rankings that are popular (Coaches polls, etc.). This time I decided to use the Rapids cuGraph library, and you can find the basic setup [here](https://www.kaggle.com/tunguz/team-rankings-with-rapids-and-cugraph/). The overall rankings did not seem as good with this approach as I had remembered them from my other work, but this feature seemed to add some signal to my models, so I had kept it. 

4. I then added the [538 features](https://www.kaggle.com/raddar/ncaa-women-538-team-ratings). 
Huge thanks to @raddar for providing those. 

5. Since we only have 538 features for the last four seasons, I needed to come up with a way to maximize their use for my models. I came up with a 2-stage modeling process: I would first create several metafeatures using stacking and data for all seasons, but with no 538 features. Then I would use those metafeatures in conjunction with the 538 features and data only for the last four seasons.

6. Finally I would do a lot of ensembling with the last four seasons, all with the usual suspects: XGBoost, LightGBM, HistGradientBoosting, etc. Do this with different subsets of features. All of the level 1 meta features were for predicting the spread, while all of the level 2 meta features were predicting the win probability. I was constantly concerned about overfitting, but it seems that in the end my ensemble did just fine. 😄

7. Postprocessing. As we all know, this competition creates an opportunity to gamble. In principle, the test set is sufficiently small that it's possible to "hand pick" all the game winners, but in practice that is a very dangerous and impossible to pull off correctly. Since we are allowed two submissions, a perfectly rational zero-risk approach would be to choose one winner for the final game from both possible options for each submission. (That's the so called "1-0" trick.) However, I  chose to instead bet on all the 1st and 2nd ranked teams in the first round, and then on Stanford and Connecticut in the second round. Fortunately for me, all of those bets worked, but the boost I got was not that significant, and I would have won even without it. Doing the "1-0" trick would have actually boosted my score more.
