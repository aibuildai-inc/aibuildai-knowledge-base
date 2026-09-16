# 4th Place Solution

Competition: march-machine-learning-mania-2023
Rank: #4
Source: https://www.kaggle.com/c/march-machine-learning-mania-2023/discussion/401588

Details:
Private Leaderboard Score: 0.17557
Private Leaderboard Place: 4th
Name: Nicholas Hilton
Location: Brooklyn, NY
Email: nicholas.w.hilton@gmail.com

## My Background
I graduated from the University of Cambridge with a maths degree in 2015. From there I worked at Manchester City Football Club before teaching myself how to code and getting a job at Tribe Dynamics, a start up in San Francisco. There I learnt how to code professionally and data science skills. After 5 years in SF I left Tribe Dynamics to work as an Machine Learning Engineer for Gro Intelligence in New York where I am today.

I have been doing this competition since 2018 and have enjoyed it every year as I both enjoy the actual competition on TV and it gives me a chance to do some data science outside of my job. I used roughly the same method every year and it has performed a bit better every year as I make improvements. In total I spent about 3 weeks total on the competition over the last 5 years, this year making a few small tweaks before running my training process with the new data.

## Model Summary
My model implements an ELO rating system and then uses those ratings to produce predictions for each matchup, adjusting for a few other features (rebounding, field goal percentage (overall and 3 point), seed). The rating difference and adjustments are then passed into a link function which converts the adjusted rating difference to a probability for the matchup.

The training process involved tuning the parameters of the model, mainly the feature adjustments and the rate at which ELO ratings are updated after each match. A single training run would simulate all matches from 2010, keeping track of ratings over time, using the tournament score for each season as the metric to evaluate that set of params on. During the regular season there was also an adjustment for home or away teams but in the tournament all matches were treated as neutral.

I think the training process could be optimised in the future as I ran a grid search over a sensible parameter space and let my macbook run all day chugging through the param space.

### Final Params
Final ELO ratings are spread over a ~400 point range for tournament teams so weights below are related to that  rating spread.

*Mens:*
'k': 150  # how fast the ELO ratings update - at the high end of the param range tested
'seed': -40  # the seed difference to adjust - Medium End
'link': 'N'  # The response function to turn a rating diff into a probability - Normal distribution
'fgp': 1200.0  # Field Goal Percentage difference - High end
'fgp3': -10.0  # Field Goal 3pt Percentage difference - Low end
'reb': 15.0  # Rebounding Average difference - Medium End


*Womens:*
'k': 80  # Medium End
'seed': -40  # Medium End
'link': 'L'  # Logistic Function
'fgp': 1200.0  # High end
'fgp3': 0.0  #  Low end
'reb': 15.0  # Medium End

## References

Model Repo can be found [here](https://github.com/NickHilton/March-Madness)
Write up of details on the Elo system (for soccer but the initial work I based this model off) can be found [here](https://drive.google.com/file/d/1mCGCTWuxvyWAXaOhiqt0ncUExs4LsOPf/view)
The idea is similar to 538’s methodology [here](https://fivethirtyeight.com/methodology/how-our-march-madness-predictions-work-2/)

## Interesting Insights

I think a key improvement I made to the model last year (when I placed in the top 50) and which probably helped me this year, was simulating the tournament matches and updating the ratings throughout the tournament. I.e. If a matchup was a R2 matchup between W1 and W8 then W1 must have beaten W16 and W8 must have beaten W9 and so you can update the ratings before predicting each matchup in the tournament. I found a rating system which had a higher importance on recent form, performed better this year, and probably helped with the number of upsets that happened (Iowa for example in the women’s tournament I ended up rating highly at the end)

Another thing which helped me a bit was picking a team to do well and adjusting their ratings as a sort of gamble. I don’t think its as important now there are 126 games, but I picked Princeton to do well in the mens tournament and updated their ratings to do so. I figured that if you are competing in a pool of >1k entrants, who are all likely to have good models, having a ~5% chance of a big boost to your score means that whilst your mean score might go down, your likelihood of winning goes up (go big or go home). That being said, I only should have picked them to win 2 games to get the boost I needed as when they lost their third game most of the advantage I had was wiped out. In hindsight, rerunning my model without this manual choice gives a score which would still have put me in 4th place.

## Model Simplicity

My model is actually relatively simple from a Machine Learning point of view. It doesn’t use any advanced ML libraries, instead it is a tried and tested algorithm for rating sports teams/players (mainly Chess) and I was happy to have found a way to apply this to the tournament. I think the time consuming part came from building out the actual code to implement the ELO model with the data structured as it is as well as running the grid search over the param space.

## Final Thoughts

I thoroughly enjoyed this competition, I have done in prior years when I wasn’t near the top as it takes one of the highlights of the sports year and allows me to nerd out on it. I will keep doing this competition and look forward to thinking of improvements to my model for next year. 

I also appreciate that with the scores so tight at the top of the leaderboard, and games coming down to final shots regularly, a few baskets made or missed could have sent me tumbling down the leaderboard. The beauty of competitions like these based on sports is the drama that unfolds as you watch your model perform in real time, and it really was a fun couple of weeks, and a good time to get a bit lucky.
