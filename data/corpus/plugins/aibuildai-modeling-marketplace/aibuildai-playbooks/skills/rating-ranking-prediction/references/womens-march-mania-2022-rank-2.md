# 2nd Place Approach

Competition: womens-march-mania-2022
Rank: #2
Source: https://www.kaggle.com/c/womens-march-mania-2022/discussion/316966

Congratulations to all, including the South Carolina Gamecocks! Thanks to Kaggle for running the contests again and for prompt leaderboard updates! Thanks also to Mark McClure for surely the best user-generated submission scoring UI ever! Providing this tool and updating the scores on time is certainly worth some kind of *community-leader* or *forum hero* prize 👏.

My submissions this year are exactly the same as my 2021 code (10th Mens, 12th Womens).

### Base Model

For the Women's tournament it is an approximation of the [Five38 match probabilities](https://fivethirtyeight.com/features/how-our-march-madness-predictions-work-2/):

	sub["Pred"] = 1. / (1 + 10**(rdiff * (30.464 / 400)))

`rdiff` is the difference in Five38 team ratings; and using their more precise predictions for round 1:

	idx = sub.Round == 1
	sub.loc[idx, "Pred"] = sub.loc[idx, "LTeamID"].map(five38.rd2_win)

That would score 0.42217 ~ 31st place. It tends to be too confident for later matches, which might be important when you see what happens to these predictions next...

### Overrides

There are a couple of *safe* ways to split two submissions:

 - split 0/1 for all possible final games, but then what if if the final is unbalanced? e.g. 2021.
 - split the first round match closest to 0.5 &rarr; 1 in one sub and 0.5 &rarr; 0 in the other

The latter guarantees -ln(0.5)/63 = 0.011 saving of log loss.

Instead I used the same [Madtown](https://www.kaggle.com/competitions/mens-machine-learning-competition-2019/discussion/90254) technique as [last year](https://www.kaggle.com/competitions/ncaam-march-mania-2021/discussion/231068), which I now realise is a generalisation of the second option:

 - `preds[(abs(preds-0.5)<T)]` &rarr; [ `P` , `(1-P)` ]

For the second "safe R1 gamble" (R1 matches only) T is nearly zero (such that only one match is affected) and P is 0.

For "Madtown" gambling for the Women's tournament, I found more conservative settings appear to work better, and used T=0.16 and P=0.36, so all predictions between 0.34 - 0.66 are set to 0.36 in one sub & 0.64 in the other.

Here are my scored predictions:



 - Y scale is the probability for the favourite
 - X axis shows round and match winner
 - X is sorted [round 1, round 2, other rounds], then by prob
 - blue = model predictions
 - black = unused original model (0.36 in one sub, 0.64 in another)
 - red stars = log loss scores (right Y axis)
 - black line = original model score 0.42217
 - red line = best submission score 0.38864

Instead of uncertain predictions around ~0.5, which will all score ~0.693, we get -ln(0.36)=0.446285 or -ln(0.64)=1.021679, hopefully more 0.446285's in one submission, pushing average logloss down.

It turns out 13 *close* matches were scored that had these diverging predictions, and interestingly, **any** set of 0.36/0.64 overrides would have resulted in a gold medal:



So over **all** possibilities for a random seed to reassign the ~0.5 &rarr; \[ 0.36, 0.64 \] matches, the worst rank would be 8th, with a 42% chance of 8th and 58% chance of top 3! My actual score of 0.38864 is from having a 9:4 split. Of course, those particular 13 matches being scored was not a given, it does not mean I was guaranteed a top 8 finish from the outset.

I'm guessing the impressive-looking rankings above are partly due to other teams gambling with a 0.99+ override on Baylor (seeded 2), who lost in round 2.

Thanks for reading! Any questions would be welcome...

### P.S.

Same approach in Men's (but with Pystan Bayesian model as baseline) is currently a 16:14 split, 0.67280 logloss, 542nd place...

### Code

**Updated 7th April**: [2nd Place NCAAW 2022](https://www.kaggle.com/code/jtrotman/2nd-place-ncaaw-2022) notebook.
