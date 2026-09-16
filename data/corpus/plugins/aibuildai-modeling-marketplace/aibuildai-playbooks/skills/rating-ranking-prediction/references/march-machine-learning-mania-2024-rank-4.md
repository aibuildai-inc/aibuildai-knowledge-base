# March ML Mania 2024

Competition: march-machine-learning-mania-2024
Rank: #4
Source: https://www.kaggle.com/c/march-machine-learning-mania-2024/discussion/494407

**Background**

I'm a data scientist with a background in math. College basketball fan and I've been making March Madness brackets for quite a few years.

Huge thank you to the competition sponsors / organizers for putting this together! The last-minute scoring metric change made this quite stressful, but I think it was ultimately necessary. In the early stages of this competition I (with many others) was concerned that this would be less of a predictive modeling competition and more of a game theory / portfolio optimization contest.

**Overview of the Approach**

Unlike other folks who have their own pet models that they've been iterating across each year's March Mania competitions, my overall strategy was pretty heavily indexed on what performed well last year (and past years in general).

The main strategy was to start with the 2023 winning submission ([paris madness](https://www.kaggle.com/code/rustyb/paris-madness-2023)), itself based on raddar's code which so many previous competition medalists had used, then modify the feature engineering and CV setup. I updated the feature engineering based on ideas that other past top scorers shared in their own writeups ([discussion post from 2023 5th place](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/discussion/401382) was particularly informative) and tweaked the CV strat to sample by year instead of by game. The other strategy was to take Silver's, sharpened with goto conversion, as a way of producing a baseline comparison while introducing variance between my two submissions. And of course the one I spent less time on ended up scoring better...

**Details of the submission**

Because each submitted bracket had to follow valid tournament paths, assigned probs had to be nonincreasing; it seemed to me that it would be really tough to stage a comeback if you made an early-round gamble that didn't pay off. Otherwise, the intuition for odds-adjustment is well-explained in the [notebook](https://www.kaggle.com/code/kaito510/updated-1xgold-2xsilvers-key-ingredient) (long-shot bias, strong teams resting their best players during the regular season, different playing conditions during the tourney).

Given the emphasis on favorites, the other question was whether to clip / override the probabilities, which I ultimately decided against:
1. Felt too arbitrary and didn't seem to provide meaningful improvement in backtesting.
2. The baseline already bet pretty heavily against the underdogs, and it seemed like a lot of other teams were overweighting top seeds already.
 
So the final notebook looked like this:
- In order to map teams to matchups properly, I created a transition matrix / dictionary for how every seed would progress through the tournament and the round in which they would face every possible opponent.
- Apply goto conversion to matchup odds.
- Bracket simulation code was efficient enough that there wasn't really a reason not to submit the full 10k brackets for granularity.

After generating the final brackets, some quick visualizations of each team's probabilities as a sanity check, plus the change in likelihood of advancing before vs. after converting (looked slightly wonky due to play-in rounds).

**Sources**
- [Submission Notebook](https://www.kaggle.com/code/tztang/pub-march-machine-learning-mania-2024)
- Ravi's [Discussion post](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/480346) with previous years' winning materials

In particular, please go upvote the following notebooks if you haven't already. They're incredibly generous to share stuff that's this well-made with the broader community, and I know many others have benefited from them as well:
- [simulation NB](https://www.kaggle.com/code/lennarthaupts/simulate-n-brackets): Lennart is an actual lifesaver; this is far faster and cleaner than what I would've produced myself.
- [conversion NB](https://www.kaggle.com/code/kaito510/updated-1xgold-2xsilvers-key-ingredient): goto conversion has showed up in quite a few spots that aren't directly related to sports / odds betting, including the recent Optiver competition; definitely a powerful tool to have handy in the future.

**Thoughts on Future Competitions**

A lot of ink has already been spilled by smarter people on the matter, here's my 2c:

I also support MAE point differential. Anecdotally, definitely saw some 'momentum' in LB movement (see [this comment](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/486396#2716145)) where movement up/down the rankings continues as the competition progresses. Predicting point differential would allow for more (exciting) late-stage prediction upsets.

I don't support reducing two competition submissions to one: 
1. Other Kaggle competitions, across a broad range of subjects / prediction types, allow multiple submissions, with two being standard; I don't see a sufficiently compelling reason that March Madness should somehow be an exception. 
2. People naturally have multiple ways of approaching the problem, and giving room to field a more ambitious / diverse strategies is a good thing.
3. It provides some amount of insurance against one of your submission notebooks breaking.
