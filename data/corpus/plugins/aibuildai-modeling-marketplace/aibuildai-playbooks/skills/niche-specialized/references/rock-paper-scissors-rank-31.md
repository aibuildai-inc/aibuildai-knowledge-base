# Tony Robinson (temporary gold zone solution)

Competition: rock-paper-scissors
Rank: #31
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/217561

As we are all writing up, here goes.

Like others, I have an ensemble of strategies/bots that together make my agent.  However, l wasn't all that interested in throwing everything in, even though that would have resulted in higher scores, I was more interested in how all the bits fit together.  So I only had a very minimal set of predictors:

-  PRED: a set that looked back 0 to 4 'actions' where an action is a RPS choice.  Looking back 2 actions you take your own and the opponents last move into account, this I called PRED2.  Looking back 1 or 3 actions rotated/rolled the space to normalise and undid the transformation on prediction.  This game is near random so to look back n actions gives a state space of 3^n and 4^3 = 256 so I didn't think it worth going further.  These predictors output a distribution and which could be set to model the opponent or the agent.  There observations were exponentially weighted.
- LSTM: LSTMs were trained on [rps-episode](https://www.kaggle.com/tonyrobinson/rps-episode).  These got quite big, 1024 hidden units and 2 layers gave the best validation scores.   I never quite worked out if they helped, it really depended on what techniques were popular at the top of the leaderboard.

Overall I think my LSTMs were probably a waste of time so really all my work was based on nothing more than very simple counting based n-grams.  Score zero for innovation here....

However, what I was really interested in was what could be called 'non-stationary time series prediction', that is we know that the opponent is going to change strategy, so I then built a set of strategies on top of my PRED and LSTM low level bots.   By caching the results (on 'step') it's easy to wrap the low level in very many other levels for minimal computational overheard.   I used:

- ROLL:   This simply takes the ouput of another bot and calls np.roll() on it.   This is a 2pi/3 rotation not 2pi/4 like the excellent work of @superant - if I were to start again I'd use the z plane representation everywhere.
- GAIN:  This takes a distribution of expected moves, converts to expected gain (as in [From prediction to action](https://www.kaggle.com/c/rock-paper-scissors/discussion/208242)), zeros the negative entries and normalsises.
- ENHANCE:  This simply sets the minimum probablily to zeroand renormalises (again from From prediction to action](https://www.kaggle.com/c/rock-paper-scissors/discussion/208242)).
- FLATTEN:  This simply inverts the distribution so you end up playnig what the opponent probably didn't expect.   This was suprisingly effective on it's own for the first half of the competiion and one agent even came top near the end (early version at [flatten](https://www.kaggle.com/tonyrobinson/flatten))
- RLS:  This takes a set of bots and uses the [Recursive Least Squares](https://en.wikipedia.org/wiki/Recursive_least_squares_filter) from (padasip)[https://matousc89.github.io/padasip/] to combine them.   This is really helpful when combining different action predictors, I just threw everything in, assumes that RMS was valid (yikes!) and used the result.

Also, I've always wanted to write a predictor that took some time series and gave you an estimate (mean and variance) of the next observation and I'm very pleased that this contest gave me the kick-up-the-backside to do this.  (this problem occurs everywhere - for example in quant work).  For a series of windows of exponentially increasing lengths (I used 2, 4, 8, ... 128) I fitted a linear regresssion and then used the (inverse) variances to combine all the window lengths.  For some noddy test cases it worked quite well, that is the weighting tended to pick out the actual rate of change giving a predictor that was about as good as if you knew the rate of change. I picked a half sine window function because I liked it - from experiance the window function makes little difference.  The whole lot was computationally expensive and this slowed me down on some days because I ended up with 48 cores (96vCPU) all churning away to get results which would take a day (I've since made a much faster one that just works for exponential windows). this gave me another class: 

- PREDICT: take a set of bots and using the distribution of what could have been played (not the sampled version or just what was played) then give an expected gain for each bot.   Zap all those that have negative expected gain and use the expected gains to weight all the others.  Return this distribution.

So, even though I really didn't have anything special in the very basic bots, I now have an army of interesting things I can do with them.    What tended to work well was combining to do three things:

- PREDICT(ROLL(GAIN(RLS(collection ot PRED('opponent')))))
- PREDICT(ROLL(GAIN(RLS(collection ot PRED('agent')))))
- PREDICT(FLATTEN(GAIN(RLS(collection ot PRED('agent')))))

That is, for the first one I took a whole load of things that could predict the opponent (including LSTMs at some points), used RLS to combine them into one prediction, converted the distribution on actions to that which reflected the expected gain, expanded into all the 3 rotations of that distribution and then predicted which of the three rotations was going to be best, blended them all and returned that distribution.   The second is the same but it works on prediction my action and the third works on the inverse of my action.

Finally we have to put everything together.  As I have PREDICT I used that to combine everything.  I also used the beta distribution as it was popular, but I really didn't like it (I don't think we have anything like the standard reinforcement learning problem).   The basic beta distrubution works with fractional counts with an exponentially decaying window but my gut feeling is that it overexploits.  I ended up computing a winRate = alpha * winRate + (1-alpha) * currWinRate - note the (1-alpha) which keeps everything in a -1 to 1 range that you don't nornally want (it started as a bug and I kept it).

Of course it's just as intructional to list everything that didn't work.   Notably:
- teacher training from [rps-episode](https://www.kaggle.com/tonyrobinson/rps-episode).   This grew huge so my train/valid/test was huge and I used a massive amount of CPU/GPU.   I trained on both the whole archive and I filtered out the easy wins/ossesI  I never got reliable transfer from superverised training into leaderoard results, for any task (predicting opponet move, predicting which bot is going to be best or predicting wheter I was going to win or lose) although it's still a super-interesting problem.
- LSTMs - even small decreases in perplexity result in massive changes after 1000 games.   I even trained a LSTM to maximise gain, there's some interesting weight dynamics in that case.
- stop when you are ahead/behind.   I may have got a small gain from being more random when I thought I was about to lose and I had a score of less than -10, but I'm not sure.  I never got around to analysing the archive for the optimal strategy.
- I also implement a reinforcement learning value function.  This was easy to do with LSTMs, but I was never confident enough of my LSTMs to submit it.   It would have been super cool to nudge your opponent into a way of play that gave you long term reward.

I didn't really enter to win, I entered to learn and sort out some of the ideas that had been on my TO DO list for years (decades).  I could have really increased my chances of winning by:
- listening better to @superant and others who have posted a lot of interesting theory and ideas
- teaming up with someone who had implemented all of the basic bots and who had a good leaderboard simulator
- making one configurable agent, tagging every submission, scraping all games played and coming up with an internal leaderboard.  This would have allowed a much more detailed analysis of what worked and what didn't.

So thanks to @superant, @taahakahn, @nikhijohnk and everyone who has posted here, I've really enjoyed working with you.  I have to take my ideas into finacial prediction for a bit (PM me if interested) and then I really hope to see everyone at [Hungry Geese](https://www.kaggle.com/c/hungry-geese)!
