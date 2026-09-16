# My approach to the Santa 2020 challenge

Competition: santa-2020
Rank: #4
Source: https://www.kaggle.com/c/santa-2020/discussion/216536

This has been a very interesting challenge.  Originally I was disappointed that it wasn't an optimization challenge, but I think this one turned out to be more interesting than any optimization challenge could've been.  The more I worked on this challenge, the more I appreciated how deep this game turned out to be.  The multiplayer aspect, as well as the decay aspect, seemed to reward first-principles thinking, which always makes me obsessively motivated.
# General Thoughts
- My gut feeling is that this game has just one equilibrium optimum solution.  I think that if you find that optimum solution, then your opponent wouldn't be able to counter it, even if they knew exactly what it was.  I can't prove it, it's just a gut feeling.  I don't think we have a rock-paper-scissors dynamic here, I think this game is more like poker.
- I don't think that exploration/exploitation dilemma is all that relevant here.  I think both the presence of the opponent and the decay mechanic make this game mostly about immediate exploitation of any good bandit you come across.
- Our objective in this game is to make perfect use of what we learned from our own bandit pulls, infer what our opponent knows from his own bandit pulls, and make it difficult for our opponent to deduce what we know about bandits from our pulls.
- The key challenge is finding the balance between making best use of our information, and being deceptive about our own knowledge.  Deception by definition requires us to make suboptimal decisions in order to avoid being predictable, so the challenge is to find the right level of deception such that you give up less than what your opponent loses from being unable to read you effectively.
- Performance analysis is a crucial part here as well.  Given the level of noise present in outcomes, you need to be able to measure whether you're going in the right direction or the wrong direction with your agent development.  I think there is a lot of advantage to be gained from knowing how to do that effectively.
# Agent Components
- Exact calculation of discrete posterior distribution based on results of our bandit pulls.
- LSTM model to estimate opponent's posterior distribution based on our observation of their actions.
- Deception tactics to avoid tipping our hand as to what we know about the bandits.
# Exact calculation of posterior distribution
In my opinion, a lot of competitors here made their life unnecessarily hard in this part.  Using Bayes Theorem, we can calculate the posterior distribution of initial thresholds based on results of our pulls, without any approximation or modeling involved.  The initial thresholds follow a discrete uniform distribution between 0 and 100, inclusively, so we know that a priori, each threshold is 1/101 likely.  All we need to update our 101-long vector of posterior probabilities is the prior 101-long vector of probabilities, knowing how many times the bandit was pulled previously, and the result of our pull.  

For the bandit pulled, ignoring the normalization of probabilities, 
$$P_1(thr) = \frac{\lceil thr * 0.97^{pulls} \rceil}{101}$$
$$P_{posterior}(thr) \propto P_{prior}(thr) * [P_1(thr) * reward + (1 - P_1(thr)) * (1 - reward)]$$

# LSTM model
The next piece of the puzzle is figuring out what the opponent knows about the bandits, based on which bandits they decide to pull.  For this purpose, I decided to build an LSTM model with 1999 time steps.  Each bandit would have its own 1999-long sequence.  The target variable for this model is the opponent's posterior distribution at each time step, and the loss function is the KL Divergence.

For this LSTM model, there are fundamentally only two pieces of knowledge required:  did the opponent pull the bandit at step x, and did you yourself pull the bandit at step x?  Apart from these two basic binary predictors, I didn't use any other features, I let the model figure out the rest.  Apparently it did a good job figuring out the rest, because based on my analysis of matches, many of the top opponents like nagiss were essentially face up to me, my final estimates of the threshold probabilities were close to the theoretically optimal estimates that could be achieved.

Incorporating the model output into my final estimate of threshold probabilities was straightforward:
$$P_{final} \propto P_{own} * P_{model}$$

# Deception
It seems like all of the top players converged to the strategy of trying out new bandits, and then continuing to hit them if the previous outcomes indicated a high enough posterior probability of reward.  The one problem with that approach is that it leaks information to your opponent.  Here are a couple of obvious patterns that will telegraph exactly what you know:
- If you hit a fresh bandit once and move on, you probably got a 0 reward.
- If you hit a fresh bandit twice and move on, you probably got 1 the first time, and 0 the second time, at which point hitting another fresh bandit is a better bet.
- If you hit a fresh bandit four times and move on, you probably got 1-1-0-0.

In all these cases, you pretty much telegraphed exactly the outcomes that you got.  That information is very valuable to your opponent, because in Bayesian statistics, the first few updates are the most informative.

My first strategy to conceal these patterns was to hit a fresh bandit four times before moving on, unless my opponent joined before I finished the first four hits.  If I hit every fresh bandit four times, my opponent won't know with certainty as to how many rewards I got out of it.  I probably didn't get more than two, or I would keep hitting it, but he won't know whether it was zero, one, or two.

My second strategy was to use my LSTM model against myself.  If I have a model to figure out what my opponent was up to, then I assume that my opponent has some kind of model to figure out what I'm up to.  Assuming that his model works more or less the same way, I can figure out what information I'm broadcasting, and I can also project ahead and figure out what further information I would be broadcasting with each of my 200 potential move-outcome combinations.  

If my next move would make my image converge towards my actual posterior probabilities, then I would be less likely to make that move.  If my next move would reveal very little or even increase the divergence, then I would be more likely to make that move.  Ultimately it was a matter of empirically finding the right penalty for the reduction in KL Divergence of my actual posterior distribution and my "image" posterior distribution.
# Performance Analysis
Many posters noted how difficult it was to measure performance of different agents, partly due to high variance of game outcomes.  My approach to extracting the most information from game outcomes was to calculate the expected candy outcomes.  If you pull a bandit with 0.30 probability, for example, you will get either 0 or 1 candy, but on expectation basis you will get 0.30 candies.  One further step is to translate the EV difference to the chance of winning.  Empirically, the standard deviation of the difference between actual score differential and the EV score differential was 26.5.  Therefore, your chance of winning the game is $$P(win) = \Phi\Big(\frac{EV_{own} - EV_{opponent}}{26.5}\Big)$$.
For example, if your expected count of candies is 10.5 higher than your opponent's expected count, then you should expect to win the game 65.4% of the time.  When you tabulate game outcomes from local testing or from analysis of your own games, you should tabulate this expected number of games won rather than actual numbers of games won.

According to this kind of analysis, I'm actually in a good position for this competition, as long as some of the 10 copies of my best agent make it near the top of the leaderboard.  Frustratingly, my current leaderboard position is held up by agents who appear to be clearly inferior to my best agent.  My best agent can beat the opponents in the 1250+ range with 63% probability, which I imagine is very high, but they need to overcome incredibly bad early luck first.  If they do, I really like my chances.
