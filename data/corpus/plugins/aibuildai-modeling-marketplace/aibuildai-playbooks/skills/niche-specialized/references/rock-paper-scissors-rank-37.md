# Going Meta [37th place solution]

Competition: rock-paper-scissors
Rank: #37
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/217110

This has been an extremely interesting and fun competition for me and I'm very glad I took part in it. First of all, I want to thank the whole community for the friendly and helpful atmosphere. The discussions and notebooks in this competition were truly on the next level. Of course, also special thanks to Kaggle for hosting this competition and providing excellent support.

## Introduction
When you first hear of Rock, Paper, Scissors, you may wonder why this is an interesting game for a computer competition. There is simply no way to beat the Nash equilibrium strategy (playing completely randomly) statistically. However, this doesn't mean that there is no meaning in this challenge as the majority of opponents your bot may face isn't playing pure random. Therefore the goal is to try to win against these deterministic agents while trying not to lose against them at the same time (Playing deterministically is a double-edged sword.). I'll try to give a short high-level overview of the approach I chose for this task, which is an ensemble strategy, involving multiple meta-layers.

## Policies
In my code, a policy/strategy/action selector is simply a mapping of the current game history to a probability distribution over the three possible actions. A policy selecting rock all the time would yield [1, 0, 0] for example. There are multiple ways to create new policies from existing ones of which I use the following:

### Strict policies
This transformation takes the probabilities from a policy and selects the action with the highest probability, e.g. [0.6, 0.2, 0.2] becomes [1, 0, 0]. Therefore we now have a deterministic version of the original policy.

### Incremented policies
"When my opponent thinks that I think he thinks I think..."
Dan Egnor already implemented this idea in his Iocaine Powder bot. One takes a policy and shifts the probabilities by one to produce a version that would counter the current one. [0.2, 0.5, 0.3] becomes [0.3, 0.2, 0.5]. One can derive a double incremented policy by shifting the incremented version.

### Counter policies
This is an attempt to construct a policy countering a given policy. The counter policy gets constructed by assuming the opponent uses the original policy (One has to flip one's own actions and the opponent's actions in the game history.) and shifting the probability distribution produced by the policy.

### Combinations
As you can see, it is relatively easy to derive a whole set of policies from a single stochastic policy. By combining the above transformations, one can get up to twelve policies out of a single one.

### List of base policies used in my final agent
* Random
* Frequency
* Copy last action
* Transition matrix
* Transition tensor
* Random forest model
* two adapted stochastic versions of RFind
* some fixed sequences (pi, e, De-Bruijn)
* [Seed searcher](https://www.kaggle.com/taahakhan/rps-cracking-random-number-generators) by @taahakhan
* [RPS Geometry](https://www.kaggle.com/superant/rps-geometry-silver-rank-by-minimal-logic) by @superant
* [Anti-Geometry bot](https://www.kaggle.com/robga/beating-geometry-bot/output) by @robga
* [Greenberg](https://github.com/erdman/roshambo) by Andrzej Nagorko, translated into Python by Travis Erdman
* [Iocaine Powder](http://davidbau.com/downloads/rps/rps-iocaine.py) by Dan Egnor, translated into Python by David Bau

and the following RPS Contest bots, which I could integrate into my ensemble using [this method](https://www.kaggle.com/purplepuppy/running-rpscontest-bots) by Daniel Lu ( @purplepuppy ):
* [testing please ignore](http://www.rpscontest.com/entry/342001) by Daniel Lu
* [centrifugal bumblepuppy 4](http://www.rpscontest.com/entry/161004) by Daniel Lu
* [IO2_fightinguuu](http://www.rpscontest.com/entry/885001) by sdfsdf
* [dllu1](http://www.rpscontest.com/entry/498002) by Daniel Lu
* [RPS_Meta_Fix](http://www.rpscontest.com/entry/5649874456412160) by TeleZ
* [Are you a lucker?](http://www.rpscontest.com/entry/892001) by sdfsdf
* [bayes14](http://www.rpscontest.com/entry/202003) by pyfex

## Policy selector
I tried many ways of selecting the best policy for a move including weighted agent voting, Thompson sampling, a beta distribution, a Dirichlet distribution, LSTMs, CNNs, max wins, but one outperformed them all: argmax (which I didn't even consider until one week before the competition end date because it was so simple) applied to the decayed performance scores of the policies. Every step the performance of a policy is calculated like this:
`score = probabilities[winning_action] - probabilities[losing_action] `
My final version of the policy selector simply takes the three policies with the highest decayed performance scores and combines their probabilities using the weights [0.7, 0.2, 0.1], I really don't know why this performs so well, but well, it does.

## Policy selector-selector
I added a meta-layer by creating multiple policy selectors with different parameters to calculate the performance score. The parameters are decay (used to differentiate between long- and short-term success), drop probability (the probability to reset the score of a policy when it loses; didn't perform too well in combination with a decay < 0.99) and zero clip (whether the score should be clipped at zero). Then the agent chooses best performing parameter configuration (measured by a decayed configuration performance score with additional hyperparameters) using argmax on the configuration weights. These weights are computed by sampling from a Dirichlet distribution using the configuration performance scores. At first, I tried combining the probabilities of the configurations weighting them by their respective performance, but this led to way too much noise. However, only configurations that yield a decisive result at this step get considered.
If the result is too indecisive (The sum of the weights of all configurations yielding decisive probabilities is too small), my agent selects a uniformly random action instead of choosing from the distribution it got from the best parameter configuration.

## Playing randomly
This is the last meta-layer in my agent and maybe the one with the biggest impact. Choosing a random action is a great defensive tool as there is simply no way to beat it. An additional decayed performance score gets computed for the agent. At this level, I no longer use a probability distribution for score calculation, but the action the agent selected.
The agent chooses a random action with a certain probability in the following cases:
* The probability of winning by playing random is >= 92%
* The agent performance score is below a certain threshold (I tried playing the counter of the counter of the action selected by the agent with a certain probability, but this didn't perform too well.)

## Observations
So that's it. Essentially just a strange cascade of scores and meta-layers operating on top of an ensemble of other agents. Interestingly, this architecture performs way better than previous ones with more meta-layers, more policies and more sophisticated methods for combining weighted probability distributions. This could be due to a bot essentially degenerating to random at a certain degree of complexity.
The stochastic base policies, which are obviously strongly correlated to their strict versions, are almost never used by the ensemble. However, removing them resulted in a major decline in leaderboard performance.
Removing almost all policies (except Greenberg and the Geometry bot by @superant), but keeping their counter policies results in a very strong agent, even outperforming the original version on the leaderboard. My interpretation of this is that this reduces noise that would distract the agent from cracking the opponent's strategy and minimal logic suffices (already demonstrated by @superant) apart from counter policies.
In my local tests, using online-learned LSTMs as policy scoring functions vastly outperformed (up to 80 points) the approach described here, particularly because the agents with LSTMs are extremely fast at adapting to policy changes. On the leaderboard, however, this strategy performs worse. I'm convinced there is much room for further improvement using neural networks with a good architecture and maybe a combination of offline and online learning.

## Regrets
Submitting  almost all agents only once and only using a fraction of my submissions doesn't seem to be a good idea in a competition where a few matches against random agents can completely destroy the score of an agent.

## Code
My messy code can be found on [GitHub](https://github.com/0Zeta/RockPaperScissors). The main logic is in helpers.py, policies.py and statistical_policy_ensemble_agent.py

## Conclusion
RPS has been an amazing competition and I'm sure I learned a lot from it. What I will remember most, however, is the incredible collaboration and helpfulness in the discussions section of this competition. Thanks to everyone who made this challenge the great experience it was.
Bests of luck (and strong nerves) to everyone for the remaining weeks!
