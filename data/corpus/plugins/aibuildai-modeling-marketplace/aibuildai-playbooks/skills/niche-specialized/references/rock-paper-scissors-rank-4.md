# 4th Place Solution - Taaha Khan

Competition: rock-paper-scissors
Rank: #4
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/217046

## 4th Place Solution: "HydraRPS: Dynamically Weighted Multi-Armed-Bandit Action Selection for Algorithmic Rock-Paper-Scissors Gameplay"

Hello everyone!

This competition has been an amazing experience for me. I learned a lot throughout these past few months, and I hope that others enjoyed it as much as I did. This is my third simulation competition on Kaggle, as I competed in [ConnectX](https://www.kaggle.com/c/connectx/overview) and [GRF](https://www.kaggle.com/c/google-football/overview). I really like this simulation format and am looking forward to the next simulation competition! 

Here, I put together a general write-up of my approach to the game of rock-paper-scissors. It is ranked 4th with a score of 1021.0. I probably should post this later, but I just couldn’t wait to discuss strategies with others.

I used a large ensemble of strong agents and generated a distribution of values to pick the next action from. I called this approach the “Hydra” after the [Greek and Roman myth](https://en.wikipedia.org/wiki/Lernaean_Hydra) of a beast where if one head was cut off, two more would grow back in its place. This symbolizes a bandit, where if one of the agents gets exploited, two more will jump in and continue.

The core base that I used was built off a heavily adapted model of the public [MAB Notebook](https://www.kaggle.com/ilialar/multi-armed-bandit-vs-deterministic-agents) by @ilialar.

### Ensemble Population:

Here are the agents that I included in my ensemble. They consist of strong public bots, as well as old archived bots from previous competitions. I had to edit all of them to fit my format, so they would be implemented as a class with a `step(obs, config)` method. Links and credit are listed here:

- [Decision Tree](https://www.kaggle.com/alexandersamarin/decision-tree-classifier) by @alexandersamarin
- [Decision Tree 2](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-multi-stage-decision-tree) by @jamesmcgugian
- [Memory Patterns V7](https://www.kaggle.com/yegorbiryukov/rock-paper-scissors-with-memory-patterns?scriptVersionId=46447097) by @yegorbiryukov
- [Simple Rfind](https://www.kaggle.com/riccardosanson/rps-simple-rfind-agent) by @riccardosanson
- [RPS Geometry](https://www.kaggle.com/superant/rps-geometry-silver-rank-by-minimal-logic) by @superant
- [Greenberg](https://www.kaggle.com/group16/rps-roshambo-competition-greenberg) by Andrzej Nagorko
- [Iocaine Powder](https://www.kaggle.com/group16/rps-roshambo-comp-iocaine-powder) by Dan Egnor
- [IO2_fightinguuu](https://web.archive.org/web/20200812062252/http://www.rpscontest.com/entry/885001) by sdfsdf
- [Testing Please Ignore](https://web.archive.org/web/20201021153705/http://rpscontest.com/entry/342001) by @purplepuppy (Daniel Lawrence)
- [Dllu1](https://web.archive.org/web/20200812060710/http://www.rpscontest.com/entry/498002) by Daniel Lawrence
- [Centrifugal Bumblepuppy 1001](https://web.archive.org/web/20201021155550/http://rpscontest.com/entry/315005) by Daniel Lawrence
- [Centrifugal Bumblepuppy 13](http://www.rpscontest.com/entry/203004) by Daniel Lawrence
- [RPS_Meta_Fix](https://web.archive.org/web/20200220023240/http://www.rpscontest.com/entry/5649874456412160) by TeleZ
- [Are you a lucker?](https://web.archive.org/web/20191201105926/http://www.rpscontest.com/entry/892001) by sdfsdf

The inverse version of each of these agents was also included in the ensemble. This means that I would flip the `obs.lastOpponentAction` to my last action instead, and give them the inverted observation. Their output would be the action that this agent would have played in the opponent’s shoes, so I would take the action that beats that. Also, a random agent was included in the ensemble as a fallback if all others are failing.

### Forcing Actions

The action that each agent takes might not be the real action that the overall hydra agent made. Due to this, each agent needs a method that will force the last action into their personal memory so the predictions will stay accurate. I implemented this as a custom method in each class that takes the last action and places it in the proper spot of history for that specific agent. For example:

```
def set_last_action(self, action):
	self.history[-1] = action
```

### Randomness

It is a bit of a paranoid addition, but one that I implemented nevertheless. During the beginning of the competition, I published a [notebook](https://www.kaggle.com/taahakhan/rps-cracking-random-number-generators) detailing some ways of trying to outsmart pure random agents. This ended up failing, but I still used some methods to try and fight against attacks like this using stronger randomness. The `random` and `np.random` modules are predictable as they use the solved Mersenne Twister algorithm to generate their values. Instead, I used the `secrets.SystemRandom()` module to generate my randomness, which is cryptographically secure. 

```
import secrets

# Same usability as normal random
random = secrets.SystemRandom()
```

### Agent Weighting:

One of the most important aspects of ensemble methods is picking which agent should be used. Nailing this down will allow the agent to be extremely powerful, as there is bound to be at least one agent in the group that is able to predict the opponent. Here are the meta-strategy selectors in my agent:

| **Meta Strategy** | **Description** |
| --- |
| Beta Dist | A beta distribution for the decaying wins and losses of an agent |
| Score-Beta | Similar to beta distribution, but without decay in alpha/beta values |
| Win Percent | Simply getting the win percentage of the agent’s actions |
| Loss Percent | Getting the negative loss percentage of an agent’s actions |
| Win-Loss | Simple gain scoring, +1 for a win, -1 for a loss |
| Dirichlet | Suggested by @georgstreich, a Dirichlet distribution of scores |
| Non-beta | A decaying score value based on historical wins and losses |
| Epsilon-greedy | Classic epsilon-greedy algorithm for the MAB problem |
| UCB Decay | An upper confidence bound selector with 0.98 history decay |
| Thompson Sampling | Using Gaussian Thompson Sampling scoring | 

Using only one of these individually works alright, but to really get the overall best scoring agent, I combined the weighted inputs of all of these meta strategies.

Each meta-strategy would give each agent a score ranging from 0 to 1. If this was the best agent that the strategy thinks, then it would get a 1, if it was the worst, then it would get a 0. All other agents would be scored in that range based on their meta-score. The meta selector would then return a list of scores for each agent, with scores ranging from 0 to 10 (max of 1 point from each meta-selector).

### Action Selection

For every agent, their selected action would receive a vote from the agent scores, meaning that if an agent with a score of 7.34 picked R, then `dist = [1, 1, 8.34]` (the losing action of the returned action is used to develop a distribution of the probabilities that the opponent plays a certain move). This logic is taken directly from the [From Prediction to Action](https://www.kaggle.com/c/rock-paper-scissors/discussion/208242) thread by @tonyrobinson. 

This array would be normalized, then the expected gains would be taken:

`gains[action] = dist[CEDE(action)] - dist[BEAT(action)]`
e.g. `gains[R] = dist[S] - dist[P]`. 

This should give an array `gains = [0.048, 0.171, -0.219]`. The argmax value of the gains is used to return a result, other than the simple argmax of the distribution, which was observed to not be optimal. Other methods like a Sharpe ratio could also be used in this situation.

Perhaps another version like this from the parallel [MAB Competition](https://www.kaggle.com/c/santa-2020) could be ported and used in this RPS ensemble setting to even more improve our algorithms.

### Local Evaluation

I used a custom leaderboard ranking system to locally evaluate my agent. This used a pool of strong public and archived agents from RPSContest and ran them all against each other N times. The agents were then ranked by either win percent or `n_wins - n_losses` (whichever I need). This should give me a good idea of where my newest approach stands in the pool, as well as how strong new public notebooks are when I add them into the pool.



Near the end of the competition, this became less useful as my agent was consistently scoring above 80% in win rate no matter what changes I made, so I simply started submitting the agents and let the leaderboard score them.

### What didn’t work

Some things that I tried, but either didn’t perform as well or did not work at all:
- **Defensive noise:** Every now and then playing a random move. This just increased noise in games
- **Agent selection:** Picking the best agent to use for a move. This gave one agent too much power and allowed it to be exploited, so instead, I used weighted agent voting
- **Strategy selection:** Scoring each strategy by overall score, then selecting which strategy is best, and using that strategy's best agent’s move. This overcomplicated things a lot and just wasn’t as good performance-wise
- **Drop-switch:** Using a drop-switch meta-strategy. This doesn’t show the true best agent, and only shows the most recent streak of an agent, which doesn’t show performance that well.
- **Capping losses:** Playing randomly if the agent has a score of < -15, not ideal as it decreases the chance of a late-game comeback.
- **Agent Randomness:** Allowing bandit agents to return a random action sometimes. This drastically increased noise, so I made a variable `AGENT_RANDOMNESS` to toggle if agents can return a random action if they didn’t know what to do.
- **Shift Agents:** Including all shifted variations of agents. This just made too many agents to choose from, and the distribution just became noisy with a ton of irrelevant data.

### Code

This project consisted of some of the messiest code that I have ever written, but I guess it got the job done. My final agent consisted of over 2000 lines with a file size of about 70 KB. I programmed this project entirely locally instead of using notebooks.

- Full Repository: https://github.com/taaha-khan/rock-paper-scissors
- Agent Code: https://github.com/taaha-khan/rock-paper-scissors/blob/main/RockPaperScissors/hydra.py

(The code in the Repo isn’t an exact copy of the top approach, but is a slightly adapted version which I evaluated to be the best. This is also currently gold in the leaderboard.)

### Final Thoughts

This competition has been a ton of fun for me to work on, and the forums have been very helpful and supportive throughout. There were so many great discussions in this competition, making it more of a community feeling other than simply completing a problem. I’m looking forward to hearing other top approaches soon!

Special thanks to @tonyrobinson @nikhiljohnk @superant and all others who contributed for driving these discussions forward. Also thanks to the Kaggle team for organizing these great competitions! Hope to see you all in [Hungry Geese](https://www.kaggle.com/c/hungry-geese)!

Questions, comments, suggestions are welcome!

Taaha Khan
