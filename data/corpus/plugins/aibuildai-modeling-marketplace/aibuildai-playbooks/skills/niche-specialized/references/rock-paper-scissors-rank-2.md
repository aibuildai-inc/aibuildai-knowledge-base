# 2nd Place Solution

Competition: rock-paper-scissors
Rank: #2
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221506

This code ended up being my highest scoring submission, so I thought I'd do a quick write-up as to how it works.

Like many of the other top submissions, it uses an ensemble of public agents, which includes the top 50 agents from rpscontest.com and some strong agents people posted on here, notably, I want to give credit to:

* memory patterns v7 by @yegorbiryukov
* geometry bot by @superant
* multiarmed bandit v32 by @ilialar
   
I also add an evil twin agent for every one of those.

The most important thing for this competition was probably how one selects actions from the ensemble. As for that, I have a few thoughts about what properties a good selection strategy should have:

* It should be fast at adapting to the opponent
* Because the matches are so random, it should base its decision not only on a handful of observations
* It should be quite random so that it does not expose detectable patterns in the underlying agents
    
Based on those thoughts I came up with the strategy of first selecting a group of agents which performed well for some amount of time and then selecting a final agent out of this group based on its performance during the last few moves. Finally, the scores of the agents are not deterministic but drawn from a distribution, this way I introduce some randomness. I also tried out quite a few other more complicated strategies but this one seems to work best.

Finally, I want to thank Kaggle for hosting this competition and everyone else for participating and having interesting discussions in the forum.

```
import numpy as np
from copy import deepcopy
from kaggle_environments import agent
from glob import glob
from functools import partial

# path = '/kaggle_simulations/agent/'
path = 'top/'

def rps_agent(filename):
    code = compile(open(filename).read(), filename, 'exec')

    gg = {}
    def run(observation, actual):
        if observation.step > 0:
            inp = 'RPS'[observation.lastOpponentAction]
            outp = 'RPS'[int(actual)]
        else:
            inp = ''
            outp = ''

        gg['input'] = inp
        gg['output'] = outp

        exec(code, gg)

        return {'R': 0, 'P': 1, 'S': 2}[gg['output']]

    return run

def kaggle_agent(filename): return agent.get_last_callable(agent.read_file(filename))

def score(history, predictions):
    actual = history[:, -1]

    n = actual.shape[0]
    p = lambda x: np.sum(predictions == (actual[:, np.newaxis] + x) % 3, axis=0) / n

    p_win = p(1)
    p_lose = p(-1)

    return np.random.uniform(0, np.maximum(0, p_win - p_lose))

def dirichlet(history, predictions):
    actual = history[:, -1]

    n, m = predictions.shape[:2]

    n_outcome = np.array([
        np.sum(predictions == (actual[:, np.newaxis] + i) % 3, axis=0)
        for i in range(3)
    ]).T

    return np.array([np.random.dirichlet(n_outcome[i] + 1) for i in range(m)])[:, 1]

def select_best(history, predictions, w, scoring_func, k):
    w = min(history.shape[0], w)

    q = scoring_func(history[-w:], predictions[-w:])
    best = np.argpartition(q, -k)[-k:]

    return best

def reverse_agent(agent):
    def f(observation, actual):
        if observation.step > 0: actual_actual = observation.lastOpponentAction
        else: actual_actual = np.random.randint(0, 3)

        observation = deepcopy(observation)
        observation.lastOpponentAction = actual

        return agent(observation, actual_actual)

    return f

rps_agents = [
    partial(rps_agent, filename) for filename in glob(path + 'rps/*.py')
]

kaggle_agents = [
    partial(kaggle_agent, filename) for filename in glob(path + 'kaggle/*.py')
]

agents = rps_agents + kaggle_agents

instances = [
    *[agent() for agent in agents],
    *[reverse_agent(agent()) for agent in agents],
]

N = 1000

predictions = np.zeros(shape=(N, len(instances))).astype(np.int8)
predictions[0] = np.random.randint(0, 3, size=len(instances))

history = np.zeros(shape=(N, 2)).astype(np.int8)

n = 0

def run(observation, configuration):
    global n

    if n != 0:
        history[n - 1, 1] = observation.lastOpponentAction

    for i, agent in enumerate(instances):
        predictions[n, i] = agent(observation, history[n - 1, 0])

    if n > 1:
        candidates = np.hstack([(predictions[:n + 1] + i) % 3 for i in range(3)])

        inner_circle = select_best(
            history[:n],
            candidates[:n],
            150,
            score,
            7
        )
        candidates = candidates[:, inner_circle]

        best = select_best(
            history[:n],
            candidates[:n],
            np.random.randint(3, 7),
            dirichlet,
            1
        )[0]

        action = int(candidates[-1, best])
    else:
        action = np.random.randint(0, 3)

    history[n, 0] = action
    n += 1

    return action
```
