# Simple, steady, silver (20th)

Competition: rock-paper-scissors
Rank: #20
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221507

As others, I used an ensemble of known bots.

My personal contribution was what I nickname iterated iocaine, or iterioc for short. 

The original iocaine algorithm takes a strategy, generates 3 rotations, then plays it in the opponents' shoes, and generates 3 rotations of his moves. 

My idea was that, what if we iocained an iocaine? Let I(S) be the iocaine function that does what I described above onto a predictor S. We can create I^2(S) by feeding I(S) into an iocaine meta-predictor. Even I^3(S), and so on. After some optimization, I was able to linearize I^n(S) and get it in O(n)! That wasn't needed since the most I'd used is I^5(S), beyond that things got noisy and the values would stop making sense after too many metas. (A submission of I^500(rfind) failed miserably).

Iterioc proved to be a beast against metas, and a powerhouse when we create an ensemble of iterioc.

I have 11 silver agents using variations of  "ensemble of iteriocs". 

My best solution is an ensemble of I^3 predictors created from rfind, testing, lucker, 3 other rps contest algorithms, decision tree, decision forest, kumoko, and memory_patterns v7.0. Scoring was by a weighted average of the bots, where their weights are a reflection of their history. Every other move is a random move thrown to annoy the opponent, but I don't sample from pdf's in my other moves. There is no time scaling.

I probably should have delved into more sophisticated ensemble methods; apart from my signature iterioc contribution my solutions are fairly vanilla. 

I leave you with three statements without proof (I proved them but it's quite tedious):

1) There is a finite number of deterministic agents.

2) Let A be a deterministic agent,  X be the number of agents it wins against, Y be the number of agents it loses against, and Z be the agents that it draws against. All deterministic agents share a universal X,Y, and Z, which are actual numbers. As a corollary, X=Y.

2 implies that there are some agents that lose to rock but beat rfind (because rfind beats rock, and they must beat the same number of agents). All determinstic agents are fundamentally equal in theoretical strength, the ones that achieve higher rating are those that beat a larger subset of existing bots in a population.

**The big one: 3) The bot with the best win rate against any fixed known population of bots (including nondeterministic bots) is purely deterministic.**

It was fun getting involved and meeting you guys and collaborating with you in constructions and discussions :)

Starting to put my stuff here:
https://github.com/KarHam/iterioc/blob/main/iterioc.py
