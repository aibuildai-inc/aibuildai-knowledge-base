# 28th Place Solution (Silver) - Multi Stage Decision Tree + 66% Defensive Randomness

Competition: rock-paper-scissors
Rank: #28
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221508

I approached Rock Paper Scissors with the goal of simply exploring and comparing a wide variety of different techniques.

I made all my work completely public throughout the entirety of this competition and published a total of 23 notebooks and made 277 agent submissions. 

My best scoring agent (28/1667 = top 2%) was my Multi-Stage Decision Tree (v36) was published 2 months ago. 
- https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-multi-stage-decision-tree?scriptVersionId=50045978

I believe this is the highest-scoring public notebook of the competition, discounting solutions published after the final submission deadline.

This notebook got submitted a total of 75 times, and the winning hyperparameters where:
- score = 971.0 | window=9, stages=2, random_freq=0.66, warmup_period=10 

Runner up hyperparameters:
- score = 914.1 | window=10, stages=3, random_freq=0, warmup_period=0

---

The idea is quite simple, treat the game history (your actions + opponent actions) as time series data and train a Decision Tree to predict the next opponent action based on the previous window of 9 moves.

Version 1 of this idea only scored 609 and is quite easy to predict, however it worked much better when combined with two key concepts.

The first concept is from the Iocaine Powder scene from the Princess Bride, attempting to guess, second guess and triple guess what our opponent predicts what out opponent thinks we will do. 
- https://www.youtube.com/watch?v=9s0UURBihH8&ab_channel=nerfboogers



This is concept is implemented via a multi-stage decision tree. The first stage makes a prediction using a normal Decision Tree. This is saved into the game history, then fed as additional input into a second decision tree. The process can be optionally repeated using a third-stage decision tree. The question becomes how did our opponent actually act relative to how we predicted they would act.

The second concept was statistical defensiveness. The winning hyperparameters played random moves 66% of the time.  The result of this is to make the agents appear to the opponent as statistically indistinguishable from a random bot and thus be almost impossible to predict.

Playing random moves is a Nash Equilibrium statistical draw, thus a 1000 step game effectively gets compressed into 333 steps of actual predictive gameplay. We only require +-20 score to win. For every 1 round of predictive gameplay, we get 3 turns of observation for how our opponent reacts to our random moves. Our opponent on the other hand is given 2 pieces of statistical noise for each 1 round of true gameplay signal. This gives a 9-fold signal-to-noise ratio advantage in information flow.

This strategy is a slow burner. It tends to draw most of its games, but it wins almost twice as often as it loses (76 wins / 131 Draws / 44 Losses / 252 Total). It takes as baseline the Nash Equilibrium statistical draw, then attempts to gain a small statistical advantage, which over the long run results in a slow upward trend. Ironically this agent seems to do better against higher-rated opponents than it does against lower-rated opponents. 

For this effort, I have been rewarded with a Silver medal, my first solo medal, and have finally achieved the rank of 3x Kaggle Competitions Expert. I have also proved it is possible to win a Kaggle medal without secrecy using only public notebooks.

---

# Further Reading

This work is part of a series exploring Rock Paper Scissors:

Irrational
- [PI Bot](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-pi-bot)
- [Anti-PI Bot](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-anti-pi-bot)
- [Anti-Anti-PI Bot](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-anti-anti-pi-bot)
- [Irrational Agent](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-irrational-agent)
- [Irrational Search Agent](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-irrational-search-agent)
- [Random Seed Search Nash Equlibrium Opening Book](https://www.kaggle.com/jamesmcguigan/random-seed-search-nash-equlibrium-opening-book)

RNG
- [Random Agent](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-random-agent)
- [RNG Statistics](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-rng-statistics)

Sequence
- [De Bruijn Sequence](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-de-bruijn-sequence)

Opponent Response
- [Anti-Rotn](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-anti-rotn)
- [Sequential Strategies](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-sequential-strategies)

Statistical 
- [Weighted Random Agent](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-weighted-random-agent)
- [Anti-Rotn Weighted Random](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-anti-rotn-weighted-random)
- [Statistical Prediction](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-statistical-prediction)

Memory Patterns
- [Naive Bayes](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-naive-bayes)
- [Memory Patterns](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-memory-patterns)

Decision Tree
- [XGBoost](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-xgboost)
- [Multi Stage Decision Tree](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-multi-stage-decision-tree)
- [Decision Tree Ensemble](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-decision-tree-ensemble)

Neural Networks
- [LSTM](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-lstm)

Ensemble
- [Multi Armed Stats Bandit](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-multi-armed-stats-bandit)

RoShamBo Competition Winners
- [Iocaine Powder](https://www.kaggle.com/jamesmcguigan/rps-roshambo-comp-iocaine-powder)
- [Greenberg](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-greenberg)
