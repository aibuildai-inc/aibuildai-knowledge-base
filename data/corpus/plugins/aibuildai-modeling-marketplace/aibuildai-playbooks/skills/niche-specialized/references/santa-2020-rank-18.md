# [18th place] Our approach - Pure greedy Lightgbm

Competition: santa-2020
Rank: #18
Source: https://www.kaggle.com/c/santa-2020/discussion/217105

I and Felipe, which i'm grateful for teaming, developed an agent which consisted of a Lightgbm model which estimate the threshold for each bandit.

# Overview

We started by scraping over 25k different matches of the top agents on the leaderboard and we created multiple dataset with 15 features (each dataset had more than 40 MIL rows). 

We created a dataset by using the information from the perspective of:

- The top agent.
- The winner of the match if it has more than than 1100 scores on the leaderboard.

To simplify if we create a dataset from the perspective of the top-agent it means that we create a dataset as we were the top agent with its information about reward/loss.


# Dataset

Each match generates a dataset of num_round * n_bandit = 1999 * 100 = 199900 rows.
Initially we create a dataset as the one used in the kernel epsilon-greedy, but later on we observed that updating the prediction for each bandit at each round brings better performance locally.

# Model
The model is a simple Lightgbm trained on more than 250 MIL rows and 15 features (thanks to Felipe which was able to train the model on its machine). We used the score of the agent as weight in the training process.

# What didn't work

We tried lot of things which didn't work:

- Implement different hard coded strategy of exploitation, e.g. repeat a bandit if it had a reward last time.
- Softmax exploration
- UCB exploration
- Ensembling
- Blending
- LSTM
- Xgboost

We saw that the best agent was the greedy one single model.
