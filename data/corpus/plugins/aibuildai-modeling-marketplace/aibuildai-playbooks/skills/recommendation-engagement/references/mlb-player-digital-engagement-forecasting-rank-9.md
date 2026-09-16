# 9th place solution

Competition: mlb-player-digital-engagement-forecasting
Rank: #9
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/256658

This was one heck of a competition. There was no shortage of new things to try at any point given the extent of the data. For that, I'd like to thank both the MLB and Kaggle.

## High-level strategy

It's no secret that lagged features were very important for this competition. I found this, too, and for a while my solutions were heavily reliant on lagged target variables. I suspect most high scoring submissions were, as well. However, there was a lot of uncertainty about what the gap between the ground truth and the inference period would be. I had considered doing some augmentations for robustness, like dropping out the n most recent days of target lags randomly, but this just killed my model's performance. So, I ended up abandoning this strategy. While I do have some target feature aggregations in my solution, as far as lags go, my final submission relies **only on lags of known features** like box score, standings, and games features.

## Models used

I tried out a ton of different models that didn't work. Here's a quick summary:

What didn't work at all:
- Denoising autoencoders

What worked decently, but not well enough:
- LightGBM, CatBoost
- 1D CNN (from the [MoA competition](https://www.kaggle.com/c/lish-moa/discussion/202256))
- Time series models on lagged targets (GRU, Transformer, DeepAR, Temporal Fusion Transformer)

My final submission ended up being an ensemble of a GRU and a Transformer that rely on non-target lags. The GRU was my best model, achieving ~1.25 on its own. The Transformer scored about ~1.27 on its own, but it proved to be very useful in an ensemble.

## Validation
I did a simple time series split. Before the training data was updated, I used July and August in 2020, and then April in 2021. After the updated data was released, I used August 2020 and June and July in 2021.

## Features used
Below is a list of all the types of features that I used. All lagged features go back 2 weeks.
- Mean/median/std/min/max of each target from the previous month
- Embeddings of playerId, teamId, player position, and player status
- Player box score lags
    - 'gamesPlayedBatting', 'hits', 'doubles', 'triples', 'runsScored',
    'homeRuns', 'hitByPitch', 'totalBases', 'rbi', 'stolenBases', 'assists', 
    'gamesPlayedPitching', 'completeGamesPitching', 'shutoutsPitching', 'earnedRuns', 'winsPitching', 'strikeOutsPitching', 'hitsPitching', 'saveOpportunities', 'saves', 'holds', 'inningsPitched'
- Lagged game features
    - 'gameTimeUTC', 'wasSigned', 'wasTraded', 'teamWins', 'teamLosses', 'teamScore', 'isHome', 'teamWon', 'scoreDiff'
    - The game time feature only includes the hour the game was played. The idea here was a lot of the digital engagement resulting from a, say, 1 PM game would manifest itself on the day the game was played, whereas the engagement from a night game would all happen on the next day.
- Lagged standings features
    - 'wins', 'losses', 'pct', 'xWinLossPct', 'divisionRank', 'lastTenWins', 'lastTenLosses'
- Lagged cumulative features (all box score features summed up by season)
- Lagged transactional features, which were just a flag indicating if the player had either been traded or released on the particular day.
- Player followers, team followers

I tried to use features that exploited the scaling of the data, such as the figuring out the minimum increment between target features for a player and rescaling their target features to the non-scaled "actual" feature, but it didn't really help my model at all. I was pretty surprised by this.

## Final submissions

I noticed that including as much data as possible helped immensely, so I knew that I wanted to make at least one of my final solutions have models that were trained on 100% of the available data. However, my training curves for both the Transformer and GRU were [quite bumpy](https://drive.google.com/file/d/1OR2vXYhGv8W3VoDWyLDicFef79UTXjs-/view?usp=sharing). So, I took checkpoints every 200 epochs starting from step 1500 while training blind on 100% of the data. I also ensembled across several seeds.

To be safe, my second submission used models that used the last 30 days of data available as validation data. This is really just a hedge, but I'm interested to see how it does.

## Other notes
- All models were trained in PyTorch using PyTorch Lightning
- Editor was VSCode and I made heavy use of the TensorBoard integration
- @nyanpn's submission emulator was invaluable near the end of the competition
- I used Optuna for hyperparameter tuning.
- To reduce submission bugs, I used a dataset that has example versions of roster data, game data, box score data, etc. This allowed me to have a "default" version of the data to return when it is null.
- As some other competitors noted, the predictions from even my best models look somewhat suspect. I was thrilled whenever I saw my model nail an upward spike in engagement. However, I think the MLB is acutely aware that some major variations in engagement happen because of exogenous factors like news stories. Given the MLB's fairly recent expansion of their social media endeavors, it makes sense that they would want to focus on purely the game-level factors that drive engagement, as this is all they really can act on. Maybe this is a faulty argument, but because of this I think MAE actually made pretty decent sense as a competition metric -- MSE would penalize you too much for factors that are completely unpredictable from game-level data.

If you made it this far, thanks for reading! I had a lot of fun during this competition and can't wait to see how things shake out.

EDIT: I've made my inference notebook public here: https://www.kaggle.com/marktenenholtz/mlb-rnn-transformer-final-public

A basic run-down of the models is that they're essentially the same, with the GRU and Transformer swapped in as different backbones. Other than that, they both take in the embedded categorical features as well as the lagged features, feed those into the backbone as a sequence, and then there's 3 convolutions that are applied to the output (for whatever reason, this worked better than a simple head). After the convolution, there's a set of linear layers that comprises the head of the model, and there's a unique head for each target. There are some minor differences from this in the Transformer model, but this is more or less how it works.
