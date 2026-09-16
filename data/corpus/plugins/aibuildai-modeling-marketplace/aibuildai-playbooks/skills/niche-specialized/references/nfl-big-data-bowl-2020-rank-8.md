# My solution was public from the beginning

Competition: nfl-big-data-bowl-2020
Rank: #8
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119318

Sorry for the click bait title but it is 90% true. It is basically combination of these 2 kernels:
https://www.kaggle.com/divrikwicky/lightweight-version-of-2-65-custom-nn
https://www.kaggle.com/divrikwicky/nfl-lofo-importance

First one is from the chemistry competition. I have used almost the same model in this competition. Conv1D with Global Pooling on player vs rusher interaction features instead of atom interaction features. Only difference is that this time I had 2 of them: rusher vs teammates sub-model and rusher vs opponents sub-model.

Second one is Leave One Feature Out method with my open source implementation. I have used LOFO to understand overall importance of the raw and generated features. I have also run LOFO separately for different seasons. It made it very quick for me to decide on which features have stable importance over time. I first run LOFO on target MAE with lightgbm and then remove/replace the features from my NN. This way I both benefit from speed of lightgbm and having double validation by two models even though I use only NN model for my submission at the end.

Since I am the author of the repo ( https://github.com/aerdem4/lofo-importance ), I am looking for feedback from people who have tried it in this competition. Please let me know any of your positive or negative stories with lofo-importance. Any suggestion is welcome.
