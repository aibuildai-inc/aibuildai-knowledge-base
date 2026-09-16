# 8th place solution: a 10-day challenge

Competition: um-game-playing-strength-of-mcts-variants
Rank: #8
Source: https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549616

I deeply regret not starting this competition earlier, especially after seeing @jsday96 report about the slight improvement with GAVEL – that was quite exciting! Despite the short timeframe, I had a lot of fun and learned a great deal. Thanks to the hosts and all the participants!!

Here's a quick recap of my 10 days:

* I decided to join this competition as a way to demonstrate the usefulness of a VS Code extension for Kaggle that I've been developing. Please check it out and share your feedback! https://github.com/smly/vscode-fast-kaggle
* I started by cloning the best public kernel (MCTS Starter). I added features such as CLRI in the public notebook, which took a very long time to compute, and reimplemented many pandas operations using polars. This reduced the processing time from around 410 seconds to just 7 seconds, greatly improving my feature engineering efficiency.
* I spent the most time on feature engineering. To be honest, I was validating the improvements at the same time as tuning the parameters of the model, so I did not manage what was important. I browsed through the different game types on the Ludii Game Library to get a sense of how to approach feature engineering. I implemented features based on patterns common to multiple games, such as board shape, piece types, presence of randomness, rule complexity, and special movement patterns.
* As a post-processing step, I multiplied the predicted results by `1.2`. I optimized this coefficient using feedback from the public lb, so it could have been a complete disaster if the private lb had consisted entirely of unseen games. The most effective data augmentation technique was swapping agent1 and agent2. To further increase the amount of data and improve robustness, I implemented a more complex data augmentation strategy based on equivalence relations and transitivity. For example, if `A > B` and `B = C`, then I would generate the data point `A > C`. I also created a model using this augmented data and ensembled it, but it didn't seem to help in the end.
* The average of 10-fold predictions from a single catboost model resulted in a public score of 0.416 and a private score of 0.423. I didn't try XGBoost, LightGBM, or TabNet.
