# 37th place solution

Competition: um-game-playing-strength-of-mcts-variants
Rank: #37
Source: https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/550119

**Just More Model !**

**First of all**, I'd like to thank the organizers and Kaggle for hosting this competition; it was a very interesting contest. After trying a large number of local experiments, I found that CV wasn't very reliable, so I chose to trust the LB — but not entirely, because I saw many people in the discussion forum saying that the LB was unreliable. What surprised me greatly was that the difference between the public score and the private score was very small; this is a situation I've hardly ever seen before.

Below is our team's solution. The key point is integrating more models; there weren't any other impressive or special techniques used elsewhere:


**Secondly**, I'd like to share the biggest problem I encountered when submitting my solution: **Kaggle Error! **After continuous troubleshooting, I ruled out "out of memory" and "GPU memory overflow," and discovered that the root cause was insufficient disk space!!!

The reason for the disk space shortage was that in Model 3 (DeepTables NN), I had placed the code to load the model inside the "predict" function. This led to the model being loaded repeatedly during the online testing phase (with a batch size of 100, meaning the model was loaded 600 times). In fact, during debugging, after about 30 repetitions, it had already exceeded the maximum disk capacity (Max: 57.6GB). Afterwards, I moved the code to load the model above the "predict" function, and the problem was resolved. I hope my experience can help everyone.

**Finally**, I want to thank my friends @yunsuxiaozi , @andreasbis , and @yekenot for their public solutions. I greatly admire their spirit of open-sourcing their code. Therefore, I have also made public the data, models, and code I created during this competition, hoping to be of some help to everyone. It should be especially noted that Alice ranked very high on the public leaderboard, but unfortunately, he didn't seem to choose this set of solutions, causing him to miss out on a medal. I hope he can earn his first medal soon.

Code: [https://www.kaggle.com/code/faykudbq/mcts-deeptables-nn-af7adb](url)
