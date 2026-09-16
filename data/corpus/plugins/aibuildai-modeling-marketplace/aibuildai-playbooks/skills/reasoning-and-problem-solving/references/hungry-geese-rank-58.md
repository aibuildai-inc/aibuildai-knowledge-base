# How about that, my imitation learning / resnet model is my best sub

Competition: hungry-geese
Rank: #58
Source: https://www.kaggle.com/c/hungry-geese/discussion/263130

Started the competition with some testing with imitations models but left the testing and tuning to favor more and deeper knowledge in the RL area.
The RL models trained for 3-5 days performed best during the whole competition but 1-2 weeks ago the imitation model/subs from the beginning suddenly started to see the light and now 1 day left it’s my best performing model.
The imitation model is a resnet like architecture with swish activation trained with ~4M games/replays. Its weight fits barely the size limit with 96Mb. Replays all had a cutoff in the 1100 region, trained for 16 epochs.
Fun that the first models, not fully tested and tuned is now the best, sometimes one need some extra luck 😉

A reflection: Maybe one needs 1-2 weeks of RL training to beat the imitation model trained just for some hours, both using the same hw. Or it’s just a lucky week/weeks for the imitation model.
