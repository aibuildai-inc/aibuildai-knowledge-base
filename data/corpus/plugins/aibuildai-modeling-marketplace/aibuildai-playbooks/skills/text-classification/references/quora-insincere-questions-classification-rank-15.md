# 15th Solution - focus on models

Competition: quora-insincere-questions-classification
Rank: #15
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80540

it is my first competition, and our term focus on models.

now, we have release our model in https://www.kaggle.com/xiaobai1123q/15th-place-solution 

we run our model again ( because the submitted kernel is a version ), and achieve a better result than leaderboard. I hope we can bring you some help.

in the text preprocessing stage, we don't have any personal work, all of which are public kernels.

our main job lies in the four models we ensemble. then, i will briefly explain.

the first model is RCNN.
the second model is LSTM(128) + GRU(96) + maxpooling1D + dropout(0.1).
the third model is LSTM(128) + GRU(64) + Conv1D + maxpooling_concatenate.
the fourth model is LSTM(128) + GRU(64) + Conv1D + Attention.

we used the word vector concatenated by glove and fasttext.
we set max_features = None and we set max_len = 57.

questions, advises, suggestions are all welcome.
