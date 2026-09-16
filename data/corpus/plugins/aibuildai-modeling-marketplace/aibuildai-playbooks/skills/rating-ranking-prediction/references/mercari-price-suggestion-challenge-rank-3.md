# 3rd solution. 3300s to 0.3905 at public

Competition: mercari-price-suggestion-challenge
Rank: #3
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50272

Hi, everyone. it is really an exciting competition.

Here is my kernel:https://www.kaggle.com/whitebird/mercari-price-3rd-0-3905-cv-at-pb-in-3300-s

Description: My solution has two models: A CNN and a Ftrl-Fm based on @anttip's share:https://www.kaggle.com/anttip/wordbatch-ftrl-fm-lgb-lbl-0-42555?scriptVersionId=2164491. I did a lot of jobs on text normalizing to let my CNN works which cost much time. And it also improved the ftrl-fm model. The CNN is very slow with cpu, so you can see I did some tricks like the network structure and the training batchsize. I got my CNN to 0.400 and ftml-fm to 0.415 while I'm not a pro on traditional model. Maybe the feature etl could be done better, however I am just a software engineer and good at optimizing the algorithm and NN model. The code is a bit long and you can post anything here and I will try my best to answer it.


Hope you can find sth useful. Enjoy it. lol
