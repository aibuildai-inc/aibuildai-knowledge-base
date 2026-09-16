# 5th place - Partial solution

Competition: youtube8m-2019
Rank: #5
Source: https://www.kaggle.com/c/youtube8m-2019/discussion/112296

First of I would like to thank the organizers for setting up another interesting competition. I hope that the Kaggle team will also do the right thing and clear the leaderboard of all the cheaters and competitors that broke the rules. 

It has been a great pleasure working on this challenge with both Mikel and David. Unfortunately, we lacked time to go into details and explore "exotic" solutions.

Here I will describe part of the solution and Anokas will describe his part. We ended doing ranked sum ensemble for our final submission.

There are 3 part to the solution presented here:
1. Video level network
2. 5 Frame network
3. Localization network

## Video level network - P_V
This was basically based on last year’s 1st placed solution - [link](https://github.com/miha-skalic/youtube8mchallenge) . Essentially, for each fragment we would factor in prediction based on the whole video sequence.

## 5 frame network - P_5
We trained three models: 1x DBoF, 2x VLAD on sequences from 2nd year data, sampling 5 frames. Then we fine-tuned the model on annotated fragments from this year.

##Localization network – P_L
This network took in a sequence of frames and a target label. The target label would be passed to an embedding layer and then concatenated with the sequence of frames. The concatenated sequence would then be processed by an LSTM to predict whether target label was predicted for each frame. Non-annotated frames would be masked out.
Downside of this approach was that we needed to run inference 1000 times. 1x for each target label.

##Combining the 3 models

Multiplying the probabilities (weighted geometric mean) gave the best results.
Essentially probability for each fragment-class combination, p(fc), was computed as:
p(fc) = Log (P_v) + 2/3 Log (P_5-VLAD1) + 2/3 Log (P_5-VLAD2) + 2/3 Log (P_5-DBoF) + Log(P_l(*|c))
From here on we would just sort fragments based on p(fc) and for each class c report the top fragments f.

Detailed solution will be provided as workshop submission.
