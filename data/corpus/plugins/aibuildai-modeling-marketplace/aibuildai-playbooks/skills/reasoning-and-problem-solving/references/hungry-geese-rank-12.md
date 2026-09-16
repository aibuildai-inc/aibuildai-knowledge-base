# Deep Q* Learning with simplified game ~1200

Competition: hungry-geese
Rank: #12
Source: https://www.kaggle.com/c/hungry-geese/discussion/255440

Once the submission deadline has passed I would like to share my solution for the challenge. I have prepared a [presentation](https://docs.google.com/presentation/d/1Qcf1tAl4PdOCoZuRddUdJ5ESJmevntuUf9S9Th_U8Aw/edit?usp=sharing) for my work colleagues that will be done on September. It is an ongoing work so feel free to make sugestions for improving.

Also the code is available now on [github](https://github.com/ironbar/hungry_geese).

My initial goal with the challenge is to try something similar with Muzero approach, however I started simple with Q learning and I took much more time than expected.

If the challenge would have lasted 4 more months I would have love trying this: [Vector Quantized Models for Planning](https://arxiv.org/abs/2106.04615)

## Brief summary

- Deep Q* Learning
- Big model (~20M parameters)
- Simplified version of the game (just 3 movements, head centered and always looking north)
- League of agents approach with Elo rating
- Data augmentation on test (horizontal flip and adversary flip)

For more details I kindly refer to the presentation or the repo. Good luck with the last matches!
