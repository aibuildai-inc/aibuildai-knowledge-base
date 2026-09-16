# 4th place solution

Competition: hungry-geese
Rank: #4
Source: https://www.kaggle.com/c/hungry-geese/discussion/263690

First of all, I would like to thank the hosts for organizing such a fun and wonderful competition.
I would also like to thank the HandyRL team for releasing the wonderful library [HandyRL](https://github.com/DeNA/HandyRL). Thanks to them, I was able to enjoy this competition.

The following is an overview of my team's solution. For more details, please refer to [my blog post](https://www.currypurin.com/entry/hungry-geese-2), which is in Japanese.


### Input for Network

* head position: 4 channel
* tail position: 4
* body positions: 4
* body positions (reduce gradually from the tail): 4
  * starting with the tail: 1.0, 0.95, 0.9, 0.85...
* previous head position: 4
* A board that represents the certainty or possibility of losing a body on the next step: 4
  * 1.0 for the tail and the part before the tail that will definitely disappear in the next step, and 0.5 for the part that may or may not disappear.
  * This takes into account the possibility of eating food and the fact that the body will be one step shorter after 40 steps.
* A board showing the difference between the Agent's head position and its own Agent: 1
  * A board where take the difference between the length of my agent's body and the length of opponent's agent's body, and enter the difference in the opponent's head position.
  * I created it with the intention that if my agent's body is longer than opponent's, my agent should stand strong, and if own is shorter, should stand safe!


# Training

Almost using the initial parameters of HandyRL.

* For the evaluation Agent, I set up an Agent (rating of about 1090 at the end of the competition) that was additionally trained from the weights shared by HandyRL.
* The learning curve is as follows (200 new replays are created per epoch)




# Inference

* Horizontal Flip, Vertical Flip、Horizontal Flip + Vertical Flip, and no augmentation
  * This augmentation will improve my LB score by about 40.
* In addition, we randomly set n and m, and move the board n squares to the left and right and m squares up and down to perform the above augmentation.
  * This augmentation is done for about 0.8 seconds, and the direction to go is determined by simply averaging the output predictions of about 40 solid values and taking argmax.
  * Compared to no augmentation, the score gain from this augmentation is probably only about 50, so increasing the number of augmentations won't make stronger.

# Not worked

* Monte Carlo tree search and other explorations are worked on (mainly by teammates), but without improving the score. So, no exploration for the top solution of my team.
