# [12th] Unet IL

Competition: lux-ai-2021
Rank: #12
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/296406

Thanks a lot the Lux AI and Kaggle for organizing such a great competition. It was very fun and entertaining, almost like playing a game. 

I had very limited resources: less than 3 weeks, Kaggle's quota and 1060 6Gb, so my main goal was to learn a bit of RL. I decided to start with basic Unet IL, because it was successfully used by some teams in the last Halite competition, and then use it as starting policy for RL. But thanks to great [DeepMind's RL course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm) and some experiments, i understood that due to several seconds inference time per match it would require a lot of time and luck to find the right reward shaping, and chances were not that high for off-policy method with bootstrapping. So i sticked to improving my IL model.

My basic Unet model had 3 levels and 64 channels and was trained using CSE weighted loss, 1/10 for no-op. Features were pretty basic, i didn't even use distance metrics.
At first, due to memory constrains, i used only 1 Toad team submission, 300 episodes. Then i moved scalar features to separate input, embed them and concat as additional channels at top level of Unet. This with custom background loader allowed me to efficiently train model with 3 submissions, 1100 episodes. 

My basic model used deterministic action predictions only for workers and scored at 1350.
Each of these changes gave a +60 score:
- city action model
- 3 submissions
- flip-rotate augmentations
- switching to stochastic-deterministic (deterministic if probability > 60%)
- 128 channels Unet

Winning submission was slightly better and less noisy. It used 3 models for early, mid and endgame; each trained with different loss weights. Also i stacked states from current and previous timestamp but have to use fp16 mode.

Focal loss, 2head models and additional features didn't work for me that well. 

I used [catalyst framework](https://github.com/catalyst-team/catalyst) for training.

Conclusions: it is possible to score a gold having limites resources and if you want to use RL you have to start as early as possible because you will need all available time
