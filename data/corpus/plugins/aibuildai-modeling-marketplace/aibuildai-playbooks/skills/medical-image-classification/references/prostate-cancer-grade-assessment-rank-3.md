# 3rd place solution

Competition: prostate-cancer-grade-assessment
Rank: #3
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169232

Congrats to everyone and host of competition!
I am surprised that I am get 3thd place on private leaderboard. I was just lucky 🙂 . I stopped improving my solution 1 month ago, because has problems with my GPU and didn't want to spend more credits on AWS, because I have not seen any improvement on LB and CV.

I am experimented with different networks and my custom tile cropping, hard augmentations. But  the best results I get on simple sollution based on @haqishen [kernel](https://www.kaggle.com/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87).
I am trained 2 effnetb0 and used round logits before average them, this approach give me 0.880 on public and 0.934 on private.

Github with my experiments: https://github.com/Dipet/kaggle_panda
