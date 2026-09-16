# 8th place solution [My part]

Competition: happy-whale-and-dolphin
Rank: #8
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/319900

We decided to choose minimal sharing inside our team, so our solutions are different.

Please check topics from my teammates:
1. https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/319868
2. https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/319894

All my models are effnet-b7, I use CosineAnnealingLR with SGD and AMSoftmax with scale=35 and margin=0.35. I've train on whole data without folds.

I exploit only one key idea: one embedding space for all representaion of each image. It means I take several crops for each image and just add them as new images. 

I made several datasets:

1. body and fin - 806 on lb
2. body, fin anf fullframe - 811 on lb
3. body, fin, fullframe, detic box - ~815 on lb (didnt send, trust cv)

After merge this 3 models I got 838 on lb.

After team merge we got pseudo from our ensenmble and I retrained my models with pseudo (70%).

After concat 3 models without pseudo and 3 models with pseudo I have 862 on lb.

Our final ensemble - 884 on lb.

For new_individual I used fixed threshold 19%.

Thank you @kwentar, @lenny27, @ilyadobrynin for this competition!
