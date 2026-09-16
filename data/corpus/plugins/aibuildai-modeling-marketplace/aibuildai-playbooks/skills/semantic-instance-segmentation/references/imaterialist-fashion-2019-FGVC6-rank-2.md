# My solution (2nd public | 2nd Private)

Competition: imaterialist-fashion-2019-FGVC6
Rank: #2
Source: https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/discussion/95233#latest-551075

My solution can be described as `mmdetection.fit_predict ()`

I started practicing it immediately after iMet Collection 2019 - FGVC6 and there were not many attempts and experiments. But I had several devboxes. I started with two machines with 3x 1080Ti each. And for the weekend I redistributed the working tasks so can get two more.

All my models are Mask-RCNN with a Hybrid Task Cascade.
r50, r101 with an input resolution (800, 1266) and x101-d32 with (720, 1080). Starting SGD with LR 0.02. I did not throw out the weight of the last convolutions and logites but cut the tensor to fit a new number of classes.
I did not have a solid scheduling. I planned to just pre-train the model at low resolution. So each model was trained through 6-10 epochs, LR dropped 1 time.

Then I start training from the last checkpoint on resolution (1024, 1600) with the same LR from previous stage. r50 didn't perform very well and I changed it to train r101 with SyncBN. I still had doubts about unfreez BN, because the batch was 3. As a result, it did not work well. The score was much worse than the models with frozen BN.

Somewhere on Saturday, I started experimenting with multi-checkpoint TTA, flip-TTA, scale-TTA. Everything was fine. But suddenly it turned out that I forgot to exclude validation from my trainset :facepalm:
As a result, I used leaderboard as validation :facepalm:x2

r101 managed to train through 18 epoch and x101 through 9 epochs. Both models end up at LR 0.00002. My best submission is the last two checkpoints from each model with 3x scale TTA and mirror TTA.
