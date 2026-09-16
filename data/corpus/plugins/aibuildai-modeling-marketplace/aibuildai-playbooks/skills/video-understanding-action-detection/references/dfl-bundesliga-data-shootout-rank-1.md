# Team Hydrogen: 1st place solution

Competition: dfl-bundesliga-data-shootout
Rank: #1
Source: https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/359932

Thanks a lot to Kaggle and the hosts for this very, very interesting competition. A 2-stage competition is always a thrilling experience, but then seeing your models generalizing so well to completely unseen data is really exciting.

Our solution is based on a single-stage model combining 2.5D and 3D data to properly extract temporal information from the video data. This solution ranks first in both public and private leaderboards. Next, we want to give a high-level overview of this solution.

### Validation setup

In this competition, we used a fixed train/validation split to tune our models: the validation videos used for this included 4ffd5986_0, 407c5a9e_1, 9a97dae4_1, and ecf251d4_0. Our best local score was 0.857, and we observed strong correlation between local validation and public leaderboard (LB). For our final submission, we retrained our best models on the full data that was available to us, this is a general procedure we always try to take in the end.

### Model setup

As mentioned, we solely rely on a single-stage model architecture. The input for our models was a 1024x1024 grayscale image, with the channels stacked in the time dimension over three neighboring frames. We chose to use grayscale images rather than full color due to the improved generalization and runtime capabilities it offered - and it also seems to generalize extremely well to the unseen private data.

The single-stage architecture combined 2.5D and 3D techniques. The backbone of the model was either efficientnetv2_b0 or efficientnetv2_b1, as we found that larger backbones tended to overfit quickly on the small dataset. 3D convolution layers were only used in the last block and final convolution before pooling, which allowed us to stay in 2.5D for most of the backbone and achieve fast inference times with caching.

In a bit more detail, if our current frame is `t=15`, we first stack three neighboring frames `{14,15,16}` as the channels of the input, and separately pass five different time steps (2 in each direction), each again stacked with three channels. If our time step would be 3, this would mean we pass `{8,9,10},{11,12,13},{14,15,16},{17,18,19},{20,21,22}` through the backbone, and the final 3D layer aggregates these 5 individual time steps with additional pooling.

We also experimented with different stacks over the time domain and found that 15 frames (combined over 2.5D and 3D) worked best for our final submission.

We applied the usual set of augmentations and used Mixup as well. The loss function was binary cross-entropy with three target columns. One area where we had room for experimentation and improvement was in how to set the hard or soft labels for training. In the end, our best solution used hard labels in a small window around the actual event.

Most of our time during this competition was spent training on only the labeled regions of the competition data. Pre-training on some Soccernet labels improved the score by around 0.02, but we had no success with pseudo label pre-training on unlabeled regions and clips.

### Inference

Increasing the inference resolution by 128 also slightly improved our cross-validation (CV) and LB scores. We spent a significant amount of time optimizing the inference kernel to run as efficiently as possible, utilizing both CPU cores with threading. We predicted only every second frame for a single model, and our final solution blended the results of two models, alternating between them frame by frame. We then applied post-processing to reduce the number of false positives.

A single model had a runtime of 2.5 hours in the kernel, and processing a single video took 25 minutes. Our final solution with two models had a runtime of around 5 hours. There was potential for using even more models, but we opted to keep things as simple as possible.

Big shoutout to my teammates @ybabakhin and @ilu000!
