# 39th Private LB: small and simple

Competition: deepfake-detection-challenge
Rank: #37
Source: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145732

Congratulations to everyone, and I hope that those who had scores that errored out get that sorted… Very frustrating.

**My approach in a nutshell - keep it simple and write off the hard deep fakes:** 

I figured that a complex model would not get these right anyway, and would end up messing up the easy ones too because it would be prone to overfitting. I ended up with 39th on the private LB (0.477); around 100th public LB (0.328).

**Details:**

My main worry was that models were not going to generalise well, primarily because of the training data we had available to us. My reading of the rules was that any external data was at best in a grey area, and more probably out of bounds, so I only used the data given to us.

I built as simple and as small a model as possible, knowing that this wouldn’t be able to detect hard deep fakes, but figuring that it might detect the easier ones more robustly.

**Dataset:**

I used @humananalog  Blazeface solution to save off faces in 32 frames from each video to the hard drive (evenly spaced throughout the video). I screened the videos, only allowing training cases where all of the face parts (eyes, nose, mouth, etc) were visible in all frames. This reduced the training set by c.25% and I think this was important in getting the models to converge properly. 

**Sampling:**

I sampled 50-50 real/fake videos in each epoch (all real videos + 1 sampled corresponding fake, each epoch choosing a random rake). Once a video was sampled, I randomly sampled 1 frame from the 10 frames with the highest confidence in that video. Overall, I’m not sure how much the sampling scheme affected the score. It didn’t have a huge impact on my validation set or the public leaderboard, but maybe it counted for something in the private leaderboard.

**Augmentation:**

Random horizontal flips, jpeg compression, brightness contrast. Randomly selecting frames as outlined above also did some of this job.

**Models:**

I used EfficientNet-b0 as it was the smallest model that gave reasonable results.

**Resolution:**

I used three different input sizes (separate models for each): 180, 196, 224. I used smaller rather than larger resolutions in a bid to make things more robust (reasons outlined above).

**Ensemble:**

Each model was fit over 5-fold samples (folder split). The end result averaged these.
