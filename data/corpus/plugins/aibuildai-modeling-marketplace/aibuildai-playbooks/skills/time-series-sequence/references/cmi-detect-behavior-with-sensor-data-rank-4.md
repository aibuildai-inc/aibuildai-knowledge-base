# 4th place solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #4
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/4th-place-solution

Thanks to everyone and congratulations to the winners! Let me give a quick summary of my solution and share the code.

# Target
The target for all my models was a combination of the 3: gesture, orientation and hidden Relax&Move / Move from the initial state of "behavior" column. This new target has 18 * 4 * 2 potential values and no subject had more than 1 sequence of each value recorded. That was the basis of the post-processing I did.

# Post-processing
During the inference I accumulate the history of predicted probabilities per subject. When a new subject comes in, I run a maximum llh optimization under the constraint that each target class can occur no more than once. You can use the optimization output of the new record as the prediction, but to get a tiny improvement on early records, when the history is too short yet, I just force set probabilities of the classes from past sequences to be zero for the currently processed sequence. Then I sum probabilities by gesture and take argmax.
As the data came in random order, that created a significant variance in the scores, final choice was kind of a lottery. Resubmits of my final solution scored from 0.868 to 0.880, I ended up selecting 0.876.

# Models
I used a blend of 2 model architectures, each heavily bagged. I separately training imu-only and full-data models, using the same architectures with thm and tof branches cut for imu-only variants. Imu-only models gave a very good boost when blended with full-data models.
1) cnn - attention - pooling architecture with separate branches for velocity, position, thm, thm diff, tof and tof diff branches. You can check the code if you are curious about details, but I think there was nothing special there.
2) Bert-based architecture and derived features borrowed from https://www.kaggle.com/code/wasupandceacar/lb-0-82-5fold-single-bert-model. Kudos to @wasupandceacar for publishing it! I only dropped some of imu derived variables as they hurt my cv score.

# Augmentations
I am not sure how big was the benefit of the augmentations was given the high variance of LB scores, but I used:
- Drop out of first records with phase == "Transition"
- Drop out of the last records with phase == "Gesture"
- Random noise
- Mixup of "Transition" and "Gesture" parts from different training records having the same target

# Feature engineering
Quite straightforward here, I didn't find value in complex derived features on cv, so I used a very limited set of features, like scaled thm and tof, acceleration without gravity components and angles derived from rot vectors. There were a couple of records with missing rot vectors, for them I used some rough estimates from acc, and also removed moving average from acc vectors to move the values closer to the desired distribution.
And, of course, flipping the values of some variables for handedness == 0 subjects.

I am sharing my code in full here https://www.kaggle.com/code/dott1718/cmikn-ensemble3b-re5
This is the inference notebook of my top private score (the resubmit I picked). The attached datasets contain the model weights as well as the code for feature engineering, training and inference. You can also find a lot of repetitive and redundant code of my failed ideas there :)
