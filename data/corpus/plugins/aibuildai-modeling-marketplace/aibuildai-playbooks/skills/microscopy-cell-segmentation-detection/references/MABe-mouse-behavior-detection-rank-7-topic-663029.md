# 7th Place Gold - CNN Transformer with Invariant Features

Competition: MABe-mouse-behavior-detection
Rank: #7
Source: https://www.kaggle.com/c/MABe-mouse-behavior-detection/writeups/7th-place-gold-cnn-transformer-with-invariant-fe

I really enjoyed participating in this competition. Two months ago, I was immediately attracted when I saw that there were no public deep learning public notebooks because this challenge is exactly what deep learning loves! I joined this competition with the simple goal of beating the best public XGBoost notebook with an NN.

# XGBoost Public Notebook - Top 100!
Thank you Mahdi Ravaghi for the strong XGBoost baseline [here][1] (and thank you other Kagglers whose ideas are contained within). After changing that notebook to build one model per `lab_id` per action. And training on 100% train data. The notebook achieved `CV = 0.475, Public LB = 0.477, Private LB 0.450, (88th place)` top 100 rank, wow!

# Local Validation Score
In this competition, we observe a near perfect CV LB relationship when using the correct validation score. (i.e. when XGB CV was 0.475 then XGB Public LB was 0.477). The correct CV is computed using only 15 `lab_id` which excludes labs `MABe22`, `CalMS21`, and `CRIM13` because the host said these labs did not exist in test data. So all CV scores discussed in this write up, use the correct local validation score. (UPDATE: relationship with Private LB isn't as good)

# How To Build an NN
The power of NN is transfer learning from one lab to another! To make NN shine, we need to preprocess the data in some invariant way because the labs are different. Also we need to define the target labels correctly. Lastly we use data augmentation to unlock NN's full potential 🔥

We train our NN on all labs together. From seeing one lab, the NN will learn to make better predictions on another lab. To accomplish this, we need to make input features `lab_id` invariant. We can further encourage transfer learning by using data augmentation to minimize `lab_id` feature differences. (i.e. if mice are difference sizes between labs, then we data augment randomly scale sizes of mice up and down).

# => Positive, Negative, and Masked Targets
Our NN has 37 output neurons to predict the probability of each of the 37 possible actions. However not all videos nor labs contain all 37 actions as seen below:



When training our NN, we need to define the targets correctly. For every video frame, the `target=1` if `train_annotation` says 1. Otherwise it is either negative or masked. We set `target=0` when `train.csv` says `behavior_labeled` but the target is missing in `train_annotation`. And we set `target=MASK` if an action is both missing from `train_annotation` and not included in the `behavior_labeled` list. "Mask" means that we do not include loss for this action in this video frame. (And t=0 and t=1 applies loss).

# => Invariant Mouse Positions
If we input the mice body parts x,y as they are in the train_tracking files, the NN will get confused. Because the arena sizes and shapes are different for each lab. The position `x,y = 10,10` in one lab will not mean the same thing as position `x,y = 10,10` in another lab. Below we see the variation among 3 lab arenas by making a scatter plot of 1 mouse's nose over 5 videos. We see that the labs are different shapes and sizes:



There are at least 3 ways to make invariant mouse body part positions for our NN:
* Agent centric coordinate system - Shift the agent mouse to the origin with midpoint between `ears` at `(0,0)` and rotate `tail_base` pointed down negative y axis. (All other mice in the video get shifted and rotated with the same transformation).
* Min max scale (using 5% and 95% quantiles) all the `x,y` so that left side of arena is 0 and right side of arena is 1. And bottom is 0 and top is 1. (Also provide NN the `pixels_per_cm` feature).
* Use distances instead of absolute `x,y`. Instead of saying nose is at `(10,10)`, we use the distance between `nose` and other body parts such as `tail_base`. We can compute distance as either a vector or a magnitude.

The above bullet points make positions invariant. Regarding velocity, we note that velocity vector and velocity magnitude are already invariant. And we note that all distances (as vector or magnitude) between agent mouse body part and target mouse body part are already invariant. 

Below is an example of agent mouse nose position scatterplot after we transform with agent centric coordinate system. We observe in the scatter plot that all noses positions are now around `(0,2)` and when the mouse turns it head left it goes to `(-2,2)` and when it turns right it goes to `(2,2)`:



# => Invariant Mouse Body Parts
Another challenge for transfer learning between labs is body parts. We observe that different labs track different body parts:



We observe that all labs have `["nose", "ear_left", "ear_right", "tail_base"]`. Then about half of the labs use `["hip_left", "hip_right"]` and the other half use `["lateral_left", "lateral_right"]`. It turns out that `hip` and `lateral` are similar enough that we can rename them both `["side_left", "side_right"]` to maximize transfer learning!

# Build CNN Transformer
We train our NN using all labs together! Using the 8 body parts listed above and the method of positive, negative, masked loss. We build a CNN Transformer using invariant position, velocity, and distance features. The CNN layers come first and convolute in the time direction. They learn to make features based on how everything is changing. Next the NN feeds these features into layers of attention. Using cross (mouse) attention and self (mouse) attention, the transformer layers learn to compare agent and target mice and compare body parts within a single mouse. Finally the head layer outputs 37 outputs giving a probability for each of the 37 possible actions!



# Different Time Scales
We train the same CNN Transformer multiple times and use a different length sliding window each time (i.e. vary from 2 seconds to 16 seconds). We also add more dilated convolution layers to the longer time windows to capture the larger time scale. Afterward we have four different trained NN and we can ensemble them:
* window = 64 frames, stride = 16 (approx 2 second window), CNN layers = 6
* window = 128 frames, stride = 32 (approx 4 second), CNN layers = 8
* window = 256 frames, stride = 64 (approx 8 second), CNN layers = 10
* window = 512 frames, stride = 128 (approx 16 second), CNN layers = 12

# Data Augmentation
Using data augmentation gives a huge boost in performance. Data augmentation helps for two reasons. First is makes it harder for the NN to overfit (i.e. memorize instead of generalize the data) and we can train for more epochs. Second, it helps transfer learning between labs because all the labs start looking the same. For example if one lab had larger mice than another lab, after random scaling up and down, the model receives all sizes mice from every lab. We use
* time augmentation - (randomly change 30 fps to 15 or 60)
* rotation augmentation 
* x scale augmentation
* y scale augmentation
* horizontal flip
* relabel right with left body parts
* body part drop out

# Inference
We predict using the same slide window scheme that we train with. Then each video frame predicts 37 probabilities. Next we smooth probs in time and filter actions that were not in `behaviors_labeled` (per video_id, agent_id, target_id triplet). Lastly we apply a dictionary of action thresholds per lab_id. Each action under threshold gets set to prob=0. Lastly we take `argmax` to pick the frame action. (UPDATE: I have an unselected submission which achieves +0.004 private LB (rank 4th or 5th) that uses better tie breaking logic to decide what happens when multiple actions exceed threshold).

# Ensemble of NN - CV 0.546 - Public LB 0.551 - Private LB 0.518 (8th Place)
We take an equal weighted average of all our different time scale NN and achieve `CV = 0.546` and `Public LB = 0.551` and `Private LB = 0.518` (rank 8th). In this ensemble we include 3 families of NN. Each family uses a different set of invariant features and a range of time windows. In total our final ensemble use 2xNN from family one, 5xNN from family two, and 3xNN from family three (for a total of 10 NN).

# Stack XGB over NN - CV 0.559 - Public LB 0.562 - Private LB 0.523 (7th Place)
To boost our NN score, we train an XGB using one OOF from each of the 3 families of NN and all the features from Mahdi Ravaghi public notebook [here][1]. We train one XGB per lab per action. This XGB's CV achieves `CV = 0.528` and `Public LB = 0.536` and `Private LB = 0.508` (rank 12th). And when we blend it with our Ensemble NN, the result is `CV = 0.559` and `Public LB = 0.562` and `Private LB = 0.523` (rank 7th), wow!

[1]: https://www.kaggle.com/code/ravaghi/social-action-recognition-in-mice-xgboost
