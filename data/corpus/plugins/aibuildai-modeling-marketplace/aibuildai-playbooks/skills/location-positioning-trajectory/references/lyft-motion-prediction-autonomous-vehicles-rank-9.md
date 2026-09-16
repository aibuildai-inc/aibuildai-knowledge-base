# 9th place solution

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #9
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199636

First, it was a well organized competition. The engagement from @lucabergamini and @iglovikov on the forum is exemplarily, and one can see how @iglovikov corrects in this competition problems that he himself has probably experienced when he was a participant, well done. Second, I think Lyft investing and organizing Kaggle competitions is not something to be taken for granted. They give the data and lots of their time, but also expose their proprietary l5kit library to the competitors. I hope it pays off to a degree with the ideas that kagglers generate, but in any case thanks to Lyft for doing this, please continue. 

My solution at the end is surprisingly simple, because no fancy ideas have worked well. Yet, there are several things that helped me, I describe everything below. For me, it is once again a lesson to stick to the solid basics. I remember when I started 3 months ago I was excited to try GANs, as it [seemed to be the state of the art](http://papers.neurips.cc/paper/8308-social-bigat-multimodal-trajectory-forecasting-using-bicycle-gan-and-graph-attention-networks.pdf). I couldn't finish further away from it.

# Data

### Samples selection

The dataset parameters are `min_history_steps = 0` and `min_future_steps = 10`, which is standard and what was probably used to chop the test data. And I added another layer of filtering, - based on the state index. Stated index is a frame's count inside its scene, ranging from 0 to 250. In test, state index is always 99. If one decides to train on a chopped dataset he basically constraints himself to state index 99. But this cuts too much data, so I built a mask similar to the existing mask and require  `min_state_history = 30` and `min_state_future = 50`. I believe this change improved my scores. Without it I would get samples with state index starting from 0, which is not good.

### zarr files

From the beginning I used full train zarr, and two weeks before the end I added the chopped validation and the test zarrs. That last part may sound surprising at first. Basically we have 100 frames in the test and validation, and with my states index filtering I look only at frames 30 to 50 to train on. It is a legitimate data for the training, and my motivation was to expose the models to any specifics of the test dataset, like a raining day. After all the filtering I had that much data for the training:

- 140M full train
- 1.4M test, frames 30-50
- 1.9M validation, frames 30-50

### Rasterization

- 12 frames `[0,1,2,3,4,5,8,10,13,16,20,30]` into the past
- 3 channels semantic map
- another 3 channels of semantic map for history frame 30, gives traffic light 3s from the past
- future frame, when applicable (which is about `87%` of time). [Discussed here](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199498).
- AV vehicle is marked with a black dot (see the above link for an example)
- raster size `[256,192]`, all the rest of the parameters are defaults

All of the above modification have helped to improve the score. 

### Sampling

We had a lot of training data so I had a luxury to make sure that I did not attend the same sample more than once during the whole training. Basically I generated the order before the training and picked indices from it sequentially. 

# Model

The ensemble of 3 models, from [timm package](https://rwightman.github.io/pytorch-image-models/results/)

| Model| train | validation | LB public |
| --- | ---|---|---|
| mixnet_m, 3 traj | 12.10 | 12.62 | 12.976 |
| mixnet_l, 3 traj | 11.48 | 12.02 | 12.285 |
| mixnet_l, 6 traj | 6.88 | 7.10 |  |
| ensembled, 3 traj |  | 11.61 | 11.788 |

The approach to the model is straightforward 3 or 6 trajectories prediction, I initially got the idea from [this kernel](https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence), and it stuck. 

### Training

- the best model `mixnet_l` was trained for `215 (epochs) x 64 (batch size) x 2500 (iterations)=34.4M` samples
- Adam optimizer
- learning rate `4e-4` decaying to `1e-7`. Most of the training on `1e-4` - `2e-4`
- Auxiliary losses - also predicting 50 future velocities and yaws. It helped to converge faster, but not sure if it improved the accuracy. 

# Ensembling

I am proud of the ensembling procedure that I developed, however I have not tested it against the alternatives, and the benefit is only `0.5`. It seems like some people report good results with stacking, very interesting what is better. 

In my approach I gave weights `[0.8,0.3,0.5]` to my 3 models, which multiplied their own confidences. As a result I had 12 (3+3+6) trajectories with some probabilities assigned to them. How to ensemble those? I want to optimize the negative log likelihood metric and get three trajectories with confidences as output. If I assume that each of the 12 trajectories is a ground truth with its probability, then I have a well-defined optimization problem at hand. Still, it is a difficult non-convex optimization problem. By writing down the expression, requesting that the gradient equals to zero (local minimum requirement), I got to a fixed point formulation. Fixed point is an expression of form `x = f(x)`. From this by making several iterations I can get to an optimum. 

There are a couple of questions. First, does it always converge? In practice, it does always converge in 2-20 iterations. In theory it needs to be proven, I think it can be proven. Second, does it converge to a global minimum and not a local one? Given the problem structure I believe it is a global optimum, but maybe it is a global optimum only almost always. Third, does my original assumption about 12 ground truth trajectories with probabilities create a problem? It seems to me that it is a theoretically sounds approach, but it is a long discussion in itself.

The advantages of the approach is that I run it in batches on GPU, very fast. The results are stable, and it accepts any number of input trajectories. Specifically, I trained the 6 trajectory model to cover a wider range of possibilities, knowing that I can ensemble them effectively. 

# Things that did not work, all in one big pile

- weight decay, AdamP, AdamW, one-cycle learning rate policy
- freezing batch normalization layers at the last epochs of training
- changing agents box colors continuously based on their velocity. Bright - fast, dim - slow.
- GAN: train a model to differentiate between real and generated trajectories, with gradient reverse layer to propagate the gradient
- PCA: selecting 25 PCA components for 100 dimensional vector of the possible futures and learning to predict their coefficients
- predicting probability maps (like a segmentation task) and clustering those into 3 trajectories
- adding velocity, acceleration, yaws to the last layer of the network
- predicting trajectory as a difference above constant speed trajectory
- rasterizing output of a model and feeding it to a second level CNN model which outputs small corrections
- bigger raster sizes
- predicting heading and velocity instead of `x` and `y`
- adding 1-2 fully connected layers on top of the model
- learning a "specialized" model, - one output trajectory for fast modes, one for turns etc.
- augmentations: random cutout of semantic map, removing random agents
- resnet18, resnet50, mobilenetv3_large_100, densenet121
- adding constant speed and constant acceleration models to the ensembling with low weights
- smoothing final trajectories with B-splines gave indecently small improvement

Wow, I tried a lot of stuff. 

# Finally

Congratulations to the winners! It was a great competition, truly unique, challenging and rewarding.

And with that I am happy to become Kaggle competitions grandmaster, number 197. Thank you Kaggle and kagglers for that journey!
