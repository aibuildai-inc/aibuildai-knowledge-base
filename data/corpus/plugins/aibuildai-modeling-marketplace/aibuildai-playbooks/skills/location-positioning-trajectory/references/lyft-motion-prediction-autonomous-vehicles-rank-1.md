# 1st Place Solution & L5Kit Speedup

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #1
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/201493

# 1st Place Solution & L5Kit Speedup

## Introduction

First of all, I would like to thank Lyft for this great challenge and of course for their huge dataset. I must say I was a bit overwhelmed by the sheer amount of agents in the training set, but it was a fun experience to be able to train without ever using training data twice. Well, we did, when we used pre-trained models, but more on that later.

I also want to thank my Teammates @philippsinger, @christofhenkel and @nvnnghia for the great collaborative teamwork during the competition. We worked well together with a pipeline consisting of two repositories (version controlled custom l5kit and a training repo) and used logging in neptune.ai to keep track of the experiments. 

## TL;DR

Our solution is an ensemble of 4 efficientnet models trained with different rasterization parameters indivdually on train_full.zarr and stacked on validation.zarr. Key to training that many models on this huge dataset were several key improvements to the original l5kit repository to remove CPU bottleneck and enable efficient multi-GPU training using pytorch. 

## Improvements/Changes to the L5Kit

One of the major challenges in this competition was the slow rasterizer. We profiled the l5kit a lot and pinpointed the bottlenecks to speed it up by a factor of 4+. After that, our experiments were mostly GPU limited and multi-GPU experiments allowed us for "fast" iterations. Running through train_full (191_177_863+ samples) was possible within 2 days with medium sized models. 

### Speedups

Speed before changes with semantic view only (i5-3570K single thread, so ignore the actual time):

```
Raster-time per sample: 0.124 s
Hits         Time  Per Hit   % Time  Line Contents
==================================================
9   11147507.0 1238611.9     93.6    rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
```

All the speedups were achieved with plain python, no C++. The main speedups came from:

1. Batched transformation of boxes with modifications from https://github.com/lyft/l5kit/pull/167. 
   It has been said that the transformation is the bottleneck in the rasterizer as it is called a lot. We used the vectorized transformation whereever possible.
2. In `box_rasterizer.py`, using python lists of small numpy arrays instead of one two large numpy arrays (`agent_images, ego_images`).
   Python lists are surprisingly fast! `np.concatenate` on large numpy arrays is slow. Pre-allocation and writing to a single large array is even slower.
   The lists can be cast to a numpy array by `out_im = np.asarray(agents_images + ego_images, dtype=np.uint8)`
   This way, boxes will stay in `uint8`
3. Moving concatenate to the GPU
   As `np.concatenate` is slow on large arrays, we do that on the GPU.
   Replace `"image": image,` with `"image_box": image_box`, `"image_sat": image_sat`, `"image_sem": image_sem`
4. Speedups from https://github.com/lyft/l5kit/pull/140
   With the changes above, they now made a larger impact than the 7-8% stated in the PR.

Speed after changes with semantic and satellite view (i5-3570K single thread, so ignore the actual time):

```
Raster-time per sample: 0.032 s
Hits         Time  Per Hit   % Time  Line Contents
==================================================
   9    2847054.0 316339.3     84.2  rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
```


### Additions/Changes

- New rasterizer which combines satellite and semantic view (`py_sem_sat`).
- Flag: AgentID used as a value for drawing (instead of just boolean `no agent = 0`, `agent = 1`)
- Flag: AgentLabelProba used as a value for drawing (instead of just boolean `no agent = 0`, `agent = 1`)
- Flag: Velocity used as a value for drawing (instead of just boolean `no agent = 0`, `agent = 1`)
- Option to draw all objects on the semantic layer, combinable with the label probas from above.
- Filter options for agents: `th_extent_ratio, th_area, th_distance_av` to give a larger training dataset (or harder as you want to phrase it)
- Multiple parallel rasterizers for ensembling models that use different raster sizes
- Satellite view fix for non square shapes (had some weird transformation in it before)

[rasterized_image]

We will make our L5kit code changes public in the next few days and after cleanup. Most likely with PRs to the original repo.

## Journey & Experiments

With all the speed improvements from our custom L5kit we were able to eliminate the previous CPU bottleneck and run experiments on an efficient multi-GPU setup. We mostly trained on 8xV100 nodes using pytorchs awesome distributed data parallel (DDP), which enables to effectivly scale a training script to multiple GPUs. This efficient training setup was key to run quite a few experiments. (See list of stuff that did not work below) In all our experiments, we used the chopped validation set for local validation as discussed [here](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/188695). This yielded us very good prediction of the LB score (in the sub 13 range, the score was usually ~0.2 higher than local validation with a standard deviation of around 0.3). 

Originating from the provided baseline and the multi-mode prediction from [here](https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence), we experimented with raster size, pooling layers, head-modifications, usage of different backbones, learning rate scheduler, subsampling and oversampling to name a few. It was a major challenge in this competition, that experiments had to be run for at least 50 % of a full epoch of train_full.zarr or when comparing different sample sizes or learning rate scheduler even until the end to be really comparable. Subsampling the train set or stepwise decrease of the LR quickly yielded a low training and validation loss, but in the end, the performance was worse.

Quite quickly, we decided to always use all data, and settled to a setup using shuffled train_full.zarr (with a fixed seed to be able to resume training) with a linear decay learning rate scheduler. To keep the training sample fixed, we also settled to an AgentDataset with `min_frame_history=1` and `min_frame_future=10.0` to best resemble the test dataset. Opposed to the many discussions about a much lower training loss, we never experienced that. Without any augmentation, the training loss is of course slightly lower, especially in the end of the epoch, but usually within 80% of the validation score. We found that larger backbones help to reach lower loss levels, but starting at the size of an EfficientNetB7, the nets seem to be overconfident with their predictions and the LB score was sometimes shaky. 

The first half of test is public, second half is private test. We couldn't identify any major differences in the two sets. With that knowledge it was possible to hide the true score (setting some public rows to zeros), we wonder how many teams did that and were afraid of surprises in the private LB due to this. 

## Best Single Model

[model]

Our best single model is actually one that not fully finished on the last day. It's just an EfficientNetB3 with a Linear Layer Head and dropout attached trained on an extended train_full.zarr Dataset (`min_frame_history=1, min_frame_future=5.0, th_distance_av=80, th_extent_ratio=1.6, th_area=0.45`)

- history_num_frames: 30.0
- raster_size: [448, 224]
- pixel_size: [0.1875, 0.1875]
- rasterizer: py_sem_sat
- batch_size: 64
- dropout: 0.3

The model was pre-trained for 4 epochs on different lower resolution images (starting with 112, 64 and pixel size 0.75, 0.75) , so you can argue this is also an ensemble of some kind. The 5th epoch was trained with the parameters above and a very customized learning rate schedule that starts with a consine anneal and then transitions to a stepwise lr scheduler as seen in the graph below. The last evaluation score is 9.697.

[best_single]

The model achieves a public LB score of 9.776 and a private LB score of 9.070. With that, it would also have ranked 1st place this competition on it's own.

Pretraining is VERY important! We discovered it by a mistake where we loaded an EfficientNet without pretrained weights from imagenet. It performed significantly worse by ~1.00.

## Ensemble

Ensembling proved to be challenging in this competition, as traditional methods are not working well with the metric in use. We tried a lot of manual blending and form of post-processing without success. We then started building stacking models taking the raw predictions of the models as input, but this was also surprisingly not working well. At about the same time as @hengck23 posted it [here](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/193908#1077899), we had the idea of using the features/embeddings from each model as an input to a ensemble head. This worked much better and we could improve upon our single models specifically as we could introduce some diversity into the blend (different image size, pixel sizes, etc.) We also went with the dropout + single linear layer approach here, and all modification we tried with the head performed worse.

[ensemble_arch]

To prevent overfitting, the ensembling was done on the chopped validation set (without using the chopped off actual validation part), in some experiments extended to validation + test set without noticeable change in the metric but with about twice the runtime. 

Ensembles of two similar models (even just two different checkpoints) already gave a boost of about 0.2 in the metric. Naturally, with more diverse models the ensemble performed even better and we were able to achieve our best public LB score of 9.319 and private LB score of 8.579 with an ensemble of 4 models (B3, B5, B6, B6) on two different raster sizes. 

## Bootstrapped Validation

We evaluated model and ensemble performance with bootstrap method on the chopped dataset. For more diverse ensembles we also got the lowest standard deviation from the bootstrap method. For our final submission it was 0.22754. The best single models had a standard deviation of ~0.27. We also identified our B7 and B8 models to have a slightly larger standard deviation in the bootstrap method of ~0.36-0.39 which may also explain their partial performance degredations on the public LB. Despite their good validation score, for that reason we excluded them in our final ensembles and final submissions. 

[bootstrap]

We plotted CV vs public LB for our submissions and found a spread of +-0.3 for most of our single models (green corridor) with just the B7 and B8 models outside of that corridor (not shown in the picture). Our ensembles are in an even smaller corridor with a spread of +-0.1 (orange corridor), proving their superior robustness and generalization capabilities over single models. 

[cv_lb]

Given this robust CV & LB correlation, we were very confident that improvements in validation also lead to improvements on the leaderboard, so we did not have to sub it. So the sudden jump on public leaderboard only happened after we decided to sub our better models and was not some sudden magic we discovered.

## What Didn't Work

- Flag: AgentID used as a value for drawing (instead of just boolean `no agent = 0`, `agent = 1`)
- Flag: AgentLabelProba used as a value for drawing (instead of just boolean `no agent = 0`, `agent = 1`)
- Flag: AgentLabelProba used as a value for drawing (instead of just boolean `no agent = 0`, `agent = 1`)
- Option to draw all objects on the semantic layer, combinable with the label probas from above.
- Kalman filtering the output
- Any non NN prediction method
- Prediction of only the difference to a constant velocity model
- Using additional informations, such as velocity, history positions, label probas or anything else we could find in the head of the models. 
- Heavier heads
- 3D CNNs
- RNN based approaches on history frames for backbone input or outbut 
- Using additional features from shallower layers of the backbone
- Ensembling with kmeans, or other analytic clustering method that we tried. Some success at scores down to around 14, but no improvements below that. 
- Subsampling the training set. Huge speedups, but accuracy was always slightly lower
- TTA with modified yaw angles
- Unfreezing the backbone after ensembling and continue training with a second epoch

## What We Didn't Try

- Augmentation
- Multi backbone, multi rasterizer training
