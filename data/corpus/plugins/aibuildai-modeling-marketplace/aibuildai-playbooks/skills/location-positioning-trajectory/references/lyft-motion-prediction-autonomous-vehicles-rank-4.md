# 4th place solution: Ensemble with GMM

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #4
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199657

Thank you to the organizers and congratulations to all the participants. 
Also I would like to thank my team members @zaburo, @qhapaq49, @charmq for our hard work, I could enjoy the competition!
The LB was really stable in this competition due to the big amount of data, we could work on improving the model without caring about big shake-up.

We have started with my baseline kernel [Lyft: Training with multi-mode confidence](https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence).
Below items are substantial changes we made:

[Update 2020/12/9] **We have published our code:**
 - https://github.com/pfnet-research/kaggle-lyft-motion-prediction-4th-place-solution

# Short Summary



Published baseline training pipeline was indeed already very strong. 
Just modifying

1. train_full.zarr
2. l5kit==1.1.0
3. Set min_history=0, min_future=10 in AgentDataset
4. Cosine annealing for LR decrease until 0

with training 1 epoch was already enough to win the prize.



# 1. Use train_full.zarr data

Bigger data is almost always better for deep learning model training. We used [Lyft Full Training Set](https://www.kaggle.com/philculliton/lyft-full-training-set).

However its size is really large, containing 191M data for AgentDataset.

Practically we need this modifications in order to train this big dataset in real time:

## Distributed training
We implemented distributed training using `torch.distributed`.
It usually takes about 5 days to finish 1 epoch when we use 8GPUs.

## Caching some arrays into zarr beforehand to reduce on-memory usage in AgentDataset
The problem arises when we run distributed training and `DataLoader` with setting `num_workers` for multi-process data loading.

In the distributed training, 8 processes run in parallel and each process invokes `num_workers` subprocess. Therefore 8 * num_workers subprocess is launched and `AgentDataset` data is copied in each subprocess.
Then Out Of Memory error occurs because `AgentDataset` internally holds `cumulative_sizes` attribute whose size is very big ([code](https://github.com/lyft/l5kit/blob/1ea2f8cecfe7ad974419bf5c8519ab3a21300119/l5kit/l5kit/dataset/agent.py#L62)).

Instead, we pre-calculated `track_id, scene_index, state_index` and saved as the zarr format. So that we can load each data from disk, and reduce on-memory usage.

The Public Score was around **25.742** [kernel](https://www.kaggle.com/corochann/lyft-prediction-with-multi-mode-confidence) at this stage.

# 2. Use l5kit==1.1.0
As mentioned in the discussion [We did it all wrong](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/186492), image is rotated while the target value is not rotated in the previous version of l5kit==1.0.6 during the beginning of the competition.
We updated l5kit version to 1.1.0 once it is released, which fixes this behavior.

The Public LB score was jumped to **15.874** with this update.

# 3. Set min_history=0, min_future=10
As written in the “Validation Strategy” section, validation&test data is made by `create_chopped_dataset` method. We noticed that this validation/test data consists of the data with A. always contains more than 10 future frames, and B sometimes it does not contain any history frame.

To align the training `AgentDataset` to this test dataset behavior, we can set **`min_frame_history=0` and `min_frame_future=10`**.
I think **this modification is the most important part to notice in this competition**.
You need a courage to intentionally ignore l5kit library warning ;)([code](https://github.com/lyft/l5kit/blob/1ea2f8cecfe7ad974419bf5c8519ab3a21300119/l5kit/l5kit/dataset/agent.py#L15-L17)).

It’s very effective, the score jumped to **13.059**.

# 4. Training: with cosine annealing

Model: We trained & used following models for final ensemble
 - Resnet18
 - Resnet50
 - SEResNeXt50
 - ecaresnet18

However resnet18 baseline was strong, and enough to win the prize.

Image size: tried (128, 128) and (224, 224). image size of 128 training proceeds faster, but image size 224 final score was slightly better.

Optimizer: Adam with Cosine annealing
Cosine annealing was better than Exponential decay. I think decreasing the learning rate until very close to 0 is important for final tuning.

Batch size: 12 * 8 process = 96

We just trained only 1 epoch to train full data. We did not downsample any of the data.

Public LB score for single resnet18 model is **11.341**.

# Augmentation
## Image augmentation
Many of the augmentation used in natural images is not appropriate for this competition task, (for example flip augmentation flips the target value as well and not realistic since right-lane, left-lane will change). We tried
 - Cutout
 - Blur
 - Downscale
using [albumentations](https://github.com/albumentations-team/albumentations) library.

## Rasterizer-level augmentation
What is different from normal image prediction is that the image is drawn by rasterizer. We can also consider applying augmentation during rasterization.

I tried following augmentation by modifying `BoxRasterizer` ([code](https://github.com/lyft/l5kit/blob/1ea2f8cecfe7ad974419bf5c8519ab3a21300119/l5kit/l5kit/rasterization/box_rasterizer.py)).
 - Drop agent randomly
    - I assumed that the target agent’s movement does not change so much when the other agent far from the target agent exists or not. So we randomly skip drawing some of the agent boxes.
 - Scale extent size randomly
    - Even when the other agent size changes a bit, I assume that the target agent’s behavior does not change. So we scaled extent size from factor 0.9~1.1

I thought rasterizer-level augmentation was an interesting idea for this competition task. However we could not see big score improvement actually. Maybe the training dataset size is already big enough and its effect is not so big.

By **only adding cutout augmentation**, we achieved to get public LB score of **10.846**, already enough to win the prize. 

# Validation Strategy

As discussed in [Validation vs LB score](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/188695), we can run very stable validation using a chopped dataset.
However, it removes ground truth data, format is different from training AgentDataset and difficult to validate during training phase.

What we want is `agents_mask_orig_bool` ([code](https://github.com/lyft/l5kit/blob/1ea2f8cecfe7ad974419bf5c8519ab3a21300119/l5kit/l5kit/evaluation/chop_dataset.py#L68)).
We saved this `agents_mask_orig_bool` and set it to the `agent_mask` argument of `AgentDataset`.
Then we could run validation during training.

We sub-sampled 10000 dataset for fast validation during training, but it differs a lot from the score using a total 190327 validation dataset.
At the end phase of the competition, we validated the trained model using a full validation dataset.

# Ensemble: sample trajectory and GMM fitting
To improve the score further, how to ensemble is the key question in this task.
No golden method is suggested in [the discussion](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/180931) and we came up with an idea to adopt the **Gaussian Mixture Model**.

We can sample trajectory and fit the sampled points by GMM with the 3 components.
We started from the [sklearn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
Setting `n_components=3` and `covariance_type=”spherical”` achieved the good score.
However sigma is fixed to 1 in this competition [metric](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview/evaluation), so we also tried implementing own GMM model with fixing covariance to be 1.

Ensemble by GMM was really effective, we finally achieved **public LB score 10.272/private LB score 9.475**

Thus, the ensemble pushed the public LB score from 10.846 to 10.272. But it does not change the final rank this time ;)

By the way, I saw other participants used k-means clustering for ensembling coords. 
I think the behavior is quite similar with using GMM, since it calls k-means clustering in the initialization of EM algorithm.

# What we tried and not worked
We noticed Baseline model & l5kit default rasterizer was already very strong in this competition.
We really tried a lot, but many of the attempts failed to improve the scores. I’ll write in the reply section (since it’s already very long).
