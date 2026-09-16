# 22nd place journey : a completely different motion prediction approach

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #22
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199494

First of all, many thanks to the Kaggle team and Lyft team for hosting this competition, and congrats to all winners! Thanks to my teammates too, especially @doanquanvietnamca for his hard work and commitment during all those 3 laborious months.

# Preditcting agent motion with PointNet architecture ...
Our ideas are mainly based on the pointnet architechture. We totally forgot about L5kit package and deal with the raw data which was transformed into a 4D tensors of shape (num_mini_scenes, max_agents_on_frame, num_backward_frames, num_features). Those tensors are concatenated over n_batches scenes and feed to the pointnet architecture. We was able to reach a score of 21.xx  by using a single pointnet model but things become harder and we got stuck.


# Breakthrough ideas
To push our model performance a little bit, we manage to stack many pointnet models. This was doable since pointnet is very lightweight. Each model will look at a limited time step back to the agent history and output some embeddings of the scenes, the embeddging are then projected into a lower dimension space and concatenated. A simple full connected head is responsible for outputing the final 300+3 predictions.
Our best model is composed of 4 stacked models which resp. look at 10, 5, 3, and 1 frame back into the agent history. We use a simple zero padding for agents with no enough frames. With that giant pointnet model (~40 M params), we was able to reach a score of 13.353 on the public LB and 12.912 on the private.

# Custom loss and training
We implement a custom version of the competition loss in which agent's loss is ignored when it leaves the scene. This custom loss allows us to try things like sample weight, penalization ... We use **pytorch-lightening** to ease things. We maingly train on Colab, which is just owesome given the huge size of the competiion dataset. The optimizer is the classical Adam with a step learning rate scheduler, nothing fancy over there.

# Things that doesn't work
* Sample weight
* Bagging: we try a lot of ideas, and they  all fail :(
* RNN : we try a LSTM over time-stacked models without any success
* Longer history : augementing the history step leads to worse results (likely because of the many zeros coming from  our zero padding stratedy)


# Advantages of our model
* Very fast model, whole inference last 14 min's with single model and  less than 30 minutes with tens of stacked models
* Training on whole train_full  takes just 30' with a light pointnet model and 1h30' with 4 stacked models on a Colab Tesla-V100 single GPU

# Cons of our model
* Our pointnet implementation is completely road lanes blinded, even if we manage to incorporate light faces info in some extents


# Things we may like to try
* Moving from pointnet to other point-cloud models or voxel based models (pointCNN, point-Voxel, ShapeNet, ...)
* Stacking & transfer learning:  use our best models as embedders and train a simple model on top of them
* Combining our model with other image raster models (this one could make the pointnet road lanes aware)
* 3D convolutions

PS:  Our inference code by @doanquanvietnamca is available [here](https://www.kaggle.com/doanquanvietnamca/22st-solution-kkiller) .
