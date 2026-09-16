# 1st place solution summary

Competition: youtube8m-2018
Rank: #1
Source: https://www.kaggle.com/c/youtube8m-2018/discussion/62781

First off: Congratulations to everyone who participated and was burning their GPUs trying to squeeze in as much performance into 1GB as possible. Personally I would like to thank David for all the effort, time and resources he put into this competition, carrying out all of our crazy ideas.

**TLDR**: Our final model was combination of 9 submodels belonging to 4 model families - NetVlad, Deep Bag of Frames (DBoF), Fisher Vector (FV) and Recurrent neural networks (RNNs), weighting contributions based on test set performance. To make things work efficiently we used (multilayer) distillation, 8bit partial weights quantization, exponential moving averaging of weights and “smarter” inference time frame sampling. 

# Models and Distillation schema
We did not use any fancy new architectures for our models. The four used model families (NetVLAD, FV, RNNs and DBoFs) were all used in the first year competition. The models were however adjusted due to the competition constraints. We found especially useful architectures provided by Miech et al (https://arxiv.org/abs/1706.06905). Following figure shows models scores and flow of distillation training:

![enter image description here][1]
*Scores in architecture figure are public leaderboard equivalent scores.  Values were obtained through local validation which had a consistent offset to public leaderboard scores. The values are collected from quantized models.

# Tricks

## 1. Distillation

Distillation was done in a similar way as in work of Wang et. al. (https://arxiv.org/abs/1706.05150) using soft labels - mixture of ground truth and teacher model predictions. As a teacher model we used ensemble of 3 (or in case of 2nd layer model 6) models. For student networks we used same family of networks as teacher networks. This means that for a student NetVlad network we would use 3 teacher NetVlad. The idea was that similar networks could learn similar patterns from data - thus giving better results. This might not be entirely true since in some of our experiments, doing cross family training, the students performed as good as ones trained on same family while improving ensemble diversity. In general distillation allowed us to train lighter version of model with better performance.
For NetVlad, we chose 2nd layer distillation because of the positive impact it had on the overall ensemble score (+0.001) even though the individual models produced by it were not better than the 1st level distillation.

## 2. Quantization

To minimize weight of the models we casted them into 8 bit encoding (cutting weights size down to 1/4) in addition to storing the centroids. To not lose too much performance we kept the graph nodes in default float32 format and at inference time the variables are cast back to float32 format. We limited quantization only to variables with less than 1700 elements. This means that weights of typical fully connected layer would get quantized while batch norm factors wouldn’t get. We used uniform min-max quantization, since other methods (such as quantile based quantization) did not work well.

## 3. Inference time sampling
During training, for models like FV, NetVLAD and DBoF a collection of 300 frames was selected based on random sampling with replacement. During test time we ensured at each frame would be selected at least once and the remainder of the frames were chosen at random. This can be considered as a way of data augmentation.

## 4. In model averaging
To improve performance of single model we used in model averaging of weights over training time. We let the model train to convergence and then started averaging. Performance improvement was significant for our smaller models (DBoF and RNNs 0.002-0.003), while for bigger models (NetVLAD and FV) we did not see an improved prediction. We applied exponential decay averaging as well as equal weight averaging - both of them giving similar boosts, but we did not detect increase in performance if combining the two averages. 

# Train and testing

For training we used all samples except for samples in 800 validation TF records. The 800 randomly selected record files (~5% of all data) were used to monitor training, model selection and weighting of final meta-model. We observed a very consistent 0.00207 offset between local validation and public leaderboard so we used local validation for all testing.  The test set turned out to be sufficiently big enough to avoid overfitting. Training on all data with the 800 records included did not improve final score for us.

Training was performed on single GPU’s, taking between 1.5-3.5 days per model.  Approx 60 GPU-days would be required to train all models.  We varied batch size, number of clusters, and number of hidden layers between models within a given family.

Random search was performed on local system to determine final weights of the ensembles.  We observed a ~0.0050 range in GAP score that could be achieved between taking a weighted average of models vs optimized weights determined by random search in 500 iterations.

A detailed report/paper will be published at a later time in addition to making the code available.



  [1]: https://serving.photos.photobox.com/80844715f45b743d28b7d4ad3c3afcafbe48185f72e3fba054c5dbc6dba959e2bb3bf70c.jpg
