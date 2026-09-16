# Current first place solution write-up

Competition: flower-classification-with-tpus
Rank: #1
Source: https://www.kaggle.com/c/flower-classification-with-tpus/discussion/150454

**DISCLAIMER**

*I don't read competition forum very often and as a result I missed  the information that usage of external datasets (I used the ones that were shared by @kirillblinov) was banned several days ago. As a result, there is very high probability that my solution is not eligible for prizes (but I ask confirm it from the organizer's side). However, I've got some interesting findings and maybe it would be interesting for other participants.*

First of all I would like to thank organizers for the good opportunity to test and evaluate TPU technology in this competition. Also I would like to note some participants who contribute a much: @hengck23  for sharing fresh ideas, experiments and external datasets, @cdeotte  for publishing great notebook and adapting augmentations for TPU kernel and @kirillblinov  for processing and sharing external datasets.

My main goal for this competition was to understand how to use free TPU kernels and what are the limitations when it used for free. I decided to use tensorflow instead of pytorch especially due to the fact that there is a great repository with the models specially adapted for TPU kernels: https://github.com/tensorflow/tpu/tree/master/models/official. That’s was a plan and below are the results of my experiments.

**Technical challenges with free tpu kernel**

1. The most common issue I discovered is a *“file system scheme [local] is not implement”* error. You can’t save files (checkpoints, logs) on local drive or google drive. Instead, google cloud storage (GCS) is required for input data and output. Mainly due to this fact I didn’t use the models from tensorflow github repo - they built on tf 1.x (efficientnet models) and disk space on GCS is required for temporary checkpoints during training.
2. The second issue was that some useful image processing utilites (e.g. augmentations) was removed from tensorflow 2.0 core and placed to tensorflow-addons (tfa). But tfa doesnt work properly with kaggle kernel TPU. This issue was solved by @cdeotte  who provided code for main augmentations that is compatible for kaggle TPU kernel.
3. The third issue was tensorflow-related: for every kind of model you have to use different tf releases: for example classical resnets-like model was ported to tf 2.x while efficientnet models works only with tf 1.x . When I tried to use these models in tf.compat.v1 regime (with paid GCS), I’ve got multiple depreciation warnings and the training results were bad.
4. Last but interesting issue - some useful features (like mixed precision training) were not working with TF 2.1 release on TPU. However it's work well on any TF 2.2 version. But TF 2.2 version was not stable and provided some other bugs (that I could fix)

In the end, I decided to use tf2 keras. For unknown reasons Keras let to save checkpoints on local kernel disk, have pretrained models for tf 2.X and it was widely used by other participants of this competition. 

**Solution tips and tricks**

1. **Data.** I used external datasets prepared by @hengck23 and adapted by @kirillblinov . My experiment showed that openimage and inaturalist decrease accuracy, so I removed them. I used pictures with 512 and 331 sizes. Also with mixed precision training I could use large batches (192) for all model architectures.
2. **Prepocessing.** I tried autoaugment, randaugment (the basis was official tensorflow implementation for tf 1.X and then adapted code for tf 2.X with TPU) and training without any augmentations. My experiments showed that with big datasets it is better don't use any augmentations.
3. **Model selection.** I tried all major efficientnet architectures. The best for me were b5 and b6 architecture with noisy-student pre-trained weights (great thanks to @pavel92) . The classical resnets (50 and 101) as well as se-resnext 50-101 were not as well as efficientnet.
4. **Model training.** I tried different strategies for cross-validation: 5-fold training, train-validation split, training without validation. As it was noted by many participants, training without validation provide higher score on public leaderboard and my final solution included the models that were trained without validation part. I used exponential decay LR scheduler with warmup. 
5. **Ensembling.** As i decided to use aggressive training strategy (no augmentations, no cross-validation), the ensembling was essential to avoid painful falling on private leaderboard. I trained best architectures (b5, b6, b7) with different picture sizes, try to combine effnet and seresnext models. The winner blend is three models B5-512, B5-331 and B6-512. This solution also provides my  best score on public LB.

As a result, I would say that though TPU kernels have some limitations, little bugs it is great opportunity for the researchers with limited computational capacities. In majority of computer vision competitions I participated with one laptop GPU (8gb) and google colab gpu. The model that usually calculated one day locally can be processed on TPU within one hour or faster. This technology is a great equalizer on kaggle competitions and it could facilitate deep learning researches as well.

Regarding the usage of external dataset, I can confirm and assure that I don't know about the ban for usage the external datasets. Moreover, when I read the forum last time (one or two weeks ago) this topic was discussed and my understanding was that it'is ok to use this external dataset. It would be great to implement some alerting system for all competition participants to spread the news like this. 

However, despite the issue like that it was a great time to participate in this competition. Thank you very much for all participants and organizers.

**UPD**: 

Some participants ask to me describe how did I find right model configuration without having validation part. Below is the high level explanation.

1. At first step I buit multi-dimensional grid  with the parameters I would like to test:
     a. **Model architectures** (Effnet b4, b5, b6, b7; Resnet 50, 101, Se-Resnext 50, 101, Inception)
     b. **Pictures size** (512, 331, 224)
     c. **Augmentations** (none, autoaugment, randaugment with different params). I selected these 
     three  types because it is easy to implement and they work well to beat some benchmarks in image classification tasks
  d. **Losses** (cross-entropy, focal)
 e. **Optimizer and lr** 
2. Then I made grid-search of best combinations in manual mode. I don't need to test all possible combinations of my params to understand that pic size 224 provide lower score than 512. Validation part was "valid" folder and public leaderboard. 
3. Thanks to TPU and relatively small dataset, I could test a lot of hypothesis and considerably reduce my parameters grid. Then I added external datasets, **but keep the same validation part**. So after some experiments I could confirm that my validation score improved with external data and public score improved as well. At this step I defined best models (best combo of grid params)
4.  The best models I test with different combination of external datasets and realize that it is possible to eliminate some of them while improving validation and public LB score
5.  My best models was the models without augmentations (strictly speaking "no augmentation" training mode include random left-right flip). And I realized that training loss correlated very well with validation and public LB score. And adding validation part in training process increase Public LB considerably. Due to this I decided to retrain best models with validation part and use the models with the lowest training loss.
6. Finally I've got about 10 models with good score. I tried different blend combinations. I  average probabilities because the majority of these models were effnets with 512 and 331 pic sizes. Nice to try here was to test different combination of voting but it was out of my goal to test TPU capabilites and I decided to use the simplest version of blending.
